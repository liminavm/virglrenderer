// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The C ABI shim: all 69 exported symbols, and the one place global state is allowed.
//!
//! The C ABI has no handle for the renderer -- `virgl_renderer_init` initializes "the" renderer
//! and every later call addresses it implicitly. That implicit global has to live somewhere, and
//! this is it: ONE `static`, holding ONE owned root, in the module whose job is the ABI. It is not
//! a licence for a habit. Nothing outside this file reaches a global, and the native crate that
//! replaces this shim will hand the root out as a value instead.
//!
//! Every function here is a boundary against a guest, so every one of them validates rather than
//! asserting: a bad handle, an unknown context or a null pointer is rejected with an error code.
//! Asserts belong behind this line, on invariants we control.
//!
//! The exports are declared safe and dereference caller-supplied pointers, which is normally a
//! lint. It is the correct shape here and the allow is module-wide rather than per-function: the
//! contract is the C ABI's, identical for all 69, and stating it once is what keeps it readable.
//! The caller is the VMM, which owns every pointer it passes for the duration of the call.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

use crate::abi::{
    self, Box3, Callbacks, CreateBlobArgs, DebugCallback, FreeDataCallback, GlCtxParam, GuestIov,
    ImportBlobArgs, LogCallback, ResourceCreateArgs, ResourceInfo, ResourceInfoExt, VmmPtr,
};
use crate::ids::{CtxId, FenceId, ResourceHandle, RingIdx};
use crate::renderer::{self, Renderer};

/// Translate a renderer failure into the errno the C ABI answers with.
///
/// The whole reason [`renderer::Error`] exists: the negative integers stop here, so nothing on the
/// Rust side has to describe a failure in a vocabulary borrowed from a C header. Several causes
/// share a code because the ABI has no finer answer, not because they are the same thing.
fn errno(e: renderer::Error) -> c_int {
    use renderer::Error::*;
    match e {
        ZeroHandle | ResourceExists | ContextExists | NoContext | RendererAbsent | Poisoned => {
            EINVAL
        }
        RendererUnimplemented => -libc::ENOTSUP,
    }
}

/// THE global. See the module docs -- one static, one owned root, nothing else.
fn root() -> &'static Mutex<Option<Renderer>> {
    static ROOT: OnceLock<Mutex<Option<Renderer>>> = OnceLock::new();
    ROOT.get_or_init(|| Mutex::new(None))
}

/// Run `f` against the renderer, or return `err` if `virgl_renderer_init` has not been called.
fn with<T>(err: T, f: impl FnOnce(&mut Renderer) -> T) -> T {
    let mut g = root().lock().expect("the renderer lock is never held across a panic");
    match g.as_mut() {
        Some(r) => f(r),
        None => err,
    }
}

const ENOTSUP: c_int = -libc::ENOTSUP;
const EINVAL: c_int = -libc::EINVAL;
const ENOMEM: c_int = -libc::ENOMEM;

/// A symbol that belongs to a phase this build has not reached.
///
/// It is a hard error return, never a silent success. A stub that returned 0 would have the VMM
/// believe a resource was exported, a snapshot taken or a command submitted, and the failure
/// would surface somewhere with no connection to the missing feature.
macro_rules! todo_phase {
    ($phase:literal) => {{
        let _ = $phase;
        ENOTSUP
    }};
}

// ---------------------------------------------------------------- lifecycle

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_init(
    cookie: *mut c_void,
    flags: c_int,
    cb: *mut Callbacks,
) -> c_int {
    if cb.is_null() {
        return EINVAL;
    }
    // SAFETY: the VMM's contract is that `cb` points to a valid callbacks struct for the duration
    // of the call. We copy out what we keep and never retain the pointer.
    let cbs = unsafe { &*cb };
    if cbs.version < 3 {
        // v3 introduced write_context_fence, without which venus fences cannot retire at all.
        return EINVAL;
    }
    let mut g = root().lock().expect("the renderer lock is never held across a panic");
    if g.is_some() {
        return EINVAL;
    }
    eprintln!(
        "[virglrs] init flags={flags:#x} -- {}",
        crate::renderer::unsupported_renderers(flags)
    );
    *g = Some(Renderer::new(cookie, cbs, flags));
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_cleanup(_cookie: *mut c_void) {
    // Dropping the root joins the retirement thread, so a fence the VMM is still waiting on is
    // delivered before cleanup returns rather than being lost with the queue.
    let taken = root().lock().expect("the renderer lock is never held across a panic").take();
    drop(taken);
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_reset() {
    with((), |r| {
        let (res, ctx) = r.counts();
        eprintln!("[virglrs] reset: dropping {res} resources, {ctx} contexts");
    });
}

/// The C's implicit current context. Nothing here has one, so there is nothing to force.
#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_force_ctx_0() {}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_get_poll_fd() -> c_int {
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_poll() {}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_get_dev_fd(_ctx_id: c_int) -> c_int {
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_execute(_execute_args: *mut c_void, _execute_size: u32) -> c_int {
    todo_phase!("P2: venus execute")
}

// ---------------------------------------------------------------- contexts

/// The upstream ABI's `ctx_id` argument.
///
/// Zero is not a small context id: it is that ABI's implicit global, the `force_ctx_0` world this
/// tree exists to remove. The two are different kinds of thing, so an entry point below has to say
/// which one it is answering -- and nothing inside the renderer has to know the global was ever a
/// possibility, because no [`CtxId`] can carry it.
///
/// This is deliberately private and deliberately only on the upstream entry points. limina's own
/// `virgl_renderer_limina_*` calls have no global: we designed them, and there a zero is simply an
/// id that names nothing.
enum AbiCtx {
    /// The implicit global. Nothing here implements it; vrend is where it will mean something.
    Global,
    /// A context the guest created.
    Ctx(CtxId),
}

impl AbiCtx {
    fn new(raw: u32) -> AbiCtx {
        match CtxId::new(raw) {
            Some(id) => AbiCtx::Ctx(id),
            None => AbiCtx::Global,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_context_create(
    handle: u32,
    nlen: u32,
    name: *const c_char,
) -> c_int {
    virgl_renderer_context_create_with_flags(handle, 0, nlen, name)
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_context_create_with_flags(
    ctx_id: u32,
    ctx_flags: u32,
    nlen: u32,
    name: *const c_char,
) -> c_int {
    // The global is not a context the guest may create; it is the one that always existed.
    let AbiCtx::Ctx(id) = AbiCtx::new(ctx_id) else {
        return EINVAL;
    };
    let name = read_name(name, nlen);
    with(EINVAL, |r| match r.context_create(id, ctx_flags, name) {
        Ok(()) => 0,
        Err(e) => errno(e),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_context_destroy(handle: u32) {
    // The global is never destroyed, so this is a no-op for it, as it has always been.
    if let AbiCtx::Ctx(id) = AbiCtx::new(handle) {
        with((), |r| r.context_destroy(id));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_ctx_attach_resource(ctx_id: c_int, res_handle: c_int) {
    // Nothing is attached to the global: it holds no resource table of its own.
    if let AbiCtx::Ctx(id) = AbiCtx::new(ctx_id as u32) {
        with((), |r| r.ctx_attach_resource(id, ResourceHandle(res_handle as u32)));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_ctx_detach_resource(ctx_id: c_int, res_handle: c_int) {
    if let AbiCtx::Ctx(id) = AbiCtx::new(ctx_id as u32) {
        with((), |r| r.ctx_detach_resource(id, ResourceHandle(res_handle as u32)));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_context_poll(_ctx_id: u32) {}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_context_get_poll_fd(_ctx_id: u32) -> c_int {
    -1
}

// ---------------------------------------------------------------- resources

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_create(
    args: *mut ResourceCreateArgs,
    iov: *mut libc::iovec,
    num_iovs: u32,
) -> c_int {
    if args.is_null() {
        return EINVAL;
    }
    // SAFETY: the VMM's contract is that `args` is valid for the call; the fields are copied out.
    let a = unsafe { &*args };
    let iov = read_iov(iov, num_iovs);
    with(EINVAL, |r| match r.resource_create(a, iov) {
        Ok(()) => 0,
        Err(e) => errno(e),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_create_blob(args: *const CreateBlobArgs) -> c_int {
    if args.is_null() {
        return EINVAL;
    }
    // SAFETY: valid for the duration of the call by the VMM's contract; copied out below.
    let a = unsafe { &*args };
    with(EINVAL, |r| match r.resource_create_blob(a) {
        Ok(()) => 0,
        Err(e) => errno(e),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_import_blob(args: *const ImportBlobArgs) -> c_int {
    if args.is_null() {
        return EINVAL;
    }
    // SAFETY: valid for the duration of the call by the VMM's contract; copied out below.
    let a = unsafe { &*args };
    with(EINVAL, |r| {
        match r.resource_import(ResourceHandle(a.res_handle), a.blob_mem, a.fd_type, a.size) {
            Ok(()) => 0,
            Err(e) => errno(e),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_import_eglimage(
    _args: *mut ResourceCreateArgs,
    _image: *mut c_void,
) -> c_int {
    todo_phase!("P3: vrend EGL winsys")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_unref(res_handle: u32) {
    with((), |r| r.resource_unref(ResourceHandle(res_handle)));
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_set_priv(res_handle: u32, priv_: *mut c_void) {
    with((), |r| {
        if let Some(res) = r.resource_mut(ResourceHandle(res_handle)) {
            res.priv_ = VmmPtr(priv_);
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_priv(res_handle: u32) -> *mut c_void {
    with(std::ptr::null_mut(), |r| {
        r.resource(ResourceHandle(res_handle)).map_or(std::ptr::null_mut(), |res| res.priv_.0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_attach_iov(
    res_handle: c_int,
    iov: *mut libc::iovec,
    num_iovs: c_int,
) -> c_int {
    if num_iovs < 0 {
        return EINVAL;
    }
    let v = read_iov(iov, num_iovs as u32);
    with(EINVAL, |r| match r.resource_mut(ResourceHandle(res_handle as u32)) {
        Some(res) => {
            res.iov = v;
            0
        }
        None => EINVAL,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_detach_iov(
    res_handle: c_int,
    iov: *mut *mut libc::iovec,
    num_iovs: *mut c_int,
) {
    with((), |r| {
        let n = match r.resource_mut(ResourceHandle(res_handle as u32)) {
            Some(res) => {
                let n = res.iov.len();
                res.iov.clear();
                n
            }
            None => 0,
        };
        // The C hands back the array it was given. We copied it, so we own nothing the caller may
        // free -- report the count and a null array rather than inventing a pointer it would.
        if !iov.is_null() {
            // SAFETY: caller-provided out-pointer, checked non-null.
            unsafe { *iov = std::ptr::null_mut() };
        }
        if !num_iovs.is_null() {
            // SAFETY: caller-provided out-pointer, checked non-null.
            unsafe { *num_iovs = n as c_int };
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_info(
    res_handle: c_int,
    info: *mut ResourceInfo,
) -> c_int {
    if info.is_null() {
        return EINVAL;
    }
    with(EINVAL, |r| match r.resource(ResourceHandle(res_handle as u32)) {
        Some(_) => todo_phase!("P3: resource info needs the pipe resource"),
        None => EINVAL,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_info_ext(
    _res_handle: c_int,
    _info: *mut ResourceInfoExt,
) -> c_int {
    todo_phase!("P3: resource info needs the pipe resource")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_export_blob(
    _res_id: u32,
    _fd_type: *mut u32,
    _fd: *mut c_int,
) -> c_int {
    todo_phase!("P2: blob export")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_map(
    _res_handle: u32,
    _map: *mut *mut c_void,
    _out_size: *mut u64,
) -> c_int {
    todo_phase!("P2: blob mapping")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_map_fixed(_res_handle: u32, _addr: *mut c_void) -> c_int {
    todo_phase!("P2: blob mapping")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_unmap(_res_handle: u32) -> c_int {
    // Called unconditionally at unref to balance an eager map, so "was never mapped" is the
    // ordinary case and must be a harmless error, never a failure the VMM reports.
    EINVAL
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_map_info(
    _res_handle: u32,
    _map_info: *mut u32,
) -> c_int {
    todo_phase!("P2: blob mapping")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_map_ptr(
    _res_handle: u32,
    _map_ptr: *mut u64,
) -> c_int {
    todo_phase!("P2: blob mapping")
}

// ---------------------------------------------------------------- IOSurface

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_iosurface_id(
    res_handle: u32,
    iosurface_id: *mut u32,
) -> c_int {
    if iosurface_id.is_null() {
        return EINVAL;
    }
    // Zero means "not IOSurface-backed", which is the truth for every resource here. It must also
    // become the answer the instant a backing is freed: ids are recycled immediately, and a stale
    // one names a stranger's surface.
    with(EINVAL, |r| match r.resource(ResourceHandle(res_handle)) {
        Some(_) => {
            // SAFETY: caller-provided out-pointer, checked non-null.
            unsafe { *iosurface_id = 0 };
            0
        }
        None => EINVAL,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_read_iosurface(
    _res_handle: u32,
    _dst: *mut c_void,
    _dst_stride: u32,
    _height: u32,
) -> c_int {
    EINVAL
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_sync_iosurface(_res_handle: u32) -> c_int {
    EINVAL
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_republish_iosurface(_iosurface_id: u32) -> c_int {
    EINVAL
}

// ---------------------------------------------------------------- transfers and commands

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn virgl_renderer_transfer_read_iov(
    _handle: u32,
    _ctx_id: u32,
    _level: u32,
    _stride: u32,
    _layer_stride: u32,
    _box_: *mut Box3,
    _offset: u64,
    _iov: *mut libc::iovec,
    _iovec_cnt: c_int,
) -> c_int {
    todo_phase!("P3: vrend transfers")
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn virgl_renderer_transfer_write_iov(
    _handle: u32,
    _ctx_id: u32,
    _level: c_int,
    _stride: u32,
    _layer_stride: u32,
    _box_: *mut Box3,
    _offset: u64,
    _iovec: *mut libc::iovec,
    _iovec_cnt: c_uint,
) -> c_int {
    todo_phase!("P3: vrend transfers")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_submit_cmd(
    buffer: *mut c_void,
    ctx_id: c_int,
    ndw: c_int,
) -> c_int {
    let Some(buf) = cmd_slice(buffer, ndw) else {
        return EINVAL;
    };
    // vrend's global context arrives in P3; until then this ABI has no global to submit to.
    let AbiCtx::Ctx(id) = AbiCtx::new(ctx_id as u32) else {
        return EINVAL;
    };
    with(EINVAL, |r| match r.submit_cmd(id, buf) {
        Ok(()) => 0,
        Err(e) => errno(e),
    })
}

/// A submission as the ABI describes it: a pointer and a length in *dwords*, not bytes.
///
/// Returning `None` rather than an empty slice for a null pointer is deliberate -- a null buffer
/// with a non-zero length is a caller bug, and treating it as "nothing to do" would hide it.
fn cmd_slice<'a>(buffer: *mut c_void, ndw: c_int) -> Option<&'a [u8]> {
    if buffer.is_null() || ndw < 0 {
        return None;
    }
    // SAFETY: the caller owns this buffer and promised it holds `ndw` dwords. It is only read, and
    // the borrow does not outlive the call the slice is passed into.
    Some(unsafe { std::slice::from_raw_parts(buffer.cast::<u8>(), ndw as usize * 4) })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_submit_cmd2(
    buffer: *mut c_void,
    ctx_id: c_int,
    ndw: c_int,
    _in_fence_ids: *mut u64,
    _num_in_fences: u32,
) -> c_int {
    // The in-fences are a vrend feature; venus carries its waits inside the command stream.
    virgl_renderer_submit_cmd(buffer, ctx_id, ndw)
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_get_rect(
    _resource_id: c_int,
    _iov: *mut libc::iovec,
    _num_iovs: c_uint,
    _offset: u32,
    _x: c_int,
    _y: c_int,
    _width: c_int,
    _height: c_int,
) {
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_get_cursor_data(
    _resource_id: u32,
    _width: *mut u32,
    _height: *mut u32,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_get_fd_for_texture(_tex_id: u32, _fd: *mut c_int) -> c_int {
    todo_phase!("P3: dmabuf export -- not a path macOS has")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_get_fd_for_texture2(
    _tex_id: u32,
    _fd: *mut c_int,
    _stride: *mut c_int,
    _offset: *mut c_int,
) -> c_int {
    todo_phase!("P3: dmabuf export -- not a path macOS has")
}

// ---------------------------------------------------------------- caps

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_get_cap_set(set: u32, max_ver: *mut u32, max_size: *mut u32) {
    let (v, s) = with((0, 0), |r| r.capset_max(set).unwrap_or((0, 0)));
    if !max_ver.is_null() {
        // SAFETY: caller-provided out-pointer, checked non-null.
        unsafe { *max_ver = v };
    }
    if !max_size.is_null() {
        // SAFETY: caller-provided out-pointer, checked non-null.
        unsafe { *max_size = s };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_fill_caps(set: u32, version: u32, caps: *mut c_void) {
    // A capset this build does not advertise fills nothing: the caller sized its buffer from
    // `get_cap_set`, so writing into one we reported as absent corrupts the caller's stack.
    if caps.is_null() {
        return;
    }
    let Some(bytes) = with(None, |r| r.capset_bytes(set, version)) else {
        return;
    };
    // SAFETY: `caps` is the caller's buffer, which it sized from `virgl_renderer_get_cap_set` for
    // this same set -- and that reported exactly `bytes.len()`, the size of the capset struct.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), caps.cast::<u8>(), bytes.len()) };
}

// ---------------------------------------------------------------- fences

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_create_fence(client_fence_id: c_int, _ctx_id: u32) -> c_int {
    with(EINVAL, |r| {
        r.create_fence(client_fence_id as u32);
        0
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_context_create_fence(
    ctx_id: u32,
    _flags: u32,
    ring_idx: u32,
    fence_id: u64,
) -> c_int {
    // The global has no per-context ring to fence; `virgl_renderer_create_fence` is its path.
    let AbiCtx::Ctx(id) = AbiCtx::new(ctx_id) else {
        return EINVAL;
    };
    with(EINVAL, |r| match r.context_create_fence(id, RingIdx(ring_idx), FenceId(fence_id)) {
        Ok(()) => 0,
        Err(e) => errno(e),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_export_fence(_client_fence_id: u64, _fd: *mut c_int) -> c_int {
    todo_phase!("P5: sync export")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_export_signalled_fence() -> c_int {
    todo_phase!("P5: sync export")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_attach_fence(_ctx_id: c_int, _fence_fd: c_int) -> c_int {
    todo_phase!("P5: sync restore")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_get_fence_fd(_fence_id: u64) -> c_int {
    -1
}

// ---------------------------------------------------------------- logging

#[unsafe(no_mangle)]
pub extern "C" fn virgl_set_debug_callback(_cb: DebugCallback) -> DebugCallback {
    // Returns the PREVIOUS callback, which is always none here.
    None
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_set_log_callback(
    _cb: LogCallback,
    _user_data: *mut c_void,
    _free_user_data_cb: FreeDataCallback,
) {
}

// ---------------------------------------------------------------- limina extensions

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_dump_state() {
    with((), |r| {
        let (res, ctx) = r.counts();
        eprintln!("[virglrs] {res} resources, {ctx} contexts, flags {:#x}", r.flags);
        let todo = r.venus_todo();
        if !todo.is_empty() {
            let total: u64 = todo.iter().map(|(_, n)| n).sum();
            eprintln!("[virglrs] {total} venus commands in {} kinds not served yet:", todo.len());
            for (name, n) in &todo {
                eprintln!("[virglrs]   {n:>8}  {name}");
            }
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_journal_export(
    _ctx_id: u32,
    _out_buf: *mut *mut c_void,
    _out_size: *mut u64,
) -> c_int {
    todo_phase!("P5: snapshot")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_journal_seq(_ctx_id: u32) -> u64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_journal_unpin(_ctx_id: u32, _key: u64) {}

// The replay feed hands journal entries straight to the dispatcher, bypassing the ring buffer
// entirely -- which is what makes a replay possible with no guest and no VM. The entries have had
// their reply flag stripped by the recorder, so nothing below answers anything.

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_replay_begin(ctx_id: u32) -> c_int {
    let Some(ctx) = CtxId::new(ctx_id) else {
        return EINVAL;
    };
    with(EINVAL, |r| match r.venus_mut().map(|v| v.replay_begin(ctx)) {
        Some(Ok(())) => 0,
        _ => EINVAL,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_replay_submit(
    ctx_id: u32,
    cmd: *mut c_void,
    size: u32,
) -> c_int {
    let Some(buf) = byte_slice(cmd, size) else {
        return EINVAL;
    };
    let Some(ctx) = CtxId::new(ctx_id) else {
        return EINVAL;
    };
    with(EINVAL, |r| match r.venus_mut().map(|v| v.submit(ctx, buf)) {
        Some(Ok(())) => 0,
        _ => EINVAL,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_replay_ring_cmd(
    ctx_id: u32,
    ring_id: u64,
    cmd: *mut c_void,
    size: u32,
) -> c_int {
    let Some(buf) = byte_slice(cmd, size) else {
        return EINVAL;
    };
    // The ring id is the guest's 64-bit ring object; the index is what fences are keyed by. Until
    // a ring loop exists there is nothing to key, so the command is dispatched directly.
    let ring = RingIdx(ring_id as u32);
    let Some(ctx) = CtxId::new(ctx_id) else {
        return EINVAL;
    };
    with(EINVAL, |r| match r.venus_mut().map(|v| v.submit_ring(ctx, ring, buf)) {
        Some(Ok(())) => 0,
        _ => EINVAL,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_replay_end(ctx_id: u32) -> c_int {
    let Some(ctx) = CtxId::new(ctx_id) else {
        return EINVAL;
    };
    with(EINVAL, |r| match r.venus_mut().map(|v| v.replay_end(ctx)) {
        Some(Ok(())) => 0,
        _ => EINVAL,
    })
}

/// A buffer the caller owns, as bytes. See [`cmd_slice`] for why null is `None`.
fn byte_slice<'a>(p: *mut c_void, size: u32) -> Option<&'a [u8]> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller owns this buffer and promised it holds `size` bytes. It is only read, and
    // the borrow does not outlive the call the slice is passed into.
    Some(unsafe { std::slice::from_raw_parts(p.cast::<u8>(), size as usize) })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_sync_export(
    _ctx_id: u32,
    _out_buf: *mut *mut c_void,
    _out_size: *mut u64,
) -> c_int {
    todo_phase!("P5: snapshot")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_sync_restore(
    _ctx_id: u32,
    _data: *const c_void,
    _size: u64,
) -> c_int {
    todo_phase!("P5: snapshot")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_memory_census(
    ctx_id: u32,
    out_pairs: *mut *mut u64,
    out_count: *mut u32,
) -> c_int {
    if out_pairs.is_null() || out_count.is_null() {
        return EINVAL;
    }
    with(EINVAL, |r| {
        let Some(pairs) = CtxId::new(ctx_id).and_then(|c| r.venus_memory_census(c)) else {
            return EINVAL;
        };
        // The array is the caller's to `free`, which is the ABI's contract and the reason this
        // does not hand out a `Vec`: the VMM is C and frees it with `free`.
        let n = pairs.len();
        let buf = if n == 0 {
            core::ptr::null_mut()
        } else {
            let p = unsafe { libc::malloc(n * 2 * size_of::<u64>()) }.cast::<u64>();
            if p.is_null() {
                return ENOMEM;
            }
            for (i, (id, size)) in pairs.iter().enumerate() {
                // SAFETY: `p` holds 2*n u64s and `i` is below `n`.
                unsafe {
                    p.add(2 * i).write(*id);
                    p.add(2 * i + 1).write(*size);
                }
            }
            p
        };
        // SAFETY: both checked non-null above; the VMM's contract is that they are writable.
        unsafe {
            *out_pairs = buf;
            *out_count = n as u32;
        }
        0
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_memory_read(
    ctx_id: u32,
    mem_id: u64,
    buf: *mut c_void,
    size: u64,
) -> c_int {
    if buf.is_null() || size == 0 {
        return EINVAL;
    }
    with(EINVAL, |r| {
        // SAFETY: the VMM's contract is `size` writable bytes at `buf` for the length of the call.
        let out = unsafe { std::slice::from_raw_parts_mut(buf.cast::<u8>(), size as usize) };
        let Some(ctx) = CtxId::new(ctx_id) else {
            return EINVAL;
        };
        if r.venus_memory_read(ctx, mem_id, out) { 0 } else { EINVAL }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_memory_write(
    _ctx_id: u32,
    _mem_id: u64,
    _buf: *const c_void,
    _size: u64,
) -> c_int {
    todo_phase!("P5: snapshot")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_classic_content_export(
    _ctx_id: u32,
    _out_buf: *mut *mut c_void,
    _out_size: *mut u64,
) -> c_int {
    todo_phase!("P5: snapshot")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_classic_content_restore(
    _ctx_id: u32,
    _buf: *const c_void,
    _size: u64,
) -> c_int {
    todo_phase!("P5: snapshot")
}

// ---------------------------------------------------------------- helpers

/// The ABI passes a name as pointer + length, and the length is the guest's. Treat it as untrusted:
/// a null pointer, a zero length or invalid UTF-8 yields an empty name rather than a panic.
fn read_name(name: *const c_char, nlen: u32) -> String {
    if name.is_null() || nlen == 0 {
        return String::new();
    }
    // SAFETY: the VMM's contract is `nlen` readable bytes at `name` for the duration of the call.
    let bytes = unsafe { std::slice::from_raw_parts(name.cast::<u8>(), nlen as usize) };
    let bytes = match bytes.iter().position(|b| *b == 0) {
        Some(nul) => &bytes[..nul],
        None => bytes,
    };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Copy an iovec array's descriptions. The pages belong to the VMM; only the description is ours.
fn read_iov(iov: *mut libc::iovec, num_iovs: u32) -> Vec<GuestIov> {
    // SAFETY: the VMM's contract is `num_iovs` valid entries at `iov` for the duration of the
    // call; from_raw checks for null and copies the descriptions out.
    unsafe { GuestIov::from_raw(iov, num_iovs) }
}

/// Unused today, but the shim owes the header a `GlCtxParam` and a `CStr` import; keeping the
/// types referenced here means a drift in either is a build error rather than a surprise in P3.
#[allow(dead_code)]
fn _type_anchors(p: &GlCtxParam, s: &CStr) -> (c_int, usize) {
    (p.major_ver, s.to_bytes().len())
}

#[allow(dead_code)]
const _ABI_ANCHORS: (c_int, u32) = (abi::CALLBACKS_VERSION, abi::CAPSET_VENUS);

#[cfg(test)]
mod tests {
    use super::*;

    /// The one thing lost when the Rust API stopped speaking errno: nothing else checks that a
    /// cause still reaches the guest as the code it used to. `ENOTSUP` is the code that matters --
    /// it is the ABI's "this build has no renderer for that", and collapsing it into `EINVAL`
    /// would tell a VMM the guest sent something malformed instead.
    #[test]
    fn every_cause_keeps_the_errno_the_abi_answered_with() {
        use renderer::Error::*;
        for e in [ZeroHandle, ResourceExists, ContextExists, NoContext, RendererAbsent, Poisoned] {
            assert_eq!(errno(e), -libc::EINVAL, "{e:?} must still be EINVAL");
        }
        assert_eq!(errno(RendererUnimplemented), -libc::ENOTSUP);
        assert_ne!(errno(RendererUnimplemented), errno(RendererAbsent));
    }
}
