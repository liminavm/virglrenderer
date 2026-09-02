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
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::sync::{Mutex, OnceLock};

use crate::abi::{
    self, Box3, Callbacks, CreateBlobArgs, DebugCallback, FreeDataCallback, GlCtxParam, GuestIov,
    ImportBlobArgs, LogCallback, ResourceCreateArgs, ResourceInfo, ResourceInfoExt, VmmPtr,
};
use crate::config::{CapsetId, Config};
use crate::fence;
use crate::ids::{BlobId, ClientFenceId, CtxId, FenceId, ResourceHandle, RingId, RingIdx};
use crate::renderer::{self, BlobMem, FdType, ImportDesc, Renderer};

/// Decode a capset id the guest chose.
///
/// The context-create form arrives inside a flag word, whose other bits the header reserves and
/// nothing uses; `get_cap_set` passes the same id bare. Masking is done here so no unknown byte
/// reaches the renderer as a number -- an id we have no name for is [`CapsetId::Unknown`], which
/// is a value the renderer can match on rather than one it has to compare against constants.
fn capset_of(raw: u32) -> CapsetId {
    CapsetId::from_raw((raw & abi::CAPSET_MASK) as u8)
}

/// Decode `virgl_renderer_init`'s flag word into what the renderer is being asked to be.
///
/// Three of the eleven flags reach the renderer. The rest choose a winsys (EGL, GLES,
/// surfaceless, DRM), a threading model we implement unconditionally (`THREAD_SYNC`,
/// `ASYNC_FENCE_CB`, `RENDER_SERVER`), or a feature that is vrend's (`USE_VIDEO`) -- none of them
/// is a question the renderer answers, so none of them is carried inward.
///
/// Note `NO_VIRGL` inverting: the ABI names the absence, [`Config`] names the presence.
fn config_of(flags: c_int) -> Config {
    Config {
        venus: flags & abi::VENUS != 0,
        vrend: flags & abi::NO_VIRGL == 0,
        guest_vram: flags & abi::USE_GUEST_VRAM != 0,
    }
}

/// The VMM's callback table, as a place for retired fences to go.
///
/// The C ABI's half of [`FenceSink`]: it holds the two entry points the renderer can reach and
/// the opaque token they are called with. A callback the VMM did not supply means that kind of
/// fence is dropped -- which is the C table's contract, and not something a Rust implementor
/// should inherit, so the optionality stops here.
///
/// No `unsafe impl Send` is needed: [`VmmPtr`] already carries that justification in the module
/// that owns the boundary, and an `extern "C" fn` is `Send` on its own.
struct VmmFences {
    cookie: VmmPtr,
    write_fence: Option<extern "C" fn(*mut c_void, u32)>,
    write_context_fence: Option<extern "C" fn(*mut c_void, u32, u32, u64)>,
}

impl fence::FenceSink for VmmFences {
    fn context_fence(&mut self, ctx: CtxId, ring: RingIdx, fence: FenceId) {
        if let Some(f) = self.write_context_fence {
            f(self.cookie.0, ctx.get(), ring.0, fence.0);
        }
    }

    fn global_fence(&mut self, fence: ClientFenceId) {
        if let Some(f) = self.write_fence {
            f(self.cookie.0, fence.0);
        }
    }
}

/// Translate a renderer failure into the errno the C ABI answers with.
///
/// The whole reason [`renderer::Error`] exists: the negative integers stop here, so nothing on the
/// Rust side has to describe a failure in a vocabulary borrowed from a C header. Several causes
/// share a code because the ABI has no finer answer, not because they are the same thing.
fn errno(e: renderer::Error) -> c_int {
    use renderer::Error::*;
    match e {
        ResourceExists
        | NoResource
        | ContextExists
        | NoContext
        | RendererAbsent
        | Poisoned
        | NoAllocation
        | NotMappable
        | ZeroSize
        | Unmappable
        | AlreadyExported
        | NotHostVisible
        | BlobLargerThanAllocation => EINVAL,
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
    // `Callbacks` is size-versioned: the VMM allocates only the prefix its `version` names, so the
    // struct in front of us is shorter than ours whenever it was built against an older header --
    // and `get_egl_display` is v4, so even a conforming v3 caller's allocation ends before our
    // last field. A `&Callbacks` would therefore claim bytes the caller never owned. Every read
    // below projects to one field and stops there, so nothing outside the version's prefix is
    // touched.
    //
    // SAFETY: the VMM's contract is that `cb` points to a callbacks struct whose `version` field
    // is initialised and whose allocation covers that version's prefix, for the duration of the
    // call. `version` is at offset 0, inside every version.
    let version = unsafe { (&raw const (*cb).version).read() };
    if version < 3 {
        // v3 introduced write_context_fence, without which venus fences cannot retire at all.
        return EINVAL;
    }
    let mut g = root().lock().expect("the renderer lock is never held across a panic");
    if g.is_some() {
        return EINVAL;
    }
    eprintln!(
        "[virglrs] init flags={flags:#x} -- {}",
        crate::renderer::unsupported_renderers(config_of(flags))
    );
    // Only the two fence callbacks are read. The other six are vrend's winsys hooks, which
    // nothing here calls; they get a trait of their own when P3 needs one.
    //
    // SAFETY: both fields are within the v3 prefix, which the check above proved the caller
    // allocated. `callbacks_offsets_match_the_c_header` pins that claim to the header's layout.
    let sink = unsafe {
        VmmFences {
            cookie: VmmPtr(cookie),
            write_fence: (&raw const (*cb).write_fence).read(),
            write_context_fence: (&raw const (*cb).write_context_fence).read(),
        }
    };
    *g = Some(Renderer::new(Box::new(sink), config_of(flags)));
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
    with((), |r| r.reset());
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
    with(EINVAL, |r| match r.context_create(id, capset_of(ctx_flags), name) {
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
    let (AbiCtx::Ctx(id), Some(handle)) =
        (AbiCtx::new(ctx_id as u32), ResourceHandle::new(res_handle as u32))
    else {
        return;
    };
    with((), |r| r.ctx_attach_resource(id, handle));
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_ctx_detach_resource(ctx_id: c_int, res_handle: c_int) {
    let (AbiCtx::Ctx(id), Some(handle)) =
        (AbiCtx::new(ctx_id as u32), ResourceHandle::new(res_handle as u32))
    else {
        return;
    };
    with((), |r| r.ctx_detach_resource(id, handle));
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_context_poll(_ctx_id: u32) {}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_context_get_poll_fd(_ctx_id: u32) -> c_int {
    -1
}

// ---------------------------------------------------------------- resources

/// Split the ABI's create args into the handle it names and the resource it describes.
///
/// Field by field on purpose. A `From` impl would have to live beside one of the two types, which
/// means either the C layout appearing in the renderer or the renderer's type appearing in the
/// ABI module -- and the whole point of the split is that neither knows the other.
fn classic_desc(a: &ResourceCreateArgs) -> (Option<ResourceHandle>, renderer::ClassicDesc) {
    let desc = renderer::ClassicDesc {
        target: a.target,
        format: a.format,
        bind: a.bind,
        width: a.width,
        height: a.height,
        depth: a.depth,
        array_size: a.array_size,
        last_level: a.last_level,
        nr_samples: a.nr_samples,
        flags: a.flags,
    };
    (ResourceHandle::new(a.handle), desc)
}

/// The blob the ABI's create args describe.
///
/// `ctx_id` is dropped: nothing reads it today, and when host3d blobs land its Rust shape is
/// `Option<CtxId>` rather than a `u32`, because a guest-memory blob legitimately has no context
/// and zero is how the ABI spells that.
/// The C's flat argument struct as the two operations it actually encodes.
///
/// `blob_id` means something only for a blob whose storage is the host's: the C reads it solely on
/// the `HOST3D` path and ignores it everywhere else, and a guest-storage blob carrying a non-zero
/// id would otherwise arrive here as an export of memory no context was named for. Reconciled
/// once, here, because this is the boundary that knows what the ABI meant.
fn blob_desc(a: &CreateBlobArgs) -> Option<renderer::BlobDesc> {
    let source = match (a.blob_mem, a.blob_id) {
        // An export names memory in some context's table, so a request that names no context
        // names no memory either. `None` here is the refusal -- the alternative, treating it as a
        // mint, would answer with fresh zeroed pages for a guest that asked for its own bytes.
        (crate::abi::BLOB_MEM_HOST3D, id) if id != 0 => {
            renderer::BlobSource::Exported { ctx: CtxId::new(a.ctx_id)?, mem: BlobId(id) }
        }
        _ => renderer::BlobSource::HostMinted,
    };
    Some(renderer::BlobDesc {
        blob_mem: a.blob_mem,
        blob_flags: a.blob_flags,
        source,
        size: a.size,
    })
}

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
    let (handle, desc) = classic_desc(a);
    // Zero was already `Error::ZeroHandle` here, answered with EINVAL. Now it is the parse that
    // fails, and the answer is the same one.
    let Some(handle) = handle else {
        return EINVAL;
    };
    let iov = read_iov(iov, num_iovs);
    with(EINVAL, |r| match r.resource_create(handle, desc, iov) {
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
    let Some(handle) = ResourceHandle::new(a.res_handle) else {
        return EINVAL;
    };
    let Some(desc) = blob_desc(a) else {
        return EINVAL;
    };
    // SAFETY: the VMM's contract for create_blob is that `iovecs` points to `num_iovs` valid
    // entries for the duration of the call. Done here so the renderer never sees a raw pointer.
    let iov = unsafe { GuestIov::from_raw(a.iovecs, a.num_iovs) };
    with(EINVAL, |r| match r.resource_create_blob(handle, desc, iov) {
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
    // Every refusal that does not need the renderer happens first, while the descriptor is still
    // plainly the caller's -- the ABI transfers it only on success, so nothing may be constructed
    // that would close it on the way out.
    let (Some(blob_mem), Some(fd_type)) = (blob_mem_of(a.blob_mem), fd_type_of(a.fd_type)) else {
        return EINVAL;
    };
    if a.fd < 0 {
        return EINVAL;
    }
    let Some(handle) = ResourceHandle::new(a.res_handle) else {
        return EINVAL;
    };
    let desc = ImportDesc { blob_mem, fd_type, size: a.size };
    // SAFETY: `a.fd` is non-negative and the ABI's contract is that a successful import takes it.
    // Both paths below account for it: the renderer files it under the resource, or hands it back
    // and `return_fd` releases it without closing.
    let fd = unsafe { OwnedFd::from_raw_fd(a.fd) };
    // Spelled out rather than run through `with`, because `with`'s not-initialized answer is a
    // value constructed before the call and dropped after it -- and dropping this one would close
    // a descriptor the caller still owns.
    let mut g = root().lock().expect("the renderer lock is never held across a panic");
    let Some(r) = g.as_mut() else {
        return_fd(fd, a.fd);
        return EINVAL;
    };
    match r.resource_import(handle, desc, fd) {
        Ok(()) => 0,
        Err(rej) => {
            return_fd(rej.fd, a.fd);
            errno(rej.error)
        }
    }
}

/// `blob_mem` as the import path accepts it. Creation accepts more; see [`BlobMem`].
fn blob_mem_of(v: u32) -> Option<BlobMem> {
    match v {
        abi::BLOB_MEM_HOST3D => Some(BlobMem::Host3d),
        abi::BLOB_MEM_GUEST_VRAM => Some(BlobMem::GuestVram),
        _ => None,
    }
}

fn fd_type_of(v: u32) -> Option<FdType> {
    match v {
        abi::BLOB_FD_TYPE_DMABUF => Some(FdType::DmaBuf),
        abi::BLOB_FD_TYPE_OPAQUE => Some(FdType::Opaque),
        abi::BLOB_FD_TYPE_SHM => Some(FdType::Shm),
        _ => None,
    }
}

/// Release a refused import's descriptor without closing it: the caller still owns it.
fn return_fd(fd: OwnedFd, expected: c_int) {
    let raw = fd.into_raw_fd();
    assert_eq!(raw, expected, "an import handed back a descriptor other than the one it took");
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
    let Some(handle) = ResourceHandle::new(res_handle) else {
        return;
    };
    with((), |r| r.resource_unref(handle));
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_set_priv(res_handle: u32, priv_: *mut c_void) {
    let Some(handle) = ResourceHandle::new(res_handle) else {
        return;
    };
    with((), |r| {
        r.with_resource_mut(handle, |res| res.priv_ = VmmPtr(priv_));
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_priv(res_handle: u32) -> *mut c_void {
    let Some(handle) = ResourceHandle::new(res_handle) else {
        return std::ptr::null_mut();
    };
    with(std::ptr::null_mut(), |r| {
        r.with_resource(handle, |res| res.priv_.0).unwrap_or(std::ptr::null_mut())
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
    let Some(handle) = ResourceHandle::new(res_handle as u32) else {
        return EINVAL;
    };
    let v = read_iov(iov, num_iovs as u32);
    with(EINVAL, |r| {
        r.with_resource_mut(handle, |res| {
            res.iov = v;
            0
        })
        .unwrap_or(EINVAL)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_detach_iov(
    res_handle: c_int,
    iov: *mut *mut libc::iovec,
    num_iovs: *mut c_int,
) {
    let Some(handle) = ResourceHandle::new(res_handle as u32) else {
        return;
    };
    with((), |r| {
        let n = r
            .with_resource_mut(handle, |res| {
                let n = res.iov.len();
                res.iov.clear();
                n
            })
            .unwrap_or(0);
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
    let Some(handle) = ResourceHandle::new(res_handle as u32) else {
        return EINVAL;
    };
    with(EINVAL, |r| {
        r.with_resource(handle, |_| todo_phase!("P3: resource info needs the pipe resource"))
            .unwrap_or(EINVAL)
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

/// Where a blob resource lives, for the three ABI calls that each want part of the same answer.
///
/// One resolution, so a VMM that asks for the address, then the size, then the caching mode
/// cannot be told about three different states of the same resource.
fn host_mapping(res_handle: u32) -> Result<renderer::HostMapping, c_int> {
    let handle = ResourceHandle::new(res_handle).ok_or(EINVAL)?;
    with(Err(EINVAL), |r| r.resource_host_mapping(handle).map_err(errno))
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_map(
    res_handle: u32,
    map: *mut *mut c_void,
    out_size: *mut u64,
) -> c_int {
    if map.is_null() || out_size.is_null() {
        return EINVAL;
    }
    let m = match host_mapping(res_handle) {
        Ok(m) => m,
        Err(e) => return e,
    };
    // SAFETY: both were checked non-null above, and the caller owns writable storage for each --
    // this is the ABI's way of returning two values.
    unsafe {
        *map = m.addr as *mut c_void;
        *out_size = m.size;
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_map_fixed(_res_handle: u32, _addr: *mut c_void) -> c_int {
    todo_phase!("P2: blob mapping")
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_unmap(res_handle: u32) -> c_int {
    // Nothing to undo either way. A blob's mapping is owned by whatever minted it -- the shm the
    // resource holds, or the `VkDeviceMemory` that was exported -- and is released when that is,
    // so unmapping here would take the memory out from under a guest that still has the blob.
    //
    // Called unconditionally at unref to balance an eager map, so "was never mapped" is the
    // ordinary case and must be a harmless error, never a failure the VMM reports.
    match host_mapping(res_handle) {
        Ok(_) => 0,
        Err(_) => EINVAL,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_map_info(
    res_handle: u32,
    map_info: *mut u32,
) -> c_int {
    if map_info.is_null() {
        return EINVAL;
    }
    let m = match host_mapping(res_handle) {
        Ok(m) => m,
        Err(e) => return e,
    };
    let info = match m.caching {
        renderer::Caching::Cached => crate::abi::MAP_CACHE_CACHED,
        renderer::Caching::WriteCombining => crate::abi::MAP_CACHE_WC,
    };
    // SAFETY: checked non-null above; the caller owns writable storage for one `u32`.
    unsafe { *map_info = info };
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_resource_get_map_ptr(res_handle: u32, map_ptr: *mut u64) -> c_int {
    if map_ptr.is_null() {
        return EINVAL;
    }
    let m = match host_mapping(res_handle) {
        Ok(m) => m,
        Err(e) => return e,
    };
    // SAFETY: checked non-null above; the caller owns writable storage for one `u64`.
    unsafe { *map_ptr = m.addr as u64 };
    0
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
    let Some(handle) = ResourceHandle::new(res_handle) else {
        return EINVAL;
    };
    with(EINVAL, |r| {
        // A resource that is not here is the one refusal, as in the C. Everything else answers,
        // and zero is the ABI's "not IOSurface-backed" -- which is also what a resource whose
        // surface has been released says, because the id is resolved from the live surface on
        // every call and there is no longer one to ask.
        if r.with_resource(handle, |_| ()).is_none() {
            return EINVAL;
        }
        let id = r.resource_iosurface_id(handle).map_or(0, |s| s.0);
        // SAFETY: caller-provided out-pointer, checked non-null.
        unsafe { *iosurface_id = id };
        0
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
    // vrend's global context arrives in P3; until then this ABI has no global to submit to.
    let AbiCtx::Ctx(id) = AbiCtx::new(ctx_id as u32) else {
        return EINVAL;
    };
    with_cmd_bytes(buffer, ndw, |buf| {
        with(EINVAL, |r| match r.submit_cmd(id, buf) {
            Ok(()) => 0,
            Err(e) => errno(e),
        })
    })
    .unwrap_or(EINVAL)
}

/// A submission as the ABI describes it: a pointer and a length in *dwords*, not bytes.
fn with_cmd_bytes<R>(buffer: *mut c_void, ndw: c_int, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
    let dwords = usize::try_from(ndw).ok()?;
    with_bytes(buffer, dwords * 4, f)
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
    let (v, s) = with((0, 0), |r| r.capset_max(capset_of(set)).unwrap_or((0, 0)));
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
    // The version is the C caller's request, and only this side has one: `get_cap_set` told it
    // which version to ask for, and a different number names a layout we never described.
    if version != crate::venus::capset::VERSION {
        return;
    }
    let Some(capset) = with(None, |r| r.capset(capset_of(set))) else {
        return;
    };
    let bytes = capset.as_bytes();
    // SAFETY: `caps` is the caller's buffer, which it sized from `virgl_renderer_get_cap_set` for
    // this same set -- and that reported exactly `bytes.len()`, the size of the capset struct.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), caps.cast::<u8>(), bytes.len()) };
}

// ---------------------------------------------------------------- fences

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_create_fence(client_fence_id: c_int, _ctx_id: u32) -> c_int {
    with(EINVAL, |r| {
        r.create_fence(ClientFenceId(client_fence_id as u32));
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
        eprintln!("[virglrs] {res} resources, {ctx} contexts, {:?}", r.config);
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
    with(EINVAL, |r| match Some(r.venus_replay_begin(ctx)) {
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
    let Some(ctx) = CtxId::new(ctx_id) else {
        return EINVAL;
    };
    with_bytes(cmd, size as usize, |buf| {
        with(EINVAL, |r| match Some(r.venus_replay_cmd(ctx, buf)) {
            Some(Ok(())) => 0,
            _ => EINVAL,
        })
    })
    .unwrap_or(EINVAL)
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_replay_ring_cmd(
    ctx_id: u32,
    ring_id: u64,
    cmd: *mut c_void,
    size: u32,
) -> c_int {
    // The ring object the guest named, at its full width. This used to narrow into a `RingIdx`,
    // which is a fence timeline index and a different concept -- two rings whose ids differed
    // only above bit 32 became the same ring, silently.
    let ring = RingId(ring_id);
    let Some(ctx) = CtxId::new(ctx_id) else {
        return EINVAL;
    };
    with_bytes(cmd, size as usize, |buf| {
        with(EINVAL, |r| match Some(r.venus_replay_ring_cmd(ctx, ring, buf)) {
            Some(Ok(())) => 0,
            _ => EINVAL,
        })
    })
    .unwrap_or(EINVAL)
}

#[unsafe(no_mangle)]
pub extern "C" fn virgl_renderer_limina_replay_end(ctx_id: u32) -> c_int {
    let Some(ctx) = CtxId::new(ctx_id) else {
        return EINVAL;
    };
    with(EINVAL, |r| match Some(r.venus_replay_end(ctx)) {
        Some(Ok(())) => 0,
        _ => EINVAL,
    })
}

/// Runs `f` on a buffer the caller owns, as bytes.
///
/// The slice goes to a closure rather than being returned because a returned `&'a [u8]` has no
/// borrow to anchor `'a` to: the lifetime is chosen by the caller, so nothing stops it being
/// inferred as `'static` and the slice outliving the call the VMM guaranteed it for. The bound
/// here is higher-ranked over the slice's lifetime, so `R` cannot name it -- an attempt to let the
/// slice escape is a compile error rather than a convention to remember.
///
/// Returning `None` rather than running `f` on an empty slice for a null pointer is deliberate --
/// a null buffer with a non-zero length is a caller bug, and treating it as "nothing to do" would
/// hide it.
fn with_bytes<R>(p: *mut c_void, len: usize, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller owns this buffer and promised it holds `len` bytes. It is only read, and
    // the slice cannot escape `f`.
    Some(f(unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) }))
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
        let Some(ctx) = CtxId::new(ctx_id) else {
            return EINVAL;
        };
        let census = match r.venus_memory_census(ctx) {
            Ok(c) => c,
            Err(e) => return errno(e),
        };
        // The array is the caller's to `free`, which is the ABI's contract and the reason this
        // does not hand out a `Vec`: the VMM is C and frees it with `free`.
        let n = census.len();
        let buf = if n == 0 {
            core::ptr::null_mut()
        } else {
            let p = unsafe { libc::malloc(n * 2 * size_of::<u64>()) }.cast::<u64>();
            if p.is_null() {
                return ENOMEM;
            }
            // The ABI's shape: id and size alternating in one array. Flattening a named pair
            // into two anonymous words is the shim's job, not the renderer's.
            for (i, a) in census.iter().enumerate() {
                // SAFETY: `p` holds 2*n u64s and `i` is below `n`.
                unsafe {
                    p.add(2 * i).write(a.id.0);
                    p.add(2 * i + 1).write(a.size);
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
        // The ABI answers success or a code, never a count -- a caller that wants fewer bytes
        // than the census reported passes a shorter buffer and knows what it asked for.
        match r.venus_memory_read(ctx, mem_id, out) {
            Ok(_) => 0,
            Err(e) => errno(e),
        }
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
const _ABI_ANCHORS: c_int = abi::CALLBACKS_VERSION;

#[cfg(test)]
mod tests {
    use super::*;

    /// Import accepts exactly the `blob_mem` and `fd_type` values the C accepts, and no others.
    ///
    /// The shim used to pass both through as bare `u32`s, so an import naming memory this build
    /// cannot serve -- or a descriptor kind it cannot interpret -- was filed as a live resource
    /// and answered 0. Creation's accept-set is wider than import's, which is why this cannot be
    /// one list: `BLOB_MEM_GUEST` and `BLOB_MEM_HOST3D_GUEST` are creatable and not importable.
    ///
    /// Ground truth: `virgl_renderer_resource_import_blob` in `src/virglrenderer.c`.
    #[test]
    fn import_names_the_blob_mems_and_fd_types_the_c_accepts() {
        assert_eq!(blob_mem_of(0x0002), Some(BlobMem::Host3d));
        assert_eq!(blob_mem_of(0x0004), Some(BlobMem::GuestVram));
        for v in [0x0000, 0x0001, 0x0003, 0x0005, u32::MAX] {
            assert_eq!(blob_mem_of(v), None, "blob_mem {v:#x} is not importable");
        }

        assert_eq!(fd_type_of(0x0001), Some(FdType::DmaBuf));
        assert_eq!(fd_type_of(0x0002), Some(FdType::Opaque));
        assert_eq!(fd_type_of(0x0003), Some(FdType::Shm));
        for v in [0x0000, 0x0004, u32::MAX] {
            assert_eq!(fd_type_of(v), None, "fd_type {v:#x} names no descriptor kind");
        }
    }

    /// A guest picks this byte, and picking a wrong one must not be mistaken for picking venus.
    ///
    /// The venus arm is gate-covered -- the replayer creates its contexts with it -- but the mask
    /// is not: the id arrives inside a flag word whose upper bits the header reserves, so reading
    /// the whole word would turn a venus context with any reserved bit set into an unknown one,
    /// and its every submission into ENOTSUP.
    #[test]
    fn a_capset_id_is_named_and_the_flag_word_around_it_is_ignored() {
        assert_eq!(capset_of(1), CapsetId::Virgl);
        assert_eq!(capset_of(2), CapsetId::Virgl2);
        assert_eq!(capset_of(4), CapsetId::Venus);

        // Reserved bits above the low byte belong to no capset and must not change the answer.
        assert_eq!(capset_of(4 | 0xdead_ff00), CapsetId::Venus);

        // An id we have no name for keeps its value rather than becoming one we do have a name
        // for -- collapsing it onto a known capset would route a guest to the wrong renderer.
        assert!(matches!(capset_of(0), CapsetId::Unknown(_)));
        assert!(matches!(capset_of(0x42), CapsetId::Unknown(_)));
    }

    /// `NO_VIRGL` is spelled inside out, and an inverted read of it is invisible: it changes a
    /// startup log line and, in P3, whether vrend exists at all. `guest_vram` is worse -- it
    /// reaches the guest through the capset, which decides where the guest allocates from, and no
    /// gate here reads a capset.
    ///
    /// `venus` is the one bit a corpus would catch, since the replay needs the renderer to exist.
    #[test]
    fn the_init_flags_decode_into_the_configuration_they_name() {
        assert_eq!(config_of(0), Config { venus: false, vrend: true, guest_vram: false });

        // Every flag that means something, and one that does not, to show it changes nothing.
        let all = abi::VENUS | abi::NO_VIRGL | abi::USE_GUEST_VRAM | abi::USE_EGL;
        assert_eq!(config_of(all), Config { venus: true, vrend: false, guest_vram: true });

        // One at a time, so a bit read for the wrong field cannot hide behind another.
        assert!(config_of(abi::VENUS).venus);
        assert!(!config_of(abi::NO_VIRGL).vrend, "NO_VIRGL means vrend is absent");
        assert!(config_of(abi::USE_GUEST_VRAM).guest_vram);
        assert_eq!(config_of(abi::USE_EGL), config_of(0), "a winsys flag reaches the renderer");
    }

    /// Nothing else checks this. No corpus calls `virgl_renderer_resource_create` -- the venus
    /// replayer binds only the blob entry point, and the classic path belongs to vrend, which
    /// arrives in P3 -- so a transposed pair here would compile clean and leave every gate green
    /// while handing the renderer a texture with its width and height swapped.
    ///
    /// The values are deliberately all different, so any crossed pair fails rather than only the
    /// pairs someone thought to check.
    #[test]
    fn the_abi_create_args_reach_the_renderer_field_for_field() {
        let a = ResourceCreateArgs {
            handle: 1,
            target: 2,
            format: 3,
            bind: 4,
            width: 5,
            height: 6,
            depth: 7,
            array_size: 8,
            last_level: 9,
            nr_samples: 10,
            flags: 11,
        };
        let (handle, d) = classic_desc(&a);
        assert_eq!(handle, ResourceHandle::new(1));
        assert_eq!(
            d,
            renderer::ClassicDesc {
                target: 2,
                format: 3,
                bind: 4,
                width: 5,
                height: 6,
                depth: 7,
                array_size: 8,
                last_level: 9,
                nr_samples: 10,
                flags: 11,
            }
        );
    }

    /// A handle of zero never becomes one.
    ///
    /// It used to travel as an ordinary `ResourceHandle` and be caught by a single check inside
    /// `Renderer::free_handle`, which meant every lookup that did not go through that check --
    /// unref, get_priv, attach_iov -- took zero as a name and merely failed to find it. The type
    /// refuses it at the boundary now, and `Error::ZeroHandle` is gone with the check. What the
    /// ABI answers is unchanged: `EINVAL`, which is what that error mapped to.
    #[test]
    fn a_resource_handle_of_zero_does_not_parse() {
        let a = ResourceCreateArgs {
            handle: 0,
            target: 2,
            format: 3,
            bind: 4,
            width: 5,
            height: 6,
            depth: 7,
            array_size: 8,
            last_level: 9,
            nr_samples: 10,
            flags: 11,
        };
        assert_eq!(classic_desc(&a).0, None);
        assert_eq!(ResourceHandle::new(0), None);
    }

    /// Likewise, and additionally which of the two operations the flat args encode.
    ///
    /// `blob_id` is meaningful only on the `HOST3D` path. Everywhere else the ABI carries whatever
    /// the guest put there and the C ignores it, so reading it unconditionally would turn a
    /// guest-storage blob into an export of memory nobody named a context for.
    #[test]
    fn the_abi_blob_args_say_which_of_the_two_blobs_was_asked_for() {
        let host3d = crate::abi::BLOB_MEM_HOST3D;
        let a = CreateBlobArgs {
            res_handle: 1,
            ctx_id: 2,
            blob_mem: host3d,
            blob_flags: 4,
            blob_id: 5,
            size: 6,
            iovecs: core::ptr::null(),
            num_iovs: 0,
        };
        assert_eq!(
            blob_desc(&a),
            Some(renderer::BlobDesc {
                blob_mem: host3d,
                blob_flags: 4,
                source: renderer::BlobSource::Exported {
                    ctx: CtxId::new(2).unwrap(),
                    mem: BlobId(5),
                },
                size: 6,
            }),
            "a host3d blob naming an id exports that context's memory"
        );

        // A zero id on the same path is the other operation entirely.
        let minted = CreateBlobArgs { blob_id: 0, ..a };
        assert_eq!(
            blob_desc(&minted).expect("a mint needs no context").source,
            renderer::BlobSource::HostMinted
        );

        // And an id set on a path that has no host storage is the guest's leftover, not a request.
        let guest = CreateBlobArgs { blob_mem: crate::abi::BLOB_MEM_GUEST_VRAM, ..a };
        assert_eq!(
            blob_desc(&guest).expect("guest storage needs no context").source,
            renderer::BlobSource::HostMinted,
            "the C reads blob_id only for host3d; reading it here would invent an export"
        );

        // An export naming no context names no memory: there is no table to resolve the id in.
        let orphan = CreateBlobArgs { ctx_id: 0, ..a };
        assert_eq!(blob_desc(&orphan), None, "an export with no context is refused, not minted");
    }

    /// The one thing lost when the Rust API stopped speaking errno: nothing else checks that a
    /// cause still reaches the guest as the code it used to. `ENOTSUP` is the code that matters --
    /// it is the ABI's "this build has no renderer for that", and collapsing it into `EINVAL`
    /// would tell a VMM the guest sent something malformed instead.
    #[test]
    fn every_cause_keeps_the_errno_the_abi_answered_with() {
        use renderer::Error::*;
        for e in [
            ResourceExists,
            ContextExists,
            NoContext,
            RendererAbsent,
            Poisoned,
            NoAllocation,
            NotMappable,
        ] {
            assert_eq!(errno(e), -libc::EINVAL, "{e:?} must still be EINVAL");
        }
        assert_eq!(errno(RendererUnimplemented), -libc::ENOTSUP);
        assert_ne!(errno(RendererUnimplemented), errno(RendererAbsent));
    }
}
