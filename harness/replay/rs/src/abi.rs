// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The renderer under test, reached only through its public C ABI.
//!
//! The library is opened at run time rather than linked, because the point of the harness is that
//! the same binary drives either implementation: `--renderer` picks the dylib. That is also the
//! only reason `unsafe` appears in this crate at all, and it is confined here -- every symbol is
//! resolved once in [`Renderer::open`] and exposed as a safe method, so no call site outside this
//! module writes `unsafe`.
//!
//! Layouts below are transcribed from `src/virglrenderer.h`. A layout that drifts from the header
//! compiles clean and corrupts at run time, which is precisely the failure the plan's ABI fixtures
//! exist to catch; until those land, [`Renderer::open`] at least refuses a library that is missing
//! a symbol it needs.

use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;

/// `struct virgl_renderer_callbacks`, callbacks version 4.
#[repr(C)]
pub struct Callbacks {
    pub version: c_int,
    pub write_fence: Option<extern "C" fn(*mut c_void, u32)>,
    pub create_gl_context: Option<extern "C" fn(*mut c_void, c_int, *mut c_void) -> *mut c_void>,
    pub destroy_gl_context: Option<extern "C" fn(*mut c_void, *mut c_void)>,
    pub make_current: Option<extern "C" fn(*mut c_void, c_int, *mut c_void) -> c_int>,
    pub get_drm_fd: Option<extern "C" fn(*mut c_void) -> c_int>,
    pub write_context_fence: Option<extern "C" fn(*mut c_void, u32, u32, u64)>,
    pub get_server_fd: Option<extern "C" fn(*mut c_void, u32) -> c_int>,
    pub get_egl_display: Option<extern "C" fn(*mut c_void) -> *mut c_void>,
}

/// `struct virgl_renderer_resource_create_blob_args`.
#[repr(C)]
pub struct CreateBlobArgs {
    pub res_handle: u32,
    pub ctx_id: u32,
    pub blob_mem: u32,
    pub blob_flags: u32,
    pub blob_id: u64,
    pub size: u64,
    pub iovecs: *const libc::iovec,
    pub num_iovs: u32,
}

pub const FLAG_THREAD_SYNC: c_int = 2;
pub const FLAG_VENUS: c_int = 1 << 6;
pub const FLAG_NO_VIRGL: c_int = 1 << 7;
pub const FLAG_ASYNC_FENCE_CB: c_int = 1 << 8;
pub const FLAG_RENDER_SERVER: c_int = 1 << 9;

/// Venus traffic needs no vrend and therefore no EGL, no GL context and no winsys. The render
/// server flag is not optional: the `limina_replay_*` entry points return -ENOTSUP unless the
/// same-process render server is initialized.
pub const DEFAULT_FLAGS: c_int =
    FLAG_VENUS | FLAG_NO_VIRGL | FLAG_RENDER_SERVER | FLAG_THREAD_SYNC | FLAG_ASYNC_FENCE_CB;

macro_rules! syms {
    ($($field:ident : $ty:ty = $name:literal),* $(,)?) => {
        struct Syms { $($field: $ty,)* }

        impl Syms {
            /// # Safety
            /// `h` must be a live handle from `dlopen` of a virglrenderer-compatible library.
            unsafe fn resolve(h: *mut c_void) -> Result<Syms, String> {
                $(
                    let $field = {
                        let n = CString::new($name).unwrap();
                        let p = libc::dlsym(h, n.as_ptr());
                        if p.is_null() {
                            return Err(format!(
                                "the library does not export {} -- it is not a virglrenderer \
                                 build this harness can drive",
                                $name
                            ));
                        }
                        std::mem::transmute::<*mut c_void, $ty>(p)
                    };
                )*
                Ok(Syms { $($field,)* })
            }
        }
    };
}

syms! {
    init: extern "C" fn(*mut c_void, c_int, *const Callbacks) -> c_int = "virgl_renderer_init",
    cleanup: extern "C" fn(*mut c_void) = "virgl_renderer_cleanup",
    context_create_with_flags: extern "C" fn(u32, u32, u32, *const c_char) -> c_int
        = "virgl_renderer_context_create_with_flags",
    context_destroy: extern "C" fn(u32) = "virgl_renderer_context_destroy",
    resource_create_blob: extern "C" fn(*const CreateBlobArgs) -> c_int
        = "virgl_renderer_resource_create_blob",
    resource_unref: extern "C" fn(u32) = "virgl_renderer_resource_unref",
    ctx_attach_resource: extern "C" fn(c_int, c_int) = "virgl_renderer_ctx_attach_resource",
    ctx_detach_resource: extern "C" fn(c_int, c_int) = "virgl_renderer_ctx_detach_resource",
    replay_begin: extern "C" fn(u32) -> c_int = "virgl_renderer_limina_replay_begin",
    replay_submit: extern "C" fn(u32, *mut c_void, u32) -> c_int
        = "virgl_renderer_limina_replay_submit",
    replay_ring_cmd: extern "C" fn(u32, u64, *mut c_void, u32) -> c_int
        = "virgl_renderer_limina_replay_ring_cmd",
    replay_end: extern "C" fn(u32) -> c_int = "virgl_renderer_limina_replay_end",
    dump_state: extern "C" fn() = "virgl_renderer_limina_dump_state",
    memory_census: extern "C" fn(u32, *mut *mut u64, *mut u32) -> c_int
        = "virgl_renderer_limina_memory_census",
    memory_read: extern "C" fn(u32, u64, *mut c_void, u64) -> c_int
        = "virgl_renderer_limina_memory_read",
}

/// The renderer under test. Owns the `dlopen` handle for the process lifetime: the library is
/// never unloaded, because venus keeps ring threads that outlive any single call.
pub struct Renderer {
    syms: Syms,
}

/// Venus fences retire through this. A replay has no guest to notify, so the only job is to not
/// be null -- venus asserts the v3 callback exists, and a renderer whose fences never retire
/// wedges on the first queue wait.
extern "C" fn write_context_fence(_cookie: *mut c_void, _ctx_id: u32, _ring: u32, _fence: u64) {}

extern "C" fn write_fence(_cookie: *mut c_void, _fence: u32) {}

impl Renderer {
    pub fn open(path: &str) -> Result<Renderer, String> {
        let c = CString::new(path).map_err(|e| e.to_string())?;
        // SAFETY: a valid NUL-terminated path; the handle is kept for the process lifetime.
        let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if h.is_null() {
            // SAFETY: dlerror is valid immediately after a failed dlopen on this thread.
            let e = unsafe { libc::dlerror() };
            let msg = if e.is_null() {
                "unknown error".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(e) }.to_string_lossy().into_owned()
            };
            return Err(format!("dlopen({path}): {msg}"));
        }
        // SAFETY: `h` is a live handle from the dlopen above.
        let syms = unsafe { Syms::resolve(h) }?;
        Ok(Renderer { syms })
    }

    pub fn init(&self, flags: c_int) -> Result<(), String> {
        let cbs = Callbacks {
            version: 4,
            write_fence: Some(write_fence),
            create_gl_context: None,
            destroy_gl_context: None,
            make_current: None,
            get_drm_fd: None,
            write_context_fence: Some(write_context_fence),
            get_server_fd: None,
            get_egl_display: None,
        };
        // SAFETY: `cbs` outlives the call, and the renderer copies what it keeps.
        let r = (self.syms.init)(ptr::null_mut(), flags, &cbs);
        if r != 0 {
            return Err(format!("virgl_renderer_init(flags={flags:#x}) returned {r}"));
        }
        Ok(())
    }

    pub fn cleanup(&self) {
        (self.syms.cleanup)(ptr::null_mut());
    }

    pub fn context_create(&self, ctx_id: u32, ctx_flags: u32, name: &str) -> c_int {
        let c = CString::new(name).unwrap_or_default();
        (self.syms.context_create_with_flags)(
            ctx_id,
            ctx_flags,
            c.as_bytes().len() as u32,
            c.as_ptr(),
        )
    }

    pub fn context_destroy(&self, ctx_id: u32) {
        (self.syms.context_destroy)(ctx_id);
    }

    /// `iovecs` must outlive the call; the renderer copies what it keeps.
    pub fn create_blob(&self, args: &CreateBlobArgs) -> c_int {
        (self.syms.resource_create_blob)(args)
    }

    pub fn resource_unref(&self, res_handle: u32) {
        (self.syms.resource_unref)(res_handle);
    }

    pub fn attach_resource(&self, ctx_id: u32, res_handle: u32) {
        (self.syms.ctx_attach_resource)(ctx_id as c_int, res_handle as c_int);
    }

    pub fn detach_resource(&self, ctx_id: u32, res_handle: u32) {
        (self.syms.ctx_detach_resource)(ctx_id as c_int, res_handle as c_int);
    }

    pub fn replay_begin(&self, ctx_id: u32) -> c_int {
        (self.syms.replay_begin)(ctx_id)
    }

    /// The wire buffer is mutated in place: the replay path clears the reply bit at offset 4, so
    /// a shared or read-only buffer would be a write through a const pointer.
    pub fn replay_submit(&self, ctx_id: u32, wire: &mut [u8]) -> c_int {
        (self.syms.replay_submit)(ctx_id, wire.as_mut_ptr().cast(), wire.len() as u32)
    }

    pub fn replay_ring_cmd(&self, ctx_id: u32, ring_id: u64, wire: &mut [u8]) -> c_int {
        (self.syms.replay_ring_cmd)(ctx_id, ring_id, wire.as_mut_ptr().cast(), wire.len() as u32)
    }

    pub fn replay_end(&self, ctx_id: u32) -> c_int {
        (self.syms.replay_end)(ctx_id)
    }

    pub fn dump_state(&self) {
        (self.syms.dump_state)();
    }

    /// `(VkDeviceMemory object id, allocation size)` for every capturable memory in the context.
    /// An empty census after a replay that reported successes means the commands were accepted and
    /// built nothing, which is the failure mode a pass/fail count alone cannot see.
    pub fn memory_census(&self, ctx_id: u32) -> Result<Vec<(u64, u64)>, c_int> {
        let mut pairs: *mut u64 = ptr::null_mut();
        let mut count: u32 = 0;
        let r = (self.syms.memory_census)(ctx_id, &mut pairs, &mut count);
        if r != 0 {
            return Err(r);
        }
        if pairs.is_null() || count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: on success the renderer returns a malloc'd array of 2*count u64s, ours to free.
        let out = unsafe {
            let s = std::slice::from_raw_parts(pairs, count as usize * 2);
            let v = s.chunks_exact(2).map(|c| (c[0], c[1])).collect();
            libc::free(pairs.cast());
            v
        };
        Ok(out)
    }

    /// Copy a capturable memory's contents out of its host mapping. `size` should be the
    /// allocation size the census reported; the renderer copies min(size, allocation).
    pub fn memory_read(&self, ctx_id: u32, mem_id: u64, buf: &mut [u8]) -> c_int {
        (self.syms.memory_read)(ctx_id, mem_id, buf.as_mut_ptr().cast(), buf.len() as u64)
    }
}
