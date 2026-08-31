// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The renderer root: the resource table, the context table, and fence tracking.
//!
//! Everything the renderer owns hangs off one `Renderer` reached through the shim's single
//! `static`. There are no file-scope mutables and no implicit current context -- `force_ctx_0`,
//! the C's implicit global, is a no-op here because nothing reads such a thing.

use std::collections::BTreeMap;
use std::ffi::{c_int, c_void};

use crate::abi::{self, Callbacks, CreateBlobArgs, GuestIov, ResourceCreateArgs, VmmPtr};
use crate::fence::Retirement;
use crate::ids::{BlobId, CtxId, FenceId, ResourceHandle, RingIdx};
use crate::venus;

/// What a resource is backed by. A resource is exactly one of these for its whole life; the C
/// keeps overlapping fields and a set of flags saying which are meaningful.
pub enum Backing {
    /// Created from `virgl_renderer_resource_create` -- a classic texture or buffer.
    Classic(ResourceCreateArgs),
    /// Created from `virgl_renderer_resource_create_blob` -- host memory the guest maps, or a
    /// handle exported by a context.
    Blob { blob_mem: u32, blob_flags: u32, blob_id: BlobId, size: u64 },
    /// Imported from a file descriptor the VMM already owns.
    Imported { blob_mem: u32, fd_type: u32, size: u64 },
}

pub struct Resource {
    pub handle: ResourceHandle,
    pub backing: Backing,
    /// The guest pages behind the resource, when it has any. Owned by the VMM, valid until it
    /// detaches them; we keep the description, never a copy.
    pub iov: Vec<GuestIov>,
    /// The VMM's opaque per-resource token (`resource_set_priv`/`get_priv`).
    pub priv_: VmmPtr,
    /// Contexts this resource is attached to. A resource outlives the contexts that used it, so
    /// this is what says whether an unref may actually free it.
    pub attached: Vec<CtxId>,
}

pub struct Context {
    pub id: CtxId,
    pub flags: u32,
    pub name: String,
    /// The last fence id created on each ring. A ring's fences retire in creation order, so this
    /// is what a later phase checks a retirement against.
    pub last_fence: BTreeMap<RingIdx, FenceId>,
}

pub struct Renderer {
    pub flags: c_int,
    resources: BTreeMap<ResourceHandle, Resource>,
    contexts: BTreeMap<CtxId, Context>,
    fences: Retirement,
    /// The venus renderer, present only when this build was initialized to serve it.
    venus: Option<venus::vkr::Vkr>,
}

impl Renderer {
    pub fn new(cookie: *mut c_void, cb: &Callbacks, flags: c_int) -> Renderer {
        Renderer {
            flags,
            resources: BTreeMap::new(),
            contexts: BTreeMap::new(),
            fences: Retirement::start(cookie, cb),
            venus: (flags & abi::VENUS != 0).then(|| venus::vkr::Vkr::new(flags)),
        }
    }

    /// Which capsets this build advertises, given the flags it was initialized with.
    ///
    /// Honest by construction: a capset appears only when the renderer that serves it is present.
    /// A skeleton that claimed VIRGL2 would have the guest bind a classic context and submit
    /// commands into a renderer that cannot answer them.
    pub fn capset_max(&self, set: u32) -> Option<(u32, u32)> {
        match set {
            abi::CAPSET_VENUS if self.venus.is_some() => {
                Some((venus::capset::VERSION, venus::capset::size()))
            }
            _ => None,
        }
    }

    /// The capset's bytes, for a set this build advertises. `None` for anything else -- the caller
    /// sized its buffer from `capset_max`, so writing into a buffer for a capset we reported as
    /// absent would run off the end of it.
    pub fn capset_bytes(&self, set: u32, version: u32) -> Option<Vec<u8>> {
        match set {
            abi::CAPSET_VENUS
                if self.flags & abi::VENUS != 0 && version == venus::capset::VERSION =>
            {
                Some(venus::capset::Capset::new(self.flags).as_bytes().to_vec())
            }
            _ => None,
        }
    }

    // ---- resources ----

    pub fn resource_create(
        &mut self,
        args: &ResourceCreateArgs,
        iov: Vec<GuestIov>,
    ) -> Result<(), c_int> {
        let handle = ResourceHandle(args.handle);
        // A guest-chosen handle that is already live is the guest's error, not ours: reject it
        // rather than replacing an entry something else still holds.
        if handle.0 == 0 || self.resources.contains_key(&handle) {
            return Err(-libc::EINVAL);
        }
        self.resources.insert(
            handle,
            Resource {
                handle,
                backing: Backing::Classic(ResourceCreateArgs { ..*args }),
                iov,
                priv_: VmmPtr::NULL,
                attached: Vec::new(),
            },
        );
        Ok(())
    }

    pub fn resource_create_blob(&mut self, args: &CreateBlobArgs) -> Result<(), c_int> {
        let handle = ResourceHandle(args.res_handle);
        if handle.0 == 0 || self.resources.contains_key(&handle) {
            return Err(-libc::EINVAL);
        }
        // SAFETY: the VMM's contract for create_blob is that `iovecs` points to `num_iovs` valid
        // entries for the duration of the call.
        let iov = unsafe { GuestIov::from_raw(args.iovecs, args.num_iovs) };
        self.resources.insert(
            handle,
            Resource {
                handle,
                backing: Backing::Blob {
                    blob_mem: args.blob_mem,
                    blob_flags: args.blob_flags,
                    blob_id: BlobId(args.blob_id),
                    size: args.size,
                },
                iov,
                priv_: VmmPtr::NULL,
                attached: Vec::new(),
            },
        );
        Ok(())
    }

    pub fn resource_import(
        &mut self,
        handle: ResourceHandle,
        blob_mem: u32,
        fd_type: u32,
        size: u64,
    ) -> Result<(), c_int> {
        if handle.0 == 0 || self.resources.contains_key(&handle) {
            return Err(-libc::EINVAL);
        }
        self.resources.insert(
            handle,
            Resource {
                handle,
                backing: Backing::Imported { blob_mem, fd_type, size },
                iov: Vec::new(),
                priv_: VmmPtr::NULL,
                attached: Vec::new(),
            },
        );
        Ok(())
    }

    pub fn resource(&self, handle: ResourceHandle) -> Option<&Resource> {
        self.resources.get(&handle)
    }

    pub fn resource_mut(&mut self, handle: ResourceHandle) -> Option<&mut Resource> {
        self.resources.get_mut(&handle)
    }

    pub fn resource_unref(&mut self, handle: ResourceHandle) {
        // Detach from every context first. A context holding a dangling handle is how the C's
        // use-after-free reached the command stream.
        if let Some(r) = self.resources.get_mut(&handle) {
            r.attached.clear();
        }
        self.resources.remove(&handle);
    }

    // ---- contexts ----

    pub fn context_create(&mut self, id: CtxId, flags: u32, name: String) -> Result<(), c_int> {
        // Context 0 is the ABI's implicit global, never a context the guest may create.
        if !id.is_real() || self.contexts.contains_key(&id) {
            return Err(-libc::EINVAL);
        }
        self.contexts.insert(id, Context { id, flags, name, last_fence: BTreeMap::new() });
        // A venus context gets venus state. The capset the guest bound is the low byte of the
        // flags, and it is the only thing that says which renderer a submission belongs to.
        if flags & abi::CAPSET_MASK == abi::CAPSET_VENUS
            && let Some(v) = self.venus.as_mut()
        {
            v.context_create(id);
        }
        Ok(())
    }

    pub fn context_destroy(&mut self, id: CtxId) {
        if self.contexts.remove(&id).is_none() {
            return;
        }
        if let Some(v) = self.venus.as_mut() {
            v.context_destroy(id);
        }
        // A destroyed context releases its claim on every resource; the resources themselves
        // survive, because the VMM unrefs them separately and may still be holding one.
        for r in self.resources.values_mut() {
            r.attached.retain(|c| *c != id);
        }
    }

    pub fn context(&self, id: CtxId) -> Option<&Context> {
        self.contexts.get(&id)
    }

    pub fn ctx_attach_resource(&mut self, ctx: CtxId, handle: ResourceHandle) {
        let known = self.contexts.contains_key(&ctx);
        if let (true, Some(r)) = (known, self.resources.get_mut(&handle))
            && !r.attached.contains(&ctx)
        {
            r.attached.push(ctx);
        }
    }

    pub fn ctx_detach_resource(&mut self, ctx: CtxId, handle: ResourceHandle) {
        if let Some(r) = self.resources.get_mut(&handle) {
            r.attached.retain(|c| *c != ctx);
        }
    }

    // ---- fences ----

    pub fn context_create_fence(
        &mut self,
        ctx: CtxId,
        ring: RingIdx,
        fence: FenceId,
    ) -> Result<(), c_int> {
        let Some(c) = self.contexts.get_mut(&ctx) else {
            return Err(-libc::EINVAL);
        };
        c.last_fence.insert(ring, fence);
        // Nothing here submits GPU work yet, so every fence is already satisfied. It still goes
        // through the retirement thread: the asynchrony is the contract, not an optimization.
        self.fences.retire_context(ctx, ring, fence);
        Ok(())
    }

    pub fn create_fence(&mut self, client_fence_id: u32) {
        self.fences.retire_global(client_fence_id);
    }

    // ---- venus ----

    /// Route a submission to the renderer the context bound. `Err` is a context that named no
    /// renderer we have, or a stream that poisoned the one it named.
    pub fn submit_cmd(&mut self, ctx: CtxId, buf: &[u8]) -> Result<(), c_int> {
        let Some(c) = self.contexts.get(&ctx) else {
            return Err(-libc::EINVAL);
        };
        match c.flags & abi::CAPSET_MASK {
            abi::CAPSET_VENUS => {
                let v = self.venus.as_mut().ok_or(-libc::EINVAL)?;
                v.submit(ctx, buf).map_err(|_| -libc::EINVAL)
            }
            // vrend arrives in P3.
            _ => Err(-libc::ENOTSUP),
        }
    }

    /// The venus renderer, for the replay feed that drives it directly.
    pub fn venus_mut(&mut self) -> Option<&mut venus::vkr::Vkr> {
        self.venus.as_mut()
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.resources.len(), self.contexts.len())
    }
}

/// Whether the flag word asks for a renderer this build does not have yet.
pub fn unsupported_renderers(flags: c_int) -> &'static str {
    if flags & abi::VENUS != 0 && flags & abi::NO_VIRGL == 0 {
        "venus and vrend"
    } else if flags & abi::VENUS != 0 {
        "venus"
    } else {
        "vrend"
    }
}
