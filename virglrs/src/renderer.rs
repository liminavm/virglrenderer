// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The renderer root: the resource table, the context table, and fence tracking.
//!
//! Everything the renderer owns hangs off one `Renderer` reached through the shim's single
//! `static`. There are no file-scope mutables and no implicit current context -- `force_ctx_0`,
//! the C's implicit global, is a no-op here because nothing reads such a thing.

use std::collections::BTreeMap;
use std::ffi::c_int;

use crate::abi::{self, GuestIov, VmmPtr};
use crate::fence::{FenceSink, Retirement};
use crate::ids::{BlobId, ClientFenceId, CtxId, FenceId, ResourceHandle, RingIdx};
use crate::venus;

/// Why a call failed.
///
/// Named causes, not error codes: the C ABI answers in `errno`, and translating to it is the
/// shim's job -- see `ffi::errno`. A Rust caller gets to tell "that id is already live" from
/// "there is no renderer for that capset" without consulting a table of negative integers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// A handle of zero, which names nothing.
    ZeroHandle,
    /// The guest reused a resource handle that is still live.
    ResourceExists,
    /// The guest reused a context id that is still live.
    ContextExists,
    /// No context under that id.
    NoContext,
    /// The context bound a capset this build was not initialized to serve.
    RendererAbsent,
    /// The context bound a capset no build serves yet.
    RendererUnimplemented,
    /// The stream violated the protocol; its context is poisoned and accepts nothing further.
    Poisoned,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Error::ZeroHandle => "handle zero names nothing",
            Error::ResourceExists => "that resource handle is already live",
            Error::ContextExists => "that context id is already live",
            Error::NoContext => "no such context",
            Error::RendererAbsent => "this build was not initialized to serve that capset",
            Error::RendererUnimplemented => "no renderer serves that capset yet",
            Error::Poisoned => "the context is poisoned",
        };
        f.write_str(s)
    }
}

impl std::error::Error for Error {}

/// A classic texture or buffer, as the guest described it -- everything about the resource except
/// the handle it is filed under, which is the caller's to name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ClassicDesc {
    pub target: u32,
    pub format: u32,
    pub bind: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    pub last_level: u32,
    pub nr_samples: u32,
    pub flags: u32,
}

/// Host memory the guest maps, or a handle a context exported.
///
/// `blob_mem` and `blob_flags` stay bare integers until the blob path is served: naming their
/// values means rejecting the ones we do not know, and a rejection nothing exercises is a
/// rejection nobody has checked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlobDesc {
    pub blob_mem: u32,
    pub blob_flags: u32,
    pub blob_id: BlobId,
    pub size: u64,
}

/// What a resource is backed by. A resource is exactly one of these for its whole life; the C
/// keeps overlapping fields and a set of flags saying which are meaningful.
pub enum Backing {
    /// Created from `virgl_renderer_resource_create` -- a classic texture or buffer.
    Classic(ClassicDesc),
    /// Created from `virgl_renderer_resource_create_blob`.
    Blob(BlobDesc),
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
    pub fn new(fences: Box<dyn FenceSink>, flags: c_int) -> Renderer {
        Renderer {
            flags,
            resources: BTreeMap::new(),
            contexts: BTreeMap::new(),
            fences: Retirement::start(fences),
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
        handle: ResourceHandle,
        desc: ClassicDesc,
        iov: Vec<GuestIov>,
    ) -> Result<(), Error> {
        // A guest-chosen handle that is already live is the guest's error, not ours: reject it
        // rather than replacing an entry something else still holds.
        self.free_handle(handle)?;
        self.insert(handle, Backing::Classic(desc), iov);
        Ok(())
    }

    pub fn resource_create_blob(
        &mut self,
        handle: ResourceHandle,
        desc: BlobDesc,
        iov: Vec<GuestIov>,
    ) -> Result<(), Error> {
        self.free_handle(handle)?;
        self.insert(handle, Backing::Blob(desc), iov);
        Ok(())
    }

    pub fn resource_import(
        &mut self,
        handle: ResourceHandle,
        blob_mem: u32,
        fd_type: u32,
        size: u64,
    ) -> Result<(), Error> {
        self.free_handle(handle)?;
        self.insert(handle, Backing::Imported { blob_mem, fd_type, size }, Vec::new());
        Ok(())
    }

    /// File a resource under a handle `free_handle` has already cleared.
    fn insert(&mut self, handle: ResourceHandle, backing: Backing, iov: Vec<GuestIov>) {
        let r = Resource { handle, backing, iov, priv_: VmmPtr::NULL, attached: Vec::new() };
        self.resources.insert(handle, r);
    }

    /// Check a guest-chosen resource handle before anything is inserted under it.
    fn free_handle(&self, handle: ResourceHandle) -> Result<(), Error> {
        if handle.0 == 0 {
            return Err(Error::ZeroHandle);
        }
        if self.resources.contains_key(&handle) {
            return Err(Error::ResourceExists);
        }
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

    pub fn context_create(&mut self, id: CtxId, flags: u32, name: String) -> Result<(), Error> {
        // A guest reusing a live id is the guest's error, not ours: rejected rather than
        // replacing an entry it still holds. Zero needs no check -- `CtxId` cannot be zero.
        if self.contexts.contains_key(&id) {
            return Err(Error::ContextExists);
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
    ) -> Result<(), Error> {
        let Some(c) = self.contexts.get_mut(&ctx) else {
            return Err(Error::NoContext);
        };
        c.last_fence.insert(ring, fence);
        // Nothing here submits GPU work yet, so every fence is already satisfied. It still goes
        // through the retirement thread: the asynchrony is the contract, not an optimization.
        self.fences.retire_context(ctx, ring, fence);
        Ok(())
    }

    pub fn create_fence(&mut self, fence: ClientFenceId) {
        self.fences.retire_global(fence);
    }

    // ---- venus ----

    /// Route a submission to the renderer the context bound. `Err` is a context that named no
    /// renderer we have, or a stream that poisoned the one it named.
    pub fn submit_cmd(&mut self, ctx: CtxId, buf: &[u8]) -> Result<(), Error> {
        let Some(c) = self.contexts.get(&ctx) else {
            return Err(Error::NoContext);
        };
        match c.flags & abi::CAPSET_MASK {
            abi::CAPSET_VENUS => {
                let v = self.venus.as_mut().ok_or(Error::RendererAbsent)?;
                v.submit(ctx, buf).map_err(|e| match e {
                    venus::vkr::Error::NoContext => Error::NoContext,
                    venus::vkr::Error::Poisoned => Error::Poisoned,
                })
            }
            // vrend arrives in P3.
            _ => Err(Error::RendererUnimplemented),
        }
    }

    /// The venus renderer, for the replay feed that drives it directly.
    pub fn venus_mut(&mut self) -> Option<&mut venus::vkr::Vkr> {
        self.venus.as_mut()
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.resources.len(), self.contexts.len())
    }

    /// The venus commands a run asked for and this build did not serve, most-used first.
    ///
    /// This is the implementation order, and it has to come from a corpus rather than from
    /// intuition: the commands a real desktop leans on are not the ones a reading of the Vulkan
    /// spec would rank first.
    pub fn venus_todo(&self) -> Vec<(&'static str, u64)> {
        self.venus.as_ref().map(|v| v.todo.by_frequency()).unwrap_or_default()
    }

    /// One venus context's live device memory, as (guest id, size).
    ///
    /// `None` when there is no such venus context, which the ABI reports as a failure -- a census
    /// of nothing and a census that could not be taken are different answers, and the VMM decides
    /// whether to snapshot on the difference.
    pub fn venus_memory_census(&self, ctx_id: CtxId) -> Option<Vec<(u64, u64)>> {
        Some(self.venus.as_ref()?.context(ctx_id)?.driver().memory_census())
    }

    /// Copy one allocation's contents out. False when the context, the id, or the mapping fails.
    pub fn venus_memory_read(&self, ctx_id: CtxId, mem_id: u64, buf: &mut [u8]) -> bool {
        let Some(ctx) = self.venus.as_ref().and_then(|v| v.context(ctx_id)) else {
            return false;
        };
        ctx.driver().memory_read(mem_id, buf)
    }
}

/// What this build cannot do for the flags it was given, in one phrase for the startup log.
///
/// venus decodes and dispatches everything but serves only part of it -- `dump_state` prints
/// which part -- and vrend does not exist at all. Saying which is which is the difference between
/// a log line that explains a failure and one that misleads about it.
pub fn unsupported_renderers(flags: c_int) -> &'static str {
    let venus = flags & abi::VENUS != 0;
    let vrend = flags & abi::NO_VIRGL == 0;
    match (venus, vrend) {
        (true, true) => "venus serves only part of the protocol; no vrend",
        (true, false) => "venus serves only part of the protocol",
        (false, true) => "no vrend",
        (false, false) => "no renderer asked for",
    }
}
