// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The renderer root: the resource table, the context table, and fence tracking.
//!
//! Everything the renderer owns hangs off one `Renderer` reached through the shim's single
//! `static`. There are no file-scope mutables and no implicit current context -- `force_ctx_0`,
//! the C's implicit global, is a no-op here because nothing reads such a thing.

use crate::abi::{GuestIov, VmmPtr};
use crate::config::{CapsetId, Config};
use crate::fence::{FenceSink, Retirement};
use crate::guest_mem::GuestMap;
use crate::ids::{BlobId, ClientFenceId, CtxId, FenceId, ResourceHandle, RingId, RingIdx};
use crate::venus;
use crate::venus::cs::ObjectId;
use crate::venus::driver::{Allocation, MemoryError};
use std::collections::BTreeMap;
use std::os::fd::{AsFd, OwnedFd};
#[cfg(test)]
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
use std::sync::{Arc, RwLock};

/// Why a call failed.
///
/// Named causes, not error codes: the C ABI answers in `errno`, and translating to it is the
/// shim's job -- see `ffi::errno`. A Rust caller gets to tell "that id is already live" from
/// "there is no renderer for that capset" without consulting a table of negative integers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
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
    /// Nothing is allocated under that id in that context.
    NoAllocation,
    /// The allocation exists but the driver would not map it; see [`MemoryError::NotMappable`].
    NotMappable,
    /// An import of zero bytes, which names no memory.
    ZeroSize,
    /// An shm descriptor the host could not map. A resource whose memory we cannot reach is one
    /// no ring can live in, so this fails at import rather than at the first command that needs
    /// it -- the guest gets the refusal while it is still holding the thing that caused it.
    Unmappable,
}

/// venus's own refusals in the renderer's vocabulary. One function, because every venus entry
/// point owes the same translation and a second copy is a second chance to disagree.
fn venus_error(e: venus::vkr::Error) -> Error {
    match e {
        venus::vkr::Error::NoContext => Error::NoContext,
        venus::vkr::Error::Poisoned => Error::Poisoned,
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Error::ResourceExists => "that resource handle is already live",
            Error::ContextExists => "that context id is already live",
            Error::NoContext => "no such context",
            Error::RendererAbsent => "this build was not initialized to serve that capset",
            Error::RendererUnimplemented => "no renderer serves that capset yet",
            Error::Poisoned => "the context is poisoned",
            Error::NoAllocation => "no such allocation in that context",
            Error::NotMappable => "that allocation cannot be mapped for reading",
            Error::ZeroSize => "an import of zero bytes names no memory",
            Error::Unmappable => "that shm descriptor could not be mapped",
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

/// Which memory an imported blob lives in.
///
/// Named, where [`BlobDesc`]'s `blob_mem` is not, because the import path has a defined
/// accept-set: the C refuses everything outside it, so there is a rejection here to check. The two
/// sets differ -- `BLOB_MEM_GUEST` and `BLOB_MEM_HOST3D_GUEST` are creatable but not importable --
/// which is exactly the distinction a bare `u32` in this position would lose.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlobMem {
    Host3d,
    GuestVram,
}

/// What kind of descriptor an import arrived on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FdType {
    DmaBuf,
    Opaque,
    Shm,
}

/// An imported blob as the VMM described it -- everything except the descriptor itself, which is
/// passed separately because passing it is a transfer of ownership and the type says so.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImportDesc {
    pub blob_mem: BlobMem,
    pub fd_type: FdType,
    pub size: u64,
}

/// A refused import, handing the descriptor back to the caller that still owns it.
///
/// An import that fails must not close the descriptor: the caller passed it expecting the transfer
/// to happen only on success, and a caller that then closes what we already closed is corrupting
/// whatever has since been opened under that number. Returning it in the error makes forgetting
/// impossible -- there is no path that yields an `Err` without also yielding the descriptor.
#[derive(Debug)]
pub struct Rejected {
    pub error: Error,
    pub fd: OwnedFd,
}

/// Memory this renderer minted for a blob, and the descriptor that names it.
///
/// The guest asks for a blob two different ways through one entry point, and `blob_id` is what
/// separates them: a non-zero id names something the renderer already has -- a `VkDeviceMemory` to
/// export -- while a zero id asks the renderer to supply the memory itself. Only the second kind
/// has a `HostShm`, and [`HostShm::for_blob`] is the single place that decides, so nothing else
/// has to keep the id and the memory in step.
///
/// The descriptor is kept rather than closed. The C hands it straight to the VMM in `out_blob` and
/// keeps only the mapping; there is no `out_blob` here yet, and closing it now would make the
/// export path impossible to add without re-plumbing this.
pub struct HostShm {
    pub fd: OwnedFd,
    pub map: Arc<GuestMap>,
}

impl HostShm {
    /// Mint memory for a blob, if this is the kind of blob that needs it.
    ///
    /// Mirrors the C's `vkr_context_get_blob`: `blob_id == 0` reaches
    /// `vkr_context_create_resource_from_shm`, everything else exports memory that already exists.
    fn for_blob(handle: ResourceHandle, desc: &BlobDesc) -> Result<Option<HostShm>, Error> {
        if desc.blob_mem != crate::abi::BLOB_MEM_HOST3D || desc.blob_id != BlobId(0) {
            return Ok(None);
        }
        let len = usize::try_from(desc.size).map_err(|_| Error::Unmappable)?;
        match crate::guest_mem::anonymous_shm(len, "virglrs-shmem") {
            Ok((fd, map)) => Ok(Some(HostShm { fd, map: Arc::new(map) })),
            Err(e) => {
                eprintln!("[virglrs] resource {handle}: cannot mint {len} shm bytes: {e}");
                Err(Error::Unmappable)
            }
        }
    }
}

/// What a resource is backed by. A resource is exactly one of these for its whole life; the C
/// keeps overlapping fields and a set of flags saying which are meaningful.
pub enum Backing {
    /// Created from `virgl_renderer_resource_create` -- a classic texture or buffer.
    Classic(ClassicDesc),
    /// Created from `virgl_renderer_resource_create_blob`.
    ///
    /// `host` is the memory this renderer minted for the blob, present exactly when the guest
    /// asked for one the host has to supply -- see [`HostShm`] and the one place that builds it.
    Blob { desc: BlobDesc, host: Option<HostShm> },
    /// Imported from a descriptor the VMM opened and handed over. The resource owns it now, and
    /// closing it is what dropping this does.
    ///
    /// `map` is `Some` exactly when `desc.fd_type` is [`FdType::Shm`], and it is established at
    /// the one place that builds this variant so no other code has to keep the two in step. Only
    /// shm is host-addressable: a dma-buf or an opaque handle names memory belonging to a driver,
    /// which the host reaches through Vulkan and never through a pointer.
    Imported { desc: ImportDesc, fd: OwnedFd, map: Option<Arc<GuestMap>> },
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

// The resource table is shared with every ring thread, so it must be safe to read from more than
// one at a time. Checked by the compiler rather than asserted in prose: a raw pointer or a `Cell`
// appearing anywhere under `Resource` names itself here instead of at the refactor that assumed it.
const _: () = {
    const fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<BTreeMap<ResourceHandle, Resource>>();
};

/// The resource table answering the only question venus asks of it.
///
/// On the table rather than on `Renderer`, because a venus submission needs the renderer's venus
/// state mutably and its resources shared at the same time. Those are sibling fields, so the
/// borrow is only disjoint if each is named separately -- a trait on the whole struct would make
/// every submission borrow all of it.
impl venus::ring::ShmResources for BTreeMap<ResourceHandle, Resource> {
    fn shm(&self, handle: ResourceHandle) -> Option<Arc<GuestMap>> {
        self.get(&handle)?.shm().map(Arc::clone)
    }
}

impl Resource {
    /// The host mapping of this resource, for the only backing that has one.
    ///
    /// Cloning the `Arc` is how a ring takes a share of it: the resource stops being the sole
    /// owner, so a guest that unrefs the resource while a ring still lives in it loses the handle
    /// and keeps the memory, instead of leaving the ring pointing at a freed mapping.
    pub fn shm(&self) -> Option<&Arc<GuestMap>> {
        match &self.backing {
            Backing::Imported { map, .. } => map.as_ref(),
            Backing::Blob { host, .. } => host.as_ref().map(|h| &h.map),
            Backing::Classic(_) => None,
        }
    }
}

pub struct Context {
    pub id: CtxId,
    /// The renderer this context bound. It is the only thing that says which one a submission
    /// belongs to, and it cannot change once the context exists.
    pub capset: CapsetId,
    pub name: String,
    /// The last fence id created on each ring. A ring's fences retire in creation order, so this
    /// is what a later phase checks a retirement against.
    pub last_fence: BTreeMap<RingIdx, FenceId>,
}

pub struct Renderer {
    pub config: Config,
    /// Shared with every ring thread, which reads it to resolve a reply stream's resource while
    /// the caller's thread may be creating another. A read-write lock rather than a mutex because
    /// that is the actual access pattern: many readers looking up a handle, one writer when the
    /// VMM creates or unrefs. See the lock order in `venus::vkr`.
    resources: Arc<RwLock<BTreeMap<ResourceHandle, Resource>>>,
    contexts: BTreeMap<CtxId, Context>,
    fences: Retirement,
    /// The venus renderer, present only when this build was initialized to serve it.
    venus: Option<venus::vkr::Vkr>,
}

impl Renderer {
    pub fn new(fences: Box<dyn FenceSink>, config: Config) -> Renderer {
        // Built here and shared into venus, rather than reached through the renderer: a ring
        // thread needs the table long after the call that created its ring returned, and it must
        // not need the renderer to get it.
        let resources: Arc<RwLock<BTreeMap<ResourceHandle, Resource>>> = Arc::default();
        Renderer {
            config,
            resources: Arc::clone(&resources),
            contexts: BTreeMap::new(),
            fences: Retirement::start(fences),
            venus: config.venus.then(|| venus::vkr::Vkr::new(config, resources.clone())),
        }
    }

    /// What this build advertises for a capset, or `None` for one it does not serve.
    ///
    /// Honest by construction: a capset appears only when the renderer that serves it is present.
    /// A skeleton that claimed VIRGL2 would have the guest bind a classic context and submit
    /// commands into a renderer that cannot answer them.
    ///
    /// The struct, not its bytes. A caller that wants to read a field reads a field, and one about
    /// to hand the image to a guest asks for the bytes at that moment -- which is the only point
    /// at which the layout matters, and is never here. The version a C caller names is checked at
    /// the shim, where a requested version is a thing that exists.
    pub fn capset(&self, set: CapsetId) -> Option<venus::capset::Capset> {
        match set {
            CapsetId::Venus => self.venus.as_ref().map(|_| venus::capset::Capset::new(self.config)),
            _ => None,
        }
    }

    /// The version and size a caller sizes its buffer from, for a capset this build serves.
    ///
    /// Asks [`Renderer::capset`] rather than testing a flag of its own. Advertising a capset and
    /// filling one were two answers to "does this build serve venus" -- one read the config, the
    /// other read whether the renderer existed -- and a VMM that sized a buffer from the first and
    /// got nothing from the second would hand its guest an uninitialised capset.
    pub fn capset_max(&self, set: CapsetId) -> Option<(u32, u32)> {
        self.capset(set).map(|_| (venus::capset::VERSION, venus::capset::size()))
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
        let host = HostShm::for_blob(handle, &desc)?;
        self.insert(handle, Backing::Blob { desc, host }, iov);
        Ok(())
    }

    /// Take over a descriptor the VMM opened.
    ///
    /// The descriptor is *taken*: once this returns `Ok`, closing it is ours to do, and
    /// [`Self::resource_unref`] does it by dropping the resource. Every refusal hands it back --
    /// see [`Rejected`].
    pub fn resource_import(
        &mut self,
        handle: ResourceHandle,
        desc: ImportDesc,
        fd: OwnedFd,
    ) -> Result<(), Rejected> {
        if let Err(error) = self.free_handle(handle) {
            return Err(Rejected { error, fd });
        }
        if desc.size == 0 {
            return Err(Rejected { error: Error::ZeroSize, fd });
        }
        // Map shm here, once, so that everything downstream holds a mapping rather than a
        // descriptor plus a promise to map it later. A ring created from this resource clones the
        // `Arc`, which is what lets the ring keep working after the resource is unref'd -- the C
        // keeps a bare `const struct vkr_resource *` in the ring and is a use-after-free waiting
        // on the guest to get the order wrong.
        let map = match desc.fd_type {
            FdType::Shm => {
                let len = match usize::try_from(desc.size) {
                    Ok(len) => len,
                    Err(_) => return Err(Rejected { error: Error::Unmappable, fd }),
                };
                match GuestMap::shm(fd.as_fd(), len) {
                    Ok(m) => Some(Arc::new(m)),
                    Err(e) => {
                        eprintln!("[virglrs] resource {handle}: cannot map {len} shm bytes: {e}");
                        return Err(Rejected { error: Error::Unmappable, fd });
                    }
                }
            }
            FdType::DmaBuf | FdType::Opaque => None,
        };
        self.insert(handle, Backing::Imported { desc, fd, map }, Vec::new());
        Ok(())
    }

    /// File a resource under a handle `free_handle` has already cleared.
    fn insert(&mut self, handle: ResourceHandle, backing: Backing, iov: Vec<GuestIov>) {
        let r = Resource { handle, backing, iov, priv_: VmmPtr::NULL, attached: Vec::new() };
        self.resources.write().expect("the resource lock is never poisoned").insert(handle, r);
    }

    /// Check a guest-chosen resource handle before anything is inserted under it.
    ///
    /// Zero is not among the answers: [`ResourceHandle`] cannot hold one, so the shim that parsed
    /// the guest's integer has already refused it.
    fn free_handle(&self, handle: ResourceHandle) -> Result<(), Error> {
        if self.resources.read().expect("the resource lock is never poisoned").contains_key(&handle)
        {
            return Err(Error::ResourceExists);
        }
        Ok(())
    }

    /// Look at one resource, for as long as the closure runs and no longer.
    ///
    /// The table is shared, so a `&Resource` cannot outlive the lock that made it safe to read;
    /// scoping it to a closure is what says so in the type. `None` is a handle that names nothing,
    /// which stays distinguishable from a closure that returned nothing.
    pub fn with_resource<R>(
        &self,
        handle: ResourceHandle,
        f: impl FnOnce(&Resource) -> R,
    ) -> Option<R> {
        self.resources.read().expect("the resource lock is never poisoned").get(&handle).map(f)
    }

    /// Change one resource, under the same scoping and for the same reason.
    ///
    /// `&self`, because the lock is what grants the mutation -- not an exclusive borrow of the
    /// renderer, which the ring threads make impossible to hand out anyway.
    pub fn with_resource_mut<R>(
        &self,
        handle: ResourceHandle,
        f: impl FnOnce(&mut Resource) -> R,
    ) -> Option<R> {
        self.resources.write().expect("the resource lock is never poisoned").get_mut(&handle).map(f)
    }

    pub fn resource_unref(&mut self, handle: ResourceHandle) {
        // Detach from every context first. A context holding a dangling handle is how the C's
        // use-after-free reached the command stream.
        let mut resources = self.resources.write().expect("the resource lock is never poisoned");
        if let Some(r) = resources.get_mut(&handle) {
            r.attached.clear();
        }
        resources.remove(&handle);
    }

    // ---- contexts ----

    pub fn context_create(
        &mut self,
        id: CtxId,
        capset: CapsetId,
        name: String,
    ) -> Result<(), Error> {
        // A guest reusing a live id is the guest's error, not ours: rejected rather than
        // replacing an entry it still holds. Zero needs no check -- `CtxId` cannot be zero.
        if self.contexts.contains_key(&id) {
            return Err(Error::ContextExists);
        }
        self.contexts.insert(id, Context { id, capset, name, last_fence: BTreeMap::new() });
        // A venus context gets venus state; anything else gets a context and nothing behind it,
        // and finds out when it submits.
        if capset == CapsetId::Venus
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
        for r in self.resources.write().expect("the resource lock is never poisoned").values_mut() {
            r.attached.retain(|c| *c != id);
        }
    }

    pub fn context(&self, id: CtxId) -> Option<&Context> {
        self.contexts.get(&id)
    }

    pub fn ctx_attach_resource(&mut self, ctx: CtxId, handle: ResourceHandle) {
        if !self.contexts.contains_key(&ctx) {
            return;
        }
        self.with_resource_mut(handle, |r| {
            if !r.attached.contains(&ctx) {
                r.attached.push(ctx);
            }
        });
    }

    pub fn ctx_detach_resource(&mut self, ctx: CtxId, handle: ResourceHandle) {
        self.with_resource_mut(handle, |r| r.attached.retain(|c| *c != ctx));
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
        match c.capset {
            CapsetId::Venus => self.venus_mut()?.submit(ctx, buf).map_err(venus_error),
            // vrend arrives in P3.
            _ => Err(Error::RendererUnimplemented),
        }
    }

    /// The venus renderer, or the error a caller gets when this build has none.
    ///
    /// Venus takes the resource lock itself, inside `on_context`, because it is the side that
    /// knows the order the ring threads have to agree with. Taking it here as well would be the
    /// same thread reading twice, which a writer arriving in between is allowed to deadlock.
    fn venus_mut(&mut self) -> Result<&mut venus::vkr::Vkr, Error> {
        self.venus.as_mut().ok_or(Error::RendererAbsent)
    }

    /// Feed one replay journal entry to a context's default stream.
    pub fn venus_replay_cmd(&mut self, ctx: CtxId, buf: &[u8]) -> Result<(), Error> {
        self.venus_mut()?.submit(ctx, buf).map_err(venus_error)
    }

    /// Feed one replay journal entry to a named ring's stream.
    pub fn venus_replay_ring_cmd(
        &mut self,
        ctx: CtxId,
        ring: RingId,
        buf: &[u8],
    ) -> Result<(), Error> {
        self.venus_mut()?.submit_ring(ctx, ring, buf).map_err(venus_error)
    }

    pub fn venus_replay_begin(&mut self, ctx: CtxId) -> Result<(), Error> {
        self.venus.as_mut().ok_or(Error::RendererAbsent)?.replay_begin(ctx).map_err(venus_error)
    }

    pub fn venus_replay_end(&mut self, ctx: CtxId) -> Result<(), Error> {
        self.venus.as_mut().ok_or(Error::RendererAbsent)?.replay_end(ctx).map_err(venus_error)
    }

    pub fn counts(&self) -> (usize, usize) {
        (
            self.resources.read().expect("the resource lock is never poisoned").len(),
            self.contexts.len(),
        )
    }

    /// The venus commands a run asked for and this build did not serve, most-used first.
    ///
    /// This is the implementation order, and it has to come from a corpus rather than from
    /// intuition: the commands a real desktop leans on are not the ones a reading of the Vulkan
    /// spec would rank first.
    pub fn venus_todo(&self) -> Vec<(&'static str, u64)> {
        self.venus
            .as_ref()
            .map(|v| v.todo.lock().expect("the census lock is never poisoned").by_frequency())
            .unwrap_or_default()
    }

    /// One venus context's live device memory.
    ///
    /// An empty census and a census that could not be taken are different answers, and the VMM
    /// decides whether to snapshot on the difference -- so the failure says which it was.
    pub fn venus_memory_census(&self, ctx_id: CtxId) -> Result<Vec<Allocation>, Error> {
        self.venus_context(ctx_id, |ctx| ctx.driver().memory_census())
    }

    /// Copy one allocation's contents out, returning how many bytes landed in `buf`.
    pub fn venus_memory_read(
        &self,
        ctx_id: CtxId,
        mem_id: u64,
        buf: &mut [u8],
    ) -> Result<usize, Error> {
        self.venus_context(ctx_id, |ctx| ctx.memory_read(ObjectId(mem_id), buf))?.map_err(|e| {
            match e {
                MemoryError::NoSuchAllocation => Error::NoAllocation,
                MemoryError::NotMappable => Error::NotMappable,
            }
        })
    }

    /// The venus context under an id, distinguishing "no venus in this build" from "no such
    /// context" -- which a caller asking for a snapshot needs to tell apart.
    fn venus_context<R>(
        &self,
        ctx_id: CtxId,
        f: impl FnOnce(&venus::context::Context) -> R,
    ) -> Result<R, Error> {
        let v = self.venus.as_ref().ok_or(Error::RendererAbsent)?;
        v.with_context(ctx_id, f).ok_or(Error::NoContext)
    }
}

/// What this build cannot do for the configuration it was given, in one phrase for the startup
/// log.
///
/// venus decodes and dispatches everything but serves only part of it -- `dump_state` prints
/// which part -- and vrend does not exist at all. Saying which is which is the difference between
/// a log line that explains a failure and one that misleads about it.
pub fn unsupported_renderers(config: Config) -> &'static str {
    match (config.venus, config.vrend) {
        (true, true) => "venus serves only part of the protocol; no vrend",
        (true, false) => "venus serves only part of the protocol",
        (false, true) => "no vrend",
        (false, false) => "no renderer asked for",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sink that goes nowhere. Nothing here retires a fence; the renderer needs one to exist.
    struct NoSink;
    impl FenceSink for NoSink {
        fn context_fence(&mut self, _ctx: CtxId, _ring: RingIdx, _fence: FenceId) {}
        fn global_fence(&mut self, _fence: ClientFenceId) {}
    }

    fn renderer(config: Config) -> Renderer {
        Renderer::new(Box::new(NoSink), config)
    }

    /// The two questions `HostShm::for_blob` answers, and it answers them from `blob_id` alone.
    ///
    /// One entry point serves two different guest requests: a zero id asks the renderer to supply
    /// memory, any other id names memory it already has and asks for it to be exported. Minting
    /// for the second kind would hand the guest fresh zeroed pages where it expected the contents
    /// of a `VkDeviceMemory` -- a wrong answer that looks like a working one.
    #[test]
    fn only_a_blob_that_asks_the_host_for_memory_is_given_any() {
        let mut r = renderer(Config::default());

        let minted = BlobDesc {
            blob_mem: crate::abi::BLOB_MEM_HOST3D,
            blob_flags: 1,
            blob_id: BlobId(0),
            // The size the venus corpus asks for a ring resource: not a whole number of pages.
            size: 0x24000 - 1,
        };
        r.resource_create_blob(ResourceHandle::new(1).unwrap(), minted, Vec::new())
            .expect("created");
        let map = r
            .with_resource(ResourceHandle::new(1).unwrap(), |res| res.shm().cloned())
            .expect("there")
            .expect("has memory");

        // Page-rounded, because the VMM maps with MAP_FIXED and because the ring layout is
        // validated against this length -- un-rounded, a layout the C accepts would be refused.
        let page = crate::guest_mem::page_size();
        assert_eq!(map.len() % page, 0, "the mapping is a whole number of pages");
        assert!(map.len() >= 0x24000 - 1, "and covers everything that was asked for");

        // A blob naming memory that already exists gets none of its own.
        let exported = BlobDesc { blob_id: BlobId(9), ..minted };
        r.resource_create_blob(ResourceHandle::new(2).unwrap(), exported, Vec::new())
            .expect("created");
        assert!(
            r.with_resource(ResourceHandle::new(2).unwrap(), |res| res.shm().cloned())
                .expect("there")
                .is_none(),
            "an export names memory the renderer already has; minting would answer with the wrong bytes"
        );

        // And so does a blob in memory that is not the host's to mint.
        let vram = BlobDesc { blob_mem: crate::abi::BLOB_MEM_GUEST_VRAM, ..minted };
        r.resource_create_blob(ResourceHandle::new(3).unwrap(), vram, Vec::new()).expect("created");
        assert!(
            r.with_resource(ResourceHandle::new(3).unwrap(), |res| res.shm().cloned())
                .expect("there")
                .is_none()
        );
    }

    /// A real descriptor to hand over. Any would do; a pipe's read end is the cheapest.
    fn a_descriptor() -> OwnedFd {
        let mut fds = [0 as std::ffi::c_int; 2];
        // SAFETY: `pipe` fills two ints at the pointer it is given; `fds` is exactly that.
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0, "the test needs a pipe");
        // SAFETY: `pipe` succeeded, so both are open descriptors this test owns.
        unsafe {
            libc::close(fds[1]);
            OwnedFd::from_raw_fd(fds[0])
        }
    }

    fn is_open(fd: std::os::fd::RawFd) -> bool {
        // SAFETY: `F_GETFD` only reads the flags of whatever is under `fd`, or fails if nothing is.
        unsafe { libc::fcntl(fd, libc::F_GETFD) != -1 }
    }

    fn import_desc(size: u64) -> ImportDesc {
        ImportDesc { blob_mem: BlobMem::Host3d, fd_type: FdType::DmaBuf, size }
    }

    /// An imported descriptor is closed when the resource holding it goes.
    ///
    /// The shim used to read `res_handle`, `blob_mem`, `fd_type` and `size` out of the import
    /// arguments and never touch `fd` at all, so every import leaked a descriptor for the life of
    /// the process -- a VMM importing per frame runs out of them.
    #[test]
    fn unref_closes_the_descriptor_the_import_took() {
        let fd = a_descriptor();
        let raw = fd.as_raw_fd();
        let mut r = renderer(Config::default());
        let h = ResourceHandle::new(1).unwrap();
        r.resource_import(h, import_desc(4096), fd).expect("a fresh handle and a real size");
        assert!(is_open(raw), "the resource holds the descriptor while it lives");
        r.resource_unref(h);
        assert!(!is_open(raw), "dropping the resource closes it");
    }

    /// A refused import gives the descriptor back, still open.
    ///
    /// The C rejects a zero-size import before it takes the fd, so a caller whose import was
    /// refused still owns it and will close it itself. Closing it here as well would leave that
    /// caller closing a number something else has since been opened under.
    #[test]
    fn a_refused_import_hands_the_descriptor_back() {
        let fd = a_descriptor();
        let raw = fd.as_raw_fd();
        let mut r = renderer(Config::default());
        let rej = r
            .resource_import(ResourceHandle::new(1).unwrap(), import_desc(0), fd)
            .expect_err("zero bytes names no memory");
        assert_eq!(rej.error, Error::ZeroSize);
        assert_eq!(rej.fd.into_raw_fd(), raw, "the same descriptor, not another");
        assert!(is_open(raw), "and it was not closed on the way out");
        // SAFETY: `into_raw_fd` above gave up ownership without closing; this test is what owns it
        // now, and closes it here.
        unsafe { libc::close(raw) };
    }

    /// A capset is advertised and filled by the same predicate.
    ///
    /// They were two: the size a VMM sizes its buffer from asked whether the venus renderer
    /// existed, and the fill asked whether the config had venus set. They agree today only
    /// because one is built from the other at construction -- and the day they stopped agreeing,
    /// a VMM would size a buffer from the first and get nothing back from the second, handing its
    /// guest whatever was already in that memory as a capset.
    #[test]
    fn a_capset_is_advertised_by_whatever_would_fill_it() {
        let venus = Config { venus: true, ..Config::default() };
        let r = renderer(venus);
        assert!(r.capset(CapsetId::Venus).is_some(), "this build serves venus");
        assert_eq!(
            r.capset_max(CapsetId::Venus),
            Some((venus::capset::VERSION, venus::capset::size())),
            "and says so at exactly the size the fill will write"
        );

        // Nothing else is served, by either answer.
        for set in [CapsetId::Virgl, CapsetId::Virgl2, CapsetId::from_raw(9)] {
            assert!(r.capset(set).is_none(), "{set:?} has no renderer behind it");
            assert!(r.capset_max(set).is_none(), "{set:?} must not be advertised either");
        }

        // And a build without venus advertises nothing at all, rather than a size it cannot fill.
        let bare = renderer(Config::default());
        assert!(bare.capset(CapsetId::Venus).is_none());
        assert!(bare.capset_max(CapsetId::Venus).is_none());
    }

    /// What the guest reads is the configuration it was given, through the struct the Rust API
    /// hands over -- no serialization in between, which is the point of handing over the struct.
    #[test]
    fn the_capset_a_caller_receives_carries_the_configuration() {
        let on = Config { venus: true, guest_vram: true, ..Config::default() };
        let c = renderer(on).capset(CapsetId::Venus).expect("venus is served");
        assert_eq!(c.use_guest_vram, 1, "a field, read as a field");
        assert_eq!(c.as_bytes().len(), venus::capset::size() as usize);
    }
}
