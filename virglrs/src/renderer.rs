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
use crate::guest_mem::{GuestMap, Iov};
use crate::ids::{
    BlobId, ClientFenceId, CtxId, FenceId, ResourceHandle, RingId, RingIdx, SurfaceId,
};
use crate::venus;
use crate::venus::context::Submitted;
use crate::venus::cs::ObjectId;
use crate::venus::driver::{Allocation, Exported, MemoryError, Storage};
use crate::venus::objects::ObjectKey;
use crate::venus::ring::{Published, ResourceBytes};
use crate::vrend;
use crate::vrend::context::Guest;
use crate::vrend::resource::{Args as ClassicArgs, Refusal};
use crate::vrend::transfer;
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
    /// No resource under that handle.
    NoResource,
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
    /// A submission waited on a ring that is not running in that context.
    NoRing,
    /// Nothing is allocated under that id in that context.
    NoAllocation,
    /// The allocation exists but the driver would not map it; see [`MemoryError::NotMappable`].
    NotMappable,
    /// The allocation is already published as some other resource's blob. A memory backs one
    /// blob: two resources over one storage is a state neither of them could detect.
    AlreadyExported,
    /// The host cannot address the allocation, so there is nothing to publish into the guest.
    NotHostVisible,
    /// The blob is larger than the allocation behind it. Mapping it would publish whatever
    /// follows the allocation in this process.
    BlobLargerThanAllocation,
    /// An import of zero bytes, which names no memory.
    ZeroSize,
    /// An shm descriptor the host could not map. A resource whose memory we cannot reach is one
    /// no ring can live in, so this fails at import rather than at the first command that needs
    /// it -- the guest gets the refusal while it is still holding the thing that caused it.
    Unmappable,
    /// vrend refused to create a classic resource, for the reason given.
    ClassicRefused(Refusal),
    /// A classic transfer did not happen, for the reason given.
    Transfer(transfer::Error),
}

/// An export's refusals in the renderer's vocabulary, for the reason [`venus_error`] gives.
fn export_error(e: venus::driver::ExportError) -> Error {
    use venus::driver::ExportError as E;
    match e {
        E::NoSuchAllocation => Error::NoAllocation,
        E::AlreadyExported => Error::AlreadyExported,
        E::NotHostVisible => Error::NotHostVisible,
        E::LargerThanAllocation => Error::BlobLargerThanAllocation,
        E::NotMappable => Error::NotMappable,
    }
}

/// venus's own refusals in the renderer's vocabulary. One function, because every venus entry
/// point owes the same translation and a second copy is a second chance to disagree.
fn venus_error(e: venus::vkr::Error) -> Error {
    match e {
        venus::vkr::Error::NoContext => Error::NoContext,
        venus::vkr::Error::NoRing => Error::NoRing,
        venus::vkr::Error::Poisoned => Error::Poisoned,
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Error::ResourceExists => "that resource handle is already live",
            Error::NoResource => "no such resource",
            Error::ContextExists => "that context id is already live",
            Error::NoContext => "no such context",
            Error::RendererAbsent => "this build was not initialized to serve that capset",
            Error::RendererUnimplemented => "no renderer serves that capset yet",
            Error::Poisoned => "the context is poisoned",
            Error::NoRing => "no such running ring in that context",
            Error::NoAllocation => "no such allocation in that context",
            Error::NotMappable => "that allocation cannot be mapped for reading",
            Error::AlreadyExported => "that allocation is already published as a blob",
            Error::NotHostVisible => "that allocation is not addressable by the host",
            Error::BlobLargerThanAllocation => "the blob is larger than the allocation behind it",
            Error::ZeroSize => "an import of zero bytes names no memory",
            Error::Unmappable => "that shm descriptor could not be mapped",
            Error::ClassicRefused(r) => return write!(f, "vrend refused the resource: {r}"),
            Error::Transfer(e) => return write!(f, "the transfer failed: {e}"),
        };
        f.write_str(s)
    }
}

impl std::error::Error for Error {}

/// Where a blob's storage comes from.
///
/// The ABI discriminates on `blob_id == 0`, one magic number standing between two operations that
/// have nothing in common: one asks this renderer to supply memory, the other names memory a venus
/// context already holds and asks for it to be published. Naming them separates the two, and
/// carries with the export the context whose table the id means something in -- without which the
/// id names nothing at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlobSource {
    /// The guest asks the host for memory it does not yet have.
    HostMinted,
    /// A venus context publishes device memory it already holds.
    Exported { ctx: CtxId, mem: BlobId },
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
    pub source: BlobSource,
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
/// Only a [`BlobSource::HostMinted`] blob has one; an export names memory that already exists, and
/// minting for it would hand the guest fresh zeroed pages where it expected a `VkDeviceMemory`'s
/// contents.
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
    /// Mirrors the C's `vkr_context_get_blob`: a host-minted blob reaches
    /// `vkr_context_create_resource_from_shm`, an export publishes memory that already exists.
    fn for_blob(handle: ResourceHandle, desc: &BlobDesc) -> Result<Option<HostShm>, Error> {
        if desc.blob_mem != crate::abi::BLOB_MEM_HOST3D
            || !matches!(desc.source, BlobSource::HostMinted)
        {
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

/// What a blob resource's bytes are. Exactly one of these, decided when the blob is created and
/// never revisited: every path that asks about a blob's bytes -- who may reach them, where they
/// are, whether they are a surface -- is one match on this.
pub enum BlobStorage {
    /// The guest's own pages, described by the resource's iov. Nothing host-side to publish.
    Guest,
    /// Memory this renderer minted for the blob, which the guest asked the host to supply.
    Minted(HostShm),
    /// A share of storage a venus allocation was published from -- a surface, or pages this
    /// renderer minted. Holding it is what makes the resource resolvable from any context the
    /// guest attached it to, and what keeps the bytes alive for as long as the resource stands,
    /// including after the context that minted them is gone. `caching` is how the exporter's
    /// memory type said the host reaches them, decided at the export.
    Shared { storage: Storage, caching: Caching },
    /// The exporting allocation's own `vkMapMemory` pointer, resolved once at the export and
    /// kept -- what the C keeps too. Good only in the context that mapped it, and only while
    /// that allocation stands: `key` is what says whether it still does, and it says so about
    /// the *object*, not the id, which the guest may have given to something else since.
    Borrowed { ctx: CtxId, mem: BlobId, key: ObjectKey, mapping: HostMapping },
}

/// How a guest may cache memory the host published to it.
///
/// Decided by how the *host* reaches the same bytes: the guest's mapping has to be no weaker.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Caching {
    /// Write-back, coherent without explicit flushes.
    Cached,
    /// Write-combining: the guest must not read back through the cache.
    WriteCombining,
}

/// A blob resource's home in this process, as a VMM needs it to publish the blob to its guest.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HostMapping {
    /// Where it starts.
    pub addr: usize,
    /// How far it runs. Not the blob's requested size but the mapping's own extent, taken from
    /// whatever owns the memory -- which is the only thing that knows how much of it there is.
    pub size: u64,
    pub caching: Caching,
}

/// What a resource is backed by. A resource is exactly one of these for its whole life; the C
/// keeps overlapping fields and a set of flags saying which are meaningful.
pub enum Backing {
    /// Created from `virgl_renderer_resource_create` -- a classic texture or buffer. Its host
    /// side, when this build serves vrend, is vrend's under the same handle.
    Classic(ClassicArgs),
    /// Created from `virgl_renderer_resource_create_blob`. `desc` is what the guest asked for;
    /// `storage` is what it got, settled once at the create -- see [`BlobStorage`].
    Blob { desc: BlobDesc, storage: BlobStorage },
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

    fn bytes(&self, ctx: CtxId, handle: ResourceHandle) -> Option<ResourceBytes> {
        let Some(res) = self.get(&handle) else {
            eprintln!("[virglrs] ctx {}: resource {handle:?} is not in the table", ctx.get());
            return None;
        };
        // What the guest kernel attached, which is the decision virtio-gpu already made about
        // who may reach this resource. Gating on it delegates that decision rather than inventing
        // a second one; gating on who *created* the resource invents a rule the protocol has not
        // got, and a compositor importing a client's buffer trips over it. Every kind of backing
        // is behind it, a ring's shared memory included: a ring is the one resource a context
        // reads *from*, and the one it must least be able to reach by guessing a handle.
        if !res.attached.contains(&ctx) {
            eprintln!(
                "[virglrs] ctx {}: resource {handle:?} is not attached to this context",
                ctx.get(),
            );
            return None;
        }
        if let Some(map) = res.shm() {
            return Some(ResourceBytes::Host(Arc::clone(map)));
        }
        match &res.backing {
            Backing::Blob { desc, storage } => match storage {
                // A share resolves for anyone holding it: no table is asked.
                BlobStorage::Shared { storage, .. } => Some(ResourceBytes::Shared(storage.clone())),
                BlobStorage::Borrowed { ctx: owner, mem, .. } if *owner == ctx => {
                    Some(ResourceBytes::Allocation(Published {
                        memory: ObjectId(mem.0),
                        size: desc.size,
                    }))
                }
                // Named by the wrong context. Not a mistake the guest made in this command: it is
                // one context reaching for another's export, which the ids cannot express.
                // Attached, so the guest is entitled to it -- but its storage is a borrowed
                // `vkMapMemory` pointer into ctx `owner`'s device, and this renderer cannot keep
                // that alive for anyone else. Refused loudly rather than shared: handing it over
                // would dangle the moment `owner` freed the memory or went away.
                BlobStorage::Borrowed { ctx: owner, .. } => {
                    eprintln!(
                        "[virglrs] ctx {}: resource {handle:?} is ctx {}'s export, and ordinary \
                         device memory is published as a borrowed mapping this renderer cannot \
                         share across contexts",
                        ctx.get(),
                        owner.get(),
                    );
                    None
                }
                BlobStorage::Minted(_) => unreachable!("answered as shared memory above"),
                BlobStorage::Guest => {
                    eprintln!(
                        "[virglrs] ctx {}: resource {handle:?} is a blob with no host storage",
                        ctx.get(),
                    );
                    None
                }
            },
            Backing::Classic(_) | Backing::Imported { .. } => {
                eprintln!(
                    "[virglrs] ctx {}: resource {handle:?} is not host-addressable",
                    ctx.get(),
                );
                None
            }
        }
    }
}

/// The resource table answering what vrend asks of the guest side: whether a context may reach
/// a resource, and the pages behind it. Attachment is the gate here for the reason it is in
/// `ShmResources`: it is the decision virtio-gpu already made.
impl Guest for BTreeMap<ResourceHandle, Resource> {
    fn attached(&self, ctx: CtxId, handle: ResourceHandle) -> bool {
        self.get(&handle).is_some_and(|r| r.attached.contains(&ctx))
    }

    fn pages(&self, ctx: CtxId, handle: ResourceHandle) -> Option<Iov<'_>> {
        let r = self.get(&handle)?;
        if !r.attached.contains(&ctx) || r.iov.is_empty() {
            return None;
        }
        Some(Iov::new(&r.iov))
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
            Backing::Blob { storage: BlobStorage::Minted(h), .. } => Some(&h.map),
            Backing::Blob { .. } | Backing::Classic(_) => None,
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

/// One capset, as the guest reads it: which one decides both its layout and the version a
/// caller may ask for.
// A capset is built once per request and copied straight out to the caller's buffer; the size
// difference between the sets is the sets' own, not an indirection worth adding.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug)]
pub enum Capset {
    Venus(venus::capset::Capset),
    Virgl(vrend::caps::CapsV1),
    Virgl2(vrend::caps::CapsV2),
}

impl Capset {
    /// The newest version of this capset the build fills: what `get_cap_set` reports, and the
    /// most a `fill_caps` may ask for.
    pub fn version(&self) -> u32 {
        match self {
            Capset::Venus(_) => venus::capset::VERSION,
            Capset::Virgl(_) => vrend::caps::VIRGL_VERSION,
            Capset::Virgl2(_) => vrend::caps::VIRGL2_VERSION,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Capset::Venus(c) => c.as_bytes(),
            Capset::Virgl(c) => c.as_bytes(),
            Capset::Virgl2(c) => c.as_bytes(),
        }
    }
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
    /// The classic renderer, likewise.
    vrend: Option<vrend::vrend::Vrend>,
}

impl Renderer {
    /// Bring up the renderers the config asks for. vrend needs a GL context on the calling
    /// thread, and a host that cannot give it one is a host this renderer cannot serve the
    /// classic protocol on -- so that is a failure to initialise, as it is in the C.
    pub fn new(
        fences: Box<dyn FenceSink>,
        config: Config,
    ) -> Result<Renderer, vrend::vrend::InitError> {
        // Built here and shared into venus, rather than reached through the renderer: a ring
        // thread needs the table long after the call that created its ring returned, and it must
        // not need the renderer to get it.
        let resources: Arc<RwLock<BTreeMap<ResourceHandle, Resource>>> = Arc::default();
        let vrend = if config.vrend { Some(vrend::vrend::Vrend::new()?) } else { None };
        Ok(Renderer {
            config,
            resources: Arc::clone(&resources),
            contexts: BTreeMap::new(),
            fences: Retirement::start(fences),
            venus: config.venus.then(|| venus::vkr::Vkr::new(config, resources.clone())),
            vrend,
        })
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
    pub fn capset(&self, set: CapsetId) -> Option<Capset> {
        match set {
            CapsetId::Venus => {
                self.venus.as_ref().map(|_| Capset::Venus(venus::capset::Capset::new(self.config)))
            }
            CapsetId::Virgl => self.vrend.as_ref().map(|v| Capset::Virgl(v.caps().v1())),
            CapsetId::Virgl2 => self.vrend.as_ref().map(|v| Capset::Virgl2(*v.caps())),
            CapsetId::Unknown(_) => None,
        }
    }

    /// The version and size a caller sizes its buffer from, for a capset this build serves.
    ///
    /// Asks [`Renderer::capset`] rather than testing a flag of its own. Advertising a capset and
    /// filling one were two answers to "does this build serve venus" -- one read the config, the
    /// other read whether the renderer existed -- and a VMM that sized a buffer from the first and
    /// got nothing from the second would hand its guest an uninitialised capset.
    pub fn capset_max(&self, set: CapsetId) -> Option<(u32, u32)> {
        self.capset(set).map(|c| (c.version(), c.as_bytes().len() as u32))
    }

    // ---- resources ----

    pub fn resource_create(
        &mut self,
        handle: ResourceHandle,
        args: ClassicArgs,
        iov: Vec<GuestIov>,
    ) -> Result<(), Error> {
        // A guest-chosen handle that is already live is the guest's error, not ours: reject it
        // rather than replacing an entry something else still holds.
        self.free_handle(handle)?;
        // vrend's half first, because it is the half that can refuse; without vrend the resource
        // is a table entry and nothing more, which is all a venus-only build owes it.
        if let Some(v) = self.vrend.as_mut() {
            v.resource_create(handle, args).map_err(Error::ClassicRefused)?;
        }
        self.insert(handle, Backing::Classic(args), iov);
        Ok(())
    }

    /// Create a resource backed by a blob: memory this renderer mints, or memory a venus context
    /// already holds and is publishing.
    ///
    /// The export happens here, before the resource exists, because it is the half that can fail
    /// on the guest's account -- naming memory it never allocated, exporting the same memory
    /// twice, or asking for a blob bigger than what backs it. Failing before the insert is what
    /// leaves nothing behind to clean up.
    pub fn resource_create_blob(
        &mut self,
        handle: ResourceHandle,
        desc: BlobDesc,
        iov: Vec<GuestIov>,
    ) -> Result<(), Error> {
        self.free_handle(handle)?;
        let storage = match desc.source {
            BlobSource::HostMinted => match HostShm::for_blob(handle, &desc)? {
                Some(shm) => BlobStorage::Minted(shm),
                None => BlobStorage::Guest,
            },
            BlobSource::Exported { ctx, mem } => {
                match self.venus_memory_export(ctx, mem, desc.size) {
                    Ok((exported, share, key)) => {
                        let caching = if exported.write_back {
                            Caching::Cached
                        } else {
                            Caching::WriteCombining
                        };
                        match share {
                            Some(storage) => BlobStorage::Shared { storage, caching },
                            // The VMM publishes the *blob's* size from this address, which the
                            // export held to the allocation's.
                            None => BlobStorage::Borrowed {
                                ctx,
                                mem,
                                key,
                                mapping: HostMapping {
                                    addr: exported.addr,
                                    size: desc.size,
                                    caching,
                                },
                            },
                        }
                    }
                    Err(e) => {
                        // Two failures, one errno at the ABI, and they want opposite
                        // investigations. The memory not being there says the command that would
                        // have allocated it never reached us -- the transport is what to look at,
                        // and the allocation is innocent. The memory being there and the export
                        // refusing it says the opposite. The guest kernel treats CREATE_BLOB as
                        // fire-and-forget, so this line is the only account anyone gets of either;
                        // one line covering both sends the next reader to the wrong half.
                        let half = match e {
                            Error::NoAllocation | Error::NoContext => "no such allocation",
                            _ => "the allocation is there, and the export of it refused",
                        };
                        eprintln!(
                            "[virglrs] resource {handle}: CREATE_BLOB of ctx {ctx} memory {mem}, \
                         {} bytes: {half}: {e}",
                            desc.size,
                        );
                        return Err(e);
                    }
                }
            }
        };
        self.insert(handle, Backing::Blob { desc, storage }, iov);
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
        if let Some(v) = self.vrend.as_mut() {
            v.resource_destroy(handle);
        }
        // Detach from every context first. A context holding a dangling handle is how the C's
        // use-after-free reached the command stream.
        let mut resources = self.resources.write().expect("the resource lock is never poisoned");
        if let Some(r) = resources.get_mut(&handle) {
            r.attached.clear();
        }
        resources.remove(&handle);
    }

    /// The VMM attached guest pages to a resource. `Err` if it has none, or already has some:
    /// the C refuses a second attach, and a VMM that attaches twice has lost track of what it
    /// gave.
    pub fn resource_attach_iov(
        &mut self,
        handle: ResourceHandle,
        iov: Vec<GuestIov>,
    ) -> Result<(), Error> {
        let already = self.with_resource(handle, |r| !r.iov.is_empty()).ok_or(Error::NoResource)?;
        if already {
            return Err(Error::ResourceExists);
        }
        if let Some(v) = self.vrend.as_mut() {
            v.resource_attached(handle, &Iov::new(&iov));
        }
        self.with_resource_mut(handle, |r| r.iov = iov);
        Ok(())
    }

    /// The VMM is taking a resource's pages back; how many entries it had.
    pub fn resource_detach_iov(&mut self, handle: ResourceHandle) -> usize {
        let iov =
            self.with_resource_mut(handle, |r| std::mem::take(&mut r.iov)).unwrap_or_default();
        if let Some(v) = self.vrend.as_mut() {
            v.resource_detaching(handle, &Iov::new(&iov));
        }
        iov.len()
    }

    /// A classic transfer between a resource and guest pages, on the API path: the resource's
    /// own pages when `iov` is empty.
    ///
    /// `ctx` names the context the transfer runs on, or the VMM's own (ctx 0) when `None`. A
    /// named context must exist and must have the resource attached, as in the C's per-context
    /// lookup.
    pub fn transfer(
        &mut self,
        handle: ResourceHandle,
        ctx: Option<CtxId>,
        to_host: bool,
        info: &transfer::Info,
        iov: Vec<GuestIov>,
    ) -> Result<(), Error> {
        let (own, attached) = self
            .with_resource(handle, |r| (r.iov.clone(), r.attached.clone()))
            .ok_or(Error::NoResource)?;
        if let Some(c) = ctx {
            if !self.contexts.contains_key(&c) {
                return Err(Error::NoContext);
            }
            if !attached.contains(&c) {
                return Err(Error::NoResource);
            }
        }
        let v = self.vrend.as_mut().ok_or(Error::RendererAbsent)?;
        let own = Iov::new(&own);
        let given = Iov::new(&iov);
        let pages = if iov.is_empty() { &own } else { &given };
        v.transfer(ctx, handle, to_host, Some(&own), pages, info).map_err(Error::Transfer)
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
        // A venus context gets venus state, a classic one vrend's; anything else gets a context
        // and nothing behind it, and finds out when it submits.
        match capset {
            CapsetId::Venus => {
                if let Some(v) = self.venus.as_mut() {
                    v.context_create(id);
                }
            }
            CapsetId::Virgl | CapsetId::Virgl2 => {
                let table = self.resources.read().expect("the resource lock is never poisoned");
                if let Some(v) = self.vrend.as_mut()
                    && let Err(e) = v.context_create(id, &*table)
                {
                    drop(table);
                    eprintln!("[virglrs] ctx {}: no GL context: {e}", id.get());
                    self.contexts.remove(&id);
                    return Err(Error::RendererAbsent);
                }
            }
            CapsetId::Unknown(_) => {}
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
        if let Some(v) = self.vrend.as_mut() {
            let table = self.resources.read().expect("the resource lock is never poisoned");
            v.context_destroy(id, &*table);
        }
        // A destroyed context releases its claim on every resource; the resources themselves
        // survive, because the VMM unrefs them separately and may still be holding one.
        for r in self.resources.write().expect("the resource lock is never poisoned").values_mut() {
            r.attached.retain(|c| *c != id);
        }
    }

    /// Destroy every context and drop every resource, leaving the renderer as new.
    ///
    /// Contexts go first, and one at a time through the same path a single destroy takes: that
    /// path is where a venus context's rings are stopped, and a bulk clear of the map would drop
    /// the contexts while their ring threads were still running against them. The resources
    /// outlive no one afterwards, so they go once nothing is attached to them.
    ///
    /// What is left is a renderer that has been initialized and nothing more -- the config, the
    /// retirement thread and the venus renderer stay, and an id used before this call is free
    /// again.
    pub fn reset(&mut self) {
        for id in self.contexts.keys().copied().collect::<Vec<_>>() {
            self.context_destroy(id);
        }
        self.resources.write().expect("the resource lock is never poisoned").clear();
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
    ///
    /// A submission does not always finish: a `vkWaitRingSeqnoMESA` in the stream suspends it,
    /// and the answer says how much ran. The caller waits -- holding none of this renderer, which
    /// is the whole reason the wait is not taken here -- and comes back with the remainder. See
    /// [`Submitted`] and [`Renderer::ring_waiter`].
    pub fn submit_cmd(&mut self, ctx: CtxId, buf: &[u8]) -> Result<Submitted, Error> {
        let Some(c) = self.contexts.get(&ctx) else {
            return Err(Error::NoContext);
        };
        match c.capset {
            CapsetId::Venus => self.venus_mut()?.submit(ctx, buf).map_err(venus_error),
            CapsetId::Virgl | CapsetId::Virgl2 => {
                // The classic wire is dwords; a stream that is not whole dwords is not a stream.
                if !buf.len().is_multiple_of(4) {
                    return Err(Error::Poisoned);
                }
                let words: Vec<u32> =
                    buf.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
                let table = self.resources.read().expect("the resource lock is never poisoned");
                let v = self.vrend.as_mut().ok_or(Error::RendererAbsent)?;
                match v.submit(ctx, &words, &*table) {
                    None => Err(Error::NoContext),
                    Some(Ok(())) => Ok(Submitted::Done),
                    Some(Err(_fault)) => Ok(Submitted::Poisoned),
                }
            }
            CapsetId::Unknown(_) => Err(Error::RendererUnimplemented),
        }
    }

    /// The wait a suspended submission named, ready to be waited on with nothing of this renderer
    /// held. See [`Submitted::Waiting`].
    pub fn ring_waiter(
        &self,
        ctx: CtxId,
        ring: RingId,
        seqno: u32,
    ) -> Result<venus::ring_thread::RingWaiter, Error> {
        self.venus
            .as_ref()
            .ok_or(Error::RendererAbsent)?
            .ring_waiter(ctx, ring, seqno)
            .map_err(venus_error)
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
        // A journal entry never suspends: `Context::submit_ring` and the replay path refuse a
        // wait outright, because there is no thread on the other side of one during a replay.
        match self.venus_mut()?.submit(ctx, buf).map_err(venus_error)? {
            Submitted::Done => Ok(()),
            Submitted::Poisoned => Err(Error::Poisoned),
            Submitted::Waiting { .. } => Err(Error::Poisoned),
        }
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

    /// Publish one venus allocation to the VMM, handing back the host address it lives at.
    ///
    /// The address is not kept here. A resource holding it would outlive the memory it points
    /// into the first time a guest freed the memory while the resource stood -- so the resource
    /// keeps the names, and [`Self::venus_memory_map_ptr`] resolves them again each time. Memory
    /// that is gone then has no address to give, instead of having a stale one.
    pub fn venus_memory_export(
        &mut self,
        ctx_id: CtxId,
        mem: BlobId,
        blob_size: u64,
    ) -> Result<(Exported, Option<Storage>, ObjectKey), Error> {
        let v = self.venus.as_ref().ok_or(Error::RendererAbsent)?;
        v.with_context_mut(ctx_id, |ctx| ctx.memory_export(ObjectId(mem.0), blob_size))
            .ok_or(Error::NoContext)?
            .map_err(export_error)
    }

    /// The IOSurface a resource is presented from, or `None` when it is not presented from one.
    ///
    /// Resolved the whole way down on every call, exactly like [`Self::resource_host_mapping`]:
    /// resource, to the allocation it was published from, to the surface that allocation is. The
    /// id is never cached anywhere along that path -- the system recycles ids immediately, so a
    /// remembered one names whoever minted next, and releasing it would free *their* surface.
    /// A resource whose surface has gone answers `None` because there is no longer a surface to
    /// ask, which is the same thing said once instead of purged at each destroy site.
    ///
    /// A share of pages has no surface, and a compositor presenting from one gets a black
    /// window with nothing to say why. The pages know why -- which question the image failed
    /// at allocate -- and say it here, once, the first time they are asked.
    pub fn resource_iosurface_id(&self, handle: ResourceHandle) -> Option<SurfaceId> {
        if let Some(surface) = self.classic_surface(handle) {
            return Some(surface.id());
        }
        let storage = self.resource_storage(handle)?;
        match storage.surface() {
            Ok(surface) => Some(surface.id()),
            Err(why) => {
                if storage.first_refusal() {
                    eprintln!(
                        "[virglrs] resource {}: has no surface to present -- {why}",
                        handle.get()
                    );
                }
                None
            }
        }
    }

    /// The surface a classic resource is presented from, when vrend gave it one. Asked of vrend,
    /// which owns the resource's host side, rather than mirrored in the table: the surface lives
    /// and dies with the texture whose storage it is.
    fn classic_surface(&self, handle: ResourceHandle) -> Option<&crate::metal::Surface> {
        self.vrend.as_ref()?.resource_surface(handle)
    }

    /// The share of storage a resource holds, for the paths that act on the bytes themselves.
    ///
    /// One resolution, so the id a frame is published under and the pixels read out of it cannot
    /// come from two different surfaces. The share is cloned out and the resource lock released
    /// before anything is done with it: holding a read lock across work that may reach a context
    /// is how a ring thread ends up waiting on us while we wait on it.
    fn resource_storage(&self, handle: ResourceHandle) -> Option<Storage> {
        self.with_resource(handle, |r| match &r.backing {
            Backing::Blob { storage: BlobStorage::Shared { storage, .. }, .. } => {
                Some(storage.clone())
            }
            Backing::Blob { .. } | Backing::Classic(_) | Backing::Imported { .. } => None,
        })?
    }

    /// Copy a scanout resource's presented pixels into a caller's buffer, `stride` bytes per row.
    ///
    /// The headless display sink's only way to see a frame: a venus scanout blob has no CPU
    /// transfer path, because the frame exists nowhere but the surface's shared storage.
    ///
    /// The count of rows that landed, so a caller that asked for more than the surface holds is
    /// told so rather than handed a buffer with a stale tail in it.
    pub fn resource_read_iosurface(
        &self,
        handle: ResourceHandle,
        dst: &mut [u8],
        stride: usize,
        height: u32,
    ) -> Option<u32> {
        if let Some(surface) = self.classic_surface(handle) {
            return Some(surface.read_rows(dst, stride, height));
        }
        Some(self.resource_storage(handle)?.surface().ok()?.read_rows(dst, stride, height))
    }

    /// Complete a classic scanout's renders before the surface they landed in is presented.
    ///
    /// Classic only: a venus scanout renders into its surface on the guest's own timeline and
    /// must never be synced from here. `false` for a resource that has no surface -- the VMM's
    /// cue to read its pixels back the slow way.
    pub fn resource_sync_iosurface(&mut self, handle: ResourceHandle) -> bool {
        self.vrend.as_mut().is_some_and(|v| v.resource_sync_iosurface(handle))
    }

    /// Where a blob resource lives in this process, for a VMM about to publish it to the guest.
    ///
    /// The one question the mapping calls ask, in one answer: an address on its own is not enough
    /// to publish memory, and a size or a caching mode fetched separately is a second lookup that
    /// can land on a different resource -- or on one that has since been freed.
    ///
    /// Answered from what the resource holds. A share keeps its storage alive, so its address is
    /// good by construction. A borrowed mapping is good only while the allocation that lent it
    /// stands, which is asked of the exporting context every time -- of the object, by its key,
    /// so a fresh allocation the guest has since named by the same id does not answer for it.
    pub fn resource_host_mapping(&self, handle: ResourceHandle) -> Result<HostMapping, Error> {
        // What is needed is copied out and the resource lock released before venus is asked
        // anything. Holding a read lock across a call into a context is how a ring thread on the
        // other side of it ends up waiting on us while we wait on it.
        let answer = self
            .with_resource(handle, |r| match &r.backing {
                Backing::Blob { desc, storage } => match storage {
                    // The mapping answers for its own extent: `desc.size` is what was asked for,
                    // this is what was mapped, and it is what bounds what may be read. The host
                    // maps its own shm write-back and coherent, like any anonymous page.
                    BlobStorage::Minted(h) => Some(Ok(HostMapping {
                        addr: h.map.host_addr(),
                        size: h.map.len() as u64,
                        caching: Caching::Cached,
                    })),
                    // The blob's size, not the storage's: the export held the one to the other.
                    BlobStorage::Shared { storage, caching } => Some(Ok(HostMapping {
                        addr: storage.span().0,
                        size: desc.size,
                        caching: *caching,
                    })),
                    BlobStorage::Borrowed { ctx, key, mapping, .. } => {
                        Some(Err((*ctx, *key, *mapping)))
                    }
                    BlobStorage::Guest => None,
                },
                Backing::Classic(_) | Backing::Imported { .. } => None,
            })
            .ok_or(Error::NoResource)?
            .ok_or(Error::NotMappable)?;
        let (ctx, key, mapping) = match answer {
            Ok(held) => return Ok(held),
            Err(borrowed) => borrowed,
        };
        if !self.venus_context(ctx, |c| c.holds(key))? {
            return Err(Error::NoAllocation);
        }
        Ok(mapping)
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
        (true, true) => {
            "venus serves only part of the protocol; vrend serves resources and transfers"
        }
        (true, false) => "venus serves only part of the protocol",
        (false, true) => "vrend serves resources and transfers",
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
        Renderer::new(Box::new(NoSink), config).expect("no vrend is asked for")
    }

    /// The two questions `HostShm::for_blob` answers, and it answers them from the source alone.
    ///
    /// One entry point serves two different guest requests: one asks the renderer to supply
    /// memory, the other names memory a context already has and asks for it to be published.
    /// Minting for the second kind would hand the guest fresh zeroed pages where it expected the
    /// contents of a `VkDeviceMemory` -- a wrong answer that looks like a working one.
    #[test]
    fn only_a_blob_that_asks_the_host_for_memory_is_given_any() {
        let mut r = renderer(Config::default());

        let minted = BlobDesc {
            blob_mem: crate::abi::BLOB_MEM_HOST3D,
            blob_flags: 1,
            source: BlobSource::HostMinted,
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

        // A blob naming memory that already exists gets none of its own. There is no venus in
        // this build, so the export itself is refused -- what is being asked here is that the
        // refusal came from the export path and not from minting something first.
        let exported = BlobDesc {
            source: BlobSource::Exported { ctx: CtxId::new(1).unwrap(), mem: BlobId(9) },
            ..minted
        };
        assert_eq!(
            r.resource_create_blob(ResourceHandle::new(2).unwrap(), exported, Vec::new()),
            Err(Error::RendererAbsent),
            "an export names memory a context holds; minting would answer with the wrong bytes"
        );
        assert!(
            r.with_resource(ResourceHandle::new(2).unwrap(), |res| res.shm().cloned()).is_none(),
            "and a create that failed leaves no resource behind"
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

    /// What a context may reach is what the guest kernel attached to it, and nothing else.
    ///
    /// Resource ids are device-wide -- the C keeps one table for the whole device -- so "which
    /// context created it" is not a permission, and a compositor sampling a client's window is
    /// the ordinary case rather than an error. `CTX_ATTACH_RESOURCE` is where that decision is
    /// actually made, so this gate delegates it instead of inventing a second one.
    ///
    /// No corpus can reach this: the captures are one guest, replayed one context at a time.
    #[test]
    fn a_context_reaches_the_resources_the_guest_attached_to_it() {
        use crate::venus::ring::ShmResources;

        let one = CtxId::new(1).unwrap();
        let two = CtxId::new(2).unwrap();
        let blob = ResourceHandle::new(1).unwrap();

        // The exporting allocation, as its context's object table holds it: what a borrowed
        // mapping is keyed to.
        let mut objects = crate::venus::objects::Table::new();
        objects
            .add(
                ObjectId(66),
                crate::venus::proto::types::VkObjectType::VK_OBJECT_TYPE_DEVICE_MEMORY,
                crate::venus::cs::HostHandle(0x9000),
                None,
            )
            .unwrap();
        let key = objects.key_of(ObjectId(66)).unwrap();
        let borrowed = |attached: Vec<CtxId>| Resource {
            handle: blob,
            backing: Backing::Blob {
                desc: BlobDesc {
                    blob_mem: crate::abi::BLOB_MEM_HOST3D,
                    blob_flags: 1,
                    source: BlobSource::Exported { ctx: one, mem: BlobId(66) },
                    size: 4128768,
                },
                // Ordinary device memory: published as a borrowed `vkMapMemory` pointer, which
                // is storage this renderer cannot keep alive on anyone else's behalf.
                storage: BlobStorage::Borrowed {
                    ctx: one,
                    mem: BlobId(66),
                    key,
                    mapping: HostMapping { addr: 0x1000, size: 4128768, caching: Caching::Cached },
                },
            },
            iov: Vec::new(),
            priv_: VmmPtr(core::ptr::null_mut()),
            attached,
        };

        let mut table = BTreeMap::new();
        table.insert(blob, borrowed(vec![one]));
        assert_eq!(
            match table.bytes(one, blob) {
                Some(ResourceBytes::Allocation(published)) => Some(published),
                _ => None,
            },
            Some(Published { memory: ObjectId(66), size: 4128768 }),
            "the context it is attached to finds the allocation, at the size the resource has"
        );
        assert!(
            table.bytes(two, blob).is_none(),
            "a context the guest never attached it to reaches nothing, whoever exported it"
        );

        // Attached to both, and still refused for the second -- but for a reason about the
        // storage rather than about who owns the name. A borrowed mapping into ctx one's device
        // would dangle for ctx two the moment ctx one freed it or went away.
        table.insert(blob, borrowed(vec![one, two]));
        assert!(
            table.bytes(two, blob).is_none(),
            "attached is not enough when the storage behind it cannot be shared"
        );

        assert!(
            table.bytes(one, ResourceHandle::new(2).unwrap()).is_none(),
            "a resource that is not here is not reachable by anyone"
        );

        // Shared memory -- what a ring lives in -- is behind the same gate. It is the resource a
        // context reads commands from, so a context reaching one it was never attached to would
        // be reading another guest process's ring.
        let shm = ResourceHandle::new(4).unwrap();
        let map = Arc::new(crate::guest_mem::GuestMap::anonymous(4096).expect("minted"));
        table.insert(
            shm,
            Resource {
                handle: shm,
                backing: Backing::Imported {
                    desc: ImportDesc {
                        blob_mem: BlobMem::Host3d,
                        fd_type: FdType::Shm,
                        size: 4096,
                    },
                    fd: crate::guest_mem::anonymous_shm(4096, "virglrs-attach-test")
                        .expect("minted")
                        .0,
                    map: Some(Arc::clone(&map)),
                },
                iov: Vec::new(),
                priv_: VmmPtr(core::ptr::null_mut()),
                attached: vec![one],
            },
        );
        assert!(
            matches!(table.bytes(one, shm), Some(ResourceBytes::Host(m)) if Arc::ptr_eq(&m, &map)),
            "the context it is attached to reaches the shared memory"
        );
        assert!(
            table.bytes(two, shm).is_none(),
            "a context the guest never attached it to reaches no shared memory either"
        );

        // A blob the host minted and never mapped publishes no storage either: there is memory
        // somewhere, but nothing here can say where, and saying so would be inventing it.
        let minted = ResourceHandle::new(3).unwrap();
        table.insert(
            minted,
            Resource {
                handle: minted,
                backing: Backing::Blob {
                    desc: BlobDesc {
                        blob_mem: crate::abi::BLOB_MEM_HOST3D,
                        blob_flags: 1,
                        source: BlobSource::HostMinted,
                        size: 4096,
                    },
                    storage: BlobStorage::Guest,
                },
                iov: Vec::new(),
                priv_: VmmPtr(core::ptr::null_mut()),
                attached: vec![one],
            },
        );
        assert!(
            table.bytes(one, minted).is_none(),
            "a blob with no host storage resolves to nothing"
        );
    }

    /// Storage a resource holds a *share* of resolves for every context the guest attached it to,
    /// and resolves to the same bytes for each.
    ///
    /// This is the whole point of holding a share rather than a name: a name is only meaningful
    /// in the context that chose it, so a compositor could never reach a client's buffer. The
    /// surface here is a real one, because a share whose storage is faked proves nothing about
    /// whether the pages are actually there.
    #[test]
    fn a_share_resolves_for_every_context_the_resource_is_attached_to() {
        use crate::venus::driver::Storage;
        use crate::venus::ring::ShmResources;

        let one = CtxId::new(1).unwrap();
        let two = CtxId::new(2).unwrap();
        let three = CtxId::new(3).unwrap();
        let blob = ResourceHandle::new(1).unwrap();

        let surface = crate::metal::Surface::scanout(64, 8, crate::metal::PixelFormat::Bgra, 256)
            .expect("the system minted a surface");
        let account = crate::venus::budget::Account::for_test(None);
        let share = Storage::minted_for_test(surface, &account);

        let mut table = BTreeMap::new();
        table.insert(
            blob,
            Resource {
                handle: blob,
                backing: Backing::Blob {
                    desc: BlobDesc {
                        blob_mem: crate::abi::BLOB_MEM_HOST3D,
                        blob_flags: 1,
                        source: BlobSource::Exported { ctx: one, mem: BlobId(66) },
                        size: 2048,
                    },
                    storage: BlobStorage::Shared {
                        storage: share.clone(),
                        caching: Caching::Cached,
                    },
                },
                iov: Vec::new(),
                priv_: VmmPtr(core::ptr::null_mut()),
                attached: vec![one, two],
            },
        );

        let reached = |ctx| match table.bytes(ctx, blob) {
            Some(ResourceBytes::Shared(s)) => Some(s),
            _ => None,
        };
        assert_eq!(
            reached(one).as_ref(),
            Some(&share),
            "the context that exported it reaches the storage it published"
        );
        assert_eq!(
            reached(two).as_ref(),
            Some(&share),
            "and so does the other context the guest attached it to -- the same storage, not a \
             lookup of id 66 in ctx two's own table"
        );
        assert!(
            table.bytes(three, blob).is_none(),
            "a context the guest never attached it to still reaches nothing"
        );

        // The other shape of share -- pages minted for exportable memory -- resolves by the same
        // rule. The gate is about the share, not about what the share is of.
        let pages = Storage::pages_for_test(4096, &account);
        let linear = ResourceHandle::new(2).unwrap();
        table.insert(
            linear,
            Resource {
                handle: linear,
                backing: Backing::Blob {
                    desc: BlobDesc {
                        blob_mem: crate::abi::BLOB_MEM_HOST3D,
                        blob_flags: 1,
                        source: BlobSource::Exported { ctx: one, mem: BlobId(67) },
                        size: 4096,
                    },
                    storage: BlobStorage::Shared {
                        storage: pages.clone(),
                        caching: Caching::Cached,
                    },
                },
                iov: Vec::new(),
                priv_: VmmPtr(core::ptr::null_mut()),
                attached: vec![two],
            },
        );
        assert_eq!(
            match table.bytes(two, linear) {
                Some(ResourceBytes::Shared(s)) => Some(s),
                _ => None,
            }
            .as_ref(),
            Some(&pages),
            "pages resolve for the context attached to them, exporter or not"
        );
        assert!(table.bytes(one, linear).is_none(), "and not for the exporter once detached");
    }

    /// The VMM publishes a blob by asking where it lives, and it asks *after* the create --
    /// three separate ABI calls, each of which has to be answered from the resource that is
    /// standing now rather than from anything remembered at the create.
    ///
    /// The export half of this cannot be reached without a venus renderer, so what is pinned here
    /// is the half that can: a minted blob answers from the mapping it owns, and everything that
    /// is not a blob refuses instead of inventing an address.
    #[test]
    fn a_blob_says_where_it_lives_and_everything_else_refuses_to() {
        let mut r = renderer(Config::default());

        let minted = BlobDesc {
            blob_mem: crate::abi::BLOB_MEM_HOST3D,
            blob_flags: 1,
            source: BlobSource::HostMinted,
            size: 0x24000 - 1,
        };
        let blob = ResourceHandle::new(1).unwrap();
        r.resource_create_blob(blob, minted, Vec::new()).expect("created");

        let m = r.resource_host_mapping(blob).expect("a minted blob knows where it is");
        let map = r.with_resource(blob, |res| res.shm().cloned()).expect("there").expect("memory");
        assert_eq!(m.addr, map.host_addr(), "the address is the mapping's own");
        assert_eq!(m.size, map.len() as u64, "and so is the extent, not what was asked for");
        assert!(m.size > minted.size, "which here is the larger of the two");
        assert_eq!(m.caching, Caching::Cached);

        // Memory the host never mapped has no address to give, and saying so is the difference
        // between a VMM reporting a failed guest mmap and one publishing a wild pointer.
        let vram = BlobDesc { blob_mem: crate::abi::BLOB_MEM_GUEST_VRAM, ..minted };
        let elsewhere = ResourceHandle::new(2).unwrap();
        r.resource_create_blob(elsewhere, vram, Vec::new()).expect("created");
        assert_eq!(r.resource_host_mapping(elsewhere), Err(Error::NotMappable));

        assert_eq!(
            r.resource_host_mapping(ResourceHandle::new(77).unwrap()),
            Err(Error::NoResource),
            "a handle that names nothing is not the same as a resource that maps nothing"
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
    /// A borrowed mapping is good while the allocation that lent it stands, and not a moment
    /// longer -- and "stands" is asked of the object, not the id.
    ///
    /// The C caches the exporter's pointer on the resource and answers it for as long as the
    /// resource lives, which is a use after the guest frees the memory. Resolving the *name*
    /// each time instead was worse in a different way: the guest reuses ids, so after a free and
    /// a fresh allocation under the same id the name resolves to somebody else's bytes. The key
    /// answers "is it still that object", which is the only question with a safe answer.
    #[test]
    fn a_borrowed_mapping_dies_with_the_object_that_lent_it_and_not_with_its_id() {
        use crate::venus::cs::HostHandle;
        use crate::venus::proto::types::VkObjectType;

        const MEM: ObjectId = ObjectId(66);
        let mut r = renderer(Config { venus: true, ..Config::default() });
        let one = CtxId::new(1).unwrap();
        r.context_create(one, CapsetId::Venus, "exporter".into()).expect("a fresh id");

        // The allocation the blob borrows, as the context's table holds it.
        let add = |r: &Renderer, handle: u64| {
            r.venus_context(one, |c| {
                c.objects()
                    .borrow_mut()
                    .add(MEM, VkObjectType::VK_OBJECT_TYPE_DEVICE_MEMORY, HostHandle(handle), None)
                    .expect("added")
            })
            .expect("the context is here")
        };
        add(&r, 0x9000);
        let key = r
            .venus_context(one, |c| c.objects().borrow().key_of(MEM))
            .expect("the context is here")
            .expect("just added");

        let mapping = HostMapping { addr: 0x1000, size: 4096, caching: Caching::Cached };
        let blob = ResourceHandle::new(5).unwrap();
        r.insert(
            blob,
            Backing::Blob {
                desc: BlobDesc {
                    blob_mem: crate::abi::BLOB_MEM_HOST3D,
                    blob_flags: 1,
                    source: BlobSource::Exported { ctx: one, mem: BlobId(MEM.0) },
                    size: 4096,
                },
                storage: BlobStorage::Borrowed { ctx: one, mem: BlobId(MEM.0), key, mapping },
            },
            Vec::new(),
        );
        assert_eq!(r.resource_host_mapping(blob), Ok(mapping), "the allocation stands");

        // The guest frees the memory, and allocates again under the same id.
        r.venus_context(one, |c| c.objects().borrow_mut().remove(MEM)).unwrap().expect("removed");
        assert_eq!(
            r.resource_host_mapping(blob),
            Err(Error::NoAllocation),
            "the mapping went with the allocation"
        );
        add(&r, 0x9100);
        assert_eq!(
            r.resource_host_mapping(blob),
            Err(Error::NoAllocation),
            "and a fresh allocation under the same id does not answer for it"
        );
    }

    /// A scanout of pages answers no surface, and says why -- once.
    ///
    /// The line itself goes to stderr, where a test cannot read it; what is pinned is that the
    /// share knows the reason and hands out the first refusal exactly once, so a compositor
    /// asking every frame gets one line and not a log of them.
    #[test]
    fn a_scanout_of_pages_has_no_surface_and_says_so_once() {
        use crate::venus::budget::Account;
        use crate::venus::driver::NoSurface;

        let mut r = renderer(Config { venus: true, ..Config::default() });
        let one = CtxId::new(1).unwrap();
        let account = Account::for_test(None);
        let pages = Storage::pages_for_test(4096, &account);
        assert_eq!(pages.surface().err(), Some(NoSurface::NotDedicated), "the pages know why");

        let blob = ResourceHandle::new(5).unwrap();
        r.insert(
            blob,
            Backing::Blob {
                desc: BlobDesc {
                    blob_mem: crate::abi::BLOB_MEM_HOST3D,
                    blob_flags: 1,
                    source: BlobSource::Exported { ctx: one, mem: BlobId(66) },
                    size: 4096,
                },
                storage: BlobStorage::Shared { storage: pages.clone(), caching: Caching::Cached },
            },
            Vec::new(),
        );
        assert_eq!(r.resource_iosurface_id(blob), None, "nothing to present from");
        assert!(!pages.first_refusal(), "the ask above was the first, and it has been said");
        assert_eq!(r.resource_iosurface_id(blob), None);
        assert!(!pages.first_refusal(), "and it is not said again");
    }

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
        let Some(Capset::Venus(c)) = renderer(on).capset(CapsetId::Venus) else {
            panic!("venus is served");
        };
        assert_eq!(c.use_guest_vram, 1, "a field, read as a field");
        assert_eq!(c.as_bytes().len(), venus::capset::size() as usize);
    }

    /// A reset leaves nothing of what came before it.
    ///
    /// The shim used to print how many contexts and resources it was dropping and drop none of
    /// them, so a VMM resetting between guest boots carried the previous boot's contexts into the
    /// next one -- and the ids it had just been told were free came back `ContextExists`.
    #[test]
    fn a_reset_frees_the_ids_and_the_memory_it_says_it_frees() {
        let fd = a_descriptor();
        let raw = fd.as_raw_fd();
        let mut r = renderer(Config::default());
        let ctx = CtxId::new(1).unwrap();
        let res = ResourceHandle::new(1).unwrap();

        r.context_create(ctx, CapsetId::Virgl, "before".into()).expect("a fresh id");
        r.resource_import(res, import_desc(4096), fd).expect("a fresh handle and a real size");
        r.ctx_attach_resource(ctx, res);
        assert_eq!(r.counts(), (1, 1));

        r.reset();

        assert_eq!(r.counts(), (0, 0), "nothing survives a reset");
        assert!(!is_open(raw), "and the memory a resource held goes with it");
        r.context_create(ctx, CapsetId::Virgl, "after".into())
            .expect("the id the reset freed is free");
    }
}
