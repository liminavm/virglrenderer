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
use crate::guest_mem::{GuestMap, Iov, PixelSource};
use crate::ids::{
    BlobId, ClientFenceId, ContextId, FenceId, ResourceHandle, RingId, RingIdx, SurfaceId,
};
use crate::venus;
use crate::venus::context::Submitted;
use crate::venus::cs::ObjectId;
use crate::venus::driver::{Allocation, Exported, MemoryError, Storage};
use crate::venus::objects::ObjectKey;
use crate::venus::ring::ResourceBytes;
use crate::venus::vkr::ContextKey;
use crate::vrend;
use crate::vrend::content;
use crate::vrend::context::Guest;
use crate::vrend::proto::Box3;
use crate::vrend::resource::{Args as ClassicArgs, Refusal};
use crate::vrend::transfer;
use std::collections::{BTreeMap, BTreeSet};
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
    /// A restore carries more bytes than the allocation it is being put back into holds; see
    /// [`MemoryError::LargerThanAllocation`].
    ContentLargerThanAllocation,
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
    /// A restore was handed something that is not a classic content blob.
    MalformedContent(content::Malformed),
    /// A restore was handed something that is not a venus sync blob.
    MalformedSync(venus::sync::Malformed),
}

/// A read or a write of an allocation's bytes, in the renderer's vocabulary. One function for
/// both, because a route one of them refuses is a route the other refuses for the same reason.
fn memory_error(e: MemoryError) -> Error {
    match e {
        MemoryError::NoSuchAllocation => Error::NoAllocation,
        MemoryError::NotMappable => Error::NotMappable,
        MemoryError::LargerThanAllocation => Error::ContentLargerThanAllocation,
    }
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
            Error::ContentLargerThanAllocation => {
                "the restored contents are larger than the allocation they go back into"
            }
            Error::ZeroSize => "an import of zero bytes names no memory",
            Error::Unmappable => "that shm descriptor could not be mapped",
            Error::ClassicRefused(r) => return write!(f, "vrend refused the resource: {r}"),
            Error::Transfer(e) => return write!(f, "the transfer failed: {e}"),
            Error::MalformedContent(m) => return write!(f, "the contents were refused: {m}"),
            Error::MalformedSync(m) => return write!(f, "the sync state was refused: {m}"),
        };
        f.write_str(s)
    }
}

impl std::error::Error for Error {}

/// Where a blob's storage comes from.
///
/// The wire discriminates on `blob_id == 0`, one magic number standing between two operations that
/// have nothing in common: one asks this renderer to supply memory, the other names memory a
/// context already holds and asks for it to be published. Naming them separates the two, and
/// carries the context the id means something in -- without which the id names nothing at all.
///
/// What a nonzero id names is the *context's* to say, and there is nothing in the number to tell
/// the two cases apart: in a venus context it is a `VkDeviceMemory` the guest allocated, and in a
/// classic one it is a resource that context's own command stream described with
/// `PIPE_RESOURCE_CREATE`. So this says what the guest said -- a context and an id -- and
/// [`Renderer::resource_create_blob`] resolves it against the context's capset, which is the only
/// place that knows. What it resolves *to* is spelled out: see [`BlobStorage`] and [`Backing`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlobSource {
    /// The guest asks the host for memory it does not yet have.
    HostMinted,
    /// A context publishes something it already holds, under the id it holds it by.
    InContext { ctx: ContextId, id: BlobId },
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
    Shared { storage: Storage, caching: Caching, from: Exporter },
}

/// The allocation a blob was exported from: whose it is, and which object it was.
///
/// One value rather than a context beside a key, because neither means anything without the other:
/// an [`ObjectKey`] indexes one context's arena, so the same key names a different object in every
/// other context. Passing them separately would let a caller ask one context's table about another
/// context's key and get a confident wrong answer.
///
/// Both halves are generational, and for one reason: a blob outlives what it was exported from.
/// The object goes when the guest frees it, and the context goes when the guest destroys it -- and
/// the VMM is free to stand a new context up under the same id straight afterwards. A bare
/// [`ContextId`] here would let that new context find this blob among its own and be told about an
/// allocation belonging to a guest that is gone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Exporter {
    pub ctx: ContextKey,
    pub key: ObjectKey,
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
    ///
    /// `surface` is a share of the IOSurface that host side's storage is, when it is one, taken
    /// at the create because that is when the surface is minted and there is no later moment one
    /// appears. It is what a venus context imports the resource by: a share held here resolves
    /// without asking vrend's table, which is what lets a compositor keep a client's last frame
    /// after the client's context is gone.
    Classic { args: ClassicArgs, surface: Option<Arc<dyn crate::metal::Held>> },
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
    pub attached: Vec<ContextId>,
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

    fn bytes(&self, ctx: ContextId, handle: ResourceHandle) -> Option<ResourceBytes> {
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
            Backing::Blob { storage, .. } => match storage {
                // A share resolves for anyone holding it: no table is asked.
                BlobStorage::Shared { storage, .. } => Some(ResourceBytes::Shared(storage.clone())),
                BlobStorage::Minted(_) => unreachable!("answered as shared memory above"),
                BlobStorage::Guest => {
                    eprintln!(
                        "[virglrs] ctx {}: resource {handle:?} is a blob with no host storage",
                        ctx.get(),
                    );
                    None
                }
            },
            // A classic resource whose storage is an IOSurface lends a share of it, which is how
            // a GL client's window buffer reaches the Vulkan compositor importing it. The share
            // travels, never the surface's id: an id names whatever the system last minted under
            // it, so an import holding one would resolve to a stranger's surface the moment this
            // resource died. Anything else classic is a GL object with no host address at all.
            Backing::Classic { surface, .. } => match surface {
                Some(held) => Some(ResourceBytes::Shared(Storage::lent(Arc::clone(held)))),
                None => {
                    eprintln!(
                        "[virglrs] ctx {}: resource {handle:?} is a classic resource with no \
                         IOSurface storage, so there is nothing to import",
                        ctx.get(),
                    );
                    None
                }
            },
            Backing::Imported { .. } => {
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
    fn attached(&self, ctx: ContextId, handle: ResourceHandle) -> bool {
        self.get(&handle).is_some_and(|r| r.attached.contains(&ctx))
    }

    fn pages(&self, ctx: ContextId, handle: ResourceHandle) -> Option<Iov<'_>> {
        let r = self.get(&handle)?;
        if !r.attached.contains(&ctx) || r.iov.is_empty() {
            return None;
        }
        Some(Iov::new(&r.iov))
    }

    fn blob_pixels(&self, ctx: ContextId, handle: ResourceHandle) -> Option<PixelSource<'_>> {
        let r = self.get(&handle)?;
        if !r.attached.contains(&ctx) {
            return None;
        }
        let Backing::Blob { storage, .. } = &r.backing else {
            // Only a blob is typed after the fact; a classic resource was made from a description
            // and has never been in the state this answers for.
            return None;
        };
        match storage {
            // The guest's own pages. `iov` is the VMM's list, borrowed for this call only.
            BlobStorage::Guest => {
                (!r.iov.is_empty()).then(|| PixelSource::Scattered(Iov::new(&r.iov)))
            }
            BlobStorage::Minted(h) => Some(PixelSource::Mapped(&h.map)),
            // `pages()` is `None` for a surface, which is not a gap: a surface is adopted whole
            // and the texture's storage becomes it, so reading its bytes here would mint exactly
            // the copy the adopt exists to avoid.
            BlobStorage::Shared { storage, .. } => storage.mapping().map(PixelSource::Mapped),
        }
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
            Backing::Blob { .. } | Backing::Classic { .. } => None,
        }
    }
}

pub struct Context {
    pub id: ContextId,
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
    contexts: BTreeMap<ContextId, Context>,
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
        let vrend = if config.vrend { Some(vrend::vrend::Vrend::new(config)?) } else { None };
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
        // The share is taken here and not looked up later: the surface is minted inside the
        // create, and a resource that got one has it from this moment to its last.
        let surface = match self.vrend.as_mut() {
            Some(v) => {
                v.resource_create(handle, args).map_err(Error::ClassicRefused)?;
                v.resource_surface_share(handle)
            }
            None => None,
        };
        self.insert(handle, Backing::Classic { args, surface }, iov);
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
            BlobSource::InContext { ctx, id } => match self.blob_capset(ctx)? {
                CapsetId::Virgl | CapsetId::Virgl2 => {
                    return self.claim_described(handle, ctx, id, desc, iov);
                }
                // A context whose capset names no renderer we have holds nothing, so its ids
                // name nothing either.
                CapsetId::Unknown(_) => return Err(Error::RendererUnimplemented),
                CapsetId::Venus => {
                    let mem = id;
                    match self.venus_memory_export(ctx, mem, desc.size) {
                        Ok((exported, storage, from)) => BlobStorage::Shared {
                            storage,
                            caching: if exported.write_back {
                                Caching::Cached
                            } else {
                                Caching::WriteCombining
                            },
                            from,
                        },
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
            },
        };
        self.insert(handle, Backing::Blob { desc, storage }, iov);
        Ok(())
    }

    /// Which renderer a blob's id is to be read by. `Err` for a context that is not here, which
    /// is a guest naming a context it never created.
    fn blob_capset(&self, ctx: ContextId) -> Result<CapsetId, Error> {
        self.contexts.get(&ctx).map(|c| c.capset).ok_or(Error::NoContext)
    }

    /// The classic half of a nonzero `blob_id`: adopt the resource `ctx`'s command stream
    /// described under it, and publish its pages.
    ///
    /// It becomes an ordinary classic resource -- [`Backing::Classic`], vrend's host side under
    /// the same handle -- because that is what it is. Transfers, draws and views over it work
    /// with nothing new: the only thing the blob path added was a second way for the guest to
    /// learn its handle, and a mapping.
    fn claim_described(
        &mut self,
        handle: ResourceHandle,
        ctx: ContextId,
        id: BlobId,
        desc: BlobDesc,
        iov: Vec<GuestIov>,
    ) -> Result<(), Error> {
        let v = self.vrend.as_mut().ok_or(Error::RendererAbsent)?;
        let args = match v.claim_described(ctx, id, handle, desc.size) {
            Ok(args) => args,
            Err(why) => {
                // The guest kernel treats CREATE_BLOB as fire-and-forget, so this line is the
                // only account anyone gets of the failure -- and each refusal sends the reader
                // somewhere different. See [`ClaimRefused`].
                eprintln!(
                    "[virglrs] resource {handle}: CREATE_BLOB of ctx {ctx} blob {id}, \
                     {} bytes: {why}",
                    desc.size,
                );
                return Err(Error::NoAllocation);
            }
        };
        self.insert(handle, Backing::Classic { args, surface: None }, iov);
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
    ///
    /// **A refusal here is answered to the caller and never poisons `ctx`.** Poisoning is the
    /// answer to a guest that asked for something impossible, and on this path the guest did not
    /// ask -- the VMM did, naming a context only to reach its resources. The C conflates the two:
    /// its planar refusal calls `vrend_report_context_error`, so a VMM readback of a planar
    /// resource marks that guest's context in error and every later submit on it is dropped. It
    /// survives that only because vrend exempts ctx 0 from the gate and its one host-side reader
    /// happens to pass 0. Here the separation is structural instead: the command handlers turn a
    /// transfer failure into a `Fault`, which poisons, and this function returns a
    /// `transfer::Error`, which cannot. A host-initiated transfer has no way to kill a guest.
    pub fn transfer(
        &mut self,
        handle: ResourceHandle,
        ctx: Option<ContextId>,
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
        id: ContextId,
        capset: CapsetId,
        name: String,
    ) -> Result<(), Error> {
        // A guest reusing a live id is the guest's error, not ours: rejected rather than
        // replacing an entry it still holds. Zero needs no check -- `ContextId` cannot be zero.
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

    pub fn context_destroy(&mut self, id: ContextId) {
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

    pub fn context(&self, id: ContextId) -> Option<&Context> {
        self.contexts.get(&id)
    }

    pub fn ctx_attach_resource(&mut self, ctx: ContextId, handle: ResourceHandle) {
        if !self.contexts.contains_key(&ctx) {
            return;
        }
        self.with_resource_mut(handle, |r| {
            if !r.attached.contains(&ctx) {
                r.attached.push(ctx);
            }
        });
        // A blob has no host side until something types it, so this is where vrend hears about
        // one. The share travels rather than an id: an id stops naming this surface the moment
        // the surface dies, and the whole point of holding storage across contexts is that it
        // cannot. `None` is a blob with no surface to adopt -- the reason stays with the storage
        // that refused, which says it once itself.
        let blob = self
            .with_resource(handle, |r| match &r.backing {
                Backing::Blob { storage: BlobStorage::Shared { storage, .. }, .. } => {
                    Some(storage.held())
                }
                Backing::Blob { .. } => Some(None),
                Backing::Classic { .. } | Backing::Imported { .. } => None,
            })
            .flatten();
        if let Some(held) = blob
            && let Some(v) = self.vrend.as_mut()
        {
            v.resource_attach_blob(handle, held);
        }
    }

    pub fn ctx_detach_resource(&mut self, ctx: ContextId, handle: ResourceHandle) {
        self.with_resource_mut(handle, |r| r.attached.retain(|c| *c != ctx));
    }

    // ---- fences ----

    pub fn context_create_fence(
        &mut self,
        ctx: ContextId,
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
    pub fn submit_cmd(&mut self, ctx: ContextId, buf: &[u8]) -> Result<Submitted, Error> {
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
        ctx: ContextId,
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
    pub fn venus_replay_cmd(&mut self, ctx: ContextId, buf: &[u8]) -> Result<(), Error> {
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
        ctx: ContextId,
        ring: RingId,
        buf: &[u8],
    ) -> Result<(), Error> {
        self.venus_mut()?.submit_ring(ctx, ring, buf).map_err(venus_error)
    }

    pub fn venus_replay_begin(&mut self, ctx: ContextId) -> Result<(), Error> {
        self.venus.as_mut().ok_or(Error::RendererAbsent)?.replay_begin(ctx).map_err(venus_error)
    }

    pub fn venus_replay_end(&mut self, ctx: ContextId) -> Result<(), Error> {
        self.venus.as_mut().ok_or(Error::RendererAbsent)?.replay_end(ctx).map_err(venus_error)
    }

    /// Whether this context is served by the classic renderer.
    ///
    /// The snapshot entry points ask because the two renderers keep separate journals and a
    /// context belongs to exactly one of them. This is the same split the C makes by having
    /// `limina_classic_ctx_lookup` answer NULL for a venus capset -- said once, as a question
    /// about the context, rather than rediscovered at each entry point.
    pub fn is_classic(&self, ctx: ContextId) -> bool {
        self.contexts
            .get(&ctx)
            .is_some_and(|c| matches!(c.capset, CapsetId::Virgl | CapsetId::Virgl2))
    }

    /// One classic context's journal, for the VMM to store beside its own.
    pub fn vrend_journal_export(&self, id: ContextId) -> Option<Vec<u8>> {
        self.vrend.as_ref()?.journal_export(id)
    }

    /// One classic context's resource contents, for the VMM to store beside its journal.
    ///
    /// `None` for a context this renderer does not serve as a classic one; an empty world is a
    /// blob with no entries, which is a different answer and says so.
    ///
    /// The walk is over the resources attached to the context, which is the classic analogue of
    /// the C's `ctx->res_hash` -- and attachment lives here rather than in vrend, so the loop does
    /// too. What it costs is a full readback of every level of every attached resource; that is
    /// the price of the textures whose only copy is on the host, and a snapshot is not a hot path.
    pub fn vrend_content_export(&mut self, ctx: ContextId) -> Option<(Vec<u8>, content::Account)> {
        if !self.is_classic(ctx) {
            return None;
        }
        let handles: Vec<ResourceHandle> = self
            .resources
            .read()
            .expect("the resource lock is never poisoned")
            .iter()
            .filter(|(_, r)| r.attached.contains(&ctx))
            .map(|(h, _)| *h)
            .collect();

        let mut capture = content::Capture::new();
        for handle in handles {
            let Some(plan) = self.content_plan(handle) else {
                capture.excluded();
                continue;
            };
            let own = self.with_resource(handle, |r| r.iov.clone()).unwrap_or_default();
            let own = Iov::new(&own);
            for (level, region, size) in plan {
                let mut bytes = vec![0u8; size];
                let span = crate::guest_mem::HostSpanMut::new(&mut bytes);
                let info = transfer::Info {
                    level,
                    stride: 0,
                    layer_stride: 0,
                    offset: 0,
                    region,
                    synchronized: true,
                };
                let v = self.vrend.as_mut().expect("a classic context needs vrend");
                match v.transfer(Some(ctx), handle, false, Some(&own), &span.iov(), &info) {
                    // Counted, and nothing written: a level that could not be read back has no
                    // bytes, and zeros in its place would restore over whatever the guest still
                    // holds. The C writes them, having called `calloc` so at least they are
                    // deterministic zeros rather than heap garbage -- which is the tell that the
                    // entry should not have been there at all.
                    Err(_) => capture.skipped(),
                    Ok(()) => capture.push(handle, level, region, &bytes),
                }
            }
        }
        let account = capture.account();
        Some((capture.into_bytes(), account))
    }

    /// Which levels of a resource a capture reads, and how many bytes each needs. `None` for a
    /// resource deliberately left out.
    fn content_plan(&self, handle: ResourceHandle) -> Option<Vec<(u32, Box3, usize)>> {
        let res = self.vrend.as_ref()?.resource(handle)?;
        // A multisample surface cannot be read back at all, and is a render target the compositor
        // draws again anyway. A resource whose only storage is the guest's pages holds nothing
        // this renderer could restore -- the VMM's RAM dump carries those bytes.
        //
        // The C excludes one more: an immutable buffer created without `GL_MAP_READ_BIT`, which
        // its readback path would log and then capture uninitialised. This tree cannot make one
        // -- `alloc_buffer` sets `GL_MAP_READ_BIT` whenever it sets any storage flag at all -- so
        // porting the exclusion would carry a rule that guards nothing.
        if res.args.nr_samples > 0 || matches!(res.storage, vrend::resource::Storage::Guest) {
            return None;
        }
        let last = if res.args.target == crate::vrend::pipe::TextureTarget::Buffer {
            0
        } else {
            res.args.last_level
        };
        let mut plan = Vec::new();
        for level in 0..=last {
            let region = transfer::level_region(res, level);
            // A format this tree cannot describe is a resource sized by guessing, which is worse
            // than one left out.
            let size = transfer::level_span(res, level)?;
            // The blob's header field is a dword, so a level that does not fit one cannot be
            // written down. Refused rather than truncated: a length that wrapped would name a
            // fraction of the bytes that follow it and every entry after would parse as garbage.
            let Ok(size) = usize::try_from(size)
                .map_err(|_| ())
                .and_then(|n| if n > u32::MAX as usize { Err(()) } else { Ok(n) })
            else {
                return None;
            };
            plan.push((level, region, size));
        }
        Some(plan)
    }

    /// Put a captured blob back into a classic context, after its journal has rebuilt the world
    /// the resources hang in.
    ///
    /// An entry naming a resource this context cannot reach is `dropped`, not an error: the
    /// journal is allowed to lose a create whose resource died around the snapshot, and this is
    /// that loss seen from the other side. What is an error is a blob that is not one.
    pub fn vrend_content_restore(
        &mut self,
        ctx: ContextId,
        blob: &[u8],
    ) -> Result<content::Account, Error> {
        if !self.is_classic(ctx) {
            return Err(Error::NoContext);
        }
        let entries = content::entries(blob).map_err(Error::MalformedContent)?;
        let mut account = content::Account::default();
        for e in entries {
            let own =
                match self.with_resource(e.res, |r| (r.iov.clone(), r.attached.contains(&ctx))) {
                    Some((own, true)) => own,
                    _ => {
                        account.dropped += 1;
                        continue;
                    }
                };
            let own = Iov::new(&own);
            let span = crate::guest_mem::HostSpan::new(e.bytes);
            let info = transfer::Info {
                level: e.level,
                stride: 0,
                layer_stride: 0,
                offset: 0,
                region: e.region,
                synchronized: true,
            };
            let v = self.vrend.as_mut().expect("a classic context needs vrend");
            match v.transfer(Some(ctx), e.res, true, Some(&own), &span.iov(), &info) {
                Ok(()) => account.entries += 1,
                Err(_) => account.skipped += 1,
            }
        }
        Ok(account)
    }

    /// One venus context's journal.
    ///
    /// The held set is the reachability the object table cannot see. A blob resource holds a share
    /// of the storage a venus allocation was published from, and that share keeps the bytes alive
    /// after the guest has freed the allocation -- so the resource goes on working, and a restore
    /// that did not rebuild the allocation would produce a dead blob where the original had a live
    /// one. Collected here because this is the only layer that can see both tables at once.
    pub fn venus_journal_export(&self, id: ContextId) -> Option<Vec<u8>> {
        let held = self.exported_allocations(id);
        self.venus.as_ref()?.journal_export(id, &held)
    }

    /// How many of `ctx`'s allocations a resource still holds a share of.
    ///
    /// Diagnostic. The harness's rebuild gate asks because it cannot compare a journal against a
    /// rebuilt one while this is non-zero: those entries are retained on the strength of resources
    /// holding the *original* context's allocations, and reproducing them in the rebuilt context
    /// means replaying the VMM's blob creates against it -- the fence, which that gate does not
    /// exercise. Better to decline the comparison and say why than to report a difference that is
    /// the gate's own gap.
    pub fn venus_held_allocations(&self, ctx: ContextId) -> usize {
        self.exported_allocations(ctx).len()
    }

    /// The allocations of `ctx` that a resource still holds a share of.
    ///
    /// Only `BlobStorage::Shared` holds one; a blob over the guest's own pages or over memory this
    /// renderer minted for it borrows nothing from a venus context and has no allocation to name.
    ///
    /// The context is resolved to its key first, and everything is matched on that. A resource
    /// outlives the context it was exported from, so the table can hold blobs of a context that is
    /// gone -- and if the VMM has since made a new context under the same id, matching on the id
    /// would hand the newcomer a set of keys into a dead guest's arena. An unknown id answers with
    /// the empty set, which is the truth: no live context, no allocations of one.
    fn exported_allocations(&self, ctx: ContextId) -> BTreeSet<ObjectKey> {
        // Its own scope: the context lock and the resource lock are taken one after the other and
        // never nested, which is the order the ring threads take them in too.
        let Some(key) = self.venus.as_ref().and_then(|v| v.with_context(ctx, |c| c.key())) else {
            return BTreeSet::new();
        };
        let table = self.resources.read().expect("the resource lock is never poisoned");
        table
            .values()
            .filter_map(|res| match &res.backing {
                Backing::Blob { storage: BlobStorage::Shared { from, .. }, .. }
                    if from.ctx == key =>
                {
                    Some(from.key)
                }
                _ => None,
            })
            .collect()
    }

    /// How far a venus context's journal has been written, for the VMM's fence.
    pub fn venus_journal_seq(&self, id: ContextId) -> Option<u64> {
        Some(self.venus.as_ref()?.journal_seq(id)?.0)
    }

    /// Hand a venus context the journal it will be rebuilt from.
    pub fn venus_journal_restore(
        &mut self,
        ctx: ContextId,
        bytes: &[u8],
    ) -> Result<usize, &'static str> {
        self.venus.as_mut().ok_or("no venus renderer")?.journal_restore(ctx, bytes)
    }

    /// Feed a venus context's retained commands up to `upto`.
    pub fn venus_replay_upto(&mut self, ctx: ContextId, upto: u64) -> Result<(), Error> {
        self.venus_mut()?.replay_upto(ctx, crate::venus::journal::Seq(upto)).map_err(venus_error)
    }

    /// What the venus recorder dropped, by command name, most-dropped first.
    pub fn venus_journal_transient(&self) -> Vec<(&'static str, u64)> {
        self.venus.as_ref().map(|v| v.journal_transient()).unwrap_or_default()
    }

    /// Begin rebuilding a classic context from its journal.
    pub fn vrend_replay_begin(&mut self, ctx: ContextId) -> bool {
        self.vrend.as_mut().is_some_and(|v| v.replay_begin(ctx))
    }

    /// Hand a classic context the journal it will be rebuilt from.
    pub fn vrend_journal_restore(
        &mut self,
        ctx: ContextId,
        bytes: &[u8],
    ) -> Result<usize, &'static str> {
        self.vrend.as_mut().ok_or("no classic renderer")?.journal_restore(ctx, bytes)
    }

    /// Feed a classic context's retained commands up to `upto`.
    pub fn vrend_replay_upto(&mut self, ctx: ContextId, upto: u64) -> bool {
        let table = self.resources.read().expect("the resource lock is never poisoned");
        let Some(v) = self.vrend.as_mut() else {
            return false;
        };
        v.replay_upto(ctx, &*table, crate::vrend::journal::Seq(upto))
    }

    /// Finish rebuilding a classic context, and report what it could not use.
    pub fn vrend_replay_end(&mut self, ctx: ContextId) -> bool {
        self.vrend.as_mut().is_some_and(|v| v.replay_end(ctx))
    }

    /// Each classic context's journal size and entry count, round-tripped.
    pub fn vrend_journal_report(&self) -> Vec<(ContextId, usize, Result<usize, &'static str>)> {
        self.vrend.as_ref().map(|v| v.journal_report()).unwrap_or_default()
    }

    /// What the classic renderer has retained for a rebuild.
    pub fn journal_census(&self) -> crate::vrend::journal::Census {
        self.vrend.as_ref().map(|v| v.journal_census()).unwrap_or_default()
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
    pub fn venus_memory_census(&self, ctx_id: ContextId) -> Result<Vec<Allocation>, Error> {
        self.venus_context(ctx_id, |ctx| ctx.driver().memory_census())
    }

    /// Publish one venus allocation to the VMM: the host address, the storage behind it, and the
    /// generational name of the object that lent it.
    ///
    /// All three, because a blob resource needs all three and none of them can be derived from
    /// another later. The address is what the VMM maps; the storage is a share of the pages
    /// themselves, so the mapping stays good after the guest's `vkFreeMemory` retires the
    /// allocation; and the key is what a journal entry and a held-allocation census name the
    /// object by, which an id alone cannot do once the guest reuses it.
    pub fn venus_memory_export(
        &mut self,
        ctx_id: ContextId,
        mem: BlobId,
        blob_size: u64,
    ) -> Result<(Exported, Storage, Exporter), Error> {
        let v = self.venus.as_ref().ok_or(Error::RendererAbsent)?;
        // The exporter is named here and not by the caller, because this is where the live
        // context is in hand: an id the caller passed says which context it *meant*, and the key
        // says which one it got.
        v.with_context_mut(ctx_id, |ctx| {
            let ctx_key = ctx.key();
            ctx.memory_export(ObjectId(mem.0), blob_size)
                .map(|(exported, storage, key)| (exported, storage, Exporter { ctx: ctx_key, key }))
        })
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
            Backing::Blob { .. } | Backing::Classic { .. } | Backing::Imported { .. } => None,
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
    /// Answered from what the resource holds, and from nothing else. Both shapes are things the
    /// resource keeps alive -- its own pages, or a share of an allocation's storage -- so the
    /// address is good for as long as the resource is, whatever the guest has since done to the
    /// allocation it was published from. There is no longer a case that has to be re-checked
    /// against the exporting context, because there is no longer an address this renderer
    /// hands out without owning.
    pub fn resource_host_mapping(&self, handle: ResourceHandle) -> Result<HostMapping, Error> {
        self.with_resource(handle, |r| match &r.backing {
            Backing::Blob { desc, storage } => match storage {
                // The mapping answers for its own extent: `desc.size` is what was asked for,
                // this is what was mapped, and it is what bounds what may be read. The host
                // maps its own shm write-back and coherent, like any anonymous page.
                BlobStorage::Minted(h) => Some(HostMapping {
                    addr: h.map.host_addr(),
                    size: h.map.len() as u64,
                    caching: Caching::Cached,
                }),
                // The blob's size, not the storage's: the export held the one to the other.
                BlobStorage::Shared { storage, caching, .. } => {
                    Some(HostMapping { addr: storage.span().0, size: desc.size, caching: *caching })
                }
                BlobStorage::Guest => None,
            },
            // A classic resource is mappable only when something took its mapping, which only
            // the blob claim does. Asked of vrend rather than kept here: vrend owns the buffer,
            // so it owns the address into it, and a copy on this side would outlive the
            // buffer by exactly as long as it took someone to forget to clear it.
            Backing::Classic { .. } => self.vrend.as_ref().and_then(|v| {
                let (addr, size) = v.resource_mapping(handle)?;
                // Write-back: these are ordinary driver pages the host reaches through a
                // coherent mapping, which is the C's `inferred_gl_caching_type` for a buffer
                // it did not get from gbm.
                Some(HostMapping { addr, size, caching: Caching::Cached })
            }),
            Backing::Imported { .. } => None,
        })
        .ok_or(Error::NoResource)?
        .ok_or(Error::NotMappable)
    }

    /// Copy one allocation's contents out, returning how many bytes landed in `buf`.
    ///
    /// Named by the same [`ObjectId`] the census reported it under, and not by a bare integer:
    /// the two are one value, and the shim is where it is spelled as a number.
    pub fn venus_memory_read(
        &self,
        ctx_id: ContextId,
        mem: ObjectId,
        buf: &mut [u8],
    ) -> Result<usize, Error> {
        self.venus_context(ctx_id, |ctx| ctx.memory_read(mem, buf))?.map_err(memory_error)
    }

    /// Copy one allocation's contents back in, returning how many bytes landed.
    ///
    /// The restore half of [`Self::venus_memory_read`], and the only writer of an allocation's
    /// bytes that is not the guest. What it puts back is what that read handed out, so the two
    /// reach the memory by the same routes and neither knows a route the other does not.
    pub fn venus_memory_write(
        &self,
        ctx_id: ContextId,
        mem: ObjectId,
        src: &[u8],
    ) -> Result<usize, Error> {
        self.venus_context(ctx_id, |ctx| ctx.memory_write(mem, src))?.map_err(memory_error)
    }

    /// One venus context's sync objects, for the VMM to store beside its journal.
    ///
    /// Cannot fail beyond "no such context": everything it reads is a poll or a record this
    /// renderer already keeps, so a suspend is never refused on account of the GPU being busy.
    pub fn venus_sync_export(&self, ctx_id: ContextId) -> Result<Vec<u8>, Error> {
        self.venus_context(ctx_id, venus::context::Context::sync_export)
    }

    /// Put a captured sync state back, after the context's journal has replayed and before its
    /// rings start.
    pub fn venus_sync_restore(
        &self,
        ctx_id: ContextId,
        blob: &[u8],
    ) -> Result<venus::sync::Account, Error> {
        let v = self.venus.as_ref().ok_or(Error::RendererAbsent)?;
        v.with_context_mut(ctx_id, |ctx| ctx.sync_restore(blob))
            .ok_or(Error::NoContext)?
            .map_err(Error::MalformedSync)
    }

    /// The venus context under an id, distinguishing "no venus in this build" from "no such
    /// context" -- which a caller asking for a snapshot needs to tell apart.
    fn venus_context<R>(
        &self,
        ctx_id: ContextId,
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
    /// Some allocation key, for the tests whose subject is not an allocation's lifetime.
    ///
    /// `Exporter` names the allocation a blob came from, and every blob has one -- but a test about
    /// which contexts may reach a share does not care which allocation that was. Minting one here
    /// keeps those tests saying only what they are about.
    fn any_key() -> super::ObjectKey {
        let mut t = crate::venus::objects::Table::new();
        t.add(
            crate::venus::cs::ObjectId(1),
            crate::venus::proto::types::VkObjectType::VK_OBJECT_TYPE_DEVICE_MEMORY,
            crate::venus::cs::HostHandle(1),
            None,
        )
        .expect("a fresh table takes any id");
        t.key_of(crate::venus::cs::ObjectId(1)).expect("just added")
    }

    use super::*;

    /// A sink that goes nowhere. Nothing here retires a fence; the renderer needs one to exist.
    struct NoSink;
    impl FenceSink for NoSink {
        fn context_fence(&mut self, _ctx: ContextId, _ring: RingIdx, _fence: FenceId) {}
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

        // A blob naming something a context already holds gets no memory of its own. There is
        // no such context in this build, so the resolution is refused before either renderer is
        // reached -- what is being asked here is that the refusal came from resolving the id and
        // not from minting something first.
        let exported = BlobDesc {
            source: BlobSource::InContext { ctx: ContextId::new(1).unwrap(), id: BlobId(9) },
            ..minted
        };
        assert_eq!(
            r.resource_create_blob(ResourceHandle::new(2).unwrap(), exported, Vec::new()),
            Err(Error::NoContext),
            "an id means what its context says; minting would answer with the wrong bytes"
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
        use crate::venus::budget::Account;
        use crate::venus::ring::ShmResources;

        let one = ContextId::new(1).unwrap();
        let two = ContextId::new(2).unwrap();
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
        let pages = Storage::pages_for_test(4096, &Account::for_test(None));
        let exported = |attached: Vec<ContextId>| Resource {
            handle: blob,
            backing: Backing::Blob {
                desc: BlobDesc {
                    blob_mem: crate::abi::BLOB_MEM_HOST3D,
                    blob_flags: 1,
                    source: BlobSource::InContext { ctx: one, id: BlobId(66) },
                    size: 4096,
                },
                storage: BlobStorage::Shared {
                    storage: pages.clone(),
                    caching: Caching::Cached,
                    from: Exporter { ctx: ContextKey::for_test(one), key },
                },
            },
            iov: Vec::new(),
            priv_: VmmPtr(core::ptr::null_mut()),
            attached,
        };

        let mut table = BTreeMap::new();
        table.insert(blob, exported(vec![one]));
        assert!(
            matches!(table.bytes(one, blob), Some(ResourceBytes::Shared(s)) if s == pages),
            "the context it is attached to finds the storage the resource holds"
        );
        assert!(
            table.bytes(two, blob).is_none(),
            "a context the guest never attached it to reaches nothing, whoever exported it"
        );

        // Attached to both, and reached by both. Attachment is the whole gate: the storage is a
        // share, so it is not the exporting context's to be the only one that can resolve it.
        table.insert(blob, exported(vec![one, two]));
        assert!(
            matches!(table.bytes(two, blob), Some(ResourceBytes::Shared(s)) if s == pages),
            "and once the guest attaches it, the other context reaches the same bytes"
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

    /// A classic resource whose storage is an IOSurface lends a share of it to a venus context,
    /// and one whose storage is an ordinary GL texture lends nothing.
    ///
    /// This is the classic-into-venus import: a GL client's window buffer reaching the Vulkan
    /// compositor that composites it. What it must lend is the surface itself and not its id --
    /// the system recycles ids, so an importer holding one would resolve to a stranger's surface
    /// the moment this resource died -- and the span it lends must be the surface's own, because
    /// that is what the driver is handed as a host pointer.
    #[test]
    fn a_classic_resource_lends_the_surface_its_storage_is() {
        use crate::metal::{Held, PixelFormat, Surface};
        use crate::venus::driver::Storage;
        use crate::venus::ring::ShmResources;

        let one = ContextId::new(1).unwrap();
        let two = ContextId::new(2).unwrap();
        let window = ResourceHandle::new(1).unwrap();
        let plain = ResourceHandle::new(2).unwrap();

        let surface = Surface::plain(800, 600, PixelFormat::Bgra).expect("the system minted");
        let (addr, extent) = (surface.host_addr(), surface.alloc_size());
        // A share of a real surface, which is what vrend's EGL image holds. No EGL here: the
        // image is how the classic side comes by the share, not what makes it one.
        let held: Arc<dyn Held> = Arc::new(surface);

        let args = ClassicArgs {
            target: crate::vrend::pipe::TextureTarget::Texture2d,
            format: crate::vrend::proto::Format::from_wire(1).expect("a format"),
            bind: crate::vrend::resource::Bind::SHARED,
            width: 800,
            height: 600,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: crate::vrend::resource::ResourceFlags::default(),
        };
        let classic = |handle, surface, attached| Resource {
            handle,
            backing: Backing::Classic { args, surface },
            iov: Vec::new(),
            priv_: VmmPtr(core::ptr::null_mut()),
            attached,
        };

        let mut table = BTreeMap::new();
        table.insert(window, classic(window, Some(Arc::clone(&held)), vec![one, two]));
        table.insert(plain, classic(plain, None, vec![one, two]));

        let Some(ResourceBytes::Shared(lent)) = table.bytes(two, window) else {
            panic!("the compositor's context imports the client's window buffer");
        };
        assert_eq!(lent.span(), (addr, extent), "and it is the surface's own pages it imports");
        assert_eq!(
            lent,
            Storage::lent(Arc::clone(&held)),
            "the same storage, not a second one describing the same surface"
        );
        assert!(
            lent.held().is_some(),
            "a share, so the surface outlives the classic context that made it"
        );

        assert!(
            table.bytes(two, plain).is_none(),
            "a classic resource with GL storage has no host address to import"
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

        let one = ContextId::new(1).unwrap();
        let two = ContextId::new(2).unwrap();
        let three = ContextId::new(3).unwrap();
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
                        source: BlobSource::InContext { ctx: one, id: BlobId(66) },
                        size: 2048,
                    },
                    storage: BlobStorage::Shared {
                        storage: share.clone(),
                        caching: Caching::Cached,
                        from: Exporter { ctx: ContextKey::for_test(one), key: any_key() },
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
                        source: BlobSource::InContext { ctx: one, id: BlobId(67) },
                        size: 4096,
                    },
                    storage: BlobStorage::Shared {
                        storage: pages.clone(),
                        caching: Caching::Cached,
                        from: Exporter { ctx: ContextKey::for_test(one), key: any_key() },
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
    /// A blob's mapping outlives the `vkFreeMemory` that retires the allocation it was published
    /// from -- which is the fix, and the thing the guest can otherwise weaponise.
    ///
    /// The sequence is legal and unremarkable: allocate host-visible memory, `CREATE_BLOB` over
    /// it, `RESOURCE_MAP_BLOB` so the VMM `hv_vm_map`s that host address into guest physical
    /// memory, then `vkFreeMemory`. When the address was the driver's own `vkMapMemory` pointer,
    /// the free unmapped and released it and the hypervisor's mapping was left pointing at host
    /// memory this process had given back. The C has the same hole, guarded by a comment.
    ///
    /// It is a lifetime bug, so the fix is structural rather than another purge: the storage is
    /// pages this renderer minted, the resource holds a share of them, and the free retires the
    /// record while the pages stand. There is no destroy site left that could get this wrong,
    /// because there is no longer an address handed out that this renderer does not own.
    #[test]
    fn a_blobs_mapping_survives_the_guest_freeing_the_allocation() {
        use crate::venus::budget::Account;

        const MEM: ObjectId = ObjectId(66);
        let mut r = renderer(Config { venus: true, ..Config::default() });
        let one = ContextId::new(1).unwrap();
        r.context_create(one, CapsetId::Venus, "exporter".into()).expect("a fresh id");

        let account = Account::for_test(None);
        let pages = Storage::pages_for_test(4096, &account);
        let addr = pages.span().0;

        let blob = ResourceHandle::new(5).unwrap();
        r.insert(
            blob,
            Backing::Blob {
                desc: BlobDesc {
                    blob_mem: crate::abi::BLOB_MEM_HOST3D,
                    blob_flags: 1,
                    source: BlobSource::InContext { ctx: one, id: BlobId(MEM.0) },
                    size: 4096,
                },
                storage: BlobStorage::Shared {
                    storage: pages,
                    caching: Caching::Cached,
                    from: Exporter { ctx: ContextKey::for_test(one), key: any_key() },
                },
            },
            Vec::new(),
        );
        let mapping = r.resource_host_mapping(blob).expect("the blob publishes its pages");
        assert_eq!(mapping.addr, addr);

        // The whole context goes -- more than a `vkFreeMemory`, and the strictest version of the
        // question: the arena, the driver and the device are all gone.
        r.context_destroy(one);
        assert_eq!(
            r.resource_host_mapping(blob),
            Ok(mapping),
            "the address the hypervisor is mapped over is still the resource's own"
        );
    }

    /// A context id the VMM reuses does not inherit the last occupant's blobs.
    ///
    /// The state is ordinary: a guest destroys a context and the VMM stands another up under the
    /// same id, while a blob exported from the first is still a live resource -- resources are
    /// device-wide and outlive the context that exported them, which is the whole reason the
    /// share exists. Held allocations decide what a snapshot's journal retains, so attributing
    /// the old context's allocations to the new one would export a journal keeping entries for
    /// objects in a dead guest's arena.
    ///
    /// The key is what makes it impossible, rather than a purge at `context_destroy` that a
    /// future path could skip: the record names the occupant, and the next occupant of that id is
    /// simply not it.
    #[test]
    fn a_reused_context_id_does_not_inherit_the_previous_contexts_blobs() {
        use crate::venus::budget::Account;

        let mut r = renderer(Config { venus: true, ..Config::default() });
        let two = ContextId::new(2).unwrap();
        r.context_create(two, CapsetId::Venus, "first".into()).expect("a fresh id");
        let first = r
            .venus
            .as_ref()
            .and_then(|v| v.with_context(two, |c| c.key()))
            .expect("the context was just created");

        let blob = ResourceHandle::new(5).unwrap();
        let pages = Storage::pages_for_test(4096, &Account::for_test(None));
        r.insert(
            blob,
            Backing::Blob {
                desc: BlobDesc {
                    blob_mem: crate::abi::BLOB_MEM_HOST3D,
                    blob_flags: 1,
                    source: BlobSource::InContext { ctx: two, id: BlobId(66) },
                    size: 4096,
                },
                storage: BlobStorage::Shared {
                    storage: pages,
                    caching: Caching::Cached,
                    from: Exporter { ctx: first, key: any_key() },
                },
            },
            Vec::new(),
        );
        assert_eq!(r.venus_held_allocations(two), 1, "its own context holds it");

        r.context_destroy(two);
        r.context_create(two, CapsetId::Venus, "second".into()).expect("the id is free again");
        assert_eq!(
            r.venus_held_allocations(two),
            0,
            "the new occupant of the id holds nothing the old one exported"
        );
        assert!(
            r.resource_host_mapping(blob).is_ok(),
            "and the blob still publishes its pages, which is what it holds the share for"
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
        let one = ContextId::new(1).unwrap();
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
                    source: BlobSource::InContext { ctx: one, id: BlobId(66) },
                    size: 4096,
                },
                storage: BlobStorage::Shared {
                    storage: pages.clone(),
                    caching: Caching::Cached,
                    from: Exporter { ctx: ContextKey::for_test(one), key: any_key() },
                },
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
        let ctx = ContextId::new(1).unwrap();
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
