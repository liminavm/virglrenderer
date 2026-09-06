// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The classic renderer's root: the winsys, the driver, what it can do, and every GL object the
//! guests own.
//!
//! One `Vrend` per renderer, owned by it. It holds the winsys and the context every API-path
//! operation runs on (the C's `ctx0`), the feature and format tables probed once at init, the
//! host side of every classic resource keyed by the handle the renderer files it under, and one
//! GL context per guest context, all sharing ctx0's objects.
//!
//! Which context is current is tracked here and switched only when it changes, the way
//! `vrend_hw_switch_context` does: every entry point names the context it needs, and the switch
//! is one place rather than a habit.

use super::blitter;
use super::caps;
use super::context::{Context, Current, Fault, Guest, Host, Todo};
use super::egl::{self, EglError, Flavour, Version, Winsys};
use super::features::{Feature, Features};
use super::formats::Table;
use super::gl::Gl;
use super::gl::gles::GL_VERSION;
use super::journal::{Census, Seq};
use super::resource::{self, Args, Limits, Refusal, Resource};
use super::shader;
use super::transfer::{self, Info};
use crate::config::Config;
use crate::guest_mem::{Iov, PixelSource};
use crate::ids::{BlobId, ContextId, ResourceHandle};
use crate::metal;
use crate::videotoolbox;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// Why the classic renderer could not come up.
#[derive(Debug)]
pub enum InitError {
    Egl(EglError),
    /// No GLES 3.x context could be made.
    NoContext,
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InitError::Egl(e) => write!(f, "{e}"),
            InitError::NoContext => f.write_str("no GLES 3.x context could be created"),
        }
    }
}

impl From<EglError> for InitError {
    fn from(e: EglError) -> InitError {
        InitError::Egl(e)
    }
}

pub struct Vrend {
    winsys: Winsys,
    gl: Gl,
    pub features: Features,
    pub formats: Table,
    pub limits: Limits,
    /// What the translator may assume of the host, read once alongside the limits.
    shader_cfg: shader::Config,
    /// What the guest's driver is told of the host, probed once from the same answers.
    caps: caps::CapsV2,
    /// What VideoToolbox decodes here, or `None` when the caller did not ask for video.
    ///
    /// Probing is what registers the supplemental decoders, so this is also the record that
    /// registration happened: nothing can ask this host about a codec without holding one.
    video: Option<videotoolbox::Support>,
    ctx0: egl::Context,
    /// The version guest contexts are made with: the newest the driver gave ctx0.
    version: Version,
    current: Current,
    resources: BTreeMap<ResourceHandle, resource::Slot>,
    contexts: BTreeMap<ContextId, Context>,
    pub todo: Todo,
    /// The shader blitter and its GL context, built on the first blit that needs one. A renderer
    /// that never takes the blitter's path never pays for it.
    blitter: Option<blitter::Blitter>,
    /// Texture storage the guest has freed that something else still holds a share of.
    ///
    /// The share cannot delete itself when the last holder lets go, because deleting needs a
    /// current GL context and the driver, and a drop has neither. So it is parked here and swept
    /// from the next place that has both.
    doomed: Vec<Arc<resource::Texture>>,
    /// Which resources copy guest pages, as of the batch it was last asked in.
    pixels: resource::Refresh,
    /// Batches run, ever. The unit a copy of a guest's pages is kept fresh in: within one batch
    /// the guest has had no opportunity to run, so one read serves every draw in it.
    batch: u64,
}

/// The versions tried, newest first -- the GLES rows of the C's `gl_versions` ladder.
const VERSIONS: [Version; 3] = [
    Version { major: 3, minor: 2 },
    Version { major: 3, minor: 1 },
    Version { major: 3, minor: 0 },
];

/// Why a `RESOURCE_CREATE_BLOB` naming a classic context's blob id was refused.
///
/// Each is the guest's error and each wants a different investigation, which is why they are not
/// one "invalid": nothing described the id at all says the `PIPE_RESOURCE_CREATE` never arrived
/// or already went to another claim; a size mismatch says the guest's two halves disagree about
/// how big its allocation is; unmappable says the host built the buffer and the driver would not
/// hand its pages over.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ClaimRefused {
    /// No resource stands under that id in that context.
    NotDescribed,
    /// The blob is larger than the resource backing it.
    Oversize { asked: u64, allocated: u32 },
    /// The driver refused the persistent mapping, or the storage never admitted one.
    Unmappable,
}

/// Whether a blob of `size` bytes may be published from a resource of `width` bytes.
///
/// The wire carries the two independently and the C reconciles neither -- `vrend_get_blob_pipe`
/// takes `blob_size` as `UNUSED`. They are one fact with two spellings, and the guest is the only
/// thing that can make them disagree: asking to publish more than was allocated is asking the VMM
/// to map whatever the driver put after the buffer. Refused here, where both halves are in hand,
/// and never clamped -- a clamp reports success for a mapping the guest did not ask for and will
/// index past.
///
/// Smaller is not a mismatch. A guest may publish part of what it allocated, and the mapping it
/// is given is still bounded by the buffer.
fn publishable(size: u64, width: u32) -> Result<(), ClaimRefused> {
    if size > width as u64 {
        return Err(ClaimRefused::Oversize { asked: size, allocated: width });
    }
    Ok(())
}

impl fmt::Display for ClaimRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClaimRefused::NotDescribed => write!(f, "no resource was described under that blob id"),
            ClaimRefused::Oversize { asked, allocated } => {
                write!(f, "asked to publish {asked} bytes of a {allocated}-byte resource")
            }
            ClaimRefused::Unmappable => write!(f, "the buffer's pages cannot be mapped"),
        }
    }
}

impl Vrend {
    /// Open the winsys, bring ctx0 up on this thread and probe the driver.
    pub fn new(config: Config) -> Result<Vrend, InitError> {
        let winsys = Winsys::open(Flavour::Gles)?;
        let mut ctx0 = None;
        for v in VERSIONS {
            if let Ok(c) = winsys.create_context(v, None) {
                ctx0 = Some((c, v));
                break;
            }
        }
        let (ctx0, version) = ctx0.ok_or(InitError::NoContext)?;
        winsys.make_current(&ctx0)?;
        let gl = Gl::new(winsys.gles());
        let version_string = gl.get_string(GL_VERSION);
        let gles_version = parse_gles_version(&version_string);
        let mut features = Features::probe(gles_version, gl.extensions());
        if !winsys.has_extension("EGL_KHR_gl_colorspace") {
            features.clear(Feature::srgb_write_control);
        }
        features.reconcile(&gl);
        let limits = Limits::query(&gl, &features);
        let shader_cfg = shader::Config::probe(&gl, &features, &limits);
        let formats = Table::probe(&gl, &features);
        let video = config.video.then(videotoolbox::Support::probe);
        if let Some(support) = video {
            let names: Vec<&str> = videotoolbox::Codec::ALL
                .iter()
                .filter(|c| support.decodes(**c))
                .map(|c| c.name())
                .collect();
            eprintln!(
                "[virglrs] vrend: hardware video decode {}",
                if names.is_empty() {
                    "UNAVAILABLE -- this host has silicon for none of the codecs we serve".into()
                } else {
                    names.join(" ")
                },
            );
        }
        let caps = caps::CapsV2::probe(&gl, &features, &limits, &formats, video.as_ref());
        eprintln!(
            "[virglrs] vrend: {version_string} (gles {gles_version}), {} formats, {} features, \
             iosurface storage {}",
            formats.entries().count(),
            features.present().count(),
            if features.adopts_iosurfaces() {
                "available"
            } else {
                "UNAVAILABLE -- no scanout              or shared buffer can be imported without a copy, and every one will be blank"
            },
        );
        Ok(Vrend {
            winsys,
            gl,
            features,
            formats,
            limits,
            shader_cfg,
            caps,
            video,
            ctx0,
            version,
            current: Current::Ctx0,
            resources: BTreeMap::new(),
            contexts: BTreeMap::new(),
            todo: Todo::default(),
            blitter: None,
            doomed: Vec::new(),
            batch: 0,
            pixels: resource::Refresh::default(),
        })
    }

    /// The classic capsets, as probed at init.
    pub fn caps(&self) -> &caps::CapsV2 {
        &self.caps
    }

    /// What this host decodes in hardware, or `None` when the caller did not ask for video.
    ///
    /// `None` and a support that decodes nothing are different answers and stay different: the
    /// first is a configuration, the second is this machine's silicon.
    pub fn video(&self) -> Option<&videotoolbox::Support> {
        self.video.as_ref()
    }

    pub fn gl(&self) -> &Gl {
        &self.gl
    }

    /// Make ctx0 current, if it is not already.
    fn switch_ctx0(&mut self) {
        if self.current != Current::Ctx0 {
            self.winsys.make_current(&self.ctx0).expect("ctx0 was current once and still exists");
            self.current = Current::Ctx0;
        }
    }

    /// The host a context's commands run against, and the contexts beside it: two disjoint
    /// borrows of this renderer, so a context can run against the rest of it.
    fn split<'a>(
        &'a mut self,
        ctx: ContextId,
        guest: &'a dyn Guest,
    ) -> (Host<'a>, &'a mut BTreeMap<ContextId, Context>) {
        let Vrend {
            winsys,
            gl,
            features,
            formats,
            limits,
            shader_cfg,
            caps: _,
            video,
            ctx0,
            version,
            current,
            resources,
            contexts,
            todo,
            blitter,
            doomed: _,
            batch,
            pixels,
        } = self;
        let host = Host {
            batch: *batch,
            gl,
            winsys,
            version: *version,
            share: ctx0,
            features,
            formats,
            limits,
            shader_cfg,
            resources,
            pixels,
            guest,
            ctx,
            current,
            todo,
            blitter,
            video: video.as_ref(),
        };
        (host, contexts)
    }

    /// What has been retained for a rebuild: every live context's objects and current state,
    /// plus the type each live blob was given.
    ///
    /// The resources are counted here and not in `Context` because that is where they live -- one
    /// table shared by every context, so no single context can answer for it.
    pub fn journal_census(&self) -> Census {
        let mut c = Census::default();
        for ctx in self.contexts.values() {
            c += ctx.journal_census();
        }
        for wire in self.resource_preamble() {
            c.add_wire(wire.len(), true);
        }
        c
    }

    /// One context's journal, as the bytes the VMM stores and hands back.
    ///
    /// `None` for a context that is not here. An empty journal still serializes: a context that
    /// built nothing is a fact worth restoring accurately, and the alternative -- answering
    /// "no journal" -- is what the VMM reads as "this context is not mine to rebuild".
    pub fn journal_export(&self, id: ContextId) -> Option<Vec<u8>> {
        let ctx = self.contexts.get(&id)?;
        Some(crate::vrend::journal::serialize(&ctx.journal(self.resource_preamble())))
    }

    /// The commands a rebuild must send before anything else: what described each claimed blob,
    /// and what typed each attached one.
    ///
    /// They live on the resource table rather than on a context because one table serves every
    /// context, so no single context can walk it. Describe before type, per resource: a resource
    /// has at most one of the two -- a described one arrives with its shape, an attached one is
    /// told its shape later -- so the order between them never actually arises, and stating it is
    /// cheaper than relying on that staying true.
    fn resource_preamble(&self) -> impl Iterator<Item = &Vec<u32>> {
        self.resources
            .values()
            .filter_map(|s| s.resource())
            .flat_map(|r| r.described_by.as_ref().into_iter().chain(r.typed_by.as_ref()))
    }

    /// Each live context's journal: how many bytes it exports, and how many entries those bytes
    /// read back as.
    ///
    /// The round-trip is the point. A census counts what was retained, which a serializer bug
    /// would leave untouched; parsing our own output back is the cheapest thing that actually
    /// exercises the format on a real world rather than on a fixture we wrote.
    pub fn journal_report(&self) -> Vec<(ContextId, usize, Result<usize, &'static str>)> {
        self.contexts
            .keys()
            .filter_map(|id| {
                let bytes = self.journal_export(*id)?;
                let read_back = crate::vrend::journal::parse(&bytes).map(|e| e.len());
                Some((*id, bytes.len(), read_back))
            })
            .collect()
    }

    /// Begin rebuilding a classic context. `false` if it is not here.
    pub fn replay_begin(&mut self, id: ContextId) -> bool {
        match self.contexts.get_mut(&id) {
            Some(c) => {
                c.replay_begin();
                true
            }
            None => false,
        }
    }

    /// Hand a classic context the journal it will be rebuilt from.
    pub fn journal_restore(&mut self, id: ContextId, bytes: &[u8]) -> Result<usize, &'static str> {
        self.contexts.get_mut(&id).ok_or("no such context")?.replay_restore(bytes)
    }

    /// Feed a classic context's retained commands up to `upto`.
    pub fn replay_upto(&mut self, id: ContextId, guest: &dyn Guest, upto: Seq) -> bool {
        if !self.contexts.contains_key(&id) {
            return false;
        }
        let (mut host, contexts) = self.split(id, guest);
        contexts.get_mut(&id).expect("checked above").replay_upto(&mut host, upto);
        true
    }

    /// Finish rebuilding a classic context, and report what it could not use.
    pub fn replay_end(&mut self, id: ContextId) -> bool {
        match self.contexts.get_mut(&id) {
            Some(c) => {
                c.replay_end();
                true
            }
            None => false,
        }
    }

    // ---- contexts ----

    pub fn context_create(&mut self, id: ContextId, guest: &dyn Guest) -> Result<(), EglError> {
        let (mut host, contexts) = self.split(id, guest);
        let c = Context::new(&mut host)?;
        contexts.insert(id, c);
        Ok(())
    }

    pub fn context_destroy(&mut self, id: ContextId, guest: &dyn Guest) {
        let (mut host, contexts) = self.split(id, guest);
        if let Some(c) = contexts.remove(&id) {
            c.destroy(&mut host);
        }
        self.switch_ctx0();
        // The framebuffers that went with the context were the last holders of any storage the
        // guest freed while they drew into it.
        self.sweep_doomed();
    }

    pub fn has_context(&self, id: ContextId) -> bool {
        self.contexts.contains_key(&id)
    }

    /// Run a batch on a context. `None` for a context this renderer does not have.
    pub fn submit(
        &mut self,
        id: ContextId,
        words: &[u32],
        guest: &dyn Guest,
    ) -> Option<Result<(), Fault>> {
        // One tick per batch, before anything in it runs. It is what says a guest has had no
        // opportunity to rewrite its pages since a copy of them was taken -- see
        // [`resource::GuestPixels`] -- so it must move exactly when that stops being true.
        self.batch += 1;
        let (mut host, contexts) = self.split(id, guest);
        contexts.get_mut(&id).map(|c| c.submit(&mut host, words))
    }

    // ---- resources ----

    /// Create the host side of a classic resource, on ctx0.
    pub fn resource_create(&mut self, handle: ResourceHandle, args: Args) -> Result<(), Refusal> {
        assert!(!self.resources.contains_key(&handle), "the renderer checked the handle was free");
        self.switch_ctx0();
        let res = Resource::create(
            &self.gl,
            &self.winsys,
            &self.features,
            &self.formats,
            &self.limits,
            args,
        )?;
        self.resources.insert(handle, resource::Slot::Resource(res));
        Ok(())
    }

    /// Claim the resource a classic context described under `blob`, giving it `handle` and a
    /// persistent host mapping.
    ///
    /// `vrend_get_blob_pipe` plus `vrend_renderer_resource_map`, as one step. The C leaves them
    /// apart and the VMM maps later, which means a resource can exist in the table having refused
    /// the only thing it was created to do. Here the map is part of the claim: a buffer that
    /// cannot be mapped is not published, and `CREATE_BLOB` says so to the guest that asked.
    ///
    /// `size` is what the guest asked to publish, and `args.width` is what was allocated. The
    /// wire lets them disagree -- the C ignores `blob_size` entirely -- and a guest asking to
    /// publish more than it allocated is asking the VMM to map whatever follows the buffer into
    /// its address space. Refused, never clamped.
    pub fn claim_described(
        &mut self,
        ctx: ContextId,
        blob: BlobId,
        handle: ResourceHandle,
        size: u64,
    ) -> Result<Args, ClaimRefused> {
        assert!(!self.resources.contains_key(&handle), "the renderer checked the handle was free");
        let (mut res, wire) = self
            .contexts
            .get_mut(&ctx)
            .and_then(|c| c.claim_described(blob))
            .ok_or(ClaimRefused::NotDescribed)?;
        let args = res.args;
        // Any GL context of the share group can map the buffer -- ctx0 is the one that is always
        // there, and using it means the answer does not depend on which sub-context the guest
        // happened to leave current.
        self.switch_ctx0();
        let refused = publishable(size, args.width)
            .err()
            .or_else(|| (!res.map_persistent(&self.gl)).then_some(ClaimRefused::Unmappable));
        if let Some(why) = refused {
            // Nothing can be attached to it: it has had no handle, so no view, framebuffer or
            // transfer has ever been able to name it. `destroy` handing storage back here would
            // mean a described resource had been reachable, which is the thing this path exists
            // to prevent.
            assert!(
                res.destroy(&self.gl).is_none(),
                "a resource with no handle is attached to nothing"
            );
            return Err(why);
        }
        res.described_by = Some(wire);
        self.resources.insert(handle, resource::Slot::Resource(res));
        Ok(args)
    }

    /// Where a claimed resource's buffer is mapped, and how far it runs.
    ///
    /// `None` for every resource that was never published to a guest, which is every ordinary
    /// classic one: the address exists only because [`Self::claim_described`] took it.
    pub fn resource_mapping(&self, handle: ResourceHandle) -> Option<(usize, u64)> {
        let res = self.resource(handle)?;
        Some((res.mapped?, res.args.width as u64))
    }

    /// A blob attached to a classic context: storage, and nothing yet that says what it is.
    ///
    /// A blob reaches vrend only here. It is created without a type -- no format, no extent --
    /// so there is nothing for `resource_create` to make, and the `SET_TYPE` that describes it
    /// arrives later in the command stream. Holding the storage from the attach is what gives
    /// that command something to adopt, and what lets a handle touched in between fault as
    /// untyped rather than as absent.
    ///
    /// Idempotent: the guest may attach one resource to several contexts, and a later attach
    /// must not un-type what an earlier one's `SET_TYPE` already settled.
    pub fn resource_attach_blob(
        &mut self,
        handle: ResourceHandle,
        surface: Option<Arc<dyn metal::Held>>,
    ) {
        self.resources
            .entry(handle)
            .or_insert_with(|| resource::Slot::Untyped(resource::Untyped::new(surface)));
    }

    /// The IOSurface a resource is presented from, if its storage is one. Asked of the resource
    /// every time: the surface goes with the resource, and there is no other place to hold one.
    pub fn resource_surface(&self, handle: ResourceHandle) -> Option<&metal::Surface> {
        self.resources.get(&handle)?.resource()?.surface()
    }

    /// A share of that surface, for a holder outside vrend -- a venus context importing this
    /// resource, which must keep the surface alive rather than name it. See
    /// [`resource::Resource::surface_share`].
    pub fn resource_surface_share(&self, handle: ResourceHandle) -> Option<Arc<dyn metal::Held>> {
        self.resources.get(&handle)?.resource()?.surface_share()
    }

    /// `vrend_renderer_resource_sync_iosurface`: make a surface-backed resource's contents whole
    /// before the surface is presented. The texture's storage *is* the surface, so there is
    /// nothing to copy -- only the renders queued into it to complete, since the present that
    /// follows reads the bytes on another queue. `false` for a resource with no surface, which
    /// is the caller's cue to read the pixels back instead.
    ///
    /// The renders live on the queue of whichever sub-context drew them, and a finish waits for
    /// one context's queue only. The C finishes ctx0, which never draws, and the harness caught
    /// it reading the frame before last off a scanout: every GL context this renderer owns is
    /// finished, so the surface is whole whoever rendered into it.
    pub fn resource_sync_iosurface(&mut self, handle: ResourceHandle) -> bool {
        if self.resource_surface(handle).is_none() {
            return false;
        }
        for (id, ctx) in &self.contexts {
            for (sub, gl_ctx) in ctx.gl_contexts() {
                let want = Current::Sub(*id, sub);
                if self.current != want {
                    self.winsys.make_current(gl_ctx).expect("a sub-context's GL context exists");
                    self.current = want;
                }
                self.gl.finish();
            }
        }
        self.switch_ctx0();
        self.gl.finish();
        true
    }

    /// Delete the host side of a resource, on ctx0. A handle this renderer never held is
    /// nothing to delete: the renderer's table also holds resources vrend has no side of.
    pub fn resource_destroy(&mut self, handle: ResourceHandle) {
        // An untyped slot owns only a share of someone else's storage: dropping it is the
        // whole of its teardown, and it needs no GL context to do it.
        if let Some(resource::Slot::Resource(res)) = self.resources.remove(&handle) {
            self.switch_ctx0();
            if let Some(still_attached) = res.destroy(&self.gl) {
                self.doomed.push(still_attached);
            }
            self.sweep_doomed();
        }
    }

    /// Delete the parked texture storage nothing holds any more. Called where ctx0 is current.
    fn sweep_doomed(&mut self) {
        let mut i = 0;
        while i < self.doomed.len() {
            if Arc::strong_count(&self.doomed[i]) == 1 {
                let t = self.doomed.swap_remove(i);
                Arc::into_inner(t).expect("the only share").destroy(&self.gl);
            } else {
                i += 1;
            }
        }
    }

    pub fn resource(&self, handle: ResourceHandle) -> Option<&Resource> {
        self.resources.get(&handle)?.resource()
    }

    /// The guest attached pages to a resource: a host-memory buffer pays them whatever they are
    /// owed (`vrend_pipe_resource_attach_iov`).
    ///
    /// A freshly created resource owes nothing, and that is what keeps this from racing the
    /// guest -- see [`resource::Shadow`].
    pub fn resource_attached(&mut self, handle: ResourceHandle, pages: &Iov<'_>) {
        if let Some(Resource { storage: resource::Storage::Host(shadow), .. }) =
            self.resources.get_mut(&handle).and_then(resource::Slot::resource_mut)
            && !shadow.mirror_into(pages)
        {
            eprintln!(
                "[virglrs] resource {handle}: the attached pages are smaller than the buffer"
            );
        }
    }

    /// The guest is detaching a resource's pages: a host-memory buffer pulls them back first
    /// (`vrend_pipe_resource_detach_iov`).
    pub fn resource_detaching(&mut self, handle: ResourceHandle, pages: &Iov<'_>) {
        if let Some(Resource { storage: resource::Storage::Host(shadow), .. }) =
            self.resources.get_mut(&handle).and_then(resource::Slot::resource_mut)
        {
            // The pages are about to go away, so what they hold survives only here -- and is
            // owed back to whatever pages arrive next.
            let ok = pages.copy_out(0, shadow.bytes_mut());
            shadow.unmirrored();
            if !ok {
                eprintln!(
                    "[virglrs] resource {handle}: the detached pages are smaller than the buffer"
                );
            }
        }
    }

    // ---- transfers ----

    /// A transfer on the API path (`virgl_renderer_transfer_{read,write}_iov`): on the named
    /// context, or on ctx0 for the VMM's own.
    ///
    /// `own` is the resource's attached pages, `pages` the ones the transfer names -- the same
    /// ones when the caller gave none.
    pub fn transfer(
        &mut self,
        ctx: Option<ContextId>,
        handle: ResourceHandle,
        to_host: bool,
        own: Option<&Iov<'_>>,
        pages: &Iov<'_>,
        info: &Info,
    ) -> Result<(), transfer::Error> {
        match ctx {
            Some(id) => {
                if !self.contexts.contains_key(&id) {
                    return Err(transfer::Error::NoPages);
                }
                // The context's current sub-context owns the GL context to run on.
                let (mut host, contexts) = self.split(id, &NoGuest);
                contexts[&id].make_current(&mut host);
            }
            None => self.switch_ctx0(),
        }
        if pages.is_empty() {
            return Err(transfer::Error::NoPages);
        }
        let res = self
            .resources
            .get_mut(&handle)
            .and_then(resource::Slot::resource_mut)
            .ok_or(transfer::Error::NoPages)?;
        if to_host {
            transfer::write(&self.gl, &self.formats, res, own, pages, info)
        } else {
            transfer::read(&self.gl, &self.features, &self.formats, res, own, pages, info)
        }
    }
}

/// A guest that answers nothing: for a switch of GL context, which asks nothing of the guest.
struct NoGuest;

impl Guest for NoGuest {
    fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
        false
    }

    fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
        None
    }

    fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
        None
    }
}

/// `epoxy_gl_version` for a GLES version string: "OpenGL ES 3.1 Mesa ..." is 31.
fn parse_gles_version(s: &str) -> u32 {
    let rest = s.strip_prefix("OpenGL ES ").unwrap_or(s);
    let mut it = rest.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty());
    let major: u32 = it.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor: u32 = it.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    major * 10 + minor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blob_is_never_published_past_the_resource_backing_it() {
        // The size the guest asks to publish and the width it allocated are one fact sent twice,
        // and the guest is the only thing that can make them disagree. Exactly is the ordinary
        // case, and less is the guest publishing part of what it made.
        publishable(0x21000, 0x21000).expect("exactly what was allocated");
        publishable(0x1000, 0x21000).expect("part of what was allocated");

        // More is the guest asking the VMM to map whatever the driver put after the buffer into
        // its address space. There is no repair for it: mapping less than was asked reports
        // success for a mapping that was not requested, and the guest indexes past the end of it.
        assert_eq!(
            publishable(0x22000, 0x21000),
            Err(ClaimRefused::Oversize { asked: 0x22000, allocated: 0x21000 }),
            "a blob larger than its resource is refused, not trimmed to fit"
        );
    }

    /// The probed table on the live host, for the formats the classic corpus creates. Needs the
    /// zink-on-KosmicKrisp environment, so it is opted into.
    #[test]
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn the_host_table_for_the_corpus_formats() {
        let v = Vrend::new(Config::default()).expect("vrend comes up");
        let present: Vec<&str> = v.features.present().map(|f| f.name()).collect();
        eprintln!("features: {}", present.join(" "));
        for raw in [1, 2, 20, 48, 49, 64, 65, 67, 131, 134, 177, 227] {
            let f = super::super::proto::Format::from_wire(raw).unwrap();
            eprintln!("{raw:>4} {:<24} {:?}", f.name(), v.formats.get(f));
        }
    }

    #[test]
    fn the_version_string_parses_the_way_epoxy_reads_it() {
        assert_eq!(parse_gles_version("OpenGL ES 3.1 Mesa 26.0.0"), 31);
        assert_eq!(parse_gles_version("OpenGL ES 3.2 Mesa 26.0.0-devel (git-abc)"), 32);
        assert_eq!(parse_gles_version(""), 0);
    }
}
