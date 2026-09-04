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
use super::resource::{self, Args, Limits, Refusal, Resource};
use super::shader;
use super::transfer::{self, Info};
use crate::guest_mem::Iov;
use crate::ids::{CtxId, ResourceHandle};
use crate::metal;
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
    shader_cfg: shader::Cfg,
    /// What the guest's driver is told of the host, probed once from the same answers.
    caps: caps::CapsV2,
    ctx0: egl::Context,
    /// The version guest contexts are made with: the newest the driver gave ctx0.
    version: Version,
    current: Current,
    resources: BTreeMap<ResourceHandle, Resource>,
    contexts: BTreeMap<CtxId, Context>,
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
}

/// The versions tried, newest first -- the GLES rows of the C's `gl_versions` ladder.
const VERSIONS: [Version; 3] = [
    Version { major: 3, minor: 2 },
    Version { major: 3, minor: 1 },
    Version { major: 3, minor: 0 },
];

impl Vrend {
    /// Open the winsys, bring ctx0 up on this thread and probe the driver.
    pub fn new() -> Result<Vrend, InitError> {
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
        let shader_cfg = shader::Cfg::probe(&gl, &features, &limits);
        let formats = Table::probe(&gl, &features);
        let caps = caps::CapsV2::probe(&gl, &features, &limits, &formats);
        eprintln!(
            "[virglrs] vrend: {version_string} (gles {gles_version}), {} formats, {} features",
            formats.entries().count(),
            features.present().count(),
        );
        Ok(Vrend {
            winsys,
            gl,
            features,
            formats,
            limits,
            shader_cfg,
            caps,
            ctx0,
            version,
            current: Current::Ctx0,
            resources: BTreeMap::new(),
            contexts: BTreeMap::new(),
            todo: Todo::default(),
            blitter: None,
            doomed: Vec::new(),
        })
    }

    /// The classic capsets, as probed at init.
    pub fn caps(&self) -> &caps::CapsV2 {
        &self.caps
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
        ctx: CtxId,
        guest: &'a dyn Guest,
    ) -> (Host<'a>, &'a mut BTreeMap<CtxId, Context>) {
        let Vrend {
            winsys,
            gl,
            features,
            formats,
            limits,
            shader_cfg,
            caps: _,
            ctx0,
            version,
            current,
            resources,
            contexts,
            todo,
            blitter,
            doomed: _,
        } = self;
        let host = Host {
            gl,
            winsys,
            version: *version,
            share: ctx0,
            features,
            formats,
            limits,
            shader_cfg,
            resources,
            guest,
            ctx,
            current,
            todo,
            blitter,
        };
        (host, contexts)
    }

    // ---- contexts ----

    pub fn context_create(&mut self, id: CtxId, guest: &dyn Guest) -> Result<(), EglError> {
        let (mut host, contexts) = self.split(id, guest);
        let c = Context::new(&mut host)?;
        contexts.insert(id, c);
        Ok(())
    }

    pub fn context_destroy(&mut self, id: CtxId, guest: &dyn Guest) {
        let (mut host, contexts) = self.split(id, guest);
        if let Some(c) = contexts.remove(&id) {
            c.destroy(&mut host);
        }
        self.switch_ctx0();
        // The framebuffers that went with the context were the last holders of any storage the
        // guest freed while they drew into it.
        self.sweep_doomed();
    }

    pub fn has_context(&self, id: CtxId) -> bool {
        self.contexts.contains_key(&id)
    }

    /// Run a batch on a context. `None` for a context this renderer does not have.
    pub fn submit(
        &mut self,
        id: CtxId,
        words: &[u32],
        guest: &dyn Guest,
    ) -> Option<Result<(), Fault>> {
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
        self.resources.insert(handle, res);
        Ok(())
    }

    /// The IOSurface a resource is presented from, if its storage is one. Asked of the resource
    /// every time: the surface goes with the resource, and there is no other place to hold one.
    pub fn resource_surface(&self, handle: ResourceHandle) -> Option<&metal::Surface> {
        self.resources.get(&handle)?.surface()
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
        if let Some(res) = self.resources.remove(&handle) {
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
        self.resources.get(&handle)
    }

    /// The guest attached pages to a resource: a host-memory buffer pushes its contents into
    /// them (`vrend_pipe_resource_attach_iov`).
    pub fn resource_attached(&mut self, handle: ResourceHandle, pages: &Iov<'_>) {
        if let Some(Resource { storage: resource::Storage::Host(buf), .. }) =
            self.resources.get(&handle)
            && !pages.copy_in(0, buf)
        {
            eprintln!(
                "[virglrs] resource {handle}: the attached pages are smaller than the buffer"
            );
        }
    }

    /// The guest is detaching a resource's pages: a host-memory buffer pulls them back first
    /// (`vrend_pipe_resource_detach_iov`).
    pub fn resource_detaching(&mut self, handle: ResourceHandle, pages: &Iov<'_>) {
        if let Some(Resource { storage: resource::Storage::Host(buf), .. }) =
            self.resources.get_mut(&handle)
            && !pages.copy_out(0, buf)
        {
            eprintln!(
                "[virglrs] resource {handle}: the detached pages are smaller than the buffer"
            );
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
        ctx: Option<CtxId>,
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
        let res = self.resources.get_mut(&handle).ok_or(transfer::Error::NoPages)?;
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
    fn attached(&self, _: CtxId, _: ResourceHandle) -> bool {
        false
    }

    fn pages(&self, _: CtxId, _: ResourceHandle) -> Option<Iov<'_>> {
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

    /// The probed table on the live host, for the formats the classic corpus creates. Needs the
    /// zink-on-KosmicKrisp environment, so it is opted into.
    #[test]
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn the_host_table_for_the_corpus_formats() {
        let v = Vrend::new().expect("vrend comes up");
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
