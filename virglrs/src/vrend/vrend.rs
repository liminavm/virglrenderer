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

use super::egl::{self, EglError, Flavour, Version, Winsys};
use super::features::Features;
use super::formats::Table;
use super::gl::Gl;
use super::gl::gles::GL_VERSION;
use super::resource::{self, Args, Limits, Refusal, Resource};
use super::transfer::{self, Info};
use crate::guest_mem::Iov;
use crate::ids::{CtxId, ResourceHandle};
use std::collections::BTreeMap;
use std::fmt;

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

/// A guest context's host side.
pub struct Context {
    gl_ctx: egl::Context,
}

/// Which GL context is current on the calling thread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Current {
    Ctx0,
    Ctx(CtxId),
}

pub struct Vrend {
    winsys: Winsys,
    gl: Gl,
    pub features: Features,
    pub formats: Table,
    pub limits: Limits,
    ctx0: egl::Context,
    current: Current,
    resources: BTreeMap<ResourceHandle, Resource>,
    contexts: BTreeMap<CtxId, Context>,
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
                ctx0 = Some(c);
                break;
            }
        }
        let ctx0 = ctx0.ok_or(InitError::NoContext)?;
        winsys.make_current(&ctx0)?;
        let gl = Gl::new(winsys.gles());
        let version = gl.get_string(GL_VERSION);
        let gles_version = parse_gles_version(&version);
        let features = Features::probe(gles_version, gl.extensions());
        let limits = Limits::query(&gl);
        let formats = Table::probe(&gl, &features);
        eprintln!(
            "[virglrs] vrend: {version} (gles {gles_version}), {} formats, {} features",
            formats.entries().count(),
            features.present().count(),
        );
        Ok(Vrend {
            winsys,
            gl,
            features,
            formats,
            limits,
            ctx0,
            current: Current::Ctx0,
            resources: BTreeMap::new(),
            contexts: BTreeMap::new(),
        })
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

    /// Make a guest context current. `false` if there is no such context.
    fn switch_to(&mut self, id: CtxId) -> bool {
        let Some(c) = self.contexts.get(&id) else {
            return false;
        };
        if self.current != Current::Ctx(id) {
            self.winsys
                .make_current(&c.gl_ctx)
                .expect("a guest context that exists can be made current");
            self.current = Current::Ctx(id);
        }
        true
    }

    // ---- contexts ----

    pub fn context_create(&mut self, id: CtxId) -> Result<(), EglError> {
        let version = VERSIONS
            .iter()
            .copied()
            .find(|v| self.features.gles_version >= v.major * 10 + v.minor)
            .unwrap_or(VERSIONS[2]);
        let gl_ctx = self.winsys.create_context(version, Some(&self.ctx0))?;
        self.contexts.insert(id, Context { gl_ctx });
        Ok(())
    }

    pub fn context_destroy(&mut self, id: CtxId) {
        if self.current == Current::Ctx(id) {
            self.switch_ctx0();
        }
        self.contexts.remove(&id);
    }

    pub fn has_context(&self, id: CtxId) -> bool {
        self.contexts.contains_key(&id)
    }

    // ---- resources ----

    /// Create the host side of a classic resource, on ctx0.
    pub fn resource_create(&mut self, handle: ResourceHandle, args: Args) -> Result<(), Refusal> {
        assert!(!self.resources.contains_key(&handle), "the renderer checked the handle was free");
        self.switch_ctx0();
        let res = Resource::create(&self.gl, &self.features, &self.formats, &self.limits, args)?;
        self.resources.insert(handle, res);
        Ok(())
    }

    /// Delete the host side of a resource, on ctx0. A handle this renderer never held is
    /// nothing to delete: the renderer's table also holds resources vrend has no side of.
    pub fn resource_destroy(&mut self, handle: ResourceHandle) {
        if let Some(res) = self.resources.remove(&handle) {
            self.switch_ctx0();
            res.destroy(&self.gl);
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
                if !self.switch_to(id) {
                    return Err(transfer::Error::NoPages);
                }
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
            transfer::read(&self.gl, &self.formats, res, own, pages, info)
        }
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
