// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! EGL: the winsys the classic renderer draws through.
//!
//! One of the named unsafe modules (CLAUDE.md). libEGL is linked, not dlopened -- the argument is
//! `vulkan.rs`'s, unchanged -- and `eglGetProcAddress` is the one symbol taken from it: every EGL
//! and GLES entry point is resolved through it into a table generated from the Khronos registry.
//!
//! The display is surfaceless: there is no window and never will be, because what a guest sees is
//! an IOSurface a compositor presents (`metal.rs`), and vrend renders into textures that are
//! imported from those. So the platform is `EGL_PLATFORM_SURFACELESS_MESA`, every context is
//! made current with no surface, and the config is a formality the API insists on.
//!
//! What this module hands the rest of vrend is a [`Winsys`] that owns the display, [`Context`]s
//! that cannot outlive it, and a [`Gles`] table loaded once a context is current. Nothing else
//! sees an `EGLDisplay` or an `EGLContext`.

use core::ffi::CStr;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

pub(crate) use super::gl::types;
use super::gl::{Gles, ProcAddr};
use crate::metal::{Held, Surface};

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod proc {
    include!(concat!(env!("OUT_DIR"), "/gl/egl.rs"));
}

use proc::Egl;
use types::*;

// libEGL, linked. `eglGetProcAddress` is the root of every table below it: EGL 1.5 promises it
// answers for core EGL commands as well as extensions, and Mesa answers for GLES through it too.
#[link(name = "EGL")]
unsafe extern "C" {
    fn eglGetProcAddress(name: *const c_char) -> Option<ProcAddr>;
}

fn get_proc(name: &CStr) -> Option<ProcAddr> {
    // SAFETY: `name` is a `&CStr`, so it is NUL-terminated and live for the call; the function
    // takes nothing else.
    unsafe { eglGetProcAddress(name.as_ptr()) }
}

/// The EGL entry points, resolved once. Needs no display: the table is the library's, and a
/// missing entry is a missing extension, which the census below reports by name.
pub fn table() -> Egl {
    // SAFETY: `eglGetProcAddress` answers each name with null or with the address of the EGL
    // command of that name, which is the contract `Egl::load` requires.
    unsafe { Egl::load(&mut get_proc) }
}

/// An EGL call that returned failure, and what `eglGetError` said about it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EglError {
    pub call: &'static str,
    pub code: EGLint,
}

impl EglError {
    fn name(code: EGLint) -> &'static str {
        match code as u32 {
            proc::EGL_SUCCESS => "EGL_SUCCESS",
            proc::EGL_NOT_INITIALIZED => "EGL_NOT_INITIALIZED",
            proc::EGL_BAD_ACCESS => "EGL_BAD_ACCESS",
            proc::EGL_BAD_ALLOC => "EGL_BAD_ALLOC",
            proc::EGL_BAD_ATTRIBUTE => "EGL_BAD_ATTRIBUTE",
            proc::EGL_BAD_CONFIG => "EGL_BAD_CONFIG",
            proc::EGL_BAD_CONTEXT => "EGL_BAD_CONTEXT",
            proc::EGL_BAD_CURRENT_SURFACE => "EGL_BAD_CURRENT_SURFACE",
            proc::EGL_BAD_DISPLAY => "EGL_BAD_DISPLAY",
            proc::EGL_BAD_MATCH => "EGL_BAD_MATCH",
            proc::EGL_BAD_NATIVE_PIXMAP => "EGL_BAD_NATIVE_PIXMAP",
            proc::EGL_BAD_NATIVE_WINDOW => "EGL_BAD_NATIVE_WINDOW",
            proc::EGL_BAD_PARAMETER => "EGL_BAD_PARAMETER",
            proc::EGL_BAD_SURFACE => "EGL_BAD_SURFACE",
            proc::EGL_CONTEXT_LOST => "EGL_CONTEXT_LOST",
            _ => "an EGL error the registry has no name for",
        }
    }
}

impl fmt::Display for EglError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {} ({:#x})", self.call, Self::name(self.code), self.code)
    }
}

impl std::error::Error for EglError {}

/// Which client API the winsys binds. Only GLES today; a desktop-GL flavour is a later,
/// separately gated change (`docs/rust-rewrite.md`, P3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flavour {
    Gles,
}

/// A client API version to ask `eglCreateContext` for.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
}

/// The display and everything a context needs to reach it. Shared by every [`Context`] and
/// [`Winsys`] so that a context cannot outlive the display it was created on.
struct Shared {
    egl: Egl,
    display: EGLDisplay,
    config: EGLConfig,
}

// SAFETY: an `EGLDisplay`, `EGLConfig` or `EGLContext` is a token the library hands out, not
// memory this crate reads: every use goes back through an EGL call, and EGL is specified to be
// callable from any thread, serialising its own state. Which thread a context is *current* on is
// EGL's own bookkeeping (`eglMakeCurrent` binds the calling thread), not a property of the handle,
// so moving the handle between threads asserts nothing. These impls are what lets the renderer
// root, which is `Send`, own the winsys; the make-current discipline is `Vrend`'s.
unsafe impl Send for Shared {}
// SAFETY: as above -- a shared reference grants only the ability to pass the tokens to EGL.
unsafe impl Sync for Shared {}

impl Shared {
    fn error(&self, call: &'static str) -> EglError {
        // SAFETY: `eglGetError` takes nothing and is defined on any thread.
        EglError { call, code: unsafe { self.egl.eglGetError()() } }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: `display` is the initialised display this struct owns, and every context and
        // image created on it holds an `Arc` of this struct, so all of them are gone by now.
        unsafe { self.egl.eglTerminate()(self.display) };
    }
}

/// The surfaceless EGL display, initialised, with the client API bound and a config chosen.
pub struct Winsys {
    shared: Arc<Shared>,
    flavour: Flavour,
    version: Version,
    extensions: BTreeSet<String>,
}

/// An EGL context on the winsys's display. Destroyed with it; cannot outlive the display.
pub struct Context {
    shared: Arc<Shared>,
    ctx: EGLContext,
}

// SAFETY: `ctx` is an EGL token, for the reason `Shared`'s impl gives; which thread it is current
// on is EGL's bookkeeping, not the handle's.
unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: `ctx` was returned by `eglCreateContext` on this display and has not been
        // destroyed, because only this drop destroys it.
        unsafe { self.shared.egl.eglDestroyContext()(self.shared.display, self.ctx) };
    }
}

/// `EGL_IOSURFACE_LIMINA`: the `eglCreateImageKHR` target limina's Mesa accepts an `IOSurfaceRef`
/// as the client buffer of. The value is the one `egl_dri2.c` defines, and must stay so.
const EGL_IOSURFACE_LIMINA: EGLenum = 0x3B9A;

/// `EGL_IOSURFACE_PLANE_LIMINA` and `EGL_IOSURFACE_FOURCC_LIMINA`: which plane of a planar
/// surface an image is over, and how that plane's bytes are laid out. Same source as the target
/// above, and the same requirement that the values match.
const EGL_IOSURFACE_PLANE_LIMINA: EGLint = 0x3B9B;
const EGL_IOSURFACE_FOURCC_LIMINA: EGLint = 0x3B9C;

/// How one plane of a planar surface is read: the DRM FourCC limina's Mesa names it by.
///
/// A plane of a 4:2:0 biplanar surface is not YUV to the sampler -- it is one or two 8-bit
/// channels, and which one decides the texture format the driver lays over those bytes. The
/// index and the layout travel as one value because neither means anything alone: an index
/// without a layout names bytes with no interpretation, and a layout without an index names no
/// bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plane {
    /// Plane 0 of a 4:2:0 surface: one 8-bit channel, full resolution. `R8`.
    Luma,
    /// Plane 1 of a 4:2:0 biplanar surface: two interleaved 8-bit channels at half resolution.
    /// `GR88`.
    ChromaPair,
}

impl Plane {
    /// The plane's index within the surface.
    fn index(self) -> EGLint {
        match self {
            Plane::Luma => 0,
            Plane::ChromaPair => 1,
        }
    }

    /// The DRM FourCC for the plane's own layout.
    fn fourcc(self) -> EGLint {
        let code = match self {
            Plane::Luma => u32::from_le_bytes(*b"R8  "),
            Plane::ChromaPair => u32::from_le_bytes(*b"GR88"),
        };
        code as EGLint
    }
}

/// An EGL image over an IOSurface, which is the surface's bytes seen as a GL texture's storage.
///
/// Holds a share of the surface it was made from, so the image cannot outlive what it images:
/// the driver keeps its own reference to the IOSurface, but ours is what keeps the id the
/// compositor was handed naming this surface and not a stranger's minted after it.
///
/// The share is the owner's, whoever that is -- see [`Held`]. An image over a venus allocation's
/// surface keeps that allocation's charge standing for exactly as long as the texture does.
pub struct Image {
    shared: Arc<Shared>,
    image: EGLImageKHR,
    held: Arc<dyn Held>,
}

// SAFETY: `image` is an EGL token, for the reason `Shared`'s impl gives, and the surface is
// `Send + Sync` on its own account.
unsafe impl Send for Image {}
// SAFETY: as above -- a shared reference grants only the ability to pass the token to EGL or GL.
unsafe impl Sync for Image {}

impl Image {
    /// The surface this images.
    pub fn surface(&self) -> &Surface {
        self.held.surface()
    }

    /// A share of the surface this images, for a holder outside the classic side: a venus context
    /// importing this resource keeps the surface alive for as long as it can still reach it,
    /// which is past the classic context that created it and past the resource itself.
    pub fn held(&self) -> Arc<dyn Held> {
        Arc::clone(&self.held)
    }

    /// The token GL binds as texture storage (`GLeglImageOES`). For the GL bindings only, which
    /// take the `Image` by reference and so cannot hold the token past it.
    pub(crate) fn raw(&self) -> EGLImageKHR {
        self.image
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: `image` was returned by `eglCreateImageKHR` on this display and has not been
        // destroyed, because only this drop destroys it. The surface is released after, by its
        // own drop, so the driver's last look at it (if any) precedes ours.
        unsafe { self.shared.egl.eglDestroyImageKHR()(self.shared.display, self.image) };
    }
}

fn c_str_to_string(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: EGL returns a NUL-terminated string owned by the library, live for the display's
    // lifetime; it is copied out before anything else is called.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

impl Winsys {
    /// Open the surfaceless display, bind the flavour's API and choose a config.
    ///
    /// Fails, naming the call, when the library has no surfaceless platform, no display can be
    /// initialised (which is what a missing Vulkan ICD looks like from here: zink has nothing to
    /// stand on), or no config renders the API asked for.
    pub fn open(flavour: Flavour) -> Result<Winsys, EglError> {
        let egl = table();
        let client = {
            // SAFETY: `eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS)` is the client-extension
            // query and needs no display.
            let p = unsafe {
                egl.eglQueryString()(proc::EGL_NO_DISPLAY, proc::EGL_EXTENSIONS as EGLint)
            };
            c_str_to_string(p)
        };
        let has = |name: &str| client.split(' ').any(|e| e == name);
        if !has("EGL_EXT_platform_base") || !has("EGL_MESA_platform_surfaceless") {
            return Err(EglError {
                call: "eglGetPlatformDisplay (surfaceless)",
                code: proc::EGL_BAD_PARAMETER as EGLint,
            });
        }
        let get_platform_display = egl
            .try_eglGetPlatformDisplayEXT()
            .expect("EGL_EXT_platform_base is advertised and eglGetPlatformDisplayEXT is absent");
        // SAFETY: the surfaceless platform takes `EGL_DEFAULT_DISPLAY` and no attributes; the
        // extension that defines it was just checked for.
        let display = unsafe {
            get_platform_display(
                proc::EGL_PLATFORM_SURFACELESS_MESA,
                proc::EGL_DEFAULT_DISPLAY,
                core::ptr::null(),
            )
        };
        if display == proc::EGL_NO_DISPLAY {
            let code = unsafe { egl.eglGetError()() };
            return Err(EglError { call: "eglGetPlatformDisplay", code });
        }
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: `display` is the handle just returned; the out-pointers are live locals.
        if unsafe { egl.eglInitialize()(display, &mut major, &mut minor) } == proc::EGL_FALSE {
            let code = unsafe { egl.eglGetError()() };
            return Err(EglError { call: "eglInitialize", code });
        }
        // From here the display is owned, and `Shared`'s drop terminates it on any failure.
        let shared = Arc::new(Shared { egl, display, config: core::ptr::null_mut() });
        let egl = &shared.egl;

        let extensions: BTreeSet<String> = {
            // SAFETY: `display` is initialised.
            let p = unsafe { egl.eglQueryString()(display, proc::EGL_EXTENSIONS as EGLint) };
            c_str_to_string(p).split(' ').filter(|s| !s.is_empty()).map(str::to_string).collect()
        };

        let (api, renderable) = match flavour {
            Flavour::Gles => (proc::EGL_OPENGL_ES_API, proc::EGL_OPENGL_ES2_BIT),
        };
        // SAFETY: `eglBindAPI` takes an enum and nothing else.
        if unsafe { egl.eglBindAPI()(api) } == proc::EGL_FALSE {
            return Err(shared.error("eglBindAPI"));
        }

        let attribs: [EGLint; 13] = [
            proc::EGL_SURFACE_TYPE as EGLint,
            proc::EGL_PBUFFER_BIT as EGLint,
            proc::EGL_RENDERABLE_TYPE as EGLint,
            renderable as EGLint,
            proc::EGL_RED_SIZE as EGLint,
            1,
            proc::EGL_GREEN_SIZE as EGLint,
            1,
            proc::EGL_BLUE_SIZE as EGLint,
            1,
            proc::EGL_ALPHA_SIZE as EGLint,
            0,
            proc::EGL_NONE as EGLint,
        ];
        let mut config: EGLConfig = core::ptr::null_mut();
        let mut count: EGLint = 0;
        // SAFETY: `attribs` is NONE-terminated, `config` has room for the one config asked for,
        // and `count` is a live local.
        let ok =
            unsafe { egl.eglChooseConfig()(display, attribs.as_ptr(), &mut config, 1, &mut count) };
        if ok == proc::EGL_FALSE || count != 1 {
            return Err(shared.error("eglChooseConfig"));
        }
        // The config is the one field not known at construction; `Arc::get_mut` holds because
        // nothing else has cloned the `Arc` yet.
        let mut shared = shared;
        Arc::get_mut(&mut shared).expect("no context exists yet").config = config;

        Ok(Winsys {
            shared,
            flavour,
            version: Version { major: major as u32, minor: minor as u32 },
            extensions,
        })
    }

    pub fn flavour(&self) -> Flavour {
        self.flavour
    }

    /// The EGL version the display reports.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Whether the display advertises an extension, by its `EGL_*` name.
    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.contains(name)
    }

    pub fn extensions(&self) -> impl Iterator<Item = &str> {
        self.extensions.iter().map(String::as_str)
    }

    /// Create a context of the flavour's API at `version`, sharing objects with `shared` if
    /// given.
    pub fn create_context(
        &self,
        version: Version,
        shared: Option<&Context>,
    ) -> Result<Context, EglError> {
        let attribs: [EGLint; 5] = [
            proc::EGL_CONTEXT_MAJOR_VERSION as EGLint,
            version.major as EGLint,
            proc::EGL_CONTEXT_MINOR_VERSION as EGLint,
            version.minor as EGLint,
            proc::EGL_NONE as EGLint,
        ];
        let share = shared.map_or(proc::EGL_NO_CONTEXT, |c| {
            assert!(Arc::ptr_eq(&c.shared, &self.shared), "a share context from another display");
            c.ctx
        });
        let egl = &self.shared.egl;
        // SAFETY: the display is initialised and the config chosen on it; `attribs` is
        // NONE-terminated; `share` is either no context or one alive on this display.
        let ctx = unsafe {
            egl.eglCreateContext()(self.shared.display, self.shared.config, share, attribs.as_ptr())
        };
        if ctx == proc::EGL_NO_CONTEXT {
            return Err(self.shared.error("eglCreateContext"));
        }
        Ok(Context { shared: Arc::clone(&self.shared), ctx })
    }

    /// Make `ctx` current on this thread, with no surface.
    pub fn make_current(&self, ctx: &Context) -> Result<(), EglError> {
        assert!(Arc::ptr_eq(&ctx.shared, &self.shared), "a context from another display");
        let egl = &self.shared.egl;
        // SAFETY: the display is initialised, the context alive on it, and surfaceless contexts
        // are made current with `EGL_NO_SURFACE` twice.
        let ok = unsafe {
            egl.eglMakeCurrent()(
                self.shared.display,
                proc::EGL_NO_SURFACE,
                proc::EGL_NO_SURFACE,
                ctx.ctx,
            )
        };
        if ok == proc::EGL_FALSE {
            return Err(self.shared.error("eglMakeCurrent"));
        }
        Ok(())
    }

    /// Release whatever context is current on this thread.
    pub fn release_current(&self) -> Result<(), EglError> {
        let egl = &self.shared.egl;
        // SAFETY: `EGL_NO_CONTEXT` with no surfaces is the documented way to release.
        let ok = unsafe {
            egl.eglMakeCurrent()(
                self.shared.display,
                proc::EGL_NO_SURFACE,
                proc::EGL_NO_SURFACE,
                proc::EGL_NO_CONTEXT,
            )
        };
        if ok == proc::EGL_FALSE {
            return Err(self.shared.error("eglMakeCurrent"));
        }
        Ok(())
    }

    /// An EGL image whose pixels are `surface`'s, for a texture to take as its storage.
    ///
    /// Made against no context: the image belongs to the display, and any context on it may
    /// bind it. Fails, naming the call, when the driver will not import the surface -- the
    /// resource then keeps ordinary GL storage, and the caller says so.
    pub fn image_from_iosurface(&self, held: Arc<dyn Held>) -> Result<Image, EglError> {
        self.image_of_iosurface(held, None)
    }

    /// An EGL image over *one plane* of a planar surface, in that plane's own layout.
    ///
    /// The planes of a composite decode target share one allocation, so each is imaged
    /// separately and the images are what the plane resources take as storage. Each holds its
    /// own share of the surface: the surface outlives whichever plane image is dropped last,
    /// and no plane's image is a view into something already freed.
    pub fn image_from_iosurface_plane(
        &self,
        held: Arc<dyn Held>,
        plane: Plane,
    ) -> Result<Image, EglError> {
        self.image_of_iosurface(held, Some(plane))
    }

    fn image_of_iosurface(
        &self,
        held: Arc<dyn Held>,
        plane: Option<Plane>,
    ) -> Result<Image, EglError> {
        let egl = &self.shared.egl;
        let attribs = plane.map(|plane| {
            [
                EGL_IOSURFACE_PLANE_LIMINA,
                plane.index(),
                EGL_IOSURFACE_FOURCC_LIMINA,
                plane.fourcc(),
                proc::EGL_NONE as EGLint,
            ]
        });
        // SAFETY: the display is initialised; the target is the one limina's Mesa defines for an
        // `IOSurfaceRef` client buffer, and `surface` is held by the `Image` for as long as the
        // image exists, so the reference passed here outlives every use the driver makes of it.
        // The attribute list, when there is one, is a live local this call outlives and is
        // `EGL_NONE`-terminated; `NULL` is the documented empty list when there is not.
        let image = unsafe {
            egl.eglCreateImageKHR()(
                self.shared.display,
                proc::EGL_NO_CONTEXT,
                EGL_IOSURFACE_LIMINA,
                held.surface().client_buffer(),
                attribs.as_ref().map_or(core::ptr::null(), |a| a.as_ptr()),
            )
        };
        if image.is_null() {
            return Err(self.shared.error("eglCreateImageKHR"));
        }
        Ok(Image { shared: Arc::clone(&self.shared), image, held })
    }

    /// The GLES entry points. Resolved through `eglGetProcAddress`, which for Mesa answers the
    /// same addresses whichever context is current -- dispatch happens behind them -- so one
    /// table serves every context of the display.
    pub fn gles(&self) -> Gles {
        // SAFETY: `eglGetProcAddress` answers each name with null or with the address of the GL
        // command of that name, which is the contract `Gles::load` requires.
        unsafe { Gles::load(&mut get_proc) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// libEGL has to be there and answer for every core EGL command through
    /// `eglGetProcAddress`. It needs no display and no GPU: if this fails the link is wrong,
    /// not the driver.
    #[test]
    fn the_library_answers_for_every_core_command() {
        let egl = table();
        for feature in ["EGL_VERSION_1_0", "EGL_VERSION_1_4", "EGL_VERSION_1_5"] {
            assert!(egl.has_all_of(feature), "{feature}: missing {:?}", egl.missing());
        }
        assert!(egl.has_all_of("EGL_EXT_platform_base"));
        assert!(egl.has_all_of("EGL_KHR_image_base"));
    }

    /// A GLES 3.1 context on the surfaceless display, and the whole 3.1 core resolved through
    /// it. Needs the zink-on-KosmicKrisp stack: `VK_ICD_FILENAMES` at the KK ICD and
    /// `MESA_LOADER_DRIVER_OVERRIDE=zink`, the way `harness/replay/vkr-replay.sh` sets them --
    /// so it is opted into rather than run by `cargo test`, which has no GPU.
    #[test]
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn a_gles_31_context_comes_up_surfaceless() {
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        assert!(winsys.version() >= Version { major: 1, minor: 4 });
        assert!(
            winsys.has_extension("EGL_KHR_image_base"),
            "{:?}",
            winsys.extensions().collect::<Vec<_>>()
        );
        let ctx =
            winsys.create_context(Version { major: 3, minor: 1 }, None).expect("a 3.1 context");
        winsys.make_current(&ctx).expect("current");
        let gl = winsys.gles();
        for feature in ["GL_ES_VERSION_2_0", "GL_ES_VERSION_3_0", "GL_ES_VERSION_3_1"] {
            assert!(gl.has_all_of(feature), "{feature}: missing {:?}", gl.missing());
        }
        // SAFETY: a context is current on this thread and `GL_VERSION` is a valid name.
        let version =
            c_str_to_string(
                unsafe { gl.glGetString()(super::super::gl::gles::GL_VERSION) } as *const c_char
            );
        assert!(version.starts_with("OpenGL ES 3.1"), "{version}");
        let second = winsys
            .create_context(Version { major: 3, minor: 1 }, Some(&ctx))
            .expect("a shared context");
        winsys.make_current(&second).expect("current");
        winsys.release_current().expect("released");
    }

    /// The two planes of one planar surface import as two images, each in its own layout.
    ///
    /// This is the whole storage model behind a composite decode target: the planes share an
    /// allocation and are reached separately, so if the driver will not take the plane
    /// attributes there is nothing above this that can work. It asks the driver rather than
    /// asserting the attribute values, because the values are only right if Mesa agrees.
    ///
    /// Run it on its own (`--lib each_plane_of -- --ignored`): two displays opened in one
    /// process leave this driver unable to make a shared context, so the ignored tests in this
    /// module fail each other when run together.
    #[test]
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn each_plane_of_a_planar_surface_imports_as_its_own_image() {
        use crate::metal::{PlanarFormat, Surface};

        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let surface: Arc<dyn Held> =
            Arc::new(Surface::planar(64, 64, PlanarFormat::BiPlanar420).expect("a planar surface"));
        let id = surface.surface().id();

        let luma = winsys
            .image_from_iosurface_plane(Arc::clone(&surface), Plane::Luma)
            .expect("the driver imports the luma plane");
        let chroma = winsys
            .image_from_iosurface_plane(Arc::clone(&surface), Plane::ChromaPair)
            .expect("the driver imports the chroma plane");

        assert_ne!(luma.raw(), chroma.raw(), "a plane image per plane, not one image twice");
        assert_eq!(luma.surface().id(), id, "both image the surface they were made from");
        assert_eq!(chroma.surface().id(), id);

        // Each holds its own share, so dropping one leaves the other imaging a live surface.
        drop(luma);
        assert_eq!(chroma.surface().id(), id);
    }

    /// A plane image samples the plane it asked for, asked by content.
    ///
    /// The import succeeding proves only that the driver took the attributes, and the image's
    /// own width and height cannot settle it either: they are set from the geometry the import
    /// was handed, so a chroma image reports half-resolution because that is what it was told,
    /// whether or not the plane index reached Metal underneath. Both are measurements of this
    /// side's own input.
    ///
    /// Content is the oracle that cannot lie. Each plane gets a different byte through the CPU,
    /// and each image is read back through a framebuffer. If the index is dropped anywhere
    /// between EGL and Metal, both images are plane 0 and the chroma read returns the luma
    /// pattern.
    #[test]
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn a_plane_image_samples_the_plane_it_asked_for() {
        use super::super::gl::Gl;
        use super::super::gl::gles::{
            GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_RED, GL_RG,
            GL_TEXTURE_2D, GL_UNSIGNED_BYTE,
        };
        use crate::metal::{PlanarFormat, Surface};

        const LUMA_BYTE: u8 = 0x10;
        const CHROMA_BYTE: u8 = 0x80;

        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let ctx =
            winsys.create_context(Version { major: 3, minor: 1 }, None).expect("a 3.1 context");
        winsys.make_current(&ctx).expect("current");
        let gl = Gl::new(winsys.gles());

        let surface = Surface::planar(64, 64, PlanarFormat::BiPlanar420).expect("a planar surface");
        assert!(surface.fill_plane(0, LUMA_BYTE), "the luma plane fills");
        assert!(surface.fill_plane(1, CHROMA_BYTE), "the chroma plane fills");
        let surface: Arc<dyn Held> = Arc::new(surface);

        let read = |plane: Plane, format, w, h| -> Vec<u8> {
            let image = winsys
                .image_from_iosurface_plane(Arc::clone(&surface), plane)
                .expect("the driver imports the plane");
            let texture = gl.gen_texture();
            gl.bind_texture(GL_TEXTURE_2D, Some(texture));
            gl.egl_image_target_texture_2d(GL_TEXTURE_2D, &image);
            let fb = gl.gen_framebuffer();
            gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
            gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Some(texture), 0);
            assert_eq!(
                gl.check_framebuffer_status(),
                GL_FRAMEBUFFER_COMPLETE,
                "a plane texture is not renderable, so this test cannot read it"
            );
            let mut out = vec![0u8; (w * h) as usize * if format == GL_RG { 2 } else { 1 }];
            assert!(gl.read_pixels(0, 0, w, h, format, GL_UNSIGNED_BYTE, &mut out), "readback");
            gl.bind_framebuffer(GL_FRAMEBUFFER, None);
            out
        };

        let luma = read(Plane::Luma, GL_RED, 64, 64);
        let chroma = read(Plane::ChromaPair, GL_RG, 32, 32);

        assert!(luma.iter().all(|&b| b == LUMA_BYTE), "luma read {:?}", &luma[..8]);
        assert!(
            chroma.iter().all(|&b| b == CHROMA_BYTE),
            "the chroma image did not sample the chroma plane; it read {:?}. A read of \
             {LUMA_BYTE:#04x} means the plane index was dropped between EGL and Metal and both \
             images are plane 0.",
            &chroma[..8]
        );
    }
}
