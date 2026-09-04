// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! A classic resource on the host: what the guest asked for, checked, and the GL object (or the
//! host memory, or nothing) that backs it.
//!
//! The C's `vrend_resource` is one struct with a bitmask saying which of its overlapping fields
//! mean anything. Here the backing is an enum, so a transfer or a destroy is one match with no
//! flag to consult, and a buffer cannot be mistaken for a texture.

use super::egl::{Image, Winsys};
use super::features::{Feature, Features};
use super::formats::{Entry, Table};
use super::gl::gles::*;
use super::gl::{BufferName, GLbitfield, GLenum, GLint, GLsizei, Gl, TextureName};
use super::pipe::TextureTarget;
use super::proto::Format;
use crate::guest_mem::Iov;
use crate::metal::{Held, PixelFormat, Surface};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

/// `VIRGL_BIND_*`: what the guest intends to do with a resource.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Bind(pub u32);

impl Bind {
    pub const DEPTH_STENCIL: Bind = Bind(1 << 0);
    pub const RENDER_TARGET: Bind = Bind(1 << 1);
    pub const SAMPLER_VIEW: Bind = Bind(1 << 3);
    pub const VERTEX_BUFFER: Bind = Bind(1 << 4);
    pub const INDEX_BUFFER: Bind = Bind(1 << 5);
    pub const CONSTANT_BUFFER: Bind = Bind(1 << 6);
    pub const DISPLAY_TARGET: Bind = Bind(1 << 7);
    pub const COMMAND_ARGS: Bind = Bind(1 << 8);
    pub const STREAM_OUTPUT: Bind = Bind(1 << 11);
    pub const SHADER_BUFFER: Bind = Bind(1 << 14);
    pub const QUERY_BUFFER: Bind = Bind(1 << 15);
    pub const CURSOR: Bind = Bind(1 << 16);
    pub const CUSTOM: Bind = Bind(1 << 17);
    pub const SCANOUT: Bind = Bind(1 << 18);
    pub const STAGING: Bind = Bind(1 << 19);
    pub const SHARED: Bind = Bind(1 << 20);
    pub const PREFER_EMULATED_BGRA: Bind = Bind(1 << 21);
    pub const LINEAR: Bind = Bind(1 << 22);

    pub fn has(self, other: Bind) -> bool {
        self.0 & other.0 != 0
    }

    /// The buffer binds, which the C matches by *equality*: a resource is a buffer when its bind
    /// is exactly one of these, and a texture otherwise.
    fn is_buffer_bind(self) -> bool {
        matches!(
            self,
            Bind(0)
                | Bind::CUSTOM
                | Bind::STAGING
                | Bind::INDEX_BUFFER
                | Bind::STREAM_OUTPUT
                | Bind::VERTEX_BUFFER
                | Bind::CONSTANT_BUFFER
                | Bind::QUERY_BUFFER
                | Bind::COMMAND_ARGS
                | Bind::SHADER_BUFFER
        )
    }
}

/// `VIRGL_RESOURCE_*` flags.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ResourceFlags(pub u32);

impl ResourceFlags {
    pub const Y_0_TOP: ResourceFlags = ResourceFlags(1 << 0);
    pub const MAP_PERSISTENT: ResourceFlags = ResourceFlags(1 << 1);
    pub const MAP_COHERENT: ResourceFlags = ResourceFlags(1 << 2);
    const KNOWN: u32 = 0b111;

    pub fn has(self, other: ResourceFlags) -> bool {
        self.0 & other.0 != 0
    }
}

/// `vrend_renderer_resource_create_args`, typed. The wire numbers were parsed at the boundary
/// that received them; what is left to check is whether the combination makes sense.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Args {
    pub target: TextureTarget,
    pub format: Format,
    pub bind: Bind,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    pub last_level: u32,
    pub nr_samples: u32,
    pub flags: ResourceFlags,
}

/// Why a resource was not created. Each is one of `check_resource_valid`'s rejections, plus the
/// ones the C's allocation path answers with an errno and a log line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refusal {
    UnsupportedFormat,
    UnsupportedMultisampleFormat,
    MultisampleNot2d,
    MultisampleWithMipmaps,
    BufferWithMipmaps,
    RectWithMipmaps,
    TooManyLevels,
    UnknownFlags,
    Y0TopNot2d,
    CubeArraySize,
    CubeArraysUnsupported,
    CubeArrayArraySize,
    ArrayOfNonArrayTarget,
    ArraysUnsupported,
    ZeroWidth,
    BufferBindOnTexture,
    BufferNotFlat,
    QueryBuffersUnsupported,
    IndirectUnsupported,
    NoTextureBind,
    DepthOn2d,
    ZeroHeight,
    Not1dShape,
    TooLarge,
    ZeroDepth,
    ZeroArraySize,
    CubeNotSquare,
    /// A buffer target carrying texture binds, which the C lets through to an allocation with
    /// GL target zero.
    TextureBindOnBuffer,
    IllegalBufferBind,
    /// Persistent mapping was asked for and the driver has no `GL_EXT_buffer_storage`.
    NoBufferStorage,
    /// The driver refused the allocation.
    GlError(GLenum),
    /// An IOSurface was minted for the resource and the driver has no entry point to make it
    /// a texture's storage.
    NoEglImage,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Refusal::UnsupportedFormat => "unsupported texture format",
            Refusal::UnsupportedMultisampleFormat => "unsupported multisample texture format",
            Refusal::MultisampleNot2d => "multisample textures must be 2D",
            Refusal::MultisampleWithMipmaps => "multisample textures do not support mipmaps",
            Refusal::BufferWithMipmaps => "buffers do not support mipmaps",
            Refusal::RectWithMipmaps => "RECT textures do not support mipmaps",
            Refusal::TooManyLevels => "mipmap levels too large",
            Refusal::UnknownFlags => "resource flags not supported",
            Refusal::Y0TopNot2d => "Y_0_TOP is only supported for 2D or RECT",
            Refusal::CubeArraySize => "cube map: unexpected array size",
            Refusal::CubeArraysUnsupported => "cube map arrays not supported",
            Refusal::CubeArrayArraySize => "cube map array: unexpected array size",
            Refusal::ArrayOfNonArrayTarget => "texture target cannot be an array",
            Refusal::ArraysUnsupported => "texture arrays are not supported",
            Refusal::ZeroWidth => "texture width must be > 0",
            Refusal::BufferBindOnTexture => "buffer bind flags require the buffer target",
            Refusal::BufferNotFlat => "buffer target with height or depth other than 1",
            Refusal::QueryBuffersUnsupported => "query buffers are not supported",
            Refusal::IndirectUnsupported => "indirect draw buffers are not supported",
            Refusal::NoTextureBind => "invalid texture bind flags",
            Refusal::DepthOn2d => "2D texture target with depth other than 1",
            Refusal::ZeroHeight => "2D texture storage requires non-zero height",
            Refusal::Not1dShape => "1D texture with height or depth other than 1",
            Refusal::TooLarge => "texture larger than the driver allows",
            Refusal::ZeroDepth => "3D texture storage requires non-zero height and depth",
            Refusal::ZeroArraySize => "array texture storage requires non-zero array size",
            Refusal::CubeNotSquare => "cube map not square",
            Refusal::TextureBindOnBuffer => "texture bind flags on the buffer target",
            Refusal::IllegalBufferBind => "illegal buffer binding flags",
            Refusal::NoBufferStorage => "persistent mapping needs GL_EXT_buffer_storage",
            Refusal::GlError(_) => "the driver refused the allocation",
            Refusal::NoEglImage => "no GL_OES_EGL_image entry point to bind an IOSurface with",
        };
        match self {
            Refusal::GlError(e) => write!(f, "{s} (GL error {e:#x})"),
            _ => f.write_str(s),
        }
    }
}

/// The driver's size limits, read once at init.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Limits {
    pub max_texture_2d_size: u32,
    pub max_texture_3d_size: u32,
    pub max_texture_cube_size: u32,
    /// Capped at 8, as the C caps `max_draw_buffers`.
    pub max_draw_buffers: u32,
    pub max_vertex_attributes: u32,
    pub max_texture_units: u32,
    /// Zero when the host has no texture buffers.
    pub max_texture_buffer_size: u32,
}

impl Limits {
    pub fn query(gl: &Gl, features: &Features) -> Limits {
        let get = |name| gl.get_integer(name).max(1) as u32;
        Limits {
            max_texture_2d_size: get(GL_MAX_TEXTURE_SIZE),
            max_texture_3d_size: get(GL_MAX_3D_TEXTURE_SIZE),
            max_texture_cube_size: get(GL_MAX_CUBE_MAP_TEXTURE_SIZE),
            max_draw_buffers: get(GL_MAX_DRAW_BUFFERS).min(8),
            max_vertex_attributes: get(GL_MAX_VERTEX_ATTRIBS),
            max_texture_units: get(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS),
            max_texture_buffer_size: if features.has(Feature::arb_or_gles_ext_texture_buffer) {
                get(GL_MAX_TEXTURE_BUFFER_SIZE)
            } else {
                0
            },
        }
    }
}

/// What a handle names in vrend's table.
///
/// The wire lets a resource exist before anything says what it is: the VMM creates a blob and
/// attaches it, and only a later `SET_TYPE` in the command stream gives it a shape and a format.
/// That window is the protocol's, not ours -- attach precedes the command stream by ordering --
/// so it is represented rather than wished away, and it is a state with one owner rather than a
/// side table of handles waiting to be typed.
///
/// The distinction is the whole point of the type: a handle in [`Slot::Untyped`] is reachable
/// only by the upgrade, by detach and by unref. Every other command faults on it, and says which
/// of the two it is -- attached-but-untyped, or never attached at all. The C conflates them, and
/// its own comment records the cost: "indistinguishable from an attach that never happened".
pub enum Slot {
    /// Storage, and nothing yet that says what it is.
    Untyped(Untyped),
    /// Shape, format and host storage all decided.
    Resource(Resource),
}

impl Slot {
    /// The resource this names, or `None` while nothing has typed it.
    pub fn resource(&self) -> Option<&Resource> {
        match self {
            Slot::Resource(r) => Some(r),
            Slot::Untyped(_) => None,
        }
    }

    pub fn resource_mut(&mut self) -> Option<&mut Resource> {
        match self {
            Slot::Resource(r) => Some(r),
            Slot::Untyped(_) => None,
        }
    }
}

/// A resource attached to a context before anything has said what it is.
///
/// It holds the exporter's storage and no opinion about it. Whether those bytes are a surface
/// this can adopt was decided by whoever minted them, and the *reason* they are not stays with
/// them -- venus latches it beside its pages and says it once -- so nothing is re-derived or
/// guessed here.
pub struct Untyped {
    /// A share of the exporter's surface, when the storage is one. The share rather than an id:
    /// an id stops naming this surface the moment the surface dies, and the whole point of
    /// holding storage across contexts is that it cannot.
    surface: Option<Arc<dyn Held>>,
}

impl Untyped {
    /// Attached, with whatever the exporter published as its storage.
    pub fn new(surface: Option<Arc<dyn Held>>) -> Untyped {
        Untyped { surface }
    }

    /// Say what this storage is, which is the one thing that may be done to it.
    ///
    /// Consuming, so the untyped value cannot outlive the transition: there is no path that
    /// types a handle and leaves the old entry standing, because after this there is no old
    /// entry to leave.
    ///
    /// The surface is *adopted*, never minted. These bytes belong to whoever exported them and
    /// are the frame a client is presenting; a surface minted here would be a second copy of it,
    /// and the guest would composite the one nobody draws into.
    /// A refusal hands the storage back, so a rejected upgrade leaves the handle exactly as it
    /// was rather than deleting it: there is no path that loses a resource by failing to type it.
    ///
    /// Three outcomes, and each answers to whoever caused it.
    ///
    /// [`check`] asks the guest's half -- whether these arguments describe an image at all. A
    /// guest that describes none has erred, and gets a refusal that stops its own context and
    /// nobody else's.
    ///
    /// A host that cannot adopt IOSurfaces at all, or a blob whose share is not a surface,
    /// is neither party's error: the resource gets a blank texture and the log says the
    /// contents are wrong. That is a limitation, answered once at init for the first and
    /// carried by the storage for the second.
    ///
    /// A host that said it adopts IOSurfaces and then refuses this one is a host bug, and it
    /// crashes. Degrading there would hide our own defect behind a window that renders the
    /// wrong thing, which is the one outcome worth less than stopping.
    pub fn upgrade(
        self,
        gl: &Gl,
        winsys: &Winsys,
        features: &Features,
        formats: &Table,
        limits: &Limits,
        args: Args,
    ) -> Result<Resource, (Untyped, Refusal)> {
        if let Err(e) = check(features, formats, limits, &args) {
            return Err((self, e));
        }
        let image = match self.surface {
            Some(held) if features.adopts_iosurfaces() => match winsys.image_from_iosurface(held) {
                Ok(image) => Some(image),
                // The host takes IOSurfaces and would not take this one. See above: ours.
                Err(e) => panic!(
                    "the driver imports IOSurfaces but refused an exported {}x{} {} one: {e}",
                    args.width,
                    args.height,
                    args.format.name()
                ),
            },
            // Either this host adopts no surfaces -- said once at init -- or these bytes are
            // not one, which the storage that minted them already said. Neither is news here.
            Some(_) | None => {
                eprintln!(
                    "[virglrs] vrend: resource {}x{} {} has no surface to adopt; it gets a blank \
                     texture and its contents will be wrong",
                    args.width,
                    args.height,
                    args.format.name()
                );
                None
            }
        };
        // The share is gone into the image, or was never there; a refusal past this point has
        // nothing left to hand back but an empty slot, which is what the handle already was.
        let storage = match alloc_texture(gl, features, formats, &args, image) {
            Ok(s) => s,
            Err(e) => return Err((Untyped { surface: None }, e)),
        };
        // Whether an image backs the storage is read off the storage, not carried alongside it:
        // the adopt can fail at either step, and a second boolean tracking it would be a copy of
        // this fact that the retry above is exactly the thing to make disagree.
        if !matches!(&storage, Storage::Texture(t) if t.image.is_some()) {
            // `glTexStorage` leaves contents undefined, which is another context's memory read
            // as pixels -- wrong, and a leak. Blank is still wrong; it is not also a leak.
            zero_texture(gl, formats, &args, &storage);
        }
        Ok(Resource { args, storage })
    }

    /// A share of the surface these bytes are, if they are one.
    pub fn surface(&self) -> Option<&Arc<dyn Held>> {
        self.surface.as_ref()
    }
}

/// The host buffer behind a `VIRGL_BIND_CUSTOM` resource, and which side of it holds the truth.
///
/// The buffer and the guest's pages are two containers for one fact, so exactly one of them is
/// authoritative at a time and the other is refreshed from it. Which one is a *state*, not a flag
/// beside the bytes, because the refresh must not run in the state a fresh resource is in: the
/// guest kernel queues `RESOURCE_CREATE` and `ATTACH_BACKING` and hands the handle back without
/// waiting for either, so the guest is usually already writing through its mapping while we
/// process the attach. Pushing a fresh buffer's zeros then lands on top of what it wrote -- the
/// whole buffer, or everything up to wherever its copy had reached.
///
/// So the push is reachable only from [`Shadow::Unmirrored`], and nothing constructs that except
/// a detach or a transfer the pages did not receive. A newly created resource is
/// [`Shadow::Mirrored`] and there is no path from there to a push, which is the bug made
/// unrepresentable rather than guarded against.
pub enum Shadow {
    /// The guest's pages already hold everything this buffer does, so an attach owes them
    /// nothing. Where a resource starts, and where every paid attach returns it.
    Mirrored(Vec<u8>),
    /// This buffer holds bytes the guest's pages do not: a detach pulled them out of pages that
    /// then went away, or a transfer wrote them with no backing attached to receive them. The
    /// next attach pays that debt and the buffer is mirrored again.
    Unmirrored(Vec<u8>),
}

impl Shadow {
    /// A newly created resource's buffer: zeroed, and owing the guest nothing.
    pub fn fresh(size: usize) -> Shadow {
        Shadow::Mirrored(vec![0; size])
    }

    pub fn bytes(&self) -> &[u8] {
        match self {
            Shadow::Mirrored(b) | Shadow::Unmirrored(b) => b,
        }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Shadow::Mirrored(b) | Shadow::Unmirrored(b) => b,
        }
    }

    /// The guest's pages now hold what this buffer does -- they were just written from it, or
    /// they are where its bytes came from.
    pub fn mirrored(&mut self) {
        if let Shadow::Unmirrored(b) = self {
            *self = Shadow::Mirrored(std::mem::take(b));
        }
    }

    /// This buffer now holds bytes the guest's pages do not, and owes them to the next attach.
    pub fn unmirrored(&mut self) {
        if let Shadow::Mirrored(b) = self {
            *self = Shadow::Unmirrored(std::mem::take(b));
        }
    }

    /// Pay what the pages are owed, if anything. `false` if the pages could not hold it.
    ///
    /// A mirrored buffer writes nothing, which is the whole point: the attach that races the
    /// guest's first write through its own mapping has nothing to clobber it with.
    #[must_use]
    pub fn mirror_into(&mut self, pages: &Iov<'_>) -> bool {
        let Shadow::Unmirrored(b) = self else {
            return true;
        };
        let ok = pages.copy_in(0, b);
        self.mirrored();
        ok
    }
}

/// What backs a resource. Exactly one for its whole life.
pub enum Storage {
    /// `VIRGL_BIND_STAGING`: the guest's pages and nothing on the host.
    Guest,
    /// `VIRGL_BIND_CUSTOM`: a host buffer the guest's pages mirror at attach and detach.
    Host(Shadow),
    Buffer {
        name: BufferName,
        /// The binding target the buffer is created and mapped through, from its bind.
        target: GLenum,
        /// The buffer texture a sampler view of this buffer samples through, made at the first
        /// such view (`tbo_tex_id`).
        tbo: Option<TextureName>,
    },
    Texture(Arc<Texture>),
}

/// The GL textures a resource's storage is: the texture itself, and the render-target views taken
/// of it.
///
/// It is shared so that whatever uses a texture can hold it directly instead of holding a handle
/// and looking the resource up again. The guest may free a resource a framebuffer is still drawing
/// into, and a lookup at that moment finds nothing; a share is always there. GL would keep the
/// texture's storage alive on its own while an attachment names it -- deleting a texture releases
/// the name, not the object -- but leaning on that means every user must stay attached for its
/// whole life to stay correct, which is not a property any of them state. The share says it.
pub struct Texture {
    pub name: TextureName,
    /// The GL target -- not the pipe target: on GLES a 1D texture is a 2D one, a 1D array a
    /// 2D array, and a RECT a 2D.
    pub target: GLenum,
    pub immutable: bool,
    /// The EGL image that is the texture's storage, when that storage is an IOSurface: a
    /// scanout or a shared buffer, rendered into directly and presented from without a copy.
    /// The image owns the surface, so the surface's id is good exactly as long as the
    /// texture is.
    pub image: Option<Image>,
    /// The render-target views taken of this texture, one per distinct [`ViewKey`].
    ///
    /// They live here rather than on the surface objects that ask for them because a view
    /// outlives the surface, and here rather than on the [`Resource`] because it outlives that
    /// too. The lock is uncontended -- a renderer's classic side is one thread -- and buys the
    /// share the `Send` a `RefCell` would cost it.
    views: Mutex<BTreeMap<ViewKey, TextureName>>,
}

/// What a render target's texture view is a function of: the format it reinterprets the resource
/// in, and the layer range it restricts to. The level range is not part of it -- a surface's view
/// always spans the resource's whole mip chain -- and neither is anything the guest sets on the
/// view object afterwards, because nothing does. That makes two surfaces with the same key the
/// same view, which is why the resource can own it.
///
/// A *sampler* view has no such key: its swizzle, depth/stencil read mode and sRGB decode live on
/// the view object itself and are not derivable from the resource, so those stay where they are.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ViewKey {
    pub format: Format,
    pub first_layer: u32,
    pub layers: u32,
}

/// A resource the host holds. Its GL objects are deleted by [`Resource::destroy`], never by drop:
/// deleting needs the driver and a current context, which a drop does not have.
pub struct Resource {
    pub args: Args,
    pub storage: Storage,
}

impl Resource {
    pub fn y_0_top(&self) -> bool {
        self.args.flags.has(ResourceFlags::Y_0_TOP)
    }

    /// The format's table entry. Present for every texture, since creation refused the format
    /// otherwise; a buffer's format is nominal.
    pub fn entry<'t>(&self, formats: &'t Table) -> Option<&'t Entry> {
        formats.get(self.args.format)
    }

    /// `u_minify` of the base extent at `level`.
    pub fn width_at(&self, level: u32) -> u32 {
        minify(self.args.width, level)
    }

    pub fn height_at(&self, level: u32) -> u32 {
        minify(self.args.height, level)
    }

    /// The third extent a box's `z` and `depth` range over at `level`: layers for an array or
    /// cube, minified depth for a 3D texture, one for anything else.
    pub fn depth_at(&self, level: u32) -> u32 {
        match self.args.target {
            TextureTarget::Cube
            | TextureTarget::Array1d
            | TextureTarget::Array2d
            | TextureTarget::CubeArray => self.args.array_size,
            TextureTarget::Texture3d => minify(self.args.depth, level),
            TextureTarget::Buffer
            | TextureTarget::Texture1d
            | TextureTarget::Texture2d
            | TextureTarget::Rect => 1,
        }
    }

    /// The IOSurface this resource is presented from, if its storage is one.
    pub fn surface(&self) -> Option<&Surface> {
        match &self.storage {
            Storage::Texture(t) => t.image.as_ref().map(|i| i.surface()),
            _ => None,
        }
    }

    /// `vrend_format_is_bgra` of the resource's own format.
    pub fn is_bgra(&self) -> bool {
        is_bgra(self.args.format)
    }

    /// `vrend_resource_supports_view`: whether a texture view may be made of this resource.
    ///
    /// Not of an IOSurface-backed BGR* one. Its storage is natively BGRA8, where a texture this
    /// renderer allocates for a BGR* format is RGBA8 with the bytes swapped on the way through,
    /// and GL has no internal format to name the difference to `glTextureView` -- a view of one
    /// reads its channels in the wrong order. Such a resource is sampled and rendered as itself,
    /// with a swizzle where a view would have converted.
    pub fn supports_view(&self) -> bool {
        !(self.is_bgra() && self.surface().is_some())
    }

    /// `vrend_resource_needs_redblue_swizzle`: viewed as `view_format`, this resource's red and
    /// blue come out swapped and must be swapped back by hand.
    pub fn needs_redblue_swizzle(&self, view_format: Format) -> bool {
        !self.supports_view() && self.is_bgra() != is_bgra(view_format)
    }

    /// `vrend_resource_needs_srgb_decode`: an sRGB resource viewed linearly, with no view to do
    /// the decoding.
    pub fn needs_srgb_decode(&self, view_format: Format) -> bool {
        !self.supports_view() && is_srgb(self.args.format) && !is_srgb(view_format)
    }

    /// `vrend_resource_needs_srgb_encode`: a linear resource viewed as sRGB, with no view to do
    /// the encoding.
    pub fn needs_srgb_encode(&self, view_format: Format) -> bool {
        !self.supports_view() && !is_srgb(self.args.format) && is_srgb(view_format)
    }

    /// Check the guest's description and allocate on the current context.
    pub fn create(
        gl: &Gl,
        winsys: &Winsys,
        features: &Features,
        formats: &Table,
        limits: &Limits,
        args: Args,
    ) -> Result<Resource, Refusal> {
        check(features, formats, limits, &args)?;
        let storage = if args.target == TextureTarget::Buffer {
            alloc_buffer(gl, features, &args)?
        } else {
            let image = mint_surface(winsys, features, &args);
            alloc_texture(gl, features, formats, &args, image)?
        };
        Ok(Resource { args, storage })
    }

    /// The texture storage, for the operations only a texture has.
    pub fn texture(&self) -> Option<&Arc<Texture>> {
        match &self.storage {
            Storage::Texture(t) => Some(t),
            _ => None,
        }
    }

    /// Delete what only this resource holds. Texture storage a framebuffer is still attached to
    /// is handed back instead: the caller keeps it until the last attachment lets go.
    #[must_use = "texture storage still attached somewhere has to be kept, not dropped"]
    pub fn destroy(self, gl: &Gl) -> Option<Arc<Texture>> {
        match self.storage {
            Storage::Guest | Storage::Host(_) => None,
            Storage::Buffer { name, tbo, .. } => {
                if let Some(t) = tbo {
                    gl.delete_texture(t);
                }
                gl.delete_buffer(name);
                None
            }
            Storage::Texture(t) => match Arc::try_unwrap(t) {
                Ok(t) => {
                    t.destroy(gl);
                    None
                }
                Err(t) => Some(t),
            },
        }
    }
}

#[cfg(test)]
impl Texture {
    /// Texture storage that names no GL object, for tests about identity and nothing else.
    pub fn unbacked(name: TextureName) -> Texture {
        Texture { name, target: 0, immutable: true, image: None, views: Mutex::default() }
    }
}

impl fmt::Debug for Texture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Texture").field("name", &self.name).field("target", &self.target).finish()
    }
}

impl Texture {
    /// The render-target view for `key`, minted the first time it is asked for.
    ///
    /// `internalformat` is the format table's answer for `key.format`, and `levels` the texture's
    /// mip count: the caller has the format table and the resource's args, and this has neither.
    pub fn view(&self, gl: &Gl, key: ViewKey, internalformat: GLenum, levels: u32) -> TextureName {
        let mut views = self.views.lock().expect("the classic side never panics under this lock");
        *views.entry(key).or_insert_with(|| {
            let v = gl.gen_texture();
            gl.texture_view(
                v,
                self.target,
                self.name,
                internalformat,
                0,
                levels,
                key.first_layer,
                key.layers,
            );
            v
        })
    }

    /// The render-target view for `key`, if it has already been minted. Every surface mints its
    /// view at creation, so an attach only ever reads one back.
    pub fn view_texture(&self, key: ViewKey) -> Option<TextureName> {
        self.views.lock().expect("the classic side never panics under this lock").get(&key).copied()
    }

    /// Delete the GL objects, on a current context that shares with the one that made them. Only
    /// the last share does this, which is why it consumes the texture rather than taking `&self`.
    pub fn destroy(self, gl: &Gl) {
        for v in self.views.into_inner().expect("the classic side never panics under this lock") {
            gl.delete_texture(v.1);
        }
        // The image, and the surface it owns, go with the texture they were the storage of.
        gl.delete_texture(self.name);
    }
}

/// `vrend_format_is_bgra`: the formats GLES stores as RGBA and swaps on the way through.
pub fn is_bgra(format: Format) -> bool {
    matches!(format.name(), "B8G8R8A8_UNORM" | "B8G8R8X8_UNORM" | "B8G8R8A8_SRGB" | "B8G8R8X8_SRGB")
}

pub fn is_srgb(format: Format) -> bool {
    format.describe().is_some_and(|d| d.is_srgb())
}

pub fn minify(v: u32, level: u32) -> u32 {
    (v >> level.min(31)).max(1)
}

/// `check_resource_valid`, every rejection in the C's order.
fn check(features: &Features, formats: &Table, limits: &Limits, a: &Args) -> Result<(), Refusal> {
    use TextureTarget as T;
    let entry = formats.get(a.format);
    let can_texture_storage =
        features.has(Feature::texture_storage) && entry.is_some_and(|e| e.can_texture_storage);
    if a.nr_samples > 1 {
        if !entry.is_some_and(|e| e.can_multisample) {
            return Err(Refusal::UnsupportedMultisampleFormat);
        }
        if !matches!(a.target, T::Texture2d | T::Array2d) {
            return Err(Refusal::MultisampleNot2d);
        }
        if a.last_level > 0 {
            return Err(Refusal::MultisampleWithMipmaps);
        }
    }
    if a.last_level > 0 {
        if a.target == T::Buffer {
            return Err(Refusal::BufferWithMipmaps);
        }
        if a.target == T::Rect {
            return Err(Refusal::RectWithMipmaps);
        }
        // The C accepts one level more than the extent has: `floor(log2(max)) + 1`.
        let max = a.width.max(a.height).max(1);
        if a.last_level > max.ilog2() + 1 {
            return Err(Refusal::TooManyLevels);
        }
    }
    if a.flags.0 & !ResourceFlags::KNOWN != 0 {
        return Err(Refusal::UnknownFlags);
    }
    if a.flags.has(ResourceFlags::Y_0_TOP) && !matches!(a.target, T::Texture2d | T::Rect) {
        return Err(Refusal::Y0TopNot2d);
    }
    if a.target == T::Cube {
        if a.array_size != 6 {
            return Err(Refusal::CubeArraySize);
        }
    } else if a.target == T::CubeArray {
        if !features.has(Feature::cube_map_array) {
            return Err(Refusal::CubeArraysUnsupported);
        }
        if !a.array_size.is_multiple_of(6) {
            return Err(Refusal::CubeArrayArraySize);
        }
    } else if a.array_size > 1 {
        if !matches!(a.target, T::Array2d | T::Array1d) {
            return Err(Refusal::ArrayOfNonArrayTarget);
        }
        if !features.has(Feature::texture_array) {
            return Err(Refusal::ArraysUnsupported);
        }
    }
    if a.target != T::Buffer && a.width == 0 {
        return Err(Refusal::ZeroWidth);
    }
    if a.bind.is_buffer_bind() {
        if a.target != T::Buffer {
            return Err(Refusal::BufferBindOnTexture);
        }
        if a.height != 1 || a.depth != 1 {
            return Err(Refusal::BufferNotFlat);
        }
        if a.bind == Bind::QUERY_BUFFER && !features.has(Feature::qbo) {
            return Err(Refusal::QueryBuffersUnsupported);
        }
        if a.bind == Bind::COMMAND_ARGS && !features.has(Feature::indirect_draw) {
            return Err(Refusal::IndirectUnsupported);
        }
        return Ok(());
    }
    let texture_binds = Bind(
        Bind::SAMPLER_VIEW.0
            | Bind::DEPTH_STENCIL.0
            | Bind::RENDER_TARGET.0
            | Bind::CURSOR.0
            | Bind::SHARED.0
            | Bind::LINEAR.0,
    );
    if !a.bind.has(texture_binds) {
        return Err(Refusal::NoTextureBind);
    }
    // The C lets a buffer target with texture binds through to an allocation with GL target 0.
    if a.target == T::Buffer {
        return Err(Refusal::TextureBindOnBuffer);
    }
    if entry.is_none() {
        return Err(Refusal::UnsupportedFormat);
    }
    match a.target {
        T::Texture2d | T::Rect | T::Cube | T::Array2d | T::CubeArray => {
            if a.depth != 1 {
                return Err(Refusal::DepthOn2d);
            }
            if can_texture_storage && a.height == 0 {
                return Err(Refusal::ZeroHeight);
            }
        }
        T::Texture1d | T::Array1d => {
            if a.height != 1 || a.depth != 1 {
                return Err(Refusal::Not1dShape);
            }
            if a.width > limits.max_texture_2d_size {
                return Err(Refusal::TooLarge);
            }
        }
        T::Texture3d | T::Buffer => {}
    }
    match a.target {
        T::Texture2d | T::Rect | T::Array2d => {
            if a.width > limits.max_texture_2d_size || a.height > limits.max_texture_2d_size {
                return Err(Refusal::TooLarge);
            }
        }
        T::Texture3d => {
            if can_texture_storage && (a.height == 0 || a.depth == 0) {
                return Err(Refusal::ZeroDepth);
            }
            let m = limits.max_texture_3d_size;
            if a.width > m || a.height > m || a.depth > m {
                return Err(Refusal::TooLarge);
            }
        }
        _ => {}
    }
    if matches!(a.target, T::Array2d | T::CubeArray | T::Array1d)
        && can_texture_storage
        && a.array_size == 0
    {
        return Err(Refusal::ZeroArraySize);
    }
    if matches!(a.target, T::Cube | T::CubeArray) {
        if a.width != a.height {
            return Err(Refusal::CubeNotSquare);
        }
        if a.width > limits.max_texture_cube_size {
            return Err(Refusal::TooLarge);
        }
    }
    Ok(())
}

/// `vrend_resource_alloc_buffer`: the bind decides the target, and the target the store.
fn alloc_buffer(gl: &Gl, features: &Features, a: &Args) -> Result<Storage, Refusal> {
    let size = a.width as usize;
    let target = match a.bind {
        Bind::CUSTOM => return Ok(Storage::Host(Shadow::fresh(size))),
        Bind::STAGING => return Ok(Storage::Guest),
        Bind::INDEX_BUFFER => GL_ELEMENT_ARRAY_BUFFER,
        Bind::STREAM_OUTPUT => GL_TRANSFORM_FEEDBACK_BUFFER,
        Bind::VERTEX_BUFFER => GL_ARRAY_BUFFER,
        Bind::CONSTANT_BUFFER => GL_UNIFORM_BUFFER,
        Bind::QUERY_BUFFER => GL_QUERY_BUFFER_AMD,
        Bind::COMMAND_ARGS => GL_DRAW_INDIRECT_BUFFER,
        Bind(0) | Bind::SHADER_BUFFER => GL_ARRAY_BUFFER,
        b if b.has(Bind::SAMPLER_VIEW) => {
            if features.has(Feature::arb_or_gles_ext_texture_buffer) {
                GL_TEXTURE_BUFFER
            } else {
                GL_PIXEL_PACK_BUFFER
            }
        }
        _ => return Err(Refusal::IllegalBufferBind),
    };
    let mut storage_flags: GLbitfield = 0;
    if a.flags.has(ResourceFlags::MAP_PERSISTENT) {
        storage_flags |= GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
    }
    if a.flags.has(ResourceFlags::MAP_COHERENT) {
        storage_flags |= GL_MAP_COHERENT_BIT_EXT;
    }
    let name = gl.gen_buffer();
    gl.bind_buffer(target, Some(name));
    gl.drain_errors();
    let ok = if storage_flags != 0 {
        if !features.has(Feature::arb_buffer_storage) {
            // The C logs and leaves the buffer with no data store, reporting success. A buffer
            // the guest cannot map is a refusal, not a resource.
            gl.bind_buffer(target, None);
            gl.delete_buffer(name);
            return Err(Refusal::NoBufferStorage);
        }
        gl.buffer_storage_null(target, size, storage_flags)
    } else {
        gl.buffer_data_null(target, size, GL_STREAM_DRAW)
    };
    let err = gl.drain_errors();
    gl.bind_buffer(target, None);
    if !ok || err != GL_NO_ERROR {
        gl.delete_buffer(name);
        return Err(Refusal::GlError(err));
    }
    Ok(Storage::Buffer { name, target, tbo: None })
}

/// `tgsitargettogltarget`, with the GLES rewrites `vrend_resource_alloc_texture` applies after
/// it: RECT is never probed on GLES, 1D has no GL form.
pub fn gl_target(target: TextureTarget, nr_samples: u32) -> GLenum {
    match target {
        TextureTarget::Texture1d | TextureTarget::Rect => GL_TEXTURE_2D,
        TextureTarget::Texture2d if nr_samples > 1 => GL_TEXTURE_2D_MULTISAMPLE,
        TextureTarget::Texture2d => GL_TEXTURE_2D,
        TextureTarget::Texture3d => GL_TEXTURE_3D,
        TextureTarget::Cube => GL_TEXTURE_CUBE_MAP,
        TextureTarget::Array1d => GL_TEXTURE_2D_ARRAY,
        TextureTarget::Array2d if nr_samples > 1 => GL_TEXTURE_2D_MULTISAMPLE_ARRAY,
        TextureTarget::Array2d => GL_TEXTURE_2D_ARRAY,
        TextureTarget::CubeArray => GL_TEXTURE_CUBE_MAP_ARRAY,
        TextureTarget::Buffer => unreachable!("a buffer has no texture target"),
    }
}

/// `vrend_resource_iosurface_init`: the IOSurface a resource's storage is, when it is one.
///
/// A scanout is the compositor's framebuffer; a shared buffer is every buffer gbm hands out,
/// which is what a Vulkan compositor imports into venus for each client window. Both are minted
/// as surfaces so that the first is presented from without a copy and the second can be
/// imported at all -- there is no dma-buf to export on this host.
///
/// Only a single-level, single-sample 2D texture in a 32-bit format IOSurface and Metal both
/// name. Anything else keeps ordinary GL storage and the CPU readback path, as does a surface
/// the system or the driver refuses: the fallback is never removed, only reported.
fn mint_surface(winsys: &Winsys, features: &Features, a: &Args) -> Option<Image> {
    let scanout = a.bind.has(Bind::SCANOUT);
    if !scanout && !a.bind.has(Bind::SHARED) {
        return None;
    }
    if a.target != TextureTarget::Texture2d || a.last_level != 0 || a.nr_samples > 1 || a.depth != 1
    {
        return None;
    }
    let format = match a.format.name() {
        "B8G8R8A8_UNORM" | "B8G8R8X8_UNORM" => PixelFormat::Bgra,
        "R8G8B8A8_UNORM" | "R8G8B8X8_UNORM" => PixelFormat::Rgba,
        _ => return None,
    };
    let surface = match Surface::plain(a.width, a.height, format) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "[virglrs] vrend: no IOSurface for a {}x{} {} resource ({e:?}); it keeps GL storage",
                a.width,
                a.height,
                a.format.name()
            );
            return None;
        }
    };
    if !features.adopts_iosurfaces() {
        // Known at init and reported there; the resource keeps ordinary GL storage.
        return None;
    }
    match winsys.image_from_iosurface(Arc::new(surface)) {
        Ok(image) => {
            if scanout {
                eprintln!(
                    "[virglrs] vrend: iosurface scanout: {}x{} {} (IOSurface id {}); renders \
                     land in the surface directly",
                    a.width,
                    a.height,
                    a.format.name(),
                    image.surface().id().0
                );
            }
            Some(image)
        }
        // The driver said it imports IOSurfaces and then would not import one of the formats
        // and extents it accepts. That is the host contradicting itself, not the guest asking
        // for anything, and a blank window would hide it.
        Err(e) => panic!(
            "the driver imports IOSurfaces but refused a {}x{} {} one: {e}",
            a.width,
            a.height,
            a.format.name()
        ),
    }
}

/// `vrend_resource_alloc_texture`.
/// Write zeros over a texture's first level.
///
/// For storage nothing filled: an adopted surface carries the exporter's pixels, but a texture
/// that stood in for one carries whatever the driver's allocator last held there.
fn zero_texture(gl: &Gl, formats: &Table, a: &Args, storage: &Storage) {
    let (Storage::Texture(t), Some(entry)) = (storage, formats.get(a.format)) else {
        return;
    };
    let Some(bytes) = crate::vrend::gl::image_bytes(
        entry.gl.glformat,
        entry.gl.gltype,
        a.width as GLsizei,
        a.height as GLsizei,
        1,
    ) else {
        return;
    };
    let zeros = vec![0u8; bytes];
    gl.bind_texture(t.target, Some(t.name));
    gl.unpack_tight();
    gl.tex_sub_image_2d(
        t.target,
        0,
        0,
        0,
        a.width as GLsizei,
        a.height as GLsizei,
        entry.gl.glformat,
        entry.gl.gltype,
        &zeros,
    );
    gl.bind_texture(t.target, None);
}

fn alloc_texture(
    gl: &Gl,
    features: &Features,
    formats: &Table,
    a: &Args,
    image: Option<Image>,
) -> Result<Storage, Refusal> {
    let entry = formats.get(a.format).ok_or(Refusal::UnsupportedFormat)?;
    let mut immutable = features.has(Feature::texture_storage) && entry.can_texture_storage;
    let target = gl_target(a.target, a.nr_samples);
    let (ifmt, glformat, gltype) = (entry.gl.internalformat, entry.gl.glformat, entry.gl.gltype);
    let levels = (a.last_level + 1) as GLsizei;
    let (w, h) = (a.width as GLsizei, a.height as GLsizei);
    let name = gl.gen_texture();
    gl.bind_texture(target, Some(name));
    gl.drain_errors();
    if let Some(image) = image {
        // The surface becomes the texture's storage: immutable where the driver can make it so,
        // else through the older entry point, which leaves the texture mutable.
        let bound = if immutable && features.has(Feature::egl_image_storage) {
            gl.egl_image_target_tex_storage(target, &image);
            true
        } else if features.has(Feature::egl_image) {
            immutable = false;
            gl.egl_image_target_texture_2d(target, &image);
            true
        } else {
            false
        };
        let err = gl.drain_errors();
        // Nobody makes an EGL image without `Features::adopts_iosurfaces` first saying the host
        // takes them, so reaching here means the driver accepted the image and then would not
        // attach it. That is a host bug, and a resource that quietly kept GL storage instead
        // would hide it behind a window that merely renders the wrong thing.
        assert!(bound, "an EGL image with no entry point to bind it, past the feature probe");
        assert_eq!(
            err,
            GL_NO_ERROR,
            "the driver imported a {}x{} {} IOSurface and then would not bind it",
            a.width,
            a.height,
            a.format.name()
        );
        gl.bind_texture(target, None);
        return Ok(Storage::Texture(Arc::new(Texture {
            name,
            target,
            immutable,
            image: Some(image),
            views: Mutex::default(),
        })));
    }
    match target {
        _ if a.nr_samples > 1 => {
            let samples = a.nr_samples as GLsizei;
            if target == GL_TEXTURE_2D_MULTISAMPLE {
                gl.tex_storage_2d_multisample(target, samples, ifmt, w, h);
            } else {
                let d = a.array_size as GLsizei;
                if !features.has(Feature::storage_multisample_2d_array) {
                    gl.bind_texture(target, None);
                    gl.delete_texture(name);
                    return Err(Refusal::UnsupportedMultisampleFormat);
                }
                gl.tex_storage_3d_multisample(target, samples, ifmt, w, h, d);
            }
        }
        GL_TEXTURE_CUBE_MAP => {
            if immutable {
                gl.tex_storage_2d(target, levels, ifmt, w, h);
            } else {
                for face in 0..6 {
                    for level in 0..levels {
                        let l = level as u32;
                        gl.tex_image_2d_null(
                            GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                            level,
                            ifmt,
                            minify(a.width, l) as GLsizei,
                            minify(a.height, l) as GLsizei,
                            glformat,
                            gltype,
                        );
                    }
                }
            }
        }
        GL_TEXTURE_3D | GL_TEXTURE_2D_ARRAY | GL_TEXTURE_CUBE_MAP_ARRAY => {
            let layered = target != GL_TEXTURE_3D;
            if immutable {
                let d = if layered { a.array_size } else { a.depth } as GLsizei;
                gl.tex_storage_3d(target, levels, ifmt, w, h, d);
            } else {
                for level in 0..levels {
                    let l = level as u32;
                    let d = if layered { a.array_size } else { minify(a.depth, l) };
                    gl.tex_image_3d_null(
                        target,
                        level,
                        ifmt,
                        minify(a.width, l) as GLsizei,
                        minify(a.height, l) as GLsizei,
                        d as GLsizei,
                        glformat,
                        gltype,
                    );
                }
            }
        }
        _ => {
            if immutable {
                gl.tex_storage_2d(target, levels, ifmt, w, h);
            } else {
                for level in 0..levels {
                    let l = level as u32;
                    gl.tex_image_2d_null(
                        target,
                        level,
                        ifmt,
                        minify(a.width, l) as GLsizei,
                        minify(a.height, l) as GLsizei,
                        glformat,
                        gltype,
                    );
                }
            }
        }
    }
    let err = gl.drain_errors();
    if err != GL_NO_ERROR {
        gl.bind_texture(target, None);
        gl.delete_texture(name);
        return Err(Refusal::GlError(err));
    }
    if !immutable {
        gl.tex_parameter_i(target, GL_TEXTURE_BASE_LEVEL, 0);
        gl.tex_parameter_i(target, GL_TEXTURE_MAX_LEVEL, a.last_level as GLint);
    }
    gl.bind_texture(target, None);
    Ok(Storage::Texture(Arc::new(Texture {
        name,
        target,
        immutable,
        image: None,
        views: Mutex::default(),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vrend::formats::{Bindings, Entry, GlFormat, ViewClass};

    /// Guest pages a test can both hand to a `Shadow` and read back afterwards.
    ///
    /// The entry holds a raw pointer, exactly as one from the VMM does, so the buffer it points
    /// at has to outlive it -- which in each test below it does, being a local declared first.
    fn pages(buf: &mut [u8]) -> [crate::abi::GuestIov; 1] {
        [crate::abi::GuestIov { base: crate::abi::VmmPtr(buf.as_mut_ptr().cast()), len: buf.len() }]
    }

    /// The race this type exists for: the guest queues `RESOURCE_CREATE` and `ATTACH_BACKING`
    /// and starts writing without waiting for either, so a fresh resource's zeroed buffer must
    /// not be pushed on top of what it wrote.
    #[test]
    fn attaching_backing_to_a_fresh_resource_leaves_the_guest_bytes_alone() {
        let mut shadow = Shadow::fresh(4096);
        let mut guest = vec![0xa5u8; 4096];
        let entries = pages(&mut guest);

        assert!(shadow.mirror_into(&Iov::new(&entries)), "the pages hold the buffer");
        assert!(guest.iter().all(|&b| b == 0xa5), "the guest's own bytes are still there");
    }

    /// The one write-back that is owed: bytes that reached the host while no pages were there
    /// to receive them.
    #[test]
    fn attaching_backing_restores_what_the_guest_cannot_have() {
        let mut shadow = Shadow::fresh(4096);
        shadow.bytes_mut().fill(0x5a);
        shadow.unmirrored();

        let mut guest = vec![0u8; 4096];
        let entries = pages(&mut guest);
        assert!(shadow.mirror_into(&Iov::new(&entries)));
        assert!(guest.iter().all(|&b| b == 0x5a), "the host-only content is restored");
    }

    /// Paying the debt clears it, so the *next* attach is back to writing nothing -- the fresh
    /// case again, reached from a resource that has been round the loop.
    #[test]
    fn a_paid_attach_owes_the_next_one_nothing() {
        let mut shadow = Shadow::fresh(16);
        shadow.bytes_mut().fill(0x5a);
        shadow.unmirrored();

        let mut first = vec![0u8; 16];
        let entries = pages(&mut first);
        assert!(shadow.mirror_into(&Iov::new(&entries)));
        assert_eq!(first, [0x5a; 16], "the first attach is paid");

        let mut second = vec![0x3cu8; 16];
        let entries = pages(&mut second);
        assert!(shadow.mirror_into(&Iov::new(&entries)));
        assert_eq!(second, [0x3c; 16], "and the second is owed nothing");
    }

    /// A detach is what makes the buffer authoritative: the pages it captured are going away.
    #[test]
    fn what_a_detach_captured_reaches_the_next_backing() {
        let mut shadow = Shadow::fresh(16);

        let mut old = vec![0x3cu8; 16];
        let entries = pages(&mut old);
        assert!(Iov::new(&entries).copy_out(0, shadow.bytes_mut()));
        shadow.unmirrored();

        let mut new = vec![0u8; 16];
        let entries = pages(&mut new);
        assert!(shadow.mirror_into(&Iov::new(&entries)));
        assert_eq!(new, [0x3c; 16], "the detached bytes land in the fresh backing");
    }

    fn features() -> Features {
        Features::probe(31, Vec::new())
    }

    fn table() -> Table {
        let mut t = Table::empty();
        let rgba = Format::from_wire(67).unwrap();
        assert_eq!(rgba.name(), "R8G8B8A8_UNORM");
        t.insert(Entry {
            gl: GlFormat {
                format: rgba,
                internalformat: GL_RGBA8,
                glformat: GL_RGBA,
                gltype: GL_UNSIGNED_BYTE,
                swizzle: None,
                view_class: ViewClass::Bits32,
            },
            bindings: Bindings { sampler_view: true, render_target: true, depth_stencil: false },
            can_texture_storage: true,
            can_readback: true,
            can_multisample: false,
        });
        t
    }

    fn limits() -> Limits {
        Limits {
            max_texture_2d_size: 4096,
            max_texture_3d_size: 256,
            max_texture_cube_size: 4096,
            max_draw_buffers: 8,
            max_vertex_attributes: 16,
            max_texture_units: 32,
            max_texture_buffer_size: 65536,
        }
    }

    fn texture() -> Args {
        Args {
            target: TextureTarget::Texture2d,
            format: Format::from_wire(67).unwrap(),
            bind: Bind::SAMPLER_VIEW,
            width: 64,
            height: 64,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: ResourceFlags(0),
        }
    }

    #[test]
    fn the_checks_refuse_what_the_c_refuses() {
        let (f, t, l) = (features(), table(), limits());
        assert_eq!(check(&f, &t, &l, &texture()), Ok(()));
        type Tweak = fn(&mut Args);
        let cases: &[(Tweak, Refusal)] = &[
            (|a| a.format = Format::from_wire(1).unwrap(), Refusal::UnsupportedFormat),
            (|a| a.nr_samples = 4, Refusal::UnsupportedMultisampleFormat),
            (|a| a.last_level = 8, Refusal::TooManyLevels),
            (|a| a.flags = ResourceFlags(8), Refusal::UnknownFlags),
            (
                |a| {
                    a.flags = ResourceFlags::Y_0_TOP;
                    a.target = TextureTarget::Texture3d;
                },
                Refusal::Y0TopNot2d,
            ),
            (
                |a| {
                    a.target = TextureTarget::Cube;
                    a.array_size = 1;
                },
                Refusal::CubeArraySize,
            ),
            (|a| a.array_size = 2, Refusal::ArrayOfNonArrayTarget),
            (|a| a.width = 0, Refusal::ZeroWidth),
            (|a| a.bind = Bind::VERTEX_BUFFER, Refusal::BufferBindOnTexture),
            (|a| a.bind = Bind::SCANOUT, Refusal::NoTextureBind),
            (|a| a.target = TextureTarget::Buffer, Refusal::TextureBindOnBuffer),
            (|a| a.depth = 2, Refusal::DepthOn2d),
            (|a| a.height = 0, Refusal::ZeroHeight),
            (|a| a.width = 8192, Refusal::TooLarge),
            (
                |a| {
                    a.target = TextureTarget::Cube;
                    a.array_size = 6;
                    a.height = 32;
                },
                Refusal::CubeNotSquare,
            ),
        ];
        for (tweak, want) in cases {
            let mut a = texture();
            tweak(&mut a);
            assert_eq!(check(&f, &t, &l, &a), Err(*want), "{a:?}");
        }
        // One level more than the extent has is accepted, as in the C.
        let mut a = texture();
        a.last_level = 7;
        assert_eq!(check(&f, &t, &l, &a), Ok(()));
    }

    #[test]
    fn a_buffer_is_told_by_its_bind_and_must_be_flat() {
        let (f, t, l) = (features(), table(), limits());
        let buffer = Args {
            target: TextureTarget::Buffer,
            format: Format::from_wire(64).unwrap(),
            bind: Bind::VERTEX_BUFFER,
            width: 1024,
            height: 1,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: ResourceFlags(0),
        };
        assert_eq!(check(&f, &t, &l, &buffer), Ok(()));
        let mut tall = buffer;
        tall.height = 2;
        assert_eq!(check(&f, &t, &l, &tall), Err(Refusal::BufferNotFlat));
        let mut query = buffer;
        query.bind = Bind::QUERY_BUFFER;
        assert_eq!(check(&f, &t, &l, &query), Err(Refusal::QueryBuffersUnsupported));
        let mut mip = buffer;
        mip.last_level = 1;
        assert_eq!(check(&f, &t, &l, &mip), Err(Refusal::BufferWithMipmaps));
    }

    #[test]
    fn gles_has_no_1d_and_no_rect() {
        assert_eq!(gl_target(TextureTarget::Texture1d, 0), GL_TEXTURE_2D);
        assert_eq!(gl_target(TextureTarget::Rect, 0), GL_TEXTURE_2D);
        assert_eq!(gl_target(TextureTarget::Array1d, 0), GL_TEXTURE_2D_ARRAY);
        assert_eq!(gl_target(TextureTarget::Texture2d, 4), GL_TEXTURE_2D_MULTISAMPLE);
    }
}
