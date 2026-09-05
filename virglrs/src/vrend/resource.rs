// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! A classic resource on the host: what the guest asked for, checked, and the GL object (or the
//! host memory, or nothing) that backs it.
//!
//! The C's `vrend_resource` is one struct with a bitmask saying which of its overlapping fields
//! mean anything. Here the backing is an enum, so a transfer or a destroy is one match with no
//! flag to consult, and a buffer cannot be mistaken for a texture.

use super::egl::{self, Image, Winsys};
use super::features::{Feature, Features};
use super::formats::{Entry, Table};
use super::gl::gles::*;
use super::gl::{BufferName, GLbitfield, GLenum, GLint, GLsizei, Gl, TextureName};
use super::pipe::TextureTarget;
use super::proto::Format;
use super::video;
use crate::guest_mem::Iov;
use crate::metal::{Held, PixelFormat, PlanarFormat, Surface};
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

    /// What this bind names, decided once.
    ///
    /// Read it here and nowhere else. `Bind` looks like a bitmask and is one, but the buffer
    /// binds are told apart by **equality on the whole word** -- `VERTEX_BUFFER|SAMPLER_VIEW` is
    /// not a vertex buffer, it is the sampler case. The C states that twice, in
    /// `check_resource_valid` and again in `vrend_resource_alloc_buffer`, and a reader who takes
    /// either for a mask gets a different answer from the other.
    fn kind(self) -> BindKind {
        match self {
            Bind::CUSTOM => BindKind::HostShadow,
            Bind::STAGING => BindKind::GuestPages,
            Bind::INDEX_BUFFER => BindKind::Index,
            Bind::STREAM_OUTPUT => BindKind::StreamOutput,
            Bind::VERTEX_BUFFER => BindKind::Vertex,
            Bind::CONSTANT_BUFFER => BindKind::Constant,
            Bind::QUERY_BUFFER => BindKind::Query,
            Bind::COMMAND_ARGS => BindKind::CommandArgs,
            Bind(0) | Bind::SHADER_BUFFER => BindKind::Plain,
            b if b.has(Bind::SAMPLER_VIEW) => BindKind::Sampled,
            _ => BindKind::Other,
        }
    }
}

/// What a bind names. See [`Bind::kind`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BindKind {
    /// `VIRGL_BIND_CUSTOM`: host memory, and no GL object at all.
    HostShadow,
    /// `VIRGL_BIND_STAGING`: the guest's own pages are the storage.
    GuestPages,
    Index,
    StreamOutput,
    Vertex,
    Constant,
    Query,
    CommandArgs,
    /// No bind at all, or a shader buffer: a plain array buffer either way.
    Plain,
    /// Not equal to any of the above, but carrying `SAMPLER_VIEW` -- a texture buffer on a
    /// buffer target, an ordinary texture on any other.
    Sampled,
    /// Neither, which on a buffer target is nothing this can make.
    Other,
}

impl BindKind {
    /// Whether this is one of the binds the C tells buffers by, which is what selects the
    /// buffer-shaped rules rather than the texture-shaped ones.
    fn is_buffer_bind(self) -> bool {
        !matches!(self, BindKind::Sampled | BindKind::Other)
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
    /// A multi-plane format with nothing to back its planes with.
    NoPlanarStorage,
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
    /// A blob given a type that is not texture storage. Blobs are adopted as textures and there
    /// is nothing else here to make one into.
    NotTextureStorage,
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
            Refusal::NoPlanarStorage => "no planar surface to back a multi-plane target",
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
            Refusal::NotTextureStorage => "a blob typed as something other than a texture",
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
        // A blob being given a type is always given a texture's. The guest states the target,
        // so this asks rather than assumes: `gl_target` has no answer for a buffer and would
        // abort on one, and a guest must never be able to do that.
        let gl_target = match plan(features, formats, limits, &args) {
            Ok(Plan::Texture { gl_target }) => gl_target,
            Ok(_) => return Err((self, Refusal::NotTextureStorage)),
            Err(e) => return Err((self, e)),
        };
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
        // No planes: this is a resource adopting a surface it was handed, and a composite
        // target is never one of those -- it mints its own, and its planes are cut from that.
        let storage = match alloc_texture(gl, features, formats, &args, gl_target, image, None) {
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
/// The planes of a composite decode target: one image per plane of one planar IOSurface.
///
/// Held beside the texture's own storage rather than as it. A guest that samples the planar
/// format whole lands on the texture, which the decode path converts into; one that asks for a
/// plane by index lands here. Replacing the texture's storage would serve the second by breaking
/// the first.
///
/// Both planes or neither. A target holding an image for one of them is one whose chroma view
/// silently samples luma, which is a green picture and no error anywhere -- so the type does not
/// admit it.
pub struct Planes {
    luma: Image,
    chroma: Image,
    /// Whether the base texture a composite view samples is in step with these planes.
    conversion: Mutex<Conversion>,
    /// The conversion pass's own textures over the two plane images, made on its first run.
    ///
    /// Once per resource, never once per frame: each import adopts the IOSurface plane on the
    /// driver side, and importing per frame leaks that adoption as surely as a per-frame view
    /// would. Deleted with the texture they hang off.
    ///
    /// The lock is the one [`Texture::views`] documents -- uncontended, and what buys the share
    /// the `Send` a `RefCell` would cost it.
    textures: Mutex<Option<[TextureName; 2]>>,
    /// The layout the surface was minted in. Kept because the kernel does not report a plane's
    /// element size and every other half of a plane's geometry comes from the surface: keeping
    /// the format that does state it means nothing re-derives it from the resource's own.
    planar: PlanarFormat,
}

/// Whether a composite target's base texture is in step with its planes, and whether anything
/// reads it.
///
/// The C keeps two booleans, `composite_sampled` and `planes_dirty`, and every site tests the
/// pair. They only mean anything together: a target no composite view has sampled yet must still
/// remember that a picture landed, so the first such view converts what is already there rather
/// than showing an empty frame; and a target nothing has delivered into has nothing to convert
/// however it is sampled. So they are one value, and the state the pair can spell but nothing
/// establishes -- filled, with nobody looking -- is not in it.
///
/// The conversion runs at whichever of the two events comes second, and only then: a target only
/// per-plane consumers ever read pays nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Conversion {
    /// No composite view yet, and nothing delivered since the last fill.
    #[default]
    Unwatched,
    /// No composite view yet, but a picture has landed. The first composite view converts it.
    UnwatchedPending,
    /// A composite view samples the base texture, and it is in step with the planes.
    Current,
    /// A composite view samples the base texture, and a picture has landed since it was filled.
    Pending,
}

impl Conversion {
    /// A picture was delivered into the planes.
    pub fn delivered(self) -> Conversion {
        match self {
            Conversion::Unwatched | Conversion::UnwatchedPending => Conversion::UnwatchedPending,
            Conversion::Current | Conversion::Pending => Conversion::Pending,
        }
    }

    /// A composite view of the target was made. Sticky: nothing takes a target back to unwatched,
    /// because a view that existed once may be sampled again at any time.
    pub fn sampled(self) -> Conversion {
        match self {
            Conversion::Unwatched | Conversion::Current => Conversion::Current,
            Conversion::UnwatchedPending | Conversion::Pending => Conversion::Pending,
        }
    }

    /// The conversion ran and succeeded. A failed pass must not call this: leaving the state
    /// pending is what makes the next event retry, where claiming success would leave the base
    /// texture a frame behind for good.
    pub fn filled(self) -> Conversion {
        match self {
            Conversion::Pending | Conversion::Current => Conversion::Current,
            unwatched => unwatched,
        }
    }

    /// Whether the pass should run now. The one question both trigger sites ask.
    pub fn needs_fill(self) -> bool {
        self == Conversion::Pending
    }
}

/// The tight extent of one plane of a composite decode target.
///
/// Tight is the point. A plane has three row lengths around it -- the decoder's pitch, the
/// surface's pitch, and this -- and only this one is the picture. The other two are padded, and
/// neither is padded the way the other is, so copying by either shears the picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneGeometry {
    pub width: u32,
    pub height: u32,
    pub bytes_per_element: u32,
}

impl PlaneGeometry {
    /// The picture's own row.
    pub fn row_bytes(self) -> usize {
        (self.width as usize) * (self.bytes_per_element as usize)
    }
}

/// Which plane a sampler view of a plane-backed resource is asking for -- see [`PlaneRequest`].
///
/// Free of the planes themselves so it can be reasoned about without a driver to make them; the
/// only thing it needs from them is how many there are.
pub fn plane_request(
    planes: u32,
    resource_format: Format,
    view_format: Format,
    indexed: Option<u32>,
) -> PlaneRequest {
    if let Some(index) = indexed {
        // An index past the planes there are is spent, not refused: the C does the same, and a
        // refused view costs the guest its context for the rest of its life, which is far past
        // what a bad index is worth.
        return if index < planes { PlaneRequest::Plane(index) } else { PlaneRequest::Ordinary };
    }
    if view_format == resource_format {
        return PlaneRequest::Composite;
    }
    // Unindexed and naming something else: a plane, and its own format says which. Two
    // components is the interleaved chroma plane; anything else is luma. Only a two-plane
    // surface can be named this way at all -- a three-plane format has R8 for every plane and
    // so says nothing -- which is what this type being biplanar already settles.
    let components = view_format.describe().map_or(0, |d| d.nr_channels);
    PlaneRequest::Plane(u32::from(components == 2))
}

/// What a sampler view of a plane-backed resource is asking for.
///
/// One decision read from both ends. A view naming a *component* format on a planar resource is
/// asking for a plane; a view naming the resource's own planar format is a consumer asking for
/// the whole thing, which only exists once the planes have been converted into the base texture.
/// Deciding the two apart, in two places, is how they drift -- and the pair that drifts here is a
/// luma view sampling RGBA and a composite view that never arms the conversion, neither of which
/// reports anything.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlaneRequest {
    /// Plane `n` of the surface, sampled through its own image.
    Plane(u32),
    /// The planar format itself. Nothing fills the base texture on this path but the conversion,
    /// so a resource sampled this way owes one.
    Composite,
    /// Neither: an ordinary view, of an ordinary resource or of a plane-backed one in a way that
    /// names no plane.
    Ordinary,
}

impl Planes {
    /// Which plane, if any, a sampler view of this resource is asking for.
    ///
    /// `indexed` is the plane the guest packed into the layer range, which arrives only when it
    /// is nonzero: `virgl_encode_sampler_view` writes the index only then, so plane 0 has to be
    /// recognised some other way. Its view format is that other way -- a component format on a
    /// planar resource can be nothing but a plane request.
    ///
    /// Both signals are gated on the planes actually being here, so a resource that was never
    /// given any cannot reach this however it is sampled.
    pub fn request(
        &self,
        resource_format: Format,
        view_format: Format,
        indexed: Option<u32>,
    ) -> PlaneRequest {
        plane_request(self.count(), resource_format, view_format, indexed)
    }

    /// The image for plane `index`, or `None` past the planes there are.
    pub fn image(&self, index: u32) -> Option<&Image> {
        match index {
            0 => Some(&self.luma),
            1 => Some(&self.chroma),
            _ => None,
        }
    }

    /// The extent and element size of plane `index`, or `None` past the planes there are.
    ///
    /// The extent is read back from the surface rather than derived from the decode target's
    /// own width and height. Those are two separately stated numbers -- the resource's, which
    /// the surface was cut to, and the video buffer's, which the guest sends again in
    /// CREATE_VIDEO_BUFFER -- and only the first describes what is actually there.
    pub fn geometry(&self, index: u32) -> Option<PlaneGeometry> {
        let (shape, _pitch) = self.surface().plane(index)?;
        Some(PlaneGeometry {
            width: shape.width,
            height: shape.height,
            bytes_per_element: self.planar.bytes_per_element(index as usize),
        })
    }

    /// Note that a picture was delivered into the planes, and say whether the base texture must
    /// now be converted.
    pub fn delivered(&self) -> bool {
        self.transition(Conversion::delivered)
    }

    /// Note that a composite view of the target was made, and say whether the base texture must
    /// be converted before it is sampled.
    pub fn sampled(&self) -> bool {
        self.transition(Conversion::sampled)
    }

    /// Note that the conversion ran and succeeded. A failed pass does not call this -- see
    /// [`Conversion::filled`].
    pub fn filled(&self) {
        self.transition(Conversion::filled);
    }

    fn transition(&self, step: impl FnOnce(Conversion) -> Conversion) -> bool {
        let mut state =
            self.conversion.lock().expect("the classic side never panics under this lock");
        *state = step(*state);
        state.needs_fill()
    }

    /// Whether the conversion pass is owed. See [`Conversion::needs_fill`].
    pub fn needs_fill(&self) -> bool {
        self.conversion.lock().expect("the classic side never panics under this lock").needs_fill()
    }

    /// The conversion pass's textures over the two plane images, made on the first call.
    ///
    /// The caller has the blitter's context current; these are made there and used only there.
    pub fn textures(&self, gl: &Gl) -> [TextureName; 2] {
        let mut slot = self.textures.lock().expect("the classic side never panics under this lock");
        *slot.get_or_insert_with(|| {
            [&self.luma, &self.chroma].map(|image| {
                let name = gl.gen_texture();
                gl.bind_texture(GL_TEXTURE_2D, Some(name));
                gl.egl_image_target_texture_2d(GL_TEXTURE_2D, image);
                for wrap in [GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T] {
                    gl.tex_parameter_i(GL_TEXTURE_2D, wrap, GL_CLAMP_TO_EDGE as GLint);
                }
                // LINEAR is load-bearing on the chroma plane: it is half resolution and is
                // sampled at the luma's coordinates, so this filter is the upsampler.
                for filter in [GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER] {
                    gl.tex_parameter_i(GL_TEXTURE_2D, filter, GL_LINEAR as GLint);
                }
                for level in [GL_TEXTURE_BASE_LEVEL, GL_TEXTURE_MAX_LEVEL] {
                    gl.tex_parameter_i(GL_TEXTURE_2D, level, 0);
                }
                gl.bind_texture(GL_TEXTURE_2D, None);
                name
            })
        })
    }

    /// The GL objects this made, given up so the caller can delete them. Called once, by the
    /// texture these hang off, as it is destroyed.
    fn into_textures(self) -> Option<[TextureName; 2]> {
        self.textures.into_inner().expect("the classic side never panics under this lock")
    }

    /// The surface both planes are cut from. One surface, so either image answers.
    pub fn surface(&self) -> &Surface {
        self.luma.surface()
    }

    /// How many planes there are. A biplanar surface, so always two -- named rather than
    /// written as a literal at each call site.
    pub fn count(&self) -> u32 {
        2
    }
}

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
    /// The planes, when this resource is a composite decode target. See [`Planes`].
    pub planes: Option<Planes>,
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

    /// The planes, if this resource is a composite decode target.
    pub fn planes(&self) -> Option<&Planes> {
        match &self.storage {
            Storage::Texture(t) => t.planes.as_ref(),
            _ => None,
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
        let storage = match plan(features, formats, limits, &args)? {
            Plan::HostShadow => Storage::Host(Shadow::fresh(args.width as usize)),
            Plan::GuestPages => Storage::Guest,
            Plan::Buffer { gl_target, storage_flags } => {
                alloc_buffer(gl, &args, gl_target, storage_flags)?
            }
            Plan::Texture { gl_target } => {
                let planes = mint_planes(winsys, features, &args);
                // A resource in a format that has more than one plane, with nothing backing them,
                // is one whose plane views would find no image and fall through to a texture view
                // the format has no view class for -- which puts the guest's context in error for
                // its lifetime. The guest never sees this refusal: the kernel handed it the handle
                // before we were asked, so it will use the resource and poison itself either way.
                // What keeps it from asking is the capset, which offers a planar format only where
                // `composite_target_backable` says it can be backed. This fires when that contract
                // is broken -- IOSurfaces off, an allocation refused, the two sides disagreeing on
                // a format -- and it fails loudly rather than corrupting.
                if planes.is_none() && video::guest_planes(args.format) > 1 {
                    eprintln!(
                        "[virglrs] vrend: no planar surface for a {}x{} {} target; refusing the \
                         create (the guest cannot see this and will poison its context -- the \
                         capset should not have let it ask)",
                        args.width,
                        args.height,
                        args.format.name()
                    );
                    return Err(Refusal::NoPlanarStorage);
                }
                let image = mint_surface(winsys, features, &args);
                alloc_texture(gl, features, formats, &args, gl_target, image, planes)?
            }
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
        Texture {
            name,
            target: 0,
            immutable: true,
            image: None,
            planes: None,
            views: Mutex::default(),
        }
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
        // The conversion pass's textures over the plane images. The images themselves, and the
        // surface they share, go with the `Planes` this consumes.
        for t in self.planes.and_then(Planes::into_textures).into_iter().flatten() {
            gl.delete_texture(t);
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

/// What a set of creation arguments describes, once it has been understood.
///
/// **The point of this type is that it is the only description.** The bug it exists to prevent
/// is a rule in the checking that says a shape cannot be made, while the allocating has an arm
/// that makes it -- two accounts of what this build can create, in two functions, disagreeing
/// silently. A create refusal reaches no guest: the kernel handed out the handle before we were
/// asked and mesa does not read the control queue's error, so the guest transfers into a
/// resource that does not exist and dies on a later command naming a handle we never made. The
/// disagreement is therefore not a wrong error message, it is a hung desktop attributed to
/// something else entirely.
///
/// So the understanding and the refusing happen in one place and produce this, and allocating is
/// a total match on it. A rule refusing a shape now has to be written as an arm of the same match
/// that would otherwise say how to build it -- adjacent lines, not two files.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Plan {
    /// Host memory and no GL object: `VIRGL_BIND_CUSTOM`.
    HostShadow,
    /// The guest's own pages, with nothing allocated here: `VIRGL_BIND_STAGING`.
    GuestPages,
    /// A GL buffer at this target. `GL_TEXTURE_BUFFER` is one of them -- a buffer sampled
    /// through a texture is still a buffer, and the C allocates it as one.
    Buffer { gl_target: GLenum, storage_flags: GLbitfield },
    /// A GL texture at this target.
    Texture { gl_target: GLenum },
}

/// `check_resource_valid` and the head of the C's create path, every rejection in the C's order.
fn plan(features: &Features, formats: &Table, limits: &Limits, a: &Args) -> Result<Plan, Refusal> {
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
    if a.bind.kind().is_buffer_bind() {
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
        return plan_storage(features, a);
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
    // A buffer target reaching here is a texture buffer -- a buffer sampled through a texture --
    // and it is `alloc_buffer` that makes one, so it is not refused and none of the per-target
    // shape rules below name it. It arrives with a texture bind rather than a buffer one, which
    // is why it missed the buffer branch above; what decides its storage is its target, and the
    // creating path reads the target.
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
    plan_storage(features, a)
}

/// Which storage the arguments name, once they are known to be coherent.
///
/// The target alone decides buffer from texture, exactly as the C's create path does. The bind
/// says which *kind* of buffer, and that is the only place it is asked.
fn plan_storage(features: &Features, a: &Args) -> Result<Plan, Refusal> {
    if a.target != TextureTarget::Buffer {
        let gl_target = gl_target(a.target, a.nr_samples);
        // A multisample array needs the entry point that makes one. Decided here rather than
        // half-way through allocating, where the refusal arrives after a texture has been
        // generated and has to be unwound.
        if a.nr_samples > 1
            && gl_target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY
            && !features.has(Feature::storage_multisample_2d_array)
        {
            return Err(Refusal::UnsupportedMultisampleFormat);
        }
        return Ok(Plan::Texture { gl_target });
    }
    let gl_target = match a.bind.kind() {
        BindKind::HostShadow => return Ok(Plan::HostShadow),
        BindKind::GuestPages => return Ok(Plan::GuestPages),
        BindKind::Index => GL_ELEMENT_ARRAY_BUFFER,
        BindKind::StreamOutput => GL_TRANSFORM_FEEDBACK_BUFFER,
        BindKind::Vertex => GL_ARRAY_BUFFER,
        BindKind::Constant => GL_UNIFORM_BUFFER,
        BindKind::Query => GL_QUERY_BUFFER_AMD,
        BindKind::CommandArgs => GL_DRAW_INDIRECT_BUFFER,
        BindKind::Plain => GL_ARRAY_BUFFER,
        // A texture buffer. This arm is why no rule above may refuse a buffer target for
        // carrying a texture bind: the two would be contradicting each other from four lines
        // apart, which is the whole reason the decision is one function.
        BindKind::Sampled => {
            if features.has(Feature::arb_or_gles_ext_texture_buffer) {
                GL_TEXTURE_BUFFER
            } else {
                GL_PIXEL_PACK_BUFFER
            }
        }
        BindKind::Other => return Err(Refusal::IllegalBufferBind),
    };
    let mut storage_flags: GLbitfield = 0;
    if a.flags.has(ResourceFlags::MAP_PERSISTENT) {
        storage_flags |= GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
    }
    if a.flags.has(ResourceFlags::MAP_COHERENT) {
        storage_flags |= GL_MAP_COHERENT_BIT_EXT;
    }
    if storage_flags != 0 && !features.has(Feature::arb_buffer_storage) {
        // The C logs and leaves the buffer with no data store, reporting success. A buffer the
        // guest cannot map is a refusal, not a resource.
        return Err(Refusal::NoBufferStorage);
    }
    Ok(Plan::Buffer { gl_target, storage_flags })
}

/// `vrend_resource_alloc_buffer`, past the deciding: make the buffer [`plan`] asked for.
///
/// It takes the target and the storage flags rather than the bind, so there is nothing here to
/// disagree with the plan about. Every refusal that is a *decision* has already happened; what
/// is left is the driver's answer.
fn alloc_buffer(
    gl: &Gl,
    a: &Args,
    target: GLenum,
    storage_flags: GLbitfield,
) -> Result<Storage, Refusal> {
    let size = a.width as usize;
    let name = gl.gen_buffer();
    gl.bind_buffer(target, Some(name));
    gl.drain_errors();
    let ok = if storage_flags != 0 {
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

/// The planar IOSurface behind a composite decode target, and an image per plane.
///
/// A composite target is one resource in a planar format with its planes chained behind it,
/// which is the shape a guest takes when the capset says this host can back it. It is neither a
/// scanout nor shared, so it never reaches the bind gate in [`mint_surface`]: its format is what
/// identifies it, and nothing else on this host carries a planar one.
///
/// `None` for every format this build cannot back, and for a surface the system or the driver
/// refuses -- the caller turns that into a refused create rather than a resource whose planes
/// cannot be sampled.
fn mint_planes(winsys: &Winsys, features: &Features, a: &Args) -> Option<Planes> {
    if !video::composite_target_backable(features, a.format) {
        return None;
    }
    if a.target != TextureTarget::Texture2d || a.last_level != 0 || a.nr_samples > 1 || a.depth != 1
    {
        return None;
    }
    // Named once: the surface is cut to this layout and the planes are read back by it, and a
    // second statement of it is the pair that drifts.
    let planar = PlanarFormat::BiPlanar420;
    let surface = match Surface::planar(a.width, a.height, planar) {
        Ok(surface) => Arc::new(surface),
        Err(e) => {
            eprintln!(
                "[virglrs] vrend: no planar IOSurface for a {}x{} {} target ({e:?})",
                a.width,
                a.height,
                a.format.name()
            );
            return None;
        }
    };
    // Both planes of the one surface, each holding its own share of it: the surface outlives
    // whichever image is dropped last.
    let plane = |which| match winsys.image_from_iosurface_plane(Arc::clone(&surface) as _, which) {
        Ok(image) => Some(image),
        Err(e) => {
            eprintln!(
                "[virglrs] vrend: the driver refused plane {which:?} of a {}x{} {} target ({e})",
                a.width,
                a.height,
                a.format.name()
            );
            None
        }
    };
    let (luma, chroma) = (plane(egl::Plane::Luma)?, plane(egl::Plane::ChromaPair)?);
    eprintln!(
        "[virglrs] vrend: composite target: {}x{} {} on a two-plane IOSurface (id {}); plane \
         views sample the surface directly",
        a.width,
        a.height,
        a.format.name(),
        surface.id().0
    );
    Some(Planes { luma, chroma, planar, conversion: Mutex::default(), textures: Mutex::default() })
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
    target: GLenum,
    image: Option<Image>,
    planes: Option<Planes>,
) -> Result<Storage, Refusal> {
    let entry = formats.get(a.format).ok_or(Refusal::UnsupportedFormat)?;
    let mut immutable = features.has(Feature::texture_storage) && entry.can_texture_storage;
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
            planes,
            views: Mutex::default(),
        })));
    }
    match target {
        _ if a.nr_samples > 1 => {
            let samples = a.nr_samples as GLsizei;
            if target == GL_TEXTURE_2D_MULTISAMPLE {
                gl.tex_storage_2d_multisample(target, samples, ifmt, w, h);
            } else {
                // The entry point was required by `plan`, so reaching here without it is a
                // host invariant broken, not a guest asking for something.
                let d = a.array_size as GLsizei;
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
        planes,
        views: Mutex::default(),
    })))
}

#[cfg(test)]
mod tests {
    use super::Conversion;
    use super::Format;
    use super::{PlaneRequest, plane_request};

    /// The conversion state, walked as the two trigger sites walk it.
    ///
    /// Small enough to read and still get wrong: the whole point of the state is that the pass
    /// runs at whichever event comes second, so every path has to be entered from both ends.
    #[test]
    fn a_conversion_is_owed_at_whichever_of_the_two_events_comes_second() {
        // Delivery first, then the view. This is the order the C's pair of booleans exists for:
        // a frame that landed before anything looked must still be converted when something does.
        let landed = Conversion::default().delivered();
        assert!(!landed.needs_fill(), "nothing samples it yet, so nothing is owed");
        assert!(landed.sampled().needs_fill(), "the first composite view converts what is there");

        // The view first, then delivery.
        let watched = Conversion::default().sampled();
        assert!(!watched.needs_fill(), "nothing has been delivered, so there is nothing to show");
        assert!(watched.delivered().needs_fill());

        // A successful pass clears it, and the next picture owes another.
        let filled = watched.delivered().filled();
        assert!(!filled.needs_fill());
        assert!(filled.delivered().needs_fill());

        // A failed pass does not call `filled`, so the debt is still there for the next event to
        // find. Stated as a test because the retry is the only thing keeping a target from
        // sitting a frame behind for the rest of its life.
        let owed = watched.delivered();
        assert!(owed.sampled().needs_fill(), "a failed pass leaves it owed");

        // Sampling is sticky: a view made once may be sampled again at any time, and a target
        // that forgot would deliver into planes nothing converts.
        assert_eq!(watched.filled().delivered(), Conversion::Pending);

        // ... and filling an unwatched target does not make it watched. Reaching `filled` from
        // there means a pass ran for nobody, which is a bug elsewhere; it must not be recorded
        // as a composite consumer regardless.
        assert_eq!(landed.filled(), Conversion::UnwatchedPending);
    }

    /// One decision, read from both ends: which view is a plane and which is the whole thing.
    ///
    /// Both halves are here together because deciding them apart is how they drift, and the pair
    /// that drifts is a luma view sampling RGBA and a composite view that never arms the
    /// conversion -- neither of which reports anything.
    #[test]
    fn a_plane_view_and_a_composite_view_are_told_apart_by_format() {
        // By name, because a wire number written down here is a number that goes stale
        // silently: the table is generated, and the two that matter are the plane formats.
        let by_name = |name: &str| {
            (0..1024)
                .filter_map(Format::from_wire)
                .find(|f| f.describe().is_some_and(|d| d.name == name))
                .unwrap_or_else(|| panic!("no format {name}"))
        };
        let nv12 = by_name("Y8_U8V8_420_UNORM");
        let r8 = by_name("R8_UNORM");
        let rg88 = by_name("R8G8_UNORM");
        // The composite target path and the video capset name this format by two different
        // routes; they have to be the same format or the two halves guard different things.
        assert_eq!(nv12.wire(), 166, "NV12's wire number moved");
        assert_eq!(rg88.describe().expect("described").nr_channels, 2, "chroma is two channels");

        // The index arrives only when it is nonzero, so plane 0 is named by its format alone.
        assert_eq!(plane_request(2, nv12, r8, None), PlaneRequest::Plane(0));
        assert_eq!(plane_request(2, nv12, rg88, None), PlaneRequest::Plane(1));
        // ... and an index, when there is one, is the whole answer.
        assert_eq!(plane_request(2, nv12, r8, Some(1)), PlaneRequest::Plane(1));
        assert_eq!(plane_request(2, nv12, rg88, Some(0)), PlaneRequest::Plane(0));

        // The resource's own format is the consumer asking for the converted whole.
        assert_eq!(plane_request(2, nv12, nv12, None), PlaneRequest::Composite);

        // An index past the planes is spent rather than refused: a refused view would cost the
        // guest its context for the rest of its life.
        assert_eq!(plane_request(2, nv12, r8, Some(2)), PlaneRequest::Ordinary);
        assert_eq!(plane_request(2, nv12, r8, Some(9)), PlaneRequest::Ordinary);
    }

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
        // A second format that multisamples, because the whole multisample branch is otherwise
        // unreachable: every rule under `nr_samples > 1` is guarded by the entry saying the
        // format can, and one format that cannot leaves all of them unswept.
        let bgrx = Format::from_wire(2).unwrap();
        assert_eq!(bgrx.name(), "B8G8R8X8_UNORM");
        t.insert(Entry {
            gl: GlFormat {
                format: bgrx,
                internalformat: GL_RGBA8,
                glformat: GL_RGBA,
                gltype: GL_UNSIGNED_BYTE,
                swizzle: None,
                view_class: ViewClass::Bits32,
            },
            bindings: Bindings { sampler_view: true, render_target: true, depth_stencil: false },
            can_texture_storage: true,
            can_readback: true,
            can_multisample: true,
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

    /// `check_resource_valid`, transcribed from `src/vrend/vrend_renderer.c` and used for
    /// nothing but disagreeing with our own.
    ///
    /// Deliberately a second implementation rather than a call into the first: what it is worth
    /// is that it was written from the C and reads in the C's order, so a rule ours has and the
    /// C does not cannot appear in it by construction. It answers only accept-or-refuse, because
    /// the C's reason is a formatted string and ours is a variant, and pinning a mapping between
    /// them would be pinning our own translation.
    ///
    /// The GBM early return is not transcribed: this build has no GBM, so the C never takes it.
    fn c_check(features: &Features, formats: &Table, limits: &Limits, a: &Args) -> bool {
        type T = TextureTarget;
        let entry = formats.get(a.format);
        let fcts = entry.is_some_and(|e| e.can_texture_storage);
        let can_multisample = entry.is_some_and(|e| e.can_multisample);
        if a.nr_samples > 1 {
            if !can_multisample {
                return false;
            }
            if !matches!(a.target, T::Texture2d | T::Array2d) {
                return false;
            }
            if a.last_level > 0 {
                return false;
            }
        }
        if a.last_level > 0 {
            if matches!(a.target, T::Buffer | T::Rect) {
                return false;
            }
            let cap = (f64::from(a.width.max(a.height)).log2().floor() as u32) + 1;
            if a.last_level > cap {
                return false;
            }
        }
        if a.flags.0 != 0 {
            let supported = ResourceFlags::Y_0_TOP.0
                | ResourceFlags::MAP_PERSISTENT.0
                | ResourceFlags::MAP_COHERENT.0;
            if a.flags.0 & !supported != 0 {
                return false;
            }
        }
        if a.flags.0 & ResourceFlags::Y_0_TOP.0 != 0 && !matches!(a.target, T::Texture2d | T::Rect)
        {
            return false;
        }
        if a.target == T::Cube {
            if a.array_size != 6 {
                return false;
            }
        } else if a.target == T::CubeArray {
            if !features.has(Feature::cube_map_array) {
                return false;
            }
            if !a.array_size.is_multiple_of(6) {
                return false;
            }
        } else if a.array_size > 1 {
            if !matches!(a.target, T::Array2d | T::Array1d) {
                return false;
            }
            if !features.has(Feature::texture_array) {
                return false;
            }
        }
        if a.target != T::Buffer && a.width == 0 {
            return false;
        }
        // The C matches the buffer binds by equality, and everything else is a texture.
        let buffer_bind = matches!(
            a.bind,
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
        );
        if buffer_bind {
            if a.target != T::Buffer {
                return false;
            }
            if a.height != 1 || a.depth != 1 {
                return false;
            }
            if a.bind == Bind::QUERY_BUFFER && !features.has(Feature::qbo) {
                return false;
            }
            if a.bind == Bind::COMMAND_ARGS && !features.has(Feature::indirect_draw) {
                return false;
            }
            return true;
        }
        let texture_bind = a.bind.has(Bind(
            Bind::SAMPLER_VIEW.0
                | Bind::DEPTH_STENCIL.0
                | Bind::RENDER_TARGET.0
                | Bind::CURSOR.0
                | Bind::SHARED.0
                | Bind::LINEAR.0,
        ));
        if !texture_bind {
            return false;
        }
        // Note what is NOT here: nothing in this branch names PIPE_BUFFER. A buffer target with
        // a texture bind reaches the end and is accepted, and the C then allocates it as a
        // buffer, because the create path routes on the target alone.
        if matches!(a.target, T::Texture2d | T::Rect | T::Cube | T::Array2d | T::CubeArray) {
            if a.depth != 1 {
                return false;
            }
            if fcts && a.height == 0 {
                return false;
            }
        }
        if matches!(a.target, T::Texture1d | T::Array1d) {
            if a.height != 1 || a.depth != 1 {
                return false;
            }
            if a.width > limits.max_texture_2d_size {
                return false;
            }
        }
        if matches!(a.target, T::Texture2d | T::Rect | T::Array2d)
            && (a.width > limits.max_texture_2d_size || a.height > limits.max_texture_2d_size)
        {
            return false;
        }
        if a.target == T::Texture3d {
            if fcts && (a.height == 0 || a.depth == 0) {
                return false;
            }
            let m = limits.max_texture_3d_size;
            if a.width > m || a.height > m || a.depth > m {
                return false;
            }
        }
        if matches!(a.target, T::Array2d | T::CubeArray | T::Array1d) && fcts && a.array_size == 0 {
            return false;
        }
        if matches!(a.target, T::Cube | T::CubeArray) {
            if a.width != a.height {
                return false;
            }
            if a.width > limits.max_texture_cube_size {
                return false;
            }
        }
        // `vrend_resource_alloc_buffer` is the other half of the C's create path and refuses
        // too, so a reference for "does the C end up with a resource" has to include it. It
        // runs after the whole of check_resource_valid, and only on a buffer target -- every
        // other one allocates a texture.
        if a.target == T::Buffer && !buffer_bind && !a.bind.has(Bind::SAMPLER_VIEW) {
            return false;
        }
        true
    }

    /// The whole input space, ours against the C's, because the guest cannot see a refusal.
    ///
    /// A create is answered on ctx0, before any context owns the resource, and mesa does not read
    /// the control queue's error -- so a resource we refuse is one the guest goes on to attach
    /// backing to and transfer into, and the context dies on that later command naming a resource
    /// that "does not exist". The symptom is a desktop that hangs several commands after the
    /// mistake, attributed to the wrong thing. Nothing downstream can recover from it and no
    /// amount of care at the refusal site prevents it, because the refusal is invisible.
    ///
    /// So the acceptance set is the invariant, not the individual rules: whatever this build
    /// accepts must be what the C accepts, over every shape the wire can express. Sweeping it is
    /// what makes a rule refusing something real impossible to add -- it fails here on the first
    /// run, rather than in a guest, months later, as a hang.
    ///
    /// This caught a refusal of a texture buffer -- `PIPE_BUFFER` carrying `SAMPLER_VIEW`, which
    /// `alloc_buffer` has always known how to make -- that killed GNOME's overview on the first
    /// glyph it uploaded.
    #[test]
    fn nothing_is_refused_that_the_c_creates() {
        let (f, t, l) = (features(), table(), limits());
        let targets = [
            TextureTarget::Buffer,
            TextureTarget::Texture1d,
            TextureTarget::Texture2d,
            TextureTarget::Texture3d,
            TextureTarget::Cube,
            TextureTarget::Rect,
            TextureTarget::Array1d,
            TextureTarget::Array2d,
            TextureTarget::CubeArray,
        ];
        // Every single bind flag, plus the unions a real guest sends: the C tells buffers from
        // textures by *equality*, so a union of two buffer binds is a texture to it and the
        // pairs are where a reimplementation drifts.
        let singles: Vec<Bind> = (0..23).map(|i| Bind(1 << i)).collect();
        let mut binds = vec![Bind(0)];
        binds.extend(singles.iter().copied());
        for &a in &[Bind::SAMPLER_VIEW, Bind::VERTEX_BUFFER, Bind::CONSTANT_BUFFER, Bind::CUSTOM] {
            for &b in &singles {
                binds.push(Bind(a.0 | b.0));
            }
        }
        let shapes: &[(u32, u32, u32, u32, u32, u32)] = &[
            // width, height, depth, array_size, last_level, nr_samples
            (64, 64, 1, 1, 0, 0),
            (64, 64, 1, 6, 0, 0),
            (64, 32, 1, 1, 0, 0),
            (64, 1, 1, 1, 0, 0),
            (64, 64, 4, 1, 0, 0),
            (0, 64, 1, 1, 0, 0),
            (64, 0, 1, 1, 0, 0),
            (64, 64, 1, 0, 0, 0),
            (64, 64, 1, 1, 3, 0),
            (64, 64, 1, 1, 0, 4),
            (99999, 64, 1, 1, 0, 0),
            (64, 64, 999, 1, 0, 0),
        ];
        // MAP_PERSISTENT is in here for the deviation it exposes, below.
        let flags = [
            ResourceFlags(0),
            ResourceFlags::Y_0_TOP,
            ResourceFlags::MAP_PERSISTENT,
            ResourceFlags(1 << 30),
        ];
        // Both table formats: one that multisamples and one that does not, because the entry
        // gates a whole branch of the rules.
        let formats = [Format::from_wire(67).unwrap(), Format::from_wire(2).unwrap()];
        let mut checked = 0usize;
        for &target in &targets {
            for &bind in &binds {
                for &(width, height, depth, array_size, last_level, nr_samples) in shapes {
                    for &flag in &flags {
                        for &format in &formats {
                            let a = Args {
                                target,
                                format,
                                bind,
                                width,
                                height,
                                depth,
                                array_size,
                                last_level,
                                nr_samples,
                                flags: flag,
                            };
                            let ours = plan(&f, &t, &l, &a).is_ok();
                            let mut theirs = c_check(&f, &t, &l, &a);
                            // The one deviation, stated here because this is what would otherwise
                            // quietly re-litigate it: asked for a persistent or coherent mapping
                            // without `ARB_buffer_storage`, the C logs, leaves the buffer with no
                            // data store, and reports success. A buffer the guest cannot map is a
                            // refusal here. See `plan_storage`.
                            let wants_storage = a.flags.has(ResourceFlags::MAP_PERSISTENT)
                                || a.flags.has(ResourceFlags::MAP_COHERENT);
                            if theirs
                                && a.target == TextureTarget::Buffer
                                && wants_storage
                                && !f.has(Feature::arb_buffer_storage)
                                && !matches!(
                                    a.bind.kind(),
                                    BindKind::HostShadow | BindKind::GuestPages
                                )
                            {
                                theirs = false;
                            }
                            // An OPEN deviation, not a settled one. A multisample 2D *array* needs
                            // `glTexStorage3DMultisample`, which GLES has only at 3.2 or behind
                            // OES_texture_storage_multisample_2d_array; the C reaches for
                            // `glTexImage3DMultisample` instead, which GLES does not have at all, so
                            // there is no fallback to port. We refuse. The problem is that the capset
                            // has no per-target multisample bit, so the guest is told the format
                            // multisamples and then refused -- invisibly, the failure this whole test
                            // exists for. It is reachable on this host, which is GLES 3.1. Resolving
                            // it means advertising no multisample at all without the array form,
                            // which costs 2D MSAA, and that is a call to make deliberately.
                            if theirs
                                && a.nr_samples > 1
                                && gl_target(a.target, a.nr_samples)
                                    == GL_TEXTURE_2D_MULTISAMPLE_ARRAY
                                && !f.has(Feature::storage_multisample_2d_array)
                            {
                                theirs = false;
                            }
                            assert_eq!(
                                ours,
                                theirs,
                                "we {} what the C {}: {a:?}",
                                if ours { "accept" } else { "refuse" },
                                if theirs { "creates" } else { "refuses" }
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 20_000, "the sweep covered only {checked} shapes");
    }

    #[test]
    fn the_checks_refuse_what_the_c_refuses() {
        let (f, t, l) = (features(), table(), limits());
        assert!(plan(&f, &t, &l, &texture()).is_ok());
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
            assert_eq!(plan(&f, &t, &l, &a).map(|_| ()), Err(*want), "{a:?}");
        }
        // One level more than the extent has is accepted, as in the C.
        let mut a = texture();
        a.last_level = 7;
        assert!(plan(&f, &t, &l, &a).is_ok());
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
        assert!(plan(&f, &t, &l, &buffer).is_ok());
        let mut tall = buffer;
        tall.height = 2;
        assert_eq!(plan(&f, &t, &l, &tall).map(|_| ()), Err(Refusal::BufferNotFlat));
        let mut query = buffer;
        query.bind = Bind::QUERY_BUFFER;
        assert_eq!(plan(&f, &t, &l, &query).map(|_| ()), Err(Refusal::QueryBuffersUnsupported));
        let mut mip = buffer;
        mip.last_level = 1;
        assert_eq!(plan(&f, &t, &l, &mip).map(|_| ()), Err(Refusal::BufferWithMipmaps));
    }

    #[test]
    fn gles_has_no_1d_and_no_rect() {
        assert_eq!(gl_target(TextureTarget::Texture1d, 0), GL_TEXTURE_2D);
        assert_eq!(gl_target(TextureTarget::Rect, 0), GL_TEXTURE_2D);
        assert_eq!(gl_target(TextureTarget::Array1d, 0), GL_TEXTURE_2D_ARRAY);
        assert_eq!(gl_target(TextureTarget::Texture2d, 4), GL_TEXTURE_2D_MULTISAMPLE);
    }
}
