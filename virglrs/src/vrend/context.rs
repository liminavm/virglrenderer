// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! A classic context: its sub-contexts, what each has bound, and every command a guest can send
//! into one.
//!
//! The C keeps this as `vrend_context` and `vrend_sub_context`. Each sub-context owns a GL
//! context in ctx0's share group, an object table, and a copy of every piece of state the guest
//! last set; a command runs on the current sub-context and touches nothing else. What is
//! emitted to GL immediately and what is only recorded follows the C exactly, because pixels
//! come out of the order GL sees things in -- the notes behind each handler name the C function
//! it mirrors.
//!
//! Errors are one sticky fault. The C reports most of them without stopping the batch and
//! refuses the *next* submission; here the batch stops at the fault and this submission is the
//! one refused. Nothing a guest sends reaches an `assert`: a handle that is not in the table, a
//! resource the context does not have, a shape the host cannot serve -- each is a [`Fault`].

use super::blitter::Blitter;
use super::decode::Batch;
use super::dirty::Dirty;
use super::egl::{self, EglError, Version, Winsys};
use super::features::{Feature, Features};
use super::formats::{Desc, Table};
use super::gl::gles::*;
use super::gl::{
    BindingPoint, BufferName, FramebufferName, GLbitfield, GLenum, GLint, GLsizei, GLuint, Gl,
    ImageUnit, ProgramName, QueryName, SamplerName, ShaderName, TextureName, TextureUnit,
    TransformFeedbackName, UniformLocation, VertexArrayName,
};
use super::pipe::slots::{
    MAX_COLOR_BUFS, MAX_CONSTANT_BUFFERS, MAX_SAMPLERS, MAX_SHADER_BUFFERS, MAX_SHADER_IMAGES,
    MAX_VIEWPORTS,
};
use super::pipe::*;
use super::proto::{self, *};
use super::resource::{self, Limits, Resource, Storage};
use super::transfer::{self, Info};
use super::{debug, shader, tgsi};
use crate::guest_mem::{HostSpan, Iov};
use crate::ids::{CtxId, ResourceHandle};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;

#[path = "context/blit.rs"]
mod blit;
#[path = "context/draw.rs"]
mod draw;
#[path = "context/select.rs"]
mod select;

pub use draw::{HwBlend, LinkedProgram, ProgramSerial, Sysval, Xfb};
pub use select::{Bound, Program, Variant, VariantId};

const PIPE_CLEAR_DEPTH: u32 = 1 << 0;
const PIPE_CLEAR_STENCIL: u32 = 1 << 1;
const PIPE_CLEAR_COLOR0: u32 = 1 << 2;
const PIPE_CLEAR_COLOR: u32 = 0xff << 2;

/// What the guest side of the renderer answers about a resource's pages.
pub trait Guest {
    /// Whether the context may reach the resource: the C's per-context `res_hash`.
    fn attached(&self, ctx: CtxId, handle: ResourceHandle) -> bool;
    /// The resource's attached pages, when the context may reach it and it has any.
    fn pages(&self, ctx: CtxId, handle: ResourceHandle) -> Option<Iov<'_>>;
}

/// Which GL context the thread has current, by name, so a switch is one compare.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Current {
    Ctx0,
    Sub(CtxId, SubCtxId),
    /// The blitter's own GL context, for the length of one blit. It is a state of this enum and
    /// not a flag beside it because it is the same fact: a switch back that consulted a stale
    /// `Sub` would decide it had nothing to do, and every GL call after the blit -- the rest of
    /// the batch, which does not ask -- would land in the blitter's context.
    Blitter,
}

/// Commands this build could not serve, counted by shape. Printed once each as they are first
/// met, so a run's log says what it asked for that is not here.
#[derive(Default)]
pub struct Todo(BTreeMap<&'static str, u64>);

impl Todo {
    pub fn note(&mut self, what: &'static str) {
        let n = self.0.entry(what).or_insert(0);
        if *n == 0 {
            eprintln!("[virglrs] vrend: not served: {what}");
        }
        *n += 1;
    }

    pub fn by_frequency(&self) -> Vec<(&'static str, u64)> {
        let mut v: Vec<_> = self.0.iter().map(|(k, n)| (*k, *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }
}

/// Everything a command needs that the context does not own: the driver, the host's tables, the
/// resources, the guest's pages, and the winsys for sub-context switches.
pub struct Host<'a> {
    pub gl: &'a Gl,
    pub winsys: &'a Winsys,
    /// The version and share context a new sub-context's GL context is made with.
    pub version: Version,
    pub share: &'a egl::Context,
    pub features: &'a Features,
    pub formats: &'a Table,
    pub limits: &'a Limits,
    pub shader_cfg: &'a shader::Cfg,
    pub resources: &'a mut BTreeMap<ResourceHandle, Resource>,
    pub guest: &'a dyn Guest,
    pub ctx: CtxId,
    pub current: &'a mut Current,
    pub todo: &'a mut Todo,
    /// The shader blitter, built on the first blit that needs it.
    pub blitter: &'a mut Option<Blitter>,
}

impl Host<'_> {
    fn make_current(&mut self, sub: SubCtxId, gl_ctx: &egl::Context) {
        let want = Current::Sub(self.ctx, sub);
        if *self.current != want {
            self.winsys
                .make_current(gl_ctx)
                .expect("a sub-context's GL context can be made current");
            *self.current = want;
        }
    }

    /// A resource the context may reach, with vrend's side of it.
    fn resource(&self, cmd: Cmd, handle: ResourceHandle) -> Result<&Resource, Fault> {
        if !self.guest.attached(self.ctx, handle) {
            return Err(Fault::IllegalResource { cmd, handle });
        }
        self.resources.get(&handle).ok_or(Fault::IllegalResource { cmd, handle })
    }

    fn resource_mut(&mut self, cmd: Cmd, handle: ResourceHandle) -> Result<&mut Resource, Fault> {
        if !self.guest.attached(self.ctx, handle) {
            return Err(Fault::IllegalResource { cmd, handle });
        }
        self.resources.get_mut(&handle).ok_or(Fault::IllegalResource { cmd, handle })
    }

    fn has(&self, f: Feature) -> bool {
        self.features.has(f)
    }
}

/// Why a context stopped serving. Sticky: the first one is kept and every later submission is
/// refused with it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Fault {
    Wire(Refused),
    /// No object of the type the command wants under that handle.
    IllegalHandle {
        cmd: Cmd,
        handle: ObjectHandle,
    },
    /// No resource the context may reach under that handle.
    IllegalResource {
        cmd: Cmd,
        handle: ResourceHandle,
    },
    IllegalFormat {
        cmd: Cmd,
        format: Format,
    },
    /// A field outside what the state can hold: a box past a resource, a slot past a table.
    OutOfRange {
        cmd: Cmd,
        what: &'static str,
    },
    /// A shader the host cannot take: a stage without its feature, a continuation out of
    /// sequence, text without its terminator.
    Shader {
        cmd: Cmd,
        what: &'static str,
    },
    /// Shader text the TGSI layer refused.
    Tgsi {
        cmd: Cmd,
        error: tgsi::Refusal,
    },
    /// A program the translator refused under the key the state made for it.
    Glsl {
        cmd: Cmd,
        stage: ShaderStage,
        error: shader::Failure,
    },
    /// A vertex format with no GL type.
    IllegalVertexFormat(Format),
    UnsupportedTexWrap(TexWrap),
    Transfer {
        cmd: Cmd,
        error: transfer::Error,
    },
    /// The driver raised an error while the command ran.
    Gl {
        cmd: Cmd,
        error: GLenum,
    },
    /// The command needs a feature this host lacks, which the capset told the guest.
    NoFeature {
        cmd: Cmd,
        feature: Feature,
    },
    /// A command this build does not serve yet.
    Unimplemented {
        cmd: Cmd,
        what: &'static str,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Wire(r) => write!(f, "{r}"),
            Fault::IllegalHandle { cmd, handle } => {
                write!(f, "{}: no such object {handle}", cmd.name())
            }
            Fault::IllegalResource { cmd, handle } => {
                write!(f, "{}: no such resource {handle}", cmd.name())
            }
            Fault::IllegalFormat { cmd, format } => {
                write!(f, "{}: format {} is not served", cmd.name(), format.name())
            }
            Fault::OutOfRange { cmd, what } => write!(f, "{}: {what} out of range", cmd.name()),
            Fault::Shader { cmd, what } => write!(f, "{}: {what}", cmd.name()),
            Fault::Tgsi { cmd, error } => write!(f, "{}: {error}", cmd.name()),
            Fault::Glsl { cmd, stage, error } => {
                write!(f, "{}: {} shader: {error}", cmd.name(), stage.name())
            }
            Fault::IllegalVertexFormat(fmt) => {
                write!(f, "vertex format {} has no GL type", fmt.name())
            }
            Fault::UnsupportedTexWrap(w) => write!(f, "texture wrap {} is unsupported", w.name()),
            Fault::Transfer { cmd, error } => write!(f, "{}: {error}", cmd.name()),
            Fault::Gl { cmd, error } => write!(f, "{}: GL error {error:#x}", cmd.name()),
            Fault::NoFeature { cmd, feature } => {
                write!(f, "{}: the host has no {}", cmd.name(), feature.name())
            }
            Fault::Unimplemented { cmd, what } => {
                write!(f, "{}: {what} is not served by this build", cmd.name())
            }
        }
    }
}

// ---- objects ----

/// A shader object: what the guest declared it, and its program once the text is all here.
pub struct Shader {
    pub stage: ShaderStage,
    pub kind: ShaderKind,
    pub text: ShaderText,
}

/// A shader's text arrives in one command or, past a command's size, in several; the object
/// exists from the first. Nothing reads a program until it is parsed, so the two are one state
/// each rather than a buffer and a flag.
#[allow(clippy::large_enum_variant)]
pub enum ShaderText {
    Arriving {
        /// The text so far, dword-padded as sent.
        text: Vec<u8>,
        /// How long the whole is, in bytes rounded up to a dword.
        total: usize,
    },
    /// Parsed, scanned and translated the moment the text completed, as the C does, and again
    /// under every key it is selected with.
    Whole(Program),
}

#[derive(Clone, Copy, Debug)]
pub struct Element {
    pub base: VertexElement,
    pub gl_type: GLenum,
    pub normalized: bool,
    pub nr_channels: GLint,
    pub pure_integer: bool,
}

pub struct VertexElements {
    pub elements: Vec<Element>,
    /// The elements whose format is stored blue first: the vertex shader reads them `.zyxw`,
    /// since GLES has no `GL_BGRA` attribute size.
    pub zyxw_bitmask: u32,
    /// The vertex array object the elements are laid out in, made at the first bind.
    pub vao: Option<VertexArrayName>,
}

/// A sampler view as the C derives it at creation.
pub struct View {
    pub resource: ResourceHandle,
    pub format: Format,
    pub target: GLenum,
    /// A rectangle view served by a 2D texture, GLES having no rectangle target: the shader
    /// scales the coordinates.
    pub emulated_rect: bool,
    /// A linear view of an sRGB texture: sampled through the sampler that skips decoding.
    pub skip_srgb_decode: bool,
    /// A texture view of the resource, when one was needed and could be made.
    pub view: Option<TextureName>,
    pub first_layer: u32,
    pub last_layer: u32,
    pub first_level: u32,
    pub last_level: u32,
    pub first_element: u32,
    pub last_element: u32,
    pub gl_swizzle: [GLint; 4],
}

pub struct Sampler {
    pub state: SamplerState,
    /// Two sampler objects: one skipping sRGB decode, one decoding.
    pub ids: Option<[SamplerName; 2]>,
}

pub struct Surf {
    pub resource: ResourceHandle,
    pub format: Format,
    pub level: u32,
    pub first_layer: u32,
    pub last_layer: u32,
    pub nr_samples: u32,
    pub view: Option<TextureName>,
}

impl Surf {
    /// The layer the surface attaches, or `None` for every layer (the C's -1).
    fn layer(&self) -> Option<GLint> {
        (self.first_layer == self.last_layer).then_some(self.first_layer as GLint)
    }
}

pub struct Query {
    pub kind: QueryType,
    pub gl_type: GLenum,
    pub id: QueryName,
    pub resource: ResourceHandle,
    /// The C's `fake_samples_passed`: an occlusion counter served as a boolean.
    pub fake_samples_passed: bool,
}

pub struct Streamout {
    pub id: TransformFeedbackName,
    pub targets: Vec<Option<ObjectHandle>>,
    pub xfb: Xfb,
}

pub enum Object {
    Blend(BlendState),
    Rasterizer(RasterizerState),
    Dsa(DepthStencilAlpha),
    Shader(Shader),
    VertexElements(VertexElements),
    SamplerView(View),
    SamplerState(Sampler),
    Surface(Surf),
    Query(Query),
    StreamoutTarget(StreamoutTarget),
}

impl Object {
    fn kind(&self) -> ObjectType {
        match self {
            Object::Blend(_) => ObjectType::Blend,
            Object::Rasterizer(_) => ObjectType::Rasterizer,
            Object::Dsa(_) => ObjectType::Dsa,
            Object::Shader(_) => ObjectType::Shader,
            Object::VertexElements(_) => ObjectType::VertexElements,
            Object::SamplerView(_) => ObjectType::SamplerView,
            Object::SamplerState(_) => ObjectType::SamplerState,
            Object::Surface(_) => ObjectType::Surface,
            Object::Query(_) => ObjectType::Query,
            Object::StreamoutTarget(_) => ObjectType::StreamoutTarget,
        }
    }
}

// ---- bound state ----

/// A surface as bound to the framebuffer: the object's fields at the bind, which is what the C
/// keeps a reference to. Resolved against the resource when bound; the handle is kept so a
/// later bind of the same object is a no-op.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoundSurface {
    pub handle: ObjectHandle,
    pub resource: ResourceHandle,
    pub format: Format,
    pub level: u32,
    pub nr_samples: u32,
    pub tex_height: u32,
    pub y_0_top: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct ViewportHw {
    x: GLint,
    y: GLint,
    width: GLsizei,
    height: GLsizei,
    near: f64,
    far: f64,
}

/// What `vrend_hw_emit_rs` last told GL, for the toggles it only emits on change.
#[derive(Clone, Copy, Default, Debug)]
struct HwRs {
    rasterizer_discard: bool,
    flatshade: bool,
    clip_halfz: bool,
    flatshade_first: bool,
    clip_plane_enable: u8,
    scissor: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ubo {
    pub resource: ResourceHandle,
    pub offset: u32,
    pub length: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ssbo {
    pub resource: ResourceHandle,
    pub offset: u32,
    pub length: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImageView {
    pub resource: ResourceHandle,
    pub format: Format,
    pub access: ImageAccess,
    pub layer_offset: u32,
    pub level_size: u32,
}

pub struct SubCtx {
    gl_ctx: egl::Context,
    fb: FramebufferName,
    blit_fbs: [FramebufferName; 2],
    vao: VertexArrayName,
    objects: BTreeMap<ObjectHandle, Object>,
    long_shader: [Option<ObjectHandle>; ShaderStage::COUNT],

    blend: Option<BlendState>,
    /// Blend state reaches GL only at draw; this is what it last told GL, and what a clear
    /// restores.
    hw_blend: HwBlend,
    blend_dirty: bool,
    dsa: Option<(ObjectHandle, DepthStencilAlpha)>,
    rs: Option<RasterizerState>,
    hw_rs: HwRs,
    depth_test_enabled: bool,
    stencil_test_enabled: bool,
    stencil_refs: [u8; 2],
    stencil_dirty: bool,

    viewports: [ViewportHw; MAX_VIEWPORTS],
    viewport_dirty: Dirty<MAX_VIEWPORTS>,
    viewport_is_negative: bool,
    scissors: [Scissor; MAX_VIEWPORTS],
    scissor_dirty: Dirty<MAX_VIEWPORTS>,

    zsurf: Option<BoundSurface>,
    cbufs: Vec<Option<BoundSurface>>,
    fb_height: u32,
    fbo_origin_upper_left: bool,
    framebuffer_srgb_enabled: bool,
    /// Colour buffers whose resource cannot be viewed (`Resource::supports_view`) and whose
    /// surface format swaps red and blue against it: the fragment shader swaps them on its final
    /// write, one bit per attachment. Read by the shader key.
    swizzle_output_rgb_to_bgr: u8,
    /// Colour buffers whose resource cannot be viewed and whose surface format is sRGB: the
    /// fragment shader encodes on its final write, one bit per attachment. Read by the shader
    /// key.
    needs_manual_srgb_encode: u8,

    blend_color: [f32; 4],
    ve: Option<ObjectHandle>,
    vbos: Vec<VertexBuffer>,
    /// How many vertex buffers the hardware holds, written by the bind that put them there --
    /// so a shorter set unbinds exactly the ones still bound, however many state-sets went by
    /// without a draw between them.
    hw_num_vbos: usize,
    vbo_dirty: bool,
    ib: Option<IndexBuffer>,
    consts: [Vec<u32>; ShaderStage::COUNT],
    const_dirty: [bool; ShaderStage::COUNT],
    ubos: [BTreeMap<u32, Ubo>; ShaderStage::COUNT],
    ubos_dirty: [Dirty<MAX_CONSTANT_BUFFERS>; ShaderStage::COUNT],
    views: [BTreeMap<u32, ObjectHandle>; ShaderStage::COUNT],
    /// The view slots re-bound at the next draw, one bit each.
    /// A sampler unit is what a shader names, and there are [`MAX_SAMPLERS`] of them -- fewer
    /// than the view slots the decoder admits. A view set above them is held and never sampled.
    views_dirty: [Dirty<MAX_SAMPLERS>; ShaderStage::COUNT],
    /// The level count of each view the last draw bound, in sampler order, for the GLES
    /// `textureQueryLevels` emulation.
    texture_levels: [Vec<GLint>; ShaderStage::COUNT],
    samplers: [BTreeMap<u32, ObjectHandle>; ShaderStage::COUNT],
    shaders: [Option<Bound>; ShaderStage::COUNT],
    /// A stage was bound or its key's inputs changed: the draw re-selects the variants.
    shader_dirty: bool,
    /// The last draw's primitive mode, which the fragment shader's key reads; the C's is zero
    /// until a draw, and zero is points.
    prim_mode: PrimType,
    /// Every program linked for this sub-context, and the one the draws run.
    programs: Vec<LinkedProgram>,
    prog: Option<ProgramSerial>,
    next_program_serial: Cell<u64>,
    next_variant_id: u64,
    /// The `VirglBlock` contents. Each program remembers the block it last uploaded and
    /// compares by value, so there is no second record of whether this changed.
    sysval: Sysval,
    ssbos: [BTreeMap<u32, Ssbo>; ShaderStage::COUNT],
    images: [BTreeMap<u32, ImageView>; ShaderStage::COUNT],
    abos: BTreeMap<u32, Ssbo>,
    streamouts: Vec<Streamout>,
    current_so: Option<usize>,
    render_condition: Option<(ObjectHandle, bool, RenderCondMode)>,
}

impl SubCtx {
    /// `vrend_renderer_create_sub_ctx`'s GL side, on a context just made current.
    fn new(gl: &Gl, gl_ctx: egl::Context) -> SubCtx {
        let vao = gl.gen_vertex_array();
        let fb = gl.gen_framebuffer();
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
        let blit_fbs = [gl.gen_framebuffer(), gl.gen_framebuffer()];
        let vp = ViewportHw { x: 0, y: 0, width: 0, height: 0, near: 0.0, far: 1.0 };
        SubCtx {
            gl_ctx,
            fb,
            blit_fbs,
            vao,
            objects: BTreeMap::new(),
            long_shader: [None; ShaderStage::COUNT],
            blend: None,
            hw_blend: HwBlend::default(),
            blend_dirty: false,
            dsa: None,
            rs: None,
            hw_rs: HwRs::default(),
            depth_test_enabled: false,
            stencil_test_enabled: false,
            stencil_refs: [0; 2],
            stencil_dirty: false,
            viewports: [vp; MAX_VIEWPORTS],
            viewport_dirty: Dirty::none(),
            viewport_is_negative: false,
            scissors: [Scissor { minx: 0, miny: 0, maxx: 0, maxy: 0 }; MAX_VIEWPORTS],
            scissor_dirty: Dirty::none(),
            zsurf: None,
            cbufs: Vec::new(),
            fb_height: 0,
            fbo_origin_upper_left: false,
            framebuffer_srgb_enabled: false,
            swizzle_output_rgb_to_bgr: 0,
            needs_manual_srgb_encode: 0,
            blend_color: [0.0; 4],
            ve: None,
            vbos: Vec::new(),
            hw_num_vbos: 0,
            vbo_dirty: false,
            ib: None,
            consts: Default::default(),
            const_dirty: [false; ShaderStage::COUNT],
            ubos: Default::default(),
            ubos_dirty: [Dirty::none(); ShaderStage::COUNT],
            views: Default::default(),
            views_dirty: [Dirty::none(); ShaderStage::COUNT],
            texture_levels: Default::default(),
            samplers: Default::default(),
            shaders: Default::default(),
            shader_dirty: false,
            prim_mode: PrimType::Points,
            programs: Vec::new(),
            prog: None,
            next_program_serial: Cell::new(0),
            next_variant_id: 0,
            sysval: Sysval::default(),
            ssbos: Default::default(),
            images: Default::default(),
            abos: BTreeMap::new(),
            streamouts: Vec::new(),
            current_so: None,
            render_condition: None,
        }
    }

    /// `vrend_destroy_sub_context`'s GL side: every object's GL side and the sub-context's own,
    /// on its context, which the caller made current.
    fn destroy(mut self, gl: &Gl) -> egl::Context {
        gl.delete_framebuffer(self.fb);
        for fb in self.blit_fbs {
            gl.delete_framebuffer(fb);
        }
        gl.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, None);
        gl.delete_vertex_array(self.vao);
        gl.bind_vertex_array(None);
        if self.current_so.is_some() {
            gl.bind_transform_feedback(None);
        }
        for so in self.streamouts.drain(..) {
            gl.delete_transform_feedback(so.id);
        }
        for p in self.programs.drain(..) {
            if let Some(b) = p.sysval_buffer {
                gl.delete_buffer(b);
            }
            gl.delete_program(p.id);
        }
        let objects = std::mem::take(&mut self.objects);
        for (_, obj) in objects {
            match obj {
                Object::Shader(s) => draw::release_shader(&mut self, gl, s),
                other => release(gl, other),
            }
        }
        for b in std::mem::take(&mut self.shaders) {
            if let Some(Bound::Owned(s)) = b {
                draw::release_shader(&mut self, gl, s);
            }
        }
        self.gl_ctx
    }

    fn dsa_state(&self) -> DepthStencilAlpha {
        self.dsa.map(|(_, s)| s).unwrap_or(ZERO_DSA)
    }

    fn rs_state(&self) -> RasterizerState {
        self.rs.unwrap_or(ZERO_RS)
    }

    fn object(&self, cmd: Cmd, handle: ObjectHandle, kind: ObjectType) -> Result<&Object, Fault> {
        match self.objects.get(&handle) {
            Some(o) if o.kind() == kind => Ok(o),
            _ => Err(Fault::IllegalHandle { cmd, handle }),
        }
    }

    fn surface(&self, cmd: Cmd, handle: ObjectHandle) -> Result<&Surf, Fault> {
        match self.object(cmd, handle, ObjectType::Surface)? {
            Object::Surface(s) => Ok(s),
            _ => unreachable!("looked up as a surface"),
        }
    }
}

/// Empty every view slot naming `handle` and mark each for rebinding. Whether any did.
fn evict_view(
    views: &mut [BTreeMap<u32, ObjectHandle>; ShaderStage::COUNT],
    views_dirty: &mut [Dirty<MAX_SAMPLERS>; ShaderStage::COUNT],
    handle: ObjectHandle,
) -> bool {
    let mut held = false;
    for (stage, slots) in views.iter_mut().enumerate() {
        let gone: Vec<u32> = slots.iter().filter(|(_, h)| **h == handle).map(|(s, _)| *s).collect();
        for slot in gone {
            slots.remove(&slot);
            // A view above the sampler units is held but never sampled, so nothing rebinds it.
            if (slot as usize) < MAX_SAMPLERS {
                views_dirty[stage].mark(slot);
            }
            held = true;
        }
    }
    held
}

/// Empty every framebuffer slot naming `handle`: whether the depth slot held it, and the colour
/// slots that did. The attachments they name are the ones to detach.
fn evict_surface(
    zsurf: &mut Option<BoundSurface>,
    cbufs: &mut [Option<BoundSurface>],
    handle: ObjectHandle,
) -> (bool, Vec<usize>) {
    let held_z = zsurf.is_some_and(|s| s.handle == handle);
    if held_z {
        *zsurf = None;
    }
    let mut colours = Vec::new();
    for (i, slot) in cbufs.iter_mut().enumerate() {
        if slot.is_some_and(|s| s.handle == handle) {
            *slot = None;
            colours.push(i);
        }
    }
    (held_z, colours)
}

/// Release an object's GL side. The sub-context that made it is current.
fn release(gl: &Gl, obj: Object) {
    match obj {
        Object::VertexElements(v) => {
            if let Some(vao) = v.vao {
                gl.delete_vertex_array(vao);
            }
        }
        Object::SamplerView(v) => {
            if let Some(t) = v.view {
                gl.delete_texture(t);
            }
        }
        Object::SamplerState(s) => {
            if let Some(ids) = s.ids {
                for id in ids {
                    gl.delete_sampler(id);
                }
            }
        }
        Object::Surface(s) => {
            if let Some(t) = s.view {
                gl.delete_texture(t);
            }
        }
        Object::Query(q) => gl.delete_query(q.id),
        Object::Shader(_) => unreachable!("a shader leaves through draw::release_shader"),
        Object::Blend(_) | Object::Rasterizer(_) | Object::Dsa(_) | Object::StreamoutTarget(_) => {}
    }
}

const ZERO_FACE: StencilFace = StencilFace {
    enabled: false,
    func: CompareFunc::Never,
    fail_op: StencilOp::Keep,
    zpass_op: StencilOp::Keep,
    zfail_op: StencilOp::Keep,
    valuemask: 0,
    writemask: 0,
};

/// The C's zeroed `pipe_depth_stencil_alpha_state`.
const ZERO_DSA: DepthStencilAlpha = DepthStencilAlpha {
    depth: DepthState { enabled: false, writemask: false, func: CompareFunc::Never },
    alpha: AlphaState { enabled: false, func: CompareFunc::Never, ref_value: 0.0 },
    stencil: [ZERO_FACE; 2],
};

/// The C's zeroed `pipe_blend_state`.
const ZERO_BLEND: BlendState = BlendState {
    independent_blend_enable: false,
    logicop_enable: false,
    dither: false,
    alpha_to_coverage: false,
    alpha_to_one: false,
    logicop_func: LogicOp::Clear,
    rt: [RtBlend { equation: None, colormask: 0 }; 8],
};

/// The C's zeroed `pipe_rasterizer_state`.
const ZERO_RS: RasterizerState = RasterizerState {
    flatshade: false,
    depth_clip: false,
    clip_halfz: false,
    rasterizer_discard: false,
    flatshade_first: false,
    light_twoside: false,
    sprite_coord_mode: false,
    point_quad_rasterization: false,
    cull_face: CullFace::None,
    fill_front: FillMode::Fill,
    fill_back: FillMode::Fill,
    scissor: false,
    front_ccw: false,
    clamp_vertex_color: false,
    clamp_fragment_color: false,
    offset_line: false,
    offset_point: false,
    offset_tri: false,
    poly_smooth: false,
    poly_stipple_enable: false,
    point_smooth: false,
    point_size_per_vertex: false,
    multisample: false,
    line_smooth: false,
    line_stipple_enable: false,
    line_last_pixel: false,
    half_pixel_center: false,
    bottom_edge_rule: false,
    force_persample_interp: false,
    point_size: 0.0,
    sprite_coord_enable: 0,
    line_stipple_pattern: 0,
    line_stipple_factor: 0,
    clip_plane_enable: 0,
    line_width: 0.0,
    offset_units: 0.0,
    offset_scale: 0.0,
    offset_clamp: 0.0,
};

// ---- the context ----

pub struct Context {
    subs: BTreeMap<SubCtxId, SubCtx>,
    current: SubCtxId,
    fault: Option<Fault>,
}

impl Context {
    /// `vrend_create_context`: a context with sub-context 0, current on this thread.
    pub fn new(host: &mut Host<'_>) -> Result<Context, EglError> {
        let mut ctx = Context { subs: BTreeMap::new(), current: SubCtxId(0), fault: None };
        ctx.create_sub(host, SubCtxId(0))?;
        Ok(ctx)
    }

    pub fn fault(&self) -> Option<&Fault> {
        self.fault.as_ref()
    }

    /// Whether this context's GL contexts are `Current::Sub(self, ...)`.
    pub fn current_sub(&self) -> SubCtxId {
        self.current
    }

    fn sub(&self) -> &SubCtx {
        self.subs.get(&self.current).expect("the current sub-context exists")
    }

    fn sub_mut(&mut self) -> &mut SubCtx {
        self.subs.get_mut(&self.current).expect("the current sub-context exists")
    }

    /// Make the current sub-context's GL context current.
    pub fn make_current(&self, host: &mut Host<'_>) {
        host.make_current(self.current, &self.sub().gl_ctx);
    }

    /// Every sub-context's GL context, for the renderer to wait on. Each has its own command
    /// queue, so work one of them rendered is not covered by a finish on any other.
    pub fn gl_contexts(&self) -> impl Iterator<Item = (SubCtxId, &egl::Context)> {
        self.subs.iter().map(|(id, sub)| (*id, &sub.gl_ctx))
    }

    /// `vrend_destroy_context`: unbind what the C unbinds, then every sub-context.
    pub fn destroy(mut self, host: &mut Host<'_>) {
        let ids: Vec<SubCtxId> = self.subs.keys().rev().copied().collect();
        for id in ids {
            let sub = self.subs.remove(&id).expect("listed");
            host.make_current(id, &sub.gl_ctx);
            let gl_ctx = sub.destroy(host.gl);
            drop(gl_ctx);
        }
        *host.current = Current::Ctx0;
    }

    fn create_sub(&mut self, host: &mut Host<'_>, id: SubCtxId) -> Result<(), EglError> {
        if self.subs.contains_key(&id) {
            return Ok(());
        }
        let gl_ctx = host.winsys.create_context(host.version, Some(host.share))?;
        host.make_current(id, &gl_ctx);
        let sub = SubCtx::new(host.gl, gl_ctx);
        self.subs.insert(id, sub);
        Ok(())
    }

    /// Run one batch. A fault stops it and sticks.
    pub fn submit(&mut self, host: &mut Host<'_>, words: &[u32]) -> Result<(), Fault> {
        if let Some(f) = &self.fault {
            return Err(f.clone());
        }
        self.make_current(host);
        let batch = Batch::new(words);
        for item in batch {
            let cmd = match item {
                Ok(c) => c,
                Err(r) => return self.poison(Fault::Wire(r)),
            };
            let kind = cmd.kind();
            if let Err(f) = self.run(host, cmd) {
                return self.poison(f);
            }
            // `vrend_check_no_error`: any GL error a command left is the context's error.
            let err = host.gl.drain_errors();
            if err != GL_NO_ERROR {
                return self.poison(Fault::Gl { cmd: kind, error: err });
            }
        }
        Ok(())
    }

    fn poison(&mut self, f: Fault) -> Result<(), Fault> {
        eprintln!("[virglrs] vrend: context poisoned: {f}");
        self.fault = Some(f.clone());
        Err(f)
    }

    fn run(&mut self, host: &mut Host<'_>, cmd: Command<'_>) -> Result<(), Fault> {
        let kind = cmd.kind();
        match cmd {
            Command::Nop => Ok(()),
            Command::CreateObject { handle, object } => self.create_object(host, handle, object),
            Command::BindObject { kind: ty, handle } => self.bind_object(host, ty, handle),
            Command::DestroyObject { handle, .. } => {
                self.destroy_object(host, handle);
                Ok(())
            }
            Command::SetViewportState { start_slot, viewports } => {
                self.set_viewports(start_slot, &viewports);
                Ok(())
            }
            Command::SetFramebufferState { zsurf, cbufs } => {
                self.set_framebuffer_state(host, zsurf, &cbufs)
            }
            Command::SetVertexBuffers(vbos) => self.set_vertex_buffers(host, vbos),
            Command::Clear { buffers, color, depth, stencil } => {
                self.clear(host, buffers, color, depth, stencil);
                Ok(())
            }
            Command::DrawVbo(draw) => self.draw_vbo(host, draw),
            Command::ResourceInlineWrite { transfer, data } => {
                self.inline_write(host, transfer, data)
            }
            Command::SetSamplerViews { stage, start_slot, views } => {
                self.set_sampler_views(host, stage, start_slot, &views)
            }
            Command::SetIndexBuffer(ib) => self.set_index_buffer(host, ib),
            Command::SetConstantBuffer { stage, data, .. } => {
                let sub = self.sub_mut();
                sub.consts[stage.index()] = data.to_vec();
                sub.const_dirty[stage.index()] = true;
                Ok(())
            }
            Command::SetStencilRef { front, back } => {
                let sub = self.sub_mut();
                if sub.stencil_refs != [front, back] {
                    sub.stencil_refs = [front, back];
                    sub.stencil_dirty = true;
                }
                Ok(())
            }
            Command::SetBlendColor(c) => {
                self.sub_mut().blend_color = c;
                host.gl.blend_color(c);
                Ok(())
            }
            Command::SetScissorState { start_slot, scissors } => {
                let sub = self.sub_mut();
                for (i, s) in scissors.iter().enumerate() {
                    let idx = start_slot as usize + i;
                    sub.scissors[idx] = *s;
                    sub.scissor_dirty.mark(idx as u32);
                }
                Ok(())
            }
            Command::Blit(b) => self.blit(host, &b),
            Command::ResourceCopyRegion {
                dst,
                dst_level,
                dst_x,
                dst_y,
                dst_z,
                src,
                src_level,
                src_region,
            } => self.copy_region(
                host,
                dst,
                dst_level,
                [dst_x, dst_y, dst_z],
                src,
                src_level,
                src_region,
            ),
            Command::BindSamplerStates { stage, start_slot, states } => {
                self.bind_sampler_states(stage, start_slot, &states);
                Ok(())
            }
            Command::BeginQuery(h) => self.begin_query(host, h),
            Command::EndQuery(h) => self.end_query(host, h),
            Command::GetQueryResult { query, wait } => self.get_query_result(host, query, wait),
            Command::SetPolygonStipple(rows) => {
                self.sub_mut().sysval.stipple = rows;
                Ok(())
            }
            Command::SetClipState(planes) => {
                self.sub_mut().sysval.clip_planes = planes;
                Ok(())
            }
            Command::SetSampleMask(mask) => {
                if host.has(Feature::sample_mask) {
                    host.gl.sample_mask_i(0, mask);
                }
                Ok(())
            }
            Command::SetStreamoutTargets { targets, .. } => {
                self.set_streamout_targets(host, &targets)
            }
            Command::SetRenderCondition { query, condition, mode } => {
                self.set_render_condition(query, condition, mode);
                Ok(())
            }
            Command::SetUniformBuffer { stage, index, offset, length, resource } => {
                self.set_uniform_buffer(host, stage, index, offset, length, resource)
            }
            Command::SetSubCtx(id) => {
                self.set_sub_ctx(host, id);
                Ok(())
            }
            Command::CreateSubCtx(id) => self.create_sub(host, id).map_err(|e| {
                eprintln!("[virglrs] vrend: sub-context {}: no GL context: {e}", id.0);
                Fault::Unimplemented { cmd: kind, what: "a GL context for the sub-context" }
            }),
            Command::DestroySubCtx(id) => {
                self.destroy_sub_ctx(host, id);
                Ok(())
            }
            Command::BindShader { handle, stage } => {
                self.bind_shader(host, handle, stage);
                Ok(())
            }
            Command::SetTessState(_) => Ok(()),
            Command::SetMinSamples(n) => {
                self.set_min_samples(host, n);
                Ok(())
            }
            Command::SetShaderBuffers { stage, start_slot, buffers } => {
                self.set_shader_buffers(host, stage, start_slot, &buffers)
            }
            Command::SetShaderImages { stage, start_slot, images } => {
                self.set_shader_images(host, stage, start_slot, &images)
            }
            Command::MemoryBarrier(flags) => {
                self.memory_barrier(host, flags);
                Ok(())
            }
            Command::LaunchGrid { .. } => {
                host.todo.note("LAUNCH_GRID");
                Err(Fault::Unimplemented { cmd: kind, what: "compute dispatch" })
            }
            Command::SetFramebufferStateNoAttach { width, height, layers, samples } => {
                if host.has(Feature::fb_no_attach) {
                    let gl = host.gl;
                    gl.framebuffer_parameter_i(GL_FRAMEBUFFER_DEFAULT_WIDTH, width as GLint);
                    gl.framebuffer_parameter_i(GL_FRAMEBUFFER_DEFAULT_HEIGHT, height as GLint);
                    if host.features.gles_version > 31 {
                        gl.framebuffer_parameter_i(GL_FRAMEBUFFER_DEFAULT_LAYERS, layers as GLint);
                    }
                    gl.framebuffer_parameter_i(GL_FRAMEBUFFER_DEFAULT_SAMPLES, samples as GLint);
                }
                Ok(())
            }
            Command::TextureBarrier(_) => {
                // Neither `glTextureBarrier` nor `glBlendBarrierKHR` is reachable on this host
                // without its feature; the C emits nothing then either.
                Ok(())
            }
            Command::SetAtomicBuffers { start_slot, buffers } => {
                self.set_atomic_buffers(host, start_slot, &buffers)
            }
            Command::SetDebugFlags(_) => Ok(()),
            Command::GetQueryResultQbo { .. } => {
                Err(Fault::Unimplemented { cmd: kind, what: "query buffer objects on GLES" })
            }
            Command::Transfer3d { transfer, offset, direction } => {
                self.transfer3d(host, transfer, offset, direction)
            }
            Command::EndTransfers(_) => Ok(()),
            Command::CopyTransfer3d {
                direction,
                transfer,
                staging,
                staging_offset,
                synchronized,
            } => self.copy_transfer3d(
                host,
                direction,
                transfer,
                staging,
                staging_offset,
                synchronized,
            ),
            Command::SetTweaks { .. } => Ok(()),
            Command::ClearTexture { resource, level, region, data } => {
                self.clear_texture(host, resource, level, region, data)
            }
            Command::PipeResourceCreate { .. }
            | Command::PipeResourceSetType { .. }
            | Command::GetMemoryInfo(_)
            | Command::GetPipeResourceLayout { .. } => {
                host.todo.note(kind.name());
                Err(Fault::Unimplemented { cmd: kind, what: "blob resources" })
            }
            Command::SendStringMarker { .. } => Ok(()),
            Command::LinkShader(handles) => self.link_shader(host, handles),
            Command::CreateVideoCodec(_)
            | Command::DestroyVideoCodec(_)
            | Command::CreateVideoBuffer { .. }
            | Command::DestroyVideoBuffer(_)
            | Command::BeginFrame { .. }
            | Command::DecodeMacroblock(_)
            | Command::DecodeBitstream { .. }
            | Command::EncodeBitstream { .. }
            | Command::EndFrame { .. } => {
                host.todo.note(kind.name());
                Err(Fault::Unimplemented { cmd: kind, what: "video" })
            }
            Command::ClearSurface {
                render_condition_enable: _,
                buffers,
                surface,
                color,
                dst_x,
                dst_y,
                width,
                height,
            } => self.clear_surface(
                host,
                surface,
                buffers as u32,
                color,
                [dst_x, dst_y, width, height],
            ),
        }
    }
}

// ---- sub-contexts ----

impl Context {
    /// `vrend_renderer_set_sub_ctx`: an unknown id is ignored.
    fn set_sub_ctx(&mut self, host: &mut Host<'_>, id: SubCtxId) {
        if id == self.current {
            return;
        }
        if let Some(sub) = self.subs.get(&id) {
            host.make_current(id, &sub.gl_ctx);
            self.current = id;
        }
    }

    /// `vrend_renderer_destroy_sub_ctx`: sub-context 0 is never destroyed.
    fn destroy_sub_ctx(&mut self, host: &mut Host<'_>, id: SubCtxId) {
        if id.0 == 0 {
            return;
        }
        let Some(sub) = self.subs.remove(&id) else {
            return;
        };
        host.make_current(id, &sub.gl_ctx);
        drop(sub.destroy(host.gl));
        if self.current == id {
            self.current = SubCtxId(0);
        }
        self.make_current(host);
    }
}

// ---- objects ----

impl Context {
    fn create_object(
        &mut self,
        host: &mut Host<'_>,
        handle: ObjectHandle,
        object: proto::Object<'_>,
    ) -> Result<(), Fault> {
        let cmd = Cmd::CreateObject;
        let obj = match object {
            proto::Object::Blend(s) => Object::Blend(s),
            proto::Object::Rasterizer(s) => Object::Rasterizer(s),
            proto::Object::Dsa(s) => Object::Dsa(s),
            proto::Object::Shader(s) => return self.create_shader(host, handle, s),
            proto::Object::VertexElements(elements) => {
                Object::VertexElements(vertex_elements(&elements)?)
            }
            proto::Object::SamplerView(v) => {
                Object::SamplerView(self.create_sampler_view(host, v)?)
            }
            proto::Object::SamplerState(s) => Object::SamplerState(create_sampler_state(host, s)?),
            proto::Object::Surface(s) => Object::Surface(self.create_surface(host, s)?),
            proto::Object::Query(q) => Object::Query(create_query(host, q)?),
            proto::Object::StreamoutTarget(t) => {
                host.resource(cmd, t.resource)?;
                Object::StreamoutTarget(t)
            }
        };
        self.insert_object(host, handle, obj);
        Ok(())
    }

    /// Insert, replacing -- and releasing -- whatever the handle named before, as the C's hash
    /// table does.
    fn insert_object(&mut self, host: &mut Host<'_>, handle: ObjectHandle, obj: Object) {
        if let Some(old) = self.sub_mut().objects.insert(handle, obj) {
            self.on_object_gone(host, handle, old);
        }
    }

    /// `vrend_renderer_object_destroy`: the type byte is ignored, a missing handle is nothing.
    fn destroy_object(&mut self, host: &mut Host<'_>, handle: ObjectHandle) {
        if let Some(old) = self.sub_mut().objects.remove(&handle) {
            self.on_object_gone(host, handle, old);
        }
    }

    /// The per-type destroy callbacks: unbind what was bound, then release the GL side.
    fn on_object_gone(&mut self, host: &mut Host<'_>, handle: ObjectHandle, old: Object) {
        let gl = host.gl;
        match &old {
            Object::Dsa(_) => {
                if self.sub().dsa.is_some_and(|(h, _)| h == handle) {
                    self.bind_dsa(host, None);
                }
            }
            Object::VertexElements(_) => {
                let sub = self.sub_mut();
                if sub.ve == Some(handle) {
                    sub.ve = None;
                }
            }
            Object::SamplerView(_) => {
                // The C's slot holds a reference, so a view destroyed while bound keeps
                // sampling until the slot is rebound. A slot here names the view by handle,
                // and a handle the guest has freed is reused by its next create: left in
                // place, the slot would answer "already bound" to the new view and keep the
                // old texture on the unit. Every slot holding it is emptied and marked, so
                // the next draw rebinds the unit and reselects the key the view fed.
                let sub = self.sub_mut();
                if evict_view(&mut sub.views, &mut sub.views_dirty, handle) {
                    sub.shader_dirty = true;
                }
            }
            Object::SamplerState(_) => {
                // The C nulls every slot holding it and compacts the slots after each down.
                for stage in self.sub_mut().samplers.iter_mut() {
                    let slots: Vec<(u32, ObjectHandle)> =
                        stage.iter().map(|(k, v)| (*k, *v)).collect();
                    let mut rebuilt = BTreeMap::new();
                    let mut shift = 0;
                    for (slot, h) in slots {
                        if h == handle {
                            shift += 1;
                        } else {
                            rebuilt.insert(slot - shift, h);
                        }
                    }
                    *stage = rebuilt;
                }
            }
            Object::Surface(_) => {
                // A DEVIATION FROM THE C, which holds a reference from the framebuffer, so a
                // surface destroyed while attached keeps taking pixels until the next
                // SET_FRAMEBUFFER_STATE. Here the surface's view texture is deleted with it,
                // and an attachment naming a deleted texture is detached by GL only if that
                // framebuffer happens to be the bound one -- so the C's behaviour would be
                // reproduced by luck, and the slot's value copy would compare equal to an
                // identical later bind and skip re-attaching what GL had quietly dropped.
                // The attachment and the slot are emptied together instead, which costs one
                // detach for a guest that destroys a bound surface before rebinding.
                let fb = self.sub().fb;
                let sub = self.sub_mut();
                let (held_z, colours) = evict_surface(&mut sub.zsurf, &mut sub.cbufs, handle);
                for i in &colours {
                    sub.swizzle_output_rgb_to_bgr &= !(1 << i);
                    sub.needs_manual_srgb_encode &= !(1 << i);
                }
                if held_z || !colours.is_empty() {
                    gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
                    if held_z {
                        gl.framebuffer_texture_2d(
                            GL_DEPTH_STENCIL_ATTACHMENT,
                            GL_TEXTURE_2D,
                            None,
                            0,
                        );
                    }
                    for i in colours {
                        gl.framebuffer_texture_2d(
                            GL_COLOR_ATTACHMENT0 + i as GLenum,
                            GL_TEXTURE_2D,
                            None,
                            0,
                        );
                    }
                    sub.shader_dirty = true;
                    sub.blend_dirty = true;
                }
            }
            Object::StreamoutTarget(_) => {
                let sub = self.sub_mut();
                let mut i = 0;
                while i < sub.streamouts.len() {
                    if sub.streamouts[i].targets.contains(&Some(handle)) {
                        let so = sub.streamouts.remove(i);
                        if sub.current_so == Some(i) {
                            sub.current_so = None;
                        } else if let Some(c) = sub.current_so
                            && c > i
                        {
                            sub.current_so = Some(c - 1);
                        }
                        if so.xfb == Xfb::Paused {
                            gl.bind_transform_feedback(Some(so.id));
                            gl.end_transform_feedback();
                        }
                        gl.delete_transform_feedback(so.id);
                    } else {
                        i += 1;
                    }
                }
                if let Some(c) = sub.current_so {
                    gl.bind_transform_feedback(Some(sub.streamouts[c].id));
                }
            }
            Object::Shader(shader) => {
                let sub = self.sub_mut();
                for s in sub.long_shader.iter_mut() {
                    if *s == Some(handle) {
                        *s = None;
                    }
                }
                // The C holds a reference from the bound slot, so the shader outlives its
                // handle there: the slot takes it over.
                let slot = &mut sub.shaders[shader.stage.index()];
                if slot.as_ref().is_some_and(|b| b.is(handle)) {
                    let Object::Shader(shader) = old else { unreachable!() };
                    *slot = Some(Bound::Owned(shader));
                    return;
                }
                let Object::Shader(shader) = old else { unreachable!() };
                draw::release_shader(sub, gl, shader);
                return;
            }
            _ => {}
        }
        release(gl, old);
    }

    fn bind_object(
        &mut self,
        host: &mut Host<'_>,
        kind: ObjectType,
        handle: Option<ObjectHandle>,
    ) -> Result<(), Fault> {
        let cmd = Cmd::BindObject;
        match kind {
            ObjectType::Blend => {
                let Some(h) = handle else {
                    self.sub_mut().blend = None;
                    host.gl.disable(GL_BLEND);
                    return Ok(());
                };
                let Object::Blend(s) = self.sub().object(cmd, h, ObjectType::Blend)? else {
                    unreachable!()
                };
                let s = *s;
                let sub = self.sub_mut();
                sub.blend = Some(s);
                sub.shader_dirty = true;
                sub.blend_dirty = true;
                Ok(())
            }
            ObjectType::Dsa => {
                let state = match handle {
                    None => None,
                    Some(h) => {
                        let Object::Dsa(s) = self.sub().object(cmd, h, ObjectType::Dsa)? else {
                            unreachable!()
                        };
                        Some((h, *s))
                    }
                };
                self.bind_dsa(host, state);
                Ok(())
            }
            ObjectType::Rasterizer => {
                let Some(h) = handle else {
                    self.sub_mut().rs = None;
                    return Ok(());
                };
                let Object::Rasterizer(s) = self.sub().object(cmd, h, ObjectType::Rasterizer)?
                else {
                    unreachable!()
                };
                self.sub_mut().rs = Some(*s);
                self.emit_rs(host);
                Ok(())
            }
            ObjectType::VertexElements => self.bind_vertex_elements(host, handle),
            _ => Err(Fault::OutOfRange { cmd, what: "object type" }),
        }
    }

    /// `vrend_object_bind_dsa_to_sub_context` and `vrend_hw_emit_dsa`.
    fn bind_dsa(&mut self, host: &mut Host<'_>, state: Option<(ObjectHandle, DepthStencilAlpha)>) {
        let gl = host.gl;
        let sub = self.sub_mut();
        if state.is_none() && sub.dsa.is_none() {
            return;
        }
        if sub.dsa.map(|(h, _)| h) != state.map(|(h, _)| h) {
            sub.stencil_dirty = true;
            sub.shader_dirty = true;
        }
        sub.dsa = state;
        let s = sub.dsa_state();
        if state.is_some() {
            sub.sysval.alpha_ref_val = s.alpha.ref_value;
        }
        if s.depth.enabled {
            if !sub.depth_test_enabled {
                gl.enable(GL_DEPTH_TEST);
                sub.depth_test_enabled = true;
            }
            gl.depth_func(GL_NEVER + s.depth.func.wire());
            gl.depth_mask(s.depth.writemask);
        } else if sub.depth_test_enabled {
            gl.disable(GL_DEPTH_TEST);
            sub.depth_test_enabled = false;
        }
        // Alpha test is the shader's on GLES.
    }

    /// `vrend_hw_emit_rs`, the GLES branches: what the C emits on every rasterizer bind.
    fn emit_rs(&mut self, host: &mut Host<'_>) {
        let gl = host.gl;
        let features = host.features;
        let sub = self.sub_mut();
        let s = sub.rs_state();
        if features.has(Feature::depth_clamp) {
            gl.set_enabled(GL_DEPTH_CLAMP_EXT, !s.depth_clip);
        }
        gl.line_width(if s.line_width <= 0.0 { 1.0 } else { s.line_width });
        if s.rasterizer_discard != sub.hw_rs.rasterizer_discard {
            sub.hw_rs.rasterizer_discard = s.rasterizer_discard;
            gl.set_enabled(GL_RASTERIZER_DISCARD, s.rasterizer_discard);
        }
        gl.set_enabled(GL_POLYGON_OFFSET_FILL, s.offset_tri);
        sub.hw_rs.flatshade = s.flatshade;
        if s.clip_halfz != sub.hw_rs.clip_halfz && features.has(Feature::clip_control) {
            let rule = if s.clip_halfz { GL_ZERO_TO_ONE_EXT } else { GL_NEGATIVE_ONE_TO_ONE_EXT };
            gl.clip_control(GL_LOWER_LEFT_EXT, rule);
            sub.hw_rs.clip_halfz = s.clip_halfz;
        }
        sub.hw_rs.flatshade_first = s.flatshade_first;
        gl.polygon_offset(s.offset_scale, s.offset_units);
        match s.cull_face {
            CullFace::None => gl.disable(GL_CULL_FACE),
            face => {
                gl.cull_face(match face {
                    CullFace::Front => GL_FRONT,
                    CullFace::Back => GL_BACK,
                    _ => GL_FRONT_AND_BACK,
                });
                gl.enable(GL_CULL_FACE);
            }
        }
        // The C toggles GL_CLIP_PLANE0+i here even on GLES, where the enum does not exist and
        // the call is an error that poisons the context. Clip planes are the shader's on GLES;
        // the toggle is dropped, and the C's error with it.
        if s.clip_plane_enable != sub.hw_rs.clip_plane_enable {
            sub.hw_rs.clip_plane_enable = s.clip_plane_enable;
            sub.sysval.clip_plane_enabled = if s.clip_plane_enable != 0 { 1.0 } else { 0.0 };
        }
        if features.has(Feature::multisample) {
            if features.has(Feature::sample_mask) {
                gl.set_enabled(GL_SAMPLE_MASK, s.multisample);
            }
            if features.has(Feature::sample_shading) {
                gl.set_enabled(GL_SAMPLE_SHADING, s.force_persample_interp);
            }
        }
        gl.set_enabled(GL_SCISSOR_TEST, s.scissor);
        sub.hw_rs.scissor = s.scissor;
    }

    /// `vrend_bind_vertex_elements_state`: the first bind lays the elements out in a VAO.
    fn bind_vertex_elements(
        &mut self,
        host: &mut Host<'_>,
        handle: Option<ObjectHandle>,
    ) -> Result<(), Fault> {
        let cmd = Cmd::BindObject;
        let Some(h) = handle else {
            self.sub_mut().ve = None;
            return Ok(());
        };
        let gl = host.gl;
        let max_attribs = host.limits.max_vertex_attributes;
        let sub = self.sub_mut();
        let Some(Object::VertexElements(v)) = sub.objects.get_mut(&h) else {
            return Err(Fault::IllegalHandle { cmd, handle: h });
        };
        if sub.ve != Some(h) {
            sub.vbo_dirty = true;
        }
        sub.ve = Some(h);
        if v.elements.len() as u32 > max_attribs {
            return Err(Fault::OutOfRange { cmd, what: "vertex attribute count" });
        }
        if v.vao.is_none() {
            let vao = gl.gen_vertex_array();
            gl.bind_vertex_array(Some(vao));
            for (i, e) in v.elements.iter().enumerate() {
                let i = i as GLuint;
                if e.pure_integer {
                    gl.vertex_attrib_i_format(i, e.nr_channels, e.gl_type, e.base.src_offset);
                } else {
                    gl.vertex_attrib_format(
                        i,
                        e.nr_channels,
                        e.gl_type,
                        e.normalized,
                        e.base.src_offset,
                    );
                }
                gl.vertex_attrib_binding(i, e.base.vertex_buffer_index);
                gl.vertex_binding_divisor(i, e.base.instance_divisor);
                gl.enable_vertex_attrib_array(i);
            }
            v.vao = Some(vao);
        }
        Ok(())
    }

    /// `vrend_bind_shader`: a handle that is not a shader of the stage is ignored, as the C
    /// ignores it. A shader the slot owned is released with the bind that replaces it.
    fn bind_shader(
        &mut self,
        host: &mut Host<'_>,
        handle: Option<ObjectHandle>,
        stage: ShaderStage,
    ) {
        let sub = self.sub_mut();
        let bound = match handle {
            None => None,
            Some(h) => match sub.objects.get(&h) {
                Some(Object::Shader(s)) if s.stage == stage => Some(Bound::Object(h)),
                _ => return,
            },
        };
        let same = match (&bound, &sub.shaders[stage.index()]) {
            (None, None) => true,
            (Some(Bound::Object(a)), Some(b)) => b.is(*a),
            _ => false,
        };
        if !same {
            sub.shader_dirty = true;
        }
        if let Some(Bound::Owned(s)) = std::mem::replace(&mut sub.shaders[stage.index()], bound) {
            draw::release_shader(sub, host.gl, s);
        }
    }

    /// `vrend_create_shader`: the long-shader protocol, and the parse once the text is whole.
    fn create_shader(
        &mut self,
        host: &mut Host<'_>,
        handle: ObjectHandle,
        s: ShaderCreate<'_>,
    ) -> Result<(), Fault> {
        let cmd = Cmd::CreateObject;
        let missing = match s.stage {
            ShaderStage::Geometry => !host.has(Feature::geometry_shader),
            ShaderStage::TessCtrl | ShaderStage::TessEval => !host.has(Feature::tessellation),
            ShaderStage::Compute => !host.has(Feature::compute_shader),
            _ => false,
        };
        if missing {
            return Err(Fault::Shader { cmd, what: "the stage's feature is absent" });
        }
        let bytes: Vec<u8> = s.text.iter().flat_map(|w| w.to_le_bytes()).collect();
        let in_progress = self.sub().long_shader[s.stage.index()];
        match s.chunk {
            ShaderChunk::New { total_bytes } => {
                if in_progress.is_some() {
                    return Err(Fault::Shader {
                        cmd,
                        what: "a new shader while one is in progress",
                    });
                }
                let total = (total_bytes as usize).div_ceil(4) * 4;
                if total < bytes.len() {
                    return Err(Fault::Shader { cmd, what: "more text than the declared length" });
                }
                let whole = total == bytes.len();
                let text = if whole {
                    ShaderText::Whole(read_shader(&bytes, s.num_tokens)?)
                } else {
                    self.sub_mut().long_shader[s.stage.index()] = Some(handle);
                    ShaderText::Arriving { text: bytes, total }
                };
                let shader = Shader { stage: s.stage, kind: s.kind, text };
                self.insert_object(host, handle, Object::Shader(shader));
                if whole {
                    self.select_new(host, handle)?;
                }
            }
            ShaderChunk::Continuation { offset } => {
                if in_progress != Some(handle) {
                    self.destroy_object(host, handle);
                    return Err(Fault::Shader { cmd, what: "a continuation of no shader" });
                }
                let sub = self.sub_mut();
                let Some(Object::Shader(shader)) = sub.objects.get_mut(&handle) else {
                    return Err(Fault::IllegalHandle { cmd, handle });
                };
                let ShaderText::Arriving { text, total } = &mut shader.text else {
                    return Err(Fault::Shader { cmd, what: "a continuation of a whole shader" });
                };
                let fits = offset as usize == text.len() && *total - text.len() >= bytes.len();
                if !fits {
                    sub.long_shader[s.stage.index()] = None;
                    self.destroy_object(host, handle);
                    return Err(Fault::Shader { cmd, what: "a continuation out of sequence" });
                }
                text.extend_from_slice(&bytes);
                if text.len() == *total {
                    // The C measures the whole against the completing command's token count.
                    let parsed = read_shader(text, s.num_tokens);
                    sub.long_shader[s.stage.index()] = None;
                    match parsed {
                        Ok(program) => shader.text = ShaderText::Whole(program),
                        Err(e) => {
                            self.destroy_object(host, handle);
                            return Err(e);
                        }
                    }
                    self.select_new(host, handle)?;
                }
            }
        }
        Ok(())
    }

    /// `vrend_finish_shader`'s selection: the first translation, under whatever is bound when
    /// the text completes. A program the translator refuses is destroyed with its handle, as
    /// the C destroys it.
    fn select_new(&mut self, host: &mut Host<'_>, handle: ObjectHandle) -> Result<(), Fault> {
        if let Err(e) = self.select_object(host, Cmd::CreateObject, handle) {
            self.destroy_object(host, handle);
            return Err(e);
        }
        Ok(())
    }

    /// `vrend_create_sampler_view`, and the texture view it makes when the host has them.
    fn create_sampler_view(&mut self, host: &mut Host<'_>, v: SamplerView) -> Result<View, Fault> {
        let cmd = Cmd::CreateObject;
        let gl = host.gl;
        let features = host.features;
        let formats = host.formats;
        let res = host.resource(cmd, v.resource)?;
        let entry = formats.get(v.format).ok_or(Fault::IllegalFormat { cmd, format: v.format })?;
        let (is_buffer, tex_name, tex_target, immutable) = match &res.storage {
            Storage::Buffer { .. } => (true, None, GL_TEXTURE_BUFFER, false),
            Storage::Texture { name, target, immutable, .. } => {
                (false, Some(*name), *target, *immutable)
            }
            Storage::Guest | Storage::Host(_) => {
                return Err(Fault::IllegalResource { cmd, handle: v.resource });
            }
        };
        let mut target = resource::gl_target(v.target, res.args.nr_samples);
        if is_buffer {
            target = tex_target;
            // A buffer view is an element range, first to last inclusive. The C binds one
            // past the host's texel limit shortened to fit and reports the view made; the
            // range is the guest's claim about the resource, and a claim past the limit is
            // refused here instead.
            let (first, last) = (v.first_element_or_layers, v.last_element_or_levels);
            let count = u64::from(last.wrapping_sub(first)).wrapping_add(1);
            if u64::from(first) + count > u64::from(host.limits.max_texture_buffer_size) {
                return Err(Fault::OutOfRange { cmd, what: "buffer view range" });
            }
        }
        let (mut first_layer, mut last_layer, first_level, last_level) = (
            v.first_element_or_layers & 0xffff,
            (v.first_element_or_layers >> 16) & 0xffff,
            v.last_element_or_levels & 0xff,
            (v.last_element_or_levels >> 8) & 0xff,
        );
        let desc = v.format.describe();
        let mut swizzle = v.swizzle;
        if !desc.is_some_and(|d| d.has_alpha() || d.is_depth_or_stencil()) {
            for s in swizzle.iter_mut() {
                if *s == Swizzle::W {
                    *s = Swizzle::One;
                }
            }
        }
        if let Some(table) = entry.gl.swizzle {
            for s in swizzle.iter_mut() {
                if (*s as u32) <= Swizzle::W as u32 {
                    *s = table[*s as usize];
                }
            }
        }
        let mut gl_swizzle = swizzle.map(|s| to_gl_swizzle(s) as GLint);
        let mut view = None;
        if let Some(tex) = tex_name {
            let res_format = res.args.format;
            let supports_view = res.supports_view();
            let res_is_ds = res_format.describe().is_some_and(|d| d.is_depth_or_stencil());
            let mut needs_view = target != tex_target;
            let view_format = if res_is_ds { res_format } else { v.format };
            if !res_is_ds && v.format != res_format {
                needs_view = true;
            }
            // A plane index, not a layer range: see the C's comment. Nothing here has an aux
            // plane image, so the index is spent and the range is the whole texture.
            if last_layer < first_layer {
                first_layer = 0;
                last_layer = 0;
            }
            if first_layer > 0 || first_level > 0 {
                needs_view = true;
            }
            if needs_view && immutable && features.has(Feature::texture_view) {
                let levels = last_level.wrapping_sub(first_level).wrapping_add(1);
                let layers = last_layer as i64 - first_layer as i64 + 1;
                if levels == 0 || layers <= 0 {
                    return Err(Fault::OutOfRange { cmd, what: "sampler view layers or levels" });
                }
                let ifmt = formats
                    .get(view_format)
                    .ok_or(Fault::IllegalFormat { cmd, format: view_format })?
                    .gl
                    .internalformat;
                let name = gl.gen_texture();
                // A view of an IOSurface-backed BGR* texture reads its red and blue swapped
                // (`Resource::supports_view`); the sampler's swizzle is ours to set, so the
                // swap is undone there. A BGR*-to-RGB* swap the guest asked for is left alone.
                if !supports_view && resource::is_bgra(v.format) {
                    gl_swizzle.swap(0, 2);
                }
                gl.texture_view(
                    name,
                    target,
                    tex,
                    ifmt,
                    first_level,
                    levels,
                    first_layer,
                    layers as GLuint,
                );
                gl.bind_texture(target, Some(name));
                if desc.is_some_and(|d| d.is_depth_or_stencil())
                    && features.has(Feature::stencil_texturing)
                {
                    let mode = if desc.is_some_and(|d| d.has_depth()) {
                        GL_DEPTH_COMPONENT
                    } else {
                        GL_STENCIL_INDEX
                    };
                    gl.tex_parameter_i(target, GL_DEPTH_STENCIL_TEXTURE_MODE, mode as GLint);
                }
                for (i, s) in gl_swizzle.iter().enumerate() {
                    gl.tex_parameter_i(target, GL_TEXTURE_SWIZZLE_R + i as GLenum, *s);
                }
                if desc.is_some_and(|d| d.is_srgb()) && features.has(Feature::texture_srgb_decode) {
                    gl.tex_parameter_i(target, GL_TEXTURE_SRGB_DECODE_EXT, GL_DECODE_EXT as GLint);
                }
                gl.bind_texture(target, None);
                view = Some(name);
            }
        }
        Ok(View {
            resource: v.resource,
            format: v.format,
            target,
            // No format can be a rectangle target on GLES (`vrend_formats.c` probes only
            // desktop GL for it), so every rectangle view is served by a 2D texture.
            emulated_rect: !is_buffer && v.target == TextureTarget::Rect,
            skip_srgb_decode: v.format != res.args.format
                && res.args.format.describe().is_some_and(|d| d.is_srgb())
                && !desc.is_some_and(|d| d.is_srgb()),
            view,
            first_layer,
            last_layer,
            first_level,
            last_level,
            first_element: v.first_element_or_layers,
            last_element: v.last_element_or_levels,
            gl_swizzle,
        })
    }

    /// `vrend_create_surface`, and the texture view it makes when the host has them.
    fn create_surface(&mut self, host: &mut Host<'_>, s: Surface) -> Result<Surf, Fault> {
        let cmd = Cmd::CreateObject;
        let gl = host.gl;
        let res = host.resource(cmd, s.resource)?;
        let (level, first_layer, last_layer) = match res.storage {
            Storage::Buffer { .. } => (0, s.first_element_or_level, s.last_element_or_layers),
            _ => (
                s.first_element_or_level,
                s.last_element_or_layers & 0xffff,
                (s.last_element_or_layers >> 16) & 0xffff,
            ),
        };
        let mut view = None;
        if let Storage::Texture { name, target, immutable: true, .. } = res.storage
            && host.features.has(Feature::texture_view)
        {
            let max_layer = res.depth_at(level).saturating_sub(1);
            let mut needs_view =
                first_layer != last_layer && (first_layer != 0 || last_layer != max_layer);
            if !needs_view && s.format != res.args.format {
                needs_view = true;
            }
            // A resource that cannot be viewed is rendered to as itself, with the conversion the
            // view would have done moved into the writes (`set_framebuffer_state`, the clears).
            if needs_view && res.supports_view() {
                let entry = host
                    .formats
                    .get(s.format)
                    .ok_or(Fault::IllegalFormat { cmd, format: s.format })?;
                let (mut fl, mut ll) = (first_layer, last_layer);
                if target == GL_TEXTURE_CUBE_MAP && fl == ll {
                    fl = 0;
                    ll = 5;
                }
                let layers = ll as i64 - fl as i64 + 1;
                if layers <= 0 {
                    return Err(Fault::OutOfRange { cmd, what: "surface layers" });
                }
                let v = gl.gen_texture();
                gl.texture_view(
                    v,
                    target,
                    name,
                    entry.gl.internalformat,
                    0,
                    res.args.last_level + 1,
                    fl,
                    layers as GLuint,
                );
                view = Some(v);
            }
        }
        Ok(Surf {
            resource: s.resource,
            format: s.format,
            level,
            first_layer,
            last_layer,
            nr_samples: s.samples,
            view,
        })
    }
}

/// `vrend_shader_assign_tgsi` up to the translation: a complete text -- which ends in a NUL
/// somewhere in its last dword, as the C requires -- parsed and scanned.
fn read_shader(text: &[u8], num_tokens: u32) -> Result<Program, Fault> {
    let cmd = Cmd::CreateObject;
    if text.len() < 4 || !text[text.len() - 4..].contains(&0) {
        return Err(Fault::Shader { cmd, what: "text without a terminator" });
    }
    let shader =
        tgsi::Program::parse(text, num_tokens).map_err(|error| Fault::Tgsi { cmd, error })?;
    let tgsi = tgsi::Program::scan(shader).map_err(|error| Fault::Tgsi { cmd, error })?;
    Ok(Program { tgsi, info: shader::Info::default(), variants: Vec::new() })
}

fn to_gl_swizzle(s: Swizzle) -> GLenum {
    match s {
        Swizzle::X => GL_RED,
        Swizzle::Y => GL_GREEN,
        Swizzle::Z => GL_BLUE,
        Swizzle::W => GL_ALPHA,
        Swizzle::Zero => GL_ZERO,
        Swizzle::One => GL_ONE,
    }
}

/// `vrend_create_vertex_elements_state`: the GL type of each element from its format.
fn vertex_elements(elements: &[VertexElement]) -> Result<VertexElements, Fault> {
    let mut out = Vec::with_capacity(elements.len());
    let mut zyxw_bitmask = 0;
    for (i, e) in elements.iter().enumerate() {
        let desc = e.src_format.describe().ok_or(Fault::IllegalVertexFormat(e.src_format))?;
        let c0 = desc.channels[0];
        let by_channel = match (c0.ty, c0.bits) {
            (super::formats::ChannelType::Float, 16) => Some(GL_HALF_FLOAT),
            (super::formats::ChannelType::Float, 32) => Some(GL_FLOAT),
            (super::formats::ChannelType::Float, 64) => Some(GL_DOUBLE),
            (super::formats::ChannelType::Unsigned, 8) => Some(GL_UNSIGNED_BYTE),
            (super::formats::ChannelType::Unsigned, 16) => Some(GL_UNSIGNED_SHORT),
            (super::formats::ChannelType::Unsigned, 32) => Some(GL_UNSIGNED_INT),
            (super::formats::ChannelType::Signed, 8) => Some(GL_BYTE),
            (super::formats::ChannelType::Signed, 16) => Some(GL_SHORT),
            (super::formats::ChannelType::Signed, 32) => Some(GL_INT),
            _ => None,
        };
        let by_name = match desc.name {
            "R10G10B10A2_SSCALED" | "R10G10B10A2_SNORM" | "B10G10R10A2_SNORM" => {
                Some(GL_INT_2_10_10_10_REV)
            }
            "R10G10B10A2_USCALED" | "R10G10B10A2_UNORM" | "B10G10R10A2_UNORM" => {
                Some(GL_UNSIGNED_INT_2_10_10_10_REV)
            }
            "R11G11B10_FLOAT" => Some(GL_UNSIGNED_INT_10F_11F_11F_REV),
            _ => None,
        };
        let gl_type = by_channel.or(by_name).ok_or(Fault::IllegalVertexFormat(e.src_format))?;
        if desc.nr_channels == 4 && desc.swizzle[0] == Some(Swizzle::Z) {
            zyxw_bitmask |= 1 << i;
        }
        out.push(Element {
            base: *e,
            gl_type,
            normalized: c0.normalized,
            nr_channels: desc.nr_channels as GLint,
            pure_integer: desc.is_pure_integer(),
        });
    }
    Ok(VertexElements { elements: out, zyxw_bitmask, vao: None })
}

/// `vrend_create_sampler_state`: two sampler objects with the state applied.
fn create_sampler_state(host: &mut Host<'_>, s: SamplerState) -> Result<Sampler, Fault> {
    if !host.has(Feature::samplers) {
        return Ok(Sampler { state: s, ids: None });
    }
    let gl = host.gl;
    let features = host.features;
    let wrap = |w: TexWrap| -> Result<GLenum, Fault> {
        Ok(match w {
            TexWrap::Repeat => GL_REPEAT,
            TexWrap::Clamp | TexWrap::ClampToEdge => GL_CLAMP_TO_EDGE,
            TexWrap::ClampToBorder => GL_CLAMP_TO_BORDER,
            TexWrap::MirrorRepeat => GL_MIRRORED_REPEAT,
            TexWrap::MirrorClamp => {
                if features.has(Feature::texture_mirror_clamp) {
                    GL_MIRROR_CLAMP_EXT
                } else {
                    return Err(Fault::UnsupportedTexWrap(w));
                }
            }
            TexWrap::MirrorClampToEdge => {
                if features.has(Feature::texture_mirror_clamp_to_edge) {
                    GL_MIRROR_CLAMP_TO_EDGE_EXT
                } else {
                    return Err(Fault::UnsupportedTexWrap(w));
                }
            }
            TexWrap::MirrorClampToBorder => {
                if features.has(Feature::texture_mirror_clamp_to_border) {
                    GL_MIRROR_CLAMP_TO_BORDER_EXT
                } else {
                    return Err(Fault::UnsupportedTexWrap(w));
                }
            }
        })
    };
    let (ws, wt, wr) = (wrap(s.wrap_s)?, wrap(s.wrap_t)?, wrap(s.wrap_r)?);
    let mag = |f: TexFilter| if f == TexFilter::Nearest { GL_NEAREST } else { GL_LINEAR };
    let min = match (s.min_img_filter, s.min_mip_filter) {
        (f, MipFilter::None) => mag(f),
        (TexFilter::Nearest, MipFilter::Linear) => GL_NEAREST_MIPMAP_LINEAR,
        (TexFilter::Linear, MipFilter::Linear) => GL_LINEAR_MIPMAP_LINEAR,
        (TexFilter::Nearest, MipFilter::Nearest) => GL_NEAREST_MIPMAP_NEAREST,
        (TexFilter::Linear, MipFilter::Nearest) => GL_LINEAR_MIPMAP_NEAREST,
    };
    let ids = [gl.gen_sampler(), gl.gen_sampler()];
    for (i, id) in ids.iter().enumerate() {
        let id = *id;
        gl.sampler_parameter_i(id, GL_TEXTURE_WRAP_S, ws as GLint);
        gl.sampler_parameter_i(id, GL_TEXTURE_WRAP_T, wt as GLint);
        gl.sampler_parameter_i(id, GL_TEXTURE_WRAP_R, wr as GLint);
        gl.sampler_parameter_f(id, GL_TEXTURE_MIN_FILTER, min as f32);
        gl.sampler_parameter_f(id, GL_TEXTURE_MAG_FILTER, mag(s.mag_img_filter) as f32);
        gl.sampler_parameter_f(id, GL_TEXTURE_MIN_LOD, s.min_lod);
        gl.sampler_parameter_f(id, GL_TEXTURE_MAX_LOD, s.max_lod);
        let mode = if s.compare_mode { GL_COMPARE_REF_TO_TEXTURE } else { GL_NONE };
        gl.sampler_parameter_i(id, GL_TEXTURE_COMPARE_MODE, mode as GLint);
        gl.sampler_parameter_i(
            id,
            GL_TEXTURE_COMPARE_FUNC,
            (GL_NEVER + s.compare_func.wire()) as GLint,
        );
        if features.has(Feature::sampler_border_colors) {
            gl.sampler_border_color(id, &s.border_color);
        }
        if features.has(Feature::texture_srgb_decode) {
            let decode = if i == 0 { GL_SKIP_DECODE_EXT } else { GL_DECODE_EXT };
            gl.sampler_parameter_i(id, GL_TEXTURE_SRGB_DECODE_EXT, decode as GLint);
        }
    }
    Ok(Sampler { state: s, ids: Some(ids) })
}

/// `vrend_create_query`, the GLES leg.
fn create_query(host: &mut Host<'_>, q: QueryCreate) -> Result<Query, Fault> {
    let cmd = Cmd::CreateObject;
    let res = host.resource(cmd, q.resource)?;
    if !matches!(res.storage, Storage::Host(_)) {
        return Err(Fault::IllegalResource { cmd, handle: q.resource });
    }
    let mut kind = q.kind;
    let mut fake = false;
    if kind == QueryType::OcclusionCounter && !host.has(Feature::occlusion_query) {
        kind = QueryType::OcclusionPredicate;
        fake = true;
    }
    let gl_type = match kind {
        QueryType::OcclusionCounter => GL_SAMPLES_PASSED,
        QueryType::OcclusionPredicate => GL_ANY_SAMPLES_PASSED,
        QueryType::OcclusionPredicateConservative => GL_ANY_SAMPLES_PASSED_CONSERVATIVE,
        QueryType::Timestamp | QueryType::TimeElapsed => {
            if !host.has(Feature::timer_query) {
                return Err(Fault::Unimplemented { cmd, what: "timer queries" });
            }
            if kind == QueryType::Timestamp { GL_TIMESTAMP_EXT } else { GL_TIME_ELAPSED_EXT }
        }
        QueryType::PrimitivesGenerated => GL_PRIMITIVES_GENERATED,
        QueryType::PrimitivesEmitted => GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN,
        QueryType::SoOverflowPredicate
        | QueryType::SoOverflowAnyPredicate
        | QueryType::PipelineStatistics => {
            return Err(Fault::Unimplemented { cmd, what: "that query type on GLES" });
        }
        QueryType::TimestampDisjoint | QueryType::SoStatistics | QueryType::GpuFinished => 0,
    };
    let id = host.gl.gen_query();
    Ok(Query { kind, gl_type, id, resource: q.resource, fake_samples_passed: fake })
}

// ---- state ----

impl Context {
    /// `vrend_set_viewport_states`. Every call marks the viewport dirty: the C's
    /// `viewport_state_initialized &= bit` never sets the bit, so its compare always fires.
    fn set_viewports(&mut self, start_slot: u32, viewports: &[Viewport]) {
        let sub = self.sub_mut();
        let clip_halfz = sub.rs_state().clip_halfz;
        let negative = viewports.first().is_some_and(|v| v.scale[1] < 0.0);
        for (i, v) in viewports.iter().enumerate() {
            let idx = start_slot as usize + i;
            let (near, far) = if !clip_halfz {
                let near = (v.translate[2] - v.scale[2]) as f64;
                (near, near + v.scale[2] as f64 * 2.0)
            } else {
                (v.translate[2] as f64, (v.scale[2] + v.translate[2]) as f64)
            };
            sub.viewports[idx] = ViewportHw {
                x: (v.translate[0] - v.scale[0]) as GLint,
                y: (v.translate[1] - v.scale[1]) as GLint,
                width: (v.scale[0] * 2.0) as GLsizei,
                height: (v.scale[1].abs() * 2.0) as GLsizei,
                near,
                far,
            };
            sub.viewport_dirty.mark(idx as u32);
            if idx == 0 && sub.viewport_is_negative != negative {
                sub.viewport_is_negative = negative;
                sub.sysval.winsys_adjust_y = if negative { -1.0 } else { 1.0 };
            }
        }
    }

    /// The lazy state a clear or draw flushes first: front face, stencil, scissor, viewport.
    fn flush_lazy_state(&mut self, host: &mut Host<'_>) {
        let gl = host.gl;
        let sub = self.sub_mut();
        let rs = sub.rs_state();
        let front_ccw = rs.front_ccw ^ !sub.fbo_origin_upper_left;
        gl.front_face(if front_ccw { GL_CCW } else { GL_CW });
        if sub.stencil_dirty {
            sub.stencil_dirty = false;
            if let Some((_, dsa)) = sub.dsa {
                let op = |o: StencilOp| match o {
                    StencilOp::Keep => GL_KEEP,
                    StencilOp::Zero => GL_ZERO,
                    StencilOp::Replace => GL_REPLACE,
                    StencilOp::Incr => GL_INCR,
                    StencilOp::Decr => GL_DECR,
                    StencilOp::IncrWrap => GL_INCR_WRAP,
                    StencilOp::DecrWrap => GL_DECR_WRAP,
                    StencilOp::Invert => GL_INVERT,
                };
                let [front, back] = dsa.stencil;
                if !back.enabled {
                    if front.enabled {
                        if !sub.stencil_test_enabled {
                            gl.enable(GL_STENCIL_TEST);
                            sub.stencil_test_enabled = true;
                        }
                        gl.stencil_op(op(front.fail_op), op(front.zfail_op), op(front.zpass_op));
                        gl.stencil_func(
                            GL_NEVER + front.func.wire(),
                            sub.stencil_refs[0] as GLint,
                            front.valuemask as GLuint,
                        );
                        gl.stencil_mask(front.writemask as GLuint);
                    } else if sub.stencil_test_enabled {
                        gl.disable(GL_STENCIL_TEST);
                        sub.stencil_test_enabled = false;
                    }
                } else {
                    if !sub.stencil_test_enabled {
                        gl.enable(GL_STENCIL_TEST);
                        sub.stencil_test_enabled = true;
                    }
                    for (i, face) in [(0, front), (1, back)] {
                        let gl_face = if i == 1 { GL_BACK } else { GL_FRONT };
                        gl.stencil_op_separate(
                            gl_face,
                            op(face.fail_op),
                            op(face.zfail_op),
                            op(face.zpass_op),
                        );
                        gl.stencil_func_separate(
                            gl_face,
                            GL_NEVER + face.func.wire(),
                            sub.stencil_refs[i] as GLint,
                            face.valuemask as GLuint,
                        );
                        gl.stencil_mask_separate(gl_face, face.writemask as GLuint);
                    }
                }
            }
        }
        if !sub.scissor_dirty.is_empty() {
            for idx in 0..MAX_VIEWPORTS {
                if !sub.scissor_dirty.contains(idx as u32) {
                    continue;
                }
                let s = sub.scissors[idx];
                // Only viewport 0 has a scissor on this host: `glScissorIndexed` is
                // viewport-array's, which GLES lacks.
                if idx == 0 {
                    gl.scissor(
                        s.minx as GLint,
                        s.miny as GLint,
                        s.maxx as GLsizei - s.minx as GLsizei,
                        s.maxy as GLsizei - s.miny as GLsizei,
                    );
                }
            }
            sub.scissor_dirty.clear();
        }
        if !sub.viewport_dirty.is_empty() {
            for idx in 0..MAX_VIEWPORTS {
                if !sub.viewport_dirty.contains(idx as u32) || idx != 0 {
                    continue;
                }
                let v = sub.viewports[idx];
                let cy = if sub.viewport_is_negative { v.y - v.height } else { v.y };
                gl.viewport(v.x, cy, v.width, v.height);
                gl.depth_range_f(v.near as f32, v.far as f32);
            }
            sub.viewport_dirty.clear();
        }
    }

    /// `vrend_set_framebuffer_state`.
    fn set_framebuffer_state(
        &mut self,
        host: &mut Host<'_>,
        zsurf: Option<ObjectHandle>,
        cbufs: &[Option<ObjectHandle>],
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetFramebufferState;
        let gl = host.gl;
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(self.sub().fb));
        let new_z = match zsurf {
            None => None,
            Some(h) => Some(self.bound_surface(host, cmd, h)?),
        };
        if self.sub().zsurf != new_z {
            match new_z {
                None => {
                    gl.framebuffer_texture_2d(GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, None, 0)
                }
                Some(_) => self.attach_surface(host, cmd, zsurf.expect("bound"), 0)?,
            }
            self.sub_mut().zsurf = new_z;
        }
        let old_num = self.sub().cbufs.len();
        let mut new_cbufs = Vec::with_capacity(cbufs.len());
        for (i, h) in cbufs.iter().enumerate() {
            let want = match h {
                None => None,
                Some(h) => Some(self.bound_surface(host, cmd, *h)?),
            };
            let had = self.sub().cbufs.get(i).copied().flatten();
            if had != want {
                match h {
                    None => gl.framebuffer_texture_2d(
                        GL_COLOR_ATTACHMENT0 + i as GLenum,
                        GL_TEXTURE_2D,
                        None,
                        0,
                    ),
                    Some(h) => self.attach_surface(host, cmd, *h, i as u32)?,
                }
            }
            new_cbufs.push(want);
        }
        for i in cbufs.len()..old_num {
            if self.sub().cbufs[i].is_some() {
                gl.framebuffer_texture_2d(
                    GL_COLOR_ATTACHMENT0 + i as GLenum,
                    GL_TEXTURE_2D,
                    None,
                    0,
                );
            }
        }
        // `vrend_hw_emit_framebuffer_state`'s per-attachment half: what a view would have
        // converted, for the resources that cannot be viewed, becomes work for the writes.
        let (mut to_bgr, mut encode) = (0u8, 0u8);
        for (i, s) in new_cbufs.iter().enumerate() {
            let Some(s) = s else { continue };
            let res = host.resource(cmd, s.resource)?;
            if res.needs_redblue_swizzle(s.format) {
                to_bgr |= 1 << i;
            }
            if !res.supports_view() && s.format.describe().is_some_and(|d| d.is_srgb()) {
                encode |= 1 << i;
            }
        }
        let sub = self.sub_mut();
        sub.cbufs = new_cbufs;
        sub.swizzle_output_rgb_to_bgr = to_bgr;
        sub.needs_manual_srgb_encode = encode;
        let (height, upper_left) = if sub.cbufs.is_empty() && sub.zsurf.is_none() {
            (0, false)
        } else if sub.cbufs.is_empty() {
            let z = sub.zsurf.expect("checked");
            (resource::minify(z.tex_height, z.level), z.y_0_top)
        } else {
            let Some(s) = sub.cbufs.iter().flatten().next() else {
                return Err(Fault::OutOfRange { cmd, what: "a framebuffer with no surface" });
            };
            (resource::minify(s.tex_height, s.level), s.y_0_top)
        };
        if sub.fb_height != height || sub.fbo_origin_upper_left != upper_left {
            sub.fb_height = height;
            sub.fbo_origin_upper_left = upper_left;
            sub.viewport_dirty.mark(0);
        }
        // `vrend_hw_emit_framebuffer_state`.
        let srgb_control = host.features.has(Feature::srgb_write_control);
        if sub.cbufs.is_empty() {
            gl.read_buffer(GL_NONE);
            if srgb_control {
                gl.disable(GL_FRAMEBUFFER_SRGB_EXT);
                sub.framebuffer_srgb_enabled = false;
            }
        } else if srgb_control {
            let use_srgb = sub
                .cbufs
                .iter()
                .flatten()
                .any(|s| s.format.describe().is_some_and(|d| d.is_srgb()));
            gl.set_enabled(GL_FRAMEBUFFER_SRGB_EXT, use_srgb);
            sub.framebuffer_srgb_enabled = use_srgb;
        }
        let bufs: Vec<GLenum> =
            (0..sub.cbufs.len()).map(|i| GL_COLOR_ATTACHMENT0 + i as GLenum).collect();
        gl.draw_buffers(&bufs);
        if !sub.cbufs.is_empty() || sub.zsurf.is_some() {
            let status = gl.check_framebuffer_status();
            if status != GL_FRAMEBUFFER_COMPLETE {
                eprintln!("[virglrs] vrend: framebuffer incomplete: {status:#x}");
            }
        }
        sub.shader_dirty = true;
        sub.blend_dirty = true;
        Ok(())
    }

    /// A surface resolved against its resource, as the framebuffer keeps it.
    fn bound_surface(
        &self,
        host: &Host<'_>,
        cmd: Cmd,
        h: ObjectHandle,
    ) -> Result<BoundSurface, Fault> {
        let s = self.sub().surface(cmd, h)?;
        let res = host.resource(cmd, s.resource)?;
        Ok(BoundSurface {
            handle: h,
            resource: s.resource,
            format: s.format,
            level: s.level,
            nr_samples: s.nr_samples,
            tex_height: res.args.height,
            y_0_top: res.y_0_top(),
        })
    }

    /// `vrend_fb_bind_texture_id` for a surface object, on the bound framebuffer.
    fn attach_surface(
        &self,
        host: &mut Host<'_>,
        cmd: Cmd,
        h: ObjectHandle,
        idx: u32,
    ) -> Result<(), Fault> {
        let s = self.sub().surface(cmd, h)?;
        let res = host.resource(cmd, s.resource)?;
        if s.nr_samples > 0 {
            host.todo.note("implicit multisample surfaces");
            return Err(Fault::Unimplemented { cmd, what: "a multisampled surface" });
        }
        let (name, target) = match (&res.storage, s.view) {
            (Storage::Texture { target, .. }, Some(v)) => (v, *target),
            (Storage::Texture { name, target, .. }, None) => (*name, *target),
            _ => return Err(Fault::IllegalResource { cmd, handle: s.resource }),
        };
        let mut attachment = transfer::attachment_for(res, host.formats);
        if attachment == GL_COLOR_ATTACHMENT0 {
            attachment += idx;
        }
        transfer::attach_texture(
            host.gl,
            host.features,
            target,
            name,
            attachment,
            s.level as GLint,
            s.layer(),
        )
        .map_err(|feature| Fault::NoFeature { cmd, feature })
    }

    fn set_vertex_buffers(
        &mut self,
        host: &mut Host<'_>,
        vbos: Vec<VertexBuffer>,
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetVertexBuffers;
        for v in &vbos {
            if let Some(r) = v.resource {
                host.resource(cmd, r)?;
            }
        }
        let sub = self.sub_mut();
        if sub.vbos != vbos {
            sub.vbo_dirty = true;
        }
        sub.vbos = vbos;
        Ok(())
    }

    fn set_index_buffer(
        &mut self,
        host: &mut Host<'_>,
        ib: Option<IndexBuffer>,
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetIndexBuffer;
        if let Some(ib) = ib {
            let res = host.resource(cmd, ib.resource)?;
            if !matches!(res.storage, Storage::Buffer { .. } | Storage::Texture { .. }) {
                return Err(Fault::IllegalResource { cmd, handle: ib.resource });
            }
        }
        self.sub_mut().ib = ib;
        Ok(())
    }

    fn set_uniform_buffer(
        &mut self,
        host: &mut Host<'_>,
        stage: ShaderStage,
        index: u32,
        offset: u32,
        length: u32,
        resource: Option<ResourceHandle>,
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetUniformBuffer;
        let sub = self.sub_mut();
        let ubos = &mut sub.ubos[stage.index()];
        match resource {
            None => {
                ubos.remove(&index);
            }
            Some(r) => {
                let res = host.resource(cmd, r)?;
                if !matches!(res.storage, Storage::Buffer { .. } | Storage::Texture { .. }) {
                    return Err(Fault::IllegalResource { cmd, handle: r });
                }
                ubos.insert(index, Ubo { resource: r, offset, length });
            }
        }
        // The decoder refuses a constant-buffer index past MAX_CONSTANT_BUFFERS.
        sub.ubos_dirty[stage.index()].mark(index);
        Ok(())
    }

    /// `vrend_set_single_sampler_view` per slot, then `vrend_set_num_sampler_views`.
    fn set_sampler_views(
        &mut self,
        host: &mut Host<'_>,
        stage: ShaderStage,
        start_slot: u32,
        views: &[Option<ObjectHandle>],
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetSamplerViews;
        for (i, h) in views.iter().enumerate() {
            let slot = start_slot + i as u32;
            let Some(h) = h else {
                self.sub_mut().views[stage.index()].remove(&slot);
                continue;
            };
            let sub = self.sub_mut();
            let Some(Object::SamplerView(view)) = sub.objects.get(h) else {
                sub.views[stage.index()].remove(&slot);
                return Err(Fault::IllegalHandle { cmd, handle: *h });
            };
            if sub.views[stage.index()].get(&slot) == Some(h) {
                continue;
            }
            // A view above the sampler units is held but never sampled.
            if (slot as usize) < MAX_SAMPLERS {
                sub.views_dirty[stage.index()].mark(slot);
            }
            let (gl, features, formats) = (host.gl, host.features, host.formats);
            let mut buffer_view = false;
            let res = host.resource_mut(cmd, view.resource)?;
            match &mut res.storage {
                Storage::Texture { name, target, .. } if view.view.is_none() => {
                    let (name, target) = (*name, *target);
                    gl.bind_texture(view.target, Some(name));
                    let desc = view.format.describe();
                    if desc.is_some_and(|d| d.is_depth_or_stencil())
                        && features.has(Feature::stencil_texturing)
                    {
                        let mode = if desc.is_some_and(|d| d.has_depth()) {
                            GL_DEPTH_COMPONENT
                        } else {
                            GL_STENCIL_INDEX
                        };
                        gl.tex_parameter_i(target, GL_DEPTH_STENCIL_TEXTURE_MODE, mode as GLint);
                    }
                    gl.tex_parameter_i(target, GL_TEXTURE_BASE_LEVEL, view.first_level as GLint);
                    gl.tex_parameter_i(target, GL_TEXTURE_MAX_LEVEL, view.last_level as GLint);
                    for (c, s) in view.gl_swizzle.iter().enumerate() {
                        gl.tex_parameter_i(target, GL_TEXTURE_SWIZZLE_R + c as GLenum, *s);
                    }
                }
                Storage::Texture { .. } => {}
                Storage::Buffer { name, tbo, .. } => {
                    // `create_sampler_view` refused every range on a host without buffer
                    // textures, whose limit is zero; this is the same fact, said once more.
                    if !features.has(Feature::arb_or_gles_ext_texture_buffer) {
                        return Err(Fault::NoFeature {
                            cmd,
                            feature: Feature::arb_or_gles_ext_texture_buffer,
                        });
                    }
                    let tbo_tex = *tbo.get_or_insert_with(|| gl.gen_texture());
                    gl.bind_texture(GL_TEXTURE_BUFFER, Some(tbo_tex));
                    buffer_view = true;
                    let entry = formats.get(view.format);
                    let mut ifmt = entry.map_or(GL_NONE, |e| e.gl.internalformat);
                    if ifmt == GL_NONE || ifmt == GL_ALPHA8_EXT {
                        ifmt = arb_format(view.format);
                    }
                    let range = if features.has(Feature::texture_buffer_range) {
                        // Within the host's limit: `create_sampler_view` refused any other.
                        let bs = view.format.describe().map_or(1, |d| d.block_bytes()) as usize;
                        let offset = view.first_element as usize;
                        let size = (view.last_element as usize) - offset + 1;
                        Some((offset * bs, size * bs))
                    } else {
                        None
                    };
                    gl.tex_buffer(ifmt, *name, range);
                }
                Storage::Guest | Storage::Host(_) => {
                    return Err(Fault::IllegalResource { cmd, handle: view.resource });
                }
            }
            let sub = self.sub_mut();
            sub.views[stage.index()].insert(slot, *h);
            if buffer_view {
                sub.shader_dirty = true;
            }
        }
        let end = start_slot + views.len() as u32;
        self.sub_mut().views[stage.index()].retain(|slot, _| *slot < end);
        Ok(())
    }

    /// `vrend_bind_sampler_states`: a handle that is not a sampler state binds nothing, with a
    /// warning, as in the C.
    fn bind_sampler_states(
        &mut self,
        stage: ShaderStage,
        start_slot: u32,
        states: &[Option<ObjectHandle>],
    ) {
        let sub = self.sub_mut();
        for (i, h) in states.iter().enumerate() {
            let slot = start_slot + i as u32;
            match h {
                Some(h) if matches!(sub.objects.get(h), Some(Object::SamplerState(_))) => {
                    sub.samplers[stage.index()].insert(slot, *h);
                }
                Some(h) => {
                    eprintln!("[virglrs] vrend: no sampler state under handle {h}");
                    sub.samplers[stage.index()].remove(&slot);
                }
                None => {
                    sub.samplers[stage.index()].remove(&slot);
                }
            }
        }
    }

    fn set_shader_buffers(
        &mut self,
        host: &mut Host<'_>,
        stage: ShaderStage,
        start_slot: u32,
        buffers: &[ShaderBuffer],
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetShaderBuffers;
        if !host.has(Feature::ssbo) {
            return Ok(());
        }
        for (i, b) in buffers.iter().enumerate() {
            let slot = start_slot + i as u32;
            match b.resource {
                None => {
                    self.sub_mut().ssbos[stage.index()].remove(&slot);
                }
                Some(r) => {
                    let res = host.resource(cmd, r)?;
                    if b.offset > res.args.width || b.length > res.args.width - b.offset {
                        return Err(Fault::OutOfRange { cmd, what: "shader buffer range" });
                    }
                    self.sub_mut().ssbos[stage.index()]
                        .insert(slot, Ssbo { resource: r, offset: b.offset, length: b.length });
                }
            }
        }
        Ok(())
    }

    fn set_atomic_buffers(
        &mut self,
        host: &mut Host<'_>,
        start_slot: u32,
        buffers: &[ShaderBuffer],
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetAtomicBuffers;
        if !host.has(Feature::atomic_counters) {
            return Ok(());
        }
        for (i, b) in buffers.iter().enumerate() {
            let slot = start_slot + i as u32;
            match b.resource {
                None => {
                    self.sub_mut().abos.remove(&slot);
                }
                Some(r) => {
                    host.resource(cmd, r)?;
                    self.sub_mut()
                        .abos
                        .insert(slot, Ssbo { resource: r, offset: b.offset, length: b.length });
                }
            }
        }
        Ok(())
    }

    fn set_shader_images(
        &mut self,
        host: &mut Host<'_>,
        stage: ShaderStage,
        start_slot: u32,
        images: &[Option<ShaderImage>],
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetShaderImages;
        for (i, im) in images.iter().enumerate() {
            let slot = start_slot + i as u32;
            match im {
                None => {
                    self.sub_mut().images[stage.index()].remove(&slot);
                }
                Some(im) => {
                    let r = im.resource;
                    if !host.has(Feature::images) {
                        return Err(Fault::Unimplemented { cmd, what: "shader images" });
                    }
                    let res = host.resource(cmd, r)?;
                    match res.storage {
                        Storage::Texture { .. } => {
                            let (first, last) =
                                (im.layer_offset & 0xffff, (im.layer_offset >> 16) & 0xffff);
                            if last.wrapping_sub(first).wrapping_add(1) & 0xffff == 0 {
                                return Err(Fault::OutOfRange { cmd, what: "image layers" });
                            }
                        }
                        Storage::Buffer { .. } => {
                            // A buffer image is a texel range: `layer_offset` and `level_size`
                            // are its byte offset and length. The C binds a range past the
                            // host's texel limit shortened to fit, reporting success for a
                            // view it did not make; the range is the guest's claim about the
                            // resource, and a claim past the limit is refused here instead.
                            let Some(bs) = im
                                .format
                                .describe()
                                .map(|d| d.block_bytes())
                                .filter(|bs| matches!(bs, 1 | 2 | 4 | 8 | 16))
                            else {
                                return Err(Fault::IllegalFormat { cmd, format: im.format });
                            };
                            let texels =
                                u64::from(im.layer_offset / bs) + u64::from(im.level_size / bs);
                            if texels > u64::from(host.limits.max_texture_buffer_size) {
                                return Err(Fault::OutOfRange { cmd, what: "image buffer range" });
                            }
                        }
                        _ => {}
                    }
                    self.sub_mut().images[stage.index()].insert(
                        slot,
                        ImageView {
                            resource: r,
                            format: im.format,
                            access: im.access,
                            layer_offset: im.layer_offset,
                            level_size: im.level_size,
                        },
                    );
                }
            }
        }
        Ok(())
    }

    /// `vrend_set_streamout_targets`.
    fn set_streamout_targets(
        &mut self,
        host: &mut Host<'_>,
        targets: &[Option<ObjectHandle>],
    ) -> Result<(), Fault> {
        let cmd = Cmd::SetStreamoutTargets;
        if !host.has(Feature::transform_feedback) {
            return Ok(());
        }
        let gl = host.gl;
        if targets.is_empty() {
            gl.bind_transform_feedback(None);
            self.sub_mut().current_so = None;
            return Ok(());
        }
        if let Some(i) = self.sub().streamouts.iter().position(|so| so.targets == targets) {
            let sub = self.sub_mut();
            sub.current_so = Some(i);
            gl.bind_transform_feedback(Some(sub.streamouts[i].id));
            return Ok(());
        }
        let id = gl.gen_transform_feedback();
        gl.bind_transform_feedback(Some(id));
        for (i, t) in targets.iter().enumerate() {
            let Some(h) = t else {
                gl.bind_buffer_base(GL_TRANSFORM_FEEDBACK_BUFFER, BindingPoint::at(i as u32), None);
                continue;
            };
            let target = match self.sub().objects.get(h) {
                Some(Object::StreamoutTarget(t)) => *t,
                _ => {
                    gl.delete_transform_feedback(id);
                    return Err(Fault::IllegalHandle { cmd, handle: *h });
                }
            };
            let res = host.resource(cmd, target.resource)?;
            let Storage::Buffer { name, .. } = res.storage else {
                gl.delete_transform_feedback(id);
                return Err(Fault::IllegalResource { cmd, handle: target.resource });
            };
            if target.buffer_offset == 0 && target.buffer_size == res.args.width {
                gl.bind_buffer_base(
                    GL_TRANSFORM_FEEDBACK_BUFFER,
                    BindingPoint::at(i as u32),
                    Some(name),
                );
            } else {
                gl.bind_buffer_range(
                    GL_TRANSFORM_FEEDBACK_BUFFER,
                    BindingPoint::at(i as u32),
                    name,
                    target.buffer_offset as usize,
                    target.buffer_size as usize,
                );
            }
        }
        let sub = self.sub_mut();
        sub.streamouts.push(Streamout { id, targets: targets.to_vec(), xfb: Xfb::NeedBegin });
        sub.current_so = Some(sub.streamouts.len() - 1);
        Ok(())
    }

    /// `vrend_render_condition` on a host without conditional rendering: recorded, never
    /// applied; an unknown query is ignored as the C ignores it.
    fn set_render_condition(
        &mut self,
        query: Option<ObjectHandle>,
        condition: bool,
        mode: RenderCondMode,
    ) {
        let sub = self.sub_mut();
        match query {
            None => sub.render_condition = None,
            Some(h) => {
                if matches!(sub.objects.get(&h), Some(Object::Query(_))) {
                    sub.render_condition = Some((h, condition, mode));
                }
            }
        }
    }

    fn set_min_samples(&mut self, host: &mut Host<'_>, min_samples: u32) {
        if !host.has(Feature::sample_shading) {
            return;
        }
        let samples =
            self.sub().cbufs.first().copied().flatten().map_or(0, |s| s.nr_samples).max(1);
        host.gl.min_sample_shading(min_samples as f32 / samples as f32);
    }

    /// `vrend_memory_barrier`.
    fn memory_barrier(&mut self, host: &mut Host<'_>, flags: u32) {
        if !host.has(Feature::barrier) {
            return;
        }
        const ALL: u32 = 0xffff_ffff;
        let bits = if flags == ALL {
            GL_ALL_BARRIER_BITS
        } else {
            let pairs: [(u32, GLbitfield); 12] = [
                (1 << 0, GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT),
                (1 << 1, GL_ELEMENT_ARRAY_BARRIER_BIT),
                (1 << 2, GL_UNIFORM_BARRIER_BIT),
                (1 << 3, GL_TEXTURE_FETCH_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT),
                (1 << 4, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT),
                (1 << 5, GL_COMMAND_BARRIER_BIT),
                (1 << 6, GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT),
                (1 << 7, GL_FRAMEBUFFER_BARRIER_BIT),
                (1 << 8, GL_TRANSFORM_FEEDBACK_BARRIER_BIT),
                (1 << 9, GL_BUFFER_UPDATE_BARRIER_BIT),
                (1 << 10, GL_TEXTURE_UPDATE_BARRIER_BIT),
                (
                    1 << 11,
                    GL_ATOMIC_COUNTER_BARRIER_BIT
                        | if host.has(Feature::ssbo_barrier) {
                            GL_SHADER_STORAGE_BARRIER_BIT
                        } else {
                            0
                        },
                ),
            ];
            pairs.iter().filter(|(f, _)| flags & f != 0).fold(0, |acc, (_, b)| acc | b)
        };
        host.gl.memory_barrier(bits);
    }
}

/// `vrend_get_arb_format`: the R/RG internal format an A/L/I format samples as on a buffer.
fn arb_format(format: Format) -> GLenum {
    match format.name() {
        "A8_UNORM" | "L8_UNORM" | "I8_UNORM" => GL_R8,
        "A8_SINT" | "L8_SINT" | "I8_SINT" => GL_R8I,
        "A8_UINT" | "L8_UINT" | "I8_UINT" => GL_R8UI,
        "A16_FLOAT" | "L16_UNORM" | "L16_FLOAT" | "I16_FLOAT" => GL_R16F,
        "A32_FLOAT" | "L32_SINT" | "L32_FLOAT" | "I32_FLOAT" => GL_R32F,
        "L16_SINT" | "I16_SINT" => GL_R16I,
        "L16_UINT" | "I16_UINT" => GL_R16UI,
        "L32_UINT" => GL_R32I,
        "I32_SINT" => GL_R32I,
        "I32_UINT" => GL_R32UI,
        "L8A8_UNORM" => GL_RG8,
        "L8A8_SINT" => GL_RG8I,
        "L8A8_UINT" => GL_RG8UI,
        "L16A16_UNORM" => GL_RG16_EXT,
        "L16A16_SINT" => GL_RG16I,
        "L16A16_UINT" => GL_RG16UI,
        "L16A16_FLOAT" => GL_RG16F,
        "L32A32_FLOAT" => GL_RG32F,
        "L32A32_SINT" => GL_RG32I,
        "L32A32_UINT" => GL_RG32UI,
        "I16_UNORM" => GL_R16_EXT,
        _ => GL_R8,
    }
}

// ---- clears ----

/// `vrend_color_encode_as_srgb`: one linear channel to sRGB.
fn encode_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

impl Context {
    /// `vrend_clear_prepare`: unmask what the clear writes, set the values.
    ///
    /// `surf` is the surface the colour lands on: a resource that cannot be viewed as the
    /// surface's format gets the view's conversion applied to the clear colour instead.
    fn clear_prepare(
        &mut self,
        host: &mut Host<'_>,
        surf: Option<(ResourceHandle, Format)>,
        buffers: u32,
        mut color: [f32; 4],
        depth: f64,
        stencil: u32,
    ) -> Result<(), Fault> {
        let gl = host.gl;
        let indep = host.has(Feature::indep_blend);
        if let Some((handle, format)) = surf {
            let res = host.resource(Cmd::Clear, handle)?;
            if !res.supports_view() && format.describe().is_some_and(|d| d.is_srgb()) {
                for c in &mut color[..3] {
                    *c = encode_srgb(*c);
                }
            }
            if res.needs_redblue_swizzle(format) {
                color.swap(0, 2);
            }
        }
        let sub = self.sub_mut();
        if buffers & PIPE_CLEAR_COLOR != 0 {
            gl.clear_color(color);
            if sub.hw_blend.independent && indep {
                for i in 0..MAX_COLOR_BUFS {
                    gl.color_mask_i(i as GLuint, [true; 4]);
                }
            } else {
                gl.color_mask([true; 4]);
            }
        }
        if buffers & PIPE_CLEAR_DEPTH != 0 {
            gl.depth_mask(true);
            gl.clear_depth_f(depth as f32);
        }
        if buffers & PIPE_CLEAR_STENCIL != 0 {
            gl.stencil_mask(!0);
            gl.clear_stencil(stencil as GLint);
        }
        if sub.hw_rs.rasterizer_discard {
            gl.disable(GL_RASTERIZER_DISCARD);
        }
        Ok(())
    }

    /// `vrend_clear_finish`: restore the masks the clear lifted.
    fn clear_finish(&mut self, host: &mut Host<'_>, buffers: u32) {
        let gl = host.gl;
        let indep = host.has(Feature::indep_blend);
        let sub = self.sub_mut();
        if sub.hw_rs.rasterizer_discard {
            gl.enable(GL_RASTERIZER_DISCARD);
        }
        let dsa = sub.dsa_state();
        if buffers & PIPE_CLEAR_DEPTH != 0 && !dsa.depth.writemask {
            gl.depth_mask(false);
        }
        if buffers & PIPE_CLEAR_STENCIL != 0 {
            gl.stencil_mask_separate(GL_FRONT, dsa.stencil[0].writemask as GLuint);
            gl.stencil_mask_separate(GL_BACK, dsa.stencil[1].writemask as GLuint);
        }
        if buffers & PIPE_CLEAR_COLOR != 0 {
            let mask = |m: u8| [m & 1 != 0, m & 2 != 0, m & 4 != 0, m & 8 != 0];
            if sub.hw_blend.independent && indep {
                for (i, m) in sub.hw_blend.colormask.iter().enumerate() {
                    gl.color_mask_i(i as GLuint, mask(*m));
                }
            } else {
                gl.color_mask(mask(sub.hw_blend.colormask[0]));
            }
        }
        gl.set_enabled(GL_SCISSOR_TEST, sub.hw_rs.scissor);
    }

    /// `vrend_clear`.
    fn clear(
        &mut self,
        host: &mut Host<'_>,
        buffers: u32,
        color: [u32; 4],
        depth: f64,
        stencil: u32,
    ) {
        let gl = host.gl;
        self.flush_lazy_state(host);
        if host.has(Feature::separate_shader_objects) {
            gl.bind_program_pipeline_none();
        }
        gl.use_program_none();
        gl.disable(GL_SCISSOR_TEST);
        let colorf = color.map(f32::from_bits);
        let surf = self.sub().cbufs.first().copied().flatten().map(|s| (s.resource, s.format));
        if let Err(e) = self.clear_prepare(host, surf, buffers, colorf, depth, stencil) {
            // A bound surface names a resource that is gone: the C clears on regardless, with
            // no conversion, and so does this.
            eprintln!("[virglrs] vrend: clear on a surface whose resource is gone: {e:?}");
        }
        let sub = self.sub();
        let mut bits: GLbitfield = 0;
        let mask: u32 =
            sub.cbufs.iter().enumerate().filter(|(_, s)| s.is_some()).map(|(i, _)| 1 << i).sum();
        if mask == buffers >> 2 {
            if buffers & PIPE_CLEAR_COLOR != 0 {
                bits |= GL_COLOR_BUFFER_BIT;
            }
        } else {
            // The C's pure-integer tests here are on a constant, so it always clears as float.
            for i in 0..sub.cbufs.len() {
                if buffers & (PIPE_CLEAR_COLOR0 << i) != 0 && sub.cbufs[i].is_some() {
                    gl.clear_buffer_fv(GL_COLOR, i as GLint, &colorf);
                }
            }
        }
        if buffers & PIPE_CLEAR_DEPTH != 0 {
            bits |= GL_DEPTH_BUFFER_BIT;
        }
        if buffers & PIPE_CLEAR_STENCIL != 0 {
            bits |= GL_STENCIL_BUFFER_BIT;
        }
        if bits != 0 {
            gl.clear(bits);
        }
        self.clear_finish(host, buffers);
    }

    /// `vrend_clear_texture`, the GLES leg: `glClearTexSubImageEXT`, with BGRA's bytes swapped.
    fn clear_texture(
        &mut self,
        host: &mut Host<'_>,
        resource: ResourceHandle,
        level: u32,
        region: Box3,
        data: [u32; 4],
    ) -> Result<(), Fault> {
        let cmd = Cmd::ClearTexture;
        let res = host.resource(cmd, resource)?;
        let Storage::Texture { name, .. } = res.storage else {
            return Err(Fault::IllegalResource { cmd, handle: resource });
        };
        let entry =
            res.entry(host.formats).ok_or(Fault::IllegalFormat { cmd, format: res.args.format })?;
        let mut bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        if res.is_bgra() {
            bytes.swap(0, 2);
        }
        if !host.has(Feature::clear_texture) {
            return Err(Fault::NoFeature { cmd, feature: Feature::clear_texture });
        }
        host.gl.clear_tex_sub_image(
            name,
            level as GLint,
            [region.x, region.y, region.z],
            [region.width, region.height, region.depth],
            entry.gl.glformat,
            entry.gl.gltype,
            &bytes,
        );
        Ok(())
    }
}

// ---- queries ----

impl Context {
    fn query(&self, cmd: Cmd, h: ObjectHandle) -> Result<&Query, Fault> {
        match self.sub().objects.get(&h) {
            Some(Object::Query(q)) => Ok(q),
            _ => Err(Fault::IllegalHandle { cmd, handle: h }),
        }
    }

    fn begin_query(&mut self, host: &mut Host<'_>, h: ObjectHandle) -> Result<(), Fault> {
        let q = self.query(Cmd::BeginQuery, h)?;
        if q.gl_type == GL_TIMESTAMP_EXT || q.gl_type == 0 {
            return Ok(());
        }
        host.gl.begin_query(q.gl_type, q.id);
        Ok(())
    }

    fn end_query(&mut self, host: &mut Host<'_>, h: ObjectHandle) -> Result<(), Fault> {
        let q = self.query(Cmd::EndQuery, h)?;
        if q.gl_type == GL_TIMESTAMP_EXT {
            host.todo.note("timestamp queries");
            return Ok(());
        }
        if q.gl_type != 0 {
            host.gl.end_query(q.gl_type);
        }
        Ok(())
    }

    /// `vrend_get_query_result`: when the result is ready, write `virgl_host_query_state`
    /// into the query's buffer. A result that is not ready is left for a later poll, which the
    /// guest makes by asking again.
    fn get_query_result(
        &mut self,
        host: &mut Host<'_>,
        h: ObjectHandle,
        _wait: bool,
    ) -> Result<(), Fault> {
        let cmd = Cmd::GetQueryResult;
        let (id, resource, fake) = {
            let q = self.query(cmd, h)?;
            (q.id, q.resource, q.fake_samples_passed)
        };
        let gl = host.gl;
        if gl.get_query_object_uiv(id, GL_QUERY_RESULT_AVAILABLE) == 0 {
            return Ok(());
        }
        let mut result = gl.get_query_object_uiv(id, GL_QUERY_RESULT) as u64;
        if fake {
            result *= 1024;
        }
        let mut state = [0u8; 16];
        state[0..4].copy_from_slice(&1u32.to_le_bytes()); // VIRGL_QUERY_STATE_DONE
        state[4..8].copy_from_slice(&4u32.to_le_bytes());
        state[8..16].copy_from_slice(&result.to_le_bytes());
        let ctx = host.ctx;
        let guest = host.guest;
        let res = host.resource_mut(cmd, resource)?;
        if let Storage::Host(buf) = &mut res.storage
            && buf.len() >= 16
        {
            buf[..16].copy_from_slice(&state);
        }
        if let Some(pages) = guest.pages(ctx, resource) {
            let _ = pages.copy_in(0, &state);
        }
        Ok(())
    }
}

// ---- transfers ----

impl Context {
    fn info(t: &Transfer, offset: u64, synchronized: bool) -> Info {
        Info {
            level: t.level,
            stride: t.stride,
            layer_stride: t.layer_stride,
            offset,
            region: t.region,
            synchronized,
        }
    }

    /// `vrend_renderer_transfer_iov` from the stream: the resource's own pages.
    fn transfer3d(
        &mut self,
        host: &mut Host<'_>,
        t: Transfer,
        offset: u32,
        direction: TransferDirection,
    ) -> Result<(), Fault> {
        let cmd = Cmd::Transfer3d;
        let guest = host.guest;
        let ctx = host.ctx;
        let formats = host.formats;
        let features = host.features;
        let gl = host.gl;
        let Some(pages) = guest.pages(ctx, t.resource) else {
            return Err(Fault::IllegalResource { cmd, handle: t.resource });
        };
        let res = host.resource_mut(cmd, t.resource)?;
        let info = Self::info(&t, offset as u64, false);
        let r = match direction {
            TransferDirection::ToHost => {
                transfer::write(gl, formats, res, Some(&pages), &pages, &info)
            }
            TransferDirection::FromHost => {
                transfer::read(gl, features, formats, res, Some(&pages), &pages, &info)
            }
        };
        r.map_err(|error| Fault::Transfer { cmd, error })
    }

    /// `vrend_renderer_copy_transfer3d` and `_from_host`: the staging resource's pages.
    fn copy_transfer3d(
        &mut self,
        host: &mut Host<'_>,
        direction: CopyDirection,
        t: Transfer,
        staging: ResourceHandle,
        staging_offset: u32,
        synchronized: bool,
    ) -> Result<(), Fault> {
        let cmd = Cmd::CopyTransfer3d;
        let guest = host.guest;
        let ctx = host.ctx;
        let formats = host.formats;
        let features = host.features;
        let gl = host.gl;
        let Some(staging_pages) = guest.pages(ctx, staging) else {
            return Err(Fault::IllegalResource { cmd, handle: staging });
        };
        let own = guest.pages(ctx, t.resource);
        let res = host.resource_mut(cmd, t.resource)?;
        let info = Self::info(&t, staging_offset as u64, synchronized);
        let r = match direction {
            CopyDirection::ToHost => {
                transfer::write(gl, formats, res, own.as_ref(), &staging_pages, &info)
            }
            CopyDirection::FromHost => {
                transfer::read(gl, features, formats, res, own.as_ref(), &staging_pages, &info)
            }
        };
        r.map_err(|error| Fault::Transfer { cmd, error })
    }

    /// `vrend_transfer_inline_write`: the bytes are in the command.
    fn inline_write(
        &mut self,
        host: &mut Host<'_>,
        t: Transfer,
        data: &[u32],
    ) -> Result<(), Fault> {
        let cmd = Cmd::ResourceInlineWrite;
        let guest = host.guest;
        let ctx = host.ctx;
        let formats = host.formats;
        let gl = host.gl;
        let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        let span = HostSpan::new(&bytes);
        let own = guest.pages(ctx, t.resource);
        let res = host.resource_mut(cmd, t.resource)?;
        let info = Self::info(&t, 0, false);
        transfer::write(gl, formats, res, own.as_ref(), &span.iov(), &info)
            .map_err(|error| Fault::Transfer { cmd, error })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(name: &str) -> Format {
        (0..FORMAT_MAX)
            .map(|n| Format::from_wire(n).unwrap())
            .find(|f| f.name() == name)
            .unwrap_or_else(|| panic!("no format {name}"))
    }

    fn element(name: &str) -> VertexElement {
        VertexElement {
            src_offset: 0,
            instance_divisor: 0,
            vertex_buffer_index: 0,
            src_format: format(name),
        }
    }

    #[test]
    fn a_vertex_element_takes_its_gl_type_from_the_first_channel() {
        let v = vertex_elements(&[
            element("R32G32B32_FLOAT"),
            element("R8G8B8A8_UNORM"),
            element("R16G16_SINT"),
            element("R10G10B10A2_UNORM"),
            element("R11G11B10_FLOAT"),
        ])
        .expect("every element has a type");
        let got: Vec<(GLenum, bool, GLint)> =
            v.elements.iter().map(|e| (e.gl_type, e.normalized, e.nr_channels)).collect();
        assert_eq!(
            got,
            [
                (GL_FLOAT, false, 3),
                (GL_UNSIGNED_BYTE, true, 4),
                (GL_SHORT, false, 2),
                (GL_UNSIGNED_INT_2_10_10_10_REV, true, 4),
                (GL_UNSIGNED_INT_10F_11F_11F_REV, false, 3),
            ]
        );
        assert!(v.elements[2].pure_integer);
        assert!(!v.elements[1].pure_integer);
    }

    #[test]
    fn a_vertex_format_with_no_gl_type_is_refused() {
        let r = vertex_elements(&[element("DXT1_RGB")]);
        assert!(matches!(r, Err(Fault::IllegalVertexFormat(_))));
    }

    #[test]
    fn a_shader_must_end_in_its_terminator() {
        assert!(read_shader(b"VERT\0\0\0\0", 2).is_ok());
        assert!(read_shader(b"VERT\0", 2).is_ok());
        assert!(matches!(
            read_shader(b"VERTEXSH", 2),
            Err(Fault::Shader { what: "text without a terminator", .. })
        ));
        assert!(matches!(read_shader(b"VE\0", 2), Err(Fault::Shader { .. })));
    }

    #[test]
    fn shader_text_the_tgsi_layer_refuses_is_a_tgsi_fault() {
        assert!(matches!(
            read_shader(b"PIXEL\n\0\0", 2),
            Err(Fault::Tgsi { error: tgsi::Refusal::Text(_), .. })
        ));
        // The program fits the guest's count plus the C's allowance of ten, and no more.
        let text = b"VERT\nDCL IN[0]\nDCL OUT[0], POSITION\n0: MOV OUT[0], IN[0]\n1: END\n\0";
        assert!(read_shader(text, 1).is_ok());
        assert!(read_shader(text, 0).is_err());
        assert!(matches!(
            read_shader(b"VERT\nDCL IN[80]\n0: END\n\0", 20),
            Err(Fault::Tgsi { error: tgsi::Refusal::Scan(_), .. })
        ));
    }

    #[test]
    fn a_destroyed_view_leaves_every_slot_it_held_and_marks_them() {
        let h = |n| ObjectHandle::new(n).unwrap();
        let mut views: [BTreeMap<u32, ObjectHandle>; ShaderStage::COUNT] = Default::default();
        views[0].insert(3, h(9));
        views[1].insert(0, h(9));
        views[1].insert(1, h(4));
        views[1].insert(7, h(9));
        let mut dirty = [Dirty::<MAX_SAMPLERS>::none(); ShaderStage::COUNT];
        assert!(evict_view(&mut views, &mut dirty, h(9)));
        assert!(views[0].is_empty());
        assert_eq!(views[1].keys().copied().collect::<Vec<_>>(), vec![1]);
        assert!(dirty[0].contains(3));
        assert!(dirty[1].contains(0) && dirty[1].contains(7) && !dirty[1].contains(1));
        assert!(dirty[2..].iter().all(|d| d.is_empty()));
        // A handle the guest reuses for a new view then binds afresh, instead of reading as
        // already bound.
        assert!(!evict_view(&mut views, &mut dirty, h(9)));
    }

    fn bound(handle: u32) -> BoundSurface {
        BoundSurface {
            handle: ObjectHandle::new(handle).unwrap(),
            resource: ResourceHandle::new(1).unwrap(),
            format: format("B8G8R8A8_UNORM"),
            level: 0,
            nr_samples: 0,
            tex_height: 16,
            y_0_top: false,
        }
    }

    #[test]
    fn a_destroyed_surface_leaves_every_attachment_it_held() {
        let h = ObjectHandle::new(9).unwrap();
        let mut zsurf = Some(bound(9));
        let mut cbufs = vec![Some(bound(4)), None, Some(bound(9)), Some(bound(9))];
        let (held_z, colours) = evict_surface(&mut zsurf, &mut cbufs, h);
        assert!(held_z && zsurf.is_none());
        assert_eq!(colours, vec![2, 3]);
        assert_eq!(cbufs, vec![Some(bound(4)), None, None, None]);
        // The handle the guest frees is reused by its next create, so a slot left holding it
        // would answer "already bound" to a different surface.
        let (held_z, colours) = evict_surface(&mut zsurf, &mut cbufs, h);
        assert!(!held_z && colours.is_empty());
    }

    #[test]
    fn the_todo_census_counts_by_shape() {
        let mut t = Todo::default();
        t.note("draws");
        t.note("draws");
        t.note("video");
        assert_eq!(t.by_frequency(), vec![("draws", 2), ("video", 1)]);
    }
}
