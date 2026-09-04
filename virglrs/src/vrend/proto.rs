// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The classic virgl protocol, typed.
//!
//! `src/virgl_protocol.h` is the wire: a stream of dwords, each command a header word
//! (`cmd | obj << 8 | len << 16`) followed by `len` payload dwords whose layout the header
//! spells out as index defines. This module is that header as Rust types: one [`Command`] variant
//! per `VIRGL_CCMD_*`, one struct per payload, every enumerated field a `pipe` enum, every
//! handle a newtype. [`decode`](super::decode) parses the wire into these and refuses what does
//! not fit; [`encode`](super::encode) writes them back, which is what makes a recorded stream a
//! differential test of the decoder without a C dump to diff against.
//!
//! What is *not* here: anything that needs the context to judge. A resource handle is checked to
//! be non-zero, not to exist; a shader continuation carries its offset, not whether a shader is
//! in progress. Those are the handler's, because only it holds the state.

use std::num::NonZeroU32;

use super::pipe::*;
use crate::ids::{BlobId, ResourceHandle};

use super::pipe::wire_enum;

wire_enum!(
    /// `virgl_context_cmd`. The wire's command byte.
    Cmd {
        Nop = 0,
        CreateObject = 1,
        BindObject = 2,
        DestroyObject = 3,
        SetViewportState = 4,
        SetFramebufferState = 5,
        SetVertexBuffers = 6,
        Clear = 7,
        DrawVbo = 8,
        ResourceInlineWrite = 9,
        SetSamplerViews = 10,
        SetIndexBuffer = 11,
        SetConstantBuffer = 12,
        SetStencilRef = 13,
        SetBlendColor = 14,
        SetScissorState = 15,
        Blit = 16,
        ResourceCopyRegion = 17,
        BindSamplerStates = 18,
        BeginQuery = 19,
        EndQuery = 20,
        GetQueryResult = 21,
        SetPolygonStipple = 22,
        SetClipState = 23,
        SetSampleMask = 24,
        SetStreamoutTargets = 25,
        SetRenderCondition = 26,
        SetUniformBuffer = 27,
        SetSubCtx = 28,
        CreateSubCtx = 29,
        DestroySubCtx = 30,
        BindShader = 31,
        SetTessState = 32,
        SetMinSamples = 33,
        SetShaderBuffers = 34,
        SetShaderImages = 35,
        MemoryBarrier = 36,
        LaunchGrid = 37,
        SetFramebufferStateNoAttach = 38,
        TextureBarrier = 39,
        SetAtomicBuffers = 40,
        SetDebugFlags = 41,
        GetQueryResultQbo = 42,
        Transfer3d = 43,
        EndTransfers = 44,
        CopyTransfer3d = 45,
        SetTweaks = 46,
        ClearTexture = 47,
        PipeResourceCreate = 48,
        PipeResourceSetType = 49,
        GetMemoryInfo = 50,
        SendStringMarker = 51,
        LinkShader = 52,
        CreateVideoCodec = 53,
        DestroyVideoCodec = 54,
        CreateVideoBuffer = 55,
        DestroyVideoBuffer = 56,
        BeginFrame = 57,
        DecodeMacroblock = 58,
        DecodeBitstream = 59,
        EncodeBitstream = 60,
        EndFrame = 61,
        ClearSurface = 62,
        GetPipeResourceLayout = 63,
    }
);

wire_enum!(
    /// `virgl_object_type`: the header's object byte on `CREATE_OBJECT`, `BIND_OBJECT` and
    /// `DESTROY_OBJECT`. `NULL` (0) names nothing that can be created.
    ObjectType {
        Blend = 1,
        Rasterizer = 2,
        Dsa = 3,
        Shader = 4,
        VertexElements = 5,
        SamplerView = 6,
        SamplerState = 7,
        Surface = 8,
        Query = 9,
        StreamoutTarget = 10,
        MsaaSurface = 11,
    }
);

/// `VIRGL_FORMAT_MAX`: the first integer that is not a format. The two `_EMULATED` formats past
/// it are the host's own and never on the wire.
pub const FORMAT_MAX: u32 = 482;

/// A `virgl_formats` value, in range. `NONE` (0) is a format: the wire uses it for "no format",
/// and where the C refuses it (a sampler view) the decoder refuses it too.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Format(u32);

impl Format {
    pub const NONE: Format = Format(0);

    pub fn from_wire(raw: u32) -> Option<Format> {
        (raw < FORMAT_MAX).then_some(Format(raw))
    }

    pub fn wire(self) -> u32 {
        self.0
    }

    /// A format from a generated table. The generator reads the wire numbering from the same
    /// header `FORMAT_MAX` is copied from, and the assert runs at compile time in a static
    /// initialiser, so a table naming a number the wire does not have fails the build.
    pub(super) const fn table(raw: u32) -> Format {
        assert!(raw < FORMAT_MAX, "a generated format table names a number past FORMAT_MAX");
        Format(raw)
    }
}

/// A handle in a context's object table: a blend state, a shader, a surface. Guest-chosen,
/// scoped to the sub-context that created it, and reused. Never zero: on the wire zero means
/// "no object", and every field that allows that is an `Option`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct ObjectHandle(NonZeroU32);

impl ObjectHandle {
    pub const fn new(raw: u32) -> Option<ObjectHandle> {
        match NonZeroU32::new(raw) {
            Some(n) => Some(ObjectHandle(n)),
            None => None,
        }
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl std::fmt::Display for ObjectHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A sub-context id. Each context starts with sub-context 0; the guest creates more and switches
/// between them, and every object table and every piece of bound state is per sub-context.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct SubContextId(pub u32);

/// A video codec's handle, in the video context's own namespace.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct VideoCodecHandle(pub u32);

/// A video buffer's handle, in the video context's own namespace.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct VideoBufferHandle(pub u32);

/// `pipe_box`: a region of a resource. Signed because a blit flips by sending a negative extent.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Box3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub width: i32,
    pub height: i32,
    pub depth: i32,
}

/// The eleven dwords every transfer-shaped command opens with (`VIRGL_RESOURCE_IW_*`): which
/// resource, and where in it. `usage` is carried for the wire's sake; nothing reads it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Transfer {
    pub resource: ResourceHandle,
    pub level: u32,
    pub usage: u32,
    pub stride: u32,
    pub layer_stride: u32,
    pub region: Box3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlendEq {
    pub func: BlendFunc,
    pub src: BlendFactor,
    pub dst: BlendFactor,
}

/// One render target's blend state. The equation exists only when blending is on: a disabled
/// target's factor bits are zero on the wire, and zero is not a factor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RtBlend {
    pub equation: Option<RtBlendEq>,
    pub colormask: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RtBlendEq {
    pub rgb: BlendEq,
    pub alpha: BlendEq,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlendState {
    pub independent_blend_enable: bool,
    pub logicop_enable: bool,
    pub dither: bool,
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
    pub logicop_func: LogicOp,
    pub rt: [RtBlend; 8],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DepthState {
    pub enabled: bool,
    pub writemask: bool,
    pub func: CompareFunc,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AlphaState {
    pub enabled: bool,
    pub func: CompareFunc,
    pub ref_value: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StencilFace {
    pub enabled: bool,
    pub func: CompareFunc,
    pub fail_op: StencilOp,
    pub zpass_op: StencilOp,
    pub zfail_op: StencilOp,
    pub valuemask: u8,
    pub writemask: u8,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DepthStencilAlpha {
    pub depth: DepthState,
    pub alpha: AlphaState,
    /// Front, then back.
    pub stencil: [StencilFace; 2],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RasterizerState {
    pub flatshade: bool,
    pub depth_clip: bool,
    pub clip_halfz: bool,
    pub rasterizer_discard: bool,
    pub flatshade_first: bool,
    pub light_twoside: bool,
    pub sprite_coord_mode: bool,
    pub point_quad_rasterization: bool,
    pub cull_face: CullFace,
    pub fill_front: FillMode,
    pub fill_back: FillMode,
    pub scissor: bool,
    pub front_ccw: bool,
    pub clamp_vertex_color: bool,
    pub clamp_fragment_color: bool,
    pub offset_line: bool,
    pub offset_point: bool,
    pub offset_tri: bool,
    pub poly_smooth: bool,
    pub poly_stipple_enable: bool,
    pub point_smooth: bool,
    pub point_size_per_vertex: bool,
    pub multisample: bool,
    pub line_smooth: bool,
    pub line_stipple_enable: bool,
    pub line_last_pixel: bool,
    pub half_pixel_center: bool,
    pub bottom_edge_rule: bool,
    pub force_persample_interp: bool,
    pub point_size: f32,
    pub sprite_coord_enable: u32,
    pub line_stipple_pattern: u16,
    pub line_stipple_factor: u8,
    pub clip_plane_enable: u8,
    pub line_width: f32,
    pub offset_units: f32,
    pub offset_scale: f32,
    pub offset_clamp: f32,
}

/// The transform-feedback buffer an output lands in: the index into `StreamOutput::stride`.
/// The wire gives it three bits and the table has four entries, so it is parsed, not cast.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SoBuffer(u8);

impl SoBuffer {
    pub const COUNT: usize = 4;

    pub fn from_wire(raw: u32) -> Option<SoBuffer> {
        (raw < Self::COUNT as u32).then_some(SoBuffer(raw as u8))
    }

    pub fn wire(self) -> u32 {
        u32::from(self.0)
    }

    pub fn index(self) -> usize {
        usize::from(self.0)
    }
}

/// One transform-feedback output, as `pipe_stream_output`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SoOutput {
    pub register_index: u8,
    pub start_component: u8,
    pub num_components: u8,
    pub output_buffer: SoBuffer,
    pub dst_offset: u16,
    pub stream: u8,
}

/// `pipe_stream_output_info`. Present in a graphics shader's header only when it has outputs:
/// with none, the strides are not on the wire either.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct StreamOutput {
    pub stride: [u32; SoBuffer::COUNT],
    pub outputs: Vec<SoOutput>,
}

/// Where a `CREATE_OBJECT` shader's text sits in the whole. A shader longer than one command
/// arrives in pieces: the first declares the total and each continuation its offset into it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShaderChunk {
    New { total_bytes: u32 },
    Continuation { offset: u32 },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ShaderKind {
    Graphics { stream_output: StreamOutput },
    Compute { req_local_mem: u32 },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShaderCreate<'a> {
    pub stage: ShaderStage,
    pub chunk: ShaderChunk,
    pub num_tokens: u32,
    pub kind: ShaderKind,
    /// This command's piece of the TGSI text, as sent: dword-padded, NUL somewhere in the last
    /// piece.
    pub text: &'a [u32],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VertexElement {
    pub src_offset: u32,
    pub instance_divisor: u32,
    pub vertex_buffer_index: u32,
    pub src_format: Format,
}

/// A sampler view. Dwords 4 and 5 mean different things by the resource's target -- first and
/// last element of a buffer, packed layer and level ranges of a texture -- and the decoder cannot
/// see the resource, so they are carried as the wire has them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SamplerView {
    pub resource: ResourceHandle,
    pub format: Format,
    pub target: TextureTarget,
    pub first_element_or_layers: u32,
    pub last_element_or_levels: u32,
    pub swizzle: [Swizzle; 4],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SamplerState {
    pub wrap_s: TexWrap,
    pub wrap_t: TexWrap,
    pub wrap_r: TexWrap,
    pub min_img_filter: TexFilter,
    pub min_mip_filter: MipFilter,
    pub mag_img_filter: TexFilter,
    pub compare_mode: bool,
    pub compare_func: CompareFunc,
    pub seamless_cube_map: bool,
    pub max_anisotropy: u8,
    pub lod_bias: f32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: [u32; 4],
}

/// A surface, plain or multisampled (`samples` is 0 for a plain one). As with a sampler view,
/// dwords 4 and 5 are the buffer's element range or the texture's level and packed layer range.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Surface {
    pub resource: ResourceHandle,
    pub format: Format,
    pub first_element_or_level: u32,
    pub last_element_or_layers: u32,
    pub samples: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QueryCreate {
    pub kind: QueryType,
    pub index: u16,
    pub offset: u32,
    pub resource: ResourceHandle,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StreamoutTarget {
    pub resource: ResourceHandle,
    pub buffer_offset: u32,
    pub buffer_size: u32,
}

/// What a `CREATE_OBJECT` creates, by the header's object byte.
#[derive(Clone, PartialEq, Debug)]
pub enum Object<'a> {
    Blend(BlendState),
    Rasterizer(RasterizerState),
    Dsa(DepthStencilAlpha),
    Shader(ShaderCreate<'a>),
    VertexElements(Vec<VertexElement>),
    SamplerView(SamplerView),
    SamplerState(SamplerState),
    Surface(Surface),
    Query(QueryCreate),
    StreamoutTarget(StreamoutTarget),
}

impl Object<'_> {
    pub fn kind(&self) -> ObjectType {
        match self {
            Object::Blend(_) => ObjectType::Blend,
            Object::Rasterizer(_) => ObjectType::Rasterizer,
            Object::Dsa(_) => ObjectType::Dsa,
            Object::Shader(_) => ObjectType::Shader,
            Object::VertexElements(_) => ObjectType::VertexElements,
            Object::SamplerView(_) => ObjectType::SamplerView,
            Object::SamplerState(_) => ObjectType::SamplerState,
            Object::Surface(s) if s.samples != 0 => ObjectType::MsaaSurface,
            Object::Surface(_) => ObjectType::Surface,
            Object::Query(_) => ObjectType::Query,
            Object::StreamoutTarget(_) => ObjectType::StreamoutTarget,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Viewport {
    pub scale: [f32; 3],
    pub translate: [f32; 3],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Scissor {
    pub minx: u16,
    pub miny: u16,
    pub maxx: u16,
    pub maxy: u16,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VertexBuffer {
    pub stride: u32,
    pub offset: u32,
    pub resource: Option<ResourceHandle>,
}

/// The width of one index, as the wire carries it: its size in bytes. Three widths exist; the
/// C draws with `GL_UNSIGNED_INT` for any other value while sizing its bounds check by the
/// value as sent, so a width of 3 passed the check for a draw that read a third more.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IndexType {
    U8,
    U16,
    U32,
}

impl IndexType {
    pub fn from_wire(raw: u32) -> Option<IndexType> {
        match raw {
            1 => Some(IndexType::U8),
            2 => Some(IndexType::U16),
            4 => Some(IndexType::U32),
            _ => None,
        }
    }

    pub fn bytes(self) -> u32 {
        match self {
            IndexType::U8 => 1,
            IndexType::U16 => 2,
            IndexType::U32 => 4,
        }
    }

    pub fn wire(self) -> u32 {
        self.bytes()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndexBuffer {
    pub resource: ResourceHandle,
    pub index_type: IndexType,
    pub offset: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TessDraw {
    pub vertices_per_patch: u32,
    pub drawid: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndirectDraw {
    pub resource: ResourceHandle,
    pub offset: u32,
    pub stride: u32,
    pub draw_count: u32,
    pub draw_count_offset: u32,
    pub draw_count_resource: Option<ResourceHandle>,
}

/// `pipe_draw_info`, in the three sizes the wire allows: plain, with tessellation, and with both
/// tessellation and an indirect buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Draw {
    pub start: u32,
    pub count: u32,
    pub mode: PrimType,
    pub indexed: bool,
    pub instance_count: u32,
    pub index_bias: i32,
    pub start_instance: u32,
    pub primitive_restart: bool,
    pub restart_index: u32,
    pub min_index: u32,
    pub max_index: u32,
    pub count_from_so: Option<ObjectHandle>,
    pub tess: Option<TessDraw>,
    pub indirect: Option<IndirectDraw>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlitTarget {
    pub resource: ResourceHandle,
    pub level: u32,
    pub format: Format,
    pub region: Box3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Blit {
    pub mask: u8,
    pub filter: TexFilter,
    pub scissor_enable: bool,
    pub render_condition_enable: bool,
    pub alpha_blend: bool,
    pub scissor: Scissor,
    pub dst: BlitTarget,
    pub src: BlitTarget,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ShaderBuffer {
    pub offset: u32,
    pub length: u32,
    pub resource: Option<ResourceHandle>,
}

/// A bound shader image. An entry that unbinds its slot is `None` on the wire as five zero
/// dwords, and only a bound one has an access to parse.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ShaderImage {
    pub format: Format,
    pub access: ImageAccess,
    pub layer_offset: u32,
    pub level_size: u32,
    pub resource: ResourceHandle,
}

/// Which way a `COPY_TRANSFER3D` moves bytes between a resource and a staging buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CopyDirection {
    ToHost,
    FromHost,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plane {
    pub stride: u32,
    pub offset: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VideoCodec {
    pub handle: VideoCodecHandle,
    pub profile: u32,
    pub entrypoint: u32,
    pub chroma_format: u32,
    pub level: u32,
    pub width: u32,
    pub height: u32,
    /// Absent on the older, seven-dword form; the C then assumes two.
    pub max_references: Option<u32>,
}

/// One decoded command. Borrows the wire for the bulk payloads -- shader text, inline-write
/// bytes, constants -- and owns everything else.
#[derive(Clone, PartialEq, Debug)]
pub enum Command<'a> {
    Nop,
    CreateObject {
        handle: ObjectHandle,
        object: Object<'a>,
    },
    BindObject {
        kind: ObjectType,
        handle: Option<ObjectHandle>,
    },
    DestroyObject {
        kind: ObjectType,
        handle: ObjectHandle,
    },
    SetViewportState {
        start_slot: u32,
        viewports: Vec<Viewport>,
    },
    SetFramebufferState {
        zsurf: Option<ObjectHandle>,
        cbufs: Vec<Option<ObjectHandle>>,
    },
    SetVertexBuffers(Vec<VertexBuffer>),
    Clear {
        buffers: u32,
        color: [u32; 4],
        depth: f64,
        stencil: u32,
    },
    DrawVbo(Draw),
    ResourceInlineWrite {
        transfer: Transfer,
        data: &'a [u32],
    },
    SetSamplerViews {
        stage: ShaderStage,
        start_slot: u32,
        views: Vec<Option<ObjectHandle>>,
    },
    SetIndexBuffer(Option<IndexBuffer>),
    SetConstantBuffer {
        stage: ShaderStage,
        index: u32,
        data: &'a [u32],
    },
    SetStencilRef {
        front: u8,
        back: u8,
    },
    SetBlendColor([f32; 4]),
    SetScissorState {
        start_slot: u32,
        scissors: Vec<Scissor>,
    },
    Blit(Blit),
    ResourceCopyRegion {
        dst: ResourceHandle,
        dst_level: u32,
        dst_x: u32,
        dst_y: u32,
        dst_z: u32,
        src: ResourceHandle,
        src_level: u32,
        src_region: Box3,
    },
    BindSamplerStates {
        stage: ShaderStage,
        start_slot: u32,
        states: Vec<Option<ObjectHandle>>,
    },
    BeginQuery(ObjectHandle),
    EndQuery(ObjectHandle),
    GetQueryResult {
        query: ObjectHandle,
        wait: bool,
    },
    SetPolygonStipple([u32; 32]),
    SetClipState([[f32; 4]; 8]),
    SetSampleMask(u32),
    SetStreamoutTargets {
        append_bitmask: u32,
        targets: Vec<Option<ObjectHandle>>,
    },
    SetRenderCondition {
        query: Option<ObjectHandle>,
        condition: bool,
        mode: RenderCondMode,
    },
    SetUniformBuffer {
        stage: ShaderStage,
        index: u32,
        offset: u32,
        length: u32,
        resource: Option<ResourceHandle>,
    },
    SetSubCtx(SubContextId),
    CreateSubCtx(SubContextId),
    DestroySubCtx(SubContextId),
    BindShader {
        handle: Option<ObjectHandle>,
        stage: ShaderStage,
    },
    SetTessState([f32; 6]),
    SetMinSamples(u32),
    SetShaderBuffers {
        stage: ShaderStage,
        start_slot: u32,
        buffers: Vec<ShaderBuffer>,
    },
    SetShaderImages {
        stage: ShaderStage,
        start_slot: u32,
        images: Vec<Option<ShaderImage>>,
    },
    MemoryBarrier(u32),
    LaunchGrid {
        block: [u32; 3],
        grid: [u32; 3],
        indirect: Option<ResourceHandle>,
        indirect_offset: u32,
    },
    SetFramebufferStateNoAttach {
        width: u16,
        height: u16,
        layers: u16,
        samples: u8,
    },
    TextureBarrier(u32),
    SetAtomicBuffers {
        start_slot: u32,
        buffers: Vec<ShaderBuffer>,
    },
    SetDebugFlags(&'a [u32]),
    GetQueryResultQbo {
        query: ObjectHandle,
        buffer: ResourceHandle,
        wait: bool,
        result_type: QueryValueType,
        offset: u32,
        index: i32,
    },
    Transfer3d {
        transfer: Transfer,
        offset: u32,
        direction: TransferDirection,
    },
    /// Ends the transfer prologue of a batch. Mesa writes it with a length that pads the
    /// prologue out to its fixed size, so the payload is slack, carried as sent and never read.
    EndTransfers(&'a [u32]),
    CopyTransfer3d {
        direction: CopyDirection,
        transfer: Transfer,
        staging: ResourceHandle,
        staging_offset: u32,
        synchronized: bool,
    },
    SetTweaks {
        id: u32,
        value: u32,
    },
    ClearTexture {
        resource: ResourceHandle,
        level: u32,
        region: Box3,
        data: [u32; 4],
    },
    PipeResourceCreate {
        target: TextureTarget,
        format: Format,
        bind: u32,
        width: u32,
        height: u32,
        depth: u32,
        array_size: u32,
        last_level: u32,
        nr_samples: u32,
        flags: u32,
        blob_id: BlobId,
    },
    PipeResourceSetType {
        resource: ResourceHandle,
        format: Format,
        bind: u32,
        width: u32,
        height: u32,
        usage: u32,
        modifier: u64,
        planes: Vec<Plane>,
    },
    GetMemoryInfo(ResourceHandle),
    SendStringMarker {
        len: u32,
        text: &'a [u32],
    },
    LinkShader([Option<ObjectHandle>; ShaderStage::COUNT]),
    CreateVideoCodec(VideoCodec),
    DestroyVideoCodec(VideoCodecHandle),
    CreateVideoBuffer {
        handle: VideoBufferHandle,
        format: u32,
        width: u32,
        height: u32,
        planes: Vec<ResourceHandle>,
    },
    DestroyVideoBuffer(VideoBufferHandle),
    BeginFrame {
        codec: VideoCodecHandle,
        target: VideoBufferHandle,
    },
    /// The C decodes nothing of it and does nothing with it; the payload is kept as sent.
    DecodeMacroblock(&'a [u32]),
    DecodeBitstream {
        codec: VideoCodecHandle,
        target: VideoBufferHandle,
        descriptor: ResourceHandle,
        buffer: ResourceHandle,
        buffer_size: u32,
    },
    EncodeBitstream {
        codec: VideoCodecHandle,
        source: VideoBufferHandle,
        destination: ResourceHandle,
        descriptor: ResourceHandle,
        feedback: ResourceHandle,
    },
    EndFrame {
        codec: VideoCodecHandle,
        target: VideoBufferHandle,
    },
    ClearSurface {
        render_condition_enable: bool,
        buffers: u8,
        surface: ObjectHandle,
        color: [u32; 4],
        dst_x: u32,
        dst_y: u32,
        width: u32,
        height: u32,
    },
    GetPipeResourceLayout {
        out: ResourceHandle,
        target: ResourceHandle,
    },
}

impl Command<'_> {
    pub fn kind(&self) -> Cmd {
        match self {
            Command::Nop => Cmd::Nop,
            Command::CreateObject { .. } => Cmd::CreateObject,
            Command::BindObject { .. } => Cmd::BindObject,
            Command::DestroyObject { .. } => Cmd::DestroyObject,
            Command::SetViewportState { .. } => Cmd::SetViewportState,
            Command::SetFramebufferState { .. } => Cmd::SetFramebufferState,
            Command::SetVertexBuffers(_) => Cmd::SetVertexBuffers,
            Command::Clear { .. } => Cmd::Clear,
            Command::DrawVbo(_) => Cmd::DrawVbo,
            Command::ResourceInlineWrite { .. } => Cmd::ResourceInlineWrite,
            Command::SetSamplerViews { .. } => Cmd::SetSamplerViews,
            Command::SetIndexBuffer(_) => Cmd::SetIndexBuffer,
            Command::SetConstantBuffer { .. } => Cmd::SetConstantBuffer,
            Command::SetStencilRef { .. } => Cmd::SetStencilRef,
            Command::SetBlendColor(_) => Cmd::SetBlendColor,
            Command::SetScissorState { .. } => Cmd::SetScissorState,
            Command::Blit(_) => Cmd::Blit,
            Command::ResourceCopyRegion { .. } => Cmd::ResourceCopyRegion,
            Command::BindSamplerStates { .. } => Cmd::BindSamplerStates,
            Command::BeginQuery(_) => Cmd::BeginQuery,
            Command::EndQuery(_) => Cmd::EndQuery,
            Command::GetQueryResult { .. } => Cmd::GetQueryResult,
            Command::SetPolygonStipple(_) => Cmd::SetPolygonStipple,
            Command::SetClipState(_) => Cmd::SetClipState,
            Command::SetSampleMask(_) => Cmd::SetSampleMask,
            Command::SetStreamoutTargets { .. } => Cmd::SetStreamoutTargets,
            Command::SetRenderCondition { .. } => Cmd::SetRenderCondition,
            Command::SetUniformBuffer { .. } => Cmd::SetUniformBuffer,
            Command::SetSubCtx(_) => Cmd::SetSubCtx,
            Command::CreateSubCtx(_) => Cmd::CreateSubCtx,
            Command::DestroySubCtx(_) => Cmd::DestroySubCtx,
            Command::BindShader { .. } => Cmd::BindShader,
            Command::SetTessState(_) => Cmd::SetTessState,
            Command::SetMinSamples(_) => Cmd::SetMinSamples,
            Command::SetShaderBuffers { .. } => Cmd::SetShaderBuffers,
            Command::SetShaderImages { .. } => Cmd::SetShaderImages,
            Command::MemoryBarrier(_) => Cmd::MemoryBarrier,
            Command::LaunchGrid { .. } => Cmd::LaunchGrid,
            Command::SetFramebufferStateNoAttach { .. } => Cmd::SetFramebufferStateNoAttach,
            Command::TextureBarrier(_) => Cmd::TextureBarrier,
            Command::SetAtomicBuffers { .. } => Cmd::SetAtomicBuffers,
            Command::SetDebugFlags(_) => Cmd::SetDebugFlags,
            Command::GetQueryResultQbo { .. } => Cmd::GetQueryResultQbo,
            Command::Transfer3d { .. } => Cmd::Transfer3d,
            Command::EndTransfers(_) => Cmd::EndTransfers,
            Command::CopyTransfer3d { .. } => Cmd::CopyTransfer3d,
            Command::SetTweaks { .. } => Cmd::SetTweaks,
            Command::ClearTexture { .. } => Cmd::ClearTexture,
            Command::PipeResourceCreate { .. } => Cmd::PipeResourceCreate,
            Command::PipeResourceSetType { .. } => Cmd::PipeResourceSetType,
            Command::GetMemoryInfo(_) => Cmd::GetMemoryInfo,
            Command::SendStringMarker { .. } => Cmd::SendStringMarker,
            Command::LinkShader(_) => Cmd::LinkShader,
            Command::CreateVideoCodec(_) => Cmd::CreateVideoCodec,
            Command::DestroyVideoCodec(_) => Cmd::DestroyVideoCodec,
            Command::CreateVideoBuffer { .. } => Cmd::CreateVideoBuffer,
            Command::DestroyVideoBuffer(_) => Cmd::DestroyVideoBuffer,
            Command::BeginFrame { .. } => Cmd::BeginFrame,
            Command::DecodeMacroblock(_) => Cmd::DecodeMacroblock,
            Command::DecodeBitstream { .. } => Cmd::DecodeBitstream,
            Command::EncodeBitstream { .. } => Cmd::EncodeBitstream,
            Command::EndFrame { .. } => Cmd::EndFrame,
            Command::ClearSurface { .. } => Cmd::ClearSurface,
            Command::GetPipeResourceLayout { .. } => Cmd::GetPipeResourceLayout,
        }
    }
}

/// Why the wire was refused. Every variant names the command, so the report a guest gets (and
/// the sabotage sweep reads) says which one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refused {
    /// A command byte the protocol does not have. Nothing after it can be framed.
    UnknownCommand { at: usize, cmd: u32 },
    /// A command whose declared length runs past the end of the batch.
    Overrun { at: usize, cmd: Cmd, len: usize, left: usize },
    /// A length the command does not allow.
    Length { cmd: Cmd, len: usize },
    /// A field whose value is outside what it may hold.
    Field { cmd: Cmd, field: &'static str, value: u32 },
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refused::UnknownCommand { at, cmd } => {
                write!(f, "dword {at}: command {cmd} is not in the protocol")
            }
            Refused::Overrun { at, cmd, len, left } => write!(
                f,
                "dword {at}: {} declares {len} dwords with {left} left in the batch",
                cmd.name()
            ),
            Refused::Length { cmd, len } => {
                write!(f, "{} does not come in {len} dwords", cmd.name())
            }
            Refused::Field { cmd, field, value } => {
                write!(f, "{}: {field} cannot be {value:#x}", cmd.name())
            }
        }
    }
}

impl std::error::Error for Refused {}
