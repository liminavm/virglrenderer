// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! TGSI, the shader language of the classic wire: a guest sends its shaders as TGSI *text*, and
//! this module is what the text becomes -- the typed program the GLSL translator walks.
//!
//! The C keeps a program as an array of packed 32-bit tokens (`p_shader_tokens.h`), built by
//! `tgsi_text.c` and re-parsed by every consumer through `tgsi_parse.c`. Nothing here is packed:
//! a [`Shader`] is a list of typed [`Token`]s, filled by the text parser ([`text`]) and read
//! directly. Where the C's bitfields would truncate a value -- a 16-bit register index, a 10-bit
//! array id, a 24-bit label -- the parser masks to the same width, so a program that would have
//! meant one thing to the C means the same thing here.
//!
//! What the C runs after the text parse, `tgsi_sanity_check`, is not ported: it counts an error
//! only when `TGSI_PRINT_SANITY` is set in the environment, so in the C as shipped it refuses
//! nothing. The guest-facing refusals that do exist are the parser's own and the scan's
//! ([`scan`]), which is where the C's `tgsi_scan_shader` says no.

pub mod dump;
pub(crate) mod fixture;
pub mod info;
pub mod scan;
pub mod text;

pub use info::{Opcode, OutputMode};

/// Why a guest's shader text was refused: the parser said no, or the scan did.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Refusal {
    Text(text::Error),
    Scan(scan::Refusal),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Text(e) => write!(f, "{e}"),
            Refusal::Scan(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Refusal {}

/// A program read from a guest's text and scanned: what the translator takes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Program {
    pub shader: Shader,
    pub info: scan::Info,
}

impl Program {
    /// `tgsi_text_translate` with the C's allowance: the guest sizes its program at `num_tokens`
    /// packed tokens and the C parses into ten more than that.
    pub fn parse(text: &[u8], num_tokens: u32) -> Result<Shader, Refusal> {
        text::parse(text, num_tokens.saturating_add(10)).map_err(Refusal::Text)
    }

    /// `tgsi_scan_shader`, the translator's first step.
    pub fn scan(shader: Shader) -> Result<Program, Refusal> {
        let info = scan::scan(&shader).map_err(Refusal::Scan)?;
        Ok(Program { shader, info })
    }
}

/// A macro for the named enumerations the wire spells out in text: each gets its C ordinal, its
/// spelling, and a lookup by ordinal for the places the C indexes a name table.
macro_rules! named {
    ($(#[$m:meta])* $name:ident { $($variant:ident = $s:literal),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        #[repr(u8)]
        pub enum $name { $($variant),* }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant),*];

            /// The spelling the text uses.
            pub fn name(self) -> &'static str {
                match self { $($name::$variant => $s),* }
            }

            /// The variant with this C ordinal.
            pub fn from_index(i: usize) -> Option<$name> {
                Self::ALL.get(i).copied()
            }
        }
    };
}

named! {
    /// `tgsi_processor_type`.
    Processor {
        Fragment = "FRAG",
        Vertex = "VERT",
        Geometry = "GEOM",
        TessCtrl = "TESS_CTRL",
        TessEval = "TESS_EVAL",
        Compute = "COMP",
    }
}

named! {
    /// `tgsi_file_type`: the register files.
    File {
        Null = "NULL",
        Constant = "CONST",
        Input = "IN",
        Output = "OUT",
        Temporary = "TEMP",
        Sampler = "SAMP",
        Address = "ADDR",
        Immediate = "IMM",
        Predicate = "PRED",
        SystemValue = "SV",
        Image = "IMAGE",
        SamplerView = "SVIEW",
        Buffer = "BUFFER",
        Memory = "MEMORY",
        HwAtomic = "HWATOMIC",
    }
}

named! {
    /// `tgsi_semantic`.
    Semantic {
        Position = "POSITION",
        Color = "COLOR",
        BColor = "BCOLOR",
        Fog = "FOG",
        PSize = "PSIZE",
        Generic = "GENERIC",
        Normal = "NORMAL",
        Face = "FACE",
        EdgeFlag = "EDGEFLAG",
        PrimId = "PRIM_ID",
        InstanceId = "INSTANCEID",
        VertexId = "VERTEXID",
        Stencil = "STENCIL",
        ClipDist = "CLIPDIST",
        ClipVertex = "CLIPVERTEX",
        GridSize = "GRID_SIZE",
        BlockId = "BLOCK_ID",
        BlockSize = "BLOCK_SIZE",
        ThreadId = "THREAD_ID",
        TexCoord = "TEXCOORD",
        PCoord = "PCOORD",
        ViewportIndex = "VIEWPORT_INDEX",
        Layer = "LAYER",
        CullDist = "CULLDIST",
        SampleId = "SAMPLEID",
        SamplePos = "SAMPLEPOS",
        SampleMask = "SAMPLEMASK",
        InvocationId = "INVOCATIONID",
        VertexIdNoBase = "VERTEXID_NOBASE",
        BaseVertex = "BASEVERTEX",
        Patch = "PATCH",
        TessCoord = "TESSCOORD",
        TessOuter = "TESSOUTER",
        TessInner = "TESSINNER",
        VerticesIn = "VERTICESIN",
        HelperInvocation = "HELPER_INVOCATION",
        BaseInstance = "BASEINSTANCE",
        DrawId = "DRAWID",
        WorkDim = "WORK_DIM",
        SubgroupSize = "SUBGROUP_SIZE",
        SubgroupInvocation = "SUBGROUP_INVOCATION",
        SubgroupEqMask = "SUBGROUP_EQ_MASK",
        SubgroupGeMask = "SUBGROUP_GE_MASK",
        SubgroupGtMask = "SUBGROUP_GT_MASK",
        SubgroupLeMask = "SUBGROUP_LE_MASK",
        SubgroupLtMask = "SUBGROUP_LT_MASK",
        CsUserDataAmd = "CS_USER_DATA_AMD",
        ViewportMask = "VIEWPORT_MASK",
    }
}

named! {
    /// `tgsi_texture_type`.
    Texture {
        Buffer = "BUFFER",
        D1 = "1D",
        D2 = "2D",
        D3 = "3D",
        Cube = "CUBE",
        Rect = "RECT",
        Shadow1d = "SHADOW1D",
        Shadow2d = "SHADOW2D",
        ShadowRect = "SHADOWRECT",
        Array1d = "1D_ARRAY",
        Array2d = "2D_ARRAY",
        Shadow1dArray = "SHADOW1D_ARRAY",
        Shadow2dArray = "SHADOW2D_ARRAY",
        ShadowCube = "SHADOWCUBE",
        Msaa2d = "2D_MSAA",
        Msaa2dArray = "2D_ARRAY_MSAA",
        CubeArray = "CUBEARRAY",
        ShadowCubeArray = "SHADOWCUBEARRAY",
        Unknown = "UNKNOWN",
    }
}

named! {
    /// `tgsi_property_name`.
    Property {
        GsInputPrim = "GS_INPUT_PRIMITIVE",
        GsOutputPrim = "GS_OUTPUT_PRIMITIVE",
        GsMaxOutputVertices = "GS_MAX_OUTPUT_VERTICES",
        FsCoordOrigin = "FS_COORD_ORIGIN",
        FsCoordPixelCenter = "FS_COORD_PIXEL_CENTER",
        FsColor0WritesAllCbufs = "FS_COLOR0_WRITES_ALL_CBUFS",
        FsDepthLayout = "FS_DEPTH_LAYOUT",
        VsProhibitUcps = "VS_PROHIBIT_UCPS",
        GsInvocations = "GS_INVOCATIONS",
        VsWindowSpacePosition = "VS_WINDOW_SPACE_POSITION",
        TcsVerticesOut = "TCS_VERTICES_OUT",
        TesPrimMode = "TES_PRIM_MODE",
        TesSpacing = "TES_SPACING",
        TesVertexOrderCw = "TES_VERTEX_ORDER_CW",
        TesPointMode = "TES_POINT_MODE",
        NumClipdistEnabled = "NUM_CLIPDIST_ENABLED",
        NumCulldistEnabled = "NUM_CULLDIST_ENABLED",
        FsEarlyDepthStencil = "FS_EARLY_DEPTH_STENCIL",
        FsPostDepthCoverage = "FS_POST_DEPTH_COVERAGE",
        NextShader = "NEXT_SHADER",
        CsFixedBlockWidth = "CS_FIXED_BLOCK_WIDTH",
        CsFixedBlockHeight = "CS_FIXED_BLOCK_HEIGHT",
        CsFixedBlockDepth = "CS_FIXED_BLOCK_DEPTH",
        MulZeroWins = "MUL_ZERO_WINS",
        VsBlitSgprsAmd = "VS_BLIT_SGPRS_AMD",
        CsUserDataComponentsAmd = "CS_USER_DATA_COMPONENTS_AMD",
        LayerViewportRelative = "LAYER_VIEWPORT_RELATIVE",
        FsBlendEquationAdvanced = "FS_BLEND_EQUATION_ADVANCED",
        SeparableProgram = "SEPARABLE_PROGRAM",
    }
}

named! {
    /// `tgsi_return_type`, a sampler view's per-channel type.
    ReturnType {
        Unorm = "UNORM",
        Snorm = "SNORM",
        Sint = "SINT",
        Uint = "UINT",
        Float = "FLOAT",
    }
}

named! {
    /// `tgsi_interpolate_mode`.
    Interpolate {
        Constant = "CONSTANT",
        Linear = "LINEAR",
        Perspective = "PERSPECTIVE",
        Color = "COLOR",
    }
}

named! {
    /// `tgsi_interpolate_loc`.
    Location {
        Center = "CENTER",
        Centroid = "CENTROID",
        Sample = "SAMPLE",
    }
}

named! {
    /// `pipe_prim_type`, as a geometry shader's properties name it.
    Primitive {
        Points = "POINTS",
        Lines = "LINES",
        LineLoop = "LINE_LOOP",
        LineStrip = "LINE_STRIP",
        Triangles = "TRIANGLES",
        TriangleStrip = "TRIANGLE_STRIP",
        TriangleFan = "TRIANGLE_FAN",
        Quads = "QUADS",
        QuadStrip = "QUAD_STRIP",
        Polygon = "POLYGON",
        LinesAdjacency = "LINES_ADJACENCY",
        LineStripAdjacency = "LINE_STRIP_ADJACENCY",
        TrianglesAdjacency = "TRIANGLES_ADJACENCY",
        TriangleStripAdjacency = "TRIANGLE_STRIP_ADJACENCY",
        Patches = "PATCHES",
    }
}

impl Default for Semantic {
    /// The C's zero.
    fn default() -> Semantic {
        Semantic::Position
    }
}

impl Default for Interpolate {
    /// The C's zero.
    fn default() -> Interpolate {
        Interpolate::Constant
    }
}

impl Default for Location {
    /// The C's zero.
    fn default() -> Location {
        Location::Center
    }
}

impl Primitive {
    /// `u_vertices_per_prim`: how many vertices a geometry shader sees per input primitive. The
    /// primitives no geometry shader takes answer 3, as the C's default does.
    pub fn vertices_per_prim(self) -> u32 {
        match self {
            Primitive::Points => 1,
            Primitive::Lines | Primitive::LineLoop | Primitive::LineStrip => 2,
            Primitive::Triangles | Primitive::TriangleStrip | Primitive::TriangleFan => 3,
            Primitive::LinesAdjacency | Primitive::LineStripAdjacency => 4,
            Primitive::TrianglesAdjacency | Primitive::TriangleStripAdjacency => 6,
            Primitive::Polygon | Primitive::Quads | Primitive::QuadStrip | Primitive::Patches => 3,
        }
    }
}

named! {
    /// `tgsi_fs_coord_origin`.
    CoordOrigin {
        UpperLeft = "UPPER_LEFT",
        LowerLeft = "LOWER_LEFT",
    }
}

named! {
    /// `tgsi_fs_coord_pixcenter`.
    PixelCenter {
        HalfInteger = "HALF_INTEGER",
        Integer = "INTEGER",
    }
}

named! {
    /// `tgsi_imm_type`.
    ImmType {
        Float32 = "FLT32",
        Uint32 = "UINT32",
        Int32 = "INT32",
        Float64 = "FLT64",
        Uint64 = "UINT64",
        Int64 = "INT64",
    }
}

named! {
    /// `TGSI_MEMORY_*`, a load or store's qualifiers, by bit.
    MemoryQualifier {
        Coherent = "COHERENT",
        Restrict = "RESTRICT",
        Volatile = "VOLATILE",
    }
}

named! {
    /// `tgsi_memory_type`, a `MEMORY` declaration's space.
    MemoryType {
        Global = "GLOBAL",
        Shared = "SHARED",
        Private = "PRIVATE",
        Input = "INPUT",
    }
}

/// `TGSI_WRITEMASK_*`.
pub const WRITEMASK_X: u8 = 0x1;
pub const WRITEMASK_Y: u8 = 0x2;
pub const WRITEMASK_XY: u8 = 0x3;
pub const WRITEMASK_Z: u8 = 0x4;
pub const WRITEMASK_XZ: u8 = 0x5;
pub const WRITEMASK_XYZ: u8 = 0x7;
pub const WRITEMASK_W: u8 = 0x8;
pub const WRITEMASK_XYZW: u8 = 0xf;

/// `TGSI_SWIZZLE_*`: a component, as a swizzle names one.
pub const SWIZZLE_X: u8 = 0;
pub const SWIZZLE_Y: u8 = 1;
pub const SWIZZLE_Z: u8 = 2;
pub const SWIZZLE_W: u8 = 3;

pub const SWIZZLE_NAMES: [&str; 4] = ["x", "y", "z", "w"];

/// The most registers an instruction names, on each side (`TGSI_FULL_MAX_*_REGISTERS`).
pub const MAX_DST: usize = 2;
pub const MAX_SRC: usize = 5;
pub const MAX_TEX_OFFSETS: usize = 4;

/// `tgsi_ind_register`: the register an indirect index is read from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndReg {
    pub file: File,
    pub index: i16,
    pub swizzle: u8,
    /// Which declared array the access stays inside; zero for the whole file. Ten bits.
    pub array_id: u16,
}

impl Default for IndReg {
    fn default() -> IndReg {
        IndReg { file: File::Null, index: 0, swizzle: SWIZZLE_X, array_id: 0 }
    }
}

/// `tgsi_dimension`: a register's second index.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Dimension {
    pub indirect: bool,
    pub index: i16,
}

/// `tgsi_full_src_register`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Src {
    pub file: File,
    pub indirect: bool,
    pub dimension: bool,
    pub index: i16,
    pub swizzle: [u8; 4],
    pub absolute: bool,
    pub negate: bool,
    pub ind: IndReg,
    pub dim: Dimension,
    pub dim_ind: IndReg,
}

impl Default for Src {
    fn default() -> Src {
        Src {
            file: File::Null,
            indirect: false,
            dimension: false,
            index: 0,
            swizzle: [SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W],
            absolute: false,
            negate: false,
            ind: IndReg::default(),
            dim: Dimension::default(),
            dim_ind: IndReg::default(),
        }
    }
}

/// `tgsi_full_dst_register`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dst {
    pub file: File,
    pub writemask: u8,
    pub indirect: bool,
    pub dimension: bool,
    pub index: i16,
    pub ind: IndReg,
    pub dim: Dimension,
    pub dim_ind: IndReg,
}

impl Default for Dst {
    fn default() -> Dst {
        Dst {
            file: File::Null,
            writemask: WRITEMASK_XYZW,
            indirect: false,
            dimension: false,
            index: 0,
            ind: IndReg::default(),
            dim: Dimension::default(),
            dim_ind: IndReg::default(),
        }
    }
}

/// `tgsi_texture_offset`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TexOffset {
    pub file: File,
    pub index: i16,
    pub swizzle: [u8; 3],
}

impl Default for TexOffset {
    fn default() -> TexOffset {
        TexOffset { file: File::Null, index: 0, swizzle: [0; 3] }
    }
}

/// `tgsi_instruction_texture`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TextureInfo {
    pub texture: Texture,
    pub num_offsets: u8,
}

impl Default for TextureInfo {
    fn default() -> TextureInfo {
        TextureInfo { texture: Texture::Unknown, num_offsets: 0 }
    }
}

/// `tgsi_instruction_memory`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemoryInfo {
    /// `TGSI_MEMORY_*` bits, by [`MemoryQualifier`] ordinal.
    pub qualifier: u8,
    /// The image's target. The C's "none" is its zero, which is `BUFFER`: an instruction that
    /// named no target reads as a buffer one, and one that named `BUFFER` reads as having named
    /// nothing.
    pub texture: Texture,
    /// A `PIPE_FORMAT_*` wire number; zero for none.
    pub format: u16,
}

impl Default for MemoryInfo {
    fn default() -> MemoryInfo {
        MemoryInfo { qualifier: 0, texture: Texture::Buffer, format: 0 }
    }
}

/// `tgsi_full_instruction`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Instruction {
    pub opcode: Opcode,
    pub saturate: bool,
    pub precise: bool,
    pub num_dst: u8,
    pub num_src: u8,
    /// A branch target, 24 bits.
    pub label: Option<u32>,
    /// Present when the instruction carries a texture word.
    pub texture: Option<TextureInfo>,
    pub memory: Option<MemoryInfo>,
    pub dst: [Dst; MAX_DST],
    pub src: [Src; MAX_SRC],
    pub tex_offsets: [TexOffset; MAX_TEX_OFFSETS],
}

impl Instruction {
    /// The registers this instruction writes.
    pub fn dsts(&self) -> &[Dst] {
        &self.dst[..self.num_dst as usize]
    }

    /// The registers this instruction reads.
    pub fn srcs(&self) -> &[Src] {
        &self.src[..self.num_src as usize]
    }

    /// The texture word, or the C's default when the instruction has none.
    pub fn tex(&self) -> TextureInfo {
        self.texture.unwrap_or_default()
    }

    /// How many 32-bit tokens the C would pack this into.
    fn token_count(&self) -> u32 {
        let mut n = 1;
        if self.label.is_some() {
            n += 1;
        }
        if let Some(t) = self.texture {
            n += 1 + u32::from(t.num_offsets);
        }
        if self.memory.is_some() {
            n += 1;
        }
        for d in self.dsts() {
            n += 1 + u32::from(d.indirect) + u32::from(d.dimension);
            if d.dimension && d.dim.indirect {
                n += 1;
            }
        }
        for s in self.srcs() {
            n += 1 + u32::from(s.indirect) + u32::from(s.dimension);
            if s.dimension && s.dim.indirect {
                n += 1;
            }
        }
        n
    }
}

/// `tgsi_declaration_semantic`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SemanticInfo {
    pub name: Semantic,
    pub index: u16,
    /// A geometry shader output's vertex stream per component, two bits each.
    pub stream: [u8; 4],
}

impl Default for SemanticInfo {
    fn default() -> SemanticInfo {
        SemanticInfo { name: Semantic::Position, index: 0, stream: [0; 4] }
    }
}

/// `tgsi_declaration_interp`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InterpInfo {
    pub interpolate: Interpolate,
    pub location: Location,
    pub cylindrical_wrap: u8,
}

impl Default for InterpInfo {
    fn default() -> InterpInfo {
        InterpInfo {
            interpolate: Interpolate::Constant,
            location: Location::Center,
            cylindrical_wrap: 0,
        }
    }
}

/// `tgsi_declaration_image`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImageInfo {
    pub resource: Texture,
    pub raw: bool,
    pub writable: bool,
    /// A `PIPE_FORMAT_*` wire number; zero for none.
    pub format: u16,
}

impl Default for ImageInfo {
    fn default() -> ImageInfo {
        ImageInfo { resource: Texture::Buffer, raw: false, writable: false, format: 0 }
    }
}

/// `tgsi_declaration_sampler_view`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SamplerViewInfo {
    pub resource: Texture,
    pub return_type: [ReturnType; 4],
}

impl Default for SamplerViewInfo {
    fn default() -> SamplerViewInfo {
        SamplerViewInfo { resource: Texture::Buffer, return_type: [ReturnType::Unorm; 4] }
    }
}

/// `tgsi_full_declaration`. The `has_*` flags are the C's presence bits: a semantic or an
/// interpolation is read through its struct whether or not one was declared, and the defaults
/// are the C's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Declaration {
    pub file: File,
    pub usage_mask: u8,
    pub first: u16,
    pub last: u16,
    /// The second index (`Dim.Index2D`), when the declaration is two-dimensional.
    pub dimension: Option<u16>,
    pub has_semantic: bool,
    pub semantic: SemanticInfo,
    pub has_interpolate: bool,
    pub interp: InterpInfo,
    pub invariant: bool,
    pub local: bool,
    /// The array id, when the declaration names one.
    pub array: Option<u16>,
    pub atomic: bool,
    pub mem_type: MemoryType,
    pub image: ImageInfo,
    pub sampler_view: SamplerViewInfo,
}

impl Default for Declaration {
    fn default() -> Declaration {
        Declaration {
            file: File::Null,
            usage_mask: WRITEMASK_XYZW,
            first: 0,
            last: 0,
            dimension: None,
            has_semantic: false,
            semantic: SemanticInfo::default(),
            has_interpolate: false,
            interp: InterpInfo::default(),
            invariant: false,
            local: false,
            array: None,
            atomic: false,
            mem_type: MemoryType::Global,
            image: ImageInfo::default(),
            sampler_view: SamplerViewInfo::default(),
        }
    }
}

impl Declaration {
    /// How many 32-bit tokens the C would pack this into.
    fn token_count(&self) -> u32 {
        2 + u32::from(self.dimension.is_some())
            + u32::from(self.has_interpolate)
            + u32::from(self.has_semantic)
            + u32::from(self.file == File::Image)
            + u32::from(self.file == File::SamplerView)
            + u32::from(self.array.is_some())
    }
}

/// `tgsi_full_immediate`: four data words, as the text always gives four (a 64-bit type packs
/// two values into them).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Immediate {
    pub ty: ImmType,
    pub data: [u32; 4],
}

impl Immediate {
    pub fn float(&self, i: usize) -> f32 {
        f32::from_bits(self.data[i])
    }

    pub fn int(&self, i: usize) -> i32 {
        self.data[i] as i32
    }

    pub fn uint(&self, i: usize) -> u32 {
        self.data[i]
    }
}

/// `tgsi_full_property`: one data word, as the text always gives one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PropertyToken {
    pub name: Property,
    pub data: u32,
}

/// One token of a program, in the C's four kinds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Token {
    Declaration(Declaration),
    Immediate(Immediate),
    Instruction(Instruction),
    Property(PropertyToken),
}

impl Token {
    fn token_count(&self) -> u32 {
        match self {
            Token::Declaration(d) => d.token_count(),
            Token::Immediate(_) => 5,
            Token::Instruction(i) => i.token_count(),
            Token::Property(_) => 2,
        }
    }
}

/// A parsed program.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shader {
    pub processor: Processor,
    pub tokens: Vec<Token>,
}

impl Shader {
    /// How many 32-bit tokens the C would have packed this program into, header included --
    /// what the guest's `num_tokens` is measured against.
    pub fn token_count(&self) -> u32 {
        2 + self.tokens.iter().map(Token::token_count).sum::<u32>()
    }

    pub fn declarations(&self) -> impl Iterator<Item = &Declaration> {
        self.tokens.iter().filter_map(|t| match t {
            Token::Declaration(d) => Some(d),
            _ => None,
        })
    }

    pub fn instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.tokens.iter().filter_map(|t| match t {
            Token::Instruction(i) => Some(i),
            _ => None,
        })
    }

    pub fn immediates(&self) -> impl Iterator<Item = &Immediate> {
        self.tokens.iter().filter_map(|t| match t {
            Token::Immediate(i) => Some(i),
            _ => None,
        })
    }

    pub fn properties(&self) -> impl Iterator<Item = &PropertyToken> {
        self.tokens.iter().filter_map(|t| match t {
            Token::Property(p) => Some(p),
            _ => None,
        })
    }

    /// The value of a property, if the program declares it.
    pub fn property(&self, name: Property) -> Option<u32> {
        self.properties().find(|p| p.name == name).map(|p| p.data)
    }
}
