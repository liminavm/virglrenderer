// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The TGSI-to-GLSL emitter: `vrend_shader.c`'s `dump_ctx` and the passes that fill it.
//!
//! The C walks the token stream twice. The first pass (`iter_decls`, `analyze_instruction`)
//! notes the edge cases the second needs to know in advance; the second (`iter_declaration`,
//! `iter_immediate`, `iter_property`, `iter_instruction`) emits the body of `main` as it goes,
//! and the header and interface declarations are written afterwards from what the walk found.
//! The three strings the C hands GL -- version and extensions, header, main -- are kept apart
//! here too ([`Strings`]), so a log prints them the way the C's does.
//!
//! Every `printf`-style format of the C is reproduced with `format!`, and every emitted string
//! is the C's, spaces and all: the fixture diff reads the output, not the code.

mod decl;
mod exit;
mod header;
mod inst;
mod tex;

use std::fmt;

use super::{
    Array, Cfg, Info, InterpInfo, IoArray, IoArrayInfo, Key, MAX_SHADER_BUFFERS, MAX_SHADER_IMAGES,
    MAX_SO_OUTPUTS, POLYGON_STIPPLE_SIZE, VarInfo,
};
use crate::vrend::pipe::{LogicOp, PrimType};
use crate::vrend::proto::StreamOutput;
use crate::vrend::tgsi::{
    self, File, ImageInfo, ImmType, Interpolate, Location, Processor, ReturnType, Semantic, Shader,
    Texture, Token, scan,
};

/// Why a translation failed: the C's `virgl_error` line, which is all it says.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Failure(pub String);

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Failure {}

fn fail<T>(message: String) -> Result<T, Failure> {
    Err(Failure(message))
}

/// The GLSL, in the three pieces the C keeps (`SHADER_STRING_VER_EXT`, `SHADER_STRING_HDR`,
/// and main) and hands GL in that order.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Strings {
    pub ver_ext: String,
    pub hdr: String,
    pub main: String,
}

impl Strings {
    /// The whole source, as GL sees it.
    pub fn source(&self) -> String {
        format!("{}{}{}", self.ver_ext, self.hdr, self.main)
    }
}

/// `SHADER_REQ_*`: what the program turned out to need of the host, by bit.
pub(super) mod req {
    pub const SAMPLER_RECT: u64 = 1 << 0;
    pub const CUBE_ARRAY: u64 = 1 << 1;
    pub const INTS: u64 = 1 << 2;
    pub const SAMPLER_MS: u64 = 1 << 3;
    pub const INSTANCE_ID: u64 = 1 << 4;
    pub const LODQ: u64 = 1 << 5;
    pub const TXQ_LEVELS: u64 = 1 << 6;
    pub const TG4: u64 = 1 << 7;
    pub const VIEWPORT_IDX: u64 = 1 << 8;
    pub const STENCIL_EXPORT: u64 = 1 << 9;
    pub const LAYER: u64 = 1 << 10;
    pub const SAMPLE_SHADING: u64 = 1 << 11;
    pub const GPU_SHADER5: u64 = 1 << 12;
    pub const DERIVATIVE_CONTROL: u64 = 1 << 13;
    pub const FP64: u64 = 1 << 14;
    pub const IMAGE_LOAD_STORE: u64 = 1 << 15;
    pub const ES31_COMPAT: u64 = 1 << 16;
    pub const IMAGE_SIZE: u64 = 1 << 17;
    pub const TXQS: u64 = 1 << 18;
    pub const FBFETCH: u64 = 1 << 19;
    pub const SHADER_CLOCK: u64 = 1 << 20;
    pub const PSIZE: u64 = 1 << 21;
    pub const IMAGE_ATOMIC: u64 = 1 << 22;
    pub const CLIP_DISTANCE: u64 = 1 << 23;
    pub const SEPERATE_SHADER_OBJECTS: u64 = 1 << 25;
    pub const SHADER_ATOMIC_FLOAT: u64 = 1 << 28;
    pub const NV_IMAGE_FORMATS: u64 = 1 << 29;
    pub const CONSERVATIVE_DEPTH: u64 = 1 << 30;
    pub const SAMPLER_BUF: u64 = 1 << 31;
    pub const GEOMETRY_SHADER: u64 = 1 << 32;
    pub const BLEND_EQUATION_ADVANCED: u64 = 1 << 33;
    pub const EXPLICIT_ATTRIB_LOCATION: u64 = 1 << 34;
    pub const SHADER_NOPERSPECTIVE_INTERPOLATION: u64 = 1 << 35;
    pub const TEXTURE_SHADOW_LOD: u64 = 1 << 36;
    pub const AMD_VS_LAYER: u64 = 1 << 37;
    pub const SHADER_DRAW_PARAMETERS: u64 = 1 << 39;
    pub const SHADER_GROUP_VOTE: u64 = 1 << 40;
    pub const EXPLICIT_UNIFORM_LOCATION: u64 = 1 << 41;
}

/// `vrend_sysval_uniform`: the members of the `VirglBlock` uniform block a program reads, by
/// bit.
pub(super) mod sysval {
    pub const WINSYS_ADJUST_Y: u8 = 1 << 0;
    pub const CLIP_PLANE: u8 = 1 << 1;
    pub const ALPHA_REF_VAL: u8 = 1 << 2;
    pub const PSTIPPLE_SAMPLER: u8 = 1 << 3;
    pub const DRAWID_BASE: u8 = 1 << 4;
}

/// `MAX_VARYING`.
pub(super) const MAX_VARYING: u32 = 32;
/// The C's `inputs[64]` and `outputs[64]`.
pub(super) const MAX_IO: usize = 64;
/// The C's `system_values[32]`.
const MAX_SYSTEM_VALUES: usize = 32;
/// The C's `samplers[32]`.
pub(super) const MAX_SAMPLERS: usize = 32;
/// `MAX_IMMEDIATE`.
const MAX_IMMEDIATE: usize = 1024;

/// `vec_type`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum VecType {
    #[default]
    Float,
    Int,
    Uint,
}

/// `vrend_type_qualifier`: a GLSL type or bit-cast, as the C names it in a format.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum Qual {
    #[default]
    None,
    Float,
    Vec2,
    Vec3,
    Vec4,
    Int,
    IVec2,
    IVec3,
    IVec4,
    Uint,
    UVec2,
    UVec3,
    UVec4,
    FloatBitsToUint,
    UintBitsToFloat,
    FloatBitsToInt,
    IntBitsToFloat,
    Double,
    DVec2,
}

impl Qual {
    /// `get_string`.
    pub(super) fn s(self) -> &'static str {
        use Qual::*;
        match self {
            None => "",
            Float => "float",
            Vec2 => "vec2",
            Vec3 => "vec3",
            Vec4 => "vec4",
            Int => "int",
            IVec2 => "ivec2",
            IVec3 => "ivec3",
            IVec4 => "ivec4",
            Uint => "uint",
            UVec2 => "uvec2",
            UVec3 => "uvec3",
            UVec4 => "uvec4",
            FloatBitsToUint => "floatBitsToUint",
            UintBitsToFloat => "uintBitsToFloat",
            FloatBitsToInt => "floatBitsToInt",
            IntBitsToFloat => "intBitsToFloat",
            Double => "double",
            DVec2 => "dvec2",
        }
    }

    /// The C computes vector qualifiers by adding a width to a base (`FLOAT + n - 1`); this is
    /// that arithmetic on the enum's ordinals.
    pub(super) fn vec(base: Qual, width: u32) -> Qual {
        use Qual::*;
        const ALL: [Qual; 19] = [
            None,
            Float,
            Vec2,
            Vec3,
            Vec4,
            Int,
            IVec2,
            IVec3,
            IVec4,
            Uint,
            UVec2,
            UVec3,
            UVec4,
            FloatBitsToUint,
            UintBitsToFloat,
            FloatBitsToInt,
            IntBitsToFloat,
            Double,
            DVec2,
        ];
        let i = base as usize + width as usize - 1;
        ALL.get(i).copied().unwrap_or(None)
    }
}

/// `io_type`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum IoDir {
    In,
    Out,
}

/// `io_decl_type`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum IoDecl {
    Plain,
    Block,
}

/// `vrend_shader_io`: one input, output or system value as the emitter names it.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub(super) struct Io {
    pub glsl_name: String,
    /// The entry this one shares an array with, by index into the same list.
    pub overlapping_array: Option<usize>,
    pub sid: u32,
    pub first: u32,
    pub last: u32,
    pub array_id: u32,
    pub interpolate: Interpolate,
    pub location: Location,
    pub array_offset: u32,
    pub name: Semantic,
    pub stream: u32,
    pub usage_mask: u8,
    pub ty: VecType,
    pub num_components: u32,
    pub invariant: bool,
    pub precise: bool,
    pub glsl_predefined_no_emit: bool,
    pub glsl_no_index: bool,
    pub glsl_gl_block: bool,
    pub override_no_wm: bool,
    pub is_int: bool,
    pub fbfetch_used: bool,
    pub needs_override: bool,
}

/// `vrend_interface_bits`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct InterfaceBits {
    pub outputs_expected_mask: u64,
    pub inputs_emitted_mask: u64,
    pub outputs_emitted_mask: u64,
}

/// `vrend_generic_ios`. The C's also carries an input and an output range, which nothing
/// ever marks used; the branches that would read them are not here.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub(super) struct GenericIos {
    pub matched: InterfaceBits,
}

/// `vrend_temp_range`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct TempRange {
    pub first: i32,
    pub last: i32,
    pub array_id: i32,
    pub precise_result: bool,
}

/// `vrend_shader_sampler`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Sampler {
    pub ty: Texture,
    pub ret: ReturnType,
}

impl Default for Sampler {
    fn default() -> Sampler {
        Sampler { ty: Texture::Buffer, ret: ReturnType::Unorm }
    }
}

/// `vrend_shader_image`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Image {
    pub decl: ImageInfo,
    pub image_return: ReturnType,
    pub vflag: bool,
    pub coherent: bool,
}

impl Default for Image {
    fn default() -> Image {
        Image {
            decl: ImageInfo::default(),
            image_return: ReturnType::Unorm,
            vflag: false,
            coherent: false,
        }
    }
}

/// `immed`: an immediate's four words and how to read them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Immed {
    pub ty: ImmType,
    pub val: [u32; 4],
}

/// `vrend_glsl_strbufs`: the three output strings and the indentation of main. The C's
/// buffers carry an error flag that a failed emit sets and the driver checks at the end; the
/// flags are here for the same purpose.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub(super) struct Bufs {
    pub indent_level: i32,
    pub required_sysval_uniform_decls: u8,
    pub main: String,
    pub hdr: String,
    pub ver_ext: String,
    pub main_error: bool,
    pub hdr_error: bool,
}

impl Bufs {
    /// `emit_indent`: a tab per level, at most fifteen.
    fn emit_indent(&mut self) {
        if self.indent_level > 0 {
            let n = self.indent_level.min(15) as usize;
            self.main.extend(std::iter::repeat_n('\t', n));
        }
    }

    /// `emit_buf` / `emit_buff`.
    pub fn emit(&mut self, s: &str) {
        self.emit_indent();
        self.main.push_str(s);
    }

    /// `indent_buf`.
    pub fn indent(&mut self) {
        self.indent_level += 1;
    }

    /// `outdent_buf`.
    pub fn outdent(&mut self) {
        if self.indent_level <= 0 {
            self.main_error = true;
            return;
        }
        self.indent_level -= 1;
    }

    /// `set_buf_error`.
    pub fn set_error(&mut self) {
        self.main_error = true;
    }

    /// `emit_hdr` / `emit_hdrf`.
    pub fn hdr(&mut self, s: &str) {
        self.hdr.push_str(s);
    }

    /// `set_hdr_error`.
    pub fn set_hdr_error(&mut self) {
        self.hdr_error = true;
    }

    /// `emit_ver_ext` / `emit_ver_extf`.
    pub fn ver_ext(&mut self, s: &str) {
        self.ver_ext.push_str(s);
    }
}

/// `emit_buff`: a formatted line of main.
macro_rules! emit {
    ($bufs:expr, $($arg:tt)*) => { $bufs.emit(&format!($($arg)*)) };
}
/// `emit_hdrf`: a formatted line of the header.
macro_rules! hdr {
    ($bufs:expr, $($arg:tt)*) => { $bufs.hdr(&format!($($arg)*)) };
}
pub(super) use {emit, hdr};

/// `dump_ctx`: the state of one translation.
pub(super) struct Ctx<'a> {
    pub cfg: &'a Cfg,
    pub key: &'a Key,
    pub info: &'a scan::Info,
    pub prog_type: Processor,
    pub bufs: Bufs,
    pub instno: u32,

    /// The C's `src_bufs` and `dst_bufs`: one per operand slot, kept across instructions,
    /// so an operand a walk does not fill reads as the last one that did.
    pub src_bufs: [String; tgsi::MAX_SRC],
    pub dst_bufs: [String; tgsi::MAX_DST],

    pub interp_input_mask: u64,
    pub attrib_input_mask: u32,
    pub inputs: Vec<Io>,
    pub outputs: Vec<Io>,
    pub front_back_color_emitted_flags: [u8; MAX_IO],
    pub system_values: Vec<Io>,

    pub guest_sent_io_arrays: bool,
    pub texcoord_ios: InterfaceBits,
    pub generic_ios: GenericIos,

    pub temp_ranges: Vec<TempRange>,

    pub samplers: [Sampler; MAX_SAMPLERS],
    pub samplers_used: u32,

    pub ssbo_first_binding: u32,
    pub ssbo_used_mask: u32,
    pub ssbo_atomic_mask: u32,
    pub ssbo_array_base: u32,
    pub ssbo_atomic_array_base: u32,
    pub ssbo_integer_mask: u32,
    pub ssbo_memory_qualifier: [u8; MAX_SHADER_BUFFERS],
    pub ssbo_last_binding: i32,

    pub images: [Image; MAX_SHADER_IMAGES],
    pub images_used_mask: u32,
    pub image_last_binding: i32,

    pub image_arrays: Vec<Array>,
    pub sampler_arrays: Vec<Array>,

    pub fog_input_mask: u32,
    pub fog_output_mask: u32,

    pub num_consts: i32,
    pub imm: Vec<Immed>,

    pub req_local_mem: u32,
    pub integer_memory: bool,

    pub ubo_base: u32,
    pub ubo_used_mask: u32,
    pub ubo_sizes: [i32; 32],
    pub num_address: u32,

    pub abo_idx: Vec<i32>,
    pub abo_sizes: Vec<i32>,
    pub abo_offsets: Vec<i32>,

    pub shader_req_bits: u64,
    pub patches_emitted_mask: u64,

    pub so: Option<&'a StreamOutput>,
    pub so_names: Vec<Option<String>>,
    pub write_so_outputs: [bool; MAX_SO_OUTPUTS],
    pub write_all_cbufs: bool,
    pub shadow_samp_mask: u32,

    pub fs_lower_left_origin: bool,
    pub fs_integer_pixel_center: bool,
    pub fs_depth_layout: u32,
    /// `FS_BLEND_EQUATION_ADVANCED`, a bit per [`AdvancedBlend`].
    pub fs_blend_equation_advanced: u32,

    pub separable_program: bool,

    pub gs_in_prim: u32,
    pub gs_out_prim: u32,
    pub gs_max_out_verts: u32,
    pub gs_num_invocations: u32,

    pub num_in_clip_dist: i32,
    pub num_out_clip_dist: i32,
    pub fs_uses_clipdist_input: bool,
    pub glsl_ver_required: u32,
    pub color_in_mask: u32,
    pub color_out_mask: u32,
    pub num_cull_dist_prop: u8,
    pub num_clip_dist_prop: u8,
    pub has_pervertex: bool,
    pub front_face_emitted: bool,

    pub has_clipvertex: bool,
    pub has_clipvertex_so: bool,
    pub write_mul_utemp: bool,
    pub write_mul_itemp: bool,
    pub has_sample_input: bool,
    pub has_noperspective: bool,
    pub early_depth_stencil: bool,
    pub has_file_memory: bool,
    pub force_color_two_side: bool,
    pub gles_use_tex_query_level: bool,
    pub has_pointsize_input: bool,
    pub has_pointsize_output: bool,

    pub has_input_arrays: bool,
    pub has_output_arrays: bool,

    pub tcs_vertices_out: u32,
    pub tes_prim_mode: u32,
    pub tes_spacing: u32,
    pub tes_vertex_order: u32,
    pub tes_point_mode: u32,
    pub is_last_vertex_stage: bool,
    pub require_dummy_value: bool,

    pub local_cs_block_size: [u16; 3],
}

impl<'a> Ctx<'a> {
    pub(super) fn new(
        cfg: &'a Cfg,
        key: &'a Key,
        info: &'a scan::Info,
        processor: Processor,
    ) -> Ctx<'a> {
        Ctx {
            cfg,
            key,
            info,
            prog_type: processor,
            bufs: Bufs::default(),
            instno: 0,
            src_bufs: Default::default(),
            dst_bufs: Default::default(),
            interp_input_mask: 0,
            attrib_input_mask: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            front_back_color_emitted_flags: [0; MAX_IO],
            system_values: Vec::new(),
            guest_sent_io_arrays: false,
            texcoord_ios: InterfaceBits::default(),
            generic_ios: GenericIos::default(),
            temp_ranges: Vec::new(),
            samplers: [Sampler::default(); MAX_SAMPLERS],
            samplers_used: 0,
            ssbo_first_binding: u32::MAX,
            ssbo_used_mask: 0,
            ssbo_atomic_mask: 0,
            ssbo_array_base: 0xffff_ffff,
            ssbo_atomic_array_base: 0xffff_ffff,
            ssbo_integer_mask: 0,
            ssbo_memory_qualifier: [0; MAX_SHADER_BUFFERS],
            ssbo_last_binding: -1,
            images: [Image::default(); MAX_SHADER_IMAGES],
            images_used_mask: 0,
            image_last_binding: -1,
            image_arrays: Vec::new(),
            sampler_arrays: Vec::new(),
            fog_input_mask: 0,
            fog_output_mask: 0,
            num_consts: 0,
            imm: Vec::new(),
            req_local_mem: 0,
            integer_memory: false,
            ubo_base: 0,
            ubo_used_mask: 0,
            ubo_sizes: [0; 32],
            num_address: 0,
            abo_idx: Vec::new(),
            abo_sizes: Vec::new(),
            abo_offsets: Vec::new(),
            shader_req_bits: 0,
            patches_emitted_mask: 0,
            so: None,
            so_names: Vec::new(),
            write_so_outputs: [false; MAX_SO_OUTPUTS],
            write_all_cbufs: false,
            shadow_samp_mask: 0,
            fs_lower_left_origin: false,
            fs_integer_pixel_center: false,
            fs_depth_layout: 0,
            fs_blend_equation_advanced: 0,
            separable_program: false,
            gs_in_prim: 0,
            gs_out_prim: 0,
            gs_max_out_verts: 0,
            gs_num_invocations: 0,
            num_in_clip_dist: 0,
            num_out_clip_dist: 0,
            fs_uses_clipdist_input: false,
            glsl_ver_required: 0,
            color_in_mask: 0,
            color_out_mask: 0,
            num_cull_dist_prop: 0,
            num_clip_dist_prop: 0,
            has_pervertex: false,
            front_face_emitted: false,
            has_clipvertex: false,
            has_clipvertex_so: false,
            write_mul_utemp: false,
            write_mul_itemp: false,
            has_sample_input: false,
            has_noperspective: false,
            early_depth_stencil: false,
            has_file_memory: false,
            force_color_two_side: false,
            gles_use_tex_query_level: false,
            has_pointsize_input: false,
            has_pointsize_output: false,
            has_input_arrays: false,
            has_output_arrays: false,
            tcs_vertices_out: 0,
            tes_prim_mode: 0,
            tes_spacing: 0,
            tes_vertex_order: 0,
            tes_point_mode: 0,
            is_last_vertex_stage: false,
            require_dummy_value: false,
            local_cs_block_size: [0; 3],
        }
    }

    /// `require_glsl_ver`.
    pub fn require_glsl_ver(&self, v: u32) -> u32 {
        v.max(self.glsl_ver_required)
    }

    /// `prefer_generic_io_block`, GLES leg: arrays of arrays are never preferred on GLES, so
    /// the answer depends only on the stage and the direction.
    pub fn prefer_generic_io_block(&self, io: IoDir) -> bool {
        match self.prog_type {
            Processor::Fragment => false,
            Processor::TessCtrl => true,
            Processor::TessEval => io == IoDir::In || self.key.gs_present,
            Processor::Geometry => io == IoDir::In,
            Processor::Vertex => io == IoDir::Out && (self.key.gs_present || self.key.tes_present),
            Processor::Compute => false,
        }
    }

    /// `get_stage_input_name_prefix`.
    pub fn stage_input_name_prefix(&self, processor: Processor) -> &'static str {
        match processor {
            Processor::Fragment => {
                if self.key.gs_present {
                    "gso"
                } else if self.key.tes_present {
                    "teo"
                } else {
                    "vso"
                }
            }
            Processor::Geometry => {
                if self.key.tes_present {
                    "teo"
                } else {
                    "vso"
                }
            }
            Processor::TessEval => {
                if self.key.tcs_present {
                    "tco"
                } else {
                    "vso"
                }
            }
            Processor::TessCtrl => "vso",
            Processor::Vertex | Processor::Compute => "in",
        }
    }

    /// `find_temp_range`.
    pub fn find_temp_range(&self, index: i32) -> Option<usize> {
        self.temp_ranges.iter().position(|r| index >= r.first && index <= r.last)
    }

    /// `lookup_sampler_array`.
    pub fn lookup_sampler_array(&self, index: i32) -> i32 {
        self.sampler_arrays
            .iter()
            .find(|a| index >= a.first && index < a.first + a.array_size)
            .map_or(-1, |a| a.first)
    }

    /// `lookup_image_array_ptr`.
    pub fn lookup_image_array_ptr(&self, index: i32) -> Option<usize> {
        self.image_arrays.iter().position(|a| index >= a.first && index < a.first + a.array_size)
    }

    /// `lookup_image_array`.
    pub fn lookup_image_array(&self, index: i32) -> i32 {
        self.lookup_image_array_ptr(index).map_or(-1, |i| self.image_arrays[i].first)
    }

    /// `logiop_require_inout`.
    pub fn logiop_require_inout(&self) -> bool {
        match self.key.fs.logicop_func {
            None => false,
            Some(LogicOp::Clear | LogicOp::Set | LogicOp::Copy | LogicOp::CopyInverted) => false,
            Some(_) => true,
        }
    }

    /// Whether stream output `i` needs a temporary: the C's decoder marks an output that
    /// writes fewer than four components, or a register another output also writes.
    pub fn so_need_temp(&self, i: usize) -> bool {
        let so = self.so.expect("stream output present");
        let o = so.outputs[i];
        o.num_components < 4 || so.outputs[..i].iter().any(|p| p.register_index == o.register_index)
    }
}

/// `get_wm_string`.
pub(super) fn wm_string(wm: u8) -> &'static str {
    match wm {
        0 => "",
        tgsi::WRITEMASK_X => ".x",
        tgsi::WRITEMASK_XY => ".xy",
        tgsi::WRITEMASK_XYZ => ".xyz",
        tgsi::WRITEMASK_W => ".w",
        _ => {
            println!("Unable to unknown writemask");
            ""
        }
    }
}

/// `get_swizzle_string`, over `pipe_swizzle` values.
pub(super) fn swizzle_string(swizzle: u16) -> &'static str {
    match swizzle {
        0 => ".x",
        1 => ".y",
        2 => ".z",
        3 => ".w",
        4 | 5 => ".0",
        _ => "",
    }
}

/// `tgsi_proc_to_prefix`.
pub(super) fn proc_prefix(p: Processor) -> &'static str {
    match p {
        Processor::Vertex => "vs",
        Processor::Fragment => "fs",
        Processor::Geometry => "gs",
        Processor::TessCtrl => "tc",
        Processor::TessEval => "te",
        Processor::Compute => "cs",
    }
}

/// `prim_to_name`, over `pipe_prim_type` values.
pub(super) fn prim_to_name(prim: u32) -> &'static str {
    match PrimType::from_wire(prim) {
        Some(PrimType::Points) => "points",
        Some(PrimType::Lines) => "lines",
        Some(PrimType::LineStrip) => "line_strip",
        Some(PrimType::LinesAdjacency) => "lines_adjacency",
        Some(PrimType::Triangles) => "triangles",
        Some(PrimType::TriangleStrip) => "triangle_strip",
        Some(PrimType::TrianglesAdjacency) => "triangles_adjacency",
        Some(PrimType::Quads) => "quads",
        _ => "UNKNOWN",
    }
}

/// `prim_to_tes_name`.
pub(super) fn prim_to_tes_name(prim: u32) -> &'static str {
    match PrimType::from_wire(prim) {
        Some(PrimType::Quads) => "quads",
        Some(PrimType::Triangles) => "triangles",
        Some(PrimType::Lines) => "isolines",
        _ => "UNKNOWN",
    }
}

/// `get_spacing_string`, over `pipe_tess_spacing` values.
pub(super) fn spacing_string(spacing: u32) -> &'static str {
    match spacing {
        0 => "fractional_odd_spacing",
        1 => "fractional_even_spacing",
        _ => "equal_spacing",
    }
}

/// `gs_input_prim_to_size`.
pub(super) fn gs_input_prim_to_size(prim: u32) -> i32 {
    match PrimType::from_wire(prim) {
        Some(PrimType::Points) => 1,
        Some(PrimType::Lines) => 2,
        Some(PrimType::LinesAdjacency) => 4,
        Some(PrimType::Triangles) => 3,
        Some(PrimType::TrianglesAdjacency) => 6,
        _ => -1,
    }
}

/// `get_stage_output_name_prefix`.
pub(super) fn stage_output_name_prefix(p: Processor) -> &'static str {
    match p {
        Processor::Fragment => "fsout",
        Processor::Geometry => "gso",
        Processor::Vertex => "vso",
        Processor::TessCtrl => "tco",
        Processor::TessEval => "teo",
        Processor::Compute => "out",
    }
}

/// `samplertype_is_shadow`.
pub(super) fn samplertype_is_shadow(t: Texture) -> bool {
    matches!(
        t,
        Texture::Shadow1d
            | Texture::Shadow1dArray
            | Texture::Shadow2d
            | Texture::ShadowRect
            | Texture::Shadow2dArray
            | Texture::ShadowCube
            | Texture::ShadowCubeArray
    )
}

/// `samplertype_to_req_bits`.
pub(super) fn samplertype_to_req_bits(t: Texture) -> u64 {
    match t {
        Texture::ShadowCubeArray | Texture::CubeArray => req::CUBE_ARRAY,
        Texture::Msaa2d | Texture::Msaa2dArray => req::SAMPLER_MS,
        Texture::Buffer => req::SAMPLER_BUF,
        Texture::ShadowRect | Texture::Rect => req::SAMPLER_RECT,
        _ => 0,
    }
}

/// `1 << n` as the C computes it on this host's arm64: the shift count wraps at the operand's
/// width, so a guest's out-of-range index shifts by its low bits rather than trapping.
pub(super) fn bit32(n: u32) -> u32 {
    1u32.wrapping_shl(n)
}

pub(super) fn bit64(n: u32) -> u64 {
    1u64.wrapping_shl(n)
}

/// `varying_bit_from_semantic_and_index`: mesa's `gl_varying_slot` for a semantic.
pub(super) fn varying_bit_from_semantic_and_index(semantic: Semantic, index: u32) -> u32 {
    const POS: u32 = 0;
    const COL0: u32 = 1;
    const COL1: u32 = 2;
    const FOGC: u32 = 3;
    const TEX0: u32 = 4;
    const PSIZ: u32 = 12;
    const BFC0: u32 = 13;
    const BFC1: u32 = 14;
    const EDGE: u32 = 15;
    const CLIP_VERTEX: u32 = 16;
    const CLIP_DIST0: u32 = 17;
    const CLIP_DIST1: u32 = 18;
    const PRIMITIVE_ID: u32 = 21;
    const LAYER: u32 = 22;
    const VIEWPORT: u32 = 23;
    const FACE: u32 = 24;
    const PNTC: u32 = 25;
    const TESS_LEVEL_OUTER: u32 = 26;
    const TESS_LEVEL_INNER: u32 = 27;
    const VAR0: u32 = 32;
    const PATCH0: u32 = VAR0 + 31 + 9;
    match semantic {
        Semantic::Position => POS,
        Semantic::Color => {
            if index == 0 {
                COL0
            } else {
                COL1
            }
        }
        Semantic::BColor => {
            if index == 0 {
                BFC0
            } else {
                BFC1
            }
        }
        Semantic::Fog => FOGC,
        Semantic::PSize => PSIZ,
        Semantic::Generic => {
            if index >= MAX_VARYING {
                eprintln!("[virglrs] Warning: Out of range TGSI_SEMANTIC_GENERIC index: {index}");
                return VAR0;
            }
            VAR0 + index
        }
        Semantic::Face => FACE,
        Semantic::EdgeFlag => EDGE,
        Semantic::PrimId => PRIMITIVE_ID,
        Semantic::ClipDist => {
            if index == 0 {
                CLIP_DIST0
            } else {
                CLIP_DIST1
            }
        }
        Semantic::ClipVertex => CLIP_VERTEX,
        Semantic::TexCoord => {
            if index >= 8 {
                eprintln!("[virglrs] Warning: Out of range TGSI_SEMANTIC_TEXCOORD index: {index}");
                return TEX0;
            }
            TEX0 + index
        }
        Semantic::PCoord => PNTC,
        Semantic::ViewportIndex => VIEWPORT,
        Semantic::Layer => LAYER,
        Semantic::TessInner => TESS_LEVEL_INNER,
        Semantic::TessOuter => TESS_LEVEL_OUTER,
        Semantic::Patch => {
            if index >= MAX_VARYING {
                eprintln!("[virglrs] Warning: Out of range TGSI_SEMANTIC_PATCH index: {index}");
                return PATCH0;
            }
            PATCH0 + index
        }
        _ => {
            eprintln!("[virglrs] Warning: Bad TGSI semantic: {}/{index}", semantic as u8);
            0
        }
    }
}

/// `get_swiz_char`.
pub(super) fn swiz_char(swiz: u8) -> char {
    match swiz {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        3 => 'w',
        _ => '\0',
    }
}

/// `vrend_convert_shader`: translate `program` under `key`, producing the GLSL and what the
/// renderer needs to know about it. `so_info` is the stream-output layout the guest sent with
/// the shader; it comes back in the info, with the GLSL name of each output beside it.
pub fn convert(
    cfg: &Cfg,
    program: &tgsi::Program,
    req_local_mem: u32,
    key: &Key,
    so_info: &StreamOutput,
) -> Result<(Strings, Info, VarInfo), Failure> {
    let shader = &program.shader;
    let processor = shader.processor;

    // The first pass. The C's context is zeroed before it, and zero is `TGSI_PROCESSOR_FRAGMENT`:
    // the pass reads `prog_type` before anything sets it, so every stage gets the fragment
    // treatment here -- its inputs are collected and its clip-distance reads noted -- and
    // only the second pass knows the stage. Both readers are harmless for a non-fragment
    // program, and the walk is reproduced as it is.
    let mut ctx = Ctx::new(cfg, key, &program.info, Processor::Fragment);
    for token in &shader.tokens {
        match token {
            Token::Declaration(d) => decl::iter_decls(&mut ctx, d)?,
            Token::Instruction(i) => decl::analyze_instruction(&mut ctx, i),
            _ => {}
        }
    }

    ctx.is_last_vertex_stage = processor == Processor::Geometry
        || (processor == Processor::TessEval && !key.gs_present)
        || (processor == Processor::Vertex && !key.gs_present && !key.tes_present);

    ctx.inputs.clear();
    ctx.prog_type = processor;
    ctx.req_local_mem = req_local_mem;
    ctx.generic_ios.matched.outputs_expected_mask = key.out_generic_expected_mask;
    ctx.texcoord_ios.outputs_expected_mask = key.out_texcoord_expected_mask;

    if cfg.glsl_version >= 140 {
        ctx.glsl_ver_required = ctx.require_glsl_ver(140);
    }
    if processor == Processor::Geometry || key.gs_present {
        ctx.glsl_ver_required = ctx.require_glsl_ver(140);
    }
    if processor == Processor::TessEval
        || processor == Processor::TessCtrl
        || key.tes_present
        || key.tcs_present
    {
        ctx.glsl_ver_required = ctx.require_glsl_ver(150);
    }

    if !so_info.outputs.is_empty() {
        ctx.so = Some(so_info);
        ctx.so_names = vec![None; so_info.outputs.len()];
    }

    if ctx.info.is_dimension_indirect(File::Constant) {
        ctx.glsl_ver_required = ctx.require_glsl_ver(150);
    }
    if ctx.info.is_indirect(File::Buffer) || ctx.info.is_indirect(File::Image) {
        ctx.glsl_ver_required = ctx.require_glsl_ver(150);
        ctx.shader_req_bits |= req::GPU_SHADER5;
    }
    if ctx.info.is_indirect(File::Sampler) {
        ctx.shader_req_bits |= req::GPU_SHADER5;
    }

    // The second pass: `prolog`, then each token in order.
    if processor == Processor::Vertex && key.gs_present {
        ctx.glsl_ver_required = ctx.require_glsl_ver(150);
    }
    for token in &shader.tokens {
        match token {
            Token::Declaration(d) => decl::iter_declaration(&mut ctx, d)?,
            Token::Immediate(i) => decl::iter_immediate(&mut ctx, i)?,
            Token::Property(p) => decl::iter_property(&mut ctx, p)?,
            Token::Instruction(i) => inst::iter_instruction(&mut ctx, i)?,
        }
    }

    if ctx.shader_req_bits & req::FP64 != 0 {
        ctx.glsl_ver_required = ctx.require_glsl_ver(150);
    }
    if ctx.bufs.required_sysval_uniform_decls != 0 {
        ctx.glsl_ver_required = ctx.require_glsl_ver(140);
    }

    if ctx.prog_type == Processor::Fragment {
        // The C's qsort is not stable; two outputs of one semantic and index are declared
        // apart only by a guest that declares the same output twice, which the declaration
        // walk already folds.
        ctx.outputs.sort_by_key(|l| (l.name as u8, l.sid));
    }

    // The C gates this on `glsl_version < 320` for GLES and `>= 320` for desktop, which
    // between them is every version.
    let fs_info = &key.fs_info;
    if !fs_info.interps.is_empty() && fs_info.has_sample_input {
        ctx.shader_req_bits |= req::GPU_SHADER5;
    }
    if !fs_info.interps.is_empty() && fs_info.has_noperspective {
        ctx.shader_req_bits |= req::SHADER_NOPERSPECTIVE_INTERPOLATION;
    }

    header::emit_header(&mut ctx);
    ctx.glsl_ver_required = header::emit_ios(&mut ctx);

    if ctx.bufs.hdr_error {
        return fail("the header could not be emitted".to_string());
    }

    let mut var_info = VarInfo::default();
    fill_interpolants(&ctx, &mut var_info);

    let mut info = Info { so_info: so_info.clone(), ..Info::default() };
    fill_sinfo(&mut ctx, &mut info);
    fill_var_sinfo(&ctx, &mut var_info);

    emit_required_sysval_uniforms(&mut ctx.bufs);

    if ctx.bufs.main_error {
        return fail("the body could not be emitted".to_string());
    }

    let strings = Strings { ver_ext: ctx.bufs.ver_ext, hdr: ctx.bufs.hdr, main: ctx.bufs.main };
    Ok((strings, info, var_info))
}

/// `fill_interpolants` / `fill_fragment_interpolants`.
fn fill_interpolants(ctx: &Ctx<'_>, sinfo: &mut VarInfo) {
    if ctx.interp_input_mask == 0 || ctx.prog_type != Processor::Fragment {
        return;
    }
    for (i, input) in ctx.inputs.iter().enumerate() {
        if ctx.interp_input_mask & bit64(i as u32) == 0 {
            continue;
        }
        sinfo.fs_info.interps.push(InterpInfo {
            semantic_name: input.name,
            semantic_index: input.sid as u16,
            interpolate: input.interpolate,
            location: input.location,
        });
    }
}

/// `fill_var_sinfo`.
fn fill_var_sinfo(ctx: &Ctx<'_>, sinfo: &mut VarInfo) {
    sinfo.num_ucp = if ctx.is_last_vertex_stage { super::NUM_CLIP_PLANES as i32 } else { 0 };
    sinfo.fs_info.has_sample_input = ctx.has_sample_input;
    sinfo.fs_info.has_noperspective = ctx.has_noperspective;
    sinfo.fs_info.glsl_ver = ctx.glsl_ver_required;
    let has_prop = (ctx.num_clip_dist_prop + ctx.num_cull_dist_prop) > 0;
    sinfo.num_in_clip = if has_prop { ctx.num_clip_dist_prop } else { ctx.key.num_in_clip };
    sinfo.num_in_cull = if has_prop { ctx.num_cull_dist_prop } else { ctx.key.num_in_cull };
    sinfo.num_out_clip = if has_prop { ctx.num_clip_dist_prop } else { ctx.key.num_out_clip };
    sinfo.num_out_cull = if has_prop { ctx.num_cull_dist_prop } else { ctx.key.num_out_cull };
    sinfo.legacy_color_bits = ctx.color_out_mask as i32;
}

/// `fill_sinfo`.
pub(super) fn fill_sinfo(ctx: &mut Ctx<'_>, sinfo: &mut Info) {
    sinfo.use_pervertex_in = ctx.has_pervertex;
    sinfo.samplers_used_mask = ctx.samplers_used;
    sinfo.images_used_mask = ctx.images_used_mask;
    sinfo.image_binding_offset = u32::from(ctx.key.image_binding_offset);
    sinfo.image_last_binding = i32::from(ctx.key.image_binding_offset) + ctx.image_last_binding;
    sinfo.num_consts = ctx.num_consts;
    sinfo.ubo_used_mask = ctx.ubo_used_mask;
    sinfo.fog_input_mask = ctx.fog_input_mask;
    sinfo.fog_output_mask = ctx.fog_output_mask;

    let first = if ctx.ssbo_first_binding != u32::MAX { ctx.ssbo_first_binding } else { 0 };
    sinfo.ssbo_used_mask = ctx.ssbo_used_mask.wrapping_shr(first);
    sinfo.ssbo_binding_offset = u32::from(ctx.key.ssbo_binding_offset);
    sinfo.ssbo_last_binding =
        i32::from(ctx.key.ssbo_binding_offset) + ctx.ssbo_last_binding - first as i32;

    sinfo.ubo_indirect = ctx.info.is_dimension_indirect(File::Constant);

    sinfo.has_output_arrays = ctx.has_output_arrays;
    sinfo.has_input_arrays = ctx.has_input_arrays;

    sinfo.out_generic_emitted_mask = ctx.generic_ios.matched.outputs_emitted_mask;
    sinfo.out_texcoord_emitted_mask = ctx.texcoord_ios.outputs_emitted_mask as u8;
    sinfo.out_patch_emitted_mask = ctx.patches_emitted_mask;

    sinfo.num_inputs = ctx.inputs.len() as i32;
    sinfo.num_outputs = ctx.outputs.len() as i32;
    sinfo.shadow_samp_mask = ctx.shadow_samp_mask;
    sinfo.gs_out_prim = PrimType::from_wire(ctx.gs_out_prim);
    sinfo.tes_prim = PrimType::from_wire(ctx.tes_prim_mode);
    sinfo.tes_point_mode = ctx.tes_point_mode != 0;
    sinfo.fs_blend_equation_advanced = ctx.fs_blend_equation_advanced;
    sinfo.separable_program = ctx.separable_program;
    sinfo.reads_drawid = ctx.bufs.required_sysval_uniform_decls & sysval::DRAWID_BASE != 0;

    sinfo.fs_output_layout = [0; 12];
    if ctx.prog_type == Processor::Fragment {
        for (i, o) in ctx.outputs.iter().enumerate().take(sinfo.fs_output_layout.len()) {
            sinfo.fs_output_layout[i] = if o.name == Semantic::Color { o.sid as i8 } else { -1 };
        }
    }

    sinfo.so_names =
        std::mem::take(&mut ctx.so_names).into_iter().map(|n| n.unwrap_or_default()).collect();
    sinfo.attrib_input_mask = ctx.attrib_input_mask;
    sinfo.sampler_arrays = std::mem::take(&mut ctx.sampler_arrays);
    sinfo.image_arrays = std::mem::take(&mut ctx.image_arrays);
    sinfo.in_generic_emitted_mask = ctx.generic_ios.matched.inputs_emitted_mask;
    sinfo.in_texcoord_emitted_mask = ctx.texcoord_ios.inputs_emitted_mask;

    for o in &ctx.outputs {
        if o.invariant {
            let bit_pos = varying_bit_from_semantic_and_index(o.name, o.sid);
            let slot = (bit_pos / 32) as usize;
            if slot < sinfo.invariant_outputs.len() {
                sinfo.invariant_outputs[slot] |= 1u32 << (bit_pos & 0x1f);
            }
        }
    }
    sinfo.gles_use_tex_query_level = ctx.gles_use_tex_query_level;

    if ctx.guest_sent_io_arrays {
        sinfo.output_arrays = IoArrayInfo::default();
        for io in &ctx.outputs {
            if io.array_id > 0 && sinfo.output_arrays.layout.len() < 16 {
                sinfo.output_arrays.layout.push(IoArray {
                    sid: io.sid,
                    size: io.last - io.first,
                    name: io.name,
                    array_id: io.array_id,
                });
            }
        }
    }
}

/// `emit_required_sysval_uniforms`: the `VirglBlock` uniform block, whole, when any member is
/// read.
pub(super) fn emit_required_sysval_uniforms(bufs: &mut Bufs) {
    if bufs.required_sysval_uniform_decls == 0 {
        return;
    }
    bufs.hdr("layout (std140) uniform VirglBlock {\n");
    bufs.hdr("\tvec4 clipp[8];\n");
    hdr!(bufs, "\tuint stipple_pattern[{}];\n", POLYGON_STIPPLE_SIZE);
    bufs.hdr("\tfloat winsys_adjust_y;\n");
    bufs.hdr("\tfloat alpha_ref_val;\n");
    bufs.hdr("\tbool clip_plane_enabled;\n");
    bufs.hdr("\tint drawid_base;\n");
    bufs.hdr("};\n");
}

/// `vrend_shader_create_passthrough_tcs`: a tessellation control shader that copies the
/// vertex shader's outputs through, for a pipeline that has an evaluation shader but no
/// control shader of its own.
pub fn create_passthrough_tcs(
    cfg: &Cfg,
    vs: &Shader,
    key: &Key,
    tess_factors: &[f32; 6],
    vertices_per_patch: u8,
) -> Result<(Strings, Info), Failure> {
    header::passthrough_tcs(cfg, vs, key, tess_factors, vertices_per_patch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vrend::tgsi::fixture;

    /// The host's configuration when the corpus was recorded: GLES 3.1 on zink over
    /// KosmicKrisp, as `vrend_renderer.c` fills `shader_cfg`.
    pub(crate) fn corpus_cfg() -> Cfg {
        Cfg {
            glsl_version: 310,
            max_draw_buffers: 8,
            max_shader_patch_varyings: 30,
            has_arrays_of_arrays: true,
            has_gpu_shader5: true,
            has_es31_compat: true,
            has_conservative_depth: false,
            has_dual_src_blend: true,
            has_fbfetch_coherent: false,
            has_cull_distance: true,
            has_nopersective: false,
            has_texture_shadow_lod: false,
            has_vs_layer: false,
            has_vs_viewport_index: false,
        }
    }

    fn translate(block: &fixture::Block, key: &Key) -> String {
        let shader = tgsi::text::parse(block.tgsi.as_bytes(), u32::MAX).expect("the corpus parses");
        let program = tgsi::Program::scan(shader).expect("the corpus scans");
        let (strings, _, _) = convert(&corpus_cfg(), &program, 0, key, &StreamOutput::default())
            .unwrap_or_else(|e| panic!("{e}"));
        strings.source()
    }

    /// The C recorded each translation without its key. What the key contributed in this
    /// corpus is the interface matching between the vertex and fragment stages -- the
    /// generics each expects of the other, and how the fragment stage wants them
    /// interpolated -- and the C's header names it: an interpolation qualifier on a generic
    /// is the fragment stage's, and a generic declared for the other stage's sake is one it
    /// expected. This reads those back into a key.
    fn key_from_glsl(processor: Processor, glsl: &str) -> Key {
        let mut key = Key::default();
        for line in glsl.lines() {
            let words: Vec<&str> = line.split_whitespace().collect();
            let Some(name) = words.last().and_then(|w| w.strip_suffix(';')) else {
                continue;
            };
            let Some(sid) = name.strip_prefix("vso_g").and_then(|n| n.parse::<u32>().ok()) else {
                continue;
            };
            let interpolate = if words.contains(&"smooth") {
                Some(Interpolate::Perspective)
            } else if words.contains(&"flat") {
                Some(Interpolate::Constant)
            } else {
                None
            };
            match processor {
                Processor::Vertex => {
                    key.out_generic_expected_mask |= 1 << sid;
                    if let Some(interpolate) = interpolate {
                        key.fs_info.interps.push(InterpInfo {
                            semantic_name: Semantic::Generic,
                            semantic_index: sid as u16,
                            interpolate,
                            location: Location::Center,
                        });
                    }
                }
                Processor::Fragment => key.in_generic_expected_mask |= 1 << sid,
                _ => {}
            }
        }
        key
    }

    /// Every block of the corpus translates to the C's GLSL, byte for byte.
    #[test]
    fn every_corpus_shader_translates_to_the_c_glsl() {
        let mut failures = Vec::new();
        for (i, block) in fixture::blocks().iter().enumerate() {
            let shader =
                tgsi::text::parse(block.tgsi.as_bytes(), u32::MAX).expect("the corpus parses");
            let key = key_from_glsl(shader.processor, block.glsl);
            let glsl = translate(block, &key);
            if glsl != block.glsl {
                failures.push(i);
                if std::env::var_os("VIRGLRS_TEST_VERBOSE").is_some() {
                    eprintln!("--- shader {i}: C\n{}--- Rust\n{glsl}--- end", block.glsl);
                }
            }
        }
        assert!(failures.is_empty(), "shaders differing from the C: {failures:?}");
    }
}
