// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The classic capsets, `virgl_caps_v1` and `virgl_caps_v2`: what the guest's virgl driver
//! learns about this host before it draws anything. The guest reads them back by offset, so the
//! structs are `virgl_hw.h`'s to the byte -- `Pod` refuses padding at compile time, and
//! `harness/abi/layout.txt` pins the offsets the C compiler gives them.
//!
//! Filled once, at init, from the same probes the rest of vrend runs on: the C queries GL on
//! every `fill_caps`, but nothing it asks changes after the context is up.

use bytemuck::{Pod, Zeroable};

use super::features::{Feature, Features};
use super::formats::Table;
use super::gl::Gl;
use super::gl::gles::*;
use super::gl::{GLenum, GLsizei, GLuint};
use super::pipe::slots;
use super::pipe::{PrimType, ShaderStage};
use super::proto::{FORMAT_MAX, Format};
use super::resource::Limits;
use super::video;
use crate::videotoolbox;

/// `virgl_supported_format_mask`: one bit per wire format, 512 of them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct FormatMask {
    pub bitmask: [u32; 16],
}

impl CapsV1 {
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl FormatMask {
    pub fn set(&mut self, format: Format) {
        let raw = format.wire();
        self.bitmask[(raw / 32) as usize] |= 1 << (raw % 32);
    }

    pub fn has(&self, format: Format) -> bool {
        let raw = format.wire();
        self.bitmask[(raw / 32) as usize] & (1 << (raw % 32)) != 0
    }
}

/// `virgl_caps_bool_set1`, as the bit each member takes in its word. The C spells it as
/// one-bit bitfields, which the target ABI packs from bit 0 in declaration order.
pub mod bset {
    pub const INDEP_BLEND_ENABLE: u32 = 1 << 0;
    pub const INDEP_BLEND_FUNC: u32 = 1 << 1;
    pub const CUBE_MAP_ARRAY: u32 = 1 << 2;
    pub const SHADER_STENCIL_EXPORT: u32 = 1 << 3;
    pub const CONDITIONAL_RENDER: u32 = 1 << 4;
    pub const START_INSTANCE: u32 = 1 << 5;
    pub const PRIMITIVE_RESTART: u32 = 1 << 6;
    pub const BLEND_EQ_SEP: u32 = 1 << 7;
    pub const INSTANCEID: u32 = 1 << 8;
    pub const VERTEX_ELEMENT_INSTANCE_DIVISOR: u32 = 1 << 9;
    pub const SEAMLESS_CUBE_MAP: u32 = 1 << 10;
    pub const OCCLUSION_QUERY: u32 = 1 << 11;
    pub const TIMER_QUERY: u32 = 1 << 12;
    pub const STREAMOUT_PAUSE_RESUME: u32 = 1 << 13;
    pub const TEXTURE_MULTISAMPLE: u32 = 1 << 14;
    pub const FRAGMENT_COORD_CONVENTIONS: u32 = 1 << 15;
    pub const DEPTH_CLIP_DISABLE: u32 = 1 << 16;
    pub const SEAMLESS_CUBE_MAP_PER_TEXTURE: u32 = 1 << 17;
    pub const UBO: u32 = 1 << 18;
    pub const COLOR_CLAMPING: u32 = 1 << 19;
    pub const POLY_STIPPLE: u32 = 1 << 20;
    pub const MIRROR_CLAMP: u32 = 1 << 21;
    pub const TEXTURE_QUERY_LOD: u32 = 1 << 22;
    pub const HAS_FP64: u32 = 1 << 23;
    pub const HAS_TESSELLATION_SHADERS: u32 = 1 << 24;
    pub const HAS_INDIRECT_DRAW: u32 = 1 << 25;
    pub const HAS_SAMPLE_SHADING: u32 = 1 << 26;
    pub const HAS_CULL: u32 = 1 << 27;
    pub const CONDITIONAL_RENDER_INVERTED: u32 = 1 << 28;
    pub const DERIVATIVE_CONTROL: u32 = 1 << 29;
    pub const POLYGON_OFFSET_CLAMP: u32 = 1 << 30;
    pub const TRANSFORM_FEEDBACK_OVERFLOW_QUERY: u32 = 1 << 31;
}

/// `VIRGL_CAP_*`: `capability_bits`.
pub mod cap {
    pub const TGSI_INVARIANT: u32 = 1 << 0;
    pub const TEXTURE_VIEW: u32 = 1 << 1;
    pub const SET_MIN_SAMPLES: u32 = 1 << 2;
    pub const COPY_IMAGE: u32 = 1 << 3;
    pub const TGSI_PRECISE: u32 = 1 << 4;
    pub const TXQS: u32 = 1 << 5;
    pub const MEMORY_BARRIER: u32 = 1 << 6;
    pub const COMPUTE_SHADER: u32 = 1 << 7;
    pub const FB_NO_ATTACH: u32 = 1 << 8;
    pub const ROBUST_BUFFER_ACCESS: u32 = 1 << 9;
    pub const TGSI_FBFETCH: u32 = 1 << 10;
    pub const SHADER_CLOCK: u32 = 1 << 11;
    pub const TEXTURE_BARRIER: u32 = 1 << 12;
    pub const TGSI_COMPONENTS: u32 = 1 << 13;
    pub const GUEST_MAY_INIT_LOG: u32 = 1 << 14;
    pub const SRGB_WRITE_CONTROL: u32 = 1 << 15;
    pub const QBO: u32 = 1 << 16;
    pub const TRANSFER: u32 = 1 << 17;
    pub const FBO_MIXED_COLOR_FORMATS: u32 = 1 << 18;
    pub const HOST_IS_GLES: u32 = 1 << 19;
    pub const BIND_COMMAND_ARGS: u32 = 1 << 20;
    pub const MULTI_DRAW_INDIRECT: u32 = 1 << 21;
    pub const INDIRECT_PARAMS: u32 = 1 << 22;
    pub const TRANSFORM_FEEDBACK3: u32 = 1 << 23;
    pub const ASTC_3D: u32 = 1 << 24;
    pub const INDIRECT_INPUT_ADDR: u32 = 1 << 25;
    pub const COPY_TRANSFER: u32 = 1 << 26;
    pub const CLIP_HALFZ: u32 = 1 << 27;
    pub const APP_TWEAK_SUPPORT: u32 = 1 << 28;
    pub const BGRA_SRGB_IS_EMULATED: u32 = 1 << 29;
    pub const CLEAR_TEXTURE: u32 = 1 << 30;
    pub const ARB_BUFFER_STORAGE: u32 = 1 << 31;
}

/// `VIRGL_CAP_V2_*`: `capability_bits_v2`.
pub mod cap2 {
    pub const BLEND_EQUATION: u32 = 1 << 0;
    pub const UNTYPED_RESOURCE: u32 = 1 << 1;
    pub const VIDEO_MEMORY: u32 = 1 << 2;
    pub const MEMINFO: u32 = 1 << 3;
    pub const STRING_MARKER: u32 = 1 << 4;
    pub const DIFFERENT_GPU: u32 = 1 << 5;
    pub const IMPLICIT_MSAA: u32 = 1 << 6;
    pub const COPY_TRANSFER_BOTH_DIRECTIONS: u32 = 1 << 7;
    pub const SCANOUT_USES_GBM: u32 = 1 << 8;
    pub const SSO: u32 = 1 << 9;
    pub const TEXTURE_SHADOW_LOD: u32 = 1 << 10;
    pub const VS_VERTEX_LAYER: u32 = 1 << 11;
    pub const VS_VIEWPORT_INDEX: u32 = 1 << 12;
    pub const PIPELINE_STATISTICS_QUERY: u32 = 1 << 13;
    pub const DRAW_PARAMETERS: u32 = 1 << 14;
    pub const GROUP_VOTE: u32 = 1 << 15;
    pub const MIRROR_CLAMP_TO_EDGE: u32 = 1 << 16;
    pub const MIRROR_CLAMP: u32 = 1 << 17;
    pub const RESOURCE_LAYOUT: u32 = 1 << 18;
    /// The guest may back a decode target's planes with its own memory, which is what lets it
    /// export the decoded frame as a dmabuf rather than only sample it.
    pub const VIDEO_GUEST_PLANES: u32 = 1 << 19;
    /// The guest may hand over one composite planar resource as a decode target, instead of one
    /// resource per plane.
    pub const VIDEO_PLANAR_TARGET: u32 = 1 << 20;
}

/// `virgl_caps_v1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct CapsV1 {
    pub max_version: u32,
    pub sampler: FormatMask,
    pub render: FormatMask,
    pub depthstencil: FormatMask,
    pub vertexbuffer: FormatMask,
    pub bset: u32,
    pub glsl_level: u32,
    pub max_texture_array_layers: u32,
    pub max_streamout_buffers: u32,
    pub max_dual_source_render_targets: u32,
    pub max_render_targets: u32,
    pub max_samples: u32,
    pub prim_mask: u32,
    pub max_tbo_size: u32,
    pub max_uniform_blocks: u32,
    pub max_viewports: u32,
    pub max_texture_gather_components: u32,
}

/// `virgl_video_caps`: four words of bitfields.
///
/// Carried as words because that is what crosses to the guest, and built only by
/// [`VideoCaps::decode`], so the packing lives in one place and a caller names fields rather than
/// shifts. A bitfield struct written by hand at four call sites is four chances to put a level in
/// the width.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct VideoCaps {
    pub words: [u32; 4],
}

impl VideoCaps {
    /// The ceiling this build advertises for every profile. VideoToolbox goes higher on current
    /// silicon and nothing we can query says by how much; the guest only uses it to size its
    /// surfaces, and 4K fits comfortably below.
    const MAX_WIDTH: u32 = 4096;
    const MAX_HEIGHT: u32 = 4096;

    /// `PIPE_FORMAT_NV12`. The layout the guest is told to prefer -- it may allocate another,
    /// and the decode session is built around whatever it chose.
    const PREFERRED_FORMAT: u32 = 166;

    /// One advertised decode entry.
    ///
    /// Everything this build does not vary is fixed here rather than repeated at call sites:
    /// bitstream is the only entrypoint, nothing is interlaced (the serializers refuse field
    /// coding), and there are no stacked frames or temporal layers.
    pub fn decode(profile: video::Profile, max_level: u32) -> VideoCaps {
        /// `PIPE_VIDEO_ENTRYPOINT_BITSTREAM`.
        const BITSTREAM: u32 = 1;
        VideoCaps {
            words: [
                (profile as u32 & 0xff) | (BITSTREAM << 8) | ((max_level & 0xff) << 16),
                VideoCaps::MAX_WIDTH | (VideoCaps::MAX_HEIGHT << 16),
                VideoCaps::PREFERRED_FORMAT | (1 << 16), // max_macroblocks
                1 | (1 << 1),                            // npot_texture, supports_progressive
            ],
        }
    }
}

/// `virgl_caps_v2`. `v1` is its head, so a caller asking for capset 1 reads the same bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CapsV2 {
    pub v1: CapsV1,
    pub min_aliased_point_size: f32,
    pub max_aliased_point_size: f32,
    pub min_smooth_point_size: f32,
    pub max_smooth_point_size: f32,
    pub min_aliased_line_width: f32,
    pub max_aliased_line_width: f32,
    pub min_smooth_line_width: f32,
    pub max_smooth_line_width: f32,
    pub max_texture_lod_bias: f32,
    pub max_geom_output_vertices: u32,
    pub max_geom_total_output_components: u32,
    pub max_vertex_outputs: u32,
    pub max_vertex_attribs: u32,
    pub max_shader_patch_varyings: u32,
    pub min_texel_offset: i32,
    pub max_texel_offset: i32,
    pub min_texture_gather_offset: i32,
    pub max_texture_gather_offset: i32,
    pub texture_buffer_offset_alignment: u32,
    pub uniform_buffer_offset_alignment: u32,
    pub shader_buffer_offset_alignment: u32,
    pub capability_bits: u32,
    pub sample_locations: [u32; 8],
    pub max_vertex_attrib_stride: u32,
    pub max_shader_buffer_frag_compute: u32,
    pub max_shader_buffer_other_stages: u32,
    pub max_shader_image_frag_compute: u32,
    pub max_shader_image_other_stages: u32,
    pub max_image_samples: u32,
    pub max_compute_work_group_invocations: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_compute_grid_size: [u32; 3],
    pub max_compute_block_size: [u32; 3],
    pub max_texture_2d_size: u32,
    pub max_texture_3d_size: u32,
    pub max_texture_cube_size: u32,
    pub max_combined_shader_buffers: u32,
    pub max_atomic_counters: [u32; ShaderStage::COUNT],
    pub max_atomic_counter_buffers: [u32; ShaderStage::COUNT],
    pub max_combined_atomic_counters: u32,
    pub max_combined_atomic_counter_buffers: u32,
    pub host_feature_check_version: u32,
    pub supported_readback_formats: FormatMask,
    pub scanout: FormatMask,
    pub capability_bits_v2: u32,
    pub max_video_memory: u32,
    pub renderer: [u8; 64],
    pub max_anisotropy: f32,
    pub max_texture_samplers: u32,
    pub supported_multisample_formats: FormatMask,
    pub max_const_buffer_size: [u32; ShaderStage::COUNT],
    pub num_video_caps: u32,
    pub video_caps: [VideoCaps; 32],
    pub max_uniform_block_size: u32,
    pub max_tcs_outputs: u32,
    pub max_tes_outputs: u32,
    pub max_shader_storage_blocks: [u32; ShaderStage::COUNT],
}

/// `VREND_CAPSET_VIRGL_MAX_VERSION` and `VREND_CAPSET_VIRGL2_MAX_VERSION`.
pub const VIRGL_VERSION: u32 = 1;
pub const VIRGL2_VERSION: u32 = 2;

/// What the guest is told it has is what the decoder admits and what the state can hold: the
/// gallium slot caps, as a caps field's `u32`.
const PIPE_MAX_SHADER_BUFFERS: u32 = slots::MAX_SHADER_BUFFERS as u32;
const PIPE_MAX_SHADER_IMAGES: u32 = slots::MAX_SHADER_IMAGES as u32;
const PIPE_MAX_SAMPLERS: u32 = slots::MAX_SAMPLERS as u32;
const MAX_COMBINED_SSBO_BINDING_POINTS: u32 = slots::MAX_COMBINED_SSBO_BINDING_POINTS;

impl CapsV2 {
    /// The bytes a caller asking for capset 2 reads.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// What a caller asking for capset 1 reads: the head, saying it is version 1.
    pub fn v1(&self) -> CapsV1 {
        CapsV1 { max_version: VIRGL_VERSION, ..self.v1 }
    }

    /// `vrend_renderer_fill_caps` for a GLES host, ctx0 current: `vrend_fill_caps_glsl_version`,
    /// `vrend_renderer_fill_caps_v1` and `vrend_renderer_fill_caps_v2`, with the desktop-GL
    /// branches dropped -- this host is GLES, and every `gl_ver > 0` leg of the C is dead here.
    pub fn probe(
        gl: &Gl,
        features: &Features,
        limits: &Limits,
        formats: &Table,
        video_support: Option<&videotoolbox::Support>,
    ) -> CapsV2 {
        let has = |f: Feature| features.has(f);
        let get = |name: GLenum| gl.get_integer(name);
        let getu = |name: GLenum| gl.get_integer(name).max(0) as u32;
        let gles = features.gles_version;
        let mut c = CapsV2::zeroed();
        c.v1.max_version = VIRGL2_VERSION;

        // vrend_fill_caps_glsl_version
        c.v1.glsl_level = if gles >= 31 {
            310
        } else if gles >= 30 {
            130
        } else {
            120
        };
        if has(Feature::tessellation) && has(Feature::geometry_shader) && has(Feature::gpu_shader5)
        {
            // The C's own words: probably a lie, but gallium turns on OES_geometry_shader and
            // ARB_gpu_shader5 from it, and compute needs 430 unless the shader asks by name.
            c.v1.glsl_level = 400;
            if has(Feature::separate_shader_objects) {
                c.v1.glsl_level = if has(Feature::compute_shader) { 430 } else { 410 };
            }
        }

        // vrend_renderer_fill_caps_v1
        let v1 = &mut c.v1;
        v1.bset |= bset::OCCLUSION_QUERY;
        v1.prim_mask = [
            PrimType::Points,
            PrimType::Lines,
            PrimType::LineStrip,
            PrimType::LineLoop,
            PrimType::Triangles,
            PrimType::TriangleStrip,
            PrimType::TriangleFan,
        ]
        .iter()
        .fold(0, |m, p| m | (1 << p.wire()));
        if v1.glsl_level >= 150 {
            for p in [
                PrimType::LinesAdjacency,
                PrimType::LineStripAdjacency,
                PrimType::TrianglesAdjacency,
                PrimType::TriangleStripAdjacency,
            ] {
                v1.prim_mask |= 1 << p.wire();
            }
        }
        if v1.glsl_level >= 400 || has(Feature::tessellation) {
            v1.prim_mask |= 1 << PrimType::Patches.wire();
        }
        if features.has_extension("GL_ARB_vertex_type_10f_11f_11f_rev") {
            v1.vertexbuffer.set(Format::from_wire(R11G11B10_FLOAT).expect("a format"));
        }
        let mut bit = |on: bool, b: u32| {
            if on {
                v1.bset |= b;
            }
        };
        bit(
            has(Feature::nv_conditional_render) || has(Feature::gl_conditional_render),
            bset::CONDITIONAL_RENDER,
        );
        bit(has(Feature::indep_blend), bset::INDEP_BLEND_ENABLE);
        bit(has(Feature::draw_instance), bset::INSTANCEID);
        bit(has(Feature::depth_clamp), bset::DEPTH_CLIP_DISABLE);
        bit(
            features.has_extension("GL_ARB_fragment_coord_conventions"),
            bset::FRAGMENT_COORD_CONVENTIONS,
        );
        bit(
            features.has_extension("GL_ARB_seamless_cube_map") || gles >= 30,
            bset::SEAMLESS_CUBE_MAP,
        );
        bit(has(Feature::seamless_cubemap_per_texture), bset::SEAMLESS_CUBE_MAP_PER_TEXTURE);
        bit(has(Feature::texture_multisample), bset::TEXTURE_MULTISAMPLE);
        bit(has(Feature::tessellation), bset::HAS_TESSELLATION_SHADERS);
        bit(has(Feature::sample_shading), bset::HAS_SAMPLE_SHADING);
        bit(has(Feature::indirect_draw), bset::HAS_INDIRECT_DRAW);
        bit(has(Feature::indep_blend_func), bset::INDEP_BLEND_FUNC);
        bit(has(Feature::cube_map_array), bset::CUBE_MAP_ARRAY);
        bit(has(Feature::texture_query_lod), bset::TEXTURE_QUERY_LOD);
        bit(
            features.has_extension("GL_ARB_gpu_shader_fp64")
                && features.has_extension("GL_ARB_gpu_shader5"),
            bset::HAS_FP64,
        );
        bit(has(Feature::base_instance), bset::START_INSTANCE);
        bit(features.has_extension("GL_ARB_shader_stencil_export"), bset::SHADER_STENCIL_EXPORT);
        bit(has(Feature::conditional_render_inverted), bset::CONDITIONAL_RENDER_INVERTED);
        bit(has(Feature::cull_distance), bset::HAS_CULL);
        bit(features.has_extension("GL_ARB_derivative_control"), bset::DERIVATIVE_CONTROL);
        bit(has(Feature::polygon_offset_clamp), bset::POLYGON_OFFSET_CLAMP);
        bit(
            has(Feature::transform_feedback_overflow_query),
            bset::TRANSFORM_FEEDBACK_OVERFLOW_QUERY,
        );
        bit(
            has(Feature::texture_mirror_clamp) || has(Feature::texture_mirror_clamp_to_edge),
            bset::MIRROR_CLAMP,
        );
        bit(has(Feature::timer_query), bset::TIMER_QUERY);
        bit(
            has(Feature::nv_prim_restart) || has(Feature::gl_prim_restart),
            bset::PRIMITIVE_RESTART,
        );
        if has(Feature::ubo) {
            // GL's count omits the ordinary uniform block, add it; less the one the VirglBlock
            // helper may take.
            v1.max_uniform_blocks = getu(GL_MAX_VERTEX_UNIFORM_BLOCKS) + 1 - 1;
        }
        if has(Feature::texture_array) {
            v1.max_texture_array_layers = getu(GL_MAX_ARRAY_TEXTURE_LAYERS);
        }
        if has(Feature::transform_feedback) {
            if has(Feature::transform_feedback2) {
                v1.bset |= bset::STREAMOUT_PAUSE_RESUME;
            }
            if has(Feature::transform_feedback3) {
                v1.max_streamout_buffers = getu(GL_MAX_TRANSFORM_FEEDBACK_BUFFERS);
            } else if get(GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS) >= 4 {
                // As with the earlier transform feedback, this is at least four.
                v1.max_streamout_buffers = 4;
            }
        }
        if has(Feature::dual_src_blend) {
            v1.max_dual_source_render_targets = getu(GL_MAX_DUAL_SOURCE_DRAW_BUFFERS);
        }
        if has(Feature::arb_or_gles_ext_texture_buffer) {
            v1.max_tbo_size = limits.max_texture_buffer_size;
        }
        if has(Feature::texture_gather) {
            v1.max_texture_gather_components = 4;
        }
        v1.max_viewports = if has(Feature::viewport_array) { getu(GL_MAX_VIEWPORTS) } else { 1 };
        v1.max_render_targets = limits.max_draw_buffers;
        v1.max_samples = getu(GL_MAX_SAMPLES);
        let mut planar_target = false;
        for raw in 0..FORMAT_MAX {
            let format = Format::from_wire(raw).expect("below FORMAT_MAX");
            let Some(entry) = formats.get(format) else { continue };
            // A multi-plane format is samplable only as a composite decode target, and only
            // where this build can back one. The guest reads the bit as "I may create the
            // composite shape" and cannot survive being told yes and then refused.
            if video::guest_planes(format) > 1
                && !video::composite_target_backable(features, format)
            {
                continue;
            }
            if entry.bindings.sampler_view {
                v1.sampler.set(format);
                // What the capset's planar-target bit is about: not that some layout could be
                // backed, but that one was actually offered here. Read off the bit that offers
                // it rather than stated a second time, so the flag cannot outlive the format.
                planar_target |= video::guest_planes(format) > 1;
                if entry.can_render() {
                    v1.render.set(format);
                }
            }
        }

        // vrend_renderer_fill_caps_v2
        c.host_feature_check_version = 23;
        let renderer = gl.get_string(GL_RENDERER);
        // glamor rejects llvmpipe by name, and the guest's renderer string is composed from ours.
        let shown = renderer.replace("llvmpipe", "LLVMPIPE");
        let n = shown.len().min(c.renderer.len() - 1);
        c.renderer[..n].copy_from_slice(&shown.as_bytes()[..n]);

        let range = gl.get_float_range(GL_ALIASED_POINT_SIZE_RANGE);
        c.min_aliased_point_size = range[0];
        c.max_aliased_point_size = range[1];
        let range = gl.get_float_range(GL_ALIASED_LINE_WIDTH_RANGE);
        c.min_aliased_line_width = range[0];
        c.max_aliased_line_width = range[1];
        c.max_texture_lod_bias = gl.get_float(GL_MAX_TEXTURE_LOD_BIAS);
        c.max_vertex_attribs = limits.max_vertex_attributes;
        // The GL minimum where the query does not exist.
        let outputs = if gles >= 30 { get(GL_MAX_VERTEX_OUTPUT_COMPONENTS) } else { 64 };
        c.max_vertex_outputs = (outputs / 4).max(0) as u32;
        c.min_texel_offset = get(GL_MIN_PROGRAM_TEXEL_OFFSET);
        c.max_texel_offset = get(GL_MAX_PROGRAM_TEXEL_OFFSET);
        c.uniform_buffer_offset_alignment = getu(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT);
        c.max_texture_2d_size = limits.max_texture_2d_size;
        c.max_texture_3d_size = limits.max_texture_3d_size;
        c.max_texture_cube_size = limits.max_texture_cube_size;
        if has(Feature::geometry_shader) {
            c.max_geom_output_vertices = getu(GL_MAX_GEOMETRY_OUTPUT_VERTICES);
            c.max_geom_total_output_components = getu(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS);
        }
        if has(Feature::tessellation) {
            c.max_shader_patch_varyings = getu(GL_MAX_TESS_PATCH_COMPONENTS) / 4;
        }
        if has(Feature::texture_gather) {
            c.min_texture_gather_offset = get(GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET);
            c.max_texture_gather_offset = get(GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET);
        }
        if has(Feature::texture_buffer_range) {
            c.texture_buffer_offset_alignment = getu(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT);
        }
        if has(Feature::ssbo) {
            c.shader_buffer_offset_alignment = getu(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT);
            c.max_shader_storage_blocks = per_stage(
                gl,
                features,
                [
                    GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS,
                    GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,
                    GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS,
                    GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS,
                    GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS,
                    GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
                ],
            )
            .map(|n| n.min(PIPE_MAX_SHADER_BUFFERS));
            c.max_shader_buffer_other_stages =
                c.max_shader_storage_blocks[ShaderStage::Vertex.index()];
            c.max_shader_buffer_frag_compute =
                c.max_shader_storage_blocks[ShaderStage::Fragment.index()];
            // The binding points are a 32-bit mask, and must cover every stage combined.
            c.max_combined_shader_buffers =
                getu(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS).min(MAX_COMBINED_SSBO_BINDING_POINTS);
        }
        if has(Feature::images) {
            c.max_shader_image_other_stages =
                getu(GL_MAX_VERTEX_IMAGE_UNIFORMS).min(PIPE_MAX_SHADER_IMAGES);
            c.max_shader_image_frag_compute =
                getu(GL_MAX_FRAGMENT_IMAGE_UNIFORMS).min(PIPE_MAX_SHADER_IMAGES);
            // GLES has no multisample images: `max_image_samples` stays zero.
        }
        // Before the probe and not after it, so the counts we advertise and the positions we
        // publish for them come out of one pass: the probe skips every count above what it is
        // handed, so it writes positions for exactly the counts that survive. Clamping
        // afterwards would need the positions blanked to match, which also blanks the counts
        // still under the ceiling -- advertising a mode with no layout for it.
        c.v1.max_samples = capped_samples(c.v1.max_samples, sample_ceiling());
        if has(Feature::storage_multisample) {
            c.v1.max_samples =
                query_multisample_caps(gl, c.v1.max_samples, &mut c.sample_locations);
        }
        c.capability_bits |=
            cap::TGSI_INVARIANT | cap::SET_MIN_SAMPLES | cap::TGSI_PRECISE | cap::APP_TWEAK_SUPPORT;
        // Without the query, the specification's minimum.
        c.max_vertex_attrib_stride =
            if gles >= 31 { getu(GL_MAX_VERTEX_ATTRIB_STRIDE) } else { 2048 };
        if has(Feature::compute_shader) {
            c.max_compute_work_group_invocations = getu(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS);
            c.max_compute_shared_memory_size = getu(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE);
            for i in 0..3 {
                c.max_compute_grid_size[i as usize] =
                    gl.get_integer_i(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i).max(0) as u32;
                c.max_compute_block_size[i as usize] =
                    gl.get_integer_i(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i).max(0) as u32;
            }
            c.capability_bits |= cap::COMPUTE_SHADER;
        }
        if has(Feature::atomic_counters) {
            // The per-stage counters are desktop-only: on a GLES host atomics are lowered to
            // SSBOs. The counter buffers are queried on both.
            c.max_atomic_counter_buffers = per_stage(
                gl,
                features,
                [
                    GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS,
                    GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS,
                    GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS,
                    GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS,
                    GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS,
                    GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS,
                ],
            );
            if has(Feature::tessellation) {
                c.max_tcs_outputs = getu(GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS) / 4;
                c.max_tes_outputs = getu(GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS) / 4;
            }
            c.max_combined_atomic_counter_buffers = getu(GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS);
        }
        let mut capbit = |on: bool, b: u32| {
            if on {
                c.capability_bits |= b;
            }
        };
        capbit(has(Feature::fb_no_attach), cap::FB_NO_ATTACH);
        capbit(has(Feature::texture_view), cap::TEXTURE_VIEW);
        capbit(has(Feature::txqs), cap::TXQS);
        capbit(has(Feature::barrier), cap::MEMORY_BARRIER);
        capbit(has(Feature::copy_image), cap::COPY_IMAGE);
        capbit(has(Feature::robust_buffer_access), cap::ROBUST_BUFFER_ACCESS);
        capbit(has(Feature::framebuffer_fetch), cap::TGSI_FBFETCH);
        capbit(has(Feature::shader_clock), cap::SHADER_CLOCK);
        capbit(has(Feature::texture_barrier), cap::TEXTURE_BARRIER);
        capbit(true, cap::TGSI_COMPONENTS);
        capbit(has(Feature::srgb_write_control), cap::SRGB_WRITE_CONTROL);
        capbit(has(Feature::transform_feedback3), cap::TRANSFORM_FEEDBACK3);
        // Only says the command is served.
        capbit(true, cap::GUEST_MAY_INIT_LOG);
        capbit(has(Feature::qbo), cap::QBO);
        capbit(true, cap::TRANSFER);
        capbit(mixed_color_attachments_work(gl), cap::FBO_MIXED_COLOR_FORMATS);
        // ARB_gpu_shader_fp64 is exposed on top of ES.
        capbit(true, cap::HOST_IS_GLES);
        capbit(has(Feature::indirect_draw), cap::BIND_COMMAND_ARGS);
        capbit(has(Feature::multi_draw_indirect), cap::MULTI_DRAW_INDIRECT);
        capbit(has(Feature::indirect_params), cap::INDIRECT_PARAMS);
        capbit(has(Feature::clear_texture), cap::CLEAR_TEXTURE);
        capbit(has(Feature::clip_control), cap::CLIP_HALFZ);
        capbit(features.has_extension("GL_KHR_texture_compression_astc_sliced_3d"), cap::ASTC_3D);
        capbit(true, cap::INDIRECT_INPUT_ADDR);
        capbit(true, cap::COPY_TRANSFER);
        // ARB_BUFFER_STORAGE is advertised only where the C's caching heuristics name a GPU
        // whose mapping type it knows -- Mesa on Intel or AMD, or the one Nvidia it guessed at.
        // zink is Mesa too and matches none of them, and the C deliberately leaves the cap off
        // there: it would move the guest onto persistent host-visible buffers wholesale.
        if has(Feature::arb_buffer_storage) {
            let vendor = gl.get_string(GL_VENDOR);
            let is_mesa = renderer.contains("Mesa")
                || renderer.contains("DRM")
                || renderer.contains("llvmpipe");
            let known = if is_mesa {
                vendor.contains("Intel") || vendor.contains("AMD") || vendor.contains("Mesa")
            } else {
                renderer.contains("Quadro K2200")
            };
            capbit(known, cap::ARB_BUFFER_STORAGE);
        }
        for raw in 0..FORMAT_MAX {
            let format = Format::from_wire(raw).expect("below FORMAT_MAX");
            if let Some(entry) = formats.get(format) {
                if entry.can_readback {
                    c.supported_readback_formats.set(format);
                }
                if entry.can_multisample {
                    c.supported_multisample_formats.set(format);
                }
            }
            // Without GBM the C answers "scanout" for every format.
            c.scanout.set(format);
        }
        // For framebuffer_no_attachment.
        c.supported_multisample_formats.set(Format::NONE);

        let mut cap2bit = |on: bool, b: u32| {
            if on {
                c.capability_bits_v2 |= b;
            }
        };
        cap2bit(has(Feature::blend_equation_advanced), cap2::BLEND_EQUATION);
        // The winsys is EGL.
        cap2bit(true, cap2::UNTYPED_RESOURCE);
        // No GLX to ask for video memory, and NVX_gpu_memory_info is desktop-only.
        cap2bit(has(Feature::ati_meminfo) || has(Feature::nvx_gpu_memory_info), cap2::MEMINFO);
        cap2bit(has(Feature::khr_debug), cap2::STRING_MARKER);
        cap2bit(has(Feature::implicit_msaa), cap2::IMPLICIT_MSAA);
        cap2bit(has(Feature::texture_shadow_lod), cap2::TEXTURE_SHADOW_LOD);
        // Transfers of either direction may copy: the resource's size reaches the renderer on
        // the DRM path, which is the only path here.
        cap2bit(true, cap2::COPY_TRANSFER_BOTH_DIRECTIONS);
        cap2bit(has(Feature::separate_shader_objects), cap2::SSO);
        cap2bit(has(Feature::vs_layer_viewport), cap2::VS_VERTEX_LAYER);
        cap2bit(has(Feature::vs_viewport_index), cap2::VS_VIEWPORT_INDEX);
        cap2bit(has(Feature::pipeline_statistics_query), cap2::PIPELINE_STATISTICS_QUERY);
        cap2bit(has(Feature::draw_parameters), cap2::DRAW_PARAMETERS);
        cap2bit(has(Feature::group_vote), cap2::GROUP_VOTE);
        cap2bit(has(Feature::texture_mirror_clamp_to_edge), cap2::MIRROR_CLAMP_TO_EDGE);
        cap2bit(has(Feature::texture_mirror_clamp), cap2::MIRROR_CLAMP);

        if has(Feature::anisotropic_filter) {
            c.max_anisotropy = gl.get_float(GL_MAX_TEXTURE_MAX_ANISOTROPY).min(16.0);
        }
        // Informs the guest's PIPE_SHADER_CAP_MAX_TEXTURE_SAMPLERS, which is what it has always
        // read it as; not GL_MAX_TEXTURE_IMAGE_UNITS.
        c.max_texture_samplers = PIPE_MAX_SAMPLERS;
        if has(Feature::ubo) {
            c.max_uniform_block_size = getu(GL_MAX_UNIFORM_BLOCK_SIZE);
        }
        // The uniform component counts, as bytes.
        c.max_const_buffer_size = per_stage(
            gl,
            features,
            [
                GL_MAX_VERTEX_UNIFORM_COMPONENTS,
                GL_MAX_FRAGMENT_UNIFORM_COMPONENTS,
                GL_MAX_GEOMETRY_UNIFORM_COMPONENTS,
                GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS,
                GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS,
                GL_MAX_COMPUTE_UNIFORM_COMPONENTS,
            ],
        )
        .map(|n| n.saturating_mul(4));
        // Only what this build both has silicon for and has a decode path for. A host that
        // advertises neither is left with num_video_caps zero and the two bits clear, and the
        // guest's virgl_get_video_param() then reports no profiles -- the driver still loads, it
        // just offers no hardware decode, which is the whole of the fallback.
        for (i, profile) in video::advertised(video_support).into_iter().enumerate() {
            let Some(slot) = c.video_caps.get_mut(i) else {
                break;
            };
            *slot = VideoCaps::decode(profile, profile.max_level());
            c.num_video_caps = i as u32 + 1;
        }
        // The composite shape: one resource in a planar format holding every plane. Both halves
        // are required and neither implies the other -- a host that can mint the surface but
        // decodes nothing has no target to offer, and a decoder whose only planar layout cannot
        // be backed has nothing to offer it in.
        if planar_target && c.num_video_caps > 0 {
            c.capability_bits_v2 |= cap2::VIDEO_PLANAR_TARGET;
        }
        // VIDEO_GUEST_PLANES stays clear, where the C sets it alongside. The C gates it on there
        // being a decoder at all, which is right for the C because it writes the frame back into
        // the guest's own memory; here it is a promise this build cannot keep. The bit tells the
        // guest that backing a decode target's planes with its own pages is worthwhile, which is
        // true only if the host writes the frame back there -- and a guest that does it against
        // a host that does not exports an honest-looking fd naming a black frame, strictly worse
        // for it than never being offered. It turns on with the writeback, not with the decoder.
        c
    }
}

/// `VIRGL_FORMAT_R11G11B10_FLOAT`.
const R11G11B10_FLOAT: u32 = 135;

/// One limit per stage, queried only for the stages this host has: vertex and fragment always,
/// the rest behind their feature. A stage the host lacks reports zero.
fn per_stage(
    gl: &Gl,
    features: &Features,
    names: [GLenum; ShaderStage::COUNT],
) -> [u32; ShaderStage::COUNT] {
    let gate = |stage: ShaderStage| match stage {
        ShaderStage::Vertex | ShaderStage::Fragment => true,
        ShaderStage::Geometry => features.has(Feature::geometry_shader),
        ShaderStage::TessCtrl | ShaderStage::TessEval => features.has(Feature::tessellation),
        ShaderStage::Compute => features.has(Feature::compute_shader),
    };
    let mut out = [0; ShaderStage::COUNT];
    for (i, name) in names.into_iter().enumerate() {
        let stage = ShaderStage::from_wire(i as u32).expect("six stages");
        if gate(stage) {
            out[stage.index()] = gl.get_integer(name).max(0) as u32;
        }
    }
    out
}

/// `vrend_check_framebuffer_mixed_color_attachements`: whether one framebuffer takes an RGBA
/// and an R8 colour attachment together.
fn mixed_color_attachments_work(gl: &Gl) -> bool {
    let tex = [gl.gen_texture(), gl.gen_texture()];
    let fb = gl.gen_framebuffer();
    gl.bind_texture(GL_TEXTURE_2D, Some(tex[0]));
    gl.tex_image_2d_null(GL_TEXTURE_2D, 0, GL_RGBA, 32, 32, GL_RGBA, GL_UNSIGNED_BYTE);
    gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
    gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Some(tex[0]), 0);
    gl.bind_texture(GL_TEXTURE_2D, Some(tex[1]));
    gl.tex_image_2d_null(GL_TEXTURE_2D, 0, GL_RED, 32, 32, GL_RED, GL_UNSIGNED_BYTE);
    gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, Some(tex[1]), 0);
    let complete = gl.check_framebuffer_status() == GL_FRAMEBUFFER_COMPLETE;
    gl.bind_framebuffer(GL_FRAMEBUFFER, None);
    gl.delete_framebuffer(fb);
    gl.delete_texture(tex[0]);
    gl.delete_texture(tex[1]);
    complete
}

/// The name limina's worker sets, and so the only name that can be read here. A different one
/// would leave the mitigation below silently absent with nothing to say so.
const MAX_SAMPLES_ENV: &str = "VREND_MAX_SAMPLES";

/// A ceiling on the sample counts we are willing to advertise, or none.
///
/// Mechanism only: the value is policy and belongs to whoever embeds us. It exists because a host
/// whose multisample path is unsafe needs a way to degrade instead of dying, and a guest cannot
/// ask for multisampling it has not been told exists -- a WebGL context created with
/// `{antialias:true}` comes back reporting `antialias:false`, which is what the specification says
/// should happen and what every browser already handles.
fn sample_ceiling() -> Option<u32> {
    parse_ceiling(std::env::var(MAX_SAMPLES_ENV).ok().as_deref()?)
}

/// What a setting of the variable means, split from reading it so it can be tested without
/// touching an environment the rest of the process shares.
fn parse_ceiling(setting: &str) -> Option<u32> {
    match setting.trim().parse::<u32>() {
        // Zero is not a sample count, and single-sampled is how "no multisampling" is spelled in
        // this field. Reading it as "no ceiling" would take the value an operator is most likely
        // to write for "none" and turn it into the opposite.
        Ok(n) => Some(n.max(1)),
        Err(_) => {
            eprintln!("[virglrs] {MAX_SAMPLES_ENV}={setting:?} is not a sample count; no ceiling");
            None
        }
    }
}

/// The advertised maximum after the ceiling, saying so when it bites.
fn capped_samples(max_samples: u32, ceiling: Option<u32>) -> u32 {
    match ceiling {
        Some(c) if max_samples > c => {
            eprintln!(
                "[virglrs] {MAX_SAMPLES_ENV}: advertising max_samples {c}, not {max_samples}"
            );
            c
        }
        _ => max_samples,
    }
}

/// `vrend_renderer_query_multisample_caps`: the sample counts a multisample RGBA32F texture
/// can be made and rendered to at, from the driver's maximum down, and where each count's
/// samples sit -- packed four positions to a word, each as a 4-bit x and y in sixteenths.
/// A count the framebuffer refuses takes the positions of the smallest working count above it.
fn query_multisample_caps(gl: &Gl, max_samples: u32, locations: &mut [u32; 8]) -> u32 {
    const COUNTS: [u32; 4] = [2, 4, 8, 16];
    const OFFSETS: [usize; 4] = [0, 1, 2, 4];
    let mut confirmed = 1;
    let mut lowest_working: Option<usize> = None;
    *locations = [0; 8];
    let fb = gl.gen_framebuffer();
    for i in (0..4).rev() {
        let samples = COUNTS[i];
        if samples > max_samples {
            continue;
        }
        let tex = gl.gen_texture();
        gl.bind_texture(GL_TEXTURE_2D_MULTISAMPLE, Some(tex));
        gl.tex_storage_2d_multisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            samples as GLsizei,
            GL_RGBA32F,
            64,
            64,
        );
        if gl.drain_errors() == GL_NO_ERROR {
            gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
            gl.framebuffer_texture_2d(
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D_MULTISAMPLE,
                Some(tex),
                0,
            );
            if gl.check_framebuffer_status() == GL_FRAMEBUFFER_COMPLETE {
                confirmed = confirmed.max(samples);
                for k in 0..samples {
                    let p = gl.get_sample_position(k as GLuint);
                    let packed = (((p[0] * 16.0).floor() as u32) & 0xf) << 4
                        | (((p[1] * 16.0).floor() as u32) & 0xf);
                    locations[OFFSETS[i] + (k as usize >> 2)] |= packed << (8 * (k & 3));
                }
                lowest_working = Some(i);
            } else if let Some(w) = lowest_working.filter(|w| *w > 0) {
                for k in 0..samples as usize {
                    locations[OFFSETS[i] + (k >> 2)] = locations[OFFSETS[w] + (k >> 2)];
                }
            }
            gl.bind_framebuffer(GL_FRAMEBUFFER, None);
        }
        gl.delete_texture(tex);
    }
    gl.delete_framebuffer(fb);
    confirmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    /// Every field, because the guest reads every field by offset -- a struct that drifted in
    /// the middle would hand it the wrong limits with no error anywhere.
    ///
    /// Ground truth: `harness/abi/layout.txt`, dumped by the compiler from `virgl_hw.h`.
    #[test]
    fn the_capsets_match_the_c_header() {
        assert_eq!(size_of::<CapsV1>(), 308);
        assert_eq!(align_of::<CapsV1>(), 4);
        assert_eq!(offset_of!(CapsV1, max_version), 0);
        assert_eq!(offset_of!(CapsV1, sampler), 4);
        assert_eq!(offset_of!(CapsV1, render), 68);
        assert_eq!(offset_of!(CapsV1, depthstencil), 132);
        assert_eq!(offset_of!(CapsV1, vertexbuffer), 196);
        assert_eq!(offset_of!(CapsV1, bset), 260);
        assert_eq!(offset_of!(CapsV1, glsl_level), 264);
        assert_eq!(offset_of!(CapsV1, max_texture_array_layers), 268);
        assert_eq!(offset_of!(CapsV1, max_streamout_buffers), 272);
        assert_eq!(offset_of!(CapsV1, max_dual_source_render_targets), 276);
        assert_eq!(offset_of!(CapsV1, max_render_targets), 280);
        assert_eq!(offset_of!(CapsV1, max_samples), 284);
        assert_eq!(offset_of!(CapsV1, prim_mask), 288);
        assert_eq!(offset_of!(CapsV1, max_tbo_size), 292);
        assert_eq!(offset_of!(CapsV1, max_uniform_blocks), 296);
        assert_eq!(offset_of!(CapsV1, max_viewports), 300);
        assert_eq!(offset_of!(CapsV1, max_texture_gather_components), 304);
        assert_eq!(size_of::<CapsV2>(), 1408);
        assert_eq!(align_of::<CapsV2>(), 4);
        assert_eq!(offset_of!(CapsV2, v1), 0);
        assert_eq!(offset_of!(CapsV2, min_aliased_point_size), 308);
        assert_eq!(offset_of!(CapsV2, max_aliased_point_size), 312);
        assert_eq!(offset_of!(CapsV2, min_smooth_point_size), 316);
        assert_eq!(offset_of!(CapsV2, max_smooth_point_size), 320);
        assert_eq!(offset_of!(CapsV2, min_aliased_line_width), 324);
        assert_eq!(offset_of!(CapsV2, max_aliased_line_width), 328);
        assert_eq!(offset_of!(CapsV2, min_smooth_line_width), 332);
        assert_eq!(offset_of!(CapsV2, max_smooth_line_width), 336);
        assert_eq!(offset_of!(CapsV2, max_texture_lod_bias), 340);
        assert_eq!(offset_of!(CapsV2, max_geom_output_vertices), 344);
        assert_eq!(offset_of!(CapsV2, max_geom_total_output_components), 348);
        assert_eq!(offset_of!(CapsV2, max_vertex_outputs), 352);
        assert_eq!(offset_of!(CapsV2, max_vertex_attribs), 356);
        assert_eq!(offset_of!(CapsV2, max_shader_patch_varyings), 360);
        assert_eq!(offset_of!(CapsV2, min_texel_offset), 364);
        assert_eq!(offset_of!(CapsV2, max_texel_offset), 368);
        assert_eq!(offset_of!(CapsV2, min_texture_gather_offset), 372);
        assert_eq!(offset_of!(CapsV2, max_texture_gather_offset), 376);
        assert_eq!(offset_of!(CapsV2, texture_buffer_offset_alignment), 380);
        assert_eq!(offset_of!(CapsV2, uniform_buffer_offset_alignment), 384);
        assert_eq!(offset_of!(CapsV2, shader_buffer_offset_alignment), 388);
        assert_eq!(offset_of!(CapsV2, capability_bits), 392);
        assert_eq!(offset_of!(CapsV2, sample_locations), 396);
        assert_eq!(offset_of!(CapsV2, max_vertex_attrib_stride), 428);
        assert_eq!(offset_of!(CapsV2, max_shader_buffer_frag_compute), 432);
        assert_eq!(offset_of!(CapsV2, max_shader_buffer_other_stages), 436);
        assert_eq!(offset_of!(CapsV2, max_shader_image_frag_compute), 440);
        assert_eq!(offset_of!(CapsV2, max_shader_image_other_stages), 444);
        assert_eq!(offset_of!(CapsV2, max_image_samples), 448);
        assert_eq!(offset_of!(CapsV2, max_compute_work_group_invocations), 452);
        assert_eq!(offset_of!(CapsV2, max_compute_shared_memory_size), 456);
        assert_eq!(offset_of!(CapsV2, max_compute_grid_size), 460);
        assert_eq!(offset_of!(CapsV2, max_compute_block_size), 472);
        assert_eq!(offset_of!(CapsV2, max_texture_2d_size), 484);
        assert_eq!(offset_of!(CapsV2, max_texture_3d_size), 488);
        assert_eq!(offset_of!(CapsV2, max_texture_cube_size), 492);
        assert_eq!(offset_of!(CapsV2, max_combined_shader_buffers), 496);
        assert_eq!(offset_of!(CapsV2, max_atomic_counters), 500);
        assert_eq!(offset_of!(CapsV2, max_atomic_counter_buffers), 524);
        assert_eq!(offset_of!(CapsV2, max_combined_atomic_counters), 548);
        assert_eq!(offset_of!(CapsV2, max_combined_atomic_counter_buffers), 552);
        assert_eq!(offset_of!(CapsV2, host_feature_check_version), 556);
        assert_eq!(offset_of!(CapsV2, supported_readback_formats), 560);
        assert_eq!(offset_of!(CapsV2, scanout), 624);
        assert_eq!(offset_of!(CapsV2, capability_bits_v2), 688);
        assert_eq!(offset_of!(CapsV2, max_video_memory), 692);
        assert_eq!(offset_of!(CapsV2, renderer), 696);
        assert_eq!(offset_of!(CapsV2, max_anisotropy), 760);
        assert_eq!(offset_of!(CapsV2, max_texture_samplers), 764);
        assert_eq!(offset_of!(CapsV2, supported_multisample_formats), 768);
        assert_eq!(offset_of!(CapsV2, max_const_buffer_size), 832);
        assert_eq!(offset_of!(CapsV2, num_video_caps), 856);
        assert_eq!(offset_of!(CapsV2, video_caps), 860);
        assert_eq!(offset_of!(CapsV2, max_uniform_block_size), 1372);
        assert_eq!(offset_of!(CapsV2, max_tcs_outputs), 1376);
        assert_eq!(offset_of!(CapsV2, max_tes_outputs), 1380);
        assert_eq!(offset_of!(CapsV2, max_shader_storage_blocks), 1384);
    }

    /// Capset 1 is the head of capset 2, and says so: a guest that asked for the old set gets
    /// the old struct claiming the old version, not a v2 header on a v1-sized buffer.
    #[test]
    fn capset_one_is_the_head_of_capset_two_at_version_one() {
        let mut two = CapsV2::zeroed();
        two.v1.max_version = VIRGL2_VERSION;
        two.v1.glsl_level = 310;
        two.v1.max_samples = 4;
        let one = two.v1();
        assert_eq!(one.max_version, VIRGL_VERSION);
        assert_eq!(one.glsl_level, 310);
        assert_eq!(one.as_bytes().len(), size_of::<CapsV1>());
        assert_eq!(&two.as_bytes()[4..size_of::<CapsV1>()], &one.as_bytes()[4..]);
    }

    /// The ceiling is a mitigation, so every way of getting it wrong has to fail towards
    /// advertising less rather than more -- except the absent variable, which is the only case
    /// where the host has said nothing at all.
    #[test]
    fn a_sample_ceiling_never_raises_what_we_advertise() {
        assert_eq!(capped_samples(8, None), 8, "no ceiling advertises what the host has");
        assert_eq!(capped_samples(8, Some(4)), 4);
        assert_eq!(capped_samples(8, Some(1)), 1, "which is how multisampling is switched off");
        assert_eq!(
            capped_samples(2, Some(8)),
            2,
            "a ceiling above the host does not invent counts"
        );
        assert_eq!(capped_samples(0, Some(4)), 0, "nor does it invent them for a host with none");

        assert_eq!(parse_ceiling("1"), Some(1));
        assert_eq!(parse_ceiling(" 4 "), Some(4));
        assert_eq!(parse_ceiling("0"), Some(1), "zero means none, and none is single-sampled");
        assert_eq!(parse_ceiling("all"), None, "garbage caps nothing, and says so");
        assert_eq!(parse_ceiling(""), None);
    }

    #[test]
    fn a_format_mask_is_one_bit_per_wire_number() {
        let mut m = FormatMask::default();
        let f = Format::from_wire(67).unwrap();
        m.set(f);
        assert!(m.has(f));
        assert_eq!(m.bitmask[2], 1 << 3);
        assert!(!m.has(Format::from_wire(66).unwrap()));
    }
}
