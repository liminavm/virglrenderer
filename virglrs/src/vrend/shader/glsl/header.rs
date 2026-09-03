// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The header: the version and extension lines, and the interface declarations each stage
//! makes -- inputs, outputs, samplers, images, buffers, temporaries. Written after the body,
//! from what the walk found.

use super::exit::{blockname, blockvarname, emit_fog_fixup_hdr};
use super::tex::internalformat_string;
use super::{
    Ctx, Failure, Image, Io, IoDir, MAX_SO_OUTPUTS, Qual, Sampler, Strings, bit32, bit64, emit,
    gs_input_prim_to_size, hdr, prim_to_name, prim_to_tes_name, proc_prefix, req,
    samplertype_is_shadow, spacing_string, stage_output_name_prefix,
};
use crate::vrend::shader::{
    AdvancedBlend, Cfg, FsInfo, Info, Key, sampler_return_conv, sampler_type_conv,
};
use crate::vrend::tgsi::{
    Declaration, File, Interpolate, Location, Processor, Semantic, Shader, Token, scan,
};

/// `emit_ext`.
fn emit_ext(ctx: &mut Ctx<'_>, name: &str, verb: &str) {
    ctx.bufs.ver_ext(&format!("#extension GL_{name} : {verb}\n"));
}

/// `emit_header`, GLES leg.
pub(super) fn emit_header(ctx: &mut Ctx<'_>) {
    let bits = ctx.shader_req_bits;
    ctx.bufs.ver_ext(&format!("#version {} es\n", ctx.cfg.glsl_version));

    if bits & req::CLIP_DISTANCE != 0 || (ctx.cfg.has_cull_distance && ctx.num_out_clip_dist == 0) {
        emit_ext(ctx, "EXT_clip_cull_distance", "require");
    }
    if bits & req::SAMPLER_MS != 0 {
        emit_ext(ctx, "OES_texture_storage_multisample_2d_array", "require");
    }
    if bits & req::CONSERVATIVE_DEPTH != 0 {
        emit_ext(ctx, "EXT_conservative_depth", "require");
    }
    if ctx.prog_type == Processor::Fragment {
        if bits & req::FBFETCH != 0 {
            emit_ext(ctx, "EXT_shader_framebuffer_fetch", "require");
        }
        if bits & req::BLEND_EQUATION_ADVANCED != 0 {
            emit_ext(ctx, "KHR_blend_equation_advanced", "require");
        }
        if ctx.cfg.has_dual_src_blend {
            emit_ext(ctx, "EXT_blend_func_extended", "require");
        }
    }
    if bits & req::VIEWPORT_IDX != 0 {
        emit_ext(ctx, "OES_viewport_array", "require");
    }
    if ctx.prog_type == Processor::Geometry {
        emit_ext(ctx, "EXT_geometry_shader", "require");
        if bits & req::PSIZE != 0 {
            emit_ext(ctx, "OES_geometry_point_size", "enable");
        }
    }
    if bits & req::NV_IMAGE_FORMATS != 0 {
        emit_ext(ctx, "NV_image_formats", "require");
    }
    if bits & req::SEPERATE_SHADER_OBJECTS != 0 {
        emit_ext(ctx, "EXT_separate_shader_objects", "require");
    }
    if matches!(ctx.prog_type, Processor::TessCtrl | Processor::TessEval) {
        if ctx.cfg.glsl_version < 320 {
            emit_ext(ctx, "OES_tessellation_shader", "require");
        }
        emit_ext(ctx, "OES_tessellation_point_size", "enable");
    }
    if ctx.cfg.glsl_version < 320 {
        if bits & req::SAMPLER_BUF != 0 {
            emit_ext(ctx, "EXT_texture_buffer", "require");
        }
        if ctx.prefer_generic_io_block(IoDir::In) || ctx.prefer_generic_io_block(IoDir::Out) {
            emit_ext(ctx, "OES_shader_io_blocks", "require");
        }
        if bits & req::SAMPLE_SHADING != 0 {
            emit_ext(ctx, "OES_sample_variables", "require");
        }
        if bits & req::GPU_SHADER5 != 0 {
            emit_ext(ctx, "OES_gpu_shader5", "require");
            emit_ext(ctx, "OES_shader_multisample_interpolation", "require");
        }
        if bits & req::CUBE_ARRAY != 0 {
            emit_ext(ctx, "OES_texture_cube_map_array", "require");
        }
        if bits & req::LAYER != 0 {
            emit_ext(ctx, "EXT_geometry_shader", "require");
        }
        if bits & req::IMAGE_ATOMIC != 0 {
            emit_ext(ctx, "OES_shader_image_atomic", "require");
        }
        if bits & req::GEOMETRY_SHADER != 0 {
            emit_ext(ctx, "EXT_geometry_shader", "require");
        }
    }
    if ctx.logiop_require_inout() {
        if ctx.cfg.has_fbfetch_coherent {
            emit_ext(ctx, "EXT_shader_framebuffer_fetch", "require");
        } else {
            emit_ext(ctx, "EXT_shader_framebuffer_fetch_non_coherent", "require");
        }
    }
    if bits & req::TEXTURE_SHADOW_LOD != 0 {
        emit_ext(ctx, "EXT_texture_shadow_lod", "require");
    }
    if bits & req::LODQ != 0 {
        emit_ext(ctx, "EXT_texture_query_lod", "require");
    }
    if bits & req::SHADER_NOPERSPECTIVE_INTERPOLATION != 0 {
        emit_ext(ctx, "NV_shader_noperspective_interpolation", "require");
    }
    ctx.bufs.hdr("precision highp float;\n");
    ctx.bufs.hdr("precision highp int;\n");
}

/// `get_interp_string`.
fn interp_string(cfg: &Cfg, interpolate: Interpolate, flatshade: bool) -> &'static str {
    match interpolate {
        Interpolate::Linear => {
            if cfg.has_nopersective {
                "noperspective "
            } else {
                ""
            }
        }
        Interpolate::Perspective => "smooth ",
        Interpolate::Constant => "flat ",
        Interpolate::Color => {
            if flatshade {
                "flat "
            } else {
                ""
            }
        }
    }
}

/// `get_aux_string`.
fn aux_string(location: Location) -> &'static str {
    match location {
        Location::Center => "",
        Location::Centroid => "centroid ",
        Location::Sample => "sample ",
    }
}

/// `emit_sampler_decl`.
fn emit_sampler_decl(ctx: &mut Ctx<'_>, i: u32, range: i32, sampler: Sampler) {
    let sname = proc_prefix(ctx.prog_type);
    let precision = "highp";
    let ptc = sampler_return_conv(sampler.ret);
    let stc = sampler_type_conv(sampler.ty).unwrap_or("");
    let is_shad = samplertype_is_shadow(sampler.ty);
    if range != 0 {
        hdr!(
            ctx.bufs,
            "uniform {} {}sampler{} {}samp{}[{}];\n",
            precision,
            ptc,
            stc,
            sname,
            i,
            range
        );
    } else {
        hdr!(ctx.bufs, "uniform {} {}sampler{} {}samp{};\n", precision, ptc, stc, sname, i);
    }
    if is_shad {
        hdr!(ctx.bufs, "uniform {} vec4 {}shadmask{};\n", precision, sname, i);
        hdr!(ctx.bufs, "uniform {} vec4 {}shadadd{};\n", precision, sname, i);
        ctx.shadow_samp_mask |= bit32(i);
    }
}

/// `emit_image_decl`.
fn emit_image_decl(ctx: &mut Ctx<'_>, i: u32, range: i32, image: Image) {
    let volatile_str = if image.vflag { "volatile " } else { "" };
    let coherent_str = if image.coherent { "coherent " } else { "" };
    let precision = "highp ";
    let mut access = "";
    let (formatstr, itype) = internalformat_string(image.decl.format);
    let ptc = sampler_return_conv(itype);
    let sname = proc_prefix(ctx.prog_type);
    let stc = sampler_type_conv(image.decl.resource).unwrap_or("");

    // From ARB_shader_image_load_store: an image used for loads or atomics must carry a format
    // qualifier matching its unit; one used only for stores need not, but a declared one must
    // match.
    let mut require_format_specifier = true;
    let r32 = matches!(formatstr, "r32f" | "r32i" | "r32ui");
    if !image.decl.writable {
        access = "readonly ";
    } else if image.decl.format == 0 || !r32 {
        access = "writeonly ";
        require_format_specifier = !formatstr.is_empty();
    }

    let binding = i + u32::from(ctx.key.image_binding_offset);
    if require_format_specifier {
        hdr!(
            ctx.bufs,
            "layout(binding={}, {}) ",
            binding,
            if formatstr.is_empty() { "rgba32f" } else { formatstr }
        );
    } else {
        hdr!(
            ctx.bufs,
            "layout(binding={}{}{}) ",
            binding,
            if formatstr.is_empty() { ", rgba32f" } else { ", " },
            formatstr
        );
    }
    if range != 0 {
        hdr!(
            ctx.bufs,
            "{}{}{}uniform {}{}image{} {}img{}[{}];\n",
            access,
            volatile_str,
            coherent_str,
            precision,
            ptc,
            stc,
            sname,
            i,
            range
        );
    } else {
        hdr!(
            ctx.bufs,
            "{}{}{}uniform {}{}image{} {}img{};\n",
            access,
            volatile_str,
            coherent_str,
            precision,
            ptc,
            stc,
            sname,
            i
        );
    }
}

/// `emit_ios_common`.
fn emit_ios_common(ctx: &mut Ctx<'_>) -> u32 {
    let sname = proc_prefix(ctx.prog_type);
    let mut glsl_ver_required = ctx.glsl_ver_required;

    for r in ctx.temp_ranges.clone() {
        let precise = if r.precise_result { "precise" } else { "" };
        if r.array_id > 0 {
            hdr!(ctx.bufs, "{} vec4 temp{}[{}];\n", precise, r.first, r.last - r.first + 1);
        } else {
            hdr!(ctx.bufs, "{} vec4 temp{};\n", precise, r.first);
        }
    }

    if ctx.require_dummy_value {
        ctx.bufs.hdr("vec4 dummy_value = vec4(0.0, 0.0, 0.0, 0.0);\n");
    }
    if ctx.write_mul_utemp {
        ctx.bufs.hdr("uvec4 mul_utemp;\n");
        ctx.bufs.hdr("uvec4 umul_temp;\n");
    }
    if ctx.write_mul_itemp {
        ctx.bufs.hdr("ivec4 mul_itemp;\n");
        ctx.bufs.hdr("ivec4 imul_temp;\n");
    }
    if ctx.ssbo_used_mask != 0 || ctx.has_file_memory {
        ctx.bufs.hdr("uint ssbo_addr_temp;\n");
    }
    if ctx.shader_req_bits & req::FP64 != 0 {
        ctx.bufs.hdr("dvec2 fp64_dst[3];\n");
        ctx.bufs.hdr("dvec2 fp64_src[4];\n");
    }
    for i in 0..ctx.num_address {
        hdr!(ctx.bufs, "int addr{};\n", i);
    }
    if ctx.num_consts != 0 {
        hdr!(ctx.bufs, "uniform uvec4 {}const0[{}];\n", sname, ctx.num_consts);
    }

    if ctx.ubo_used_mask != 0 {
        if ctx.info.is_dimension_indirect(File::Constant) {
            glsl_ver_required = ctx.require_glsl_ver(150);
            let first = ctx.ubo_used_mask.trailing_zeros() as usize;
            let num_ubo = ctx.ubo_used_mask.count_ones();
            hdr!(
                ctx.bufs,
                "uniform {}ubo {{ vec4 ubocontents[{}]; }} {}uboarr[{}];\n",
                sname,
                ctx.ubo_sizes[first],
                sname,
                num_ubo
            );
        } else {
            let mut mask = ctx.ubo_used_mask;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                hdr!(
                    ctx.bufs,
                    "uniform {}ubo{} {{ vec4 {}ubo{}contents[{}]; }};\n",
                    sname,
                    i,
                    sname,
                    i,
                    ctx.ubo_sizes[i]
                );
            }
        }
    }

    if ctx.info.is_indirect(File::Sampler) {
        for a in ctx.sampler_arrays.clone() {
            let sampler = ctx.samplers[a.first as usize];
            emit_sampler_decl(ctx, a.first as u32, a.array_size, sampler);
        }
    } else {
        let nsamp = 32 - ctx.samplers_used.leading_zeros();
        for i in 0..nsamp {
            if ctx.samplers_used & bit32(i) == 0 {
                continue;
            }
            let sampler = ctx.samplers[i as usize];
            emit_sampler_decl(ctx, i, 0, sampler);
        }
    }

    if ctx.gles_use_tex_query_level {
        hdr!(
            ctx.bufs,
            "uniform int {}_texlod[{}];\n",
            proc_prefix(ctx.info.processor),
            ctx.samplers_used.count_ones()
        );
    }

    if ctx.info.is_indirect(File::Image) {
        for a in ctx.image_arrays.clone() {
            let image = ctx.images[a.first as usize];
            emit_image_decl(ctx, a.first as u32, a.array_size, image);
        }
    } else {
        let mut mask = ctx.images_used_mask;
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            let image = ctx.images[i as usize];
            emit_image_decl(ctx, i, 0, image);
        }
    }

    for i in 0..ctx.abo_idx.len() {
        let (idx, off, size) = (ctx.abo_idx[i], ctx.abo_offsets[i] * 4, ctx.abo_sizes[i]);
        hdr!(
            ctx.bufs,
            "layout (binding = {}, offset = {}) uniform atomic_uint ac{}_{}",
            idx,
            off,
            idx,
            off
        );
        if size > 1 {
            hdr!(ctx.bufs, "[{}]", size);
        }
        ctx.bufs.hdr(";\n");
    }

    let first_binding = ctx.ssbo_first_binding as i64;
    if ctx.info.is_indirect(File::Buffer) {
        let mut mask = ctx.ssbo_used_mask;
        while mask != 0 {
            let start = mask.trailing_zeros();
            let count = (!(mask >> start)).trailing_zeros();
            mask &= !((1u32.wrapping_shl(count)).wrapping_sub(1).wrapping_shl(start));
            if count == 32 {
                mask = 0;
            }
            let binding = start as i64 + i64::from(ctx.key.ssbo_binding_offset) - first_binding;
            let atomic = if ctx.ssbo_atomic_mask & bit32(start) != 0 { "atomic" } else { "" };
            hdr!(
                ctx.bufs,
                "layout (binding = {}, std430) buffer {}ssbo{} {{ uint {}ssbocontents{}[]; }} {}ssboarr{}[{}];\n",
                binding as i32,
                sname,
                start,
                sname,
                start,
                sname,
                atomic,
                count
            );
        }
    } else {
        let mut mask = ctx.ssbo_used_mask;
        while mask != 0 {
            let id = mask.trailing_zeros();
            mask &= mask - 1;
            let binding = id as i64 + i64::from(ctx.key.ssbo_binding_offset) - first_binding;
            let ty = if ctx.ssbo_integer_mask & bit32(id) != 0 { Qual::Int } else { Qual::Uint };
            let coherent =
                if ctx.ssbo_memory_qualifier[id as usize] == 1 { "coherent" } else { "" };
            hdr!(
                ctx.bufs,
                "layout (binding = {}, std430) {} buffer {}ssbo{} {{ {} {}ssbocontents{}[]; }};\n",
                binding as i32,
                coherent,
                sname,
                id,
                ty.s(),
                sname,
                id
            );
        }
    }

    glsl_ver_required
}

/// `emit_ios_streamout`.
fn emit_ios_streamout(ctx: &mut Ctx<'_>) {
    let Some(so) = ctx.so else {
        return;
    };
    for (i, o) in so.outputs.iter().enumerate() {
        if !ctx.write_so_outputs[i] {
            continue;
        }
        let outtype = if o.num_components == 1 {
            "float".to_string()
        } else {
            format!("vec{}", o.num_components)
        };
        if o.stream != 0 && ctx.prog_type == Processor::Geometry {
            hdr!(ctx.bufs, "layout (stream={}) out {} tfout{};\n", o.stream, outtype, i);
        } else {
            let Some(output) = ctx.outputs.iter().find(|s| {
                s.first <= u32::from(o.register_index) && s.last >= u32::from(o.register_index)
            }) else {
                ctx.bufs.set_hdr_error();
                return;
            };
            if ctx.so_need_temp(i)
                || output.name == Semantic::ClipDist
                || ctx.prog_type == Processor::Geometry
                || output.glsl_predefined_no_emit
            {
                if ctx.prog_type == Processor::TessCtrl {
                    hdr!(ctx.bufs, "out {} tfout{}[];\n", outtype, i);
                } else {
                    hdr!(ctx.bufs, "out {} tfout{};\n", outtype, i);
                }
            }
        }
    }
}

/// `emit_ios_generic`.
fn emit_ios_generic(
    ctx: &mut Ctx<'_>,
    iot: IoDir,
    prefix: &str,
    io: &Io,
    inout: &str,
    postfix: &str,
) {
    let t = match io.ty {
        super::VecType::Float => " vec4",
        super::VecType::Int => "ivec4",
        super::VecType::Uint => "uvec4",
    };
    if io.overlapping_array.is_some() {
        return;
    }
    let precise = if io.precise { "precise" } else { "" };
    let invariant = if io.invariant { "invariant" } else { "" };

    if io.first == io.last {
        // Ugly: spaces are left to patch the interpolation in later.
        hdr!(
            ctx.bufs,
            "{}{} {}  {} {} {}{};\n",
            precise,
            invariant,
            prefix,
            inout,
            t,
            io.glsl_name,
            postfix
        );
        if io.name == Semantic::Generic {
            if io.sid >= 64 {
                ctx.bufs.set_error();
                return;
            }
            if iot == IoDir::In {
                ctx.generic_ios.matched.inputs_emitted_mask |= 1 << io.sid;
            } else {
                ctx.generic_ios.matched.outputs_emitted_mask |= 1 << io.sid;
            }
        } else if io.name == Semantic::TexCoord {
            if io.sid >= 8 {
                ctx.bufs.set_error();
                return;
            }
            if iot == IoDir::In {
                ctx.texcoord_ios.inputs_emitted_mask |= 1 << io.sid;
            } else {
                ctx.texcoord_ios.outputs_emitted_mask |= 1 << io.sid;
            }
        }
    } else {
        let array_size = io.last as i32 - io.first as i32 + 1;
        if ctx.prefer_generic_io_block(iot) {
            let stage_prefix = if iot == IoDir::In {
                ctx.stage_input_name_prefix(ctx.prog_type)
            } else {
                stage_output_name_prefix(ctx.prog_type)
            };
            let block = blockname(stage_prefix, io);
            let blockvar = blockvarname(stage_prefix, io, postfix);
            hdr!(ctx.bufs, "{} {} {{\n", inout, block);
            hdr!(
                ctx.bufs,
                "{}{}\n{}     {} {}[{}]; \n}} {};\n",
                precise,
                invariant,
                prefix,
                t,
                io.glsl_name,
                array_size,
                blockvar
            );
        } else {
            hdr!(
                ctx.bufs,
                "{}{}\n{}       {} {} {}{}[{}];\n",
                precise,
                invariant,
                prefix,
                inout,
                t,
                io.glsl_name,
                postfix,
                array_size
            );
            let mask = bit64(array_size as u32).wrapping_sub(1).wrapping_shl(io.sid);
            if io.name == Semantic::Generic {
                if io.sid as i32 + array_size >= 64 {
                    ctx.bufs.set_error();
                    return;
                }
                if iot == IoDir::In {
                    ctx.generic_ios.matched.inputs_emitted_mask |= mask;
                } else {
                    ctx.generic_ios.matched.outputs_emitted_mask |= mask;
                }
            } else if io.name == Semantic::TexCoord {
                if io.sid as i32 + array_size > 8 {
                    ctx.bufs.set_error();
                    return;
                }
                if iot == IoDir::In {
                    ctx.texcoord_ios.inputs_emitted_mask |= mask;
                } else {
                    ctx.texcoord_ios.outputs_emitted_mask |= mask;
                }
            }
        }
    }
}

/// `get_semantic_to_compare`: front and back colour of one index share interpolators, and a
/// shader may define only one of them, so both compare as `COLOR`.
fn semantic_to_compare(name: Semantic) -> Semantic {
    match name {
        Semantic::Color | Semantic::BColor => Semantic::Color,
        n => n,
    }
}

/// `get_interpolator_prefix`.
fn interpolator_prefix(cfg: &Cfg, io: &Io, fs_info: &FsInfo, flatshade: bool) -> String {
    if matches!(
        io.name,
        Semantic::Generic | Semantic::TexCoord | Semantic::Color | Semantic::BColor
    ) {
        let name = semantic_to_compare(io.name);
        for interp in &fs_info.interps {
            if semantic_to_compare(interp.semantic_name) == name
                && u32::from(interp.semantic_index) == io.sid
            {
                return format!(
                    "{} {}",
                    interp_string(cfg, interp.interpolate, flatshade),
                    aux_string(interp.location)
                );
            }
        }
    }
    String::new()
}

const FRONT_COLOR_EMITTED: u8 = 1 << 0;
const BACK_COLOR_EMITTED: u8 = 1 << 1;

/// `emit_ios_generic_outputs`.
fn emit_ios_generic_outputs(ctx: &mut Ctx<'_>, can_emit_generic: fn(&Io) -> bool) {
    let mut fc_emitted = 0u64;
    let mut bc_emitted = 0u64;
    for i in 0..ctx.outputs.len() {
        let output = ctx.outputs[i].clone();
        if !output.glsl_predefined_no_emit {
            // GS stream outputs are handled separately.
            if !can_emit_generic(&output) {
                continue;
            }
            let prefix = interpolator_prefix(ctx.cfg, &output, &ctx.key.fs_info, ctx.key.flatshade);
            if output.name == Semantic::Color {
                if output.sid >= 64 {
                    eprintln!("[virglrs] Number of output id exceeded, max is 64");
                    ctx.bufs.set_error();
                    return;
                }
                ctx.front_back_color_emitted_flags[output.sid as usize] |= FRONT_COLOR_EMITTED;
                fc_emitted |= 1 << output.sid;
            }
            if output.name == Semantic::BColor {
                if output.sid >= 64 {
                    eprintln!("[virglrs] Number of output id exceeded, max is 64");
                    ctx.bufs.set_error();
                    return;
                }
                ctx.front_back_color_emitted_flags[output.sid as usize] |= BACK_COLOR_EMITTED;
                bc_emitted |= 1 << output.sid;
            }
            emit_ios_generic(
                ctx,
                IoDir::Out,
                &prefix,
                &output,
                if output.fbfetch_used { "inout" } else { "out" },
                "",
            );
        } else if output.invariant || output.precise {
            hdr!(
                ctx.bufs,
                "{}{};\n",
                if output.precise {
                    "precise "
                } else if output.invariant {
                    "invariant "
                } else {
                    ""
                },
                output.glsl_name
            );
        }
    }
    // A back colour emitted without its front colour forces two-side colouring, because the
    // fragment shader may expect a front colour too.
    if bc_emitted & !fc_emitted != 0 {
        ctx.force_color_two_side = true;
    }
}

/// `emit_ios_patch`.
fn emit_ios_patch(ctx: &mut Ctx<'_>, prefix: &str, io: &Io, inout: &str, size: i32) -> u64 {
    let mut emitted_patches = 0u64;
    if io.last == io.first {
        hdr!(ctx.bufs, "{} {} vec4 {};\n", prefix, inout, io.glsl_name);
        emitted_patches |= bit64(io.sid);
    } else {
        hdr!(ctx.bufs, "{} {} vec4 {}[{}];\n", prefix, inout, io.glsl_name, size);
        let mask = bit64(size as u32).wrapping_sub(1);
        emitted_patches |= mask.wrapping_shl(io.sid);
    }
    emitted_patches
}

fn can_emit_generic_default(_: &Io) -> bool {
    true
}

fn can_emit_generic_geom(io: &Io) -> bool {
    io.stream == 0
}

/// `emit_ios_vs`.
fn emit_ios_vs(ctx: &mut Ctx<'_>) {
    for i in 0..ctx.inputs.len() {
        let input = ctx.inputs[i].clone();
        if !input.glsl_predefined_no_emit {
            let postfix = if input.first != input.last {
                format!("[{}]", input.last as i32 - input.first as i32 + 1)
            } else {
                String::new()
            };
            let vtype = match input.ty {
                super::VecType::Float => "vec4",
                super::VecType::Int => "ivec4",
                super::VecType::Uint => "uvec4",
            };
            hdr!(ctx.bufs, "in {} {}{};\n", vtype, input.glsl_name, postfix);
        }
    }

    emit_ios_generic_outputs(ctx, can_emit_generic_default);
    if ctx.bufs.main_error {
        return;
    }

    if ctx.key.color_two_side || ctx.force_color_two_side {
        let mut interpolators = [Interpolate::Color; 2];
        let mut interp_loc = [Location::Center; 2];
        for interp in &ctx.key.fs_info.interps {
            if interp.semantic_name == Semantic::Color || interp.semantic_name == Semantic::BColor {
                // The C indexes the pair with the semantic index unchecked.
                let k = usize::from(interp.semantic_index);
                if k < 2 {
                    interpolators[k] = interp.interpolate;
                    interp_loc[k] = interp.location;
                }
            }
        }
        for i in 0..ctx.outputs.len() {
            let sid = ctx.outputs[i].sid;
            if sid >= 2 {
                continue;
            }
            let flags = ctx.front_back_color_emitted_flags[sid as usize];
            let fcolor_emitted = flags & FRONT_COLOR_EMITTED != 0;
            let bcolor_emitted = flags & BACK_COLOR_EMITTED != 0;
            if fcolor_emitted && !bcolor_emitted {
                hdr!(
                    ctx.bufs,
                    "{} {} out vec4 vso_bc{};\n",
                    interp_string(ctx.cfg, interpolators[sid as usize], ctx.key.flatshade),
                    aux_string(interp_loc[sid as usize]),
                    sid
                );
                ctx.front_back_color_emitted_flags[sid as usize] |= BACK_COLOR_EMITTED;
            }
            if bcolor_emitted && !fcolor_emitted {
                hdr!(
                    ctx.bufs,
                    "{} {} out vec4 vso_c{};\n",
                    interp_string(ctx.cfg, interpolators[sid as usize], ctx.key.flatshade),
                    aux_string(interp_loc[sid as usize]),
                    sid
                );
                ctx.front_back_color_emitted_flags[sid as usize] |= FRONT_COLOR_EMITTED;
            }
        }
    }

    if ctx.key.vs.fog_fixup_mask != 0 {
        emit_fog_fixup_hdr(ctx);
    }

    if ctx.has_clipvertex && ctx.is_last_vertex_stage {
        hdr!(ctx.bufs, "{}vec4 clipv_tmp;\n", if ctx.has_clipvertex_so { "out " } else { "" });
    }

    let mut cull_buf = String::new();
    let mut clip_buf = String::new();
    if ctx.cfg.has_cull_distance && (ctx.num_out_clip_dist != 0 || ctx.is_last_vertex_stage) {
        let mut num_clip_dists = i32::from(ctx.num_clip_dist_prop);
        let num_cull_dists = i32::from(ctx.num_cull_dist_prop);
        if ctx.num_out_clip_dist != 0 && num_clip_dists + num_cull_dists == 0 {
            num_clip_dists = ctx.num_out_clip_dist;
        }
        if num_clip_dists != 0 {
            clip_buf = format!("out float gl_ClipDistance[{num_clip_dists}];\n");
        }
        if num_cull_dists != 0 {
            cull_buf = format!("out float gl_CullDistance[{num_cull_dists}];\n");
        }
        if ctx.is_last_vertex_stage {
            hdr!(ctx.bufs, "{}{}", clip_buf, cull_buf);
        }
        ctx.bufs.hdr("vec4 clip_dist_temp[2];\n");
    }

    let psize_buf = if ctx.has_pointsize_output { "out float gl_PointSize;\n" } else { "" };
    if !ctx.is_last_vertex_stage && ctx.key.use_pervertex_in {
        hdr!(
            ctx.bufs,
            "out gl_PerVertex {{\n vec4 gl_Position;\n {}{}{}}};\n",
            clip_buf,
            cull_buf,
            psize_buf
        );
    }
}

/// `get_depth_layout`.
fn depth_layout(layout: u32) -> Option<&'static str> {
    match layout {
        1 => Some("depth_any"),
        2 => Some("depth_greater"),
        3 => Some("depth_less"),
        4 => Some("depth_unchanged"),
        _ => None,
    }
}

/// `emit_ios_fs`.
fn emit_ios_fs(ctx: &mut Ctx<'_>) {
    // `fs_emit_layout` only chooses a `gl_FragCoord` layout on desktop GL.
    if ctx.early_depth_stencil {
        ctx.bufs.hdr("layout(early_fragment_tests) in;\n");
    }

    for i in 0..ctx.inputs.len() {
        let input = ctx.inputs[i].clone();
        if input.glsl_predefined_no_emit {
            continue;
        }
        let mut prefix = "";
        let mut auxprefix = "";

        if input.name == Semantic::Color
            && u32::from(ctx.key.fs.available_color_in_bits) & bit32(input.sid) == 0
        {
            hdr!(ctx.bufs, "vec4 {} = vec4(0.0, 0.0, 0.0, 0.0);\n", input.glsl_name);
            continue;
        }
        if input.name == Semantic::BColor
            && u32::from(ctx.key.fs.available_color_in_bits) & (bit32(input.sid) << 2) == 0
        {
            hdr!(ctx.bufs, "vec4 {} = vec4(0.0, 0.0, 0.0, 0.0);\n", input.glsl_name);
            continue;
        }

        if matches!(input.name, Semantic::Generic | Semantic::Color | Semantic::BColor) {
            prefix = interp_string(ctx.cfg, input.interpolate, ctx.key.flatshade);
            auxprefix = aux_string(input.location);
            ctx.interp_input_mask |= bit64(i as u32);
        }
        let prefixes = format!("{prefix} {auxprefix}");
        emit_ios_generic(ctx, IoDir::In, &prefixes, &input, "in", "");
    }

    if ctx.key.color_two_side {
        if ctx.color_in_mask & 1 != 0 {
            ctx.bufs.hdr("vec4 realcolor0;\n");
        }
        if ctx.color_in_mask & 2 != 0 {
            ctx.bufs.hdr("vec4 realcolor1;\n");
        }
    }

    let mut choices = ctx.fs_blend_equation_advanced;
    while choices != 0 {
        let choice = choices.trailing_zeros();
        choices &= choices - 1;
        let name =
            AdvancedBlend::from_property(choice).map_or("UNKNOWN", AdvancedBlend::layout_name);
        hdr!(ctx.bufs, "layout(blend_support_{}) out;\n", name);
    }

    if ctx.write_all_cbufs {
        let ty = if ctx.key.fs.cbufs_unsigned_int_bitmask != 0 {
            "uvec4"
        } else if ctx.key.fs.cbufs_signed_int_bitmask != 0 {
            "ivec4"
        } else {
            "vec4"
        };
        for i in 0..ctx.cfg.max_draw_buffers {
            if ctx.key.fs.logicop_func.is_some() {
                hdr!(ctx.bufs, "{} fsout_tmp_c{};\n", ty, i);
            }
            if ctx.logiop_require_inout() {
                let noncoherent = if ctx.cfg.has_fbfetch_coherent { "" } else { ", noncoherent" };
                hdr!(
                    ctx.bufs,
                    "layout (location={}{}) inout highp {} fsout_c{};\n",
                    i,
                    noncoherent,
                    ty,
                    i
                );
            } else {
                hdr!(ctx.bufs, "layout (location={}) out {} fsout_c{};\n", i, ty, i);
            }
        }
    } else {
        for i in 0..ctx.outputs.len() {
            let output = ctx.outputs[i].clone();
            if !output.glsl_predefined_no_emit {
                let prefix = if output.name == Semantic::Color && !ctx.cfg.has_dual_src_blend {
                    format!("layout(location = {})", output.sid)
                } else {
                    String::new()
                };
                emit_ios_generic(
                    ctx,
                    IoDir::Out,
                    &prefix,
                    &output,
                    if output.fbfetch_used { "inout" } else { "out" },
                    "",
                );
            } else if output.invariant || output.precise {
                hdr!(
                    ctx.bufs,
                    "{}{};\n",
                    if output.precise {
                        "precise "
                    } else if output.invariant {
                        "invariant "
                    } else {
                        ""
                    },
                    output.glsl_name
                );
            }
        }
    }

    if ctx.fs_depth_layout != 0
        && let Some(layout) = depth_layout(ctx.fs_depth_layout)
    {
        hdr!(ctx.bufs, "layout ({}) out float gl_FragDepth;\n", layout);
    }

    if ctx.num_in_clip_dist != 0 {
        if ctx.key.num_in_clip != 0 {
            hdr!(ctx.bufs, "in float gl_ClipDistance[{}];\n", ctx.key.num_in_clip);
        } else if ctx.num_in_clip_dist > 4 && ctx.key.num_in_cull == 0 {
            hdr!(ctx.bufs, "in float gl_ClipDistance[{}];\n", ctx.num_in_clip_dist);
        }
        if ctx.key.num_in_cull != 0 {
            hdr!(ctx.bufs, "in float gl_CullDistance[{}];\n", ctx.key.num_in_cull);
        }
        if ctx.fs_uses_clipdist_input {
            ctx.bufs.hdr("vec4 clip_dist_temp[2];\n");
        }
    }
}

/// `emit_ios_per_vertex_in`.
fn emit_ios_per_vertex_in(ctx: &mut Ctx<'_>) {
    if ctx.num_in_clip_dist == 0 {
        return;
    }
    let mut clip_dist = i32::from(ctx.key.num_in_clip);
    let cull_dist = i32::from(ctx.key.num_in_cull);
    if clip_dist + cull_dist == 0 {
        clip_dist = ctx.num_in_clip_dist;
    }
    let clip_var = if clip_dist != 0 {
        format!("float gl_ClipDistance[{clip_dist}];\n")
    } else {
        String::new()
    };
    let cull_var = if cull_dist != 0 {
        format!("float gl_CullDistance[{cull_dist}];\n")
    } else {
        String::new()
    };
    ctx.has_pervertex = true;
    hdr!(
        ctx.bufs,
        "in gl_PerVertex {{\n vec4 gl_Position; \n {}{}{}\n}} gl_in[];\n",
        clip_var,
        cull_var,
        if ctx.has_pointsize_input { "float gl_PointSize;\n" } else { "" }
    );
}

/// `emit_ios_per_vertex_out`.
fn emit_ios_per_vertex_out(ctx: &mut Ctx<'_>, instance_var: &str) {
    let mut clip_dist = if ctx.num_clip_dist_prop != 0 {
        i32::from(ctx.num_clip_dist_prop)
    } else {
        i32::from(ctx.key.num_out_clip)
    };
    let cull_dist = if ctx.num_cull_dist_prop != 0 {
        i32::from(ctx.num_cull_dist_prop)
    } else {
        i32::from(ctx.key.num_out_cull)
    };
    if ctx.num_out_clip_dist != 0 && clip_dist + cull_dist == 0 {
        clip_dist = ctx.num_out_clip_dist;
    }
    if ctx.key.use_pervertex_in {
        let cull_var = if cull_dist != 0 {
            format!("float gl_CullDistance[{cull_dist}];\n")
        } else {
            String::new()
        };
        let clip_var = if clip_dist != 0 {
            format!("float gl_ClipDistance[{clip_dist}];\n")
        } else {
            String::new()
        };
        hdr!(
            ctx.bufs,
            "out gl_PerVertex {{\n vec4 gl_Position; \n {}{}{}\n}} {};\n",
            clip_var,
            cull_var,
            if ctx.has_pointsize_output { "float gl_PointSize;\n" } else { "" },
            instance_var
        );
    }
    if clip_dist + cull_dist > 0 {
        ctx.bufs.hdr("vec4 clip_dist_temp[2];\n");
    }
}

/// `emit_ios_geom`.
fn emit_ios_geom(ctx: &mut Ctx<'_>) {
    let invocbuf = if ctx.gs_num_invocations > 1 {
        format!(", invocations = {}", ctx.gs_num_invocations)
    } else {
        String::new()
    };
    hdr!(ctx.bufs, "layout({}{}) in;\n", prim_to_name(ctx.gs_in_prim), invocbuf);
    hdr!(
        ctx.bufs,
        "layout({}, max_vertices = {}) out;\n",
        prim_to_name(ctx.gs_out_prim),
        ctx.gs_max_out_verts
    );

    for i in 0..ctx.inputs.len() {
        let input = ctx.inputs[i].clone();
        if !input.glsl_predefined_no_emit {
            let postfix = format!("[{}]", gs_input_prim_to_size(ctx.gs_in_prim));
            emit_ios_generic(ctx, IoDir::In, "", &input, "in", &postfix);
        }
    }

    for i in 0..ctx.outputs.len() {
        let output = ctx.outputs[i].clone();
        if !output.glsl_predefined_no_emit {
            if output.stream == 0 {
                continue;
            }
            if matches!(output.name, Semantic::Generic | Semantic::Color | Semantic::BColor) {
                ctx.interp_input_mask |= bit64(i as u32);
            }
            hdr!(
                ctx.bufs,
                "layout (stream = {}) {}{}{}out vec4 {};\n",
                output.stream,
                "",
                if output.precise { "precise " } else { "" },
                if output.invariant { "invariant " } else { "" },
                output.glsl_name
            );
        }
    }

    emit_ios_generic_outputs(ctx, can_emit_generic_geom);
    if ctx.bufs.main_error {
        return;
    }

    emit_ios_per_vertex_in(ctx);

    if ctx.has_clipvertex {
        hdr!(ctx.bufs, "{}vec4 clipv_tmp;\n", if ctx.has_clipvertex_so { "out " } else { "" });
    }

    if ctx.num_out_clip_dist != 0 {
        let has_prop = ctx.num_clip_dist_prop + ctx.num_cull_dist_prop > 0;
        let num_clip_dists = if has_prop {
            i32::from(ctx.num_clip_dist_prop)
        } else if ctx.num_out_clip_dist != 0 {
            ctx.num_out_clip_dist
        } else {
            8
        };
        let num_cull_dists = if has_prop { i32::from(ctx.num_cull_dist_prop) } else { 0 };
        let clip_buf = if num_clip_dists != 0 {
            format!("out float gl_ClipDistance[{num_clip_dists}];\n")
        } else {
            String::new()
        };
        let cull_buf = if num_cull_dists != 0 {
            format!("out float gl_CullDistance[{num_cull_dists}];\n")
        } else {
            String::new()
        };
        hdr!(ctx.bufs, "{}{}\n", clip_buf, cull_buf);
        ctx.bufs.hdr("vec4 clip_dist_temp[2];\n");
    }
}

/// `emit_ios_tcs`.
fn emit_ios_tcs(ctx: &mut Ctx<'_>) {
    for i in 0..ctx.inputs.len() {
        let input = ctx.inputs[i].clone();
        if !input.glsl_predefined_no_emit {
            if input.name == Semantic::Patch {
                emit_ios_patch(ctx, "", &input, "in", input.last as i32 - input.first as i32 + 1);
            } else {
                emit_ios_generic(ctx, IoDir::In, "", &input, "in", "[]");
            }
        }
    }

    let mut emitted_patches = 0u64;
    hdr!(ctx.bufs, "layout(vertices = {}) out;\n", ctx.tcs_vertices_out);

    for i in 0..ctx.outputs.len() {
        let output = ctx.outputs[i].clone();
        if !output.glsl_predefined_no_emit {
            if output.name == Semantic::Patch {
                emitted_patches |= emit_ios_patch(
                    ctx,
                    "patch",
                    &output,
                    "out",
                    output.last as i32 - output.first as i32 + 1,
                );
            } else {
                emit_ios_generic(ctx, IoDir::Out, "", &output, "out", "[]");
            }
        } else if (output.invariant || output.precise) && !output.glsl_gl_block {
            hdr!(
                ctx.bufs,
                "{}{};\n",
                if output.precise {
                    "precise "
                } else if output.invariant {
                    "invariant "
                } else {
                    ""
                },
                output.glsl_name
            );
        }
    }

    emit_ios_per_vertex_in(ctx);
    emit_ios_per_vertex_out(ctx, " gl_out[]");
    ctx.patches_emitted_mask = emitted_patches;
}

/// `emit_ios_tes`.
fn emit_ios_tes(ctx: &mut Ctx<'_>) {
    for i in 0..ctx.inputs.len() {
        let input = ctx.inputs[i].clone();
        if !input.glsl_predefined_no_emit {
            if input.name == Semantic::Patch {
                emit_ios_patch(
                    ctx,
                    "patch",
                    &input,
                    "in",
                    input.last as i32 - input.first as i32 + 1,
                );
            } else {
                emit_ios_generic(ctx, IoDir::In, "", &input, "in", "[]");
            }
        }
    }

    hdr!(
        ctx.bufs,
        "layout({}, {}, {}{}) in;\n",
        prim_to_tes_name(ctx.tes_prim_mode),
        spacing_string(ctx.tes_spacing),
        if ctx.tes_vertex_order != 0 { "cw" } else { "ccw" },
        if ctx.tes_point_mode != 0 { ", point_mode" } else { "" }
    );

    emit_ios_generic_outputs(ctx, can_emit_generic_default);
    if ctx.bufs.main_error {
        return;
    }

    emit_ios_per_vertex_in(ctx);
    emit_ios_per_vertex_out(ctx, "");

    if ctx.has_clipvertex && !ctx.key.gs_present {
        hdr!(ctx.bufs, "{}vec4 clipv_tmp;\n", if ctx.has_clipvertex_so { "out " } else { "" });
    }
}

/// `emit_ios_cs`.
fn emit_ios_cs(ctx: &mut Ctx<'_>) {
    let [x, y, z] = ctx.local_cs_block_size;
    hdr!(
        ctx.bufs,
        "layout (local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n",
        x,
        y,
        z
    );
    if ctx.req_local_mem != 0 {
        let ty = if ctx.integer_memory { Qual::Int } else { Qual::Uint };
        hdr!(ctx.bufs, "shared {} values[{}];\n", ty.s(), ctx.req_local_mem / 4);
    }
}

/// `emit_interp_info`.
fn emit_interp_info(ctx: &mut Ctx<'_>, semantic: Semantic, sid: u32) {
    for interp in &ctx.key.fs_info.interps {
        if interp.semantic_name == semantic && u32::from(interp.semantic_index) == sid {
            let s = format!(
                "{} {} ",
                interp_string(ctx.cfg, interp.interpolate, ctx.key.flatshade),
                aux_string(interp.location)
            );
            ctx.bufs.hdr(&s);
            break;
        }
    }
}

/// `emit_match_interfaces`: the outputs the next stage expects and this one did not emit.
fn emit_match_interfaces(
    ctx: &mut Ctx<'_>,
    expected: u64,
    emitted: u64,
    semantic: Semantic,
    prefix: char,
) {
    let mut mask = (expected | emitted) ^ emitted;
    while mask != 0 {
        let i = mask.trailing_zeros();
        mask &= mask - 1;
        emit_interp_info(ctx, semantic, i);
        hdr!(
            ctx.bufs,
            "out vec4 {}_{}{}{};\n",
            stage_output_name_prefix(ctx.prog_type),
            prefix,
            i,
            if ctx.prog_type == Processor::TessCtrl { "[]" } else { "" }
        );
    }
}

/// `emit_ios`.
pub(super) fn emit_ios(ctx: &mut Ctx<'_>) -> u32 {
    ctx.interp_input_mask = 0;
    let mut glsl_ver_required = ctx.glsl_ver_required;

    if ctx.so.is_some_and(|so| so.outputs.len() >= MAX_SO_OUTPUTS) {
        eprintln!("[virglrs] Num outputs exceeded, max is {MAX_SO_OUTPUTS}");
        ctx.bufs.set_hdr_error();
        return glsl_ver_required;
    }

    match ctx.prog_type {
        Processor::Vertex => emit_ios_vs(ctx),
        Processor::Fragment => emit_ios_fs(ctx),
        Processor::Geometry => emit_ios_geom(ctx),
        Processor::TessCtrl => emit_ios_tcs(ctx),
        Processor::TessEval => emit_ios_tes(ctx),
        Processor::Compute => emit_ios_cs(ctx),
    }

    if ctx.bufs.main_error {
        return glsl_ver_required;
    }

    let generic = ctx.generic_ios.matched;
    emit_match_interfaces(
        ctx,
        generic.outputs_expected_mask,
        generic.outputs_emitted_mask,
        Semantic::Generic,
        'g',
    );
    let texcoord = ctx.texcoord_ios;
    emit_match_interfaces(
        ctx,
        texcoord.outputs_expected_mask,
        texcoord.outputs_emitted_mask,
        Semantic::TexCoord,
        't',
    );

    emit_ios_streamout(ctx);
    glsl_ver_required = emit_ios_common(ctx);

    if ctx.prog_type == Processor::Fragment && ctx.key.pstipple_enabled {
        ctx.bufs.hdr("uint stip_temp;\n");
    }
    glsl_ver_required
}

/// `iter_vs_declaration`: a vertex shader's outputs become the passthrough TCS's inputs and
/// outputs both.
fn iter_vs_declaration(ctx: &mut Ctx<'_>, decl: &Declaration) {
    let shader_in_prefix = "vso";
    let shader_out_prefix = "tco";
    if decl.file != File::Output {
        return;
    }
    let sindex = u32::from(decl.semantic.index);
    let first = u32::from(decl.first);
    let last = u32::from(decl.last);
    let array_id = decl.array.map_or(0, u32::from);
    if ctx.inputs.iter().any(|j| {
        j.name == decl.semantic.name
            && j.sid == sindex
            && j.first == first
            && j.usage_mask == decl.usage_mask
            && ((decl.array.is_none() && j.array_id == 0) || j.array_id == array_id)
    }) {
        return;
    }
    let mut io = Io {
        name: decl.semantic.name,
        sid: sindex,
        interpolate: decl.interp.interpolate,
        location: decl.interp.location,
        first,
        last,
        array_id,
        usage_mask: decl.usage_mask,
        num_components: 4,
        ..Io::default()
    };
    io.override_no_wm = io.num_components == 1;
    let mut name_prefix = "";
    match io.name {
        Semantic::PSize => {
            name_prefix = "gl_PointSize";
            io.glsl_predefined_no_emit = true;
            io.glsl_no_index = true;
            io.override_no_wm = true;
            io.glsl_gl_block = true;
            ctx.shader_req_bits |= req::PSIZE;
        }
        Semantic::ClipDist => {
            name_prefix = "gl_ClipDistance";
            io.glsl_predefined_no_emit = true;
            io.glsl_no_index = true;
            io.glsl_gl_block = true;
            ctx.num_in_clip_dist += 4 * (io.last as i32 - io.first as i32 + 1);
            ctx.shader_req_bits |= req::CLIP_DISTANCE;
            if io.last != io.first {
                ctx.guest_sent_io_arrays = true;
            }
        }
        Semantic::Position => {
            name_prefix = "gl_Position";
            io.glsl_predefined_no_emit = true;
            io.glsl_no_index = true;
            io.glsl_gl_block = true;
        }
        Semantic::Patch | Semantic::Generic if (io.first != io.last || io.array_id > 0) => {
            ctx.guest_sent_io_arrays = true;
        }
        _ => {}
    }

    let mut output = io.clone();
    if io.glsl_no_index {
        io.glsl_name = name_prefix.to_string();
        output.glsl_name = name_prefix.to_string();
    } else {
        match io.name {
            Semantic::Fog => {
                io.usage_mask = 0xf;
                io.num_components = 4;
                io.override_no_wm = false;
                io.glsl_name = format!("{shader_in_prefix}_f{}", io.sid);
                output.glsl_name = format!("{shader_out_prefix}_f{}", io.sid);
            }
            Semantic::Color => {
                io.glsl_name = format!("{shader_in_prefix}_c{}", io.sid);
                output.glsl_name = format!("{shader_out_prefix}_c{}", io.sid);
            }
            Semantic::Generic => {
                io.glsl_name = format!("{shader_in_prefix}_g{}", io.sid);
                output.glsl_name = format!("{shader_out_prefix}_g{}", io.sid);
            }
            _ => {
                // The C swaps the prefixes here.
                output.glsl_name = format!("{shader_in_prefix}_{}", io.first);
                io.glsl_name = format!("{shader_out_prefix}_{}", io.first);
            }
        }
    }
    ctx.inputs.push(io);
    ctx.outputs.push(output);
}

/// `vrend_shader_create_passthrough_tcs`.
pub(super) fn passthrough_tcs(
    cfg: &Cfg,
    vs: &Shader,
    key: &Key,
    tess_factors: &[f32; 6],
    vertices_per_patch: u8,
) -> Result<(Strings, Info), Failure> {
    // The C's context is zeroed and never scanned here: no file is indirect.
    let info = scan::Info {
        processor: Processor::Fragment,
        indirect_files: 0,
        dimension_indirect_files: 0,
    };
    let mut ctx = Ctx::new(cfg, key, &info, Processor::TessCtrl);

    for token in &vs.tokens {
        if let Token::Declaration(d) = token {
            iter_vs_declaration(&mut ctx, d);
        }
    }

    ctx.tcs_vertices_out = u32::from(vertices_per_patch);
    super::inst::handle_io_arrays(&mut ctx);

    emit_header(&mut ctx);
    ctx.glsl_ver_required = emit_ios(&mut ctx);

    ctx.bufs.emit("void main() {\n");
    for i in 0..ctx.inputs.len() {
        let input = ctx.inputs[i].clone();
        let output_name = ctx.outputs[i].glsl_name.clone();
        let (out_prefix, in_prefix, postfix) = if input.glsl_gl_block {
            ("gl_out[gl_InvocationID].", "gl_in[gl_InvocationID].", "")
        } else {
            ("", "", "[gl_InvocationID]")
        };
        if input.first == input.last {
            emit!(
                ctx.bufs,
                "{}{}{} = {}{}{};\n",
                out_prefix,
                output_name,
                postfix,
                in_prefix,
                input.glsl_name,
                postfix
            );
        } else {
            // The C computes `size` as a comparison, so a range copies one element or none.
            let size = u32::from(input.last == input.first + 1);
            for k in 0..size {
                emit!(
                    ctx.bufs,
                    "{}{}{}[{}] = {}{}{}[{}];\n",
                    out_prefix,
                    output_name,
                    postfix,
                    k,
                    in_prefix,
                    input.glsl_name,
                    postfix,
                    k
                );
            }
        }
    }
    for (i, f) in tess_factors[..4].iter().enumerate() {
        emit!(ctx.bufs, "gl_TessLevelOuter[{}] = {:.6};\n", i, f);
    }
    for (i, f) in tess_factors[4..].iter().enumerate() {
        emit!(ctx.bufs, "gl_TessLevelInner[{}] = {:.6};\n", i, f);
    }
    ctx.bufs.emit("}\n");

    let mut sinfo = Info::default();
    super::fill_sinfo(&mut ctx, &mut sinfo);
    super::emit_required_sysval_uniforms(&mut ctx.bufs);
    if ctx.bufs.main_error || ctx.bufs.hdr_error {
        return Err(Failure("the passthrough TCS could not be emitted".to_string()));
    }
    Ok((Strings { ver_ext: ctx.bufs.ver_ext, hdr: ctx.bufs.hdr, main: ctx.bufs.main }, sinfo))
}
