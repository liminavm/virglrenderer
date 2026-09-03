// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The instruction walk: how each operand is spelled, and what each opcode becomes.

use super::exit::{
    blockvarname, emit_clip_dist_movs, emit_color_select, emit_fs_clipdistance_load, emit_prescale,
    emit_so_movs, handle_fragment_proc_exit, handle_vertex_proc_exit, prepare_so_movs,
};
use super::tex::{
    emit_lodq, emit_txq, emit_txqs, get_temp, make_ssbo_varstring, translate_atomic,
    translate_load, translate_resq, translate_store, translate_tex,
};
use super::{
    Ctx, Failure, Io, IoDecl, IoDir, MAX_IMMEDIATE, MAX_IO, Qual, VecType, bit32, bit64, emit,
    fail, proc_prefix, req, stage_output_name_prefix, swiz_char,
};
use crate::vrend::shader::Key;
use crate::vrend::tgsi::info::OpType;
use crate::vrend::tgsi::{
    Dst, File, ImmType, Instruction, Opcode, Processor, Semantic, Src, Texture, WRITEMASK_XYZW,
};

/// `dest_info`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct DestInfo {
    pub dtypeprefix: Qual,
    pub dstconv: Qual,
    pub udstconv: Qual,
    pub idstconv: Qual,
    pub dst_override_no_wm: [bool; 2],
    pub dest_index: i32,
}

/// `source_info`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct SourceInfo {
    pub svec4: Qual,
    pub sreg_index: i32,
    pub tg4_has_component: bool,
    pub override_no_wm: [bool; 5],
    pub override_no_cast: [bool; 5],
    pub imm_value: i32,
}

/// C's `%.*g`: `precision` significant digits, the shorter of fixed and exponent notation,
/// trailing zeros dropped.
pub(super) fn fmt_g(v: f64, precision: usize) -> String {
    let p = precision.max(1);
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".to_string() } else { "0".to_string() };
    }
    let e = format!("{:.*e}", p - 1, v);
    let (mant, exp) = e.split_once('e').expect("exponent form");
    let x: i32 = exp.parse().expect("exponent");
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if x >= -4 && x < p as i32 {
        strip(&format!("{:.*}", (p as i32 - 1 - x) as usize, v))
    } else {
        let sign = if x < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", strip(mant), sign, x.abs())
    }
}

/// `reswizzle_dest`: the writemask an IO element that uses fewer than four components takes.
fn reswizzle_dest<'a>(
    io: &Io,
    dst_reg: &Dst,
    reswizzled: &'a mut String,
    writemask: &'a str,
) -> &'a str {
    if io.usage_mask != 0xf {
        if io.num_components > 1 {
            reswizzled.push('.');
            for i in 0..io.num_components {
                if dst_reg.writemask & (1 << i) != 0 {
                    reswizzled.push(swiz_char(i as u8));
                }
            }
        }
        return reswizzled.as_str();
    }
    writemask
}

/// `vrend_shader_write_io_as_src`.
pub(super) fn write_io_as_src(
    result: &mut String,
    array_or_varname: &str,
    io: &Io,
    list: &[Io],
    src: &Src,
    decl_type: IoDecl,
) {
    if io.first == io.last && io.overlapping_array.is_none() {
        result.push_str(&format!("{}{}", io.glsl_name, array_or_varname));
        return;
    }
    let base = io.overlapping_array.map_or(io, |i| &list[i]);
    let offset = i32::from(src.index) - io.first as i32 + io.array_offset as i32;
    let s = match (decl_type, src.indirect) {
        (IoDecl::Block, true) => {
            format!("{}.{}[addr{} + {}]", array_or_varname, base.glsl_name, src.ind.index, offset)
        }
        (IoDecl::Block, false) => format!("{}.{}[{}]", array_or_varname, base.glsl_name, offset),
        (IoDecl::Plain, true) => {
            format!("{}{}[addr{} + {}]", base.glsl_name, array_or_varname, src.ind.index, offset)
        }
        (IoDecl::Plain, false) => format!("{}{}[{}]", base.glsl_name, array_or_varname, offset),
    };
    result.push_str(&s);
}

/// `vrend_shader_write_io_as_dst`.
pub(super) fn write_io_as_dst(
    result: &mut String,
    array_or_varname: &str,
    io: &Io,
    list: &[Io],
    dst: &Dst,
    decl_type: IoDecl,
) {
    if io.first == io.last {
        let s = match io.overlapping_array {
            Some(o) => format!("{}{}[{}]", list[o].glsl_name, array_or_varname, io.array_offset),
            None => format!("{}{}", io.glsl_name, array_or_varname),
        };
        result.push_str(&s);
        return;
    }
    let base = io.overlapping_array.map_or(io, |i| &list[i]);
    let offset = i32::from(dst.index) - io.first as i32 + io.array_offset as i32;
    let s = match (decl_type, dst.indirect) {
        (IoDecl::Block, true) => {
            format!("{}.{}[addr{} + {}]", array_or_varname, base.glsl_name, dst.ind.index, offset)
        }
        (IoDecl::Block, false) => format!("{}.{}[{}]", array_or_varname, base.glsl_name, offset),
        (IoDecl::Plain, true) => {
            format!("{}{}[addr{} + {}]", base.glsl_name, array_or_varname, dst.ind.index, offset)
        }
        (IoDecl::Plain, false) => format!("{}{}[{}]", base.glsl_name, array_or_varname, offset),
    };
    result.push_str(&s);
}

/// `get_destination_info_generic`.
fn destination_info_generic(ctx: &Ctx<'_>, dst_reg: &Dst, io: &Io, writemask: &str) -> String {
    let stage_prefix = stage_output_name_prefix(ctx.prog_type);
    let mut reswizzled = String::new();
    let wm = reswizzle_dest(io, dst_reg, &mut reswizzled, writemask);

    let mut result = String::new();
    let mut blkarray = if ctx.prog_type == Processor::TessCtrl {
        "[gl_InvocationID]".to_string()
    } else {
        String::new()
    };
    let mut decl_type = IoDecl::Plain;
    if io.first != io.last && ctx.prefer_generic_io_block(IoDir::Out) {
        blkarray = blockvarname(stage_prefix, io, &blkarray);
        decl_type = IoDecl::Block;
    }
    write_io_as_dst(&mut result, &blkarray, io, &ctx.outputs, dst_reg, decl_type);
    result.push_str(wm);
    result
}

/// `find_io_index`.
fn find_io_index(io: &[Io], index: i32) -> Option<usize> {
    io.iter().position(|j| j.first as i32 <= index && j.last as i32 >= index)
}

/// `get_destination_info`. Fills `ctx.dst_bufs`, the fp64 originals, and the writemask.
fn get_destination_info(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    dinfo: &mut DestInfo,
    fp64_dsts: &mut [String; 2],
    writemask: &mut String,
) -> bool {
    let dtype = inst.opcode.dst_type();
    if dtype == OpType::Signed || dtype == OpType::Unsigned {
        ctx.shader_req_bits |= req::INTS;
    }
    if dtype == OpType::Double {
        // Doubles need the uvec2 conversion.
        ctx.shader_req_bits |= req::INTS | req::FP64;
    }

    if inst.opcode == Opcode::Txq {
        dinfo.dtypeprefix = Qual::IntBitsToFloat;
    } else {
        match dtype {
            OpType::Unsigned => dinfo.dtypeprefix = Qual::UintBitsToFloat,
            OpType::Signed => dinfo.dtypeprefix = Qual::IntBitsToFloat,
            _ => {}
        }
    }

    for (i, fp64_dst) in fp64_dsts.iter_mut().enumerate().take(usize::from(inst.num_dst)) {
        let dst_reg = inst.dst[i];
        let mut fp64_writemask = String::new();
        dinfo.dst_override_no_wm[i] = false;
        if dst_reg.writemask != WRITEMASK_XYZW {
            writemask.clear();
            writemask.push('.');
            fp64_writemask.push('.');
            for (bit, c) in [(0x1, 'x'), (0x2, 'y'), (0x4, 'z'), (0x8, 'w')] {
                if dst_reg.writemask & bit != 0 {
                    writemask.push(c);
                }
            }
            if dtype == OpType::Double {
                if dst_reg.writemask & 0x3 != 0 {
                    fp64_writemask.push('x');
                }
                if dst_reg.writemask & 0xc != 0 {
                    fp64_writemask.push('y');
                }
                dinfo.dstconv = if fp64_writemask.len() == 2 { Qual::Double } else { Qual::DVec2 };
            } else {
                let width = writemask.len() as u32 - 1;
                dinfo.dstconv = Qual::vec(Qual::Float, width);
                dinfo.udstconv = Qual::vec(Qual::Uint, width);
                dinfo.idstconv = Qual::vec(Qual::Int, width);
            }
        } else {
            dinfo.dstconv = if dtype == OpType::Double { Qual::DVec2 } else { Qual::Vec4 };
            dinfo.udstconv = Qual::UVec4;
            dinfo.idstconv = Qual::IVec4;
        }

        match dst_reg.file {
            File::Output => {
                let Some(j) = find_io_index(&ctx.outputs, i32::from(dst_reg.index)) else {
                    return false;
                };
                if inst.precise
                    && !ctx.outputs[j].invariant
                    && ctx.outputs[j].name != Semantic::ClipVertex
                    && ctx.cfg.has_gpu_shader5
                {
                    ctx.outputs[j].precise = true;
                    ctx.shader_req_bits |= req::GPU_SHADER5;
                }
                let output = ctx.outputs[j].clone();

                if ctx.glsl_ver_required >= 140 && output.name == Semantic::ClipVertex {
                    ctx.dst_bufs[i] = if ctx.prog_type == Processor::TessCtrl {
                        format!("{}[gl_InvocationID]", output.glsl_name)
                    } else if ctx.is_last_vertex_stage {
                        "clipv_tmp".to_string()
                    } else {
                        output.glsl_name.clone()
                    };
                } else if output.name == Semantic::ClipDist {
                    let mut clip_indirect = String::new();
                    if output.first != output.last {
                        clip_indirect = if dst_reg.indirect {
                            format!("+ addr{}", dst_reg.ind.index)
                        } else {
                            format!("+ {}", i32::from(dst_reg.index) - output.first as i32)
                        };
                    }
                    ctx.dst_bufs[i] =
                        format!("clip_dist_temp[{} {}]{}", output.sid, clip_indirect, writemask);
                } else if matches!(
                    output.name,
                    Semantic::TessOuter | Semantic::TessInner | Semantic::SampleMask
                ) {
                    let idx = match dst_reg.writemask {
                        0x1 => 0,
                        0x2 => 1,
                        0x4 => 2,
                        0x8 => 3,
                        _ => 0,
                    };
                    ctx.dst_bufs[i] = format!("{}[{}]", output.glsl_name, idx);
                    if output.is_int {
                        dinfo.dtypeprefix = Qual::FloatBitsToInt;
                        dinfo.dstconv = Qual::Int;
                    }
                } else {
                    let wm_or_none = if output.override_no_wm { "" } else { writemask.as_str() };
                    if output.glsl_gl_block {
                        ctx.dst_bufs[i] = format!(
                            "gl_out[{}].{}{}",
                            if ctx.prog_type == Processor::TessCtrl {
                                "gl_InvocationID"
                            } else {
                                "0"
                            },
                            output.glsl_name,
                            wm_or_none
                        );
                    } else if output.name == Semantic::Generic || output.name == Semantic::TexCoord
                    {
                        ctx.dst_bufs[i] =
                            destination_info_generic(ctx, &dst_reg, &output, writemask);
                        dinfo.dst_override_no_wm[i] = output.override_no_wm;
                    } else if output.name == Semantic::Patch {
                        let mut reswizzled = String::new();
                        let wm = reswizzle_dest(&output, &dst_reg, &mut reswizzled, writemask)
                            .to_string();
                        let mut s = String::new();
                        write_io_as_dst(&mut s, "", &output, &ctx.outputs, &dst_reg, IoDecl::Plain);
                        if !output.override_no_wm {
                            s.push_str(&wm);
                        }
                        ctx.dst_bufs[i] = s;
                        dinfo.dst_override_no_wm[i] = output.override_no_wm;
                    } else {
                        ctx.dst_bufs[i] = if ctx.prog_type == Processor::TessCtrl {
                            format!("{}[gl_InvocationID]{}", output.glsl_name, wm_or_none)
                        } else {
                            format!("{}{}", output.glsl_name, wm_or_none)
                        };
                        dinfo.dst_override_no_wm[i] = output.override_no_wm;
                    }
                    if output.is_int {
                        if dinfo.dtypeprefix == Qual::None {
                            dinfo.dtypeprefix = Qual::FloatBitsToInt;
                        }
                        dinfo.dstconv = Qual::Int;
                    } else if output.ty == VecType::Uint {
                        if dinfo.dtypeprefix == Qual::None {
                            dinfo.dtypeprefix = Qual::FloatBitsToUint;
                        }
                        dinfo.dstconv = dinfo.udstconv;
                    } else if output.ty == VecType::Int {
                        if dinfo.dtypeprefix == Qual::None {
                            dinfo.dtypeprefix = Qual::FloatBitsToInt;
                        }
                        dinfo.dstconv = dinfo.idstconv;
                    }
                    if output.name == Semantic::PSize {
                        dinfo.dstconv = Qual::Float;
                    }
                }
            }
            File::Temporary => {
                let temp = get_temp(ctx, dst_reg.indirect, 0, i32::from(dst_reg.index));
                ctx.dst_bufs[i] = format!("{temp}{writemask}");
                if inst.precise
                    && let Some(r) = ctx.find_temp_range(i32::from(dst_reg.index))
                    && ctx.cfg.has_gpu_shader5
                {
                    ctx.temp_ranges[r].precise_result = true;
                    ctx.shader_req_bits |= req::GPU_SHADER5;
                }
            }
            File::Image => {
                let cname = proc_prefix(ctx.prog_type);
                if ctx.info.is_indirect(File::Image) {
                    let basearrayidx = ctx.lookup_image_array(i32::from(dst_reg.index));
                    if dst_reg.indirect {
                        if dst_reg.ind.file != File::Address {
                            return false;
                        }
                        ctx.dst_bufs[i] = format!(
                            "{}img{}[addr{} + {}]",
                            cname,
                            basearrayidx,
                            dst_reg.ind.index,
                            i32::from(dst_reg.index) - basearrayidx
                        );
                    } else {
                        ctx.dst_bufs[i] = format!(
                            "{}img{}[{}]",
                            cname,
                            basearrayidx,
                            i32::from(dst_reg.index) - basearrayidx
                        );
                    }
                } else {
                    ctx.dst_bufs[i] = format!("{}img{}", cname, dst_reg.index);
                }
                dinfo.dest_index = i32::from(dst_reg.index);
            }
            File::Buffer => {
                ctx.dst_bufs[i] = make_ssbo_varstring(ctx, dst_reg.index as u32);
                dinfo.dest_index = i32::from(dst_reg.index);
            }
            File::Memory => ctx.dst_bufs[i] = "values".to_string(),
            File::Address => ctx.dst_bufs[i] = format!("addr{}", dst_reg.index),
            _ => return false,
        }

        if dtype == OpType::Double {
            *fp64_dst = ctx.dst_bufs[i].clone();
            ctx.dst_bufs[i] = format!("fp64_dst[{i}]{fp64_writemask}");
            writemask.clear();
        }
    }
    true
}

/// `shift_swizzles`: an IO element using fewer than four components shifts the swizzle's
/// names.
fn shift_swizzles(io: &Io, src: &Src, shifted: &mut String, swizzle: &str) -> bool {
    if io.usage_mask != 0xf && !swizzle.is_empty() {
        if io.num_components > 1 {
            shifted.push('.');
            shifted.push(swiz_char(src.swizzle[0]));
            shifted.push(swiz_char(src.swizzle[1]));
            shifted.push(if u32::from(src.swizzle[2]) < io.num_components {
                swiz_char(src.swizzle[2])
            } else {
                'x'
            });
            shifted.push(if u32::from(src.swizzle[3]) < io.num_components {
                swiz_char(src.swizzle[3])
            } else {
                'x'
            });
        }
        return true;
    }
    false
}

/// `get_source_info_generic`.
#[allow(clippy::too_many_arguments)]
fn source_info_generic(
    ctx: &Ctx<'_>,
    iot: IoDir,
    srcstypeprefix: Qual,
    prefix: &str,
    src: &Src,
    io: &Io,
    list: &[Io],
    arrayname: &str,
    swizzle: &str,
) -> String {
    let mut shifted = String::new();
    if swizzle.starts_with(')') {
        shifted.push(')');
    }
    let swizzle =
        if shift_swizzles(io, src, &mut shifted, swizzle) { shifted.as_str() } else { swizzle };

    let mut result = format!("{}({}", srcstypeprefix.s(), prefix);
    let mut decl_type = IoDecl::Plain;
    let mut arrayname = arrayname.to_string();
    if (io.first != io.last || io.overlapping_array.is_some()) && ctx.prefer_generic_io_block(iot) {
        let array = io.overlapping_array.map_or(io, |i| &list[i]);
        let stage_prefix = if iot == IoDir::In {
            ctx.stage_input_name_prefix(ctx.prog_type)
        } else {
            stage_output_name_prefix(ctx.prog_type)
        };
        arrayname = blockvarname(stage_prefix, array, &arrayname);
        decl_type = IoDecl::Block;
    }
    write_io_as_src(&mut result, &arrayname, io, list, src, decl_type);
    result.push_str(&format!("{})", if io.is_int { "" } else { swizzle }));
    result
}

/// `get_source_info_patch`.
fn source_info_patch(
    srcstypeprefix: Qual,
    prefix: &str,
    src: &Src,
    io: &Io,
    list: &[Io],
    arrayname: &str,
    swizzle: &str,
) -> String {
    let mut shifted = String::new();
    if swizzle.starts_with(')') {
        shifted.push(')');
    }
    let swizzle =
        if shift_swizzles(io, src, &mut shifted, swizzle) { shifted.as_str() } else { swizzle };
    let mut result = format!("{}({}", srcstypeprefix.s(), prefix);
    write_io_as_src(
        &mut result,
        if io.last == io.first { arrayname } else { "" },
        io,
        list,
        src,
        IoDecl::Plain,
    );
    result.push_str(&format!("{})", if io.is_int { "" } else { swizzle }));
    result
}

/// `get_tesslevel_as_source`.
fn tesslevel_as_source(prefix: &str, name: &str, src: &Src) -> String {
    format!(
        "{}(vec4({}[{}], {}[{}], {}[{}], {}[{}]))",
        prefix,
        name,
        src.swizzle[0],
        name,
        src.swizzle[1],
        name,
        src.swizzle[2],
        name,
        src.swizzle[3]
    )
}

/// `get_source_swizzle`.
fn source_swizzle(src: &Src) -> String {
    if src.swizzle != [0, 1, 2, 3] {
        let mut s = String::from(".");
        for c in src.swizzle {
            s.push(swiz_char(c));
        }
        s
    } else {
        String::new()
    }
}

/// `create_swizzled_clipdist`.
#[allow(clippy::too_many_arguments)]
fn swizzled_clipdist(
    ctx: &Ctx<'_>,
    src: &Src,
    input_idx: usize,
    gl_in: bool,
    stypeprefix: &str,
    prefix: &str,
    arrayname: &str,
    offset: i32,
) -> String {
    let has_prop = (ctx.num_cull_dist_prop + ctx.num_clip_dist_prop) > 0;
    let num_culls = i32::from(if has_prop { ctx.num_cull_dist_prop } else { ctx.key.num_out_cull });
    let mut num_clips =
        i32::from(if has_prop { ctx.num_clip_dist_prop } else { ctx.key.num_out_clip });
    if ctx.num_in_clip_dist != 0 && num_culls + num_clips == 0 {
        num_clips = ctx.num_in_clip_dist;
    }
    let base_idx = ctx.inputs[input_idx].sid as i32 * 4;
    // This does not work for indirect addressing.
    let base_offset = (i32::from(src.index) - offset) * 4;

    // With arrays enabled, and only when gl_ClipDistance or gl_CullDistance are emitted (>4),
    // indirect addressing is added.
    let clip_indirect =
        if src.indirect && ((num_clips > 4 && base_idx < num_clips) || num_culls > 4) {
            format!("4*addr{} +", src.ind.index)
        } else if i32::from(src.index) != offset {
            format!("4*{} +", i32::from(src.index) - offset)
        } else {
            String::new()
        };

    let mut vec = Vec::with_capacity(4);
    for cc in 0..4 {
        let mut cc_name = ctx.inputs[input_idx].glsl_name.as_str();
        let mut idx = base_idx + i32::from(src.swizzle[cc]);
        if num_culls != 0 && idx + base_offset >= num_clips {
            idx -= num_clips;
            cc_name = "gl_CullDistance";
        }
        vec.push(if gl_in {
            format!("{}gl_in{}.{}[{} {}]", prefix, arrayname, cc_name, clip_indirect, idx)
        } else {
            format!("{}{}{}[{} {}]", prefix, arrayname, cc_name, clip_indirect, idx)
        });
    }
    format!("{}(vec4({},{},{},{}))", stypeprefix, vec[0], vec[1], vec[2], vec[3])
}

/// `load_clipdist_fs`.
fn load_clipdist_fs(
    ctx: &Ctx<'_>,
    src: &Src,
    input_idx: usize,
    stypeprefix: &str,
    offset: i32,
) -> String {
    let swz: String = src.swizzle.iter().map(|&c| swiz_char(c)).collect();
    let base_idx = ctx.inputs[input_idx].sid;
    let clip_indirect = if src.indirect {
        format!("addr{} + {}", src.ind.index, base_idx)
    } else {
        format!("{} + {}", i32::from(src.index) - offset, base_idx)
    };
    format!("{}(clip_dist_temp[{}].{})", stypeprefix, clip_indirect, swz)
}

/// `get_source_info`. Fills `ctx.src_bufs` and, for the interpolation opcodes, the swizzle of
/// the first operand that the caller applies to the result instead.
fn get_source_info(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sinfo: &mut SourceInfo,
    src_swizzle0: &mut String,
) -> bool {
    let mut stprefix = false;
    let mut stypeprefix = Qual::None;
    let mut stype = inst.opcode.src_type();

    if stype == OpType::Signed || stype == OpType::Unsigned {
        ctx.shader_req_bits |= req::INTS;
    }
    if stype == OpType::Double {
        ctx.shader_req_bits |= req::INTS | req::FP64;
    }

    match stype {
        OpType::Double => {
            stypeprefix = Qual::FloatBitsToUint;
            sinfo.svec4 = Qual::DVec2;
            stprefix = true;
        }
        OpType::Unsigned => {
            stypeprefix = Qual::FloatBitsToUint;
            sinfo.svec4 = Qual::UVec4;
            stprefix = true;
        }
        OpType::Signed => {
            stypeprefix = Qual::FloatBitsToInt;
            sinfo.svec4 = Qual::IVec4;
            stprefix = true;
        }
        _ => {}
    }

    let interp =
        matches!(inst.opcode, Opcode::InterpSample | Opcode::InterpOffset | Opcode::InterpCentroid);

    for i in 0..usize::from(inst.num_src).min(5) {
        let src = inst.src[i];
        let isfloatabsolute = src.absolute && stype != OpType::Double;

        sinfo.override_no_wm[i] = false;
        sinfo.override_no_cast[i] = false;

        let mut prefix = String::new();
        if src.negate {
            prefix.push('-');
        }
        if isfloatabsolute {
            prefix.push_str("abs(");
        }

        let mut arrayname = String::new();
        if src.dimension {
            if src.dim.indirect {
                if src.dim_ind.file != File::Address {
                    return false;
                }
                arrayname = format!("[addr{}]", src.dim_ind.index);
            } else {
                arrayname = format!("[{}]", src.dim.index);
            }
        }

        // The swizzle, as the C builds it into `swizzle`; for the interpolation opcodes the
        // first operand's swizzle goes to the caller and the operand's own stays empty.
        let mut w = String::new();
        if isfloatabsolute {
            w.push(')');
        }
        let swz_idx = w.len();
        w.push_str(&source_swizzle(&src));
        if src.file == File::Input
            && ctx.prog_type == Processor::Vertex
            && let Some(j) = find_io_index(&ctx.inputs, i32::from(src.index))
            && ctx.key.vs.attrib_zyxw_bitmask & bit32(ctx.inputs[j].first) != 0
        {
            w.truncate(swz_idx);
            w.push_str(".zyxw");
            w.push_str(&source_swizzle(&src));
        }
        let redirected = interp && i == 0;
        if redirected {
            *src_swizzle0 = w.clone();
        }
        let swizzle: &str = if redirected { "" } else { w.as_str() };
        let stp = stypeprefix.s();

        match src.file {
            File::Input => {
                let Some(j) = find_io_index(&ctx.inputs, i32::from(src.index)) else {
                    return false;
                };
                let input = ctx.inputs[j].clone();

                let buf = if ctx.prog_type == Processor::Fragment
                    && ctx.key.color_two_side
                    && input.name == Semantic::Color
                {
                    format!(
                        "{}({}{}{}{}{})",
                        stp, prefix, "realcolor", input.sid, arrayname, swizzle
                    )
                } else if input.glsl_gl_block {
                    // A GS clip distance input needs a conversion.
                    if input.name == Semantic::ClipDist {
                        swizzled_clipdist(
                            ctx,
                            &src,
                            j,
                            true,
                            stp,
                            &prefix,
                            &arrayname,
                            input.first as i32,
                        )
                    } else {
                        format!(
                            "{}(vec4({}gl_in{}.{}){})",
                            stp, prefix, arrayname, input.glsl_name, swizzle
                        )
                    }
                } else if input.name == Semantic::PrimId {
                    format!("{}(vec4(intBitsToFloat({})))", stp, input.glsl_name)
                } else if input.name == Semantic::Face {
                    format!("{}({} ? 1.0 : -1.0)", stp, input.glsl_name)
                } else if input.name == Semantic::ClipDist {
                    if ctx.prog_type == Processor::Fragment {
                        load_clipdist_fs(ctx, &src, j, stp, input.first as i32)
                    } else {
                        swizzled_clipdist(
                            ctx,
                            &src,
                            j,
                            false,
                            stp,
                            &prefix,
                            &arrayname,
                            input.first as i32,
                        )
                    }
                } else if input.name == Semantic::TessOuter || input.name == Semantic::TessInner {
                    tesslevel_as_source(&prefix, &input.glsl_name, &src)
                } else {
                    let mut srcstypeprefix = stypeprefix;
                    if input.ty != VecType::Float {
                        srcstypeprefix = if stype == OpType::Unsigned {
                            Qual::UVec4
                        } else if stype == OpType::Signed {
                            Qual::IVec4
                        } else if input.ty == VecType::Int {
                            Qual::IntBitsToFloat
                        } else {
                            Qual::UintBitsToFloat
                        };
                    }
                    let sw = if input.is_int { "" } else { swizzle };
                    if inst.opcode == Opcode::InterpSample && i == 1 {
                        format!(
                            "floatBitsToInt({}{}{}{})",
                            prefix, input.glsl_name, arrayname, swizzle
                        )
                    } else if input.name == Semantic::Generic || input.name == Semantic::TexCoord {
                        source_info_generic(
                            ctx,
                            IoDir::In,
                            srcstypeprefix,
                            &prefix,
                            &src,
                            &input,
                            &ctx.inputs,
                            &arrayname,
                            swizzle,
                        )
                    } else if input.name == Semantic::Patch {
                        source_info_patch(
                            srcstypeprefix,
                            &prefix,
                            &src,
                            &input,
                            &ctx.inputs,
                            &arrayname,
                            swizzle,
                        )
                    } else if input.name == Semantic::Position
                        && ctx.prog_type == Processor::Vertex
                        && input.first != input.last
                    {
                        if src.indirect {
                            format!(
                                "{}({}{}{}[addr{} + {}]{})",
                                srcstypeprefix.s(),
                                prefix,
                                input.glsl_name,
                                arrayname,
                                src.ind.index,
                                src.index,
                                sw
                            )
                        } else {
                            format!(
                                "{}({}{}{}[{}]{})",
                                srcstypeprefix.s(),
                                prefix,
                                input.glsl_name,
                                arrayname,
                                src.index,
                                sw
                            )
                        }
                    } else {
                        format!(
                            "{}({}{}{}{})",
                            srcstypeprefix.s(),
                            prefix,
                            input.glsl_name,
                            arrayname,
                            sw
                        )
                    }
                };
                ctx.src_bufs[i] = buf;
                sinfo.override_no_wm[i] = input.override_no_wm;
            }
            File::Output => {
                let Some(j) = find_io_index(&ctx.outputs, i32::from(src.index)) else {
                    return false;
                };
                if inst.opcode == Opcode::Fbfetch {
                    ctx.outputs[j].fbfetch_used = true;
                    ctx.shader_req_bits |= req::FBFETCH;
                }
                let output = ctx.outputs[j].clone();
                let mut srcstypeprefix = stypeprefix;
                if stype == OpType::Unsigned && output.is_int {
                    srcstypeprefix = Qual::None;
                }
                if output.glsl_gl_block {
                    if output.name == Semantic::ClipDist {
                        let mut clip_indirect = String::new();
                        if output.first != output.last {
                            clip_indirect = if src.indirect {
                                format!("+ addr{}", src.ind.index)
                            } else {
                                format!("+ {}", i32::from(src.index) - output.first as i32)
                            };
                        }
                        ctx.src_bufs[i] =
                            format!("clip_dist_temp[{}{}]", output.sid, clip_indirect);
                    }
                } else if output.name == Semantic::Generic {
                    ctx.src_bufs[i] = source_info_generic(
                        ctx,
                        IoDir::Out,
                        srcstypeprefix,
                        &prefix,
                        &src,
                        &output,
                        &ctx.outputs,
                        &arrayname,
                        swizzle,
                    );
                } else if output.name == Semantic::Patch {
                    ctx.src_bufs[i] = source_info_patch(
                        srcstypeprefix,
                        &prefix,
                        &src,
                        &output,
                        &ctx.outputs,
                        &arrayname,
                        swizzle,
                    );
                } else if output.name == Semantic::TessOuter || output.name == Semantic::TessInner {
                    ctx.src_bufs[i] = tesslevel_as_source(&prefix, &output.glsl_name, &src);
                } else {
                    ctx.src_bufs[i] = format!(
                        "{}({}{}{}{})",
                        srcstypeprefix.s(),
                        prefix,
                        output.glsl_name,
                        arrayname,
                        if output.is_int { "" } else { swizzle }
                    );
                }
                sinfo.override_no_wm[i] = output.override_no_wm;
            }
            File::Temporary => {
                if ctx.find_temp_range(i32::from(src.index)).is_none() {
                    return false;
                }
                if inst.opcode == Opcode::InterpSample && i == 1 {
                    stprefix = true;
                    stypeprefix = Qual::FloatBitsToInt;
                }
                let temp =
                    get_temp(ctx, src.indirect, i32::from(src.ind.index), i32::from(src.index));
                ctx.src_bufs[i] = format!(
                    "{}{}vec4({}{}){}{}",
                    stypeprefix.s(),
                    if stprefix { '(' } else { ' ' },
                    prefix,
                    temp,
                    swizzle,
                    if stprefix { ')' } else { ' ' }
                );
            }
            File::Constant => {
                let cname = proc_prefix(ctx.prog_type);
                if src.dimension && src.dim.index != 0 {
                    let dim = i32::from(src.dim.index);
                    if src.dim.indirect {
                        if src.dim_ind.file != File::Address {
                            return false;
                        }
                        ctx.shader_req_bits |= req::GPU_SHADER5;
                        if src.indirect {
                            if src.ind.file != File::Address {
                                return false;
                            }
                            ctx.src_bufs[i] = format!(
                                "{}({}{}uboarr[addr{}].ubocontents[addr{} + {}]{})",
                                stp,
                                prefix,
                                cname,
                                src.dim_ind.index,
                                src.ind.index,
                                src.index,
                                swizzle
                            );
                        } else {
                            ctx.src_bufs[i] = format!(
                                "{}({}{}uboarr[addr{}].ubocontents[{}]{})",
                                stp, prefix, cname, src.dim_ind.index, src.index, swizzle
                            );
                        }
                    } else if ctx.info.is_dimension_indirect(File::Constant) {
                        let arr = dim - ctx.ubo_base as i32;
                        if src.indirect {
                            ctx.src_bufs[i] = format!(
                                "{}({}{}uboarr[{}].ubocontents[addr{} + {}]{})",
                                stp, prefix, cname, arr, src.ind.index, src.index, swizzle
                            );
                        } else {
                            ctx.src_bufs[i] = format!(
                                "{}({}{}uboarr[{}].ubocontents[{}]{})",
                                stp, prefix, cname, arr, src.index, swizzle
                            );
                        }
                    } else if src.indirect {
                        if src.ind.file != File::Address {
                            return false;
                        }
                        ctx.src_bufs[i] = format!(
                            "{}({}{}ubo{}contents[addr{} + {}]{})",
                            stp, prefix, cname, dim, src.ind.index, src.index, swizzle
                        );
                    } else {
                        ctx.src_bufs[i] = format!(
                            "{}({}{}ubo{}contents[{}]{})",
                            stp, prefix, cname, dim, src.index, swizzle
                        );
                    }
                } else {
                    let mut csp = Qual::None;
                    ctx.shader_req_bits |= req::INTS;
                    if inst.opcode == Opcode::InterpSample && i == 1 {
                        csp = Qual::IVec4;
                    } else if stype == OpType::Float || stype == OpType::Untyped {
                        csp = Qual::UintBitsToFloat;
                    } else if stype == OpType::Signed {
                        csp = Qual::IVec4;
                    }
                    if src.indirect {
                        if src.ind.file != File::Address {
                            return false;
                        }
                        ctx.src_bufs[i] = format!(
                            "{}{}({}const{}[addr{} + {}]{})",
                            prefix,
                            csp.s(),
                            cname,
                            0,
                            src.ind.index,
                            src.index,
                            swizzle
                        );
                    } else {
                        ctx.src_bufs[i] = format!(
                            "{}{}({}const{}[{}]{})",
                            prefix,
                            csp.s(),
                            cname,
                            0,
                            src.index,
                            swizzle
                        );
                    }
                }
            }
            File::Sampler => {
                let cname = proc_prefix(ctx.prog_type);
                if ctx.info.is_indirect(File::Sampler) {
                    let basearrayidx = ctx.lookup_sampler_array(i32::from(src.index));
                    if src.indirect {
                        ctx.src_bufs[i] = format!(
                            "{}samp{}[addr{}+{}]{}",
                            cname,
                            basearrayidx,
                            src.ind.index,
                            i32::from(src.index) - basearrayidx,
                            swizzle
                        );
                    } else {
                        ctx.src_bufs[i] = format!(
                            "{}samp{}[{}]{}",
                            cname,
                            basearrayidx,
                            i32::from(src.index) - basearrayidx,
                            swizzle
                        );
                    }
                } else {
                    ctx.src_bufs[i] = format!("{}samp{}{}", cname, src.index, swizzle);
                }
                sinfo.sreg_index = i32::from(src.index);
            }
            File::Image => {
                let cname = proc_prefix(ctx.prog_type);
                if ctx.info.is_indirect(File::Image) {
                    let basearrayidx = ctx.lookup_image_array(i32::from(src.index));
                    if src.indirect {
                        if src.ind.file != File::Address {
                            return false;
                        }
                        ctx.src_bufs[i] = format!(
                            "{}img{}[addr{} + {}]",
                            cname,
                            basearrayidx,
                            src.ind.index,
                            i32::from(src.index) - basearrayidx
                        );
                    } else {
                        ctx.src_bufs[i] = format!(
                            "{}img{}[{}]",
                            cname,
                            basearrayidx,
                            i32::from(src.index) - basearrayidx
                        );
                    }
                } else {
                    ctx.src_bufs[i] = format!("{}img{}{}", cname, src.index, swizzle);
                }
                sinfo.sreg_index = i32::from(src.index);
            }
            File::Buffer => {
                ctx.src_bufs[i] = make_ssbo_varstring(ctx, src.index as u32);
                sinfo.sreg_index = i32::from(src.index);
            }
            File::Memory => {
                ctx.src_bufs[i] = "values".to_string();
                sinfo.sreg_index = i32::from(src.index);
            }
            File::Immediate => {
                if src.index < 0 || src.index as usize >= MAX_IMMEDIATE {
                    eprintln!("[virglrs] Immediate exceeded, max is {MAX_IMMEDIATE}");
                    return false;
                }
                // A slot the program never filled reads as the C's zeroed one.
                let imd = ctx
                    .imm
                    .get(src.index as usize)
                    .copied()
                    .unwrap_or(super::Immed { ty: ImmType::Float32, val: [0; 4] });
                let mut vtype = Qual::Vec4;
                let mut imm_stypeprefix = stypeprefix;

                if (inst.opcode == Opcode::Tg4 && i == 1)
                    || (inst.opcode == Opcode::InterpSample && i == 1)
                {
                    stype = OpType::Signed;
                }

                match imd.ty {
                    ImmType::Int32 => {
                        vtype = Qual::IVec4;
                        if stype == OpType::Signed {
                            imm_stypeprefix = Qual::None;
                        } else if stype == OpType::Unsigned {
                            imm_stypeprefix = Qual::UVec4;
                        } else if stype == OpType::Float || stype == OpType::Untyped {
                            imm_stypeprefix = Qual::IntBitsToFloat;
                        }
                    }
                    ImmType::Uint32 => {
                        vtype = Qual::UVec4;
                        if stype == OpType::Unsigned {
                            imm_stypeprefix = Qual::None;
                        } else if stype == OpType::Signed {
                            imm_stypeprefix = Qual::IVec4;
                        } else if stype == OpType::Float || stype == OpType::Untyped {
                            imm_stypeprefix = Qual::UintBitsToFloat;
                        }
                    }
                    ImmType::Float64 => {
                        vtype = Qual::UVec4;
                        imm_stypeprefix = if stype == OpType::Double {
                            Qual::None
                        } else {
                            Qual::UintBitsToFloat
                        };
                    }
                    ImmType::Int64 | ImmType::Uint64 | ImmType::Float32 => {}
                }

                // A vec4 of immediates.
                let mut buf = format!("{}{}({}(", prefix, imm_stypeprefix.s(), vtype.s());
                for j in 0..4 {
                    let idx = (src.swizzle[j] & 3) as usize;
                    if inst.opcode == Opcode::Tg4 && i == 1 && j == 0 && imd.val[idx] > 0 {
                        sinfo.tg4_has_component = true;
                    }
                    let temp = match imd.ty {
                        ImmType::Float32 => {
                            let f = f32::from_bits(imd.val[idx]);
                            if f.is_infinite() || f.is_nan() {
                                ctx.shader_req_bits |= req::INTS;
                                format!("uintBitsToFloat({}U)", imd.val[idx])
                            } else {
                                fmt_g(f64::from(f), 8)
                            }
                        }
                        ImmType::Uint32 => format!("{}U", imd.val[idx]),
                        ImmType::Int32 => {
                            sinfo.imm_value = imd.val[idx] as i32;
                            format!("{}", imd.val[idx] as i32)
                        }
                        ImmType::Float64 => format!("{}U", imd.val[idx]),
                        other => {
                            eprintln!("[virglrs] Unhandled imm type: {:x}", other as u8);
                            return false;
                        }
                    };
                    buf.push_str(&temp);
                    if j < 3 {
                        buf.push(',');
                    } else {
                        buf.push_str("))");
                        if isfloatabsolute {
                            buf.push(')');
                        }
                    }
                }
                ctx.src_bufs[i] = buf;
            }
            File::SystemValue => {
                let Some(sv) = ctx
                    .system_values
                    .iter()
                    .find(|s| s.first as i32 == i32::from(src.index))
                    .cloned()
                else {
                    return false;
                };
                let name = sv.glsl_name.as_str();
                let sw = |k: usize| swiz_char(src.swizzle[k]);
                match sv.name {
                    Semantic::VertexId
                    | Semantic::VertexIdNoBase
                    | Semantic::InstanceId
                    | Semantic::PrimId
                    | Semantic::VerticesIn
                    | Semantic::InvocationId
                    | Semantic::SampleId => {
                        ctx.src_bufs[i] = if inst.opcode == Opcode::InterpSample && i == 1 {
                            format!("ivec4({name})")
                        } else {
                            format!("{}(vec4(intBitsToFloat({})))", stp, name)
                        };
                    }
                    Semantic::HelperInvocation => ctx.src_bufs[i] = format!("uvec4({name})"),
                    Semantic::TessInner | Semantic::TessOuter => {
                        ctx.src_bufs[i] = format!(
                            "{}(vec4({}[{}], {}[{}], {}[{}], {}[{}]))",
                            prefix,
                            name,
                            src.swizzle[0],
                            name,
                            src.swizzle[1],
                            name,
                            src.swizzle[2],
                            name,
                            src.swizzle[3]
                        );
                    }
                    Semantic::SamplePos => {
                        // gl_SamplePosition is a vec2; the semantic is a vec4 with z = w = 0.
                        let components =
                            ["gl_SamplePosition.x", "gl_SamplePosition.y", "0.0", "0.0"];
                        let c = |k: usize| components[(src.swizzle[k] & 3) as usize];
                        ctx.src_bufs[i] =
                            format!("{}(vec4({}, {}, {}, {}))", prefix, c(0), c(1), c(2), c(3));
                    }
                    Semantic::TessCoord => {
                        ctx.src_bufs[i] = format!(
                            "{}(vec4({}.{}, {}.{}, {}.{}, {}.{}))",
                            prefix,
                            name,
                            sw(0),
                            name,
                            sw(1),
                            name,
                            sw(2),
                            name,
                            sw(3)
                        );
                    }
                    Semantic::GridSize | Semantic::ThreadId | Semantic::BlockId => {
                        let mov_conv =
                            if inst.opcode == Opcode::Mov && inst.dst[0].file == File::Temporary {
                                Qual::UintBitsToFloat
                            } else {
                                Qual::None
                            };
                        ctx.src_bufs[i] = format!(
                            "{}(uvec4({}.{}, {}.{}, {}.{}, {}.{}))",
                            mov_conv.s(),
                            name,
                            sw(0),
                            name,
                            sw(1),
                            name,
                            sw(2),
                            name,
                            sw(3)
                        );
                        sinfo.override_no_cast[i] = true;
                    }
                    Semantic::SampleMask => {
                        let mut vec_type = "ivec4";
                        let mut srcstypeprefix = Qual::None;
                        if stypeprefix == Qual::None {
                            srcstypeprefix = Qual::IntBitsToFloat;
                        } else if stype == OpType::Unsigned {
                            vec_type = "uvec4";
                        }
                        ctx.shader_req_bits |= req::SAMPLE_SHADING | req::INTS;
                        let c = |k: usize| if src.swizzle[k] == 0 { name } else { "0" };
                        ctx.src_bufs[i] = format!(
                            "{}({}({}, {}, {}, {}))",
                            srcstypeprefix.s(),
                            vec_type,
                            c(0),
                            c(1),
                            c(2),
                            c(3)
                        );
                    }
                    _ => {
                        ctx.src_bufs[i] = format!("{}{}", prefix, name);
                        sinfo.override_no_wm[i] = sv.override_no_wm;
                    }
                }
            }
            File::HwAtomic => {
                for j in 0..ctx.abo_idx.len() {
                    if i32::from(src.dim.index) == ctx.abo_idx[j]
                        && i32::from(src.index) >= ctx.abo_offsets[j]
                        && i32::from(src.index) < ctx.abo_offsets[j] + ctx.abo_sizes[j]
                    {
                        let abo_idx = ctx.abo_idx[j];
                        let abo_offset = ctx.abo_offsets[j] * 4;
                        if ctx.abo_sizes[j] > 1 {
                            let offset = i32::from(src.index) - ctx.abo_offsets[j];
                            if src.indirect {
                                if src.ind.file != File::Address {
                                    return false;
                                }
                                ctx.src_bufs[i] = format!(
                                    "ac{}_{}[addr{} + {}]",
                                    abo_idx, abo_offset, src.ind.index, offset
                                );
                            } else {
                                ctx.src_bufs[i] =
                                    format!("ac{}_{}[{}]", abo_idx, abo_offset, offset);
                            }
                        } else {
                            ctx.src_bufs[i] = format!("ac{}_{}", abo_idx, abo_offset);
                        }
                        break;
                    }
                }
                sinfo.sreg_index = i32::from(src.index);
            }
            _ => return false,
        }

        if stype == OpType::Double {
            let isabsolute = src.absolute;
            let fp64_src = ctx.src_bufs[i].clone();
            ctx.src_bufs[i] = format!("fp64_src[{i}]");
            emit!(
                ctx.bufs,
                "{}.x = {}packDouble2x32(uvec2({}{})){};\n",
                ctx.src_bufs[i].clone(),
                if isabsolute { "abs(" } else { "" },
                fp64_src,
                swizzle,
                if isabsolute { ")" } else { "" }
            );
        }
    }
    true
}

/// `rewrite_1d_image_coordinate`, GLES leg: a 1D image is a 2D one with a zero row.
fn rewrite_1d_image_coordinate(ctx: &mut Ctx<'_>, inst: &Instruction) {
    let texture = inst.memory.map_or(Texture::Buffer, |m| m.texture);
    if inst.src[0].file == File::Image && (texture == Texture::D1 || texture == Texture::Array1d) {
        let buf = ctx.src_bufs[1].clone();
        ctx.src_bufs[1] = if texture == Texture::D1 {
            format!("vec2(vec4({buf}).x, 0)")
        } else {
            format!("vec3({buf}.xy, 0).xzy")
        };
    }
}

/// `make_array_from_semantic`: fold the entries of one semantic with consecutive indices,
/// starting at `start`, into the array at `start`.
fn make_array_from_semantic(io: &mut [Io], start: usize, semantic: Semantic) -> usize {
    let mut last_sid = io[start].sid;
    for i in start + 1..io.len() {
        if io[i].name == semantic && io[i].sid.wrapping_sub(last_sid) == 1 {
            io[i].glsl_predefined_no_emit = true;
            last_sid = io[i].sid;
            io[i].array_offset = io[i].sid - io[start].sid;
            io[start].last = io[start].first + io[i].array_offset;
            io[i].overlapping_array = Some(start);
        } else {
            break;
        }
    }
    (io[start].last + 1) as usize
}

/// `collapse_vars_to_arrays`.
fn collapse_vars_to_arrays(io: &mut [Io], semantic: Semantic) -> bool {
    let mut retval = false;
    let mut start = 0;
    while start < io.len() {
        if io[start].name == semantic && !io[start].glsl_predefined_no_emit {
            let next = make_array_from_semantic(io, start, semantic);
            retval |= io[start].first != io[start].last;
            start = next;
        } else {
            start += 1;
        }
    }
    if let Some(first) = io.first_mut() {
        first.num_components = 4;
        first.usage_mask = 0xf;
    }
    retval
}

/// `rewrite_io_ranged`: with indirect IO access but separately sent values, arrays are
/// emulated by putting values into arrays by semantic.
fn rewrite_io_ranged(ctx: &mut Ctx<'_>) {
    if ctx.info.is_indirect(File::Input) || ctx.key.require_input_arrays {
        let generic_array = collapse_vars_to_arrays(&mut ctx.inputs, Semantic::Generic);
        let patch_array = collapse_vars_to_arrays(&mut ctx.inputs, Semantic::Patch);
        ctx.has_input_arrays = generic_array || patch_array;
        if ctx.prefer_generic_io_block(IoDir::In) {
            ctx.glsl_ver_required = ctx.require_glsl_ver(150);
        }
    }
    if ctx.info.is_indirect(File::Output) || ctx.key.require_output_arrays {
        let generic_array = collapse_vars_to_arrays(&mut ctx.outputs, Semantic::Generic);
        let patch_array = collapse_vars_to_arrays(&mut ctx.outputs, Semantic::Patch);
        ctx.has_output_arrays = generic_array || patch_array;
        if ctx.prefer_generic_io_block(IoDir::Out) {
            ctx.glsl_ver_required = ctx.require_glsl_ver(150);
        }
    }
}

/// `rewrite_vs_pos_array`.
fn rewrite_vs_pos_array(ctx: &mut Ctx<'_>) {
    let mut range_start = 0xffff;
    let mut range_end = 0;
    let mut io_idx = 0;
    for i in 0..ctx.inputs.len() {
        if ctx.inputs[i].name == Semantic::Position {
            ctx.inputs[i].glsl_predefined_no_emit = true;
            if ctx.inputs[i].first < range_start {
                io_idx = i;
                range_start = ctx.inputs[i].first;
            }
            if ctx.inputs[i].last > range_end {
                range_end = ctx.inputs[i].last;
            }
        }
    }
    if range_start != range_end {
        ctx.inputs[io_idx].first = range_start;
        ctx.inputs[io_idx].last = range_end;
        ctx.inputs[io_idx].glsl_predefined_no_emit = false;
        ctx.glsl_ver_required = ctx.require_glsl_ver(150);
    }
}

/// `renumber_io_arrays`: array ids are not ordered across shaders, so generics and patches
/// are renumbered.
fn renumber_io_arrays(io: &mut [Io]) {
    let mut next_array_id = 1;
    for e in io {
        if e.name != Semantic::Generic && e.name != Semantic::Patch {
            continue;
        }
        if e.array_id > 0 {
            e.array_id = next_array_id;
            next_array_id += 1;
        }
    }
}

/// `handle_io_arrays`.
pub(super) fn handle_io_arrays(ctx: &mut Ctx<'_>) {
    if ctx.guest_sent_io_arrays {
        renumber_io_arrays(&mut ctx.inputs);
        renumber_io_arrays(&mut ctx.outputs);
    } else {
        rewrite_io_ranged(ctx);
    }
}

/// `add_missing_semantic_inputs`.
fn add_missing_semantic_inputs(
    inputs: &mut Vec<Io>,
    next_location: &mut u32,
    mut sids_missing: u64,
    prefix: &str,
    type_prefix: &str,
    name: Semantic,
    key: &Key,
) {
    while sids_missing != 0 {
        let sid = sids_missing.trailing_zeros();
        sids_missing &= sids_missing - 1;
        let mut io = Io {
            sid,
            first: *next_location,
            last: *next_location,
            name,
            ty: VecType::Float,
            ..Io::default()
        };
        let mut sids_added = u64::from(bit32(sid));
        for array in &key.in_arrays.layout {
            if array.name == name && array.sid <= sid && array.sid + array.size >= sid {
                io.last = io.first + array.size;
                io.sid = array.sid;
                sids_added = u64::from((bit32(array.size).wrapping_sub(1)).wrapping_shl(sid));
                break;
            }
        }
        *next_location += io.last - io.first + 1;
        sids_missing &= !sids_added;
        io.glsl_name = format!("{prefix}{type_prefix}{sid}");
        if inputs.len() < MAX_IO {
            inputs.push(io);
        }
    }
}

/// `add_missing_inputs`: inputs the stage before emits but this one did not declare.
fn add_missing_inputs(ctx: &mut Ctx<'_>) {
    let mut generics_declared = 0u64;
    let mut patches_declared = 0u64;
    let mut texcoord_declared = 0u64;
    let mut next_location = 0;
    for input in &ctx.inputs {
        for offset in 0..=(input.last.wrapping_sub(input.first)) {
            let sid = input.sid + offset;
            match input.name {
                Semantic::Generic => generics_declared |= bit64(sid),
                Semantic::Patch => patches_declared |= bit64(sid),
                Semantic::TexCoord => texcoord_declared |= bit64(sid),
                _ => {}
            }
            if input.last < input.first {
                break;
            }
        }
        if next_location < input.last {
            next_location = input.last;
        }
    }
    next_location += 1;

    let generics_missing = ctx.key.in_generic_expected_mask & !generics_declared;
    let patches_missing = ctx.key.in_patch_expected_mask & !patches_declared;
    let texcoord_missing = ctx.key.in_texcoord_expected_mask & !texcoord_declared;

    let prefix = ctx.stage_input_name_prefix(ctx.prog_type);
    let key = ctx.key;
    let mut inputs = std::mem::take(&mut ctx.inputs);
    add_missing_semantic_inputs(
        &mut inputs,
        &mut next_location,
        generics_missing,
        prefix,
        "_g",
        Semantic::Generic,
        key,
    );
    add_missing_semantic_inputs(
        &mut inputs,
        &mut next_location,
        texcoord_missing,
        prefix,
        "_t",
        Semantic::TexCoord,
        key,
    );
    add_missing_semantic_inputs(
        &mut inputs,
        &mut next_location,
        patches_missing,
        "patch",
        "",
        Semantic::Patch,
        key,
    );
    // The C's qsort is not stable; the entries it would leave in either order share a name
    // and an index, which the declaration walk does not produce.
    inputs.sort_by_key(|l| (l.name as u8, l.sid));
    ctx.inputs = inputs;
}

/// `iter_instruction`.
pub(super) fn iter_instruction(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), Failure> {
    let mut dinfo = DestInfo::default();
    let mut sinfo = SourceInfo { svec4: Qual::Vec4, ..SourceInfo::default() };
    let mut fp64_dsts: [String; 2] = [String::new(), String::new()];
    let mut writemask = String::new();
    let mut src_swizzle0 = String::new();
    let instno = ctx.instno;
    ctx.instno += 1;
    let processor = ctx.prog_type;

    if instno == 0 {
        if ctx.prog_type != Processor::Vertex {
            add_missing_inputs(ctx);
        }
        handle_io_arrays(ctx);

        // Vertex shader inputs are not sent as arrays, but the access may still be indirect.
        if ctx.prog_type == Processor::Vertex && ctx.info.is_indirect(File::Input) {
            rewrite_vs_pos_array(ctx);
        }

        ctx.bufs.emit("void main(void)\n{\n");
        if processor == Processor::Fragment {
            emit_color_select(ctx);
            if ctx.fs_uses_clipdist_input {
                emit_fs_clipdistance_load(ctx);
            }
        }
        if ctx.so.is_some() {
            prepare_so_movs(ctx);
        }
        // GLES allows no invariant specifiers on inputs, so the key's forced invariants are
        // desktop only.
    }

    if !get_destination_info(ctx, inst, &mut dinfo, &mut fp64_dsts, &mut writemask) {
        return fail("illegal destination".to_string());
    }
    if !get_source_info(ctx, inst, &mut sinfo, &mut src_swizzle0) {
        return fail("illegal source".to_string());
    }

    let srcs: Vec<String> = ctx.src_bufs[..4].to_vec();
    let dsts: Vec<String> = ctx.dst_bufs.to_vec();
    let dst0 = dsts[0].as_str();
    let dstconv = dinfo.dstconv.s();
    let dtp = dinfo.dtypeprefix.s();
    let wm = writemask.as_str();

    macro_rules! arit_op2 {
        ($op:literal) => {
            emit!(
                ctx.bufs,
                "{} = {}({}(({} {} {})){});\n",
                dst0,
                dstconv,
                dtp,
                srcs[0],
                $op,
                srcs[1],
                wm
            )
        };
    }
    macro_rules! op1 {
        ($op:literal) => {
            emit!(ctx.bufs, "{} = {}({}({}({})){});\n", dst0, dstconv, dtp, $op, srcs[0], wm)
        };
    }
    macro_rules! compare {
        ($op:literal) => {
            emit!(
                ctx.bufs,
                "{} = {}({}(({}({}({}), {}({})))){});\n",
                dst0,
                dstconv,
                dtp,
                $op,
                sinfo.svec4.s(),
                srcs[0],
                sinfo.svec4.s(),
                srcs[1],
                wm
            )
        };
    }
    macro_rules! ucompare {
        ($op:literal, $wm:expr) => {
            emit!(
                ctx.bufs,
                "{} = {}(uintBitsToFloat({}({}({}({}), {}({})){}) * {}(0xffffffff)));\n",
                dst0,
                dstconv,
                dinfo.udstconv.s(),
                $op,
                sinfo.svec4.s(),
                srcs[0],
                sinfo.svec4.s(),
                srcs[1],
                $wm,
                dinfo.udstconv.s()
            )
        };
    }

    use Opcode::*;
    match inst.opcode {
        Sqrt | Dsqrt => emit!(ctx.bufs, "{} = sqrt(vec4({})){};\n", dst0, srcs[0], wm),
        Lrp => emit!(
            ctx.bufs,
            "{} = mix(vec4({}), vec4({}), vec4({})){};\n",
            dst0,
            srcs[2],
            srcs[1],
            srcs[0],
            wm
        ),
        Dp2 => {
            emit!(ctx.bufs, "{} = {}(dot(vec2({}), vec2({})));\n", dst0, dstconv, srcs[0], srcs[1])
        }
        Dp3 => {
            emit!(ctx.bufs, "{} = {}(dot(vec3({}), vec3({})));\n", dst0, dstconv, srcs[0], srcs[1])
        }
        Dp4 => {
            emit!(ctx.bufs, "{} = {}(dot(vec4({}), vec4({})));\n", dst0, dstconv, srcs[0], srcs[1])
        }
        Dph => emit!(
            ctx.bufs,
            "{} = {}(dot(vec4(vec3({}), 1.0), vec4({})));\n",
            dst0,
            dstconv,
            srcs[0],
            srcs[1]
        ),
        Max | Dmax | Imax | Umax => emit!(
            ctx.bufs,
            "{} = {}({}(max({}, {})){});\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            srcs[1],
            wm
        ),
        Min | Dmin | Imin | Umin => emit!(
            ctx.bufs,
            "{} = {}({}(min({}, {})){});\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            srcs[1],
            wm
        ),
        Abs | Iabs | Dabs => op1!("abs"),
        KillIf => emit!(ctx.bufs, "if (any(lessThan({}, vec4(0.0))))\ndiscard;\n", srcs[0]),
        If | Uif => {
            emit!(ctx.bufs, "if (bool({}.x)) {{\n", srcs[0]);
            ctx.bufs.indent();
        }
        Else => {
            ctx.bufs.outdent();
            ctx.bufs.emit("} else {\n");
            ctx.bufs.indent();
        }
        Endif => {
            ctx.bufs.emit("}\n");
            ctx.bufs.outdent();
        }
        Kill => ctx.bufs.emit("discard;\n"),
        Dst => emit!(
            ctx.bufs,
            "{} = vec4(1.0, {}.y * {}.y, {}.z, {}.w);\n",
            dst0,
            srcs[0],
            srcs[1],
            srcs[0],
            srcs[1]
        ),
        Lit => emit!(
            ctx.bufs,
            "{} = {}(vec4(1.0, max({}.x, 0.0), step(0.0, {}.x) * pow(max(0.0, {}.y), clamp({}.w, -128.0, 128.0)), 1.0){});\n",
            dst0,
            dstconv,
            srcs[0],
            srcs[0],
            srcs[0],
            srcs[0],
            wm
        ),
        Ex2 => op1!("exp2"),
        Lg2 => op1!("log2"),
        Exp => emit!(
            ctx.bufs,
            "{} = {}(vec4(pow(2.0, floor({}.x)), {}.x - floor({}.x), exp2({}.x), 1.0){});\n",
            dst0,
            dstconv,
            srcs[0],
            srcs[0],
            srcs[0],
            srcs[0],
            wm
        ),
        Log => emit!(
            ctx.bufs,
            "{} = {}(vec4(floor(log2({}.x)), {}.x / pow(2.0, floor(log2({}.x))), log2({}.x), 1.0){});\n",
            dst0,
            dstconv,
            srcs[0],
            srcs[0],
            srcs[0],
            srcs[0],
            wm
        ),
        Cos => op1!("cos"),
        Sin => op1!("sin"),
        Scs => emit!(
            ctx.bufs,
            "{} = {}(vec4(cos({}.x), sin({}.x), 0, 1){});\n",
            dst0,
            dstconv,
            srcs[0],
            srcs[0],
            wm
        ),
        Ddx => op1!("dFdx"),
        Ddy => op1!("dFdy"),
        DdxFine => {
            ctx.shader_req_bits |= req::DERIVATIVE_CONTROL;
            op1!("dFdxFine");
        }
        DdyFine => {
            ctx.shader_req_bits |= req::DERIVATIVE_CONTROL;
            op1!("dFdyFine");
        }
        Rcp => emit!(ctx.bufs, "{} = {}(1.0/({}));\n", dst0, dstconv, srcs[0]),
        Drcp => emit!(ctx.bufs, "{} = {}(1.0LF/({}));\n", dst0, dstconv, srcs[0]),
        Flr | Dflr => op1!("floor"),
        Round | Dround => {
            // There is no TGSI opcode for roundEven; a guest's roundEven arrives as ROUND and
            // goes back out as roundEven.
            if ctx.cfg.glsl_version >= 300 {
                op1!("roundEven");
            } else {
                op1!("round");
            }
        }
        Issg => op1!("sign"),
        Ceil | Dceil => op1!("ceil"),
        Frc | Dfrac => op1!("fract"),
        Trunc | Dtrunc => op1!("trunc"),
        Ssg | Dssg => op1!("sign"),
        VoteAll => {
            emit!(ctx.bufs, "{} = {}(allInvocationsARB(bool({}.x)));\n", dst0, dstconv, srcs[0]);
            ctx.shader_req_bits |= req::SHADER_GROUP_VOTE;
            ctx.glsl_ver_required = ctx.require_glsl_ver(430);
        }
        VoteAny => {
            emit!(ctx.bufs, "{} = {}(anyInvocationARB(bool({}.x)));\n", dst0, dstconv, srcs[0]);
            ctx.shader_req_bits |= req::SHADER_GROUP_VOTE;
            ctx.glsl_ver_required = ctx.require_glsl_ver(430);
        }
        VoteEq => {
            emit!(
                ctx.bufs,
                "{} = {}(allInvocationsEqualARB(bool({}.x)));\n",
                dst0,
                dstconv,
                srcs[0]
            );
            ctx.shader_req_bits |= req::SHADER_GROUP_VOTE;
            ctx.glsl_ver_required = ctx.require_glsl_ver(430);
        }
        Rsq | Drsq => emit!(ctx.bufs, "{} = {}(inversesqrt({}.x));\n", dst0, dstconv, srcs[0]),
        Fbfetch | Mov => emit!(
            ctx.bufs,
            "{} = {}({}({}{}));\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            if sinfo.override_no_wm[0] { "" } else { wm }
        ),
        Add | Dadd => arit_op2!("+"),
        Uadd => emit!(
            ctx.bufs,
            "{} = {}({}(uvec4({}) + uvec4({})){});\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            srcs[1],
            wm
        ),
        Sub => arit_op2!("-"),
        Mul | Dmul => arit_op2!("*"),
        Div | Ddiv => arit_op2!("/"),
        Umul => emit!(
            ctx.bufs,
            "{} = {}({}((uvec4({}) * uvec4({}))){});\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            srcs[1],
            wm
        ),
        Umod => emit!(
            ctx.bufs,
            "{} = {}({}((uvec4({}) % uvec4({}))){});\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            srcs[1],
            wm
        ),
        Idiv => emit!(
            ctx.bufs,
            "{} = {}({}((ivec4({}) / ivec4({}))){});\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            srcs[1],
            wm
        ),
        Udiv => emit!(
            ctx.bufs,
            "{} = {}({}((uvec4({}) / uvec4({}))){});\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            srcs[1],
            wm
        ),
        Ishr | Ushr => arit_op2!(">>"),
        Shl => arit_op2!("<<"),
        Mad => emit!(
            ctx.bufs,
            "{} = {}(({} * {} + {}){});\n",
            dst0,
            dstconv,
            srcs[0],
            srcs[1],
            srcs[2],
            wm
        ),
        Umad | Dmad => emit!(
            ctx.bufs,
            "{} = {}({}(({} * {} + {}){}));\n",
            dst0,
            dstconv,
            dtp,
            srcs[0],
            srcs[1],
            srcs[2],
            wm
        ),
        Or => arit_op2!("|"),
        And => arit_op2!("&"),
        Xor => arit_op2!("^"),
        Mod => arit_op2!("%"),
        Tex | Tex2 | Txb | Txl | Txb2 | Txl2 | Txd | Txf | Tg4 | Txp => {
            // Lower a 2D_ARRAY fetch to 2D when a 2D texture is what is actually bound.
            // Rewriting the instruction rather than each use point also fixes the
            // declaration: set_texture_reqs records the declared sampler type straight off
            // this instruction.
            let mut linst = *inst;
            if inst.tex().texture == Texture::Array2d
                && sinfo.sreg_index >= 0
                && Key::view_mask_get(
                    &ctx.key.sampler_views_lower_array_mask,
                    sinfo.sreg_index as usize,
                )
            {
                let mut t = linst.tex();
                t.texture = Texture::D2;
                linst.texture = Some(t);
            }
            translate_tex(ctx, &linst, &sinfo, &dinfo, &srcs, dst0, wm);
        }
        Lodq => emit_lodq(ctx, inst, &sinfo, &dinfo, &srcs, dst0, wm),
        Txq => emit_txq(ctx, inst, sinfo.sreg_index, &srcs, dst0, wm),
        Txqs => emit_txqs(ctx, inst, sinfo.sreg_index, &srcs, dst0),
        I2f => emit!(ctx.bufs, "{} = {}(ivec4({}){});\n", dst0, dstconv, srcs[0], wm),
        I2d => emit!(ctx.bufs, "{} = {}(ivec4({}));\n", dst0, dstconv, srcs[0]),
        D2f => emit!(ctx.bufs, "{} = {}({});\n", dst0, dstconv, srcs[0]),
        U2f => emit!(ctx.bufs, "{} = {}(uvec4({}){});\n", dst0, dstconv, srcs[0], wm),
        U2d => emit!(ctx.bufs, "{} = {}(uvec4({}));\n", dst0, dstconv, srcs[0]),
        F2i => emit!(ctx.bufs, "{} = {}({}(ivec4({})){});\n", dst0, dstconv, dtp, srcs[0], wm),
        D2i => emit!(
            ctx.bufs,
            "{} = {}({}({}({})));\n",
            dst0,
            dstconv,
            dtp,
            dinfo.idstconv.s(),
            srcs[0]
        ),
        F2u => emit!(ctx.bufs, "{} = {}({}(uvec4({})){});\n", dst0, dstconv, dtp, srcs[0], wm),
        D2u => emit!(
            ctx.bufs,
            "{} = {}({}({}({})));\n",
            dst0,
            dstconv,
            dtp,
            dinfo.udstconv.s(),
            srcs[0]
        ),
        F2d => emit!(ctx.bufs, "{} = {}({}({}));\n", dst0, dstconv, dtp, srcs[0]),
        Not => emit!(ctx.bufs, "{} = {}(uintBitsToFloat(~(uvec4({}))));\n", dst0, dstconv, srcs[0]),
        Ineg => emit!(ctx.bufs, "{} = {}(intBitsToFloat(-(ivec4({}))));\n", dst0, dstconv, srcs[0]),
        Dneg => emit!(ctx.bufs, "{} = {}(-{});\n", dst0, dstconv, srcs[0]),
        Seq => compare!("equal"),
        Useq | Fseq | Dseq => ucompare!("equal", if inst.opcode == Dseq { ".x" } else { wm }),
        Slt => compare!("lessThan"),
        Sle => compare!("lessThanEqual"),
        Sgt => compare!("greaterThan"),
        Islt | Uslt | Fslt | Dslt => {
            ucompare!("lessThan", if inst.opcode == Dslt { ".x" } else { wm })
        }
        Sne => compare!("notEqual"),
        Usne | Fsne | Dsne => ucompare!("notEqual", if inst.opcode == Dsne { ".x" } else { wm }),
        Sge => compare!("greaterThanEqual"),
        Isge | Usge | Fsge | Dsge => {
            ucompare!("greaterThanEqual", if inst.opcode == Dsge { ".x" } else { wm })
        }
        Pow => emit!(ctx.bufs, "{} = {}(pow({}, {}));\n", dst0, dstconv, srcs[0], srcs[1]),
        Cmp => emit!(
            ctx.bufs,
            "{} = mix({}, {}, greaterThanEqual({}, vec4(0.0))){};\n",
            dst0,
            srcs[1],
            srcs[2],
            srcs[0],
            wm
        ),
        Ucmp => emit!(
            ctx.bufs,
            "{} = mix({}, {}, notEqual(floatBitsToUint({}), uvec4(0.0))){};\n",
            dst0,
            srcs[2],
            srcs[1],
            srcs[0],
            wm
        ),
        End => {
            match processor {
                Processor::Vertex => handle_vertex_proc_exit(ctx),
                Processor::TessCtrl if ctx.cfg.has_cull_distance => emit_clip_dist_movs(ctx),
                Processor::TessEval if ctx.cfg.has_cull_distance => {
                    if ctx.so.is_some() && !ctx.key.gs_present {
                        emit_so_movs(ctx);
                    }
                    emit_clip_dist_movs(ctx);
                    if !ctx.key.gs_present {
                        emit_prescale(ctx);
                    }
                }
                Processor::Fragment => handle_fragment_proc_exit(ctx),
                _ => {}
            }
            ctx.bufs.emit("}\n");
        }
        Ret => {
            match processor {
                Processor::Vertex => handle_vertex_proc_exit(ctx),
                Processor::Fragment => handle_fragment_proc_exit(ctx),
                _ => {}
            }
            ctx.bufs.emit("return;\n");
        }
        Arl => emit!(ctx.bufs, "{} = int(floor({}){});\n", dst0, srcs[0], wm),
        Uarl => emit!(ctx.bufs, "{} = int({});\n", dst0, srcs[0]),
        Xpd => emit!(
            ctx.bufs,
            "{} = {}(cross(vec3({}), vec3({})));\n",
            dst0,
            dstconv,
            srcs[0],
            srcs[1]
        ),
        Bgnloop => {
            ctx.bufs.emit("do {\n");
            ctx.bufs.indent();
        }
        Endloop => {
            ctx.bufs.outdent();
            ctx.bufs.emit("} while(true);\n");
        }
        Brk => ctx.bufs.emit("break;\n"),
        Emit => {
            let val = immediate_word(ctx, inst)?;
            if ctx.so.is_some() && ctx.key.gs_present {
                emit_so_movs(ctx);
            }
            if ctx.cfg.has_cull_distance && ctx.key.gs.emit_clip_distance {
                emit_clip_dist_movs(ctx);
            }
            emit_prescale(ctx);
            if val > 0 {
                ctx.shader_req_bits |= req::GPU_SHADER5;
                emit!(ctx.bufs, "EmitStreamVertex({});\n", val);
            } else {
                ctx.bufs.emit("EmitVertex();\n");
            }
        }
        Endprim => {
            let val = immediate_word(ctx, inst)?;
            if val > 0 {
                ctx.shader_req_bits |= req::GPU_SHADER5;
                emit!(ctx.bufs, "EndStreamPrimitive({});\n", val);
            } else {
                ctx.bufs.emit("EndPrimitive();\n");
            }
        }
        InterpCentroid => {
            emit!(
                ctx.bufs,
                "{} = {}({}(vec4(interpolateAtCentroid({}){})));\n",
                dst0,
                dstconv,
                dtp,
                srcs[0],
                src_swizzle0
            );
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        InterpSample => {
            emit!(
                ctx.bufs,
                "{} = {}({}(vec4(interpolateAtSample({}, {}.x){})));\n",
                dst0,
                dstconv,
                dtp,
                srcs[0],
                srcs[1],
                src_swizzle0
            );
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        InterpOffset => {
            emit!(
                ctx.bufs,
                "{} = {}({}(vec4(interpolateAtOffset({}, {}.xy){})));\n",
                dst0,
                dstconv,
                dtp,
                srcs[0],
                srcs[1],
                src_swizzle0
            );
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        UmulHi => {
            emit!(ctx.bufs, "umulExtended({}, {}, umul_temp, mul_utemp);\n", srcs[0], srcs[1]);
            emit!(ctx.bufs, "{} = {}({}(umul_temp{}));\n", dst0, dstconv, dtp, wm);
            ctx.write_mul_utemp = true;
        }
        ImulHi => {
            emit!(ctx.bufs, "imulExtended({}, {}, imul_temp, mul_itemp);\n", srcs[0], srcs[1]);
            emit!(ctx.bufs, "{} = {}({}(imul_temp{}));\n", dst0, dstconv, dtp, wm);
            ctx.write_mul_itemp = true;
        }
        Ibfe | Ubfe => {
            emit!(
                ctx.bufs,
                "{} = {}({}(bitfieldExtract({}, int({}.x), int({}.x))));\n",
                dst0,
                dstconv,
                dtp,
                srcs[0],
                srcs[1],
                srcs[2]
            );
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        Bfi => {
            emit!(
                ctx.bufs,
                "{} = {}(uintBitsToFloat(bitfieldInsert({}, {}, int({}), int({}))));\n",
                dst0,
                dstconv,
                srcs[0],
                srcs[1],
                srcs[2],
                srcs[3]
            );
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        Brev => {
            emit!(ctx.bufs, "{} = {}({}(bitfieldReverse({})));\n", dst0, dstconv, dtp, srcs[0]);
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        Popc => {
            emit!(ctx.bufs, "{} = {}({}(bitCount({})));\n", dst0, dstconv, dtp, srcs[0]);
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        Lsb => {
            emit!(ctx.bufs, "{} = {}({}(findLSB({})));\n", dst0, dstconv, dtp, srcs[0]);
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        Imsb | Umsb => {
            emit!(ctx.bufs, "{} = {}({}(findMSB({})));\n", dst0, dstconv, dtp, srcs[0]);
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        Barrier => ctx.bufs.emit("barrier();\n"),
        Membar => {
            let val = immediate_word(ctx, inst)?;
            const SHADER_BUFFER: u32 = 1 << 0;
            const ATOMIC_BUFFER: u32 = 1 << 1;
            const SHADER_IMAGE: u32 = 1 << 2;
            const SHARED: u32 = 1 << 3;
            const THREAD_GROUP: u32 = 1 << 4;
            let all_val = SHADER_BUFFER | ATOMIC_BUFFER | SHADER_IMAGE | SHARED;
            if val & THREAD_GROUP != 0 {
                ctx.bufs.emit("groupMemoryBarrier();\n");
            } else if val & all_val == all_val {
                ctx.bufs.emit("memoryBarrier();\n");
                ctx.shader_req_bits |= req::IMAGE_LOAD_STORE;
            } else {
                if val & SHADER_BUFFER != 0 {
                    ctx.bufs.emit("memoryBarrierBuffer();\n");
                }
                if val & ATOMIC_BUFFER != 0 {
                    ctx.bufs.emit("memoryBarrierAtomicCounter();\n");
                }
                if val & SHADER_IMAGE != 0 {
                    ctx.bufs.emit("memoryBarrierImage();\n");
                }
                if val & SHARED != 0 {
                    ctx.bufs.emit("memoryBarrierShared();\n");
                }
            }
        }
        Store => {
            rewrite_1d_image_coordinate(ctx, inst);
            let srcs: Vec<String> = ctx.src_bufs[..4].to_vec();
            // A destination with a negative index is not written.
            if dinfo.dest_index >= 0 {
                translate_store(ctx, inst, &sinfo, &srcs, &dinfo, dst0);
            }
        }
        Load => {
            rewrite_1d_image_coordinate(ctx, inst);
            let srcs: Vec<String> = ctx.src_bufs[..4].to_vec();
            // An obvious out-of-bounds load loads zero.
            if sinfo.sreg_index < 0 || !translate_load(ctx, inst, &sinfo, &dinfo, &srcs, dst0, wm) {
                emit!(ctx.bufs, "{} = vec4(0.0, 0.0, 0.0, 0.0){};\n", dst0, wm);
            }
        }
        Atomuadd | Atomxchg | Atomcas | Atomand | Atomor | Atomxor | Atomumin | Atomumax
        | Atomimin | Atomimax => {
            rewrite_1d_image_coordinate(ctx, inst);
            let srcs: Vec<String> = ctx.src_bufs[..4].to_vec();
            translate_atomic(ctx, inst, &sinfo, &srcs, dst0);
        }
        Resq => translate_resq(ctx, inst, &srcs, dst0, wm),
        Clock => {
            ctx.shader_req_bits |= req::SHADER_CLOCK;
            emit!(ctx.bufs, "{} = uintBitsToFloat(clock2x32ARB());\n", dst0);
        }
        other => {
            eprintln!("[virglrs] Failed to convert opcode {}", other as u8);
        }
    }

    if inst.opcode.dst_type() == OpType::Double {
        emit!(ctx.bufs, "{} = uintBitsToFloat(unpackDouble2x32({}));\n", fp64_dsts[0], dst0);
    }
    if inst.saturate {
        emit!(ctx.bufs, "{} = clamp({}, 0.0, 1.0);\n", dst0, dst0);
    }

    if ctx.bufs.main_error {
        return fail("the body could not be emitted".to_string());
    }
    Ok(())
}

/// The first source's immediate, at its X swizzle: what `EMIT`, `ENDPRIM` and `MEMBAR` carry.
fn immediate_word(ctx: &Ctx<'_>, inst: &Instruction) -> Result<u32, Failure> {
    let src = &inst.src[0];
    if src.index < 0 || src.index as usize >= MAX_IMMEDIATE {
        return fail(format!("Immediate range exceeded, max is {MAX_IMMEDIATE}"));
    }
    Ok(ctx.imm.get(src.index as usize).map_or(0, |imd| imd.val[(src.swizzle[0] & 3) as usize]))
}
