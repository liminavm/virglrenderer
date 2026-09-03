// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Texture sampling and queries, and image, buffer and shared-memory access: the instructions
//! whose translation depends on the resource they touch.

use super::inst::{DestInfo, SourceInfo};
use super::{
    Ctx, MAX_IMMEDIATE, MAX_SAMPLERS, Qual, bit32, emit, proc_prefix, samplertype_is_shadow,
    samplertype_to_req_bits, swiz_char, swizzle_string, wm_string,
};
use crate::vrend::proto::Format;
use crate::vrend::shader::{Key, MAX_SHADER_BUFFERS, MAX_SHADER_IMAGES};
use crate::vrend::tgsi::{
    File, Instruction, MemoryQualifier, Opcode, ReturnType, Texture, WRITEMASK_W, WRITEMASK_X,
    WRITEMASK_XY, WRITEMASK_XYZ,
};

/// `get_internalformat_string`: the GLSL image format layout for a wire format, and the
/// type it returns. An unknown format is warned about and reads as `UNORM` with no layout.
pub(super) fn internalformat_string(virgl_format: u16) -> (&'static str, ReturnType) {
    use ReturnType::*;
    let format = Format::from_wire(u32::from(virgl_format));
    if format == Some(Format::NONE) {
        return ("", Unorm);
    }
    let name = format.and_then(Format::describe).map(|d| d.name);

    match name {
        Some("R11G11B10_FLOAT") => ("r11f_g11f_b10f", Float),
        Some("R10G10B10A2_UNORM") => ("rgb10_a2", Unorm),
        Some("R10G10B10A2_UINT") => ("rgb10_a2ui", Uint),
        Some("R8_UNORM") => ("r8", Unorm),
        Some("R8_SNORM") => ("r8_snorm", Snorm),
        Some("R8_UINT") => ("r8ui", Uint),
        Some("R8_SINT") => ("r8i", Sint),
        Some("R8G8_UNORM") => ("rg8", Unorm),
        Some("R8G8_SNORM") => ("rg8_snorm", Snorm),
        Some("R8G8_UINT") => ("rg8ui", Uint),
        Some("R8G8_SINT") => ("rg8i", Sint),
        Some("R8G8B8A8_UNORM") => ("rgba8", Unorm),
        Some("R8G8B8A8_SNORM") => ("rgba8_snorm", Snorm),
        Some("R8G8B8A8_UINT") => ("rgba8ui", Uint),
        Some("R8G8B8A8_SINT") => ("rgba8i", Sint),
        Some("R16_UNORM") => ("r16", Unorm),
        Some("R16_SNORM") => ("r16_snorm", Snorm),
        Some("R16_UINT") => ("r16ui", Uint),
        Some("R16_SINT") => ("r16i", Sint),
        Some("R16_FLOAT") => ("r16f", Float),
        Some("R16G16_UNORM") => ("rg16", Unorm),
        Some("R16G16_SNORM") => ("rg16_snorm", Snorm),
        Some("R16G16_UINT") => ("rg16ui", Uint),
        Some("R16G16_SINT") => ("rg16i", Sint),
        Some("R16G16_FLOAT") => ("rg16f", Float),
        Some("R16G16B16A16_UNORM") => ("rgba16", Unorm),
        Some("R16G16B16A16_SNORM") => ("rgba16_snorm", Snorm),
        Some("R16G16B16A16_FLOAT") => ("rgba16f", Float),
        Some("R32_FLOAT") => ("r32f", Float),
        Some("R32_UINT") => ("r32ui", Uint),
        Some("R32_SINT") => ("r32i", Sint),
        Some("R32G32_FLOAT") => ("rg32f", Float),
        Some("R32G32_UINT") => ("rg32ui", Uint),
        Some("R32G32_SINT") => ("rg32i", Sint),
        Some("R32G32B32A32_FLOAT") => ("rgba32f", Float),
        Some("R32G32B32A32_UINT") => ("rgba32ui", Uint),
        Some("R16G16B16A16_UINT") => ("rgba16ui", Uint),
        Some("R16G16B16A16_SINT") => ("rgba16i", Sint),
        Some("R32G32B32A32_SINT") => ("rgba32i", Sint),
        _ => {
            eprintln!("[virglrs] Illegal format {virgl_format}");
            ("", Unorm)
        }
    }
}

/// Whether a wire format is one of the three whose images GLES lets a shader both read and
/// write (`R32_FLOAT`, `R32_SINT`, `R32_UINT`).
fn is_r32_format(virgl_format: u16) -> bool {
    let name =
        Format::from_wire(u32::from(virgl_format)).and_then(Format::describe).map(|d| d.name);
    matches!(name, Some("R32_FLOAT" | "R32_SINT" | "R32_UINT"))
}

/// `set_texture_reqs`.
pub(super) fn set_texture_reqs(ctx: &mut Ctx<'_>, inst: &Instruction, sreg_index: i32) -> bool {
    if sreg_index < 0 || sreg_index as usize >= MAX_SAMPLERS {
        eprintln!("[virglrs] Sampler view exceeded, max is {MAX_SAMPLERS}");
        return false;
    }
    let texture = inst.tex().texture;
    ctx.samplers[sreg_index as usize].ty = texture;
    ctx.shader_req_bits |= samplertype_to_req_bits(texture);
    if ctx.cfg.glsl_version >= 140
        && ctx.shader_req_bits & (super::req::SAMPLER_RECT | super::req::SAMPLER_BUF) != 0
    {
        ctx.glsl_ver_required = ctx.require_glsl_ver(140);
    }
    true
}

/// `emit_txq`.
pub(super) fn emit_txq(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sreg_index: i32,
    srcs: &[String],
    dst: &str,
    writemask: &str,
) {
    let mut twm: u8 = 0;
    let mut bias = String::new();
    let sampler_index = 1;
    let dtypeprefix = Qual::IntBitsToFloat;
    let texture = inst.tex().texture;

    set_texture_reqs(ctx, inst, sreg_index);

    // No LOD for these texture types; RECT is emulated with a plain 2D texture, which wants
    // LOD 0.
    match texture {
        Texture::Rect | Texture::ShadowRect => bias = ", 0".to_string(),
        Texture::Buffer | Texture::Msaa2d | Texture::Msaa2dArray => {}
        _ => bias = format!(", int({}.x)", srcs[0]),
    }

    let wm = inst.dst[0].writemask;
    if wm & 0x8 != 0 {
        if !matches!(
            texture,
            Texture::Buffer | Texture::Rect | Texture::Msaa2d | Texture::Msaa2dArray
        ) {
            ctx.shader_req_bits |= super::req::TXQ_LEVELS;
            if wm & 0x7 != 0 {
                twm = WRITEMASK_W;
            }
            let src = &inst.src[1];
            let mut gles_sampler_index = 0;
            for i in 0..src.index.max(0) as u32 {
                if ctx.samplers_used & bit32(i) != 0 {
                    gles_sampler_index += 1;
                }
            }
            let sampler_str = if ctx.info.is_indirect(File::Sampler) && src.indirect {
                format!("addr{}+{}", src.ind.index, gles_sampler_index)
            } else {
                format!("{gles_sampler_index}")
            };
            emit!(
                ctx.bufs,
                "{}{} = {}({}_texlod[{}]);\n",
                dst,
                wm_string(twm),
                dtypeprefix.s(),
                proc_prefix(ctx.info.processor),
                sampler_str
            );
            ctx.gles_use_tex_query_level = true;
        }

        if wm & 0x7 != 0 {
            twm = match texture {
                Texture::D1 | Texture::Buffer | Texture::Shadow1d => WRITEMASK_X,
                Texture::Array1d
                | Texture::Shadow1dArray
                | Texture::D2
                | Texture::Shadow2d
                | Texture::Rect
                | Texture::ShadowRect
                | Texture::Cube
                | Texture::ShadowCube
                | Texture::Msaa2d => WRITEMASK_XY,
                Texture::D3
                | Texture::Array2d
                | Texture::Shadow2dArray
                | Texture::ShadowCubeArray
                | Texture::CubeArray
                | Texture::Msaa2dArray => WRITEMASK_XYZ,
                _ => twm,
            };
        }
    }

    if wm & 0x7 != 0 {
        let txq_returns_vec = texture != Texture::Buffer;
        let wm_buffer;
        let writemask = if matches!(texture, Texture::Array1d | Texture::Shadow1dArray) {
            wm_buffer = format!(".xz{writemask}");
            wm_buffer.as_str()
        } else {
            writemask
        };
        emit!(
            ctx.bufs,
            "{}{} = {}(textureSize({}{})){};\n",
            dst,
            wm_string(twm),
            dtypeprefix.s(),
            srcs[sampler_index],
            bias,
            if txq_returns_vec { writemask } else { "" }
        );
    }
}

/// `emit_txqs`: sample queries.
pub(super) fn emit_txqs(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sreg_index: i32,
    srcs: &[String],
    dst: &str,
) {
    let sampler_index = 0;
    let dtypeprefix = Qual::IntBitsToFloat;
    ctx.shader_req_bits |= super::req::TXQS;
    set_texture_reqs(ctx, inst, sreg_index);
    let texture = inst.tex().texture;
    if texture != Texture::Msaa2d && texture != Texture::Msaa2dArray {
        ctx.bufs.set_error();
        return;
    }
    emit!(ctx.bufs, "{} = {}(textureSamples({}));\n", dst, dtypeprefix.s(), srcs[sampler_index]);
}

/// `get_tex_inst_ext`.
fn tex_inst_ext(inst: &Instruction) -> &'static str {
    let tex = inst.tex();
    match inst.opcode {
        Opcode::Txp => {
            if matches!(tex.texture, Texture::Cube | Texture::Array2d | Texture::Array1d) {
                ""
            } else if tex.num_offsets == 1 {
                "ProjOffset"
            } else {
                "Proj"
            }
        }
        Opcode::Txl | Opcode::Txl2 => {
            if tex.num_offsets == 1 {
                "LodOffset"
            } else {
                "Lod"
            }
        }
        Opcode::Txd => {
            if tex.num_offsets == 1 {
                "GradOffset"
            } else {
                "Grad"
            }
        }
        Opcode::Tg4 => {
            if tex.num_offsets == 4 {
                "GatherOffsets"
            } else if tex.num_offsets == 1 {
                "GatherOffset"
            } else {
                "Gather"
            }
        }
        _ => {
            if tex.num_offsets == 1 {
                "Offset"
            } else {
                ""
            }
        }
    }
}

/// `get_temp`: the GLSL for a temporary register.
pub(super) fn get_temp(ctx: &mut Ctx<'_>, indirect_dim: bool, dim: i32, reg: i32) -> String {
    match ctx.find_temp_range(reg) {
        Some(i) => {
            let range = ctx.temp_ranges[i];
            if indirect_dim {
                format!("temp{}[addr{} + {}]", range.first, dim, reg - range.first)
            } else if range.array_id > 0 {
                format!("temp{}[{}]", range.first, reg - range.first)
            } else {
                format!("temp{reg}")
            }
        }
        None => {
            ctx.require_dummy_value = true;
            "dummy_value".to_string()
        }
    }
}

/// `fill_offset_buffer`: the offset argument of a texture instruction.
fn fill_offset_buffer(ctx: &mut Ctx<'_>, inst: &Instruction, offset_buf: &mut String) -> bool {
    let off = inst.tex_offsets[0];
    let texture = inst.tex().texture;
    let sw = |i: usize| swiz_char(off.swizzle[i]);
    match off.file {
        File::Immediate => {
            if off.index < 0 || off.index as usize >= MAX_IMMEDIATE {
                eprintln!("[virglrs] Immediate exceeded, max is {MAX_IMMEDIATE}");
                return false;
            }
            let Some(imd) = ctx.imm.get(off.index as usize).copied() else {
                // The C reads an immediate slot the program never filled: zeros.
                return fill_immediate_offset(texture, [0; 4], off.swizzle, offset_buf);
            };
            fill_immediate_offset(texture, imd.val, off.swizzle, offset_buf)
        }
        File::Temporary => {
            let temp_buf = get_temp(ctx, false, 0, i32::from(off.index));
            match texture {
                Texture::D1 | Texture::Array1d | Texture::Shadow1d | Texture::Shadow1dArray => {
                    offset_buf.push_str(&format!(", int(floatBitsToInt({temp_buf}.{}))", sw(0)));
                }
                Texture::Rect
                | Texture::ShadowRect
                | Texture::D2
                | Texture::Array2d
                | Texture::Shadow2d
                | Texture::Shadow2dArray => {
                    offset_buf.push_str(&format!(
                        ", ivec2(floatBitsToInt({temp_buf}.{}), floatBitsToInt({temp_buf}.{}))",
                        sw(0),
                        sw(1)
                    ));
                }
                Texture::D3 => {
                    offset_buf.push_str(&format!(
                        ", ivec3(floatBitsToInt({temp_buf}.{}), floatBitsToInt({temp_buf}.{}), floatBitsToInt({temp_buf}.{})",
                        sw(0),
                        sw(1),
                        sw(2)
                    ));
                }
                _ => {
                    eprintln!("[virglrs] Unhandled texture: {:x}", texture as u8);
                    return false;
                }
            }
            true
        }
        File::Input => {
            for j in 0..ctx.inputs.len() {
                if ctx.inputs[j].first as i32 != i32::from(off.index) {
                    continue;
                }
                let name = ctx.inputs[j].glsl_name.clone();
                match texture {
                    Texture::D1 | Texture::Array1d | Texture::Shadow1d | Texture::Shadow1dArray => {
                        offset_buf.push_str(&format!(", int(floatBitsToInt({name}.{}))", sw(0)));
                    }
                    Texture::Rect
                    | Texture::ShadowRect
                    | Texture::D2
                    | Texture::Array2d
                    | Texture::Shadow2d
                    | Texture::Shadow2dArray => {
                        offset_buf.push_str(&format!(
                            ", ivec2(floatBitsToInt({name}.{}), floatBitsToInt({name}.{}))",
                            sw(0),
                            sw(1)
                        ));
                    }
                    Texture::D3 => {
                        offset_buf.push_str(&format!(
                            ", ivec3(floatBitsToInt({name}.{}), floatBitsToInt({name}.{}), floatBitsToInt({name}.{})",
                            sw(0),
                            sw(1),
                            sw(2)
                        ));
                    }
                    _ => {
                        eprintln!("[virglrs] Unhandled texture: {:x}", texture as u8);
                        return false;
                    }
                }
            }
            true
        }
        _ => true,
    }
}

fn fill_immediate_offset(
    texture: Texture,
    val: [u32; 4],
    swizzle: [u8; 3],
    offset_buf: &mut String,
) -> bool {
    let v = |i: usize| val[(swizzle[i] & 3) as usize] as i32;
    match texture {
        Texture::D1 | Texture::Array1d | Texture::Shadow1d | Texture::Shadow1dArray => {
            offset_buf.push_str(&format!(", ivec2({}, 0)", v(0)));
        }
        Texture::Rect
        | Texture::ShadowRect
        | Texture::D2
        | Texture::Array2d
        | Texture::Shadow2d
        | Texture::Shadow2dArray => {
            offset_buf.push_str(&format!(", ivec2({}, {})", v(0), v(1)));
        }
        Texture::D3 => {
            offset_buf.push_str(&format!(", ivec3({}, {}, {})", v(0), v(1), v(2)));
        }
        _ => {
            eprintln!("[virglrs] Unhandled texture: {:x}", texture as u8);
            return false;
        }
    }
    true
}

/// `emit_lodq`.
pub(super) fn emit_lodq(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sinfo: &SourceInfo,
    dinfo: &DestInfo,
    srcs: &[String],
    dst: &str,
    writemask: &str,
) {
    ctx.shader_req_bits |= super::req::LODQ;
    set_texture_reqs(ctx, inst, sinfo.sreg_index);

    emit!(ctx.bufs, "{} = {}(textureQueryLOD({}, ", dst, dinfo.dstconv.s(), srcs[1]);
    match inst.tex().texture {
        Texture::D1 | Texture::Array1d | Texture::Shadow1d | Texture::Shadow1dArray => {
            emit!(ctx.bufs, "vec2({}.x, 0)", srcs[0]);
        }
        Texture::D2
        | Texture::Array2d
        | Texture::Msaa2d
        | Texture::Msaa2dArray
        | Texture::Rect
        | Texture::Shadow2d
        | Texture::Shadow2dArray
        | Texture::ShadowRect => {
            emit!(ctx.bufs, "{}.xy", srcs[0]);
        }
        Texture::D3
        | Texture::Cube
        | Texture::ShadowCube
        | Texture::ShadowCubeArray
        | Texture::CubeArray => {
            emit!(ctx.bufs, "{}.xyz", srcs[0]);
        }
        _ => {
            emit!(ctx.bufs, "{}", srcs[0]);
        }
    }
    emit!(ctx.bufs, "){});\n", writemask);
}

/// `translate_tex`.
pub(super) fn translate_tex(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sinfo: &SourceInfo,
    dinfo: &DestInfo,
    srcs: &[String],
    dst: &str,
    writemask: &str,
) {
    let txfi: Qual;
    let src_swizzle: &str;
    let mut dtypeprefix = Qual::None;
    let mut sampler_index = 1;
    let mut bias_buf = String::new();
    let mut offset_buf = String::new();
    let texture = inst.tex().texture;
    let num_offsets = inst.tex().num_offsets;

    if !set_texture_reqs(ctx, inst, sinfo.sreg_index) {
        ctx.bufs.set_error();
        return;
    }

    let is_shad = samplertype_is_shadow(texture);

    match ctx.samplers[sinfo.sreg_index as usize].ret {
        ReturnType::Sint => {
            if dinfo.dstconv != Qual::Int {
                dtypeprefix = Qual::IntBitsToFloat;
            }
        }
        ReturnType::Uint if dinfo.dstconv != Qual::Int => {
            dtypeprefix = Qual::UintBitsToFloat;
        }
        _ => {}
    }

    match texture {
        Texture::D1 | Texture::Buffer => {
            src_swizzle = if inst.opcode == Opcode::Txp { "" } else { ".x" };
            txfi = Qual::Int;
        }
        Texture::Array1d => {
            src_swizzle = ".xy";
            txfi = Qual::IVec2;
        }
        Texture::D2 | Texture::Rect => {
            src_swizzle = if inst.opcode == Opcode::Txp { "" } else { ".xy" };
            txfi = Qual::IVec2;
        }
        Texture::Shadow1d
        | Texture::Shadow2d
        | Texture::Shadow1dArray
        | Texture::ShadowRect
        | Texture::D3 => {
            src_swizzle = if inst.opcode == Opcode::Txp {
                ""
            } else if inst.opcode == Opcode::Tg4 {
                ".xy"
            } else {
                ".xyz"
            };
            txfi = Qual::IVec3;
        }
        Texture::Cube | Texture::Array2d => {
            src_swizzle = ".xyz";
            txfi = Qual::IVec3;
        }
        Texture::Msaa2d => {
            src_swizzle = ".xy";
            txfi = Qual::IVec2;
        }
        Texture::Msaa2dArray => {
            src_swizzle = ".xyz";
            txfi = Qual::IVec3;
        }
        _ => {
            src_swizzle = if inst.opcode == Opcode::Tg4
                && texture != Texture::CubeArray
                && texture != Texture::ShadowCubeArray
            {
                ".xyz"
            } else {
                ""
            };
            txfi = Qual::None;
        }
    }

    match inst.opcode {
        Opcode::Tex2 => {
            sampler_index = 2;
            if texture == Texture::ShadowCubeArray {
                bias_buf.push_str(&format!(", {}.x", srcs[1]));
            }
        }
        Opcode::Txb2 | Opcode::Txl2 => {
            sampler_index = 2;
            bias_buf.push_str(&format!(", {}.x", srcs[1]));
            if texture == Texture::ShadowCubeArray {
                bias_buf.push_str(&format!(", {}.y", srcs[1]));
            }
        }
        Opcode::Txb | Opcode::Txl => {
            // A 1D array is emulated with a 2D array, which has no shadow lookup with bias
            // unless EXT_texture_shadow_lod is there; the bias is dropped rather than
            // compiling a shader that cannot compile.
            if !(!ctx.cfg.has_texture_shadow_lod && texture == Texture::Shadow1dArray) {
                bias_buf.push_str(&format!(", {}.w", srcs[0]));
            }
        }
        Opcode::Txf => {
            if matches!(
                texture,
                Texture::D1
                    | Texture::D2
                    | Texture::Msaa2d
                    | Texture::Msaa2dArray
                    | Texture::D3
                    | Texture::Array1d
                    | Texture::Array2d
            ) {
                bias_buf.push_str(&format!(", int({}.w)", srcs[0]));
            }
        }
        Opcode::Txd => {
            sampler_index = 3;
            match texture {
                Texture::D1 | Texture::Shadow1d | Texture::Array1d | Texture::Shadow1dArray => {
                    bias_buf.push_str(&format!(", vec2({}.x, 0), vec2({}.x, 0)", srcs[1], srcs[2]));
                }
                Texture::D2
                | Texture::Shadow2d
                | Texture::Array2d
                | Texture::Shadow2dArray
                | Texture::Rect
                | Texture::ShadowRect => {
                    bias_buf.push_str(&format!(", {}.xy, {}.xy", srcs[1], srcs[2]));
                }
                Texture::D3 | Texture::Cube | Texture::ShadowCube | Texture::CubeArray => {
                    bias_buf.push_str(&format!(", {}.xyz, {}.xyz", srcs[1], srcs[2]));
                }
                _ => {
                    bias_buf.push_str(&format!(", {}, {}", srcs[1], srcs[2]));
                }
            }
        }
        Opcode::Tg4 => {
            sampler_index = 2;
            ctx.shader_req_bits |= super::req::TG4;
            if num_offsets == 1 && inst.tex_offsets[0].file != File::Immediate {
                ctx.shader_req_bits |= super::req::GPU_SHADER5;
            }
            if is_shad {
                if texture == Texture::ShadowCube || texture == Texture::Shadow2dArray {
                    bias_buf.push_str(&format!(", {}.w", srcs[0]));
                } else if texture == Texture::ShadowCubeArray {
                    bias_buf.push_str(&format!(", {}.x", srcs[1]));
                } else {
                    bias_buf.push_str(&format!(", {}.z", srcs[0]));
                }
            } else if sinfo.tg4_has_component {
                if num_offsets == 0 {
                    if matches!(
                        texture,
                        Texture::D2
                            | Texture::Rect
                            | Texture::Cube
                            | Texture::Array2d
                            | Texture::CubeArray
                    ) {
                        bias_buf.push_str(&format!(", int({})", srcs[1]));
                    }
                } else if matches!(texture, Texture::D2 | Texture::Rect | Texture::Array2d) {
                    bias_buf.push_str(&format!(", int({})", srcs[1]));
                }
            }
        }
        _ => {}
    }

    let mut exchange_bias_offset = false;
    if num_offsets == 1 {
        let index = inst.tex_offsets[0].index;
        if index < 0 || index as usize >= MAX_IMMEDIATE {
            eprintln!("[virglrs] Immediate exceeded, max is {MAX_IMMEDIATE}");
            ctx.bufs.set_error();
            return;
        }
        if !fill_offset_buffer(ctx, inst, &mut offset_buf) {
            ctx.bufs.set_error();
            return;
        }
        exchange_bias_offset = matches!(inst.opcode, Opcode::Txl | Opcode::Txl2 | Opcode::Txd)
            || (inst.opcode == Opcode::Tg4 && is_shad);
    }

    let has_bias = !bias_buf.is_empty();
    let has_offset = !offset_buf.is_empty();
    // EXT_texture_shadow_lod defines a few more functions handling bias.
    if has_bias
        && matches!(
            texture,
            Texture::Shadow2dArray | Texture::ShadowCube | Texture::ShadowCubeArray
        )
    {
        ctx.shader_req_bits |= super::req::TEXTURE_SHADOW_LOD;
    }
    // EXT_texture_shadow_lod also adds the missing textureOffset for 2DArrayShadow in GLES.
    if (has_bias || has_offset)
        && matches!(texture, Texture::Shadow1dArray | Texture::Shadow2dArray)
    {
        ctx.shader_req_bits |= super::req::TEXTURE_SHADOW_LOD;
    }

    let tex_ext = tex_inst_ext(inst);
    let (bias, offset) = if exchange_bias_offset {
        (offset_buf.as_str(), bias_buf.as_str())
    } else {
        (bias_buf.as_str(), offset_buf.as_str())
    };

    // The coordinate is unnormalised for all but the texel fetch.
    let mut coord = srcs[0].clone();
    if inst.opcode != Opcode::Txf
        && Key::view_mask_get(&ctx.key.sampler_views_emulated_rect_mask, sinfo.sreg_index as usize)
    {
        // No LOD for these texture types; RECT is emulated with a plain 2D texture, which
        // wants LOD 0.
        let lod = match texture {
            Texture::Buffer | Texture::Msaa2d | Texture::Msaa2dArray => "",
            _ => ", 0",
        };
        coord = match inst.opcode {
            Opcode::Txp => {
                format!("vec4({})/vec4(textureSize({}{}), 1, 1)", srcs[0], srcs[sampler_index], lod)
            }
            Opcode::Tg4 => {
                format!("{}.xy/vec2(textureSize({}{}))", srcs[0], srcs[sampler_index], lod)
            }
            _ => {
                // Non-TG4 ops have the compare value in the z component.
                if texture == Texture::ShadowRect {
                    format!(
                        "vec3({}.xy/vec2(textureSize({}{})), {}.z)",
                        srcs[0], srcs[sampler_index], lod, srcs[0]
                    )
                } else {
                    format!("{}.xy/vec2(textureSize({}{}))", srcs[0], srcs[sampler_index], lod)
                }
            }
        };
    }
    let src0 = coord.as_str();
    let sampler = srcs[sampler_index].as_str();
    let dstconv = dinfo.dstconv.s();
    let dtp = dtypeprefix.s();
    let wm_or_none = if dinfo.dst_override_no_wm[0] { "" } else { writemask };

    if inst.opcode == Opcode::Txf {
        if matches!(texture, Texture::D1 | Texture::Array1d | Texture::Rect) {
            if texture == Texture::D1 {
                emit!(
                    ctx.bufs,
                    "{} = {}({}(texelFetch{}({}, ivec2({}({}{}), 0){}{}){}));\n",
                    dst,
                    dstconv,
                    dtp,
                    tex_ext,
                    sampler,
                    txfi.s(),
                    src0,
                    src_swizzle,
                    bias,
                    offset,
                    wm_or_none
                );
            } else if texture == Texture::Array1d {
                // The y coordinate goes into the z element, and y is zero.
                emit!(
                    ctx.bufs,
                    "{} = {}({}(texelFetch{}({}, ivec3({}({}{}), 0).xzy{}{}){}));\n",
                    dst,
                    dstconv,
                    dtp,
                    tex_ext,
                    sampler,
                    txfi.s(),
                    src0,
                    src_swizzle,
                    bias,
                    offset,
                    wm_or_none
                );
            } else {
                emit!(
                    ctx.bufs,
                    "{} = {}({}(texelFetch{}({}, {}({}{}), 0{}){}));\n",
                    dst,
                    dstconv,
                    dtp,
                    tex_ext,
                    sampler,
                    txfi.s(),
                    src0,
                    src_swizzle,
                    offset,
                    wm_or_none
                );
            }
        } else {
            // The swizzle for texture buffers with emulated formats is injected as
            //   { vec4 val = texelFetch(); val = vec4(0/1/swizzle_x, ...); dest.wm = val.wm; }
            emit!(
                ctx.bufs,
                "{{\n  vec4 val = {}(texelFetch{}({}, {}({}{}){}{}));\n",
                dtp,
                tex_ext,
                sampler,
                txfi.s(),
                src0,
                src_swizzle,
                bias,
                offset
            );

            if Key::view_mask_get(
                &ctx.key.sampler_views_lower_swizzle_mask,
                sinfo.sreg_index as usize,
            ) {
                let packed_swizzles = ctx.key.tex_swizzle[sinfo.sreg_index as usize];
                ctx.bufs.emit("   val = vec4(");
                for i in 0..4 {
                    if i > 0 {
                        ctx.bufs.emit(", ");
                    }
                    let swz = (packed_swizzles >> (i * 3)) & 7;
                    match swz {
                        4 => ctx.bufs.emit("0.0"),
                        5 => match dtypeprefix {
                            Qual::UintBitsToFloat => ctx.bufs.emit("uintBitsToFloat(1u)"),
                            Qual::IntBitsToFloat => ctx.bufs.emit("intBitsToFloat(1)"),
                            _ => ctx.bufs.emit("1.0"),
                        },
                        _ => emit!(ctx.bufs, "val{}", swizzle_string(swz)),
                    }
                }
                ctx.bufs.emit(");\n");
            }

            emit!(ctx.bufs, "  {}  = val{};\n}}\n", dst, wm_or_none);
        }
    } else if is_shad && inst.opcode != Opcode::Tg4 {
        // TGSI returns 1.0 in alpha.
        let cname = proc_prefix(ctx.prog_type);
        let src_index = inst.src[sampler_index].index;
        if texture == Texture::Shadow1d {
            if inst.opcode == Opcode::Txp {
                emit!(
                    ctx.bufs,
                    "{} = {}({}(vec4(vec4(texture{}({}, vec4({}{}.xzw, 0).xwyz {}{})) * {}shadmask{} + {}shadadd{}){}));\n",
                    dst,
                    dstconv,
                    dtp,
                    tex_ext,
                    sampler,
                    src0,
                    src_swizzle,
                    offset,
                    bias,
                    cname,
                    src_index,
                    cname,
                    src_index,
                    writemask
                );
            } else {
                emit!(
                    ctx.bufs,
                    "{} = {}({}(vec4(vec4(texture{}({}, vec3({}{}.xz, 0).xzy {}{})) * {}shadmask{} + {}shadadd{}){}));\n",
                    dst,
                    dstconv,
                    dtp,
                    tex_ext,
                    sampler,
                    src0,
                    src_swizzle,
                    offset,
                    bias,
                    cname,
                    src_index,
                    cname,
                    src_index,
                    writemask
                );
            }
        } else if texture == Texture::Shadow1dArray {
            emit!(
                ctx.bufs,
                "{} = {}({}(vec4(vec4(texture{}({}, vec4({}{}, 0).xwyz {}{})) * {}shadmask{} + {}shadadd{}){}));\n",
                dst,
                dstconv,
                dtp,
                tex_ext,
                sampler,
                src0,
                src_swizzle,
                offset,
                bias,
                cname,
                src_index,
                cname,
                src_index,
                writemask
            );
        } else {
            emit!(
                ctx.bufs,
                "{} = {}({}(vec4(vec4(texture{}({}, {}{}{}{})) * {}shadmask{} + {}shadadd{}){}));\n",
                dst,
                dstconv,
                dtp,
                tex_ext,
                sampler,
                src0,
                src_swizzle,
                offset,
                bias,
                cname,
                src_index,
                cname,
                src_index,
                writemask
            );
        }
    } else if texture == Texture::D1 {
        // GLES has no 1D texture: a 2D texture is sampled at 0.5.
        if inst.opcode == Opcode::Txp {
            emit!(
                ctx.bufs,
                "{} = {}({}(texture{}({}, vec3({}.xw, 0).xzy {}{}){}));\n",
                dst,
                dstconv,
                dtp,
                tex_ext,
                sampler,
                src0,
                offset,
                bias,
                wm_or_none
            );
        } else {
            emit!(
                ctx.bufs,
                "{} = {}({}(texture{}({}, vec2({}{}, 0.5) {}{}){}));\n",
                dst,
                dstconv,
                dtp,
                tex_ext,
                sampler,
                src0,
                src_swizzle,
                offset,
                bias,
                wm_or_none
            );
        }
    } else if texture == Texture::Array1d {
        if inst.opcode == Opcode::Txp {
            emit!(
                ctx.bufs,
                "{} = {}({}(texture{}({}, vec3({}.x / {}.w, 0, {}.y) {}{}){}));\n",
                dst,
                dstconv,
                dtp,
                tex_ext,
                sampler,
                src0,
                src0,
                src0,
                offset,
                bias,
                wm_or_none
            );
        } else {
            emit!(
                ctx.bufs,
                "{} = {}({}(texture{}({}, vec3({}{}, 0).xzy {}{}){}));\n",
                dst,
                dstconv,
                dtp,
                tex_ext,
                sampler,
                src0,
                src_swizzle,
                offset,
                bias,
                wm_or_none
            );
        }
    } else {
        emit!(
            ctx.bufs,
            "{} = {}({}(texture{}({}, {}{}{}{}){}));\n",
            dst,
            dstconv,
            dtp,
            tex_ext,
            sampler,
            src0,
            src_swizzle,
            offset,
            bias,
            wm_or_none
        );
    }
}

/// `get_coord_prefix`, GLES leg.
fn coord_prefix(resource: Texture) -> (Qual, bool) {
    match resource {
        Texture::D1 => (Qual::IVec2, false),
        Texture::Buffer => (Qual::Int, false),
        Texture::Array1d => (Qual::IVec3, false),
        Texture::D2 | Texture::Rect => (Qual::IVec2, false),
        Texture::D3 | Texture::Cube | Texture::Array2d | Texture::CubeArray => (Qual::IVec3, false),
        Texture::Msaa2d => (Qual::IVec2, true),
        Texture::Msaa2dArray => (Qual::IVec3, true),
        _ => (Qual::None, false),
    }
}

/// `is_integer_memory`.
fn is_integer_memory(ctx: &Ctx<'_>, file: File, index: u32) -> bool {
    match file {
        File::Buffer => ctx.ssbo_integer_mask & bit32(index) != 0,
        File::Memory => ctx.integer_memory,
        other => {
            eprintln!("[virglrs] Invalid file type: {}", other as u8);
            false
        }
    }
}

fn is_coherent(inst: &Instruction) -> bool {
    inst.memory.is_some_and(|m| m.qualifier == 1 << MemoryQualifier::Coherent as u8)
}

/// `set_image_qualifier`.
fn set_image_qualifier(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    reg_index: i32,
    indirect: bool,
) -> bool {
    if is_coherent(inst) {
        if indirect {
            let mut mask = ctx.images_used_mask;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                ctx.images[i].coherent = true;
            }
        } else if reg_index >= 0 && (reg_index as usize) < MAX_SHADER_IMAGES {
            ctx.images[reg_index as usize].coherent = true;
        } else {
            return false;
        }
    }
    true
}

/// `set_memory_qualifier`.
fn set_memory_qualifier(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    reg_index: i32,
    indirect: bool,
) -> bool {
    if is_coherent(inst) {
        if indirect {
            let mut mask = ctx.ssbo_used_mask;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                ctx.ssbo_memory_qualifier[i] = 1 << MemoryQualifier::Coherent as u8;
            }
        } else if reg_index >= 0 && (reg_index as usize) < MAX_SHADER_BUFFERS {
            ctx.ssbo_memory_qualifier[reg_index as usize] = 1 << MemoryQualifier::Coherent as u8;
        } else {
            return false;
        }
    }
    true
}

/// `u_bit_scan_consecutive_range`: the first set bit of `mask` and how many set bits follow
/// it without a gap.
fn bit_scan_consecutive_range(mask: u32) -> (i32, i32) {
    if mask == 0 {
        return (0, 0);
    }
    let start = mask.trailing_zeros();
    let count = (!(mask >> start)).trailing_zeros();
    (start as i32, count as i32)
}

/// `emit_store_mem`.
fn emit_store_mem(ctx: &mut Ctx<'_>, dst: &str, writemask: u8, srcs: &[String], conversion: &str) {
    for (i, swizzle) in ['x', 'y', 'z', 'w'].into_iter().enumerate() {
        if writemask & (1 << i) != 0 {
            emit!(
                ctx.bufs,
                "{}[(uint(floatBitsToUint({})) >> 2) + {}u] = {}({}).{};\n",
                dst,
                srcs[0],
                i,
                conversion,
                srcs[1],
                swizzle
            );
        }
    }
}

/// `make_ssbo_varstring`, GLES leg: an indirect index never reaches the name here, the
/// callers switch over the array instead.
pub(super) fn make_ssbo_varstring(ctx: &Ctx<'_>, register_index: u32) -> String {
    let cname = proc_prefix(ctx.prog_type);
    let atomic_ssbo = ctx.ssbo_atomic_mask & bit32(register_index) != 0;
    let atomic_str = if atomic_ssbo { "atomic" } else { "" };
    let base = if atomic_ssbo { ctx.ssbo_atomic_array_base } else { ctx.ssbo_array_base };
    if ctx.info.is_indirect(File::Buffer) {
        format!(
            "{cname}ssboarr{atomic_str}[{}].{cname}ssbocontents{base}",
            register_index.wrapping_sub(base) as i32
        )
    } else {
        format!("{cname}ssbocontents{register_index}")
    }
}

/// `translate_store`.
pub(super) fn translate_store(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sinfo: &SourceInfo,
    srcs: &[String],
    dinfo: &DestInfo,
    dst: &str,
) {
    let dst_reg = &inst.dst[0];
    if dinfo.dest_index < 0 {
        ctx.bufs.set_error();
        return;
    }
    if dst_reg.file == File::Image {
        if dinfo.dest_index as usize >= MAX_SHADER_IMAGES {
            ctx.bufs.set_error();
            return;
        }
        // A write to an image that does not exist is dropped.
        if bit32(dinfo.dest_index as u32) & ctx.images_used_mask == 0 {
            return;
        }
        if !set_image_qualifier(ctx, inst, i32::from(inst.src[0].index), inst.src[0].indirect) {
            ctx.bufs.set_error();
            return;
        }

        let resource = ctx
            .images
            .get(dst_reg.index.max(0) as usize)
            .map_or(Texture::Buffer, |i| i.decl.resource);
        let (coord_prefix, is_ms) = coord_prefix(resource);
        let conversion = if sinfo.override_no_cast[0] { "" } else { Qual::FloatBitsToInt.s() };
        let (_, itype) = internalformat_string(inst.memory.map_or(0, |m| m.format));
        let ms_str = if is_ms { format!("int({}.w),", srcs[0]) } else { String::new() };
        let stypeprefix = match itype {
            ReturnType::Uint => Qual::FloatBitsToUint,
            ReturnType::Sint => Qual::FloatBitsToInt,
            _ => Qual::None,
        };
        if !dst_reg.indirect {
            emit!(
                ctx.bufs,
                "imageStore({},{}({}({})),{}{}({}));\n",
                dst,
                coord_prefix.s(),
                conversion,
                srcs[0],
                ms_str,
                stypeprefix.s(),
                srcs[1]
            );
        } else if let Some(image) = ctx.lookup_image_array_ptr(i32::from(dst_reg.index)) {
            let image = ctx.image_arrays[image];
            let basearrayidx = image.first;
            let array_size = image.array_size;
            emit!(
                ctx.bufs,
                "switch (addr{} + {}) {{\n",
                dst_reg.ind.index,
                i32::from(dst_reg.index) - basearrayidx
            );
            let cname = proc_prefix(ctx.prog_type);
            for i in 0..array_size {
                emit!(
                    ctx.bufs,
                    "case {}: imageStore({}img{}[{}],{}({}({})),{}{}({})); break;\n",
                    i,
                    cname,
                    basearrayidx,
                    i,
                    coord_prefix.s(),
                    conversion,
                    srcs[0],
                    ms_str,
                    stypeprefix.s(),
                    srcs[1]
                );
            }
            ctx.bufs.emit("}\n");
        }
    } else if dst_reg.file == File::Buffer || dst_reg.file == File::Memory {
        if dinfo.dest_index as usize >= MAX_SHADER_BUFFERS {
            ctx.bufs.set_error();
            return;
        }
        if !set_memory_qualifier(ctx, inst, i32::from(dst_reg.index), dst_reg.indirect) {
            ctx.bufs.set_error();
            return;
        }
        let dtypeprefix = if is_integer_memory(ctx, dst_reg.file, dst_reg.index as u32) {
            Qual::FloatBitsToInt
        } else {
            Qual::FloatBitsToUint
        };
        let conversion = if sinfo.override_no_cast[1] { "" } else { dtypeprefix.s() };

        if !dst_reg.indirect {
            emit_store_mem(ctx, dst, dst_reg.writemask, srcs, conversion);
        } else {
            let atomic_ssbo = ctx.ssbo_atomic_mask & bit32(dst_reg.index as u32) != 0;
            let base = if atomic_ssbo { ctx.ssbo_atomic_array_base } else { ctx.ssbo_array_base };
            let (start, array_count) = bit_scan_consecutive_range(ctx.ssbo_used_mask);
            emit!(
                ctx.bufs,
                "switch (addr{} + {}) {{\n",
                dst_reg.ind.index,
                i32::from(dst_reg.index).wrapping_sub(base as i32)
            );
            for i in 0..array_count {
                emit!(ctx.bufs, "case {}:\n", i);
                let dst_tmp = make_ssbo_varstring(ctx, (i + start) as u32);
                emit_store_mem(ctx, &dst_tmp, dst_reg.writemask, srcs, conversion);
                ctx.bufs.emit("break;\n");
            }
            ctx.bufs.emit("}\n");
        }
    }
}

/// `emit_load_mem`.
fn emit_load_mem(
    ctx: &mut Ctx<'_>,
    dst: &str,
    writemask: u8,
    conversion: &str,
    atomic_op: &str,
    src0: &str,
    atomic_src: &str,
) {
    for (i, swizzle) in ['x', 'y', 'z', 'w'].into_iter().enumerate() {
        if writemask & (1 << i) != 0 {
            emit!(
                ctx.bufs,
                "{}.{} = ({}({}({}[ssbo_addr_temp + {}u]{})));\n",
                dst,
                swizzle,
                conversion,
                atomic_op,
                src0,
                i,
                atomic_src
            );
        }
    }
}

/// `translate_load`.
pub(super) fn translate_load(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sinfo: &SourceInfo,
    dinfo: &DestInfo,
    srcs: &[String],
    dst: &str,
    writemask: &str,
) -> bool {
    let src = &inst.src[0];
    if src.file == File::Image {
        // A load from an image that is not used is dropped.
        if sinfo.sreg_index < 0 || sinfo.sreg_index as usize > MAX_SHADER_IMAGES {
            return false;
        }
        if bit32(sinfo.sreg_index as u32) & ctx.images_used_mask == 0 {
            return false;
        }
        if !set_image_qualifier(ctx, inst, i32::from(src.index), src.indirect) {
            ctx.bufs.set_error();
            return false;
        }
        let sreg = sinfo.sreg_index as usize;
        let (coord_prefix, is_ms) = coord_prefix(ctx.images[sreg].decl.resource);
        let conversion = if sinfo.override_no_cast[1] { "" } else { Qual::FloatBitsToInt.s() };
        let (_, itype) = internalformat_string(ctx.images[sreg].decl.format);
        let ms_str = if is_ms { format!(", int({}.w)", srcs[1]) } else { String::new() };
        let wm = if dinfo.dst_override_no_wm[0] { "" } else { writemask };
        let dtypeprefix = match itype {
            ReturnType::Uint => Qual::UintBitsToFloat,
            ReturnType::Sint => Qual::IntBitsToFloat,
            _ => Qual::None,
        };

        // On GLES `WR` becomes `writeonly`, since most formats have to be one or the other:
        // an image declared `WR` and read from loses its writable flag. Formats that allow
        // both are unaffected; for the others a write fails instead of the read, which is no
        // regression, as both were never possible.
        if ctx.images[sreg].decl.writable && !is_r32_format(ctx.images[sreg].decl.format) {
            ctx.images[sreg].decl.writable = false;
        }

        if !src.indirect {
            emit!(
                ctx.bufs,
                "{} = {}(imageLoad({}, {}({}({})){}){});\n",
                dst,
                dtypeprefix.s(),
                srcs[0],
                coord_prefix.s(),
                conversion,
                srcs[1],
                ms_str,
                wm
            );
        } else if let Some(image) = ctx.lookup_image_array_ptr(i32::from(src.index)) {
            let image = ctx.image_arrays[image];
            let basearrayidx = image.first;
            let array_size = image.array_size;
            emit!(
                ctx.bufs,
                "switch (addr{} + {}) {{\n",
                src.ind.index,
                i32::from(src.index) - basearrayidx
            );
            let cname = proc_prefix(ctx.prog_type);
            for i in 0..array_size {
                let s = format!("{cname}img{basearrayidx}[{i}]");
                emit!(
                    ctx.bufs,
                    "case {}: {} = {}(imageLoad({}, {}({}({})){}){});break;\n",
                    i,
                    dst,
                    dtypeprefix.s(),
                    s,
                    coord_prefix.s(),
                    conversion,
                    srcs[1],
                    ms_str,
                    wm
                );
            }
            ctx.bufs.emit("}\n");
        }
    } else if src.file == File::Buffer || src.file == File::Memory {
        if !set_memory_qualifier(ctx, inst, i32::from(src.index), src.indirect) {
            ctx.bufs.set_error();
            return false;
        }
        // The destination up to its first '.'.
        let mydst: String = dst.chars().take_while(|&c| c != '.').take(254).collect();

        emit!(ctx.bufs, "ssbo_addr_temp = uint(floatBitsToUint({})) >> 2;\n", srcs[1]);

        let (atomic_op, atomic_src) = if ctx.ssbo_atomic_mask & bit32(src.index as u32) != 0 {
            // atomicCounter is emulated with atomicOr.
            ("atomicOr", ", uint(0)")
        } else {
            ("", "")
        };
        let dtypeprefix = if is_integer_memory(ctx, src.file, src.index as u32) {
            Qual::IntBitsToFloat
        } else {
            Qual::UintBitsToFloat
        };

        if !src.indirect {
            emit_load_mem(
                ctx,
                &mydst,
                inst.dst[0].writemask,
                dtypeprefix.s(),
                atomic_op,
                &srcs[0],
                atomic_src,
            );
        } else {
            let atomic_ssbo = ctx.ssbo_atomic_mask & bit32(src.index as u32) != 0;
            let base = if atomic_ssbo { ctx.ssbo_atomic_array_base } else { ctx.ssbo_array_base };
            let (start, array_count) = bit_scan_consecutive_range(ctx.ssbo_used_mask);
            emit!(
                ctx.bufs,
                "switch (addr{} + {}) {{\n",
                src.ind.index,
                i32::from(src.index).wrapping_sub(base as i32)
            );
            for i in 0..array_count {
                emit!(ctx.bufs, "case {}:\n", i);
                let s = make_ssbo_varstring(ctx, (i + start) as u32);
                emit_load_mem(
                    ctx,
                    &mydst,
                    inst.dst[0].writemask,
                    dtypeprefix.s(),
                    atomic_op,
                    &s,
                    atomic_src,
                );
                ctx.bufs.emit("  break;\n");
            }
            ctx.bufs.emit("}\n");
        }
    } else if src.file == File::HwAtomic {
        emit!(ctx.bufs, "{} = uintBitsToFloat(atomicCounter({}));\n", dst, srcs[0]);
    }
    true
}

/// `get_atomic_opname`.
fn atomic_opname(opcode: Opcode) -> Option<(&'static str, bool)> {
    Some(match opcode {
        Opcode::Atomuadd => ("Add", false),
        Opcode::Atomxchg => ("Exchange", false),
        Opcode::Atomcas => ("CompSwap", true),
        Opcode::Atomand => ("And", false),
        Opcode::Atomor => ("Or", false),
        Opcode::Atomxor => ("Xor", false),
        Opcode::Atomumin => ("Min", false),
        Opcode::Atomumax => ("Max", false),
        Opcode::Atomimin => ("Min", false),
        Opcode::Atomimax => ("Max", false),
        other => {
            eprintln!("[virglrs] Illegal atomic opcode: {}", other as u8);
            return None;
        }
    })
}

/// `translate_resq`.
pub(super) fn translate_resq(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    srcs: &[String],
    dst: &str,
    writemask: &str,
) {
    let src = &inst.src[0];
    let mem_texture = inst.memory.map_or(Texture::Buffer, |m| m.texture);
    if src.file == File::Image {
        if inst.dst[0].writemask & 0x8 != 0 {
            ctx.shader_req_bits |= super::req::TXQS | super::req::INTS;
            emit!(ctx.bufs, "{} = {}(imageSamples({}));\n", dst, Qual::IntBitsToFloat.s(), srcs[0]);
        }
        if inst.dst[0].writemask & 0x7 != 0 {
            let swizzle_mask = if mem_texture == Texture::Array1d { ".xz" } else { "" };
            ctx.shader_req_bits |= super::req::IMAGE_SIZE | super::req::INTS;
            let skip_emit_writemask = mem_texture == Texture::Buffer;
            emit!(
                ctx.bufs,
                "{} = {}(imageSize({}){}{});\n",
                dst,
                Qual::IntBitsToFloat.s(),
                srcs[0],
                swizzle_mask,
                if skip_emit_writemask { "" } else { writemask }
            );
        }
    } else if src.file == File::Buffer {
        emit!(
            ctx.bufs,
            "{} = {}(int({}.length()) << 2);\n",
            dst,
            Qual::IntBitsToFloat.s(),
            srcs[0]
        );
    }
}

/// `translate_atomic`.
pub(super) fn translate_atomic(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sinfo: &SourceInfo,
    srcs: &[String],
    dst: &str,
) {
    let src = &inst.src[0];
    let mut stypeprefix = Qual::None;
    let mut dtypeprefix = Qual::None;
    let stypecast;
    let mut cas_str = String::new();

    if src.file == File::Image {
        if sinfo.sreg_index < 0 || sinfo.sreg_index as usize >= MAX_SHADER_IMAGES {
            ctx.bufs.set_error();
            return;
        }
        let (_, itype) = internalformat_string(ctx.images[sinfo.sreg_index as usize].decl.format);
        match itype {
            ReturnType::Sint => {
                stypeprefix = Qual::FloatBitsToInt;
                dtypeprefix = Qual::IntBitsToFloat;
                stypecast = Qual::Int;
            }
            ReturnType::Float => {
                if ctx.cfg.has_es31_compat {
                    ctx.shader_req_bits |= super::req::ES31_COMPAT;
                } else {
                    ctx.shader_req_bits |= super::req::SHADER_ATOMIC_FLOAT;
                }
                stypecast = Qual::Float;
            }
            _ => {
                stypeprefix = Qual::FloatBitsToUint;
                dtypeprefix = Qual::UintBitsToFloat;
                stypecast = Qual::Uint;
            }
        }
    } else {
        stypeprefix = Qual::FloatBitsToUint;
        dtypeprefix = Qual::UintBitsToFloat;
        stypecast = Qual::Uint;
    }

    let Some((opname, is_cas)) = atomic_opname(inst.opcode) else {
        ctx.bufs.set_error();
        return;
    };

    if is_cas {
        cas_str = format!(", {}({}({}))", stypecast.s(), stypeprefix.s(), srcs[3]);
    }

    if src.file == File::Image {
        let sreg = sinfo.sreg_index as usize;
        let (coord_prefix, is_ms) = coord_prefix(ctx.images[sreg].decl.resource);
        let conversion = if sinfo.override_no_cast[1] { "" } else { Qual::FloatBitsToInt.s() };
        let ms_str = if is_ms { format!(", int({}.w)", srcs[1]) } else { String::new() };

        if !set_image_qualifier(ctx, inst, i32::from(src.index), src.indirect) {
            ctx.bufs.set_error();
            return;
        }

        if !src.indirect {
            emit!(
                ctx.bufs,
                "{} = {}(imageAtomic{}({}, {}({}({})){}, {}({}({})){}));\n",
                dst,
                dtypeprefix.s(),
                opname,
                srcs[0],
                coord_prefix.s(),
                conversion,
                srcs[1],
                ms_str,
                stypecast.s(),
                stypeprefix.s(),
                srcs[2],
                cas_str
            );
        } else if let Some(image) = ctx.lookup_image_array_ptr(i32::from(src.index)) {
            let image = ctx.image_arrays[image];
            let basearrayidx = image.first;
            let array_size = image.array_size;
            emit!(
                ctx.bufs,
                "switch (addr{} + {}) {{\n",
                src.ind.index,
                i32::from(src.index) - basearrayidx
            );
            let cname = proc_prefix(ctx.prog_type);
            for i in 0..array_size {
                let s = format!("{cname}img{basearrayidx}[{i}]");
                emit!(
                    ctx.bufs,
                    "case {}: {} = {}(imageAtomic{}({}, {}({}({})){}, {}({}({})){}));\n",
                    i,
                    dst,
                    dtypeprefix.s(),
                    opname,
                    s,
                    coord_prefix.s(),
                    conversion,
                    srcs[1],
                    ms_str,
                    stypecast.s(),
                    stypeprefix.s(),
                    srcs[2],
                    cas_str
                );
            }
            ctx.bufs.emit("}\n");
        }
        ctx.shader_req_bits |= super::req::IMAGE_ATOMIC;
    }
    if src.file == File::Buffer || src.file == File::Memory {
        if src.index < 0 || src.index as usize >= MAX_SHADER_BUFFERS {
            ctx.bufs.set_error();
            return;
        }
        let ty;
        if is_integer_memory(ctx, src.file, src.index as u32) {
            ty = Qual::Int;
            dtypeprefix = Qual::IntBitsToFloat;
            stypeprefix = Qual::FloatBitsToInt;
        } else {
            ty = Qual::Uint;
            dtypeprefix = Qual::UintBitsToFloat;
            stypeprefix = Qual::FloatBitsToUint;
        }
        if is_cas {
            cas_str = format!(", {}({}({}))", ty.s(), stypeprefix.s(), srcs[3]);
        }
        emit!(
            ctx.bufs,
            "{} = {}(atomic{}({}[int(floatBitsToInt({})) >> 2], {}({}({}).x){}));\n",
            dst,
            dtypeprefix.s(),
            opname,
            srcs[0],
            srcs[1],
            ty.s(),
            stypeprefix.s(),
            srcs[2],
            cas_str
        );
    }
    if src.file == File::HwAtomic {
        if sinfo.imm_value == -1 {
            emit!(
                ctx.bufs,
                "{} = {}(atomicCounterDecrement({}) + 1u);\n",
                dst,
                dtypeprefix.s(),
                srcs[0]
            );
        } else if sinfo.imm_value == 1 {
            emit!(
                ctx.bufs,
                "{} = {}(atomicCounterIncrement({}));\n",
                dst,
                dtypeprefix.s(),
                srcs[0]
            );
        } else {
            emit!(
                ctx.bufs,
                "{} = {}(atomicCounter{}ARB({}, floatBitsToUint({}).x{}));\n",
                dst,
                dtypeprefix.s(),
                opname,
                srcs[0],
                srcs[2],
                cas_str
            );
        }
    }
}
