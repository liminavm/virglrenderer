// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! What a stage does on the way out of `main`: stream-out copies, clip and cull distance
//! moves, the prescale, and the fragment stage's tests and fixups.

use super::{
    Ctx, Io, IoDir, MAX_SO_OUTPUTS, MAX_VARYING, Processor, bit32, emit, stage_output_name_prefix,
    sysval,
};
use crate::vrend::pipe::{CompareFunc, LogicOp};
use crate::vrend::tgsi::Semantic;

/// `emit_cbuf_writes`.
fn emit_cbuf_writes(ctx: &mut Ctx<'_>) {
    for i in ctx.outputs.len() as u32..ctx.cfg.max_draw_buffers {
        emit!(ctx.bufs, "fsout_c{} = fsout_c0;\n", i);
    }
}

/// `emit_a8_swizzle`.
fn emit_a8_swizzle(ctx: &mut Ctx<'_>) {
    ctx.bufs.emit("fsout_c0.x = fsout_c0.w;\n");
}

/// `emit_alpha_test`.
fn emit_alpha_test(ctx: &mut Ctx<'_>) {
    if ctx.outputs.is_empty() {
        return;
    }
    // The alpha stanza is only emitted when the first output is 0.
    if !ctx.write_all_cbufs && ctx.outputs[0].sid != 0 {
        return;
    }
    let comp_buf = match ctx.key.alpha_test {
        CompareFunc::Never => "false".to_string(),
        CompareFunc::Always => "true".to_string(),
        f => {
            let op = match f {
                CompareFunc::Less => "<",
                CompareFunc::Equal => "==",
                CompareFunc::LessEqual => "<=",
                CompareFunc::Greater => ">",
                CompareFunc::NotEqual => "!=",
                _ => ">=",
            };
            ctx.bufs.required_sysval_uniform_decls |= sysval::ALPHA_REF_VAL;
            format!("fsout_c0.w {op} alpha_ref_val")
        }
    };
    emit!(ctx.bufs, "if (!({})) {{\n\tdiscard;\n}}\n", comp_buf);
}

/// `emit_pstipple_pass`.
fn emit_pstipple_pass(ctx: &mut Ctx<'_>) {
    let mask = super::super::POLYGON_STIPPLE_SIZE - 1;
    ctx.bufs.emit("{\n");
    emit!(ctx.bufs, "   int spx = int(gl_FragCoord.x) & {};\n", mask);
    emit!(ctx.bufs, "   int spy = int(gl_FragCoord.y) & {};\n", mask);
    ctx.bufs.emit("   stip_temp = stipple_pattern[spy] & (0x80000000u >> spx);\n");
    ctx.bufs.emit("   if (stip_temp == 0u) {\n      discard;\n   }\n");
    ctx.bufs.emit("}\n");
    ctx.bufs.required_sysval_uniform_decls |= sysval::PSTIPPLE_SAMPLER;
}

/// `emit_color_select`.
pub(super) fn emit_color_select(ctx: &mut Ctx<'_>) {
    if !ctx.key.color_two_side || ctx.color_in_mask & 0x3 == 0 {
        return;
    }
    let name_prefix = ctx.stage_input_name_prefix(ctx.prog_type);
    if ctx.color_in_mask & 1 != 0 {
        emit!(
            ctx.bufs,
            "realcolor0 = gl_FrontFacing ? {}_c0 : {}_bc0;\n",
            name_prefix,
            name_prefix
        );
    }
    if ctx.color_in_mask & 2 != 0 {
        emit!(
            ctx.bufs,
            "realcolor1 = gl_FrontFacing ? {}_c1 : {}_bc1;\n",
            name_prefix,
            name_prefix
        );
    }
}

/// `emit_prescale`.
pub(super) fn emit_prescale(ctx: &mut Ctx<'_>) {
    ctx.bufs.emit("gl_Position.y = gl_Position.y * winsys_adjust_y;\n");
    ctx.bufs.required_sysval_uniform_decls |= sysval::WINSYS_ADJUST_Y;
}

/// `prepare_so_movs`.
pub(super) fn prepare_so_movs(ctx: &mut Ctx<'_>) {
    let so = ctx.so.expect("stream output present");
    for (i, o) in so.outputs.iter().enumerate().take(MAX_SO_OUTPUTS) {
        ctx.write_so_outputs[i] = true;
        if o.start_component != 0 || o.num_components != 4 {
            continue;
        }
        let Some(out) = ctx.outputs.get(usize::from(o.register_index)) else {
            continue;
        };
        if out.name == Semantic::ClipDist || out.name == Semantic::Position {
            continue;
        }
        ctx.outputs[usize::from(o.register_index)].stream = u32::from(o.stream);
        if ctx.prog_type == Processor::Geometry && o.stream != 0 {
            ctx.shader_req_bits |= super::req::GPU_SHADER5;
        }
        ctx.write_so_outputs[i] = false;
    }
}

/// `get_io_slot`: the output covering register `idx`.
fn io_slot(slots: &[Io], idx: u32) -> Option<usize> {
    slots.iter().position(|s| s.first <= idx && s.last >= idx)
}

/// `get_blockname`.
pub(super) fn blockname(stage_prefix: &str, io: &Io) -> String {
    format!("block_{stage_prefix}g{}", io.sid)
}

/// `get_blockvarname`.
pub(super) fn blockvarname(stage_prefix: &str, io: &Io, postfix: &str) -> String {
    format!("{stage_prefix}g{}{postfix}", io.first)
}

/// `get_so_name`.
fn so_name(ctx: &Ctx<'_>, from_block: bool, output: &Io, index: u32, wm: &str) -> String {
    if output.first == output.last
        || (output.name != Semantic::Generic && output.name != Semantic::TexCoord)
    {
        format!("{}{}", output.glsl_name, wm)
    } else if output.name == Semantic::Generic && ctx.prefer_generic_io_block(IoDir::Out) {
        let stage_prefix = stage_output_name_prefix(ctx.prog_type);
        let block = if from_block {
            blockname(stage_prefix, output)
        } else {
            blockvarname(stage_prefix, output, "")
        };
        format!("{}.{}[{}]{}", block, output.glsl_name, index as i32 - output.first as i32, wm)
    } else {
        format!("{}[{}]{}", output.glsl_name, index as i32 - output.first as i32, wm)
    }
}

/// `emit_so_movs`.
pub(super) fn emit_so_movs(ctx: &mut Ctx<'_>) {
    let so = ctx.so.expect("stream output present");
    if so.outputs.len() >= MAX_SO_OUTPUTS {
        eprintln!("[virglrs] Num outputs exceeded, max is {MAX_SO_OUTPUTS}");
        ctx.bufs.set_error();
        return;
    }

    for (i, o) in so.outputs.iter().enumerate() {
        let register_index = u32::from(o.register_index);
        // The C asserts the register is an output; a stream output naming none is the guest's
        // error, and fails the translation here rather than the process.
        let Some(output) = io_slot(&ctx.outputs, register_index).map(|s| ctx.outputs[s].clone())
        else {
            ctx.bufs.set_error();
            return;
        };

        let mut writemask = String::new();
        if o.start_component != 0 {
            writemask.push('.');
            for j in 0..o.num_components {
                let idx = o.start_component + j;
                if idx >= 4 {
                    break;
                }
                writemask.push(if idx <= 2 { (b'x' + idx) as char } else { 'w' });
            }
        }

        if !ctx.write_so_outputs[i] {
            if register_index > ctx.outputs.len() as u32 {
                ctx.so_names[i] = None;
            } else if output.name == Semantic::ClipVertex && ctx.has_clipvertex {
                ctx.so_names[i] = Some("clipv_tmp".to_string());
                ctx.has_clipvertex_so = true;
            } else {
                ctx.so_names[i] = Some(so_name(ctx, true, &output, register_index, ""));
            }
        } else if ctx.so_names[i].is_none() {
            ctx.so_names[i] = Some(format!("tfout{i}"));
        }

        let outtype = if o.num_components == 1 {
            if output.is_int { "intBitsToFloat".to_string() } else { "float".to_string() }
        } else {
            format!("vec{}", o.num_components)
        };

        if output.name == Semantic::ClipDist {
            if output.first == output.last {
                emit!(
                    ctx.bufs,
                    "tfout{} = {}(clip_dist_temp[{}]{});\n",
                    i,
                    outtype,
                    output.sid,
                    writemask
                );
            } else {
                emit!(
                    ctx.bufs,
                    "tfout{} = {}(clip_dist_temp[{}]{});\n",
                    i,
                    outtype,
                    output.sid as i32 + register_index as i32 - output.first as i32,
                    writemask
                );
            }
        } else if ctx.write_so_outputs[i] {
            if ctx.so_need_temp(i)
                || ctx.prog_type == Processor::Geometry
                || output.glsl_predefined_no_emit
            {
                let out_var = so_name(ctx, false, &output, register_index, &writemask);
                emit!(ctx.bufs, "tfout{} = {}({});\n", i, outtype, out_var);
            } else {
                let out_var = so_name(ctx, true, &output, register_index, &writemask);
                ctx.so_names[i] = Some(out_var);
            }
        }
    }
}

/// `emit_clip_dist_movs`.
pub(super) fn emit_clip_dist_movs(ctx: &mut Ctx<'_>) {
    let has_prop = (ctx.num_clip_dist_prop + ctx.num_cull_dist_prop) > 0;
    let mut num_clip =
        i32::from(if has_prop { ctx.num_clip_dist_prop } else { ctx.key.num_out_clip });
    let num_cull = i32::from(if has_prop { ctx.num_cull_dist_prop } else { ctx.key.num_out_cull });

    let num_clip_cull = num_cull + num_clip;
    if ctx.num_out_clip_dist != 0 && num_clip_cull == 0 {
        num_clip = ctx.num_out_clip_dist;
    }

    let prefix = if ctx.prog_type == Processor::TessCtrl { "gl_out[gl_InvocationID]." } else { "" };

    if ctx.num_out_clip_dist == 0
        && ctx.is_last_vertex_stage
        && ctx.outputs.len() as u32 + 2 <= MAX_VARYING
    {
        ctx.bufs.emit("if (clip_plane_enabled) {\n");
        for i in 0..8 {
            emit!(
                ctx.bufs,
                "  {}gl_ClipDistance[{}] = dot({}, clipp[{}]);\n",
                prefix,
                i,
                if ctx.has_clipvertex { "clipv_tmp" } else { "gl_Position" },
                i
            );
        }
        ctx.bufs.emit("}\n");
        ctx.bufs.required_sysval_uniform_decls |= sysval::CLIP_PLANE;
    }
    let mut ndists = ctx.num_out_clip_dist;
    if has_prop {
        ndists = num_clip + num_cull;
    }
    for i in 0..ndists {
        let clipidx = if i < 4 { 0 } else { 1 };
        let wm = ['x', 'y', 'z', 'w'][(i & 3) as usize];
        let (is_cull, clip_cull) = if i >= num_clip { (true, "Cull") } else { (false, "Clip") };
        emit!(
            ctx.bufs,
            "{}gl_{}Distance[{}] = clip_dist_temp[{}].{};\n",
            prefix,
            clip_cull,
            if is_cull { i - num_clip } else { i },
            clipidx,
            wm
        );
    }
}

/// `emit_fog_fixup_hdr`.
pub(super) fn emit_fog_fixup_hdr(ctx: &mut Ctx<'_>) {
    let mut fixup_mask = ctx.key.vs.fog_fixup_mask;
    let prefix = stage_output_name_prefix(Processor::Vertex);
    while fixup_mask != 0 {
        let semantic = fixup_mask.trailing_zeros();
        super::hdr!(ctx.bufs, "out vec4 {}_f{};\n", prefix, semantic);
        fixup_mask &= !bit32(semantic);
    }
}

/// `emit_fog_fixup_write`: unwritten fog outputs are forced to (0, 0, 0, 1).
fn emit_fog_fixup_write(ctx: &mut Ctx<'_>) {
    let mut fixup_mask = ctx.key.vs.fog_fixup_mask;
    let prefix = stage_output_name_prefix(Processor::Vertex);
    while fixup_mask != 0 {
        let semantic = fixup_mask.trailing_zeros();
        emit!(ctx.bufs, "{}_f{} = vec4(0.0, 0.0, 0.0, 1.0);\n", prefix, semantic);
        fixup_mask &= !bit32(semantic);
    }
}

/// `handle_vertex_proc_exit`.
pub(super) fn handle_vertex_proc_exit(ctx: &mut Ctx<'_>) {
    if ctx.so.is_some() && !ctx.key.gs_present && !ctx.key.tes_present {
        emit_so_movs(ctx);
    }
    if ctx.cfg.has_cull_distance {
        emit_clip_dist_movs(ctx);
    }
    if !ctx.key.gs_present && !ctx.key.tes_present {
        emit_prescale(ctx);
    }
    if ctx.key.vs.fog_fixup_mask != 0 {
        emit_fog_fixup_write(ctx);
    }
}

/// `emit_fragment_logicop`.
fn emit_fragment_logicop(ctx: &mut Ctx<'_>) {
    let Some(func) = ctx.key.fs.logicop_func else {
        return;
    };
    let n = ctx.outputs.len();
    let mut src = vec![String::new(); n];
    let mut src_fb = vec![String::new(); n];
    let mut scale = vec![0f64; n];
    let mut mask = vec![0i32; n];

    for i in 0..n {
        let bits = ctx.key.fs.surface_component_bits.get(i).copied().unwrap_or(0);
        mask[i] = (1i32.wrapping_shl(u32::from(bits))).wrapping_sub(1);
        scale[i] = f64::from(mask[i]);
        match func {
            LogicOp::Invert => {
                src_fb[i] = format!("ivec4({:.6} * fsout_c{} + 0.5)", scale[i], i);
            }
            LogicOp::Nor
            | LogicOp::AndInverted
            | LogicOp::AndReverse
            | LogicOp::Xor
            | LogicOp::Nand
            | LogicOp::And
            | LogicOp::Equiv
            | LogicOp::OrInverted
            | LogicOp::OrReverse
            | LogicOp::Or => {
                src_fb[i] = format!("ivec4({:.6} * fsout_c{} + 0.5)", scale[i], i);
                src[i] = format!("ivec4({:.6} * fsout_tmp_c{} + 0.5)", scale[i], i);
            }
            LogicOp::CopyInverted => {
                src[i] = format!("ivec4({:.6} * fsout_tmp_c{} + 0.5)", scale[i], i);
            }
            LogicOp::Copy | LogicOp::Noop | LogicOp::Clear | LogicOp::Set => {}
        }
    }

    let mut full_op = vec![String::new(); n];
    for i in 0..n {
        full_op[i] = match func {
            LogicOp::Clear => "vec4(0)".to_string(),
            LogicOp::Noop => String::new(),
            LogicOp::Set => "vec4(1)".to_string(),
            LogicOp::Copy => format!("fsout_tmp_c{i}"),
            LogicOp::CopyInverted => format!("~{}", src[i]),
            LogicOp::Invert => format!("~{}", src_fb[i]),
            LogicOp::And => format!("{} & {}", src[i], src_fb[i]),
            LogicOp::Nand => format!("~( {} & {} )", src[i], src_fb[i]),
            LogicOp::Nor => format!("~( {} | {} )", src[i], src_fb[i]),
            LogicOp::AndInverted => format!("~{} & {}", src[i], src_fb[i]),
            LogicOp::AndReverse => format!("{} & ~{}", src[i], src_fb[i]),
            LogicOp::Xor => format!("{} ^{}", src[i], src_fb[i]),
            LogicOp::Equiv => format!("~( {} ^ {} )", src[i], src_fb[i]),
            LogicOp::OrInverted => format!("~{} | {}", src[i], src_fb[i]),
            LogicOp::OrReverse => format!("{} | ~{}", src[i], src_fb[i]),
            LogicOp::Or => format!("{} | {}", src[i], src_fb[i]),
        };
    }

    for i in 0..n {
        match func {
            LogicOp::Noop => {}
            LogicOp::Copy | LogicOp::Clear | LogicOp::Set => {
                emit!(ctx.bufs, "fsout_c{} = {};\n", i, full_op[i]);
            }
            _ => {
                emit!(
                    ctx.bufs,
                    "fsout_c{} = vec4(({}) & {}) / {:.6};\n",
                    i,
                    full_op[i],
                    mask[i],
                    scale[i]
                );
            }
        }
    }
}

/// `emit_cbuf_swizzle`.
fn emit_cbuf_swizzle(ctx: &mut Ctx<'_>) {
    let mut cbuf_id = 0u32;
    for i in 0..ctx.outputs.len() {
        if ctx.outputs[i].name == Semantic::Color {
            if u32::from(ctx.key.fs.swizzle_output_rgb_to_bgr) & bit32(cbuf_id) != 0 {
                emit!(ctx.bufs, "fsout_c{} = fsout_c{}.zyxw;\n", cbuf_id, cbuf_id);
            }
            cbuf_id += 1;
        }
    }
}

/// `emit_cbuf_colorspace_convert`.
fn emit_cbuf_colorspace_convert(ctx: &mut Ctx<'_>) {
    for i in 0..ctx.outputs.len() as u32 {
        if u32::from(ctx.key.fs.needs_manual_srgb_encode_bitmask) & bit32(i) != 0 {
            emit!(
                ctx.bufs,
                "{{\n   vec3 temp = fsout_c{}.xyz;\n   bvec3 thresh = lessThanEqual(temp, vec3(0.0031308));\n   vec3 a = temp * vec3(12.92);\n   vec3 b = ( vec3(1.055) * pow(temp, vec3(1.0/2.4)) ) - vec3(0.055);\n   fsout_c{}.xyz = mix(b, a, thresh);\n}}\n",
                i,
                i
            );
        }
    }
}

/// `handle_fragment_proc_exit`.
pub(super) fn handle_fragment_proc_exit(ctx: &mut Ctx<'_>) {
    if ctx.key.pstipple_enabled {
        emit_pstipple_pass(ctx);
    }
    if ctx.key.fs.cbufs_are_a8_bitmask != 0 {
        emit_a8_swizzle(ctx);
    }
    if ctx.key.add_alpha_test {
        emit_alpha_test(ctx);
    }
    if ctx.key.fs.logicop_func.is_some() {
        emit_fragment_logicop(ctx);
    }
    if ctx.key.fs.swizzle_output_rgb_to_bgr != 0 {
        emit_cbuf_swizzle(ctx);
    }
    if ctx.key.fs.needs_manual_srgb_encode_bitmask != 0 {
        emit_cbuf_colorspace_convert(ctx);
    }
    if ctx.write_all_cbufs {
        emit_cbuf_writes(ctx);
    }
}

/// `emit_fs_clipdistance_load`.
pub(super) fn emit_fs_clipdistance_load(ctx: &mut Ctx<'_>) {
    if !ctx.fs_uses_clipdist_input {
        return;
    }
    let num_in_clip = i32::from(ctx.key.num_in_clip);
    let prev_num = num_in_clip + i32::from(ctx.key.num_in_cull);
    let prefix = if ctx.prog_type == Processor::TessCtrl { "gl_out[gl_InvocationID]." } else { "" };
    let ndists = if prev_num > 0 { prev_num } else { ctx.num_in_clip_dist };
    for i in 0..ndists {
        let clipidx = if i < 4 { 0 } else { 1 };
        let wm = ['x', 'y', 'z', 'w'][(i & 3) as usize];
        let is_cull = prev_num > 0 && i >= num_in_clip && i < prev_num;
        let clip_cull = if is_cull { "Cull" } else { "Clip" };
        emit!(
            ctx.bufs,
            "clip_dist_temp[{}].{} = {}gl_{}Distance[{}];\n",
            clipidx,
            wm,
            prefix,
            clip_cull,
            if is_cull { i - num_in_clip } else { i }
        );
    }
}
