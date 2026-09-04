// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The declaration side of the walk: what each `DCL`, `IMM` and `PROPERTY` records in the
//! context, and the first pass that runs ahead of it.

use super::{
    Context, Failure, Immed, Io, MAX_IMMEDIATE, MAX_IO, MAX_SAMPLERS, MAX_SYSTEM_VALUES, TempRange,
    VecType, bit32, fail, req, samplertype_is_shadow, samplertype_to_req_bits,
    stage_output_name_prefix, sysval,
};
use crate::vrend::pipe::Swizzle;
use crate::vrend::proto::Format;
use crate::vrend::shader::{Array, MAX_COMBINED_SSBO_BINDING_POINTS, MAX_SHADER_IMAGES};
use crate::vrend::tgsi::{
    Declaration, File, ImageInfo, ImmType, Immediate, Instruction, Interpolate, Location, Opcode,
    Processor, Property, PropertyToken, ReturnType, Semantic, Texture,
};

/// `iter_decls`: the first pass over the declarations.
pub(super) fn iter_decls(ctx: &mut Context<'_>, decl: &Declaration) -> Result<(), Failure> {
    match decl.file {
        File::Input => {
            if decl.semantic.name == Semantic::Fog {
                ctx.fog_input_mask |= bit32(u32::from(decl.semantic.index));
            }
            if ctx.prog_type == Processor::Fragment {
                if ctx.inputs.len() >= MAX_IO {
                    return fail(format!("Number of inputs exceeded, max is {MAX_IO}"));
                }
                let name = decl.semantic.name;
                let first = u32::from(decl.first);
                if ctx.inputs.iter().any(|i| {
                    i.name == name && i.sid == u32::from(decl.semantic.index) && i.first == first
                }) {
                    return Ok(());
                }
                ctx.inputs.push(Io { name, first, last: u32::from(decl.last), ..Io::default() });
            }
        }
        File::Output => {
            if decl.semantic.name == Semantic::Fog {
                ctx.fog_output_mask |= bit32(u32::from(decl.semantic.index));
            }
        }
        File::Buffer if ctx.ssbo_first_binding > u32::from(decl.first) => {
            ctx.ssbo_first_binding = u32::from(decl.first);
        }
        _ => {}
    }
    Ok(())
}

/// `analyze_instruction`: the first pass over the instructions.
pub(super) fn analyze_instruction(ctx: &mut Context<'_>, inst: &Instruction) {
    if inst.opcode == Opcode::Atomimin || inst.opcode == Opcode::Atomimax {
        let src = &inst.src[0];
        if src.file == File::Buffer {
            ctx.ssbo_integer_mask |= bit32(src.index as u32);
        }
        if src.file == File::Memory {
            ctx.integer_memory = true;
        }
    }

    if !ctx.fs_uses_clipdist_input && ctx.prog_type == Processor::Fragment {
        for src in inst.srcs() {
            if src.file == File::Input {
                let idx = i32::from(src.index);
                if ctx.inputs.iter().any(|i| {
                    i.first as i32 <= idx && i.last as i32 >= idx && i.name == Semantic::ClipDist
                }) {
                    ctx.fs_uses_clipdist_input = true;
                    break;
                }
            }
        }
    }
}

/// `get_type`.
fn get_type(signed_int_mask: u32, unsigned_int_mask: u32, bit: u32) -> VecType {
    if signed_int_mask & bit32(bit) != 0 {
        VecType::Int
    } else if unsigned_int_mask & bit32(bit) != 0 {
        VecType::Uint
    } else {
        VecType::Float
    }
}

/// `find_overlapping_io`: an earlier entry of `io` (the last one excluded) whose range
/// overlaps the declaration's.
fn find_overlapping_io(io: &[Io], decl: &Declaration) -> Option<usize> {
    let first = u32::from(decl.first);
    let last = u32::from(decl.last);
    io[..io.len() - 1].iter().position(|j| {
        j.interpolate == decl.interp.interpolate
            && j.name == decl.semantic.name
            && ((j.first <= first && j.last > first) || (j.first < last && j.last >= last))
    })
}

/// `map_overlapping_io_array`: fold the newest entry of `io` into the array it overlaps.
fn map_overlapping_io_array(io: &mut [Io], new: usize, decl: &Declaration) {
    let Some(overlap) = find_overlapping_io(io, decl) else {
        return;
    };
    if io[overlap].needs_override {
        return;
    }
    let delta = io[new].first as i64 - io[overlap].first as i64;
    if delta >= 0 {
        io[new].array_offset = delta as u32;
        io[new].overlapping_array = Some(overlap);
        io[overlap].last = io[overlap].last.max(io[new].last);
    } else {
        io[overlap].overlapping_array = Some(new);
        io[overlap].array_offset = (-delta) as u32;
        io[new].last = io[overlap].last.max(io[new].last);
    }
    io[overlap].usage_mask |= io[new].usage_mask;
    io[new].usage_mask = io[overlap].usage_mask;
}

/// `sysvalue_map`: the GLSL for a system value, what it requires, and whether its writemask
/// is dropped.
fn sysvalue_map(name: Semantic) -> Option<(&'static str, u64, bool)> {
    Some(match name {
        Semantic::InstanceId => ("gl_InstanceID", req::INSTANCE_ID | req::INTS, true),
        Semantic::VertexId => ("gl_VertexID", req::INTS, true),
        Semantic::HelperInvocation => ("gl_HelperInvocation", req::ES31_COMPAT, true),
        Semantic::SampleId => ("gl_SampleID", req::SAMPLE_SHADING | req::INTS, true),
        Semantic::SamplePos => ("gl_SamplePosition", req::SAMPLE_SHADING, true),
        Semantic::InvocationId => ("gl_InvocationID", req::INTS | req::GPU_SHADER5, true),
        Semantic::VertexIdNoBase => (
            "(gl_VertexID - gl_BaseVertexARB)",
            req::SHADER_DRAW_PARAMETERS | req::INTS | req::GPU_SHADER5,
            true,
        ),
        Semantic::SampleMask => ("gl_SampleMaskIn[0]", req::INTS | req::GPU_SHADER5, true),
        Semantic::PrimId => ("gl_PrimitiveID", req::INTS | req::GPU_SHADER5, true),
        Semantic::TessCoord => ("gl_TessCoord", 0, false),
        Semantic::VerticesIn => ("gl_PatchVerticesIn", req::INTS, true),
        Semantic::TessOuter => ("gl_TessLevelOuter", 0, true),
        Semantic::TessInner => ("gl_TessLevelInner", 0, true),
        Semantic::ThreadId => ("gl_LocalInvocationID", 0, false),
        Semantic::BlockId => ("gl_WorkGroupID", 0, false),
        Semantic::GridSize => ("gl_NumWorkGroups", 0, false),
        Semantic::BaseVertex => ("gl_BaseVertexARB", req::SHADER_DRAW_PARAMETERS | req::INTS, true),
        Semantic::BaseInstance => {
            ("gl_BaseInstanceARB", req::SHADER_DRAW_PARAMETERS | req::INTS, true)
        }
        Semantic::DrawId => {
            ("gl_DrawIDARB + drawid_base", req::SHADER_DRAW_PARAMETERS | req::INTS, true)
        }
        _ => return None,
    })
}

/// `allocate_temp_range`.
fn allocate_temp_range(ctx: &mut Context<'_>, first: i32, last: i32, array_id: i32) {
    if array_id > 0 {
        ctx.temp_ranges.push(TempRange { first, last, array_id, precise_result: false });
    } else {
        for i in first..=last {
            ctx.temp_ranges.push(TempRange {
                first: i,
                last: i,
                array_id: 0,
                precise_result: false,
            });
        }
    }
}

/// `add_images`.
fn add_images(ctx: &mut Context<'_>, first: usize, last: usize, img_decl: &ImageInfo) {
    let descr = Format::from_wire(u32::from(img_decl.format)).and_then(Format::describe);
    if let Some(d) = descr {
        let sw = |i: usize| d.swizzle[i];
        let two_channel = d.nr_channels == 2
            && sw(0) == Some(Swizzle::X)
            && sw(1) == Some(Swizzle::Y)
            && sw(2) == Some(Swizzle::Zero)
            && sw(3) == Some(Swizzle::One);
        let packed = matches!(
            d.name,
            "R11G11B10_FLOAT"
                | "R10G10B10A2_UINT"
                | "R10G10B10A2_UNORM"
                | "R16G16B16A16_UNORM"
                | "R16G16B16A16_SNORM"
        );
        let one_channel = d.nr_channels == 1
            && sw(0) == Some(Swizzle::X)
            && sw(1) == Some(Swizzle::Zero)
            && sw(2) == Some(Swizzle::Zero)
            && sw(3) == Some(Swizzle::One)
            && (d.channels[0].bits == 8 || d.channels[0].bits == 16);
        if two_channel || packed || one_channel {
            ctx.shader_req_bits |= req::NV_IMAGE_FORMATS;
        }
    }

    for i in first..=last {
        ctx.images[i].decl = *img_decl;
        ctx.images[i].vflag = false;
        ctx.images_used_mask |= bit32(i as u32);
        if !samplertype_is_shadow(ctx.images[i].decl.resource) {
            ctx.shader_req_bits |= samplertype_to_req_bits(ctx.images[i].decl.resource);
        }
    }

    if ctx.info.is_indirect(File::Image) {
        if let Some(last_array) = ctx.image_arrays.last().copied() {
            // A run consecutive to the last array with the same declaration extends it.
            if last_array.first + last_array.array_size == first as i32
                && ctx.images[last_array.first as usize].decl == ctx.images[first].decl
                && ctx.images[last_array.first as usize].image_return
                    == ctx.images[first].image_return
            {
                ctx.image_arrays.last_mut().expect("checked").array_size +=
                    (last - first + 1) as i32;
                if ctx.image_last_binding < last as i32 {
                    ctx.image_last_binding = last as i32;
                }
                return;
            }
        }
        ctx.image_arrays.push(Array { first: first as i32, array_size: (last - first + 1) as i32 });
    }

    if ctx.image_last_binding < last as i32 {
        ctx.image_last_binding = last as i32;
    }
}

/// `add_samplers`.
fn add_samplers(ctx: &mut Context<'_>, first: usize, last: usize, ty: Texture, ret: ReturnType) {
    if ret == ReturnType::Sint || ret == ReturnType::Uint {
        ctx.shader_req_bits |= req::INTS;
    }
    for i in first..=last {
        ctx.samplers[i].ret = ret;
        ctx.samplers[i].ty = ty;
    }
    if ctx.info.is_indirect(File::Sampler) {
        if let Some(last_array) = ctx.sampler_arrays.last().copied()
            && last_array.first + last_array.array_size == first as i32
            && ctx.samplers[last_array.first as usize].ty == ty
            && ctx.samplers[last_array.first as usize].ret == ret
        {
            ctx.sampler_arrays.last_mut().expect("checked").array_size += (last - first + 1) as i32;
            return;
        }
        ctx.sampler_arrays
            .push(Array { first: first as i32, array_size: (last - first + 1) as i32 });
    }
}

/// `iter_declaration`: the second pass over a declaration.
pub(super) fn iter_declaration(ctx: &mut Context<'_>, decl: &Declaration) -> Result<(), Failure> {
    let processor = ctx.prog_type;
    let first = u32::from(decl.first);
    let last = u32::from(decl.last);
    let sindex = u32::from(decl.semantic.index);
    let array_id = decl.array.map_or(0, u32::from);

    match decl.file {
        File::Input => {
            if ctx.inputs.iter().any(|j| {
                j.name == decl.semantic.name
                    && j.sid == sindex
                    && j.first == first
                    && ((decl.array.is_none() && j.array_id == 0) || j.array_id == array_id)
            }) {
                return Ok(());
            }

            let i = ctx.inputs.len();
            if i + 1 > MAX_IO {
                return fail(format!("Number of inputs exceeded, max is {MAX_IO}"));
            }
            if first > last {
                return fail(format!("Wrong range: First ({first}) > Last ({last})"));
            }
            if last as usize >= MAX_IO {
                return fail(format!("Input slot out of range, want {last}, max is {MAX_IO}"));
            }

            let mut io = Io::default();
            if processor == Processor::Vertex {
                ctx.attrib_input_mask |= bit32(first);
                // The C types the attribute from the key's integer masks, which are only ever
                // filled under `use_integer`; this host never sets it.
                io.ty = get_type(0, 0, first);
            }
            io.name = decl.semantic.name;
            io.sid = sindex;
            io.interpolate = decl.interp.interpolate;
            io.location = decl.interp.location;
            io.first = first;
            io.last = last;
            io.array_id = array_id;
            io.usage_mask = decl.usage_mask;
            io.num_components = 4;
            io.glsl_predefined_no_emit = false;
            io.glsl_no_index = false;
            io.override_no_wm = io.num_components == 1;
            io.glsl_gl_block = false;
            io.overlapping_array = None;
            ctx.inputs.push(io);

            if processor == Processor::Fragment {
                if decl.interp.location == Location::Sample {
                    ctx.shader_req_bits |= req::GPU_SHADER5;
                    ctx.has_sample_input = true;
                }
                if decl.interp.interpolate == Interpolate::Linear && ctx.cfg.has_nopersective {
                    ctx.shader_req_bits |= req::SHADER_NOPERSPECTIVE_INTERPOLATION;
                    ctx.has_noperspective = true;
                }
            }

            map_overlapping_io_array(&mut ctx.inputs, i, decl);

            if !ctx.inputs[i].glsl_predefined_no_emit {
                // If the output of the previous shader contained arrays, a non-array input
                // here may be part of one.
                for array in &ctx.key.in_arrays.layout {
                    if array.name == decl.semantic.name
                        && array.sid <= sindex
                        && array.sid + array.size >= sindex
                    {
                        ctx.inputs[i].sid = array.sid;
                        ctx.inputs[i].last =
                            (ctx.inputs[i].first + array.size).max(ctx.inputs[i].last);
                        break;
                    }
                }
            }

            if ctx.inputs[i].first != ctx.inputs[i].last {
                ctx.glsl_ver_required = ctx.require_glsl_ver(150);
            }

            let mut name_prefix = ctx.stage_input_name_prefix(processor);
            let mut add_two_side = false;

            match ctx.inputs[i].name {
                Semantic::Color => {
                    if processor == Processor::Fragment {
                        if ctx.glsl_ver_required < 140 {
                            if sindex == 0 {
                                name_prefix = "gl_Color";
                            } else if sindex == 1 {
                                name_prefix = "gl_SecondaryColor";
                            } else {
                                eprintln!("[virglrs] got illegal color semantic index {sindex}");
                            }
                            ctx.inputs[i].glsl_no_index = true;
                        } else if ctx.key.color_two_side {
                            if ctx.inputs.len() + 1 >= MAX_IO {
                                return fail(format!("Number of inputs exceeded, max is {MAX_IO}"));
                            }
                            ctx.inputs.push(Io {
                                name: Semantic::BColor,
                                sid: sindex,
                                interpolate: decl.interp.interpolate,
                                location: decl.interp.location,
                                first,
                                last,
                                glsl_predefined_no_emit: false,
                                glsl_no_index: false,
                                override_no_wm: false,
                                ..Io::default()
                            });
                            ctx.color_in_mask |= bit32(sindex);

                            if !ctx.front_face_emitted {
                                if ctx.inputs.len() + 1 >= MAX_IO {
                                    return fail(format!(
                                        "Number of inputs exceeded, max is {MAX_IO}"
                                    ));
                                }
                                ctx.inputs.push(Io {
                                    name: Semantic::Face,
                                    sid: 0,
                                    interpolate: Interpolate::Constant,
                                    location: Location::Center,
                                    first: 0,
                                    override_no_wm: false,
                                    glsl_predefined_no_emit: true,
                                    glsl_no_index: true,
                                    ..Io::default()
                                });
                            }
                            add_two_side = true;
                        }
                    }
                }
                Semantic::PrimId => {
                    if processor == Processor::Geometry {
                        name_prefix = "gl_PrimitiveIDIn";
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.inputs[i].override_no_wm = true;
                        ctx.shader_req_bits |= req::INTS;
                    } else if processor == Processor::Fragment {
                        name_prefix = "gl_PrimitiveID";
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.glsl_ver_required = ctx.require_glsl_ver(150);
                        ctx.shader_req_bits |= req::GEOMETRY_SHADER;
                    }
                }
                Semantic::ViewportIndex => {
                    if processor == Processor::Fragment {
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.inputs[i].is_int = true;
                        ctx.inputs[i].ty = VecType::Int;
                        ctx.inputs[i].override_no_wm = true;
                        name_prefix = "gl_ViewportIndex";
                        ctx.shader_req_bits |= req::LAYER;
                        ctx.shader_req_bits |= req::VIEWPORT_IDX;
                    }
                }
                Semantic::Layer => {
                    if processor == Processor::Fragment {
                        name_prefix = "gl_Layer";
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.inputs[i].is_int = true;
                        ctx.inputs[i].ty = VecType::Int;
                        ctx.inputs[i].override_no_wm = true;
                        ctx.shader_req_bits |= req::LAYER;
                    }
                }
                Semantic::PSize => {
                    if matches!(
                        processor,
                        Processor::Geometry | Processor::TessCtrl | Processor::TessEval
                    ) {
                        name_prefix = "gl_PointSize";
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.inputs[i].override_no_wm = true;
                        ctx.inputs[i].glsl_gl_block = true;
                        ctx.shader_req_bits |= req::PSIZE;
                        ctx.has_pointsize_input = true;
                    }
                }
                Semantic::ClipDist => {
                    if matches!(
                        processor,
                        Processor::Geometry | Processor::TessCtrl | Processor::TessEval
                    ) {
                        name_prefix = "gl_ClipDistance";
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.inputs[i].glsl_gl_block = true;
                        ctx.num_in_clip_dist +=
                            4 * (ctx.inputs[i].last as i32 - ctx.inputs[i].first as i32 + 1);
                        ctx.shader_req_bits |= req::CLIP_DISTANCE;
                        if ctx.inputs[i].last != ctx.inputs[i].first {
                            ctx.guest_sent_io_arrays = true;
                        }
                    } else if processor == Processor::Fragment {
                        name_prefix = "gl_ClipDistance";
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.num_in_clip_dist +=
                            4 * (ctx.inputs[i].last as i32 - ctx.inputs[i].first as i32 + 1);
                        ctx.shader_req_bits |= req::CLIP_DISTANCE;
                        if ctx.inputs[i].last != ctx.inputs[i].first {
                            ctx.guest_sent_io_arrays = true;
                        }
                    }
                }
                Semantic::Position => {
                    if matches!(
                        processor,
                        Processor::Geometry | Processor::TessCtrl | Processor::TessEval
                    ) {
                        name_prefix = "gl_Position";
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.inputs[i].glsl_gl_block = true;
                    } else if processor == Processor::Fragment {
                        name_prefix = if ctx.fs_integer_pixel_center {
                            "(gl_FragCoord - vec4(0.5, 0.5, 0.0, 0.0))"
                        } else {
                            "gl_FragCoord"
                        };
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                    }
                }
                Semantic::Face => {
                    if processor == Processor::Fragment {
                        if ctx.front_face_emitted {
                            ctx.inputs.pop();
                            return Ok(());
                        }
                        name_prefix = "gl_FrontFacing";
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.front_face_emitted = true;
                    }
                }
                Semantic::PCoord => {
                    if processor == Processor::Fragment {
                        name_prefix = "vec4(gl_PointCoord.x, mix(1.0 - gl_PointCoord.y, gl_PointCoord.y, clamp(winsys_adjust_y, 0.0, 1.0)), 0.0, 1.0)";
                        ctx.bufs.required_sysval_uniform_decls |= sysval::WINSYS_ADJUST_Y;
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.inputs[i].num_components = 4;
                        ctx.inputs[i].usage_mask = 0xf;
                    }
                }
                Semantic::Patch | Semantic::Generic | Semantic::TexCoord => {
                    if ctx.inputs[i].name == Semantic::Patch && processor == Processor::TessEval {
                        name_prefix = "patch";
                    }
                    let mut coord_replaced = false;
                    if processor == Processor::Fragment
                        && ctx.key.fs.coord_replace & bit32(ctx.inputs[i].sid) != 0
                    {
                        name_prefix = "vec4(gl_PointCoord.x, mix(1.0 - gl_PointCoord.y, gl_PointCoord.y, clamp(winsys_adjust_y, 0.0, 1.0)), 0.0, 1.0)";
                        ctx.bufs.required_sysval_uniform_decls |= sysval::WINSYS_ADJUST_Y;
                        ctx.inputs[i].glsl_predefined_no_emit = true;
                        ctx.inputs[i].glsl_no_index = true;
                        ctx.inputs[i].num_components = 4;
                        ctx.inputs[i].usage_mask = 0xf;
                        coord_replaced = true;
                    }
                    if !coord_replaced
                        && (ctx.inputs[i].first != ctx.inputs[i].last || ctx.inputs[i].array_id > 0)
                    {
                        ctx.guest_sent_io_arrays = true;
                    }
                }
                other => {
                    eprintln!("[virglrs] Unhandled input semantic: {:x}", other as u8);
                }
            }

            let io = &mut ctx.inputs[i];
            if io.glsl_no_index {
                io.glsl_name = name_prefix.to_string();
            } else {
                io.glsl_name = match io.name {
                    Semantic::Fog => {
                        io.usage_mask = 0xf;
                        io.num_components = 4;
                        io.override_no_wm = false;
                        format!("{name_prefix}_f{}", io.sid)
                    }
                    Semantic::Color => format!("{name_prefix}_c{}", io.sid),
                    Semantic::BColor => format!("{name_prefix}_bc{}", io.sid),
                    Semantic::Generic => format!("{name_prefix}_g{}", io.sid),
                    Semantic::Patch => format!("{name_prefix}{}", io.sid),
                    Semantic::TexCoord => format!("{name_prefix}_t{}", io.sid),
                    _ => format!("{name_prefix}_{}", io.first),
                };
            }
            if add_two_side {
                let sid = ctx.inputs[i + 1].sid;
                ctx.inputs[i + 1].glsl_name = format!("{name_prefix}_bc{sid}");
                if !ctx.front_face_emitted {
                    ctx.inputs[i + 2].glsl_name = "gl_FrontFacing".to_string();
                    ctx.front_face_emitted = true;
                }
            }
        }
        File::Output => {
            if ctx.outputs.iter().any(|j| {
                j.name == decl.semantic.name
                    && j.sid == sindex
                    && j.first == first
                    && ((decl.array.is_none() && j.array_id == 0) || j.array_id == array_id)
            }) {
                return Ok(());
            }
            let i = ctx.outputs.len();
            if i + 1 > MAX_IO {
                return fail(format!("Number of outputs exceeded, max is {MAX_IO}"));
            }
            if last as usize >= MAX_IO {
                return fail(format!("Input slot out of range, want {last}, max is {MAX_IO}"));
            }
            if first > last {
                return fail(format!("Wrong range: First ({first}) > Last ({last})"));
            }

            let mut io = Io::default();
            io.name = decl.semantic.name;
            io.sid = sindex;
            io.interpolate = decl.interp.interpolate;
            io.invariant = decl.invariant;
            io.precise = false;
            io.first = first;
            io.last = last;
            io.array_id = array_id;
            io.usage_mask = decl.usage_mask;
            io.num_components = 4;
            io.glsl_predefined_no_emit = false;
            io.glsl_no_index = false;
            io.override_no_wm = io.num_components == 1;
            io.is_int = false;
            io.fbfetch_used = false;
            io.overlapping_array = None;
            ctx.outputs.push(io);

            map_overlapping_io_array(&mut ctx.outputs, i, decl);

            let mut name_prefix = stage_output_name_prefix(processor);
            let mut color_offset: i32 = 0;

            match ctx.outputs[i].name {
                Semantic::Position => {
                    if matches!(
                        processor,
                        Processor::Vertex
                            | Processor::Geometry
                            | Processor::TessCtrl
                            | Processor::TessEval
                    ) {
                        if ctx.outputs[i].first > 0 {
                            eprintln!("[virglrs] Illegal position input");
                        }
                        name_prefix = "gl_Position";
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        if processor == Processor::TessCtrl {
                            ctx.outputs[i].glsl_gl_block = true;
                        }
                    } else if processor == Processor::Fragment {
                        name_prefix = "gl_FragDepth";
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                    }
                }
                Semantic::Stencil => {
                    if processor == Processor::Fragment {
                        name_prefix = "gl_FragStencilRefARB";
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                        ctx.outputs[i].is_int = true;
                        ctx.shader_req_bits |= req::INTS | req::STENCIL_EXPORT;
                    }
                }
                Semantic::ClipDist => {
                    ctx.shader_req_bits |= req::CLIP_DISTANCE;
                    name_prefix = "gl_ClipDistance";
                    ctx.outputs[i].glsl_predefined_no_emit = true;
                    ctx.outputs[i].glsl_no_index = true;
                    ctx.num_out_clip_dist +=
                        4 * (ctx.outputs[i].last as i32 - ctx.outputs[i].first as i32 + 1);
                    if processor == Processor::Vertex && (ctx.key.gs_present || ctx.key.tcs_present)
                    {
                        ctx.glsl_ver_required = ctx.require_glsl_ver(150);
                    }
                    if processor == Processor::TessCtrl {
                        ctx.outputs[i].glsl_gl_block = true;
                    }
                    if ctx.outputs[i].last != ctx.outputs[i].first {
                        ctx.guest_sent_io_arrays = true;
                    }
                }
                Semantic::ClipVertex => {
                    ctx.outputs[i].override_no_wm = true;
                    ctx.outputs[i].invariant = false;
                    if ctx.glsl_ver_required >= 140 {
                        ctx.has_clipvertex = true;
                        name_prefix = stage_output_name_prefix(processor);
                    } else {
                        name_prefix = "gl_ClipVertex";
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                    }
                }
                Semantic::SampleMask => {
                    if processor == Processor::Fragment {
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                        ctx.outputs[i].is_int = true;
                        ctx.shader_req_bits |= req::INTS | req::SAMPLE_SHADING;
                        name_prefix = "gl_SampleMask";
                    }
                }
                Semantic::Color => {
                    if processor == Processor::Fragment {
                        ctx.outputs[i].ty = get_type(
                            u32::from(ctx.key.fs.cbufs_signed_int_bitmask),
                            u32::from(ctx.key.fs.cbufs_unsigned_int_bitmask),
                            ctx.outputs[i].sid,
                        );
                        name_prefix =
                            if ctx.key.fs.logicop_func.is_some() { "fsout_tmp" } else { "fsout" };
                    } else if ctx.glsl_ver_required < 140 {
                        ctx.outputs[i].glsl_no_index = true;
                        if ctx.outputs[i].sid == 0 {
                            name_prefix = "gl_FrontColor";
                        } else if ctx.outputs[i].sid == 1 {
                            name_prefix = "gl_FrontSecondaryColor";
                        }
                    } else {
                        ctx.color_out_mask |= bit32(sindex);
                    }
                    ctx.outputs[i].override_no_wm = false;
                }
                Semantic::BColor => {
                    if ctx.glsl_ver_required < 140 {
                        ctx.outputs[i].glsl_no_index = true;
                        if ctx.outputs[i].sid == 0 {
                            name_prefix = "gl_BackColor";
                        } else if ctx.outputs[i].sid == 1 {
                            name_prefix = "gl_BackSecondaryColor";
                        }
                    } else {
                        ctx.outputs[i].override_no_wm = false;
                        ctx.color_out_mask |= bit32(sindex) << 2;
                    }
                }
                Semantic::PSize => {
                    if matches!(
                        processor,
                        Processor::Vertex
                            | Processor::Geometry
                            | Processor::TessCtrl
                            | Processor::TessEval
                    ) {
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                        ctx.shader_req_bits |= req::PSIZE;
                        name_prefix = "gl_PointSize";
                        ctx.has_pointsize_output = true;
                        if processor == Processor::TessCtrl {
                            ctx.outputs[i].glsl_gl_block = true;
                        }
                    }
                }
                Semantic::Layer => {
                    if processor == Processor::Geometry
                        || (processor == Processor::Vertex && ctx.cfg.has_vs_layer)
                    {
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                        ctx.outputs[i].is_int = true;
                        name_prefix = "gl_Layer";
                        if processor == Processor::Vertex {
                            ctx.shader_req_bits |= req::AMD_VS_LAYER;
                        }
                    }
                }
                Semantic::PrimId => {
                    if processor == Processor::Geometry {
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                        ctx.outputs[i].is_int = true;
                        name_prefix = "gl_PrimitiveID";
                    }
                }
                Semantic::ViewportIndex => {
                    // The vertex-shader leg is desktop only (`!use_gles`).
                    if processor == Processor::Geometry {
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                        ctx.outputs[i].is_int = true;
                        name_prefix = "gl_ViewportIndex";
                        ctx.shader_req_bits |= req::VIEWPORT_IDX;
                        ctx.glsl_ver_required = ctx.require_glsl_ver(140);
                    }
                }
                Semantic::TessOuter => {
                    if processor == Processor::TessCtrl {
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                        name_prefix = "gl_TessLevelOuter";
                    }
                }
                Semantic::TessInner => {
                    if processor == Processor::TessCtrl {
                        ctx.outputs[i].glsl_predefined_no_emit = true;
                        ctx.outputs[i].glsl_no_index = true;
                        ctx.outputs[i].override_no_wm = true;
                        name_prefix = "gl_TessLevelInner";
                    }
                }
                Semantic::Patch | Semantic::Generic | Semantic::TexCoord => {
                    if ctx.outputs[i].name == Semantic::Patch && processor == Processor::TessCtrl {
                        name_prefix = "patch";
                    }
                    if processor == Processor::Vertex && ctx.outputs[i].name == Semantic::Generic {
                        color_offset = -1;
                    }
                    if ctx.outputs[i].first != ctx.outputs[i].last || ctx.outputs[i].array_id > 0 {
                        ctx.guest_sent_io_arrays = true;
                    }
                }
                other => {
                    eprintln!("[virglrs] Unhandled output semantic: {:x}", other as u8);
                }
            }

            let io = &mut ctx.outputs[i];
            if io.glsl_no_index {
                io.glsl_name = name_prefix.to_string();
            } else {
                io.glsl_name = match io.name {
                    Semantic::Fog => {
                        io.usage_mask = 0xf;
                        io.num_components = 4;
                        io.override_no_wm = false;
                        format!("{name_prefix}_f{}", io.sid)
                    }
                    Semantic::Color => format!("{name_prefix}_c{}", io.sid),
                    Semantic::BColor => format!("{name_prefix}_bc{}", io.sid),
                    Semantic::Patch => format!("{name_prefix}{}", io.sid),
                    Semantic::Generic => format!("{name_prefix}_g{}", io.sid),
                    Semantic::TexCoord => format!("{name_prefix}_t{}", io.sid),
                    _ => format!("{name_prefix}_{}", io.first as i32 + color_offset),
                };
            }
        }
        File::Temporary => {
            if first > last {
                return fail(format!("Wrong range: First ({first}) > Last ({last})"));
            }
            allocate_temp_range(ctx, first as i32, last as i32, array_id as i32);
        }
        File::Sampler => {
            ctx.samplers_used |= bit32(last);
        }
        File::SamplerView => {
            if first > last {
                return fail(format!("Wrong range: First ({first}) > Last ({last})"));
            }
            if last as usize >= MAX_SAMPLERS {
                return fail(format!("Sampler view exceeded, max is {MAX_SAMPLERS}"));
            }
            add_samplers(
                ctx,
                first as usize,
                last as usize,
                decl.sampler_view.resource,
                decl.sampler_view.return_type[0],
            );
        }
        File::Image => {
            if first > last {
                return fail(format!("Wrong range: First ({first}) > Last ({last})"));
            }
            ctx.shader_req_bits |= req::IMAGE_LOAD_STORE;
            ctx.shader_req_bits |= req::EXPLICIT_UNIFORM_LOCATION;
            ctx.shader_req_bits |= req::EXPLICIT_ATTRIB_LOCATION;
            if last as usize >= MAX_SHADER_IMAGES {
                return fail(format!("Image view exceeded, max is {MAX_SHADER_IMAGES}"));
            }
            add_images(ctx, first as usize, last as usize, &decl.image);
        }
        File::Buffer => {
            if first + u32::from(ctx.key.ssbo_binding_offset) >= MAX_COMBINED_SSBO_BINDING_POINTS {
                return fail(format!(
                    "Buffer view exceeded, max is {MAX_COMBINED_SSBO_BINDING_POINTS}"
                ));
            }
            ctx.ssbo_used_mask |= bit32(first);
            if decl.atomic {
                if first < ctx.ssbo_atomic_array_base {
                    ctx.ssbo_atomic_array_base = first;
                }
                ctx.ssbo_atomic_mask |= bit32(first);
            } else if first < ctx.ssbo_array_base {
                ctx.ssbo_array_base = first;
            }
            if ctx.ssbo_last_binding < last as i32 {
                ctx.ssbo_last_binding = last as i32;
            }
            ctx.glsl_ver_required = ctx.require_glsl_ver(140);
        }
        File::Constant => {
            if let Some(index2d) = decl.dimension.filter(|&d| d != 0) {
                if index2d > 31 {
                    return fail("Number of uniforms exceeded, max is 32".to_string());
                }
                if ctx.ubo_used_mask & (1 << index2d) != 0 {
                    return fail(format!("UBO #{index2d} is already defined"));
                }
                ctx.ubo_used_mask |= 1 << index2d;
                ctx.ubo_sizes[index2d as usize] = last as i32 + 1;
            } else {
                // A plain constant set puts the UBO base at 1.
                ctx.ubo_base = 1;
                if last != 0 {
                    if last as i32 + 1 > ctx.num_consts {
                        ctx.num_consts = last as i32 + 1;
                    }
                } else {
                    ctx.num_consts += 1;
                }
            }
        }
        File::Address => {
            ctx.num_address = last + 1;
        }
        File::SystemValue => {
            if ctx.system_values.len() + 1 > MAX_SYSTEM_VALUES {
                return fail(format!(
                    "Number of system values exceeded, max is {MAX_SYSTEM_VALUES}"
                ));
            }
            let Some((glsl_name, required_ext, override_no_wm)) = sysvalue_map(decl.semantic.name)
            else {
                return fail(format!("Unsupported system value {}", decl.semantic.name as u8));
            };
            ctx.shader_req_bits |= required_ext;
            ctx.system_values.push(Io {
                name: decl.semantic.name,
                sid: sindex,
                glsl_predefined_no_emit: true,
                glsl_no_index: true,
                first,
                override_no_wm,
                glsl_name: glsl_name.to_string(),
                ..Io::default()
            });
            if decl.semantic.name == Semantic::DrawId {
                ctx.bufs.required_sysval_uniform_decls |= sysval::DRAWID_BASE;
            }
        }
        File::Memory => {
            ctx.has_file_memory = true;
        }
        File::HwAtomic => {
            if first > last {
                return fail(format!("Wrong range: First ({first}) > Last ({last})"));
            }
            if ctx.abo_idx.len() >= 32 {
                return fail("Number of atomic counter buffers exceeded, max is 32".to_string());
            }
            ctx.abo_idx.push(decl.dimension.map_or(0, i32::from));
            ctx.abo_sizes.push(last as i32 - first as i32 + 1);
            ctx.abo_offsets.push(first as i32);
            ctx.glsl_ver_required = ctx.require_glsl_ver(140);
        }
        other => {
            eprintln!("[virglrs] Unsupported file {} declaration", other as u8);
        }
    }
    Ok(())
}

/// `iter_property`.
pub(super) fn iter_property(ctx: &mut Context<'_>, prop: &PropertyToken) -> Result<(), Failure> {
    let data = prop.data;
    match prop.name {
        Property::FsColor0WritesAllCbufs => {
            if data == 1 {
                ctx.write_all_cbufs = true;
            }
        }
        Property::FsCoordOrigin => ctx.fs_lower_left_origin = data != 0,
        Property::FsCoordPixelCenter => ctx.fs_integer_pixel_center = data != 0,
        Property::FsDepthLayout => {
            // Without host support this is only a lost optimisation.
            if ctx.cfg.has_conservative_depth {
                ctx.shader_req_bits |= req::CONSERVATIVE_DEPTH;
                ctx.fs_depth_layout = data;
            }
        }
        Property::GsInputPrim => ctx.gs_in_prim = data,
        Property::GsOutputPrim => ctx.gs_out_prim = data,
        Property::GsMaxOutputVertices => ctx.gs_max_out_verts = data,
        Property::GsInvocations => {
            ctx.gs_num_invocations = data;
            ctx.shader_req_bits |= req::GPU_SHADER5;
        }
        Property::NumClipdistEnabled => {
            ctx.shader_req_bits |= req::CLIP_DISTANCE;
            ctx.num_clip_dist_prop = data as u8;
        }
        Property::NumCulldistEnabled => ctx.num_cull_dist_prop = data as u8,
        Property::TcsVerticesOut => ctx.tcs_vertices_out = data,
        Property::TesPrimMode => ctx.tes_prim_mode = data,
        Property::TesSpacing => ctx.tes_spacing = data,
        Property::TesVertexOrderCw => ctx.tes_vertex_order = data,
        Property::TesPointMode => ctx.tes_point_mode = data,
        Property::FsEarlyDepthStencil => {
            ctx.early_depth_stencil = data > 0;
            if ctx.early_depth_stencil {
                ctx.glsl_ver_required = ctx.require_glsl_ver(150);
                ctx.shader_req_bits |= req::IMAGE_LOAD_STORE;
            }
        }
        Property::CsFixedBlockWidth => ctx.local_cs_block_size[0] = data as u16,
        Property::CsFixedBlockHeight => ctx.local_cs_block_size[1] = data as u16,
        Property::CsFixedBlockDepth => ctx.local_cs_block_size[2] = data as u16,
        Property::FsBlendEquationAdvanced => {
            ctx.fs_blend_equation_advanced = data;
            if ctx.cfg.glsl_version < 320 {
                ctx.glsl_ver_required = ctx.require_glsl_ver(150);
                ctx.shader_req_bits |= req::BLEND_EQUATION_ADVANCED;
            }
        }
        Property::SeparableProgram => {
            // GLES is strict about how separable interfaces match -- it refuses, for one,
            // an input without a matching output -- so separable programs stay off there.
        }
        other => {
            return fail(format!("Unhandled property: {:x}", other as u8));
        }
    }
    Ok(())
}

/// `iter_immediate`.
pub(super) fn iter_immediate(ctx: &mut Context<'_>, imm: &Immediate) -> Result<(), Failure> {
    if ctx.imm.len() >= MAX_IMMEDIATE {
        return fail(format!("Number of immediates exceeded, max is: {MAX_IMMEDIATE}"));
    }
    let mut val = [0u32; 4];
    for (i, v) in val.iter_mut().enumerate() {
        match imm.ty {
            ImmType::Float32 => *v = imm.data[i],
            ImmType::Uint32 | ImmType::Float64 => {
                ctx.shader_req_bits |= req::INTS;
                *v = imm.data[i];
            }
            ImmType::Int32 => {
                ctx.shader_req_bits |= req::INTS;
                *v = imm.data[i];
            }
            other => {
                eprintln!("[virglrs] Unhandled immediate type, ignoring: {:x}", other as u8);
            }
        }
    }
    ctx.imm.push(Immed { ty: imm.ty, val });
    Ok(())
}
