// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Draws: the program a draw runs, linked from the variants selected for the bound stages, and
//! everything the C binds around it before the primitives go out (`vrend_draw_vbo`).
//!
//! The order is the C's, because pixels come out of the order GL sees things in: the lazy state
//! first, then the program, then per stage the uniform buffers, the constants, the samplers, the
//! images and the storage buffers, then the vertex layout and the index buffer, and only then
//! the draw call the wire's `pipe_draw_info` names. What the C skips on a clean dirty flag is
//! skipped here on the same flag, since a re-bind is a GL call the C did not make.

use super::*;

/// `sysval_uniform_block`: what the shaders read through the `VirglBlock` uniform block, laid
/// out as std140 lays it out -- every array element on sixteen bytes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Sysval {
    pub clip_planes: [[f32; 4]; shader::NUM_CLIP_PLANES],
    pub stipple: [u32; shader::POLYGON_STIPPLE_SIZE],
    pub winsys_adjust_y: f32,
    pub alpha_ref_val: f32,
    pub clip_plane_enabled: f32,
    pub drawid_base: i32,
}

impl Default for Sysval {
    /// The C's zeroed block, with `winsys_adjust_y` at one as the sub-context sets it.
    fn default() -> Sysval {
        Sysval {
            clip_planes: [[0.0; 4]; shader::NUM_CLIP_PLANES],
            stipple: [0; shader::POLYGON_STIPPLE_SIZE],
            winsys_adjust_y: 1.0,
            alpha_ref_val: 0.0,
            clip_plane_enabled: 0.0,
            drawid_base: 0,
        }
    }
}

impl Sysval {
    /// The block's bytes, in the layout the shader declares.
    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Sysval::SIZE);
        for p in &self.clip_planes {
            for c in p {
                out.extend_from_slice(&c.to_ne_bytes());
            }
        }
        for s in &self.stipple {
            out.extend_from_slice(&s.to_ne_bytes());
            out.extend_from_slice(&[0; 12]);
        }
        out.extend_from_slice(&self.winsys_adjust_y.to_ne_bytes());
        out.extend_from_slice(&self.alpha_ref_val.to_ne_bytes());
        out.extend_from_slice(&self.clip_plane_enabled.to_ne_bytes());
        out.extend_from_slice(&self.drawid_base.to_ne_bytes());
        debug_assert_eq!(out.len(), Sysval::SIZE);
        out
    }

    const SIZE: usize = shader::NUM_CLIP_PLANES * 16 + shader::POLYGON_STIPPLE_SIZE * 16 + 16;
}

/// What `vrend_hw_emit_blend` last told GL, for what it only emits on change.
#[derive(Clone, Copy, Debug)]
pub struct HwBlend {
    pub logicop_enable: bool,
    pub logicop_func: LogicOp,
    pub independent: bool,
    /// `hw_blend_state.rt[i].colormask`: what a clear restores.
    pub colormask: [u8; MAX_COLOR_BUFS],
}

impl Default for HwBlend {
    fn default() -> HwBlend {
        HwBlend {
            logicop_enable: false,
            logicop_func: LogicOp::Clear,
            independent: false,
            colormask: [0xf; MAX_COLOR_BUFS],
        }
    }
}

/// Where a transform feedback object stands, as the draws move it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Xfb {
    /// Bound, not yet begun: the first draw begins it.
    NeedBegin,
    Started,
    Paused,
}

/// `vrend_linked_shader_program`: the variants of the bound stages linked into one GL program,
/// and every location the draw path writes through.
pub struct LinkedProgram {
    /// Names this program in the sub-context's list; stable while the program lives.
    pub serial: ProgramSerial,
    pub id: ProgramName,
    /// The variant of each stage, in stage order, compute excluded.
    pub stages: [Option<VariantId>; 5],
    pub dual_src_linked: bool,
    pub last_stage: ShaderStage,
    pub ubo_used_mask: [u32; ShaderStage::COUNT],
    pub samplers_used_mask: [u32; ShaderStage::COUNT],
    pub shadow_samp_mask: [u32; ShaderStage::COUNT],
    /// One location per set bit of `samplers_used_mask`, in bit order.
    pub sampler_locs: [Vec<Option<UniformLocation>>; ShaderStage::COUNT],
    pub shadow_samp_mask_locs: [Vec<Option<UniformLocation>>; ShaderStage::COUNT],
    pub shadow_samp_add_locs: [Vec<Option<UniformLocation>>; ShaderStage::COUNT],
    pub const_location: [Option<UniformLocation>; ShaderStage::COUNT],
    pub num_consts: [usize; ShaderStage::COUNT],
    pub ssbo_used_mask: [u32; ShaderStage::COUNT],
    pub ssbo_binding_offset: [u32; ShaderStage::COUNT],
    pub images_used_mask: [u32; ShaderStage::COUNT],
    /// One location per image slot up to the highest used; `None` where the compiler dropped
    /// the image and the draw has nothing to write.
    pub img_locs: [Vec<Option<UniformLocation>>; ShaderStage::COUNT],
    pub image_binding_offset: [u32; ShaderStage::COUNT],
    pub tex_levels_uniform_id: [Option<UniformLocation>; ShaderStage::COUNT],
    /// The `VirglBlock` binding and the buffer behind it, once a stage declared the block.
    pub virgl_block_bind: Option<BindingPoint>,
    pub sysval_buffer: Option<BufferName>,
    /// The block the buffer holds, so a draw uploads only a block that changed.
    pub sysval_uploaded: Option<Sysval>,
    pub reads_drawid: bool,
    pub fs_blend_equation_advanced: u32,
}

impl LinkedProgram {
    /// Whether this program links `variant`.
    pub fn links(&self, variant: VariantId) -> bool {
        self.stages.contains(&Some(variant))
    }
}

/// The vertex buffer slots a bind must clear: the ones the hardware still holds past the set
/// the guest has now. `hw` is what the last bind left bound -- not what the last state-set
/// replaced, which is a different number the moment the guest sets twice before drawing.
fn stale_vbo_slots(bound: usize, hw: usize) -> std::ops::Range<usize> {
    bound..hw.max(bound)
}

fn stage_prefix(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vs",
        ShaderStage::Fragment => "fs",
        ShaderStage::Geometry => "gs",
        ShaderStage::TessCtrl => "tc",
        ShaderStage::TessEval => "te",
        ShaderStage::Compute => "cs",
    }
}

const GRAPHICS_STAGES: [ShaderStage; 5] = [
    ShaderStage::Vertex,
    ShaderStage::TessCtrl,
    ShaderStage::TessEval,
    ShaderStage::Geometry,
    ShaderStage::Fragment,
];

/// Stage order as the C walks it: vertex, fragment, geometry, control, evaluation.
const C_STAGE_ORDER: [ShaderStage; 5] = [
    ShaderStage::Vertex,
    ShaderStage::Fragment,
    ShaderStage::Geometry,
    ShaderStage::TessCtrl,
    ShaderStage::TessEval,
];

fn blend_func(f: BlendFunc) -> GLenum {
    match f {
        BlendFunc::Add => GL_FUNC_ADD,
        BlendFunc::Subtract => GL_FUNC_SUBTRACT,
        BlendFunc::ReverseSubtract => GL_FUNC_REVERSE_SUBTRACT,
        BlendFunc::Min => GL_MIN,
        BlendFunc::Max => GL_MAX,
    }
}

fn blend_factor(f: BlendFactor) -> GLenum {
    use BlendFactor::*;
    match f {
        One => GL_ONE,
        SrcColor => GL_SRC_COLOR,
        SrcAlpha => GL_SRC_ALPHA,
        DstColor => GL_DST_COLOR,
        DstAlpha => GL_DST_ALPHA,
        ConstColor => GL_CONSTANT_COLOR,
        ConstAlpha => GL_CONSTANT_ALPHA,
        Src1Color => GL_SRC1_COLOR_EXT,
        Src1Alpha => GL_SRC1_ALPHA_EXT,
        SrcAlphaSaturate => GL_SRC_ALPHA_SATURATE,
        Zero => GL_ZERO,
        InvSrcColor => GL_ONE_MINUS_SRC_COLOR,
        InvSrcAlpha => GL_ONE_MINUS_SRC_ALPHA,
        InvDstColor => GL_ONE_MINUS_DST_COLOR,
        InvDstAlpha => GL_ONE_MINUS_DST_ALPHA,
        InvConstColor => GL_ONE_MINUS_CONSTANT_COLOR,
        InvConstAlpha => GL_ONE_MINUS_CONSTANT_ALPHA,
        InvSrc1Color => GL_ONE_MINUS_SRC1_COLOR_EXT,
        InvSrc1Alpha => GL_ONE_MINUS_SRC1_ALPHA_EXT,
    }
}

fn is_dst_blend(f: BlendFactor) -> bool {
    matches!(f, BlendFactor::DstAlpha | BlendFactor::InvDstAlpha)
}

fn conv_dst_blend(f: BlendFactor) -> BlendFactor {
    match f {
        BlendFactor::DstAlpha => BlendFactor::One,
        BlendFactor::InvDstAlpha => BlendFactor::Zero,
        f => f,
    }
}

/// `util_blend_state_is_dual`: whether target `i` reads a second colour output.
fn blend_is_dual(state: &BlendState, i: usize) -> bool {
    let Some(eq) = state.rt[i].equation else {
        return false;
    };
    use BlendFactor::*;
    let dual = |f| matches!(f, Src1Color | Src1Alpha | InvSrc1Color | InvSrc1Alpha);
    dual(eq.rgb.src) || dual(eq.rgb.dst) || dual(eq.alpha.src) || dual(eq.alpha.dst)
}

/// `get_xfb_mode`.
fn xfb_mode(mode: PrimType) -> GLenum {
    match mode {
        PrimType::Points => GL_POINTS,
        PrimType::Lines
        | PrimType::LineLoop
        | PrimType::LineStrip
        | PrimType::LinesAdjacency
        | PrimType::LineStripAdjacency => GL_LINES,
        _ => GL_TRIANGLES,
    }
}

/// `get_gs_xfb_mode`.
fn gs_xfb_mode(prim: Option<PrimType>) -> GLenum {
    match prim {
        Some(PrimType::Points) => GL_POINTS,
        Some(PrimType::LineStrip) => GL_LINES,
        Some(PrimType::TriangleStrip) => GL_TRIANGLES,
        _ => GL_POINTS,
    }
}

/// `get_tess_xfb_mode`.
fn tess_xfb_mode(prim: Option<PrimType>, point_mode: bool) -> GLenum {
    if point_mode {
        return GL_POINTS;
    }
    match prim {
        Some(PrimType::Quads) | Some(PrimType::Triangles) => GL_TRIANGLES,
        Some(PrimType::Lines) => GL_LINES,
        _ => GL_POINTS,
    }
}

/// Gallium numbers its primitives as GL numbers its modes, and the C passes the one as the
/// other; the quad strip and polygon the desktop enums name are among them.
fn prim_mode(mode: PrimType) -> GLenum {
    debug_assert_eq!(PrimType::Patches.wire(), GL_PATCHES);
    mode.wire() as GLenum
}

/// `get_skip_str`: the `gl_SkipComponents` varying that covers up to four of `skip`.
fn skip_varying(skip: &mut i32) -> Option<&'static str> {
    if *skip < 0 {
        *skip = 0;
        return None;
    }
    let (name, n) = match *skip {
        0 => return None,
        1 => ("gl_SkipComponents1", 1),
        2 => ("gl_SkipComponents2", 2),
        3 => ("gl_SkipComponents3", 3),
        _ => ("gl_SkipComponents4", 4),
    };
    *skip -= n;
    Some(name)
}

/// `set_stream_out_varyings`: the interleaved varying list a stream-output layout asks of GL.
fn set_stream_out_varyings(gl: &Gl, program: ProgramName, info: &shader::Info) {
    let so = &info.so_info;
    if so.outputs.is_empty() {
        return;
    }
    const MAX: usize = shader::MAX_SO_OUTPUTS * 2;
    let mut varyings: Vec<String> = Vec::new();
    let mut last_buffer = 0usize;
    let mut buf_offset = 0i32;
    for (i, out) in so.outputs.iter().enumerate() {
        let buffer = out.output_buffer.index();
        if last_buffer != buffer {
            let mut skip = so.stride[last_buffer] as i32 - buf_offset;
            while skip != 0 && varyings.len() < MAX {
                if let Some(s) = skip_varying(&mut skip) {
                    varyings.push(s.to_string());
                }
            }
            let mut j = last_buffer;
            while j < buffer && varyings.len() < MAX {
                varyings.push("gl_NextBuffer".to_string());
                j += 1;
            }
            last_buffer = buffer;
            buf_offset = 0;
        }
        let mut skip = out.dst_offset as i32 - buf_offset;
        while skip != 0 && varyings.len() < MAX {
            if let Some(s) = skip_varying(&mut skip) {
                varyings.push(s.to_string());
            }
        }
        buf_offset = out.dst_offset as i32 + out.num_components as i32;
        if let Some(name) = info.so_names.get(i)
            && !name.is_empty()
            && varyings.len() < MAX
        {
            varyings.push(name.clone());
        }
    }
    let mut skip = so.stride[last_buffer] as i32 - buf_offset;
    while skip != 0 && varyings.len() < MAX {
        if let Some(s) = skip_varying(&mut skip) {
            varyings.push(s.to_string());
        }
    }
    gl.transform_feedback_varyings(program, &varyings);
}

/// The variant a bound stage's program currently selects, with the program's info.
struct Linked<'a> {
    stage: ShaderStage,
    info: &'a shader::Info,
    variant: &'a Variant,
    gl: ShaderName,
}

/// A linked program's name for as long as its sub-context lives: minted once and never reused,
/// so a sub-context pointing at one cannot come to mean a later program.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ProgramSerial(u64);

impl SubContext {
    /// The next name, taken through a `Cell` so minting one does not need the whole
    /// sub-context: the program being named is built from a borrow of it.
    fn mint_program_serial(&self) -> ProgramSerial {
        let serial = self.next_program_serial.get();
        self.next_program_serial.set(serial + 1);
        ProgramSerial(serial)
    }

    fn program(&self) -> Option<&LinkedProgram> {
        let serial = self.prog?;
        self.programs.iter().find(|p| p.serial == serial)
    }

    fn program_mut(&mut self) -> Option<&mut LinkedProgram> {
        let serial = self.prog?;
        self.programs.iter_mut().find(|p| p.serial == serial)
    }

    /// The current variant of each bound graphics stage, compiled -- `None` for a stage that
    /// is bound but has no compiled variant, which is the C's `!current` failure.
    fn linked_stages(&self) -> Result<Vec<Linked<'_>>, ShaderStage> {
        let mut out = Vec::new();
        for stage in GRAPHICS_STAGES {
            if self.shaders[stage.index()].is_none() {
                continue;
            }
            let Some(program) = self.bound_program(stage) else {
                return Err(stage);
            };
            let Some(variant) = program.variants.first() else {
                return Err(stage);
            };
            let Some(gl) = variant.gl else {
                return Err(stage);
            };
            out.push(Linked { stage, info: &program.info, variant, gl });
        }
        Ok(out)
    }

    /// `vrend_destroy_program` for every program linking `variant`, as the C destroys them
    /// with the variant.
    fn forget_programs_of(&mut self, gl: &Gl, variant: VariantId) {
        let mut i = 0;
        while i < self.programs.len() {
            if self.programs[i].links(variant) {
                let p = self.programs.remove(i);
                if self.prog == Some(p.serial) {
                    self.prog = None;
                }
                if let Some(b) = p.sysval_buffer {
                    gl.delete_buffer(b);
                }
                gl.delete_program(p.id);
            } else {
                i += 1;
            }
        }
    }
}

/// The C's `vrend_shader_destroy`, and the one place a shader leaves a sub-context: the
/// programs linking each variant, then the variant's GL shader.
pub(super) fn release_shader(sub: &mut SubContext, gl: &Gl, shader: Shader) {
    if let ShaderText::Whole(p) = shader.text {
        for v in p.variants {
            sub.forget_programs_of(gl, v.id);
            if let Some(name) = v.gl {
                gl.delete_shader(name);
            }
        }
    }
}

/// `add_shader_program`, the non-separable leg: the bound stages' variants linked into one
/// program, with every location the draw path will write through looked up once.
fn add_shader_program(
    host: &mut Host<'_>,
    cmd: Cmd,
    serial: ProgramSerial,
    linked: &[Linked<'_>],
    dual_src: bool,
) -> Result<LinkedProgram, Fault> {
    let gl = host.gl;
    let features = host.features;
    let Some(id) = gl.create_program() else {
        return Err(Fault::Shader { cmd, what: "the driver refused a program object" });
    };
    let mut stages = [None; 5];
    for l in linked {
        gl.attach_shader(id, l.gl);
        stages[l.stage.index()] = Some(l.variant.id);
    }
    let by_stage = |s: ShaderStage| linked.iter().find(|l| l.stage == s);
    let vs = by_stage(ShaderStage::Vertex).expect("a program has a vertex stage");
    let fs = by_stage(ShaderStage::Fragment).expect("a program has a fragment stage");
    let gs = by_stage(ShaderStage::Geometry);
    let tes = by_stage(ShaderStage::TessEval);

    // The stream-out layout belongs to the last stage before the rasterizer.
    set_stream_out_varyings(gl, id, gs.or(tes).unwrap_or(vs).info);

    let mut dual_src_linked = false;
    if fs.info.num_outputs > 1 {
        dual_src_linked = dual_src;
        if dual_src_linked {
            if features.has(Feature::dual_src_blend) {
                gl.bind_frag_data_location_indexed(id, 0, 0, "fsout_c0");
                gl.bind_frag_data_location_indexed(id, 0, 1, "fsout_c1");
            } else {
                return Err(Fault::Shader { cmd, what: "dual-source blending the host lacks" });
            }
        }
        // Without dual-source blending the GLES shader carries its output layout itself.
    }

    if features.has(Feature::gles31_vertex_attrib_binding) {
        let mut mask = vs.info.attrib_input_mask;
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            gl.bind_attrib_location(id, i, &format!("in_{i}"));
        }
    }

    if let Err(log) = gl.link_program(id) {
        gl.delete_program(id);
        eprintln!("[virglrs] vrend: error linking program:\n{log}");
        for l in linked {
            eprintln!("{}: GLSL:\n{}", stage_prefix(l.stage), l.variant.strings.source());
        }
        return Err(Fault::Shader { cmd, what: "a program the driver refused to link" });
    }

    let last_stage = if tes.is_some() {
        ShaderStage::TessEval
    } else if gs.is_some() {
        ShaderStage::Geometry
    } else {
        ShaderStage::Fragment
    };
    let mut prog = LinkedProgram {
        serial,
        id,
        stages,
        dual_src_linked,
        last_stage,
        ubo_used_mask: [0; ShaderStage::COUNT],
        samplers_used_mask: [0; ShaderStage::COUNT],
        shadow_samp_mask: [0; ShaderStage::COUNT],
        sampler_locs: Default::default(),
        shadow_samp_mask_locs: Default::default(),
        shadow_samp_add_locs: Default::default(),
        const_location: [None; ShaderStage::COUNT],
        num_consts: [0; ShaderStage::COUNT],
        ssbo_used_mask: [0; ShaderStage::COUNT],
        ssbo_binding_offset: [0; ShaderStage::COUNT],
        images_used_mask: [0; ShaderStage::COUNT],
        img_locs: Default::default(),
        image_binding_offset: [0; ShaderStage::COUNT],
        tex_levels_uniform_id: [None; ShaderStage::COUNT],
        virgl_block_bind: None,
        sysval_buffer: None,
        sysval_uploaded: None,
        reads_drawid: false,
        fs_blend_equation_advanced: fs.info.fs_blend_equation_advanced,
    };

    gl.use_program(Some(id));

    // Stage order as the C walks it, vertex through the last stage.
    let walk: Vec<&Linked<'_>> = C_STAGE_ORDER
        .iter()
        .filter(|s| s.index() <= last_stage.index())
        .filter_map(|s| by_stage(*s))
        .collect();
    for l in &walk {
        let s = l.stage.index();
        let prefix = stage_prefix(l.stage);
        // bind_const_locs
        if l.info.num_consts > 0 {
            prog.const_location[s] = gl.get_uniform_location(id, &format!("{prefix}const0"));
            prog.num_consts[s] = l.info.num_consts as usize;
        }
        // bind_image_locs
        let mask = l.info.images_used_mask;
        if (mask != 0 || !l.info.image_arrays.is_empty()) && features.has(Feature::images) {
            let nsamp = (32 - mask.leading_zeros()) as usize;
            let mut locs = vec![None; nsamp];
            if !l.info.image_arrays.is_empty() {
                for arr in &l.info.image_arrays {
                    for j in 0..arr.array_size {
                        let name = format!("{prefix}img{}[{j}]", arr.first);
                        // An image the compiler dropped has no location; the draw skips it.
                        let loc = gl.get_uniform_location(id, &name);
                        let slot = (arr.first + j) as usize;
                        if slot >= locs.len() {
                            locs.resize(slot + 1, None);
                        }
                        locs[slot] = loc;
                    }
                }
            } else {
                for (i, loc) in locs.iter_mut().enumerate() {
                    if mask & (1 << i) != 0 {
                        let name = format!("{prefix}img{i}");
                        *loc = gl.get_uniform_location(id, &name);
                    }
                }
            }
            prog.img_locs[s] = locs;
            prog.images_used_mask[s] = mask;
            prog.image_binding_offset[s] = l.info.image_binding_offset;
        }
        // bind_ssbo_locs
        if features.has(Feature::ssbo) {
            prog.ssbo_used_mask[s] = l.info.ssbo_used_mask;
            prog.ssbo_binding_offset[s] = l.info.ssbo_binding_offset;
        }
        if l.info.reads_drawid {
            prog.reads_drawid = true;
        }
    }

    // rebind_ubo_and_sampler_locs
    let mut next_ubo_id = BindingPoint::FIRST;
    for l in &walk {
        let s = l.stage.index();
        let prefix = stage_prefix(l.stage);
        // bind_sampler_locs
        let mut mask = l.info.samplers_used_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as i32;
            mask &= mask - 1;
            let name = if !l.info.sampler_arrays.is_empty() {
                let first = l.info.lookup_sampler_array(i);
                format!("{prefix}samp{first}[{}]", i - first)
            } else {
                format!("{prefix}samp{i}")
            };
            prog.sampler_locs[s].push(gl.get_uniform_location(id, &name));
            let (mask_loc, add_loc) = if l.info.shadow_samp_mask & (1 << i) != 0 {
                (
                    gl.get_uniform_location(id, &format!("{prefix}shadmask{i}")),
                    gl.get_uniform_location(id, &format!("{prefix}shadadd{i}")),
                )
            } else {
                (None, None)
            };
            prog.shadow_samp_mask_locs[s].push(mask_loc);
            prog.shadow_samp_add_locs[s].push(add_loc);
        }
        prog.samplers_used_mask[s] = l.info.samplers_used_mask;
        prog.shadow_samp_mask[s] = l.info.shadow_samp_mask;
        // bind_ubo_locs
        let mut mask = l.info.ubo_used_mask;
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            let name = if l.info.ubo_indirect {
                format!("{prefix}ubo[{}]", i.wrapping_sub(1) as i32)
            } else {
                format!("{prefix}ubo{i}")
            };
            if let Some(block) = gl.get_uniform_block_index(id, &name) {
                gl.uniform_block_binding(id, block, next_ubo_id);
            }
            next_ubo_id = next_ubo_id.next();
        }
        prog.ubo_used_mask[s] = l.info.ubo_used_mask;
    }
    // bind_virgl_block_loc: the block binds after the last UBO.
    for _ in &walk {
        let Some(block) = gl.get_uniform_block_index(id, "VirglBlock") else {
            continue;
        };
        let mut created = false;
        if prog.virgl_block_bind.is_none() {
            prog.virgl_block_bind = Some(next_ubo_id);
            if prog.sysval_buffer.is_none() {
                prog.sysval_buffer = Some(gl.gen_buffer());
                created = true;
            }
        }
        let bind = prog.virgl_block_bind.expect("set a moment ago");
        gl.uniform_block_binding(id, block, bind);
        let size = gl.uniform_block_data_size(id, block);
        assert!(
            size as usize >= Sysval::SIZE,
            "the VirglBlock the shader declares holds the sysval block"
        );
        if created {
            let buf = prog.sysval_buffer.expect("made a moment ago");
            gl.bind_buffer(GL_UNIFORM_BUFFER, Some(buf));
            gl.buffer_data_null(GL_UNIFORM_BUFFER, size as usize, GL_DYNAMIC_DRAW);
            gl.bind_buffer(GL_UNIFORM_BUFFER, None);
        }
    }

    // The texture-level uniforms the GLES `textureQueryLevels` emulation reads, looked up
    // once here where the C looks them up on every draw.
    for l in &walk {
        if l.info.gles_use_tex_query_level {
            let name = format!("{}_texlod", stage_prefix(l.stage));
            prog.tex_levels_uniform_id[l.stage.index()] = gl.get_uniform_location(id, &name);
        }
    }

    Ok(prog)
}

impl Context {
    /// `vrend_select_program`, the program half: the variants selected and compiled by
    /// [`Context::select_program`], then the program that links them found or made. Answers
    /// whether the sub-context's program changed.
    pub(super) fn select_linked_program(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
    ) -> Result<bool, Fault> {
        self.select_program(host, cmd)?;
        let sub = self.sub();
        let dual_src = sub.blend.as_ref().is_some_and(|b| blend_is_dual(b, 0));
        let linked = match sub.linked_stages() {
            Ok(l) => l,
            Err(_) => return Err(Fault::Shader { cmd, what: "a stage with no compiled variant" }),
        };
        let fs = linked
            .iter()
            .find(|l| l.stage == ShaderStage::Fragment)
            .expect("select_program required a fragment stage");
        let dual_src = dual_src && fs.info.num_outputs > 1;
        let mut ids = [None; 5];
        for l in &linked {
            ids[l.stage.index()] = Some(l.variant.id);
        }
        let same = sub.program().is_some_and(|p| p.stages == ids && p.dual_src_linked == dual_src);
        if same {
            // The selection is settled either way; a flag left standing here would run the
            // nine key passes again on every draw of this program.
            self.sub_mut().shader_dirty = false;
            return Ok(false);
        }
        let found = sub
            .programs
            .iter()
            .find(|p| p.stages == ids && p.dual_src_linked == dual_src)
            .map(|p| p.serial);
        let serial = match found {
            Some(s) => s,
            None => {
                let serial = sub.mint_program_serial();
                let prog = add_shader_program(host, cmd, serial, &linked, dual_src)?;
                self.sub_mut().programs.push(prog);
                serial
            }
        };
        let sub = self.sub_mut();
        let changed = sub.prog != Some(serial);
        sub.prog = Some(serial);
        if changed {
            // Every constant buffer and view is re-bound for a new program.
            for stage in [ShaderStage::Vertex, ShaderStage::Fragment] {
                sub.ubos_dirty[stage.index()] = Dirty::all();
                sub.views_dirty[stage.index()] = Dirty::all();
            }
        }
        sub.shader_dirty = false;
        Ok(changed)
    }

    /// `vrend_patch_blend_state` and `vrend_hw_emit_blend`: the blend state as bound, patched
    /// for targets without alpha, told to GL.
    pub(super) fn patch_blend_state(&mut self, host: &mut Host<'_>) {
        let gl = host.gl;
        let features = host.features;
        let sub = self.sub_mut();
        if sub.cbufs.iter().all(Option::is_none) {
            sub.blend_dirty = false;
            return;
        }
        let state = sub.blend.unwrap_or(ZERO_BLEND);
        let mut new_state = state;
        let targets = if state.independent_blend_enable { MAX_COLOR_BUFS } else { 1 };
        for i in 0..targets {
            let Some(Some(surf)) = sub.cbufs.get(i) else {
                continue;
            };
            // Emulated alpha is a desktop-only case; on GLES only an alpha-less target patches.
            let has_alpha = surf.format.describe().is_some_and(|d| d.has_alpha());
            if has_alpha {
                continue;
            }
            let Some(eq) = state.rt[i].equation else {
                continue;
            };
            let dst = [eq.rgb.src, eq.rgb.dst, eq.alpha.src, eq.alpha.dst];
            if !dst.iter().any(|f| is_dst_blend(*f)) {
                continue;
            }
            new_state.rt[i].equation = Some(RtBlendEq {
                rgb: BlendEq {
                    func: eq.rgb.func,
                    src: conv_dst_blend(eq.rgb.src),
                    dst: conv_dst_blend(eq.rgb.dst),
                },
                alpha: BlendEq {
                    func: eq.alpha.func,
                    src: conv_dst_blend(eq.alpha.src),
                    dst: conv_dst_blend(eq.alpha.dst),
                },
            });
        }

        // vrend_hw_emit_blend
        let mut logicop_changed = false;
        if new_state.logicop_enable != sub.hw_blend.logicop_enable {
            sub.hw_blend.logicop_enable = new_state.logicop_enable;
            logicop_changed = true;
        }
        if new_state.logicop_enable && new_state.logicop_func != sub.hw_blend.logicop_func {
            sub.hw_blend.logicop_func = new_state.logicop_func;
            logicop_changed = true;
        }
        if logicop_changed {
            // GLES has no logic op: the shader does it when it can.
            if select::can_emulate_logicop(features, new_state.logicop_func) {
                sub.shader_dirty = true;
            } else {
                host.todo.note("a logic op the shader cannot emulate");
            }
        }
        let mask = |m: u8| [m & 1 != 0, m & 2 != 0, m & 4 != 0, m & 8 != 0];
        if new_state.independent_blend_enable
            && features.has(Feature::indep_blend)
            && features.has(Feature::indep_blend_func)
        {
            for i in 0..MAX_COLOR_BUFS {
                let rt = new_state.rt[i];
                if let Some(eq) = rt.equation {
                    if blend_is_dual(&state, i) && !features.has(Feature::dual_src_blend) {
                        eprintln!(
                            "[virglrs] vrend: dual src blend requested but not supported for rt {i}"
                        );
                        continue;
                    }
                    gl.blend_func_separate_i(
                        i as GLuint,
                        blend_factor(eq.rgb.src),
                        blend_factor(eq.rgb.dst),
                        blend_factor(eq.alpha.src),
                        blend_factor(eq.alpha.dst),
                    );
                    gl.blend_equation_separate_i(
                        i as GLuint,
                        blend_func(eq.rgb.func),
                        blend_func(eq.alpha.func),
                    );
                    gl.set_enabled_i(GL_BLEND, i as GLuint, true);
                } else {
                    gl.set_enabled_i(GL_BLEND, i as GLuint, false);
                }
                if rt.colormask != sub.hw_blend.colormask[i] {
                    sub.hw_blend.colormask[i] = rt.colormask;
                    gl.color_mask_i(i as GLuint, mask(rt.colormask));
                }
            }
        } else {
            let rt = new_state.rt[0];
            if let Some(eq) = rt.equation {
                if blend_is_dual(&state, 0) && !features.has(Feature::dual_src_blend) {
                    eprintln!(
                        "[virglrs] vrend: dual src blend requested but not supported for rt 0"
                    );
                }
                gl.blend_func_separate(
                    blend_factor(eq.rgb.src),
                    blend_factor(eq.rgb.dst),
                    blend_factor(eq.alpha.src),
                    blend_factor(eq.alpha.dst),
                );
                gl.blend_equation_separate(blend_func(eq.rgb.func), blend_func(eq.alpha.func));
                gl.enable(GL_BLEND);
            } else {
                gl.disable(GL_BLEND);
            }
            if rt.colormask != sub.hw_blend.colormask[0]
                || (sub.hw_blend.independent && !new_state.independent_blend_enable)
            {
                gl.color_mask(mask(rt.colormask));
                sub.hw_blend.colormask = [rt.colormask; MAX_COLOR_BUFS];
            }
        }
        sub.hw_blend.independent = new_state.independent_blend_enable;
        if features.has(Feature::multisample) {
            gl.set_enabled(GL_SAMPLE_ALPHA_TO_COVERAGE, new_state.alpha_to_coverage);
        }
        gl.set_enabled(GL_DITHER, new_state.dither);

        // Only emulated alpha swizzles the blend colour, and that is not a GLES case.
        gl.blend_color(sub.blend_color);
        sub.blend_dirty = false;
    }

    /// `vrend_draw_bind_ubo_shader`.
    fn draw_bind_ubo(
        &mut self,
        host: &mut Host<'_>,
        stage: ShaderStage,
        mut next_ubo_id: BindingPoint,
    ) -> BindingPoint {
        let gl = host.gl;
        let s = stage.index();
        let sub = self.sub_mut();
        let Some(prog) = sub.program() else {
            return next_ubo_id;
        };
        let mut mask = prog.ubo_used_mask[s];
        // The decoder refuses an index past the mask, so every key is a slot it holds.
        let mut used = Dirty::none();
        for slot in sub.ubos[s].keys() {
            used.mark(*slot);
        }
        let mut dirty = sub.ubos_dirty[s];
        let update = dirty.intersect(used);
        if update.is_empty() {
            return next_ubo_id.plus(mask.count_ones());
        }
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            if update.contains(i)
                && let Some(cb) = sub.ubos[s].get(&i)
                && let Some(res) = host.resources.get(&cb.resource)
                && let Storage::Buffer { name, .. } = res.storage
            {
                gl.bind_buffer_range(
                    GL_UNIFORM_BUFFER,
                    next_ubo_id,
                    name,
                    cb.offset as usize,
                    cb.length as usize,
                );
                dirty.unmark(i);
            }
            next_ubo_id = next_ubo_id.next();
        }
        sub.ubos_dirty[s] = dirty;
        next_ubo_id
    }

    /// `vrend_draw_bind_const_shader`: the inline constants, as one `uvec4` array uniform.
    fn draw_bind_const(&mut self, host: &mut Host<'_>, stage: ShaderStage, new_program: bool) {
        let gl = host.gl;
        let s = stage.index();
        let sub = self.sub_mut();
        let Some(prog) = sub.program() else {
            return;
        };
        let num_consts = prog.num_consts[s];
        if let Some(loc) = prog.const_location[s]
            && !sub.consts[s].is_empty()
            && sub.shaders[s].is_some()
            && (sub.const_dirty[s] || new_program)
        {
            let n = (num_consts * 4).min(sub.consts[s].len());
            gl.uniform_4uiv(loc, &sub.consts[s][..n]);
            sub.const_dirty[s] = false;
        }
        // The C also bridges uniform buffer 0 into plain constants for a shader that declared
        // them one-dimensional, out of the resource's guest pages; that is the video
        // compositor's shape and lands with video.
    }

    /// `vrend_draw_bind_samplers_shader`.
    fn draw_bind_samplers(
        &mut self,
        host: &mut Host<'_>,
        stage: ShaderStage,
        mut next_sampler_id: TextureUnit,
    ) -> TextureUnit {
        let gl = host.gl;
        let max_units = host.limits.max_texture_units;
        let s = stage.index();
        let sub = self.sub_mut();
        let Some(prog) = sub.program() else {
            return next_sampler_id;
        };
        let dirty = sub.views_dirty[s];
        let mut mask = prog.samplers_used_mask[s];
        let shadow_mask = prog.shadow_samp_mask[s];
        let sampler_locs = prog.sampler_locs[s].clone();
        let mask_locs = prog.shadow_samp_mask_locs[s].clone();
        let add_locs = prog.shadow_samp_add_locs[s].clone();
        let mut sampler_index = 0usize;
        let mut levels_out: Vec<GLint> = Vec::new();
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            let view = sub.views[s].get(&i).and_then(|h| match sub.objects.get(h) {
                Some(Object::SamplerView(v)) => Some(v),
                _ => None,
            });
            if dirty.contains(i)
                && let Some(view) = view
            {
                gl.active_texture(next_sampler_id);
                if let Some(loc) = sampler_locs[sampler_index] {
                    gl.uniform_1i(loc, next_sampler_id.uniform_value());
                }
                let res = host.resources.get(&view.resource);
                if shadow_mask & (1 << i) != 0 {
                    // A depth texture read through a shadow sampler compares, and the
                    // luminance-style swizzles must not apply: the texture goes back to
                    // identity and the view's swizzle becomes a mask and an add.
                    if let Some(res) = res
                        && let Storage::Texture(t) = &res.storage
                    {
                        let target = t.target;
                        gl.bind_texture(target, Some(t.name));
                        for (c, sw) in [GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA].iter().enumerate() {
                            gl.tex_parameter_i(
                                target,
                                GL_TEXTURE_SWIZZLE_R + c as GLenum,
                                *sw as GLint,
                            );
                        }
                    }
                    let one_or_zero = |g: GLint| g == GL_ZERO as GLint || g == GL_ONE as GLint;
                    let m = view.gl_swizzle.map(|g| if one_or_zero(g) { 0.0 } else { 1.0 });
                    let a = view.gl_swizzle.map(|g| if g == GL_ONE as GLint { 1.0 } else { 0.0 });
                    if let Some(loc) = mask_locs[sampler_index] {
                        gl.uniform_4f(loc, m);
                    }
                    if let Some(loc) = add_locs[sampler_index] {
                        gl.uniform_4f(loc, a);
                    }
                }
                if let Some(res) = res {
                    let (id, target, is_buffer, multisampled) = match &res.storage {
                        Storage::Buffer { tbo, .. } => (*tbo, GL_TEXTURE_BUFFER, true, false),
                        Storage::Texture(t) => (
                            Some(view.view.unwrap_or(t.name)),
                            view.target,
                            false,
                            res.args.nr_samples > 1,
                        ),
                        _ => (None, view.target, false, false),
                    };
                    if let Some(id) = id {
                        gl.bind_texture(target, Some(id));
                    }
                    // vrend_apply_sampler_state, the sampler-objects leg: a buffer or
                    // multisampled texture takes no sampler.
                    if !is_buffer && !multisampled {
                        let sampler =
                            sub.samplers[s].get(&i).and_then(|h| match sub.objects.get(h) {
                                Some(Object::SamplerState(st)) => st.ids,
                                _ => None,
                            });
                        if let Some(ids) = sampler {
                            let id = if view.skip_srgb_decode { ids[0] } else { ids[1] };
                            gl.bind_sampler(next_sampler_id, Some(id));
                        }
                    }
                    let levels = view.last_level.wrapping_sub(view.first_level).wrapping_add(1);
                    let levels = if levels != 0 { levels } else { res.args.last_level + 1 };
                    if levels_out.len() <= sampler_index {
                        levels_out.resize(sampler_index + 1, 0);
                    }
                    levels_out[sampler_index] = levels as GLint;
                }
            }
            sampler_index += 1;
            next_sampler_id = next_sampler_id.next();
        }
        let sub = self.sub_mut();
        let tl = &mut sub.texture_levels[s];
        if tl.len() < sampler_index {
            tl.resize(sampler_index, 0);
        }
        tl.truncate(sampler_index);
        for (i, l) in levels_out.into_iter().enumerate() {
            if i < tl.len() {
                tl[i] = l;
            }
        }
        sub.views_dirty[s].clear();
        // A later glBindTexture for another reason must not disturb the units just bound.
        gl.active_texture(TextureUnit::at(max_units.saturating_sub(1)));
        next_sampler_id
    }

    /// `vrend_draw_bind_ssbo_shader`.
    fn draw_bind_ssbo(&mut self, host: &mut Host<'_>, stage: ShaderStage) {
        let gl = host.gl;
        if !host.has(Feature::ssbo) {
            return;
        }
        let s = stage.index();
        let sub = self.sub();
        let Some(prog) = sub.program() else {
            return;
        };
        let offset = prog.ssbo_binding_offset[s];
        let prog_mask = prog.ssbo_used_mask[s];
        for (&i, ssbo) in &sub.ssbos[s] {
            if i as usize >= MAX_SHADER_BUFFERS || prog_mask & (1 << i) == 0 {
                continue;
            }
            if let Some(res) = host.resources.get(&ssbo.resource)
                && let Storage::Buffer { name, .. } = res.storage
            {
                gl.bind_buffer_range(
                    GL_SHADER_STORAGE_BUFFER,
                    BindingPoint::at(i + offset),
                    name,
                    ssbo.offset as usize,
                    ssbo.length as usize,
                );
            }
        }
    }

    /// `vrend_draw_bind_abo_shader`.
    fn draw_bind_abo(&mut self, host: &mut Host<'_>) {
        let gl = host.gl;
        if !host.has(Feature::atomic_counters) {
            return;
        }
        for (&i, abo) in &self.sub().abos {
            if let Some(res) = host.resources.get(&abo.resource)
                && let Storage::Buffer { name, .. } = res.storage
            {
                gl.bind_buffer_range(
                    GL_ATOMIC_COUNTER_BUFFER,
                    BindingPoint::at(i),
                    name,
                    abo.offset as usize,
                    abo.length as usize,
                );
            }
        }
    }

    /// `vrend_draw_bind_images_shader`.
    fn draw_bind_images(&mut self, host: &mut Host<'_>, stage: ShaderStage) {
        let gl = host.gl;
        let features = host.features;
        let formats = host.formats;
        let s = stage.index();
        let sub = self.sub();
        let Some(prog) = sub.program() else {
            return;
        };
        if sub.images[s].is_empty() || prog.img_locs[s].is_empty() || !features.has(Feature::images)
        {
            return;
        }
        let prog_mask = prog.images_used_mask[s];
        let offset = prog.image_binding_offset[s];
        for (&i, iview) in &sub.images[s] {
            if i as usize >= MAX_SHADER_IMAGES || prog_mask & (1 << i) == 0 {
                continue;
            }
            let image_unit = ImageUnit::at(i + offset);
            if prog.img_locs[s].get(i as usize).copied().flatten().is_none() {
                continue;
            }
            let Some(res) = host.resources.get_mut(&iview.resource) else {
                continue;
            };
            let Some(entry) = formats.get(iview.format) else {
                continue;
            };
            let (tex_id, level, first_layer, layered) = match &mut res.storage {
                Storage::Buffer { name, tbo, .. } => {
                    let tbo_tex = *tbo.get_or_insert_with(|| gl.gen_texture());
                    // `set_shader_images` admits a buffer image only with one of these widths.
                    let bs = iview.format.describe().map_or(1, |d| d.block_bytes());
                    let format = match bs {
                        16 => GL_RGBA32UI,
                        8 => GL_RG32UI,
                        4 => GL_R32UI,
                        2 => GL_R16UI,
                        _ => GL_R8UI,
                    };
                    gl.bind_buffer(GL_TEXTURE_BUFFER, Some(*name));
                    gl.bind_texture(GL_TEXTURE_BUFFER, Some(tbo_tex));
                    if features.has(Feature::arb_or_gles_ext_texture_buffer) {
                        let range = if features.has(Feature::texture_buffer_range) {
                            let bs = bs as usize;
                            let size = iview.level_size as usize / bs;
                            Some((iview.layer_offset as usize, size * bs))
                        } else {
                            None
                        };
                        gl.tex_buffer(format, *name, range);
                    }
                    (tbo_tex, 0, 0, true)
                }
                Storage::Texture(t) => {
                    let level = iview.level_size;
                    let first = iview.layer_offset & 0xffff;
                    let last = (iview.layer_offset >> 16) & 0xffff;
                    let depth = res.args.array_size.max(res.args.depth);
                    let layered =
                        !((res.args.array_size > 1 || res.args.depth > 1) && first == last);
                    let num_layers = last.wrapping_sub(first).wrapping_add(1);
                    if layered && (first != 0 || num_layers != depth) {
                        host.todo.note("image views of a layer subset");
                        continue;
                    }
                    (t.name, level as GLint, first as GLint, layered)
                }
                _ => continue,
            };
            let access = match iview.access {
                ImageAccess::Read => GL_READ_ONLY,
                ImageAccess::Write => GL_WRITE_ONLY,
                ImageAccess::ReadWrite => GL_READ_WRITE,
            };
            gl.bind_image_texture(
                image_unit,
                tex_id,
                level,
                layered,
                first_layer,
                access,
                entry.gl.internalformat,
            );
        }
    }

    /// `vrend_fill_sysval_uniform_block`.
    fn fill_sysval_uniform_block(&mut self, host: &mut Host<'_>) {
        let gl = host.gl;
        let sub = self.sub_mut();
        let sysval = sub.sysval;
        let Some(prog) = sub.program_mut() else {
            return;
        };
        if prog.virgl_block_bind.is_none() {
            return;
        }
        if prog.sysval_uploaded != Some(sysval) {
            let buf = prog.sysval_buffer.expect("a bound block has its buffer");
            gl.bind_buffer(GL_UNIFORM_BUFFER, Some(buf));
            gl.buffer_sub_data(GL_UNIFORM_BUFFER, 0, &sysval.bytes());
            gl.bind_buffer(GL_UNIFORM_BUFFER, None);
            prog.sysval_uploaded = Some(sysval);
        }
    }

    /// `vrend_draw_bind_objects`.
    fn draw_bind_objects(&mut self, host: &mut Host<'_>, new_program: bool) {
        let gl = host.gl;
        let Some(prog) = self.sub().program() else {
            return;
        };
        let last = prog.last_stage;
        let mut next_ubo_id = BindingPoint::FIRST;
        let mut next_sampler_id = TextureUnit::FIRST;
        for stage in C_STAGE_ORDER {
            if stage.index() > last.index() {
                continue;
            }
            next_ubo_id = self.draw_bind_ubo(host, stage, next_ubo_id);
            self.draw_bind_const(host, stage, new_program);
            next_sampler_id = self.draw_bind_samplers(host, stage, next_sampler_id);
            self.draw_bind_images(host, stage);
            self.draw_bind_ssbo(host, stage);
            let sub = self.sub();
            if let Some(prog) = sub.program()
                && let Some(loc) = prog.tex_levels_uniform_id[stage.index()]
            {
                gl.uniform_1iv(loc, &sub.texture_levels[stage.index()]);
            }
        }
        let sub = self.sub();
        if let Some(prog) = sub.program()
            && let Some(bind) = prog.virgl_block_bind
            && let Some(buf) = prog.sysval_buffer
        {
            gl.bind_buffer_range(GL_UNIFORM_BUFFER, bind, buf, 0, Sysval::SIZE);
        }
        self.draw_bind_abo(host);
    }

    /// `vrend_draw_bind_vertex_binding`: the bound layout's VAO, and the vertex buffers when
    /// they changed.
    fn draw_bind_vertex_binding(&mut self, host: &mut Host<'_>) {
        let gl = host.gl;
        let sub = self.sub_mut();
        let vao = sub.ve.and_then(|h| match sub.objects.get(&h) {
            Some(Object::VertexElements(v)) => v.vao,
            _ => None,
        });
        let Some(vao) = vao else {
            gl.bind_vertex_array(Some(sub.vao));
            return;
        };
        gl.bind_vertex_array(Some(vao));
        if !sub.vbo_dirty {
            return;
        }
        for (i, vbo) in sub.vbos.iter().enumerate() {
            let name = vbo.resource.and_then(|r| host.resources.get(&r)).and_then(|res| match res
                .storage
            {
                Storage::Buffer { name, .. } => Some(name),
                _ => None,
            });
            match name {
                Some(name) => {
                    gl.bind_vertex_buffer(i as GLuint, Some(name), vbo.offset, vbo.stride)
                }
                None => gl.bind_vertex_buffer(i as GLuint, None, 0, 0),
            }
        }
        for i in stale_vbo_slots(sub.vbos.len(), sub.hw_num_vbos) {
            gl.bind_vertex_buffer(i as GLuint, None, 0, 0);
        }
        sub.hw_num_vbos = sub.vbos.len();
        sub.vbo_dirty = false;
    }

    /// `vrend_draw_vbo`.
    pub(super) fn draw_vbo(&mut self, host: &mut Host<'_>, draw: Draw) -> Result<(), Fault> {
        let cmd = Cmd::DrawVbo;
        let gl = host.gl;
        let features = host.features;
        let need = |feature: Feature| {
            if features.has(feature) { Ok(()) } else { Err(Fault::NoFeature { cmd, feature }) }
        };
        if draw.instance_count > 0 {
            need(Feature::draw_instance)?;
        }
        if draw.start_instance > 0 {
            need(Feature::base_instance)?;
        }
        let indirect = draw.indirect;
        if let Some(ind) = indirect {
            if ind.draw_count > 1 {
                need(Feature::multi_draw_indirect)?;
            }
            need(Feature::indirect_draw)?;
            if ind.draw_count_resource.is_some() {
                // `feat_indirect_params` is desktop-only.
                return Err(Fault::Unimplemented { cmd, what: "an indirect draw count" });
            }
        }
        // GL takes every count as a signed 32-bit value. Past that, GL would raise an error the
        // batch turns into a fault at its end; saying so here names the draw that asked.
        let sized = |v: u32, what: &'static str| -> Result<GLsizei, Fault> {
            GLsizei::try_from(v).map_err(|_| Fault::OutOfRange { cmd, what })
        };
        let start = sized(draw.start, "a draw start")?;
        let count = sized(draw.count, "a draw count")?;
        let instances = sized(draw.instance_count, "an instance count")?;
        let indirect_counts = match indirect {
            Some(ind) => Some((
                sized(ind.draw_count, "an indirect draw count")?,
                sized(ind.stride, "an indirect stride")?,
            )),
            None => None,
        };
        let vertices_per_patch = match draw.tess {
            Some(t) => sized(t.vertices_per_patch, "a patch size")?,
            None => 0,
        };
        let indirect_buffer = match indirect {
            Some(ind) => match host.resource(cmd, ind.resource)?.storage {
                Storage::Buffer { name, .. } => Some(name),
                _ => return Err(Fault::IllegalResource { cmd, handle: ind.resource }),
            },
            None => None,
        };

        self.flush_lazy_state(host);
        if self.sub().blend_dirty {
            self.patch_blend_state(host);
        }

        let sub = self.sub_mut();
        if sub.prim_mode != draw.mode {
            // Only a switch in or out of points changes the shader variants.
            if sub.prim_mode == PrimType::Points || draw.mode == PrimType::Points {
                sub.shader_dirty = true;
            }
            sub.prim_mode = draw.mode;
        }

        let mut new_program = false;
        let sub = self.sub();
        if sub.shader_dirty
            || sub.swizzle_output_rgb_to_bgr != 0
            || sub.needs_manual_srgb_encode != 0
            || sub.vbo_dirty
        {
            new_program = self.select_linked_program(host, cmd)?;
        }
        // The C drops the draw with a warning; a draw with nothing to run it is a fault here.
        let Some(prog) = self.sub().program() else {
            return Err(Fault::Shader { cmd, what: "a draw with no program" });
        };
        let prog_id = prog.id;
        let reads_drawid = prog.reads_drawid;
        let fs_blend_advanced = prog.fs_blend_equation_advanced;
        gl.use_program(Some(prog_id));

        if features.has(Feature::draw_parameters) && reads_drawid {
            let drawid = draw.tess.map_or(0, |t| t.drawid) as i32;
            self.sub_mut().sysval.drawid_base = drawid;
        }

        self.draw_bind_objects(host, new_program);
        self.fill_sysval_uniform_block(host);
        self.draw_bind_vertex_binding(host);

        let mut index_type = GL_UNSIGNED_INT;
        let mut ib_offset = 0;
        if draw.indexed {
            // The C skips an indexed draw with no index buffer, or one that reads past it,
            // with a warning and success. Both are the guest's claim about its own buffer,
            // and a claim the buffer cannot meet is refused.
            let Some(ib) = self.sub().ib else {
                return Err(Fault::OutOfRange {
                    cmd,
                    what: "an indexed draw with no index buffer",
                });
            };
            let res = host.resource(cmd, ib.resource)?;
            if indirect.is_none() {
                let expected = ib.index_type.bytes() as u64 * draw.count as u64 + ib.offset as u64;
                if expected > res.args.width as u64 {
                    return Err(Fault::OutOfRange { cmd, what: "a draw past its index buffer" });
                }
            }
            let Storage::Buffer { name, .. } = res.storage else {
                return Err(Fault::IllegalResource { cmd, handle: ib.resource });
            };
            gl.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, Some(name));
            index_type = match ib.index_type {
                IndexType::U8 => GL_UNSIGNED_BYTE,
                IndexType::U16 => GL_UNSIGNED_SHORT,
                IndexType::U32 => GL_UNSIGNED_INT,
            };
            ib_offset = ib.offset;
        } else {
            gl.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, None);
        }

        let sub = self.sub_mut();
        if let Some(i) = sub.current_so {
            match sub.streamouts[i].xfb {
                Xfb::NeedBegin => {
                    let mode = if let Some(gs) = sub.bound_program(ShaderStage::Geometry) {
                        gs_xfb_mode(gs.info.gs_out_prim)
                    } else if let Some(tes) = sub.bound_program(ShaderStage::TessEval) {
                        tess_xfb_mode(tes.info.tes_prim, tes.info.tes_point_mode)
                    } else {
                        xfb_mode(draw.mode)
                    };
                    gl.begin_transform_feedback(mode);
                    sub.streamouts[i].xfb = Xfb::Started;
                }
                Xfb::Paused => {
                    gl.resume_transform_feedback();
                    sub.streamouts[i].xfb = Xfb::Started;
                }
                Xfb::Started => {}
            }
        }

        if draw.primitive_restart {
            gl.enable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        }
        if features.has(Feature::indirect_draw) {
            gl.bind_buffer(GL_DRAW_INDIRECT_BUFFER, indirect_buffer);
        }
        if vertices_per_patch > 0 && features.has(Feature::tessellation) {
            gl.patch_parameter_i(GL_PATCH_VERTICES, vertices_per_patch);
        }

        // A host with advanced blend equations but no framebuffer fetch takes the equation the
        // guest sent through the blend state. The wire carries it in the alpha factors of a
        // target whose blending is off, which the decoder does not keep; the shape is counted
        // until it does.
        if fs_blend_advanced != 0
            && !features.has(Feature::framebuffer_fetch)
            && features.has(Feature::blend_equation_advanced)
        {
            host.todo.note("advanced blend equations");
        }

        let mode = prim_mode(draw.mode);
        if !draw.indexed {
            // The C draws `cso` vertices from zero when the wire names a stream-out object to
            // count from -- the handle's number, as it is: the count is never read from the
            // object. A handle is not a count, and no corpus has asked; refused until one does.
            if draw.count_from_so.is_some() {
                host.todo.note("a draw counted from a stream-out object");
                return Err(Fault::Unimplemented {
                    cmd,
                    what: "a draw counted from a stream-out object",
                });
            }
            if let Some(ind) = indirect {
                let (draw_count, stride) = indirect_counts.expect("counted with the buffer");
                if draw_count > 1 {
                    gl.multi_draw_arrays_indirect(mode, ind.offset, draw_count, stride);
                } else {
                    gl.draw_arrays_indirect(mode, ind.offset);
                }
            } else if draw.instance_count > 0 {
                if draw.start_instance > 0 {
                    gl.draw_arrays_instanced_base_instance(
                        mode,
                        start,
                        count,
                        instances,
                        draw.start_instance,
                    );
                } else {
                    gl.draw_arrays_instanced(mode, start, count, instances);
                }
            } else {
                gl.draw_arrays(mode, start, count);
            }
        } else {
            let ranged = draw.min_index != 0 || draw.max_index != u32::MAX;
            if let Some(ind) = indirect {
                let (draw_count, stride) = indirect_counts.expect("counted with the buffer");
                if draw_count > 1 {
                    gl.multi_draw_elements_indirect(
                        mode, index_type, ind.offset, draw_count, stride,
                    );
                } else {
                    gl.draw_elements_indirect(mode, index_type, ind.offset);
                }
            } else if draw.index_bias != 0 {
                if draw.instance_count > 0 {
                    if draw.start_instance > 0 {
                        gl.draw_elements_instanced_base_vertex_base_instance(
                            mode,
                            count,
                            index_type,
                            ib_offset,
                            instances,
                            draw.index_bias,
                            draw.start_instance,
                        );
                    } else {
                        gl.draw_elements_instanced_base_vertex(
                            mode,
                            count,
                            index_type,
                            ib_offset,
                            instances,
                            draw.index_bias,
                        );
                    }
                } else if ranged {
                    gl.draw_range_elements_base_vertex(
                        mode,
                        draw.min_index,
                        draw.max_index,
                        count,
                        index_type,
                        ib_offset,
                        draw.index_bias,
                    );
                } else {
                    gl.draw_elements_base_vertex(
                        mode,
                        count,
                        index_type,
                        ib_offset,
                        draw.index_bias,
                    );
                }
            } else if draw.instance_count > 0 {
                if draw.start_instance > 0 {
                    gl.draw_elements_instanced_base_instance(
                        mode,
                        count,
                        index_type,
                        ib_offset,
                        instances,
                        draw.start_instance,
                    );
                } else {
                    gl.draw_elements_instanced(mode, count, index_type, ib_offset, instances);
                }
            } else if ranged {
                gl.draw_range_elements(
                    mode,
                    draw.min_index,
                    draw.max_index,
                    count,
                    index_type,
                    ib_offset,
                );
            } else {
                gl.draw_elements(mode, count, index_type, ib_offset);
            }
        }

        if draw.primitive_restart {
            gl.disable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        }
        let sub = self.sub_mut();
        if let Some(i) = sub.current_so
            && features.has(Feature::transform_feedback2)
            && sub.streamouts[i].xfb == Xfb::Started
        {
            gl.pause_transform_feedback();
            sub.streamouts[i].xfb = Xfb::Paused;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_that_never_reached_a_draw_still_leaves_its_buffers_unbound() {
        // Four bound, then the guest sets two and one before drawing again: the slots to clear
        // are those the hardware holds, not the ones the last set replaced.
        assert_eq!(stale_vbo_slots(4, 0), 4..4);
        assert_eq!(stale_vbo_slots(1, 4), 1..4);
        assert_eq!(stale_vbo_slots(4, 4), 4..4);
        assert_eq!(stale_vbo_slots(6, 4), 6..6);
    }
}
