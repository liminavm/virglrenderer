// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The shader blitter: a blit no framebuffer blit can do, done as a textured quad.
//!
//! `glBlitFramebuffer` reads a framebuffer's storage and writes another's. That is the wrong
//! operation whenever the two ends disagree about what their storage *means* -- an X-channel
//! format whose alpha is a texture swizzle rather than a stored byte, an IOSurface-backed BGRA
//! texture whose red and blue arrive the other way round, a colourspace one end converts and the
//! other does not. Each of those is a per-texel function, so it belongs in a fragment shader, and
//! this module is that shader plus the one quad it runs on.
//!
//! It renders in its own GL context, shared with the renderer's, exactly as the C's does. That is
//! not an implementation detail to tidy away: the blit sets a program, a framebuffer, a vertex
//! array, a viewport and the depth state, and doing that in the calling sub-context would
//! invalidate every piece of state the draw path tracks. A separate context means the caller's
//! state is untouched and its dirty masks stay true. The caller's context is made current again
//! before the blit returns, because the commands after a `BLIT` in the same batch do not ask.
//!
//! The C's blit context is a file-scope static; here it hangs off the renderer root and is built
//! on the first blit that needs it.

use std::collections::HashMap;

use super::features::{Feature, Features};
use super::gl::gles::*;
use super::gl::{
    BufferName, FramebufferName, GLenum, GLint, GLsizei, Gl, ProgramName, ShaderName, TextureName,
    VertexArrayName,
};
use super::pipe::{Swizzle, TexFilter, TextureTarget};
use super::proto::Format;
use super::shader::{sampler_return_conv, sampler_type_conv};
use super::transfer;
use super::{egl, tgsi};
use crate::vrend::egl::{Version, Winsys};

/// The vertex the quad is drawn from: a clip-space position and a texture coordinate, eight
/// floats, which is also the stride the attribute pointers are set with.
const FLOATS_PER_VERTEX: usize = 8;
const VERTICES: usize = 4;

/// `VS_PASSTHROUGH_GLES`.
const VS_PASSTHROUGH: &str = "#version 310 es\n\
// Blitter\n\
precision mediump float;\n\
in vec4 arg0;\n\
in vec4 arg1;\n\
out vec4 tc;\n\
void main() {\n\
\x20  gl_Position = arg0;\n\
\x20  tc = arg1;\n\
}\n";

/// `FS_FUNC_COL_SRGB_DECODE`.
const SRGB_DECODE: &str = "cvec4 srgb_decode(cvec4 col) {\n\
\x20  vec3 temp = vec3(col.rgb);\n\
\x20  bvec3 thresh = lessThanEqual(temp, vec3(0.04045));\n\
\x20  vec3 a = temp / vec3(12.92);\n\
\x20  vec3 b = pow((temp + vec3(0.055)) / vec3(1.055), vec3(2.4));\n\
\x20  return cvec4(clamp(mix(b, a, thresh), 0.0, 1.0), col.a);\n\
}\n";

/// `FS_FUNC_COL_SRGB_ENCODE`.
const SRGB_ENCODE: &str = "cvec4 srgb_encode(cvec4 col) {\n\
\x20  vec3 temp = vec3(col.rgb);\n\
\x20  bvec3 thresh = lessThanEqual(temp, vec3(0.0031308));\n\
\x20  vec3 a = temp * vec3(12.92);\n\
\x20  vec3 b = (vec3(1.055) * pow(temp, vec3(1.0 / 2.4))) - vec3(0.055);\n\
\x20  return cvec4(mix(b, a, thresh), col.a);\n\
}\n";

/// What a blit asks of the shader. Every field the shader's text depends on is here and nothing
/// else, so two blits sharing a key share a program and two that do not cannot collide.
///
/// The C packs the same fields into a `uint64_t` through a bitfield union, because its hash table
/// takes a `u64` key. Nothing here needs that, and the packing is where a widened enum would
/// silently start aliasing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProgramKey {
    /// Whether this blit writes a colour. A depth-writing blit is a different program with a
    /// different output -- `gl_FragDepth` rather than a draw buffer -- so it is the first thing
    /// the key distinguishes, and none of the colour fields below say anything when it is false.
    color: bool,
    manual_srgb_decode: bool,
    manual_srgb_encode: bool,
    target: TextureTarget,
    /// The source's sample count, as the resource carries it -- not the count the shader loops
    /// over, which is 1 for an integer format. Both are needed: the loop count is what the text
    /// says, and the raw count is what distinguishes two resolves that resolve differently.
    num_samples: u32,
    src_format: Format,
    /// `None` is the identity, which emits no swizzle snippet at all.
    swizzle: Option<[Swizzle; 4]>,
}

/// The GL objects a blit runs on, and the programs built for the keys seen so far.
pub struct Blitter {
    ctx: egl::Context,
    vao: VertexArrayName,
    vbo: BufferName,
    fbo: FramebufferName,
    vs: ShaderName,
    programs: HashMap<ProgramKey, ProgramName>,
}

/// One end of the quad in the destination's pixels, or the source's texels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

/// Why a blit the blitter was handed did not happen. Never a guest fault -- the guest's blit was
/// legal, and this is the host coming up short.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unserved {
    /// The driver refused the shader or the link, which it has already been told about.
    NoProgram,
    /// Attaching the destination needs a feature this host lacks.
    NoFeature(Feature),
}

/// One blit, with every end already resolved: the textures to sample and render into, the
/// rectangles in each, and what the shader must do between them.
///
/// The caller assembles this from the resources and the wire's `BLIT`, which is the boundary that
/// knows the truth about both. Nothing here is a handle the blitter would have to look up, and
/// nothing is a size that could disagree with the thing it measures -- `src_w`/`src_h` are the
/// source's extent *at `src_level`*, taken once.
pub struct Job {
    pub src: TextureName,
    /// The GL target the source texture is bound with.
    pub src_gl_target: GLenum,
    /// The gallium target, which is what the sampler's declaration and the layer arithmetic read.
    pub src_target: TextureTarget,
    pub src_w: u32,
    pub src_h: u32,
    pub src_level: u32,
    pub src_samples: u32,
    /// The format the shader samples as: the one the BLIT named for its source, whose table entry
    /// decides the sampler's return type.
    pub src_format: Format,
    /// `vrend_set_tex_param`: the table swizzle the source's format is stored with, set on the
    /// texture object because that is where GL keeps it.
    pub src_table_swizzle: Option<[Swizzle; 4]>,
    /// Whether to re-assert `GL_TEXTURE_SRGB_DECODE_EXT`, in case stale state disabled it.
    pub set_srgb_decode: bool,
    pub filter: TexFilter,
    pub src_box: (Point, i32, i32),
    pub src_z: i32,
    /// The depth of the source *box*: how many slices this blit reads.
    pub src_depth: i32,
    /// The depth of the source *texture* at `src_level`, which is what a 3D texture's layer
    /// coordinate is normalised by. Not the same number as `src_depth`, and dividing by that one
    /// makes a sub-range blit sample the wrong slices.
    pub src_texture_depth: u32,

    /// Whether this blit writes a colour. The C's `blit_depth`, negated: both ends of the blit
    /// carry depth and the guest asked for the Z channel. It decides the program, and it is a
    /// separate fact from `dst_attachment` -- one is about the formats the BLIT names, the other
    /// about the format the destination RESOURCE was made with.
    pub color: bool,
    pub dst: TextureName,
    pub dst_gl_target: GLenum,
    /// Where the destination hangs on the blitter's framebuffer: colour, depth, or both. The C
    /// reads it off the destination resource inside `vrend_fb_bind_texture_id`; the blitter here
    /// holds no resources, so the caller resolves it once and passes the answer.
    pub dst_attachment: GLenum,
    /// The destination's gallium target, which decides whether the attached layer is the one the
    /// guest named or this pass's slice.
    pub dst_target: TextureTarget,
    pub dst_w: u32,
    pub dst_h: u32,
    pub dst_level: u32,
    pub dst_layer: i32,
    pub dst_box: (Point, i32, i32),
    pub dst_depth: i32,

    /// What the fragment shader does to a sampled texel, or `None` for the identity.
    pub swizzle: Option<[Swizzle; 4]>,
    pub manual_srgb_decode: bool,
    pub manual_srgb_encode: bool,
    /// `Some` only where the host has sRGB write control; then it is whether to enable it.
    pub framebuffer_srgb: Option<bool>,
    /// `glScissor`'s x, y, width, height.
    pub scissor: Option<[GLint; 4]>,
}

/// `vrend_set_tex_param`, which sets on the source *texture object* what a sampler object cannot
/// carry: the format's stored swizzle, and the level range the fetch is confined to.
fn set_tex_param(gl: &Gl, job: &Job) {
    let t = job.src_gl_target;
    if let Some(sw) = job.src_table_swizzle {
        for (i, s) in sw.iter().enumerate() {
            gl.tex_parameter_i(t, GL_TEXTURE_SWIZZLE_R + i as GLenum, gl_swizzle(*s));
        }
    }
    if job.set_srgb_decode && job.src_samples < 1 {
        gl.tex_parameter_i(t, GL_TEXTURE_SRGB_DECODE_EXT, GL_DECODE_EXT as GLint);
    }
    if job.src_samples < 1 {
        for wrap in [GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R] {
            gl.tex_parameter_i(t, wrap, GL_CLAMP_TO_EDGE as GLint);
        }
    }
    gl.tex_parameter_i(t, GL_TEXTURE_BASE_LEVEL, job.src_level as GLint);
    gl.tex_parameter_i(t, GL_TEXTURE_MAX_LEVEL, job.src_level as GLint);
    if job.src_samples < 1 {
        let f = if job.filter == TexFilter::Nearest { GL_NEAREST } else { GL_LINEAR } as GLint;
        gl.tex_parameter_i(t, GL_TEXTURE_MAG_FILTER, f);
        gl.tex_parameter_i(t, GL_TEXTURE_MIN_FILTER, f);
    }
}

/// `to_gl_swizzle`.
fn gl_swizzle(s: Swizzle) -> GLint {
    (match s {
        Swizzle::X => GL_RED,
        Swizzle::Y => GL_GREEN,
        Swizzle::Z => GL_BLUE,
        Swizzle::W => GL_ALPHA,
        Swizzle::Zero => GL_ZERO,
        Swizzle::One => GL_ONE,
    }) as GLint
}

/// `vrend_set_vertex_param`: the two attributes of the passthrough vertex shader, both reading
/// the one interleaved buffer.
fn set_vertex_param(gl: &Gl, prog: ProgramName) {
    let stride = (FLOATS_PER_VERTEX * 4) as GLsizei;
    for (name, offset) in [("arg0", 0u32), ("arg1", 16)] {
        let Some(loc) = gl.get_attrib_location(prog, name) else {
            continue;
        };
        gl.vertex_attrib_pointer(loc, 4, GL_FLOAT, false, stride, offset);
        gl.enable_vertex_attrib_array_at(loc);
    }
}

/// The vertex buffer's bytes. GL wants the floats as the host stores them, and a `&[f32]` cannot
/// become a `&[u8]` without unsafe -- which this module does not have and does not need for
/// thirty-two floats.
fn vertex_bytes(
    floats: &[f32; FLOATS_PER_VERTEX * VERTICES],
) -> [u8; FLOATS_PER_VERTEX * VERTICES * 4] {
    let mut out = [0u8; FLOATS_PER_VERTEX * VERTICES * 4];
    for (i, f) in floats.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&f.to_ne_bytes());
    }
    out
}

impl Blitter {
    /// Bring the blit context up: its own GL context sharing the renderer's objects, and the one
    /// vertex array, buffer, framebuffer and passthrough vertex shader every blit reuses.
    ///
    /// Leaves its own context current -- the caller is switching to it anyway.
    pub fn open(
        winsys: &Winsys,
        gl: &Gl,
        version: Version,
        share: &egl::Context,
    ) -> Result<Blitter, egl::EglError> {
        let ctx = winsys.create_context(version, Some(share))?;
        winsys.make_current(&ctx)?;
        let vao = gl.gen_vertex_array();
        let vbo = gl.gen_buffer();
        let fbo = gl.gen_framebuffer();
        let vs = gl.create_shader(GL_VERTEX_SHADER).expect("the driver makes a vertex shader");
        gl.compile_shader(vs, VS_PASSTHROUGH).expect("the blitter's passthrough shader compiles");
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(GL_ARRAY_BUFFER, Some(vbo));
        Ok(Blitter { ctx, vao, vbo, fbo, vs, programs: HashMap::new() })
    }

    pub fn context(&self) -> &egl::Context {
        &self.ctx
    }

    /// `vrend_renderer_blit_gl`: draw the quad, once per destination layer.
    ///
    /// The caller has made this blitter's context current and resolved every end of the blit into
    /// [`Job`]; this touches nothing the caller owns but the two textures the job names.
    pub fn run(&mut self, gl: &Gl, features: &Features, job: &Job) -> Result<(), Unserved> {
        let key = ProgramKey {
            color: job.color,
            manual_srgb_decode: job.manual_srgb_decode,
            manual_srgb_encode: job.manual_srgb_encode,
            target: job.src_target,
            num_samples: job.src_samples,
            src_format: job.src_format,
            swizzle: job.swizzle,
        };
        let prog = self.program(gl, key).ok_or(Unserved::NoProgram)?;
        let (src0, src1, dst0, dst1) =
            bounded_points(job.src_w, job.src_h, job.src_box, job.dst_box);
        gl.use_program(Some(prog));
        gl.bind_vertex_array(Some(self.vao));
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(self.fbo));
        gl.draw_buffers(&[GL_COLOR_ATTACHMENT0]);
        gl.bind_texture(job.src_gl_target, Some(job.src));
        set_tex_param(gl, job);
        set_vertex_param(gl, prog);
        // `set_dsa_write_depth_keep_stencil`: the quad must not be depth-tested away by whatever
        // the destination happens to carry.
        gl.disable(GL_STENCIL_TEST);
        gl.enable(GL_DEPTH_TEST);
        gl.depth_func(GL_ALWAYS);
        gl.depth_mask(true);
        match job.scissor {
            Some([x, y, w, h]) => {
                gl.scissor(x, y, w, h);
                gl.enable(GL_SCISSOR_TEST);
            }
            None => gl.disable(GL_SCISSOR_TEST),
        }
        if let Some(on) = job.framebuffer_srgb {
            if on {
                gl.enable(GL_FRAMEBUFFER_SRGB);
            } else {
                gl.disable(GL_FRAMEBUFFER_SRGB);
            }
        }
        // The viewport is the whole destination level; the quad's clip-space corners carry the
        // rectangle, so a scaled blit is a smaller quad rather than a smaller viewport.
        gl.viewport(0, 0, job.dst_w as GLsizei, job.dst_h as GLsizei);
        let normalized = job.src_gl_target != GL_TEXTURE_RECTANGLE && job.src_samples < 1;
        let mut vertices = [0f32; FLOATS_PER_VERTEX * VERTICES];
        for dst_z in 0..job.dst_depth {
            // The layer sampled for this destination slice, at the middle of the source's share
            // of it -- what the C's dst2src_scale and dst_offset compute.
            let scale = job.src_depth as f32 / job.dst_depth as f32;
            let offset = ((job.src_depth - 1) as f32 - (job.dst_depth - 1) as f32 * scale) * 0.5;
            let src_z = (dst_z as f32 + offset) * scale;
            // The DESTINATION's target decides this: a layered destination is attached at the
            // layer the guest named, and only a plain one walks its slices.
            let layer = match job.dst_target {
                TextureTarget::Cube | TextureTarget::Array1d | TextureTarget::Array2d => {
                    job.dst_layer
                }
                _ => dst_z,
            };
            transfer::attach_texture(
                gl,
                features,
                job.dst_gl_target,
                job.dst,
                job.dst_attachment,
                job.dst_level as GLint,
                Some(layer),
            )
            .map_err(Unserved::NoFeature)?;
            let coord = texcoords(normalized, job.src_w, job.src_h, src0, src1);
            let pos = quad_positions(job.dst_w, job.dst_h, dst0, dst1);
            let tex = quad_texcoords(coord);
            let layer_coord = job.src_z as f32 + src_z;
            for i in 0..VERTICES {
                let v = &mut vertices[i * FLOATS_PER_VERTEX..(i + 1) * FLOATS_PER_VERTEX];
                v[0] = pos[i][0];
                v[1] = pos[i][1];
                v[2] = 0.0;
                v[3] = 1.0;
                v[4] = tex[i][0];
                v[5] = tex[i][1];
                v[6] = 0.0;
                v[7] = 0.0;
                match job.src_target {
                    TextureTarget::Texture3d => {
                        v[6] = layer_coord / job.src_texture_depth.max(1) as f32
                    }
                    TextureTarget::Array1d => v[5] = layer_coord,
                    TextureTarget::Array2d => v[6] = layer_coord,
                    _ => {}
                }
            }
            gl.bind_buffer(GL_ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data(GL_ARRAY_BUFFER, &vertex_bytes(&vertices), GL_STATIC_DRAW);
            gl.draw_arrays(GL_TRIANGLE_FAN, 0, VERTICES as GLsizei);
        }
        gl.use_program(None);
        gl.framebuffer_texture_2d(GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, None, 0);
        gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, None, 0);
        gl.bind_texture(job.src_gl_target, None);
        Ok(())
    }

    /// The program for this key, built and cached on first use. `None` when the shader would not
    /// compile or the program would not link, which is reported once by the caller.
    fn program(&mut self, gl: &Gl, key: ProgramKey) -> Option<ProgramName> {
        if let Some(p) = self.programs.get(&key) {
            return Some(*p);
        }
        let source = fragment_source(key);
        let fs = gl.create_shader(GL_FRAGMENT_SHADER)?;
        if let Err(log) = gl.compile_shader(fs, &source) {
            eprintln!("[virglrs] vrend: the blitter's fragment shader failed to compile: {log}");
            eprintln!("{source}");
            gl.delete_shader(fs);
            return None;
        }
        let prog = gl.create_program()?;
        gl.attach_shader(prog, self.vs);
        gl.attach_shader(prog, fs);
        let linked = gl.link_program(prog);
        gl.delete_shader(fs);
        if let Err(log) = linked {
            eprintln!("[virglrs] vrend: the blitter's program failed to link: {log}");
            gl.delete_program(prog);
            return None;
        }
        self.programs.insert(key, prog);
        Some(prog)
    }
}

/// `util_pipe_tex_to_tgsi_tex`, the subset a blit can name: the target the sampler is declared
/// with, which is the multisample spelling when the source carries samples.
fn tgsi_texture(target: TextureTarget, num_samples: u32) -> tgsi::Texture {
    use TextureTarget::*;
    match (target, num_samples > 1) {
        (Buffer, _) => tgsi::Texture::Buffer,
        (Texture1d, _) => tgsi::Texture::D1,
        (Texture2d, false) => tgsi::Texture::D2,
        (Texture2d, true) => tgsi::Texture::Msaa2d,
        (Texture3d, _) => tgsi::Texture::D3,
        (Cube, _) => tgsi::Texture::Cube,
        (Rect, _) => tgsi::Texture::Rect,
        (Array1d, _) => tgsi::Texture::Array1d,
        (Array2d, false) => tgsi::Texture::Array2d,
        (Array2d, true) => tgsi::Texture::Msaa2dArray,
        (CubeArray, _) => tgsi::Texture::CubeArray,
    }
}

/// `blit_get_swizzle`, GLES leg without depth: the components of `tc` the fetch takes, and the
/// integer coordinate type a `texelFetch` needs. The bool is whether that type is an array one,
/// which is the only thing that decides which GLES header the shader gets.
fn coord_swizzle_and_type(
    target: tgsi::Texture,
    msaa: bool,
    depth: bool,
) -> (&'static str, &'static str, bool) {
    use tgsi::Texture::*;
    match target {
        // `BLIT_USE_GLES | BLIT_USE_DEPTH`: GLES has no 1D sampler, so a depth 1D blit samples
        // the 2D one the shader declared and needs the second coordinate the colour path fakes
        // inline. This is the only thing depth changes about the coordinates.
        D1 if depth => (".xy", "", false),
        Buffer | D1 => (".x", "", false),
        Msaa2d if msaa => (".xy", "ivec2", false),
        Msaa2d => (".xy", "", false),
        Array1d => (".xyz", "", false),
        D2 | Rect => (".xy", "", false),
        Msaa2dArray if msaa => (".xyz", "ivec3", true),
        Msaa2dArray | Shadow1d | Shadow2d | Shadow1dArray | ShadowRect | D3 | Cube | Array2d => {
            (".xyz", "", false)
        }
        ShadowCube | Shadow2dArray | ShadowCubeArray | CubeArray => ("", "", false),
        Unknown => (".xy", "", false),
    }
}

/// `vec4_type_for_tgsi_ret`.
fn vec4_type(ret: tgsi::ReturnType) -> &'static str {
    match ret {
        tgsi::ReturnType::Sint => "ivec4",
        tgsi::ReturnType::Uint => "uvec4",
        _ => "vec4",
    }
}

/// `tgsi_ret_for_format`.
fn return_type_for(format: Format) -> tgsi::ReturnType {
    let desc = format.describe();
    if desc.is_some_and(|d| d.is_pure_uint()) {
        tgsi::ReturnType::Uint
    } else if desc.is_some_and(|d| d.is_pure_sint()) {
        tgsi::ReturnType::Sint
    } else {
        tgsi::ReturnType::Unorm
    }
}

/// `create_dest_swizzle_snippet`: the expression that reorders a sampled texel into the
/// destination's channel order.
///
/// The swizzle names, for each destination channel, which *source* channel it reads. The shader
/// needs the inverse -- for each output channel, where it comes from -- so this inverts the map,
/// and a channel nothing maps to is 0 in the colours and 1 in the alpha. A channel named twice
/// keeps its first claimant, which is the C's rule and matters only for a swizzle the format
/// table cannot produce.
pub fn dest_swizzle_snippet(swizzle: [Swizzle; 4]) -> String {
    let mut inverse: [Option<usize>; 4] = [None; 4];
    for (i, s) in swizzle.iter().enumerate() {
        let c = *s as usize;
        if c > 3 {
            continue;
        }
        if inverse[c].is_none() {
            inverse[c] = Some(i);
        }
    }
    let mut out = String::new();
    for (i, from) in inverse.iter().enumerate() {
        match from {
            Some(c) => out.push_str(&format!("texel.{}", "rgba".as_bytes()[*c] as char)),
            None if i < 3 => out.push_str("0.0f"),
            None => out.push_str("1.0f"),
        }
        if i < 3 {
            out.push_str(", ");
        }
    }
    out
}

/// The fragment shader for a key, as the C's `blit_build_frag_tex_col` prints it -- GLES leg,
/// which is the only one this tree has a host for.
/// `blit_build_frag_depth`: the fragment shader a depth-writing blit runs.
///
/// None of the colour shader's machinery applies -- no return-type conversion, no destination
/// swizzle, no sRGB -- because the one channel that exists goes to `gl_FragDepth`, and a depth
/// has no colourspace. The header is the C's plain `HEADER_GLES`, which differs from the colour
/// path's `FS_HEADER_GLES` only in the extension line this shader never needs.
fn depth_fragment_source(tex: tgsi::Texture, msaa: bool) -> String {
    let (coord, fetch_type, is_array) = coord_swizzle_and_type(tex, msaa, true);
    let sampler = sampler_type_conv(tex).unwrap_or("2D");
    let header = if msaa && is_array {
        "#version 310 es\n// Blitter\n#extension GL_OES_texture_storage_multisample_2d_array: \
         require\nprecision mediump float;\n"
    } else {
        "#version 310 es\n// Blitter\nprecision mediump float;\n"
    };
    // A multisample source has no `texture`, so the C reads sample 0 and does not average the
    // way the colour path does: a depth is a position, and the mean of two positions is a third
    // one that neither sample saw.
    let body = if msaa {
        format!(
            "void main() {{\n   gl_FragDepth = float(texelFetch(samp, \
             {fetch_type}(tc{coord}), 0).x);\n}}\n"
        )
    } else {
        format!("void main() {{\n   gl_FragDepth = float(texture(samp, tc{coord}).x);\n}}\n")
    };
    format!("{header}uniform mediump sampler{sampler} samp;\nin vec4 tc;\n{body}")
}

pub fn fragment_source(key: ProgramKey) -> String {
    let tex = tgsi_texture(key.target, key.num_samples);
    let msaa = key.num_samples > 1;
    if !key.color {
        return depth_fragment_source(tex, msaa);
    }
    let ret = return_type_for(key.src_format);
    // The C loops over every sample only where averaging them means something: an integer format
    // has no meaningful average, so it reads one.
    let loop_samples =
        if msaa && ret == tgsi::ReturnType::Unorm { key.num_samples } else { u32::from(msaa) };
    let (coord, fetch_type, is_array) = coord_swizzle_and_type(tex, msaa, false);
    let sampler = sampler_type_conv(tex).unwrap_or("2D");
    let prefix = sampler_return_conv(ret);
    let cvec4 = vec4_type(ret);
    // The C always prints the snippet, identity included -- `info->swizzle` is an array and its
    // "no swizzle" value is {0,1,2,3}, not a null pointer. `None` here is that same identity, kept
    // apart in the key only so two spellings of it cannot make two programs.
    let texel = dest_swizzle_snippet(key.swizzle.unwrap_or([
        Swizzle::X,
        Swizzle::Y,
        Swizzle::Z,
        Swizzle::W,
    ]));
    let decode_fn = if key.manual_srgb_decode { SRGB_DECODE } else { "" };
    let encode_fn = if key.manual_srgb_encode { SRGB_ENCODE } else { "" };
    let decode = if key.manual_srgb_decode { "srgb_decode" } else { "" };
    let encode = if key.manual_srgb_encode { "srgb_encode" } else { "" };
    // `FS_HEADER_GLES` / `FS_HEADER_GLES_MS_ARRAY`: the multisample-array sampler is an extension
    // even where the rest of 3.1 is core, and the `%s` the C passes for the extension line is
    // empty for every other shader.
    let header = if msaa && is_array {
        "#version 310 es\n// Blitter\n#extension GL_OES_texture_storage_multisample_2d_array: \
         require\n\nprecision mediump float;\n"
    } else {
        "#version 310 es\n// Blitter\n\nprecision mediump float;\n"
    };
    let body = if msaa {
        format!(
            "void main() {{\n   const int num_samples = {loop_samples};\n   cvec4 texel = \
             cvec4(0);\n   for (int i = 0; i < num_samples; ++i) \n      texel += \
             decode(texelFetch(samp, {fetch_type}(tc{coord}), i));\n   texel = texel / \
             cvec4(num_samples);\n   FragColor = encode(cvec4({texel}));\n}}\n"
        )
    } else if tex == tgsi::Texture::D1 {
        // GLES has no 1D sampler, so the C samples the 2D one it declared at the middle of the
        // one row that exists.
        format!(
            "void main() {{\n   cvec4 texel = decode(texture(samp, vec2(tc{coord}, 0.5)));\n   \
             FragColor = encode(cvec4({texel}));\n}}\n"
        )
    } else {
        format!(
            "void main() {{\n   cvec4 texel = decode(cvec4(texture(samp, tc{coord})));\n   \
             FragColor = encode(cvec4({texel}));\n}}\n"
        )
    };
    format!(
        "{header}#define cvec4 {cvec4}\n{decode_fn}\n{encode_fn}\n#define decode {decode}\n\
         #define encode {encode}\nuniform mediump {prefix}sampler{sampler} samp;\nin vec4 tc;\n\
         out cvec4 FragColor;\n{body}"
    )
}

/// `calc_delta_for_bound`: how far `v` must move to land inside `[0, max]`.
pub fn delta_for_bound(v: i32, max: i32) -> i32 {
    if v < 0 {
        -v
    } else if v > max {
        -(v - max)
    } else {
        0
    }
}

/// `blitter_set_points`: the source and destination rectangles, with the source clamped into the
/// texture it reads and the destination moved by the same fraction.
///
/// A guest may name a source box that runs off the texture. Sampling it would read whatever
/// `GL_CLAMP_TO_EDGE` gives back and stretch it across the destination; the C instead pulls the
/// source rectangle back inside and shortens the destination in proportion, so the pixels that
/// exist land where they would have.
pub fn bounded_points(
    src_w: u32,
    src_h: u32,
    src: (Point, i32, i32),
    dst: (Point, i32, i32),
) -> (Point, Point, Point, Point) {
    let (src0, src_width, src_height) = src;
    let (dst0, dst_width, dst_height) = dst;
    let max_x = src_w as i32 - 1;
    let max_y = src_h as i32 - 1;
    // Whether a point's bound is inclusive depends on which way the blit reads: for a flipped
    // box the first point is the exclusive end and the second the inclusive one.
    let x_excl = i32::from(src_width < 0);
    let y_excl = i32::from(src_height < 0);
    let s0 = Point {
        x: delta_for_bound(src0.x, max_x + x_excl),
        y: delta_for_bound(src0.y, max_y + y_excl),
    };
    let s1 = Point {
        x: delta_for_bound(src0.x + src_width, max_x + 1 - x_excl),
        y: delta_for_bound(src0.y + src_height, max_y + 1 - y_excl),
    };
    let scale_x = dst_width as f32 / src_width as f32;
    let scale_y = dst_height as f32 / src_height as f32;
    (
        Point { x: src0.x + s0.x, y: src0.y + s0.y },
        Point { x: src0.x + src_width + s1.x, y: src0.y + src_height + s1.y },
        Point {
            x: dst0.x + (s0.x as f32 * scale_x) as i32,
            y: dst0.y + (s0.y as f32 * scale_y) as i32,
        },
        Point {
            x: dst0.x + dst_width + (s1.x as f32 * scale_x) as i32,
            y: dst0.y + dst_height + (s1.y as f32 * scale_y) as i32,
        },
    )
}

/// `blitter_set_rectangle`: the destination rectangle as clip-space positions.
pub fn quad_positions(dst_w: u32, dst_h: u32, p0: Point, p1: Point) -> [[f32; 2]; 4] {
    let x = |v: i32| v as f32 / dst_w as f32 * 2.0 - 1.0;
    let y = |v: i32| v as f32 / dst_h as f32 * 2.0 - 1.0;
    [[x(p0.x), y(p0.y)], [x(p1.x), y(p0.y)], [x(p1.x), y(p1.y)], [x(p0.x), y(p1.y)]]
}

/// `get_texcoords`: the source rectangle as texture coordinates, normalised unless the source is
/// a multisample texture, whose fetches are by texel.
pub fn texcoords(normalized: bool, w: u32, h: u32, p0: Point, p1: Point) -> [f32; 4] {
    if normalized {
        [
            p0.x as f32 / w as f32,
            p0.y as f32 / h as f32,
            p1.x as f32 / w as f32,
            p1.y as f32 / h as f32,
        ]
    } else {
        [p0.x as f32, p0.y as f32, p1.x as f32, p1.y as f32]
    }
}

/// `set_texcoords_in_vertices`: the four corners, in the triangle-fan order the quad is drawn in.
pub fn quad_texcoords(coord: [f32; 4]) -> [[f32; 2]; 4] {
    [[coord[0], coord[1]], [coord[2], coord[1]], [coord[2], coord[3]], [coord[0], coord[3]]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_identity_swizzle_reads_every_channel_where_it_lies() {
        use Swizzle::*;
        assert_eq!(dest_swizzle_snippet([X, Y, Z, W]), "texel.r, texel.g, texel.b, texel.a");
    }

    #[test]
    fn a_red_blue_swap_is_its_own_inverse() {
        use Swizzle::*;
        assert_eq!(dest_swizzle_snippet([Z, Y, X, W]), "texel.b, texel.g, texel.r, texel.a");
    }

    #[test]
    fn a_channel_nothing_maps_to_is_zero_in_colour_and_one_in_alpha() {
        use Swizzle::*;
        // What an X-channel format's table swizzle is: alpha comes from nowhere, so it is 1.
        assert_eq!(dest_swizzle_snippet([X, Y, Z, One]), "texel.r, texel.g, texel.b, 1.0f");
        // And a lone red, which is what a luminance format's swizzle inverts to.
        assert_eq!(dest_swizzle_snippet([X, X, X, One]), "texel.r, 0.0f, 0.0f, 1.0f");
    }

    #[test]
    fn a_point_inside_its_bound_does_not_move() {
        assert_eq!(delta_for_bound(0, 63), 0);
        assert_eq!(delta_for_bound(63, 63), 0);
        assert_eq!(delta_for_bound(-4, 63), 4);
        assert_eq!(delta_for_bound(70, 63), -7);
    }

    #[test]
    fn a_source_box_that_runs_off_the_texture_shortens_its_destination_with_it() {
        // A 64-wide source read from -8 to 56 into a 64-wide destination: the eight texels that
        // do not exist are dropped from both ends of the pair, not stretched.
        let (s0, s1, d0, d1) =
            bounded_points(64, 64, (Point { x: -8, y: 0 }, 64, 64), (Point { x: 0, y: 0 }, 64, 64));
        assert_eq!(s0, Point { x: 0, y: 0 });
        assert_eq!(s1, Point { x: 56, y: 64 });
        assert_eq!(d0, Point { x: 8, y: 0 });
        assert_eq!(d1, Point { x: 64, y: 64 });
    }

    #[test]
    fn a_blit_wholly_inside_its_source_is_the_box_it_was_given() {
        let (s0, s1, d0, d1) =
            bounded_points(64, 64, (Point { x: 0, y: 0 }, 64, 64), (Point { x: 0, y: 0 }, 32, 32));
        assert_eq!((s0, s1), (Point { x: 0, y: 0 }, Point { x: 64, y: 64 }));
        assert_eq!((d0, d1), (Point { x: 0, y: 0 }, Point { x: 32, y: 32 }));
    }

    #[test]
    fn the_quad_covers_the_whole_destination_in_clip_space() {
        let p = quad_positions(64, 64, Point { x: 0, y: 0 }, Point { x: 64, y: 64 });
        assert_eq!(p, [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]);
    }

    #[test]
    fn texcoords_are_normalised_except_for_a_multisample_source() {
        let p0 = Point { x: 0, y: 0 };
        let p1 = Point { x: 32, y: 64 };
        assert_eq!(texcoords(true, 64, 64, p0, p1), [0.0, 0.0, 0.5, 1.0]);
        assert_eq!(texcoords(false, 64, 64, p0, p1), [0.0, 0.0, 32.0, 64.0]);
    }

    #[test]
    fn a_depth_blit_writes_the_fragment_depth_and_nothing_else() {
        let src = depth_fragment_source(tgsi::Texture::D2, false);
        assert!(src.contains("gl_FragDepth = float(texture(samp, tc.xy).x);"), "{src}");
        // The colour shader's whole apparatus is absent, not merely unused: a depth has no
        // colourspace and no destination channels to reorder.
        for absent in ["FragColor", "srgb", "cvec4", "#define"] {
            assert!(!src.contains(absent), "{absent} in {src}");
        }
    }

    #[test]
    fn a_1d_depth_source_needs_the_second_coordinate_a_colour_one_fakes() {
        // GLES has no 1D sampler. The colour path declares a 2D one and writes the missing
        // coordinate into the call; the depth path takes it from the vertex instead, and this is
        // the only thing `BLIT_USE_DEPTH` changes about the coordinates.
        assert_eq!(coord_swizzle_and_type(tgsi::Texture::D1, false, true).0, ".xy");
        assert_eq!(coord_swizzle_and_type(tgsi::Texture::D1, false, false).0, ".x");
        assert_eq!(coord_swizzle_and_type(tgsi::Texture::D2, false, true).0, ".xy");
    }

    #[test]
    fn a_multisample_depth_source_reads_one_sample_rather_than_averaging() {
        let src = depth_fragment_source(tgsi::Texture::Msaa2d, true);
        assert!(src.contains("texelFetch(samp, ivec2(tc.xy), 0)"), "{src}");
        assert!(!src.contains("num_samples"), "{src}");
    }
}
