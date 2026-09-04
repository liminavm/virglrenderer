// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The pixel-moving commands that need no guest shader: blit, resource copy, surface clear.
//!
//! Each follows the C's choice of GL path -- `glCopyImageSubData` when the two images are
//! copy-compatible and nothing but pixels move, `glBlitFramebuffer` when a framebuffer can hold
//! both ends, the shader blitter otherwise -- because the path decides the pixels: a
//! framebuffer blit converts and filters where a copy does neither. The blitter's colour path is
//! in [`super::super::blitter`]; a blit that would have to write depth through it is counted and
//! skipped, not faked through another path.

use super::*;
use crate::vrend::blitter;

const PIPE_MASK_RGBA: u8 = 0xf;
const PIPE_MASK_Z: u8 = 0x10;
const PIPE_MASK_S: u8 = 0x20;

/// `format_is_copy_compatible`: whether `glCopyImageSubData` may move pixels between the two.
///
/// Plain pairs follow gallium's `util_is_format_compatible`; compressed pairs follow GL's view
/// classes, which is the rule the driver enforces and the C's hand-written list approximates.
fn copy_compatible(formats: &Table, src: Format, dst: Format, allow_compressed: bool) -> bool {
    if src == dst {
        return true;
    }
    let (Some(sd), Some(dd)) = (src.describe(), dst.describe()) else {
        return false;
    };
    if sd.is_plain() && dd.is_plain() {
        return plain_compatible(sd, dd);
    }
    if !allow_compressed {
        return false;
    }
    let (Some(se), Some(de)) = (formats.get(src), formats.get(dst)) else {
        return false;
    };
    use super::super::formats::ViewClass as V;
    match (se.gl.view_class, de.gl.view_class) {
        (V::Unsupported, _) | (_, V::Unsupported) => false,
        (a, b) if a == b => true,
        (a, b) => {
            let bytes = |v: V| match v {
                V::Bits128
                | V::Rgtc2Rg
                | V::BptcUnorm
                | V::BptcFloat
                | V::Dxt3Rgba
                | V::Dxt5Rgba
                | V::Etc2EacRgba => Some(16),
                V::Bits64 | V::Rgtc1Red | V::Dxt1Rgb | V::Dxt1Rgba | V::Etc2Rgb | V::Etc2Rgba => {
                    Some(8)
                }
                _ => None,
            };
            let compressed = |v: V| {
                !matches!(v, V::Bits128 | V::Bits96 | V::Bits64 | V::Bits32 | V::Bits16 | V::Bits8)
            };
            compressed(a) != compressed(b) && bytes(a).is_some() && bytes(a) == bytes(b)
        }
    }
}

/// gallium's `util_is_format_compatible`.
fn plain_compatible(s: &Desc, d: &Desc) -> bool {
    if s.block.bits != d.block.bits
        || s.nr_channels != d.nr_channels
        || s.colorspace != d.colorspace
    {
        return false;
    }
    if (0..4).any(|c| s.channels[c].bits != d.channels[c].bits) {
        return false;
    }
    for c in 0..4 {
        if let Some(sw) = d.swizzle[c]
            && (sw as u32) < 4
        {
            if s.swizzle[c] != Some(sw) {
                return false;
            }
            let (sc, dc) = (s.channels[sw as usize], d.channels[sw as usize]);
            if sc.ty != dc.ty || sc.normalized != dc.normalized {
                return false;
            }
        }
    }
    true
}

/// `vrend_blit_needs_swizzle`: the two formats' table swizzles differ, so a framebuffer blit
/// would carry channels across in the wrong order.
fn needs_swizzle(formats: &Table, a: Format, b: Format) -> bool {
    let sw = |f: Format| formats.get(f).and_then(|e| e.gl.swizzle).unwrap_or([Swizzle::X; 4]);
    sw(a) != sw(b)
}

/// A texture as one end of a blit: the resource's own, or a view reinterpreting it.
struct End {
    name: TextureName,
    target: GLenum,
    /// Made for this blit, deleted after it.
    temporary: bool,
}

/// `vrend_make_view`: a view of the resource in `format`, or the resource itself when the
/// internal formats agree, the view classes disagree, or the storage is not immutable.
fn make_view(
    gl: &Gl,
    features: &Features,
    formats: &Table,
    res: &Resource,
    format: Format,
) -> Option<End> {
    let Storage::Texture { name, target, immutable, .. } = res.storage else {
        return None;
    };
    let base = End { name, target, temporary: false };
    if res.args.format == format || !features.has(Feature::texture_view) || !res.supports_view() {
        return Some(base);
    }
    let (Some(te), Some(ve)) = (formats.get(res.args.format), formats.get(format)) else {
        return Some(base);
    };
    if te.gl.internalformat == ve.gl.internalformat
        || te.gl.view_class != ve.gl.view_class
        || te.gl.view_class == super::super::formats::ViewClass::Unsupported
        || !immutable
    {
        return Some(base);
    }
    let view = gl.gen_texture();
    gl.texture_view(
        view,
        target,
        name,
        ve.gl.internalformat,
        0,
        res.args.last_level + 1,
        0,
        res.args.array_size,
    );
    Some(End { name: view, target, temporary: true })
}

fn unattachable(cmd: Cmd, handle: ResourceHandle, e: transfer::Unattachable) -> Fault {
    match e {
        transfer::Unattachable::NotATexture => Fault::IllegalResource { cmd, handle },
        transfer::Unattachable::NoFeature(feature) => Fault::NoFeature { cmd, feature },
    }
}

/// `vrend_fb_bind_texture_id` for a named texture of a resource, on the bound framebuffer.
fn bind_fb_texture(
    host: &Host<'_>,
    cmd: Cmd,
    res: &Resource,
    end: &End,
    level: i32,
    layer: Option<GLint>,
) -> Result<(), Fault> {
    let attachment = transfer::attachment_for(res, host.formats);
    transfer::attach_texture(host.gl, host.features, end.target, end.name, attachment, level, layer)
        .map_err(|feature| Fault::NoFeature { cmd, feature })
}

fn detach_all(gl: &Gl) {
    gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, None, 0);
    gl.framebuffer_texture_2d(GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, None, 0);
}

impl Context {
    /// `vrend_renderer_blit`.
    pub(super) fn blit(&mut self, host: &mut Host<'_>, b: &Blit) -> Result<(), Fault> {
        let cmd = Cmd::Blit;
        let formats = host.formats;
        let (src, dst) = (b.src, b.dst);
        let src_res = host.resource(cmd, src.resource)?;
        let dst_res = host.resource(cmd, dst.resource)?;
        for f in [src.format, dst.format] {
            if formats.get(f).is_none() {
                return Err(Fault::IllegalFormat { cmd, format: f });
            }
        }
        if src_res.args.target == TextureTarget::Cube && src.region.depth + src.region.z > 6 {
            return Err(Fault::OutOfRange { cmd, what: "source cube face" });
        }
        if dst_res.args.target == TextureTarget::Cube && dst.region.depth + dst.region.z > 6 {
            return Err(Fault::OutOfRange { cmd, what: "destination cube face" });
        }
        let dst_depth =
            resource::minify(dst_res.args.depth, dst.level).max(dst_res.args.array_size) as i32;
        if dst.region.depth > dst_depth || dst.region.z > dst_depth {
            return Err(Fault::OutOfRange { cmd, what: "destination depth" });
        }
        let (sw, sh) = (src_res.width_at(src.level) as i32, src_res.height_at(src.level) as i32);
        let (dw, dh) = (dst_res.width_at(dst.level) as i32, dst_res.height_at(dst.level) as i32);
        let same_samples = src_res.args.nr_samples == dst_res.args.nr_samples;
        let condition_free = !b.render_condition_enable || self.sub().render_condition.is_none();
        // A resource that cannot be viewed gets its colourspace conversion from a shader, which
        // only the blitter has.
        let eglimage_copy_compatible =
            !(src_res.needs_srgb_decode(src.format) || dst_res.needs_srgb_encode(dst.format));
        let copy_path = host.has(Feature::copy_image)
            && condition_free
            && copy_compatible(formats, src.format, dst.format, false)
            && eglimage_copy_compatible
            && !b.scissor_enable
            && b.filter == TexFilter::Nearest
            && !b.alpha_blend
            && b.mask == PIPE_MASK_RGBA
            && same_samples
            && src.region.x + src.region.width <= sw
            && dst.region.x + dst.region.width <= dw
            && src.region.y + src.region.height <= sh
            && dst.region.y + dst.region.height <= dh
            && src.region.width == dst.region.width
            && src.region.height == dst.region.height
            && src.region.depth == dst.region.depth;
        if copy_path {
            return self.copy_sub_image(
                host,
                cmd,
                src.resource,
                src.level,
                src.region,
                dst.resource,
                dst.level,
                [dst.region.x, dst.region.y, dst.region.z],
            );
        }
        self.blit_int(host, b)
    }

    /// `vrend_copy_sub_image`.
    #[allow(clippy::too_many_arguments)]
    fn copy_sub_image(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
        src: ResourceHandle,
        src_level: u32,
        src_box: Box3,
        dst: ResourceHandle,
        dst_level: u32,
        dst_origin: [i32; 3],
    ) -> Result<(), Fault> {
        let src_res = host.resource(cmd, src)?;
        let dst_res = host.resource(cmd, dst)?;
        let (
            Storage::Texture { name: sn, target: st, .. },
            Storage::Texture { name: dn, target: dt, .. },
        ) = (&src_res.storage, &dst_res.storage)
        else {
            return Err(Fault::IllegalResource { cmd, handle: dst });
        };
        host.gl.copy_image_sub_data(
            *sn,
            *st,
            src_level as GLint,
            [src_box.x, src_box.y, src_box.z],
            *dn,
            *dt,
            dst_level as GLint,
            dst_origin,
            [src_box.width, src_box.height, src_box.depth],
        );
        Ok(())
    }

    /// `vrend_renderer_blit_int`: a framebuffer blit where one can serve, else the blitter.
    fn blit_int(&mut self, host: &mut Host<'_>, b: &Blit) -> Result<(), Fault> {
        let cmd = Cmd::Blit;
        let formats = host.formats;
        let (src_res, dst_res, redblue) = {
            let s = host.resource(cmd, b.src.resource)?;
            let d = host.resource(cmd, b.dst.resource)?;
            // `vrend_blit_needs_redblue_swizzle`: one end reads its red and blue swapped and the
            // other does not, so the blit has to swap them -- which only the blitter can.
            let redblue =
                s.needs_redblue_swizzle(b.src.format) != d.needs_redblue_swizzle(b.dst.format);
            ((s.args, s.y_0_top()), (d.args, d.y_0_top()), redblue)
        };
        let (gl, features) = (host.gl, host.features);
        let src_end =
            make_view(gl, features, formats, host.resource(cmd, b.src.resource)?, b.src.format);
        let dst_end =
            make_view(gl, features, formats, host.resource(cmd, b.dst.resource)?, b.dst.format);
        let (Some(src_end), Some(dst_end)) = (src_end, dst_end) else {
            return Err(Fault::IllegalResource { cmd, handle: b.dst.resource });
        };
        let cleanup = |host: &mut Host<'_>| {
            if src_end.temporary {
                host.gl.delete_texture(src_end.name);
            }
            if dst_end.temporary {
                host.gl.delete_texture(dst_end.name);
            }
        };
        // `vrend_renderer_prepare_blit_extra_info`.
        let mut can_fbo = true;
        let mut gl_filter = if b.filter == TexFilter::Nearest { GL_NEAREST } else { GL_LINEAR };
        let ys = |y_0_top: bool, height0: u32, y: i32, h: i32| {
            if !y_0_top { (y + h, y) } else { (height0 as i32 - y - h, height0 as i32 - y) }
        };
        let (dst_y1, dst_y2) = ys(dst_res.1, dst_res.0.height, b.dst.region.y, b.dst.region.height);
        let (src_y1, src_y2) = ys(src_res.1, src_res.0.height, b.src.region.y, b.src.region.height);
        if needs_swizzle(formats, b.dst.format, b.src.format) || redblue {
            can_fbo = false;
        }
        if b.mask & PIPE_MASK_RGBA != 0
            && src_res.0.nr_samples > 1
            && src_res.0.nr_samples != dst_res.0.nr_samples
            && (b.src.region.width != b.dst.region.width
                || b.src.region.height != b.dst.region.height)
        {
            if host.has(Feature::ms_scaled_blit) {
                gl_filter = GL_SCALED_RESOLVE_NICEST_EXT;
            } else {
                can_fbo = false;
            }
        }
        // `vrend_renderer_prepare_blit`.
        let src_entry = formats.get(src_res.0.format);
        let dst_entry = formats.get(dst_res.0.format);
        let src_ds = src_entry.is_some_and(|e| e.is_ds());
        let dst_ds = dst_entry.is_some_and(|e| e.is_ds());
        if !src_entry.is_some_and(|e| e.can_render()) && !src_ds {
            can_fbo = false;
        }
        if src_ds && dst_ds && src_res.0.format != dst_res.0.format {
            let pair = (src_res.0.format.name(), dst_res.0.format.name());
            if pair != ("S8_UINT_Z24_UNORM", "Z24X8_UNORM") {
                can_fbo = false;
            }
        }
        if b.mask & (PIPE_MASK_Z | PIPE_MASK_S) != 0 && gl_filter != GL_NEAREST {
            can_fbo = false;
        }
        if dst_res.0.nr_samples > 1
            || (b.mask & PIPE_MASK_RGBA != 0
                && src_res.0.nr_samples > 1
                && (b.src.region.x != b.dst.region.x
                    || b.src.region.width != b.dst.region.width
                    || dst_y1 != src_y1
                    || dst_y2 != src_y2
                    || b.src.format != b.dst.format))
        {
            can_fbo = false;
        }
        if b.src.region.depth != b.dst.region.depth {
            can_fbo = false;
        }
        if !can_fbo {
            let r = self.blit_shader(host, b, &src_end, &dst_end, redblue);
            cleanup(host);
            return r;
        }
        let r =
            self.blit_fbo(host, b, &src_end, &dst_end, gl_filter, [src_y1, src_y2, dst_y1, dst_y2]);
        cleanup(host);
        r
    }

    /// `vrend_renderer_blit_gl`'s caller half: resolve both ends into a [`blitter::Job`], run it
    /// in the blitter's own GL context, and put the caller's context back.
    fn blit_shader(
        &mut self,
        host: &mut Host<'_>,
        b: &Blit,
        src_end: &End,
        dst_end: &End,
        redblue: bool,
    ) -> Result<(), Fault> {
        let cmd = Cmd::Blit;
        // `blit_depth`: both ends carry depth and the guest asked for the Z channel, so this
        // blit writes `gl_FragDepth` and hangs its destination off the depth attachment. None of
        // the colour work below applies to it -- the swizzle, the sRGB pair -- and the C skips
        // computing any of it too.
        let src_desc = b.src.format.describe();
        let dst_desc = b.dst.format.describe();
        let color = !(src_desc.is_some_and(|d| d.has_depth())
            && dst_desc.is_some_and(|d| d.has_depth())
            && b.mask & PIPE_MASK_Z != 0);
        let formats = host.formats;
        let src_res = host.resource(cmd, b.src.resource)?;
        let dst_res = host.resource(cmd, b.dst.resource)?;
        // `vrend_renderer_prepare_blit_extra_info`'s swizzle: the destination's own stored order,
        // then red and blue traded if exactly one end is an IOSurface-backed BGRA texture.
        let mut swizzle = [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W];
        if needs_swizzle(formats, b.dst.format, b.src.format)
            && let Some(s) = formats.get(dst_res.args.format).and_then(|e| e.gl.swizzle)
        {
            swizzle = s;
        }
        if redblue {
            swizzle.swap(0, 2);
        }
        let identity = swizzle == [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W];
        let manual_srgb_decode = src_res.needs_srgb_decode(b.src.format);
        let manual_srgb_encode = dst_res.needs_srgb_encode(b.dst.format);
        // The C reads these two the wrong way round -- it fills `has_srgb_write_control` from
        // `feat_texture_srgb_decode` and `has_texture_srgb_decode` from `feat_srgb_write_control`.
        // Written straight here; the pinned score says whether the transposition is observable.
        let has_srgb_write_control = host.has(Feature::srgb_write_control);
        let has_texture_srgb_decode = host.has(Feature::texture_srgb_decode);
        let job = blitter::Job {
            src: src_end.name,
            src_gl_target: src_end.target,
            src_target: src_res.args.target,
            src_w: src_res.width_at(b.src.level),
            src_h: src_res.height_at(b.src.level),
            src_level: b.src.level,
            src_samples: src_res.args.nr_samples,
            src_format: b.src.format,
            src_table_swizzle: formats.get(b.src.format).and_then(|e| e.gl.swizzle),
            set_srgb_decode: has_texture_srgb_decode
                && !manual_srgb_decode
                && resource::is_srgb(b.src.format),
            filter: b.filter,
            src_box: (
                blitter::Point { x: b.src.region.x, y: b.src.region.y },
                b.src.region.width,
                b.src.region.height,
            ),
            src_z: b.src.region.z,
            src_depth: b.src.region.depth,
            src_texture_depth: src_res.depth_at(b.src.level),
            dst: dst_end.name,
            color,
            dst_gl_target: dst_end.target,
            dst_attachment: transfer::attachment_for(dst_res, formats),
            dst_target: dst_res.args.target,
            dst_w: dst_res.width_at(b.dst.level),
            dst_h: dst_res.height_at(b.dst.level),
            dst_level: b.dst.level,
            dst_layer: b.dst.region.z,
            dst_box: (
                blitter::Point { x: b.dst.region.x, y: b.dst.region.y },
                b.dst.region.width,
                b.dst.region.height,
            ),
            dst_depth: b.dst.region.depth,
            swizzle: (!identity).then_some(swizzle),
            manual_srgb_decode,
            manual_srgb_encode,
            framebuffer_srgb: has_srgb_write_control.then(|| {
                !manual_srgb_encode
                    && (resource::is_srgb(b.dst.format) || resource::is_srgb(b.src.format))
            }),
            scissor: b.scissor_enable.then(|| {
                let s = b.scissor;
                [
                    s.minx as GLint,
                    s.miny as GLint,
                    s.maxx as GLint - s.minx as GLint,
                    s.maxy as GLint - s.miny as GLint,
                ]
            }),
        };
        let (gl, winsys, features) = (host.gl, host.winsys, host.features);
        if host.blitter.is_none() {
            match blitter::Blitter::open(winsys, gl, host.version, host.share) {
                Ok(b) => *host.blitter = Some(b),
                Err(e) => {
                    eprintln!("[virglrs] vrend: no GL context for the blitter ({e}); no blit");
                    host.todo.note("the shader blitter");
                    return Ok(());
                }
            }
            // `Blitter::open` left its own context current.
            *host.current = Current::Blitter;
        }
        let blitter = host.blitter.as_mut().expect("just built");
        winsys.make_current(blitter.context()).expect("the blitter's context can be made current");
        *host.current = Current::Blitter;
        let outcome = blitter.run(gl, features, &job);
        // Unconditionally, and before anything else can run: the commands after this one in the
        // batch do not switch contexts, they assume the sub-context's is current.
        self.make_current(host);
        match outcome {
            Ok(()) => Ok(()),
            Err(blitter::Unserved::NoFeature(feature)) => Err(Fault::NoFeature { cmd, feature }),
            Err(blitter::Unserved::NoProgram) => {
                host.todo.note("the shader blitter");
                Ok(())
            }
        }
    }

    /// `vrend_renderer_blit_fbo`.
    fn blit_fbo(
        &mut self,
        host: &mut Host<'_>,
        b: &Blit,
        src_end: &End,
        dst_end: &End,
        gl_filter: GLenum,
        y: [i32; 4],
    ) -> Result<(), Fault> {
        let cmd = Cmd::Blit;
        let gl = host.gl;
        let [src_y1, src_y2, dst_y1, dst_y2] = y;
        let mut glmask: GLbitfield = 0;
        if b.mask & PIPE_MASK_Z != 0 {
            glmask |= GL_DEPTH_BUFFER_BIT;
        }
        if b.mask & PIPE_MASK_S != 0 {
            glmask |= GL_STENCIL_BUFFER_BIT;
        }
        if b.mask & PIPE_MASK_RGBA != 0 {
            glmask |= GL_COLOR_BUFFER_BIT;
        }
        if b.scissor_enable {
            let s = b.scissor;
            gl.scissor(
                s.minx as GLint,
                s.miny as GLint,
                s.maxx as GLsizei - s.minx as GLsizei,
                s.maxy as GLsizei - s.miny as GLsizei,
            );
            self.sub_mut().scissor_dirty = Dirty::just(0);
            gl.enable(GL_SCISSOR_TEST);
        } else {
            gl.disable(GL_SCISSOR_TEST);
        }
        let src_ms = host.resource(cmd, b.src.resource)?.args.nr_samples;
        let dst_ms = host.resource(cmd, b.dst.resource)?.args.nr_samples;
        if b.mask & (PIPE_MASK_Z | PIPE_MASK_S) != 0
            && src_ms > 1
            && src_ms != dst_ms
            && (b.src.region.x != b.dst.region.x
                || src_y1 != dst_y1
                || b.src.region.width != b.dst.region.width
                || src_y2 != dst_y2)
        {
            host.todo.note("multisample depth blits through an intermediate copy");
            return Err(Fault::Unimplemented {
                cmd,
                what: "a resolving depth blit with differing rectangles",
            });
        }
        let [fb0, fb1] = self.sub().blit_fbs;
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb0));
        if b.mask & PIPE_MASK_RGBA != 0 {
            gl.framebuffer_texture_2d(GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, None, 0);
        } else {
            gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, None, 0);
        }
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb1));
        if b.mask & PIPE_MASK_RGBA != 0 {
            gl.framebuffer_texture_2d(GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, None, 0);
        } else if b.mask & (PIPE_MASK_Z | PIPE_MASK_S) != 0 {
            gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, None, 0);
        }
        let srgb_control = host.has(Feature::srgb_write_control);
        let any_srgb =
            [b.src.format, b.dst.format].iter().any(|f| f.describe().is_some_and(|d| d.is_srgb()));
        let n_layers =
            if b.src.region.depth == b.dst.region.depth { b.dst.region.depth } else { 1 };
        for i in 0..n_layers {
            gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb0));
            let src_res = host.resource(cmd, b.src.resource)?;
            bind_fb_texture(
                host,
                cmd,
                src_res,
                src_end,
                b.src.level as GLint,
                Some(b.src.region.z + i),
            )?;
            gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb1));
            let dst_res = host.resource(cmd, b.dst.resource)?;
            bind_fb_texture(
                host,
                cmd,
                dst_res,
                dst_end,
                b.dst.level as GLint,
                Some(b.dst.region.z + i),
            )?;
            gl.bind_framebuffer(GL_DRAW_FRAMEBUFFER, Some(fb1));
            if srgb_control {
                gl.set_enabled(GL_FRAMEBUFFER_SRGB_EXT, any_srgb);
            }
            gl.bind_framebuffer(GL_READ_FRAMEBUFFER, Some(fb0));
            gl.blit_framebuffer(
                [b.src.region.x, src_y1, b.src.region.x + b.src.region.width, src_y2],
                [b.dst.region.x, dst_y1, b.dst.region.x + b.dst.region.width, dst_y2],
                glmask,
                gl_filter,
            );
        }
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb1));
        detach_all(gl);
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb0));
        detach_all(gl);
        let sub = self.sub();
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(sub.fb));
        if srgb_control {
            gl.set_enabled(GL_FRAMEBUFFER_SRGB_EXT, sub.framebuffer_srgb_enabled);
        }
        gl.set_enabled(GL_SCISSOR_TEST, sub.rs_state().scissor);
        Ok(())
    }

    /// `vrend_renderer_resource_copy_region`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_region(
        &mut self,
        host: &mut Host<'_>,
        dst: ResourceHandle,
        dst_level: u32,
        dst_origin: [u32; 3],
        src: ResourceHandle,
        src_level: u32,
        src_box: Box3,
    ) -> Result<(), Fault> {
        let cmd = Cmd::ResourceCopyRegion;
        let gl = host.gl;
        let formats = host.formats;
        let src_res = host.resource(cmd, src)?;
        let dst_res = host.resource(cmd, dst)?;
        if !transfer::contains_box(src_res, &src_box, src_level) {
            return Err(Fault::OutOfRange { cmd, what: "source box" });
        }
        let mut dst_box = Box3 {
            x: dst_origin[0] as i32,
            y: dst_origin[1] as i32,
            z: dst_origin[2] as i32,
            ..src_box
        };
        let bw = |r: &Resource| {
            r.args.format.describe().map_or((1, 1), |d| (d.block.width, d.block.height))
        };
        let ((sbw, sbh), (dbw, dbh)) = (bw(src_res), bw(dst_res));
        if sbw > 1 && dbw == 1 {
            dst_box.width /= sbw as i32;
            dst_box.height /= sbh as i32;
        } else if sbw == 1 && dbw > 1 {
            dst_box.width *= dbw as i32;
            dst_box.height *= dbh as i32;
        }
        if !transfer::contains_box(dst_res, &dst_box, dst_level) {
            return Err(Fault::OutOfRange { cmd, what: "destination box" });
        }
        if let (Storage::Buffer { name: sn, .. }, Storage::Buffer { name: dn, .. }) =
            (&src_res.storage, &dst_res.storage)
        {
            gl.bind_buffer(GL_COPY_READ_BUFFER, Some(*sn));
            gl.bind_buffer(GL_COPY_WRITE_BUFFER, Some(*dn));
            gl.copy_buffer_sub_data(
                src_box.x as usize,
                dst_origin[0] as usize,
                src_box.width as usize,
            );
            gl.bind_buffer(GL_COPY_READ_BUFFER, None);
            gl.bind_buffer(GL_COPY_WRITE_BUFFER, None);
            return Ok(());
        }
        let (sf, df) = (src_res.args.format, dst_res.args.format);
        if host.has(Feature::copy_image)
            && copy_compatible(formats, sf, df, true)
            && src_res.args.nr_samples == dst_res.args.nr_samples
        {
            let origin = [dst_origin[0] as i32, dst_origin[1] as i32, dst_origin[2] as i32];
            return self.copy_sub_image(host, cmd, src, src_level, src_box, dst, dst_level, origin);
        }
        let can_render = |f: Format| formats.get(f).is_some_and(|e| e.can_render());
        if !can_render(sf) || !can_render(df) {
            host.todo.note("the resource copy fallback through guest memory");
            return Err(Fault::Unimplemented { cmd, what: "a copy between unrenderable formats" });
        }
        // The framebuffer blit.
        let (src_y0top, src_h) = (src_res.y_0_top(), src_res.args.height as i32);
        let (dst_y0top, dst_h) = (dst_res.y_0_top(), dst_res.args.height as i32);
        let [fb0, fb1] = self.sub().blit_fbs;
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb0));
        gl.framebuffer_texture_2d(GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, None, 0);
        let src_res = host.resource(cmd, src)?;
        transfer::attach(
            gl,
            host.features,
            src_res,
            transfer::attachment_for(src_res, formats),
            src_level as GLint,
            Some(src_box.z),
        )
        .map_err(|e| unattachable(cmd, src, e))?;
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb1));
        gl.framebuffer_texture_2d(GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, None, 0);
        let dst_res = host.resource(cmd, dst)?;
        transfer::attach(
            gl,
            host.features,
            dst_res,
            transfer::attachment_for(dst_res, formats),
            dst_level as GLint,
            Some(dst_origin[2] as GLint),
        )
        .map_err(|e| unattachable(cmd, dst, e))?;
        gl.bind_framebuffer(GL_DRAW_FRAMEBUFFER, Some(fb1));
        gl.bind_framebuffer(GL_READ_FRAMEBUFFER, Some(fb0));
        gl.disable(GL_SCISSOR_TEST);
        let (sy1, sy2) = if !src_y0top {
            (src_box.y, src_box.y + src_box.height)
        } else {
            (src_h - src_box.y - src_box.height, src_h - src_box.y)
        };
        let dy = dst_origin[1] as i32;
        let (dy1, dy2) = if !dst_y0top {
            (dy, dy + src_box.height)
        } else {
            (dst_h - dy - src_box.height, dst_h - dy)
        };
        let dx = dst_origin[0] as i32;
        gl.blit_framebuffer(
            [src_box.x, sy1, src_box.x + src_box.width, sy2],
            [dx, dy1, dx + src_box.width, dy2],
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST,
        );
        // The C detaches through GL_FRAMEBUFFER twice here, which is the draw framebuffer both
        // times; the read framebuffer keeps its colour attachment. Reproduced as it is.
        gl.bind_framebuffer(GL_READ_FRAMEBUFFER, Some(fb0));
        gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, None, 0);
        gl.bind_framebuffer(GL_DRAW_FRAMEBUFFER, Some(fb1));
        gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, None, 0);
        let sub = self.sub();
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(sub.fb));
        if sub.rs_state().scissor {
            gl.enable(GL_SCISSOR_TEST);
        }
        Ok(())
    }

    /// `vrend_clear_surface`.
    pub(super) fn clear_surface(
        &mut self,
        host: &mut Host<'_>,
        surface: ObjectHandle,
        buffers: u32,
        color: [u32; 4],
        rect: [u32; 4],
    ) -> Result<(), Fault> {
        let cmd = Cmd::ClearSurface;
        let gl = host.gl;
        let (resource, format, level, layer, view) = {
            let s = self.sub().surface(cmd, surface)?;
            (s.resource, s.format, s.level, s.layer(), s.view)
        };
        let entry = host.formats.get(format).ok_or(Fault::IllegalFormat { cmd, format })?;
        if !entry.can_render() && !entry.is_ds() {
            return Err(Fault::IllegalFormat { cmd, format });
        }
        gl.scissor(rect[0] as GLint, rect[1] as GLint, rect[2] as GLsizei, rect[3] as GLsizei);
        gl.enable(GL_SCISSOR_TEST);
        self.sub_mut().scissor_dirty = Dirty::just(0);
        let fb0 = self.sub().blit_fbs[0];
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb0));
        let res = host.resource(cmd, resource)?;
        let Storage::Texture { name, target, .. } = res.storage else {
            return Err(Fault::IllegalResource { cmd, handle: resource });
        };
        let name = match view {
            None => name,
            Some(key) => {
                res.view_texture(key).ok_or(Fault::IllegalResource { cmd, handle: resource })?
            }
        };
        let end = End { name, target, temporary: false };
        bind_fb_texture(host, cmd, res, &end, level as GLint, layer)?;
        let colorf = color.map(f32::from_bits);
        let depth = f64::from_bits(color[0] as u64 | (color[1] as u64) << 32);
        let stencil = color[3];
        self.clear_prepare(host, Some((resource, format)), buffers, colorf, depth, stencil)?;
        let mut bits: GLbitfield = 0;
        if buffers & PIPE_CLEAR_COLOR0 != 0 {
            bits |= GL_COLOR_BUFFER_BIT;
        }
        if buffers & PIPE_CLEAR_DEPTH != 0 {
            bits |= GL_DEPTH_BUFFER_BIT;
        }
        if buffers & PIPE_CLEAR_STENCIL != 0 {
            bits |= GL_STENCIL_BUFFER_BIT;
        }
        gl.clear(bits);
        self.clear_finish(host, buffers);
        detach_all(gl);
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(self.sub().fb));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(name: &str) -> Format {
        (0..FORMAT_MAX)
            .map(|n| Format::from_wire(n).unwrap())
            .find(|f| f.name() == name)
            .unwrap_or_else(|| panic!("no format {name}"))
    }

    #[test]
    fn plain_formats_are_copy_compatible_when_gallium_says_so() {
        let t = Table::empty();
        let compat = |a: &str, b: &str| copy_compatible(&t, format(a), format(b), false);
        assert!(compat("R8G8B8A8_UNORM", "R8G8B8A8_UNORM"));
        // An X channel is a swizzle of 1, which the compare skips.
        assert!(compat("R8G8B8A8_UNORM", "R8G8B8X8_UNORM"));
        // Different channel order.
        assert!(!compat("R8G8B8A8_UNORM", "B8G8R8A8_UNORM"));
        // Different colourspace.
        assert!(!compat("R8G8B8A8_UNORM", "R8G8B8A8_SRGB"));
        // Different type in a channel.
        assert!(!compat("R8G8B8A8_UNORM", "R8G8B8A8_UINT"));
        // Compressed pairs are never copy-compatible for a blit.
        assert!(!compat("DXT1_RGB", "DXT1_SRGB"));
    }
}
