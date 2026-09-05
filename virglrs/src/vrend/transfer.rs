// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Transfers: guest pages to a host resource and back.
//!
//! Every transfer goes through a tightly packed host buffer. The C hands the driver a pointer
//! into guest memory whenever it can, and pays for it with a pixel-store state machine, a
//! zero-copy path with different rules from the copying path, and a depth rescale that runs over
//! whatever guest memory follows the box. Here the box is gathered from the pages by stride,
//! worked on as a slice, and handed to the driver tight; the readback is the mirror. What the
//! guest sees in its pages is the same bytes at the same places -- the stride and layer-stride
//! contract of `harness/README.md`'s invariants -- with one copy in between.
//!
//! Bounds are checked in `u64` and refused, never truncated: the C computes its sizes in
//! `GLuint`, and a box past 4 GiB wraps into an accepted transfer.

use super::features::{Feature, Features};
use super::formats::{Entry, Table};
use super::gl::gles::*;
use super::gl::{GLenum, GLint, GLsizei, Gl, TextureName, pixel_bytes};
use super::proto::Box3;
use super::resource::{Resource, Storage};
use crate::guest_mem::Iov;
use std::fmt;

/// Where a transfer lands in the resource, and how the guest laid it out in the pages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Info {
    pub level: u32,
    /// Bytes between rows in the pages; zero for the level's tight pitch.
    pub stride: u32,
    /// Bytes between layers in the pages; zero for the level's tight 2D size.
    pub layer_stride: u32,
    /// Where the box starts in the pages.
    pub offset: u64,
    pub region: Box3,
    /// Only a `COPY_TRANSFER3D` sets this; it decides whether a buffer write may orphan.
    pub synchronized: bool,
}

/// Why a transfer did not happen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// The box is not inside the resource at that level.
    BoxOutOfRange,
    /// The pages do not hold the bytes the box and strides describe.
    IovOutOfRange,
    /// The resource has no pages attached and none were given.
    NoPages,
    /// A shape the host cannot serve: a format with no GL triple, a multisample upload.
    Unsupported,
    /// The format cannot be read back through GL, and the destination is not the resource's own
    /// pages -- so there is nothing to say the guest does not already hold.
    NotReadable,
    /// The driver refused.
    GlError(GLenum),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BoxOutOfRange => f.write_str("the box is outside the resource"),
            Error::IovOutOfRange => f.write_str("the pages do not hold the transfer"),
            Error::NoPages => f.write_str("the resource has no pages attached"),
            Error::Unsupported => f.write_str("the host cannot serve that transfer"),
            Error::NotReadable => f.write_str("the format cannot be read back"),
            Error::GlError(e) => write!(f, "GL error {e:#x}"),
        }
    }
}

/// The box's geometry in blocks and bytes, settled once from the format and the level.
#[derive(Clone)]
struct Layout {
    /// Bytes per block.
    block: u64,
    /// Blocks across the box.
    blocks_wide: u64,
    /// Block rows down the box.
    blocks_high: u64,
    depth: u64,
    /// The row pitch in the pages.
    stride: u64,
    /// The layer pitch in the pages.
    layer_stride: u64,
    compressed: bool,
}

impl Layout {
    /// Tight bytes per row of the box.
    fn row(&self) -> u64 {
        self.blocks_wide * self.block
    }

    /// Tight bytes per layer of the box.
    fn layer(&self) -> u64 {
        self.row() * self.blocks_high
    }

    /// Tight bytes for the whole box.
    fn total(&self) -> u64 {
        self.layer() * self.depth
    }

    /// `vrend_transfer_size`: the bytes the box spans in the pages, last row and layer tight.
    fn span(&self) -> u64 {
        (self.depth - 1) * self.layer_stride + (self.blocks_high - 1) * self.stride + self.row()
    }

    /// The layout the driver reads and writes, which is not always the one gallium describes.
    ///
    /// The bounds check is gallium's -- a block of `block_bytes` per `block.width` pixels -- but
    /// the driver moves `w` pixels of `pixel_bytes(glformat, gltype)` each, and the C's
    /// zero-copy path hands it the guest's pointer with a row length of `stride / block_bytes`
    /// *pixels*. Where the two sizes agree, which is every plain format, this is the identity.
    /// Where they do not -- `R8G8_R8B8_422_UNORM`, which the guest creates as a 4-byte-per-pixel
    /// RGBA8 and gallium describes as 4 bytes per two pixels; `X24S8_UINT`, 4 bytes described and
    /// one byte moved -- the bytes land where the C's driver call put them, at the driver's pitch.
    /// Compressed formats hand the driver the described bytes as-is.
    fn as_gl(&self, entry: &Entry, w: u64, h: u64) -> Layout {
        if self.compressed {
            return self.clone();
        }
        let Some(px) = pixel_bytes(entry.gl.glformat, entry.gl.gltype) else {
            return self.clone();
        };
        let px = px as u64;
        let pitch = self.stride / self.block * px;
        Layout {
            block: px,
            blocks_wide: w,
            blocks_high: h,
            depth: self.depth,
            stride: pitch,
            layer_stride: self.layer_stride / self.stride * pitch,
            compressed: false,
        }
    }
}

/// `resource_contains_box`.
pub fn contains_box(res: &Resource, b: &Box3, level: u32) -> bool {
    if level > res.args.last_level {
        return false;
    }
    let extent = [res.width_at(level), res.height_at(level), res.depth_at(level)];
    let start = [b.x, b.y, b.z];
    let size = [b.width, b.height, b.depth];
    (0..3).all(|i| {
        start[i] >= 0
            && size[i] >= 0
            && (start[i] as u32) <= extent[i]
            && (start[i] as i64 + size[i] as i64) <= extent[i] as i64
    })
}

/// `check_iov_bounds`, with the layout it computes on the way.
fn layout(res: &Resource, info: &Info, pages: &Iov<'_>) -> Result<Layout, Error> {
    let desc = res.args.format.describe().ok_or(Error::Unsupported)?;
    let b = &info.region;
    let (w, h, d) = (b.width.max(1) as u32, b.height.max(1) as u32, b.depth.max(1) as u64);
    let block = desc.block_bytes() as u64;
    let blocks_wide = desc.blocks_wide(w) as u64;
    let blocks_high = desc.blocks_high(h) as u64;
    let stride = if info.stride != 0 {
        if (info.stride as u64) < blocks_wide * block {
            return Err(Error::IovOutOfRange);
        }
        info.stride as u64
    } else {
        desc.stride(res.width_at(info.level)) as u64
    };
    let layer_stride = if info.layer_stride != 0 {
        if (info.layer_stride as u64) < blocks_high * stride {
            return Err(Error::IovOutOfRange);
        }
        info.layer_stride as u64
    } else {
        desc.blocks_high(res.height_at(info.level)) as u64 * stride
    };
    let l = Layout {
        block,
        blocks_wide,
        blocks_high,
        depth: d,
        stride,
        layer_stride,
        compressed: desc.is_compressed(),
    };
    let size = pages.len();
    let end = info.offset.checked_add(l.span()).ok_or(Error::IovOutOfRange)?;
    if end > size {
        return Err(Error::IovOutOfRange);
    }
    Ok(l)
}

/// Gather the box out of the pages into a tight buffer, rows in the order they are in the
/// pages. `false` if a row fell outside the pages, which `layout` has already ruled out.
fn gather(pages: &Iov<'_>, info: &Info, l: &Layout, out: &mut [u8]) -> bool {
    let row = l.row() as usize;
    for d in 0..l.depth {
        for r in 0..l.blocks_high {
            let at = info.offset + d * l.layer_stride + r * l.stride;
            let into = ((d * l.blocks_high + r) as usize) * row;
            if !pages.copy_out(at, &mut out[into..into + row]) {
                return false;
            }
        }
    }
    true
}

/// Scatter a tight buffer into the pages, the inverse of [`gather`].
fn scatter(pages: &Iov<'_>, info: &Info, l: &Layout, data: &[u8]) -> bool {
    let row = l.row() as usize;
    for d in 0..l.depth {
        for r in 0..l.blocks_high {
            let at = info.offset + d * l.layer_stride + r * l.stride;
            let from = ((d * l.blocks_high + r) as usize) * row;
            if !pages.copy_in(at, &data[from..from + row]) {
                return false;
            }
        }
    }
    true
}

/// Reverse the row order of every layer in place: a `y_0_top` resource's rows are stored the
/// other way up from GL's.
fn flip_rows(data: &mut [u8], l: &Layout) {
    let row = l.row() as usize;
    let layer = l.layer() as usize;
    for d in 0..l.depth as usize {
        let rows = &mut data[d * layer..(d + 1) * layer];
        let n = l.blocks_high as usize;
        for r in 0..n / 2 {
            let (a, b) = rows.split_at_mut((n - 1 - r) * row);
            a[r * row..(r + 1) * row].swap_with_slice(&mut b[..row]);
        }
    }
}

/// `vrend_swizzle_data_bgra`: swap the first and third byte of every four.
fn swizzle_bgra(data: &mut [u8]) {
    for px in data.as_chunks_mut::<4>().0 {
        px.swap(0, 2);
    }
}

/// `vrend_scale_depth`: `Z24X8_UNORM` travels with its depth in the high 24 bits, and GL wants
/// it scaled by 256 one way and back the other. The float round trip is the C's, rounding and
/// all, because the golden readbacks carry it.
fn scale_depth(data: &mut [u8], scale: f32) {
    const MYSCALE: f32 = 1.0 / 0xff_ffff as f32;
    for word in data.as_chunks_mut::<4>().0 {
        let v = u32::from_ne_bytes(*word);
        let d = (((v >> 8) as f32) * MYSCALE * scale).clamp(0.0, 1.0);
        *word = (((d / MYSCALE) as u32) << 8).to_ne_bytes();
    }
}

/// The GL target the FBO attaches, and the `x`/`y` of the box for the driver: cube faces and
/// arrays go by layer, and `y_0_top` flips the row the box starts at.
fn upload_y(res: &Resource, b: &Box3, invert: bool) -> GLint {
    if invert {
        // The C uses the base height here whatever the level; kept, because the goldens
        // were recorded through it.
        res.args.height as GLint - b.y - b.height
    } else {
        b.y
    }
}

/// Copy the box from the pages into the resource: `vrend_renderer_transfer_write_iov`.
///
/// `pages` is where the bytes come from -- the resource's own pages for a `TRANSFER3D`, another
/// resource's for a `COPY_TRANSFER3D`; `own` is the resource's own pages, which a host-side
/// buffer mirrors.
pub fn write(
    gl: &Gl,
    formats: &Table,
    res: &mut Resource,
    own: Option<&Iov<'_>>,
    pages: &Iov<'_>,
    info: &Info,
) -> Result<(), Error> {
    let b = info.region;
    if !contains_box(res, &b, info.level) {
        return Err(Error::BoxOutOfRange);
    }
    let l = layout(res, info, pages)?;
    match &mut res.storage {
        Storage::Guest => {
            // The guest's own pages are the storage: a transfer from them to themselves is
            // done, and one from another resource's pages is a copy between the two.
            if let Some(own) = own
                && !own.same_pages(pages)
            {
                let mut tmp = vec![0u8; b.width as usize];
                if !pages.copy_out(info.offset, &mut tmp) || !own.copy_in(b.x as u64, &tmp) {
                    return Err(Error::IovOutOfRange);
                }
            }
            Ok(())
        }
        Storage::Host(shadow) => {
            let (x, w) = (b.x as usize, b.width as usize);
            let from_own_backing = own.is_some_and(|own| own.same_pages(pages));
            let dst = shadow.bytes_mut().get_mut(x..x + w).ok_or(Error::BoxOutOfRange)?;
            if !pages.copy_out(info.offset, dst) {
                return Err(Error::IovOutOfRange);
            }
            // Where the bytes came from decides who holds the truth now. Written from the
            // resource's own backing, the pages already have them; written from anywhere else --
            // another resource's pages, or an iov the API path supplied to a resource with no
            // backing at all -- they do not, and the next attach owes them.
            if from_own_backing {
                shadow.mirrored();
            } else {
                shadow.unmirrored();
            }
            Ok(())
        }
        Storage::Buffer { name, target, .. } => {
            let (name, target) = (*name, *target);
            let (x, w) = (b.x as usize, b.width as usize);
            let mut flags = GL_MAP_INVALIDATE_RANGE_BIT;
            if !info.synchronized {
                // limina's rule: a write from the start of the buffer orphans it, so a guest
                // rewriting a buffer every frame never stalls on the frame still reading it;
                // any other unsynchronized write is a plain unsynchronized map.
                if x == 0 {
                    flags = GL_MAP_INVALIDATE_BUFFER_BIT;
                } else {
                    flags |= GL_MAP_UNSYNCHRONIZED_BIT;
                }
            }
            gl.bind_buffer(target, Some(name));
            gl.drain_errors();
            let mapped =
                gl.map_buffer_write(target, x, w, flags, |dst| pages.copy_out(info.offset, dst));
            let ok = match mapped {
                Some(ok) => ok,
                None => {
                    let mut tmp = vec![0u8; w];
                    pages.copy_out(info.offset, &mut tmp) && gl.buffer_sub_data(target, x, &tmp)
                }
            };
            let err = gl.drain_errors();
            gl.bind_buffer(target, None);
            if !ok {
                return Err(Error::IovOutOfRange);
            }
            if err != GL_NO_ERROR {
                return Err(Error::GlError(err));
            }
            Ok(())
        }
        Storage::Texture(t) => {
            let (name, target) = (t.name, t.target);
            if matches!(target, GL_TEXTURE_2D_MULTISAMPLE | GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
                return Err(Error::Unsupported);
            }
            let entry = res.entry(formats).ok_or(Error::Unsupported)?;
            let format_name = res.args.format.name();
            let l = l.as_gl(entry, b.width as u64, b.height as u64);
            let total = usize::try_from(l.total()).map_err(|_| Error::IovOutOfRange)?;
            let mut data = vec![0u8; total];
            if !gather(pages, info, &l, &mut data) {
                return Err(Error::IovOutOfRange);
            }
            let invert = res.y_0_top();
            if invert {
                flip_rows(&mut data, &l);
            }
            if res.is_bgra() {
                swizzle_bgra(&mut data);
            }
            if format_name == "Z24X8_UNORM" {
                scale_depth(&mut data, 256.0);
            }
            let (x, y) = (b.x, upload_y(res, &b, invert));
            let (w, h, d) = (b.width, b.height, b.depth);
            gl.use_program_none();
            gl.bind_texture(target, Some(name));
            gl.unpack_tight();
            gl.drain_errors();
            let (glformat, gltype, ifmt) =
                (entry.gl.glformat, entry.gl.gltype, entry.gl.internalformat);
            let sent = match target {
                GL_TEXTURE_CUBE_MAP => {
                    // One face per upload: the C refuses a cube box deeper than one face.
                    if d != 1 {
                        gl.bind_texture(target, None);
                        return Err(Error::Unsupported);
                    }
                    let face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + b.z as GLenum;
                    if l.compressed {
                        gl.compressed_tex_sub_image_2d(
                            face,
                            info.level as GLint,
                            x,
                            y,
                            w,
                            h,
                            ifmt,
                            &data,
                        )
                    } else {
                        gl.tex_sub_image_2d(
                            face,
                            info.level as GLint,
                            x,
                            y,
                            w,
                            h,
                            glformat,
                            gltype,
                            &data,
                        )
                    }
                }
                GL_TEXTURE_3D | GL_TEXTURE_2D_ARRAY | GL_TEXTURE_CUBE_MAP_ARRAY => {
                    let lv = info.level as GLint;
                    if l.compressed {
                        gl.compressed_tex_sub_image_3d(target, lv, x, y, b.z, w, h, d, ifmt, &data)
                    } else {
                        gl.tex_sub_image_3d(target, lv, x, y, b.z, w, h, d, glformat, gltype, &data)
                    }
                }
                _ => {
                    let lv = info.level as GLint;
                    if l.compressed {
                        gl.compressed_tex_sub_image_2d(target, lv, x, y, w, h, ifmt, &data)
                    } else {
                        gl.tex_sub_image_2d(target, lv, x, y, w, h, glformat, gltype, &data)
                    }
                }
            };
            let err = gl.drain_errors();
            gl.bind_texture(target, None);
            if !sent {
                return Err(Error::Unsupported);
            }
            if err != GL_NO_ERROR {
                return Err(Error::GlError(err));
            }
            // A surface-backed resource has no host copy: the upload above is queued on this
            // context's queue and the consumer -- a venus context that imported the surface --
            // submits on its own, and Metal does not order work across queues. The guest's fence
            // for this transfer signals when this returns, so this is where the upload has to
            // have landed.
            if res.surface().is_some() {
                gl.finish();
            }
            Ok(())
        }
    }
}

/// The framebuffer attachment a format's texture goes on: `vrend_fb_bind_texture_id`.
pub fn attachment_for(res: &Resource, formats: &Table) -> GLenum {
    let entry = res.entry(formats);
    let desc = res.args.format.describe();
    if entry.is_some_and(|e| e.is_ds()) {
        match desc.map(|d| (d.has_depth(), d.has_stencil())) {
            Some((true, true)) => GL_DEPTH_STENCIL_ATTACHMENT,
            Some((false, true)) => GL_STENCIL_ATTACHMENT,
            _ => GL_DEPTH_ATTACHMENT,
        }
    } else {
        GL_COLOR_ATTACHMENT0
    }
}

/// Why a texture could not be attached to a framebuffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unattachable {
    /// The resource has no texture behind it.
    NotATexture,
    /// The shape needs an entry point this host lacks.
    NoFeature(Feature),
}

/// Attach `level` of a texture to the bound framebuffer at `attachment`: one `layer` of it, or
/// every layer when `None` (`vrend_fb_bind_texture_id` with a layer of -1).
pub fn attach(
    gl: &Gl,
    features: &Features,
    res: &Resource,
    attachment: GLenum,
    level: GLint,
    layer: Option<GLint>,
) -> Result<(), Unattachable> {
    let Storage::Texture(t) = &res.storage else {
        return Err(Unattachable::NotATexture);
    };
    attach_texture(gl, features, t.target, t.name, attachment, level, layer)
        .map_err(Unattachable::NoFeature)
}

/// [`attach`] for a texture named directly -- a resource's own, or a view of it. Every layer
/// of a layered texture at once needs `glFramebufferTexture`, which GLES has with geometry
/// shaders; one slice of a 3D texture needs `GL_OES_texture_3D`.
#[allow(clippy::too_many_arguments)]
pub fn attach_texture(
    gl: &Gl,
    features: &Features,
    target: GLenum,
    name: TextureName,
    attachment: GLenum,
    level: GLint,
    layer: Option<GLint>,
) -> Result<(), Feature> {
    let need = |feature: Feature| if features.has(feature) { Ok(()) } else { Err(feature) };
    let name = &name;
    match (target, layer) {
        (
            GL_TEXTURE_2D_ARRAY
            | GL_TEXTURE_2D_MULTISAMPLE_ARRAY
            | GL_TEXTURE_CUBE_MAP_ARRAY
            | GL_TEXTURE_3D
            | GL_TEXTURE_CUBE_MAP,
            None,
        ) => {
            need(Feature::geometry_shader)?;
            gl.framebuffer_texture(attachment, Some(*name), level);
        }
        (
            GL_TEXTURE_2D_ARRAY | GL_TEXTURE_2D_MULTISAMPLE_ARRAY | GL_TEXTURE_CUBE_MAP_ARRAY,
            Some(layer),
        ) => gl.framebuffer_texture_layer(attachment, Some(*name), level, layer),
        (GL_TEXTURE_3D, Some(layer)) => {
            need(Feature::texture_3d_attach)?;
            gl.framebuffer_texture_3d(attachment, Some(*name), level, layer);
        }
        (GL_TEXTURE_CUBE_MAP, Some(layer)) => gl.framebuffer_texture_2d(
            attachment,
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer as GLenum,
            Some(*name),
            level,
        ),
        (t, _) => gl.framebuffer_texture_2d(attachment, t, Some(*name), level),
    }
    if attachment == GL_DEPTH_ATTACHMENT {
        gl.framebuffer_texture_2d(GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, None, 0);
    }
    Ok(())
}

/// Read one layer of the box out of a texture through a framebuffer: `do_readpixels`.
#[allow(clippy::too_many_arguments)]
fn read_layer(
    gl: &Gl,
    features: &Features,
    formats: &Table,
    res: &Resource,
    level: u32,
    layer: GLint,
    x: GLint,
    y: GLint,
    w: GLsizei,
    h: GLsizei,
    dst: &mut [u8],
) -> Result<(), Error> {
    let entry = res.entry(formats).ok_or(Error::Unsupported)?;
    // A readback borrows the framebuffer binding and has to give it back. The guest is entitled
    // to transfer out of a resource between binding its framebuffer and drawing into it, and a
    // readback that left the default framebuffer bound would send that draw nowhere -- on a
    // surfaceless context the default framebuffer is not complete at all.
    let previous = gl.framebuffer_binding();
    let fb = gl.gen_framebuffer();
    gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
    let attachment = attachment_for(res, formats);
    let attached = attach(gl, features, res, attachment, level as GLint, Some(layer)).is_ok();
    gl.drain_errors();
    let read = attached && gl.read_pixels(x, y, w, h, entry.gl.glformat, entry.gl.gltype, dst);
    let err = gl.drain_errors();
    gl.bind_framebuffer(GL_FRAMEBUFFER, previous);
    gl.delete_framebuffer(fb);
    if !read {
        return Err(Error::Unsupported);
    }
    if err != GL_NO_ERROR {
        return Err(Error::GlError(err));
    }
    Ok(())
}

/// Copy the box from the resource into the pages: `vrend_renderer_transfer_send_iov`.
pub fn read(
    gl: &Gl,
    features: &Features,
    formats: &Table,
    res: &Resource,
    own: Option<&Iov<'_>>,
    pages: &Iov<'_>,
    info: &Info,
) -> Result<(), Error> {
    let b = info.region;
    if !contains_box(res, &b, info.level) {
        return Err(Error::BoxOutOfRange);
    }
    let l = layout(res, info, pages)?;
    match &res.storage {
        Storage::Guest => {
            if let Some(own) = own
                && !own.same_pages(pages)
            {
                let mut tmp = vec![0u8; b.width as usize];
                if !own.copy_out(b.x as u64, &mut tmp) || !pages.copy_in(info.offset, &tmp) {
                    return Err(Error::IovOutOfRange);
                }
            }
            Ok(())
        }
        Storage::Host(shadow) => {
            let (x, w) = (b.x as usize, b.width as usize);
            let src = shadow.bytes().get(x..x + w).ok_or(Error::BoxOutOfRange)?;
            if !pages.copy_in(info.offset, src) {
                return Err(Error::IovOutOfRange);
            }
            Ok(())
        }
        Storage::Buffer { name, target, .. } => {
            let (x, w) = (b.x as usize, b.width as usize);
            gl.bind_buffer(*target, Some(*name));
            gl.drain_errors();
            let copied = gl.map_buffer_read(*target, x, w, |src| pages.copy_in(info.offset, src));
            let err = gl.drain_errors();
            gl.bind_buffer(*target, None);
            match copied {
                // The C reports success for a map the driver refused, with nothing written.
                // A readback that wrote nothing is a failure here.
                None => Err(Error::GlError(err)),
                Some(false) => Err(Error::IovOutOfRange),
                Some(true) => Ok(()),
            }
        }
        Storage::Texture { .. } => {
            let entry = res.entry(formats).ok_or(Error::Unsupported)?;
            let can_readpixels = entry.can_render() || entry.is_ds();
            let readonly = || -> Result<(), Error> {
                // `vrend_transfer_send_readonly`: the guest is reading its own upload back into
                // the pages it uploaded from, and nothing host-side changed the texture.
                if own.is_some_and(|own| own.same_pages(pages)) {
                    Ok(())
                } else {
                    Err(Error::NotReadable)
                }
            };
            if !can_readpixels {
                return readonly();
            }
            let format_name = res.args.format.name();
            let invert = res.y_0_top();
            let l = l.as_gl(entry, b.width as u64, b.height as u64);
            let total = usize::try_from(l.total()).map_err(|_| Error::IovOutOfRange)?;
            let mut data = vec![0u8; total];
            let layer = l.layer() as usize;
            let y = if invert { res.height_at(info.level) as GLint - b.y - b.height } else { b.y };
            gl.use_program_none();
            gl.pack_tight();
            for d in 0..l.depth as usize {
                let dst = &mut data[d * layer..(d + 1) * layer];
                let r = read_layer(
                    gl,
                    features,
                    formats,
                    res,
                    info.level,
                    b.z + d as GLint,
                    b.x,
                    y,
                    b.width,
                    b.height,
                    dst,
                );
                if let Err(e) = r {
                    // A read the driver refused falls back to the readonly answer, as in the C:
                    // the guest is told either that it already holds the bytes, or nothing.
                    eprintln!("[virglrs] readback of {}: {e}", format_name);
                    return readonly();
                }
            }
            if res.is_bgra() {
                swizzle_bgra(&mut data);
            }
            if format_name == "Z24X8_UNORM" {
                scale_depth(&mut data, 1.0 / 256.0);
            }
            if invert {
                flip_rows(&mut data, &l);
            }
            if !scatter(pages, info, &l, &data) {
                return Err(Error::IovOutOfRange);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_flip_within_each_layer() {
        let l = Layout {
            block: 1,
            blocks_wide: 2,
            blocks_high: 3,
            depth: 2,
            stride: 2,
            layer_stride: 6,
            compressed: false,
        };
        let mut d: Vec<u8> = (0..12).collect();
        flip_rows(&mut d, &l);
        assert_eq!(d, [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7]);
    }

    #[test]
    fn bgra_swaps_and_depth_scales_like_the_c() {
        let mut px = [1u8, 2, 3, 4, 5, 6, 7, 8];
        swizzle_bgra(&mut px);
        assert_eq!(px, [3, 2, 1, 4, 7, 6, 5, 8]);
        // The C's arithmetic: `1/0xffffff` rounds to 2^-24 in f32, so a 24-bit value scaled by
        // 256 lands one ulp above the C's `CLAMP` would suggest, and the round trip is not the
        // identity. Both are what the goldens were recorded through.
        let mut z = 0x0080_0000u32.to_ne_bytes().to_vec();
        scale_depth(&mut z, 256.0);
        assert_eq!(u32::from_ne_bytes(z[..4].try_into().unwrap()), 0x8000_0000);
        let mut back = 0x8000_0000u32.to_ne_bytes().to_vec();
        scale_depth(&mut back, 1.0 / 256.0);
        assert_eq!(u32::from_ne_bytes(back[..4].try_into().unwrap()), 0x0080_0000);
        // Saturates: the top of the range clamps to 1.0 and comes back one ulp short of it.
        let mut full = 0xffff_ff00u32.to_ne_bytes().to_vec();
        scale_depth(&mut full, 256.0);
        assert_eq!(u32::from_ne_bytes(full[..4].try_into().unwrap()), 0xffff_fe00);
    }

    /// The planar bug the C carries, expressed as the property that makes it impossible here.
    ///
    /// vrend registers NV12 with the `GL_RGBA8/GL_RGBA/GL_UNSIGNED_BYTE` triple while gallium
    /// describes it as one byte per block, then bounds-checks the guest's pages with gallium's
    /// number and hands GL the driver's -- so GL moves four times what was checked and runs off
    /// the end of a correctly sized iov. Two descriptions of one format, and the bound taken from
    /// the wrong one.
    ///
    /// Every route a guest has into a transfer converges on `write` and `read`: TRANSFER3D,
    /// COPY_TRANSFER3D, RESOURCE_INLINE_WRITE and the C ABI's own `transfer_*_iov`. None of them
    /// bounds-checks first and then moves bytes itself, and none of them can -- the layout
    /// reconciliation and the gather are private to this module, so the only thing a fifth route
    /// could call is the pair that already gets this right.
    ///
    /// Here there is one description at the moment of use: the staging buffer is sized from
    /// `as_gl`, the same layout the driver reads, and `gather` fills it out of the guest's pages a
    /// row at a time through `copy_out`, which answers false when the pages do not hold the row.
    /// So the same transfer is refused rather than overrun, and it is refused because the buffer
    /// and the bound are one number.
    #[test]
    fn a_planar_transfer_is_refused_because_its_buffer_and_its_bound_are_one_number() {
        use crate::vrend::proto::Format;
        let wire = super::super::formats::DESCRIPTIONS
            .iter()
            .position(|d| d.is_some_and(|d| d.name == "Y8_U8V8_420_UNORM"))
            .expect("NV12 is a described format");
        let nv12 = Format::from_wire(wire as u32).unwrap();
        let desc = nv12.describe().unwrap();
        assert_eq!(desc.block_bytes(), 1, "gallium describes NV12 as one byte per block");

        let entry = Entry {
            gl: super::super::formats::GlFormat {
                format: nv12,
                internalformat: GL_RGBA8,
                glformat: GL_RGBA,
                gltype: GL_UNSIGNED_BYTE,
                swizzle: None,
                view_class: super::super::formats::ViewClass::Bits32,
            },
            bindings: super::super::formats::Bindings {
                sampler_view: true,
                render_target: false,
                depth_stencil: false,
            },
            can_texture_storage: true,
            can_readback: false,
            can_multisample: false,
        };

        // What gallium says a tight 64x64 NV12 box spans, which is what the C bounds-checks.
        let gallium = Layout {
            block: 1,
            blocks_wide: 64,
            blocks_high: 64,
            depth: 1,
            stride: 64,
            layer_stride: 64 * 64,
            compressed: false,
        };
        assert_eq!(gallium.total(), 64 * 64);

        // What the driver actually moves, which is what this path allocates and fills.
        let driver = gallium.as_gl(&entry, 64, 64);
        assert_eq!(driver.block, 4);
        assert_eq!(driver.total(), 4 * gallium.total());

        // A guest that attached exactly the bytes gallium describes -- the shape the C reads four
        // times over -- cannot satisfy the driver's layout, and gather says so instead.
        let mut guest = vec![0xa5u8; (gallium.total()) as usize];
        let entries = [crate::abi::GuestIov {
            base: crate::abi::VmmPtr(guest.as_mut_ptr().cast()),
            len: guest.len(),
        }];
        let pages = Iov::new(&entries);
        let info = Info {
            level: 0,
            stride: 64,
            layer_stride: 64 * 64,
            offset: 0,
            region: Box3 { x: 0, y: 0, z: 0, width: 64, height: 64, depth: 1 },
            synchronized: false,
        };
        let mut by_gallium = vec![0u8; gallium.total() as usize];
        assert!(
            gather(&pages, &info, &gallium, &mut by_gallium),
            "gallium's layout fits, which is why the C's bounds check passes this transfer"
        );
        let mut staging = vec![0u8; driver.total() as usize];
        assert!(
            !gather(&pages, &info, &driver, &mut staging),
            "the driver's layout does not, and that is the answer the guest gets"
        );
    }

    #[test]
    fn a_span_counts_the_last_row_and_layer_tight() {
        let l = Layout {
            block: 4,
            blocks_wide: 10,
            blocks_high: 4,
            depth: 3,
            stride: 64,
            layer_stride: 1024,
            compressed: false,
        };
        assert_eq!(l.span(), 2 * 1024 + 3 * 64 + 40);
        assert_eq!(l.total(), 40 * 4 * 3);
    }
}
