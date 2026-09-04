// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! GLES, as the driver exports it -- and the safe surface the rest of vrend calls it through.
//!
//! One of the named unsafe modules (CLAUDE.md). The table is generated from the Khronos registry
//! by `gl-gen`, for the reason `vulkan.rs` gives: a binding transcribed by hand can disagree with
//! the driver about a parameter, and the disagreement is a stack smash rather than a compile
//! error. Every entry point is resolved through `eglGetProcAddress` ([`super::egl`]), which is
//! what makes the transmute in `Gles::load` sound: EGL promises the address of the function of
//! that name, and the registry gives that name its signature.
//!
//! [`Gl`] is the safe layer over the table, and the only thing the resource, transfer and blit
//! code sees. Each method owns the one hazard its entry point has: a pointer argument is a slice
//! whose length is checked against what GL will read or write, computed here from the format and
//! type the way the driver computes it, with the pixel-store state pinned to tightly packed so
//! there is nothing else for the size to depend on. A call with an argument this crate cannot size
//! is refused before it reaches the driver, never passed through on trust.
//!
//! No method here needs a context to be current for *soundness* -- Mesa dispatches a call with no
//! current context to a stub -- so currency is a correctness property `Vrend` upholds, not a
//! precondition these wrappers encode.

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod types {
    include!(concat!(env!("OUT_DIR"), "/gl/types.rs"));
}

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod gles {
    include!(concat!(env!("OUT_DIR"), "/gl/gles.rs"));
}

use core::ffi::CStr;

use super::egl::Image;

pub use gles::Gles;
use gles::*;
pub use types::*;

/// A texture name the driver handed out. Never zero: zero is "no texture", and is spelled `None`
/// wherever a binding allows it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FramebufferName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RenderbufferName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VertexArrayName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplerName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TransformFeedbackName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QueryName(GLuint);

/// A shader object the driver handed out. Never zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ShaderName(GLuint);

/// A program object the driver handed out. Never zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProgramName(GLuint);

impl ShaderName {
    pub fn raw(self) -> GLuint {
        self.0
    }
}

/// The C's `GLvoid const *` offset into a bound buffer, as the draw and indirect calls spell it.
fn offset_ptr(offset: u32) -> *const core::ffi::c_void {
    offset as usize as *const core::ffi::c_void
}

impl TextureName {
    pub fn raw(self) -> GLuint {
        self.0
    }
}

impl BufferName {
    pub fn raw(self) -> GLuint {
        self.0
    }
}

/// The number of bytes one pixel of `format`/`ty` takes in client memory, tightly packed --
/// `_mesa_bytes_per_pixel`, over the GLES pairs this renderer uploads and reads. `None` for a
/// pair the driver would refuse too, and one the callers refuse first.
pub fn pixel_bytes(format: GLenum, ty: GLenum) -> Option<usize> {
    let components = match format {
        GL_RED | GL_RED_INTEGER | GL_DEPTH_COMPONENT | GL_STENCIL_INDEX | GL_ALPHA
        | GL_LUMINANCE => 1,
        GL_RG | GL_RG_INTEGER | GL_LUMINANCE_ALPHA | GL_DEPTH_STENCIL => 2,
        GL_RGB | GL_RGB_INTEGER => 3,
        GL_RGBA | GL_RGBA_INTEGER | GL_BGRA | GL_BGRA_INTEGER | GL_ABGR_EXT => 4,
        _ => return None,
    };
    let per_component = match ty {
        GL_UNSIGNED_BYTE | GL_BYTE => 1,
        GL_UNSIGNED_SHORT | GL_SHORT | GL_HALF_FLOAT | GL_HALF_FLOAT_OES => 2,
        GL_UNSIGNED_INT | GL_INT | GL_FLOAT => 4,
        // Packed types: the whole pixel in one word, whatever the component count.
        GL_UNSIGNED_SHORT_5_6_5
        | GL_UNSIGNED_SHORT_4_4_4_4
        | GL_UNSIGNED_SHORT_5_5_5_1
        | GL_UNSIGNED_SHORT_5_6_5_REV
        | GL_UNSIGNED_SHORT_4_4_4_4_REV
        | GL_UNSIGNED_SHORT_1_5_5_5_REV => return Some(2),
        GL_UNSIGNED_INT_8_8_8_8
        | GL_UNSIGNED_INT_8_8_8_8_REV
        | GL_UNSIGNED_INT_2_10_10_10_REV
        | GL_UNSIGNED_INT_10F_11F_11F_REV
        | GL_UNSIGNED_INT_5_9_9_9_REV
        | GL_UNSIGNED_INT_24_8 => return Some(4),
        GL_FLOAT_32_UNSIGNED_INT_24_8_REV => return Some(8),
        _ => return None,
    };
    Some(components * per_component)
}

/// The bytes a `w`×`h`×`d` image of `format`/`ty` takes tightly packed, or `None` when the pair is
/// unknown or the product overflows.
fn image_bytes(format: GLenum, ty: GLenum, w: GLsizei, h: GLsizei, d: GLsizei) -> Option<usize> {
    let px = pixel_bytes(format, ty)?;
    let dim = |v: GLsizei| usize::try_from(v).ok();
    px.checked_mul(dim(w)?)?.checked_mul(dim(h)?)?.checked_mul(dim(d)?)
}

/// The driver's entry points behind a safe surface.
pub struct Gl {
    t: Gles,
}

impl Gl {
    pub fn new(t: Gles) -> Gl {
        Gl { t }
    }

    /// The raw table, for the census of what the driver exports.
    pub fn table(&self) -> &Gles {
        &self.t
    }

    // ---- queries ----

    pub fn get_error(&self) -> GLenum {
        // SAFETY: takes nothing.
        unsafe { self.t.glGetError()() }
    }

    /// Drain the error queue, returning the first error or `GL_NO_ERROR`.
    pub fn drain_errors(&self) -> GLenum {
        let first = self.get_error();
        if first != GL_NO_ERROR {
            while self.get_error() != GL_NO_ERROR {}
        }
        first
    }

    pub fn get_integer(&self, name: GLenum) -> GLint {
        let mut v: GLint = 0;
        // SAFETY: every name this crate asks for writes exactly one integer.
        unsafe { self.t.glGetIntegerv()(name, &mut v) };
        v
    }

    /// `glGetIntegeri_v`: one integer of an indexed state.
    pub fn get_integer_i(&self, name: GLenum, index: GLuint) -> GLint {
        let mut v: GLint = 0;
        // SAFETY: every indexed name this crate asks for writes exactly one integer.
        unsafe { self.t.glGetIntegeri_v()(name, index, &mut v) };
        v
    }

    pub fn get_float(&self, name: GLenum) -> GLfloat {
        let mut v: GLfloat = 0.0;
        // SAFETY: every name this crate asks for through here writes exactly one float.
        unsafe { self.t.glGetFloatv()(name, &mut v) };
        v
    }

    /// `glGetFloatv` for a two-float range (`GL_ALIASED_POINT_SIZE_RANGE` and its kin).
    pub fn get_float_range(&self, name: GLenum) -> [GLfloat; 2] {
        let mut v: [GLfloat; 2] = [0.0; 2];
        // SAFETY: every name this crate asks for through here writes exactly two floats.
        unsafe { self.t.glGetFloatv()(name, v.as_mut_ptr()) };
        v
    }

    /// `glGetMultisamplefv(GL_SAMPLE_POSITION, index)`: where a sample sits in its pixel.
    pub fn get_sample_position(&self, index: GLuint) -> [GLfloat; 2] {
        let mut v: [GLfloat; 2] = [0.0; 2];
        // SAFETY: `GL_SAMPLE_POSITION` writes exactly two floats.
        unsafe { self.t.glGetMultisamplefv()(GL_SAMPLE_POSITION, index, v.as_mut_ptr()) };
        v
    }

    pub fn get_string(&self, name: GLenum) -> String {
        // SAFETY: `glGetString` returns null or a NUL-terminated string owned by the driver, live
        // while the context is; it is copied out immediately.
        let p = unsafe { self.t.glGetString()(name) };
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p.cast::<c_char>()) }.to_string_lossy().into_owned()
    }

    pub fn get_string_i(&self, name: GLenum, index: GLuint) -> String {
        // SAFETY: as `get_string`; an out-of-range index answers null.
        let p = unsafe { self.t.glGetStringi()(name, index) };
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p.cast::<c_char>()) }.to_string_lossy().into_owned()
    }

    /// Every extension the context advertises.
    pub fn extensions(&self) -> Vec<String> {
        let n = self.get_integer(GL_NUM_EXTENSIONS).max(0) as GLuint;
        (0..n).map(|i| self.get_string_i(GL_EXTENSIONS, i)).collect()
    }

    // ---- objects ----

    pub fn gen_texture(&self) -> TextureName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenTextures()(1, &mut id) };
        TextureName(id)
    }

    pub fn delete_texture(&self, tex: TextureName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteTextures()(1, &tex.0) };
    }

    pub fn bind_texture(&self, target: GLenum, tex: Option<TextureName>) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindTexture()(target, tex.map_or(0, |t| t.0)) };
    }

    pub fn gen_buffer(&self) -> BufferName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenBuffers()(1, &mut id) };
        BufferName(id)
    }

    pub fn delete_buffer(&self, buf: BufferName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteBuffers()(1, &buf.0) };
    }

    pub fn bind_buffer(&self, target: GLenum, buf: Option<BufferName>) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindBuffer()(target, buf.map_or(0, |b| b.0)) };
    }

    pub fn gen_framebuffer(&self) -> FramebufferName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenFramebuffers()(1, &mut id) };
        FramebufferName(id)
    }

    pub fn delete_framebuffer(&self, fb: FramebufferName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteFramebuffers()(1, &fb.0) };
    }

    pub fn bind_framebuffer(&self, target: GLenum, fb: Option<FramebufferName>) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindFramebuffer()(target, fb.map_or(0, |f| f.0)) };
    }

    pub fn gen_renderbuffer(&self) -> RenderbufferName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenRenderbuffers()(1, &mut id) };
        RenderbufferName(id)
    }

    pub fn delete_renderbuffer(&self, rb: RenderbufferName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteRenderbuffers()(1, &rb.0) };
    }

    // ---- texture allocation ----

    /// `glTexImage2D` with no data: allocate a level and nothing more.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_image_2d_null(
        &self,
        target: GLenum,
        level: GLint,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        ty: GLenum,
    ) {
        // SAFETY: a null pixel pointer with no pixel unpack buffer bound reads nothing.
        unsafe {
            self.t.glTexImage2D()(
                target,
                level,
                internalformat as GLint,
                w,
                h,
                0,
                format,
                ty,
                core::ptr::null(),
            )
        };
    }

    /// `glTexImage3D` with no data.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_image_3d_null(
        &self,
        target: GLenum,
        level: GLint,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
        format: GLenum,
        ty: GLenum,
    ) {
        // SAFETY: a null pixel pointer with no pixel unpack buffer bound reads nothing.
        unsafe {
            self.t.glTexImage3D()(
                target,
                level,
                internalformat as GLint,
                w,
                h,
                d,
                0,
                format,
                ty,
                core::ptr::null(),
            )
        };
    }

    pub fn tex_storage_2d(
        &self,
        target: GLenum,
        levels: GLsizei,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glTexStorage2D()(target, levels, internalformat, w, h) };
    }

    pub fn tex_storage_3d(
        &self,
        target: GLenum,
        levels: GLsizei,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glTexStorage3D()(target, levels, internalformat, w, h, d) };
    }

    pub fn tex_storage_2d_multisample(
        &self,
        target: GLenum,
        samples: GLsizei,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glTexStorage2DMultisample()(target, samples, internalformat, w, h, GL_TRUE as _)
        };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tex_storage_3d_multisample(
        &self,
        target: GLenum,
        samples: GLsizei,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
    ) -> bool {
        let Some(f) = self.t.try_glTexStorage3DMultisample() else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe { f(target, samples, internalformat, w, h, d, GL_TRUE as _) };
        true
    }

    pub fn tex_parameter_i(&self, target: GLenum, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glTexParameteri()(target, name, value) };
    }

    pub fn pixel_store_i(&self, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glPixelStorei()(name, value) };
    }

    // ---- texture data ----

    /// Upload a tightly packed `w`×`h` image. Refuses, without calling the driver, a slice shorter
    /// than the image or a format/type pair this crate cannot size.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_sub_image_2d(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        ty: GLenum,
        data: &[u8],
    ) -> bool {
        let Some(need) = image_bytes(format, ty, w, h, 1) else {
            return false;
        };
        if data.len() < need {
            return false;
        }
        // SAFETY: with unpack row length and image height zero and alignment 1 -- which
        // `Gl::unpack_tight` sets and every caller uses -- the driver reads exactly `need` bytes
        // from `data`, and `data` holds at least that many.
        unsafe {
            self.t.glTexSubImage2D()(target, level, x, y, w, h, format, ty, data.as_ptr().cast())
        };
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tex_sub_image_3d(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        z: GLint,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
        format: GLenum,
        ty: GLenum,
        data: &[u8],
    ) -> bool {
        let Some(need) = image_bytes(format, ty, w, h, d) else {
            return false;
        };
        if data.len() < need {
            return false;
        }
        // SAFETY: as `tex_sub_image_2d`, over `d` tightly packed layers.
        unsafe {
            self.t.glTexSubImage3D()(
                target,
                level,
                x,
                y,
                z,
                w,
                h,
                d,
                format,
                ty,
                data.as_ptr().cast(),
            )
        };
        true
    }

    /// Upload compressed blocks. GL reads exactly `data.len()` bytes -- the length is the
    /// `imageSize` argument -- so there is nothing to check but that the slice fits a `GLsizei`.
    #[allow(clippy::too_many_arguments)]
    pub fn compressed_tex_sub_image_2d(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        data: &[u8],
    ) -> bool {
        let Ok(size) = GLsizei::try_from(data.len()) else {
            return false;
        };
        // SAFETY: the driver reads `size` bytes from `data`, and `size` is `data`'s length.
        unsafe {
            self.t.glCompressedTexSubImage2D()(
                target,
                level,
                x,
                y,
                w,
                h,
                format,
                size,
                data.as_ptr().cast(),
            )
        };
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressed_tex_sub_image_3d(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        z: GLint,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
        format: GLenum,
        data: &[u8],
    ) -> bool {
        let Ok(size) = GLsizei::try_from(data.len()) else {
            return false;
        };
        // SAFETY: as `compressed_tex_sub_image_2d`.
        unsafe {
            self.t.glCompressedTexSubImage3D()(
                target,
                level,
                x,
                y,
                z,
                w,
                h,
                d,
                format,
                size,
                data.as_ptr().cast(),
            )
        };
        true
    }

    /// Read a tightly packed `w`×`h` image out of the read framebuffer into `dst`. Refuses a
    /// destination shorter than the image.
    #[allow(clippy::too_many_arguments)]
    pub fn read_pixels(
        &self,
        x: GLint,
        y: GLint,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        ty: GLenum,
        dst: &mut [u8],
    ) -> bool {
        let Some(need) = image_bytes(format, ty, w, h, 1) else {
            return false;
        };
        if dst.len() < need {
            return false;
        }
        // Robust readback where the driver has it: the bound is the slice's own length, so
        // whatever the driver believes the image is, it cannot write past `dst`.
        if let Some(f) = self.t.try_glReadnPixelsKHR()
            && let Ok(size) = GLsizei::try_from(dst.len())
        {
            // SAFETY: the driver writes at most `size` bytes into `dst`, which holds `size`.
            unsafe { f(x, y, w, h, format, ty, size, dst.as_mut_ptr().cast()) };
            return true;
        }
        // SAFETY: with pack row length zero and alignment 1 -- `Gl::pack_tight`, which every
        // caller sets -- the driver writes exactly `need` bytes, and `dst` holds at least that.
        unsafe { self.t.glReadPixels()(x, y, w, h, format, ty, dst.as_mut_ptr().cast()) };
        true
    }

    /// Pin the unpack state to tightly packed rows, which is what every upload here assumes.
    pub fn unpack_tight(&self) {
        self.pixel_store_i(GL_UNPACK_ROW_LENGTH, 0);
        self.pixel_store_i(GL_UNPACK_IMAGE_HEIGHT, 0);
        self.pixel_store_i(GL_UNPACK_SKIP_PIXELS, 0);
        self.pixel_store_i(GL_UNPACK_SKIP_ROWS, 0);
        self.pixel_store_i(GL_UNPACK_ALIGNMENT, 1);
    }

    /// Pin the pack state to tightly packed rows, which is what every readback here assumes.
    pub fn pack_tight(&self) {
        self.pixel_store_i(GL_PACK_ROW_LENGTH, 0);
        self.pixel_store_i(GL_PACK_SKIP_PIXELS, 0);
        self.pixel_store_i(GL_PACK_SKIP_ROWS, 0);
        self.pixel_store_i(GL_PACK_ALIGNMENT, 1);
    }

    // ---- buffers ----

    /// `glBufferData` with no data: allocate the store.
    pub fn buffer_data_null(&self, target: GLenum, size: usize, usage: GLenum) -> bool {
        let Ok(size) = GLsizeiptr::try_from(size) else {
            return false;
        };
        // SAFETY: a null data pointer allocates without reading.
        unsafe { self.t.glBufferData()(target, size, core::ptr::null(), usage) };
        true
    }

    pub fn buffer_storage_null(&self, target: GLenum, size: usize, flags: GLbitfield) -> bool {
        let (Some(f), Ok(size)) = (self.t.try_glBufferStorageEXT(), GLsizeiptr::try_from(size))
        else {
            return false;
        };
        // SAFETY: a null data pointer allocates without reading.
        unsafe { f(target, size, core::ptr::null(), flags) };
        true
    }

    pub fn buffer_sub_data(&self, target: GLenum, offset: usize, data: &[u8]) -> bool {
        let (Ok(offset), Ok(size)) = (GLintptr::try_from(offset), GLsizeiptr::try_from(data.len()))
        else {
            return false;
        };
        // SAFETY: the driver reads `size` bytes from `data`, and `size` is `data`'s length.
        unsafe { self.t.glBufferSubData()(target, offset, size, data.as_ptr().cast()) };
        true
    }

    /// Map `len` bytes of the bound buffer for writing and hand them to `f` as a slice; unmapped
    /// before this returns. `None` if the driver refused the map.
    pub fn map_buffer_write<R>(
        &self,
        target: GLenum,
        offset: usize,
        len: usize,
        flags: GLbitfield,
        f: impl FnOnce(&mut [u8]) -> R,
    ) -> Option<R> {
        let (Ok(off), Ok(size)) = (GLintptr::try_from(offset), GLsizeiptr::try_from(len)) else {
            return None;
        };
        // SAFETY: plain scalars; the returned pointer is null or addresses `len` writable bytes
        // until `glUnmapBuffer`.
        let p = unsafe { self.t.glMapBufferRange()(target, off, size, flags | GL_MAP_WRITE_BIT) };
        if p.is_null() {
            return None;
        }
        // SAFETY: the driver mapped `len` bytes at `p`, exclusively ours until unmapped, which
        // happens after `f` returns and before the slice can be used again.
        let r = f(unsafe { core::slice::from_raw_parts_mut(p.cast::<u8>(), len) });
        unsafe { self.t.glUnmapBuffer()(target) };
        Some(r)
    }

    /// Map `len` bytes of the bound buffer for reading.
    pub fn map_buffer_read<R>(
        &self,
        target: GLenum,
        offset: usize,
        len: usize,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Option<R> {
        let (Ok(off), Ok(size)) = (GLintptr::try_from(offset), GLsizeiptr::try_from(len)) else {
            return None;
        };
        // SAFETY: as `map_buffer_write`, for reading.
        let p = unsafe { self.t.glMapBufferRange()(target, off, size, GL_MAP_READ_BIT) };
        if p.is_null() {
            return None;
        }
        // SAFETY: the driver mapped `len` readable bytes at `p` until unmapped.
        let r = f(unsafe { core::slice::from_raw_parts(p.cast::<u8>(), len) });
        unsafe { self.t.glUnmapBuffer()(target) };
        Some(r)
    }

    // ---- framebuffers ----

    pub fn framebuffer_texture_2d(
        &self,
        attachment: GLenum,
        textarget: GLenum,
        tex: Option<TextureName>,
        level: GLint,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glFramebufferTexture2D()(
                GL_FRAMEBUFFER,
                attachment,
                textarget,
                tex.map_or(0, |t| t.0),
                level,
            )
        };
    }

    pub fn framebuffer_texture_layer(
        &self,
        attachment: GLenum,
        tex: Option<TextureName>,
        level: GLint,
        layer: GLint,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glFramebufferTextureLayer()(
                GL_FRAMEBUFFER,
                attachment,
                tex.map_or(0, |t| t.0),
                level,
                layer,
            )
        };
    }

    /// `glFramebufferTexture3DOES`: one slice of a 3D texture. `false` if the driver lacks it.
    pub fn framebuffer_texture_3d(
        &self,
        attachment: GLenum,
        tex: Option<TextureName>,
        level: GLint,
        layer: GLint,
    ) -> bool {
        let Some(f) = self.t.try_glFramebufferTexture3DOES() else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe {
            f(GL_FRAMEBUFFER, attachment, GL_TEXTURE_3D, tex.map_or(0, |t| t.0), level, layer)
        };
        true
    }

    pub fn check_framebuffer_status(&self) -> GLenum {
        // SAFETY: plain scalar.
        unsafe { self.t.glCheckFramebufferStatus()(GL_FRAMEBUFFER) }
    }

    pub fn draw_buffers(&self, bufs: &[GLenum]) {
        let Ok(n) = GLsizei::try_from(bufs.len()) else {
            return;
        };
        // SAFETY: the driver reads `n` enums from `bufs`, which holds `n`.
        unsafe { self.t.glDrawBuffers()(n, bufs.as_ptr()) };
    }

    pub fn read_buffer(&self, src: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glReadBuffer()(src) };
    }

    /// `glFramebufferTexture`: every layer of a layered texture. `false` if the driver has none
    /// of the three spellings.
    pub fn framebuffer_texture(
        &self,
        attachment: GLenum,
        tex: Option<TextureName>,
        level: GLint,
    ) -> bool {
        let f = self
            .t
            .try_glFramebufferTexture()
            .or_else(|| self.t.try_glFramebufferTextureEXT())
            .or_else(|| self.t.try_glFramebufferTextureOES());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe { f(GL_FRAMEBUFFER, attachment, tex.map_or(0, |t| t.0), level) };
        true
    }

    /// `glTextureView` in whichever spelling the driver exports: `view` becomes a view of
    /// `levels` levels from `first_level` and `layers` layers from `first_layer` of `tex`.
    /// `false` if the driver has none.
    #[allow(clippy::too_many_arguments)]
    pub fn texture_view(
        &self,
        view: TextureName,
        target: GLenum,
        tex: TextureName,
        internalformat: GLenum,
        first_level: GLuint,
        levels: GLuint,
        first_layer: GLuint,
        layers: GLuint,
    ) -> bool {
        let f = self.t.try_glTextureViewOES().or_else(|| self.t.try_glTextureViewEXT());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe {
            f(view.0, target, tex.0, internalformat, first_level, levels, first_layer, layers)
        };
        true
    }

    /// `glEGLImageTargetTexStorageEXT`: the bound texture takes `image` as immutable storage.
    /// `false` if the driver has none.
    pub fn egl_image_target_tex_storage(&self, target: GLenum, image: &Image) -> bool {
        let Some(f) = self.t.try_glEGLImageTargetTexStorageEXT() else {
            return false;
        };
        // SAFETY: `image` is a live EGL image on this display, held by the caller across the
        // call; a null attribute list is the documented empty one.
        unsafe { f(target, image.raw(), core::ptr::null()) };
        true
    }

    /// `glEGLImageTargetTexture2DOES`: the bound texture takes `image` as (mutable) storage.
    /// `false` if the driver has none.
    pub fn egl_image_target_texture_2d(&self, target: GLenum, image: &Image) -> bool {
        let Some(f) = self.t.try_glEGLImageTargetTexture2DOES() else {
            return false;
        };
        // SAFETY: as above.
        unsafe { f(target, image.raw()) };
        true
    }

    pub fn framebuffer_parameter_i(&self, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glFramebufferParameteri()(GL_FRAMEBUFFER, name, value) };
    }

    /// `glBlitFramebuffer` from the read framebuffer's rectangle to the draw framebuffer's.
    #[allow(clippy::too_many_arguments)]
    pub fn blit_framebuffer(
        &self,
        src: [GLint; 4],
        dst: [GLint; 4],
        mask: GLbitfield,
        filter: GLenum,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glBlitFramebuffer()(
                src[0], src[1], src[2], src[3], dst[0], dst[1], dst[2], dst[3], mask, filter,
            )
        };
    }

    /// `glCopyImageSubData`, in whichever spelling the driver exports. `false` if none.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_image_sub_data(
        &self,
        src: TextureName,
        src_target: GLenum,
        src_level: GLint,
        src_origin: [GLint; 3],
        dst: TextureName,
        dst_target: GLenum,
        dst_level: GLint,
        dst_origin: [GLint; 3],
        extent: [GLsizei; 3],
    ) -> bool {
        let f = self
            .t
            .try_glCopyImageSubData()
            .or_else(|| self.t.try_glCopyImageSubDataEXT())
            .or_else(|| self.t.try_glCopyImageSubDataOES());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe {
            f(
                src.0,
                src_target,
                src_level,
                src_origin[0],
                src_origin[1],
                src_origin[2],
                dst.0,
                dst_target,
                dst_level,
                dst_origin[0],
                dst_origin[1],
                dst_origin[2],
                extent[0],
                extent[1],
                extent[2],
            )
        };
        true
    }

    pub fn copy_buffer_sub_data(
        &self,
        read_offset: usize,
        write_offset: usize,
        size: usize,
    ) -> bool {
        let (Ok(r), Ok(w), Ok(s)) = (
            GLintptr::try_from(read_offset),
            GLintptr::try_from(write_offset),
            GLsizeiptr::try_from(size),
        ) else {
            return false;
        };
        // SAFETY: plain scalars; the copy is between the two bound copy buffers.
        unsafe { self.t.glCopyBufferSubData()(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, r, w, s) };
        true
    }

    /// `glTexBufferRange` on the bound `GL_TEXTURE_BUFFER`, or `glTexBuffer` when `range` is
    /// `None`. `false` if the driver has neither spelling of the one asked for.
    pub fn tex_buffer(
        &self,
        internalformat: GLenum,
        buf: BufferName,
        range: Option<(usize, usize)>,
    ) -> bool {
        match range {
            Some((offset, size)) => {
                let f = self
                    .t
                    .try_glTexBufferRange()
                    .or_else(|| self.t.try_glTexBufferRangeEXT())
                    .or_else(|| self.t.try_glTexBufferRangeOES());
                let (Some(f), Ok(offset), Ok(size)) =
                    (f, GLintptr::try_from(offset), GLsizeiptr::try_from(size))
                else {
                    return false;
                };
                // SAFETY: plain scalars.
                unsafe { f(GL_TEXTURE_BUFFER, internalformat, buf.0, offset, size) };
            }
            None => {
                let f = self
                    .t
                    .try_glTexBuffer()
                    .or_else(|| self.t.try_glTexBufferEXT())
                    .or_else(|| self.t.try_glTexBufferOES());
                let Some(f) = f else {
                    return false;
                };
                // SAFETY: plain scalars.
                unsafe { f(GL_TEXTURE_BUFFER, internalformat, buf.0) };
            }
        }
        true
    }

    /// `glClearTexSubImageEXT` with one pixel of `format`/`ty` as the value. Refuses a value
    /// shorter than the pixel or a pair this crate cannot size, and a driver without the
    /// extension.
    #[allow(clippy::too_many_arguments)]
    pub fn clear_tex_sub_image(
        &self,
        tex: TextureName,
        level: GLint,
        origin: [GLint; 3],
        extent: [GLsizei; 3],
        format: GLenum,
        ty: GLenum,
        value: &[u8],
    ) -> bool {
        let (Some(f), Some(need)) = (self.t.try_glClearTexSubImageEXT(), pixel_bytes(format, ty))
        else {
            return false;
        };
        if value.len() < need {
            return false;
        }
        // SAFETY: the driver reads one pixel -- `need` bytes -- from `value`, which holds them.
        unsafe {
            f(
                tex.0,
                level,
                origin[0],
                origin[1],
                origin[2],
                extent[0],
                extent[1],
                extent[2],
                format,
                ty,
                value.as_ptr().cast(),
            )
        };
        true
    }

    // ---- fixed-function state ----

    pub fn enable(&self, cap: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glEnable()(cap) };
    }

    pub fn disable(&self, cap: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDisable()(cap) };
    }

    pub fn set_enabled(&self, cap: GLenum, on: bool) {
        if on {
            self.enable(cap);
        } else {
            self.disable(cap);
        }
    }

    pub fn depth_func(&self, func: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDepthFunc()(func) };
    }

    pub fn depth_mask(&self, on: bool) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDepthMask()(on as GLboolean) };
    }

    pub fn stencil_op(&self, fail: GLenum, zfail: GLenum, zpass: GLenum) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilOp()(fail, zfail, zpass) };
    }

    pub fn stencil_op_separate(&self, face: GLenum, fail: GLenum, zfail: GLenum, zpass: GLenum) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilOpSeparate()(face, fail, zfail, zpass) };
    }

    pub fn stencil_func(&self, func: GLenum, reference: GLint, mask: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilFunc()(func, reference, mask) };
    }

    pub fn stencil_func_separate(
        &self,
        face: GLenum,
        func: GLenum,
        reference: GLint,
        mask: GLuint,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilFuncSeparate()(face, func, reference, mask) };
    }

    pub fn stencil_mask(&self, mask: GLuint) {
        // SAFETY: plain scalar.
        unsafe { self.t.glStencilMask()(mask) };
    }

    pub fn stencil_mask_separate(&self, face: GLenum, mask: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilMaskSeparate()(face, mask) };
    }

    pub fn color_mask(&self, rgba: [bool; 4]) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glColorMask()(
                rgba[0] as GLboolean,
                rgba[1] as GLboolean,
                rgba[2] as GLboolean,
                rgba[3] as GLboolean,
            )
        };
    }

    /// `glColorMaski` in whichever spelling the driver exports. `false` if none.
    pub fn color_mask_i(&self, index: GLuint, rgba: [bool; 4]) -> bool {
        let f = self
            .t
            .try_glColorMaski()
            .or_else(|| self.t.try_glColorMaskiEXT())
            .or_else(|| self.t.try_glColorMaskiOES());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe {
            f(
                index,
                rgba[0] as GLboolean,
                rgba[1] as GLboolean,
                rgba[2] as GLboolean,
                rgba[3] as GLboolean,
            )
        };
        true
    }

    pub fn clear_color(&self, rgba: [f32; 4]) {
        // SAFETY: plain scalars.
        unsafe { self.t.glClearColor()(rgba[0], rgba[1], rgba[2], rgba[3]) };
    }

    pub fn clear_depth_f(&self, depth: f32) {
        // SAFETY: plain scalar.
        unsafe { self.t.glClearDepthf()(depth) };
    }

    pub fn clear_stencil(&self, stencil: GLint) {
        // SAFETY: plain scalar.
        unsafe { self.t.glClearStencil()(stencil) };
    }

    pub fn clear(&self, mask: GLbitfield) {
        // SAFETY: plain scalar.
        unsafe { self.t.glClear()(mask) };
    }

    pub fn clear_buffer_fv(&self, buffer: GLenum, drawbuffer: GLint, value: &[f32; 4]) {
        // SAFETY: the driver reads four floats for `GL_COLOR`, one for `GL_DEPTH`; `value` holds
        // four.
        unsafe { self.t.glClearBufferfv()(buffer, drawbuffer, value.as_ptr()) };
    }

    pub fn blend_color(&self, rgba: [f32; 4]) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBlendColor()(rgba[0], rgba[1], rgba[2], rgba[3]) };
    }

    pub fn line_width(&self, width: f32) {
        // SAFETY: plain scalar.
        unsafe { self.t.glLineWidth()(width) };
    }

    pub fn polygon_offset(&self, factor: f32, units: f32) {
        // SAFETY: plain scalars.
        unsafe { self.t.glPolygonOffset()(factor, units) };
    }

    pub fn cull_face(&self, mode: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glCullFace()(mode) };
    }

    pub fn front_face(&self, mode: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glFrontFace()(mode) };
    }

    pub fn sample_mask_i(&self, index: GLuint, mask: GLbitfield) {
        // SAFETY: plain scalars.
        unsafe { self.t.glSampleMaski()(index, mask) };
    }

    /// `glMinSampleShading`. `false` if the driver has neither spelling.
    pub fn min_sample_shading(&self, value: f32) -> bool {
        let f = self.t.try_glMinSampleShading().or_else(|| self.t.try_glMinSampleShadingOES());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: plain scalar.
        unsafe { f(value) };
        true
    }

    pub fn viewport(&self, x: GLint, y: GLint, w: GLsizei, h: GLsizei) {
        // SAFETY: plain scalars.
        unsafe { self.t.glViewport()(x, y, w, h) };
    }

    pub fn scissor(&self, x: GLint, y: GLint, w: GLsizei, h: GLsizei) {
        // SAFETY: plain scalars.
        unsafe { self.t.glScissor()(x, y, w, h) };
    }

    pub fn depth_range_f(&self, near: f32, far: f32) {
        // SAFETY: plain scalars.
        unsafe { self.t.glDepthRangef()(near, far) };
    }

    /// `glClipControlEXT`. `false` if the driver lacks it.
    pub fn clip_control(&self, origin: GLenum, depth: GLenum) -> bool {
        let Some(f) = self.t.try_glClipControlEXT() else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe { f(origin, depth) };
        true
    }

    pub fn memory_barrier(&self, barriers: GLbitfield) {
        // SAFETY: plain scalar.
        unsafe { self.t.glMemoryBarrier()(barriers) };
    }

    // ---- vertex arrays ----

    pub fn gen_vertex_array(&self) -> VertexArrayName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenVertexArrays()(1, &mut id) };
        VertexArrayName(id)
    }

    pub fn delete_vertex_array(&self, vao: VertexArrayName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteVertexArrays()(1, &vao.0) };
    }

    pub fn bind_vertex_array(&self, vao: Option<VertexArrayName>) {
        // SAFETY: plain scalar.
        unsafe { self.t.glBindVertexArray()(vao.map_or(0, |v| v.0)) };
    }

    pub fn vertex_attrib_format(
        &self,
        index: GLuint,
        size: GLint,
        ty: GLenum,
        normalized: bool,
        offset: GLuint,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glVertexAttribFormat()(index, size, ty, normalized as GLboolean, offset) };
    }

    pub fn vertex_attrib_i_format(&self, index: GLuint, size: GLint, ty: GLenum, offset: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glVertexAttribIFormat()(index, size, ty, offset) };
    }

    pub fn vertex_attrib_binding(&self, index: GLuint, binding: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glVertexAttribBinding()(index, binding) };
    }

    pub fn vertex_binding_divisor(&self, binding: GLuint, divisor: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glVertexBindingDivisor()(binding, divisor) };
    }

    pub fn enable_vertex_attrib_array(&self, index: GLuint) {
        // SAFETY: plain scalar.
        unsafe { self.t.glEnableVertexAttribArray()(index) };
    }

    // ---- samplers ----

    pub fn gen_sampler(&self) -> SamplerName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenSamplers()(1, &mut id) };
        SamplerName(id)
    }

    pub fn delete_sampler(&self, s: SamplerName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteSamplers()(1, &s.0) };
    }

    pub fn sampler_parameter_i(&self, s: SamplerName, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glSamplerParameteri()(s.0, name, value) };
    }

    pub fn sampler_parameter_f(&self, s: SamplerName, name: GLenum, value: f32) {
        // SAFETY: plain scalars.
        unsafe { self.t.glSamplerParameterf()(s.0, name, value) };
    }

    /// `glSamplerParameterIuiv` for the border colour. `false` if the driver has no spelling.
    pub fn sampler_border_color(&self, s: SamplerName, color: &[GLuint; 4]) -> bool {
        let f = self
            .t
            .try_glSamplerParameterIuiv()
            .or_else(|| self.t.try_glSamplerParameterIuivEXT())
            .or_else(|| self.t.try_glSamplerParameterIuivOES());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: the driver reads four values for `GL_TEXTURE_BORDER_COLOR`; `color` holds four.
        unsafe { f(s.0, GL_TEXTURE_BORDER_COLOR, color.as_ptr()) };
        true
    }

    // ---- transform feedback ----

    pub fn gen_transform_feedback(&self) -> TransformFeedbackName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenTransformFeedbacks()(1, &mut id) };
        TransformFeedbackName(id)
    }

    pub fn delete_transform_feedback(&self, tf: TransformFeedbackName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteTransformFeedbacks()(1, &tf.0) };
    }

    pub fn bind_transform_feedback(&self, tf: Option<TransformFeedbackName>) {
        // SAFETY: plain scalar.
        unsafe { self.t.glBindTransformFeedback()(GL_TRANSFORM_FEEDBACK, tf.map_or(0, |t| t.0)) };
    }

    pub fn end_transform_feedback(&self) {
        // SAFETY: takes nothing.
        unsafe { self.t.glEndTransformFeedback()() };
    }

    pub fn bind_buffer_base(&self, target: GLenum, index: GLuint, buf: Option<BufferName>) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindBufferBase()(target, index, buf.map_or(0, |b| b.0)) };
    }

    pub fn bind_buffer_range(
        &self,
        target: GLenum,
        index: GLuint,
        buf: BufferName,
        offset: usize,
        size: usize,
    ) -> bool {
        let (Ok(offset), Ok(size)) = (GLintptr::try_from(offset), GLsizeiptr::try_from(size))
        else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe { self.t.glBindBufferRange()(target, index, buf.0, offset, size) };
        true
    }

    // ---- queries ----

    pub fn gen_query(&self) -> QueryName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenQueries()(1, &mut id) };
        QueryName(id)
    }

    pub fn delete_query(&self, q: QueryName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteQueries()(1, &q.0) };
    }

    pub fn begin_query(&self, target: GLenum, q: QueryName) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBeginQuery()(target, q.0) };
    }

    pub fn end_query(&self, target: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glEndQuery()(target) };
    }

    pub fn get_query_object_uiv(&self, q: QueryName, name: GLenum) -> GLuint {
        let mut v: GLuint = 0;
        // SAFETY: every name asked for writes exactly one integer.
        unsafe { self.t.glGetQueryObjectuiv()(q.0, name, &mut v) };
        v
    }

    // ---- shaders ----

    /// `glCreateShader`; `None` when the driver refused the kind.
    pub fn create_shader(&self, kind: GLenum) -> Option<ShaderName> {
        // SAFETY: plain scalar.
        let id = unsafe { self.t.glCreateShader()(kind) };
        (id != 0).then_some(ShaderName(id))
    }

    pub fn delete_shader(&self, shader: ShaderName) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDeleteShader()(shader.0) };
    }

    /// `glShaderSource` and `glCompileShader`, and the driver's log when it refused the source.
    pub fn compile_shader(&self, shader: ShaderName, source: &str) -> Result<(), String> {
        let ptr = source.as_ptr().cast::<GLchar>();
        let len = GLint::try_from(source.len()).expect("a shader's source fits a GLint");
        // SAFETY: one string, with its length given, so the driver reads exactly `source`.
        unsafe { self.t.glShaderSource()(shader.0, 1, &ptr, &len) };
        // SAFETY: plain scalar.
        unsafe { self.t.glCompileShader()(shader.0) };
        let mut status: GLint = 0;
        // SAFETY: `GL_COMPILE_STATUS` writes exactly one integer.
        unsafe { self.t.glGetShaderiv()(shader.0, GL_COMPILE_STATUS, &mut status) };
        if status != 0 {
            return Ok(());
        }
        let mut log = vec![0u8; 65536];
        let mut written: GLsizei = 0;
        // SAFETY: the driver writes at most the capacity given, NUL included, and reports how
        // many bytes it wrote without the NUL.
        unsafe {
            self.t.glGetShaderInfoLog()(
                shader.0,
                log.len() as GLsizei,
                &mut written,
                log.as_mut_ptr().cast::<GLchar>(),
            )
        };
        log.truncate(written.max(0) as usize);
        Err(String::from_utf8_lossy(&log).into_owned())
    }

    // ---- programs ----

    pub fn create_program(&self) -> Option<ProgramName> {
        // SAFETY: no arguments.
        let id = unsafe { self.t.glCreateProgram()() };
        (id != 0).then_some(ProgramName(id))
    }

    pub fn delete_program(&self, program: ProgramName) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDeleteProgram()(program.0) };
    }

    pub fn attach_shader(&self, program: ProgramName, shader: ShaderName) {
        // SAFETY: plain scalars.
        unsafe { self.t.glAttachShader()(program.0, shader.0) };
    }

    /// `glLinkProgram`, and the driver's log when the link failed.
    pub fn link_program(&self, program: ProgramName) -> Result<(), String> {
        // SAFETY: plain scalar.
        unsafe { self.t.glLinkProgram()(program.0) };
        let mut status: GLint = 0;
        // SAFETY: `GL_LINK_STATUS` writes exactly one integer.
        unsafe { self.t.glGetProgramiv()(program.0, GL_LINK_STATUS, &mut status) };
        if status != 0 {
            return Ok(());
        }
        let mut log = vec![0u8; 65536];
        let mut written: GLsizei = 0;
        // SAFETY: the driver writes at most the capacity given, NUL included, and reports how
        // many bytes it wrote without the NUL.
        unsafe {
            self.t.glGetProgramInfoLog()(
                program.0,
                log.len() as GLsizei,
                &mut written,
                log.as_mut_ptr().cast::<GLchar>(),
            )
        };
        log.truncate(written.max(0) as usize);
        Err(String::from_utf8_lossy(&log).into_owned())
    }

    pub fn use_program(&self, program: Option<ProgramName>) {
        // SAFETY: plain scalar; zero is "no program".
        unsafe { self.t.glUseProgram()(program.map_or(0, |p| p.0)) };
    }

    /// `glGetUniformLocation`; -1 when the program has no such uniform.
    pub fn get_uniform_location(&self, program: ProgramName, name: &str) -> GLint {
        let name = std::ffi::CString::new(name).expect("a uniform name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        unsafe { self.t.glGetUniformLocation()(program.0, name.as_ptr().cast::<GLchar>()) }
    }

    /// `glGetUniformBlockIndex`; `None` when the program has no such block.
    pub fn get_uniform_block_index(&self, program: ProgramName, name: &str) -> Option<GLuint> {
        let name = std::ffi::CString::new(name).expect("a block name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        let i =
            unsafe { self.t.glGetUniformBlockIndex()(program.0, name.as_ptr().cast::<GLchar>()) };
        (i != GL_INVALID_INDEX).then_some(i)
    }

    pub fn uniform_block_binding(&self, program: ProgramName, block: GLuint, binding: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glUniformBlockBinding()(program.0, block, binding) };
    }

    /// `GL_UNIFORM_BLOCK_DATA_SIZE` of a block.
    pub fn uniform_block_data_size(&self, program: ProgramName, block: GLuint) -> GLint {
        let mut v: GLint = 0;
        // SAFETY: the query writes exactly one integer.
        unsafe {
            self.t.glGetActiveUniformBlockiv()(program.0, block, GL_UNIFORM_BLOCK_DATA_SIZE, &mut v)
        };
        v
    }

    pub fn bind_attrib_location(&self, program: ProgramName, index: GLuint, name: &str) {
        let name = std::ffi::CString::new(name).expect("an attribute name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        unsafe { self.t.glBindAttribLocation()(program.0, index, name.as_ptr().cast::<GLchar>()) };
    }

    /// `glTransformFeedbackVaryings`, interleaved.
    pub fn transform_feedback_varyings(&self, program: ProgramName, varyings: &[String]) {
        let owned: Vec<std::ffi::CString> = varyings
            .iter()
            .map(|v| std::ffi::CString::new(v.as_str()).expect("a varying name has no NUL"))
            .collect();
        let ptrs: Vec<*const GLchar> = owned.iter().map(|c| c.as_ptr().cast::<GLchar>()).collect();
        let count = GLsizei::try_from(ptrs.len()).expect("a varying count fits a GLsizei");
        // SAFETY: `count` NUL-terminated strings, all live for the call.
        unsafe {
            self.t.glTransformFeedbackVaryings()(
                program.0,
                count,
                ptrs.as_ptr(),
                GL_INTERLEAVED_ATTRIBS,
            )
        };
    }

    /// `glBindFragDataLocationIndexedEXT`. `false` if the driver has no spelling.
    pub fn bind_frag_data_location_indexed(
        &self,
        program: ProgramName,
        color: GLuint,
        index: GLuint,
        name: &str,
    ) -> bool {
        let Some(f) = self.t.try_glBindFragDataLocationIndexedEXT() else {
            return false;
        };
        let name = std::ffi::CString::new(name).expect("an output name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        unsafe { f(program.0, color, index, name.as_ptr().cast::<GLchar>()) };
        true
    }

    pub fn uniform_1i(&self, location: GLint, v: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glUniform1i()(location, v) };
    }

    pub fn uniform_1iv(&self, location: GLint, v: &[GLint]) {
        let count = GLsizei::try_from(v.len()).expect("a uniform count fits a GLsizei");
        // SAFETY: the driver reads `count` integers from a slice of that length.
        unsafe { self.t.glUniform1iv()(location, count, v.as_ptr()) };
    }

    pub fn uniform_4f(&self, location: GLint, v: [f32; 4]) {
        // SAFETY: plain scalars.
        unsafe { self.t.glUniform4f()(location, v[0], v[1], v[2], v[3]) };
    }

    /// `glUniform4uiv` over `v`, which holds `count` vectors of four.
    pub fn uniform_4uiv(&self, location: GLint, v: &[u32]) {
        let count = GLsizei::try_from(v.len() / 4).expect("a uniform count fits a GLsizei");
        // SAFETY: the driver reads `count` vectors of four, which the slice holds.
        unsafe { self.t.glUniform4uiv()(location, count, v.as_ptr()) };
    }

    pub fn active_texture(&self, unit: GLuint) {
        // SAFETY: plain scalar.
        unsafe { self.t.glActiveTexture()(GL_TEXTURE0 + unit) };
    }

    pub fn bind_sampler(&self, unit: GLuint, sampler: Option<SamplerName>) {
        // SAFETY: plain scalars; zero is "no sampler".
        unsafe { self.t.glBindSampler()(unit, sampler.map_or(0, |s| s.0)) };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_image_texture(
        &self,
        unit: GLuint,
        texture: TextureName,
        level: GLint,
        layered: bool,
        layer: GLint,
        access: GLenum,
        format: GLenum,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glBindImageTexture()(
                unit,
                texture.0,
                level,
                layered as GLboolean,
                layer,
                access,
                format,
            )
        };
    }

    pub fn bind_vertex_buffer(
        &self,
        binding: GLuint,
        buf: Option<BufferName>,
        offset: u32,
        stride: u32,
    ) {
        // SAFETY: plain scalars; zero is "no buffer".
        unsafe {
            self.t.glBindVertexBuffer()(
                binding,
                buf.map_or(0, |b| b.0),
                offset as GLintptr,
                stride as GLsizei,
            )
        };
    }

    // ---- blending ----

    pub fn blend_func_separate(
        &self,
        src_rgb: GLenum,
        dst_rgb: GLenum,
        src_a: GLenum,
        dst_a: GLenum,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBlendFuncSeparate()(src_rgb, dst_rgb, src_a, dst_a) };
    }

    pub fn blend_equation_separate(&self, rgb: GLenum, alpha: GLenum) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBlendEquationSeparate()(rgb, alpha) };
    }

    pub fn blend_equation(&self, mode: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glBlendEquation()(mode) };
    }

    /// `glBlendFuncSeparatei` in whichever spelling the driver exports. `false` if none.
    pub fn blend_func_separate_i(
        &self,
        buf: GLuint,
        src_rgb: GLenum,
        dst_rgb: GLenum,
        src_a: GLenum,
        dst_a: GLenum,
    ) -> bool {
        let f = self
            .t
            .try_glBlendFuncSeparatei()
            .or_else(|| self.t.try_glBlendFuncSeparateiEXT())
            .or_else(|| self.t.try_glBlendFuncSeparateiOES());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe { f(buf, src_rgb, dst_rgb, src_a, dst_a) };
        true
    }

    /// `glBlendEquationSeparatei` in whichever spelling the driver exports. `false` if none.
    pub fn blend_equation_separate_i(&self, buf: GLuint, rgb: GLenum, alpha: GLenum) -> bool {
        let f = self
            .t
            .try_glBlendEquationSeparatei()
            .or_else(|| self.t.try_glBlendEquationSeparateiEXT())
            .or_else(|| self.t.try_glBlendEquationSeparateiOES());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe { f(buf, rgb, alpha) };
        true
    }

    /// `glEnablei`/`glDisablei` in whichever spelling the driver exports. `false` if none.
    pub fn set_enabled_i(&self, cap: GLenum, index: GLuint, on: bool) -> bool {
        if on {
            let f = self
                .t
                .try_glEnablei()
                .or_else(|| self.t.try_glEnableiEXT())
                .or_else(|| self.t.try_glEnableiOES());
            let Some(f) = f else {
                return false;
            };
            // SAFETY: plain scalars.
            unsafe { f(cap, index) };
        } else {
            let f = self
                .t
                .try_glDisablei()
                .or_else(|| self.t.try_glDisableiEXT())
                .or_else(|| self.t.try_glDisableiOES());
            let Some(f) = f else {
                return false;
            };
            // SAFETY: plain scalars.
            unsafe { f(cap, index) };
        }
        true
    }

    // ---- draws ----

    pub fn draw_arrays(&self, mode: GLenum, first: GLint, count: GLsizei) {
        // SAFETY: plain scalars; the driver reads the bound arrays, which it bounds itself.
        unsafe { self.t.glDrawArrays()(mode, first, count) };
    }

    pub fn draw_arrays_instanced(
        &self,
        mode: GLenum,
        first: GLint,
        count: GLsizei,
        instances: GLsizei,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glDrawArraysInstanced()(mode, first, count, instances) };
    }

    /// `false` if the driver has no spelling.
    pub fn draw_arrays_instanced_base_instance(
        &self,
        mode: GLenum,
        first: GLint,
        count: GLsizei,
        instances: GLsizei,
        base_instance: GLuint,
    ) -> bool {
        let Some(f) = self.t.try_glDrawArraysInstancedBaseInstanceEXT() else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe { f(mode, first, count, instances, base_instance) };
        true
    }

    /// `glDrawElements` with the indices at `offset` into the bound element buffer.
    pub fn draw_elements(&self, mode: GLenum, count: GLsizei, ty: GLenum, offset: u32) {
        // SAFETY: an offset into the bound element array buffer, which the driver bounds.
        unsafe { self.t.glDrawElements()(mode, count, ty, offset_ptr(offset)) };
    }

    pub fn draw_range_elements(
        &self,
        mode: GLenum,
        start: GLuint,
        end: GLuint,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe { self.t.glDrawRangeElements()(mode, start, end, count, ty, offset_ptr(offset)) };
    }

    pub fn draw_elements_base_vertex(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        base_vertex: GLint,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe {
            self.t.glDrawElementsBaseVertex()(mode, count, ty, offset_ptr(offset), base_vertex)
        };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_range_elements_base_vertex(
        &self,
        mode: GLenum,
        start: GLuint,
        end: GLuint,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        base_vertex: GLint,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe {
            self.t.glDrawRangeElementsBaseVertex()(
                mode,
                start,
                end,
                count,
                ty,
                offset_ptr(offset),
                base_vertex,
            )
        };
    }

    pub fn draw_elements_instanced(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        instances: GLsizei,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe { self.t.glDrawElementsInstanced()(mode, count, ty, offset_ptr(offset), instances) };
    }

    pub fn draw_elements_instanced_base_vertex(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        instances: GLsizei,
        base_vertex: GLint,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe {
            self.t.glDrawElementsInstancedBaseVertex()(
                mode,
                count,
                ty,
                offset_ptr(offset),
                instances,
                base_vertex,
            )
        };
    }

    /// `false` if the driver has no spelling.
    pub fn draw_elements_instanced_base_instance(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        instances: GLsizei,
        base_instance: GLuint,
    ) -> bool {
        let Some(f) = self.t.try_glDrawElementsInstancedBaseInstanceEXT() else {
            return false;
        };
        // SAFETY: as `draw_elements`.
        unsafe { f(mode, count, ty, offset_ptr(offset), instances, base_instance) };
        true
    }

    /// `false` if the driver has no spelling.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_elements_instanced_base_vertex_base_instance(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        instances: GLsizei,
        base_vertex: GLint,
        base_instance: GLuint,
    ) -> bool {
        let Some(f) = self.t.try_glDrawElementsInstancedBaseVertexBaseInstanceEXT() else {
            return false;
        };
        // SAFETY: as `draw_elements`.
        unsafe { f(mode, count, ty, offset_ptr(offset), instances, base_vertex, base_instance) };
        true
    }

    /// `glDrawArraysIndirect` with the command at `offset` into the bound indirect buffer.
    pub fn draw_arrays_indirect(&self, mode: GLenum, offset: u32) {
        // SAFETY: an offset into the bound indirect buffer, which the driver bounds.
        unsafe { self.t.glDrawArraysIndirect()(mode, offset_ptr(offset)) };
    }

    pub fn draw_elements_indirect(&self, mode: GLenum, ty: GLenum, offset: u32) {
        // SAFETY: as `draw_arrays_indirect`.
        unsafe { self.t.glDrawElementsIndirect()(mode, ty, offset_ptr(offset)) };
    }

    /// `false` if the driver has no spelling.
    pub fn multi_draw_arrays_indirect(
        &self,
        mode: GLenum,
        offset: u32,
        draw_count: GLsizei,
        stride: GLsizei,
    ) -> bool {
        let Some(f) = self.t.try_glMultiDrawArraysIndirectEXT() else {
            return false;
        };
        // SAFETY: as `draw_arrays_indirect`.
        unsafe { f(mode, offset_ptr(offset), draw_count, stride) };
        true
    }

    /// `false` if the driver has no spelling.
    pub fn multi_draw_elements_indirect(
        &self,
        mode: GLenum,
        ty: GLenum,
        offset: u32,
        draw_count: GLsizei,
        stride: GLsizei,
    ) -> bool {
        let Some(f) = self.t.try_glMultiDrawElementsIndirectEXT() else {
            return false;
        };
        // SAFETY: as `draw_arrays_indirect`.
        unsafe { f(mode, ty, offset_ptr(offset), draw_count, stride) };
        true
    }

    pub fn patch_parameter_i(&self, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glPatchParameteri()(name, value) };
    }

    pub fn begin_transform_feedback(&self, mode: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glBeginTransformFeedback()(mode) };
    }

    pub fn pause_transform_feedback(&self) {
        // SAFETY: no arguments.
        unsafe { self.t.glPauseTransformFeedback()() };
    }

    pub fn resume_transform_feedback(&self) {
        // SAFETY: no arguments.
        unsafe { self.t.glResumeTransformFeedback()() };
    }

    // ---- misc ----

    pub fn use_program_none(&self) {
        // SAFETY: zero is "no program".
        unsafe { self.t.glUseProgram()(0) };
    }

    /// `glBindProgramPipeline(0)`. `false` if the driver has no spelling.
    pub fn bind_program_pipeline_none(&self) -> bool {
        let f =
            self.t.try_glBindProgramPipeline().or_else(|| self.t.try_glBindProgramPipelineEXT());
        let Some(f) = f else {
            return false;
        };
        // SAFETY: zero is "no pipeline".
        unsafe { f(0) };
        true
    }

    pub fn finish(&self) {
        // SAFETY: takes nothing.
        unsafe { self.t.glFinish()() };
    }

    pub fn flush(&self) {
        // SAFETY: takes nothing.
        unsafe { self.t.glFlush()() };
    }
}
