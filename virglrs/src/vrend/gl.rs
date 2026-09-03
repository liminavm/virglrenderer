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

    // ---- misc ----

    pub fn use_program_none(&self) {
        // SAFETY: zero is "no program".
        unsafe { self.t.glUseProgram()(0) };
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
