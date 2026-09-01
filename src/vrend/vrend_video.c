/**************************************************************************
 *
 * Copyright (C) 2022 Kylin Software Co., Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/**
 * @file
 * The video implementation of the vrend renderer.
 *
 * It is based on the general virgl video submodule and handles data transfer
 * and synchronization between host and guest.
 *
 * The relationship between vaSurface and video buffer objects:
 *
 *           GUEST (Mesa)           |       HOST (Virglrenderer)
 *                                  |
 *         +------------+           |          +------------+
 *         | vaSurface  |           |          | vaSurface  | <------+
 *         +------------+           |          +------------+        |
 *               |                  |                                |
 *  +---------------------------+   |   +-------------------------+  |
 *  |    virgl_video_buffer     |   |   |    vrend_video_buffer   |  |
 *  | +-----------------------+ |   |   |  +-------------------+  |  |
 *  | |    vl_video_buffer    | |   |   |  | vrend_resource(s) |  |  |
 *  | | +-------------------+ | |<--+-->|  +-------------------+  |  |
 *  | | | virgl_resource(s) | | |   |   |  +--------------------+ |  |
 *  | | +-------------------+ | |   |   |  | virgl_video_buffer |-+--+
 *  | +-----------------------+ |   |   |  +--------------------+ |
 *  +---------------------------+   |   +-------------------------+
 *
 * The relationship between vaContext and video codec objects:
 *
 *           GUEST (Mesa)         |         HOST (Virglrenderer)
 *                                |
 *         +------------+         |           +------------+
 *         | vaContext  |         |           | vaContext  | <-------+
 *         +------------+         |           +------------+         |
 *               |                |                                  |
 *  +------------------------+    |    +--------------------------+  |
 *  |    virgl_video_codec   | <--+--> |    vrend_video_codec     |  |
 *  +------------------------+    |    |  +--------------------+  |  |
 *                                |    |  | virgl_video_codec  | -+--+
 *                                |    |  +--------------------+  |
 *                                |    +--------------------------+
 *
 * @author Feng Jiang <jiangfeng@kylinos.cn>
 */


#include "util/u_format.h"

#include "virgl_video.h"
#include "virgl_video_hw.h"

#include "vrend_debug.h"
#include "vrend_winsys.h"
#include "vrend_renderer.h"
#include "vrend_iov.h"
#include "vrend_video.h"

#ifdef __APPLE__
/* From vkr_metal_helpers (same library), forward-declared rather than included: the
 * header needs Vulkan types, which vrend deliberately keeps out. Same reason and same
 * shape as the declarations in vrend_renderer.c. */
struct vkr_mtl_iosurface;
int vkr_mtl_iosurface_plane_write(struct vkr_mtl_iosurface *surf, uint32_t plane,
                                  const void *src, uint32_t src_stride, uint32_t rows,
                                  uint32_t row_bytes);
#endif

struct vrend_context;

struct vrend_video_context {
    struct vrend_context *ctx;
    struct list_head codecs;
    struct list_head buffers;
};

struct vrend_video_codec {
    struct virgl_video_codec *codec;
    uint32_t handle;
    struct vrend_resource *feed_res;    /* encoding feedback */
    struct vrend_resource *dest_res;    /* encoding coded buffer */
    struct vrend_video_context *ctx;
    struct list_head head;
};

struct vrend_video_plane {
    uint32_t res_handle;
    GLuint texture;         /* texture for temporary use */
    GLuint framebuffer;     /* framebuffer for temporary use */
    EGLImageKHR egl_image;  /* egl image for temporary use */
};

struct vrend_video_buffer {
    struct virgl_video_buffer *buffer;

    uint32_t handle;
    struct vrend_video_context *ctx;
    struct list_head head;

    uint32_t num_planes;
    struct vrend_video_plane planes[3];
};

static struct vrend_video_codec *vrend_video_codec(
        struct virgl_video_codec *codec)
{
    return virgl_video_codec_opaque_data(codec);
}

static struct vrend_video_buffer *vrend_video_buffer(
        struct virgl_video_buffer *buffer)
{
    return virgl_video_buffer_opaque_data(buffer);
}

static struct vrend_video_codec *get_video_codec(
                                        struct vrend_video_context *ctx,
                                        uint32_t cdc_handle)
{
    list_for_each_entry(struct vrend_video_codec, cdc, &ctx->codecs, head) {
        if (cdc->handle == cdc_handle)
            return cdc;
    }

    return NULL;
}

static struct vrend_video_buffer *get_video_buffer(
                                        struct vrend_video_context *ctx,
                                        uint32_t buf_handle)
{
    list_for_each_entry(struct vrend_video_buffer, buf, &ctx->buffers, head) {
        if (buf->handle == buf_handle)
            return buf;
    }

    return NULL;
}


/* Clamp a plane copy to what the SOURCE actually holds.
 *
 * Every reader below sizes its copy from the resource, and the resource is the aligned
 * allocation while the source is CoreVideo's actual plane -- which holds exactly the
 * rows the picture has. Copying the resource's height out of it reads off the end of the
 * mapping, at any resolution where the two differ. It faults only once the pool's slack
 * runs out before the next page, which is what makes it intermittent.
 *
 * The 2026-09-01 dogfood SIGSEGV is that read: a byte read, translation fault
 * (esr 0x92000007), at a page-aligned address that was the memmove source, copying a
 * 480-byte row. The picture's dimensions are not recoverable from the report -- the
 * video traces were not armed -- so the row length is what is measured and the height
 * is not. The clamp does not depend on knowing them.
 *
 * The backend already knows the answer and no reader asked: plane->size is the mapped
 * extent. Derive the copy from it. A backend that leaves size zero -- the dmabuf path,
 * which never maps -- keeps the caller's dimensions, since there is nothing to clamp to.
 *
 * The defect predates the composite decode target (it arrived with the mapped-plane
 * delivery itself) and outlived the build that crashed, so it is not something the
 * planar work introduced or fixed. */
static bool clamp_plane_to_source(const struct virgl_video_dma_buf_plane *plane,
                                  unsigned blocksize, const char *what,
                                  unsigned *width, unsigned *height)
{
    /* Falling short is the steady state for 1080p and would log every frame. */
    static int trace = -1;
    if (trace < 0)
        trace = getenv("LIMINA_VIDEO_CLAMP_TRACE") ? 1 : 0;

    if (!plane->size || !plane->pitch)
        return *width && *height;

    const unsigned rows = plane->size / plane->pitch;
    const unsigned cols = blocksize ? plane->pitch / blocksize : *width;
    const unsigned want_h = *height, want_w = *width;

    if (rows < *height)
        *height = rows;
    if (cols < *width)
        *width = cols;

    if (trace && (want_h != *height || want_w != *width))
        virgl_warn("%s: clamped %ux%u to %ux%u — the source holds %u rows of %u bytes "
                   "(size %u, pitch %u)\n", what, want_w, want_h, *width, *height,
                   rows, plane->pitch, plane->size, plane->pitch);

    return *width && *height;
}

/* Upload one CPU-mapped plane of a decoded picture into the guest-visible resource.
 *
 * The dmabuf path below hands the picture over as an EGLImage and blits; a backend
 * with no dmabuf to export — VideoToolbox, whose output is a CVPixelBuffer — maps
 * the plane instead and we copy. */
static void upload_mapped_plane(struct vrend_resource *res, unsigned plane_idx,
                                const struct virgl_video_dma_buf_plane *plane)
{
#ifdef __APPLE__
    /* limina: a composite target has one resource for the whole picture, so there is no
     * per-plane GL texture to upload into -- res->base.format is the planar format and
     * res->gl_id is the RGBA texture composite consumers sample. The plane's pixels
     * belong in the IOSurface plane the guest's own plane view is bound to, which is
     * where a plane sampler reads them and needs no upload at all. */
    if (plane_idx < res->iosurf_planes) {
        struct guest_plane geom[VIRGL_GBM_MAX_PLANES];
        uint32_t plane_count = 0;

        vrend_guest_plane_layout(res->base.format, res->base.width0, res->base.height0,
                                 geom, &plane_count);
        if (plane_idx < plane_count) {
            unsigned w = geom[plane_idx].width, h = geom[plane_idx].height;

            if (clamp_plane_to_source(plane, geom[plane_idx].bpp, "iosurface plane", &w, &h))
                vkr_mtl_iosurface_plane_write(res->iosurface, plane_idx, plane->map,
                                              plane->pitch, h, w * geom[plane_idx].bpp);
            return;
        }
    }
#endif

    /* Ask vrend how this resource was actually created rather than deriving a GL
     * format from the plane's size: an R8 luma plane and an RG8 chroma plane both
     * arrive here, and a mismatched format silently uploads the wrong bytes. */
    const struct vrend_format_table *entry = vrend_get_format_table_entry(res->base.format);
    unsigned blocksize = util_format_get_blocksize(res->base.format);

    if (!entry || !blocksize) {
        virgl_error("%s: no format table entry for %d\n", __func__, res->base.format);
        return;
    }

    unsigned up_w = res->base.width0, up_h = res->base.height0;

    if (!clamp_plane_to_source(plane, blocksize, "plane upload", &up_w, &up_h))
        return;

    /* VideoToolbox pads plane rows, so the source stride is not the width. */
    glBindTexture(GL_TEXTURE_2D, res->gl_id);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, plane->pitch / blocksize);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, up_w, up_h,
                    entry->glformat, entry->gltype, plane->map);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
}

/* limina: also land the decoded plane in the guest's own memory.
 *
 * upload_mapped_plane() above puts the pixels in a host GL texture, which is all a
 * guest that samples the target needs. A guest that *exports* it needs more: the fd
 * has to name storage that actually holds the frame. Guest mesa refuses an export
 * whose laid-out size exceeds the storage behind it -- correctly, since the classic
 * per-plane resource has a one-page stub BO however large the picture -- and the
 * refusal costs Firefox its hardware decoder.
 *
 * So when the guest allocated the plane in guest memory (VIRGL_CAP_V2_VIDEO_GUEST_PLANES,
 * which is what tells it that doing so is worthwhile), write the frame there too. The
 * texture upload stays: sampling must not start depending on the guest's copy being
 * read back, and the readback only happens once per batch.
 *
 * Layout is the guest's, in vrend_resource::guest_pixels_*. Which entry applies depends
 * on the shape the guest asked for: with one resource per plane every target is plane 0
 * of its own resource, while a composite target is one resource holding all the planes,
 * so the index has to come from the caller either way.
 */
static void writeback_plane_to_guest(struct vrend_resource *res, unsigned plane_idx,
                                     const struct virgl_video_dma_buf_plane *plane)
{
    struct guest_plane geom[VIRGL_GBM_MAX_PLANES];
    uint32_t plane_count = 0;
    unsigned blocksize, width, height_in;

    vrend_guest_plane_layout(res->base.format, res->base.width0, res->base.height0,
                             geom, &plane_count);
    /* A per-plane resource describes one plane and nothing else, so its own plane 0 is
     * the answer however many planes the video buffer has. */
    if (plane_count < 2)
        plane_idx = 0;
    else if (plane_idx >= plane_count)
        return;

    blocksize = geom[plane_idx].bpp;
    width = geom[plane_idx].width;
    height_in = geom[plane_idx].height;

    /* The extent check below bounds the DESTINATION. The source walk further down reads
     * plane->map + y * plane->pitch and needs its own bound, for the same reason the
     * uploads do. */
    if (!clamp_plane_to_source(plane, blocksize, "writeback", &width, &height_in))
        return;
    size_t row, stride, offset, height, extent, storage;
    /* Every reason to skip is a legitimate steady state, so none of them can log per
     * frame. But when the guest ends up reading an unwritten target the skip IS the
     * fault, and silence is exactly the wrong answer -- hence one env-gated trace
     * naming the numbers that decided it. */
    static int trace = -1;
    if (trace < 0)
        trace = getenv("LIMINA_VIDEO_WRITEBACK_TRACE") ? 1 : 0;

    if (!blocksize) {
        if (trace)
            virgl_warn("writeback: res fmt %d plane %u has no blocksize\n",
                       res->base.format, plane_idx);
        return;
    }

    /* Nothing to write into: a host-only resource, which is the pre-existing case. */
    if (!res->guest_pixels_map && (!res->iov || !res->num_iovs)) {
        if (trace)
            virgl_warn("writeback: no guest storage (map %p, iov %p, num_iovs %d)\n",
                       res->guest_pixels_map, (void *)res->iov, res->num_iovs);
        return;
    }

    row = (size_t)width * blocksize;
    height = height_in;
    stride = res->guest_pixels_stride[plane_idx] ? res->guest_pixels_stride[plane_idx] : row;
    offset = res->guest_pixels_offset[plane_idx];

    if (stride < row) {
        virgl_error("%s: guest stride %zu < row %zu for plane %u of %ux%u %s\n", __func__,
                    stride, row, plane_idx, res->base.width0, res->base.height0,
                    util_format_name(res->base.format));
        return;
    }

    /* Measure the whole extent before copying any of it. A half-written frame is worse
     * than an unwritten one: it plays, and only a checksum would ever catch it.
     *
     * Falling short is the normal case, not an error, and must stay silent. A classic
     * per-plane resource has guest iovecs too -- the one-page shadow -- so every decoded
     * frame reaches here with storage far too small until guest mesa starts allocating
     * these in guest memory. Whether the storage is big enough IS the test for "the
     * guest wants the frame here"; there is nothing else to ask. */
    extent = offset + (height ? (height - 1) * stride + row : 0);
    storage = res->guest_pixels_map ? res->guest_pixels_map_size
                                    : vrend_get_iovec_size(res->iov, res->num_iovs);
    if (trace)
        virgl_warn("writeback: plane %u %ux%u of %ux%u %s row %zu stride %zu off %zu "
                   "extent %zu storage %zu src pitch %u -> %s\n",
                   plane_idx, width, height_in, res->base.width0, res->base.height0,
                   util_format_name(res->base.format), row, stride, offset, extent,
                   storage, plane->pitch, extent > storage ? "SKIP" : "write");
    if (extent > storage)
        return;

    for (size_t y = 0; y < height; y++) {
        const char *src = (const char *)plane->map + y * plane->pitch;
        size_t dst = offset + y * stride;

        if (res->guest_pixels_map)
            memcpy((char *)res->guest_pixels_map + dst, src, row);
        else
            vrend_write_to_iovec(res->iov, res->num_iovs, dst, src, row);
    }
}

static int sync_dmabuf_to_video_buffer(struct vrend_video_buffer *buf,
                                       const struct virgl_video_dma_buf *dmabuf)
{
    if (!(dmabuf->flags & VIRGL_VIDEO_DMABUF_READ_ONLY)) {
        virgl_error("%s: dmabuf is not readable\n", __func__);
        return -1;
    }

    for (unsigned i = 0; i < dmabuf->num_planes && i < buf->num_planes; i++) {
        struct vrend_video_plane *plane = &buf->planes[i];
        struct vrend_resource *res;

        res = vrend_renderer_ctx_res_lookup(buf->ctx->ctx, plane->res_handle);
        if (!res) {
            virgl_error("%s: res %d not found\n", __func__, plane->res_handle);
            continue;
        }

        if (dmabuf->planes[i].fd < 0 && dmabuf->planes[i].map) {
            upload_mapped_plane(res, i, &dmabuf->planes[i]);
            writeback_plane_to_guest(res, i, &dmabuf->planes[i]);
            continue;
        }

        /* dmabuf -> eglimage */
        if (EGL_NO_IMAGE_KHR == plane->egl_image) {
            EGLint img_attrs[16] = {
                EGL_LINUX_DRM_FOURCC_EXT,       dmabuf->planes[i].drm_format,
                EGL_WIDTH,                      dmabuf->width / (i + 1),
                EGL_HEIGHT,                     dmabuf->height / (i + 1),
                EGL_DMA_BUF_PLANE0_FD_EXT,      dmabuf->planes[i].fd,
                EGL_DMA_BUF_PLANE0_OFFSET_EXT,  dmabuf->planes[i].offset,
                EGL_DMA_BUF_PLANE0_PITCH_EXT,   dmabuf->planes[i].pitch,
                EGL_NONE
            };

            plane->egl_image = eglCreateImageKHR(eglGetCurrentDisplay(),
                    EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, img_attrs);
        }

        if (EGL_NO_IMAGE_KHR == plane->egl_image) {
            virgl_error("%s: create egl image failed\n", __func__);
            continue;
        }

        /* eglimage -> texture */
        glBindTexture(GL_TEXTURE_2D, plane->texture);
        glEGLImageTargetTexture2DOES(GL_TEXTURE_2D,
                                    (GLeglImageOES)(plane->egl_image));

        /* texture -> framebuffer */
        glBindFramebuffer(GL_READ_FRAMEBUFFER, plane->framebuffer);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, plane->texture, 0);

        /* framebuffer -> vrend_video_buffer.planes[i] */
        glBindTexture(GL_TEXTURE_2D, res->gl_id);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0,
                            res->base.width0, res->base.height0);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

static int sync_video_buffer_to_dmabuf(struct vrend_video_buffer *buf,
                                       const struct virgl_video_dma_buf *dmabuf)
{
    if (!(dmabuf->flags & VIRGL_VIDEO_DMABUF_WRITE_ONLY)) {
        virgl_error("%s: dmabuf is not writable\n", __func__);
        return -1;
    }

    for (unsigned i = 0; i < dmabuf->num_planes && i < buf->num_planes; i++) {
        struct vrend_video_plane *plane = &buf->planes[i];
        struct vrend_resource *res;

        res = vrend_renderer_ctx_res_lookup(buf->ctx->ctx, plane->res_handle);
        if (!res) {
            virgl_error("%s: res %d not found\n", __func__, plane->res_handle);
            continue;
        }

        /* dmabuf -> eglimage */
        if (EGL_NO_IMAGE_KHR == plane->egl_image) {
            EGLint img_attrs[16] = {
                EGL_LINUX_DRM_FOURCC_EXT,       dmabuf->planes[i].drm_format,
                EGL_WIDTH,                      dmabuf->width / (i + 1),
                EGL_HEIGHT,                     dmabuf->height / (i + 1),
                EGL_DMA_BUF_PLANE0_FD_EXT,      dmabuf->planes[i].fd,
                EGL_DMA_BUF_PLANE0_OFFSET_EXT,  dmabuf->planes[i].offset,
                EGL_DMA_BUF_PLANE0_PITCH_EXT,   dmabuf->planes[i].pitch,
                EGL_NONE
            };

            plane->egl_image = eglCreateImageKHR(eglGetCurrentDisplay(),
                    EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, img_attrs);
        }

        if (EGL_NO_IMAGE_KHR == plane->egl_image) {
            virgl_error("%s: create egl image failed\n", __func__);
            continue;
        }

        /* eglimage -> texture */
        glBindTexture(GL_TEXTURE_2D, plane->texture);
        glEGLImageTargetTexture2DOES(GL_TEXTURE_2D,
                                    (GLeglImageOES)(plane->egl_image));

        /* vrend_video_buffer.planes[i] -> framebuffer */
        glBindFramebuffer(GL_READ_FRAMEBUFFER, plane->framebuffer);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, res->gl_id, 0);

        /* framebuffer -> texture */
        glBindTexture(GL_TEXTURE_2D, plane->texture);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0,
                            res->base.width0, res->base.height0);

    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}


static void vrend_video_decode_completed(
                                struct virgl_video_codec *codec,
                                const struct virgl_video_dma_buf *dmabuf)
{
    struct vrend_video_buffer *buf = vrend_video_buffer(dmabuf->buf);

    (void)codec;

    sync_dmabuf_to_video_buffer(buf, dmabuf);
}


static void vrend_video_enocde_upload_picture(
                                struct virgl_video_codec *codec,
                                const struct virgl_video_dma_buf *dmabuf)
{
    struct vrend_video_buffer *buf = vrend_video_buffer(dmabuf->buf);

    (void)codec;

    sync_video_buffer_to_dmabuf(buf, dmabuf);
}

static void vrend_video_encode_completed(
                                struct virgl_video_codec *codec,
                                const struct virgl_video_dma_buf *src_buf,
                                const struct virgl_video_dma_buf *ref_buf,
                                unsigned num_coded_bufs,
                                const void * const *coded_bufs,
                                const unsigned *coded_sizes)
{
    void *buf;
    unsigned i, size, data_size;
    struct virgl_video_encode_feedback feedback;
    struct vrend_video_codec *cdc = vrend_video_codec(codec);

    (void)src_buf;
    (void)ref_buf;

    if (!cdc->dest_res || !cdc->feed_res)
        return;

    memset(&feedback, 0, sizeof(feedback));

    /* sync coded data to guest */
    if (has_bit(cdc->dest_res->storage_bits, VREND_STORAGE_GL_BUFFER)) {
        glBindBufferARB(cdc->dest_res->target, cdc->dest_res->gl_id);
        buf = glMapBufferRange(cdc->dest_res->target, 0,
                               cdc->dest_res->base.width0, GL_MAP_WRITE_BIT);
        for (i = 0, data_size = 0; i < num_coded_bufs &&
                    data_size < cdc->dest_res->base.width0; i++) {
            size = MIN2(cdc->dest_res->base.width0 - data_size, coded_sizes[i]);
            memcpy((uint8_t *)buf + data_size, coded_bufs[i], size);
            vrend_write_to_iovec(cdc->dest_res->iov, cdc->dest_res->num_iovs,
                                 data_size, coded_bufs[i], size);
            data_size += size;
        }
        glUnmapBuffer(cdc->dest_res->target);
        glBindBufferARB(cdc->dest_res->target, 0);
        feedback.stat = VIRGL_VIDEO_ENCODE_STAT_SUCCESS;
        feedback.coded_size = data_size;
    } else {
        virgl_warn("unexcepted coded res type\n");
        feedback.stat = VIRGL_VIDEO_ENCODE_STAT_FAILURE;
        feedback.coded_size = 0;
    }

    /* send feedback */
    vrend_write_to_iovec(cdc->feed_res->iov, cdc->feed_res->num_iovs,
                         0, (char *)(&feedback),
                         MIN2(cdc->feed_res->base.width0, sizeof(feedback)));

    cdc->dest_res = NULL;
    cdc->feed_res = NULL;
}

static struct virgl_video_callbacks video_callbacks = {
    .decode_completed           = vrend_video_decode_completed,
    .encode_upload_picture      = vrend_video_enocde_upload_picture,
    .encode_completed           = vrend_video_encode_completed,
};

int vrend_video_init(int drm_fd)
{
#ifndef __APPLE__
    /* The VA-API backend opens its display on this fd; the VideoToolbox one has
     * no device node to open and is passed -1. */
    if (drm_fd < 0)
        return -1;
#endif

    return virgl_video_init(drm_fd, &video_callbacks, 0);
}

void vrend_video_fini(void)
{
    virgl_video_destroy();
}

int vrend_video_fill_caps(union virgl_caps *caps)
{
    return virgl_video_fill_caps(caps);
}

int vrend_video_create_codec(struct vrend_video_context *ctx,
                             uint32_t handle,
                             uint32_t profile,
                             uint32_t entrypoint,
                             uint32_t chroma_format,
                             uint32_t level,
                             uint32_t width,
                             uint32_t height,
                             uint32_t max_ref,
                             uint32_t flags)
{
    struct vrend_video_codec *cdc = get_video_codec(ctx, handle);
    struct virgl_video_create_codec_args args;

    if (cdc)
        return 0;

    if (profile <= PIPE_VIDEO_PROFILE_UNKNOWN ||
        profile >= PIPE_VIDEO_PROFILE_MAX)
        return -1;

    if (entrypoint <= PIPE_VIDEO_ENTRYPOINT_UNKNOWN ||
        entrypoint > PIPE_VIDEO_ENTRYPOINT_ENCODE)
        return -1;

    if (chroma_format >= PIPE_VIDEO_CHROMA_FORMAT_NONE)
        return -1;

    if (!width || !height)
        return -1;

    cdc = (struct vrend_video_codec *)calloc(1, sizeof(*cdc));
    if (!cdc)
        return -1;

    args.profile = profile;
    args.entrypoint = entrypoint;
    args.chroma_format = chroma_format;
    args.level = level;
    args.width = width;
    args.height = height;
    args.max_references = max_ref;
    args.flags = flags;
    args.opaque = cdc;
    cdc->codec = virgl_video_create_codec(&args);
    if (!cdc->codec) {
        free(cdc);
        return -1;
    }

    cdc->handle = handle;
    cdc->ctx = ctx;
    list_add(&cdc->head, &ctx->codecs);

    return 0;
}

static void destroy_video_codec(struct vrend_video_codec *cdc)
{
    if (cdc) {
        list_del(&cdc->head);
        virgl_video_destroy_codec(cdc->codec);
        free(cdc);
    }
}

void vrend_video_destroy_codec(struct vrend_video_context *ctx,
                               uint32_t handle)
{
    struct vrend_video_codec *cdc = get_video_codec(ctx, handle);

    destroy_video_codec(cdc);
}

int vrend_video_create_buffer(struct vrend_video_context *ctx,
                              uint32_t handle,
                              uint32_t format,
                              uint32_t width,
                              uint32_t height,
                              uint32_t *res_handles,
                              unsigned int num_res)
{
    unsigned i;
    struct vrend_video_plane *plane;
    struct vrend_video_buffer *buf = get_video_buffer(ctx, handle);
    struct virgl_video_create_buffer_args args;

    if (buf)
        return 0;

    if (format <= PIPE_FORMAT_NONE || format >= PIPE_FORMAT_COUNT){
        virgl_error("Invalid vrend video buffer format: %d\n", format);
        return -1;
    }

    if (!width || !height || !res_handles || !num_res)
        return -1;

    buf = (struct vrend_video_buffer *)calloc(1, sizeof(*buf));
    if (!buf)
        return -1;

    args.format = format;
    args.width = width;
    args.height = height;
    args.interlaced = 0;
    args.opaque = buf;
    buf->buffer = virgl_video_create_buffer(&args);
    if (!buf->buffer) {
        free(buf);
        return -1;
    }

    for (i = 0; i < ARRAY_SIZE(buf->planes); i++)
        buf->planes[i].egl_image = EGL_NO_IMAGE_KHR;

    for (i = 0, buf->num_planes = 0;
         i < num_res && buf->num_planes < ARRAY_SIZE(buf->planes); i++) {

        if (!res_handles[i])
            continue;

        plane = &buf->planes[buf->num_planes++];
        plane->res_handle = res_handles[i];
        glGenFramebuffers(1, &plane->framebuffer);
        glGenTextures(1, &plane->texture);
        glBindTexture(GL_TEXTURE_2D, plane->texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    buf->handle = handle;
    buf->ctx = ctx;
    list_add(&buf->head, &ctx->buffers);

    return 0;
}

static void destroy_video_buffer(struct vrend_video_buffer *buf)
{
    unsigned i;
    struct vrend_video_plane *plane;

    if (!buf)
        return;

    list_del(&buf->head);

    for (i = 0; i < buf->num_planes; i++) {
        plane = &buf->planes[i];

        glDeleteTextures(1, &plane->texture);
        glDeleteFramebuffers(1, &plane->framebuffer);
        if (plane->egl_image == EGL_NO_IMAGE_KHR)
            eglDestroyImageKHR(eglGetCurrentDisplay(), plane->egl_image);
    }

    virgl_video_destroy_buffer(buf->buffer);

    free(buf);
}

void vrend_video_destroy_buffer(struct vrend_video_context *ctx,
                                uint32_t handle)
{
    struct vrend_video_buffer *buf = get_video_buffer(ctx, handle);

    destroy_video_buffer(buf);
}

struct vrend_video_context *vrend_video_create_context(struct vrend_context *ctx)
{
    struct vrend_video_context *vctx;

    vctx = (struct vrend_video_context *)calloc(1, sizeof(*vctx));
    if (vctx) {
        vctx->ctx = ctx;
        list_inithead(&vctx->codecs);
        list_inithead(&vctx->buffers);
    }

    return vctx;
}

void vrend_video_destroy_context(struct vrend_video_context *ctx)
{
   list_for_each_entry_safe(struct vrend_video_codec, vcdc, &ctx->codecs, head)
      destroy_video_codec(vcdc);

   list_for_each_entry_safe(struct vrend_video_buffer, vbuf, &ctx->buffers, head)
      destroy_video_buffer(vbuf);

   free(ctx);
}

int vrend_video_begin_frame(struct vrend_video_context *ctx,
                            uint32_t cdc_handle,
                            uint32_t tgt_handle)
{
    struct vrend_video_codec *cdc = get_video_codec(ctx, cdc_handle);
    struct vrend_video_buffer *tgt = get_video_buffer(ctx, tgt_handle);

    if (!cdc || !tgt)
        return -1;

    return virgl_video_begin_frame(cdc->codec, tgt->buffer);
}

static void modify_h264_picture_desc(struct vrend_video_codec *cdc,
                                     struct vrend_video_buffer *tgt,
                                     struct virgl_h264_picture_desc *desc)
{
    unsigned i;
    struct vrend_video_buffer *vbuf;

    (void)tgt;

    for (i = 0; i < ARRAY_SIZE(desc->buffer_id); i++) {
        vbuf = get_video_buffer(cdc->ctx, desc->buffer_id[i]);
        desc->buffer_id[i] = virgl_video_buffer_id(vbuf ? vbuf->buffer : NULL);
    }
}

static void modify_h265_picture_desc(struct vrend_video_codec *cdc,
                                     struct vrend_video_buffer *tgt,
                                     struct virgl_h265_picture_desc *desc)
{
    unsigned i;
    struct vrend_video_buffer *vbuf;

    (void)tgt;

    for (i = 0; i < ARRAY_SIZE(desc->ref); i++) {
        vbuf = get_video_buffer(cdc->ctx, desc->ref[i]);
        desc->ref[i] = virgl_video_buffer_id(vbuf ? vbuf->buffer : NULL);
    }
}

static void modify_mpeg12_picture_desc(struct vrend_video_codec *cdc,
                                       struct vrend_video_buffer *tgt,
                                       struct virgl_mpeg12_picture_desc *desc)
{
    unsigned i;
    struct vrend_video_buffer *vbuf;

    (void)tgt;

    for (i = 0; i < ARRAY_SIZE(desc->ref); i++) {
        vbuf = get_video_buffer(cdc->ctx, desc->ref[i]);
        desc->ref[i] = virgl_video_buffer_id(vbuf ? vbuf->buffer : NULL);
    }
}


static void modify_mjpeg_picture_desc(struct vrend_video_codec *cdc,
                                      struct vrend_video_buffer *tgt,
                                      struct virgl_mjpeg_picture_desc *desc)
{
    (void)cdc;
    (void)tgt;
    (void)desc;
}

static void modify_vc1_picture_desc(struct vrend_video_codec *cdc,
                                    struct vrend_video_buffer *tgt,
                                    struct virgl_vc1_picture_desc *desc)
{
    unsigned i;
    struct vrend_video_buffer *vbuf;

    (void)tgt;

    for (i = 0; i < ARRAY_SIZE(desc->ref); i++) {
        vbuf = get_video_buffer(cdc->ctx, desc->ref[i]);
        desc->ref[i] = virgl_video_buffer_id(vbuf ? vbuf->buffer : NULL);
    }
}

static void modify_vp9_picture_desc(struct vrend_video_codec *cdc,
                                     struct vrend_video_buffer *tgt,
                                     struct virgl_vp9_picture_desc *desc)
{
    unsigned i;
    struct vrend_video_buffer *vbuf;

    (void)tgt;

    for (i = 0; i < ARRAY_SIZE(desc->ref); i++) {
        vbuf = get_video_buffer(cdc->ctx, desc->ref[i]);
        desc->ref[i] = virgl_video_buffer_id(vbuf ? vbuf->buffer : NULL);
    }
}

static void modify_av1_picture_desc(struct vrend_video_codec *cdc,
                                    struct vrend_video_buffer *tgt,
                                    struct virgl_av1_picture_desc *desc)
{
    unsigned i;
    struct vrend_video_buffer *vbuf;

    (void)tgt;

    for (i = 0; i < ARRAY_SIZE(desc->ref); i++) {
        vbuf = get_video_buffer(cdc->ctx, desc->ref[i]);
        desc->ref[i] = virgl_video_buffer_id(vbuf ? vbuf->buffer : NULL);
    }

    vbuf = get_video_buffer(cdc->ctx, desc->film_grain_target);
    desc->film_grain_target = virgl_video_buffer_id(vbuf ? vbuf->buffer : NULL);
}

static void modify_picture_desc(struct vrend_video_codec *cdc,
                                struct vrend_video_buffer *tgt,
                                union virgl_picture_desc *desc)
{
    switch(virgl_video_codec_profile(cdc->codec)) {
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_CONSTRAINED_BASELINE:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_MAIN:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_EXTENDED:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH10:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH422:
    case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH444:
        modify_h264_picture_desc(cdc, tgt, &desc->h264);
        break;
    case PIPE_VIDEO_PROFILE_HEVC_MAIN:
    case PIPE_VIDEO_PROFILE_HEVC_MAIN_10:
    case PIPE_VIDEO_PROFILE_HEVC_MAIN_STILL:
    case PIPE_VIDEO_PROFILE_HEVC_MAIN_12:
    case PIPE_VIDEO_PROFILE_HEVC_MAIN_444:
        modify_h265_picture_desc(cdc, tgt, &desc->h265);
        break;
    case PIPE_VIDEO_PROFILE_MPEG2_MAIN:
    case PIPE_VIDEO_PROFILE_MPEG2_SIMPLE:
        modify_mpeg12_picture_desc(cdc, tgt, &desc->mpeg12);
        break;
    case PIPE_VIDEO_PROFILE_JPEG_BASELINE:
        modify_mjpeg_picture_desc(cdc, tgt, &desc->mjpeg);
        break;
    case PIPE_VIDEO_PROFILE_VC1_SIMPLE:
    case PIPE_VIDEO_PROFILE_VC1_MAIN:
    case PIPE_VIDEO_PROFILE_VC1_ADVANCED:
        modify_vc1_picture_desc(cdc, tgt, &desc->vc1);
        break;
    case PIPE_VIDEO_PROFILE_VP9_PROFILE0:
    case PIPE_VIDEO_PROFILE_VP9_PROFILE2:
        modify_vp9_picture_desc(cdc, tgt, &desc->vp9);
        break;
    case PIPE_VIDEO_PROFILE_AV1_MAIN:
        modify_av1_picture_desc(cdc, tgt, &desc->av1);
        break;
    default:
        break;
    }
}

int vrend_video_decode_bitstream(struct vrend_video_context *ctx,
                                 uint32_t cdc_handle,
                                 uint32_t tgt_handle,
                                 uint32_t desc_handle,
                                 unsigned num_buffers,
                                 const uint32_t *buffer_handles,
                                 const uint32_t *buffer_sizes)
{
    int err = -1;
    unsigned i, num_bs, *bs_sizes = NULL;
    void **bs_buffers = NULL;
    struct vrend_resource *res;
    struct vrend_video_codec  *cdc = get_video_codec(ctx, cdc_handle);
    struct vrend_video_buffer *tgt = get_video_buffer(ctx, tgt_handle);
    union virgl_picture_desc desc;

    if (!cdc || !tgt){
        virgl_error("video codec: %p, video buffer: %p, invalid.\n", (void *)cdc, (void *)tgt);
        return -1;
    }

    bs_buffers = calloc(num_buffers, sizeof(void *));
    if (!bs_buffers) {
        virgl_error("%s: alloc bs_buffers failed\n", __func__);
        return -1;
    }

    bs_sizes = calloc(num_buffers, sizeof(unsigned));
    if (!bs_sizes) {
        virgl_error("%s: alloc bs_sizes failed\n", __func__);
        goto err;
    }

    for (i = 0, num_bs = 0; i < num_buffers; i++) {
        res = vrend_renderer_ctx_res_lookup(ctx->ctx, buffer_handles[i]);
        if (!res || !res->ptr) {
            virgl_warn("%s: bs res %d invalid or not found",
                       __func__, buffer_handles[i]);
            continue;
        }

        vrend_read_from_iovec(res->iov, res->num_iovs, 0,
                              res->ptr, buffer_sizes[i]);
        bs_buffers[num_bs] = res->ptr;
        bs_sizes[num_bs] = buffer_sizes[i];
        num_bs++;
    }

    res = vrend_renderer_ctx_res_lookup(ctx->ctx, desc_handle);
    if (!res) {
        virgl_error("%s: desc res %d not found\n", __func__, desc_handle);
        goto err;
    }
    memset(&desc, 0, sizeof(desc));
    vrend_read_from_iovec(res->iov, res->num_iovs, 0, (char *)(&desc),
                          MIN2(res->base.width0, sizeof(desc)));
    modify_picture_desc(cdc, tgt, &desc);

    err = virgl_video_decode_bitstream(cdc->codec, tgt->buffer, &desc,
                           num_bs, (const void * const *)bs_buffers, bs_sizes);

err:
    free(bs_buffers);
    free(bs_sizes);

    return err;
}

int vrend_video_encode_bitstream(struct vrend_video_context *ctx,
                                 uint32_t cdc_handle,
                                 uint32_t src_handle,
                                 uint32_t dest_handle,
                                 uint32_t desc_handle,
                                 uint32_t feed_handle)
{
    union virgl_picture_desc desc;
    struct vrend_resource *dest_res, *desc_res, *feed_res;
    struct vrend_video_codec  *cdc = get_video_codec(ctx, cdc_handle);
    struct vrend_video_buffer *src = get_video_buffer(ctx, src_handle);

    if (!cdc || !src)
        return -1;

    /* Feedback resource */
    feed_res = vrend_renderer_ctx_res_lookup(ctx->ctx, feed_handle);
    if (!feed_res) {
        virgl_error("%s: feedback res %d not found\n", __func__, feed_handle);
        return -1;
    }

    /* Picture descriptor resource */
    desc_res = vrend_renderer_ctx_res_lookup(ctx->ctx, desc_handle);
    if (!desc_res) {
        virgl_error("%s: desc res %d not found\n", __func__, desc_handle);
        return -1;
    }
    memset(&desc, 0, sizeof(desc));
    vrend_read_from_iovec(desc_res->iov, desc_res->num_iovs, 0, (char *)(&desc),
                          MIN2(desc_res->base.width0, sizeof(desc)));

    /* Destination buffer resource. */
    dest_res = vrend_renderer_ctx_res_lookup(ctx->ctx, dest_handle);
    if (!dest_res) {
        virgl_error("%s: dest res %d not found\n", __func__, dest_handle);
        return -1;
    }

    cdc->feed_res = feed_res;
    cdc->dest_res = dest_res;

    return virgl_video_encode_bitstream(cdc->codec, src->buffer, &desc);
}

int vrend_video_end_frame(struct vrend_video_context *ctx,
                          uint32_t cdc_handle,
                          uint32_t tgt_handle)
{
    struct vrend_video_codec *cdc = get_video_codec(ctx, cdc_handle);
    struct vrend_video_buffer *tgt = get_video_buffer(ctx, tgt_handle);

    if (!cdc || !tgt)
        return -1;

    return virgl_video_end_frame(cdc->codec, tgt->buffer);
}

