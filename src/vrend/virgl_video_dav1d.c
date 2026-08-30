/**************************************************************************
 *
 * Copyright (C) 2026 The limina authors
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
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include "virgl_video_dav1d.h"

/* virgl_util.h carries virgl_error as a static inline and pulls mesa's util headers with
 * it. Those are unavailable outside a configured virglrenderer build, and this file is
 * deliberately buildable on its own so the decoder wrapper can be exercised against the
 * fixtures without the rest of the renderer (spikes/av1-obu-serializer/sw-oracle.c). */
#ifdef VIRGL_DAV1D_STANDALONE
void virgl_error(const char *fmt, ...);
#else
#include "virgl_util.h"
#endif

#include <stdlib.h>

#ifdef HAVE_DAV1D

#include <dav1d/dav1d.h>
#include <errno.h>
#include <string.h>

struct virgl_video_dav1d {
    Dav1dContext *ctx;
    Dav1dPicture pic;
    bool holding;               /* `pic` is lent out and owes an unref */
};

bool virgl_dav1d_available(void)
{
    return true;
}

struct virgl_video_dav1d *virgl_dav1d_open(void)
{
    struct virgl_video_dav1d *d = calloc(1, sizeof(*d));
    Dav1dSettings s;
    int err;

    if (!d)
        return NULL;

    dav1d_default_settings(&s);
    /* In-order and synchronous. This decoder exists to be correct where the hardware one
     * is not, and it is driven one unit at a time from the vrend thread; frame threading
     * would buy throughput at the cost of a picture arriving after the frame that asked
     * for it has already been answered. */
    s.n_threads = 1;
    s.max_frame_delay = 1;
    /* The guest allocates a surface for EVERY decoded frame, hidden ones included -- a
     * later show_existing_frame displays one without any of it reaching us. dav1d emits
     * only shown pictures by default, which would leave those surfaces never written.
     * Asking for invisible frames too restores one-picture-per-unit, which is both what
     * the hardware path does and what the caller's one-unit-one-target model assumes. */
    s.output_invisible_frames = 1;

    if ((err = dav1d_open(&d->ctx, &s))) {
        virgl_error("video: dav1d_open failed (%d)\n", err);
        free(d);
        return NULL;
    }
    return d;
}

void virgl_dav1d_close(struct virgl_video_dav1d *d)
{
    if (!d)
        return;
    virgl_dav1d_release(d);
    if (d->ctx)
        dav1d_close(&d->ctx);
    free(d);
}

void virgl_dav1d_release(struct virgl_video_dav1d *d)
{
    if (d && d->holding) {
        dav1d_picture_unref(&d->pic);
        d->holding = false;
    }
}

int virgl_dav1d_decode(struct virgl_video_dav1d *d, const uint8_t *unit, size_t len,
                       struct virgl_dav1d_picture *out)
{
    Dav1dData data = {0};
    uint8_t *buf;
    int err;

    if (!d || !d->ctx || !unit || !len)
        return -1;

    virgl_dav1d_release(d);

    /* dav1d keeps a reference to what it is given, so it gets its own copy rather than a
     * wrap of the caller's unit buffer, which is reused for the next frame. */
    buf = dav1d_data_create(&data, len);
    if (!buf) {
        virgl_error("video: dav1d_data_create failed for %zu bytes\n", len);
        return -1;
    }
    memcpy(buf, unit, len);

    while (data.sz) {
        err = dav1d_send_data(d->ctx, &data);
        if (err < 0 && err != DAV1D_ERR(EAGAIN)) {
            virgl_error("video: dav1d_send_data failed (%d) with %zu bytes left\n",
                        err, data.sz);
            dav1d_data_unref(&data);
            return -1;
        }
        /* EAGAIN means a picture must be drained before the rest of the unit fits. */
        if (err == DAV1D_ERR(EAGAIN))
            break;
    }

    err = dav1d_get_picture(d->ctx, &d->pic);
    if (err == DAV1D_ERR(EAGAIN)) {
        /* A hidden frame: decoded, stored as a reference, shown by a later frame. */
        dav1d_data_unref(&data);
        return 0;
    }
    if (err < 0) {
        virgl_error("video: dav1d_get_picture failed (%d)\n", err);
        dav1d_data_unref(&data);
        return -1;
    }
    d->holding = true;
    dav1d_data_unref(&data);

    memset(out, 0, sizeof(*out));
    out->plane[0] = d->pic.data[0];
    out->plane[1] = d->pic.data[1];
    out->plane[2] = d->pic.data[2];
    /* dav1d carries one stride for luma and one shared by both chroma planes. */
    out->pitch[0] = (uint32_t)d->pic.stride[0];
    out->pitch[1] = (uint32_t)d->pic.stride[1];
    out->pitch[2] = (uint32_t)d->pic.stride[1];
    out->width = (uint32_t)d->pic.p.w;
    out->height = (uint32_t)d->pic.p.h;
    out->bpc = (uint8_t)d->pic.p.bpc;
    return 1;
}

#else /* !HAVE_DAV1D */

bool virgl_dav1d_available(void)
{
    return false;
}

struct virgl_video_dav1d *virgl_dav1d_open(void)
{
    return NULL;
}

void virgl_dav1d_close(struct virgl_video_dav1d *d)
{
    (void)d;
}

void virgl_dav1d_release(struct virgl_video_dav1d *d)
{
    (void)d;
}

int virgl_dav1d_decode(struct virgl_video_dav1d *d, const uint8_t *unit, size_t len,
                       struct virgl_dav1d_picture *out)
{
    (void)d; (void)unit; (void)len; (void)out;
    return -1;
}

#endif /* HAVE_DAV1D */
