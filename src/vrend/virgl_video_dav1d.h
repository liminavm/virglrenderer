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

/* A software AV1 decoder, used where the host's hardware decoder cannot be trusted with
 * the stream. Today that is exactly one case -- super-resolution, which VideoToolbox
 * reconstructs correctly but does not hand back correctly (docs/hardening-backlog.md) --
 * and the backend switches to this for the rest of the codec's life once it sees one.
 *
 * The unit of work is the same synthesized temporal unit the hardware path submits, so
 * the two decoders are fed identical bytes and the serializer stays the single source of
 * frame headers.
 *
 * Deliberately not a general "software fallback" abstraction: it decodes AV1 and nothing
 * else, because AV1 is the only codec whose hardware path has a hole in it. */

#ifndef VIRGL_VIDEO_DAV1D_H
#define VIRGL_VIDEO_DAV1D_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct virgl_video_dav1d;

/* A picture the decoder is lending out. Valid until the next decode or release. */
struct virgl_dav1d_picture {
    const uint8_t *plane[3];    /* Y, U, V -- always three, always planar */
    uint32_t pitch[3];
    uint32_t width, height;
    uint8_t bpc;                /* bits per component: 8, or 10/12 in 16-bit containers */
};

/* Whether this build can decode in software at all. False when built without dav1d, in
 * which case the caller keeps refusing the frames it cannot trust. */
bool virgl_dav1d_available(void);

struct virgl_video_dav1d *virgl_dav1d_open(void);
void virgl_dav1d_close(struct virgl_video_dav1d *d);

/* Decode one temporal unit. Returns 1 when `out` was filled, 0 when the unit showed no
 * picture (a hidden frame -- normal, and the reason this cannot be a plain
 * one-unit-one-picture call), and <0 on error. The picture is owned by the decoder;
 * virgl_dav1d_release() returns it. */
int virgl_dav1d_decode(struct virgl_video_dav1d *d, const uint8_t *unit, size_t len,
                       struct virgl_dav1d_picture *out);

void virgl_dav1d_release(struct virgl_video_dav1d *d);

#endif /* VIRGL_VIDEO_DAV1D_H */
