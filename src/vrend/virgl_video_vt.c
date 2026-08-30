/**************************************************************************
 *
 * Copyright (C) 2026 Gustavo Noronha Silva
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
 * The macOS video codec backend: VideoToolbox behind the virgl_video.h interface.
 *
 * virgl_video.c implements the same header against VA-API, which does not exist on
 * macOS. The two are never built together — src/meson.build picks one by host OS —
 * so this file owns the definitions of `struct virgl_video_codec` and
 * `struct virgl_video_buffer` on darwin.
 *
 * The two backends are shaped very differently, and the difference is all in our
 * favour. VA-API takes a *parsed* picture: the caller hands it slice parameters,
 * quantization tables and an explicit list of reference surfaces, and the backend
 * must maintain the decoded-picture buffer itself. VideoToolbox takes whole
 * compressed frames and parses them itself, tracking its own reference state. So
 * for VP9 — where the guest's slice data buffers already contain the complete frame
 * — this backend is mostly plumbing: concatenate the buffers, wrap them in a
 * CMSampleBuffer, and hand the decoded CVPixelBuffer's planes back through the
 * decode_completed callback. The whole `desc->ref[16]` apparatus is simply unused.
 *
 * Scope: VP9 profile 0 decode. H.264 and HEVC are absent from stock Fedora's mesa
 * (built `-Dvideo-codecs=all_free`, enforced driver-independently in the VA
 * frontend's vl_codec.c), so they cannot be exercised by an unmodified guest and
 * would additionally need SPS/PPS re-serialization, which VP9 does not. AV1 would
 * need an OBU serializer built from parsed VA parameters — a much larger job than
 * this file, and only useful on M3 and later. Encode is not implemented.
 *
 * Everything asserted here about VideoToolbox's behaviour was measured first;
 * see spikes/vt-vp9-decode in the limina tree.
 */

#include <CoreFoundation/CoreFoundation.h>
#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <VideoToolbox/VideoToolbox.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pipe/p_video_state.h"
#include "util/u_formats.h"
#include "util/u_memory.h"
#include "virgl_hw.h"
#include "virgl_video_hw.h"
#include "virgl_util.h"
#include "virgl_video.h"

/* Advertised ceiling. VideoToolbox's VP9 decoder goes higher on current silicon,
 * but nothing we can query says by how much, and the guest only uses this to size
 * its surfaces — 4K fits comfortably below it. */
#define VT_MAX_WIDTH  4096
#define VT_MAX_HEIGHT 4096

/* Declared in the vpcC box. Levels constrain the *stream*, so declaring the highest
 * one avoids a needless rejection; VideoToolbox parses the real bitstream anyway. */
#define VP9_LEVEL_MAX 61

struct virgl_video_codec {
    enum pipe_video_profile profile;
    enum pipe_video_entrypoint entrypoint;
    uint32_t width;
    uint32_t height;
    void *opaque;

    /* Rebuilt whenever the stream's shape changes; see ensure_session(). */
    VTDecompressionSessionRef session;
    CMVideoFormatDescriptionRef format;
    uint32_t session_width;
    uint32_t session_height;
    uint8_t session_profile;
    uint8_t session_bit_depth;
    uint8_t session_subsampling;
    enum pipe_format session_target_format;

    /* AV1 fixture capture only: how many frames have been recorded. */
    unsigned av1_capture_seq;

    /* The picture in flight: bitstream accumulated across decode_bitstream calls,
     * and the target the completed picture belongs to. */
    uint8_t *bitstream;
    size_t bitstream_len;
    size_t bitstream_cap;
    struct virgl_video_buffer *target;

    /* The decoded picture, parked by the output callback for end_frame to deliver.
     * See decode_output() for why it cannot be delivered where it arrives. */
    CVPixelBufferRef pending;

    /* Frame shape from the current picture descriptor, used to build the vpcC. */
    uint8_t frame_profile;
    uint8_t frame_bit_depth;
    uint8_t frame_subsampling;
    uint32_t frame_width;
    uint32_t frame_height;
};

struct virgl_video_buffer {
    enum pipe_format format;
    uint32_t width;
    uint32_t height;
    uint32_t id;
    void *opaque;
};

static struct virgl_video_callbacks *video_cbs;
static uint32_t next_buffer_id = 1;

/* LIMINA_VIDEO_TRACE=1 narrates the decode path to the worker log. virgl_error()
 * goes to a logger the embedder may not have installed, and the failure modes here
 * are all "nothing happened", which a silent path cannot distinguish. */
static int vt_trace = -1;

#define VT_TRACE(...) do {                                              \
    if (vt_trace < 0)                                                   \
        vt_trace = getenv("LIMINA_VIDEO_TRACE") ? 1 : 0;                \
    if (vt_trace)                                                       \
        fprintf(stderr, "[VIDEO] " __VA_ARGS__);                        \
} while (0)

/* Where to record AV1 fixtures, or NULL when not capturing. Set LIMINA_AV1_CAPTURE to a
 * directory; see docs/design/av1-decode.md, phase 0. */
static const char *av1_capture_dir(void)
{
    static const char *dir;
    static bool looked;

    if (!looked) {
        dir = getenv("LIMINA_AV1_CAPTURE");
        if (dir && !*dir)
            dir = NULL;
        looked = true;
    }
    return dir;
}

/* Whether the running Mac has silicon for a codec. Must be asked after the
 * supplemental registration below: without it VP9 reads as unsupported on hardware
 * that has it. */
static bool vt_can_decode(CMVideoCodecType codec)
{
    return VTIsHardwareDecodeSupported(codec);
}

int virgl_video_init(int drm_fd, struct virgl_video_callbacks *cbs, unsigned int flags)
{
    (void)drm_fd; /* VideoToolbox opens no device node. */
    (void)flags;

    /* VP9 and AV1 arrive as supplemental decoders; VTIsHardwareDecodeSupported
     * answers "no" for them until this call has been made. */
    VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_VP9);
    VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_AV1);

    video_cbs = cbs;

    return 0;
}

void virgl_video_destroy(void)
{
    video_cbs = NULL;
}

int virgl_video_fill_caps(union virgl_caps *caps)
{
    struct virgl_video_caps *vcaps;

    if (!caps)
        return -1;

    caps->v2.num_video_caps = 0;

    /* A host with no VP9 silicon advertises nothing, and the guest's
     * virgl_get_video_param() then reports no profiles — the driver still loads and
     * initializes, it just offers no hardware decode. That is the whole of the
     * stock-tier fallback: there is nothing to gate. */
    if (vt_can_decode(kCMVideoCodecType_VP9)) {
        VT_TRACE("fill_caps: advertising VP9 profile0 decode\n");
        vcaps = &caps->v2.video_caps[caps->v2.num_video_caps++];
        memset(vcaps, 0, sizeof(*vcaps));
        vcaps->profile = PIPE_VIDEO_PROFILE_VP9_PROFILE0;
        vcaps->entrypoint = PIPE_VIDEO_ENTRYPOINT_BITSTREAM;
        vcaps->max_level = 0;
        vcaps->stacked_frames = 0;
        vcaps->max_width = VT_MAX_WIDTH;
        vcaps->max_height = VT_MAX_HEIGHT;
        vcaps->prefered_format = PIPE_FORMAT_NV12;
        vcaps->max_macroblocks = 1;
        vcaps->npot_texture = 1;
        vcaps->supports_progressive = 1;
        vcaps->supports_interlaced = 0;
        vcaps->prefers_interlaced = 0;
        vcaps->max_temporal_layers = 0;
    }

    /* AV1 needs M3-or-later silicon. A capture build forces the advertisement on hardware
     * that cannot decode it, because capture never decodes: the picture descriptors are a
     * pure function of the *guest's* own bitstream parse, so a machine with no AV1 silicon
     * produces exactly the fixtures an M3 would. */
    if (vt_can_decode(kCMVideoCodecType_AV1) || av1_capture_dir()) {
        VT_TRACE("fill_caps: advertising AV1 main decode%s\n",
                 vt_can_decode(kCMVideoCodecType_AV1) ? "" : " (capture build, no silicon)");
        vcaps = &caps->v2.video_caps[caps->v2.num_video_caps++];
        memset(vcaps, 0, sizeof(*vcaps));
        vcaps->profile = PIPE_VIDEO_PROFILE_AV1_MAIN;
        vcaps->entrypoint = PIPE_VIDEO_ENTRYPOINT_BITSTREAM;
        vcaps->max_level = 0;
        vcaps->stacked_frames = 0;
        vcaps->max_width = VT_MAX_WIDTH;
        vcaps->max_height = VT_MAX_HEIGHT;
        vcaps->prefered_format = PIPE_FORMAT_NV12;
        vcaps->max_macroblocks = 1;
        vcaps->npot_texture = 1;
        vcaps->supports_progressive = 1;
        vcaps->supports_interlaced = 0;
        vcaps->prefers_interlaced = 0;
        vcaps->max_temporal_layers = 0;
    }

    return 0;
}

struct virgl_video_codec *virgl_video_create_codec(
        const struct virgl_video_create_codec_args *args)
{
    struct virgl_video_codec *codec;

    if (!args)
        return NULL;

    /* Decode only, and only what fill_caps advertised. */
    if (args->entrypoint != PIPE_VIDEO_ENTRYPOINT_BITSTREAM) {
        virgl_error("video: entrypoint %d not supported by the VideoToolbox backend\n",
                    args->entrypoint);
        return NULL;
    }
    if (args->profile != PIPE_VIDEO_PROFILE_VP9_PROFILE0 &&
        args->profile != PIPE_VIDEO_PROFILE_AV1_MAIN) {
        virgl_error("video: profile %d not supported by the VideoToolbox backend\n",
                    args->profile);
        return NULL;
    }

    codec = CALLOC_STRUCT(virgl_video_codec);
    if (!codec)
        return NULL;

    codec->profile = args->profile;
    codec->entrypoint = args->entrypoint;
    codec->width = args->width;
    codec->height = args->height;
    codec->opaque = args->opaque;

    VT_TRACE("create_codec: profile %d %ux%u\n", args->profile, args->width, args->height);

    return codec;
}

static void destroy_session(struct virgl_video_codec *codec)
{
    if (codec->session) {
        VTDecompressionSessionInvalidate(codec->session);
        CFRelease(codec->session);
        codec->session = NULL;
    }
    if (codec->format) {
        CFRelease(codec->format);
        codec->format = NULL;
    }
}

void virgl_video_destroy_codec(struct virgl_video_codec *codec)
{
    if (!codec)
        return;

    destroy_session(codec);
    if (codec->pending)
        CFRelease(codec->pending);
    free(codec->bitstream);
    free(codec);
}

enum pipe_video_profile virgl_video_codec_profile(const struct virgl_video_codec *codec)
{
    return codec ? codec->profile : PIPE_VIDEO_PROFILE_UNKNOWN;
}

void *virgl_video_codec_opaque_data(struct virgl_video_codec *codec)
{
    return codec ? codec->opaque : NULL;
}

struct virgl_video_buffer *virgl_video_create_buffer(
        const struct virgl_video_create_buffer_args *args)
{
    struct virgl_video_buffer *buffer;

    if (!args)
        return NULL;

    buffer = CALLOC_STRUCT(virgl_video_buffer);
    if (!buffer)
        return NULL;

    buffer->format = args->format;
    buffer->width = args->width;
    buffer->height = args->height;
    buffer->opaque = args->opaque;
    buffer->id = next_buffer_id++;

    return buffer;
}

void virgl_video_destroy_buffer(struct virgl_video_buffer *buffer)
{
    free(buffer);
}

uint32_t virgl_video_buffer_id(const struct virgl_video_buffer *buffer)
{
    return buffer ? buffer->id : 0;
}

void *virgl_video_buffer_opaque_data(struct virgl_video_buffer *buffer)
{
    return buffer ? buffer->opaque : NULL;
}

/* Which CoreVideo layout to decode into.
 *
 * The guest picks the layout when it allocates the target, and it does not always
 * pick the one we advertise as preferred: ffmpeg's VA-API path allocates I420
 * (three planes) for decode targets while asking for NV12 elsewhere. VideoToolbox
 * will produce either, so the cheapest correct thing is to ask it for whatever the
 * target already is rather than converting afterwards.
 *
 * YV12 differs from I420 only in plane order (Y,V,U against Y,U,V), so it shares
 * the planar output and swaps planes on delivery. */
static int32_t cv_format_for(enum pipe_format format)
{
    switch (format) {
    case PIPE_FORMAT_NV12:
        return kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    case PIPE_FORMAT_IYUV:
    case PIPE_FORMAT_YV12:
        return kCVPixelFormatType_420YpCbCr8Planar;
    default:
        return 0;
    }
}

/* The VP9 codec configuration record VideoToolbox wants in the format
 * description's extension atoms: a 4-byte FullBox header then the record itself. */
static CFDataRef build_vpcc(uint8_t profile, uint8_t bit_depth, uint8_t subsampling)
{
    uint8_t box[12] = {0};

    box[0] = 1;                 /* version; box[1..3] are flags and stay zero */
    box[4] = profile;
    box[5] = VP9_LEVEL_MAX;
    box[6] = (uint8_t)((bit_depth << 4) | (subsampling << 1) | 0 /* studio range */);
    box[7] = 2;                 /* colourPrimaries: unspecified */
    box[8] = 2;                 /* transferCharacteristics: unspecified */
    box[9] = 2;                 /* matrixCoefficients: unspecified */
    /* box[10..11]: codecInitializationDataSize = 0 */

    return CFDataCreate(kCFAllocatorDefault, box, sizeof(box));
}

/* Hand the decoded picture to vrend as a "dmabuf" whose planes are mapped rather
 * than exported — see the `map` member of struct virgl_video_dma_buf_plane. The
 * mapping lives only for the duration of the callback, which is all vrend needs:
 * it copies each plane into the guest-visible resource and returns. */
static void deliver_picture(struct virgl_video_codec *codec, CVPixelBufferRef pixbuf)
{
    struct virgl_video_dma_buf dmabuf;
    size_t num_planes;

    if (!video_cbs || !video_cbs->decode_completed)
        return;

    if (CVPixelBufferLockBaseAddress(pixbuf, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
        virgl_error("video: could not map the decoded picture\n");
        return;
    }

    num_planes = CVPixelBufferGetPlaneCount(pixbuf);
    if (num_planes > ARRAY_SIZE(dmabuf.planes))
        num_planes = ARRAY_SIZE(dmabuf.planes);

    memset(&dmabuf, 0, sizeof(dmabuf));
    dmabuf.buf = codec->target;
    dmabuf.width = (uint32_t)CVPixelBufferGetWidth(pixbuf);
    dmabuf.height = (uint32_t)CVPixelBufferGetHeight(pixbuf);
    dmabuf.flags = VIRGL_VIDEO_DMABUF_READ_ONLY;
    dmabuf.num_planes = (uint32_t)num_planes;

    /* CoreVideo's planar output is Y,Cb,Cr; YV12 wants Y,Cr,Cb. */
    const bool swap_chroma = codec->target && codec->target->format == PIPE_FORMAT_YV12;

    for (size_t i = 0; i < num_planes; i++) {
        size_t src = i;
        if (swap_chroma && num_planes == 3 && i > 0)
            src = 3 - i;

        dmabuf.planes[i].fd = -1;
        dmabuf.planes[i].map = CVPixelBufferGetBaseAddressOfPlane(pixbuf, src);
        dmabuf.planes[i].pitch = (uint32_t)CVPixelBufferGetBytesPerRowOfPlane(pixbuf, src);
        dmabuf.planes[i].size =
            dmabuf.planes[i].pitch * (uint32_t)CVPixelBufferGetHeightOfPlane(pixbuf, src);
    }

    VT_TRACE("deliver: %u planes, %ux%u, pitch0 %u\n", dmabuf.num_planes,
             dmabuf.width, dmabuf.height, dmabuf.planes[0].pitch);
    video_cbs->decode_completed(codec, &dmabuf);

    CVPixelBufferUnlockBaseAddress(pixbuf, kCVPixelBufferLock_ReadOnly);
}

/* Decoding is synchronous in ORDER — the callback always runs before
 * VTDecompressionSessionDecodeFrame returns, because
 * kVTDecodeFrame_EnableAsynchronousDecompression is not passed — but it runs on one
 * of VideoToolbox's own threads, not the caller's. That thread has no EGL context
 * current, so every GL call made from here is dropped with "called without a
 * rendering context" and the guest silently reads an untouched surface.
 *
 * So the picture is only parked here; end_frame delivers it once DecodeFrame has
 * returned and we are back on the vrend thread that holds the context. Ordering
 * makes that safe: exactly one picture is outstanding per frame. */
static void decode_output(void *codec_ref, void *frame_ref, OSStatus status,
                          VTDecodeInfoFlags info_flags, CVImageBufferRef image,
                          CMTime pts, CMTime duration)
{
    struct virgl_video_codec *codec = codec_ref;

    (void)frame_ref;
    (void)info_flags;
    (void)pts;
    (void)duration;

    if (status != noErr) {
        virgl_error("video: decode failed, status %d\n", (int)status);
        return;
    }
    if (!image) {
        /* VideoToolbox emits a picture even for frames the stream never shows, so a
         * missing one is a real failure rather than the hidden-alt-ref case. */
        virgl_error("video: decode returned no image buffer\n");
        return;
    }

    VT_TRACE("decode_output: got a picture\n");

    if (codec->pending)
        CFRelease(codec->pending);
    codec->pending = (CVPixelBufferRef)CFRetain(image);
}

/* Create (or recreate) the decompression session. VP9 streams can change resolution
 * or bit depth at a keyframe, so the session is keyed on the shape of the frame we
 * are about to decode rather than built once from the codec's creation arguments. */
static int ensure_session(struct virgl_video_codec *codec)
{
    CFDataRef vpcc;
    CFMutableDictionaryRef atoms, extensions, pixel_attrs, empty;
    CFNumberRef pixel_format_num;
    VTDecompressionOutputCallbackRecord callback;
    enum pipe_format target_format = codec->target ? codec->target->format : PIPE_FORMAT_NV12;
    int32_t pixel_format = cv_format_for(target_format);
    OSStatus status;

    if (!pixel_format) {
        virgl_error("video: no CoreVideo layout for target format %d\n", target_format);
        return -1;
    }

    if (codec->session &&
        codec->session_width == codec->frame_width &&
        codec->session_height == codec->frame_height &&
        codec->session_profile == codec->frame_profile &&
        codec->session_bit_depth == codec->frame_bit_depth &&
        codec->session_subsampling == codec->frame_subsampling &&
        codec->session_target_format == target_format)
        return 0;

    destroy_session(codec);

    vpcc = build_vpcc(codec->frame_profile, codec->frame_bit_depth,
                      codec->frame_subsampling);
    if (!vpcc)
        return -1;

    atoms = CFDictionaryCreateMutable(kCFAllocatorDefault, 1,
                                      &kCFTypeDictionaryKeyCallBacks,
                                      &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(atoms, CFSTR("vpcC"), vpcc);
    CFRelease(vpcc);

    extensions = CFDictionaryCreateMutable(kCFAllocatorDefault, 1,
                                           &kCFTypeDictionaryKeyCallBacks,
                                           &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(extensions,
                         kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
                         atoms);
    CFRelease(atoms);

    status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault, kCMVideoCodecType_VP9,
                                            codec->frame_width, codec->frame_height,
                                            extensions, &codec->format);
    CFRelease(extensions);
    if (status != noErr) {
        virgl_error("video: CMVideoFormatDescriptionCreate failed, status %d\n", (int)status);
        return -1;
    }

    /* The target's own layout, with an IOSurface behind it: the planes map straight
     * onto the guest's video buffer resources, and the IOSurface backing is what a
     * future zero-copy import would need. */
    pixel_attrs = CFDictionaryCreateMutable(kCFAllocatorDefault, 2,
                                            &kCFTypeDictionaryKeyCallBacks,
                                            &kCFTypeDictionaryValueCallBacks);
    pixel_format_num = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pixel_format);
    CFDictionarySetValue(pixel_attrs, kCVPixelBufferPixelFormatTypeKey, pixel_format_num);
    CFRelease(pixel_format_num);
    empty = (CFMutableDictionaryRef)CFDictionaryCreate(kCFAllocatorDefault, NULL, NULL, 0,
                                                       &kCFTypeDictionaryKeyCallBacks,
                                                       &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(pixel_attrs, kCVPixelBufferIOSurfacePropertiesKey, empty);
    CFRelease(empty);

    callback.decompressionOutputCallback = decode_output;
    callback.decompressionOutputRefCon = codec;

    VT_TRACE("ensure_session: %ux%u prof %u depth %u sub %u target fmt %d cv '%.4s'\n",
             codec->frame_width, codec->frame_height, codec->frame_profile,
             codec->frame_bit_depth, codec->frame_subsampling, target_format,
             (const char *)&(uint32_t){ __builtin_bswap32((uint32_t)pixel_format) });
    status = VTDecompressionSessionCreate(kCFAllocatorDefault, codec->format, NULL,
                                          pixel_attrs, &callback, &codec->session);
    CFRelease(pixel_attrs);
    if (status != noErr) {
        virgl_error("video: VTDecompressionSessionCreate failed, status %d\n", (int)status);
        CFRelease(codec->format);
        codec->format = NULL;
        return -1;
    }

    codec->session_width = codec->frame_width;
    codec->session_height = codec->frame_height;
    codec->session_profile = codec->frame_profile;
    codec->session_bit_depth = codec->frame_bit_depth;
    codec->session_subsampling = codec->frame_subsampling;
    codec->session_target_format = target_format;

    return 0;
}

int virgl_video_begin_frame(struct virgl_video_codec *codec,
                            struct virgl_video_buffer *target)
{
    if (!codec || !target)
        return -1;

    codec->bitstream_len = 0;
    codec->target = target;
    VT_TRACE("begin_frame: target %u\n", target->id);

    return 0;
}

/* The guest may split one picture across several calls, so this only accumulates;
 * the decode itself happens in end_frame, mirroring vaEndPicture. */
/* AV1, phase 0: record the descriptor and the tile bytes, decode nothing.
 *
 * The guest hands us a fully parsed picture descriptor plus tile payload only -- the frame
 * header was consumed by ffmpeg's own parse and destroyed at the ffmpeg -> VA-API boundary
 * (libavcodec/av1dec.c passes `raw_tile_group->tile_data.data` to its hwaccel). Synthesizing
 * a conformant OBU_FRAME_HEADER from this descriptor is the whole of the AV1 work, and these
 * fixtures are what let that be developed and tested offline, against dav1d, on a machine
 * with no AV1 silicon.
 *
 * The descriptor is written raw. It is an ABI-shaped dump consumed only by a spike built from
 * this same tree on the same machine, never a persisted format. */
static int av1_decode_bitstream(struct virgl_video_codec *codec,
                                const union virgl_picture_desc *desc,
                                unsigned num_buffers,
                                const void * const *buffers,
                                const unsigned *sizes)
{
    const char *dir = av1_capture_dir();
    const struct virgl_av1_picture_desc *av1 = &desc->av1;
    char path[1024];
    FILE *f;

    if (!dir) {
        virgl_error("video: AV1 decode is not implemented yet "
                    "(set LIMINA_AV1_CAPTURE to record fixtures)\n");
        return -1;
    }

    snprintf(path, sizeof(path), "%s/frame%05u.desc", dir, codec->av1_capture_seq);
    if ((f = fopen(path, "wb"))) {
        fwrite(av1, sizeof(*av1), 1, f);
        fclose(f);
    } else {
        virgl_error("video: cannot write AV1 fixture %s\n", path);
        return -1;
    }

    snprintf(path, sizeof(path), "%s/frame%05u.tile", dir, codec->av1_capture_seq);
    if ((f = fopen(path, "wb"))) {
        for (unsigned i = 0; i < num_buffers; i++)
            if (buffers[i] && sizes[i])
                fwrite(buffers[i], sizes[i], 1, f);
        fclose(f);
    }

    VT_TRACE("av1 capture %u: %ux%u frame_type %u show %u primary_ref %u tiles %ux%u "
             "slices %u, %u buffers\n",
             codec->av1_capture_seq,
             av1->picture_parameter.frame_width, av1->picture_parameter.frame_height,
             av1->picture_parameter.pic_info_fields.frame_type,
             av1->picture_parameter.pic_info_fields.show_frame,
             av1->picture_parameter.primary_ref_frame,
             av1->picture_parameter.tile_cols, av1->picture_parameter.tile_rows,
             av1->slice_parameter.slice_count, num_buffers);

    codec->av1_capture_seq++;

    /* Report success with no picture. ffmpeg keeps submitting frames because every
     * descriptor comes from its own parse, not from anything we return. */
    return 0;
}

int virgl_video_decode_bitstream(struct virgl_video_codec *codec,
                                 struct virgl_video_buffer *target,
                                 const union virgl_picture_desc *desc,
                                 unsigned num_buffers,
                                 const void * const *buffers,
                                 const unsigned *sizes)
{
    const struct virgl_vp9_picture_desc *vp9;
    size_t needed = codec->bitstream_len;

    if (!codec || !target || !desc || !buffers || !sizes)
        return -1;

    if (codec->profile == PIPE_VIDEO_PROFILE_AV1_MAIN)
        return av1_decode_bitstream(codec, desc, num_buffers, buffers, sizes);

    if (codec->profile != PIPE_VIDEO_PROFILE_VP9_PROFILE0)
        return -1;

    vp9 = &desc->vp9;
    codec->frame_profile = vp9->picture_parameter.profile;
    codec->frame_bit_depth =
        vp9->picture_parameter.bit_depth ? vp9->picture_parameter.bit_depth : 8;
    codec->frame_subsampling =
        (vp9->picture_parameter.pic_fields.subsampling_x &&
         vp9->picture_parameter.pic_fields.subsampling_y) ? 1 : 3;
    codec->frame_width = vp9->picture_parameter.frame_width
                       ? vp9->picture_parameter.frame_width : codec->width;
    codec->frame_height = vp9->picture_parameter.frame_height
                        ? vp9->picture_parameter.frame_height : codec->height;

    for (unsigned i = 0; i < num_buffers; i++)
        needed += sizes[i];

    if (needed > codec->bitstream_cap) {
        uint8_t *grown = realloc(codec->bitstream, needed);
        if (!grown) {
            virgl_error("video: out of memory growing the bitstream buffer\n");
            return -1;
        }
        codec->bitstream = grown;
        codec->bitstream_cap = needed;
    }

    for (unsigned i = 0; i < num_buffers; i++) {
        if (!buffers[i] || !sizes[i])
            continue;
        memcpy(codec->bitstream + codec->bitstream_len, buffers[i], sizes[i]);
        codec->bitstream_len += sizes[i];
    }

    VT_TRACE("decode_bitstream: %u buffers, %zu bytes total, frame %ux%u prof %u depth %u\n",
             num_buffers, codec->bitstream_len, codec->frame_width, codec->frame_height,
             codec->frame_profile, codec->frame_bit_depth);

    return 0;
}

int virgl_video_encode_bitstream(struct virgl_video_codec *codec,
                                 struct virgl_video_buffer *source,
                                 const union virgl_picture_desc *desc)
{
    (void)codec;
    (void)source;
    (void)desc;

    /* Not implemented: fill_caps advertises no encode entrypoint, so nothing should
     * reach here. */
    return -1;
}

int virgl_video_end_frame(struct virgl_video_codec *codec,
                          struct virgl_video_buffer *target)
{
    CMBlockBufferRef block = NULL;
    CMSampleBufferRef sample = NULL;
    VTDecodeInfoFlags info = 0;
    OSStatus status;
    int err = 0;

    if (!codec || !target)
        return -1;

    if (codec->profile == PIPE_VIDEO_PROFILE_AV1_MAIN)
        return 0; /* phase 0 capture: nothing was accumulated and nothing decodes */

    VT_TRACE("end_frame: %zu bytes\n", codec->bitstream_len);

    if (!codec->bitstream_len)
        return 0;

    codec->target = target;

    if (ensure_session(codec) < 0)
        return -1;

    /* kCFAllocatorNull: the sample borrows our accumulation buffer rather than
     * copying it. Safe because the decode is synchronous and the buffer outlives
     * the call. */
    status = CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, codec->bitstream,
                                                codec->bitstream_len, kCFAllocatorNull,
                                                NULL, 0, codec->bitstream_len, 0, &block);
    if (status != noErr) {
        virgl_error("video: CMBlockBufferCreateWithMemoryBlock failed, status %d\n",
                    (int)status);
        return -1;
    }

    const size_t sample_size = codec->bitstream_len;
    status = CMSampleBufferCreate(kCFAllocatorDefault, block, TRUE, NULL, NULL,
                                  codec->format, 1, 0, NULL, 1, &sample_size, &sample);
    if (status != noErr) {
        virgl_error("video: CMSampleBufferCreate failed, status %d\n", (int)status);
        CFRelease(block);
        return -1;
    }

    status = VTDecompressionSessionDecodeFrame(codec->session, sample, 0, NULL, &info);
    if (status != noErr) {
        virgl_error("video: VTDecompressionSessionDecodeFrame failed, status %d\n",
                    (int)status);
        err = -1;
    }

    CFRelease(sample);
    CFRelease(block);

    codec->bitstream_len = 0;

    /* Back on the vrend thread, with the GL context current: now the picture can be
     * copied into the guest's resources. */
    if (codec->pending) {
        deliver_picture(codec, codec->pending);
        CFRelease(codec->pending);
        codec->pending = NULL;
    } else if (!err) {
        virgl_error("video: decode produced no picture\n");
        err = -1;
    }

    return err;
}
