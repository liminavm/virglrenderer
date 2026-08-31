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
#include "virgl_video_av1_obu.h"
#include "virgl_video_dav1d.h"
#include "virgl_video_h264_ps.h"
#include "virgl_video_h265_ps.h"

/* H.264 arrives as a family of profiles rather than one, and every one of them takes the
 * same path here -- only profile_idc in the synthesized SPS differs. */
static bool is_h264(enum pipe_video_profile profile)
{
    return profile >= PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE &&
           profile <= PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH444;
}

static bool is_h265(enum pipe_video_profile profile)
{
    return profile == PIPE_VIDEO_PROFILE_HEVC_MAIN;
}

/* Both codecs synthesize parameter sets into frame_config and re-frame Annex-B access
 * units; only the syntax inside differs. */
static bool is_ps_codec(enum pipe_video_profile profile)
{
    return is_h264(profile) || is_h265(profile);
}

/* Advertised ceiling. VideoToolbox's VP9 decoder goes higher on current silicon,
 * but nothing we can query says by how much, and the guest only uses this to size
 * its surfaces — 4K fits comfortably below it. */
#define VT_MAX_WIDTH  4096
#define VT_MAX_HEIGHT 4096

/* Declared in the vpcC box. Levels constrain the *stream*, so declaring the highest
 * one avoids a needless rejection; VideoToolbox parses the real bitstream anyway. */
#define VP9_LEVEL_MAX 61

/* One temporal unit as it was submitted, kept so the software decoder can be brought up
 * to the same reference state the hardware decoder reached. */
struct unit_log {
    uint8_t *data;
    size_t len;
};

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

    /* Whether this host's VideoToolbox can decode AV1 at all. When it cannot, the codec is
     * still created so fixtures can be recorded, but nothing is submitted. */
    bool av1_decode;

    /* AV1 only. The guest's descriptor carries no bitstream, so one is synthesized: the
     * serializer holds the reference-slot model, and one hidden frame when the guest's DPB
     * is full, because which picture it evicted is only visible in the next descriptor. */
    struct virgl_av1_obu_state av1;
    struct virgl_av1_picture_desc av1_desc;
    bool av1_desc_valid;

    /* The target of a frame the serializer is holding. The guest has moved on to later
     * frames by the time that unit is emitted, so its destination has to be kept here.
     * NULL means the buffer was destroyed underneath us: the frame still has to be
     * decoded, or the decoder's reference list loses it, but its picture goes nowhere. */
    struct virgl_video_buffer *held_target;

    /* One temporal unit, built here and submitted as its own sample. */
    uint8_t *unit;
    size_t unit_cap;

    /* Codec configuration record (vpcC or av1C), and the one the live session was built
     * from -- an AV1 sequence header can change mid-stream.
     *
     * H.264 and HEVC reuse this as the carrier for their synthesized parameter sets, laid
     * end to end, so the existing "config changed => rebuild the session" comparison covers
     * a mid-stream parameter-set change for free. ps_len below says where the splits are. */
    uint8_t frame_config[256];
    size_t frame_config_len;
    uint8_t session_config[256];
    size_t session_config_len;

    /* How frame_config splits into parameter sets: SPS, PPS for H.264; VPS, SPS, PPS for
     * HEVC. Zero sets means nothing has been synthesized yet. */
    size_t ps_len[3];
    unsigned ps_count;

    /* The picture in flight: the bitstream accumulated across decode_bitstream calls. The
     * target is not kept here -- every path that needs one is handed it explicitly, because
     * a held frame's target is not the current frame's. */
    uint8_t *bitstream;
    size_t bitstream_len;
    size_t bitstream_cap;

    /* The decoded picture, parked by the output callback for end_frame to deliver.
     * See decode_output() for why it cannot be delivered where it arrives. */
    CVPixelBufferRef pending;

    /* Frame shape from the current picture descriptor, used to build the vpcC. */
    uint8_t frame_profile;
    uint8_t frame_bit_depth;
    uint8_t frame_subsampling;
    uint32_t frame_width;
    bool hw_av1;                   /* this host has AV1 silicon */

    /* Software fallback. Once the stream shows a frame the hardware decoder cannot be
     * trusted with, everything from the last full refresh onward is re-decoded here and
     * the codec stays on this path for the rest of its life. */
    struct virgl_video_dav1d *sw;
    struct unit_log *replay;       /* units since the last shown key frame */
    unsigned replay_n, replay_cap;
    size_t replay_bytes;
    bool replay_lost;              /* the cap was hit: a replay from here would be wrong */
    uint8_t *interleave;           /* scratch for NV12 targets, which dav1d cannot fill directly */
    size_t interleave_cap;

    /* The frame declares AV1 super-resolution. Read from the stream, not inferred from what
     * the host hands back, so the refusal below does not depend on the host's bug staying
     * the shape we measured it in. */
    bool frame_superres;
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

/* AV1 codecs currently alive. destroy_buffer is handed a buffer and nothing else, so
 * finding the codec that is holding a frame for it needs a way back. Only AV1 codecs are
 * registered, since only they hold frames. */
#define MAX_LIVE_CODECS 16
static struct virgl_video_codec *live_codecs[MAX_LIVE_CODECS];
static unsigned num_live_codecs;

static void codec_register(struct virgl_video_codec *codec)
{
    if (num_live_codecs < MAX_LIVE_CODECS)
        live_codecs[num_live_codecs++] = codec;
    else
        virgl_error("video: too many live codecs to track held frames\n");
}

static void codec_unregister(struct virgl_video_codec *codec)
{
    for (unsigned i = 0; i < num_live_codecs; i++)
        if (live_codecs[i] == codec) {
            live_codecs[i] = live_codecs[--num_live_codecs];
            return;
        }
}

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

    /*
     * H.264. Every Apple silicon Mac decodes it, but the advertisement is still gated on
     * the query rather than assumed -- a host that says no must not be told otherwise.
     *
     * Only the profiles this backend can actually serialize a parameter set for are
     * offered. High covers Baseline and Main streams too (a decoder for High decodes
     * both), and offering High10/422/444 would promise bit depths and chroma formats
     * virgl_h264_build_parameter_sets refuses.
     */
    if (vt_can_decode(kCMVideoCodecType_H264)) {
        static const enum pipe_video_profile h264_profiles[] = {
            PIPE_VIDEO_PROFILE_MPEG4_AVC_CONSTRAINED_BASELINE,
            PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE,
            PIPE_VIDEO_PROFILE_MPEG4_AVC_MAIN,
            PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH,
        };

        VT_TRACE("fill_caps: advertising H.264 decode\n");
        for (unsigned i = 0; i < ARRAY_SIZE(h264_profiles); i++) {
            vcaps = &caps->v2.video_caps[caps->v2.num_video_caps++];
            memset(vcaps, 0, sizeof(*vcaps));
            vcaps->profile = h264_profiles[i];
            vcaps->entrypoint = PIPE_VIDEO_ENTRYPOINT_BITSTREAM;
            vcaps->max_level = 52;      /* 5.2: above 4K; VideoToolbox parses the real
                                         * stream anyway, so this only sizes the guest. */
            vcaps->stacked_frames = 0;
            vcaps->max_width = VT_MAX_WIDTH;
            vcaps->max_height = VT_MAX_HEIGHT;
            vcaps->prefered_format = PIPE_FORMAT_NV12;
            vcaps->max_macroblocks = 1;
            vcaps->npot_texture = 1;
            vcaps->supports_progressive = 1;
            vcaps->supports_interlaced = 0;   /* field coding is refused by the serializer */
            vcaps->prefers_interlaced = 0;
            vcaps->max_temporal_layers = 0;
        }
    }

    /* AV1 needs M3-or-later silicon. dav1d is present as the fallback for frames the
     * hardware decoder returns wrongly -- super-resolution -- and not as a decoder in its
     * own right: on a host with no AV1 silicon at all, offering AV1 would take the whole
     * stream off the guest's own dav1d, which is better tested than ours, and onto a host
     * software path for no gain. Such a host advertises nothing and the guest decodes AV1
     * itself.
     *
     * A capture build forces the advertisement on anyway, because capture never decodes:
     * the picture descriptors are a pure function of the *guest's* own bitstream parse, so
     * a machine with no AV1 silicon produces exactly the fixtures an M3 would. */
    if (vt_can_decode(kCMVideoCodecType_AV1) || av1_capture_dir()) {
        VT_TRACE("fill_caps: advertising AV1 main decode%s\n",
                 vt_can_decode(kCMVideoCodecType_AV1) ? ""
                     : " (capture build, no silicon)");
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

    /*
     * HEVC Main. Only Main: Main 10 would promise a 10-bit delivery path this backend does
     * not have, and the serializer refuses anything but 8-bit 4:2:0 anyway.
     */
    if (vt_can_decode(kCMVideoCodecType_HEVC)) {
        VT_TRACE("fill_caps: advertising HEVC decode\n");
        vcaps = &caps->v2.video_caps[caps->v2.num_video_caps++];
        memset(vcaps, 0, sizeof(*vcaps));
        vcaps->profile = PIPE_VIDEO_PROFILE_HEVC_MAIN;
        vcaps->entrypoint = PIPE_VIDEO_ENTRYPOINT_BITSTREAM;
        vcaps->max_level = 153;     /* 5.1, matching the level the serializer declares. */
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
        args->profile != PIPE_VIDEO_PROFILE_AV1_MAIN &&
        !is_ps_codec(args->profile)) {
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

    if (args->profile == PIPE_VIDEO_PROFILE_AV1_MAIN) {
        codec->hw_av1 = vt_can_decode(kCMVideoCodecType_AV1);
        codec->av1_decode = codec->hw_av1 || virgl_dav1d_available();
        virgl_av1_obu_state_init(&codec->av1);
        codec_register(codec);
    }

    VT_TRACE("create_codec: profile %d %ux%u%s\n", args->profile, args->width, args->height,
             args->profile != PIPE_VIDEO_PROFILE_AV1_MAIN ? ""
                 : !codec->av1_decode ? " (no AV1 silicon; capture only)"
                 : !codec->hw_av1     ? " (no AV1 silicon; decoding in software)" : "");

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

static void replay_reset(struct virgl_video_codec *codec);

void virgl_video_destroy_codec(struct virgl_video_codec *codec)
{
    if (!codec)
        return;

    codec_unregister(codec);

    /* Drop a held frame rather than decoding it. Its picture could only be collected by a
     * guest that is tearing the codec down, and decoding here would do GL work on a
     * teardown path for a result nobody reads. */
    virgl_av1_drop_held(&codec->av1);
    virgl_av1_obu_state_fini(&codec->av1);

    destroy_session(codec);
    if (codec->pending)
        CFRelease(codec->pending);
    virgl_dav1d_close(codec->sw);
    replay_reset(codec);
    free(codec->replay);
    free(codec->interleave);
    free(codec->unit);
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
    /* A held frame's target can be destroyed before the frame is emitted, which would
     * leave a dangling pointer to write a picture into. The frame itself still has to be
     * decoded when its turn comes -- the decoder's reference list needs it -- so only the
     * destination is dropped. */
    for (unsigned i = 0; i < num_live_codecs; i++)
        if (live_codecs[i]->held_target == buffer) {
            VT_TRACE("av1: the held frame's target was destroyed; it will decode "
                     "without being delivered\n");
            live_codecs[i]->held_target = NULL;
        }

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
static void deliver_picture(struct virgl_video_codec *codec, CVPixelBufferRef pixbuf,
                            struct virgl_video_buffer *target)
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
    dmabuf.buf = target;
    dmabuf.width = (uint32_t)CVPixelBufferGetWidth(pixbuf);
    dmabuf.height = (uint32_t)CVPixelBufferGetHeight(pixbuf);
    dmabuf.flags = VIRGL_VIDEO_DMABUF_READ_ONLY;
    dmabuf.num_planes = (uint32_t)num_planes;

    /* CoreVideo's planar output is Y,Cb,Cr; YV12 wants Y,Cr,Cb. */
    const bool swap_chroma = target && target->format == PIPE_FORMAT_YV12;

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

/*
 * Build a format description from the parameter sets in frame_config. H.264 and HEVC differ
 * only in the entry point and the number of sets; nal_length_size 4 must match the framing
 * end_frame produces for both.
 */
static OSStatus build_ps_format(struct virgl_video_codec *codec,
                                CMVideoFormatDescriptionRef *out)
{
    const uint8_t *sets[3];
    size_t offset = 0;

    for (unsigned i = 0; i < codec->ps_count; i++) {
        sets[i] = codec->frame_config + offset;
        offset += codec->ps_len[i];
    }

    if (is_h265(codec->profile))
        return CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            kCFAllocatorDefault, codec->ps_count, sets, codec->ps_len, 4, NULL, out);

    return CMVideoFormatDescriptionCreateFromH264ParameterSets(
        kCFAllocatorDefault, codec->ps_count, sets, codec->ps_len, 4, out);
}

/* Create (or recreate) the decompression session. VP9 streams can change resolution
 * or bit depth at a keyframe, so the session is keyed on the shape of the frame we
 * are about to decode rather than built once from the codec's creation arguments. */
static int ensure_session(struct virgl_video_codec *codec,
                          struct virgl_video_buffer *target)
{
    CFDataRef config;
    CFMutableDictionaryRef atoms, extensions, pixel_attrs, empty;
    CFNumberRef pixel_format_num;
    VTDecompressionOutputCallbackRecord callback;
    const bool av1 = codec->profile == PIPE_VIDEO_PROFILE_AV1_MAIN;
    enum pipe_format target_format = target ? target->format : PIPE_FORMAT_NV12;
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
        codec->session_target_format == target_format &&
        codec->session_config_len == codec->frame_config_len &&
        !memcmp(codec->session_config, codec->frame_config, codec->frame_config_len))
        return 0;

    /*
     * H.264's parameter sets are not constant across a stream, and tearing the session down
     * when they change is not survivable. num_ref_idx_lX_active_minus1 reaches us as the
     * EFFECTIVE per-slice count, so a slice that overrides the PPS default changes the PPS
     * we synthesize by a byte or two mid-GOP; keying the session on those bytes rebuilt the
     * decompression session there and took the DPB with it. Every frame after the first
     * override then referenced an empty DPB -- decoding "succeeded" and the pixels were
     * wrong, which is far quieter than a rejected parameter set.
     *
     * So the config bytes drive the FORMAT DESCRIPTION only. If VideoToolbox will accept the
     * new description on the live session, swap it in and keep the session: end_frame hangs
     * the current description on each sample buffer, so the next frame decodes against it.
     * Falling through to the rebuild stays correct, just lossy, and says so.
     */
    if (is_ps_codec(codec->profile) && codec->session && codec->format &&
        codec->session_width == codec->frame_width &&
        codec->session_height == codec->frame_height &&
        codec->session_target_format == target_format &&
        codec->ps_count) {
        CMVideoFormatDescriptionRef fresh = NULL;

        status = build_ps_format(codec, &fresh);
        if (status == noErr && fresh) {
            if (VTDecompressionSessionCanAcceptFormatDescription(codec->session, fresh)) {
                CFRelease(codec->format);
                codec->format = fresh;
                memcpy(codec->session_config, codec->frame_config, codec->frame_config_len);
                codec->session_config_len = codec->frame_config_len;
                VT_TRACE("ps: parameter sets changed (%u sets, %zu bytes), session kept\n",
                         codec->ps_count, codec->frame_config_len);
                return 0;
            }
            CFRelease(fresh);
            virgl_error("video: parameter set change needs a new session;"
                        " the reference picture buffer is lost across it\n");
        }
    }

    destroy_session(codec);

    /*
     * H.264 does not go through a config atom at all: VideoToolbox takes the parameter set
     * NALs themselves and derives the format description (including the dimensions) from
     * them. nal_length_size 4 must match the framing end_frame produces.
     */
    if (is_ps_codec(codec->profile)) {
        if (!codec->ps_count) {
            virgl_error("video: session asked for before any parameter set\n");
            return -1;
        }

        status = build_ps_format(codec, &codec->format);
        if (status != noErr) {
            /* A rejection here is the loud half of a malformed parameter set. The quiet
             * half -- accepted but subtly wrong -- is what the serializer's spike guards
             * against; see spikes/h264-ps-synth. */
            virgl_error("video: building the %s format description failed, status %d\n",
                        is_h265(codec->profile) ? "HEVC" : "H.264", (int)status);
            return -1;
        }
        goto have_format;
    }

    /* AV1 carries a whole sequence header OBU in its av1C, so unlike VP9's six scalars it
     * is built by the serializer and simply handed over here. */
    if (av1)
        config = CFDataCreate(kCFAllocatorDefault, codec->frame_config,
                              (CFIndex)codec->frame_config_len);
    else
        config = build_vpcc(codec->frame_profile, codec->frame_bit_depth,
                            codec->frame_subsampling);
    if (!config)
        return -1;

    atoms = CFDictionaryCreateMutable(kCFAllocatorDefault, 1,
                                      &kCFTypeDictionaryKeyCallBacks,
                                      &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(atoms, av1 ? CFSTR("av1C") : CFSTR("vpcC"), config);
    CFRelease(config);

    extensions = CFDictionaryCreateMutable(kCFAllocatorDefault, 1,
                                           &kCFTypeDictionaryKeyCallBacks,
                                           &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(extensions,
                         kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
                         atoms);
    CFRelease(atoms);

    status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault,
                                            av1 ? kCMVideoCodecType_AV1
                                                : kCMVideoCodecType_VP9,
                                            codec->frame_width, codec->frame_height,
                                            extensions, &codec->format);
    CFRelease(extensions);
    if (status != noErr) {
        virgl_error("video: CMVideoFormatDescriptionCreate failed, status %d\n", (int)status);
        return -1;
    }

have_format:

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

    /*
     * VTIsHardwareDecodeSupported answers for the CODEC at capability time; it says nothing
     * about the session we just built, which VideoToolbox may still service in software.
     * Guest CPU proves only that the work left the guest, so ask the session itself -- this
     * is the one place the distinction is observable.
     */
    {
        CFBooleanRef hw = NULL;
        if (VTSessionCopyProperty(codec->session,
                kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
                kCFAllocatorDefault, &hw) == noErr && hw) {
            VT_TRACE("ensure_session: hardware accelerated: %s\n",
                     CFBooleanGetValue(hw) ? "yes" : "NO (VideoToolbox software path)");
            CFRelease(hw);
        } else {
            VT_TRACE("ensure_session: hardware acceleration state unavailable\n");
        }
    }

    codec->session_width = codec->frame_width;
    codec->session_height = codec->frame_height;
    codec->session_profile = codec->frame_profile;
    codec->session_bit_depth = codec->frame_bit_depth;
    codec->session_subsampling = codec->frame_subsampling;
    codec->session_target_format = target_format;
    memcpy(codec->session_config, codec->frame_config, codec->frame_config_len);
    codec->session_config_len = codec->frame_config_len;

    return 0;
}

int virgl_video_begin_frame(struct virgl_video_codec *codec,
                            struct virgl_video_buffer *target)
{
    if (!codec || !target)
        return -1;

    codec->bitstream_len = 0;
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
static int av1_flush_held(struct virgl_video_codec *codec,
                          const struct virgl_av1_picture_desc *desc);

static int av1_decode_bitstream(struct virgl_video_codec *codec,
                                const union virgl_picture_desc *desc,
                                unsigned num_buffers,
                                const void * const *buffers,
                                const unsigned *sizes)
{
    const char *dir = av1_capture_dir();
    const struct virgl_av1_picture_desc *av1 = &desc->av1;
    const __typeof__(av1->picture_parameter) *p = &av1->picture_parameter;
    size_t needed = codec->bitstream_len;
    ssize_t av1c;

    if (!codec->av1_decode && !dir) {
        virgl_error("video: this host cannot decode AV1 "
                    "(set LIMINA_AV1_CAPTURE to record fixtures instead)\n");
        return -1;
    }

    if (dir) {
        char path[1024];
        FILE *f;

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
    }

    if (!codec->av1_decode)
        return 0;

    /* First: emit whatever frame is being held. This descriptor is what makes its
     * reference slot exact, and the serializer ignores the repeat calls a frame whose tile
     * data arrives in several buffers produces. */
    if (av1_flush_held(codec, av1) < 0)
        return -1;

    /* The frame's own unit cannot be built yet -- more tile data may still arrive -- so
     * keep the descriptor for end_frame. Every call in a frame carries the same one. */
    codec->av1_desc = *av1;
    codec->av1_desc_valid = true;

    codec->frame_width = p->frame_width ? p->frame_width : codec->width;
    codec->frame_height = p->frame_height ? p->frame_height : codec->height;
    codec->frame_superres = p->pic_info_fields.use_superres;
    codec->frame_bit_depth = p->bit_depth_idx ? 10 : 8;
    codec->frame_profile = p->profile;
    codec->frame_subsampling = 1;

    av1c = virgl_av1_build_av1c(&codec->av1, av1, codec->frame_config,
                                sizeof(codec->frame_config));
    if (av1c < 0) {
        virgl_error("video: could not build the AV1 configuration record\n");
        return -1;
    }
    codec->frame_config_len = (size_t)av1c;

    for (unsigned i = 0; i < num_buffers; i++)
        needed += sizes[i];

    if (needed > codec->bitstream_cap) {
        uint8_t *grown = realloc(codec->bitstream, needed);
        if (!grown) {
            virgl_error("video: out of memory growing the tile buffer\n");
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

    return 0;
}

/*
 * H.264: accumulate the guest's slice NALs and synthesize the parameter sets they need.
 *
 * Unlike AV1, the bitstream itself is real -- mesa's VA frontend prepends an Annex-B start
 * code to every slice, so what arrives is the encoder's own slice layer. Only the SPS and
 * PPS are missing, and they can only be built once the slices are here: the id the slice
 * headers reference is not on the wire and has to be read back out of them.
 */
static int h264_decode_bitstream(struct virgl_video_codec *codec,
                                 const union virgl_picture_desc *desc,
                                 unsigned num_buffers,
                                 const void * const *buffers,
                                 const unsigned *sizes)
{
    struct virgl_h264_parameter_sets ps;
    size_t needed = codec->bitstream_len;
    unsigned pps_id;

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

    /* The session key. H.264 has no per-frame profile or bit depth to track the way VP9
     * does -- those live in the SPS, and a change there changes the config below. */
    codec->frame_profile = 0;
    codec->frame_bit_depth = 8;
    codec->frame_subsampling = 1;
    codec->frame_width = codec->width;
    codec->frame_height = codec->height;
    codec->frame_superres = false;

    if (virgl_h264_slice_pps_id(codec->bitstream, codec->bitstream_len, &pps_id)) {
        /* No slice NAL yet: this call carried only a fragment. Leave the config alone and
         * wait -- guessing an id would produce a PPS the slices never reference. */
        VT_TRACE("h264: no slice header yet, %zu bytes buffered\n", codec->bitstream_len);
        return 0;
    }

    if (virgl_h264_build_parameter_sets(&desc->h264, codec->width, codec->height,
                                        codec->profile, pps_id, &ps))
        return -1;

    if (ps.sps_len + ps.pps_len > sizeof(codec->frame_config)) {
        virgl_error("video: h264 parameter sets are %zu bytes, over the %zu-byte config\n",
                    ps.sps_len + ps.pps_len, sizeof(codec->frame_config));
        return -1;
    }

    memcpy(codec->frame_config, ps.sps, ps.sps_len);
    memcpy(codec->frame_config + ps.sps_len, ps.pps, ps.pps_len);
    codec->frame_config_len = ps.sps_len + ps.pps_len;
    codec->ps_len[0] = ps.sps_len;
    codec->ps_len[1] = ps.pps_len;
    codec->ps_count = 2;

    VT_TRACE("h264: %u buffers, %zu bytes, pps_id %u, sps %zu pps %zu, %ux%u\n",
             num_buffers, codec->bitstream_len, pps_id, ps.sps_len, ps.pps_len,
             codec->frame_width, codec->frame_height);
    return 0;
}

static int h265_decode_bitstream(struct virgl_video_codec *codec,
                                 const union virgl_picture_desc *desc,
                                 unsigned num_buffers,
                                 const void * const *buffers,
                                 const unsigned *sizes)
{
    struct virgl_h265_parameter_sets ps;
    size_t needed = codec->bitstream_len;
    size_t total;
    unsigned pps_id;
    int rc;

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

    /* Like H.264: the profile and bit depth live in the SPS, so a change there is a change
     * to the config below and needs no separate tracking. */
    codec->frame_profile = 0;
    codec->frame_bit_depth = 8;
    codec->frame_subsampling = 1;
    codec->frame_width = codec->width;
    codec->frame_height = codec->height;
    codec->frame_superres = false;

    /*
     * The inspection is not only for the pps id: it establishes that this stream does not
     * depend on SPS reference picture sets, which are absent from the wire and which we
     * therefore emit empty. A stream that does is refused here rather than decoded into
     * quietly wrong pixels.
     */
    rc = virgl_h265_slice_inspect(codec->bitstream, codec->bitstream_len, &desc->h265,
                                  &pps_id);
    if (rc < 0)
        return -1;
    if (rc > 0) {
        VT_TRACE("h265: no slice header yet, %zu bytes buffered\n", codec->bitstream_len);
        return 0;
    }

    if (virgl_h265_build_parameter_sets(&desc->h265, codec->width, codec->height,
                                        codec->profile, &ps))
        return -1;

    total = ps.vps_len + ps.sps_len + ps.pps_len;
    if (total > sizeof(codec->frame_config)) {
        virgl_error("video: h265 parameter sets are %zu bytes, over the %zu-byte config\n",
                    total, sizeof(codec->frame_config));
        return -1;
    }

    memcpy(codec->frame_config, ps.vps, ps.vps_len);
    memcpy(codec->frame_config + ps.vps_len, ps.sps, ps.sps_len);
    memcpy(codec->frame_config + ps.vps_len + ps.sps_len, ps.pps, ps.pps_len);
    codec->frame_config_len = total;
    codec->ps_len[0] = ps.vps_len;
    codec->ps_len[1] = ps.sps_len;
    codec->ps_len[2] = ps.pps_len;
    codec->ps_count = 3;

    VT_TRACE("h265: %u buffers, %zu bytes, pps_id %u, vps %zu sps %zu pps %zu, %ux%u\n",
             num_buffers, codec->bitstream_len, pps_id, ps.vps_len, ps.sps_len, ps.pps_len,
             codec->frame_width, codec->frame_height);
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

    if (is_h264(codec->profile))
        return h264_decode_bitstream(codec, desc, num_buffers, buffers, sizes);

    if (is_h265(codec->profile))
        return h265_decode_bitstream(codec, desc, num_buffers, buffers, sizes);

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
    codec->frame_superres = false;   /* VP9's scaling is per-reference, not a frame upscale */

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

/*
 * Decode one temporal unit and put its picture in `target`.
 *
 * One unit per sample, always. A sample carrying two would come back as one picture (or
 * two callbacks, the second releasing the first), and the frame that lost its picture is
 * silently never delivered -- the guest reads whatever its surface held before.
 *
 * `target` may be NULL: the unit is still decoded, because the decoder's own reference
 * list needs it, but nothing collects the picture.
 */
static int submit_unit(struct virgl_video_codec *codec, const uint8_t *data, size_t len,
                       struct virgl_video_buffer *target)
{
    CMBlockBufferRef block = NULL;
    CMSampleBufferRef sample = NULL;
    VTDecodeInfoFlags info = 0;
    OSStatus status;
    int err = 0;

    if (!len)
        return 0;

    if (ensure_session(codec, target) < 0)
        return -1;

    /* kCFAllocatorNull: the sample borrows our buffer rather than copying it. Safe because
     * the decode is synchronous and the buffer outlives the call. */
    status = CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, (void *)(uintptr_t)data,
                                                len, kCFAllocatorNull,
                                                NULL, 0, len, 0, &block);
    if (status != noErr) {
        virgl_error("video: CMBlockBufferCreateWithMemoryBlock failed, status %d\n",
                    (int)status);
        return -1;
    }

    status = CMSampleBufferCreate(kCFAllocatorDefault, block, TRUE, NULL, NULL,
                                  codec->format, 1, 0, NULL, 1, &len, &sample);
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

    /* Back on the vrend thread, with the GL context current: now the picture can be
     * copied into the guest's resources. */
    if (codec->pending) {
        /* What this host returns for an AV1 super-resolution frame is not that frame. It
         * comes back at the CODED width holding, near enough, the rightmost coded_width
         * columns of the correctly upscaled picture -- not the pre-upscale picture, which
         * could have been upscaled here (measured: 6.9% of pixels match the pre-upscale
         * reading, 76% match the right-hand crop). Asking for output buffers at the
         * sequence's own size only stretches the same wrong pixels. Half the picture is
         * simply absent from the buffer, so there is nothing to reconstruct it from, and
         * delivering it would put visibly wrong content on screen with nothing anywhere
         * reporting a problem. Refuse, and let the guest fall back to a software decoder.
         * See docs/hardening-backlog.md.
         *
         * Keyed on the stream's own `use_superres`, not on the width that came back: a host
         * that returns a full-width buffer holding the same wrong pixels must still be
         * refused, and the width is checked separately below as its own sanity error.
         *
         * Note what this does NOT do: the frame is still submitted and decoded. Only its
         * DELIVERY is refused. This host's reconstruction is internally correct -- the
         * frames that predict from super-resolution references come back bit-exact -- so
         * skipping the submission would silently corrupt every later frame. */
        size_t got = CVPixelBufferGetWidth(codec->pending);

        /* Only a delivered frame can put wrong pixels on screen; a held frame is decoded
         * purely to keep the reference state right and is never copied out. */
        if (target && codec->frame_superres) {
            virgl_error("video: this host does not return AV1 super-resolution frames "
                        "correctly (frame declares %u wide, got %zu); refusing the frame "
                        "rather than delivering wrong pixels\n",
                        codec->frame_width, got);
            err = -1;
            target = NULL;
        } else if (target && got != codec->frame_width) {
            virgl_error("video: the host returned a %zu-wide picture for a frame that "
                        "declares %u; refusing it\n", got, codec->frame_width);
            err = -1;
            target = NULL;
        }
        if (target)
            deliver_picture(codec, codec->pending, target);
        CFRelease(codec->pending);
        codec->pending = NULL;
    } else if (!err) {
        virgl_error("video: decode produced no picture\n");
        err = -1;
    }

    return err;
}

/* --- software fallback ------------------------------------------------------------ */

#define REPLAY_MAX_BYTES (64u << 20)

static void replay_reset(struct virgl_video_codec *codec)
{
    for (unsigned i = 0; i < codec->replay_n; i++)
        free(codec->replay[i].data);
    codec->replay_n = 0;
    codec->replay_bytes = 0;
    codec->replay_lost = false;
}

/* Keep a copy of a unit so a later switch can rebuild the reference state. Dropping one
 * is not fatal on its own, but it makes any replay from here wrong, so it is recorded:
 * a decoder brought up on a gapped history produces headers that parse and pictures that
 * are subtly wrong, which is the failure mode this whole path exists to avoid. */
static void replay_append(struct virgl_video_codec *codec, const uint8_t *data, size_t len)
{
    struct unit_log *grown;
    uint8_t *copy;

    if (codec->replay_lost)
        return;

    if (codec->replay_bytes + len > REPLAY_MAX_BYTES) {
        VT_TRACE("replay: over %u bytes, dropping the history\n", REPLAY_MAX_BYTES);
        replay_reset(codec);
        codec->replay_lost = true;
        return;
    }

    if (codec->replay_n == codec->replay_cap) {
        unsigned want = codec->replay_cap ? codec->replay_cap * 2 : 64;

        grown = realloc(codec->replay, want * sizeof(*grown));
        if (!grown) {
            replay_reset(codec);
            codec->replay_lost = true;
            return;
        }
        codec->replay = grown;
        codec->replay_cap = want;
    }

    copy = malloc(len);
    if (!copy) {
        replay_reset(codec);
        codec->replay_lost = true;
        return;
    }
    memcpy(copy, data, len);
    codec->replay[codec->replay_n].data = copy;
    codec->replay[codec->replay_n].len = len;
    codec->replay_n++;
    codec->replay_bytes += len;
}

/* Hand a dav1d picture to the guest. The hardware path's delivery is reused wholesale
 * where the target is planar -- which is what mesa asks for on decode targets -- and the
 * planes are interleaved into scratch first for a biplanar (NV12) target, which dav1d
 * cannot produce. */
static void deliver_dav1d_picture(struct virgl_video_codec *codec,
                                  const struct virgl_dav1d_picture *pic,
                                  struct virgl_video_buffer *target)
{
    struct virgl_video_dma_buf dmabuf;
    const bool nv12 = target && target->format == PIPE_FORMAT_NV12;
    const bool swap_chroma = target && target->format == PIPE_FORMAT_YV12;

    if (!video_cbs || !video_cbs->decode_completed)
        return;

    if (pic->bpc != 8) {
        virgl_error("video: the software decoder only delivers 8-bit pictures (got %u)\n",
                    pic->bpc);
        return;
    }

    memset(&dmabuf, 0, sizeof(dmabuf));
    dmabuf.buf = target;
    dmabuf.width = pic->width;
    dmabuf.height = pic->height;
    dmabuf.flags = VIRGL_VIDEO_DMABUF_READ_ONLY;

    if (nv12) {
        const uint32_t cw = (pic->width + 1) / 2, ch = (pic->height + 1) / 2;
        const size_t need = (size_t)cw * 2 * ch;

        if (need > codec->interleave_cap) {
            uint8_t *grown = realloc(codec->interleave, need);

            if (!grown) {
                virgl_error("video: out of memory interleaving chroma\n");
                return;
            }
            codec->interleave = grown;
            codec->interleave_cap = need;
        }
        for (uint32_t y = 0; y < ch; y++) {
            const uint8_t *u = pic->plane[1] + (size_t)y * pic->pitch[1];
            const uint8_t *v = pic->plane[2] + (size_t)y * pic->pitch[2];
            uint8_t *dst = codec->interleave + (size_t)y * cw * 2;

            for (uint32_t x = 0; x < cw; x++) {
                dst[2 * x] = u[x];
                dst[2 * x + 1] = v[x];
            }
        }

        dmabuf.num_planes = 2;
        dmabuf.planes[0].fd = -1;
        dmabuf.planes[0].map = (void *)(uintptr_t)pic->plane[0];
        dmabuf.planes[0].pitch = pic->pitch[0];
        dmabuf.planes[0].size = (size_t)pic->pitch[0] * pic->height;
        dmabuf.planes[1].fd = -1;
        dmabuf.planes[1].map = codec->interleave;
        dmabuf.planes[1].pitch = cw * 2;
        dmabuf.planes[1].size = need;
    } else {
        dmabuf.num_planes = 3;
        for (unsigned i = 0; i < 3; i++) {
            unsigned src = i;

            /* YV12 is I420 with the chroma planes the other way round. */
            if (swap_chroma && i > 0)
                src = 3 - i;

            dmabuf.planes[i].fd = -1;
            dmabuf.planes[i].map = (void *)(uintptr_t)pic->plane[src];
            dmabuf.planes[i].pitch = pic->pitch[src];
            dmabuf.planes[i].size = (size_t)pic->pitch[src] *
                                    (src ? (pic->height + 1) / 2 : pic->height);
        }
    }

    VT_TRACE("deliver (sw): %u planes, %ux%u, pitch0 %u\n", dmabuf.num_planes,
             dmabuf.width, dmabuf.height, dmabuf.planes[0].pitch);
    video_cbs->decode_completed(codec, &dmabuf);
}

static int sw_submit_unit(struct virgl_video_codec *codec, const uint8_t *data, size_t len,
                          struct virgl_video_buffer *target)
{
    struct virgl_dav1d_picture pic;
    int r = virgl_dav1d_decode(codec->sw, data, len, &pic);

    if (r < 0)
        return -1;
    if (r == 0)
        return 0;               /* decoded, nothing to show for it */
    if (target)
        deliver_dav1d_picture(codec, &pic, target);
    virgl_dav1d_release(codec->sw);
    return 0;
}

/* Bring the software decoder up to the reference state the hardware one already reached,
 * by re-decoding every unit since the last full refresh. The unit that triggered the
 * switch has already been logged and is decoded by the caller, so it is left out here.
 *
 * Failure is not fatal: the codec simply stays on the hardware path, where the frame it
 * cannot deliver correctly is refused rather than shown wrong. */
static bool sw_begin(struct virgl_video_codec *codec, const char *why)
{
    if (!virgl_dav1d_available()) {
        virgl_error("video: no software AV1 decoder in this build (%s)\n", why);
        return false;
    }
    if (codec->frame_bit_depth != 8) {
        virgl_error("video: the software AV1 fallback is 8-bit only (stream is %u-bit, "
                    "%s)\n", codec->frame_bit_depth, why);
        return false;
    }
    if (codec->replay_lost || !codec->replay_n) {
        virgl_error("video: no usable frame history to start the software AV1 decoder "
                    "from (%s)\n", why);
        return false;
    }

    codec->sw = virgl_dav1d_open();
    if (!codec->sw)
        return false;

    for (unsigned i = 0; i + 1 < codec->replay_n; i++) {
        struct virgl_dav1d_picture pic;

        if (virgl_dav1d_decode(codec->sw, codec->replay[i].data, codec->replay[i].len,
                               &pic) < 0) {
            virgl_error("video: could not replay unit %u of %u into the software decoder; "
                        "staying on the hardware path\n", i, codec->replay_n);
            virgl_dav1d_close(codec->sw);
            codec->sw = NULL;
            return false;
        }
        /* These frames were already delivered by the hardware path; they are re-decoded
         * only so the references they leave behind are right. */
        virgl_dav1d_release(codec->sw);
    }

    virgl_error("video: decoding this AV1 stream in software (%s); replayed %u unit(s)\n",
                why, codec->replay_n - 1);
    return true;
}

/* Every AV1 unit goes through here, so the replay history and the hardware/software
 * choice are decided in exactly one place. */
/* Append every reconstructed temporal unit to one file, in submission order, so the
 * rebuilt stream can be diffed field by field against the clip it came from
 * (spikes/av1-obu-serializer/fhparse.py). A debugging aid, off unless asked for. */
static void av1_dump_unit(const uint8_t *data, size_t len)
{
    static const char *path;
    static bool looked;
    FILE *f;

    if (!looked) {
        path = getenv("LIMINA_AV1_DUMP");
        if (path && !*path)
            path = NULL;
        looked = true;
    }
    if (!path || !data || !len)
        return;
    if ((f = fopen(path, "ab"))) {
        fwrite(data, len, 1, f);
        fclose(f);
    }
}

static int av1_route_unit(struct virgl_video_codec *codec, const uint8_t *data, size_t len,
                          struct virgl_video_buffer *target, bool starts_dpb)
{
    av1_dump_unit(data, len);

    if (starts_dpb)
        replay_reset(codec);
    replay_append(codec, data, len);

    /* Two reasons to leave the hardware decoder. Neither is recoverable, so the codec
     * stays in software once it switches: a host with no AV1 silicon never gains any, and
     * a stream that used super-resolution once will use it again. */
    if (!codec->sw) {
        if (!codec->hw_av1)
            sw_begin(codec, "this host has no AV1 silicon");
        else if (codec->frame_superres)
            sw_begin(codec, "this host does not return super-resolution frames correctly");
    }

    if (codec->sw)
        return sw_submit_unit(codec, data, len, target);

    return submit_unit(codec, data, len, target);
}

/* Grow the temporal-unit buffer to at least `want` bytes. */
static bool ensure_unit(struct virgl_video_codec *codec, size_t want)
{
    uint8_t *grown;

    if (want <= codec->unit_cap)
        return true;
    grown = realloc(codec->unit, want);
    if (!grown) {
        virgl_error("video: out of memory growing the temporal unit buffer\n");
        return false;
    }
    codec->unit = grown;
    codec->unit_cap = want;
    return true;
}

/*
 * Emit the frame the serializer is holding, if any. Called as soon as a descriptor is in
 * hand, because that descriptor is what settles the held frame's reference slot -- and
 * because a stream may display a hidden frame just one decode later, which leaves no
 * margin: nothing waits on delivery, so a picture that arrives late is read as a stale
 * surface rather than waited for.
 */
static int av1_flush_held(struct virgl_video_codec *codec,
                          const struct virgl_av1_picture_desc *desc)
{
    struct virgl_video_buffer *target = codec->held_target;
    ssize_t n;

    if (!ensure_unit(codec, virgl_av1_held_bound(&codec->av1) + VIRGL_AV1_UNIT_OVERHEAD))
        return -1;

    n = virgl_av1_flush_held(&codec->av1, desc, codec->unit, codec->unit_cap);
    if (n < 0) {
        virgl_error("video: could not serialize the held AV1 frame\n");
        return -1;
    }
    if (!n)
        return 0;

    codec->held_target = NULL;
    VT_TRACE("av1: flushing the held frame, %zd bytes\n", n);
    /* Never a full refresh: a key frame resets the model and is never held. */
    return av1_route_unit(codec, codec->unit, (size_t)n, target, false);
}

int virgl_video_end_frame(struct virgl_video_codec *codec,
                          struct virgl_video_buffer *target)
{
    if (!codec || !target)
        return -1;

    if (codec->profile == PIPE_VIDEO_PROFILE_AV1_MAIN) {
        ssize_t n;

        if (!codec->av1_decode || !codec->av1_desc_valid) {
            codec->bitstream_len = 0;
            return 0;   /* capture-only: nothing was accumulated to decode */
        }

        /* A shown key frame refreshes every reference slot, so the replay history can
         * start again here -- and must, or it grows for the life of the stream. Read
         * before the build, which clears the descriptor. */
        const bool starts_dpb =
            codec->av1_desc.picture_parameter.pic_info_fields.frame_type == 0 &&
            codec->av1_desc.picture_parameter.pic_info_fields.show_frame;

        if (!ensure_unit(codec, codec->bitstream_len + VIRGL_AV1_UNIT_OVERHEAD))
            return -1;

        n = virgl_av1_build_temporal_unit(&codec->av1, &codec->av1_desc,
                                          codec->bitstream, codec->bitstream_len,
                                          codec->unit, codec->unit_cap);
        codec->bitstream_len = 0;
        codec->av1_desc_valid = false;
        if (n < 0) {
            virgl_error("video: could not serialize the AV1 frame\n");
            return -1;
        }

        /* Nothing emitted means the serializer is holding this frame until the next
         * descriptor says which slot the guest stored it in. Its target has to be kept:
         * by the time it is emitted, the guest is several frames further on. */
        if (!n) {
            codec->held_target = target;
            VT_TRACE("av1: holding this frame\n");
            return 0;
        }

        VT_TRACE("av1: %zd bytes\n", n);
        return av1_route_unit(codec, codec->unit, (size_t)n, target, starts_dpb);
    }

    VT_TRACE("end_frame: %zu bytes\n", codec->bitstream_len);

    if (!codec->bitstream_len)
        return 0;

    /*
     * H.264 and HEVC arrive Annex-B (mesa prepends a start code per slice) and VideoToolbox
     * accepts
     * only length-prefixed NALs, so the access unit is re-framed here. This rewrites the
     * framing and nothing else -- emulation-prevention bytes inside each NAL stay as the
     * encoder wrote them.
     */
    if (is_ps_codec(codec->profile)) {
        const size_t len = codec->bitstream_len;
        ssize_t need, got;

        codec->bitstream_len = 0;

        if (!codec->ps_count) {
            virgl_error("video: access unit with no parameter set; dropping\n");
            return -1;
        }

        need = virgl_h264_annexb_to_avcc(codec->bitstream, len, NULL, 0);
        if (need < 0) {
            virgl_error("video: access unit is not Annex-B framed\n");
            return -1;
        }
        if (!ensure_unit(codec, (size_t)need))
            return -1;

        got = virgl_h264_annexb_to_avcc(codec->bitstream, len, codec->unit, codec->unit_cap);
        if (got < 0) {
            virgl_error("video: re-framing overflowed a %zu-byte unit\n", codec->unit_cap);
            return -1;
        }

        VT_TRACE("ps: %zu annexb -> %zd avcc bytes\n", len, got);
        return submit_unit(codec, codec->unit, (size_t)got, target);
    }

    {
        const size_t len = codec->bitstream_len;

        codec->bitstream_len = 0;
        return submit_unit(codec, codec->bitstream, len, target);
    }
}
