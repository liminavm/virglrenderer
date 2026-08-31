/**************************************************************************
 *
 * Copyright (C) 2026 Gustavo Noronha Silva
 *
 * SPDX-License-Identifier: MIT
 *
 **************************************************************************/

/*
 * H.264 parameter-set synthesis for the VideoToolbox backend.
 *
 * VideoToolbox builds its format description from real SPS/PPS *bytes*
 * (CMVideoFormatDescriptionCreateFromH264ParameterSets). The guest sends the parsed
 * semantic content instead -- struct virgl_h264_pps, which is the gallium
 * hardware-decoder shape -- so the bytes have to be written here.
 *
 * The slice layer needs no such treatment: mesa's VA frontend prepends an Annex-B start
 * code to every slice NAL (src/gallium/frontends/va/decode.c), so the guest's slice
 * bitstream arrives intact. Only the parameter sets are missing, and only they are built.
 *
 * Two things the wire does not carry, and where they come from:
 *
 *   - Geometry. There is no pic_width_in_mbs on the wire (HEVC, by contrast, does carry
 *     pic_width_in_luma_samples). It comes from the codec object, rounded up to whole
 *     macroblocks, with frame cropping making up the difference -- without the cropping a
 *     stream whose width is not a multiple of 16 decodes at the padded size.
 *
 *   - Identifiers. Nothing on the wire says which pic_parameter_set_id the guest's slice
 *     headers reference, and a PPS bearing the wrong one is simply not found. It is parsed
 *     back out of the first slice (virgl_h264_slice_pps_id): it is the third ue(v) in the
 *     slice header. The PPS->SPS link is ours to choose because nothing else refers to it.
 */

#ifndef VIRGL_VIDEO_H264_PS_H
#define VIRGL_VIDEO_H264_PS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "pipe/p_video_enums.h"
#include "virgl_video_hw.h"

#ifdef __cplusplus
extern "C" {
#endif

/* An SPS carrying every scaling list is well under 512 bytes; a PPS likewise. */
#define VIRGL_H264_PS_MAX 512

struct virgl_h264_parameter_sets {
   uint8_t sps[VIRGL_H264_PS_MAX];
   size_t sps_len;
   uint8_t pps[VIRGL_H264_PS_MAX];
   size_t pps_len;
};

/*
 * Write the SPS and PPS for `desc`, as complete NAL units (header byte + RBSP with
 * emulation-prevention bytes), ready to hand to
 * CMVideoFormatDescriptionCreateFromH264ParameterSets.
 *
 * `width`/`height` are the display dimensions from the codec object. `profile` selects
 * profile_idc. `pps_id` must be the id the stream's own slice headers use --
 * see virgl_h264_slice_pps_id().
 *
 * Returns 0, or -1 if the descriptor asks for something this cut does not serialize
 * (interlaced coding, or a chroma format other than 4:2:0).
 */
int virgl_h264_build_parameter_sets(const struct virgl_h264_picture_desc *desc,
                                    uint32_t width, uint32_t height,
                                    enum pipe_video_profile profile,
                                    unsigned pps_id,
                                    struct virgl_h264_parameter_sets *out);

/*
 * Read pic_parameter_set_id out of the first slice NAL in an Annex-B buffer.
 *
 * Returns 0 and sets *out_id, or -1 when no slice NAL is found (in which case the caller
 * has no basis for choosing an id and must not guess -- a PPS with an id the slices do not
 * reference makes VideoToolbox reject the picture, which at least fails loudly).
 */
int virgl_h264_slice_pps_id(const uint8_t *annexb, size_t len, unsigned *out_id);

/*
 * Rewrite Annex-B start codes as 4-byte big-endian NAL lengths (AVCC), which is the only
 * framing VideoToolbox accepts. Emulation-prevention bytes inside each NAL are left
 * alone: this re-frames NALs, it does not rewrite them.
 *
 * With `out` NULL, returns the size the conversion needs. Otherwise returns the bytes
 * written, or -1 if `out_cap` is too small.
 */
ssize_t virgl_h264_annexb_to_avcc(const uint8_t *in, size_t in_len,
                                  uint8_t *out, size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* VIRGL_VIDEO_H264_PS_H */
