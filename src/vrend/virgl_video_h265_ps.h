/*
 * Synthesize the HEVC VPS, SPS and PPS that VideoToolbox needs, from the picture
 * parameters the guest puts on the VA-API wire.
 *
 * The wire carries a hardware decoder's view of a stream, not a bitstream writer's, so
 * three things have to be invented rather than copied: the whole VPS, the tier/level, and
 * the short_term_ref_pic_set structures. The first two are free -- nothing consults a VPS
 * beyond its profile_tier_level, and the level is ours to over-declare. The sets are not:
 * they are absent from VA-API by design, and a slice that indexes one cannot be served.
 * virgl_h265_slice_inspect exists to detect exactly that case and refuse it loudly, which
 * is the condition under which the placeholder sets this writes are sound.
 *
 * See spikes/hevc-vt-probe for the measurements, and docs/design/h264-hevc-decode.md.
 */
#ifndef VIRGL_VIDEO_H265_PS_H
#define VIRGL_VIDEO_H265_PS_H

#include <stddef.h>
#include <stdint.h>

#include "pipe/p_video_enums.h"
#include "virgl_video_hw.h"

#define VIRGL_H265_PS_MAX 512

struct virgl_h265_parameter_sets {
   uint8_t vps[VIRGL_H265_PS_MAX];
   size_t vps_len;
   uint8_t sps[VIRGL_H265_PS_MAX];
   size_t sps_len;
   uint8_t pps[VIRGL_H265_PS_MAX];
   size_t pps_len;
};

/*
 * Build all three sets as raw NAL payloads (no start codes, emulation prevention applied).
 * width/height are the DISPLAY size: the coded size is on the wire, and the difference
 * becomes the conformance window. Returns 0, or -1 having said why.
 */
int virgl_h265_build_parameter_sets(const struct virgl_h265_picture_desc *desc,
                                    uint32_t width, uint32_t height,
                                    enum pipe_video_profile profile,
                                    struct virgl_h265_parameter_sets *out);

/*
 * Parse the first independent slice segment header of an Annex-B submission far enough to
 * learn its pic_parameter_set_id, and to establish that the stream does not depend on
 * reference picture sets we cannot reproduce. Returns 0 with *out_id set, 1 if no slice
 * header is present yet, or -1 for a stream we must not decode (having logged the reason).
 */
int virgl_h265_slice_inspect(const uint8_t *annexb, size_t len,
                             const struct virgl_h265_picture_desc *desc,
                             unsigned *out_id);

#endif /* VIRGL_VIDEO_H265_PS_H */
