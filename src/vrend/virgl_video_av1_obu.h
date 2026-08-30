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

/*
 * Rebuild AV1 OBUs from a parsed picture descriptor.
 *
 * A decode backend that takes whole compressed frames (VideoToolbox, and any other
 * bitstream-in decoder) cannot use the VA-API-shaped picture the virgl protocol carries:
 * the frame header the encoder wrote was consumed and discarded by the guest's own parser
 * long before the descriptor was built. This turns the descriptor back into a conformant
 * temporal unit.
 *
 * All state needed across frames lives in struct virgl_av1_obu_state, which the caller
 * owns per codec. It is *not* a decoded-picture buffer: the decoder keeps its own. It holds
 * only what the frame-header *syntax* is coded relative to, which is the saved global-motion
 * warp parameters -- everything else a frame inherits is written absolutely behind an
 * inherit flag, and can simply be re-emitted from the descriptor.
 */

#ifndef VIRGL_VIDEO_AV1_OBU_H
#define VIRGL_VIDEO_AV1_OBU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct virgl_av1_picture_desc;

#define VIRGL_AV1_NUM_REF_FRAMES     8
#define VIRGL_AV1_REFS_PER_FRAME     7
#define VIRGL_AV1_WARP_PARAMS        6

struct virgl_av1_obu_state {
   /* Saved warp parameters per reference slot. global_motion_param() codes each value as a
    * subexp delta against the primary reference's saved params, so a writer holding only
    * reconstructed values (which is all the descriptor carries) needs the base to compute
    * the delta. */
   int32_t saved_gm[VIRGL_AV1_NUM_REF_FRAMES][VIRGL_AV1_REFS_PER_FRAME][VIRGL_AV1_WARP_PARAMS];

   /* Order hint per reference slot. Not needed to *write* a value, but to decide whether a
    * bit is present at all: skip_mode_params() searches the references for a forward and a
    * backward one, and only writes skip_mode_present when that search succeeds. Getting this
    * wrong desynchronises the bit position of everything after it, so it is not optional. */
   uint8_t saved_order_hint[VIRGL_AV1_NUM_REF_FRAMES];

   /* Slots written at least once; before that a reference reads as order hint 0 and default
    * warp parameters, which is what a decoder starting from a key frame also assumes. */
   uint8_t slot_valid;

   /* Set once the first sequence header has been derived, so callers can build an av1C
    * record before the first frame is submitted. */
   bool seq_valid;
};

/* Reset to the state implied by "no frames decoded yet". */
void virgl_av1_obu_state_init(struct virgl_av1_obu_state *state);

/*
 * Build one temporal unit -- temporal delimiter, sequence header, frame header and tile
 * group -- for the frame the descriptor describes.
 *
 * Each frame gets its own temporal delimiter deliberately: a stream's natural framing
 * bundles a no-show frame with the frame that displays it into one temporal unit, and a
 * decoder then returns a single picture for the pair, while the virgl protocol submits one
 * *frame* at a time and expects one picture back. Repeating the sequence header in every
 * unit is legal and avoids tracking when a new one is owed.
 *
 * `tiles`/`tiles_size` is the tile payload exactly as the guest sent it. Returns the number
 * of bytes written to `out`, or a negative value on error. Call with out == NULL to size.
 */
ssize_t virgl_av1_build_temporal_unit(struct virgl_av1_obu_state *state,
                                      const struct virgl_av1_picture_desc *desc,
                                      const void *tiles, size_t tiles_size,
                                      uint8_t *out, size_t out_size);

/*
 * Build the av1C configuration record a container-shaped decoder wants: a four-byte
 * configuration record followed by the sequence header OBU. Unlike VP9's vpcC, which is six
 * scalars, this embeds real bitstream.
 */
ssize_t virgl_av1_build_av1c(struct virgl_av1_obu_state *state,
                             const struct virgl_av1_picture_desc *desc,
                             uint8_t *out, size_t out_size);

#endif /* VIRGL_VIDEO_AV1_OBU_H */
