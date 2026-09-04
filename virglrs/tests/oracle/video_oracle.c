/* SPDX-License-Identifier: MIT
 * Copyright © 2026 the limina authors
 *
 * The C side of the video serializers' differential test.
 *
 * The bit writer and reader are `static inline` in a header, so there is nothing to link
 * against and nothing to diff without a shim. These wrappers are that shim: one entry point per
 * thing the Rust port has to reproduce, driven from the Rust tests with the same inputs.
 *
 * The oracle is the C in this tree, which is the only reference these serializers have -- there
 * is no second implementation to check against and no conformance vector for a synthesized
 * parameter set. That is exactly why the differential is worth having: a port that agrees with
 * it byte for byte inherits whatever correctness the C earned in the streams it has played.
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

#include "virgl_video_bitstream.h"

/*
 * The serializers say why they refused a stream through this. The tests read the refusal, not the
 * reason -- both sides must reject the same inputs, and only one of them writes English -- so it
 * goes nowhere.
 */
void virgl_error(const char *fmt, ...) { (void)fmt; }

/* A script for the bit writer, so one call can exercise an arbitrary sequence: the Rust side
 * sends the same script to its own writer and the two buffers are compared. Each op is a tag
 * and a payload, which keeps the wire between the two sides trivially checkable. */
enum { OP_RAW = 0, OP_U = 1, OP_FLAG = 2, OP_UE = 3, OP_SE = 4, OP_TRAILING = 5 };

struct oracle_op {
   uint32_t tag;
   uint32_t n;      /* bit count for OP_U */
   int64_t value;   /* signed for OP_SE, unsigned otherwise */
};

/*
 * Run a script through the bit writer. Returns the bytes written, or -1 on overflow -- which
 * the Rust writer cannot report, because it grows instead of overflowing, so a script that
 * overflows here is one the test must not generate.
 */
ssize_t virgl_oracle_bs_run(const struct oracle_op *ops, size_t n_ops, int escape,
                            uint8_t *out, size_t out_cap)
{
   struct bs w;
   bs_init(&w, out, out_cap, escape != 0);

   for (size_t i = 0; i < n_ops; i++) {
      switch (ops[i].tag) {
      case OP_RAW:      bs_raw_byte(&w, (uint8_t)ops[i].value); break;
      case OP_U:        u(&w, ops[i].n, (uint32_t)ops[i].value); break;
      case OP_FLAG:     flag(&w, ops[i].value != 0); break;
      case OP_UE:       ue(&w, (uint32_t)ops[i].value); break;
      case OP_SE:       se(&w, (int32_t)ops[i].value); break;
      case OP_TRAILING: rbsp_trailing(&w); break;
      default:          return -1;
      }
   }
   if (w.overflow)
      return -1;
   return (ssize_t)w.pos;
}

/* Read `n_ops` values back out of an RBSP. `tags` says what to read; the values land in `out`.
 * Returns the number read, stopping at the first truncation -- which is itself the answer the
 * Rust reader has to agree on. */
size_t virgl_oracle_br_run(const uint8_t *buf, size_t len, const uint32_t *tags,
                           const uint32_t *widths, size_t n_ops, int64_t *out)
{
   struct br r = { .buf = buf, .len = len };

   for (size_t i = 0; i < n_ops; i++) {
      switch (tags[i]) {
      case OP_U: {
         uint32_t v;
         if (br_u(&r, widths[i], &v)) return i;
         out[i] = v;
         break;
      }
      case OP_UE: {
         uint32_t v;
         if (br_ue(&r, &v)) return i;
         out[i] = v;
         break;
      }
      case OP_SE: {
         int32_t v;
         if (br_se(&r, &v)) return i;
         out[i] = v;
         break;
      }
      default:
         return i;
      }
   }
   return n_ops;
}

/* ---------------------------------------------------------------- H.264 parameter sets */

#include "virgl_video_h264_ps.h"

/*
 * Where each field the Rust reader looks for actually sits. The order is the one the Rust `at`
 * module declares them in; the test compares the whole array, so a field inserted on one side and
 * not the other shifts everything after it and is caught rather than skipped.
 */
size_t virgl_oracle_h264_offsets(size_t *out, size_t cap)
{
   const size_t offsets[] = {
      offsetof(struct virgl_h264_picture_desc, pps.sps.level_idc),
      offsetof(struct virgl_h264_picture_desc, pps.sps.chroma_format_idc),
      offsetof(struct virgl_h264_picture_desc, pps.sps.separate_colour_plane_flag),
      offsetof(struct virgl_h264_picture_desc, pps.sps.bit_depth_luma_minus8),
      offsetof(struct virgl_h264_picture_desc, pps.sps.bit_depth_chroma_minus8),
      offsetof(struct virgl_h264_picture_desc, pps.sps.log2_max_frame_num_minus4),
      offsetof(struct virgl_h264_picture_desc, pps.sps.pic_order_cnt_type),
      offsetof(struct virgl_h264_picture_desc, pps.sps.log2_max_pic_order_cnt_lsb_minus4),
      offsetof(struct virgl_h264_picture_desc, pps.sps.delta_pic_order_always_zero_flag),
      offsetof(struct virgl_h264_picture_desc, pps.sps.offset_for_non_ref_pic),
      offsetof(struct virgl_h264_picture_desc, pps.sps.offset_for_top_to_bottom_field),
      offsetof(struct virgl_h264_picture_desc, pps.sps.offset_for_ref_frame),
      offsetof(struct virgl_h264_picture_desc, pps.sps.num_ref_frames_in_pic_order_cnt_cycle),
      offsetof(struct virgl_h264_picture_desc, pps.sps.frame_mbs_only_flag),
      offsetof(struct virgl_h264_picture_desc, pps.sps.direct_8x8_inference_flag),

      offsetof(struct virgl_h264_picture_desc, pps.entropy_coding_mode_flag),
      offsetof(struct virgl_h264_picture_desc, pps.bottom_field_pic_order_in_frame_present_flag),
      offsetof(struct virgl_h264_picture_desc, pps.num_slice_groups_minus1),
      offsetof(struct virgl_h264_picture_desc, pps.weighted_pred_flag),
      offsetof(struct virgl_h264_picture_desc, pps.weighted_bipred_idc),
      offsetof(struct virgl_h264_picture_desc, pps.pic_init_qp_minus26),
      offsetof(struct virgl_h264_picture_desc, pps.pic_init_qs_minus26),
      offsetof(struct virgl_h264_picture_desc, pps.chroma_qp_index_offset),
      offsetof(struct virgl_h264_picture_desc, pps.deblocking_filter_control_present_flag),
      offsetof(struct virgl_h264_picture_desc, pps.constrained_intra_pred_flag),
      offsetof(struct virgl_h264_picture_desc, pps.redundant_pic_cnt_present_flag),
      offsetof(struct virgl_h264_picture_desc, pps.transform_8x8_mode_flag),
      offsetof(struct virgl_h264_picture_desc, pps.ScalingList4x4),
      offsetof(struct virgl_h264_picture_desc, pps.ScalingList8x8),
      offsetof(struct virgl_h264_picture_desc, pps.second_chroma_qp_index_offset),

      offsetof(struct virgl_h264_picture_desc, field_pic_flag),
      offsetof(struct virgl_h264_picture_desc, num_ref_idx_l0_active_minus1),
      offsetof(struct virgl_h264_picture_desc, num_ref_idx_l1_active_minus1),
      offsetof(struct virgl_h264_picture_desc, num_ref_frames),
   };
   const size_t n = sizeof(offsets) / sizeof(offsets[0]);

   if (cap < n)
      return 0;
   memcpy(out, offsets, sizeof(offsets));
   return n;
}

size_t virgl_oracle_h264_desc_bytes(void)
{
   return sizeof(struct virgl_h264_picture_desc);
}

static uint64_t oracle_rng(uint64_t *s)
{
   *s = *s * 6364136223846793005ull + 1442695040888963407ull;
   return *s >> 11;
}

/*
 * Fill a descriptor the way a guest plausibly would, so that the differential mostly compares
 * emitted bytes rather than refusals.
 *
 * Everything starts as noise -- including the fields the serializer must NOT read, the dead
 * max_num_ref_frames and the dead PPS num_ref_idx defaults, so that reading one shows up as a
 * mismatch rather than passing by luck. The guarded fields are then set to what a 4:2:0
 * progressive stream carries, and `break_guard` picks one to violate, so the two sides are
 * compared on their refusals as well as on their output.
 */
void virgl_oracle_h264_desc_fill(uint8_t *out, uint64_t seed, int break_guard)
{
   struct virgl_h264_picture_desc *d = (struct virgl_h264_picture_desc *)out;
   uint64_t s = seed;

   for (size_t i = 0; i < sizeof(*d); i++)
      out[i] = (uint8_t)oracle_rng(&s);

   /* A scaling matrix is refused, and random bytes are always one. Send what a guest that has no
    * IQMatrix sends, or a flat list, which decodes the same as none. */
   memset(d->pps.ScalingList4x4, oracle_rng(&s) & 1 ? 0 : 16, sizeof(d->pps.ScalingList4x4));
   memset(d->pps.ScalingList8x8, oracle_rng(&s) & 1 ? 0 : 16, sizeof(d->pps.ScalingList8x8));

   d->field_pic_flag = 0;
   d->pps.sps.chroma_format_idc = 1;
   d->pps.sps.separate_colour_plane_flag = 0;
   d->pps.sps.bit_depth_luma_minus8 = 0;
   d->pps.sps.bit_depth_chroma_minus8 = 0;
   d->pps.sps.frame_mbs_only_flag = 1;
   d->pps.num_slice_groups_minus1 = 0;

   d->pps.sps.pic_order_cnt_type = oracle_rng(&s) % 3;
   /* Bounded so the C's fixed 512-byte SPS buffer can hold the cycle: 256 entries of signed
    * Exp-Golomb would overflow it, which the growing Rust writer has no equivalent for. */
   d->pps.sps.num_ref_frames_in_pic_order_cnt_cycle = oracle_rng(&s) % 17;
   for (unsigned i = 0; i < 256; i++)
      d->pps.sps.offset_for_ref_frame[i] = (int32_t)(oracle_rng(&s) % 512) - 256;

   switch (break_guard) {
   case 1: d->field_pic_flag = 1; break;
   case 2: d->pps.sps.chroma_format_idc = 3; break;
   case 3: d->pps.sps.separate_colour_plane_flag = 1; break;
   case 4: d->pps.sps.bit_depth_luma_minus8 = 2; break;
   case 5: d->pps.sps.bit_depth_chroma_minus8 = 2; break;
   case 6: d->pps.sps.frame_mbs_only_flag = 0; break;
   case 7: d->pps.num_slice_groups_minus1 = 1; break;
   case 8: d->pps.ScalingList4x4[0][0] = 17; break;
   case 9: d->pps.ScalingList8x8[1][63] = 17; break;
   /* Past the two lists mesa fills: untouched struct memory, which must not be read. */
   case 10: d->pps.ScalingList8x8[2][0] = 17; break;
   default: break;
   }
}

/*
 * Build the parameter sets, returning their lengths through `sps_len` / `pps_len`, or -1 for the
 * refusal the Rust reports by name.
 */
int virgl_oracle_h264_build(const uint8_t *desc, uint32_t width, uint32_t height,
                            uint32_t profile, unsigned pps_id,
                            uint8_t *sps, size_t *sps_len, uint8_t *pps, size_t *pps_len)
{
   struct virgl_h264_parameter_sets out;

   if (virgl_h264_build_parameter_sets((const struct virgl_h264_picture_desc *)desc,
                                       width, height, (enum pipe_video_profile)profile,
                                       pps_id, &out))
      return -1;

   memcpy(sps, out.sps, out.sps_len);
   memcpy(pps, out.pps, out.pps_len);
   *sps_len = out.sps_len;
   *pps_len = out.pps_len;
   return 0;
}

/* ---------------------------------------------------------------- HEVC parameter sets */

/* HEVC's default 8x8 scaling lists (Table 7-5/7-6), for seeding a descriptor that carries them. */
const uint8_t virgl_oracle_default_intra_8x8[64] = {
   16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18,
   17, 18, 18, 17, 18, 21, 19, 20, 21, 20, 19, 21, 24, 22, 22, 24,
   24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
   29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
};
const uint8_t virgl_oracle_default_inter_8x8[64] = {
   16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18,
   18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24,
   24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
   28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
};

#include "virgl_video_h265_ps.h"

/* The order the Rust `at` module declares them in; see the H.264 table for why it is compared
 * whole rather than field by field. */
size_t virgl_oracle_h265_offsets(size_t *out, size_t cap)
{
   const size_t offsets[] = {
      offsetof(struct virgl_h265_picture_desc, pps.sps.pic_width_in_luma_samples),
      offsetof(struct virgl_h265_picture_desc, pps.sps.pic_height_in_luma_samples),
      offsetof(struct virgl_h265_picture_desc, pps.sps.chroma_format_idc),
      offsetof(struct virgl_h265_picture_desc, pps.sps.separate_colour_plane_flag),
      offsetof(struct virgl_h265_picture_desc, pps.sps.bit_depth_luma_minus8),
      offsetof(struct virgl_h265_picture_desc, pps.sps.bit_depth_chroma_minus8),
      offsetof(struct virgl_h265_picture_desc, pps.sps.log2_max_pic_order_cnt_lsb_minus4),
      offsetof(struct virgl_h265_picture_desc, pps.sps.sps_max_dec_pic_buffering_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.sps.log2_min_luma_coding_block_size_minus3),
      offsetof(struct virgl_h265_picture_desc, pps.sps.log2_diff_max_min_luma_coding_block_size),
      offsetof(struct virgl_h265_picture_desc, pps.sps.log2_min_transform_block_size_minus2),
      offsetof(struct virgl_h265_picture_desc, pps.sps.log2_diff_max_min_transform_block_size),
      offsetof(struct virgl_h265_picture_desc, pps.sps.max_transform_hierarchy_depth_inter),
      offsetof(struct virgl_h265_picture_desc, pps.sps.max_transform_hierarchy_depth_intra),
      offsetof(struct virgl_h265_picture_desc, pps.sps.ScalingList4x4),
      offsetof(struct virgl_h265_picture_desc, pps.sps.ScalingList8x8),
      offsetof(struct virgl_h265_picture_desc, pps.sps.ScalingList16x16),
      offsetof(struct virgl_h265_picture_desc, pps.sps.ScalingList32x32),
      offsetof(struct virgl_h265_picture_desc, pps.sps.ScalingListDCCoeff16x16),
      offsetof(struct virgl_h265_picture_desc, pps.sps.ScalingListDCCoeff32x32),
      offsetof(struct virgl_h265_picture_desc, pps.sps.scaling_list_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.sps.amp_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.sps.sample_adaptive_offset_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.sps.pcm_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.sps.pcm_sample_bit_depth_luma_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.sps.pcm_sample_bit_depth_chroma_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.sps.log2_min_pcm_luma_coding_block_size_minus3),
      offsetof(struct virgl_h265_picture_desc, pps.sps.log2_diff_max_min_pcm_luma_coding_block_size),
      offsetof(struct virgl_h265_picture_desc, pps.sps.pcm_loop_filter_disabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.sps.num_short_term_ref_pic_sets),
      offsetof(struct virgl_h265_picture_desc, pps.sps.long_term_ref_pics_present_flag),
      offsetof(struct virgl_h265_picture_desc, pps.sps.num_long_term_ref_pics_sps),
      offsetof(struct virgl_h265_picture_desc, pps.sps.sps_temporal_mvp_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.sps.strong_intra_smoothing_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.dependent_slice_segments_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.output_flag_present_flag),
      offsetof(struct virgl_h265_picture_desc, pps.num_extra_slice_header_bits),
      offsetof(struct virgl_h265_picture_desc, pps.sign_data_hiding_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.cabac_init_present_flag),
      offsetof(struct virgl_h265_picture_desc, pps.num_ref_idx_l0_default_active_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.num_ref_idx_l1_default_active_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.init_qp_minus26),
      offsetof(struct virgl_h265_picture_desc, pps.constrained_intra_pred_flag),
      offsetof(struct virgl_h265_picture_desc, pps.transform_skip_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.cu_qp_delta_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.diff_cu_qp_delta_depth),
      offsetof(struct virgl_h265_picture_desc, pps.pps_cb_qp_offset),
      offsetof(struct virgl_h265_picture_desc, pps.pps_cr_qp_offset),
      offsetof(struct virgl_h265_picture_desc, pps.pps_slice_chroma_qp_offsets_present_flag),
      offsetof(struct virgl_h265_picture_desc, pps.weighted_pred_flag),
      offsetof(struct virgl_h265_picture_desc, pps.weighted_bipred_flag),
      offsetof(struct virgl_h265_picture_desc, pps.transquant_bypass_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.tiles_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.entropy_coding_sync_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.column_width_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.row_height_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.num_tile_columns_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.num_tile_rows_minus1),
      offsetof(struct virgl_h265_picture_desc, pps.uniform_spacing_flag),
      offsetof(struct virgl_h265_picture_desc, pps.loop_filter_across_tiles_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.pps_loop_filter_across_slices_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.deblocking_filter_control_present_flag),
      offsetof(struct virgl_h265_picture_desc, pps.deblocking_filter_override_enabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.pps_deblocking_filter_disabled_flag),
      offsetof(struct virgl_h265_picture_desc, pps.pps_beta_offset_div2),
      offsetof(struct virgl_h265_picture_desc, pps.pps_tc_offset_div2),
      offsetof(struct virgl_h265_picture_desc, pps.lists_modification_present_flag),
      offsetof(struct virgl_h265_picture_desc, pps.log2_parallel_merge_level_minus2),
      offsetof(struct virgl_h265_picture_desc, pps.slice_segment_header_extension_present_flag),
   };
   const size_t n = sizeof(offsets) / sizeof(offsets[0]);

   if (cap < n)
      return 0;
   memcpy(out, offsets, sizeof(offsets));
   return n;
}

size_t virgl_oracle_h265_desc_bytes(void)
{
   return sizeof(struct virgl_h265_picture_desc);
}

/*
 * Fill a descriptor the way a guest plausibly would. As for H.264: noise everywhere first, so a
 * field read from the wrong place shows up rather than passing by luck, then the guarded fields
 * set to what a 4:2:0 Main stream carries, then one guard broken on request.
 */
void virgl_oracle_h265_desc_fill(uint8_t *out, uint64_t seed, int break_guard)
{
   struct virgl_h265_picture_desc *d = (struct virgl_h265_picture_desc *)out;
   struct virgl_h265_sps *sps = &d->pps.sps;
   uint64_t s = seed;

   for (size_t i = 0; i < sizeof(*d); i++)
      out[i] = (uint8_t)oracle_rng(&s);

   sps->separate_colour_plane_flag = 0;
   sps->chroma_format_idc = 1;
   sps->num_long_term_ref_pics_sps = 0;

   /* Random lists are never the defaults, and the refusal would be the only thing measured.
    * Send the defaults, in a shuffled order half the time -- the comparison is on the sorted
    * values, and a scan order the wire has not pinned down must not change the answer. */
   const bool shuffle = oracle_rng(&s) & 1;
   memset(sps->ScalingList4x4, 16, sizeof(sps->ScalingList4x4));
   memset(sps->ScalingListDCCoeff16x16, 16, sizeof(sps->ScalingListDCCoeff16x16));
   memset(sps->ScalingListDCCoeff32x32, 16, sizeof(sps->ScalingListDCCoeff32x32));
   for (unsigned m = 0; m < 6; m++) {
      const uint8_t *def = m < 3 ? virgl_oracle_default_intra_8x8 : virgl_oracle_default_inter_8x8;
      memcpy(sps->ScalingList8x8[m], def, 64);
      memcpy(sps->ScalingList16x16[m], def, 64);
      if (m < 2)
         memcpy(sps->ScalingList32x32[m],
                m ? virgl_oracle_default_inter_8x8 : virgl_oracle_default_intra_8x8, 64);
   }
   if (shuffle) {
      /* One swap per list is enough to prove the comparison does not depend on the order. */
      for (unsigned m = 0; m < 6; m++) {
         unsigned a = oracle_rng(&s) % 64, b = oracle_rng(&s) % 64;
         uint8_t t = sps->ScalingList8x8[m][a];
         sps->ScalingList8x8[m][a] = sps->ScalingList8x8[m][b];
         sps->ScalingList8x8[m][b] = t;
      }
   }

   /* The coded size must cover the display sizes the test asks for, and the sets and tile counts
    * must stay small enough for the C's fixed 512-byte buffers. */
   sps->pic_width_in_luma_samples = 1920 + 64 * (oracle_rng(&s) % 4);
   sps->pic_height_in_luma_samples = 1088 + 64 * (oracle_rng(&s) % 4);
   sps->num_short_term_ref_pic_sets = oracle_rng(&s) % 9;
   sps->log2_max_pic_order_cnt_lsb_minus4 = oracle_rng(&s) % 13;
   d->pps.num_tile_columns_minus1 = oracle_rng(&s) % 20;
   d->pps.num_tile_rows_minus1 = oracle_rng(&s) % 22;
   for (unsigned i = 0; i < 20; i++)
      d->pps.column_width_minus1[i] = oracle_rng(&s) % 4096;
   for (unsigned i = 0; i < 22; i++)
      d->pps.row_height_minus1[i] = oracle_rng(&s) % 4096;

   switch (break_guard) {
   case 1: sps->scaling_list_enabled_flag = 1; sps->ScalingList4x4[0][0] = 17; break;
   case 2: sps->separate_colour_plane_flag = 1; break;
   case 3: sps->chroma_format_idc = 3; break;
   case 4: sps->num_long_term_ref_pics_sps = 2; break;
   case 5: sps->pic_width_in_luma_samples = 16; break;
   case 6: sps->pic_height_in_luma_samples = 16; break;
   /* A tile count past the end of the array the wire carries sizes in. The C reads past it. */
   case 7: d->pps.tiles_enabled_flag = 1; d->pps.uniform_spacing_flag = 0;
           d->pps.num_tile_columns_minus1 = 40; break;
   case 8: sps->scaling_list_enabled_flag = 1; sps->ScalingList8x8[0][0] += 1; break;
   case 9: sps->scaling_list_enabled_flag = 1; sps->ScalingListDCCoeff32x32[1] = 17; break;
   default: break;
   }
}

int virgl_oracle_h265_build(const uint8_t *desc, uint32_t width, uint32_t height,
                            uint32_t profile,
                            uint8_t *vps, size_t *vps_len,
                            uint8_t *sps, size_t *sps_len,
                            uint8_t *pps, size_t *pps_len)
{
   struct virgl_h265_parameter_sets out;

   if (virgl_h265_build_parameter_sets((const struct virgl_h265_picture_desc *)desc,
                                       width, height, (enum pipe_video_profile)profile, &out))
      return -1;

   memcpy(vps, out.vps, out.vps_len);
   memcpy(sps, out.sps, out.sps_len);
   memcpy(pps, out.pps, out.pps_len);
   *vps_len = out.vps_len;
   *sps_len = out.sps_len;
   *pps_len = out.pps_len;
   return 0;
}

int virgl_oracle_h265_slice_inspect(const uint8_t *annexb, size_t len, const uint8_t *desc,
                                    unsigned *out_id)
{
   return virgl_h265_slice_inspect(annexb, len,
                                   (const struct virgl_h265_picture_desc *)desc, out_id);
}
