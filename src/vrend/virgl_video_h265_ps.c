/*
 * HEVC parameter set synthesis for the VideoToolbox backend. See the header for why the
 * short_term_ref_pic_sets are placeholders and what makes that sound.
 */
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "virgl_util.h"
#include "virgl_video_bitstream.h"
#include "virgl_video_h265_ps.h"

/* NAL unit types we care about. */
#define H265_NAL_IDR_W_RADL 19
#define H265_NAL_IDR_N_LP   20
#define H265_NAL_BLA_W_LP   16
#define H265_NAL_RSV_IRAP23 23
#define H265_NAL_VPS        32
#define H265_NAL_SPS        33
#define H265_NAL_PPS        34

/*
 * Level 5.1. Not on the wire, and deliberately generous: it must cover anything the guest
 * can hand us, and it must never change. A level change is the one format-description
 * delta a live decompression session refuses outright (-12916, measured in
 * spikes/hevc-vt-probe), so a level derived from the stream would turn a resolution
 * change into a lost reference picture buffer.
 */
#define H265_LEVEL_IDC 153

/* ---------------------------------------------------------------- scaling lists */

/*
 * HEVC's default 8x8 scaling lists (Table 7-5/7-6), used for the 8x8, 16x16 and 32x32
 * sizes: matrix ids 0..2 are intra, 3..5 inter.
 */
static const uint8_t default_intra_8x8[64] = {
   16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18,
   17, 18, 18, 17, 18, 21, 19, 20, 21, 20, 19, 21, 24, 22, 22, 24,
   24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
   29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
};
static const uint8_t default_inter_8x8[64] = {
   16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18,
   18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24,
   24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
   28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
};

static int u8cmp(const void *a, const void *b)
{
   return (int)*(const uint8_t *)a - (int)*(const uint8_t *)b;
}

/*
 * Are these the default lists?
 *
 * We never emit scaling list DATA -- only the enable flag, with
 * sps_scaling_list_data_present_flag = 0, which selects the defaults. That is exact when
 * the stream did the same, and silently wrong when it carried custom lists, so the
 * distinction has to be made here. It is not on the wire: mesa hands us the effective
 * lists either way.
 *
 * The comparison is on the SORTED values, because the scan order VA-API delivers these in
 * is not established (the same reason the H.264 serializer refuses custom matrices), and a
 * multiset comparison does not depend on it. A custom list that is a permutation of the
 * default would slip through; nothing plausible produces one.
 */
static bool lists_are_default(const struct virgl_h265_sps *sps)
{
   uint8_t got[64], want[64];

   for (unsigned m = 0; m < 6; m++)
      for (unsigned i = 0; i < 16; i++)
         if (sps->ScalingList4x4[m][i] != 16)
            return false;

   for (unsigned m = 0; m < 6; m++)
      if (sps->ScalingListDCCoeff16x16[m] != 16)
         return false;
   for (unsigned m = 0; m < 2; m++)
      if (sps->ScalingListDCCoeff32x32[m] != 16)
         return false;

   for (unsigned m = 0; m < 6; m++) {
      const uint8_t *def = m < 3 ? default_intra_8x8 : default_inter_8x8;

      memcpy(want, def, 64);
      qsort(want, 64, 1, u8cmp);

      memcpy(got, sps->ScalingList8x8[m], 64);
      qsort(got, 64, 1, u8cmp);
      if (memcmp(got, want, 64))
         return false;

      memcpy(got, sps->ScalingList16x16[m], 64);
      qsort(got, 64, 1, u8cmp);
      if (memcmp(got, want, 64))
         return false;

      if (m < 2) {
         memcpy(want, m ? default_inter_8x8 : default_intra_8x8, 64);
         qsort(want, 64, 1, u8cmp);
         memcpy(got, sps->ScalingList32x32[m], 64);
         qsort(got, 64, 1, u8cmp);
         if (memcmp(got, want, 64))
            return false;
      }
   }

   return true;
}

static uint8_t profile_idc_of(enum pipe_video_profile profile)
{
   switch (profile) {
   case PIPE_VIDEO_PROFILE_HEVC_MAIN_10:
      return 2;
   case PIPE_VIDEO_PROFILE_HEVC_MAIN:
   default:
      return 1;
   }
}

/* profile_tier_level(1, 0) -- 96 bits, no sub-layers. */
static void write_ptl(struct bs *w, uint8_t profile_idc)
{
   u(w, 2, 0);                /* general_profile_space */
   flag(w, 0);                /* general_tier_flag: main tier */
   u(w, 5, profile_idc);      /* general_profile_idc */

   /* general_profile_compatibility_flag[32]: only our own profile's bit. */
   for (unsigned i = 0; i < 32; i++)
      flag(w, i == profile_idc);

   flag(w, 1);                /* general_progressive_source_flag */
   flag(w, 0);                /* general_interlaced_source_flag */
   flag(w, 1);                /* general_non_packed_constraint_flag */
   flag(w, 1);                /* general_frame_only_constraint_flag */

   /* general_reserved_zero_43bits, then general_inbld_flag/reserved. */
   u(w, 22, 0);
   u(w, 21, 0);
   flag(w, 0);

   u(w, 8, H265_LEVEL_IDC);
}

static void write_nal_header(struct bs *w, unsigned type)
{
   flag(w, 0);                /* forbidden_zero_bit */
   u(w, 6, type);             /* nal_unit_type */
   u(w, 6, 0);                /* nuh_layer_id */
   u(w, 3, 1);                /* nuh_temporal_id_plus1 */
}

/*
 * A whole VPS from nothing. Nothing downstream reads more of it than the
 * profile_tier_level and the buffering values, both of which have SPS twins on the wire.
 */
static void write_vps(struct bs *w, const struct virgl_h265_picture_desc *desc,
                      uint8_t profile_idc)
{
   const struct virgl_h265_sps *sps = &desc->pps.sps;

   write_nal_header(w, H265_NAL_VPS);

   u(w, 4, 0);                /* vps_video_parameter_set_id */
   flag(w, 1);                /* vps_base_layer_internal_flag */
   flag(w, 1);                /* vps_base_layer_available_flag */
   u(w, 6, 0);                /* vps_max_layers_minus1 */
   u(w, 3, 0);                /* vps_max_sub_layers_minus1 */
   flag(w, 1);                /* vps_temporal_id_nesting_flag */
   u(w, 16, 0xffff);          /* vps_reserved_0xffff_16bits */

   write_ptl(w, profile_idc);

   flag(w, 1);                /* vps_sub_layer_ordering_info_present_flag */
   ue(w, sps->sps_max_dec_pic_buffering_minus1);
   ue(w, sps->sps_max_dec_pic_buffering_minus1);   /* vps_max_num_reorder_pics */
   ue(w, 0);                  /* vps_max_latency_increase_plus1: no limit */

   u(w, 6, 0);                /* vps_max_layer_id */
   ue(w, 0);                  /* vps_num_layer_sets_minus1 */
   flag(w, 0);                /* vps_timing_info_present_flag */
   flag(w, 0);                /* vps_extension_flag */
   rbsp_trailing(w);
}

static void write_sps(struct bs *w, const struct virgl_h265_picture_desc *desc,
                      uint32_t width, uint32_t height, uint8_t profile_idc)
{
   const struct virgl_h265_sps *sps = &desc->pps.sps;
   const unsigned sub_w = sps->chroma_format_idc == 3 ? 1 : 2;
   const unsigned sub_h = sps->chroma_format_idc == 1 ? 2 : 1;

   write_nal_header(w, H265_NAL_SPS);

   u(w, 4, 0);                /* sps_video_parameter_set_id */
   u(w, 3, 0);                /* sps_max_sub_layers_minus1 */
   flag(w, 1);                /* sps_temporal_id_nesting_flag */

   write_ptl(w, profile_idc);

   ue(w, 0);                  /* sps_seq_parameter_set_id */
   ue(w, sps->chroma_format_idc);
   if (sps->chroma_format_idc == 3)
      flag(w, sps->separate_colour_plane_flag);

   ue(w, sps->pic_width_in_luma_samples);
   ue(w, sps->pic_height_in_luma_samples);

   /*
    * The coded size is on the wire; the display size is the codec's. Their difference is
    * the conformance window, in chroma units. Unlike H.264 the geometry needs no
    * derivation -- only the crop does.
    */
   if (sps->pic_width_in_luma_samples > width || sps->pic_height_in_luma_samples > height) {
      flag(w, 1);             /* conformance_window_flag */
      ue(w, 0);               /* conf_win_left_offset */
      ue(w, (sps->pic_width_in_luma_samples - width) / sub_w);
      ue(w, 0);               /* conf_win_top_offset */
      ue(w, (sps->pic_height_in_luma_samples - height) / sub_h);
   } else {
      flag(w, 0);
   }

   ue(w, sps->bit_depth_luma_minus8);
   ue(w, sps->bit_depth_chroma_minus8);
   ue(w, sps->log2_max_pic_order_cnt_lsb_minus4);

   flag(w, 1);                /* sps_sub_layer_ordering_info_present_flag */
   ue(w, sps->sps_max_dec_pic_buffering_minus1);
   ue(w, sps->sps_max_dec_pic_buffering_minus1);   /* sps_max_num_reorder_pics */
   ue(w, 0);                  /* sps_max_latency_increase_plus1 */

   ue(w, sps->log2_min_luma_coding_block_size_minus3);
   ue(w, sps->log2_diff_max_min_luma_coding_block_size);
   ue(w, sps->log2_min_transform_block_size_minus2);
   ue(w, sps->log2_diff_max_min_transform_block_size);
   ue(w, sps->max_transform_hierarchy_depth_inter);
   ue(w, sps->max_transform_hierarchy_depth_intra);

   flag(w, sps->scaling_list_enabled_flag);
   if (sps->scaling_list_enabled_flag)
      flag(w, 0);             /* sps_scaling_list_data_present_flag: the defaults,
                              * which lists_are_default() has confirmed */
   flag(w, sps->amp_enabled_flag);
   flag(w, sps->sample_adaptive_offset_enabled_flag);

   flag(w, sps->pcm_enabled_flag);
   if (sps->pcm_enabled_flag) {
      u(w, 4, sps->pcm_sample_bit_depth_luma_minus1);
      u(w, 4, sps->pcm_sample_bit_depth_chroma_minus1);
      ue(w, sps->log2_min_pcm_luma_coding_block_size_minus3);
      ue(w, sps->log2_diff_max_min_pcm_luma_coding_block_size);
      flag(w, sps->pcm_loop_filter_disabled_flag);
   }

   /*
    * The sets themselves are not on the wire and cannot be. Only the COUNT matters to a
    * slice header, which reads an index whose width derives from it; the contents are read
    * only by a slice that indexes one, and virgl_h265_slice_inspect refuses those. Note
    * st_ref_pic_set(i) carries inter_ref_pic_set_prediction_flag for every i != 0 --
    * omitting it desyncs the parse of this SPS.
    */
   ue(w, sps->num_short_term_ref_pic_sets);
   for (unsigned i = 0; i < sps->num_short_term_ref_pic_sets; i++) {
      if (i)
         flag(w, 0);          /* inter_ref_pic_set_prediction_flag */
      ue(w, 0);               /* num_negative_pics */
      ue(w, 0);               /* num_positive_pics */
   }

   flag(w, sps->long_term_ref_pics_present_flag);
   if (sps->long_term_ref_pics_present_flag)
      ue(w, 0);               /* num_long_term_ref_pics_sps; refused above if nonzero */

   flag(w, sps->sps_temporal_mvp_enabled_flag);
   flag(w, sps->strong_intra_smoothing_enabled_flag);
   flag(w, 0);                /* vui_parameters_present_flag */
   flag(w, 0);                /* sps_extension_present_flag */
   rbsp_trailing(w);
}

static void write_pps(struct bs *w, const struct virgl_h265_picture_desc *desc)
{
   const struct virgl_h265_pps *pps = &desc->pps;

   write_nal_header(w, H265_NAL_PPS);

   ue(w, 0);                  /* pps_pic_parameter_set_id */
   ue(w, 0);                  /* pps_seq_parameter_set_id */

   flag(w, pps->dependent_slice_segments_enabled_flag);
   flag(w, pps->output_flag_present_flag);
   u(w, 3, pps->num_extra_slice_header_bits);
   flag(w, pps->sign_data_hiding_enabled_flag);
   flag(w, pps->cabac_init_present_flag);

   ue(w, pps->num_ref_idx_l0_default_active_minus1);
   ue(w, pps->num_ref_idx_l1_default_active_minus1);
   se(w, pps->init_qp_minus26);

   flag(w, pps->constrained_intra_pred_flag);
   flag(w, pps->transform_skip_enabled_flag);
   flag(w, pps->cu_qp_delta_enabled_flag);
   if (pps->cu_qp_delta_enabled_flag)
      ue(w, pps->diff_cu_qp_delta_depth);

   se(w, pps->pps_cb_qp_offset);
   se(w, pps->pps_cr_qp_offset);
   flag(w, pps->pps_slice_chroma_qp_offsets_present_flag);
   flag(w, pps->weighted_pred_flag);
   flag(w, pps->weighted_bipred_flag);
   flag(w, pps->transquant_bypass_enabled_flag);
   flag(w, pps->tiles_enabled_flag);
   flag(w, pps->entropy_coding_sync_enabled_flag);

   if (pps->tiles_enabled_flag) {
      ue(w, pps->num_tile_columns_minus1);
      ue(w, pps->num_tile_rows_minus1);
      flag(w, pps->uniform_spacing_flag);
      if (!pps->uniform_spacing_flag) {
         for (unsigned i = 0; i < pps->num_tile_columns_minus1; i++)
            ue(w, pps->column_width_minus1[i]);
         for (unsigned i = 0; i < pps->num_tile_rows_minus1; i++)
            ue(w, pps->row_height_minus1[i]);
      }
      flag(w, pps->loop_filter_across_tiles_enabled_flag);
   }

   flag(w, pps->pps_loop_filter_across_slices_enabled_flag);
   flag(w, pps->deblocking_filter_control_present_flag);
   if (pps->deblocking_filter_control_present_flag) {
      flag(w, pps->deblocking_filter_override_enabled_flag);
      flag(w, pps->pps_deblocking_filter_disabled_flag);
      if (!pps->pps_deblocking_filter_disabled_flag) {
         se(w, pps->pps_beta_offset_div2);
         se(w, pps->pps_tc_offset_div2);
      }
   }

   flag(w, 0);                /* pps_scaling_list_data_present_flag */
   flag(w, pps->lists_modification_present_flag);
   ue(w, pps->log2_parallel_merge_level_minus2);
   flag(w, pps->slice_segment_header_extension_present_flag);
   flag(w, 0);                /* pps_extension_present_flag */
   rbsp_trailing(w);
}

int virgl_h265_build_parameter_sets(const struct virgl_h265_picture_desc *desc,
                                    uint32_t width, uint32_t height,
                                    enum pipe_video_profile profile,
                                    struct virgl_h265_parameter_sets *out)
{
   const struct virgl_h265_sps *sps;
   uint8_t profile_idc;
   struct bs w;

   if (!desc || !out || !width || !height)
      return -1;

   sps = &desc->pps.sps;
   profile_idc = profile_idc_of(profile);

   /*
    * Refuse rather than emit something plausible. Each of these decodes to subtly wrong
    * pixels if guessed, which is the failure mode that costs days.
    */
   if (sps->scaling_list_enabled_flag && !lists_are_default(sps)) {
      virgl_error("video: h265 stream carries custom scaling lists, whose scan order on the"
                  " VA-API wire is not established; refusing rather than dequantizing wrong\n");
      return -1;
   }
   if (sps->separate_colour_plane_flag) {
      virgl_error("video: h265 separate colour planes are not supported\n");
      return -1;
   }
   if (sps->chroma_format_idc != 1) {
      virgl_error("video: h265 chroma_format_idc %u (only 4:2:0 is supported)\n",
                  sps->chroma_format_idc);
      return -1;
   }
   if (sps->num_long_term_ref_pics_sps) {
      /* The SPS long-term entries are missing from the wire exactly as the short-term
       * sets are. Slice-carried long-term references are self-contained and fine. */
      virgl_error("video: h265 stream declares %u long term ref pics in the SPS,"
                  " whose contents the wire does not carry\n",
                  sps->num_long_term_ref_pics_sps);
      return -1;
   }
   if (sps->pic_width_in_luma_samples < width || sps->pic_height_in_luma_samples < height) {
      virgl_error("video: h265 coded size %ux%u is smaller than the display size %ux%u\n",
                  sps->pic_width_in_luma_samples, sps->pic_height_in_luma_samples,
                  width, height);
      return -1;
   }

   bs_init(&w, out->vps, sizeof(out->vps), true);
   write_vps(&w, desc, profile_idc);
   if (w.overflow)
      return -1;
   out->vps_len = w.pos;

   bs_init(&w, out->sps, sizeof(out->sps), true);
   write_sps(&w, desc, width, height, profile_idc);
   if (w.overflow)
      return -1;
   out->sps_len = w.pos;

   bs_init(&w, out->pps, sizeof(out->pps), true);
   write_pps(&w, desc);
   if (w.overflow)
      return -1;
   out->pps_len = w.pos;

   return 0;
}

/* ---------------------------------------------------------------- slice inspection */

int virgl_h265_slice_inspect(const uint8_t *annexb, size_t len,
                             const struct virgl_h265_picture_desc *desc,
                             unsigned *out_id)
{
   const struct virgl_h265_pps *pps;
   const uint8_t *p = annexb, *end;
   unsigned sc;

   if (!annexb || !desc || !out_id)
      return -1;

   pps = &desc->pps;
   end = annexb + len;

   while ((p = next_start_code(p, end, &sc)) != NULL) {
      const uint8_t *nal = p + sc;
      unsigned type;
      struct br r;
      uint32_t first, v;

      if (nal + 2 >= end)
         break;

      type = (nal[0] >> 1) & 0x3f;
      if (type > 31) {         /* not a VCL NAL */
         p = nal;
         continue;
      }

      /* The two-byte NAL header is not part of the RBSP. */
      r = (struct br){ .buf = nal + 2, .len = (size_t)(end - nal - 2) };

      if (br_u(&r, 1, &first))
         return -1;
      if (!first) {
         /* A dependent or later slice segment. Its header needs the CTB address width to
          * parse, and the picture's first slice has already answered every question we
          * have, so there is nothing to learn here. */
         p = nal;
         continue;
      }

      if (type >= H265_NAL_BLA_W_LP && type <= H265_NAL_RSV_IRAP23 &&
          br_u(&r, 1, &v))     /* no_output_of_prior_pics_flag */
         return -1;

      if (br_ue(&r, &v))       /* slice_pic_parameter_set_id */
         return -1;
      if (v > 63)
         return -1;
      *out_id = v;

      /* An IDR carries no reference picture set, so there is nothing left to check. */
      if (type == H265_NAL_IDR_W_RADL || type == H265_NAL_IDR_N_LP)
         return 0;

      for (unsigned i = 0; i < pps->num_extra_slice_header_bits; i++)
         if (br_u(&r, 1, &v))
            return -1;

      if (br_ue(&r, &v))       /* slice_type */
         return -1;
      if (pps->output_flag_present_flag && br_u(&r, 1, &v))
         return -1;
      if (pps->sps.separate_colour_plane_flag && br_u(&r, 2, &v))
         return -1;

      /* slice_pic_order_cnt_lsb */
      if (br_u(&r, pps->sps.log2_max_pic_order_cnt_lsb_minus4 + 4, &v))
         return -1;

      if (br_u(&r, 1, &v))     /* short_term_ref_pic_set_sps_flag */
         return -1;
      if (v) {
         /*
          * The slice indexes one of the SPS sets, whose contents are not on the wire and
          * which we therefore emitted empty. Decoding would silently use the wrong
          * reference pictures. Nothing about this is visible before the first inter
          * predicted slice, so the refusal necessarily lands one frame into playback.
          */
         virgl_error("video: h265 slice indexes an SPS short term ref pic set;"
                     " the set contents are not on the VA-API wire, refusing the stream\n");
         return -1;
      }
      if (pps->sps.num_short_term_ref_pic_sets) {
         if (br_u(&r, 1, &v)) /* inter_ref_pic_set_prediction_flag */
            return -1;
         if (v) {
            virgl_error("video: h265 slice predicts its ref pic set from an SPS set,"
                        " whose contents the wire does not carry, refusing the stream\n");
            return -1;
         }
      }
      return 0;
   }

   return 1;                   /* no slice header in this submission yet */
}
