/**************************************************************************
 *
 * Copyright (C) 2026 Gustavo Noronha Silva
 *
 * SPDX-License-Identifier: MIT
 *
 **************************************************************************/

/*
 * See virgl_video_h264_ps.h for what this exists to do and why.
 *
 * Everything here is written against ITU-T H.264 (7.3.2.1.1 for the SPS, 7.3.2.2 for the
 * PPS, 7.3.3 for the slice header, 7.4.1.1 for emulation prevention). Where the wire's
 * struct virgl_h264_sps has no field for a syntax element, the value is derived and the
 * derivation is commented -- those are the interesting lines.
 */

#include <string.h>

#include "virgl_util.h"
#include "virgl_video_h264_ps.h"

/* ---------------------------------------------------------------- bit writer */

struct bs {
   uint8_t *buf;
   size_t cap;
   size_t pos;
   uint32_t acc;
   unsigned nbits;
   bool overflow;
   /* Emulation prevention state: how many consecutive zero bytes have been emitted.
    * Reset when the caller starts a new NAL. */
   unsigned zeros;
   bool escape;
};

static void bs_init(struct bs *w, uint8_t *buf, size_t cap, bool escape)
{
   memset(w, 0, sizeof(*w));
   w->buf = buf;
   w->cap = cap;
   w->escape = escape;
}

static void bs_raw_byte(struct bs *w, uint8_t b)
{
   if (w->pos >= w->cap) {
      w->overflow = true;
      return;
   }
   w->buf[w->pos++] = b;
}

/*
 * Emit one RBSP byte, inserting the emulation-prevention byte where the spec requires it.
 * Any 00 00 00, 00 00 01, 00 00 02 or 00 00 03 in the payload would otherwise be
 * indistinguishable from a start code, so 00 00 0x becomes 00 00 03 0x.
 */
static void bs_byte(struct bs *w, uint8_t b)
{
   if (w->escape && w->zeros >= 2 && b <= 0x03) {
      bs_raw_byte(w, 0x03);
      w->zeros = 0;
   }
   bs_raw_byte(w, b);
   w->zeros = b ? 0 : w->zeros + 1;
}

/* u(n) / f(n): n bits, most significant first. */
static void u(struct bs *w, unsigned n, uint32_t v)
{
   for (unsigned i = 0; i < n; i++) {
      w->acc = (w->acc << 1) | ((v >> (n - 1 - i)) & 1u);
      if (++w->nbits == 8) {
         bs_byte(w, (uint8_t)w->acc);
         w->acc = 0;
         w->nbits = 0;
      }
   }
}

static void flag(struct bs *w, bool v) { u(w, 1, v ? 1 : 0); }

/* ue(v): Exp-Golomb. codeNum v is written as a prefix of N zeros, a 1, then N bits. */
static void ue(struct bs *w, uint32_t v)
{
   unsigned n = 0;
   uint64_t x = (uint64_t)v + 1;   /* 64-bit: v == UINT32_MAX would overflow 32 bits */

   while ((x >> (n + 1)) != 0)
      n++;

   u(w, n, 0);
   u(w, 1, 1);
   if (n)
      u(w, n, (uint32_t)(x & ((1u << n) - 1)));
}

/* se(v): signed Exp-Golomb, mapped 0, 1, -1, 2, -2, ... */
static void se(struct bs *w, int32_t v)
{
   uint32_t code = v <= 0 ? (uint32_t)(-2 * (int64_t)v)
                          : (uint32_t)(2 * (int64_t)v - 1);
   ue(w, code);
}

/* rbsp_trailing_bits(): a 1 bit, then zeros to the byte boundary. */
static void rbsp_trailing(struct bs *w)
{
   flag(w, 1);
   while (w->nbits)
      flag(w, 0);
}

/* ---------------------------------------------------------------- derivations */

static uint8_t profile_idc_of(enum pipe_video_profile profile)
{
   switch (profile) {
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE:
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_CONSTRAINED_BASELINE:
      return 66;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_MAIN:      return 77;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_EXTENDED:  return 88;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH:      return 100;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH10:    return 110;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH422:   return 122;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH444:   return 244;
   default:                                     return 100;
   }
}

/*
 * The chroma/bit-depth/scaling block is only present for the High family. Emitting it for
 * a Baseline/Main SPS would not merely be redundant, it would be misparsed -- the decoder
 * reads the next fields as log2_max_frame_num_minus4 and everything after shifts.
 */
static bool profile_has_chroma_block(uint8_t idc)
{
   switch (idc) {
   case 100: case 110: case 122: case 244:
   case 44: case 83: case 86: case 118:
   case 128: case 138: case 139: case 134: case 135:
      return true;
   default:
      return false;
   }
}

/* A flat list decodes identically to no list at all, so it needs no signalling. */
static bool list_is_flat(const uint8_t *list, size_t n)
{
   for (size_t i = 0; i < n; i++)
      if (list[i] != 16)
         return false;
   return true;
}

static bool list_is_zero(const uint8_t *list, size_t n)
{
   for (size_t i = 0; i < n; i++)
      if (list[i])
         return false;
   return true;
}

/*
 * Whether this descriptor carries a scaling matrix we would have to serialize.
 *
 * We deliberately refuse those for now rather than emit one. The wire lists come straight
 * from VA-API's VAIQMatrixBufferH264 (mesa's picture_h264.c memcpy's them), and the scan
 * order those are in is not something this code has verified against a real stream with a
 * non-flat matrix. Emitting them in the wrong order is invisible -- the stream decodes,
 * with subtly wrong dequantization -- which is exactly the failure this file must not
 * produce. An all-zero array means the guest sent no IQMatrix at all and is treated as
 * flat, not as a matrix.
 */
static bool has_scaling_matrix(const struct virgl_h264_pps *pps)
{
   if (!list_is_zero(&pps->ScalingList4x4[0][0], 6 * 16) &&
       !list_is_flat(&pps->ScalingList4x4[0][0], 6 * 16))
      return true;
   /* Only two 8x8 lists are meaningful for 4:2:0 -- mesa copies exactly 2 * 64 bytes from
    * VA-API and leaves the rest of the [6][64] array untouched, so reading all six would
    * be reading whatever was in the struct. */
   if (!list_is_zero(&pps->ScalingList8x8[0][0], 2 * 64) &&
       !list_is_flat(&pps->ScalingList8x8[0][0], 2 * 64))
      return true;
   return false;
}

/* ---------------------------------------------------------------- SPS / PPS */

static void write_sps(struct bs *w, const struct virgl_h264_picture_desc *desc,
                      uint32_t width, uint32_t height, uint8_t idc, unsigned sps_id)
{
   const struct virgl_h264_sps *sps = &desc->pps.sps;

   /* NAL header: forbidden_zero=0, nal_ref_idc=3, nal_unit_type=7 (SPS). */
   bs_raw_byte(w, 0x67);

   u(w, 8, idc);
   u(w, 8, 0);                 /* constraint_set0..5 + 2 reserved: all clear. They only
                                * ever narrow a profile, so clearing them is always safe. */
   u(w, 8, sps->level_idc);
   ue(w, sps_id);

   if (profile_has_chroma_block(idc)) {
      ue(w, sps->chroma_format_idc);
      if (sps->chroma_format_idc == 3)
         flag(w, sps->separate_colour_plane_flag);
      ue(w, sps->bit_depth_luma_minus8);
      ue(w, sps->bit_depth_chroma_minus8);
      flag(w, 0);              /* qpprime_y_zero_transform_bypass_flag: lossless coding,
                                * not on the wire and not in this cut's scope. */
      flag(w, 0);              /* seq_scaling_matrix_present_flag -- see has_scaling_matrix */
   }

   ue(w, sps->log2_max_frame_num_minus4);
   ue(w, sps->pic_order_cnt_type);
   if (sps->pic_order_cnt_type == 0) {
      ue(w, sps->log2_max_pic_order_cnt_lsb_minus4);
   } else if (sps->pic_order_cnt_type == 1) {
      flag(w, sps->delta_pic_order_always_zero_flag);
      se(w, sps->offset_for_non_ref_pic);
      se(w, sps->offset_for_top_to_bottom_field);
      ue(w, sps->num_ref_frames_in_pic_order_cnt_cycle);
      for (unsigned i = 0; i < sps->num_ref_frames_in_pic_order_cnt_cycle; i++)
         se(w, sps->offset_for_ref_frame[i]);
   }

   /*
    * The DPB size, and a trap: struct virgl_h264_sps HAS a max_num_ref_frames field and it
    * is dead on the decode path. mesa's VA frontend puts VA-API's num_ref_frames into the
    * *picture descriptor* instead (picture_h264.c: `desc.h264.num_ref_frames`), and only
    * the ENCODER frontend ever writes the SPS one. Reading the SPS field yields 0, which
    * VideoToolbox honours: it sizes the DPB at zero, drops every reference, and rejects the
    * third frame onward with kVTVideoDecoderBadDataErr while the first two decode fine.
    *
    * Clamped to at least 1: a stream with inter prediction needs somewhere to keep the
    * picture it predicts from, and 0 is what "the guest did not say" looks like.
    */
   ue(w, desc->num_ref_frames ? desc->num_ref_frames : 1);
   flag(w, 0);                 /* gaps_in_frame_num_value_allowed_flag */

   /*
    * Geometry. The wire has no macroblock counts, so they come from the codec object's
    * display size rounded up, and the remainder is cropped away below. Chroma units for
    * 4:2:0 are 2 luma samples horizontally, and 2 * (2 - frame_mbs_only_flag) vertically.
    */
   {
      const uint32_t mbs_w = (width + 15) / 16;
      const uint32_t map_h = (height + 15) / 16;   /* frame_mbs_only_flag == 1 here */
      const uint32_t crop_r = (mbs_w * 16 - width) / 2;
      const uint32_t crop_b = (map_h * 16 - height) / 2;

      ue(w, mbs_w - 1);
      ue(w, map_h - 1);
      flag(w, 1);              /* frame_mbs_only_flag: progressive only, enforced by the
                                * caller rejecting field_pic_flag */
      flag(w, sps->direct_8x8_inference_flag);

      if (crop_r || crop_b) {
         flag(w, 1);
         ue(w, 0);             /* left   */
         ue(w, crop_r);        /* right  */
         ue(w, 0);             /* top    */
         ue(w, crop_b);        /* bottom */
      } else {
         flag(w, 0);
      }
   }

   flag(w, 0);                 /* vui_parameters_present_flag: VUI carries timing and
                                * colour metadata the guest already applies itself. */
   rbsp_trailing(w);
}

static void write_pps(struct bs *w, const struct virgl_h264_picture_desc *desc,
                      unsigned pps_id, unsigned sps_id)
{
   const struct virgl_h264_pps *pps = &desc->pps;

   /* NAL header: nal_ref_idc=3, nal_unit_type=8 (PPS). */
   bs_raw_byte(w, 0x68);

   ue(w, pps_id);
   ue(w, sps_id);
   flag(w, pps->entropy_coding_mode_flag);
   flag(w, pps->bottom_field_pic_order_in_frame_present_flag);
   ue(w, 0);                   /* num_slice_groups_minus1: FMO is Baseline-only and the
                                * slice-group map syntax that follows a non-zero value is
                                * not on the wire, so a stream using it is refused. */
   /*
    * Reference list sizes, and the same trap as max_num_ref_frames: the PPS fields
    * num_ref_idx_l{0,1}_default_active_minus1 exist on the wire and are DEAD on the decode
    * path. mesa fills the per-slice values into the picture descriptor instead
    * (picture_h264.c: `desc.h264.num_ref_idx_l0_active_minus1`, from the slice parameter
    * buffer). Emitting the PPS fields yields 0, so the decoder builds a one-entry list and
    * every P slice that indexes past it is bad data.
    *
    * Using the per-slice value as the PPS default is correct in both directions: a slice
    * that overrides it carries num_ref_idx_active_override_flag in its own header, which
    * passes through untouched and still wins.
    */
   ue(w, desc->num_ref_idx_l0_active_minus1);
   ue(w, desc->num_ref_idx_l1_active_minus1);
   flag(w, pps->weighted_pred_flag);
   u(w, 2, pps->weighted_bipred_idc);
   se(w, pps->pic_init_qp_minus26);
   se(w, pps->pic_init_qs_minus26);
   se(w, pps->chroma_qp_index_offset);
   flag(w, pps->deblocking_filter_control_present_flag);
   flag(w, pps->constrained_intra_pred_flag);
   flag(w, pps->redundant_pic_cnt_present_flag);

   /*
    * The optional tail. transform_8x8_mode_flag is a real decoding parameter and is on the
    * wire, so it must be emitted when set -- a High-profile stream that uses 8x8 transforms
    * decodes into mush without it. Reaching it requires writing the whole trailing group.
    */
   if (pps->transform_8x8_mode_flag || pps->second_chroma_qp_index_offset) {
      flag(w, pps->transform_8x8_mode_flag);
      flag(w, 0);              /* pic_scaling_matrix_present_flag -- see has_scaling_matrix */
      se(w, pps->second_chroma_qp_index_offset);
   }

   rbsp_trailing(w);
}

int virgl_h264_build_parameter_sets(const struct virgl_h264_picture_desc *desc,
                                    uint32_t width, uint32_t height,
                                    enum pipe_video_profile profile,
                                    unsigned pps_id,
                                    struct virgl_h264_parameter_sets *out)
{
   const uint8_t idc = profile_idc_of(profile);
   struct bs w;

   if (!desc || !out || !width || !height)
      return -1;

   /* Progressive 4:2:0 8-bit only in this cut; anything else is refused rather than
    * serialized approximately. separate_colour_plane_flag implies 4:4:4. */
   if (desc->field_pic_flag) {
      virgl_error("video: h264 field coding is not supported\n");
      return -1;
   }
   if (desc->pps.sps.chroma_format_idc != 1 ||
       desc->pps.sps.separate_colour_plane_flag) {
      virgl_error("video: h264 chroma_format_idc %u is not supported (4:2:0 only)\n",
                  desc->pps.sps.chroma_format_idc);
      return -1;
   }
   if (desc->pps.sps.bit_depth_luma_minus8 || desc->pps.sps.bit_depth_chroma_minus8) {
      virgl_error("video: h264 bit depth > 8 is not supported\n");
      return -1;
   }
   /* The height derivation below counts map units as whole-frame macroblock rows, which is
    * only true for frame_mbs_only_flag == 1. A field-capable stream would need the map
    * units halved and the crop scaled, so it is refused rather than mis-sized. */
   if (!desc->pps.sps.frame_mbs_only_flag) {
      virgl_error("video: h264 field-capable streams (frame_mbs_only_flag 0) are not supported\n");
      return -1;
   }
   if (desc->pps.num_slice_groups_minus1) {
      virgl_error("video: h264 slice groups (FMO) are not supported\n");
      return -1;
   }
   if (has_scaling_matrix(&desc->pps)) {
      virgl_error("video: h264 custom scaling matrices are not supported yet\n");
      return -1;
   }

   memset(out, 0, sizeof(*out));

   /* seq_parameter_set_id is 0 throughout: the PPS below is the only thing that refers to
    * it, and nothing in the guest's slice data does. */
   bs_init(&w, out->sps, sizeof(out->sps), true);
   write_sps(&w, desc, width, height, idc, 0);
   if (w.overflow)
      return -1;
   out->sps_len = w.pos;

   bs_init(&w, out->pps, sizeof(out->pps), true);
   write_pps(&w, desc, pps_id, 0);
   if (w.overflow)
      return -1;
   out->pps_len = w.pos;

   return 0;
}

/* ---------------------------------------------------------------- slice / framing */

/* Minimal RBSP reader: strips emulation-prevention bytes as it goes. */
struct br {
   const uint8_t *buf;
   size_t len;
   size_t pos;
   unsigned bit;
   unsigned zeros;
};

static int br_bit(struct br *r)
{
   if (r->pos >= r->len)
      return -1;

   /* An emulation-prevention 0x03 after two zero bytes is not part of the RBSP. */
   if (r->bit == 0 && r->zeros >= 2 && r->buf[r->pos] == 0x03) {
      r->pos++;
      r->zeros = 0;
      if (r->pos >= r->len)
         return -1;
   }

   int v = (r->buf[r->pos] >> (7 - r->bit)) & 1;
   if (++r->bit == 8) {
      r->zeros = r->buf[r->pos] ? 0 : r->zeros + 1;
      r->bit = 0;
      r->pos++;
   }
   return v;
}

static int br_ue(struct br *r, uint32_t *out)
{
   unsigned n = 0;
   int b;

   while ((b = br_bit(r)) == 0) {
      if (++n > 32)
         return -1;
   }
   if (b < 0)
      return -1;

   uint32_t v = 1;
   for (unsigned i = 0; i < n; i++) {
      b = br_bit(r);
      if (b < 0)
         return -1;
      v = (v << 1) | (uint32_t)b;
   }
   *out = v - 1;
   return 0;
}

/* Walk Annex-B start codes, calling back with each NAL's payload. */
static const uint8_t *next_start_code(const uint8_t *p, const uint8_t *end, unsigned *sc_len)
{
   for (; p + 2 < end; p++) {
      if (p[0] == 0 && p[1] == 0) {
         if (p[2] == 1) { *sc_len = 3; return p; }
         if (p + 3 < end && p[2] == 0 && p[3] == 1) { *sc_len = 4; return p; }
      }
   }
   return NULL;
}

int virgl_h264_slice_pps_id(const uint8_t *annexb, size_t len, unsigned *out_id)
{
   const uint8_t *p = annexb, *end = annexb + len;
   unsigned sc;

   if (!annexb || !out_id)
      return -1;

   while ((p = next_start_code(p, end, &sc)) != NULL) {
      const uint8_t *nal = p + sc;
      if (nal >= end)
         break;

      const unsigned type = nal[0] & 0x1f;
      /* 1 = non-IDR slice, 5 = IDR slice. Both carry the header we want; other types
       * (SEI, AUD, parameter sets) do not. */
      if (type == 1 || type == 5) {
         struct br r = { .buf = nal + 1, .len = (size_t)(end - nal - 1) };
         uint32_t first_mb, slice_type, pps_id;

         if (br_ue(&r, &first_mb) || br_ue(&r, &slice_type) || br_ue(&r, &pps_id))
            return -1;
         if (pps_id > 255)     /* 7.4.2.2: pic_parameter_set_id is 0..255 */
            return -1;

         *out_id = pps_id;
         return 0;
      }
      p = nal;
   }

   return -1;
}

ssize_t virgl_h264_annexb_to_avcc(const uint8_t *in, size_t in_len,
                                  uint8_t *out, size_t out_cap)
{
   const uint8_t *p = in, *end = in + in_len;
   size_t written = 0;
   unsigned sc;

   if (!in)
      return -1;

   p = next_start_code(p, end, &sc);
   if (!p)
      return -1;   /* not Annex-B at all; the caller must not guess */

   while (p) {
      const uint8_t *nal = p + sc;
      unsigned next_sc = 0;
      const uint8_t *next = next_start_code(nal, end, &next_sc);
      const size_t nal_len = (size_t)((next ? next : end) - nal);

      if (nal_len) {
         if (out) {
            if (written + 4 + nal_len > out_cap)
               return -1;
            out[written + 0] = (uint8_t)(nal_len >> 24);
            out[written + 1] = (uint8_t)(nal_len >> 16);
            out[written + 2] = (uint8_t)(nal_len >> 8);
            out[written + 3] = (uint8_t)(nal_len);
            memcpy(out + written + 4, nal, nal_len);
         }
         written += 4 + nal_len;
      }

      p = next;
      sc = next_sc;
   }

   return (ssize_t)written;
}
