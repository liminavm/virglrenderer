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
 * AV1 OBU synthesis from a parsed picture descriptor. See virgl_video_av1_obu.h for why
 * this exists at all; in short, the frame header is destroyed at the guest's
 * decoder -> VA-API boundary and a bitstream-in decoder needs it back.
 *
 * Section numbers below refer to the AV1 Bitstream & Decoding Process Specification v1.0.0
 * with Errata 1. The order of every field here is the order that document gives; there is
 * no latitude in it, because a single misplaced bit shifts everything after it.
 *
 * Where the descriptor cannot tell us something the syntax needs, this file picks the
 * choice that removes state rather than the one that reproduces the original stream most
 * closely: we are not trying to reconstruct the encoder's bits, only to produce *a*
 * conformant stream that decodes to the same pixels. So frame ids are switched off,
 * inherit flags are forced on so values can be written outright, and the sequence header
 * is repeated in every temporal unit rather than tracked.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "virgl_video_av1_obu.h"
#include "virgl_video_hw.h"

#define AV1_OBU_SEQUENCE_HEADER   1
#define AV1_OBU_TEMPORAL_DELIMITER 2
#define AV1_OBU_FRAME_HEADER      3
#define AV1_OBU_TILE_GROUP        4
#define AV1_OBU_FRAME             6

#define AV1_FRAME_KEY        0
#define AV1_FRAME_INTER      1
#define AV1_FRAME_INTRA_ONLY 2
#define AV1_FRAME_SWITCH     3

#define AV1_PRIMARY_REF_NONE 7
#define AV1_SELECT_SCREEN_CONTENT_TOOLS 2
#define AV1_SELECT_INTEGER_MV 2

#define AV1_MAX_TILE_WIDTH  4096
#define AV1_MAX_TILE_AREA   (4096 * 2304)
#define AV1_MAX_TILE_COLS   64
#define AV1_MAX_TILE_ROWS   64

#define WARPEDMODEL_PREC_BITS    16
#define GM_ABS_TRANS_BITS        12
#define GM_ABS_TRANS_ONLY_BITS    9
#define GM_ABS_ALPHA_BITS        12
#define GM_ALPHA_PREC_BITS       15
#define GM_TRANS_PREC_BITS        6
#define GM_TRANS_ONLY_PREC_BITS   3

/* How many bytes each tile's size field takes. Ours to choose; four keeps any tile a
 * decoder will meet representable without a size survey. */
#define TILE_SIZE_BYTES 4

/* ---------------------------------------------------------------- bit writer */

struct bw {
   uint8_t *buf;      /* NULL to size only */
   size_t cap;
   size_t pos;        /* bytes written */
   uint32_t acc;      /* partial byte, MSB-first */
   unsigned nbits;    /* bits held in acc */
   bool overflow;
};

static void bw_init(struct bw *w, uint8_t *buf, size_t cap)
{
   memset(w, 0, sizeof(*w));
   w->buf = buf;
   w->cap = cap;
}

static void bw_byte(struct bw *w, uint8_t b)
{
   if (w->buf) {
      if (w->pos >= w->cap) {
         w->overflow = true;
         return;
      }
      w->buf[w->pos] = b;
   }
   w->pos++;
}

/* f(n) in the spec: n bits, most significant first. */
static void f(struct bw *w, unsigned n, uint32_t v)
{
   for (unsigned i = 0; i < n; i++) {
      w->acc = (w->acc << 1) | ((v >> (n - 1 - i)) & 1);
      if (++w->nbits == 8) {
         bw_byte(w, (uint8_t)w->acc);
         w->acc = 0;
         w->nbits = 0;
      }
   }
}

static void flag(struct bw *w, bool v) { f(w, 1, v ? 1 : 0); }

/* su(n): n bits total, two's complement. */
static void su(struct bw *w, unsigned n, int32_t v)
{
   f(w, n, (uint32_t)v & ((n >= 32) ? 0xffffffffu : ((1u << n) - 1)));
}

/* ns(n): non-symmetric unsigned, for values in [0, n). */
static unsigned floor_log2(uint32_t x)
{
   unsigned s = 0;
   while (x >>= 1)
      s++;
   return s;
}

static void ns(struct bw *w, uint32_t n, uint32_t v)
{
   unsigned width;
   uint32_t m;

   if (n <= 1)
      return;
   width = floor_log2(n) + 1;
   m = (1u << width) - n;
   if (v < m) {
      f(w, width - 1, v);
   } else {
      uint32_t val = v + m;
      f(w, width - 1, val >> 1);
      f(w, 1, val & 1);
   }
}

/* increment(low, high): unary, terminated by a zero unless the ceiling is reached. */
static void increment(struct bw *w, uint32_t low, uint32_t high, uint32_t v)
{
   for (uint32_t i = low; i < v; i++)
      f(w, 1, 1);
   if (v < high)
      f(w, 1, 0);
}

static void leb128(struct bw *w, uint64_t v)
{
   do {
      uint8_t b = v & 0x7f;
      v >>= 7;
      if (v)
         b |= 0x80;
      f(w, 8, b);
   } while (v);
}

static void le(struct bw *w, unsigned nbytes, uint64_t v)
{
   for (unsigned i = 0; i < nbytes; i++)
      f(w, 8, (v >> (i * 8)) & 0xff);
}

/* trailing_bits: a one, then zeroes to the byte boundary. */
static void trailing_bits(struct bw *w)
{
   f(w, 1, 1);
   while (w->nbits)
      f(w, 1, 0);
}

static void byte_alignment(struct bw *w)
{
   while (w->nbits)
      f(w, 1, 0);
}

/* ------------------------------------------------------------- subexp coding */

static uint32_t recenter(uint32_t r, uint32_t v)
{
   if (v > 2 * r)
      return v;
   if (v >= r)
      return (v - r) * 2;
   return (r - v) * 2 - 1;
}

/* 5.9.27, written rather than read. Values in [mk, mk+a) terminate with a zero bit and a
 * b2-bit remainder; anything larger costs a one bit and moves the window up. */
static void write_subexp(struct bw *w, uint32_t v, uint32_t num_syms)
{
   uint32_t i = 0, mk = 0;
   const uint32_t k = 3;

   while (1) {
      uint32_t b2 = i ? k + i - 1 : k;
      uint32_t a = 1u << b2;

      if (num_syms <= mk + 3 * a) {
         ns(w, num_syms - mk, v - mk);
         return;
      }
      if (v >= mk + a) {
         f(w, 1, 1);
         i++;
         mk += a;
      } else {
         f(w, 1, 0);
         f(w, b2, v - mk);
         return;
      }
   }
}

static void write_unsigned_subexp_with_ref(struct bw *w, uint32_t v, uint32_t mx, uint32_t r)
{
   if ((r << 1) <= mx)
      write_subexp(w, recenter(r, v), mx);
   else
      write_subexp(w, recenter(mx - 1 - r, mx - 1 - v), mx);
}

static void write_signed_subexp_with_ref(struct bw *w, int32_t v, int32_t low, int32_t high,
                                         int32_t r)
{
   write_unsigned_subexp_with_ref(w, (uint32_t)(v - low), (uint32_t)(high - low),
                                  (uint32_t)(r - low));
}

/* ------------------------------------------------------------------ helpers */

static int tile_log2(int blk_size, int target)
{
   int k;

   for (k = 0; (blk_size << k) < target; k++)
      ;
   return k;
}

static int get_relative_dist(int order_hint_bits, int a, int b)
{
   int diff, m;

   if (!order_hint_bits)
      return 0;
   diff = a - b;
   m = 1 << (order_hint_bits - 1);
   return (diff & (m - 1)) - (diff & m);
}

struct seq_params {
   uint8_t profile;
   uint8_t level_idx;
   uint8_t order_hint_bits;      /* 0 when order hints are off */
   uint8_t bit_depth;            /* 8, 10 or 12 */
   bool mono_chrome;
   bool use_128x128_superblock;
   bool enable_order_hint;
   bool enable_ref_frame_mvs;
   bool film_grain_params_present;
   /* These gate symbol decoding inside the tiles, not just the header, so they have to be
    * what the encoder used -- a permissive 1 here silently corrupts every tile. The
    * descriptor carries all of them. */
   bool enable_filter_intra;
   bool enable_intra_edge_filter;
   bool enable_interintra_compound;
   bool enable_masked_compound;
   bool enable_dual_filter;
   bool enable_jnt_comp;
   bool enable_cdef;
   uint16_t max_width, max_height;
   uint8_t width_bits, height_bits;   /* minus 1, as coded */
};

/* The level is not in the descriptor and does not affect decoding, only conformance
 * signalling. Pick the lowest one whose limits cover the frame, so a decoder that does
 * enforce them is not handed something it must reject. Levels are (major-2)*4+minor. */
static uint8_t pick_level(uint32_t w, uint32_t h)
{
   const struct { uint32_t pix; uint32_t w, h; uint8_t idx; } levels[] = {
      {   147456,  2048, 1152,  0 },  /* 2.0 */
      {   278784,  2816, 1584,  1 },  /* 2.1 */
      {   665856,  4352, 2448,  4 },  /* 3.0 */
      {  1065024,  5504, 3096,  5 },  /* 3.1 */
      {  2359296,  6144, 3456,  8 },  /* 4.0 */
      {  2359296,  6144, 3456,  9 },  /* 4.1 */
      {  8912896,  8192, 4352, 12 },  /* 5.0 */
      { 35651584, 16384, 8704, 16 },  /* 6.0 */
   };

   for (size_t i = 0; i < sizeof(levels) / sizeof(levels[0]); i++)
      if (w <= levels[i].w && h <= levels[i].h && (uint64_t)w * h <= levels[i].pix)
         return levels[i].idx;
   return 19; /* 6.3 — beyond this nothing we can decode exists anyway */
}

static void derive_seq(const struct virgl_av1_picture_desc *d, struct seq_params *s)
{
   const uint16_t maxw = d->picture_parameter.max_width
                       ? d->picture_parameter.max_width : d->picture_parameter.frame_width;
   const uint16_t maxh = d->picture_parameter.max_height
                       ? d->picture_parameter.max_height : d->picture_parameter.frame_height;

   memset(s, 0, sizeof(*s));
   s->profile = d->picture_parameter.profile;
   s->mono_chrome = d->picture_parameter.seq_info_fields.mono_chrome;
   s->use_128x128_superblock = d->picture_parameter.seq_info_fields.use_128x128_superblock;
   s->enable_order_hint = d->picture_parameter.seq_info_fields.enable_order_hint;
   s->enable_ref_frame_mvs = d->picture_parameter.seq_info_fields.ref_frame_mvs;
   s->film_grain_params_present =
      d->picture_parameter.seq_info_fields.film_grain_params_present;
   s->order_hint_bits = s->enable_order_hint
                      ? d->picture_parameter.order_hint_bits_minus_1 + 1 : 0;
   s->enable_filter_intra = d->picture_parameter.seq_info_fields.enable_filter_intra;
   s->enable_intra_edge_filter =
      d->picture_parameter.seq_info_fields.enable_intra_edge_filter;
   s->enable_interintra_compound =
      d->picture_parameter.seq_info_fields.enable_interintra_compound;
   s->enable_masked_compound = d->picture_parameter.seq_info_fields.enable_masked_compound;
   s->enable_dual_filter = d->picture_parameter.seq_info_fields.enable_dual_filter;
   s->enable_jnt_comp = d->picture_parameter.seq_info_fields.enable_jnt_comp;
   s->enable_cdef = d->picture_parameter.seq_info_fields.enable_cdef;

   switch (d->picture_parameter.bit_depth_idx) {
   case 1:  s->bit_depth = 10; break;
   case 2:  s->bit_depth = 12; break;
   default: s->bit_depth = 8;  break;
   }

   s->max_width = maxw;
   s->max_height = maxh;
   s->width_bits = (uint8_t)floor_log2(maxw ? maxw - 1 : 1) + 1;
   s->height_bits = (uint8_t)floor_log2(maxh ? maxh - 1 : 1) + 1;
   if (s->width_bits < 1) s->width_bits = 1;
   if (s->height_bits < 1) s->height_bits = 1;
   s->level_idx = pick_level(maxw, maxh);
}

/* 5.5.2 color_config. The descriptor carries matrix_coefficients but not primaries or
 * transfer, and none of the three changes a decoded sample -- they are display metadata --
 * so the description is simply marked absent, which defaults all three to unspecified.
 * separate_uv_delta_q is set because the descriptor really does carry independent U and V
 * quantizer deltas, and there would otherwise be no way to express them. */
static void write_color_config(struct bw *w, const struct seq_params *s)
{
   const bool high_bitdepth = s->bit_depth > 8;

   flag(w, high_bitdepth);
   if (s->profile == 2 && high_bitdepth)
      flag(w, s->bit_depth == 12);          /* twelve_bit */
   if (s->profile != 1)
      flag(w, s->mono_chrome);
   flag(w, false);                          /* color_description_present_flag */
   if (s->mono_chrome) {
      flag(w, false);                       /* color_range */
      return;
   }
   /* Profile 0 is 4:2:0 and both subsampling flags are inferred to 1. */
   flag(w, false);                          /* color_range: studio swing */
   if (s->profile == 0) {
      f(w, 2, 0);                           /* chroma_sample_position: unknown */
   } else if (s->profile == 1) {
      f(w, 2, 0);
   } else {
      f(w, 2, 0);
   }
   flag(w, true);                           /* separate_uv_delta_q */
}

/* 5.5.1 sequence_header_obu, payload only. */
static void write_sequence_header(struct bw *w, const struct seq_params *s)
{
   f(w, 3, s->profile);
   flag(w, false);                 /* still_picture */
   flag(w, false);                 /* reduced_still_picture_header */
   flag(w, false);                 /* timing_info_present_flag */
   flag(w, false);                 /* initial_display_delay_present_flag */
   f(w, 5, 0);                     /* operating_points_cnt_minus_1 */
   f(w, 12, 0);                    /* operating_point_idc[0] */
   f(w, 5, s->level_idx);
   if (s->level_idx > 7)
      flag(w, false);              /* seq_tier[0] */

   f(w, 4, s->width_bits - 1);
   f(w, 4, s->height_bits - 1);
   f(w, s->width_bits, s->max_width ? s->max_width - 1u : 0u);
   f(w, s->height_bits, s->max_height ? s->max_height - 1u : 0u);

   flag(w, false);                 /* frame_id_numbers_present_flag */
   flag(w, s->use_128x128_superblock);
   flag(w, s->enable_filter_intra);
   flag(w, s->enable_intra_edge_filter);

   flag(w, s->enable_interintra_compound);
   flag(w, s->enable_masked_compound);
   /* Not in the descriptor, and safe to enable: it only decides whether the frame header
    * carries allow_warped_motion, which we then write from the descriptor. Reader and
    * writer stay in step either way. */
   flag(w, true);                  /* enable_warped_motion */
   flag(w, s->enable_dual_filter);
   flag(w, s->enable_order_hint);
   if (s->enable_order_hint) {
      flag(w, s->enable_jnt_comp);
      flag(w, s->enable_ref_frame_mvs);
   }
   /* SELECT for both, so each frame states its own choice and nothing is inherited from
    * a sequence-level decision the descriptor does not record. */
   flag(w, true);                  /* seq_choose_screen_content_tools */
   flag(w, true);                  /* seq_choose_integer_mv */
   if (s->enable_order_hint)
      f(w, 3, s->order_hint_bits - 1);

   /* enable_superres and enable_restoration are likewise absent from the descriptor and
    * only gate header fields we then write from it. enable_cdef is carried, and matters. */
   flag(w, true);                  /* enable_superres */
   flag(w, s->enable_cdef);
   flag(w, true);                  /* enable_restoration */

   write_color_config(w, s);

   flag(w, s->film_grain_params_present);
}

/* --------------------------------------------------------------- frame header */

struct frame_ctx {
   const struct virgl_av1_picture_desc *d;
   const struct seq_params *s;
   bool frame_is_intra;
   bool coded_lossless;
   bool all_lossless;
   unsigned num_planes;
   uint32_t upscaled_width;   /* what the frame header codes */
   uint32_t frame_width;      /* after superres downscale */
   uint32_t frame_height;
   uint8_t frame_type;
   bool show_frame;
   bool error_resilient;
   uint8_t primary_ref_frame;
   uint32_t tile_cols, tile_rows;
   uint32_t tile_cols_log2, tile_rows_log2;
   uint8_t refresh;                                   /* ours, not the guest's */
   uint8_t our_ref_idx[VIRGL_AV1_REFS_PER_FRAME];     /* refs in our slot numbering */
};

/* 7.12.2. Lossless has to be derived because it gates whether the loop filter, CDEF and
 * loop restoration sections appear in the bitstream at all -- not merely what they say. */
static void derive_lossless(struct frame_ctx *c)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;

   c->coded_lossless = true;
   for (int i = 0; i < 8; i++) {
      int qindex = p->base_qindex;

      /* Segment feature 0 is SEG_LVL_ALT_Q. */
      if (p->seg_info.segment_info_fields.enabled && (p->seg_info.feature_mask[i] & 1))
         qindex += p->seg_info.feature_data[i][0];
      if (qindex < 0)
         qindex = 0;
      else if (qindex > 255)
         qindex = 255;

      if (qindex || p->y_dc_delta_q || p->u_ac_delta_q || p->u_dc_delta_q ||
          p->v_ac_delta_q || p->v_dc_delta_q) {
         c->coded_lossless = false;
         break;
      }
   }
   c->all_lossless = c->coded_lossless && c->frame_width == c->upscaled_width;
}

static void write_superres_params(struct bw *w, struct frame_ctx *c)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;
   uint32_t denom = 8; /* SUPERRES_NUM */

   flag(w, p->pic_info_fields.use_superres);
   if (p->pic_info_fields.use_superres) {
      denom = p->superres_scale_denominator;
      if (denom < 9)
         denom = 9;
      f(w, 3, denom - 9); /* coded_denom, SUPERRES_DENOM_MIN */
   }
   c->frame_width = (c->upscaled_width * 8 + (denom / 2)) / denom;
}

static void write_frame_size(struct bw *w, struct frame_ctx *c, bool size_override)
{
   if (size_override) {
      f(w, c->s->width_bits, c->upscaled_width - 1);
      f(w, c->s->height_bits, c->frame_height - 1);
   }
   write_superres_params(w, c);
}

static void write_render_size(struct bw *w)
{
   flag(w, false); /* render_and_frame_size_different */
}

/* 5.9.15. The descriptor hands us explicit tile boundaries, so the non-uniform form is
 * always expressible; the uniform form is used when the guest says the spacing was uniform,
 * because it is both shorter and what the original stream said. */
static void write_tile_info(struct bw *w, struct frame_ctx *c)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;
   const int mi_cols = 2 * ((c->frame_width + 7) >> 3);
   const int mi_rows = 2 * ((c->frame_height + 7) >> 3);
   const int sb_shift = c->s->use_128x128_superblock ? 5 : 4;
   const int sb_size = sb_shift + 2;
   const int sb_cols = c->s->use_128x128_superblock ? ((mi_cols + 31) >> 5)
                                                    : ((mi_cols + 15) >> 4);
   const int sb_rows = c->s->use_128x128_superblock ? ((mi_rows + 31) >> 5)
                                                    : ((mi_rows + 15) >> 4);
   const int max_tile_width_sb = AV1_MAX_TILE_WIDTH >> sb_size;
   int max_tile_area_sb = AV1_MAX_TILE_AREA >> (2 * sb_size);
   const int min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
   const int max_log2_tile_cols =
      tile_log2(1, sb_cols < AV1_MAX_TILE_COLS ? sb_cols : AV1_MAX_TILE_COLS);
   const int max_log2_tile_rows =
      tile_log2(1, sb_rows < AV1_MAX_TILE_ROWS ? sb_rows : AV1_MAX_TILE_ROWS);
   const int min_log2_tiles_area = tile_log2(max_tile_area_sb, sb_rows * sb_cols);
   const int min_log2_tiles = min_log2_tile_cols > min_log2_tiles_area
                            ? min_log2_tile_cols : min_log2_tiles_area;
   const bool uniform = p->pic_info_fields.uniform_tile_spacing_flag;

   c->tile_cols = p->tile_cols ? p->tile_cols : 1;
   c->tile_rows = p->tile_rows ? p->tile_rows : 1;

   flag(w, uniform);

   if (uniform) {
      int cols_log2 = tile_log2(1, (int)c->tile_cols);
      int rows_log2, min_log2_tile_rows, tile_width_sb, tile_height_sb;

      if (cols_log2 < min_log2_tile_cols)
         cols_log2 = min_log2_tile_cols;
      if (cols_log2 > max_log2_tile_cols)
         cols_log2 = max_log2_tile_cols;
      increment(w, min_log2_tile_cols, max_log2_tile_cols, cols_log2);

      tile_width_sb = (sb_cols + (1 << cols_log2) - 1) >> cols_log2;
      c->tile_cols = (sb_cols + tile_width_sb - 1) / tile_width_sb;
      c->tile_cols_log2 = cols_log2;

      min_log2_tile_rows = min_log2_tiles - cols_log2;
      if (min_log2_tile_rows < 0)
         min_log2_tile_rows = 0;
      rows_log2 = tile_log2(1, (int)c->tile_rows);
      if (rows_log2 < min_log2_tile_rows)
         rows_log2 = min_log2_tile_rows;
      if (rows_log2 > max_log2_tile_rows)
         rows_log2 = max_log2_tile_rows;
      increment(w, min_log2_tile_rows, max_log2_tile_rows, rows_log2);

      tile_height_sb = (sb_rows + (1 << rows_log2) - 1) >> rows_log2;
      c->tile_rows = (sb_rows + tile_height_sb - 1) / tile_height_sb;
      c->tile_rows_log2 = rows_log2;
   } else {
      int start_sb = 0, widest = 0, i;
      int max_tile_height_sb;

      for (i = 0; start_sb < sb_cols && i < AV1_MAX_TILE_COLS; i++) {
         int remaining = sb_cols - start_sb;
         int max_width = remaining < max_tile_width_sb ? remaining : max_tile_width_sb;
         int size_sb = (i < (int)c->tile_cols && p->width_in_sbs[i])
                     ? p->width_in_sbs[i] : max_width;

         if (size_sb > max_width)
            size_sb = max_width;
         ns(w, (uint32_t)max_width, (uint32_t)(size_sb - 1));
         if (size_sb > widest)
            widest = size_sb;
         start_sb += size_sb;
      }
      c->tile_cols = i;
      c->tile_cols_log2 = tile_log2(1, i);

      if (min_log2_tiles > 0)
         max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
      else
         max_tile_area_sb = sb_rows * sb_cols;
      max_tile_height_sb = widest ? max_tile_area_sb / widest : 1;
      if (max_tile_height_sb < 1)
         max_tile_height_sb = 1;

      start_sb = 0;
      for (i = 0; start_sb < sb_rows && i < AV1_MAX_TILE_ROWS; i++) {
         int remaining = sb_rows - start_sb;
         int max_height = remaining < max_tile_height_sb ? remaining : max_tile_height_sb;
         int size_sb = (i < (int)c->tile_rows && p->height_in_sbs[i])
                     ? p->height_in_sbs[i] : max_height;

         if (size_sb > max_height)
            size_sb = max_height;
         ns(w, (uint32_t)max_height, (uint32_t)(size_sb - 1));
         start_sb += size_sb;
      }
      c->tile_rows = i;
      c->tile_rows_log2 = tile_log2(1, i);
   }

   if (c->tile_cols_log2 > 0 || c->tile_rows_log2 > 0) {
      f(w, c->tile_cols_log2 + c->tile_rows_log2, p->context_update_tile_id);
      f(w, 2, TILE_SIZE_BYTES - 1);
   }
}

static void write_delta_q(struct bw *w, int8_t v)
{
   flag(w, v != 0);
   if (v)
      su(w, 1 + 6, v);
}

static void write_quantization_params(struct bw *w, struct frame_ctx *c)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;

   f(w, 8, p->base_qindex);
   write_delta_q(w, p->y_dc_delta_q);

   if (c->num_planes > 1) {
      /* separate_uv_delta_q was set in the sequence header, so diff_uv_delta is present. */
      const bool diff_uv = p->u_dc_delta_q != p->v_dc_delta_q ||
                           p->u_ac_delta_q != p->v_ac_delta_q;

      flag(w, diff_uv);
      write_delta_q(w, p->u_dc_delta_q);
      write_delta_q(w, p->u_ac_delta_q);
      if (diff_uv) {
         write_delta_q(w, p->v_dc_delta_q);
         write_delta_q(w, p->v_ac_delta_q);
      }
   }

   flag(w, p->qmatrix_fields.using_qmatrix);
   if (p->qmatrix_fields.using_qmatrix) {
      f(w, 4, p->qmatrix_fields.qm_y);
      f(w, 4, p->qmatrix_fields.qm_u);
      f(w, 4, p->qmatrix_fields.qm_v);  /* separate_uv_delta_q is on */
   }
}

/* 5.9.14. Feature values are written outright rather than inherited: update_data is forced
 * on, which is always legal, and removes any need to know what the reference held. */
static void write_segmentation_params(struct bw *w, struct frame_ctx *c)
{
   static const uint8_t bits[8] = { 8, 6, 6, 6, 6, 3, 0, 0 };
   static const uint8_t is_signed[8] = { 1, 1, 1, 1, 1, 0, 0, 0 };
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;
   const bool enabled = p->seg_info.segment_info_fields.enabled;

   flag(w, enabled);
   if (!enabled)
      return;

   if (c->primary_ref_frame != AV1_PRIMARY_REF_NONE) {
      flag(w, p->seg_info.segment_info_fields.update_map);
      if (p->seg_info.segment_info_fields.update_map)
         flag(w, p->seg_info.segment_info_fields.temporal_update);
      flag(w, true);   /* update_data */
   }

   for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
         const bool on = (p->seg_info.feature_mask[i] >> j) & 1;

         flag(w, on);
         if (on && bits[j]) {
            if (is_signed[j])
               su(w, 1 + bits[j], p->seg_info.feature_data[i][j]);
            else
               f(w, bits[j], (uint32_t)p->seg_info.feature_data[i][j]);
         }
      }
   }
}

static void write_loop_filter_params(struct bw *w, struct frame_ctx *c)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;

   if (c->coded_lossless || p->pic_info_fields.allow_intrabc)
      return;

   f(w, 6, p->filter_level[0]);
   f(w, 6, p->filter_level[1]);
   if (c->num_planes > 1 && (p->filter_level[0] || p->filter_level[1])) {
      f(w, 6, p->filter_level_u);
      f(w, 6, p->filter_level_v);
   }
   f(w, 3, p->loop_filter_info_fields.sharpness_level);

   flag(w, p->loop_filter_info_fields.mode_ref_delta_enabled);
   if (!p->loop_filter_info_fields.mode_ref_delta_enabled)
      return;

   /* Force the update on and write every delta. The descriptor carries resolved values but
    * no per-index update flags, and writing them all is both legal and independent of what
    * the reference frame held. */
   flag(w, true);   /* loop_filter_delta_update */
   for (int i = 0; i < 8; i++) {
      flag(w, true);
      su(w, 1 + 6, p->ref_deltas[i]);
   }
   for (int i = 0; i < 2; i++) {
      flag(w, true);
      su(w, 1 + 6, p->mode_deltas[i]);
   }
}

static void write_cdef_params(struct bw *w, struct frame_ctx *c)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;

   if (c->coded_lossless || p->pic_info_fields.allow_intrabc || !c->s->enable_cdef)
      return;

   f(w, 2, p->cdef_damping_minus_3);
   f(w, 2, p->cdef_bits);
   for (int i = 0; i < (1 << p->cdef_bits); i++) {
      /* VA packs each strength as (primary << 2) | secondary. */
      f(w, 4, p->cdef_y_strengths[i] >> 2);
      f(w, 2, p->cdef_y_strengths[i] & 3);
      if (c->num_planes > 1) {
         f(w, 4, p->cdef_uv_strengths[i] >> 2);
         f(w, 2, p->cdef_uv_strengths[i] & 3);
      }
   }
}

/* The bitstream's lr_type is not the restoration-type enum: it is remapped through
 * Remap_Lr_Type = { NONE, SWITCHABLE, WIENER, SGRPROJ }, while the descriptor carries the
 * enum itself. Inverting that is the whole of this table, and getting it wrong silently
 * swaps two filters. */
static uint32_t lr_type_to_coded(uint32_t type)
{
   switch (type) {
   case 0:  return 0;   /* NONE */
   case 1:  return 2;   /* WIENER */
   case 2:  return 3;   /* SGRPROJ */
   default: return 1;   /* SWITCHABLE */
   }
}

static void write_lr_params(struct bw *w, struct frame_ctx *c)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;
   uint32_t types[3];
   bool uses_lr = false, uses_chroma_lr = false;

   if (c->all_lossless || p->pic_info_fields.allow_intrabc)
      return;

   types[0] = p->loop_restoration_fields.yframe_restoration_type;
   types[1] = p->loop_restoration_fields.cbframe_restoration_type;
   types[2] = p->loop_restoration_fields.crframe_restoration_type;

   for (unsigned i = 0; i < c->num_planes; i++) {
      f(w, 2, lr_type_to_coded(types[i]));
      if (types[i]) {
         uses_lr = true;
         if (i > 0)
            uses_chroma_lr = true;
      }
   }

   if (uses_lr) {
      if (c->s->use_128x128_superblock)
         increment(w, 1, 2, p->loop_restoration_fields.lr_unit_shift);
      else
         increment(w, 0, 2, p->loop_restoration_fields.lr_unit_shift);
      /* Profile 0 is 4:2:0, so both subsampling flags are set. */
      if (!c->s->mono_chrome && uses_chroma_lr)
         f(w, 1, p->loop_restoration_fields.lr_uv_shift);
   }
}

/* One warp parameter, 5.9.24 global_param, written rather than read. */
static void write_gm_param(struct bw *w, struct frame_ctx *c, const int32_t *prev,
                           const int32_t *wmmat, uint32_t type, int idx)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;
   unsigned abs_bits, prec_bits, prec_diff;
   int32_t round, sub, mx, r, v;

   if (idx < 2) {
      if (type == 1) {   /* TRANSLATION */
         abs_bits  = GM_ABS_TRANS_ONLY_BITS - !p->pic_info_fields.allow_high_precision_mv;
         prec_bits = GM_TRANS_ONLY_PREC_BITS - !p->pic_info_fields.allow_high_precision_mv;
      } else {
         abs_bits  = GM_ABS_TRANS_BITS;
         prec_bits = GM_TRANS_PREC_BITS;
      }
   } else {
      abs_bits  = GM_ABS_ALPHA_BITS;
      prec_bits = GM_ALPHA_PREC_BITS;
   }

   prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
   round = ((idx % 3) == 2) ? (1 << WARPEDMODEL_PREC_BITS) : 0;
   sub   = ((idx % 3) == 2) ? (1 << prec_bits) : 0;
   mx    = 1 << abs_bits;

   r = (prev[idx] >> prec_diff) - sub;
   v = ((wmmat[idx] - round) >> prec_diff) - sub;

   /* The syntax cannot express anything outside [-mx, mx]; a descriptor that somehow
    * carries more would otherwise write a symbol the decoder cannot read back. */
   if (v < -mx) v = -mx;
   if (v >  mx) v =  mx;
   if (r < -mx) r = -mx;
   if (r >  mx) r =  mx;

   write_signed_subexp_with_ref(w, v, -mx, mx + 1, r);
}

/* 5.9.24. This is the only part of the frame header coded *relative* to a reference, which
 * is why the shadow state in struct virgl_av1_obu_state exists at all. Note the emission
 * order: the two-by-two block (indices 2..5) precedes the translation pair (0,1). */
static void write_global_motion_params(struct bw *w, struct frame_ctx *c,
                                       struct virgl_av1_obu_state *state)
{
   static const int32_t default_warp[VIRGL_AV1_WARP_PARAMS] = {
      0, 0, 1 << WARPEDMODEL_PREC_BITS, 0, 0, 1 << WARPEDMODEL_PREC_BITS
   };
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;

   if (c->frame_is_intra)
      return;

   for (int ref = 0; ref < VIRGL_AV1_REFS_PER_FRAME; ref++) {
      const uint32_t type = p->wm[ref].wmtype;
      const int32_t *prev;

      if (c->primary_ref_frame == AV1_PRIMARY_REF_NONE)
         prev = default_warp;
      else
         prev = state->saved_gm[c->our_ref_idx[c->primary_ref_frame] & 7][ref];

      flag(w, type != 0);                 /* is_global */
      if (type != 0) {
         flag(w, type == 2);              /* is_rot_zoom */
         if (type != 2)
            flag(w, type == 1);           /* is_translation */
      }

      if (type >= 2) {
         write_gm_param(w, c, prev, p->wm[ref].wmmat, type, 2);
         write_gm_param(w, c, prev, p->wm[ref].wmmat, type, 3);
         if (type == 3) {
            write_gm_param(w, c, prev, p->wm[ref].wmmat, type, 4);
            write_gm_param(w, c, prev, p->wm[ref].wmmat, type, 5);
         }
      }
      if (type >= 1) {
         write_gm_param(w, c, prev, p->wm[ref].wmmat, type, 0);
         write_gm_param(w, c, prev, p->wm[ref].wmmat, type, 1);
      }
   }
}

/* 5.9.30. update_grain is forced on and every parameter written outright: the descriptor
 * carries resolved grain parameters but neither update_grain nor the reference index it
 * would otherwise point at, so inheriting is not expressible while writing always is. */
static void write_film_grain_params(struct bw *w, struct frame_ctx *c)
{
   const typeof(c->d->picture_parameter.film_grain_info) *g =
      &c->d->picture_parameter.film_grain_info;
   const bool showable = c->d->picture_parameter.pic_info_fields.showable_frame;
   unsigned num_pos_luma, num_pos_chroma;

   if (!c->s->film_grain_params_present || (!c->show_frame && !showable))
      return;

   flag(w, g->film_grain_info_fields.apply_grain);
   if (!g->film_grain_info_fields.apply_grain)
      return;

   f(w, 16, g->grain_seed);
   if (c->frame_type == AV1_FRAME_INTER)
      flag(w, true);                       /* update_grain */

   f(w, 4, g->num_y_points);
   for (unsigned i = 0; i < g->num_y_points; i++) {
      f(w, 8, g->point_y_value[i]);
      f(w, 8, g->point_y_scaling[i]);
   }

   bool chroma_from_luma = g->film_grain_info_fields.chroma_scaling_from_luma;
   if (!c->s->mono_chrome)
      flag(w, chroma_from_luma);
   else
      chroma_from_luma = false;

   unsigned num_cb = g->num_cb_points, num_cr = g->num_cr_points;
   /* Profile 0 is 4:2:0, so an achromatic luma-less frame codes no chroma points. */
   if (c->s->mono_chrome || chroma_from_luma || g->num_y_points == 0) {
      num_cb = num_cr = 0;
   } else {
      f(w, 4, num_cb);
      for (unsigned i = 0; i < num_cb; i++) {
         f(w, 8, g->point_cb_value[i]);
         f(w, 8, g->point_cb_scaling[i]);
      }
      f(w, 4, num_cr);
      for (unsigned i = 0; i < num_cr; i++) {
         f(w, 8, g->point_cr_value[i]);
         f(w, 8, g->point_cr_scaling[i]);
      }
   }

   f(w, 2, g->film_grain_info_fields.grain_scaling_minus_8);
   f(w, 2, g->film_grain_info_fields.ar_coeff_lag);

   num_pos_luma = 2 * g->film_grain_info_fields.ar_coeff_lag *
                      (g->film_grain_info_fields.ar_coeff_lag + 1);
   if (g->num_y_points) {
      num_pos_chroma = num_pos_luma + 1;
      for (unsigned i = 0; i < num_pos_luma; i++)
         f(w, 8, (uint32_t)(g->ar_coeffs_y[i] + 128));
   } else {
      num_pos_chroma = num_pos_luma;
   }

   if (chroma_from_luma || num_cb)
      for (unsigned i = 0; i < num_pos_chroma; i++)
         f(w, 8, (uint32_t)(g->ar_coeffs_cb[i] + 128));
   if (chroma_from_luma || num_cr)
      for (unsigned i = 0; i < num_pos_chroma; i++)
         f(w, 8, (uint32_t)(g->ar_coeffs_cr[i] + 128));

   f(w, 2, g->film_grain_info_fields.ar_coeff_shift_minus_6);
   f(w, 2, g->film_grain_info_fields.grain_scale_shift);

   if (num_cb) {
      f(w, 8, g->cb_mult);
      f(w, 8, g->cb_luma_mult);
      f(w, 9, g->cb_offset);
   }
   if (num_cr) {
      f(w, 8, g->cr_mult);
      f(w, 8, g->cr_luma_mult);
      f(w, 9, g->cr_offset);
   }

   flag(w, g->film_grain_info_fields.overlap_flag);
   flag(w, g->film_grain_info_fields.clip_to_restricted_range);
}

/* 5.9.11 skip_mode_params. Only the *presence* of the bit is at stake, but getting that
 * wrong shifts every bit after it, so the reference search is reproduced exactly. */
static bool skip_mode_allowed(struct frame_ctx *c, struct virgl_av1_obu_state *state)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;
   const int bits = c->s->order_hint_bits;
   int forward_idx = -1, backward_idx = -1;
   int forward_hint = 0, backward_hint = 0;

   if (c->frame_is_intra || !p->mode_control_fields.reference_select ||
       !c->s->enable_order_hint)
      return false;

   for (int i = 0; i < VIRGL_AV1_REFS_PER_FRAME; i++) {
      const int hint = state->saved_order_hint[c->our_ref_idx[i] & 7];
      const int dist = get_relative_dist(bits, hint, p->order_hint);

      if (dist < 0) {
         if (forward_idx < 0 || get_relative_dist(bits, hint, forward_hint) > 0) {
            forward_idx = i;
            forward_hint = hint;
         }
      } else if (dist > 0) {
         if (backward_idx < 0 || get_relative_dist(bits, hint, backward_hint) < 0) {
            backward_idx = i;
            backward_hint = hint;
         }
      }
   }

   if (forward_idx < 0)
      return false;
   if (backward_idx >= 0)
      return true;

   for (int i = 0; i < VIRGL_AV1_REFS_PER_FRAME; i++) {
      const int hint = state->saved_order_hint[c->our_ref_idx[i] & 7];

      if (get_relative_dist(bits, hint, forward_hint) < 0)
         return true;
   }
   return false;
}

/* 5.9.2 uncompressed_header. */
static void write_uncompressed_header(struct bw *w, struct frame_ctx *c,
                                      struct virgl_av1_obu_state *state)
{
   const typeof(c->d->picture_parameter) *p = &c->d->picture_parameter;
   const bool size_override = c->upscaled_width != c->s->max_width ||
                              c->frame_height != c->s->max_height;
   /* Inferred, not what the descriptor holds: a switch frame and a shown key frame refresh
    * every slot by definition, and the *inferred* value is what the reader tests. Using the
    * descriptor's raw field here made the ref_order_hint loop below fire on a key frame and
    * emit 56 bits nobody reads, which desynchronised the whole rest of the header. */
   const uint8_t refresh = c->refresh;
   const bool all_frames_refreshed = refresh == 0xff;

   flag(w, false);                       /* show_existing_frame */
   f(w, 2, c->frame_type);
   flag(w, c->show_frame);
   if (!c->show_frame)
      flag(w, p->pic_info_fields.showable_frame);

   if (!(c->frame_type == AV1_FRAME_SWITCH ||
         (c->frame_type == AV1_FRAME_KEY && c->show_frame)))
      flag(w, c->error_resilient);

   flag(w, p->pic_info_fields.disable_cdf_update);

   /* seq_choose_screen_content_tools was SELECT, so both are stated per frame. */
   flag(w, p->pic_info_fields.allow_screen_content_tools);
   if (p->pic_info_fields.allow_screen_content_tools)
      flag(w, p->pic_info_fields.force_integer_mv);

   /* frame_id_numbers_present_flag is off, so no current_frame_id. */

   if (c->frame_type != AV1_FRAME_SWITCH)
      flag(w, size_override);

   if (c->s->order_hint_bits)
      f(w, c->s->order_hint_bits, p->order_hint);

   if (!(c->frame_is_intra || c->error_resilient))
      f(w, 3, c->primary_ref_frame);

   if (!(c->frame_type == AV1_FRAME_SWITCH ||
         (c->frame_type == AV1_FRAME_KEY && c->show_frame)))
      f(w, 8, refresh);

   if ((!c->frame_is_intra || !all_frames_refreshed) && c->s->enable_order_hint &&
       c->error_resilient)
      for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++)
         f(w, c->s->order_hint_bits, state->saved_order_hint[i]);

   if (c->frame_is_intra) {
      write_frame_size(w, c, size_override);
      write_render_size(w);
      if (p->pic_info_fields.allow_screen_content_tools &&
          c->upscaled_width == c->frame_width)
         flag(w, p->pic_info_fields.allow_intrabc);
   } else {
      if (c->s->enable_order_hint)
         flag(w, false);                 /* frame_refs_short_signaling */

      for (int i = 0; i < VIRGL_AV1_REFS_PER_FRAME; i++)
         f(w, 3, c->our_ref_idx[i]);

      if (size_override && !c->error_resilient) {
         /* frame_size_with_refs: decline every reference-derived size and state it. */
         for (int i = 0; i < VIRGL_AV1_REFS_PER_FRAME; i++)
            flag(w, false);              /* found_ref */
         write_frame_size(w, c, size_override);
         write_render_size(w);
      } else {
         write_frame_size(w, c, size_override);
         write_render_size(w);
      }

      if (!p->pic_info_fields.force_integer_mv)
         flag(w, p->pic_info_fields.allow_high_precision_mv);

      /* interpolation_filter: 4 is SWITCHABLE in the descriptor's enum. */
      if (p->interp_filter == 4) {
         flag(w, true);
      } else {
         flag(w, false);
         f(w, 2, p->interp_filter);
      }

      flag(w, p->pic_info_fields.is_motion_mode_switchable);

      if (!(c->error_resilient || !c->s->enable_ref_frame_mvs))
         flag(w, p->pic_info_fields.use_ref_frame_mvs);
   }

   if (!p->pic_info_fields.disable_cdf_update)
      flag(w, p->pic_info_fields.disable_frame_end_update_cdf);

   write_tile_info(w, c);
   write_quantization_params(w, c);
   write_segmentation_params(w, c);

   /* delta_q_params / delta_lf_params */
   if (p->base_qindex > 0)
      flag(w, p->mode_control_fields.delta_q_present_flag);
   if (p->mode_control_fields.delta_q_present_flag) {
      f(w, 2, p->mode_control_fields.log2_delta_q_res);
      if (!p->pic_info_fields.allow_intrabc)
         flag(w, p->mode_control_fields.delta_lf_present_flag);
      if (p->mode_control_fields.delta_lf_present_flag) {
         f(w, 2, p->mode_control_fields.log2_delta_lf_res);
         flag(w, p->mode_control_fields.delta_lf_multi);
      }
   }

   derive_lossless(c);

   write_loop_filter_params(w, c);
   write_cdef_params(w, c);
   write_lr_params(w, c);

   /* read_tx_mode */
   if (!c->coded_lossless)
      increment(w, 1, 2, p->mode_control_fields.tx_mode);

   /* frame_reference_mode */
   if (!c->frame_is_intra)
      flag(w, p->mode_control_fields.reference_select);

   {
      const bool allowed = skip_mode_allowed(c, state);

      /* The guest can only have resolved skip_mode_present = 1 if its decoder judged the
       * mode allowed, so a disagreement here is ours and is fatal: the bit is either
       * written or not, and getting that wrong shifts everything after it. */
      if (getenv("LIMINA_AV1_SLOT_TRACE") && !allowed &&
          p->mode_control_fields.skip_mode_present)
         fprintf(stderr, "[AV1SKIP] order_hint=%u: we say skip mode is not allowed but the "
                 "guest resolved skip_mode_present=1\n", p->order_hint);

      if (allowed)
         flag(w, p->mode_control_fields.skip_mode_present);
   }

   if (!(c->frame_is_intra || c->error_resilient))
      flag(w, p->pic_info_fields.allow_warped_motion);

   flag(w, p->mode_control_fields.reduced_tx_set_used);

   write_global_motion_params(w, c, state);
   write_film_grain_params(w, c);
}

/* ------------------------------------------------------- OBU and unit assembly */

/* An OBU is a header byte, a leb128 payload length, then the payload. has_size_field is
 * always set: a decoder handed a temporal unit needs to walk it, and sizes are the only
 * framing that survives being concatenated. */
static void emit_obu(struct bw *out, unsigned type, const uint8_t *payload, size_t len)
{
   f(out, 1, 0);              /* obu_forbidden_bit */
   f(out, 4, type);
   f(out, 1, 0);              /* obu_extension_flag */
   f(out, 1, 1);              /* obu_has_size_field */
   f(out, 1, 0);              /* obu_reserved_1bit */
   leb128(out, len);
   for (size_t i = 0; i < len; i++)
      f(out, 8, payload[i]);
}

/* 5.11.1 tile_group_obu, with every tile in one group. Each tile but the last is preceded
 * by its size; the last takes whatever remains, which is why it needs none. */
static void write_tile_group(struct bw *w, const struct virgl_av1_picture_desc *d,
                             const struct frame_ctx *c,
                             const uint8_t *tiles, size_t tiles_size)
{
   const uint32_t num_tiles = c->tile_cols * c->tile_rows;
   const uint16_t count = d->slice_parameter.slice_count;

   if (num_tiles > 1)
      flag(w, false);         /* tile_start_and_end_present_flag */
   byte_alignment(w);

   for (uint16_t i = 0; i < count; i++) {
      const uint32_t off = d->slice_parameter.slice_data_offset[i];
      const uint32_t size = d->slice_parameter.slice_data_size[i];

      if (off > tiles_size || (size_t)off + size > tiles_size)
         continue;

      if (i + 1 < count)
         le(w, TILE_SIZE_BYTES, size - 1);   /* tile_size_minus_1 */
      for (uint32_t b = 0; b < size; b++)
         f(w, 8, tiles[off + b]);
   }
}

/* Learn, from this frame's ref[], where the *previous* frame's picture landed, and pick a
 * slot for this one. See the comment on virgl_av1_obu_state for why this is needed at all:
 * the guest never tells us refresh_frame_flags, so we assign our own slots and remap.
 *
 * Returns the mask to emit. Also fills our_ref_idx[] with this frame's references expressed
 * in our slot numbering. */
/*
 * refresh_frame_flags is not in the descriptor. VA-API does not carry it -- mesa writes a
 * constant 1 (src/gallium/frontends/va/picture_av1.c) -- because a VA driver never needs it:
 * the application hands it the whole reference map every  frame and manages the DPB itself. A
 * bitstream writer does need it, so slots are assigned here and ref_frame_idx remapped onto
 * them.
 *
 * The assignment must not merely be self-consistent, it must not evict a picture a later
 * frame still references. Once eight distinct pictures are live -- which happens in any
 * pyramid GOP -- storing a ninth means evicting one, and which one the guest chose is
 * information that exists only in the *next* frame's ref[]. Guessing costs a reference the
 * decoder needs; the frame header still parses field-for-field correctly and the loss
 * surfaces much later as an entropy-decode desync.
 *
 * So a hidden frame is held for one submission and emitted once the next descriptor reveals
 * the guest's choice exactly. Only hidden frames are held: nothing waits on their pixels
 * until a later show_existing, whereas holding a shown frame would stall a caller that reads
 * its surface back before submitting the next one. A shown frame is emitted at once into a
 * slot nothing live occupies, which is free of that risk -- in a pyramid GOP shown frames are
 * never stored at all, and a low-delay stream, where they are, never fills eight slots.
 */

/* The picture the frame between two reference maps produced: the surface `now` lists that
 * `before` did not. Exactly one decode separates them, so a surface that has appeared can
 * only be its output, and a guest that stored nothing leaves the map unchanged.
 *
 * Compared as a set difference against the previous map, not against our own slots and not
 * per-slot. Surface ids are recycled -- a capture cycles through a handful of them -- so a
 * freshly reused id reads as one we already hold and teaches us nothing, while a per-slot
 * diff misses a frame that lands in a slot whose contents we had seen elsewhere. The guest
 * never reuses an id still in its own live map, which is what makes the set difference
 * exact. */
static uint32_t new_surface(const uint32_t *before, const uint32_t *now)
{
   for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++) {
      const uint32_t h = now[i];
      bool seen = (h == 0);

      for (int k = 0; k < VIRGL_AV1_NUM_REF_FRAMES && !seen; k++)
         seen = before[k] == h;
      if (!seen)
         return h;
   }
   return 0;
}

/* Drop pictures the guest no longer lists. Nothing can reference them again, and leaving
 * them behind lets a recycled id match a slot holding a picture that is long gone. */
static void prune_slots(struct virgl_av1_obu_state *state, const uint32_t *live)
{
   for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++) {
      bool alive = false;

      if (!state->slot_surface[i])
         continue;
      for (int k = 0; k < VIRGL_AV1_NUM_REF_FRAMES; k++)
         if (live[k] == state->slot_surface[i])
            alive = true;
      if (!alive)
         state->slot_surface[i] = 0;
   }
}

/* A slot we may overwrite: one holding nothing, a second copy of a picture we keep elsewhere,
 * or one holding a picture the guest no longer lists. -1 when all eight are needed. */
static int dead_slot(const struct virgl_av1_obu_state *state, const uint32_t *live)
{
   for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++)
      if (!state->slot_surface[i])
         return i;
   for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++)
      for (int k = 0; k < i; k++)
         if (state->slot_surface[k] == state->slot_surface[i])
            return i;
   for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++) {
      bool alive = false;

      for (int k = 0; k < VIRGL_AV1_NUM_REF_FRAMES; k++)
         if (live[k] == state->slot_surface[i])
            alive = true;
      if (!alive)
         return i;
   }
   return -1;
}

/* Map the guest's ref_frame_idx, which indexes its reference map, onto our own slots. */
static void resolve_refs(const struct virgl_av1_obu_state *state,
                         const struct virgl_av1_picture_desc *d,
                         uint8_t our_ref_idx[VIRGL_AV1_REFS_PER_FRAME])
{
   const typeof(d->picture_parameter) *p = &d->picture_parameter;

   for (int j = 0; j < VIRGL_AV1_REFS_PER_FRAME; j++) {
      const uint32_t want = d->ref[p->ref_frame_idx[j] & 7];
      bool found = false;

      our_ref_idx[j] = 0;
      for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++)
         if (want && state->slot_surface[i] == want) {
            our_ref_idx[j] = (uint8_t)i;
            found = true;
            break;
         }
      if (!found && want && getenv("LIMINA_AV1_SLOT_TRACE"))
         fprintf(stderr, "[AV1SLOT] ref %d wants surface %u (guest slot %u): not in our DPB\n",
                 j, want, p->ref_frame_idx[j] & 7);
   }
}

static void init_frame_ctx(struct frame_ctx *c, const struct virgl_av1_picture_desc *d,
                           const struct seq_params *s)
{
   const typeof(d->picture_parameter) *p = &d->picture_parameter;

   memset(c, 0, sizeof(*c));
   c->d = d;
   c->s = s;
   c->frame_type = p->pic_info_fields.frame_type;
   c->show_frame = p->pic_info_fields.show_frame;
   c->frame_is_intra = c->frame_type == AV1_FRAME_KEY ||
                       c->frame_type == AV1_FRAME_INTRA_ONLY;
   c->num_planes = s->mono_chrome ? 1 : 3;
   c->upscaled_width = p->frame_width ? p->frame_width : s->max_width;
   c->frame_width = c->upscaled_width;
   c->frame_height = p->frame_height ? p->frame_height : s->max_height;

   /* error_resilient_mode is inferred, not coded, for switch frames and shown key frames;
    * the descriptor's copy is authoritative everywhere else. */
   if (c->frame_type == AV1_FRAME_SWITCH ||
       (c->frame_type == AV1_FRAME_KEY && c->show_frame))
      c->error_resilient = true;
   else
      c->error_resilient = p->pic_info_fields.error_resilient_mode;

   /* Likewise primary_ref_frame: intra and error-resilient frames inherit nothing. */
   if (c->frame_is_intra || c->error_resilient)
      c->primary_ref_frame = AV1_PRIMARY_REF_NONE;
   else
      c->primary_ref_frame = p->primary_ref_frame & 7;
}

/* The reference update of 7.20, restricted to what the *writer* has to remember. */
static void update_state(struct virgl_av1_obu_state *state,
                         const struct virgl_av1_picture_desc *d,
                         const struct frame_ctx *c)
{
   const typeof(d->picture_parameter) *p = &d->picture_parameter;
   const uint8_t refresh = c->refresh;

   for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++) {
      if (!(refresh & (1 << i)))
         continue;
      state->saved_order_hint[i] = p->order_hint;
      state->slot_valid |= (uint8_t)(1 << i);
      for (int ref = 0; ref < VIRGL_AV1_REFS_PER_FRAME; ref++)
         for (int j = 0; j < VIRGL_AV1_WARP_PARAMS; j++)
            state->saved_gm[i][ref][j] = p->wm[ref].wmmat[j];
   }
}

void virgl_av1_obu_state_init(struct virgl_av1_obu_state *state)
{
   memset(state, 0, sizeof(*state));
   for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++)
      for (int ref = 0; ref < VIRGL_AV1_REFS_PER_FRAME; ref++) {
         state->saved_gm[i][ref][2] = 1 << WARPEDMODEL_PREC_BITS;
         state->saved_gm[i][ref][5] = 1 << WARPEDMODEL_PREC_BITS;
      }
}

/* Scratch for one OBU payload. A frame header is a few hundred bytes at the very most;
 * the tile group is written straight into the caller's buffer instead. */
#define OBU_SCRATCH 4096

static ssize_t emit_frame(struct virgl_av1_obu_state *state,
                          const struct virgl_av1_picture_desc *desc,
                          const void *tiles, size_t tiles_size,
                          uint8_t refresh, uint8_t *out, size_t out_size)
{
   uint8_t scratch[OBU_SCRATCH];
   struct seq_params seq;
   struct frame_ctx ctx;
   struct bw w, payload;
   size_t tile_payload_len;

   derive_seq(desc, &seq);
   init_frame_ctx(&ctx, desc, &seq);
   ctx.refresh = refresh;
   resolve_refs(state, desc, ctx.our_ref_idx);

   bw_init(&w, out, out ? out_size : 0);

   /* One temporal delimiter per frame, deliberately. A stream's natural framing bundles a
    * no-show frame with the frame that displays it into a single unit, and a decoder then
    * returns one picture for the pair -- while the protocol above us submits one frame at a
    * time and expects one picture back. */
   emit_obu(&w, AV1_OBU_TEMPORAL_DELIMITER, NULL, 0);

   bw_init(&payload, scratch, sizeof(scratch));
   write_sequence_header(&payload, &seq);
   trailing_bits(&payload);
   if (payload.overflow)
      return -1;
   emit_obu(&w, AV1_OBU_SEQUENCE_HEADER, scratch, payload.pos);
   state->seq_valid = true;

   /* One OBU_FRAME rather than a separate header and tile group. Both are legal, but this
    * is the form every real stream uses, so it is the path decoders actually exercise --
    * and inside it the frame header is byte-aligned rather than terminated with trailing
    * bits, which is the one syntactic difference. */
   bw_init(&payload, scratch, sizeof(scratch));
   write_uncompressed_header(&payload, &ctx, state);
   byte_alignment(&payload);
   if (payload.overflow)
      return -1;

   /* Size the whole payload before emitting: the OBU length precedes it. */
   {
      struct bw sizer;

      bw_init(&sizer, NULL, 0);
      write_tile_group(&sizer, desc, &ctx, tiles, tiles_size);
      tile_payload_len = payload.pos + sizer.pos;
   }

   f(&w, 1, 0);
   f(&w, 4, AV1_OBU_FRAME);
   f(&w, 1, 0);
   f(&w, 1, 1);
   f(&w, 1, 0);
   leb128(&w, tile_payload_len);
   for (size_t i = 0; i < payload.pos; i++)
      f(&w, 8, scratch[i]);
   write_tile_group(&w, desc, &ctx, tiles, tiles_size);

   if (w.overflow)
      return -1;

   update_state(state, desc, &ctx);

   return (ssize_t)w.pos;
}

/* Emit the held frame. `live` is the guest reference map from the descriptor that follows it,
 * which is exactly where the guest's own choice of slot becomes visible: if the held picture
 * is in it, the guest stored it and so must we; if it is absent, the guest kept nothing and a
 * frame it never stored can never be referenced. */
static ssize_t flush_held(struct virgl_av1_obu_state *state,
                          const uint32_t *before, const uint32_t *live,
                          uint8_t *out, size_t out_size)
{
   uint32_t surface;
   uint8_t refresh = 0;
   int slot = -1;
   ssize_t n;

   if (!state->held)
      return 0;

   surface = live ? new_surface(before, live) : 0;
   if (surface) {
      slot = dead_slot(state, live);
      if (slot >= 0)
         refresh = (uint8_t)(1u << slot);
      else if (getenv("LIMINA_AV1_SLOT_TRACE"))
         fprintf(stderr, "[AV1SLOT] held frame stored by the guest but our slots are all live\n");
   }

   n = emit_frame(state, &state->held_desc, state->held_tiles, state->held_tiles_size,
                  refresh, out, out_size);
   state->held = false;
   if (n < 0)
      return n;

   if (slot >= 0) {
      state->slot_surface[slot] = surface;
      state->slot_valid |= (uint8_t)(1u << slot);
   }
   return n;
}

static bool hold_frame(struct virgl_av1_obu_state *state,
                       const struct virgl_av1_picture_desc *desc,
                       const void *tiles, size_t tiles_size)
{
   if (tiles_size > state->held_tiles_cap) {
      uint8_t *p = realloc(state->held_tiles, tiles_size);

      if (!p)
         return false;
      state->held_tiles = p;
      state->held_tiles_cap = tiles_size;
   }
   if (tiles_size)
      memcpy(state->held_tiles, tiles, tiles_size);
   state->held_desc = *desc;
   state->held_tiles_size = tiles_size;
   state->held = true;
   return true;
}

ssize_t virgl_av1_build_temporal_unit(struct virgl_av1_obu_state *state,
                                      const struct virgl_av1_picture_desc *desc,
                                      const void *tiles, size_t tiles_size,
                                      uint8_t *out, size_t out_size)
{
   const typeof(desc->picture_parameter) *p;
   uint8_t refresh;
   ssize_t n = 0, r;
   int slot;

   if (!state || !desc)
      return -1;
   p = &desc->picture_parameter;

   /* A frame held from the previous submission goes first: decode order is preserved, and
    * this descriptor's ref[] is what makes its refresh exact. */
   r = flush_held(state, state->prev_ref, desc->ref, out, out_size);
   if (r < 0)
      return r;
   n = r;

   /* Learn where a frame we emitted immediately ended up. Its slot was chosen when it was
    * written; only the surface that landed there was still unknown. */
   if (state->pending_unlearned) {
      const uint32_t surface = new_surface(state->prev_ref, desc->ref);

      if (surface) {
         state->slot_surface[state->pending_slot] = surface;
         state->slot_valid |= (uint8_t)(1u << state->pending_slot);
      }
      state->pending_unlearned = false;
   }

   prune_slots(state, desc->ref);
   memcpy(state->prev_ref, desc->ref, sizeof(state->prev_ref));

   /* A key frame refreshes everything, by inference rather than by choice, and resets our
    * model with it. Its own surface is learned on the next submission, like any frame we
    * emit immediately. */
   if (p->pic_info_fields.frame_type == AV1_FRAME_SWITCH ||
       (p->pic_info_fields.frame_type == AV1_FRAME_KEY &&
        p->pic_info_fields.show_frame)) {
      for (int i = 0; i < VIRGL_AV1_NUM_REF_FRAMES; i++)
         state->slot_surface[i] = 0;
      r = emit_frame(state, desc, tiles, tiles_size, 0xff, out ? out + n : NULL,
                     out ? out_size - n : 0);
      if (r < 0)
         return r;
      state->pending_slot = 0;
      state->pending_unlearned = true;
      return n + r;
   }

   /* A hidden frame waits one submission so its refresh can be exact. */
   if (!p->pic_info_fields.show_frame) {
      if (!hold_frame(state, desc, tiles, tiles_size))
         return -1;
      return n;
   }

   /* A shown frame is emitted now, into a slot nothing live occupies. When there is none,
    * store nothing rather than evict a picture a later frame may still want. */
   slot = dead_slot(state, desc->ref);
   refresh = slot >= 0 ? (uint8_t)(1u << slot) : 0;
   r = emit_frame(state, desc, tiles, tiles_size, refresh, out ? out + n : NULL,
                  out ? out_size - n : 0);
   if (r < 0)
      return r;
   if (slot >= 0) {
      state->slot_surface[slot] = 0;   /* the surface is learned on the next submission */
      state->slot_valid |= (uint8_t)(1u << slot);
      state->pending_slot = (uint8_t)slot;
      state->pending_unlearned = true;
   }
   return n + r;
}

/*
 * Emit whatever is still held. Nothing follows to reveal where the guest stored it, and
 * nothing that follows can reference it either, so it is written storing nothing.
 */
ssize_t virgl_av1_flush_temporal_unit(struct virgl_av1_obu_state *state,
                                      uint8_t *out, size_t out_size)
{
   if (!state)
      return -1;
   return flush_held(state, state->prev_ref, NULL, out, out_size);
}

void virgl_av1_obu_state_fini(struct virgl_av1_obu_state *state)
{
   if (!state)
      return;
   free(state->held_tiles);
   state->held_tiles = NULL;
   state->held_tiles_cap = state->held_tiles_size = 0;
   state->held = false;
}

ssize_t virgl_av1_build_av1c(struct virgl_av1_obu_state *state,
                             const struct virgl_av1_picture_desc *desc,
                             uint8_t *out, size_t out_size)
{
   uint8_t scratch[OBU_SCRATCH];
   struct seq_params seq;
   struct bw w, payload;

   if (!desc)
      return -1;
   (void)state;

   derive_seq(desc, &seq);

   bw_init(&w, out, out ? out_size : 0);

   f(&w, 1, 1);                       /* marker */
   f(&w, 7, 1);                       /* version */
   f(&w, 3, seq.profile);
   f(&w, 5, seq.level_idx);
   f(&w, 1, 0);                       /* seq_tier_0 */
   f(&w, 1, seq.bit_depth > 8);       /* high_bitdepth */
   f(&w, 1, seq.bit_depth == 12);     /* twelve_bit */
   f(&w, 1, seq.mono_chrome);
   f(&w, 1, 1);                       /* chroma_subsampling_x: 4:2:0 */
   f(&w, 1, 1);                       /* chroma_subsampling_y */
   f(&w, 2, 0);                       /* chroma_sample_position: unknown */
   f(&w, 3, 0);                       /* reserved */
   f(&w, 1, 0);                       /* initial_presentation_delay_present */
   f(&w, 4, 0);                       /* initial_presentation_delay_minus_one */

   bw_init(&payload, scratch, sizeof(scratch));
   write_sequence_header(&payload, &seq);
   trailing_bits(&payload);
   if (payload.overflow)
      return -1;
   emit_obu(&w, AV1_OBU_SEQUENCE_HEADER, scratch, payload.pos);

   if (w.overflow)
      return -1;

   return (ssize_t)w.pos;
}
