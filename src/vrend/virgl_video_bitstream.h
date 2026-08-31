/*
 * Bit-level reading and writing for the parameter sets the VideoToolbox backend has to
 * synthesize. Shared by the H.264 and H.265 serializers, which write the same Exp-Golomb
 * and fixed-width syntax elements over the same RBSP escaping rules.
 *
 * Nothing here knows about either codec: the two serializers own their syntax, this owns
 * the bits underneath it.
 */
#ifndef VIRGL_VIDEO_BITSTREAM_H
#define VIRGL_VIDEO_BITSTREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

static inline void bs_init(struct bs *w, uint8_t *buf, size_t cap, bool escape)
{
   memset(w, 0, sizeof(*w));
   w->buf = buf;
   w->cap = cap;
   w->escape = escape;
}

static inline void bs_raw_byte(struct bs *w, uint8_t b)
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
static inline void bs_byte(struct bs *w, uint8_t b)
{
   if (w->escape && w->zeros >= 2 && b <= 0x03) {
      bs_raw_byte(w, 0x03);
      w->zeros = 0;
   }
   bs_raw_byte(w, b);
   w->zeros = b ? 0 : w->zeros + 1;
}

/* u(n) / f(n): n bits, most significant first. */
static inline void u(struct bs *w, unsigned n, uint32_t v)
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

static inline void flag(struct bs *w, bool v) { u(w, 1, v ? 1 : 0); }

/* ue(v): Exp-Golomb. codeNum v is written as a prefix of N zeros, a 1, then N bits. */
static inline void ue(struct bs *w, uint32_t v)
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
static inline void se(struct bs *w, int32_t v)
{
   uint32_t code = v <= 0 ? (uint32_t)(-2 * (int64_t)v)
                          : (uint32_t)(2 * (int64_t)v - 1);
   ue(w, code);
}

/* rbsp_trailing_bits(): a 1 bit, then zeros to the byte boundary. */
static inline void rbsp_trailing(struct bs *w)
{
   flag(w, 1);
   while (w->nbits)
      flag(w, 0);
}


/* ---------------------------------------------------------------- bit reader */

/* Minimal RBSP reader: strips emulation-prevention bytes as it goes. */
struct br {
   const uint8_t *buf;
   size_t len;
   size_t pos;
   unsigned bit;
   unsigned zeros;
};

static inline int br_bit(struct br *r)
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

static inline int br_ue(struct br *r, uint32_t *out)
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
static inline const uint8_t *next_start_code(const uint8_t *p, const uint8_t *end, unsigned *sc_len)
{
   for (; p + 2 < end; p++) {
      if (p[0] == 0 && p[1] == 0) {
         if (p[2] == 1) { *sc_len = 3; return p; }
         if (p + 3 < end && p[2] == 0 && p[3] == 1) { *sc_len = 4; return p; }
      }
   }
   return NULL;
}

/* u(n): n bits, most significant first. Negative on truncation. */
static inline int br_u(struct br *r, unsigned n, uint32_t *out)
{
   uint32_t v = 0;

   for (unsigned i = 0; i < n; i++) {
      int b = br_bit(r);
      if (b < 0)
         return -1;
      v = (v << 1) | (uint32_t)b;
   }
   *out = v;
   return 0;
}

/* se(v): signed Exp-Golomb, mapped 0, 1, -1, 2, -2, ... */
static inline int br_se(struct br *r, int32_t *out)
{
   uint32_t k;

   if (br_ue(r, &k))
      return -1;
   *out = (k & 1) ? (int32_t)((k + 1) / 2) : -(int32_t)(k / 2);
   return 0;
}

#endif /* VIRGL_VIDEO_BITSTREAM_H */
