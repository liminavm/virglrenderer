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
