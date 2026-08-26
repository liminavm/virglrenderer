/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * limina: an in-memory ring recorder for everything vrend is asked to do, for the
 * notification-text spike (spikes/notification-text-corruption/RESULTS.md).
 *
 * WHY IN MEMORY. The fault under investigation is cured by a bare glFlush, so it lives at a
 * submission/batch boundary. A tracer that writes a line per command would move exactly the
 * boundary it is trying to observe -- the classic way to build a heisenbug and then "fix" it.
 * So the hot path does a bounded, allocation-free, syscall-free append into a preallocated
 * ring, and file I/O happens only when a dump is explicitly requested.
 *
 * WHAT IT RECORDS. Not only draws: the question is what other work runs alongside them, and
 * for a compositor that is transfers and fences as much as draws. Guest->host transfers do NOT
 * arrive through the command stream, so hooking the decode loop alone would miss the
 * glyph-atlas uploads -- precisely the upload-versus-sample class we are hunting. Record types
 * cover the batch boundary, every decoded command with its FULL payload, transfers, and fences.
 *
 * Payloads are complete rather than summarised so the captured stream stays replayable later;
 * a few summary words per record would foreclose that and leave only diffing.
 */

#ifndef VREND_TRACE_H
#define VREND_TRACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Record types. Kept stable: the decoder in the spike reads these by number. */
enum vrend_trace_type {
   VREND_TRACE_SUBMIT   = 1,  /* a command batch begins; aux0 = byte size */
   VREND_TRACE_CMD      = 2,  /* one decoded command; cmd = VIRGL_CCMD_*, payload = its dwords */
   VREND_TRACE_DRAW_FB  = 3,  /* enrichment emitted with a draw: the bound target's dimensions */
   VREND_TRACE_TRANSFER = 4,  /* a transfer, which never appears in the command stream */
   VREND_TRACE_FENCE    = 5,
   VREND_TRACE_RETIRE   = 6,
   VREND_TRACE_PAD      = 7,  /* filler to the end of the ring; carries no meaning */
   VREND_TRACE_XFERDATA = 9,  /* the BYTES a guest->host transfer carried; aux0 = res handle */
};

/* Resource creation does NOT pass through the command stream -- it arrives on the control path
 * (virgl_renderer_resource_create / _create_blob), so a trace of decoded commands contains zero
 * PIPE_RESOURCE_CREATE and a replay built from it cannot construct a single resource.
 *
 * These records live in a SIDE STORE that is never evicted, not in the ring. The resources a
 * replay most needs -- the cached index buffer, the rolling constant-attribute buffer, the glyph
 * atlas -- are created once at client startup, so they are the OLDEST records in the stream and
 * would be the first thing FIFO eviction eats once transfer payloads inflate the ring. */
/* An ORDERED log of create/destroy events, not a table keyed by handle: resource handles are
 * reused, so a keyed table would silently merge two different resources that happened to share
 * an id. `seq` interleaves these against the ring's records. */
enum vrend_trace_res_kind {
   VREND_TRACE_RES_CREATE = 0,
   VREND_TRACE_RES_BLOB   = 1,
   VREND_TRACE_RES_UNREF  = 2,
};

struct vrend_trace_res {
   uint64_t seq;
   uint32_t kind;
   uint32_t handle, target, format, bind;
   uint32_t width, height, depth, array_size;
   uint32_t last_level, nr_samples, flags;
};

/* 32-byte header, 8-aligned, followed by payload_len bytes of payload. */
struct vrend_trace_rec {
   uint32_t total_len;    /* header + payload + alignment padding, for walking the ring */
   uint8_t  type;
   uint8_t  cmd;
   uint16_t ctx_id;
   uint64_t seq;
   uint64_t mono_ns;
   uint32_t payload_len;
   uint32_t aux_count;    /* aux words follow the header, before the payload */
};

/* Lazy, idempotent. LIMINA_VREND_TRACE=<MB> arms it; unset or 0 leaves every hook a
 * predictable-branch no-op. */
void vrend_trace_init(void);
bool vrend_trace_enabled(void);

void vrend_trace_submit(uint32_t ctx_id, size_t bytes);
void vrend_trace_cmd(uint32_t ctx_id, uint32_t cmd, const uint32_t *buf, uint32_t dwords);
void vrend_trace_draw_fb(uint32_t ctx_id, uint32_t width, uint32_t height,
                         uint32_t nr_cbufs, uint32_t gl_id);
void vrend_trace_transfer(uint32_t ctx_id, uint32_t res_handle, int mode, uint32_t level,
                          uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                          uint32_t stride, uint64_t offset);
void vrend_trace_fence(uint32_t ctx_id, uint32_t flags, uint64_t fence_id);
void vrend_trace_res_event(struct vrend_trace_res *res);
/* The bytes a guest->host transfer carried, captured where vrend consumes them -- one site
 * covering TRANSFER3D and COPY_TRANSFER3D alike, so a replayer never has to reconstruct the
 * blob/iov machinery: it synthesizes one backing iov per resource and memcpys these in. */
void vrend_trace_transfer_data(uint32_t ctx_id, uint32_t res_handle, uint64_t offset,
                               const void *data, uint32_t len);
void vrend_trace_retire_fence(uint32_t ctx_id, uint64_t fence_id);

/* Called at a safe point (the top of a submit) -- a relaxed atomic load when idle. The dump runs
 * on the calling thread and takes the same lock the recorder does: retires arrive on vrend's
 * fence-poll thread, so there is more than one writer. */
void vrend_trace_maybe_dump(void);

#endif /* VREND_TRACE_H */
