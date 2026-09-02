/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * limina: classic-vrend snapshot-replay re-creation journal (task #19; design in
 * limina docs/design/vrend-snapshot-replay.md). Sibling of the venus vkr_journal:
 * retains the durable subset of a context's SUBMIT_3D command stream — sub-context
 * and object creates, the current bound state (gallium only re-emits binds on
 * CHANGE, so "current binds" are part of the structural world, not transient), and
 * blob-defining PIPE_RESOURCE_CREATEs — so a snapshot restore can re-create the
 * in-renderer world the guest still believes in. Destroys prune (tombstones, not
 * retained); per-key state is latest-wins. Live size is bounded by live objects
 * and the finite set of state slots, not uptime.
 *
 * Unlike vkr_journal this is synchronous: recording happens on the single thread
 * that owns the classic submit path, entries are only creates/binds/state (draw
 * commands are never retained), and the copies are small.
 */

#ifndef VREND_JOURNAL_H
#define VREND_JOURNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct vrend_journal;

/* NULL when disabled (VREND_JOURNAL=0). */
struct vrend_journal *
vrend_journal_create(uint32_t ctx_id);

void
vrend_journal_destroy(struct vrend_journal *j);

/* Tee: called from the decode loop AFTER a command dispatched successfully.
 * `buf` points at the command header dword; `ndw` is the total dword count
 * including the header (len + 1). Classification and keying happen inside. */
void
vrend_journal_record(struct vrend_journal *j, const uint32_t *buf, uint32_t ndw);

/* Census counters for the [GPUTRACE]-style dump. */
struct vrend_journal_census {
   uint64_t recorded;   /* entries ever accepted (incl. later replaced/pruned) */
   uint64_t replaced;   /* latest-wins replacements */
   uint64_t pruned;     /* tombstoned by destroys */
   uint64_t skipped;    /* durable-unknown commands seen (no rule for them) */
   uint32_t live;       /* current entry count */
   uint64_t live_bytes; /* payload bytes across live entries */
};

void
vrend_journal_census(const struct vrend_journal *j, struct vrend_journal_census *out);

/* Log a one-line census (and with `verbose` a per-command breakdown). */
void
vrend_journal_dump(const struct vrend_journal *j);

/* P1 snapshot export, in the SAME 'VKJR' v1 wire format vkr_journal_export
 * emits — the libkrun consumer parses both identically and routes entries back
 * through the context's submit path at restore. Entry mapping: cmd_type = the
 * virgl command id, ring_key = 0 (classic has no rings), klass = 6 for creates
 * (a dropped create is load-bearing and warns at replay) and 3 for latest-wins
 * state (the benign stale-reference drop class). malloc'd; caller frees. */
bool
vrend_journal_export(const struct vrend_journal *j, void **out_buf, size_t *out_size);

/* The journal's last-assigned sequence — the cross-layer fence stamped on a
 * control-queue CREATE_BLOB so replay feeds the wire PIPE_RESOURCE_CREATE that
 * defines its blob_id first. */
uint64_t
vrend_journal_seq(const struct vrend_journal *j);

/* A blob resource's GLOBAL unref retires the wire PIPE_RESOURCE_CREATE that
 * defined its blob_id (the classic analogue of the venus create-pin release). */
void
vrend_journal_unpin_blob(struct vrend_journal *j, uint64_t blob_id);

#endif /* VREND_JOURNAL_H */
