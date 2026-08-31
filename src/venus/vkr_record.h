/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * Full venus ring-stream recorder — the capture side of harness layer 2
 * (docs/rust-rewrite.md, harness/README.md).
 *
 * WHY THIS EXISTS. vkr_journal records only the commands that BUILD state, and
 * prunes them as their objects die: it answers "how do I rebuild this context's
 * object world", which is what snapshot/resume needs. A test corpus needs the
 * other thing — every command the guest actually sent, in order, so a renderer
 * can be replayed against it with no VM. Without that there is nothing to record
 * a golden from, and the Rust venus implementation has no oracle but a boot.
 *
 * WHAT IT PRODUCES. A prologue plus a stream:
 *
 *   prologue  the vkr_journal export for each context, taken at the instant that
 *             context's first command is recorded — the state the stream starts
 *             from;
 *   stream    the wire bytes of every command dispatched after that point, in a
 *             single sequence across contexts and rings.
 *
 * Taking the prologue at ARM time rather than at dump time is what makes the two
 * sections disjoint. A prologue taken at dump time would describe state the
 * stream also builds, and replaying both would apply those commands twice.
 *
 * TRUNCATION IS A PREFIX, NEVER A WINDOW. When the cap is reached, or a decode
 * goes fatal, recording STOPS and the header says so. The alternative — a FIFO
 * ring, as the vrend tracer uses — keeps the most recent commands, which is the
 * right choice for observing a fault and the wrong one here: a stream missing
 * its beginning cannot be replayed at all. A prefix always can.
 *
 * ARMING. LIMINA_VKR_RECORD=<MB>. Requires the journal (it supplies the
 * prologue), so VKR_JOURNAL=0 refuses to arm, loudly. Dump by writing a byte to
 * LIMINA_VKR_RECORD_FIFO (default /tmp/limina-vkr-record.fifo); output goes to
 * LIMINA_VKR_RECORD_OUT (default /tmp/limina-vkr-record.bin). Unarmed, every
 * hook is a predictable-branch no-op.
 *
 * THREADING. Ring threads dispatch concurrently. Sequence numbers are assigned
 * inside the same critical section that appends the bytes: assigning a seq
 * outside the lock lets two threads append in inverted seq order, and a replayer
 * that trusts stream order would then replay a serialization that never
 * happened.
 */

#ifndef VKR_RECORD_H
#define VKR_RECORD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct vkr_context;

/*
 * Serialized format. All little-endian; every byte payload padded to 4.
 *
 *   header:
 *     u32 magic 'VKRC', u32 version, u32 flags, u32 ctx_count,
 *     u64 record_count, u64 prologue_bytes, u64 stream_bytes
 *   prologue section (ctx_count entries):
 *     u32 ctx_id, u32 pad, u64 size, <size bytes of a 'VKJR' journal export>
 *   stream section (record_count entries):
 *     u64 seq, u64 ring_id, u32 ctx_id, u32 cmd_type, u32 size, u32 pad,
 *     <size bytes of raw venus wire>
 *
 * REPLAY CONTRACT, which the harness replayer is written against:
 *   1. virgl_renderer_limina_replay_begin(ctx) for every context in the
 *      prologue;
 *   2. each context's journal blob per the vkr_journal export contract —
 *      entries with ring_key != 0 through replay_ring_cmd on that ring, the
 *      rest through replay_submit, in seq order;
 *   3. the stream in seq order across all contexts: ring_id != 0 goes to
 *      replay_ring_cmd(ctx_id, ring_id, ...), ring_id == 0 to
 *      replay_submit(ctx_id, ...);
 *   4. virgl_renderer_limina_replay_end(ctx) for every context, last.
 *
 * The stream is fed BEFORE replay_end, not after: replay_end starts the ring
 * threads, and a running ring thread would race the replayed commands into a
 * different order than the one recorded.
 */
#define VKR_RECORD_MAGIC 0x43524b56u /* 'VKRC' LE */
#define VKR_RECORD_VERSION 1u

#define VKR_RECORD_FLAG_TRUNC_FULL 0x1u  /* hit the cap */
#define VKR_RECORD_FLAG_TRUNC_FATAL 0x2u /* a decode went fatal */

/* Lazy and idempotent; safe to call from any thread. */
void
vkr_record_init(void);

bool
vkr_record_enabled(void);

/*
 * The dispatch tee. Called from vkr_journal_post_dispatch, which is bracketed
 * around every vn_dispatch_command call site and already knows the command's
 * exact wire extent. `ring_id` is 0 for the context decoder.
 *
 * `fatal` records that this dispatch failed; the command itself is not appended
 * (it did not take effect) and recording stops, because a corpus that silently
 * omitted a failed command would claim a stream that continued cleanly.
 */
void
vkr_record_dispatch(struct vkr_context *ctx,
                    uint64_t ring_id,
                    uint32_t cmd_type,
                    const void *data,
                    size_t size,
                    bool fatal);

/* Drop a context's prologue and stop recording it (called at context teardown,
 * so a dump never references a context that no longer exists). */
void
vkr_record_context_gone(uint32_t ctx_id);

#endif /* VKR_RECORD_H */
