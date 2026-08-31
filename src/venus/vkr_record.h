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
 *     u32 ctx_id, u32 generation, u64 size, <size bytes of a 'VKJR' journal export>
 *   stream section (record_count entries):
 *     u64 seq, u64 ring_id, u32 ctx_id, u32 generation, u32 kind, u32 op, u32 size,
 *     u32 reserved, <size bytes of payload>
 *
 * KIND splits the stream in two. VKR_RECORD_KIND_CMD records are venus wire bytes and `op` is
 * the command type; VKR_RECORD_KIND_CTL records are host-side control-path events and `op` is a
 * VKR_RECORD_CTL_* code over a fixed payload struct. They share one sequence because the ORDER
 * BETWEEN THEM IS LOAD-BEARING: a HOST3D blob exports a VkDeviceMemory that a venus command
 * allocated, and vkCreateRingMESA reads a blob created before it. Hoisting either group ahead of
 * the other breaks both dependencies, in opposite directions.
 *
 * WHY CONTROL EVENTS ARE HERE AT ALL. Resources are not created by the command stream. The guest
 * asks the VMM, the VMM calls virgl_renderer_resource_create_blob, and only then does a venus
 * command name the resource id. A corpus of commands alone therefore references resources that do
 * not exist: vkCreateRingMESA rejects its resourceId outright (vkr_transport.c requires
 * fd_type == VIRGL_RESOURCE_FD_SHM), so a stream recorded without these events cannot replay past
 * its first ring. This mirrors what vrend's tracer already does one level down with its resource
 * side store, and what libkrun's own snapshot journal does for resume.
 *
 * GENERATION is what makes ctx_id usable. The guest reuses context ids -- a context is destroyed
 * and the next one is handed the same number -- so a corpus keyed on ctx_id alone silently merges
 * two unrelated contexts. One run of vulkaninfo followed by one of vkcube produced exactly that:
 * two prologues both claiming context 8. The generation is a monotonic counter over adopted
 * contexts, so (ctx_id, generation) names one context for the life of the capture, and a replayer
 * must key on the pair. CTL records carry the owning context's pair where there is one, and
 * (0, 0) where there is none (a resource unref names no context).
 *
 * REPLAY CONTRACT, which the harness replayer is written against:
 *   1. virgl_renderer_limina_replay_begin(ctx) for every context in the
 *      prologue, in generation order -- two prologues may share a ctx_id, and
 *      the later generation is a DIFFERENT context that reused the number;
 *   2. each context's journal blob per the vkr_journal export contract --
 *      entries with ring_key != 0 through replay_ring_cmd on that ring, the
 *      rest through replay_submit, in seq order;
 *   3. the stream in seq order across all contexts, applying each record AT ITS
 *      RECORDED POSITION and never hoisting one kind ahead of the other:
 *        KIND_CMD, ring_id != 0  -> replay_ring_cmd(ctx_id, ring_id, ...)
 *        KIND_CMD, ring_id == 0  -> replay_submit(ctx_id, ...)
 *        KIND_CTL                -> the matching public-ABI call (below);
 *   4. virgl_renderer_limina_replay_end(ctx) for every context, last.
 *
 * The stream is fed BEFORE replay_end, not after: replay_end starts the ring
 * threads, and a running ring thread would race the replayed commands into a
 * different order than the one recorded.
 *
 * A replayer synthesizes what the VM supplied. A guest-storage blob's iovecs pointed into guest
 * RAM that no longer exists, so only their count and total size are recorded and the replayer
 * supplies its own backing of that size. An imported blob arrived as a host fd from outside the
 * renderer entirely and cannot be reconstructed at all -- it is recorded so a replayer can say so
 * rather than fail obscurely on the first command that names it.
 */
#define VKR_RECORD_KIND_CMD 0u
#define VKR_RECORD_KIND_CTL 1u

/*
 * Control-path ops and their payloads. Field order is the payload layout: u32 unless marked,
 * little-endian, padded to 8-byte alignment before any u64.
 */
#define VKR_RECORD_CTL_CTX_CREATE 1u      /* ctx_id, context_init, name_len, pad, name[] */
#define VKR_RECORD_CTL_CTX_DESTROY 2u     /* ctx_id, pad */
#define VKR_RECORD_CTL_CREATE_BLOB 3u     /* res_handle, ctx_id, blob_mem, blob_flags,
                                             u64 blob_id, u64 size, num_iovs, pad */
#define VKR_RECORD_CTL_IMPORT_BLOB 4u     /* res_handle, fd_type, u64 size */
#define VKR_RECORD_CTL_ATTACH_RESOURCE 5u /* ctx_id, res_handle */
#define VKR_RECORD_CTL_DETACH_RESOURCE 6u /* ctx_id, res_handle */
#define VKR_RECORD_CTL_RESOURCE_UNREF 7u  /* res_handle, pad */

#define VKR_RECORD_MAGIC 0x43524b56u /* 'VKRC' LE */
#define VKR_RECORD_VERSION 3u

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

/*
 * Control-path tees, called from the public ABI entry points in virglrenderer.c. They append into
 * the same sequence as vkr_record_dispatch, under the same lock, which is the whole point: the
 * recorded interleaving IS the dependency order between resources and the commands that use them.
 *
 * These record the ARGUMENTS, not the outcome, and are called before the work is attempted. A
 * recorded event for a call that then failed is harmless -- the replayer creates a resource
 * nothing references -- while a missed event is not.
 */
void
vkr_record_ctx_create(uint32_t ctx_id, uint32_t context_init, const char *name, uint32_t nlen);

void
vkr_record_ctx_destroy(uint32_t ctx_id);

void
vkr_record_create_blob(uint32_t res_handle,
                       uint32_t ctx_id,
                       uint32_t blob_mem,
                       uint32_t blob_flags,
                       uint64_t blob_id,
                       uint64_t size,
                       uint32_t num_iovs);

void
vkr_record_import_blob(uint32_t res_handle, uint32_t fd_type, uint64_t size);

void
vkr_record_attach_resource(uint32_t ctx_id, uint32_t res_handle);

void
vkr_record_detach_resource(uint32_t ctx_id, uint32_t res_handle);

void
vkr_record_resource_unref(uint32_t res_handle);

#endif /* VKR_RECORD_H */
