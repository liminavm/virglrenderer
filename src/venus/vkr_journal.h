/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * gkvm: venus snapshot-replay re-creation journal (limina M9.3 seamless resume).
 *
 * Records, per context, the wire bytes of every state-building venus command so
 * the host-side object world can be rebuilt at snapshot-restore by replaying
 * them through the normal decoder (vkr_context_submit_cmd). Guest-assigned
 * object IDs make wire replay ID-faithful. Design:
 * limina docs/design/venus-snapshot-replay.md.
 *
 * Classification is effect-driven, not table-driven, wherever possible:
 *  - a command that ADDED objects to the context object table is a CREATE,
 *    keyed by the added ids (batch allocs get all their ids; pruned when ALL
 *    are dead);
 *  - a command that REMOVED objects prunes every entry keyed by those ids and
 *    is itself not retained (destroys need no replay);
 *  - vkCmd* / Begin/End/ResetCommandBuffer are RECORDING entries keyed by the
 *    command buffer (peeked from the wire: every venus handle is a plain LE
 *    u64 object id, and the first handle sits right after the 8-byte
 *    cmd_type+flags header). Begin/Reset prune the buffer's prior recording;
 *  - binds and descriptor updates are keyed by ids the vkr handlers note
 *    explicitly (vkr_journal_note_keys), pruned when ANY key dies;
 *  - vkFree{CommandBuffers,DescriptorSets} are retained keyed to the pool,
 *    with the freed ids as aux data: a partially-dead batch alloc replays all
 *    its ids and the retained free then kills the dead subset, keeping the id
 *    space exact (guest id reuse stays collision-free). Serialization-time
 *    validation drops frees whose alloc entry is gone (P1);
 *  - ring create/destroy and per-ring reply-stream state are RING entries
 *    (reply-stream set/seek are latest-wins per ring);
 *  - everything else is transient (counted, not retained).
 *
 * Always-on (kill switch: VKR_JOURNAL=0). Thread-safety: dispatch hooks use a
 * TLS frame (ring threads dispatch concurrently); the journal itself locks its
 * own mutex, taken after ctx->object_mutex in the add/remove hooks — never the
 * reverse order anywhere.
 */

#ifndef VKR_JOURNAL_H
#define VKR_JOURNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <vulkan/vulkan.h>

struct vkr_context;
struct vn_dispatch_context;

struct vkr_journal_stats {
   uint64_t entries_live;
   uint64_t bytes_live;
   uint64_t recorded_creates;
   uint64_t recorded_recordings;
   uint64_t recorded_noted;
   uint64_t recorded_ring;
   uint64_t recorded_frees;
   uint64_t recorded_pool_resets;
   uint64_t pruned_entries;
   uint64_t transient_cmds;
   uint64_t orphan_adds;      /* object added outside any dispatch frame */
   uint64_t dropped_fatal;    /* frames dropped because the command failed */
   uint64_t noted_multi_key;  /* noted-mutate with >1 key (shared-prune risk) */
};

struct vkr_journal *
vkr_journal_create(uint32_t ctx_id);

void
vkr_journal_destroy(struct vkr_journal *j);

/* enabled unless VKR_JOURNAL=0 (checked once) */
bool
vkr_journal_enabled(void);

/* The dispatch tee (vkr_journal_pre_dispatch / vkr_journal_post_dispatch) is
 * declared in the generated vn_protocol_renderer_dispatches.h next to its one
 * caller, vn_dispatch_command — its signature uses the protocol's
 * VkCommandTypeEXT, which plain vulkan.h consumers of this header lack. */

/* object-table hooks — called from vkr_context_{add,remove}_object under
 * ctx->object_mutex; out-of-line so vkr_context.h needs no journal internals */
void
vkr_journal_object_added(struct vkr_context *ctx, uint64_t id, VkObjectType type);

void
vkr_journal_object_removed(struct vkr_context *ctx, uint64_t id);

/* in-handler key attribution for commands whose keys live in decoded args
 * (descriptor updates, memory binds); attaches to the current TLS frame */
void
vkr_journal_note_keys(struct vkr_context *ctx, const uint64_t *ids, uint32_t count);

/* in-handler attribution for partial frees: retained keyed to `pool_id`,
 * freed ids kept as aux data for serialization-time validation */
void
vkr_journal_note_free(struct vkr_context *ctx,
                      uint64_t pool_id,
                      const uint64_t *ids,
                      uint32_t count);

void
vkr_journal_get_stats(struct vkr_journal *j, struct vkr_journal_stats *out);

/* defined in vkr_renderer.c; declared here (not vkr_renderer.h) because that
 * header is consumed by non-Vulkan TUs that lack VkObjectType */
const char *
vkr_object_type_name(VkObjectType type);

/* census + stats via vkr_log, for the LIMINA_GPU_TRACE_VKR state dump; the
 * live-create tally by VkObjectType is cross-checkable against the context's
 * object-table tally */
void
vkr_journal_dump(struct vkr_journal *j);

#endif /* VKR_JOURNAL_H */
