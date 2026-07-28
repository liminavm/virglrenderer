/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * limina: venus snapshot-replay re-creation journal. See vkr_journal.h for the
 * model and limina docs/design/venus-snapshot-replay.md for the design.
 */

#include "vkr_journal.h"

#include <stdlib.h>
#include <string.h>

#define XXH_INLINE_ALL
#include "util/u_atomic.h"
#include "util/u_thread.h"
#include "util/xxhash.h"

#include "vkr_common.h"
#include "vkr_context.h"
#include "vkr_cs.h"
#include "vkr_renderer.h"
#include "vkr_ring.h"

#include "vn_protocol_renderer.h"

/* mirrors the dispatch-table bound in vn_protocol_renderer_dispatches.h */
#define VKR_JOURNAL_NUM_CMD_TYPES 346

enum vkr_journal_class {
   VKR_JOURNAL_TRANSIENT = 0, /* not retained */
   VKR_JOURNAL_CREATE,        /* keyed by added ids; pruned when ALL dead */
   VKR_JOURNAL_RECORDING,     /* vkCmd... + Begin/End/Reset; keyed by cmd_buf @8 */
   VKR_JOURNAL_NOTED,         /* keys noted by the handler; pruned on ANY death */
   VKR_JOURNAL_FREE,          /* partial batch free; keyed to pool, aux = freed ids */
   VKR_JOURNAL_POOL_RESET,    /* vkReset{Command,Descriptor}Pool; keyed by pool @16 */
   VKR_JOURNAL_RING_CREATE,   /* vkCreateRingMESA; keyed by ring id @8 */
   VKR_JOURNAL_RING_DESTROY,  /* vkDestroyRingMESA; prunes ring id @8, not retained */
   VKR_JOURNAL_RING_STREAM,   /* reply-stream set/seek; latest-wins per ring */
};

/* class-table flag: a Begin/Reset drops the target's prior RECORDING entries */
#define VKR_JOURNAL_F_RESETS_PRIOR 0x80
#define VKR_JOURNAL_CLASS_MASK 0x7f

struct vkr_journal_entry;
struct vkr_journal_msg;

static bool
vkr_journal_u64_append(uint64_t **arr, uint32_t *n, uint32_t *cap, uint64_t v);
static int
vkr_journal_thread(void *arg);
static void
vkr_journal_push(struct vkr_journal *j, struct vkr_journal_msg *m);
static void
vkr_journal_msg_apply(struct vkr_journal *j, struct vkr_journal_msg *m);
static void
vkr_journal_msg_free(struct vkr_journal_msg *m);
static void
vkr_journal_quiesce(struct vkr_journal *j);

struct vkr_journal_keynode {
   uint64_t id;
   struct list_head refs; /* vkr_journal_keyref.link */
   /* limina snapshot-replay: entry pinning. A blob resource created from a
    * VkDeviceMemory pins the memory's key: the guest may vkFreeMemory while the
    * resource lives on (Xwayland cross-context buffers), and replay still needs
    * the alloc entry to re-create the memory BEFORE the blob create. While
    * pinned, prune_key defers (prune_deferred) and the freeing command itself is
    * retained keyed by this id — replay then re-runs alloc → blob create → free
    * in original seq order, reproducing the live world. The deferred prune fires
    * when the last pinning resource dies (vkr_journal_unpin_key). */
   uint32_t pinned;
   bool prune_deferred;
   /* pinned alongside this key (the alloc's dedicated buffer/image), unpinned
    * with it — the alloc entry cannot replay if the dedicated object is gone */
   uint64_t dep_id;
};

struct vkr_journal_keyref {
   uint64_t id;
   bool dead;
   struct vkr_journal_entry *entry;
   struct vkr_journal_keynode *node;
   struct list_head link; /* in node->refs */
};

struct vkr_journal_entry {
   uint64_t seq;
   uint16_t cmd_type;
   uint8_t klass;
   uint32_t size;
   uint8_t *data;

   uint32_t nkeys;
   uint32_t keys_dead;
   struct vkr_journal_keyref *refs; /* array[nkeys] */
   VkObjectType *key_types;         /* CREATE only (census); else NULL */

   uint32_t naux;
   uint64_t *aux; /* FREE: the freed ids, for serialization-time validation */

   /* limina snapshot-replay: create-arg closure (the 2026-07-20 vkmark crash).
    * A CREATE's wire args may reference objects the guest legally destroys
    * while the created object lives on — pipeline←shader modules/layout is
    * the canonical case. Their creates must stay replayable, so a CREATE
    * entry pins every object id its decode looked up; the pins drop when
    * THIS entry dies. See vkr_journal_note_lookup / vkr_journal_pin_refs. */
   uint32_t npinned;
   uint64_t *pinned_refs;

   struct list_head link; /* global order */
};

struct vkr_journal {
   uint32_t ctx_id;
   mtx_t mutex;
   struct list_head entries;
   struct hash_table *keys; /* &keynode->id -> keynode */
   uint64_t seq_next;
   struct vkr_journal_stats stats;

   /* limina (2026-07-27, two-lane journal): retention runs on a per-journal consumer
    * thread, OFF the decode path. The decode threads (context + rings) only classify,
    * copy the command payload once, and push a message; every hash/list/pin/prune walk
    * happens here. Motivation: the compositor session correlated presentation misses
    * with DRAW COUNT at fixed GPU time (present-misses.md §17.3, partial rho +0.807),
    * and per-vkCmd retention on the decode thread is exactly a per-draw host cost.
    * Ordering: every producer is serialized by the push mutex and messages are applied
    * strictly in queue order, which preserves each decode thread's program order —
    * cross-ring interleaving was arbitrary under the old shared mutex too. Sync
    * readers (export/seq/stats/dump — snapshot-time, quiesced VM) drain the queue
    * first via vkr_journal_quiesce(). */
   mtx_t q_mutex;
   cnd_t q_cond;
   cnd_t flush_cond;
   struct list_head q; /* vkr_journal_msg.link */
   thrd_t thread;
   bool thread_live;
   bool q_stop;
   uint64_t flush_gen_next;
   uint64_t flush_gen_done;
   /* queue-depth gauge (under q_mutex): current + high-water. Depth at dump
    * time should be ~0 (dump quiesces first) — the peak is the debug signal
    * for backlog corner cases (producers outpacing the consumer). */
   uint32_t q_depth;
   uint32_t q_depth_peak;

   /* decode-thread stat bumps that must not take any lock */
   uint32_t transient_fast; /* p_atomic */
   uint32_t orphan_adds_fast;
   uint32_t dropped_fatal_fast;
   uint32_t dropped_oom_fast;
};

enum vkr_journal_msg_type {
   VKR_JOURNAL_MSG_INSERT = 0,
   VKR_JOURNAL_MSG_PRUNE_KEY,
   VKR_JOURNAL_MSG_PIN,
   VKR_JOURNAL_MSG_UNPIN,
   VKR_JOURNAL_MSG_FLUSH,
};

/* small-command payloads are captured into the message itself: the decode
 * thread then allocates exactly ONE thing per retained command (the msg), and
 * the long-term heap copy happens on the consumer (2026-07-28 drawstorm
 * decomposition: the per-command malloc(data)+malloc(keys) pair was a top
 * decode-lane cost at 20k cmds/frame). Covers every vkCmd* hot command;
 * big payloads (multi-KB descriptor updates) keep the malloc path. */
#define VKR_JOURNAL_INLINE_DATA 96

/* one captured command (or key op), produced on a decode thread, consumed in order
 * by the journal thread. All pointer fields are OWNED by the message, except
 * data/keys when the corresponding *_inline flag says they point into the msg. */
struct vkr_journal_msg {
   struct list_head link;
   uint8_t type;

   /* INSERT */
   uint8_t klass; /* insert class (already routed by the producer) */
   uint16_t cmd_type;
   uint32_t size;
   uint8_t *data;
   bool data_inline; /* data == inline_data (consumer materializes on insert) */
   bool keys_inline; /* keys == &inline_key */
   uint32_t nkeys;
   uint64_t *keys;
   uint64_t inline_key;
   uint8_t inline_data[VKR_JOURNAL_INLINE_DATA];
   VkObjectType *key_types; /* CREATE only */
   uint32_t naux;
   uint64_t *aux;
   /* CREATE-closure pinning inputs (moved out of the dispatch frame; the
    * created-id dedupe set is m->keys itself) */
   uint32_t nrefs;
   uint64_t *refs;
   /* ids removed during the dispatch — pruned before the insert */
   uint32_t nremoved;
   uint64_t *removed;
   /* drop-before-insert (RESETS_PRIOR begin/reset, RING_STREAM latest-wins) */
   bool drop_first;
   uint8_t drop_klass;
   int32_t drop_cmd_type; /* -1 = any */
   uint64_t drop_key;

   /* PRUNE_KEY / PIN / UNPIN */
   uint64_t id;
   uint64_t dep_id;

   /* FLUSH */
   uint64_t flush_gen;
};

/* per-dispatch recording frame; a stack because vkExecuteCommandStreamsMESA
 * nests one level of inner commands through the same vn_dispatch_command */
struct vkr_journal_frame {
   struct vkr_journal *j;
   struct vkr_context *ctx;
   struct vn_dispatch_context *dctx;
   const uint8_t *start;

   uint64_t *created;
   VkObjectType *created_types;
   uint32_t ncreated, created_cap;

   uint64_t *noted;
   uint32_t nnoted, noted_cap;

   /* every object id the decoder successfully looked up while dispatching this
    * command (vkr_journal_note_lookup from vkr_cs_decoder_lookup_object) — the
    * create-arg dependencies when the command turns out to be a CREATE */
   uint64_t *refs;
   uint32_t nrefs, refs_cap;

   /* object ids removed during this dispatch (vkr_journal_object_removed);
    * pruned by the consumer BEFORE the command's own insert. A prune the
    * consumer defers (pinned key; see vkr_journal_keynode) retains this
    * command NOTED-keyed by the pinned id — decided there, not here. */
   uint64_t *removed;
   uint32_t nremoved, removed_cap;

   bool have_free;
   uint64_t free_pool;
   uint64_t *freed;
   uint32_t nfreed, freed_cap;

   struct vkr_journal_frame *parent;
};

static _Thread_local struct vkr_journal_frame *vkr_journal_frame_cur;

/* limina (2026-07-28, decode-lane batching): while a decode thread drains one
 * ring/context command batch (vkr_ring_submit_cmd / vkr_context_submit_cmd),
 * retained messages accumulate here and reach the consumer queue in ONE
 * lock+signal at batch end, instead of a mutex round trip per command — at
 * 20k cmds/frame the per-command push (lock, contended wake, signal) was a
 * top decode-lane cost in the drawstorm decomposition. Program order within
 * the thread is preserved (the batch splices in order); cross-thread order
 * was arbitrary under per-command pushes too. The batch NEVER outlives the
 * submit_cmd call (flushed on every exit path), so quiesce() — which runs on
 * quiesced-VM readers only — still observes every message. */
struct vkr_journal_batch {
   struct vkr_journal *j;
   struct list_head msgs;
   uint32_t n;
   int depth; /* active while > 0; defensive against nested submit paths */
};
static _Thread_local struct vkr_journal_batch vkr_journal_batch_tls;

static void
vkr_journal_push_now(struct vkr_journal *j, struct vkr_journal_msg *m);

void
vkr_journal_batch_begin(void)
{
   struct vkr_journal_batch *b = &vkr_journal_batch_tls;
   if (b->depth++ == 0) {
      list_inithead(&b->msgs);
      b->j = NULL;
      b->n = 0;
   }
}

/* splice the pending batch into j's queue NOW, keeping the batch scope open.
 * Called whenever queue arrival order must catch up with program order — see
 * vkr_journal_msg_batchable for why only RECORDING inserts may stay behind. */
static void
vkr_journal_batch_drain(struct vkr_journal *j)
{
   struct vkr_journal_batch *b = &vkr_journal_batch_tls;
   if (b->depth <= 0 || b->j != j || !b->n)
      return;

   if (!j->thread_live) {
      /* no consumer: apply inline in order, mirroring vkr_journal_push */
      list_for_each_entry_safe (struct vkr_journal_msg, m, &b->msgs, link) {
         list_del(&m->link);
         vkr_journal_msg_apply(j, m);
         vkr_journal_msg_free(m);
      }
      b->n = 0;
      return;
   }

   mtx_lock(&j->q_mutex);
   list_splicetail(&b->msgs, &j->q);
   j->q_depth += b->n;
   if (j->q_depth > j->q_depth_peak)
      j->q_depth_peak = j->q_depth;
   cnd_signal(&j->q_cond);
   mtx_unlock(&j->q_mutex);

   list_inithead(&b->msgs);
   b->n = 0;
}

void
vkr_journal_batch_flush(void)
{
   struct vkr_journal_batch *b = &vkr_journal_batch_tls;
   assert(b->depth > 0);
   if (--b->depth > 0)
      return;

   struct vkr_journal *j = b->j;
   if (j)
      vkr_journal_batch_drain(j);
   b->j = NULL;
}

static uint32_t
vkr_journal_hash_u64(const void *key)
{
   return XXH32(key, sizeof(uint64_t), 0);
}

static bool
vkr_journal_key_u64_equal(const void *key1, const void *key2)
{
   return *(const uint64_t *)key1 == *(const uint64_t *)key2;
}

/* limina (2026-07-27, decode-path cost attribution): VKR_JOURNAL=norecord keeps the
 * journal alive but skips RECORDING-class retention (per-vkCmd* capture into live
 * command buffers) — snapshot correctness is intentionally SACRIFICED in this mode
 * (restored command buffers replay empty); it exists ONLY to measure how much of the
 * journal's hot-path cost is the per-command recording lane. Not a shipping config. */
static int
vkr_journal_mode(void)
{
   static int mode = -1;
   if (mode < 0) {
      const char *env = getenv("VKR_JOURNAL");
      if (env && env[0] == '0')
         mode = 0;
      else if (env && !strcmp(env, "norecord"))
         mode = 2;
      else
         mode = 1;
   }
   return mode;
}

bool
vkr_journal_enabled(void)
{
   return vkr_journal_mode() != 0;
}

/* --- classification table (built once; effect-driven classes override) --- */

static uint8_t vkr_journal_class_table[VKR_JOURNAL_NUM_CMD_TYPES];
static once_flag vkr_journal_class_once = ONCE_FLAG_INIT;

static void
vkr_journal_build_class_table(void)
{
   uint8_t *t = vkr_journal_class_table;

   for (uint32_t i = 0; i < VKR_JOURNAL_NUM_CMD_TYPES; i++) {
      const char *name = vn_dispatch_command_name((VkCommandTypeEXT)i);
      t[i] = (name && !strncmp(name, "vkCmd", 5)) ? VKR_JOURNAL_RECORDING
                                                  : VKR_JOURNAL_TRANSIENT;
   }

   t[VK_COMMAND_TYPE_vkBeginCommandBuffer_EXT] =
      VKR_JOURNAL_RECORDING | VKR_JOURNAL_F_RESETS_PRIOR;
   t[VK_COMMAND_TYPE_vkEndCommandBuffer_EXT] = VKR_JOURNAL_RECORDING;
   t[VK_COMMAND_TYPE_vkResetCommandBuffer_EXT] =
      VKR_JOURNAL_RECORDING | VKR_JOURNAL_F_RESETS_PRIOR;

   t[VK_COMMAND_TYPE_vkResetCommandPool_EXT] = VKR_JOURNAL_POOL_RESET;
   t[VK_COMMAND_TYPE_vkResetDescriptorPool_EXT] = VKR_JOURNAL_POOL_RESET;

   t[VK_COMMAND_TYPE_vkUpdateDescriptorSets_EXT] = VKR_JOURNAL_NOTED;
   t[VK_COMMAND_TYPE_vkUpdateDescriptorSetWithTemplate_EXT] = VKR_JOURNAL_NOTED;
   t[VK_COMMAND_TYPE_vkBindBufferMemory_EXT] = VKR_JOURNAL_NOTED;
   t[VK_COMMAND_TYPE_vkBindBufferMemory2_EXT] = VKR_JOURNAL_NOTED;
   t[VK_COMMAND_TYPE_vkBindImageMemory_EXT] = VKR_JOURNAL_NOTED;
   t[VK_COMMAND_TYPE_vkBindImageMemory2_EXT] = VKR_JOURNAL_NOTED;

   t[VK_COMMAND_TYPE_vkFreeCommandBuffers_EXT] = VKR_JOURNAL_FREE;
   t[VK_COMMAND_TYPE_vkFreeDescriptorSets_EXT] = VKR_JOURNAL_FREE;

   t[VK_COMMAND_TYPE_vkCreateRingMESA_EXT] = VKR_JOURNAL_RING_CREATE;
   t[VK_COMMAND_TYPE_vkDestroyRingMESA_EXT] = VKR_JOURNAL_RING_DESTROY;
   t[VK_COMMAND_TYPE_vkSetReplyCommandStreamMESA_EXT] = VKR_JOURNAL_RING_STREAM;
   t[VK_COMMAND_TYPE_vkSeekReplyCommandStreamMESA_EXT] = VKR_JOURNAL_RING_STREAM;
}

static uint8_t
vkr_journal_classify(VkCommandTypeEXT cmd_type)
{
   call_once(&vkr_journal_class_once, vkr_journal_build_class_table);
   return (uint32_t)cmd_type < VKR_JOURNAL_NUM_CMD_TYPES
             ? vkr_journal_class_table[cmd_type]
             : VKR_JOURNAL_TRANSIENT;
}

/* --- journal lifecycle --- */

struct vkr_journal *
vkr_journal_create(uint32_t ctx_id)
{
   struct vkr_journal *j = calloc(1, sizeof(*j));
   if (!j)
      return NULL;

   j->ctx_id = ctx_id;
   list_inithead(&j->entries);
   j->keys =
      _mesa_hash_table_create(NULL, vkr_journal_hash_u64, vkr_journal_key_u64_equal);
   if (!j->keys) {
      free(j);
      return NULL;
   }
   if (mtx_init(&j->mutex, mtx_plain) != thrd_success) {
      _mesa_hash_table_destroy(j->keys, NULL);
      free(j);
      return NULL;
   }
   j->seq_next = 1;

   list_inithead(&j->q);
   if (mtx_init(&j->q_mutex, mtx_plain) != thrd_success ||
       cnd_init(&j->q_cond) != thrd_success || cnd_init(&j->flush_cond) != thrd_success ||
       thrd_create(&j->thread, vkr_journal_thread, j) != thrd_success) {
      /* no consumer: fall back to fully-synchronous application at push time
       * (vkr_journal_push applies inline when !thread_live) — correct, just slow */
      vkr_log("journal(ctx %u): consumer thread unavailable, applying inline",
              ctx_id);
   } else {
      j->thread_live = true;
   }

   return j;
}

static void
vkr_journal_entry_free_data(struct vkr_journal_entry *e)
{
   free(e->data);
   free(e->refs);
   free(e->key_types);
   free(e->aux);
   free(e->pinned_refs);
   free(e);
}

/* prune ids queued by unpin-drains during a kill sweep — processed iteratively
 * by the sweep's owner (never recursively: a fired prune kills more entries,
 * possibly on keynode lists an outer loop is iterating) */
struct vkr_journal_fires {
   uint64_t *ids;
   uint32_t n, cap;
};

/* unlink from the global list and from every keynode; keynodes emptied by this
 * removal are freed, except `keep` (the node the caller is iterating). Unpins
 * the entry's create-arg refs; a pin draining to zero with a deferred prune
 * queues that prune on `fires` (NULL only at whole-journal teardown). */
static void
vkr_journal_entry_kill_locked(struct vkr_journal *j,
                              struct vkr_journal_entry *e,
                              struct vkr_journal_keynode *keep,
                              struct vkr_journal_fires *fires)
{
   for (uint32_t i = 0; i < e->npinned; i++) {
      struct hash_entry *he = _mesa_hash_table_search(j->keys, &e->pinned_refs[i]);
      if (!he)
         continue;
      struct vkr_journal_keynode *node = he->data;
      if (node->pinned && --node->pinned == 0 && node->prune_deferred && fires)
         vkr_journal_u64_append(&fires->ids, &fires->n, &fires->cap, node->id);
   }

   for (uint32_t i = 0; i < e->nkeys; i++) {
      struct vkr_journal_keyref *ref = &e->refs[i];
      struct vkr_journal_keynode *node = ref->node;
      list_del(&ref->link);
      if (node != keep && list_is_empty(&node->refs)) {
         struct hash_entry *he = _mesa_hash_table_search(j->keys, &node->id);
         if (he)
            _mesa_hash_table_remove(j->keys, he);
         free(node);
      }
   }

   list_del(&e->link);
   j->stats.entries_live--;
   j->stats.bytes_live -= e->size;
   j->stats.pruned_entries++;
   vkr_journal_entry_free_data(e);
}

void
vkr_journal_destroy(struct vkr_journal *j)
{
   if (!j)
      return;

   if (j->thread_live) {
      mtx_lock(&j->q_mutex);
      j->q_stop = true;
      cnd_signal(&j->q_cond);
      mtx_unlock(&j->q_mutex);
      thrd_join(j->thread, NULL);
      /* consumer drained the queue before exiting; free anything left (it only
       * leaves messages behind if it never ran) */
      list_for_each_entry_safe (struct vkr_journal_msg, m, &j->q, link) {
         list_del(&m->link);
         vkr_journal_msg_free(m);
      }
      mtx_destroy(&j->q_mutex);
      cnd_destroy(&j->q_cond);
      cnd_destroy(&j->flush_cond);
   }

   list_for_each_entry_safe (struct vkr_journal_entry, e, &j->entries, link) {
      list_del(&e->link);
      vkr_journal_entry_free_data(e);
   }
   hash_table_foreach (j->keys, he)
      free(he->data);
   _mesa_hash_table_destroy(j->keys, NULL);
   mtx_destroy(&j->mutex);
   free(j);
}

/* --- keyed insert / prune (all under j->mutex) --- */

static struct vkr_journal_keynode *
vkr_journal_keynode_get_locked(struct vkr_journal *j, uint64_t id)
{
   struct hash_entry *he = _mesa_hash_table_search(j->keys, &id);
   if (he)
      return he->data;

   struct vkr_journal_keynode *node = malloc(sizeof(*node));
   if (!node)
      return NULL;
   node->id = id;
   list_inithead(&node->refs);
   _mesa_hash_table_insert(j->keys, &node->id, node);
   return node;
}

/* process prune ids queued by unpin-drains (see vkr_journal_fires); each fired
 * prune may kill entries whose unpins queue further ids — loop until drained */
static void
vkr_journal_prune_fires_locked(struct vkr_journal *j, struct vkr_journal_fires *fires);

/* drop existing entries on key `id` that `pred`-match (RESETS_PRIOR, latest-wins) */
static void
vkr_journal_drop_on_key_locked(struct vkr_journal *j,
                               uint64_t id,
                               uint8_t klass,
                               int match_cmd_type /* -1 = any of klass */)
{
   struct hash_entry *he = _mesa_hash_table_search(j->keys, &id);
   if (!he)
      return;
   struct vkr_journal_keynode *node = he->data;
   struct vkr_journal_fires fires = { 0 };

   /* a latest-wins drop is not a lifetime prune — it proceeds even on a pinned
    * key (the superseding entry is already being inserted in our place) */
   list_for_each_entry_safe (struct vkr_journal_keyref, ref, &node->refs, link) {
      struct vkr_journal_entry *e = ref->entry;
      if ((e->klass & VKR_JOURNAL_CLASS_MASK) != klass)
         continue;
      if (match_cmd_type >= 0 && e->cmd_type != match_cmd_type)
         continue;
      vkr_journal_entry_kill_locked(j, e, node, &fires);
   }

   if (list_is_empty(&node->refs)) {
      struct hash_entry *he2 = _mesa_hash_table_search(j->keys, &node->id);
      if (he2)
         _mesa_hash_table_remove(j->keys, he2);
      free(node);
   }

   vkr_journal_prune_fires_locked(j, &fires);
   free(fires.ids);
}

static struct vkr_journal_entry *
vkr_journal_insert_locked(struct vkr_journal *j,
                          uint8_t klass_flags,
                          VkCommandTypeEXT cmd_type,
                          const uint8_t *data,
                          size_t size,
                          const uint64_t *keys,
                          const VkObjectType *key_types,
                          uint32_t nkeys,
                          const uint64_t *aux,
                          uint32_t naux)
{
   const uint8_t klass = klass_flags & VKR_JOURNAL_CLASS_MASK;

   /* two-lane journal: `data`, `key_types` and `aux` arrive as OWNED buffers from the
    * message (the producer already made the one payload copy on the decode thread);
    * on success their ownership moves into the entry, on NULL return the caller keeps
    * it (vkr_journal_msg_free). `keys` stays read-only. */
   struct vkr_journal_entry *e = calloc(1, sizeof(*e));
   if (!e)
      return NULL;
   e->seq = j->seq_next++;
   e->cmd_type = cmd_type;
   e->klass = klass;
   e->size = size;
   e->data = (uint8_t *)data;

   if (nkeys) {
      e->refs = calloc(nkeys, sizeof(*e->refs));
      if (!e->refs) {
         j->seq_next--;
         free(e);
         return NULL;
      }
      e->key_types = (VkObjectType *)key_types;
   }
   if (naux && aux) {
      e->aux = (uint64_t *)aux;
      e->naux = naux;
   }

   for (uint32_t i = 0; i < nkeys; i++) {
      struct vkr_journal_keynode *node = vkr_journal_keynode_get_locked(j, keys[i]);
      if (!node) {
         /* OOM: entry survives unkeyed-for-this-id (never pruned by it) */
         continue;
      }
      struct vkr_journal_keyref *ref = &e->refs[e->nkeys];
      ref->id = keys[i];
      ref->entry = e;
      ref->node = node;
      list_addtail(&ref->link, &node->refs);
      e->nkeys++;
   }

   list_addtail(&e->link, &j->entries);
   j->stats.entries_live++;
   j->stats.bytes_live += size;

   switch (klass) {
   case VKR_JOURNAL_CREATE:
      j->stats.recorded_creates++;
      break;
   case VKR_JOURNAL_RECORDING:
      j->stats.recorded_recordings++;
      break;
   case VKR_JOURNAL_NOTED:
      j->stats.recorded_noted++;
      break;
   case VKR_JOURNAL_FREE:
      j->stats.recorded_frees++;
      break;
   case VKR_JOURNAL_POOL_RESET:
      j->stats.recorded_pool_resets++;
      break;
   default:
      j->stats.recorded_ring++;
      break;
   }
   return e;
}

/* limina snapshot-replay create-arg closure: pin every object id this CREATE's
 * decode looked up (its wire-arg dependencies — modules, layouts, pools, the
 * device …), so their create entries stay replayable even if the guest legally
 * destroys them while the created object lives. The pins drop when this entry
 * dies (vkr_journal_entry_kill_locked). Self-created ids are skipped; a ref
 * with no keynode (its create was never journaled) is counted and skipped —
 * replay could not have re-created it before this fix either. */
static void
vkr_journal_pin_refs_locked(struct vkr_journal *j,
                            struct vkr_journal_entry *e,
                            const uint64_t *refs,
                            uint32_t nrefs,
                            const uint64_t *created,
                            uint32_t ncreated)
{
   if (!nrefs)
      return;
   uint64_t *pins = malloc(nrefs * sizeof(*pins));
   if (!pins)
      return; /* unpinned refs degrade to the pre-fix (drop-at-replay) behavior */

   uint32_t n = 0;
   for (uint32_t i = 0; i < nrefs; i++) {
      const uint64_t id = refs[i];
      bool skip = false;
      for (uint32_t k = 0; k < n && !skip; k++)
         skip = pins[k] == id; /* dedupe (creates reference few distinct ids) */
      for (uint32_t k = 0; k < ncreated && !skip; k++)
         skip = created[k] == id;
      if (skip)
         continue;
      struct hash_entry *he = _mesa_hash_table_search(j->keys, &id);
      if (!he) {
         j->stats.pin_ref_misses++;
         continue;
      }
      ((struct vkr_journal_keynode *)he->data)->pinned++;
      pins[n++] = id;
   }

   if (!n) {
      free(pins);
      return;
   }
   e->pinned_refs = pins;
   e->npinned = n;
   j->stats.pinned_refs += n;
}

/* the single-key prune body: kill (or defer, when pinned) the entries keyed by
 * `id`, queueing any unpin-drained deferred prunes on `fires`. Returns true when
 * the prune was DEFERRED (the key is pinned by a live create-arg reference). */
static bool
vkr_journal_prune_one_locked(struct vkr_journal *j,
                             uint64_t id,
                             struct vkr_journal_fires *fires)
{
   struct hash_entry *he = _mesa_hash_table_search(j->keys, &id);
   if (!he)
      return false;
   struct vkr_journal_keynode *node = he->data;

   if (node->pinned) {
      node->prune_deferred = true;
      return true;
   }

   list_for_each_entry_safe (struct vkr_journal_keyref, ref, &node->refs, link) {
      struct vkr_journal_entry *e = ref->entry;
      if (ref->dead)
         continue;
      ref->dead = true;
      e->keys_dead++;

      /* a CREATE (batch alloc) lives while any of its ids lives; everything
       * else dies with its first dead key */
      const bool kill = (e->klass == VKR_JOURNAL_CREATE) ? e->keys_dead == e->nkeys
                                                         : true;
      if (kill)
         vkr_journal_entry_kill_locked(j, e, node, fires);
   }

   if (list_is_empty(&node->refs)) {
      struct hash_entry *he2 = _mesa_hash_table_search(j->keys, &node->id);
      if (he2)
         _mesa_hash_table_remove(j->keys, he2);
      free(node);
   }
   return false;
}

static void
vkr_journal_prune_fires_locked(struct vkr_journal *j, struct vkr_journal_fires *fires)
{
   /* fires only queue at pinned==0, so a fired prune can never re-defer; it can
    * only kill further entries whose unpins append to the same list */
   for (uint32_t i = 0; i < fires->n; i++)
      vkr_journal_prune_one_locked(j, fires->ids[i], fires);
}

static void
vkr_journal_prune_key(struct vkr_journal *j, uint64_t id)
{
   struct vkr_journal_fires fires = { 0 };

   mtx_lock(&j->mutex);
   vkr_journal_prune_one_locked(j, id, &fires);
   vkr_journal_prune_fires_locked(j, &fires);
   mtx_unlock(&j->mutex);
   free(fires.ids);
}

/* --- TLS frame helpers --- */

static bool
vkr_journal_u64_append(uint64_t **arr, uint32_t *n, uint32_t *cap, uint64_t v)
{
   if (*n == *cap) {
      uint32_t newcap = *cap ? *cap * 2 : 8;
      uint64_t *p = realloc(*arr, newcap * sizeof(**arr));
      if (!p)
         return false;
      *arr = p;
      *cap = newcap;
   }
   (*arr)[(*n)++] = v;
   return true;
}

static struct vkr_journal *
vkr_journal_from_dispatch(struct vn_dispatch_context *dctx)
{
   if (!vkr_journal_enabled() || !dctx)
      return NULL;
   struct vkr_context *ctx = dctx->data;
   return ctx ? ctx->journal : NULL;
}

/* two-lane journal: frames are REUSED via a per-thread freelist (chained through
 * ->parent) so the per-command decode-path cost is pointer swaps, not calloc/free.
 * Arrays keep their grown capacity across reuse; ownership-transferred arrays are
 * nulled before the frame is pooled. */
static _Thread_local struct vkr_journal_frame *vkr_journal_frame_pool;

void
vkr_journal_pre_dispatch(struct vn_dispatch_context *dctx)
{
   /* The incoming command's type sits at the decoder cursor (vn_dispatch_command
    * decodes it right after this hook). If it is anything but a RECORDING
    * command, drain this thread's batch BEFORE the dispatch starts: several
    * non-recording commands block mid-dispatch (vkWaitRingSeqnoMESA and
    * friends), and pending recording messages must not sit out a block — a
    * snapshot export quiescing the VM at that moment would miss them (the
    * gen-2 half of the 2026-07-28 suite failure). */
   struct vkr_journal_batch *b = &vkr_journal_batch_tls;
   if (b->n) {
      const struct vkr_cs_decoder *dec = (const struct vkr_cs_decoder *)dctx->decoder;
      uint32_t next_type;
      if (dec->cur + sizeof(next_type) > dec->end) {
         vkr_journal_batch_drain(b->j);
      } else {
         memcpy(&next_type, dec->cur, sizeof(next_type));
         if ((vkr_journal_classify((VkCommandTypeEXT)next_type) &
              VKR_JOURNAL_CLASS_MASK) != VKR_JOURNAL_RECORDING)
            vkr_journal_batch_drain(b->j);
      }
   }

   struct vkr_journal *j = vkr_journal_from_dispatch(dctx);
   if (!j)
      return;

   struct vkr_journal_frame *frame = vkr_journal_frame_pool;
   if (frame)
      vkr_journal_frame_pool = frame->parent;
   else
      frame = calloc(1, sizeof(*frame));
   if (!frame)
      return; /* post_dispatch tolerates a missing frame */

   frame->j = j;
   frame->ctx = dctx->data;
   frame->dctx = dctx;
   frame->start = ((struct vkr_cs_decoder *)dctx->decoder)->cur;
   frame->parent = vkr_journal_frame_cur;
   vkr_journal_frame_cur = frame;
}

static void
vkr_journal_frame_free(struct vkr_journal_frame *frame)
{
   /* reset counters, keep array capacity, return to the per-thread pool */
   frame->ncreated = 0;
   frame->nnoted = 0;
   frame->nrefs = 0;
   frame->nremoved = 0;
   frame->nfreed = 0;
   frame->have_free = false;
   frame->free_pool = 0;
   frame->j = NULL;
   frame->ctx = NULL;
   frame->dctx = NULL;
   frame->start = NULL;
   frame->parent = vkr_journal_frame_pool;
   vkr_journal_frame_pool = frame;
}

/* limina snapshot-replay: called from vkr_cs_decoder_lookup_object on every
 * SUCCESSFUL decode-time handle lookup. Outside a dispatch (no TLS frame, or
 * journal off) it is a no-op; the noted ids only matter when the dispatched
 * command turns out to be a CREATE (vkr_journal_pin_refs_locked). */
void
vkr_journal_note_lookup(uint64_t id)
{
   struct vkr_journal_frame *frame = vkr_journal_frame_cur;
   if (!frame || !id)
      return;
   vkr_journal_u64_append(&frame->refs, &frame->nrefs, &frame->refs_cap, id);
}

/* wire peek: header is cmd_type (u32) + flags (u32); every handle a LE u64 */
static bool
vkr_journal_peek_u64(const uint8_t *start, size_t size, size_t offset, uint64_t *out)
{
   if (offset + sizeof(uint64_t) > size)
      return false;
   memcpy(out, start + offset, sizeof(uint64_t));
   return true;
}

/* which ring's decoder is dispatching, 0 = the context decoder */
static uint64_t
vkr_journal_ring_id(struct vkr_context *ctx, struct vn_dispatch_context *dctx)
{
   uint64_t id = 0;
   mtx_lock(&ctx->ring_mutex);
   list_for_each_entry (struct vkr_ring, ring, &ctx->rings, head) {
      if (&ring->dispatch == dctx) {
         id = ring->id;
         break;
      }
   }
   mtx_unlock(&ctx->ring_mutex);
   return id;
}

/* hand a dropped frame's in-dispatch removals to the consumer anyway — the
 * prunes must happen even when the command itself is not retained */
static void
vkr_journal_flush_removed(struct vkr_journal *j, struct vkr_journal_frame *frame)
{
   if (!frame->nremoved)
      return;
   struct vkr_journal_msg *m = calloc(1, sizeof(*m));
   if (!m) {
      p_atomic_inc(&j->dropped_oom_fast);
      return;
   }
   m->type = VKR_JOURNAL_MSG_PRUNE_KEY;
   m->removed = frame->removed;
   m->nremoved = frame->nremoved;
   frame->removed = NULL;
   frame->nremoved = frame->removed_cap = 0;
   vkr_journal_push(j, m);
}

void
vkr_journal_post_dispatch(struct vn_dispatch_context *dctx, VkCommandTypeEXT cmd_type)
{
   struct vkr_journal *j = vkr_journal_from_dispatch(dctx);
   if (!j)
      return;

   struct vkr_journal_frame *frame = vkr_journal_frame_cur;
   if (!frame || frame->j != j)
      return; /* pre_dispatch OOM'd; nothing to pop */
   vkr_journal_frame_cur = frame->parent;

   struct vkr_cs_decoder *dec = (struct vkr_cs_decoder *)dctx->decoder;
   const uint8_t *end = dec->cur;

   if (vkr_cs_decoder_get_fatal(dec) || end <= frame->start) {
      p_atomic_inc(&j->dropped_fatal_fast);
      vkr_journal_flush_removed(j, frame);
      vkr_journal_frame_free(frame);
      return;
   }
   const size_t size = end - frame->start;

   const uint8_t klass_flags = vkr_journal_classify(cmd_type);
   const uint8_t klass = klass_flags & VKR_JOURNAL_CLASS_MASK;

   /* attribution mode: drop the RECORDING lane before any capture work */
   if (klass == VKR_JOURNAL_RECORDING && !frame->ncreated && vkr_journal_mode() == 2) {
      vkr_journal_flush_removed(j, frame);
      vkr_journal_frame_free(frame);
      return;
   }

   /* two-lane journal, decode-thread lane: the fast path — the overwhelmingly common
    * transient command retains nothing and touches NO lock, queue, or allocator. */
   if (klass == VKR_JOURNAL_TRANSIENT && !frame->ncreated && !frame->nremoved) {
      p_atomic_inc(&j->transient_fast);
      vkr_journal_frame_free(frame);
      return;
   }

   /* everything else: capture (one payload copy + ownership transfer of the frame's
    * arrays) into a message; retention happens on the journal thread in queue order */
   struct vkr_journal_msg *m = calloc(1, sizeof(*m));
   if (!m) {
      p_atomic_inc(&j->dropped_oom_fast);
      vkr_journal_frame_free(frame);
      return;
   }
   m->type = VKR_JOURNAL_MSG_INSERT;
   m->cmd_type = cmd_type;
   m->size = size;
   if (size <= VKR_JOURNAL_INLINE_DATA) {
      /* every hot vkCmd* fits here; the consumer materializes the heap copy */
      m->data = m->inline_data;
      m->data_inline = true;
   } else {
      m->data = malloc(size);
      if (!m->data) {
         p_atomic_inc(&j->dropped_oom_fast);
         free(m);
         vkr_journal_frame_free(frame);
         return;
      }
   }
   memcpy(m->data, frame->start, size);

   bool retain = false;
   if (frame->ncreated) {
      /* effect-driven CREATE: keyed by every added id; the frame's created/refs
       * arrays move into the message (created ids double as the pin-dedupe set) */
      m->klass = VKR_JOURNAL_CREATE;
      m->keys = frame->created;
      m->key_types = frame->created_types;
      m->nkeys = frame->ncreated;
      m->refs = frame->refs;
      m->nrefs = frame->nrefs;
      frame->created = NULL;
      frame->created_types = NULL;
      frame->ncreated = frame->created_cap = 0;
      frame->refs = NULL;
      frame->nrefs = frame->refs_cap = 0;
      retain = true;
   } else {
      switch (klass) {
      case VKR_JOURNAL_RECORDING: {
         uint64_t cmd_buf;
         if (vkr_journal_peek_u64(m->data, size, 8, &cmd_buf)) {
            m->klass = klass;
            m->inline_key = cmd_buf;
            m->keys = &m->inline_key;
            m->keys_inline = true;
            m->nkeys = 1;
            if (klass_flags & VKR_JOURNAL_F_RESETS_PRIOR) {
               m->drop_first = true;
               m->drop_klass = VKR_JOURNAL_RECORDING;
               m->drop_cmd_type = -1;
               m->drop_key = cmd_buf;
            }
            retain = true;
         }
         break;
      }
      case VKR_JOURNAL_NOTED:
         if (frame->nnoted) {
            m->klass = klass;
            m->keys = frame->noted;
            m->nkeys = frame->nnoted;
            frame->noted = NULL;
            frame->nnoted = frame->noted_cap = 0;
            retain = true;
         }
         break;
      case VKR_JOURNAL_FREE:
         if (frame->have_free) {
            m->klass = klass;
            m->inline_key = frame->free_pool;
            m->keys = &m->inline_key;
            m->keys_inline = true;
            m->nkeys = 1;
            m->aux = frame->freed;
            m->naux = frame->nfreed;
            frame->freed = NULL;
            frame->nfreed = frame->freed_cap = 0;
            retain = true;
         }
         break;
      case VKR_JOURNAL_POOL_RESET:
      case VKR_JOURNAL_RING_CREATE: {
         uint64_t key;
         const size_t off = (klass == VKR_JOURNAL_POOL_RESET) ? 16 : 8;
         if (vkr_journal_peek_u64(m->data, size, off, &key)) {
            m->klass = klass;
            m->inline_key = key;
            m->keys = &m->inline_key;
            m->keys_inline = true;
            m->nkeys = 1;
            retain = true;
         }
         break;
      }
      case VKR_JOURNAL_RING_DESTROY: {
         uint64_t ring;
         if (vkr_journal_peek_u64(m->data, size, 8, &ring)) {
            m->type = VKR_JOURNAL_MSG_PRUNE_KEY;
            m->id = ring;
            if (!m->data_inline)
               free(m->data);
            m->data = NULL;
            m->data_inline = false;
            m->size = 0;
            retain = true;
         }
         break;
      }
      case VKR_JOURNAL_RING_STREAM: {
         const uint64_t ring = vkr_journal_ring_id(frame->ctx, dctx);
         m->klass = klass;
         m->inline_key = ring;
         m->keys = &m->inline_key;
         m->keys_inline = true;
         m->nkeys = 1;
         m->drop_first = true;
         m->drop_klass = VKR_JOURNAL_RING_STREAM;
         m->drop_cmd_type = (int32_t)cmd_type;
         m->drop_key = ring;
         retain = true;
         break;
      }
      default:
         /* transient that removed objects (we're only here when nremoved != 0
          * — see the fast path). Whether it is retained is the consumer's
          * call: a prune deferred by a pin — e.g. vkFreeMemory of a
          * blob-exported memory — retains this command NOTED-keyed by the
          * pinned id, so replay re-runs it after the blob create and the
          * final prune at unpin kills it together with the alloc entry. */
         m->klass = VKR_JOURNAL_TRANSIENT;
         break;
      }
   }

   /* in-dispatch object removals ride the same message; the consumer prunes
    * them (in queue order, so after the creates they refer to) BEFORE the
    * insert — matching the old in-dispatch prune timing */
   if (frame->nremoved) {
      m->removed = frame->removed;
      m->nremoved = frame->nremoved;
      frame->removed = NULL;
      frame->nremoved = frame->removed_cap = 0;
      retain = true;
   }

   if (retain)
      vkr_journal_push(j, m);
   else
      vkr_journal_msg_free(m);
   vkr_journal_frame_free(frame);
}

/* --- object-table hooks (called under ctx->object_mutex) --- */

void
vkr_journal_object_added(struct vkr_context *ctx, uint64_t id, VkObjectType type)
{
   struct vkr_journal *j = ctx->journal;
   if (!j)
      return;

   struct vkr_journal_frame *frame = vkr_journal_frame_cur;
   if (!frame || frame->j != j) {
      p_atomic_inc(&j->orphan_adds_fast);
      return;
   }

   uint32_t typecap = frame->created_cap;
   if (!vkr_journal_u64_append(&frame->created, &frame->ncreated, &frame->created_cap,
                               id))
      return;
   if (frame->created_cap != typecap) {
      VkObjectType *p = realloc(frame->created_types,
                                frame->created_cap * sizeof(*frame->created_types));
      if (p)
         frame->created_types = p;
   }
   if (frame->created_types)
      frame->created_types[frame->ncreated - 1] = type;
}

void
vkr_journal_object_removed(struct vkr_context *ctx, uint64_t id)
{
   struct vkr_journal *j = ctx->journal;
   if (!j)
      return;

   /* mid-dispatch: just record the id in the frame — the prune rides the
    * command's own message and runs on the consumer, in queue order (so
    * after the queued create it refers to). No lock on the decode path. */
   struct vkr_journal_frame *frame = vkr_journal_frame_cur;
   if (frame && frame->j == j) {
      vkr_journal_u64_append(&frame->removed, &frame->nremoved, &frame->removed_cap,
                             id);
      return;
   }

   struct vkr_journal_msg *m = calloc(1, sizeof(*m));
   if (!m) {
      p_atomic_inc(&j->dropped_oom_fast);
      return;
   }
   m->type = VKR_JOURNAL_MSG_PRUNE_KEY;
   m->id = id;
   vkr_journal_push(j, m);
}

/* limina snapshot-replay: see the pin comment on vkr_journal_keynode. Returns
 * false when there is nothing to pin (journal off, or the key was never
 * journaled — then replay could not re-create the memory anyway). */
static bool
vkr_journal_pin_one_locked(struct vkr_journal *j, uint64_t id, uint64_t dep_id)
{
   struct hash_entry *he = _mesa_hash_table_search(j->keys, &id);
   if (!he) {
      vkr_log("journal: pin MISS key %" PRIu64 " (never journaled)", id);
      return false;
   }
   struct vkr_journal_keynode *node = he->data;
   node->pinned++;
   if (dep_id)
      node->dep_id = dep_id;
   return true;
}


bool
vkr_journal_pin_key(struct vkr_context *ctx, uint64_t id, uint64_t dep_id)
{
   struct vkr_journal *j = ctx->journal;
   if (!j)
      return false;

   /* applied on the consumer in queue order — after the queued create of the
    * pinned key. The return value only says "the pin is on its way" (both
    * call sites ignore it); a pin MISS is logged by the consumer. */
   struct vkr_journal_msg *m = calloc(1, sizeof(*m));
   if (!m) {
      p_atomic_inc(&j->dropped_oom_fast);
      return false;
   }
   m->type = VKR_JOURNAL_MSG_PIN;
   m->id = id;
   m->dep_id = dep_id;
   vkr_journal_push(j, m);
   return true;
}

static void
vkr_journal_unpin_one(struct vkr_journal *j, uint64_t id, uint64_t *out_dep)
{
   bool fire = false;
   mtx_lock(&j->mutex);
   struct hash_entry *he = _mesa_hash_table_search(j->keys, &id);
   if (he) {
      struct vkr_journal_keynode *node = he->data;
      if (out_dep) {
         *out_dep = node->dep_id;
         node->dep_id = 0;
      }
      if (node->pinned && --node->pinned == 0 && node->prune_deferred)
         fire = true;
   } else if (out_dep) {
      *out_dep = 0;
   }
   mtx_unlock(&j->mutex);

   if (fire)
      vkr_journal_prune_key(j, id);
}

void
vkr_journal_unpin_key(struct vkr_context *ctx, uint64_t id)
{
   struct vkr_journal *j = ctx->journal;
   if (!j)
      return;

   struct vkr_journal_msg *m = calloc(1, sizeof(*m));
   if (!m) {
      p_atomic_inc(&j->dropped_oom_fast);
      return;
   }
   m->type = VKR_JOURNAL_MSG_UNPIN;
   m->id = id;
   vkr_journal_push(j, m);
}

/* --- the consumer lane (two-lane journal) ------------------------------------------
 * Everything below applies messages IN QUEUE ORDER on the journal thread; the only
 * other holders of j->mutex are the quiesced snapshot-time readers. */

static void
vkr_journal_msg_free(struct vkr_journal_msg *m)
{
   if (!m->data_inline)
      free(m->data);
   if (!m->keys_inline)
      free(m->keys);
   free(m->key_types);
   free(m->aux);
   free(m->refs);
   free(m->removed);
   free(m);
}

static void
vkr_journal_msg_apply(struct vkr_journal *j, struct vkr_journal_msg *m)
{
   switch (m->type) {
   case VKR_JOURNAL_MSG_INSERT: {
      mtx_lock(&j->mutex);

      /* in-dispatch removals first (the old code pruned them mid-dispatch,
       * before the command's own retention). A prune deferred by a pin
       * retains an otherwise-transient command keyed by the pinned id. */
      uint64_t *def = NULL;
      uint32_t ndef = 0, defcap = 0;
      if (m->nremoved) {
         struct vkr_journal_fires fires = { 0 };
         for (uint32_t i = 0; i < m->nremoved; i++) {
            if (vkr_journal_prune_one_locked(j, m->removed[i], &fires))
               vkr_journal_u64_append(&def, &ndef, &defcap, m->removed[i]);
         }
         vkr_journal_prune_fires_locked(j, &fires);
         free(fires.ids);
      }
      if (!m->nkeys && ndef && m->klass == VKR_JOURNAL_TRANSIENT) {
         /* e.g. vkFreeMemory of a blob-exported memory: retain it so replay
          * re-runs it after the blob create (its seq is greater than the
          * blob's record fence); the final prune at unpin kills it together
          * with the alloc entry */
         m->klass = VKR_JOURNAL_NOTED;
         m->keys = def;
         m->nkeys = ndef;
         def = NULL;
      }
      free(def);

      if (m->drop_first)
         vkr_journal_drop_on_key_locked(j, m->drop_key, m->drop_klass, m->drop_cmd_type);

      /* an inline payload lives in the message; the entry needs its own heap
       * copy (ownership of `data` moves in). Materializing HERE is the point:
       * the malloc runs on the consumer, not the decode thread. */
      if (m->nkeys && m->data_inline) {
         uint8_t *heap = malloc(m->size);
         if (heap) {
            memcpy(heap, m->inline_data, m->size);
            m->data = heap;
            m->data_inline = false;
         } else {
            p_atomic_inc(&j->dropped_oom_fast);
         }
      }
      if (m->nkeys && !m->data_inline) {
         if (m->klass == VKR_JOURNAL_NOTED && m->nkeys > 1)
            j->stats.noted_multi_key++;
         struct vkr_journal_entry *e =
            vkr_journal_insert_locked(j, m->klass, m->cmd_type, m->data, m->size,
                                      m->keys, m->key_types, m->nkeys, m->aux, m->naux);
         if (e) {
            /* data/key_types/aux ownership moved into the entry */
            m->data = NULL;
            m->key_types = NULL;
            m->aux = NULL;
            if (m->nrefs)
               vkr_journal_pin_refs_locked(j, e, m->refs, m->nrefs, m->keys, m->nkeys);
         }
      } else if (m->klass == VKR_JOURNAL_TRANSIENT) {
         j->stats.transient_cmds++;
      }

      mtx_unlock(&j->mutex);
      break;
   }
   case VKR_JOURNAL_MSG_PRUNE_KEY:
      if (m->id)
         vkr_journal_prune_key(j, m->id);
      for (uint32_t i = 0; i < m->nremoved; i++)
         vkr_journal_prune_key(j, m->removed[i]);
      break;
   case VKR_JOURNAL_MSG_PIN:
      mtx_lock(&j->mutex);
      if (vkr_journal_pin_one_locked(j, m->id, m->dep_id) && m->dep_id)
         vkr_journal_pin_one_locked(j, m->dep_id, 0);
      mtx_unlock(&j->mutex);
      break;
   case VKR_JOURNAL_MSG_UNPIN: {
      uint64_t dep_id = 0;
      vkr_journal_unpin_one(j, m->id, &dep_id);
      if (dep_id)
         vkr_journal_unpin_one(j, dep_id, NULL);
      break;
   }
   case VKR_JOURNAL_MSG_FLUSH:
      mtx_lock(&j->q_mutex);
      j->flush_gen_done = m->flush_gen;
      cnd_broadcast(&j->flush_cond);
      mtx_unlock(&j->q_mutex);
      break;
   default:
      break;
   }
}

static int
vkr_journal_thread(void *arg)
{
   struct vkr_journal *j = arg;
   u_thread_setname("vkr-journal");

   mtx_lock(&j->q_mutex);
   while (true) {
      while (list_is_empty(&j->q) && !j->q_stop)
         cnd_wait(&j->q_cond, &j->q_mutex);
      if (list_is_empty(&j->q) && j->q_stop)
         break;
      struct vkr_journal_msg *m =
         list_first_entry(&j->q, struct vkr_journal_msg, link);
      list_del(&m->link);
      j->q_depth--;
      mtx_unlock(&j->q_mutex);

      vkr_journal_msg_apply(j, m);
      vkr_journal_msg_free(m);

      mtx_lock(&j->q_mutex);
   }
   mtx_unlock(&j->q_mutex);
   return 0;
}

static void
vkr_journal_push_now(struct vkr_journal *j, struct vkr_journal_msg *m)
{
   if (!j->thread_live) {
      /* no consumer (thread creation failed): apply inline, original behavior */
      vkr_journal_msg_apply(j, m);
      vkr_journal_msg_free(m);
      return;
   }
   mtx_lock(&j->q_mutex);
   list_addtail(&m->link, &j->q);
   if (++j->q_depth > j->q_depth_peak)
      j->q_depth_peak = j->q_depth;
   cnd_signal(&j->q_cond);
   mtx_unlock(&j->q_mutex);
}

/* Only RECORDING-class inserts may sit in a batch. Everything else can carry a
 * CROSS-THREAD causal edge that per-command pushes ordered by real time and a
 * batch would reorder: a CREATE is pinned from the virtqueue worker the moment
 * the guest learns the command was consumed (blob create against a fresh
 * VkDeviceMemory), and a RING_CREATE decoded on the context thread starts a
 * ring thread whose very first journaled command (reply-stream set) must land
 * in the queue AFTER it — the 2026-07-28 suite caught exactly that: the
 * RING_STREAM entry serialized with a smaller seq than its ring's create, and
 * restore dropped it as unreplayable. RECORDING entries have no cross-thread
 * dependents in valid usage (a command buffer cannot be recorded and consumed
 * from two threads at once), so per-thread program order — which the batch
 * preserves — is enough for them. */
static bool
vkr_journal_msg_batchable(const struct vkr_journal_msg *m)
{
   return m->type == VKR_JOURNAL_MSG_INSERT &&
          (m->klass & VKR_JOURNAL_CLASS_MASK) == VKR_JOURNAL_RECORDING &&
          !m->nremoved;
}

static void
vkr_journal_push(struct vkr_journal *j, struct vkr_journal_msg *m)
{
   struct vkr_journal_batch *b = &vkr_journal_batch_tls;
   if (b->depth <= 0) {
      vkr_journal_push_now(j, m);
      return;
   }

   /* a different journal's message has no order contract with the batch, but
    * flushing keeps this thread's queue arrivals in program order regardless */
   if (b->n && b->j != j)
      vkr_journal_batch_drain(b->j);

   if (!vkr_journal_msg_batchable(m)) {
      /* per-thread order: everything batched so far must reach the queue first */
      if (b->n)
         vkr_journal_batch_drain(b->j);
      vkr_journal_push_now(j, m);
      return;
   }

   b->j = j;
   list_addtail(&m->link, &b->msgs);
   b->n++;
}

/* drain the queue: on return every message pushed before the call is applied.
 * Callers are the snapshot-time readers (export/seq/stats/dump) on a quiesced VM. */
static void
vkr_journal_quiesce(struct vkr_journal *j)
{
   if (!j->thread_live)
      return;
   struct vkr_journal_msg *m = calloc(1, sizeof(*m));
   if (!m)
      return;
   m->type = VKR_JOURNAL_MSG_FLUSH;
   mtx_lock(&j->q_mutex);
   m->flush_gen = ++j->flush_gen_next;
   const uint64_t gen = m->flush_gen;
   list_addtail(&m->link, &j->q);
   if (++j->q_depth > j->q_depth_peak)
      j->q_depth_peak = j->q_depth;
   cnd_signal(&j->q_cond);
   while (j->flush_gen_done < gen)
      cnd_wait(&j->flush_cond, &j->q_mutex);
   mtx_unlock(&j->q_mutex);
}

/* --- in-handler attribution --- */

void
vkr_journal_note_keys(struct vkr_context *ctx, const uint64_t *ids, uint32_t count)
{
   struct vkr_journal *j = ctx->journal;
   if (!j)
      return;
   struct vkr_journal_frame *frame = vkr_journal_frame_cur;
   if (!frame || frame->j != j)
      return;

   for (uint32_t i = 0; i < count; i++) {
      /* dedupe: zink updates typically hit one set many writes */
      bool seen = false;
      for (uint32_t k = 0; k < frame->nnoted; k++) {
         if (frame->noted[k] == ids[i]) {
            seen = true;
            break;
         }
      }
      if (!seen)
         vkr_journal_u64_append(&frame->noted, &frame->nnoted, &frame->noted_cap,
                                ids[i]);
   }
}

void
vkr_journal_note_free(struct vkr_context *ctx,
                      uint64_t pool_id,
                      const uint64_t *ids,
                      uint32_t count)
{
   struct vkr_journal *j = ctx->journal;
   if (!j)
      return;
   struct vkr_journal_frame *frame = vkr_journal_frame_cur;
   if (!frame || frame->j != j)
      return;

   frame->have_free = true;
   frame->free_pool = pool_id;
   for (uint32_t i = 0; i < count; i++)
      vkr_journal_u64_append(&frame->freed, &frame->nfreed, &frame->freed_cap, ids[i]);
}

/* --- stats / census --- */

/* fold the decode-thread atomic counters into a stats copy */
static void
vkr_journal_merge_fast_stats(struct vkr_journal *j, struct vkr_journal_stats *out)
{
   out->transient_cmds += p_atomic_read(&j->transient_fast);
   out->orphan_adds += p_atomic_read(&j->orphan_adds_fast);
   out->dropped_fatal += p_atomic_read(&j->dropped_fatal_fast);
   /* OOM-dropped messages have no dedicated stats field (the struct crosses
    * into libkrun); count them with the fatal drops — both mean "a command
    * the journal failed to consider" */
   out->dropped_fatal += p_atomic_read(&j->dropped_oom_fast);
}

void
vkr_journal_get_stats(struct vkr_journal *j, struct vkr_journal_stats *out)
{
   vkr_journal_quiesce(j);
   mtx_lock(&j->mutex);
   *out = j->stats;
   mtx_unlock(&j->mutex);
   vkr_journal_merge_fast_stats(j, out);
}

uint64_t
vkr_journal_seq(struct vkr_journal *j)
{
   vkr_journal_quiesce(j);
   mtx_lock(&j->mutex);
   const uint64_t seq = j->seq_next - 1;
   mtx_unlock(&j->mutex);
   return seq;
}

bool
vkr_journal_export(struct vkr_journal *j, void **out_buf, size_t *out_size)
{
   vkr_journal_quiesce(j);
   mtx_lock(&j->mutex);

   size_t total = 4 * sizeof(uint32_t);
   uint32_t count = 0;
   list_for_each_entry (struct vkr_journal_entry, e, &j->entries, link) {
      total += sizeof(uint64_t) + sizeof(uint32_t) + 4 + sizeof(uint64_t) +
               sizeof(uint32_t) + ALIGN_POT((size_t)e->size, 4);
      count++;
   }

   uint8_t *buf = malloc(total);
   if (!buf) {
      mtx_unlock(&j->mutex);
      return false;
   }

   uint8_t *p = buf;
#define VKR_JOURNAL_PUT(val)                                                             \
   do {                                                                                  \
      memcpy(p, &(val), sizeof(val));                                                    \
      p += sizeof(val);                                                                  \
   } while (0)
   const uint32_t magic = VKR_JOURNAL_EXPORT_MAGIC;
   const uint32_t version = VKR_JOURNAL_EXPORT_VERSION;
   const uint32_t reserved = 0;
   VKR_JOURNAL_PUT(magic);
   VKR_JOURNAL_PUT(version);
   VKR_JOURNAL_PUT(count);
   VKR_JOURNAL_PUT(reserved);

   list_for_each_entry (struct vkr_journal_entry, e, &j->entries, link) {
      const uint32_t cmd_type = e->cmd_type;
      const uint8_t klass = e->klass;
      const uint8_t pad[3] = { 0 };
      /* ring-scoped entries carry the ring id so the replayer can route them to
       * that ring's decoder; everything else replays on the context decoder */
      const uint64_t ring_key =
         (e->klass == VKR_JOURNAL_RING_STREAM || e->klass == VKR_JOURNAL_RING_CREATE) &&
               e->nkeys
            ? e->refs[0].id
            : 0;
      const uint32_t size = e->size;
      VKR_JOURNAL_PUT(e->seq);
      VKR_JOURNAL_PUT(cmd_type);
      VKR_JOURNAL_PUT(klass);
      memcpy(p, pad, sizeof(pad));
      p += sizeof(pad);
      VKR_JOURNAL_PUT(ring_key);
      VKR_JOURNAL_PUT(size);
      memcpy(p, e->data, e->size);
      p += e->size;
      const size_t padding = ALIGN_POT((size_t)e->size, 4) - e->size;
      memset(p, 0, padding);
      p += padding;
   }
#undef VKR_JOURNAL_PUT

   assert((size_t)(p - buf) == total);
   mtx_unlock(&j->mutex);

   *out_buf = buf;
   *out_size = total;
   return true;
}

void
vkr_journal_dump(struct vkr_journal *j)
{
   if (!j) {
      vkr_log("journal: disabled");
      return;
   }

   vkr_journal_quiesce(j);
   mtx_lock(&j->mutex);

   struct vkr_journal_stats merged = j->stats;
   vkr_journal_merge_fast_stats(j, &merged);
   const struct vkr_journal_stats *s = &merged;

   uint32_t q_now = 0, q_peak = 0;
   if (j->thread_live) {
      mtx_lock(&j->q_mutex);
      q_now = j->q_depth;
      q_peak = j->q_depth_peak;
      mtx_unlock(&j->q_mutex);
   }

   vkr_log("journal ctx %u: %" PRIu64 " entries (%" PRIu64 " KiB) live; recorded "
           "create=%" PRIu64 " recording=%" PRIu64 " noted=%" PRIu64 " ring=%" PRIu64
           " free=%" PRIu64 " pool_reset=%" PRIu64 "; pruned=%" PRIu64
           " transient=%" PRIu64 " orphan_adds=%" PRIu64 " dropped_fatal=%" PRIu64
           " noted_multi_key=%" PRIu64 " pinned_refs=%" PRIu64 " pin_ref_misses=%" PRIu64
           " q_peak=%u q_now=%u dropped_oom=%u",
           j->ctx_id, s->entries_live, s->bytes_live / 1024, s->recorded_creates,
           s->recorded_recordings, s->recorded_noted, s->recorded_ring,
           s->recorded_frees, s->recorded_pool_resets, s->pruned_entries,
           s->transient_cmds, s->orphan_adds, s->dropped_fatal, s->noted_multi_key,
           s->pinned_refs, s->pin_ref_misses, q_peak, q_now,
           p_atomic_read(&j->dropped_oom_fast));

   /* live-create census by VkObjectType — cross-check against the context's
    * object-table tally in vkr_renderer_dump_state */
   struct {
      VkObjectType type;
      uint32_t count;
   } tally[48];
   uint32_t ntally = 0;

   list_for_each_entry (struct vkr_journal_entry, e, &j->entries, link) {
      if (e->klass != VKR_JOURNAL_CREATE || !e->key_types)
         continue;
      for (uint32_t i = 0; i < e->nkeys; i++) {
         if (e->refs[i].dead)
            continue;
         const VkObjectType type = e->key_types[i];
         uint32_t k;
         for (k = 0; k < ntally; k++) {
            if (tally[k].type == type)
               break;
         }
         if (k == ntally && ntally < ARRAY_SIZE(tally)) {
            tally[ntally].type = type;
            tally[ntally].count = 0;
            ntally++;
         }
         if (k < ntally)
            tally[k].count++;
      }
   }

   char buf[512];
   int off = 0;
   for (uint32_t k = 0; k < ntally && off < (int)sizeof(buf) - 32; k++) {
      const char *name = vkr_object_type_name(tally[k].type);
      if (name)
         off += snprintf(buf + off, sizeof(buf) - off, " %s=%u", name, tally[k].count);
      else
         off += snprintf(buf + off, sizeof(buf) - off, " type%u=%u", tally[k].type,
                         tally[k].count);
   }
   vkr_log("journal ctx %u live creates by type:%s", j->ctx_id, ntally ? buf : " none");

   mtx_unlock(&j->mutex);
}
