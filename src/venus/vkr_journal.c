/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * gkvm: venus snapshot-replay re-creation journal. See vkr_journal.h for the
 * model and limina docs/design/venus-snapshot-replay.md for the design.
 */

#include "vkr_journal.h"

#include <stdlib.h>
#include <string.h>

#define XXH_INLINE_ALL
#include "util/xxhash.h"

#include "vkr_common.h"
#include "vkr_context.h"
#include "vkr_cs.h"
#include "vkr_renderer.h"
#include "vkr_ring.h"

#include "venus-protocol/vn_protocol_renderer.h"

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

static bool
vkr_journal_u64_append(uint64_t **arr, uint32_t *n, uint32_t *cap, uint64_t v);

struct vkr_journal_keynode {
   uint64_t id;
   struct list_head refs; /* vkr_journal_keyref.link */
   /* gkvm snapshot-replay: entry pinning. A blob resource created from a
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

   /* gkvm snapshot-replay: create-arg closure (the 2026-07-20 vkmark crash).
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

   /* keys whose prune this command deferred (pinned; see vkr_journal_keynode) */
   uint64_t *deferred;
   uint32_t ndeferred, deferred_cap;

   bool have_free;
   uint64_t free_pool;
   uint64_t *freed;
   uint32_t nfreed, freed_cap;

   struct vkr_journal_frame *parent;
};

static _Thread_local struct vkr_journal_frame *vkr_journal_frame_cur;

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

/* gkvm (2026-07-27, decode-path cost attribution): VKR_JOURNAL=norecord keeps the
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

   struct vkr_journal_entry *e = calloc(1, sizeof(*e));
   if (!e)
      return NULL;
   e->seq = j->seq_next++;
   e->cmd_type = cmd_type;
   e->klass = klass;
   e->size = size;
   e->data = malloc(size);
   if (!e->data) {
      free(e);
      return NULL;
   }
   memcpy(e->data, data, size);

   if (nkeys) {
      e->refs = calloc(nkeys, sizeof(*e->refs));
      if (!e->refs) {
         vkr_journal_entry_free_data(e);
         return NULL;
      }
      if (key_types) {
         e->key_types = malloc(nkeys * sizeof(*e->key_types));
         if (e->key_types)
            memcpy(e->key_types, key_types, nkeys * sizeof(*e->key_types));
      }
   }
   if (naux && aux) {
      e->aux = malloc(naux * sizeof(*e->aux));
      if (e->aux) {
         memcpy(e->aux, aux, naux * sizeof(*e->aux));
         e->naux = naux;
      }
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

/* gkvm snapshot-replay create-arg closure: pin every object id this CREATE's
 * decode looked up (its wire-arg dependencies — modules, layouts, pools, the
 * device …), so their create entries stay replayable even if the guest legally
 * destroys them while the created object lives. The pins drop when this entry
 * dies (vkr_journal_entry_kill_locked). Self-created ids are skipped; a ref
 * with no keynode (its create was never journaled) is counted and skipped —
 * replay could not have re-created it before this fix either. */
static void
vkr_journal_pin_refs_locked(struct vkr_journal *j,
                            struct vkr_journal_entry *e,
                            const struct vkr_journal_frame *frame)
{
   if (!frame->nrefs)
      return;
   uint64_t *pins = malloc(frame->nrefs * sizeof(*pins));
   if (!pins)
      return; /* unpinned refs degrade to the pre-fix (drop-at-replay) behavior */

   uint32_t n = 0;
   for (uint32_t i = 0; i < frame->nrefs; i++) {
      const uint64_t id = frame->refs[i];
      bool skip = false;
      for (uint32_t k = 0; k < n && !skip; k++)
         skip = pins[k] == id; /* dedupe (creates reference few distinct ids) */
      for (uint32_t k = 0; k < frame->ncreated && !skip; k++)
         skip = frame->created[k] == id;
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
   const bool deferred = vkr_journal_prune_one_locked(j, id, &fires);
   vkr_journal_prune_fires_locked(j, &fires);
   mtx_unlock(&j->mutex);
   free(fires.ids);

   if (deferred) {
      /* retain the pruning command (frame active ⇒ we are mid-dispatch of the
       * free on this thread) so replay re-runs it after the dependent create */
      struct vkr_journal_frame *frame = vkr_journal_frame_cur;
      if (frame && frame->j == j)
         vkr_journal_u64_append(&frame->deferred, &frame->ndeferred,
                                &frame->deferred_cap, id);
   }
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

void
vkr_journal_pre_dispatch(struct vn_dispatch_context *dctx)
{
   struct vkr_journal *j = vkr_journal_from_dispatch(dctx);
   if (!j)
      return;

   struct vkr_journal_frame *frame = calloc(1, sizeof(*frame));
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
   free(frame->created);
   free(frame->created_types);
   free(frame->noted);
   free(frame->refs);
   free(frame->deferred);
   free(frame->freed);
   free(frame);
}

/* gkvm snapshot-replay: called from vkr_cs_decoder_lookup_object on every
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
      mtx_lock(&j->mutex);
      j->stats.dropped_fatal++;
      mtx_unlock(&j->mutex);
      vkr_journal_frame_free(frame);
      return;
   }
   const size_t size = end - frame->start;

   const uint8_t klass_flags = vkr_journal_classify(cmd_type);
   const uint8_t klass = klass_flags & VKR_JOURNAL_CLASS_MASK;

   /* attribution mode: drop the RECORDING lane before it touches the mutex */
   if (klass == VKR_JOURNAL_RECORDING && !frame->ncreated && vkr_journal_mode() == 2) {
      vkr_journal_frame_free(frame);
      return;
   }

   mtx_lock(&j->mutex);

   if (frame->ncreated) {
      /* effect-driven: this command built objects — retain keyed by them all,
       * pinning the creates of every object its args reference (closure) */
      struct vkr_journal_entry *e = vkr_journal_insert_locked(
         j, VKR_JOURNAL_CREATE, cmd_type, frame->start, size, frame->created,
         frame->created_types, frame->ncreated, NULL, 0);
      if (e)
         vkr_journal_pin_refs_locked(j, e, frame);
   } else {
      switch (klass) {
      case VKR_JOURNAL_RECORDING: {
         uint64_t cmd_buf;
         if (vkr_journal_peek_u64(frame->start, size, 8, &cmd_buf)) {
            if (klass_flags & VKR_JOURNAL_F_RESETS_PRIOR)
               vkr_journal_drop_on_key_locked(j, cmd_buf, VKR_JOURNAL_RECORDING, -1);
            vkr_journal_insert_locked(j, klass, cmd_type, frame->start, size, &cmd_buf,
                                      NULL, 1, NULL, 0);
         }
         break;
      }
      case VKR_JOURNAL_NOTED:
         if (frame->nnoted) {
            if (frame->nnoted > 1)
               j->stats.noted_multi_key++;
            vkr_journal_insert_locked(j, klass, cmd_type, frame->start, size,
                                      frame->noted, NULL, frame->nnoted, NULL, 0);
         }
         break;
      case VKR_JOURNAL_FREE:
         if (frame->have_free)
            vkr_journal_insert_locked(j, klass, cmd_type, frame->start, size,
                                      &frame->free_pool, NULL, 1, frame->freed,
                                      frame->nfreed);
         break;
      case VKR_JOURNAL_POOL_RESET: {
         uint64_t pool;
         if (vkr_journal_peek_u64(frame->start, size, 16, &pool))
            vkr_journal_insert_locked(j, klass, cmd_type, frame->start, size, &pool,
                                      NULL, 1, NULL, 0);
         break;
      }
      case VKR_JOURNAL_RING_CREATE: {
         uint64_t ring;
         if (vkr_journal_peek_u64(frame->start, size, 8, &ring))
            vkr_journal_insert_locked(j, klass, cmd_type, frame->start, size, &ring,
                                      NULL, 1, NULL, 0);
         break;
      }
      case VKR_JOURNAL_RING_DESTROY: {
         uint64_t ring;
         mtx_unlock(&j->mutex);
         if (vkr_journal_peek_u64(frame->start, size, 8, &ring))
            vkr_journal_prune_key(j, ring);
         goto out_free;
      }
      case VKR_JOURNAL_RING_STREAM: {
         mtx_unlock(&j->mutex);
         const uint64_t ring = vkr_journal_ring_id(frame->ctx, dctx);
         mtx_lock(&j->mutex);
         vkr_journal_drop_on_key_locked(j, ring, VKR_JOURNAL_RING_STREAM, cmd_type);
         vkr_journal_insert_locked(j, klass, cmd_type, frame->start, size, &ring, NULL,
                                   1, NULL, 0);
         break;
      }
      default:
         if (frame->ndeferred) {
            /* this (otherwise transient) command pruned a pinned key — e.g.
             * vkFreeMemory of a blob-exported memory. Retain it keyed by the
             * pinned id: replay re-runs it after the blob create (its seq is
             * greater than the blob's record fence), and the final prune at
             * unpin kills it together with the alloc entry. */
            vkr_journal_insert_locked(j, VKR_JOURNAL_NOTED, cmd_type, frame->start,
                                      size, frame->deferred, NULL, frame->ndeferred,
                                      NULL, 0);
         } else {
            j->stats.transient_cmds++;
         }
         break;
      }
   }

   mtx_unlock(&j->mutex);

out_free:
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
      mtx_lock(&j->mutex);
      j->stats.orphan_adds++;
      mtx_unlock(&j->mutex);
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
   vkr_journal_prune_key(j, id);
}

/* gkvm snapshot-replay: see the pin comment on vkr_journal_keynode. Returns
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

   mtx_lock(&j->mutex);
   const bool ok = vkr_journal_pin_one_locked(j, id, dep_id);
   /* the dedicated object rides the memory's pin lifetime */
   if (ok && dep_id)
      vkr_journal_pin_one_locked(j, dep_id, 0);
   mtx_unlock(&j->mutex);
   return ok;
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

   uint64_t dep_id = 0;
   vkr_journal_unpin_one(j, id, &dep_id);
   if (dep_id)
      vkr_journal_unpin_one(j, dep_id, NULL);
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

void
vkr_journal_get_stats(struct vkr_journal *j, struct vkr_journal_stats *out)
{
   mtx_lock(&j->mutex);
   *out = j->stats;
   mtx_unlock(&j->mutex);
}

uint64_t
vkr_journal_seq(struct vkr_journal *j)
{
   mtx_lock(&j->mutex);
   const uint64_t seq = j->seq_next - 1;
   mtx_unlock(&j->mutex);
   return seq;
}

bool
vkr_journal_export(struct vkr_journal *j, void **out_buf, size_t *out_size)
{
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

   mtx_lock(&j->mutex);

   const struct vkr_journal_stats *s = &j->stats;
   vkr_log("journal ctx %u: %" PRIu64 " entries (%" PRIu64 " KiB) live; recorded "
           "create=%" PRIu64 " recording=%" PRIu64 " noted=%" PRIu64 " ring=%" PRIu64
           " free=%" PRIu64 " pool_reset=%" PRIu64 "; pruned=%" PRIu64
           " transient=%" PRIu64 " orphan_adds=%" PRIu64 " dropped_fatal=%" PRIu64
           " noted_multi_key=%" PRIu64 " pinned_refs=%" PRIu64 " pin_ref_misses=%" PRIu64,
           j->ctx_id, s->entries_live, s->bytes_live / 1024, s->recorded_creates,
           s->recorded_recordings, s->recorded_noted, s->recorded_ring,
           s->recorded_frees, s->recorded_pool_resets, s->pruned_entries,
           s->transient_cmds, s->orphan_adds, s->dropped_fatal, s->noted_multi_key,
           s->pinned_refs, s->pin_ref_misses);

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
