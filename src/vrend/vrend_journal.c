/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * limina: classic-vrend snapshot-replay re-creation journal. See vrend_journal.h
 * for the model and limina docs/design/vrend-snapshot-replay.md for the design.
 */

#include "vrend_journal.h"

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#define XXH_INLINE_ALL
#include "util/hash_table.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/xxhash.h"

#include "virgl_protocol.h"
#include "virgl_util.h"

/* Retention classes. Draw-time commands are never retained: the live clients
 * re-issue them every frame. What must survive a snapshot is what gallium
 * emits only ONCE (creates) or only ON CHANGE (binds and state). */
enum vrend_journal_class {
   VREND_JOURNAL_TRANSIENT = 0, /* not retained */
   VREND_JOURNAL_CREATE,        /* retained until its destroy tombstone */
   VREND_JOURNAL_LATEST,        /* latest-wins per key */
   VREND_JOURNAL_TOMBSTONE,     /* prunes; not itself retained */
   VREND_JOURNAL_UNKNOWN,       /* durable-unknown (e.g. video): counted, not kept */
};

struct vrend_journal_key {
   uint32_t cmd;
   uint32_t obj_type; /* CREATE/BIND_OBJECT object type from the header */
   uint32_t sub_ctx;  /* owning sub-context (object namespace is per-sub) */
   uint32_t k1;
   uint32_t k2;
};

struct vrend_journal_entry {
   struct list_head link; /* vrend_journal.entries, stream order */
   struct vrend_journal_key key;
   uint64_t seq;
   uint32_t ndw;
   uint32_t *data;
};

struct vrend_journal {
   uint32_t ctx_id;
   uint32_t cur_sub; /* mirrored from SET_SUB_CTX as commands stream by */
   uint64_t seq_next;

   struct list_head entries;
   struct hash_table *index; /* vrend_journal_key* -> vrend_journal_entry* */

   struct vrend_journal_census census;
   uint32_t live_per_cmd[VIRGL_MAX_COMMANDS];
};

static uint32_t
key_hash(const void *k)
{
   return (uint32_t)XXH64(k, sizeof(struct vrend_journal_key), 0);
}

static bool
key_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(struct vrend_journal_key)) == 0;
}

struct vrend_journal *
vrend_journal_create(uint32_t ctx_id)
{
   const char *env = getenv("VREND_JOURNAL");
   if (env && env[0] == '0' && env[1] == '\0')
      return NULL;

   struct vrend_journal *j = calloc(1, sizeof(*j));
   if (!j)
      return NULL;
   j->ctx_id = ctx_id;
   j->seq_next = 1;
   list_inithead(&j->entries);
   j->index = _mesa_hash_table_create(NULL, key_hash, key_equal);
   if (!j->index) {
      free(j);
      return NULL;
   }
   return j;
}

static void
entry_free(struct vrend_journal *j, struct vrend_journal_entry *e)
{
   j->census.live_bytes -= (uint64_t)e->ndw * 4;
   j->census.live--;
   j->live_per_cmd[e->key.cmd]--;
   list_del(&e->link);
   _mesa_hash_table_remove_key(j->index, &e->key);
   free(e->data);
   free(e);
}

void
vrend_journal_destroy(struct vrend_journal *j)
{
   if (!j)
      return;
   list_for_each_entry_safe (struct vrend_journal_entry, e, &j->entries, link) {
      list_del(&e->link);
      free(e->data);
      free(e);
   }
   _mesa_hash_table_destroy(j->index, NULL);
   free(j);
}

/* Retain `buf` (ndw dwords incl. header) under `key`, latest-wins. */
static void
retain(struct vrend_journal *j,
       const struct vrend_journal_key *key,
       const uint32_t *buf,
       uint32_t ndw)
{
   struct hash_entry *he = _mesa_hash_table_search(j->index, key);
   if (he) {
      struct vrend_journal_entry *old = he->data;
      j->census.replaced++;
      entry_free(j, old);
   }

   struct vrend_journal_entry *e = malloc(sizeof(*e));
   if (!e)
      return;
   e->data = malloc((size_t)ndw * 4);
   if (!e->data) {
      free(e);
      return;
   }
   e->key = *key;
   e->seq = j->seq_next++;
   e->ndw = ndw;
   memcpy(e->data, buf, (size_t)ndw * 4);
   list_addtail(&e->link, &j->entries);
   _mesa_hash_table_insert(j->index, &e->key, e);
   j->census.recorded++;
   j->census.live++;
   j->census.live_bytes += (uint64_t)ndw * 4;
   j->live_per_cmd[key->cmd]++;
}

/* Prune every live entry `pred` matches (tombstone semantics). */
static void
prune_matching(struct vrend_journal *j,
               bool (*pred)(const struct vrend_journal_entry *, uint32_t, uint32_t),
               uint32_t a,
               uint32_t b)
{
   list_for_each_entry_safe (struct vrend_journal_entry, e, &j->entries, link) {
      if (pred(e, a, b)) {
         j->census.pruned++;
         entry_free(j, e);
      }
   }
}

static bool
pred_object(const struct vrend_journal_entry *e, uint32_t sub, uint32_t handle)
{
   /* An object death takes its create chunks and its BEGIN_QUERY (if any).
    * Binds referencing the handle stay: replay tolerates a dangling bind the
    * same way the live stream would (the client re-binds or dies with it). */
   return e->key.sub_ctx == sub && e->key.k1 == handle &&
          (e->key.cmd == VIRGL_CCMD_CREATE_OBJECT || e->key.cmd == VIRGL_CCMD_BEGIN_QUERY);
}

static bool
pred_sub_ctx(const struct vrend_journal_entry *e, uint32_t sub, UNUSED uint32_t unused)
{
   return e->key.sub_ctx == sub;
}

static bool
pred_begin_query(const struct vrend_journal_entry *e, uint32_t sub, uint32_t handle)
{
   return e->key.cmd == VIRGL_CCMD_BEGIN_QUERY && e->key.sub_ctx == sub &&
          e->key.k1 == handle;
}

void
vrend_journal_record(struct vrend_journal *j, const uint32_t *buf, uint32_t ndw)
{
   if (!j || !ndw)
      return;

   const uint32_t header = buf[0];
   const uint32_t cmd = header & 0xff;
   const uint32_t obj_type = (header >> 8) & 0xff;

   struct vrend_journal_key key = {
      .cmd = cmd,
      .obj_type = 0,
      .sub_ctx = j->cur_sub,
      .k1 = 0,
      .k2 = 0,
   };

   switch (cmd) {
   /* ---- creates (tombstoned by their destroys) ---- */
   case VIRGL_CCMD_CREATE_OBJECT:
      key.obj_type = obj_type;
      key.k1 = ndw > 1 ? buf[VIRGL_OBJ_CREATE_HANDLE] : 0;
      /* shaders arrive in continuation chunks discriminated by the offset word;
       * retain the whole chunk set per handle */
      if (obj_type == VIRGL_OBJECT_SHADER && ndw > VIRGL_OBJ_SHADER_OFFSET)
         key.k2 = buf[VIRGL_OBJ_SHADER_OFFSET];
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_CREATE_SUB_CTX:
      key.sub_ctx = ndw > 1 ? buf[1] : 0; /* keyed to the sub it creates */
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_PIPE_RESOURCE_CREATE:
      /* ctx-global; keyed by the blob id the control queue will instantiate */
      key.sub_ctx = 0;
      key.k1 = ndw > VIRGL_PIPE_RES_CREATE_BLOB_ID ? buf[VIRGL_PIPE_RES_CREATE_BLOB_ID] : 0;
      retain(j, &key, buf, ndw);
      return;

   /* ---- tombstones (prune, not retained) ---- */
   case VIRGL_CCMD_DESTROY_OBJECT:
      prune_matching(j, pred_object, j->cur_sub, ndw > 1 ? buf[1] : 0);
      return;
   case VIRGL_CCMD_DESTROY_SUB_CTX:
      prune_matching(j, pred_sub_ctx, ndw > 1 ? buf[1] : 0, 0);
      return;
   case VIRGL_CCMD_END_QUERY:
      prune_matching(j, pred_begin_query, j->cur_sub, ndw > 1 ? buf[1] : 0);
      return;

   /* ---- current-state, latest-wins ---- */
   case VIRGL_CCMD_SET_SUB_CTX:
      /* ctx-level latest-wins; also advances the recording mirror so later
       * entries land under the right sub */
      key.sub_ctx = 0;
      retain(j, &key, buf, ndw);
      j->cur_sub = ndw > 1 ? buf[1] : 0;
      return;
   case VIRGL_CCMD_BIND_OBJECT:
      key.obj_type = obj_type;
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_BIND_SHADER:
      key.k1 = ndw > VIRGL_BIND_SHADER_TYPE ? buf[VIRGL_BIND_SHADER_TYPE] : 0;
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_BEGIN_QUERY:
      key.k1 = ndw > 1 ? buf[1] : 0;
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_BIND_SAMPLER_STATES:
   case VIRGL_CCMD_SET_SAMPLER_VIEWS:
   case VIRGL_CCMD_SET_CONSTANT_BUFFER:
   case VIRGL_CCMD_SET_UNIFORM_BUFFER:
   case VIRGL_CCMD_SET_SHADER_BUFFERS:
   case VIRGL_CCMD_SET_SHADER_IMAGES:
      /* (stage, start-slot/index) discriminated */
      key.k1 = ndw > 1 ? buf[1] : 0;
      key.k2 = ndw > 2 ? buf[2] : 0;
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_SET_ATOMIC_BUFFERS:
   case VIRGL_CCMD_SET_VIEWPORT_STATE:
   case VIRGL_CCMD_SET_SCISSOR_STATE:
      key.k1 = ndw > 1 ? buf[1] : 0; /* start slot */
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_SET_TWEAKS:
      key.sub_ctx = 0; /* ctx-global */
      key.k1 = ndw > VIRGL_SET_TWEAKS_ID ? buf[VIRGL_SET_TWEAKS_ID] : 0;
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_PIPE_RESOURCE_SET_TYPE:
      key.sub_ctx = 0; /* ctx-global, keyed by the typed resource */
      key.k1 = ndw > VIRGL_PIPE_RES_SET_TYPE_RES_HANDLE
                  ? buf[VIRGL_PIPE_RES_SET_TYPE_RES_HANDLE]
                  : 0;
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_LINK_SHADER:
      /* keyed by the program's shader-handle tuple; (vs, fs) discriminates in
       * practice, the folded tail covers gs/tcs/tes/cs-only programs */
      key.k1 = ndw > 2 ? buf[1] ^ buf[2] : 0;
      key.k2 = ndw > 6 ? (buf[3] ^ buf[4] ^ buf[5] ^ buf[6]) : 0;
      retain(j, &key, buf, ndw);
      return;
   case VIRGL_CCMD_SET_FRAMEBUFFER_STATE:
   case VIRGL_CCMD_SET_FRAMEBUFFER_STATE_NO_ATTACH:
   case VIRGL_CCMD_SET_VERTEX_BUFFERS:
   case VIRGL_CCMD_SET_INDEX_BUFFER:
   case VIRGL_CCMD_SET_STENCIL_REF:
   case VIRGL_CCMD_SET_BLEND_COLOR:
   case VIRGL_CCMD_SET_CLIP_STATE:
   case VIRGL_CCMD_SET_SAMPLE_MASK:
   case VIRGL_CCMD_SET_MIN_SAMPLES:
   case VIRGL_CCMD_SET_POLYGON_STIPPLE:
   case VIRGL_CCMD_SET_STREAMOUT_TARGETS:
   case VIRGL_CCMD_SET_TESS_STATE:
   case VIRGL_CCMD_SET_RENDER_CONDITION:
      /* whole-array semantics: one live entry per (sub, cmd) */
      retain(j, &key, buf, ndw);
      return;

   /* ---- transient: re-issued by live clients, never retained ---- */
   case VIRGL_CCMD_NOP:
   case VIRGL_CCMD_CLEAR:
   case VIRGL_CCMD_CLEAR_TEXTURE:
   case VIRGL_CCMD_CLEAR_SURFACE:
   case VIRGL_CCMD_DRAW_VBO:
   case VIRGL_CCMD_RESOURCE_INLINE_WRITE:
   case VIRGL_CCMD_BLIT:
   case VIRGL_CCMD_RESOURCE_COPY_REGION:
   case VIRGL_CCMD_GET_QUERY_RESULT:
   case VIRGL_CCMD_GET_QUERY_RESULT_QBO:
   case VIRGL_CCMD_MEMORY_BARRIER:
   case VIRGL_CCMD_TEXTURE_BARRIER:
   case VIRGL_CCMD_LAUNCH_GRID:
   case VIRGL_CCMD_TRANSFER3D:
   case VIRGL_CCMD_END_TRANSFERS:
   case VIRGL_CCMD_COPY_TRANSFER3D:
   case VIRGL_CCMD_SEND_STRING_MARKER:
   case VIRGL_CCMD_SET_DEBUG_FLAGS:
   case VIRGL_CCMD_GET_MEMORY_INFO:
   case VIRGL_CCMD_GET_PIPE_RESOURCE_LAYOUT:
      return;

   default:
      /* durable-unknown (video codec etc.): counted so a guest using them is
       * loud in the census instead of silently losing state at restore */
      j->census.skipped++;
      return;
   }
}

void
vrend_journal_census(const struct vrend_journal *j, struct vrend_journal_census *out)
{
   if (!j) {
      memset(out, 0, sizeof(*out));
      return;
   }
   *out = j->census;
}

/* Same header + entry layout as vkr_journal_export ('VKJR' v1): the libkrun
 * consumer must not care which journal produced the bytes. */
#define VREND_JOURNAL_EXPORT_MAGIC 0x524a4b56u /* 'VKJR' LE */
#define VREND_JOURNAL_EXPORT_VERSION 1u

#define VREND_JOURNAL_KLASS_STATE 3u  /* benign drop (stale reference) */
#define VREND_JOURNAL_KLASS_CREATE 6u /* load-bearing drop (warns at replay) */

static uint32_t
entry_klass(const struct vrend_journal_entry *e)
{
   switch (e->key.cmd) {
   case VIRGL_CCMD_CREATE_OBJECT:
   case VIRGL_CCMD_CREATE_SUB_CTX:
   case VIRGL_CCMD_PIPE_RESOURCE_CREATE:
      return VREND_JOURNAL_KLASS_CREATE;
   default:
      return VREND_JOURNAL_KLASS_STATE;
   }
}

/* Sub-ctx scope: object/state entries live in the sub-context that was current
 * when they recorded; these classes are context-global (or define subs) and
 * must NOT be bracketed by a SET_SUB_CTX. */
static bool
entry_is_sub_scoped(const struct vrend_journal_entry *e)
{
   switch (e->key.cmd) {
   case VIRGL_CCMD_CREATE_SUB_CTX:
   case VIRGL_CCMD_SET_SUB_CTX:
   case VIRGL_CCMD_SET_TWEAKS:
   case VIRGL_CCMD_PIPE_RESOURCE_CREATE:
   case VIRGL_CCMD_PIPE_RESOURCE_SET_TYPE:
      return false;
   default:
      return true;
   }
}

/* The replay stream must re-establish each entry's owning sub-context: only the
 * LATEST SET_SUB_CTX is retained (latest-wins), but entries under other subs
 * still replay. The export walks in seq order and injects a synthesized
 * SET_SUB_CTX at every sub transition. Seqs on synthesized entries duplicate
 * their successor's — the replayer's `seq <= fence` walk needs monotone seqs,
 * never unique ones. A real (retained) SET_SUB_CTX in the stream also switches
 * the tracked sub, exactly like it will at replay. */
struct vrend_journal_export_walk {
   uint32_t cur_sub;
   bool sub_known;
};

static bool
export_needs_sub_switch(struct vrend_journal_export_walk *w,
                        const struct vrend_journal_entry *e,
                        uint32_t *out_sub)
{
   if (e->key.cmd == VIRGL_CCMD_SET_SUB_CTX) {
      /* the retained latest-wins entry: replaying it switches the sub */
      w->cur_sub = e->ndw > 1 ? e->data[1] : 0;
      w->sub_known = true;
      return false;
   }
   if (!entry_is_sub_scoped(e))
      return false;
   if (w->sub_known && w->cur_sub == e->key.sub_ctx)
      return false;
   w->cur_sub = e->key.sub_ctx;
   w->sub_known = true;
   *out_sub = e->key.sub_ctx;
   return true;
}

bool
vrend_journal_export(const struct vrend_journal *j, void **out_buf, size_t *out_size)
{
   if (!j)
      return false;

   const size_t entry_overhead = sizeof(uint64_t) + sizeof(uint32_t) + 4 +
                                 sizeof(uint64_t) + sizeof(uint32_t);
   const size_t set_sub_size = 2 * sizeof(uint32_t); /* header + sub id */

   size_t total = 4 * sizeof(uint32_t);
   uint32_t count = 0;
   struct vrend_journal_export_walk cw = { 0, false };
   list_for_each_entry (struct vrend_journal_entry, e, &j->entries, link) {
      uint32_t sub;
      if (export_needs_sub_switch(&cw, e, &sub)) {
         total += entry_overhead + set_sub_size;
         count++;
      }
      total += entry_overhead + ALIGN_POT((size_t)e->ndw * 4, 4);
      count++;
   }

   uint8_t *buf = malloc(total);
   if (!buf)
      return false;

   uint8_t *p = buf;
#define VREND_JOURNAL_PUT(val)                                                           \
   do {                                                                                  \
      memcpy(p, &(val), sizeof(val));                                                    \
      p += sizeof(val);                                                                  \
   } while (0)
   const uint32_t magic = VREND_JOURNAL_EXPORT_MAGIC;
   const uint32_t version = VREND_JOURNAL_EXPORT_VERSION;
   const uint32_t reserved = 0;
   VREND_JOURNAL_PUT(magic);
   VREND_JOURNAL_PUT(version);
   VREND_JOURNAL_PUT(count);
   VREND_JOURNAL_PUT(reserved);

   struct vrend_journal_export_walk ww = { 0, false };
   list_for_each_entry (struct vrend_journal_entry, e, &j->entries, link) {
      const uint8_t pad[3] = { 0 };
      const uint64_t ring_key = 0;
      uint32_t sub;
      if (export_needs_sub_switch(&ww, e, &sub)) {
         const uint32_t cmd_type = VIRGL_CCMD_SET_SUB_CTX;
         const uint8_t klass = VREND_JOURNAL_KLASS_STATE;
         const uint32_t size = (uint32_t)set_sub_size;
         const uint32_t set_sub_cmd[2] = { VIRGL_CMD0(VIRGL_CCMD_SET_SUB_CTX, 0, 1), sub };
         VREND_JOURNAL_PUT(e->seq);
         VREND_JOURNAL_PUT(cmd_type);
         VREND_JOURNAL_PUT(klass);
         memcpy(p, pad, sizeof(pad));
         p += sizeof(pad);
         VREND_JOURNAL_PUT(ring_key);
         VREND_JOURNAL_PUT(size);
         memcpy(p, set_sub_cmd, size);
         p += size;
      }
      const uint32_t cmd_type = e->key.cmd;
      const uint8_t klass = (uint8_t)entry_klass(e);
      const uint32_t size = e->ndw * 4;
      VREND_JOURNAL_PUT(e->seq);
      VREND_JOURNAL_PUT(cmd_type);
      VREND_JOURNAL_PUT(klass);
      memcpy(p, pad, sizeof(pad));
      p += sizeof(pad);
      VREND_JOURNAL_PUT(ring_key);
      VREND_JOURNAL_PUT(size);
      memcpy(p, e->data, size);
      p += size;
   }
#undef VREND_JOURNAL_PUT

   assert((size_t)(p - buf) == total);
   *out_buf = buf;
   *out_size = total;
   return true;
}

uint64_t
vrend_journal_seq(const struct vrend_journal *j)
{
   return j ? j->seq_next - 1 : 0;
}

static bool
pred_pipe_res(const struct vrend_journal_entry *e, uint32_t blob_lo, uint32_t unused)
{
   (void)unused;
   return e->key.cmd == VIRGL_CCMD_PIPE_RESOURCE_CREATE && e->key.k1 == blob_lo;
}

void
vrend_journal_unpin_blob(struct vrend_journal *j, uint64_t blob_id)
{
   if (!j)
      return;
   prune_matching(j, pred_pipe_res, (uint32_t)blob_id, 0);
}

void
vrend_journal_dump(const struct vrend_journal *j)
{
   if (!j)
      return;
   virgl_info("[GPUTRACE] vrend journal ctx %u: live=%u bytes=%" PRIu64
              " recorded=%" PRIu64 " replaced=%" PRIu64 " pruned=%" PRIu64
              " skipped=%" PRIu64 "\n",
              j->ctx_id, j->census.live, j->census.live_bytes, j->census.recorded,
              j->census.replaced, j->census.pruned, j->census.skipped);
   for (uint32_t c = 0; c < VIRGL_MAX_COMMANDS; c++) {
      if (j->live_per_cmd[c])
         virgl_info("[GPUTRACE]   cmd %u: live=%u\n", c, j->live_per_cmd[c]);
   }
}
