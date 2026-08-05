/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * limina: classic-vrend snapshot-replay re-creation journal. See vrend_journal.h
 * for the model and limina docs/design/vrend-snapshot-replay.md for the design.
 */

#include "vrend_journal.h"

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
