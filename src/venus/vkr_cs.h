/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VKR_CS_H
#define VKR_CS_H

#include "vkr_common.h"

/* This is to avoid integer overflows and to catch bogus allocations (e.g.,
 * the guest driver encodes an uninitialized value).  In practice, the largest
 * allocations we've seen are from vkGetPipelineCacheData and are dozens of
 * MBs.
 */
#define VKR_CS_DECODER_TEMP_POOL_MAX_SIZE (1u * 1024 * 1024 * 1024)

/* limina snapshot-replay: the journal's create-arg closure hook (vkr_journal.h);
 * declared here so the inline lookup below can call it without an include cycle */
void
vkr_journal_note_lookup(uint64_t id);

/* limina ghost containment (see vkr_context.h tombstones): the decoder consults
 * and extends the context's tombstone set on its cold paths. Declared here —
 * with vkr_context opaque — because vkr_context.h includes THIS header. */
struct vkr_context;
struct vkr_cs_decoder;

/* Cap on the per-context tombstone ring. Sized for the realistic burst (a
 * compositor failing a few imports per frame) with room to spare; overflow only
 * costs the oldest entry its grace. */
#define VKR_CONTEXT_TOMBSTONE_MAX 256

bool
vkr_cs_decoder_is_tombstoned(const struct vkr_cs_decoder *dec, vkr_object_id id);

/* limina snapshot-restore: is this context one whose lookup misses are skipped
 * rather than fatal? True for the life of any restored context — see
 * fatal_contained in vkr_context.h for why it cannot be narrower yet. */
bool
vkr_cs_decoder_fatal_contained(const struct vkr_cs_decoder *dec);

void
vkr_cs_decoder_finish_soft_error(struct vkr_cs_decoder *dec);

struct vkr_cs_encoder {
   bool *fatal_error;

   /* protect stream resource */
   mtx_t mutex;

   struct {
      const struct vkr_resource *resource;
      size_t offset;
      size_t size;
      bool busy;
   } stream;

   uint8_t *cur;
   const uint8_t *end;
};

struct vkr_cs_decoder_saved_state {
   const uint8_t *cur;
   const uint8_t *end;

   uint32_t pool_buffer_count;
   uint8_t *pool_reset_to;
};

/*
 * We usually need many small allocations during decoding.  Those allocations
 * are suballocated from the temp pool.
 *
 * After a command is decoded, vkr_cs_decoder_reset_temp_pool is called to
 * reset pool->cur.  After an entire command stream is decoded,
 * vkr_cs_decoder_gc_temp_pool is called to garbage collect pool->buffers.
 */
struct vkr_cs_decoder_temp_pool {
   uint8_t **buffers;
   uint32_t buffer_count;
   uint32_t buffer_max;
   size_t total_size;

   uint8_t *reset_to;

   uint8_t *cur;
   const uint8_t *end;
};

/* limina: per-decoder repeat-lookup cache size. 4 covers the hot recording
 * pattern (command buffer + pipeline layout alternating per command, plus a
 * pipeline / descriptor set in the mix) without a real search cost. */
#define VKR_CS_LOOKUP_CACHE_SIZE 4

struct vkr_cs_decoder {
   const struct hash_table *object_table;
   mtx_t *object_mutex;
   /* limina: lock-free repeat-lookup cache (vkr_cs_decoder_lookup_object).
    * Entries are trusted only while *object_gen still equals cache_gen, the
    * generation captured under object_mutex when they were filled; every
    * table insert/remove bumps the generation (vkr_context.h), so id reuse
    * can never serve a stale object. */
   const uint64_t *object_gen;
   uint64_t cache_gen;
   struct {
      vkr_object_id id;
      struct vkr_object *obj;
   } lookup_cache[VKR_CS_LOOKUP_CACHE_SIZE];
   uint32_t lookup_cache_next;

   /* Attribution for FATAL logs: which guest context (and process, via its
    * debug name) this decoder serves. ctx_name borrows ctx->debug_name, which
    * outlives every decoder embedded in the context or its rings. */
   uint32_t ctx_id;
   const char *ctx_name;

   bool *fatal_error;

   /* limina ghost containment. ctx backs the tombstone set (vkr_context.h);
    * soft_error marks the CURRENT command as "skip me" — a command that named a
    * tombstoned id, i.e. an object the host never created. It reads as fatal to
    * the generated dispatch wrappers (they consult vkr_cs_decoder_get_fatal to
    * decide whether to call the handler and encode a reply), so the command is
    * dropped without ever reaching the driver, while the ring loops use
    * vkr_cs_decoder_get_hard_fatal and keep going.
    *
    * touched[] holds the other object ids this command looked up. When the
    * command is dropped, they are tombstoned too: an object whose bind, write or
    * recording never happened is not safe to hand the driver later (KK asserts
    * on, say, an image with no memory bound, and an abort in the worker takes
    * the whole VM — strictly worse than one guest losing a context). The poison
    * therefore spreads only along real usage chains, and stops as soon as the
    * guest's own error handling tears the objects down. */
   struct vkr_context *ctx;
   bool soft_error;
   vkr_object_id touched[8];
   uint32_t touched_count;

   struct vkr_cs_decoder_temp_pool temp_pool;

   /* Support vkExecuteCommandStreamsMESA for command buffer recording and indirect
    * submission. Only a single level nested decoder state is needed. Base level is
    * always from context or ring submit buffer, and no resource tracking is needed.
    */
   struct vkr_cs_decoder_saved_state saved_state;
   bool saved_state_valid;

   /* protect against resource destroy */
   mtx_t resource_mutex;
   const struct vkr_resource *resource;

   /* limina ring-FATAL attribution: which command this decoder is inside, so a
    * fatal names itself. Maintained by the dispatch tee (vkr_journal_pre/
    * post_dispatch), which already brackets all four vn_dispatch_command call
    * sites, keeping the generated venus-protocol headers stock. Every one of the
    * hundreds of generated set_fatal call sites reports the same shim func/line,
    * so the COMMAND is what localises a failure, not the call site.
    *
    * These are deliberately not cleared when a command ends: at depth 0 they name
    * the last command that completed, which is what a fatal raised outside any
    * dispatch (ring header, transport framing) needs to say. dispatch_depth is a
    * count, not a flag — vkExecuteCommandStreamsMESA nests one level through the
    * same dispatcher. */
   uint32_t cur_cmd_type;
   const uint8_t *cur_cmd_start;
   uint64_t dispatch_seq;
   uint32_t dispatch_depth;

   const uint8_t *cur;
   const uint8_t *end;
};

static inline int
vkr_cs_encoder_init(struct vkr_cs_encoder *enc, bool *fatal_error)
{
   memset(enc, 0, sizeof(*enc));
   enc->fatal_error = fatal_error;

   return mtx_init(&enc->mutex, mtx_plain);
}

static inline void
vkr_cs_encoder_fini(struct vkr_cs_encoder *enc)
{
   mtx_destroy(&enc->mutex);
}

static inline void
vkr_cs_encoder_set_fatal_at(const struct vkr_cs_encoder *enc, const char *func, int line)
{
   if (!*enc->fatal_error)
      vkr_log_error("cs encoder: ring FATAL set at %s:%d", func, line);
   *enc->fatal_error = true;
}

#define vkr_cs_encoder_set_fatal(enc) vkr_cs_encoder_set_fatal_at((enc), __func__, __LINE__)

void
vkr_cs_encoder_set_stream_locked(struct vkr_cs_encoder *enc,
                                 const struct vkr_resource *res,
                                 size_t offset,
                                 size_t size);

void
vkr_cs_encoder_seek_stream_locked(struct vkr_cs_encoder *enc, size_t pos);

static inline void
vkr_cs_encoder_set_stream(struct vkr_cs_encoder *enc,
                          const struct vkr_resource *res,
                          size_t offset,
                          size_t size)
{

   mtx_lock(&enc->mutex);
   vkr_cs_encoder_set_stream_locked(enc, res, offset, size);
   mtx_unlock(&enc->mutex);
}

static inline void
vkr_cs_encoder_seek_stream(struct vkr_cs_encoder *enc, size_t pos)
{
   mtx_lock(&enc->mutex);
   vkr_cs_encoder_seek_stream_locked(enc, pos);
   mtx_unlock(&enc->mutex);
}

static inline bool
vkr_cs_encoder_check_stream(struct vkr_cs_encoder *enc, const struct vkr_resource *res)
{
   mtx_lock(&enc->mutex);
   if (enc->stream.resource && enc->stream.resource == res) {
      if (enc->stream.busy) {
         mtx_unlock(&enc->mutex);
         return false;
      }
      /* TODO vkSetReplyCommandStreamMESA should support res_id 0 to unset. Until then,
       * and until we can ignore older guests, treat this as non-fatal. This can happen
       * when the driver side reply shmem has lost its last ref for being used as reply
       * shmem (it can still live in the driver side shmem cache but will be used for
       * other purposes the next time being allocated out).
       */
      vkr_cs_encoder_set_stream_locked(enc, NULL, 0, 0);
   }
   mtx_unlock(&enc->mutex);

   return true;
}

static inline bool
vkr_cs_encoder_acquire(struct vkr_cs_encoder *enc)
{
   mtx_lock(&enc->mutex);
   if (unlikely(!enc->stream.resource)) {
      vkr_cs_encoder_set_fatal(enc);
      mtx_unlock(&enc->mutex);
      return false;
   }
   enc->stream.busy = true;
   mtx_unlock(&enc->mutex);
   return true;
}

static inline void
vkr_cs_encoder_release(struct vkr_cs_encoder *enc)
{
   mtx_lock(&enc->mutex);
   assert(enc->stream.resource);
   enc->stream.busy = false;
   mtx_unlock(&enc->mutex);
}

static inline void
vkr_cs_encoder_write(struct vkr_cs_encoder *enc,
                     size_t size,
                     const void *val,
                     size_t val_size)
{
   assert(val_size <= size);

   if (unlikely(size > (size_t)(enc->end - enc->cur))) {
      vkr_log_error("failed to write the reply stream");
      vkr_cs_encoder_set_fatal(enc);
      return;
   }

   /* we should not rely on the compiler to optimize away memcpy... */
   if (enc->cur != val)
      memcpy(enc->cur, val, val_size);
   enc->cur += size;
}

int
vkr_cs_decoder_init(struct vkr_cs_decoder *dec, struct vkr_context *ctx);

void
vkr_cs_decoder_fini(struct vkr_cs_decoder *dec);

void
vkr_cs_decoder_reset(struct vkr_cs_decoder *dec);

/* Names a command type for the fatal report; "unknown" for anything the
 * generated table does not know. */
const char *
vkr_cs_command_name(uint32_t cmd_type);

/* Out of line on purpose: this header is included by every generated protocol
 * TU, and the report is cold — it runs at most once per context, on the way to
 * a dead ring. */
void
vkr_cs_decoder_report_fatal(const struct vkr_cs_decoder *dec, const char *func, int line);

static inline void
vkr_cs_decoder_set_fatal_at(const struct vkr_cs_decoder *dec, const char *func, int line)
{
   if (!*((struct vkr_cs_decoder *)dec)->fatal_error)
      vkr_cs_decoder_report_fatal(dec, func, line);
   *((struct vkr_cs_decoder *)dec)->fatal_error = true;
}

#define vkr_cs_decoder_set_fatal(dec) vkr_cs_decoder_set_fatal_at((dec), __func__, __LINE__)

/* What the GENERATED dispatch wrappers see: a soft error must make them skip
 * the handler (and the reply) exactly as a fatal one does — that skip is the
 * whole containment mechanism, and routing it through this one accessor keeps
 * the generated code untouched. */
static inline bool
vkr_cs_decoder_get_fatal(const struct vkr_cs_decoder *dec)
{
   return *dec->fatal_error || dec->soft_error;
}

/* What the RING/TRANSPORT loops must see: only a real fatal stops the ring. A
 * soft error is per-command and cleared when the command ends. */
static inline bool
vkr_cs_decoder_get_hard_fatal(const struct vkr_cs_decoder *dec)
{
   return *dec->fatal_error;
}

static inline void
vkr_cs_decoder_set_buffer_stream(struct vkr_cs_decoder *dec,
                                 const void *data,
                                 size_t size)
{
   dec->cur = data;
   dec->end = dec->cur + size;
}

bool
vkr_cs_decoder_set_resource_stream(struct vkr_cs_decoder *dec,
                                   struct vkr_context *ctx,
                                   uint32_t res_id,
                                   size_t offset,
                                   size_t size);

static inline bool
vkr_cs_decoder_check_stream(struct vkr_cs_decoder *dec, const struct vkr_resource *res)
{
   mtx_lock(&dec->resource_mutex);
   const bool ok = dec->resource != res;
   mtx_unlock(&dec->resource_mutex);
   return ok;
}

static inline bool
vkr_cs_decoder_has_command(const struct vkr_cs_decoder *dec)
{
   return dec->cur < dec->end;
}

static inline bool
vkr_cs_decoder_has_saved_state(struct vkr_cs_decoder *dec)
{
   return dec->saved_state_valid;
}

void
vkr_cs_decoder_save_state(struct vkr_cs_decoder *dec);

void
vkr_cs_decoder_restore_state(struct vkr_cs_decoder *dec);

static inline bool
vkr_cs_decoder_peek_internal(const struct vkr_cs_decoder *dec,
                             size_t size,
                             void *val,
                             size_t val_size)
{
   assert(val_size <= size);

   if (unlikely(size > (size_t)(dec->end - dec->cur))) {
      vkr_log_error("failed to peek %zu bytes", size);
      vkr_cs_decoder_set_fatal(dec);
      memset(val, 0, val_size);
      return false;
   }

   /* we should not rely on the compiler to optimize away memcpy... */
   if (dec->cur != val)
      memcpy(val, dec->cur, val_size);
   return true;
}

static inline void
vkr_cs_decoder_read(struct vkr_cs_decoder *dec, size_t size, void *val, size_t val_size)
{
   if (vkr_cs_decoder_peek_internal(dec, size, val, val_size))
      dec->cur += size;
}

static inline void
vkr_cs_decoder_peek(const struct vkr_cs_decoder *dec,
                    size_t size,
                    void *val,
                    size_t val_size)
{
   vkr_cs_decoder_peek_internal(dec, size, val, val_size);
}

static inline struct vkr_object *
vkr_cs_decoder_lookup_object(const struct vkr_cs_decoder *dec,
                             vkr_object_id id,
                             VkObjectType type)
{
   struct vkr_cs_decoder *mut_dec = (struct vkr_cs_decoder *)dec;
   struct vkr_object *obj = NULL;

   if (!id)
      return NULL;

   /* limina: generation-checked fast path — a recording stream looks up the
    * same command buffer (and usually the same layout) for every command, and
    * taking the table mutex per handle was ~10% of the decode lane. A relaxed
    * generation read is sufficient: any staleness only diverts us to the
    * locked path, and a use of a just-created/just-reused id can only be
    * decoded after the guest saw the create reply, whose ring transport
    * orders the generation bump before the command bytes we are decoding. */
   const uint64_t gen = p_atomic_read(dec->object_gen);
   if (likely(gen == dec->cache_gen)) {
      for (uint32_t i = 0; i < VKR_CS_LOOKUP_CACHE_SIZE; i++) {
         if (dec->lookup_cache[i].id == id) {
            obj = dec->lookup_cache[i].obj;
            break;
         }
      }
   }

   if (!obj) {
      mtx_lock(dec->object_mutex);
      const struct hash_entry *entry =
         _mesa_hash_table_search((struct hash_table *)dec->object_table, &id);
      obj = likely(entry) ? entry->data : NULL;
      /* the generation an entry is tagged with must never be newer than the
       * lookup that produced it, so capture it while the mutex still
       * excludes table mutations */
      const uint64_t gen_locked = *dec->object_gen;
      mtx_unlock(dec->object_mutex);

      if (obj) {
         if (mut_dec->cache_gen != gen_locked) {
            memset(mut_dec->lookup_cache, 0, sizeof(mut_dec->lookup_cache));
            mut_dec->lookup_cache_next = 0;
            mut_dec->cache_gen = gen_locked;
         }
         const uint32_t slot = mut_dec->lookup_cache_next++ % VKR_CS_LOOKUP_CACHE_SIZE;
         mut_dec->lookup_cache[slot].id = id;
         mut_dec->lookup_cache[slot].obj = obj;
      }
   }

   if (unlikely(!obj || obj->type != type)) {
      /* limina ghost containment: a miss on an id we KNOW failed host-side is
       * an expected runtime state, not protocol corruption — the guest was told
       * VK_SUCCESS by an async submission the host later refused. Drop just this
       * command (see soft_error) instead of poisoning the ring. Tombstones are
       * consulted only here, after the table missed, so they never shadow a live
       * object and never enter the lookup cache (only found objects are cached).
       *
       * A miss on an id we have NO record of stays FATAL: that IS a protocol
       * violation, and the detector has to keep working. */
      if (!obj && (vkr_cs_decoder_is_tombstoned(dec, id) ||
                   vkr_cs_decoder_fatal_contained(dec))) {
         ((struct vkr_cs_decoder *)dec)->soft_error = true;
         return NULL;
      }

      /* ERROR, not INFO: this accompanies a ring FATAL, and the id + type are
       * the attribution a production log needs. A miss here usually means a
       * host-side create failed earlier on an async command, so the guest
       * never learned (see the create-failure logging in vkr_device_object.py)
       * — the 2026-07-10 dogfood poisons were exactly this shape. */
      if (obj)
         vkr_log_error("object %" PRIu64 " has type %d, not %d", id, obj->type, type);
      else
         vkr_log_error("failed to look up object %" PRIu64 " of type %d", id, type);
      vkr_cs_decoder_set_fatal(dec);
   } else {
      /* limina ghost containment: remember what else this command names, so a
       * drop can tombstone the objects it would have modified (see touched[]).
       * Device/queue/instance/physical-device are the command's *subject*, never
       * a casualty of it — tombstoning a VkDevice would skip every subsequent
       * command in the context, i.e. reinvent the ring death this exists to
       * prevent. */
      struct vkr_cs_decoder *mut = (struct vkr_cs_decoder *)dec;
      if (type != VK_OBJECT_TYPE_DEVICE && type != VK_OBJECT_TYPE_QUEUE &&
          type != VK_OBJECT_TYPE_INSTANCE && type != VK_OBJECT_TYPE_PHYSICAL_DEVICE &&
          mut->touched_count < ARRAY_SIZE(mut->touched))
         mut->touched[mut->touched_count++] = id;

      /* limina snapshot-replay: feed the journal's create-arg closure — see
       * vkr_journal_note_lookup (no-op outside a journal dispatch frame) */
      vkr_journal_note_lookup(id);
   }

   return obj;
}

/* Every generated dispatch wrapper ends with this call, which makes it the one
 * per-command epilogue vkr owns — so the dropped command's co-named objects are
 * tombstoned here. soft_error itself is NOT cleared: the journal inspects it
 * after the wrapper returns (a command that did not run must not be recorded for
 * replay), so the dispatch loops clear it just before the next command. */
static inline void
vkr_cs_decoder_reset_temp_pool(struct vkr_cs_decoder *dec)
{
   if (unlikely(dec->soft_error))
      vkr_cs_decoder_finish_soft_error(dec);
   dec->touched_count = 0;

   struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;
   pool->cur = pool->reset_to;
}

/* Called by each dispatch loop before handing the decoder the next command. */
static inline void
vkr_cs_decoder_clear_soft_error(struct vkr_cs_decoder *dec)
{
   dec->soft_error = false;
   dec->touched_count = 0;
}

bool
vkr_cs_decoder_alloc_temp_internal(struct vkr_cs_decoder *dec, size_t size);

static inline void *
vkr_cs_decoder_alloc_temp(struct vkr_cs_decoder *dec, size_t size)
{
   struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;

   if (unlikely(size > (size_t)(pool->end - pool->cur))) {
      if (!vkr_cs_decoder_alloc_temp_internal(dec, size)) {
         vkr_log_error("failed to suballocate %zu bytes from the temp pool", size);
         vkr_cs_decoder_set_fatal(dec);
         return NULL;
      }
   }

   /* align to 64-bit after we know size is at most
    * VKR_CS_DECODER_TEMP_POOL_MAX_SIZE and cannot overflow
    */
   size = align64(size, 8);
   assert(size <= (size_t)(pool->end - pool->cur));

   void *ptr = pool->cur;
   pool->cur += size;
   return ptr;
}

static inline void *
vkr_cs_decoder_alloc_temp_array(struct vkr_cs_decoder *dec, size_t size, size_t count)
{
   size_t alloc_size;
   if (unlikely(__builtin_mul_overflow(size, count, &alloc_size))) {
      vkr_log_error("overflow in array allocation of %zu * %zu bytes", size, count);
      vkr_cs_decoder_set_fatal(dec);
      return NULL;
   }

   return vkr_cs_decoder_alloc_temp(dec, alloc_size);
}

static inline void *
vkr_cs_decoder_get_blob_storage(struct vkr_cs_decoder *dec, size_t size)
{
   return unlikely(size > (size_t)(dec->end - dec->cur)) ? NULL : (void *)dec->cur;
}

static inline void *
vkr_cs_encoder_get_blob_storage(struct vkr_cs_encoder *enc, size_t offset, size_t size)
{
   return unlikely(offset + size > (size_t)(enc->end - enc->cur))
             ? NULL
             : (void *)(enc->cur + offset);
}

static inline bool
vkr_cs_handle_indirect_id(VkObjectType type)
{
   /* Dispatchable handles may or may not have enough bits to store
    * vkr_object_id.  Non-dispatchable handles always have enough bits to
    * store vkr_object_id.
    *
    * This should compile to a constant after inlining.
    */
   switch (type) {
   case VK_OBJECT_TYPE_INSTANCE:
   case VK_OBJECT_TYPE_PHYSICAL_DEVICE:
   case VK_OBJECT_TYPE_DEVICE:
   case VK_OBJECT_TYPE_QUEUE:
   case VK_OBJECT_TYPE_COMMAND_BUFFER:
      return sizeof(VkInstance) < sizeof(vkr_object_id);
   default:
      return false;
   }
}

static inline vkr_object_id
vkr_cs_handle_load_id(const void **handle, VkObjectType type)
{
   const vkr_object_id *p = vkr_cs_handle_indirect_id(type)
                               ? *(const vkr_object_id **)handle
                               : (const vkr_object_id *)handle;
   return *p;
}

static inline void
vkr_cs_handle_store_id(void **handle, vkr_object_id id, VkObjectType type)
{
   vkr_object_id *p = vkr_cs_handle_indirect_id(type) ? *(vkr_object_id **)handle
                                                      : (vkr_object_id *)handle;
   *p = id;
}

#endif /* VKR_CS_H */
