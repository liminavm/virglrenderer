/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_cs.h"

#include "vkr_context.h"

/* limina ghost containment — see the tombstones field in vkr_context.h. Out of
 * line because the inline lookup in vkr_cs.h cannot see struct vkr_context, and
 * because this is a cold path: it runs only after the object table missed, which
 * used to be the last thing a context ever did. */
bool
vkr_cs_decoder_is_tombstoned(const struct vkr_cs_decoder *dec, vkr_object_id id)
{
   struct vkr_context *ctx = dec->ctx;
   if (!ctx)
      return false;

   bool found = false;
   mtx_lock(&ctx->object_mutex);
   for (uint32_t i = 0; i < VKR_CONTEXT_TOMBSTONE_MAX; i++) {
      if (ctx->tombstones[i] == id) {
         found = true;
         break;
      }
   }
   mtx_unlock(&ctx->object_mutex);

   return found;
}

/* limina snapshot-restore containment — same cold path, same reason it lives out
 * of line. See restored_lossy in vkr_context.h. */
bool
vkr_cs_decoder_restored_lossy(const struct vkr_cs_decoder *dec)
{
   return dec->ctx && dec->ctx->restored_lossy;
}

/* The dropped command's epilogue: everything else it named becomes a tombstone
 * too, because the bind/write/recording it was going to perform did not happen.
 * Loud at ERROR — the guest is now running degraded, and this is the only record
 * of where its objects started diverging from what the host holds. */
void
vkr_cs_decoder_finish_soft_error(struct vkr_cs_decoder *dec)
{
   struct vkr_context *ctx = dec->ctx;
   if (!ctx)
      return;

   for (uint32_t i = 0; i < dec->touched_count; i++)
      vkr_context_tombstone_object(ctx, dec->touched[i]);

   vkr_log_error("context %u [%s]: command naming a failed (tombstoned) object "
                 "SKIPPED; %u co-named object(s) tombstoned with it — the guest "
                 "runs degraded, the ring stays alive (%" PRIu64 " total)",
                 dec->ctx_id, dec->ctx_name ? dec->ctx_name : "?", dec->touched_count,
                 ctx->tombstone_total);
}

void
vkr_cs_encoder_set_stream_locked(struct vkr_cs_encoder *enc,
                                 const struct vkr_resource *res,
                                 size_t offset,
                                 size_t size)
{
   if (!res) {
      memset(&enc->stream, 0, sizeof(enc->stream));
      enc->cur = NULL;
      enc->end = NULL;
      return;
   }

   if (unlikely(size > res->size || offset > res->size - size)) {
      vkr_log(
         "failed to set the reply stream: offset(%zu) + size(%zu) exceeds res size(%zu)",
         offset, size, res->size);
      vkr_cs_encoder_set_fatal(enc);
      return;
   }

   enc->stream.resource = res;
   enc->stream.offset = offset;
   enc->stream.size = size;

   enc->end = res->u.data + res->size;

   vkr_cs_encoder_seek_stream_locked(enc, 0);
}

void
vkr_cs_encoder_seek_stream_locked(struct vkr_cs_encoder *enc, size_t pos)
{
   if (unlikely(!enc->stream.resource || pos > enc->stream.size)) {
      vkr_log_error("failed to seek the reply stream to %zu", pos);
      vkr_cs_encoder_set_fatal(enc);
      return;
   }

   enc->cur = enc->stream.resource->u.data + enc->stream.offset + pos;
}

int
vkr_cs_decoder_init(struct vkr_cs_decoder *dec, struct vkr_context *ctx)
{
   memset(dec, 0, sizeof(*dec));
   dec->ctx = ctx;
   dec->ctx_id = ctx->ctx_id;
   dec->ctx_name = ctx->debug_name;
   dec->fatal_error = &ctx->cs_fatal_error;
   dec->object_table = ctx->object_table;
   dec->object_mutex = &ctx->object_mutex;
   dec->object_gen = &ctx->object_gen;
   return mtx_init(&dec->resource_mutex, mtx_plain);
}

void
vkr_cs_decoder_fini(struct vkr_cs_decoder *dec)
{
   struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;
   for (uint32_t i = 0; i < pool->buffer_count; i++)
      free(pool->buffers[i]);
   if (pool->buffers)
      free(pool->buffers);

   mtx_destroy(&dec->resource_mutex);
}

bool
vkr_cs_decoder_set_resource_stream(struct vkr_cs_decoder *dec,
                                   struct vkr_context *ctx,
                                   uint32_t res_id,
                                   size_t offset,
                                   size_t size)
{
   mtx_lock(&dec->resource_mutex);
   struct vkr_resource *res = vkr_context_get_resource(ctx, res_id);
   if (unlikely(!res || res->fd_type != VIRGL_RESOURCE_FD_SHM || size > res->size ||
                offset > res->size - size)) {
      mtx_unlock(&dec->resource_mutex);
      return false;
   }

   dec->resource = res;
   dec->cur = res->u.data + offset;
   dec->end = dec->cur + size;
   mtx_unlock(&dec->resource_mutex);
   return true;
}

static void
vkr_cs_decoder_sanity_check(const struct vkr_cs_decoder *dec)
{
   const struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;
   assert(pool->buffer_count <= pool->buffer_max);
   if (pool->buffer_count) {
      assert(pool->buffers[pool->buffer_count - 1] <= pool->reset_to);
      assert(pool->reset_to <= pool->cur);
      assert(pool->cur <= pool->end);
   }

   assert(dec->cur <= dec->end);
}

static void
vkr_cs_decoder_gc_temp_pool(struct vkr_cs_decoder *dec)
{
   struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;
   if (!pool->buffer_count)
      return;

   /* free all but the last buffer */
   if (pool->buffer_count > 1) {
      for (uint32_t i = 0; i < pool->buffer_count - 1; i++)
         free(pool->buffers[i]);

      pool->buffers[0] = pool->buffers[pool->buffer_count - 1];
      pool->buffer_count = 1;
   }

   pool->reset_to = pool->buffers[0];
   pool->cur = pool->buffers[0];

   pool->total_size = pool->end - pool->cur;

   vkr_cs_decoder_sanity_check(dec);
}

/**
 * Reset a decoder for reuse.
 */
void
vkr_cs_decoder_reset(struct vkr_cs_decoder *dec)
{
   /* dec->fatal_error is sticky */

   vkr_cs_decoder_gc_temp_pool(dec);

   dec->saved_state_valid = false;
   /* no need to lock decoder here */
   dec->resource = NULL;
   dec->cur = NULL;
   dec->end = NULL;
}

void
vkr_cs_decoder_save_state(struct vkr_cs_decoder *dec)
{
   assert(!dec->saved_state_valid);
   dec->saved_state_valid = true;

   struct vkr_cs_decoder_saved_state *saved = &dec->saved_state;
   saved->cur = dec->cur;
   saved->end = dec->end;

   struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;
   saved->pool_buffer_count = pool->buffer_count;
   saved->pool_reset_to = pool->reset_to;
   /* avoid temp data corruption */
   pool->reset_to = pool->cur;

   vkr_cs_decoder_sanity_check(dec);
}

void
vkr_cs_decoder_restore_state(struct vkr_cs_decoder *dec)
{
   assert(dec->saved_state_valid);
   dec->saved_state_valid = false;

   /* no need to lock decoder here */
   dec->resource = NULL;

   const struct vkr_cs_decoder_saved_state *saved = &dec->saved_state;
   dec->cur = saved->cur;
   dec->end = saved->end;

   /* restore only if pool->reset_to points to the same buffer */
   struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;
   if (pool->buffer_count == saved->pool_buffer_count)
      pool->reset_to = saved->pool_reset_to;

   vkr_cs_decoder_sanity_check(dec);
}

static uint32_t
next_array_size(uint32_t cur_size, uint32_t min_size)
{
   const uint32_t next_size = cur_size ? cur_size * 2 : min_size;
   return next_size > cur_size ? next_size : 0;
}

static size_t
next_buffer_size(size_t cur_size, size_t min_size, size_t need)
{
   size_t next_size = cur_size ? cur_size * 2 : min_size;
   while (next_size < need) {
      next_size *= 2;
      if (!next_size)
         return 0;
   }
   return next_size;
}

static bool
vkr_cs_decoder_grow_temp_pool(struct vkr_cs_decoder *dec)
{
   struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;
   const uint32_t buf_max = next_array_size(pool->buffer_max, 4);
   if (!buf_max)
      return false;

   uint8_t **bufs = realloc(pool->buffers, sizeof(*pool->buffers) * buf_max);
   if (!bufs)
      return false;

   pool->buffers = bufs;
   pool->buffer_max = buf_max;

   return true;
}

bool
vkr_cs_decoder_alloc_temp_internal(struct vkr_cs_decoder *dec, size_t size)
{
   struct vkr_cs_decoder_temp_pool *pool = &dec->temp_pool;

   if (pool->buffer_count >= pool->buffer_max) {
      if (!vkr_cs_decoder_grow_temp_pool(dec))
         return false;
      assert(pool->buffer_count < pool->buffer_max);
   }

   const size_t cur_buf_size =
      pool->buffer_count ? pool->end - pool->buffers[pool->buffer_count - 1] : 0;
   const size_t buf_size = next_buffer_size(cur_buf_size, 4096, size);
   if (!buf_size)
      return false;

   if (buf_size > VKR_CS_DECODER_TEMP_POOL_MAX_SIZE - pool->total_size)
      return false;

   uint8_t *buf = malloc(buf_size);
   if (!buf)
      return false;

   pool->total_size += buf_size;
   pool->buffers[pool->buffer_count++] = buf;
   pool->reset_to = buf;
   pool->cur = buf;
   pool->end = buf + buf_size;

   vkr_cs_decoder_sanity_check(dec);

   return true;
}
