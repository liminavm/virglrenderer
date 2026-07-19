/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VKR_CONTEXT_H
#define VKR_CONTEXT_H

#include "vkr_common.h"
#include "vkr_library.h"

#include "vn_protocol_renderer_defines.h"
#include "vn_protocol_renderer_util.h"
#include "virgl_resource.h"

#include "vkr_cs.h"
#include "vkr_journal.h"

/*
 * When vkr_context_create_resource or vkr_context_import_resource is called, a
 * vkr_resource is created, and is valid until vkr_context_destroy_resource.
 */
struct vkr_resource {
   uint32_t res_id;

   enum virgl_resource_fd_type fd_type;

   /* Which union member is live. Upstream infers it from fd_type (SHM ⇒ data,
    * dma_buf/opaque ⇒ fd), but the limina macOS carrier path breaks that mapping:
    * an export-side mtl_shm carrier blob is FD_SHM *with an owned fd* (there is
    * no server-side mmap of it). Inferring from fd_type made the free path
    * munmap(u.data) — a garbage pointer that is really the fd — so the fd leaked,
    * one per exported carrier blob, until the worker hit EMFILE (2026-07-10
    * dogfood: 12k PSXSHM fds, login storm starved every new venus context). */
   bool u_is_fd;

   union {
      /* valid when u_is_fd (dma_buf/opaque exports, or an FD_SHM mtl_shm carrier) */
      int fd;
      /* valid when !u_is_fd (fd_type is shm and the server holds the mapping) */
      uint8_t *data;
   } u;

   size_t size;

   /* limina tier-2 (macOS): for a cross-context import of a venus-exported blob, where
    * the exporter's pixel bytes actually live — a GLOBAL IOSurface id (window buffers;
    * the SHM above is only a carrier) and/or the exporter's host mapping (#28 map_ptr).
    * vkAllocateMemory(VkImportMemoryResourceInfoMESA) host-pointer-imports these.
    * 0 = not present. */
   uint32_t iosurface_id;
   uint64_t map_ptr;
};

enum vkr_context_validate_level {
   /* no validation */
   VKR_CONTEXT_VALIDATE_NONE,
   /* force enabling a subset of the validation layer */
   VKR_CONTEXT_VALIDATE_ON,
   /* force enabling the validation layer */
   VKR_CONTEXT_VALIDATE_FULL,
};

struct vkr_context {
   uint32_t ctx_id;
   vkr_renderer_retire_fence_callback_type retire_fence;

   char *debug_name;
   enum vkr_context_validate_level validate_level;
   bool validate_fatal;

   /* true if ENABLE_RENDER_SERVER_WORKER_THREAD is defined */
   bool on_worker_thread;

   mtx_t ring_mutex;
   struct list_head rings;

   struct {
      mtx_t mutex;
      cnd_t cond;
      uint64_t id;
      /* This represents the ring head position to be waited on. The protocol supports
       * 64bit seqno and we only use 32bit internally because the delta between the ring
       * head and ring current will never exceed the ring size, which is far smaller than
       * 32bit int limit in practice.
       */
      uint32_t seqno;
   } wait_ring;

   struct {
      mtx_t mutex;
      cnd_t cond;
      thrd_t thread;
      atomic_bool started;

      /* When monitoring multiple rings, wake to report on all rings at the minimum of
       * per-ring maxReportingPeriodMicroseconds to ensure that every ring is marked ALIVE
       * before the next renderer check.
       */
      atomic_uint report_period_us;
   } ring_monitor;

   mtx_t object_mutex;
   struct hash_table *object_table;

   /* limina: snapshot-replay re-creation journal (vkr_journal.h); NULL when
    * disabled (VKR_JOURNAL=0) or after teardown starts */
   struct vkr_journal *journal;

   mtx_t resource_mutex;
   struct hash_table *resource_table;

   bool cs_fatal_error;
   struct vkr_cs_encoder encoder;
   struct vkr_cs_decoder decoder;
   struct vn_dispatch_context dispatch;

   PFN_vkGetInstanceProcAddr get_proc_addr;
   struct vn_global_proc_table proc_table;

   struct vkr_queue *sync_queues[64];

   struct vkr_instance *instance;
   char *instance_name;

   struct list_head head;
   struct vulkan_library vulkan_library;
};

struct vkr_context *
vkr_context_create(uint32_t ctx_id,
                   vkr_renderer_retire_fence_callback_type cb,
                   size_t debug_len,
                   const char *debug_name);

void
vkr_context_destroy(struct vkr_context *ctx);

bool
vkr_context_ring_monitor_init(struct vkr_context *ctx, uint32_t report_period_us);

bool
vkr_context_submit_fence(struct vkr_context *ctx,
                         uint32_t flags,
                         uint32_t ring_idx,
                         uint64_t fence_id);

/* limina (#8): release one phase-1 (ring barrier) reference on a present fence;
 * the last release submits the phase-2 GPU syncs (or retires if idle). */
struct vkr_present_fence;
void
vkr_limina_present_barrier_release(struct vkr_present_fence *pf);

bool
vkr_context_submit_cmd(struct vkr_context *ctx, const void *buffer, size_t size);

bool
vkr_context_create_resource(struct vkr_context *ctx,
                            uint32_t res_id,
                            uint64_t blob_id,
                            uint64_t blob_size,
                            uint32_t blob_flags,
                            struct virgl_context_blob *out_blob);

bool
vkr_context_import_resource(struct vkr_context *ctx,
                            uint32_t res_id,
                            enum virgl_resource_fd_type fd_type,
                            int fd,
                            uint64_t size,
                            uint32_t iosurface_id,
                            uint64_t map_ptr);

void
vkr_context_destroy_resource(struct vkr_context *ctx, uint32_t res_id);

static inline struct vkr_resource *
vkr_context_get_resource(struct vkr_context *ctx, uint32_t res_id)
{
   mtx_lock(&ctx->resource_mutex);
   const struct hash_entry *entry = _mesa_hash_table_search(ctx->resource_table, &res_id);
   mtx_unlock(&ctx->resource_mutex);

   return likely(entry) ? entry->data : NULL;
}

/* Poisoning the context kills the guest's venus ring: mesa aborts the whole
 * guest process on the FATAL bit (vn_relax). Always name the site in the log
 * at ERROR so a production deployment records WHICH handler poisoned the ring
 * (2026-07-09 dogfood: three silent aborts, unattributable at default level). */
static inline void
vkr_context_set_fatal_at(struct vkr_context *ctx, const char *func, int line)
{
   if (!ctx->cs_fatal_error)
      vkr_log_error("context %u: ring FATAL set at %s:%d", ctx->ctx_id, func, line);
   ctx->cs_fatal_error = true;
}

#define vkr_context_set_fatal(ctx) vkr_context_set_fatal_at((ctx), __func__, __LINE__)

/* For paths where FATAL is the expected corollary of client teardown (kernel
 * DRM-fd cleanup detaching a ring's backing after the client exited), not a
 * defect: park the context without the ERROR-level site log. Callers log the
 * specifics at INFO instead. */
static inline void
vkr_context_set_fatal_quiet(struct vkr_context *ctx)
{
   ctx->cs_fatal_error = true;
}

static inline bool
vkr_context_get_fatal(struct vkr_context *ctx)
{
   return ctx->cs_fatal_error;
}

static inline bool
vkr_context_validate_object_id(struct vkr_context *ctx, vkr_object_id id)
{
   mtx_lock(&ctx->object_mutex);
   if (unlikely(!id || _mesa_hash_table_search(ctx->object_table, &id))) {
      mtx_unlock(&ctx->object_mutex);
      vkr_log_error("invalid object id %" PRIu64, id);
      vkr_context_set_fatal(ctx);
      return false;
   }
   mtx_unlock(&ctx->object_mutex);

   return true;
}

static inline void *
vkr_context_alloc_object(UNUSED struct vkr_context *ctx,
                         size_t size,
                         VkObjectType type,
                         const void *id_handle)
{
   const vkr_object_id id = vkr_cs_handle_load_id((const void **)id_handle, type);
   if (!vkr_context_validate_object_id(ctx, id))
      return NULL;

   return vkr_object_alloc(size, type, id);
}

void
vkr_context_free_object(struct hash_entry *entry);

static inline void
vkr_context_add_object(struct vkr_context *ctx, struct vkr_object *obj)
{
   assert(vkr_is_recognized_object_type(obj->type));
   assert(obj->id);

   mtx_lock(&ctx->object_mutex);
   assert(!_mesa_hash_table_search(ctx->object_table, &obj->id));
   _mesa_hash_table_insert(ctx->object_table, &obj->id, obj);
   /* limina journal: key the current command's entry by this id (TLS append;
    * lock order here and in the remove hook: object_mutex -> journal mutex) */
   vkr_journal_object_added(ctx, obj->id, obj->type);
   mtx_unlock(&ctx->object_mutex);
}

static inline void
vkr_context_remove_object_locked(struct vkr_context *ctx, struct vkr_object *obj)
{
   assert(_mesa_hash_table_search(ctx->object_table, &obj->id));

   /* limina journal: prune every retained entry keyed by this id (before the
    * object is freed, while object_mutex still serializes id reuse) */
   vkr_journal_object_removed(ctx, obj->id);

   struct hash_entry *entry = _mesa_hash_table_search(ctx->object_table, &obj->id);
   if (likely(entry)) {
      vkr_context_free_object(entry);
      _mesa_hash_table_remove(ctx->object_table, entry);
   }
}

static inline void
vkr_context_remove_object(struct vkr_context *ctx, struct vkr_object *obj)
{
   mtx_lock(&ctx->object_mutex);
   vkr_context_remove_object_locked(ctx, obj);
   mtx_unlock(&ctx->object_mutex);
}

static inline void
vkr_context_remove_objects(struct vkr_context *ctx, struct list_head *objects)
{
   mtx_lock(&ctx->object_mutex);
   list_for_each_entry_safe (struct vkr_object, obj, objects, track_head)
      vkr_context_remove_object_locked(ctx, obj);
   mtx_unlock(&ctx->object_mutex);
   /* objects should be reinitialized if to be reused */
}

static inline void *
vkr_context_get_object(struct vkr_context *ctx, vkr_object_id obj_id)
{
   mtx_lock(&ctx->object_mutex);
   const struct hash_entry *entry = _mesa_hash_table_search(ctx->object_table, &obj_id);
   void *obj = likely(entry) ? entry->data : NULL;
   mtx_unlock(&ctx->object_mutex);
   return obj;
}

void
vkr_context_on_ring_seqno_update(struct vkr_context *ctx,
                                 uint64_t ring_id,
                                 uint32_t ring_seqno);

bool
vkr_context_get_wait_ring_seqno(struct vkr_context *ctx,
                                uint64_t ring_id,
                                uint32_t *out_seqno);

void
vkr_context_on_ring_fatal(struct vkr_context *ctx);

bool
vkr_context_wait_ring_seqno(struct vkr_context *ctx,
                            struct vkr_ring *ring,
                            uint64_t ring_seqno);

void
vkr_context_add_instance(struct vkr_context *ctx,
                         struct vkr_instance *instance,
                         const char *name);

void
vkr_context_remove_instance(struct vkr_context *ctx, struct vkr_instance *instance);

#endif /* VKR_CONTEXT_H */
