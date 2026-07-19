/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_context.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#ifdef __APPLE__
#include <dirent.h>
#include <pthread/qos.h>
#include <sys/resource.h>
#endif
#include <sys/types.h>
#include <unistd.h>

#include "util/anon_file.h"
#include "venus-protocol/vn_protocol_renderer_dispatches.h"

#define XXH_INLINE_ALL
#include "util/xxhash.h"

#include "vkr_acceleration_structure.h"
#include "vkr_buffer.h"
#include "vkr_command_buffer.h"
#include "vkr_context.h"
#include "vkr_cs.h"
#include "vkr_descriptor_heap.h"
#include "vkr_descriptor_set.h"
#include "vkr_device.h"
#include "vkr_device_memory.h"
#include "vkr_host_copy.h"
#include "vkr_image.h"
#include "vkr_instance.h"
#include "vkr_physical_device.h"
#include "vkr_pipeline.h"
#include "vkr_query_pool.h"
#include "vkr_queue.h"
#include "vkr_render_pass.h"
#include "vkr_ring.h"
#include "vkr_transport.h"

void
vkr_context_add_instance(struct vkr_context *ctx,
                         struct vkr_instance *instance,
                         const char *name)
{
   vkr_context_add_object(ctx, &instance->base);

   assert(!ctx->instance);
   ctx->instance = instance;

   if (name && name[0] != '\0') {
      assert(!ctx->instance_name);
      ctx->instance_name = strdup(name);
   }
}

void
vkr_context_remove_instance(struct vkr_context *ctx, struct vkr_instance *instance)
{
   assert(ctx->instance && ctx->instance == instance);
   ctx->instance = NULL;

   if (ctx->instance_name) {
      free(ctx->instance_name);
      ctx->instance_name = NULL;
   }

   vkr_context_remove_object(ctx, &instance->base);
}

static void
vkr_dispatch_debug_log(UNUSED struct vn_dispatch_context *dispatch, const char *msg)
{
   vkr_log(msg);
}

static void
vkr_context_init_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->data = ctx;
   dispatch->debug_log = vkr_dispatch_debug_log;

   dispatch->encoder = (struct vn_cs_encoder *)&ctx->encoder;
   dispatch->decoder = (struct vn_cs_decoder *)&ctx->decoder;

   vkr_context_init_transport_dispatch(ctx);

   vkr_context_init_instance_dispatch(ctx);
   vkr_context_init_physical_device_dispatch(ctx);
   vkr_context_init_device_dispatch(ctx);

   vkr_context_init_queue_dispatch(ctx);
   vkr_context_init_fence_dispatch(ctx);
   vkr_context_init_semaphore_dispatch(ctx);
   vkr_context_init_event_dispatch(ctx);

   vkr_context_init_device_memory_dispatch(ctx);

   vkr_context_init_buffer_dispatch(ctx);
   vkr_context_init_buffer_view_dispatch(ctx);

   vkr_context_init_image_dispatch(ctx);
   vkr_context_init_image_view_dispatch(ctx);
   vkr_context_init_sampler_dispatch(ctx);
   vkr_context_init_sampler_ycbcr_conversion_dispatch(ctx);

   vkr_context_init_descriptor_heap_dispatch(ctx);
   vkr_context_init_descriptor_set_layout_dispatch(ctx);
   vkr_context_init_descriptor_pool_dispatch(ctx);
   vkr_context_init_descriptor_set_dispatch(ctx);
   vkr_context_init_descriptor_update_template_dispatch(ctx);

   vkr_context_init_render_pass_dispatch(ctx);
   vkr_context_init_framebuffer_dispatch(ctx);

   vkr_context_init_query_pool_dispatch(ctx);

   vkr_context_init_shader_module_dispatch(ctx);
   vkr_context_init_pipeline_layout_dispatch(ctx);
   vkr_context_init_pipeline_cache_dispatch(ctx);
   vkr_context_init_pipeline_dispatch(ctx);

   vkr_context_init_command_pool_dispatch(ctx);
   vkr_context_init_command_buffer_dispatch(ctx);

   vkr_context_init_host_copy_dispatch(ctx);

   vkr_context_init_acceleration_structure_dispatch(ctx);
}

static inline void
vkr_context_init_proc_table(struct vkr_context *ctx)
{
   /* Get vkGetInstanceProcAddr from libvulkan */
   ctx->get_proc_addr = ctx->vulkan_library.GetInstanceProcAddr;
   vn_util_init_global_proc_table(ctx->get_proc_addr, &ctx->proc_table);
}

/* limina (#8): phase-1 (ring barrier) release. The release that drops the count
 * to zero means every ring has decoded past its registration point, so the
 * frame's vkQueueSubmit reached the driver — submit phase-2 GPU syncs. */
void
vkr_limina_present_barrier_release(struct vkr_present_fence *pf)
{
   if (atomic_fetch_sub(&pf->pending, 1) != 1)
      return;

   struct vkr_context *ctx = pf->ctx;

   /* collect the context's distinct queues (sync_queues maps ring_idx ->
    * queue, indices from 1; a queue can appear under several indices) */
   struct vkr_queue *queues[ARRAY_SIZE(ctx->sync_queues)];
   uint32_t count = 0;
   for (uint32_t i = 1; i < ARRAY_SIZE(ctx->sync_queues); i++) {
      struct vkr_queue *q = ctx->sync_queues[i];
      if (!q)
         continue;
      bool seen = false;
      for (uint32_t j = 0; j < count; j++)
         seen |= queues[j] == q;
      if (!seen)
         queues[count++] = q;
   }

   if (!count) {
      /* nothing on the GPU — retire now */
      ctx->retire_fence(ctx->ctx_id, VKR_LIMINA_PRESENT_RING, pf->fence_id);
      free(pf);
      return;
   }

   /* sole owner here: re-arm the count for phase 2 */
   atomic_store(&pf->pending, (int)count);
   for (uint32_t i = 0; i < count; i++) {
      if (!vkr_queue_sync_submit_present(queues[i], pf))
         vkr_present_fence_release(pf);
   }
}

/* limina (#8): entry point for a VMM-injected scanout-flush fence (reserved ring
 * VKR_LIMINA_PRESENT_RING). Registers a barrier on every ring, then GPU syncs on
 * every queue; retire_fence fires with the same ring/fence_id when all pass. */
static bool
vkr_context_limina_present_fence(struct vkr_context *ctx, uint64_t fence_id)
{
   struct vkr_present_fence *pf = malloc(sizeof(*pf));
   if (!pf)
      return false;

   pf->ctx = ctx;
   pf->fence_id = fence_id;
   atomic_init(&pf->pending, 1); /* registrar guard */

   mtx_lock(&ctx->ring_mutex);
   list_for_each_entry (struct vkr_ring, ring, &ctx->rings, head) {
      atomic_fetch_add(&pf->pending, 1);
      vkr_ring_add_limina_barrier(ring, pf);
   }
   mtx_unlock(&ctx->ring_mutex);

   vkr_limina_present_barrier_release(pf); /* drop the guard */
   return true;
}

bool
vkr_context_submit_fence(struct vkr_context *ctx,
                         uint32_t flags,
                         uint32_t ring_idx,
                         uint64_t fence_id)
{
   /* limina (#8): the reserved present ring takes the barrier+sync path */
   if (ring_idx == VKR_LIMINA_PRESENT_RING)
      return vkr_context_limina_present_fence(ctx, fence_id);

   /* retire fence on cpu timeline directly */
   if (ring_idx == 0) {
      ctx->retire_fence(ctx->ctx_id, ring_idx, fence_id);
      return true;
   }

   if (ring_idx >= ARRAY_SIZE(ctx->sync_queues) || !ctx->sync_queues[ring_idx]) {
      vkr_log("submit_fence: invalid ring_idx %u", ring_idx);
      return false;
   }

   /* always merge fences */
   assert(!(flags & ~VIRGL_RENDERER_FENCE_FLAG_MERGEABLE));
   flags = VIRGL_RENDERER_FENCE_FLAG_MERGEABLE;
   bool ok = vkr_queue_sync_submit(ctx->sync_queues[ring_idx], flags, ring_idx, fence_id);

   return ok;
}

bool
vkr_context_submit_cmd(struct vkr_context *ctx, const void *buffer, size_t size)
{
   /* CS error is considered fatal (destroy the context?) */
   if (vkr_context_get_fatal(ctx)) {
      vkr_log("submit_cmd: early bail due to fatal decoder state");
      return false;
   }

   vkr_cs_decoder_set_buffer_stream(&ctx->decoder, buffer, size);

   while (vkr_cs_decoder_has_command(&ctx->decoder)) {
      vn_dispatch_command(&ctx->dispatch);
      if (vkr_context_get_fatal(ctx)) {
         vkr_log("submit_cmd: vn_dispatch_command failed");

         vkr_cs_decoder_reset(&ctx->decoder);
         return false;
      }
   }

   vkr_cs_decoder_reset(&ctx->decoder);
   return true;
}

static inline void
vkr_context_free_resource(struct hash_entry *entry)
{
   struct vkr_resource *res = entry->data;
   if (vkr_fd_trace())
      vkr_log_error("[FDTRACE] vkr free_resource res=%u %s=%d", res->res_id,
                    res->u_is_fd ? "fd" : "mapped size",
                    res->u_is_fd ? res->u.fd : (int)res->size);
   if (res->u_is_fd) {
      if (res->u.fd >= 0)
         close(res->u.fd);
   } else {
      munmap(res->u.data, res->size);
   }
   free(res);
}

static inline bool
vkr_context_add_resource(struct vkr_context *ctx, struct vkr_resource *res)
{
   mtx_lock(&ctx->resource_mutex);
   assert(!_mesa_hash_table_search(ctx->resource_table, &res->res_id));
   struct hash_entry *entry =
      _mesa_hash_table_insert(ctx->resource_table, &res->res_id, res);
   mtx_unlock(&ctx->resource_mutex);

   return entry;
}

static inline void
vkr_context_remove_resource(struct vkr_context *ctx, uint32_t res_id)
{
   mtx_lock(&ctx->resource_mutex);
   struct hash_entry *entry = _mesa_hash_table_search(ctx->resource_table, &res_id);
   if (likely(entry)) {
      vkr_context_free_resource(entry);
      _mesa_hash_table_remove(ctx->resource_table, entry);
   }
   mtx_unlock(&ctx->resource_mutex);
   /* limina snapshot-replay: NO unpin here — this is a per-context DETACH, and a
    * cross-context-shared blob (Xwayland window buffers) outlives the exporting
    * context's attachment. The pin is released at the GLOBAL resource unref,
    * driven by the libkrun rutabaga journal (virgl_renderer_limina_journal_unpin). */
}

static bool
vkr_context_import_resource_internal(struct vkr_context *ctx,
                                     uint32_t res_id,
                                     uint64_t blob_size,
                                     enum virgl_resource_fd_type fd_type,
                                     int fd,
                                     void *mmap_ptr)
{
   assert(!vkr_context_get_resource(ctx, res_id));

   struct vkr_resource *res = malloc(sizeof(*res));
   if (!res)
      return false;

   res->res_id = res_id;
   res->fd_type = fd_type;
   res->size = blob_size;
   res->iosurface_id = 0;
   res->map_ptr = 0;

   /* fd and mmap_ptr cannot be valid at the same time, but allowed to be -1 and NULL */
   assert(fd < 0 || !mmap_ptr);
   if (mmap_ptr) {
      res->u_is_fd = false;
      res->u.data = mmap_ptr;
   } else {
      res->u_is_fd = true;
      res->u.fd = fd;
   }

   if (!vkr_context_add_resource(ctx, res)) {
      free(res);
      return false;
   }

   return true;
}

static bool
vkr_context_import_resource_from_shm(struct vkr_context *ctx,
                                     uint32_t res_id,
                                     uint64_t blob_size,
                                     int fd)
{
   assert(!vkr_context_get_resource(ctx, res_id));

   void *mmap_ptr = mmap(NULL, blob_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
   if (mmap_ptr == MAP_FAILED)
      return false;

   if (!vkr_context_import_resource_internal(ctx, res_id, blob_size,
                                             VIRGL_RESOURCE_FD_SHM, -1, mmap_ptr)) {
      munmap(mmap_ptr, blob_size);
      return false;
   }

   /* Import CONSUMES the fd on success, like the internal path consumes it by storing
    * it in the vkr_resource. Only the mapping is kept here — an unclosed fd is a
    * straight leak, one per cross-context attach of an SHM blob (the render-server
    * dispatch only closes received fds on dispatch FAILURE). On a macOS host every
    * exportable VkDeviceMemory is an SHM carrier, so compositor dmabuf-import churn
    * ratchets the worker to RLIMIT_NOFILE (EMFILE) without this. */
   if (vkr_fd_trace())
      vkr_log_error("[FDTRACE] shm import res=%u consumed fd=%d", res_id, fd);
   close(fd);

   return true;
}

static bool
vkr_context_create_resource_from_shm(struct vkr_context *ctx,
                                     uint32_t res_id,
                                     uint64_t blob_size,
                                     struct virgl_context_blob *out_blob)
{
   assert(!vkr_context_get_resource(ctx, res_id));

   /* Round up to host page size. The VMM maps this resource with
    * MAP_FIXED which requires page-aligned sizes.
    */
   const size_t page_size = getpagesize();
   const uint64_t alloc_size = (blob_size + page_size - 1) & ~(page_size - 1);

   int fd = os_create_anonymous_file(alloc_size, "vkr-shmem");
   if (fd < 0) {
      vkr_log_error("context %u: CREATE_BLOB res %u: anonymous shm of %" PRIu64
                    " bytes failed (%s) — guest holds a dead resource id",
                    ctx->ctx_id, res_id, alloc_size, strerror(errno));
      return false;
   }

   void *mmap_ptr = mmap(NULL, alloc_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
   if (mmap_ptr == MAP_FAILED) {
      vkr_log_error("context %u: CREATE_BLOB res %u: mmap of %" PRIu64
                    " bytes failed (%s) — guest holds a dead resource id",
                    ctx->ctx_id, res_id, alloc_size, strerror(errno));
      close(fd);
      return false;
   }

   if (!vkr_context_import_resource_internal(ctx, res_id, alloc_size,
                                             VIRGL_RESOURCE_FD_SHM, -1, mmap_ptr)) {
      munmap(mmap_ptr, alloc_size);
      close(fd);
      return false;
   }

   *out_blob = (struct virgl_context_blob){
      .type = VIRGL_RESOURCE_FD_SHM,
      .u.fd = fd,
      .map_info = VIRGL_RENDERER_MAP_CACHE_CACHED,
   };

   return true;
}

static bool
vkr_context_create_resource_from_device_memory(struct vkr_context *ctx,
                                               uint32_t res_id,
                                               uint64_t blob_id,
                                               uint64_t blob_size,
                                               uint32_t blob_flags,
                                               struct virgl_context_blob *out_blob)
{
   assert(!vkr_context_get_resource(ctx, res_id));

   struct vkr_device_memory *mem = vkr_context_get_object(ctx, blob_id);
   if (!mem || mem->base.type != VK_OBJECT_TYPE_DEVICE_MEMORY) {
      /* The guest kernel treats CREATE_BLOB as fire-and-forget, so this error
       * only reaches it as an async dmesg line — the guest process holds a
       * live-looking handle to a resource that never existed. Name the cause
       * here; the dead id will resurface downstream (map/props/import). */
      vkr_log_error("context %u: CREATE_BLOB res %u: blob_id %" PRIu64
                    " is not a live VkDeviceMemory (%s) — guest holds a dead "
                    "resource id",
                    ctx->ctx_id, res_id, blob_id, mem ? "wrong type" : "not found");
      return false;
   }

   struct virgl_context_blob blob;
   if (!vkr_device_memory_export_blob(mem, blob_size, blob_flags, &blob)) {
      vkr_log_error("context %u: CREATE_BLOB res %u: export of VkDeviceMemory "
                    "blob_id %" PRIu64 " failed (size %" PRIu64 " flags 0x%x) — "
                    "guest holds a dead resource id",
                    ctx->ctx_id, res_id, blob_id, blob_size, blob_flags);
      return false;
   }

#ifdef __APPLE__
   /* limina #28: HOST_VISIBLE blob shares MoltenVK's own vkMapMemory pointer (blob.map_ptr) — no
    * fd. Register the vkr_resource with no fd (FD_OPAQUE, fd=-1 → vkr_context_free_resource
    * no-ops: not SHM, and u.fd < 0). The borrowed pointer rides on the virgl_resource via
    * out_blob.map_ptr; the VkDeviceMemory owns/unmaps it (vkr_device_memory_release). */
   if (blob.map_ptr) {
      if (!vkr_context_import_resource_internal(ctx, res_id, blob_size,
                                                VIRGL_RESOURCE_FD_OPAQUE, -1, NULL))
         return false;

      /* Self-imports (vkAllocateMemory with this res in the SAME ctx) resolve the
       * bytes via these — u.fd is -1 here and useless. */
      struct vkr_resource *limina_res = vkr_context_get_resource(ctx, res_id);
      limina_res->map_ptr = (uintptr_t)blob.map_ptr;
      limina_res->iosurface_id = blob.iosurface_id;
      vkr_journal_pin_key(ctx, blob_id, mem->limina_dedicated_id);

      *out_blob = blob;
      return true;
   }
#endif

   /* If memory might get exported, store a dup'ed fd in vkr_resource for:
    * - vkAllocateMemory for dma_buf import
    * - vkGetMemoryFdPropertiesKHR for dma_buf fd properties query
    */
   int res_fd = -1;
   if (mem->might_export) {
      res_fd = os_dupfd_cloexec(blob.u.fd);
      if (res_fd < 0) {
         close(blob.u.fd);
         return false;
      }
      if (vkr_fd_trace())
         vkr_log_error("[FDTRACE] create_blob res=%u might_export src_fd=%d dup=%d",
                       res_id, blob.u.fd, res_fd);
   }

   if (!vkr_context_import_resource_internal(ctx, res_id, blob_size, blob.type, res_fd,
                                             NULL)) {
      if (res_fd >= 0)
         close(res_fd);
      close(blob.u.fd);
      return false;
   }

#ifdef __APPLE__
   /* limina: SHM blobs of scanout memories are CARRIERS (pixels live in the IOSurface);
    * record the id so a self-import of this resource aliases the real bytes. NOTE this
    * resource's u.fd holds an fd, NOT a mapping — never read u.data on this path. */
   if (blob.iosurface_id) {
      struct vkr_resource *limina_res = vkr_context_get_resource(ctx, res_id);
      limina_res->iosurface_id = blob.iosurface_id;
   }
#endif

   /* limina snapshot-replay: pin the backing memory's journal entries for this
    * resource's lifetime (guest may free the memory while the blob lives) */
   vkr_journal_pin_key(ctx, blob_id, mem->limina_dedicated_id);

   *out_blob = blob;

   return true;
}

bool
vkr_context_create_resource(struct vkr_context *ctx,
                            uint32_t res_id,
                            uint64_t blob_id,
                            uint64_t blob_size,
                            uint32_t blob_flags,
                            struct virgl_context_blob *out_blob)
{
   /* blob_id == 0 does not refer to an existing VkDeviceMemory, but implies a shm
    * allocation. It is logically contiguous and it can be exported.
    */
   if (!blob_id && blob_flags == VIRGL_RENDERER_BLOB_FLAG_USE_MAPPABLE)
      return vkr_context_create_resource_from_shm(ctx, res_id, blob_size, out_blob);

   return vkr_context_create_resource_from_device_memory(ctx, res_id, blob_id, blob_size,
                                                         blob_flags, out_blob);
}

bool
vkr_context_import_resource(struct vkr_context *ctx,
                            uint32_t res_id,
                            enum virgl_resource_fd_type fd_type,
                            int fd,
                            uint64_t size,
                            uint32_t iosurface_id,
                            uint64_t map_ptr)
{
   bool ok;
   if (fd_type == VIRGL_RESOURCE_FD_SHM)
      ok = vkr_context_import_resource_from_shm(ctx, res_id, size, fd);
   else
      ok = vkr_context_import_resource_internal(ctx, res_id, size, fd_type, fd, NULL);

   /* limina tier-2 (macOS): remember where the exporter's pixel bytes live so memory
    * imports can alias them (see struct vkr_resource). */
   if (ok && (iosurface_id || map_ptr)) {
      struct vkr_resource *res = vkr_context_get_resource(ctx, res_id);
      res->iosurface_id = iosurface_id;
      res->map_ptr = map_ptr;
   }

   return ok;
}

void
vkr_context_destroy_resource(struct vkr_context *ctx, uint32_t res_id)
{
   struct vkr_resource *res = vkr_context_get_resource(ctx, res_id);
   if (!res)
      return;

   /* A detach that yanks the reply stream or a live ring's backing is almost
    * always kernel DRM-fd cleanup of an exited client — few clients destroy
    * their VkInstance before exit, so the ring outlives its blob and this
    * path fires on every venus client exit (2026-07-10 dogfood: all such
    * events were session churn, zero guest crashes). Park the context with
    * the quiet setter and log at INFO; the FATAL bit still stops the ring in
    * case a live client really did detach its own load-bearing blob. */
   if (!vkr_cs_encoder_check_stream(&ctx->encoder, res)) {
      vkr_log("context %u: resource %u detached while backing the reply "
              "stream (normal at client teardown); parking the context",
              ctx->ctx_id, res_id);
      vkr_context_set_fatal_quiet(ctx);
   }

   mtx_lock(&ctx->ring_mutex);
   list_for_each_entry_safe (struct vkr_ring, ring, &ctx->rings, head) {
      if (ring->resource == res ||
          !vkr_cs_decoder_check_stream(&ring->decoder, res) ||
          !vkr_cs_encoder_check_stream(&ring->encoder, res)) {
         vkr_log("context %u: resource %u detached while backing a live ring "
                 "(normal at client teardown); stopping the ring",
                 ctx->ctx_id, res_id);
         vkr_context_set_fatal_quiet(ctx);

         mtx_unlock(&ctx->ring_mutex);
         vkr_ring_stop(ring);
         mtx_lock(&ctx->ring_mutex);

         vkr_ring_destroy(ring);
      }
   }
   mtx_unlock(&ctx->ring_mutex);

   vkr_context_remove_resource(ctx, res_id);
}

void
vkr_context_on_ring_seqno_update(struct vkr_context *ctx,
                                 uint64_t ring_id,
                                 uint32_t ring_seqno)
{
   mtx_lock(&ctx->wait_ring.mutex);
   if (ctx->wait_ring.id == ring_id && vkr_seqno_ge(ring_seqno, ctx->wait_ring.seqno))
      cnd_signal(&ctx->wait_ring.cond);
   mtx_unlock(&ctx->wait_ring.mutex);
}

bool
vkr_context_get_wait_ring_seqno(struct vkr_context *ctx,
                                uint64_t ring_id,
                                uint32_t *out_seqno)
{
   bool wait_ring = false;
   mtx_lock(&ctx->wait_ring.mutex);
   if (ctx->wait_ring.id == ring_id) {
      wait_ring = true;
      *out_seqno = ctx->wait_ring.seqno;
   }
   mtx_unlock(&ctx->wait_ring.mutex);
   return wait_ring;
}

void
vkr_context_on_ring_fatal(struct vkr_context *ctx)
{
   vkr_context_set_fatal(ctx);

   mtx_lock(&ctx->wait_ring.mutex);
   cnd_signal(&ctx->wait_ring.cond);
   mtx_unlock(&ctx->wait_ring.mutex);
}

bool
vkr_context_wait_ring_seqno(struct vkr_context *ctx,
                            struct vkr_ring *ring,
                            uint64_t ring_seqno)
{
   TRACE_FUNC();

   bool ok = true;

   mtx_lock(&ctx->wait_ring.mutex);
   ctx->wait_ring.id = ring->id;
   ctx->wait_ring.seqno = ring_seqno;
   while (!vkr_context_get_fatal(ctx) && ok &&
          !vkr_seqno_ge(vkr_ring_load_head(ring), ring_seqno)) {
      ok = cnd_wait(&ctx->wait_ring.cond, &ctx->wait_ring.mutex) == thrd_success;
   }
   ctx->wait_ring.id = 0;
   mtx_unlock(&ctx->wait_ring.mutex);

   return ok;
}

static inline const char *
vkr_context_get_name(const struct vkr_context *ctx)
{
   /* ctx->instance_name is the application name while ctx->debug_name is
    * usually the guest process name or the hypervisor name.  This never
    * returns NULL because ctx->debug_name is never NULL.
    */
   return ctx->instance_name ? ctx->instance_name : ctx->debug_name;
}

static inline void
vkr_context_wait_ring_fini(struct vkr_context *ctx)
{
   cnd_destroy(&ctx->wait_ring.cond);
   mtx_destroy(&ctx->wait_ring.mutex);
}

static bool
vkr_context_wait_ring_init(struct vkr_context *ctx)
{
   if (mtx_init(&ctx->wait_ring.mutex, mtx_plain) != thrd_success)
      return false;

   if (cnd_init(&ctx->wait_ring.cond) != thrd_success) {
      mtx_destroy(&ctx->wait_ring.mutex);
      return false;
   }

   return true;
}

static struct timespec
timespec_add(struct timespec a, struct timespec b)
{
   /* handle only the non-negative case, unless needed. */
   assert(a.tv_sec >= 0 && a.tv_nsec >= 0 && b.tv_sec >= 0 && b.tv_nsec >= 0);

#define NS_PER_SEC 1000000000
   a.tv_sec += b.tv_sec;
   a.tv_nsec += b.tv_nsec;
   if (a.tv_nsec >= NS_PER_SEC) {
      a.tv_sec += 1;
      a.tv_nsec -= NS_PER_SEC;
   }
#undef NS_PER_SEC

   return a;
}

static int
vkr_context_ring_monitor_thread(void *arg)
{
   struct vkr_context *ctx = arg;

   char thread_name[16];
   snprintf(thread_name, ARRAY_SIZE(thread_name), "vkr-ringmon-%d", ctx->ctx_id);
   u_thread_setname(thread_name);

#ifdef __APPLE__
   /* limina: the guest venus watchdog aborts the WHOLE guest process when an
    * ALIVE stamp lands late (mesa clears the bit at wait start and re-checks
    * ~3.48s later). This thread's work is trivial; make sure the scheduler
    * treats its deadline as user-critical so host memory pressure/thrash
    * can't starve it. */
   pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

   /* limina: stamp at 1/3 of the requested MAX reporting period (an upper
    * bound per the protocol, so stamping more often is always compliant).
    * mesa requests 3.0s and re-checks ~3.48s after clearing ALIVE — stamping
    * at exactly the max leaves ~480ms of worst-case scheduler slack, and one
    * late monitor wakeup aborts the guest process (observed 489ms late on a
    * near-idle host, 2026-07-10, despite the QoS pin). Oversampling widens
    * the tolerated lateness to ~2.5s for EVERY guest, stock mesa included,
    * at the cost of a few extra atomic stores per second. */
   const uint32_t stamp_period_us =
      ctx->ring_monitor.report_period_us > 300000
         ? ctx->ring_monitor.report_period_us / 3
         : ctx->ring_monitor.report_period_us;

   struct timespec abs_ts;
   int ret = thrd_busy;
   int64_t last_stamp_ns = 0;
   assert(ctx->ring_monitor.started);
   while (ctx->ring_monitor.started) {
      if (ret == thrd_busy) {
         /* limina: a stamp gap beyond the REQUESTED period means the guest
          * watchdog's contract was busted and it may abort the guest
          * process; make that loud so the next dogfood incident is
          * attributable (2026-07-09: three silent alive-expiry-suspected
          * aborts). Measure the gap at the STAMP itself — after the wake
          * AND after ring_mutex is acquired — so time lost blocked on the
          * lock is charged too, not just scheduler lateness. (2026-07-11:
          * a clean alive-expiry abort left ZERO late lines because the gap
          * was checked pre-lock and the next iteration never ran — the
          * victim's teardown signals this thread awake to exit, silencing
          * the one log that would have named the stall.) */
         mtx_lock(&ctx->ring_mutex);
         struct timespec mono;
         if (!clock_gettime(CLOCK_MONOTONIC, &mono)) {
            const int64_t now_ns = (int64_t)mono.tv_sec * 1000000000 + mono.tv_nsec;
            const int64_t period_ns =
               (int64_t)ctx->ring_monitor.report_period_us * 1000;
            if (last_stamp_ns && now_ns - last_stamp_ns > period_ns)
               vkr_log_error("ringmon-%d: ALIVE stamp late: %" PRId64
                             " ms since last (guest tolerates ~%u ms) — guest "
                             "venus watchdog may abort the guest process",
                             ctx->ctx_id, (now_ns - last_stamp_ns) / 1000000,
                             ctx->ring_monitor.report_period_us / 1000);
            last_stamp_ns = now_ns;
         }
         list_for_each_entry (struct vkr_ring, ring, &ctx->rings, head) {
            if (ring->monitor)
               vkr_ring_set_status_bits(ring, VK_RING_STATUS_ALIVE_BIT_MESA);
         }
         mtx_unlock(&ctx->ring_mutex);

#ifdef __APPLE__
         /* limina: fd-pressure canary. The worker is a long-lived singleton and a
          * carrier-fd ratchet walks it to EMFILE, where every CREATE_BLOB dies and
          * fresh guest sessions starve at bring-up (2026-07-10 incident; 2026-07-11:
          * 14k+ fds from a still-unattributed dup site). Audit once a minute from
          * whichever ring monitor gets here first and confess ABOVE 50% of the soft
          * limit, so the wall is visible days before it hits. Benign data race on
          * the timestamp: worst case two monitors audit the same minute. */
         {
            static int64_t last_audit_ns;
            struct timespec am;
            if (!clock_gettime(CLOCK_MONOTONIC, &am)) {
               const int64_t now_ns = (int64_t)am.tv_sec * 1000000000 + am.tv_nsec;
               if (now_ns - last_audit_ns > 60000000000ll) {
                  last_audit_ns = now_ns;
                  struct rlimit rl;
                  DIR *d;
                  if (!getrlimit(RLIMIT_NOFILE, &rl) && (d = opendir("/dev/fd"))) {
                     long nfds = 0;
                     while (readdir(d))
                        nfds++;
                     closedir(d);
                     if (rl.rlim_cur && nfds > (long)(rl.rlim_cur / 2))
                        vkr_log_error("fd audit: %ld open of %llu soft limit — a "
                                      "carrier-fd ratchet is under way; relaunch "
                                      "with VKR_FD_TRACE=1 to attribute it",
                                      nfds, (unsigned long long)rl.rlim_cur);
                  }
               }
            }
         }
#endif

         ret = clock_gettime(CLOCK_REALTIME, &abs_ts);
         if (ret)
            break;

         const struct timespec rel_ts = {
            .tv_sec = stamp_period_us / 1000000,
            .tv_nsec = (stamp_period_us % 1000000) * 1000,
         };
         abs_ts = timespec_add(abs_ts, rel_ts);
      } else if (ret)
         break;

      /* abs_ts is immutable for spurious wakeups */
      mtx_lock(&ctx->ring_monitor.mutex);
      ret = cnd_timedwait(&ctx->ring_monitor.cond, &ctx->ring_monitor.mutex, &abs_ts);
      mtx_unlock(&ctx->ring_monitor.mutex);
   }

   /* Exit-path confession: if this monitor is being torn down while its last
    * stamp is already stale, say so — teardown after a guest abort is exactly
    * when the in-loop check can no longer run, and a silent exit here is what
    * made the 2026-07-11 class-2 incident unattributable host-side. */
   struct timespec mono;
   if (last_stamp_ns && !clock_gettime(CLOCK_MONOTONIC, &mono)) {
      const int64_t now_ns = (int64_t)mono.tv_sec * 1000000000 + mono.tv_nsec;
      const int64_t period_ns = (int64_t)ctx->ring_monitor.report_period_us * 1000;
      if (now_ns - last_stamp_ns > period_ns)
         vkr_log_error("ringmon-%d: exiting with a stale ALIVE stamp: %" PRId64
                       " ms since last (guest tolerates ~%u ms) — if the guest "
                       "aborted on expired ring alive status, this was why",
                       ctx->ctx_id, (now_ns - last_stamp_ns) / 1000000,
                       ctx->ring_monitor.report_period_us / 1000);
   }

   return ret;
}

bool
vkr_context_ring_monitor_init(struct vkr_context *ctx, uint32_t report_period_us)
{
   int ret;
   assert(report_period_us > 0);
   assert(!ctx->ring_monitor.started);

   if (mtx_init(&ctx->ring_monitor.mutex, mtx_plain) != thrd_success)
      goto err_mtx_init;
   if (cnd_init(&ctx->ring_monitor.cond) != thrd_success)
      goto err_cnd_init;

   ctx->ring_monitor.report_period_us = report_period_us;
   ctx->ring_monitor.started = true;
   ret = thrd_create(&ctx->ring_monitor.thread, vkr_context_ring_monitor_thread, ctx);
   if (ret != thrd_success)
      goto err_monitor_thrd_create;

   return true;

err_monitor_thrd_create:
   cnd_destroy(&ctx->ring_monitor.cond);
err_cnd_init:
   mtx_destroy(&ctx->ring_monitor.mutex);
err_mtx_init:
   return false;
}

static void
vkr_context_ring_monitor_fini(struct vkr_context *ctx)
{
   mtx_lock(&ctx->ring_monitor.mutex);
   assert(ctx->ring_monitor.started);
   ctx->ring_monitor.started = false;
   cnd_signal(&ctx->ring_monitor.cond);
   mtx_unlock(&ctx->ring_monitor.mutex);

   thrd_join(ctx->ring_monitor.thread, NULL);

   cnd_destroy(&ctx->ring_monitor.cond);
   mtx_destroy(&ctx->ring_monitor.mutex);
}

void
vkr_context_destroy(struct vkr_context *ctx)
{
   /* TODO Move the entire teardown process to a separate thread so that the main thread
    * cannot get blocked by the vkDeviceWaitIdle upon device destruction.
    */
   list_for_each_entry_safe (struct vkr_ring, ring, &ctx->rings, head) {
      vkr_ring_stop(ring);
      vkr_ring_destroy(ring);
   }
   mtx_destroy(&ctx->ring_mutex);

   vkr_context_wait_ring_fini(ctx);

   if (ctx->ring_monitor.started)
      vkr_context_ring_monitor_fini(ctx);

   /* limina journal: detach before the teardown sweeps below so their object
    * removals skip pointless pruning (ring threads are already joined) */
   struct vkr_journal *journal = ctx->journal;
   ctx->journal = NULL;

   if (ctx->instance) {
      vkr_log("destroying context %d (%s) with a valid instance", ctx->ctx_id,
              vkr_context_get_name(ctx));

      vkr_instance_destroy(ctx, ctx->instance, false);
   }

   vkr_log("destroying context %u (%s): instance was %s, %u objects and %u resources "
           "left in the tables",
           ctx->ctx_id, vkr_context_get_name(ctx), ctx->instance ? "live" : "gone",
           _mesa_hash_table_num_entries(ctx->object_table),
           _mesa_hash_table_num_entries(ctx->resource_table));
   /* Leak canary: with no live instance to sweep, anything still in the object table
    * gets a bare free() below — its host-side allocations (mtl_shm carrier fd,
    * IOSurface refs, gbm bo, udmabuf fd) leak. The worker is a long-lived singleton,
    * so per-session leaks ratchet until EMFILE kills blob creation for every future
    * context (2026-07-10 dogfood: 12k PSXSHM fds, fresh niri starved at login). */
   hash_table_foreach (ctx->object_table, entry) {
      const struct vkr_object *obj = entry->data;
      if (obj->type == VK_OBJECT_TYPE_DEVICE_MEMORY &&
          ((const struct vkr_device_memory *)obj)->mtl_shm) {
         vkr_log_error("context %u: DEVICE_MEMORY id %" PRIu64 " still holds an mtl_shm "
                       "carrier at context destroy — its shm fd is about to leak",
                       ctx->ctx_id, (uint64_t)obj->id);
      }
   }

   _mesa_hash_table_destroy(ctx->resource_table, vkr_context_free_resource);
   mtx_destroy(&ctx->resource_mutex);

   _mesa_hash_table_destroy(ctx->object_table, vkr_context_free_object);
   mtx_destroy(&ctx->object_mutex);

   vkr_journal_destroy(journal);

   vkr_cs_encoder_fini(&ctx->encoder);
   vkr_cs_decoder_fini(&ctx->decoder);

   vkr_library_unload(&ctx->vulkan_library);

   free(ctx->debug_name);
   free(ctx);
}

static uint32_t
vkr_hash_u64(const void *key)
{
   return XXH32(key, sizeof(uint64_t), 0);
}

static bool
vkr_key_u64_equal(const void *key1, const void *key2)
{
   return *(const uint64_t *)key1 == *(const uint64_t *)key2;
}

void
vkr_context_free_object(struct hash_entry *entry)
{
   struct vkr_object *obj = entry->data;
   free(obj);
}

struct vkr_context *
vkr_context_create(uint32_t ctx_id,
                   vkr_renderer_retire_fence_callback_type cb,
                   size_t debug_len,
                   const char *debug_name)
{
   struct vkr_context *ctx = calloc(1, sizeof(*ctx));
   if (!ctx)
      return NULL;

   bool ret = vkr_library_load(&ctx->vulkan_library);
   if (!ret) {
      free(ctx);
      return NULL;
   }

   ctx->ctx_id = ctx_id;
   ctx->retire_fence = cb;
   ctx->debug_name = malloc(debug_len + 1);
   if (!ctx->debug_name)
      goto err_debug_name;

   memcpy(ctx->debug_name, debug_name, debug_len);
   ctx->debug_name[debug_len] = '\0';

   ctx->validate_level = VKR_CONTEXT_VALIDATE_NONE;
   ctx->validate_fatal = false;
   if (VKR_DEBUG(VALIDATE))
      ctx->validate_level = VKR_CONTEXT_VALIDATE_FULL;

#ifdef ENABLE_RENDER_SERVER_WORKER_THREAD
   ctx->on_worker_thread = true;
#else
   ctx->on_worker_thread = false;
#endif

   if (!vkr_context_wait_ring_init(ctx))
      goto err_ctx_wait_ring_init;

   if (mtx_init(&ctx->object_mutex, mtx_plain) != thrd_success)
      goto err_ctx_object_mutex;

   ctx->object_table = _mesa_hash_table_create(NULL, vkr_hash_u64, vkr_key_u64_equal);
   if (!ctx->object_table)
      goto err_ctx_object_table;

   /* limina: snapshot-replay journal; best-effort (NULL disables recording) */
   if (vkr_journal_enabled())
      ctx->journal = vkr_journal_create(ctx_id);

   if (mtx_init(&ctx->resource_mutex, mtx_plain) != thrd_success)
      goto err_ctx_resource_mutex;

   ctx->resource_table =
      _mesa_hash_table_create(NULL, _mesa_hash_u32, _mesa_key_u32_equal);
   if (!ctx->resource_table)
      goto err_ctx_resource_table;

   if (vkr_cs_decoder_init(&ctx->decoder, ctx))
      goto err_cs_decoder_init;

   if (vkr_cs_encoder_init(&ctx->encoder, &ctx->cs_fatal_error))
      goto err_cs_encoder_init;

   vkr_context_init_dispatch(ctx);
   vkr_context_init_proc_table(ctx);

   if (mtx_init(&ctx->ring_mutex, mtx_plain) != thrd_success)
      goto err_ctx_ring_mutex;

   list_inithead(&ctx->rings);

   return ctx;

err_ctx_ring_mutex:
   vkr_cs_encoder_fini(&ctx->encoder);
err_cs_encoder_init:
   vkr_cs_decoder_fini(&ctx->decoder);
err_cs_decoder_init:
   _mesa_hash_table_destroy(ctx->resource_table, vkr_context_free_resource);
err_ctx_resource_table:
   mtx_destroy(&ctx->resource_mutex);
err_ctx_resource_mutex:
   vkr_journal_destroy(ctx->journal);
   _mesa_hash_table_destroy(ctx->object_table, vkr_context_free_object);
err_ctx_object_table:
   mtx_destroy(&ctx->object_mutex);
err_ctx_object_mutex:
   vkr_context_wait_ring_fini(ctx);
err_ctx_wait_ring_init:
   free(ctx->debug_name);
err_debug_name:
   free(ctx);
   return NULL;
}
