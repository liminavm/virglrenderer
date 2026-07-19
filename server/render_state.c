/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "render_state.h"

#include <inttypes.h>

#ifdef ENABLE_RENDER_SERVER_WORKER_THREAD
#include "c11/threads.h"
#endif

#include "render_context.h"
#include "vkr_renderer.h"

/* Workers call into vkr renderer.  When they are processes, not much care is
 * required. But when workers are threads, we need to grab a lock to protect
 * vkr renderer.
 */
struct render_state {
#ifdef ENABLE_RENDER_SERVER_WORKER_THREAD
   /* protect renderer interface */
   mtx_t renderer_mutex;
   /* protect the below global states */
   mtx_t state_mutex;
#endif

   /* track and init/fini just once */
   int init_count;

   /* track the render_context */
   struct list_head contexts;
};

struct render_state state = {
#ifdef ENABLE_RENDER_SERVER_WORKER_THREAD
   .renderer_mutex = _MTX_INITIALIZER_NP,
   .state_mutex = _MTX_INITIALIZER_NP,
#endif
   .init_count = 0,
};

#ifdef ENABLE_RENDER_SERVER_WORKER_THREAD
static inline mtx_t *
render_state_lock(mtx_t *mtx)
{
   mtx_lock(mtx);
   return mtx;
}

static void
render_state_unlock(mtx_t **mtx)
{
   mtx_unlock(*mtx);
}

#define SCOPE_LOCK_STATE()                                                               \
   mtx_t *_state_mtx __attribute__((cleanup(render_state_unlock), unused)) =             \
      render_state_lock(&state.state_mutex)

#define SCOPE_LOCK_RENDERER()                                                            \
   mtx_t *_renderer_mtx __attribute__((cleanup(render_state_unlock), unused)) =          \
      render_state_lock(&state.renderer_mutex)

#else

#define SCOPE_LOCK_STATE()
#define SCOPE_LOCK_RENDERER()

#endif /* ENABLE_RENDER_SERVER_WORKER_THREAD */

/* limina M9.3 diagnostics: dump the vkr context table under the same lock
 * discipline every other vkr entry point uses, so it is safe to call from any
 * thread (in the same-process model, the VMM's gpu worker thread calls this
 * through virgl_renderer_limina_dump_state()). */
void
render_state_limina_dump_state(void)
{
   SCOPE_LOCK_RENDERER();
   if (!state.init_count) {
      render_log("[GPUTRACE] vkr state: renderer not initialized");
      return;
   }
   vkr_renderer_dump_state();
}

/* limina snapshot-replay (limina M9.3 P1): journal export + replay entry points,
 * under the same renderer lock the normal submit path takes (render_state.c's
 * submit is SCOPE_LOCK_RENDERER + vkr_renderer_submit_cmd). */

bool
render_state_limina_journal_export(uint32_t ctx_id, void **out_buf, size_t *out_size)
{
   SCOPE_LOCK_RENDERER();
   if (!state.init_count)
      return false;
   return vkr_renderer_journal_export(ctx_id, out_buf, out_size);
}

uint64_t
render_state_limina_journal_seq(uint32_t ctx_id)
{
   SCOPE_LOCK_RENDERER();
   if (!state.init_count)
      return 0;
   return vkr_renderer_journal_seq(ctx_id);
}

void
render_state_limina_journal_unpin(uint32_t ctx_id, uint64_t key)
{
   SCOPE_LOCK_RENDERER();
   if (!state.init_count)
      return;
   vkr_renderer_journal_unpin(ctx_id, key);
}

bool
render_state_limina_replay_begin(uint32_t ctx_id)
{
   SCOPE_LOCK_RENDERER();
   if (!state.init_count)
      return false;
   return vkr_renderer_replay_begin(ctx_id);
}

bool
render_state_limina_replay_submit(uint32_t ctx_id, void *cmd, uint32_t size)
{
   SCOPE_LOCK_RENDERER();
   if (!state.init_count)
      return false;
   return vkr_renderer_replay_submit(ctx_id, cmd, size);
}

bool
render_state_limina_replay_ring_cmd(uint32_t ctx_id,
                                    uint64_t ring_id,
                                    void *cmd,
                                    uint32_t size)
{
   SCOPE_LOCK_RENDERER();
   if (!state.init_count)
      return false;
   return vkr_renderer_replay_ring_cmd(ctx_id, ring_id, cmd, size);
}

bool
render_state_limina_replay_end(uint32_t ctx_id)
{
   SCOPE_LOCK_RENDERER();
   if (!state.init_count)
      return false;
   return vkr_renderer_replay_end(ctx_id);
}

static struct render_context *
render_state_lookup_context(uint32_t ctx_id)
{
   struct render_context *ctx = NULL;

   SCOPE_LOCK_STATE();
#ifdef ENABLE_RENDER_SERVER_WORKER_THREAD
   list_for_each_entry (struct render_context, iter, &state.contexts, head) {
      if (iter->ctx_id == ctx_id) {
         ctx = iter;
         break;
      }
   }
#else
   assert(list_is_singular(&state.contexts));
   ctx = list_first_entry(&state.contexts, struct render_context, head);
   assert(ctx->ctx_id == ctx_id);
   (void)ctx_id;
#endif

   return ctx;
}

#ifndef ENABLE_SAME_PROCESS_RENDER_SERVER
static void
render_state_cb_debug_logger(UNUSED enum virgl_log_level_flags log_level,
                             const char *message,
                             UNUSED void* user_data)
{
   render_log(message);
}
#endif

static void
render_state_cb_retire_fence(uint32_t ctx_id, uint32_t ring_idx, uint64_t fence_id)
{
   struct render_context *ctx = render_state_lookup_context(ctx_id);
   assert(ctx);

   const uint32_t seqno = (uint32_t)fence_id;
   render_context_update_timeline(ctx, ring_idx, seqno);
}

static const struct vkr_renderer_callbacks render_state_cbs = {
#ifndef ENABLE_SAME_PROCESS_RENDER_SERVER
   .debug_logger = render_state_cb_debug_logger,
#endif
   .retire_fence = render_state_cb_retire_fence,
};

static void
render_state_add_context(struct render_context *ctx)
{
   SCOPE_LOCK_STATE();
   list_addtail(&ctx->head, &state.contexts);
}

static void
render_state_remove_context(struct render_context *ctx)
{
   SCOPE_LOCK_STATE();
   list_del(&ctx->head);
}

void
render_state_fini(void)
{
   SCOPE_LOCK_STATE();
   if (state.init_count) {
      state.init_count--;
      if (!state.init_count)
         vkr_renderer_fini();
   }
}

bool
render_state_init(uint32_t init_flags)
{
   static const uint32_t required_flags = VIRGL_RENDERER_VENUS | VIRGL_RENDERER_NO_VIRGL;
   if ((init_flags & required_flags) != required_flags)
      return false;

   SCOPE_LOCK_STATE();
   if (!state.init_count) {
      /* always use sync thread and async fence cb for low latency */
      static const uint32_t vkr_flags =
         VKR_RENDERER_THREAD_SYNC | VKR_RENDERER_ASYNC_FENCE_CB;
      if (!vkr_renderer_init(vkr_flags, &render_state_cbs))
         return false;

      list_inithead(&state.contexts);
   }

   state.init_count++;

   return true;
}

bool
render_state_create_context(struct render_context *ctx,
                            uint32_t flags,
                            uint32_t name_len,
                            const char *name)
{
   {
      SCOPE_LOCK_RENDERER();
      if (!vkr_renderer_create_context(ctx->ctx_id, flags, name_len, name))
         return false;
   }

   render_state_add_context(ctx);

   return true;
}

void
render_state_destroy_context(uint32_t ctx_id)
{
   struct render_context *ctx = render_state_lookup_context(ctx_id);
   if (!ctx)
      return;

   {
      SCOPE_LOCK_RENDERER();
      vkr_renderer_destroy_context(ctx_id);
   }

   render_state_remove_context(ctx);
}

bool
render_state_submit_cmd(uint32_t ctx_id, void *cmd, uint32_t size)
{
   SCOPE_LOCK_RENDERER();
   return vkr_renderer_submit_cmd(ctx_id, cmd, size);
}

bool
render_state_submit_fence(uint32_t ctx_id,
                          uint32_t flags,
                          uint64_t ring_idx,
                          uint64_t fence_id)
{
   SCOPE_LOCK_RENDERER();
   return vkr_renderer_submit_fence(ctx_id, flags, ring_idx, fence_id);
}

bool
render_state_create_resource(uint32_t ctx_id,
                             uint32_t res_id,
                             uint64_t blob_id,
                             uint64_t blob_size,
                             uint32_t blob_flags,
                             enum virgl_resource_fd_type *out_fd_type,
                             int *out_res_fd,
                             uint32_t *out_map_info,
                             struct virgl_resource_vulkan_info *out_vulkan_info,
                             uint32_t *out_iosurface_id,
                             uint64_t *out_map_ptr)
{
   SCOPE_LOCK_RENDERER();
   return vkr_renderer_create_resource(ctx_id, res_id, blob_id, blob_size, blob_flags,
                                       out_fd_type, out_res_fd, out_map_info,
                                       out_vulkan_info, out_iosurface_id, out_map_ptr);
}

bool
render_state_import_resource(uint32_t ctx_id,
                             uint32_t res_id,
                             enum virgl_resource_fd_type fd_type,
                             int fd,
                             uint64_t size,
                             uint32_t iosurface_id,
                             uint64_t map_ptr)
{
   SCOPE_LOCK_RENDERER();
   return vkr_renderer_import_resource(ctx_id, res_id, fd_type, fd, size, iosurface_id,
                                       map_ptr);
}

void
render_state_destroy_resource(uint32_t ctx_id, uint32_t res_id)
{
   SCOPE_LOCK_RENDERER();
   vkr_renderer_destroy_resource(ctx_id, res_id);
}
