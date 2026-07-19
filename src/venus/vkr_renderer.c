/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_common.h"

#include <stdio.h>

#include "venus-protocol/vn_protocol_renderer_info.h"
#include "virtgpu_drm.h"
#include "venus_hw.h"

#include "vkr_context.h"
#include "vkr_ring.h"

struct vkr_renderer_state {
   const struct vkr_renderer_callbacks *cbs;

   /* track the vkr_context */
   struct list_head contexts;
};

struct vkr_renderer_state vkr_state;

size_t
vkr_get_capset(void *capset, uint32_t flags)
{
   struct virgl_renderer_capset_venus *c = capset;
   if (c) {
      memset(c, 0, sizeof(*c));
      c->wire_format_version = vn_info_wire_format_version();
      c->vk_xml_version = vn_info_vk_xml_version();
      c->vk_ext_command_serialization_spec_version =
         vkr_extension_get_spec_version("VK_EXT_command_serialization");
      c->vk_mesa_venus_protocol_spec_version =
         vkr_extension_get_spec_version("VK_MESA_venus_protocol");
      /* After https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/688,
       * this flag is used to indicate render server config.
       */
      c->supports_blob_id_0 = true;

      uint32_t ext_mask[VN_INFO_EXTENSION_MAX_NUMBER / 32 + 1] = { 0 };
      vn_info_extension_mask_init(ext_mask);

      static_assert(sizeof(ext_mask) <= sizeof(c->vk_extension_mask1),
                    "Time to extend venus capset with vk_extension_mask2");
      memcpy(c->vk_extension_mask1, ext_mask, sizeof(ext_mask));

      /* set bit 0 to enable the extension mask(s) */
      assert(!(c->vk_extension_mask1[0] & 0x1u));
      c->vk_extension_mask1[0] |= 0x1u;

      c->allow_vk_wait_syncs = 1;
      c->supports_multiple_timelines = 1;

      c->use_guest_vram = (bool)(flags & VIRGL_RENDERER_USE_GUEST_VRAM);
   }

   return sizeof(*c);
}

bool
vkr_renderer_init(uint32_t flags, const struct vkr_renderer_callbacks *cbs)
{
   TRACE_INIT();
   TRACE_FUNC();

   static const uint32_t required_flags =
      VKR_RENDERER_THREAD_SYNC | VKR_RENDERER_ASYNC_FENCE_CB;
   if ((flags & required_flags) != required_flags)
      return false;

   vkr_debug_init();

   if (cbs->debug_logger)
      virgl_log_set_handler(cbs->debug_logger, NULL, NULL);

   vkr_state.cbs = cbs;
   list_inithead(&vkr_state.contexts);

   return true;
}

void
vkr_renderer_fini(void)
{
   list_for_each_entry_safe (struct vkr_context, ctx, &vkr_state.contexts, head)
      vkr_context_destroy(ctx);

   list_inithead(&vkr_state.contexts);

   vkr_state.cbs = NULL;
}

static struct vkr_context *
vkr_renderer_lookup_context(uint32_t ctx_id)
{
   list_for_each_entry (struct vkr_context, ctx, &vkr_state.contexts, head) {
      if (ctx->ctx_id == ctx_id)
         return ctx;
   }

   return NULL;
}

bool
vkr_renderer_create_context(uint32_t ctx_id,
                            uint32_t ctx_flags,
                            uint32_t nlen,
                            const char *name)
{
   TRACE_FUNC();

   assert(ctx_id);
   assert(!(ctx_flags & ~VIRGL_RENDERER_CONTEXT_FLAG_CAPSET_ID_MASK));

   if ((ctx_flags & VIRGL_RENDERER_CONTEXT_FLAG_CAPSET_ID_MASK) !=
       VIRTGPU_DRM_CAPSET_VENUS)
      return false;

   /* duplicate ctx creation between server and vkr is invalid */
   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (ctx)
      return false;

   ctx = vkr_context_create(ctx_id, vkr_state.cbs->retire_fence, nlen, name);
   if (!ctx)
      return false;

   list_addtail(&ctx->head, &vkr_state.contexts);

   return true;
}

void
vkr_renderer_destroy_context(uint32_t ctx_id)
{
   TRACE_FUNC();

   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx)
      return;

   list_del(&ctx->head);
   vkr_context_destroy(ctx);
}

/* limina M9.3 diagnostics: short name for the object types a venus context holds,
 * so the per-context dump doubles as a retain-and-replay sizing measurement.
 * Non-static: the snapshot-replay journal census (vkr_journal.c) shares it. */
const char *
vkr_object_type_name(VkObjectType type)
{
   switch (type) {
   case VK_OBJECT_TYPE_INSTANCE:
      return "instance";
   case VK_OBJECT_TYPE_PHYSICAL_DEVICE:
      return "phys_dev";
   case VK_OBJECT_TYPE_DEVICE:
      return "device";
   case VK_OBJECT_TYPE_QUEUE:
      return "queue";
   case VK_OBJECT_TYPE_SEMAPHORE:
      return "semaphore";
   case VK_OBJECT_TYPE_COMMAND_BUFFER:
      return "cmd_buf";
   case VK_OBJECT_TYPE_FENCE:
      return "fence";
   case VK_OBJECT_TYPE_DEVICE_MEMORY:
      return "memory";
   case VK_OBJECT_TYPE_BUFFER:
      return "buffer";
   case VK_OBJECT_TYPE_IMAGE:
      return "image";
   case VK_OBJECT_TYPE_EVENT:
      return "event";
   case VK_OBJECT_TYPE_QUERY_POOL:
      return "query_pool";
   case VK_OBJECT_TYPE_BUFFER_VIEW:
      return "buffer_view";
   case VK_OBJECT_TYPE_IMAGE_VIEW:
      return "image_view";
   case VK_OBJECT_TYPE_SHADER_MODULE:
      return "shader";
   case VK_OBJECT_TYPE_PIPELINE_CACHE:
      return "pipe_cache";
   case VK_OBJECT_TYPE_PIPELINE_LAYOUT:
      return "pipe_layout";
   case VK_OBJECT_TYPE_RENDER_PASS:
      return "render_pass";
   case VK_OBJECT_TYPE_PIPELINE:
      return "pipeline";
   case VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT:
      return "dset_layout";
   case VK_OBJECT_TYPE_SAMPLER:
      return "sampler";
   case VK_OBJECT_TYPE_DESCRIPTOR_POOL:
      return "dpool";
   case VK_OBJECT_TYPE_DESCRIPTOR_SET:
      return "dset";
   case VK_OBJECT_TYPE_FRAMEBUFFER:
      return "framebuffer";
   case VK_OBJECT_TYPE_COMMAND_POOL:
      return "cmd_pool";
   case VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION:
      return "ycbcr_conv";
   case VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE:
      return "dupdate_tmpl";
   case VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR:
      return "accel_struct";
   default:
      return NULL;
   }
}

void
vkr_renderer_dump_state(void)
{
   /* Caller must hold the render_state renderer lock (SCOPE_LOCK_RENDERER), which
    * serializes us against every other vkr entry point; the per-context mutexes
    * below additionally cover tables their own worker threads mutate. */
   unsigned n_ctx = list_length(&vkr_state.contexts);
   vkr_log("[GPUTRACE] vkr state: %u context(s)", n_ctx);

   list_for_each_entry (struct vkr_context, ctx, &vkr_state.contexts, head) {
      mtx_lock(&ctx->ring_mutex);
      unsigned rings = list_length(&ctx->rings);
      mtx_unlock(&ctx->ring_mutex);

      mtx_lock(&ctx->resource_mutex);
      uint32_t resources = _mesa_hash_table_num_entries(ctx->resource_table);
      mtx_unlock(&ctx->resource_mutex);

      unsigned queues = 0;
      for (unsigned i = 0; i < ARRAY_SIZE(ctx->sync_queues); i++) {
         if (ctx->sync_queues[i])
            queues++;
      }

      /* Tally the object table by type: this is what a seamless restore would have
       * to re-create, so the breakdown is the replay's bill of materials. */
      struct {
         VkObjectType type;
         uint32_t count;
      } tally[48];
      unsigned n_tally = 0;
      uint32_t objects = 0;

      mtx_lock(&ctx->object_mutex);
      objects = _mesa_hash_table_num_entries(ctx->object_table);
      hash_table_foreach (ctx->object_table, entry) {
         const struct vkr_object *obj = entry->data;
         unsigned i;
         for (i = 0; i < n_tally; i++) {
            if (tally[i].type == obj->type)
               break;
         }
         if (i == n_tally && n_tally < ARRAY_SIZE(tally)) {
            tally[n_tally].type = obj->type;
            tally[n_tally].count = 0;
            n_tally++;
         }
         if (i < ARRAY_SIZE(tally))
            tally[i].count++;
      }
      mtx_unlock(&ctx->object_mutex);

      vkr_log("[GPUTRACE]   ctx %u (%s): rings=%u objects=%" PRIu32
              " resources=%" PRIu32 " sync_queues=%u",
              ctx->ctx_id, ctx->debug_name ? ctx->debug_name : "?", rings, objects,
              resources, queues);

      if (n_tally) {
         char buf[512];
         int len = 0;
         for (unsigned i = 0; i < n_tally && len < (int)sizeof(buf) - 32; i++) {
            const char *name = vkr_object_type_name(tally[i].type);
            if (name) {
               len += snprintf(buf + len, sizeof(buf) - len, "%s%s=%" PRIu32,
                               i ? " " : "", name, tally[i].count);
            } else {
               len += snprintf(buf + len, sizeof(buf) - len, "%stype%u=%" PRIu32,
                               i ? " " : "", tally[i].type, tally[i].count);
            }
         }
         vkr_log("[GPUTRACE]     objects: %s", buf);
      }

      /* limina journal census — cross-check its live-create tally against the
       * object-table tally above (they must agree on a healthy context) */
      vkr_journal_dump(ctx->journal);
   }
}

bool
vkr_renderer_submit_cmd(uint32_t ctx_id, void *cmd, uint32_t size)
{
   TRACE_FUNC();

   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx)
      return false;

   return vkr_context_submit_cmd(ctx, cmd, size);
}

/* --- limina snapshot-replay (limina M9.3 P1) --------------------------------
 *
 * Export a context's re-creation journal for the snapshot file, and replay a
 * saved journal into a freshly-created context at restore. Replay strips each
 * command's VK_COMMAND_GENERATE_REPLY_BIT (the guest consumed the original
 * replies pre-suspend; re-emitting them would corrupt the restored reply
 * stream), routes ring-scoped stream state onto the (not-yet-started) target
 * ring's decoder, and starts every deferred ring at replay_end. */

bool
vkr_renderer_journal_export(uint32_t ctx_id, void **out_buf, size_t *out_size)
{
   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx || !ctx->journal)
      return false;
   return vkr_journal_export(ctx->journal, out_buf, out_size);
}

uint64_t
vkr_renderer_journal_seq(uint32_t ctx_id)
{
   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx || !ctx->journal)
      return 0;
   return vkr_journal_seq(ctx->journal);
}

void
vkr_renderer_journal_unpin(uint32_t ctx_id, uint64_t key)
{
   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx)
      return;
   vkr_journal_unpin_key(ctx, key);
}

bool
vkr_renderer_replay_begin(uint32_t ctx_id)
{
   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx)
      return false;
   ctx->replaying = true;
   return true;
}

/* clear VK_COMMAND_GENERATE_REPLY_BIT_EXT in the (single) command's flags word
 * at wire offset 4; the buffer is caller-owned and mutable by contract */
static bool
vkr_replay_strip_reply(void *cmd, uint32_t size)
{
   if (size < 8)
      return false;
   uint32_t flags;
   memcpy(&flags, (uint8_t *)cmd + 4, sizeof(flags));
   flags &= ~(uint32_t)VK_COMMAND_GENERATE_REPLY_BIT_EXT;
   memcpy((uint8_t *)cmd + 4, &flags, sizeof(flags));
   return true;
}

/* A replayed entry can legitimately fail: a retained state-mutating command (e.g.
 * vkUpdateDescriptorSets) may reference an object destroyed before the snapshot —
 * its create was pruned, so the lookup misses and trips the context FATAL. The
 * pre-suspend dset slot held a dangling reference (garbage-if-accessed); dropping
 * the write leaves the slot unwritten (also garbage-if-accessed) — semantically
 * equivalent. FATAL is sticky by design for live traffic, but during replay it
 * must not cascade to every later entry: clear it and let the caller count. */
static bool
vkr_replay_recover_fatal(struct vkr_context *ctx)
{
   if (!ctx->cs_fatal_error)
      return false;
   vkr_log("replay: entry failed (stale reference?); clearing FATAL, continuing");
   ctx->cs_fatal_error = false;
   return true;
}

bool
vkr_renderer_replay_submit(uint32_t ctx_id, void *cmd, uint32_t size)
{
   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx || !vkr_replay_strip_reply(cmd, size))
      return false;
   bool ok = vkr_context_submit_cmd(ctx, cmd, size);
   if (!ok && ctx->replaying)
      vkr_replay_recover_fatal(ctx);
   return ok;
}

bool
vkr_renderer_replay_ring_cmd(uint32_t ctx_id, uint64_t ring_id, void *cmd, uint32_t size)
{
   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx || !vkr_replay_strip_reply(cmd, size))
      return false;

   struct vkr_ring *target = NULL;
   mtx_lock(&ctx->ring_mutex);
   list_for_each_entry (struct vkr_ring, ring, &ctx->rings, head) {
      if (ring->id == ring_id) {
         target = ring;
         break;
      }
   }
   mtx_unlock(&ctx->ring_mutex);
   if (!target) {
      vkr_log("replay_ring_cmd: no ring %" PRIu64 " in ctx %u", ring_id, ctx_id);
      return false;
   }
   bool ok = vkr_ring_replay_cmd(target, cmd, size);
   if (!ok && ctx->replaying)
      vkr_replay_recover_fatal(ctx);
   return ok;
}

bool
vkr_renderer_replay_end(uint32_t ctx_id)
{
   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx)
      return false;

   mtx_lock(&ctx->ring_mutex);
   list_for_each_entry (struct vkr_ring, ring, &ctx->rings, head) {
      if (!atomic_load(&ring->started))
         vkr_ring_start(ring);
   }
   mtx_unlock(&ctx->ring_mutex);

   ctx->replaying = false;
   return true;
}

bool
vkr_renderer_submit_fence(uint32_t ctx_id,
                          uint32_t flags,
                          uint64_t ring_idx,
                          uint64_t fence_id)
{
   TRACE_FUNC();

   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx)
      return false;

   assert(vkr_state.cbs->retire_fence);
   return vkr_context_submit_fence(ctx, flags, ring_idx, fence_id);
}

bool
vkr_renderer_create_resource(uint32_t ctx_id,
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
   TRACE_FUNC();

   assert(res_id);
   assert(blob_size);

   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx)
      return false;

   struct virgl_context_blob blob;
   if (!vkr_context_create_resource(ctx, res_id, blob_id, blob_size, blob_flags, &blob))
      return false;

   /* limina tier-2 (macOS) #28: a HOST_VISIBLE blob is shared by pointer (OPAQUE_HANDLE +
    * map_ptr, no fd) — see vkr_device_memory_export_blob. Carry map_ptr across the boundary
    * (same-process thread server) and skip the fd; the caller sends an fd-less reply. */
   *out_map_ptr = blob.map_ptr;
   if (blob.map_ptr) {
      *out_fd_type = VIRGL_RESOURCE_FD_INVALID;
      *out_res_fd = -1;
      *out_map_info = blob.map_info;
      *out_iosurface_id = blob.iosurface_id;
      return true;
   }

   assert(blob.type == VIRGL_RESOURCE_FD_SHM || blob.type == VIRGL_RESOURCE_FD_DMABUF ||
          blob.type == VIRGL_RESOURCE_FD_OPAQUE);

   *out_fd_type = blob.type;
   *out_res_fd = blob.u.fd;
   *out_map_info = blob.map_info;
   /* limina tier-2 (macOS): carry the scanout IOSurface id across the render-server boundary
    * so the proxy can rebuild it into virgl_context_blob for zero-copy SET_SCANOUT_BLOB. */
   *out_iosurface_id = blob.iosurface_id;

   if (blob.type == VIRGL_RESOURCE_FD_OPAQUE) {
      assert(out_vulkan_info);
      *out_vulkan_info = blob.vulkan_info;
   }

   return true;
}

bool
vkr_renderer_import_resource(uint32_t ctx_id,
                             uint32_t res_id,
                             enum virgl_resource_fd_type fd_type,
                             int fd,
                             uint64_t size,
                             uint32_t iosurface_id,
                             uint64_t map_ptr)
{
   TRACE_FUNC();

   assert(res_id);
   /* limina tier-2 (macOS): SHM (carrier or plain) and fd-less map_ptr blobs are also
    * importable cross-context — the pixel bytes are reached via iosurface_id/map_ptr. */
   assert(fd_type == VIRGL_RESOURCE_FD_DMABUF || fd_type == VIRGL_RESOURCE_FD_OPAQUE ||
          fd_type == VIRGL_RESOURCE_FD_SHM ||
          (fd_type == VIRGL_RESOURCE_FD_INVALID && map_ptr));
   assert(fd >= 0 || map_ptr);
   assert(size);

   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (!ctx)
      return false;

   return vkr_context_import_resource(ctx, res_id, fd_type, fd, size, iosurface_id,
                                      map_ptr);
}

void
vkr_renderer_destroy_resource(uint32_t ctx_id, uint32_t res_id)
{
   TRACE_FUNC();

   struct vkr_context *ctx = vkr_renderer_lookup_context(ctx_id);
   if (ctx)
      vkr_context_destroy_resource(ctx, res_id);
}
