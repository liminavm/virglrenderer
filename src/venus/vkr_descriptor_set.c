/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_descriptor_set.h"

#include "vkr_buffer.h"
#include "vkr_descriptor_set_gen.h"
#include "vkr_image.h"

static void
vkr_dispatch_vkGetDescriptorSetLayoutSupport(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetDescriptorSetLayoutSupport *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetDescriptorSetLayoutSupport_args_handle(args);
   vk->GetDescriptorSetLayoutSupport(args->device, args->pCreateInfo, args->pSupport);
}

static void
vkr_dispatch_vkCreateDescriptorSetLayout(
   struct vn_dispatch_context *dispatch,
   struct vn_command_vkCreateDescriptorSetLayout *args)
{
   vkr_descriptor_set_layout_create_and_add(dispatch->data, args);
}

static void
vkr_dispatch_vkDestroyDescriptorSetLayout(
   struct vn_dispatch_context *dispatch,
   struct vn_command_vkDestroyDescriptorSetLayout *args)
{
   vkr_descriptor_set_layout_destroy_and_remove(dispatch->data, args);
}

static void
vkr_dispatch_vkCreateDescriptorPool(struct vn_dispatch_context *dispatch,
                                    struct vn_command_vkCreateDescriptorPool *args)
{
   struct vkr_descriptor_pool *pool =
      vkr_descriptor_pool_create_and_add(dispatch->data, args);
   if (!pool)
      return;

   pool->flags = args->pCreateInfo->flags;

   list_inithead(&pool->descriptor_sets);
}

static void
vkr_dispatch_vkDestroyDescriptorPool(struct vn_dispatch_context *dispatch,
                                     struct vn_command_vkDestroyDescriptorPool *args)
{
   struct vkr_context *ctx = dispatch->data;
   struct vkr_descriptor_pool *pool =
      vkr_descriptor_pool_from_handle(args->descriptorPool);

   if (!pool)
      return;

   vkr_descriptor_pool_release(ctx, pool);
   vkr_descriptor_pool_destroy_and_remove(ctx, args);
}

static void
vkr_dispatch_vkResetDescriptorPool(struct vn_dispatch_context *dispatch,
                                   struct vn_command_vkResetDescriptorPool *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   struct vkr_context *ctx = dispatch->data;

   struct vkr_descriptor_pool *pool =
      vkr_descriptor_pool_from_handle(args->descriptorPool);
   if (!pool) {
      vkr_context_set_fatal(ctx);
      return;
   }

   vn_replace_vkResetDescriptorPool_args_handle(args);
   args->ret = vk->ResetDescriptorPool(args->device, args->descriptorPool, args->flags);

   vkr_descriptor_pool_release(ctx, pool);
   list_inithead(&pool->descriptor_sets);
}

static void
vkr_dispatch_vkAllocateDescriptorSets(struct vn_dispatch_context *dispatch,
                                      struct vn_command_vkAllocateDescriptorSets *args)
{
   struct vkr_context *ctx = dispatch->data;
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vkr_descriptor_pool *pool =
      vkr_descriptor_pool_from_handle(args->pAllocateInfo->descriptorPool);
   struct object_array arr;
   VkResult result;

   if (!pool) {
      vkr_context_set_fatal(ctx);
      return;
   }

   result = vkr_descriptor_set_create_array(ctx, args, &arr);
   if (result != VK_SUCCESS) {
      if (!(pool->flags & VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT))
         vkr_log("Warning: vkAllocateDescriptorSets failed(%u)", result);
      return;
   }

   vkr_descriptor_set_add_array(ctx, dev, pool, &arr);
}

static void
vkr_dispatch_vkFreeDescriptorSets(struct vn_dispatch_context *dispatch,
                                  struct vn_command_vkFreeDescriptorSets *args)
{
   struct vkr_context *ctx = dispatch->data;
   struct list_head free_list;

   /* args->pDescriptorSets is marked noautovalidity="true" */
   if (args->descriptorSetCount && !args->pDescriptorSets) {
      vkr_context_set_fatal(ctx);
      return;
   }

   /* gkvm journal: retain this (possibly partial) free keyed to the pool, with
    * the freed guest ids as aux — replay must free the dead subset of a batch
    * alloc so the id space evolves identically (pre-replace lookups) */
   if (ctx->journal && args->descriptorSetCount) {
      const struct vkr_descriptor_pool *jpool =
         vkr_descriptor_pool_from_handle(args->descriptorPool);
      uint64_t *ids = malloc(args->descriptorSetCount * sizeof(*ids));
      if (ids) {
         for (uint32_t i = 0; i < args->descriptorSetCount; i++) {
            const struct vkr_descriptor_set *set =
               vkr_descriptor_set_from_handle(args->pDescriptorSets[i]);
            ids[i] = set ? set->base.id : 0;
         }
         vkr_journal_note_free(ctx, jpool ? jpool->base.id : 0, ids,
                               args->descriptorSetCount);
         free(ids);
      }
   }

   vkr_descriptor_set_destroy_driver_handles(ctx, args, &free_list);
   vkr_context_remove_objects(ctx, &free_list);

   args->ret = VK_SUCCESS;
}

static void
vkr_dispatch_vkUpdateDescriptorSets(struct vn_dispatch_context *dispatch,
                                    struct vn_command_vkUpdateDescriptorSets *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   /* gkvm journal: key this update by every touched set (guest ids, pre-replace;
    * note_keys dedupes — zink typically hits one set with many writes) AND by every
    * referenced view/sampler/buffer: a NOTED entry dies with its first dead key, so
    * keying on the referenced objects prunes the entry the moment any of them is
    * destroyed — exactly the entries that would otherwise pile up as stale writes
    * and fail (noisily, recoverably) at replay (11k on a lived-in session). At
    * replay one dead reference drops the whole entry anyway, so the eager prune is
    * behavior-equivalent and keeps the journal lean. */
   for (uint32_t i = 0; i < args->descriptorWriteCount; i++) {
      const VkWriteDescriptorSet *w = &args->pDescriptorWrites[i];
      const struct vkr_descriptor_set *set =
         vkr_descriptor_set_from_handle(w->dstSet);
      if (set)
         vkr_journal_note_keys(dispatch->data, &set->base.id, 1);

      switch (w->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t k = 0; w->pImageInfo && k < w->descriptorCount; k++) {
            const struct vkr_image_view *view =
               vkr_image_view_from_handle(w->pImageInfo[k].imageView);
            const struct vkr_sampler *sampler =
               vkr_sampler_from_handle(w->pImageInfo[k].sampler);
            if (view)
               vkr_journal_note_keys(dispatch->data, &view->base.id, 1);
            if (sampler)
               vkr_journal_note_keys(dispatch->data, &sampler->base.id, 1);
         }
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t k = 0; w->pTexelBufferView && k < w->descriptorCount; k++) {
            const struct vkr_buffer_view *bview =
               vkr_buffer_view_from_handle(w->pTexelBufferView[k]);
            if (bview)
               vkr_journal_note_keys(dispatch->data, &bview->base.id, 1);
         }
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t k = 0; w->pBufferInfo && k < w->descriptorCount; k++) {
            const struct vkr_buffer *buf =
               vkr_buffer_from_handle(w->pBufferInfo[k].buffer);
            if (buf)
               vkr_journal_note_keys(dispatch->data, &buf->base.id, 1);
         }
         break;
      default:
         break;
      }
   }
   for (uint32_t i = 0; i < args->descriptorCopyCount; i++) {
      const struct vkr_descriptor_set *dst =
         vkr_descriptor_set_from_handle(args->pDescriptorCopies[i].dstSet);
      const struct vkr_descriptor_set *src =
         vkr_descriptor_set_from_handle(args->pDescriptorCopies[i].srcSet);
      if (dst)
         vkr_journal_note_keys(dispatch->data, &dst->base.id, 1);
      if (src)
         vkr_journal_note_keys(dispatch->data, &src->base.id, 1);
   }

   vn_replace_vkUpdateDescriptorSets_args_handle(args);
   vk->UpdateDescriptorSets(args->device, args->descriptorWriteCount,
                            args->pDescriptorWrites, args->descriptorCopyCount,
                            args->pDescriptorCopies);
}

static void
vkr_dispatch_vkCreateDescriptorUpdateTemplate(
   struct vn_dispatch_context *dispatch,
   struct vn_command_vkCreateDescriptorUpdateTemplate *args)
{
   vkr_descriptor_update_template_create_and_add(dispatch->data, args);
}

static void
vkr_dispatch_vkDestroyDescriptorUpdateTemplate(
   struct vn_dispatch_context *dispatch,
   struct vn_command_vkDestroyDescriptorUpdateTemplate *args)
{
   vkr_descriptor_update_template_destroy_and_remove(dispatch->data, args);
}

void
vkr_context_init_descriptor_set_layout_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkGetDescriptorSetLayoutSupport =
      vkr_dispatch_vkGetDescriptorSetLayoutSupport;
   dispatch->dispatch_vkCreateDescriptorSetLayout =
      vkr_dispatch_vkCreateDescriptorSetLayout;
   dispatch->dispatch_vkDestroyDescriptorSetLayout =
      vkr_dispatch_vkDestroyDescriptorSetLayout;
}

void
vkr_context_init_descriptor_pool_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkCreateDescriptorPool = vkr_dispatch_vkCreateDescriptorPool;
   dispatch->dispatch_vkDestroyDescriptorPool = vkr_dispatch_vkDestroyDescriptorPool;
   dispatch->dispatch_vkResetDescriptorPool = vkr_dispatch_vkResetDescriptorPool;
}

void
vkr_context_init_descriptor_set_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkAllocateDescriptorSets = vkr_dispatch_vkAllocateDescriptorSets;
   dispatch->dispatch_vkFreeDescriptorSets = vkr_dispatch_vkFreeDescriptorSets;
   dispatch->dispatch_vkUpdateDescriptorSets = vkr_dispatch_vkUpdateDescriptorSets;
}

void
vkr_context_init_descriptor_update_template_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkCreateDescriptorUpdateTemplate =
      vkr_dispatch_vkCreateDescriptorUpdateTemplate;
   dispatch->dispatch_vkDestroyDescriptorUpdateTemplate =
      vkr_dispatch_vkDestroyDescriptorUpdateTemplate;
   dispatch->dispatch_vkUpdateDescriptorSetWithTemplate = NULL;
}
