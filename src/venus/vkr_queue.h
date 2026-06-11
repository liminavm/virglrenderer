/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VKR_QUEUE_H
#define VKR_QUEUE_H

#include "vkr_common.h"

/* gkvm fence-accurate present (#8/#31): a host-injected fence submitted on the
 * reserved VKR_GKVM_PRESENT_RING when the VMM receives a scanout RESOURCE_FLUSH.
 * It retires only after (a) every ring of the context has decoded past the
 * commands buffered at injection time (so the frame's vkQueueSubmit reached the
 * driver) and (b) a zero-command queue sync on each of the context's queues
 * signals (true GPU completion). `pending` counts outstanding ring barriers
 * during phase (a), then outstanding queue syncs during phase (b); whoever
 * drops it to zero advances/retires.
 */
struct vkr_present_fence {
   struct vkr_context *ctx;
   uint64_t fence_id;
   atomic_int pending;
};

struct vkr_queue_sync {
   VkFence fence;
   bool device_lost;

   uint32_t flags;
   uint32_t ring_idx;
   uint64_t fence_id;

   /* gkvm: non-NULL for a present-fence sync; retire decrements the present
    * fence instead of calling retire_fence directly. */
   struct vkr_present_fence *gkvm_present;

   struct list_head head;
};

struct vkr_queue {
   struct vkr_object base;

   struct vkr_context *context;
   struct vkr_device *device;

   VkDeviceQueueCreateFlags flags;
   uint32_t family;
   uint32_t index;

   /* only used when client driver uses multiple timelines */
   uint32_t ring_idx;

   /* Ensure host access to VkQueue being externally synchronized between renderer main
    * thread and ring thread.
    */
   mtx_t vk_mutex;

   /* Submitted fences are added to sync_thread.syncs first. With required
    * VKR_RENDERER_THREAD_SYNC and VKR_RENDERER_ASYNC_FENCE_CB in render server, the sync
    * thread calls vkWaitForFences and retires signaled fences in order.
    */
   struct {
      mtx_t mutex;
      cnd_t cond;
      struct list_head syncs;
      thrd_t thread;
      bool join;
   } sync_thread;
};
VKR_DEFINE_OBJECT_CAST(queue, VK_OBJECT_TYPE_QUEUE, VkQueue)

struct vkr_fence {
   struct vkr_object base;
};
VKR_DEFINE_OBJECT_CAST(fence, VK_OBJECT_TYPE_FENCE, VkFence)

struct vkr_semaphore {
   struct vkr_object base;
};
VKR_DEFINE_OBJECT_CAST(semaphore, VK_OBJECT_TYPE_SEMAPHORE, VkSemaphore)

struct vkr_event {
   struct vkr_object base;
};
VKR_DEFINE_OBJECT_CAST(event, VK_OBJECT_TYPE_EVENT, VkEvent)

void
vkr_context_init_queue_dispatch(struct vkr_context *ctx);

void
vkr_context_init_fence_dispatch(struct vkr_context *ctx);

void
vkr_context_init_semaphore_dispatch(struct vkr_context *ctx);

void
vkr_context_init_event_dispatch(struct vkr_context *ctx);

struct vkr_queue *
vkr_queue_create(struct vkr_context *ctx,
                 struct vkr_device *dev,
                 VkDeviceQueueCreateFlags flags,
                 uint32_t family,
                 uint32_t index,
                 VkQueue handle);

void
vkr_queue_destroy(struct vkr_context *ctx, struct vkr_queue *queue);

/* gkvm: release one reference on a present fence; the release that drops it to
 * zero retires it toward the VMM (write_context_fence on VKR_GKVM_PRESENT_RING)
 * and frees it. */
void
vkr_present_fence_release(struct vkr_present_fence *pf);

/* gkvm: submit a zero-command queue sync that releases `pf` when the GPU
 * completes all work submitted to `queue` so far. */
bool
vkr_queue_sync_submit_present(struct vkr_queue *queue, struct vkr_present_fence *pf);

bool
vkr_queue_sync_submit(struct vkr_queue *queue,
                      uint32_t flags,
                      uint32_t ring_idx,
                      uint64_t fence_id);

#endif /* VKR_QUEUE_H */
