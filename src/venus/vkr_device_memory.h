/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VKR_DEVICE_MEMORY_H
#define VKR_DEVICE_MEMORY_H

#include "vkr_common.h"

struct gbm_bo;
struct vkr_mtl_shm;

struct vkr_device_memory {
   struct vkr_object base;

   struct vkr_device *device;

   bool might_export;

   /* limina snapshot-replay: guest id of the dedicated-alloc VkBuffer/VkImage this
    * memory's vkAllocateMemory references (0 = none). A blob resource created
    * from this memory pins it too: the alloc entry cannot REPLAY if its
    * dedicated object's create entry was pruned at destroy — the lookup fails,
    * the alloc drops, and the blob create finds no live memory (the run-25
    * failure: pinning the alloc alone was not enough). */
   uint64_t limina_dedicated_id;

   uint32_t property_flags;
   uint32_t valid_fd_types;

   /* gbm bo backing non-external mappable memory */
   struct gbm_bo *gbm_bo;

   /* udmabuf backing non-external mappable memory */
   int udmabuf_fd;

   /* Metal buffer backed by POSIX shared memory */
   struct vkr_mtl_shm *mtl_shm;

   /* limina tier-2 (macOS): borrowed ref to the global IOSurface backing the VkImage this
    * memory is bound to (set at vkBindImageMemory2 from vkr_image.mtl_iosurface). Lets the
    * scanout present resolve resource -> memory -> IOSurface id zero-copy. NOT owned here
    * (the vkr_image owns/frees it); we only read its id. */
   void *mtl_iosurface;

   /* limina tier-2 (macOS): RETAINED IOSurfaceRef of another context's window buffer this
    * memory host-pointer-imported (cross-context wl_buffer import); the ref keeps the
    * pixel bytes alive for this memory's lifetime. Released in vkr_device_memory_release. */
   void *imported_iosurface;

   uint64_t allocation_size;
   uint32_t memory_type_index;

   /* limina host-memory budget (vkr_budget.h): bytes charged for this memory and the
    * context they were charged to. Zero when the allocation only aliased memory someone
    * else had already paid for. Kept here because the credit happens at release, long
    * after the allocating thread's context binding is gone. */
   uint64_t limina_budget_size;
   uint32_t limina_budget_ctx;

   bool exported;

   /* limina snapshot-replay P2: this memory was created by importing a virgl
    * resource's bytes (VkImportMemoryResourceInfoMESA) — it aliases ANOTHER
    * storage (the exporter's IOSurface / blob / shm bytes), so content
    * capture/restore happens at the source, never here. */
   bool limina_res_imported;
};
VKR_DEFINE_OBJECT_CAST(device_memory, VK_OBJECT_TYPE_DEVICE_MEMORY, VkDeviceMemory)

void
vkr_context_init_device_memory_dispatch(struct vkr_context *ctx);

void
vkr_device_memory_release(struct vkr_device_memory *mem);

bool
vkr_device_memory_export_blob(struct vkr_device_memory *mem,
                              uint64_t blob_size,
                              uint32_t blob_flags,
                              struct virgl_context_blob *out_blob);

/* limina snapshot-replay P2: whether this memory's bytes must be captured by the
 * snapshot (vs. being covered elsewhere: map_ptr blobs by the VMM's mapped-blob
 * capture, imports by their exporting storage). */
bool
vkr_device_memory_capturable(const struct vkr_device_memory *mem);

/* limina snapshot-replay P2: copy min(size, allocation_size) bytes between `buf`
 * and the memory's host mapping (vkMapMemory/vkUnmapMemory round-trip; KK/UMA
 * memory is always host-visible — the placement heap's CPU alias, so raw image
 * bytes included). to_mem=false reads (capture), to_mem=true writes (restore). */
bool
vkr_device_memory_content_copy(struct vkr_device_memory *mem,
                               void *buf,
                               uint64_t size,
                               bool to_mem);

#endif /* VKR_DEVICE_MEMORY_H */
