/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_device_memory.h"

#include <math.h>

#include "venus-protocol/vn_protocol_renderer_transport.h"

#include "vkr_device_memory_gen.h"
#include "vkr_image.h" /* #30 Path A: vkr_image_from_handle for the dedicated scanout image */
#include "vkr_metal_helpers.h"
#include "vkr_physical_device.h"

static bool
vkr_get_fd_info_from_resource_info(struct vkr_context *ctx,
                                   const VkImportMemoryResourceInfoMESA *res_info,
                                   VkImportMemoryFdInfoKHR *out)
{
   struct vkr_resource *res = vkr_context_get_resource(ctx, res_info->resourceId);
   if (!res) {
      /* NOT protocol corruption: guest CREATE_BLOB is fire-and-forget, so when the
       * host fails a create (e.g. under memory pressure) the guest still holds a
       * live-looking handle and legitimately imports the dead id (2026-07-09
       * dogfood crash #1). Callers translate `false` into
       * VK_ERROR_INVALID_EXTERNAL_HANDLE — recoverable, ring stays alive. */
      vkr_log_error("failed to import resource: invalid res_id %u (returning "
                    "VK_ERROR_INVALID_EXTERNAL_HANDLE, ring stays alive)",
                    res_info->resourceId);
      return false;
   }

   VkExternalMemoryHandleTypeFlagBits handle_type;
   switch (res->fd_type) {
   case VIRGL_RESOURCE_FD_DMABUF:
      handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
      break;
   case VIRGL_RESOURCE_FD_OPAQUE:
      handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      break;
   default:
      return false;
   }

   int fd = os_dupfd_cloexec(res->u.fd);
   if (fd < 0)
      return false;

   *out = (VkImportMemoryFdInfoKHR){
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = res_info->pNext,
      .fd = fd,
      .handleType = handle_type,
   };
   return true;
}

#if defined(HAVE_LINUX_UDMABUF_H) && defined(HAVE_MEMFD_CREATE)
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

static VkResult
vkr_udmabuf_get_fd_info_from_allocation_info(struct vkr_physical_device *physical_dev,
                                             const VkMemoryAllocateInfo *alloc_info,
                                             int *out_udmabuf_fd,
                                             VkImportMemoryFdInfoKHR *out_fd_info)
{
   int memfd = -1;
   int udmabuf_fd = -1;
   int fd = -1;

   memfd = memfd_create("vkr-udmabuf", MFD_CLOEXEC | MFD_ALLOW_SEALING);
   if (memfd < 0) {
      vkr_log("memfd_create failed (%s)", strerror(errno));
      goto fail;
   }

   const size_t size = align(alloc_info->allocationSize, getpagesize());
   int ret = ftruncate(memfd, size);
   if (ret) {
      vkr_log("ftruncate failed (%s)", strerror(errno));
      goto fail;
   }

   ret = fcntl(memfd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW);
   if (ret) {
      vkr_log("fcntl F_ADD_SEALS failed (%s)", strerror(errno));
      goto fail;
   }

   const struct udmabuf_create create = {
      .memfd = memfd,
      .flags = UDMABUF_FLAGS_CLOEXEC,
      .size = size,
   };
   udmabuf_fd = ioctl(physical_dev->udmabuf_dev_fd, UDMABUF_CREATE, &create);
   if (udmabuf_fd < 0) {
      vkr_log("ioctl UDMABUF_CREATE failed (%s)", strerror(errno));
      goto fail;
   }

   fd = os_dupfd_cloexec(udmabuf_fd);
   if (fd < 0) {
      vkr_log("os_dupfd_cloexec failed (%s)", strerror(errno));
      goto fail;
   }

   close(memfd);

   *out_udmabuf_fd = udmabuf_fd;
   *out_fd_info = (VkImportMemoryFdInfoKHR){
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = alloc_info->pNext,
      .fd = fd,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };

   return VK_SUCCESS;

fail:
   if (udmabuf_fd >= 0)
      close(udmabuf_fd);
   if (memfd >= 0)
      close(memfd);
   return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

#else  /* HAVE_LINUX_UDMABUF_H && HAVE_MEMFD_CREATE */

static inline VkResult
vkr_udmabuf_get_fd_info_from_allocation_info(
   UNUSED struct vkr_physical_device *physical_dev,
   UNUSED const VkMemoryAllocateInfo *alloc_info,
   UNUSED int *out_udmabuf_fd,
   UNUSED VkImportMemoryFdInfoKHR *out_fd_info)
{
   vkr_log("udmabuf_allocation is not enabled");
   return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

#endif /* HAVE_LINUX_UDMABUF_H && HAVE_MEMFD_CREATE */

#ifdef ENABLE_GBM_ALLOCATION
#include <gbm.h>

#define GBM_BO_USE_SW_READ_RARELY (1 << 10)
#define GBM_BO_USE_SW_WRITE_RARELY (1 << 12)

static inline int
vkr_gbm_bo_get_fd(void *gbm_bo)
{
   assert(gbm_bo);

   /* gbm_bo_get_fd returns negative error code on failure */
   return gbm_bo_get_fd(gbm_bo);
}

static inline void
vkr_gbm_bo_destroy(void *gbm_bo)
{
   gbm_bo_destroy(gbm_bo);
}

static VkResult
vkr_gbm_get_fd_info_from_allocation_info(struct vkr_physical_device *physical_dev,
                                         const VkMemoryAllocateInfo *alloc_info,
                                         void **out_gbm_bo,
                                         VkImportMemoryFdInfoKHR *out_fd_info)
{
   const uint32_t flags =
      GBM_BO_USE_LINEAR | GBM_BO_USE_SW_READ_RARELY | GBM_BO_USE_SW_WRITE_RARELY;
   struct gbm_bo *gbm_bo;
   int fd = -1;

   assert(physical_dev->gbm_device);

   /*
    * Reject here for simplicity. Letting VkPhysicalDeviceVulkan11Properties return
    * min(maxMemoryAllocationSize, UINT32_MAX) will affect unmappable scenarios.
    */
   if (alloc_info->allocationSize > UINT32_MAX)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   /* Page alignment is used on all implementations we support. */
   const uint32_t alloc_size = align(alloc_info->allocationSize, getpagesize());
#ifdef MINIGBM
   const uint32_t format = GBM_FORMAT_R8;
   const uint32_t width = alloc_size;
   const uint32_t height = 1;
#else
   /* Mesa gbm has texture size limitations, so we can't rely on R8 here. Instead, we
    * allocate a large enough linear rgba8 buffer.
    */
   const uint32_t format = GBM_FORMAT_ABGR8888;
   const uint8_t pixel_bytes = 4;
   const uint32_t width =
      (uint32_t)ceil(sqrt((alloc_size + pixel_bytes - 1) / pixel_bytes));
   const uint32_t height = width;
#endif /* MINIGBM */

   gbm_bo = gbm_bo_create(physical_dev->gbm_device, width, height, format, flags);
   if (!gbm_bo)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   fd = vkr_gbm_bo_get_fd(gbm_bo);
   if (fd < 0) {
      vkr_gbm_bo_destroy(gbm_bo);
      return fd == -EMFILE ? VK_ERROR_TOO_MANY_OBJECTS : VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   *out_gbm_bo = (void *)gbm_bo;
   *out_fd_info = (VkImportMemoryFdInfoKHR){
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = alloc_info->pNext,
      .fd = fd,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   return VK_SUCCESS;
}

#else

static inline int
vkr_gbm_bo_get_fd(ASSERTED void *gbm_bo)
{
   vkr_log("minigbm_allocation is not enabled");
   assert(!gbm_bo);
   return -1;
}

static inline void
vkr_gbm_bo_destroy(ASSERTED void *gbm_bo)
{
   vkr_log("minigbm_allocation is not enabled");
   assert(!gbm_bo);
}

static inline VkResult
vkr_gbm_get_fd_info_from_allocation_info(UNUSED struct vkr_physical_device *physical_dev,
                                         UNUSED const VkMemoryAllocateInfo *alloc_info,
                                         UNUSED void **out_gbm_bo,
                                         UNUSED VkImportMemoryFdInfoKHR *out_fd_info)
{
   vkr_log("minigbm_allocation is not enabled");
   return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

#endif /* ENABLE_GBM_ALLOCATION */

static void
vkr_dispatch_vkAllocateMemory(struct vn_dispatch_context *dispatch,
                              struct vn_command_vkAllocateMemory *args)
{
   TRACE_FUNC();
   struct vkr_context *ctx = dispatch->data;
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vkr_physical_device *physical_dev = dev->physical_device;

   VkMemoryAllocateInfo *alloc_info = (VkMemoryAllocateInfo *)args->pAllocateInfo;
   const uint32_t mem_type_index = alloc_info->memoryTypeIndex;
   if (unlikely(mem_type_index >= physical_dev->memory_properties.memoryTypeCount)) {
      args->ret = VK_ERROR_UNKNOWN;
      return;
   }

   /* translate VkImportMemoryResourceInfoMESA into VkImportMemoryFdInfoKHR in place */
   VkImportMemoryFdInfoKHR local_import_info = { .fd = -1 };
   VkImportMemoryResourceInfoMESA *res_info = NULL;
#ifdef __APPLE__
   /* gkvm tier-2: cross-context import (compositor importing a client's wl_buffer
    * dmabuf). macOS has no fd-flavored external memory; the exporter's pixel bytes
    * live in a GLOBAL IOSurface (window buffers — the resource's SHM fd is only a
    * carrier) or behind the #28 map_ptr / SHM mapping. Translate the import into
    * VkImportMemoryHostPointerInfoEXT over those bytes (function scope — chained
    * into alloc_info and must outlive the driver call). */
   VkImportMemoryHostPointerInfoEXT gkvm_res_import = { 0 };
   void *gkvm_imported_iosurface = NULL;
   bool gkvm_res_imported = false;
#endif
   VkBaseInStructure *prev_of_res_info = vkr_find_prev_struct(
      alloc_info, VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA);
   if (prev_of_res_info) {
      res_info = (VkImportMemoryResourceInfoMESA *)prev_of_res_info->pNext;
#ifdef __APPLE__
      struct vkr_resource *gkvm_res = vkr_context_get_resource(ctx, res_info->resourceId);
      if (gkvm_res) {
         void *ptr = NULL;
         uint64_t span = 0;
         if (gkvm_res->iosurface_id) {
            gkvm_imported_iosurface =
               vkr_mtl_iosurface_lookup(gkvm_res->iosurface_id, &ptr, &span);
            if (!gkvm_imported_iosurface)
               vkr_log("gkvm: import res %u: IOSurface id=%u lookup failed",
                       res_info->resourceId, gkvm_res->iosurface_id);
         }
         if (!ptr && gkvm_res->map_ptr) {
            ptr = (void *)(uintptr_t)gkvm_res->map_ptr;
            span = gkvm_res->size;
         }
         if (!ptr && gkvm_res->fd_type == VIRGL_RESOURCE_FD_SHM && !gkvm_res->u_is_fd) {
            /* Cross-context SHM imports store the server-side mmap in u.data; a
             * SAME-context (export-side) SHM carrier stores an owned fd in the
             * union instead (u_is_fd) — those resolve via iosurface_id above. */
            ptr = gkvm_res->u.data;
            span = gkvm_res->size;
         }
         if (ptr) {
            /* VK_EXT_external_memory_host needs page-aligned ptr + size multiple of
             * minImportedHostPointerAlignment; blob sizes are guest-page (16K) multiples
             * and IOSurface alloc sizes page-rounded already — round up defensively. */
            const uint64_t page_size = getpagesize();
            span = (span + page_size - 1) & ~(page_size - 1);
            prev_of_res_info->pNext = res_info->pNext; /* unlink res_info */
            gkvm_res_import.sType =
               VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
            gkvm_res_import.handleType =
               VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
            gkvm_res_import.pHostPointer = ptr;
            gkvm_res_import.pNext = alloc_info->pNext;
            alloc_info->pNext = &gkvm_res_import;
            /* The host pointer backs exactly `span` bytes. allocationSize is
             * guest-declared (venus); a value larger than the backing would make the
             * imported VK_EXT_external_memory_host memory (and KK's
             * newBufferWithBytesNoCopy over it) run past the allocation -> host OOB.
             * Clamp to the backing in both directions: a smaller request still maps
             * the whole page-rounded backing, a larger one is capped. */
            alloc_info->allocationSize = span;
            gkvm_res_imported = true;
            vkr_log("gkvm: import res %u <- host-pointer (IOSurface id=%u base=%p "
                    "size=%" PRIu64 ")",
                    res_info->resourceId, gkvm_res->iosurface_id, ptr, span);
         }
      }
      if (!gkvm_res_imported) {
         if (!vkr_get_fd_info_from_resource_info(ctx, res_info, &local_import_info)) {
            /* loud: a silent error here corrupts the ring (the guest treats the alloc
             * as async-success and the follow-up bind goes CS-fatal). */
            vkr_log("gkvm: import res %u failed (fd_type=%d, no IOSurface/map_ptr/shm "
                    "bytes) -> VK_ERROR_INVALID_EXTERNAL_HANDLE",
                    res_info->resourceId,
                    gkvm_res ? (int)gkvm_res->fd_type : -999);
            args->ret = VK_ERROR_INVALID_EXTERNAL_HANDLE;
            return;
         }
         prev_of_res_info->pNext = (const struct VkBaseInStructure *)&local_import_info;
      }
#else
      if (!vkr_get_fd_info_from_resource_info(ctx, res_info, &local_import_info)) {
         args->ret = VK_ERROR_INVALID_EXTERNAL_HANDLE;
         return;
      }

      prev_of_res_info->pNext = (const struct VkBaseInStructure *)&local_import_info;
#endif
   }

   VkExportMemoryAllocateInfo *export_info =
      vkr_find_struct(alloc_info->pNext, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);

   /* track if driver has requested export allocation */
   const bool might_export = export_info && export_info->handleTypes;

   /* XXX Force dma_buf/opaque fd export or gbm bo import until a new extension that
    * supports direct export from host visible memory
    *
    * Most VkImage and VkBuffer are non-external while most VkDeviceMemory are external
    * if allocated with a host visible memory type. We still violate the spec by binding
    * external memory to non-external image or buffer, which needs spec changes with a
    * new extension.
    *
    * Skip forcing external if a valid VkImportMemoryResourceInfoMESA is provided, since
    * the mapping will be directly set up from the existing virgl resource.
    */
   const uint32_t property_flags =
      physical_dev->memory_properties.memoryTypes[mem_type_index].propertyFlags;
   uint32_t valid_fd_types = 0;
   int udmabuf_fd = -1;
   void *gbm_bo = NULL;
   VkExportMemoryAllocateInfo local_export_info;

   /* macOS: use shared memory + Metal buffer for HOST_VISIBLE cross-process sharing. */
   struct vkr_mtl_shm *mtl_shm = NULL;
#ifdef __APPLE__
   /* gkvm #30 Path A: the IOSurface of the fix-A scanout image this memory is dedicated to.
    * The image renders into this IOSurface via VkImportMetalIOSurfaceInfoEXT (vkr_image.c);
    * here it only marks the memory as needing an exportable mtl_shm blob carrier. NULL for
    * non-scanout memory. */
   struct vkr_mtl_iosurface *gkvm_scanout_surf = NULL;
   /* gkvm KK path: host-pointer import of the scanout IOSurface bytes (function scope —
    * chained into alloc_info and must outlive the driver call). */
   VkImportMemoryHostPointerInfoEXT gkvm_host_import = { 0 };

   /* gkvm #28: plain HOST_VISIBLE memory is NO LONGER backed by an imported mtl_shm carrier.
    * The old path allocated a POSIX shm + Metal buffer (newBufferWithBytesNoCopy) and imported
    * it as the VkDeviceMemory; the blob then exported the shm fd, which the VMM mmap'd a SECOND
    * time for the guest. Guest CPU writes to that second mapping never reached the GPU's view of
    * the first mapping — the #28 "venus renders black" data bug. Instead we let MoltenVK allocate
    * its own Shared MTLBuffer and, in vkr_device_memory_export_blob, vkMapMemory it and share
    * MoltenVK's own pointer with the VMM (one mapping, guest CPU + host GPU coherent — the slp /
    * krunkit model). HOST_VISIBLE now falls through to the generic block below, which is a no-op
    * on macOS (no dma_buf/opaque-fd/gbm/udmabuf), leaving valid_fd_types == 0 and mtl_shm NULL.
    * The dedicated #30 scanout image still gets a carrier further down (gkvm_scanout_surf). */
#endif /* __APPLE__ */
   if ((property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && !res_info) {
      /* An implementation can support dma_buf import along with opaque fd export/import.
       * If the client driver is using external memory and requesting dma_buf, without
       * dma_buf fd export support, we must use gbm bo import path instead of forcing
       * opaque fd export. e.g. the client driver uses external memory for wsi image.
       */
      const bool no_dma_buf_export =
         !export_info ||
         !(export_info->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
      const bool force_gbm_import = !!physical_dev->gbm_device;
      const bool force_udmabuf_import = physical_dev->udmabuf_dev_fd >= 0;
      if (!(force_gbm_import || force_udmabuf_import) &&
          (physical_dev->is_dma_buf_fd_export_supported ||
           (physical_dev->is_opaque_fd_export_supported && no_dma_buf_export))) {
         const VkExternalMemoryHandleTypeFlagBits handle_type =
            physical_dev->is_dma_buf_fd_export_supported
               ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
               : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
         if (export_info) {
            export_info->handleTypes |= handle_type;
         } else {
            local_export_info = (const VkExportMemoryAllocateInfo){
               .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
               .pNext = alloc_info->pNext,
               .handleTypes = handle_type,
            };
            export_info = &local_export_info;
            alloc_info->pNext = &local_export_info;

            if (handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
               /* Guest virtgpu kernel aligns up blob mem size to the page boundary. No
                * matter dma-buf or opaque fd export allocation, the actual allocation
                * in most cases would follow the same padding. For dma-buf, we are able
                * to allocate and validate via lseek below. For opaque fd, Vulkan spec
                * requires to use the same size with allocation time for the mapper,
                * either vkr_allocator or vulkano, to import and populate the mapping.
                * Accessible mapping given by vkMapMemory is normally limited to just the
                * alloc size. When KVM register the mappings to guest pci bar, that
                * involves an invalid chunk towards the end of the page boundary. As a
                * result, any guest side accelerated instructions that rely on the
                * paddings can end up with Illegal instruction error. The most common
                * trigger is via memcpy'ing valid range to the guest side mapped buffer
                * memory, and then, e.g. __memcpy_avx_unaligned_erms  can hit the error.
                */
               alloc_info->allocationSize =
                  align(alloc_info->allocationSize, getpagesize());
            }
         }
      } else if (physical_dev->EXT_external_memory_dma_buf) {
         /* Allocate dma_buf externally and force to import. */
         if (export_info) {
            /* Strip export info since valid_fd_types can only be dma_buf here. */
            VkBaseInStructure *prev_of_export_info = vkr_find_prev_struct(
               alloc_info, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);

            prev_of_export_info->pNext = export_info->pNext;
            export_info = NULL;
         }

         if (force_udmabuf_import) {
            /* To be noted, there are 2 limits for udmabuf:
             * - list_limit: udmabuf_create_list->count limit. Default is 1024.
             * - size_limit_mb: Max size of a dmabuf, in megabytes. Default is 64.
             */
            args->ret = vkr_udmabuf_get_fd_info_from_allocation_info(
               physical_dev, alloc_info, &udmabuf_fd, &local_import_info);
         } else {
            args->ret = vkr_gbm_get_fd_info_from_allocation_info(
               physical_dev, alloc_info, &gbm_bo, &local_import_info);
         }
         if (args->ret != VK_SUCCESS)
            return;

         alloc_info->pNext = &local_import_info;
         valid_fd_types = 1 << VIRGL_RESOURCE_FD_DMABUF;
      }
   }

   if (export_info) {
      if (export_info->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
         valid_fd_types |= 1 << VIRGL_RESOURCE_FD_OPAQUE;
      if (export_info->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)
         valid_fd_types |= 1 << VIRGL_RESOURCE_FD_DMABUF;
   }

#ifdef __APPLE__
   /* gkvm #30 (seated desktop): mutter allocates the KMS scanout framebuffer's
    * VkDeviceMemory with VkExportMemoryAllocateInfo(OPAQUE_FD/DMA_BUF), but MoltenVK
    * cannot export device-local memory that way -> vkAllocateMemory fails with
    * VK_ERROR_INITIALIZATION_FAILED, the object is never created, and the subsequent
    * vkBindImageMemory2 hits "failed to look up object" -> CS fatal -> the venus context
    * is destroyed -> gnome-shell crash-loops. The scanout VkImage is already backed by an
    * IOSurface (fix A), so this memory only needs to be a valid placeholder bind target.
    *
    * Strip the unsupported export so the raw allocation succeeds, BUT give the memory an
    * mtl_shm carrier so it is still exportable as a virtio-gpu blob: the guest exports this
    * memory as the KMS scanout dmabuf, and that export crosses the render-server proxy
    * socket, which needs a real fd (a non-exportable memory yields "proxy: invalid reply for
    * blob N" -> context destroyed -> crash-loop). The SHM bytes are NOT the scanout pixels;
    * present resolves resource -> memory -> the bound image's IOSurface (linked at
    * vkBindImageMemory2, see vkr_image.c) and scans that out zero-copy. The carrier is a
    * side allocation -- NOT chained into alloc_info -- so the VkDeviceMemory stays a plain
    * device-local placeholder MoltenVK is happy with. (The host-visible mtl_shm path above
    * is untouched.) */
   /* gkvm #30 (seated desktop): if this memory is the *dedicated* allocation for a fix-A
    * scanout VkImage, the image is already IOSurface-backed via VkImportMetalIOSurfaceInfoEXT
    * at image create (vkr_image.c) — MoltenVK's MVKImagePlane::getMTLTexture renders into the
    * IOSurface (the `_image->_ioSurface` branch). We do NOT import the MTLTexture as this
    * memory: that would take priority (the `dvcMem->_mtlTexture` branch) and use our texture
    * verbatim, silently no-op'ing the render. We only need a real, exportable mtl_shm carrier
    * so the guest can export this dedicated memory as the virtio-gpu scanout blob (present
    * resolves resource -> memory -> IOSurface, see vkr_device_memory_export_blob); the
    * carrier's bytes are never the scanout pixels. */
   /* Imported memories (res_info) already alias the EXPORTER's bytes — never re-route
    * them to the dedicated image's own (fresh, wrong) IOSurface. */
   if (!res_info) {
      const VkMemoryDedicatedAllocateInfo *ded = vkr_find_struct(
         alloc_info->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
      if (ded && ded->image != VK_NULL_HANDLE) {
         struct vkr_image *img = vkr_image_from_handle(ded->image);
         if (img && img->mtl_iosurface)
            gkvm_scanout_surf = img->mtl_iosurface;
      }
   }
   if (gkvm_scanout_surf && !mtl_shm) {
      if (export_info) {
         VkBaseInStructure *prev_of_export =
            vkr_find_prev_struct(alloc_info, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
         if (prev_of_export)
            prev_of_export->pNext = export_info->pNext;
         export_info = NULL;
      }
      valid_fd_types = 0;

      /* Side carrier: a real fd for the blob export only (bytes unused for pixels). */
      mtl_shm = vkr_mtl_shm_alloc(dev->mtl_device, alloc_info->allocationSize);
      if (!mtl_shm) {
         args->ret = VK_ERROR_OUT_OF_HOST_MEMORY;
         return;
      }

      if (!physical_dev->EXT_metal_objects) {
         /* KosmicKrisp: the image can't adopt the IOSurface via VK_EXT_metal_objects
          * (vkr_image.c made it LINEAR instead) — back the memory itself with the
          * IOSurface bytes via VK_EXT_external_memory_host. KK builds linear-image
          * textures from the memory's MTLBuffer (newBufferWithBytesNoCopy over this
          * pointer), so the render lands in the IOSurface. The pointer is page-aligned;
          * the length passed to Metal must be page-aligned too, so round the allocation
          * up to the IOSurface's own (page-rounded) alloc size. */
         struct vkr_mtl_iosurface *gkvm_io =
            (struct vkr_mtl_iosurface *)gkvm_scanout_surf;
         gkvm_host_import.sType =
            VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
         gkvm_host_import.handleType =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
         gkvm_host_import.pHostPointer = gkvm_io->base_addr;
         gkvm_host_import.pNext = alloc_info->pNext;
         alloc_info->pNext = &gkvm_host_import;
         /* Clamp to the IOSurface's backing size: a guest-declared allocationSize
          * larger than the backing would import past it (host OOB). See the matching
          * clamp on the resource-import path above. */
         alloc_info->allocationSize = gkvm_io->alloc_size;
         vkr_log("gkvm: KK scanout memory <- host-pointer import of IOSurface id=%u "
                 "(base=%p size=%" PRIu64 ")",
                 gkvm_io->id, gkvm_io->base_addr, (uint64_t)gkvm_io->alloc_size);
      } else {
         vkr_log("gkvm: scanout memory -> mtl_shm carrier for blob export; image renders "
                 "into its IOSurface (id=%u) via useIOSurface",
                 ((struct vkr_mtl_iosurface *)gkvm_scanout_surf)->id);
      }
   }

   if (export_info && !mtl_shm &&
       (export_info->handleTypes & (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
                                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT))) {
      VkBaseInStructure *prev_of_export =
         vkr_find_prev_struct(alloc_info, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
      if (prev_of_export) {
         prev_of_export->pNext = export_info->pNext;
         export_info = NULL;
         valid_fd_types = 0;

         mtl_shm = vkr_mtl_shm_alloc(dev->mtl_device, alloc_info->allocationSize);
         if (!mtl_shm) {
            args->ret = VK_ERROR_OUT_OF_HOST_MEMORY;
            return;
         }
         vkr_log("gkvm: scanout memory -> stripped OPAQUE/DMA_BUF export, attached mtl_shm "
                 "carrier (size=%llu); present uses the bound image's IOSurface",
                 (unsigned long long)alloc_info->allocationSize);
      }
   }
#endif

   struct vkr_device_memory *mem = vkr_device_memory_create_and_add(ctx, args);
   if (!mem) {
      if (local_import_info.fd >= 0)
         close(local_import_info.fd);
      if (gbm_bo)
         vkr_gbm_bo_destroy(gbm_bo);
      vkr_mtl_shm_free(mtl_shm);
#ifdef __APPLE__
      vkr_mtl_iosurface_release_ref(gkvm_imported_iosurface);
#endif
      return;
   }

   mem->device = dev;
   mem->might_export = might_export;
   mem->property_flags = property_flags;
   mem->valid_fd_types = valid_fd_types;
   mem->udmabuf_fd = udmabuf_fd;
   mem->gbm_bo = gbm_bo;
   mem->mtl_shm = mtl_shm;
   mem->allocation_size = alloc_info->allocationSize;
   mem->memory_type_index = mem_type_index;
#ifdef __APPLE__
   /* gkvm #30: link the scanout image's IOSurface onto this dedicated memory now, so
    * vkr_device_memory_export_blob carries the id even before vkBindImageMemory2. */
   if (gkvm_scanout_surf)
      mem->mtl_iosurface = gkvm_scanout_surf;
   /* gkvm: hold the imported window buffer's IOSurface for this memory's lifetime. */
   mem->imported_iosurface = gkvm_imported_iosurface;
#endif
}

static void
vkr_dispatch_vkFreeMemory(struct vn_dispatch_context *dispatch,
                          struct vn_command_vkFreeMemory *args)
{
   TRACE_FUNC();
   struct vkr_device_memory *mem = vkr_device_memory_from_handle(args->memory);
   if (!mem)
      return;

   vkr_device_memory_release(mem);
   vkr_device_memory_destroy_and_remove(dispatch->data, args);
}

static void
vkr_dispatch_vkGetDeviceMemoryCommitment(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetDeviceMemoryCommitment *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetDeviceMemoryCommitment_args_handle(args);
   vk->GetDeviceMemoryCommitment(args->device, args->memory,
                                 args->pCommittedMemoryInBytes);
}

static void
vkr_dispatch_vkGetDeviceMemoryOpaqueCaptureAddress(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetDeviceMemoryOpaqueCaptureAddress *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetDeviceMemoryOpaqueCaptureAddress_args_handle(args);
   args->ret = vk->GetDeviceMemoryOpaqueCaptureAddress(args->device, args->pInfo);
}

static void
vkr_dispatch_vkGetMemoryResourcePropertiesMESA(
   struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetMemoryResourcePropertiesMESA *args)
{
   struct vkr_context *ctx = dispatch->data;
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   struct vkr_resource *res = vkr_context_get_resource(ctx, args->resourceId);
   if (!res) {
      /* Same rationale as vkr_get_fd_info_from_resource_info: a dead resource id is
       * a reachable runtime state (failed fire-and-forget CREATE_BLOB), not protocol
       * corruption. Fail just this query; poisoning the ring here SIGABRTs the whole
       * guest process (mesa vn_relax aborts on the FATAL bit). */
      vkr_log_error("failed to query resource props: invalid res_id %u (returning "
                    "VK_ERROR_INVALID_EXTERNAL_HANDLE, ring stays alive)",
                    args->resourceId);
      args->ret = VK_ERROR_INVALID_EXTERNAL_HANDLE;
      return;
   }

#ifdef __APPLE__
   /* macOS: virtio-gpu resources are IOSurface / host-memory / shm backed, NOT Linux dmabufs, so
    * the dmabuf-only gate below would reject every resource here. The vkAllocateMemory import path
    * (see the gkvm branch above) resolves such a resource to a host pointer and imports it as
    * HOST_ALLOCATION_BIT_EXT — this property query MUST agree with that, or the guest's
    * vkGetMemoryFdPropertiesKHR returns memoryTypeBits=0 (VK_ERROR_INVALID_EXTERNAL_HANDLE) for a
    * buffer the very next vkAllocateMemory then imports and binds (the venus-bugs "#1" inconsistency
    * the compositor hit). Mirror the import: resolve the host pointer and report the memory types
    * VK_EXT_external_memory_host accepts for it. */
   {
      void *ptr = NULL;
      uint64_t span = 0;
      if (res->iosurface_id)
         vkr_mtl_iosurface_lookup(res->iosurface_id, &ptr, &span);
      if (!ptr && res->map_ptr)
         ptr = (void *)(uintptr_t)res->map_ptr;
      if (!ptr && res->fd_type == VIRGL_RESOURCE_FD_SHM && !res->u_is_fd)
         /* cross-context SHM stores the server-side mmap in u.data; a same-context
          * carrier resource stores an owned fd instead (u_is_fd). */
         ptr = res->u.data;
      (void)span;
      if (ptr) {
         /* Host-pointer-backed resource: report the host-visible memory types — the set a
          * VK_EXT_external_memory_host / HOST_ALLOCATION_BIT_EXT import accepts (which is exactly
          * how the vkAllocateMemory gkvm branch imports this resource). The guest intersects this
          * with the image's own memory_type_bits, so the query now AGREES with the import that
          * follows it. We compute it from the cached device memory properties rather than calling
          * vkGetMemoryHostPointerPropertiesEXT, because VK_EXT_external_memory_host is not in the
          * host device's enabled-extension list (KK accepts the import struct regardless), so
          * vkGetDeviceProcAddr returns NULL for that entrypoint. */
         const VkPhysicalDeviceMemoryProperties *mp = &dev->physical_device->memory_properties;
         uint32_t host_visible_bits = 0;
         for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
            if (mp->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
               host_visible_bits |= (1u << i);
         args->pMemoryResourceProperties->memoryTypeBits = host_visible_bits;
         VkMemoryResourceAllocationSizePropertiesMESA *alloc_size_props =
            vkr_find_struct(args->pMemoryResourceProperties->pNext,
                            VK_STRUCTURE_TYPE_MEMORY_RESOURCE_ALLOCATION_SIZE_PROPERTIES_MESA);
         if (alloc_size_props)
            alloc_size_props->allocationSize = res->size;
         args->ret = VK_SUCCESS;
         vkr_log("gkvm: query res %u fd_type=%d -> host-visible memoryTypeBits=0x%x "
                 "(IOSurface id=%u; dmabuf-only gate bypassed)",
                 args->resourceId, (int)res->fd_type, host_visible_bits, res->iosurface_id);
         return;
      }
      /* no host pointer resolved: fall through to the dmabuf path (a genuine dmabuf resource, e.g.
       * the udmabuf debug allocator, still queries correctly). */
   }
#endif /* __APPLE__ */

   if (res->fd_type != VIRGL_RESOURCE_FD_DMABUF) {
      args->ret = VK_ERROR_INVALID_EXTERNAL_HANDLE;
      return;
   }

   static const VkExternalMemoryHandleTypeFlagBits handle_type =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
   VkMemoryFdPropertiesKHR mem_fd_props = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
      .pNext = NULL,
      .memoryTypeBits = 0,
   };
   vn_replace_vkGetMemoryResourcePropertiesMESA_args_handle(args);
   args->ret =
      vk->GetMemoryFdPropertiesKHR(args->device, handle_type, res->u.fd, &mem_fd_props);
   if (args->ret != VK_SUCCESS)
      return;

   args->pMemoryResourceProperties->memoryTypeBits = mem_fd_props.memoryTypeBits;

   VkMemoryResourceAllocationSizePropertiesMESA *alloc_size_props =
      vkr_find_struct(args->pMemoryResourceProperties->pNext,
                      VK_STRUCTURE_TYPE_MEMORY_RESOURCE_ALLOCATION_SIZE_PROPERTIES_MESA);
   if (alloc_size_props)
      alloc_size_props->allocationSize = res->size;
}

void
vkr_context_init_device_memory_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkAllocateMemory = vkr_dispatch_vkAllocateMemory;
   dispatch->dispatch_vkFreeMemory = vkr_dispatch_vkFreeMemory;
   dispatch->dispatch_vkMapMemory = NULL;
   dispatch->dispatch_vkUnmapMemory = NULL;
   dispatch->dispatch_vkFlushMappedMemoryRanges = NULL;
   dispatch->dispatch_vkInvalidateMappedMemoryRanges = NULL;
   dispatch->dispatch_vkGetDeviceMemoryCommitment =
      vkr_dispatch_vkGetDeviceMemoryCommitment;
   dispatch->dispatch_vkGetDeviceMemoryOpaqueCaptureAddress =
      vkr_dispatch_vkGetDeviceMemoryOpaqueCaptureAddress;

   dispatch->dispatch_vkGetMemoryResourcePropertiesMESA =
      vkr_dispatch_vkGetMemoryResourcePropertiesMESA;
}

void
vkr_device_memory_release(struct vkr_device_memory *mem)
{
   /* gkvm #28: a HOST_VISIBLE blob shared MoltenVK's own vkMapMemory pointer (no mtl_shm carrier).
    * We do NOT vkUnmapMemory it here: vkFreeMemory implicitly unmaps a mapped memory (Vulkan
    * spec), and both teardown paths call FreeMemory. In the device-destroy path FreeMemory runs
    * BEFORE this release (vkr_device.c) — an explicit unmap here would hit already-freed memory
    * (a MoltenVK use-after-free / SIGSEGV). The borrowed pointer in the resource layer is only
    * read while the guest holds the mapping; by free time the guest side is already gone. */
   vkr_mtl_shm_free(mem->mtl_shm);
   vkr_mtl_iosurface_release_ref(mem->imported_iosurface);
   if (mem->gbm_bo)
      vkr_gbm_bo_destroy(mem->gbm_bo);
   if (mem->udmabuf_fd >= 0)
      close(mem->udmabuf_fd);
}

bool
vkr_device_memory_export_blob(struct vkr_device_memory *mem,
                              uint64_t blob_size,
                              uint32_t blob_flags,
                              struct virgl_context_blob *out_blob)
{
   TRACE_FUNC();

   /* a memory can only be exported once; we don't want two resources to point
    * to the same storage.
    */
   if (mem->exported) {
      vkr_log_error("mem has been exported");
      return false;
   }

   /* macOS: a Metal-buffer-over-POSIX-shm carrier is always host-mappable shared memory, so
    * it can back the blob regardless of the VkDeviceMemory's own property flags. This covers
    * both the HOST_VISIBLE path and the device-local scanout carrier (gkvm #30): the latter
    * is bound to an IOSurface-backed image and the SHM bytes are never the scanout pixels —
    * present resolves resource -> memory -> IOSurface (vkr_device_memory_get_iosurface_id);
    * the SHM only exists so the proxy/rutabaga resource creation has a real fd to pass. */
   if (mem->mtl_shm && mem->mtl_shm->shm_fd >= 0) {
      mem->exported = true;
      *out_blob = (struct virgl_context_blob){
         .type = VIRGL_RESOURCE_FD_SHM,
         .u.fd = os_dupfd_cloexec(mem->mtl_shm->shm_fd),
         .map_info = VIRGL_RENDERER_MAP_CACHE_CACHED,
      };
#ifdef __APPLE__
      /* Carry the bound scanout image's IOSurface id so SET_SCANOUT_BLOB presents it
       * zero-copy instead of reading the SHM carrier (see vkr_image.c bind linkage). */
      if (mem->mtl_iosurface)
         out_blob->iosurface_id = ((const struct vkr_mtl_iosurface *)mem->mtl_iosurface)->id;
#endif
      if (out_blob->u.fd < 0)
         vkr_log_error("mtl_shm carrier fd dup failed (%s)", strerror(errno));
      return out_blob->u.fd >= 0;
   }

   uint32_t map_info = VIRGL_RENDERER_MAP_CACHE_NONE;
   if (blob_flags & VIRGL_RENDERER_BLOB_FLAG_USE_MAPPABLE) {
      const bool visible = mem->property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      const bool coherent = mem->property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      const bool cached = mem->property_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      if (!visible) {
         vkr_log_error("mem cannot support mappable blob");
         return false;
      }

      /* XXX guessed */
      map_info = (coherent && cached) ? VIRGL_RENDERER_MAP_CACHE_CACHED
                                      : VIRGL_RENDERER_MAP_CACHE_WC;
   }

   const bool can_export_dma_buf = mem->valid_fd_types & (1 << VIRGL_RESOURCE_FD_DMABUF);
   const bool can_export_opaque = mem->valid_fd_types & (1 << VIRGL_RESOURCE_FD_OPAQUE);
   enum virgl_resource_fd_type fd_type;
   VkExternalMemoryHandleTypeFlagBits handle_type;
   struct virgl_resource_vulkan_info vulkan_info;
   if (blob_flags & VIRGL_RENDERER_BLOB_FLAG_USE_CROSS_DEVICE) {
      if (!can_export_dma_buf) {
         vkr_log_error("mem cannot export to dma_buf for cross device blob sharing");
         return false;
      }
      fd_type = VIRGL_RESOURCE_FD_DMABUF;
      handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
   } else if (can_export_dma_buf) {
      /* prefer dmabuf for easier mapping? */
      fd_type = VIRGL_RESOURCE_FD_DMABUF;
      handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
   } else if (can_export_opaque) {
      /* prefer opaque for performance? */
      fd_type = VIRGL_RESOURCE_FD_OPAQUE;
      handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

      STATIC_ASSERT(sizeof(vulkan_info.device_uuid) == VK_UUID_SIZE);
      STATIC_ASSERT(sizeof(vulkan_info.driver_uuid) == VK_UUID_SIZE);

      const VkPhysicalDeviceIDProperties *id_props =
         &mem->device->physical_device->id_properties;
      memcpy(vulkan_info.device_uuid, id_props->deviceUUID, VK_UUID_SIZE);
      memcpy(vulkan_info.driver_uuid, id_props->driverUUID, VK_UUID_SIZE);

      vulkan_info.allocation_size = mem->allocation_size;
      vulkan_info.memory_type_index = mem->memory_type_index;
   } else {
#ifdef __APPLE__
      /* gkvm #28: HOST_VISIBLE memory has no fd to export on macOS (MoltenVK uses Metal handles,
       * not opaque/dma_buf fds), but it CAN be shared by pointer. vkMapMemory hands back
       * mtlBuffer.contents — the exact bytes the GPU reads/writes — which we carry in
       * blob.map_ptr; the VMM hv_vm_maps THAT pointer into the guest (one mapping, coherent).
       * The mapping is owned by this VkDeviceMemory and released by vkUnmapMemory in
       * vkr_device_memory_release (see mem->exported). This is the krunkit (slp) host-visible
       * path; upstream 1.3.0 dropped the blob.map_ptr field, re-added in virgl_context.h. */
      if (mem->property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
         struct vn_device_proc_table *vk = &mem->device->proc_table;
         void *ptr = NULL;
         VkResult ret = vk->MapMemory(mem->device->base.handle.device,
                                      mem->base.handle.device_memory, 0, mem->allocation_size, 0,
                                      &ptr);
         if (ret != VK_SUCCESS || !ptr) {
            vkr_log_error("gkvm #28: host-visible vkMapMemory for blob export failed (vk ret %d)", ret);
            return false;
         }

         mem->exported = true;
         /* MoltenVK HOST_VISIBLE memory is a Shared MTLBuffer = write-back cacheable + coherent,
          * so the guest maps it CACHED. Hardcode CACHED (never NONE) like the old mtl_shm carrier
          * path: the computed `map_info` above is only set when blob_flags has USE_MAPPABLE, and a
          * NONE map_info makes virgl_renderer_resource_get_map_info return -EINVAL, which the VMM
          * (resource_map_blob -> rutabaga.map_info) reports as a failed guest vkMapMemory (#28). */
         *out_blob = (struct virgl_context_blob){
            /* OPAQUE_HANDLE: a no-fd resource vehicle (create_from_fd asserts fd>=0). The handle
             * is never resolved — map/get_map_ptr short-circuit on blob.map_ptr. */
            .type = VIRGL_RESOURCE_OPAQUE_HANDLE,
            .u.opaque_handle = 0,
            .map_info = VIRGL_RENDERER_MAP_CACHE_CACHED,
            .map_ptr = (uint64_t)(uintptr_t)ptr,
         };
         vkr_log("gkvm #28: host-visible blob via vkMapMemory ptr=%p size=%llu (no shm carrier)",
                 ptr, (unsigned long long)mem->allocation_size);
         return true;
      }
#endif
      vkr_log_error("mem is not exportable");
      return false;
   }

   int fd;
   if (mem->udmabuf_fd >= 0) {
      fd = os_dupfd_cloexec(mem->udmabuf_fd);
      if (fd < 0) {
         vkr_log_error("mem udmabuf fd dup failed (%s)", strerror(errno));
         return false;
      }
   } else if (mem->gbm_bo) {
      assert(handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
      assert(can_export_dma_buf && !can_export_opaque);

      fd = vkr_gbm_bo_get_fd(mem->gbm_bo);
      if (fd < 0) {
         vkr_log_error("mem gbm bo export failed (ret %d)", fd);
         return false;
      }
   } else {
      struct vn_device_proc_table *vk = &mem->device->proc_table;
      const VkMemoryGetFdInfoKHR fd_info = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
         .memory = mem->base.handle.device_memory,
         .handleType = handle_type,
      };
      VkResult ret = vk->GetMemoryFdKHR(mem->device->base.handle.device, &fd_info, &fd);
      if (ret != VK_SUCCESS) {
         vkr_log_error("mem fd export failed (vk ret %d)", ret);
         return false;
      }
   }

   if (fd_type == VIRGL_RESOURCE_FD_DMABUF) {
      const off_t dma_buf_size = lseek(fd, 0, SEEK_END);
      if (dma_buf_size < 0 || (uint64_t)dma_buf_size < blob_size) {
         vkr_log_error("mem dma_buf_size %lld < blob_size %" PRIu64, (long long)dma_buf_size,
                 blob_size);
         close(fd);
         return false;
      }
   }

   mem->exported = true;

   *out_blob = (struct virgl_context_blob){
      .type = fd_type,
      .u.fd = fd,
      .map_info = map_info,
      .vulkan_info = vulkan_info,
   };

   return true;
}
