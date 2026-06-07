/*
 * Copyright 2026 Lucas Amaral
 * SPDX-License-Identifier: MIT
 */

#ifdef __APPLE__

#include "vkr_common.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "util/anon_file.h"
#include "venus-protocol/vulkan_metal.h"

#include "vkr_metal_helpers.h"

void *
vkr_metal_get_device(VkDevice vk_device, PFN_vkGetDeviceProcAddr GetDeviceProcAddr)
{
   PFN_vkExportMetalObjectsEXT pfn = (PFN_vkExportMetalObjectsEXT)GetDeviceProcAddr(
      vk_device, "vkExportMetalObjectsEXT");
   if (!pfn)
      return NULL;

   VkExportMetalDeviceInfoEXT device_info = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT,
   };
   VkExportMetalObjectsInfoEXT export_info = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
      .pNext = &device_info,
   };
   pfn(vk_device, &export_info);
   return (void *)device_info.mtlDevice;
}

struct vkr_mtl_shm *
vkr_mtl_shm_alloc(void *mtl_device, uint64_t size)
{
   if (!mtl_device)
      return NULL;

   const size_t page_size = getpagesize();
   const size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

   int shm_fd = os_create_anonymous_file(aligned_size, "vkr-metal-mem");
   if (shm_fd < 0)
      return NULL;

   void *shm_ptr =
      mmap(NULL, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   if (shm_ptr == MAP_FAILED) {
      close(shm_fd);
      return NULL;
   }

   id<MTLDevice> device = (id<MTLDevice>)mtl_device;
   id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:shm_ptr
                                                    length:aligned_size
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
   if (!buffer) {
      munmap(shm_ptr, aligned_size);
      close(shm_fd);
      return NULL;
   }

   struct vkr_mtl_shm *shm = calloc(1, sizeof(*shm));
   if (!shm) {
      CFRelease(buffer);
      munmap(shm_ptr, aligned_size);
      close(shm_fd);
      return NULL;
   }

   shm->shm_fd = shm_fd;
   shm->shm_ptr = shm_ptr;
   shm->shm_size = aligned_size;
   shm->mtl_buffer = (void *)buffer;
   return shm;
}

void
vkr_mtl_shm_free(struct vkr_mtl_shm *shm)
{
   if (!shm)
      return;
   if (shm->mtl_buffer)
      CFRelease(shm->mtl_buffer);
   if (shm->shm_ptr)
      munmap(shm->shm_ptr, shm->shm_size);
   if (shm->shm_fd >= 0)
      close(shm->shm_fd);
   free(shm);
}

struct vkr_mtl_iosurface *
vkr_mtl_iosurface_alloc(void *mtl_device,
                        uint32_t width,
                        uint32_t height,
                        uint32_t mtl_pixel_format,
                        uint32_t iosurface_pixel_format,
                        uint32_t bytes_per_element)
{
   if (!mtl_device || !width || !height)
      return NULL;

   IOSurfaceRef io = NULL;
   id<MTLTexture> tex = nil;

   @autoreleasepool {
      /* kIOSurfaceIsGlobal: the gkvm supervisor (a separate process) looks the surface
       * up by IOSurfaceGetID for present — the existing gkvm-display transport. IOSurface
       * picks/aligns bytesPerRow itself; the IOSurface-backed MTLTexture honors it. */
      NSDictionary *props = @{
         (id)kIOSurfaceWidth : @(width),
         (id)kIOSurfaceHeight : @(height),
         (id)kIOSurfaceBytesPerElement : @(bytes_per_element),
         (id)kIOSurfacePixelFormat : @(iosurface_pixel_format),
         (id)kIOSurfaceIsGlobal : @YES,
      };
      io = IOSurfaceCreate((__bridge CFDictionaryRef)props);
      if (!io)
         return NULL;

      id<MTLDevice> device = (id<MTLDevice>)mtl_device;
      MTLTextureDescriptor *td =
         [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)mtl_pixel_format
                                                            width:width
                                                           height:height
                                                        mipmapped:NO];
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
      td.storageMode = MTLStorageModeShared;
      tex = [device newTextureWithDescriptor:td iosurface:io plane:0];
      if (!tex) {
         CFRelease(io);
         return NULL;
      }
   }

   struct vkr_mtl_iosurface *surf = calloc(1, sizeof(*surf));
   if (!surf) {
      CFRelease(tex);
      CFRelease(io);
      return NULL;
   }

   surf->io_surface = (void *)io;  /* +1 from IOSurfaceCreate */
   surf->mtl_texture = (void *)tex; /* +1 from -newTexture... */
   surf->id = IOSurfaceGetID(io);
   surf->width = width;
   surf->height = height;
   surf->bytes_per_row = (uint32_t)IOSurfaceGetBytesPerRow(io);
   return surf;
}

void
vkr_mtl_iosurface_free(struct vkr_mtl_iosurface *surf)
{
   if (!surf)
      return;
   if (surf->mtl_texture)
      CFRelease(surf->mtl_texture);
   if (surf->io_surface)
      CFRelease(surf->io_surface);
   free(surf);
}

#endif /* __APPLE__ */
