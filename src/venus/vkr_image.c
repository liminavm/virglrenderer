/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_image.h"

#include "vkr_device_memory.h"
#include "vkr_image_gen.h"
#include "vkr_physical_device.h"

#ifdef __APPLE__
#include "venus-protocol/vulkan_metal.h"
#include "vkr_metal_helpers.h"

/* MTLPixelFormat values (Metal/MTLPixelFormat.h) used for IOSurface backing. */
#define GKVM_MTLPixelFormatRGBA8Unorm 70
#define GKVM_MTLPixelFormatBGRA8Unorm 80

/* Map a VkFormat to (MTLPixelFormat, IOSurface FourCC, bytes/pixel) for an IOSurface-backed
 * scanout image. Returns false for formats we don't back (caller forwards unchanged). The
 * UNORM base also serves its sRGB sibling (MUTABLE_FORMAT view list). */
static bool
gkvm_vkformat_to_iosurface(VkFormat fmt, uint32_t *mtl, uint32_t *fourcc, uint32_t *bpe)
{
   switch (fmt) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      *mtl = GKVM_MTLPixelFormatBGRA8Unorm;
      *fourcc = (uint32_t)'BGRA';
      *bpe = 4;
      return true;
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
      *mtl = GKVM_MTLPixelFormatRGBA8Unorm;
      *fourcc = (uint32_t)'RGBA';
      *bpe = 4;
      return true;
   default:
      return false;
   }
}
#endif /* __APPLE__ */

static void
vkr_dispatch_vkCreateImage(struct vn_dispatch_context *dispatch,
                           struct vn_command_vkCreateImage *args)
{
   const VkImageCreateInfo *ci = args->pCreateInfo;

   /* XXX If VkExternalMemoryImageCreateInfo is chained by the app, all is
    * good.  If it is not chained, we might still bind an external memory to
    * the image, because vkr_dispatch_vkAllocateMemory makes any HOST_VISIBLE
    * memory external.  That is a spec violation.
    *
    * The discussions in vkr_dispatch_vkCreateBuffer are applicable to both
    * buffers and images.  Additionally, drivers usually use
    * VkExternalMemoryImageCreateInfo to pick a well-defined image layout for
    * interoperability with foreign queues.  However, a well-defined layout
    * might not exist for some images.  When it does, it might still require a
    * dedicated allocation or might have a degraded performance.
    *
    * On the other hand, binding an external memory to an image created
    * without VkExternalMemoryImageCreateInfo usually works.  Yes, it will
    * explode if the external memory is accessed by foreign queues due to the
    * lack of a well-defined image layout.  But we never end up in that
    * situation because the app does not consider the memory external.
    */

#ifdef __APPLE__
   /* gkvm fix A (tier-2 crossing B/D): MoltenVK can't honor a guest "external" image
    * (VkExternalMemoryImageCreateInfo handleTypes != 0) — host vkCreateImage returns
    * VK_ERROR_FEATURE_NOT_PRESENT (the #30 scanout failure). Back such an image with an
    * IOSurface instead, via VK_EXT_metal_objects: chain VkImportMetalIOSurfaceInfoEXT so
    * MoltenVK calls MVKImage::useIOSurface() and builds the image's MTLTexture *from the
    * IOSurface with its own correct descriptor* (MVKImagePlane::getMTLTexture, the
    * `_image->_ioSurface` branch). This is more robust than importing a raw MTLTexture as the
    * bound memory (the `dvcMem->_mtlTexture` branch): that path uses our texture verbatim and
    * any usage/format mismatch silently no-ops the render, leaving the IOSurface untouched
    * (proven: a magenta-prefilled scanout surface stayed pure magenta — the GPU never wrote
    * it). The IOSurface is global, so the supervisor presents it zero-copy. The dedicated
    * bound memory is still allocated (the guest binds it + exports it as the scanout blob) but
    * its bytes are unused for pixels — getMTLTexture uses the IOSurface. Only formats we can
    * map are backed; others forward unchanged. */
   struct vkr_mtl_iosurface *gkvm_surf = NULL;
   bool gkvm_kk_linear = false;
   uint32_t gkvm_mtl = 0, gkvm_fourcc = 0, gkvm_bpe = 0;
   /* Captured PRE-create: vkr_image_create_and_add replaces args->device with the raw
    * driver handle, so vkr_device_from_handle(args->device) is INVALID afterwards. */
   struct vkr_device *gkvm_dev = NULL;
   VkImportMetalIOSurfaceInfoEXT gkvm_io_import = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_METAL_IO_SURFACE_INFO_EXT,
   };
   if (ci && ci->pNext) {
      const VkExternalMemoryImageCreateInfo *ext = NULL;
      for (const VkBaseInStructure *s = ci->pNext; s; s = s->pNext) {
         if (s->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO) {
            ext = (const VkExternalMemoryImageCreateInfo *)s;
            break;
         }
      }
      if (ext && ext->handleTypes) {
         /* No macOS driver honors fd-flavored external images (MoltenVK tolerates the
          * structs; KosmicKrisp rejects them at create) — vkr implements the export
          * contract itself. ALWAYS drop the external-memory / DRM-format-modifier
          * structs and normalize DRM_FORMAT_MODIFIER tiling to OPTIMAL; scanout-capable
          * formats additionally get IOSurface backing below. */
         VkImageCreateInfo *mci = (VkImageCreateInfo *)ci;
         VkBaseInStructure *prev = (VkBaseInStructure *)mci;
         for (VkBaseInStructure *s = (VkBaseInStructure *)mci->pNext; s;
              s = (VkBaseInStructure *)prev->pNext) {
            const int t = (int)s->sType;
            if (t == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO ||
                t == VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT ||
                t == VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT)
               prev->pNext = s->pNext; /* unlink */
            else
               prev = s;
         }
         if (mci->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT)
            mci->tiling = VK_IMAGE_TILING_OPTIMAL;

         if (gkvm_vkformat_to_iosurface(ci->format, &gkvm_mtl, &gkvm_fourcc, &gkvm_bpe)) {
            struct vkr_device *dev = vkr_device_from_handle(args->device);
            gkvm_dev = dev;
            if (dev->physical_device->EXT_metal_objects) {
               /* MoltenVK: chain VkImportMetalIOSurfaceInfoEXT so the driver backs this
                * image with our global IOSurface (useIOSurface) — render lands in the
                * surface, presented zero-copy. */
               gkvm_surf =
                  vkr_mtl_iosurface_alloc(dev->mtl_device, ci->extent.width,
                                          ci->extent.height, gkvm_mtl, gkvm_fourcc,
                                          gkvm_bpe, 0);
               if (gkvm_surf) {
                  gkvm_io_import.ioSurface = (IOSurfaceRef)gkvm_surf->io_surface;
                  gkvm_io_import.pNext = mci->pNext;
                  mci->pNext = &gkvm_io_import;
               }
            } else {
               /* KosmicKrisp (no VK_EXT_metal_objects): force LINEAR tiling — KK creates
                * linear-image MTLTextures from the bound memory's MTLBuffer, and the bound
                * memory will be a host-pointer import of the IOSurface bytes
                * (vkr_device_memory.c) — render lands in the surface, presented zero-copy.
                * The IOSurface is allocated AFTER create, with the driver's linear rowPitch
                * (vkGetImageSubresourceLayout). */
               mci->tiling = VK_IMAGE_TILING_LINEAR;
               /* INPUT_ATTACHMENT usage makes KK promote the image layout to
                * 2DArray (vk_image_to_mtl_texture_type) — but Metal buffer-backed
                * linear textures must be plain 2D, and a layout/texture array-ness
                * mismatch makes render passes silently drop every draw (clears
                * still land). zink only adds the bit speculatively (fb-fetch);
                * scanout buffers are never fb-fetched. */
               mci->usage &= ~VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
               gkvm_kk_linear = true;
            }
         }
      }
   }
#endif /* __APPLE__ */

   struct vkr_image *gkvm_obj = vkr_image_create_and_add(dispatch->data, args);

#ifdef __APPLE__
   if (gkvm_surf) {
      if (gkvm_obj && args->ret == VK_SUCCESS) {
         gkvm_obj->mtl_iosurface = gkvm_surf;
      } else {
         vkr_mtl_iosurface_free(gkvm_surf);
      }
      vkr_log("IOSurface-backed scanout image %ux%u id=%u -> ret=%d", ci->extent.width,
              ci->extent.height, gkvm_surf->id, args->ret);
   } else if (gkvm_kk_linear && gkvm_dev && gkvm_obj && args->ret == VK_SUCCESS) {
      /* KosmicKrisp path: image is LINEAR; allocate the IOSurface with the driver's
       * rowPitch so the surface bytes can directly back the image memory. */
      struct vkr_device *dev = gkvm_dev;
      struct vn_device_proc_table *vk = &dev->proc_table;
      VkImageSubresource subres = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      VkSubresourceLayout layout = { 0 };
      vk->GetImageSubresourceLayout(dev->base.handle.device, gkvm_obj->base.handle.image,
                                    &subres, &layout);
      struct vkr_mtl_iosurface *gkvm_kk_surf = NULL;
      if (layout.rowPitch && layout.rowPitch == (uint32_t)layout.rowPitch) {
         gkvm_kk_surf = vkr_mtl_iosurface_alloc(dev->mtl_device, ci->extent.width,
                                                ci->extent.height, gkvm_mtl, gkvm_fourcc,
                                                gkvm_bpe, (uint32_t)layout.rowPitch);
         /* If IOSurface overrode the pitch (alignment minimums), the bytes can't back the
          * image — drop the surface rather than scan out sheared pixels. */
         if (gkvm_kk_surf && gkvm_kk_surf->bytes_per_row != (uint32_t)layout.rowPitch) {
            vkr_log("KK scanout: IOSurface pitch %u != image rowPitch %u — no zero-copy",
                    gkvm_kk_surf->bytes_per_row, (uint32_t)layout.rowPitch);
            vkr_mtl_iosurface_free(gkvm_kk_surf);
            gkvm_kk_surf = NULL;
         }
      }
      gkvm_obj->mtl_iosurface = gkvm_kk_surf;
      vkr_log("KK linear scanout image %ux%u rowPitch=%u -> IOSurface id=%u bpr=%u",
              ci->extent.width, ci->extent.height, (uint32_t)layout.rowPitch,
              gkvm_kk_surf ? gkvm_kk_surf->id : 0,
              gkvm_kk_surf ? gkvm_kk_surf->bytes_per_row : 0);
   }
#else
   (void)gkvm_obj;
#endif
}

static void
vkr_dispatch_vkDestroyImage(struct vn_dispatch_context *dispatch,
                            struct vn_command_vkDestroyImage *args)
{
#ifdef __APPLE__
   struct vkr_image *obj = vkr_image_from_handle(args->image);
   if (obj && obj->mtl_iosurface) {
      vkr_mtl_iosurface_free(obj->mtl_iosurface);
      obj->mtl_iosurface = NULL;
   }
#endif
   vkr_image_destroy_and_remove(dispatch->data, args);
}

static void
vkr_dispatch_vkGetImageMemoryRequirements(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetImageMemoryRequirements *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetImageMemoryRequirements_args_handle(args);
   vk->GetImageMemoryRequirements(args->device, args->image, args->pMemoryRequirements);
}

static void
vkr_dispatch_vkGetImageMemoryRequirements2(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetImageMemoryRequirements2 *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

#ifdef __APPLE__
   /* gkvm KK scanout: capture before vn_replace swaps in the raw driver handle. */
   struct vkr_image *gkvm_img = args->pInfo ? vkr_image_from_handle(args->pInfo->image) : NULL;
#endif

   vn_replace_vkGetImageMemoryRequirements2_args_handle(args);
   vk->GetImageMemoryRequirements2(args->device, args->pInfo, args->pMemoryRequirements);

#ifdef __APPLE__
   /* gkvm KK scanout: the IOSurface bytes back this image's memory via a host-pointer
    * import at vkAllocateMemory — but that path can only find the image through
    * VkMemoryDedicatedAllocateInfo. zink only chains it when the driver asks
    * (prefers/requiresDedicatedAllocation, zink_resource.c), and KosmicKrisp doesn't.
    * Force it for IOSurface-backed scanout images so the allocation is dedicated and
    * the import can fire. */
   if (gkvm_img && gkvm_img->mtl_iosurface && args->pMemoryRequirements) {
      VkMemoryDedicatedRequirements *ded = vkr_find_struct(
         args->pMemoryRequirements->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS);
      if (ded) {
         ded->prefersDedicatedAllocation = VK_TRUE;
         ded->requiresDedicatedAllocation = VK_TRUE;
      }
   }
#endif
}

static void
vkr_dispatch_vkGetImageSparseMemoryRequirements(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetImageSparseMemoryRequirements *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetImageSparseMemoryRequirements_args_handle(args);
   vk->GetImageSparseMemoryRequirements(args->device, args->image,
                                        args->pSparseMemoryRequirementCount,
                                        args->pSparseMemoryRequirements);
}

static void
vkr_dispatch_vkGetImageSparseMemoryRequirements2(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetImageSparseMemoryRequirements2 *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetImageSparseMemoryRequirements2_args_handle(args);
   vk->GetImageSparseMemoryRequirements2(args->device, args->pInfo,
                                         args->pSparseMemoryRequirementCount,
                                         args->pSparseMemoryRequirements);
}

static void
vkr_dispatch_vkBindImageMemory(UNUSED struct vn_dispatch_context *dispatch,
                               struct vn_command_vkBindImageMemory *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkBindImageMemory_args_handle(args);
   args->ret =
      vk->BindImageMemory(args->device, args->image, args->memory, args->memoryOffset);
}

static void
vkr_dispatch_vkBindImageMemory2(UNUSED struct vn_dispatch_context *dispatch,
                                struct vn_command_vkBindImageMemory2 *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

#ifdef __APPLE__
   /* gkvm tier-2 (#30 seated scanout): when an IOSurface-backed (fix A) image is bound to a
    * memory, record the IOSurface on the memory so the scanout present can resolve
    * resource -> memory -> IOSurface id zero-copy (the memory is what the guest later
    * exports as the KMS scanout blob; see tier2-iosurface-zerocopy-present.md part-(b)).
    * Done BEFORE handle replacement so the venus object lookups still resolve. */
   for (uint32_t i = 0; i < args->bindInfoCount; i++) {
      const VkBindImageMemoryInfo *b = &args->pBindInfos[i];
      struct vkr_image *img = vkr_image_from_handle(b->image);
      struct vkr_device_memory *mem = vkr_device_memory_from_handle(b->memory);
      if (img && img->mtl_iosurface && mem) {
         mem->mtl_iosurface = img->mtl_iosurface;
         vkr_log("gkvm: scanout bind image=%llu mem=%llu -> IOSurface linked",
                 (unsigned long long)b->image, (unsigned long long)b->memory);
      }
   }
#endif

   vn_replace_vkBindImageMemory2_args_handle(args);
   args->ret = vk->BindImageMemory2(args->device, args->bindInfoCount, args->pBindInfos);
}

static void
vkr_dispatch_vkGetImageSubresourceLayout(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetImageSubresourceLayout *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetImageSubresourceLayout_args_handle(args);
   vk->GetImageSubresourceLayout(args->device, args->image, args->pSubresource,
                                 args->pLayout);
}

static void
vkr_dispatch_vkGetImageSubresourceLayout2(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetImageSubresourceLayout2 *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetImageSubresourceLayout2_args_handle(args);
   vk->GetImageSubresourceLayout2(args->device, args->image, args->pSubresource,
                                  args->pLayout);
}

static void
vkr_dispatch_vkGetDeviceImageSubresourceLayout(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetDeviceImageSubresourceLayout *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetDeviceImageSubresourceLayout_args_handle(args);
   vk->GetDeviceImageSubresourceLayout(args->device, args->pInfo, args->pLayout);
}

static void
vkr_dispatch_vkGetImageDrmFormatModifierPropertiesEXT(
   UNUSED struct vn_dispatch_context *dispatch,
   struct vn_command_vkGetImageDrmFormatModifierPropertiesEXT *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetImageDrmFormatModifierPropertiesEXT_args_handle(args);
   args->ret = vk->GetImageDrmFormatModifierPropertiesEXT(args->device, args->image,
                                                          args->pProperties);
}

static void
vkr_dispatch_vkCreateImageView(struct vn_dispatch_context *dispatch,
                               struct vn_command_vkCreateImageView *args)
{
   vkr_image_view_create_and_add(dispatch->data, args);
}

static void
vkr_dispatch_vkDestroyImageView(struct vn_dispatch_context *dispatch,
                                struct vn_command_vkDestroyImageView *args)
{
   vkr_image_view_destroy_and_remove(dispatch->data, args);
}

static void
vkr_dispatch_vkCreateSampler(struct vn_dispatch_context *dispatch,
                             struct vn_command_vkCreateSampler *args)
{
   vkr_sampler_create_and_add(dispatch->data, args);
}

static void
vkr_dispatch_vkDestroySampler(struct vn_dispatch_context *dispatch,
                              struct vn_command_vkDestroySampler *args)
{
   vkr_sampler_destroy_and_remove(dispatch->data, args);
}

static void
vkr_dispatch_vkCreateSamplerYcbcrConversion(
   struct vn_dispatch_context *dispatch,
   struct vn_command_vkCreateSamplerYcbcrConversion *args)
{
   vkr_sampler_ycbcr_conversion_create_and_add(dispatch->data, args);
}

static void
vkr_dispatch_vkDestroySamplerYcbcrConversion(
   struct vn_dispatch_context *dispatch,
   struct vn_command_vkDestroySamplerYcbcrConversion *args)
{
   vkr_sampler_ycbcr_conversion_destroy_and_remove(dispatch->data, args);
}

static void
vkr_dispatch_vkGetDeviceImageMemoryRequirements(
   UNUSED struct vn_dispatch_context *ctx,
   struct vn_command_vkGetDeviceImageMemoryRequirements *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetDeviceImageMemoryRequirements_args_handle(args);
   vk->GetDeviceImageMemoryRequirements(args->device, args->pInfo,
                                        args->pMemoryRequirements);
}

static void
vkr_dispatch_vkGetDeviceImageSparseMemoryRequirements(
   UNUSED struct vn_dispatch_context *ctx,
   struct vn_command_vkGetDeviceImageSparseMemoryRequirements *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   vn_replace_vkGetDeviceImageSparseMemoryRequirements_args_handle(args);
   vk->GetDeviceImageSparseMemoryRequirements(args->device, args->pInfo,
                                              args->pSparseMemoryRequirementCount,
                                              args->pSparseMemoryRequirements);
}

void
vkr_context_init_image_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkCreateImage = vkr_dispatch_vkCreateImage;
   dispatch->dispatch_vkDestroyImage = vkr_dispatch_vkDestroyImage;
   dispatch->dispatch_vkGetImageMemoryRequirements =
      vkr_dispatch_vkGetImageMemoryRequirements;
   dispatch->dispatch_vkGetImageMemoryRequirements2 =
      vkr_dispatch_vkGetImageMemoryRequirements2;
   dispatch->dispatch_vkGetImageSparseMemoryRequirements =
      vkr_dispatch_vkGetImageSparseMemoryRequirements;
   dispatch->dispatch_vkGetImageSparseMemoryRequirements2 =
      vkr_dispatch_vkGetImageSparseMemoryRequirements2;
   dispatch->dispatch_vkBindImageMemory = vkr_dispatch_vkBindImageMemory;
   dispatch->dispatch_vkBindImageMemory2 = vkr_dispatch_vkBindImageMemory2;
   dispatch->dispatch_vkGetImageSubresourceLayout =
      vkr_dispatch_vkGetImageSubresourceLayout;
   dispatch->dispatch_vkGetImageSubresourceLayout2 =
      vkr_dispatch_vkGetImageSubresourceLayout2;
   dispatch->dispatch_vkGetDeviceImageSubresourceLayout =
      vkr_dispatch_vkGetDeviceImageSubresourceLayout;

   dispatch->dispatch_vkGetImageDrmFormatModifierPropertiesEXT =
      vkr_dispatch_vkGetImageDrmFormatModifierPropertiesEXT;
   dispatch->dispatch_vkGetDeviceImageMemoryRequirements =
      vkr_dispatch_vkGetDeviceImageMemoryRequirements;
   dispatch->dispatch_vkGetDeviceImageSparseMemoryRequirements =
      vkr_dispatch_vkGetDeviceImageSparseMemoryRequirements;
}

void
vkr_context_init_image_view_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkCreateImageView = vkr_dispatch_vkCreateImageView;
   dispatch->dispatch_vkDestroyImageView = vkr_dispatch_vkDestroyImageView;
}

void
vkr_context_init_sampler_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkCreateSampler = vkr_dispatch_vkCreateSampler;
   dispatch->dispatch_vkDestroySampler = vkr_dispatch_vkDestroySampler;
}

void
vkr_context_init_sampler_ycbcr_conversion_dispatch(struct vkr_context *ctx)
{
   struct vn_dispatch_context *dispatch = &ctx->dispatch;

   dispatch->dispatch_vkCreateSamplerYcbcrConversion =
      vkr_dispatch_vkCreateSamplerYcbcrConversion;
   dispatch->dispatch_vkDestroySamplerYcbcrConversion =
      vkr_dispatch_vkDestroySamplerYcbcrConversion;
}
