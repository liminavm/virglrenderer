/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_image.h"

#include "vkr_device_memory.h"
#include "vkr_image_gen.h"
#include "vkr_physical_device.h"

#ifdef __APPLE__
#include "vulkan/vulkan_metal.h"
#include "vkr_metal_helpers.h"

/* MTLPixelFormat values (Metal/MTLPixelFormat.h) used for IOSurface backing. */
#define LIMINA_MTLPixelFormatRGBA8Unorm 70
#define LIMINA_MTLPixelFormatRGBA8Unorm_sRGB 71
#define LIMINA_MTLPixelFormatBGRA8Unorm 80
#define LIMINA_MTLPixelFormatBGRA8Unorm_sRGB 81

/* EXACT MTLPixelFormat for a VkFormat — sRGB stays sRGB.
 *
 * This is deliberately NOT what limina_vkformat_to_iosurface() returns. That one folds sRGB onto
 * its UNORM base, which is right for the MoltenVK path (MVK builds its own view over the
 * IOSurface, from the image's real format, via a MUTABLE_FORMAT view list). It is fatal for the
 * MTLTEXTURE-import path, where KosmicKrisp adopts our texture *verbatim*: an sRGB image over an
 * UNORM texture is a format mismatch that fails the bind — observed 2026-08-04, vkmark's
 * R8G8B8A8_SRGB swapchain (image fmt 71 vs texture fmt 70). Writes through an sRGB texture also
 * carry sRGB encoding, which is exactly what the guest asked for and what the forced-LINEAR path
 * already did (KK built its own texture from the image's format). The IOSurface's fourcc is
 * unchanged either way, so presentation is unaffected. */
bool
limina_vkformat_to_mtl_exact(VkFormat fmt, uint32_t *mtl)
{
   switch (fmt) {
   case VK_FORMAT_B8G8R8A8_UNORM:
      *mtl = LIMINA_MTLPixelFormatBGRA8Unorm;
      return true;
   case VK_FORMAT_B8G8R8A8_SRGB:
      *mtl = LIMINA_MTLPixelFormatBGRA8Unorm_sRGB;
      return true;
   case VK_FORMAT_R8G8B8A8_UNORM:
      *mtl = LIMINA_MTLPixelFormatRGBA8Unorm;
      return true;
   case VK_FORMAT_R8G8B8A8_SRGB:
      *mtl = LIMINA_MTLPixelFormatRGBA8Unorm_sRGB;
      return true;
   default:
      return false;
   }
}

/* Map a VkFormat to (MTLPixelFormat, IOSurface FourCC, bytes/pixel) for an IOSurface-backed
 * scanout image. Returns false for formats we don't back (caller forwards unchanged). The
 * UNORM base also serves its sRGB sibling (MUTABLE_FORMAT view list). */
static bool
limina_vkformat_to_iosurface(VkFormat fmt, uint32_t *mtl, uint32_t *fourcc, uint32_t *bpe)
{
   switch (fmt) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      *mtl = LIMINA_MTLPixelFormatBGRA8Unorm;
      *fourcc = (uint32_t)'BGRA';
      *bpe = 4;
      return true;
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
      *mtl = LIMINA_MTLPixelFormatRGBA8Unorm;
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
   /* limina fix A (tier-2 crossing B/D): MoltenVK can't honor a guest "external" image
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
   struct vkr_mtl_iosurface *limina_surf = NULL;
   bool limina_kk_linear = false;
   uint32_t limina_mtl = 0, limina_fourcc = 0, limina_bpe = 0;
   /* Captured PRE-create: vkr_image_create_and_add replaces args->device with the raw
    * driver handle, so vkr_device_from_handle(args->device) is INVALID afterwards. */
   struct vkr_device *limina_dev = NULL;
   VkImportMetalIOSurfaceInfoEXT limina_io_import = {
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
         /* limina probe (2026-08-04): venus stamps renderer_handle_type here, so this is a
          * direct read of WHICH branch of vn_physical_device_init_external_memory fired.
          * DMA_BUF (0x200) => upstream's own branch (vkr advertises the extension) and
          * mesa 0010(a) is dead code; OPAQUE_FD (0x2) => still 0010(a)'s else-if. */
         static uint32_t limina_ht_seen;
         if (!(limina_ht_seen & ext->handleTypes)) {
            limina_ht_seen |= ext->handleTypes;
            fprintf(stderr, "[LIMINA-VKR-HT] image external handleTypes=0x%x (%s%s)\n",
                    ext->handleTypes,
                    (ext->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)
                       ? "DMA_BUF " : "",
                    (ext->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
                       ? "OPAQUE_FD" : "");
         }
         /* No macOS driver honors fd-flavored external images (MoltenVK tolerates the
          * structs; KosmicKrisp rejects them at create) — vkr implements the export
          * contract itself. ALWAYS drop the external-memory / DRM-format-modifier
          * structs and normalize DRM_FORMAT_MODIFIER tiling to OPTIMAL; scanout-capable
          * formats additionally get IOSurface backing below. */
         /* An EXPLICIT modifier struct marks an IMPORT (zink passes the exporter's
          * layout); the pixel bytes already live in the EXPORTER's IOSurface and the
          * memory bind aliases them (vkr_device_memory.c host-pointer import) — do NOT
          * allocate a fresh (wrong) IOSurface for such an image. Allocation-side
          * creates use the modifier LIST form. */
         bool limina_is_import = false;
         VkImageCreateInfo *mci = (VkImageCreateInfo *)ci;
         VkBaseInStructure *prev = (VkBaseInStructure *)mci;
         /* LIMINA_VKR_KEEP_MODIFIER_STRUCTS=1 leaves the DRM-format-modifier structs
          * chained (the external-memory one is still unlinked) — the A/B for whether
          * this unlink is load-bearing against a host driver that never advertises
          * VK_EXT_image_drm_format_modifier. */
         static int limina_keep_structs = -1;
         if (limina_keep_structs < 0) {
            const char *e = getenv("LIMINA_VKR_KEEP_MODIFIER_STRUCTS");
            limina_keep_structs = (e && *e == '1') ? 1 : 0;
         }
         for (VkBaseInStructure *s = (VkBaseInStructure *)mci->pNext; s;
              s = (VkBaseInStructure *)prev->pNext) {
            const int t = (int)s->sType;
            const bool is_mod =
               t == VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT ||
               t == VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT;
            if (t == VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT)
               limina_is_import = true;
            if (t == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO ||
                (is_mod && !limina_keep_structs))
               prev->pNext = s->pNext; /* unlink */
            else
               prev = s;
         }
         if (mci->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) {
            /* limina probe (2026-08-04): count how often a guest image actually
             * arrives with modifier tiling, i.e. how much of this normalization is
             * live traffic vs MoltenVK-era dead code. */
            static unsigned limina_mod_hits;
            if (++limina_mod_hits <= 4 || (limina_mod_hits % 256) == 0)
               fprintf(stderr, "[LIMINA-VKRMOD] modifier-tiled guest image #%u %s import=%d\n",
                       limina_mod_hits, limina_is_import ? "EXPLICIT" : "LIST", limina_is_import);
            /* LIMINA_VKR_KEEP_MODIFIER_TILING=1 leaves the modifier tiling in place so
             * the image reaches KK as the guest asked — the A/B that tells us whether
             * kk_image_layout's modifier carve-out is a live guard or dead code. */
            static int limina_keep = -1;
            if (limina_keep < 0) {
               const char *e = getenv("LIMINA_VKR_KEEP_MODIFIER_TILING");
               limina_keep = (e && *e == '1') ? 1 : 0;
            }
            if (!limina_keep)
               mci->tiling = VK_IMAGE_TILING_OPTIMAL;
         }

         if (limina_vkformat_to_iosurface(ci->format, &limina_mtl, &limina_fourcc, &limina_bpe)) {
            struct vkr_device *dev = vkr_device_from_handle(args->device);
            limina_dev = dev;
            if (limina_is_import) {
               /* Import: keep KK's LINEAR/usage normalization below, skip IOSurface
                * allocation (the bound memory aliases the exporter's bytes). */
               if (!dev->physical_device->EXT_metal_objects) {
                  /* …unless the MTLTEXTURE path is on, in which case the bound memory is
                   * the exporter's TEXTURE, not its bytes. Forcing LINEAR here is what
                   * sheared every GTK4 window on 2026-08-04: the exporter wrote in the
                   * texture's layout while this side declared a linear rowPitch. Stay
                   * OPTIMAL so KK's bind can adopt the texture (it refuses LINEAR). */
                  static int limina_mtltex_imp = -1;
                  if (limina_mtltex_imp < 0) {
                     const char *e = getenv("LIMINA_KK_MTLTEXTURE_SCANOUT");
                     limina_mtltex_imp = (e && *e == '1') ? 1 : 0;
                  }
                  if (limina_mtltex_imp)
                     mci->tiling = VK_IMAGE_TILING_OPTIMAL;
                  else
                     mci->tiling = VK_IMAGE_TILING_LINEAR;
                  mci->usage &= ~VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
               }
            } else if (dev->physical_device->EXT_metal_objects) {
               /* MoltenVK: chain VkImportMetalIOSurfaceInfoEXT so the driver backs this
                * image with our global IOSurface (useIOSurface) — render lands in the
                * surface, presented zero-copy. */
               limina_surf =
                  vkr_mtl_iosurface_alloc(dev->mtl_device, ci->extent.width,
                                          ci->extent.height, limina_mtl, limina_fourcc,
                                          limina_bpe, 0);
               if (limina_surf) {
                  limina_io_import.ioSurface = (IOSurfaceRef)limina_surf->io_surface;
                  limina_io_import.pNext = mci->pNext;
                  mci->pNext = &limina_io_import;
               }
            } else {
               /* KosmicKrisp (no VK_EXT_metal_objects): force LINEAR tiling — KK creates
                * linear-image MTLTextures from the bound memory's MTLBuffer, and the bound
                * memory will be a host-pointer import of the IOSurface bytes
                * (vkr_device_memory.c) — render lands in the surface, presented zero-copy.
                * The IOSurface is allocated AFTER create, with the driver's linear rowPitch
                * (vkGetImageSubresourceLayout). */
               /* LIMINA_VKR_NO_KK_FORCE_LINEAR=1 leaves the guest's tiling alone, so the
                * image reaches KK as it was asked for (modifier-tiled, if the guest said
                * so) and kk_image_layout's carve-out becomes reachable. Costs zero-copy:
                * without a defined linear rowPitch no IOSurface can alias the image
                * memory, so the present path falls back to a readback blit. This is the
                * A/B for "is the forced LINEAR still the only way to scan out on KK". */
               static int limina_no_force = -1;
               if (limina_no_force < 0) {
                  const char *e = getenv("LIMINA_VKR_NO_KK_FORCE_LINEAR");
                  limina_no_force = (e && *e == '1') ? 1 : 0;
               }
               /* LIMINA_KK_MTLTEXTURE_SCANOUT=1: the replacement for the forced LINEAR.
                * KosmicKrisp now implements VK_EXT_external_memory_metal's MTLTEXTURE
                * handle type, so instead of contorting the image into a linear layout
                * whose rowPitch an IOSurface can be made to match, we allocate the
                * IOSurface up front and hand KK its texture as the image's memory
                * (vkr_device_memory.c does the import). No pitch to match, so the
                * "IOSurface pitch != image rowPitch -> no zero-copy" failure mode goes
                * away, and the guest's own tiling survives. Kept behind a gate until it
                * is measured against the forced-LINEAR path. */
               static int limina_mtltex = -1;
               if (limina_mtltex < 0) {
                  const char *e = getenv("LIMINA_KK_MTLTEXTURE_SCANOUT");
                  limina_mtltex = (e && *e == '1') ? 1 : 0;
               }
               if (limina_mtltex) {
                  /* OPTIMAL keeps kk_image_layout on the plain-2D, non-linear path that
                   * matches an IOSurface-backed MTLTexture; INPUT_ATTACHMENT would push
                   * the layout type to 2DArray and fail KK's bind-time type check. */
                  mci->tiling = VK_IMAGE_TILING_OPTIMAL;
                  mci->usage &= ~VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
                  /* EXACT format, sRGB included — KK adopts this texture verbatim, so
                   * the UNORM-base folding used for the MoltenVK path would fail the
                   * bind on an sRGB image. */
                  uint32_t limina_mtl_exact = limina_mtl;
                  limina_vkformat_to_mtl_exact(ci->format, &limina_mtl_exact);
                  limina_surf = vkr_mtl_iosurface_alloc(dev->mtl_device, ci->extent.width,
                                                      ci->extent.height, limina_mtl_exact,
                                                      limina_fourcc, limina_bpe, 0);
                  fprintf(stderr,
                          "[LIMINA-VKR-MTLTEX] KK MTLTEXTURE scanout %ux%u -> IOSurface "
                          "id=%u tex=%p\n",
                          ci->extent.width, ci->extent.height,
                          limina_surf ? limina_surf->id : 0,
                          limina_surf ? limina_surf->mtl_texture : NULL);
                  vkr_log("limina: KK MTLTEXTURE scanout %ux%u -> IOSurface id=%u tex=%p",
                          ci->extent.width, ci->extent.height,
                          limina_surf ? limina_surf->id : 0,
                          limina_surf ? limina_surf->mtl_texture : NULL);
                  /* On failure fall through to the forced-LINEAR path below. */
               }
               /* Fall back to forced LINEAR when the import path is off OR its IOSurface
                * allocation failed — never leave the image with neither backing. */
               if ((!limina_mtltex || !limina_surf) && !limina_no_force) {
                  mci->tiling = VK_IMAGE_TILING_LINEAR;
                  /* INPUT_ATTACHMENT usage makes KK promote the image layout to
                   * 2DArray (vk_image_to_mtl_texture_type) — but Metal buffer-backed
                   * linear textures must be plain 2D, and a layout/texture array-ness
                   * mismatch makes render passes silently drop every draw (clears
                   * still land). zink only adds the bit speculatively (fb-fetch);
                   * scanout buffers are never fb-fetched. */
                  mci->usage &= ~VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
                  limina_kk_linear = true;
               } else {
                  vkr_log("LIMINA-NOFORCE: leaving scanout image tiling=%d usage=0x%x",
                          (int)mci->tiling, mci->usage);
               }
            }
         }
      }
   }
#endif /* __APPLE__ */

   struct vkr_image *limina_obj = vkr_image_create_and_add(dispatch->data, args);

#ifdef __APPLE__
   /* Remember the create format for the MTLTEXTURE import path (vkr_device_memory.c):
    * a dedicated import allocation names this image, and the adopted texture's pixel
    * format has to match it exactly. Recorded for EVERY image — the import side has no
    * other way back to the format, and the field is inert otherwise. */
   if (limina_obj && ci)
      limina_obj->limina_vk_format = (uint32_t)ci->format;

   if (limina_surf) {
      if (limina_obj && args->ret == VK_SUCCESS) {
         limina_obj->mtl_iosurface = limina_surf;
      } else {
         vkr_mtl_iosurface_free(limina_surf);
      }
      vkr_log("IOSurface-backed scanout image %ux%u id=%u -> ret=%d", ci->extent.width,
              ci->extent.height, limina_surf->id, args->ret);
   } else if (limina_kk_linear && limina_dev && limina_obj && args->ret == VK_SUCCESS) {
      /* KosmicKrisp path: image is LINEAR; allocate the IOSurface with the driver's
       * rowPitch so the surface bytes can directly back the image memory. */
      struct vkr_device *dev = limina_dev;
      struct vn_device_proc_table *vk = &dev->proc_table;
      VkImageSubresource subres = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      VkSubresourceLayout layout = { 0 };
      vk->GetImageSubresourceLayout(dev->base.handle.device, limina_obj->base.handle.image,
                                    &subres, &layout);
      struct vkr_mtl_iosurface *limina_kk_surf = NULL;
      if (layout.rowPitch && layout.rowPitch == (uint32_t)layout.rowPitch) {
         limina_kk_surf = vkr_mtl_iosurface_alloc(dev->mtl_device, ci->extent.width,
                                                ci->extent.height, limina_mtl, limina_fourcc,
                                                limina_bpe, (uint32_t)layout.rowPitch);
         /* If IOSurface overrode the pitch (alignment minimums), the bytes can't back the
          * image — drop the surface rather than scan out sheared pixels. */
         if (limina_kk_surf && limina_kk_surf->bytes_per_row != (uint32_t)layout.rowPitch) {
            vkr_log("KK scanout: IOSurface pitch %u != image rowPitch %u — no zero-copy",
                    limina_kk_surf->bytes_per_row, (uint32_t)layout.rowPitch);
            vkr_mtl_iosurface_free(limina_kk_surf);
            limina_kk_surf = NULL;
         }
      }
      limina_obj->mtl_iosurface = limina_kk_surf;
      vkr_log("KK linear scanout image %ux%u rowPitch=%u -> IOSurface id=%u bpr=%u",
              ci->extent.width, ci->extent.height, (uint32_t)layout.rowPitch,
              limina_kk_surf ? limina_kk_surf->id : 0,
              limina_kk_surf ? limina_kk_surf->bytes_per_row : 0);
   }
#else
   (void)limina_obj;
#endif
}

void
vkr_image_release(struct vkr_image *img)
{
#ifdef __APPLE__
   if (img && img->mtl_iosurface) {
      vkr_mtl_iosurface_free(img->mtl_iosurface);
      img->mtl_iosurface = NULL;
   }
#else
   (void)img;
#endif
}

static void
vkr_dispatch_vkDestroyImage(struct vn_dispatch_context *dispatch,
                            struct vn_command_vkDestroyImage *args)
{
   vkr_image_release(vkr_image_from_handle(args->image));
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
   /* limina KK scanout: capture before vn_replace swaps in the raw driver handle. */
   struct vkr_image *limina_img = args->pInfo ? vkr_image_from_handle(args->pInfo->image) : NULL;
#endif

   vn_replace_vkGetImageMemoryRequirements2_args_handle(args);
   vk->GetImageMemoryRequirements2(args->device, args->pInfo, args->pMemoryRequirements);

#ifdef __APPLE__
   /* limina KK scanout: the IOSurface bytes back this image's memory via a host-pointer
    * import at vkAllocateMemory — but that path can only find the image through
    * VkMemoryDedicatedAllocateInfo. zink only chains it when the driver asks
    * (prefers/requiresDedicatedAllocation, zink_resource.c), and KosmicKrisp doesn't.
    * Force it for IOSurface-backed scanout images so the allocation is dedicated and
    * the import can fire. */
   if (limina_img && limina_img->mtl_iosurface && args->pMemoryRequirements) {
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
vkr_dispatch_vkBindImageMemory(struct vn_dispatch_context *dispatch,
                               struct vn_command_vkBindImageMemory *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   /* limina journal: key this bind by its image (guest id, pre-replace) */
   const struct vkr_image *img = vkr_image_from_handle(args->image);
   if (img)
      vkr_journal_note_keys(dispatch->data, &img->base.id, 1);

   vn_replace_vkBindImageMemory_args_handle(args);
   args->ret =
      vk->BindImageMemory(args->device, args->image, args->memory, args->memoryOffset);
}

static void
vkr_dispatch_vkBindImageMemory2(struct vn_dispatch_context *dispatch,
                                struct vn_command_vkBindImageMemory2 *args)
{
   struct vkr_device *dev = vkr_device_from_handle(args->device);
   struct vn_device_proc_table *vk = &dev->proc_table;

   /* limina journal: key this bind by its images (guest ids, pre-replace) */
   for (uint32_t i = 0; i < args->bindInfoCount; i++) {
      const struct vkr_image *jimg = vkr_image_from_handle(args->pBindInfos[i].image);
      if (jimg)
         vkr_journal_note_keys(dispatch->data, &jimg->base.id, 1);
   }

#ifdef __APPLE__
   /* limina tier-2 (#30 seated scanout): when an IOSurface-backed (fix A) image is bound to a
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
         vkr_log("limina: scanout bind image=%llu mem=%llu -> IOSurface linked",
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
