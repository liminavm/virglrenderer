/*
 * Copyright 2026 Lucas Amaral
 * SPDX-License-Identifier: MIT
 */

#ifndef VKR_METAL_HELPERS_H
#define VKR_METAL_HELPERS_H

#include <stddef.h>

#include <stdint.h>

/*
 * Metal shared memory (opaque, allocated/freed in vkr_metal_helpers.m)
 */
struct vkr_mtl_shm {
   int shm_fd;
   void *shm_ptr;
   size_t shm_size;
   void *mtl_buffer;
   /* limina host-memory budget (vkr_budget.h): the context this allocation was charged to. */
   uint32_t limina_budget_ctx;
};

/*
 * A GLOBAL IOSurface + an IOSurface-backed MTLTexture, for zero-copy scanout present
 * (limina tier-2 crossings B+D — see docs/design/tier2-iosurface-zerocopy-present.md).
 *
 * The MTLTexture is imported into a venus VkImage (VkImportMetalTextureInfoEXT) so the
 * guest renders straight into the IOSurface. The IOSurface is created kIOSurfaceIsGlobal
 * so the limina supervisor process can IOSurfaceLookup(id) it for present (the existing
 * limina-display "frame <id>" transport) with no copy.  Opaque; alloc/free in the .m.
 */
struct vkr_mtl_iosurface {
   void *io_surface;    /* IOSurfaceRef (retained) */
   void *mtl_texture;   /* id<MTLTexture> (retained), IOSurface-backed */
   uint32_t id;         /* IOSurfaceGetID — the global id the supervisor looks up */
   uint32_t width;
   uint32_t height;
   uint32_t bytes_per_row;
   void *base_addr;     /* IOSurfaceGetBaseAddress — page-aligned host pointer */
   uint64_t alloc_size; /* IOSurfaceGetAllocSize */
   /* limina host-memory budget (vkr_budget.h): the context alloc_size was charged to. */
   uint32_t limina_budget_ctx;
};

#ifdef __APPLE__

#include <stdint.h>

/*
 * Metal helper functions (implemented in vkr_metal_helpers.m)
 */

/* Return the MTLDevice backing a VkDevice.
 * Uses vkExportMetalObjectsEXT to query the device, or NULL on failure.
 */
void *
vkr_metal_get_device(VkDevice vk_device, PFN_vkGetDeviceProcAddr GetDeviceProcAddr);

/* Fallback MTLDevice for drivers without VK_EXT_metal_objects (KosmicKrisp):
 * MTLCreateSystemDefaultDevice — same physical GPU on Apple Silicon. */
void *
vkr_metal_get_system_device(void);

/* Allocate Metal shared memory: create anonymous SHM file, mmap it,
 * wrap as MTLBuffer.  Returns a populated vkr_mtl_shm, or NULL on failure.
 * Caller must free with vkr_mtl_shm_free().
 */
struct vkr_mtl_shm *
vkr_mtl_shm_alloc(void *mtl_device, uint64_t size);

/* Release all resources held by a vkr_mtl_shm and free the struct. */
void
vkr_mtl_shm_free(struct vkr_mtl_shm *shm);

/* Allocate a GLOBAL IOSurface + an IOSurface-backed MTLTexture (StorageModeShared,
 * usage ShaderRead|RenderTarget).  `mtl_pixel_format` is an MTLPixelFormat value,
 * `iosurface_pixel_format` an IOSurface FourCC (e.g. 'BGRA'), `bytes_per_element` the
 * pixel size.  `bytes_per_row` forces the IOSurface row pitch (0 = let IOSurface pick) —
 * used to match a driver's linear-image rowPitch so the surface bytes can back the image
 * memory directly (KosmicKrisp host-pointer import).  Returns a populated
 * vkr_mtl_iosurface (free with vkr_mtl_iosurface_free), or NULL on failure. */
struct vkr_mtl_iosurface *
vkr_mtl_iosurface_alloc(void *mtl_device,
                        uint32_t width,
                        uint32_t height,
                        uint32_t mtl_pixel_format,
                        uint32_t iosurface_pixel_format,
                        uint32_t bytes_per_element,
                        uint32_t bytes_per_row);

/* Release the IOSurface + MTLTexture and free the struct. */
void
vkr_mtl_iosurface_free(struct vkr_mtl_iosurface *surf);

/* Allocate an IOSurface WITHOUT a Metal texture (mtl_texture = NULL) — same scoping,
 * registry, and supervisor Mach-port publication as vkr_mtl_iosurface_alloc. Used by
 * vrend's zero-copy scanout (docs/design/vrend-iosurface-scanout.md): the GL side reaches
 * the pixels through a GL_AMD_pinned_memory PBO over base_addr, no Metal object needed.
 * Free with vkr_mtl_iosurface_free. */
struct vkr_mtl_iosurface *
vkr_mtl_iosurface_alloc_plain(uint32_t width,
                              uint32_t height,
                              uint32_t iosurface_pixel_format,
                              uint32_t bytes_per_element);

/* Field accessors for TUs that must not include this (Vulkan-typed) header — vrend
 * forward-declares these instead. */
uint32_t
vkr_mtl_iosurface_get_id(const struct vkr_mtl_iosurface *surf);
void *
vkr_mtl_iosurface_get_ref(const struct vkr_mtl_iosurface *surf);
void
vkr_mtl_iosurface_get_layout(const struct vkr_mtl_iosurface *surf,
                             void **out_base,
                             uint32_t *out_bytes_per_row,
                             uint64_t *out_alloc_size);

/* Look up a GLOBAL IOSurface by id (cross-context import of a venus-exported window
 * buffer: the importing VkDeviceMemory host-pointer-imports the exporter's pixel
 * bytes).  Returns a RETAINED IOSurfaceRef (release with vkr_mtl_iosurface_release_ref)
 * and its page-aligned base address + page-rounded alloc size, or NULL. */
void *
vkr_mtl_iosurface_lookup(uint32_t id, void **out_base, uint64_t *out_alloc_size);

void
vkr_mtl_iosurface_release_ref(void *io_surface);

/* Create a RETAINED id<MTLTexture> over an EXISTING IOSurface (the import half of the
 * MTLTEXTURE scanout path — the exporter's surface, looked up by id). The descriptor
 * mirrors vkr_mtl_iosurface_alloc's, so KosmicKrisp's superset-usage / exact-format bind
 * check passes. Release with vkr_mtl_texture_release. NULL on failure. */
void *
vkr_mtl_texture_from_iosurface(void *mtl_device, void *io_surface, uint32_t mtl_pixel_format);

/* Format the +1/-1 balance of every host-side reference we take on an IOSurface, into buf.
 * The ledger says whether we allocated and released a surface; it cannot say whether some
 * OTHER reference is keeping the storage alive after our release. Each pair below is a
 * place that takes a retain, so a growing difference names the holder directly. */
void
vkr_mtl_refcount_census(char *buf, unsigned long len);

void
vkr_mtl_texture_release(void *mtl_texture);

/* limina: copy a registered scanout IOSurface's pixels into dst (top-down native BGRA, `height`
 * rows of min(dst_stride, surface bytesPerRow)). Used by the headless capture display sink, which
 * has no zero-copy present and no CPU transfer_read for venus blobs. Returns 1 on success, 0 on
 * failure (unknown id). */
int
vkr_mtl_iosurface_read(uint32_t id, void *dst, uint32_t dst_stride, uint32_t height);

#else /* !__APPLE__ */

static inline void *
vkr_metal_get_device(VkDevice vk_device, PFN_vkGetDeviceProcAddr GetDeviceProcAddr)
{
   (void)vk_device;
   (void)GetDeviceProcAddr;
   return NULL;
}

static inline void *
vkr_metal_get_system_device(void)
{
   return NULL;
}

static inline struct vkr_mtl_shm *
vkr_mtl_shm_alloc(void *mtl_device, uint64_t size)
{
   (void)mtl_device;
   (void)size;
   return NULL;
}

static inline void
vkr_mtl_shm_free(struct vkr_mtl_shm *shm)
{
   (void)shm;
}

static inline struct vkr_mtl_iosurface *
vkr_mtl_iosurface_alloc(void *mtl_device,
                        uint32_t width,
                        uint32_t height,
                        uint32_t mtl_pixel_format,
                        uint32_t iosurface_pixel_format,
                        uint32_t bytes_per_element,
                        uint32_t bytes_per_row)
{
   (void)mtl_device;
   (void)width;
   (void)height;
   (void)mtl_pixel_format;
   (void)iosurface_pixel_format;
   (void)bytes_per_row;
   (void)bytes_per_element;
   return NULL;
}

static inline void
vkr_mtl_iosurface_free(struct vkr_mtl_iosurface *surf)
{
   (void)surf;
}

static inline void *
vkr_mtl_iosurface_lookup(uint32_t id, void **out_base, uint64_t *out_alloc_size)
{
   (void)id;
   (void)out_base;
   (void)out_alloc_size;
   return NULL;
}

static inline void
vkr_mtl_iosurface_release_ref(void *io_surface)
{
   (void)io_surface;
}

static inline void
vkr_mtl_refcount_census(char *buf, unsigned long len)
{
   if (len)
      buf[0] = '\0';
}

static inline int
vkr_mtl_iosurface_read(uint32_t id, void *dst, uint32_t dst_stride, uint32_t height)
{
   (void)id;
   (void)dst;
   (void)dst_stride;
   (void)height;
   return 0;
}

#endif /* __APPLE__ */

#endif /* VKR_METAL_HELPERS_H */
