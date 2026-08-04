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
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mach/mach.h>
#include <pthread.h>
#include <servers/bootstrap.h>

#include "util/anon_file.h"
#include "vulkan/vulkan_metal.h"

#include "vkr_metal_helpers.h"

/*
 * limina: capability-scope the zero-copy scanout IOSurfaces.
 *
 * By default the scanout surfaces are created kIOSurfaceIsGlobal so the supervisor (a separate
 * process) can IOSurfaceLookup(id) them for present. Global surfaces are readable by ANY same-user
 * process (`iosdump <id>` dumps the guest screen), which Apple itself flags as insecure. When the
 * supervisor exposes a surface-port receiver (env LIMINA_SURFACE_PORT_NAME, set by limina-vmm from
 * --surface-port-name), we instead create the surfaces NON-global and hand each one's Mach port to
 * the supervisor at alloc time — the exact bootstrap rendezvous + wire format used by the sw2d path
 * in crates/limina-surfaceport. The supervisor then resolves the "frame <id>" transport from its
 * Mach map, and no stranger can read the screen. LIMINA_GLOBAL_SCANOUT=1 forces the old global
 * behaviour (so `iosdump` still works as a debug oracle).
 *
 * A small id->IOSurfaceRef registry replaces the cross-context global IOSurfaceLookup (which fails
 * on non-global surfaces) for vkr_mtl_iosurface_lookup.
 */

/* Wire message: byte-for-byte the SendMsg in crates/limina-surfaceport (id in msgh_id, one
 * MOVE_SEND port descriptor carrying the IOSurface Mach port). */
typedef struct {
   mach_msg_header_t header;
   mach_msg_body_t body;
   mach_msg_port_descriptor_t port;
} limina_surface_msg_t;

static pthread_mutex_t g_limina_lock = PTHREAD_MUTEX_INITIALIZER;
static CFMutableDictionaryRef g_limina_registry; /* id (CFNumber) -> IOSurfaceRef (retained) */
static mach_port_t g_limina_supervisor = MACH_PORT_NULL;
static int g_limina_state; /* 0 = not yet looked up, 1 = scoping on, -1 = global (off) */

/* The supervisor's surface-port receiver, looked up once. MACH_PORT_NULL ⇒ no Mach present
 * transport (no receiver name, or the lookup failed). NOTE: LIMINA_GLOBAL_SCANOUT does NOT
 * disable this — the debug oracle marks surfaces global ADDITIVELY (see limina_mark_global)
 * while keeping the scoped Mach present, so the venus desktop still displays under the flag
 * (matching the sw2d WindowBackend, whose `also_global` is likewise additive). */
static mach_port_t
limina_supervisor_port(void)
{
   pthread_mutex_lock(&g_limina_lock);
   if (g_limina_state == 0) {
      g_limina_state = -1;
      const char *name = getenv("LIMINA_SURFACE_PORT_NAME");
      if (name && name[0]) {
         mach_port_t sp = MACH_PORT_NULL;
         if (bootstrap_look_up(bootstrap_port, name, &sp) == KERN_SUCCESS &&
             sp != MACH_PORT_NULL) {
            g_limina_supervisor = sp;
            g_limina_state = 1;
         }
      }
   }
   mach_port_t ret = g_limina_state == 1 ? g_limina_supervisor : MACH_PORT_NULL;
   pthread_mutex_unlock(&g_limina_lock);
   return ret;
}

/* True when scanout surfaces should be handed over scoped (the supervisor receiver is reachable).
 * Independent of the debug oracle: present is via the Mach port whenever a receiver exists. */
static bool
limina_scope_surfaces(void)
{
   return limina_supervisor_port() != MACH_PORT_NULL;
}

/* True when scanout surfaces must ALSO be marked kIOSurfaceIsGlobal — i.e. resolvable by any
 * process via IOSurfaceLookup(id). Two cases: (1) the LIMINA_GLOBAL_SCANOUT debug oracle, so
 * `iosdump` can read the venus screen by global id; (2) no supervisor receiver, where global is
 * the only present transport. This is ADDITIVE to scoping — under the debug flag a surface is
 * BOTH global (for iosdump) AND Mach-published (for the real present), so the desktop keeps
 * displaying. Marking a surface global is intentionally insecure; it is a debug-only widening. */
static bool
limina_mark_global(void)
{
   return getenv("LIMINA_GLOBAL_SCANOUT") != NULL || limina_supervisor_port() == MACH_PORT_NULL;
}

static void
limina_registry_insert(uint32_t id, IOSurfaceRef io)
{
   pthread_mutex_lock(&g_limina_lock);
   if (!g_limina_registry) {
      g_limina_registry =
         CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                   &kCFTypeDictionaryValueCallBacks);
   }
   CFNumberRef key = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &id);
   CFDictionarySetValue(g_limina_registry, key, io); /* retains io */
   CFRelease(key);
   pthread_mutex_unlock(&g_limina_lock);
}

static void
limina_registry_remove(uint32_t id)
{
   pthread_mutex_lock(&g_limina_lock);
   if (g_limina_registry) {
      CFNumberRef key = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &id);
      CFDictionaryRemoveValue(g_limina_registry, key); /* releases io */
      CFRelease(key);
   }
   pthread_mutex_unlock(&g_limina_lock);
}

/* RETAINED IOSurfaceRef for a registered id, or NULL. */
static IOSurfaceRef
limina_registry_lookup(uint32_t id)
{
   IOSurfaceRef io = NULL;
   pthread_mutex_lock(&g_limina_lock);
   if (g_limina_registry) {
      CFNumberRef key = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &id);
      io = (IOSurfaceRef)CFDictionaryGetValue(g_limina_registry, key);
      if (io)
         CFRetain(io);
      CFRelease(key);
   }
   pthread_mutex_unlock(&g_limina_lock);
   return io;
}

/* Hand a non-global scanout surface to the supervisor by Mach port, keyed by its id. */
static void
limina_publish_surface(uint32_t id, IOSurfaceRef io)
{
   mach_port_t sup = limina_supervisor_port();
   if (sup == MACH_PORT_NULL)
      return;
   mach_port_t port = IOSurfaceCreateMachPort(io); /* +1 send right */
   if (port == MACH_PORT_NULL)
      return;
   limina_surface_msg_t msg;
   memset(&msg, 0, sizeof(msg));
   msg.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0) | MACH_MSGH_BITS_COMPLEX;
   msg.header.msgh_size = sizeof(msg);
   msg.header.msgh_remote_port = sup;
   msg.header.msgh_id = (mach_msg_id_t)id;
   msg.body.msgh_descriptor_count = 1;
   msg.port.name = port;
   msg.port.disposition = MACH_MSG_TYPE_MOVE_SEND;
   msg.port.type = MACH_MSG_PORT_DESCRIPTOR;
   kern_return_t kr = mach_msg(&msg.header, MACH_SEND_MSG, sizeof(msg), 0, MACH_PORT_NULL,
                               MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
   if (kr != KERN_SUCCESS) {
      /* MOVE_SEND only consumes the right on success; drop it otherwise. */
      mach_port_deallocate(mach_task_self(), port);
   }
}

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

void *
vkr_metal_get_system_device(void)
{
   /* Fallback when the driver can't export its MTLDevice (KosmicKrisp has no
    * VK_EXT_metal_objects). The mtl_device is only used for blob-export carriers
    * (vkr_mtl_shm_alloc's no-copy MTLBuffer) and IOSurface allocation — Apple
    * Silicon machines have a single GPU, so the system default device is the
    * same device the driver renders on. */
   return (void *)MTLCreateSystemDefaultDevice();
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
   if (vkr_fd_trace())
      vkr_log_error("[FDTRACE] shm_alloc fd=%d size=%zu", shm_fd, aligned_size);
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
   if (shm->shm_fd >= 0) {
      if (vkr_fd_trace())
         vkr_log_error("[FDTRACE] shm_free fd=%d", shm->shm_fd);
      close(shm->shm_fd);
   }
   free(shm);
}

struct vkr_mtl_iosurface *
vkr_mtl_iosurface_alloc(void *mtl_device,
                        uint32_t width,
                        uint32_t height,
                        uint32_t mtl_pixel_format,
                        uint32_t iosurface_pixel_format,
                        uint32_t bytes_per_element,
                        uint32_t bytes_per_row)
{
   if (!mtl_device || !width || !height)
      return NULL;

   IOSurfaceRef io = NULL;
   id<MTLTexture> tex = nil;

   /* Scope (hand over by Mach port) whenever the supervisor exposes a receiver — the default,
    * secure present path. SEPARATELY decide whether to ALSO mark the surface global: the
    * LIMINA_GLOBAL_SCANOUT debug oracle or the no-receiver fallback. The two are independent, so
    * under the debug flag a surface is both scoped-for-present AND global-for-iosdump. */
   bool scope = limina_scope_surfaces();
   bool mark_global = limina_mark_global();

   @autoreleasepool {
      /* kIOSurfaceIsGlobal: a global surface is looked up by IOSurfaceGetID for present (the
       * legacy limina-display transport). When scoping, the surface is NON-global and its Mach
       * port is handed to the supervisor below instead. IOSurface picks/aligns bytesPerRow
       * itself unless the caller forces one (bytes_per_row != 0, the KosmicKrisp linear-image
       * path: the surface bytes back the image memory, so the pitch must equal the driver's
       * linear rowPitch). */
      NSMutableDictionary *props = [@{
         (id)kIOSurfaceWidth : @(width),
         (id)kIOSurfaceHeight : @(height),
         (id)kIOSurfaceBytesPerElement : @(bytes_per_element),
         (id)kIOSurfacePixelFormat : @(iosurface_pixel_format),
         (id)kIOSurfaceIsGlobal : @(mark_global),
      } mutableCopy];
      if (bytes_per_row)
         props[(id)kIOSurfaceBytesPerRow] = @(bytes_per_row);
      io = IOSurfaceCreate((__bridge CFDictionaryRef)props);
      if (!io)
         return NULL;

      id<MTLDevice> device = (id<MTLDevice>)mtl_device;
      MTLTextureDescriptor *td =
         [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)mtl_pixel_format
                                                            width:width
                                                           height:height
                                                        mipmapped:NO];
      /* PixelFormatView: scanout images are MUTABLE_FORMAT with a UNORM+sRGB view list. */
      /* ShaderWrite is here for the MTLTEXTURE-import scanout path: KosmicKrisp derives
       * a Metal usage mask from the guest's VkImageUsageFlags, and a scanout image that
       * carries TRANSFER_DST or STORAGE maps to MTL_TEXTURE_USAGE_SHADER_WRITE. KK's bind
       * requires the imported texture's usage to be a SUPERSET of the image's, so a
       * missing bit fails the bind outright (by design — the alternative is Metal
       * silently no-op'ing the render into this surface). Extra bits are harmless. */
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                 MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
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
   surf->base_addr = IOSurfaceGetBaseAddress(io);
   surf->alloc_size = IOSurfaceGetAllocSize(io);

   /* Scoped surfaces are non-global: register id->IOSurfaceRef so vkr_mtl_iosurface_lookup can
    * still resolve them, and hand the Mach port to the supervisor for present. */
   if (scope) {
      limina_registry_insert(surf->id, io);
      limina_publish_surface(surf->id, io);
   }
   return surf;
}

struct vkr_mtl_iosurface *
vkr_mtl_iosurface_alloc_plain(uint32_t width,
                              uint32_t height,
                              uint32_t iosurface_pixel_format,
                              uint32_t bytes_per_element)
{
   if (!width || !height)
      return NULL;

   bool scope = limina_scope_surfaces();
   bool mark_global = limina_mark_global();

   IOSurfaceRef io = NULL;
   @autoreleasepool {
      NSDictionary *props = @{
         (id)kIOSurfaceWidth : @(width),
         (id)kIOSurfaceHeight : @(height),
         (id)kIOSurfaceBytesPerElement : @(bytes_per_element),
         (id)kIOSurfacePixelFormat : @(iosurface_pixel_format),
         (id)kIOSurfaceIsGlobal : @(mark_global),
      };
      io = IOSurfaceCreate((__bridge CFDictionaryRef)props);
   }
   if (!io)
      return NULL;

   struct vkr_mtl_iosurface *surf = calloc(1, sizeof(*surf));
   if (!surf) {
      CFRelease(io);
      return NULL;
   }
   surf->io_surface = (void *)io; /* +1 from IOSurfaceCreate */
   surf->mtl_texture = NULL;
   surf->id = IOSurfaceGetID(io);
   surf->width = width;
   surf->height = height;
   surf->bytes_per_row = (uint32_t)IOSurfaceGetBytesPerRow(io);
   surf->base_addr = IOSurfaceGetBaseAddress(io);
   surf->alloc_size = IOSurfaceGetAllocSize(io);

   if (scope) {
      limina_registry_insert(surf->id, io);
      limina_publish_surface(surf->id, io);
   }
   return surf;
}

uint32_t
vkr_mtl_iosurface_get_id(const struct vkr_mtl_iosurface *surf)
{
   return surf ? surf->id : 0;
}

/* Raw IOSurfaceRef (borrowed, no ref transfer) — for vrend's EGLImage-backed
 * scanout, which hands it to virgl_egl_image_from_iosurface. */
void *
vkr_mtl_iosurface_get_ref(const struct vkr_mtl_iosurface *surf)
{
   return surf ? surf->io_surface : NULL;
}

void
vkr_mtl_iosurface_get_layout(const struct vkr_mtl_iosurface *surf,
                             void **out_base,
                             uint32_t *out_bytes_per_row,
                             uint64_t *out_alloc_size)
{
   *out_base = surf ? surf->base_addr : NULL;
   *out_bytes_per_row = surf ? surf->bytes_per_row : 0;
   *out_alloc_size = surf ? surf->alloc_size : 0;
}

void
vkr_mtl_iosurface_free(struct vkr_mtl_iosurface *surf)
{
   if (!surf)
      return;
   /* Drop the registry's retain before ours (no-op if the surface was global). */
   limina_registry_remove(surf->id);
   if (surf->mtl_texture)
      CFRelease(surf->mtl_texture);
   if (surf->io_surface)
      CFRelease(surf->io_surface);
   free(surf);
}

void *
vkr_mtl_iosurface_lookup(uint32_t id, void **out_base, uint64_t *out_alloc_size)
{
   /* Scoped surfaces are non-global → resolve via our registry; fall back to the global
    * IOSurfaceLookup for the LIMINA_GLOBAL_SCANOUT / no-receiver path. Both return +1. */
   IOSurfaceRef io = limina_registry_lookup(id);
   if (!io)
      io = IOSurfaceLookup(id);
   if (!io)
      return NULL;
   *out_base = IOSurfaceGetBaseAddress(io);
   *out_alloc_size = IOSurfaceGetAllocSize(io);
   return (void *)io;
}

void *
vkr_mtl_texture_from_iosurface(void *mtl_device, void *io_surface, uint32_t mtl_pixel_format)
{
   if (!mtl_device || !io_surface)
      return NULL;

   IOSurfaceRef io = (IOSurfaceRef)io_surface;
   id<MTLTexture> tex = nil;
   @autoreleasepool {
      /* The IMPORT half of the MTLTEXTURE scanout path: the exporter already backed its
       * image with this IOSurface's texture, so the importer must adopt the SAME object
       * rather than aliasing the bytes as a linear host pointer. A host-pointer import
       * reads the surface at the importing image's own linear rowPitch, which is not the
       * layout the exporter's texture writes — the mismatch shears every row (2026-08-04:
       * every GTK4 window skewed while the compositor's own scanout, which has no guest
       * importer, stayed correct).
       *
       * Descriptor must mirror vkr_mtl_iosurface_alloc's exactly: KosmicKrisp's bind
       * demands the imported texture's usage be a SUPERSET of the image's, and an exact
       * pixel-format match (sRGB included — see limina_vkformat_to_mtl_exact). */
      MTLTextureDescriptor *td = [MTLTextureDescriptor
         texture2DDescriptorWithPixelFormat:(MTLPixelFormat)mtl_pixel_format
                                      width:IOSurfaceGetWidth(io)
                                     height:IOSurfaceGetHeight(io)
                                  mipmapped:NO];
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                 MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
      td.storageMode = MTLStorageModeShared;
      tex = [(id<MTLDevice>)mtl_device newTextureWithDescriptor:td iosurface:io plane:0];
   }
   return (void *)tex; /* +1 from newTextureWithDescriptor (ARC-retained by the cast) */
}

void
vkr_mtl_texture_release(void *mtl_texture)
{
   if (mtl_texture)
      CFRelease(mtl_texture);
}

void
vkr_mtl_iosurface_release_ref(void *io_surface)
{
   if (io_surface)
      CFRelease(io_surface);
}

/* limina: copy a registered scanout IOSurface's pixels (top-down, the surface's native BGRA)
 * into dst — `height` rows of min(dst_stride, surface bytesPerRow). The scanout textures are
 * MTLStorageModeShared (CPU-visible), so the headless capture display sink — which has no
 * zero-copy present_surface and no CPU transfer_read for venus blobs — reads the presented frame
 * this way. IOSurfaceLock flushes the GPU's writes for a coherent read. Returns 1 on success. */
int
vkr_mtl_iosurface_read(uint32_t id, void *dst, uint32_t dst_stride, uint32_t height)
{
   if (!dst || !dst_stride || !height)
      return 0;
   void *base = NULL;
   uint64_t alloc_size = 0;
   IOSurfaceRef io = (IOSurfaceRef)vkr_mtl_iosurface_lookup(id, &base, &alloc_size); /* +1 */
   if (!io)
      return 0;
   IOSurfaceLock(io, kIOSurfaceLockReadOnly, NULL);
   const uint8_t *src = (const uint8_t *)IOSurfaceGetBaseAddress(io);
   size_t src_stride = (size_t)IOSurfaceGetBytesPerRow(io);
   size_t row_bytes = (size_t)dst_stride < src_stride ? (size_t)dst_stride : src_stride;
   for (uint32_t row = 0; row < height; row++)
      memcpy((uint8_t *)dst + (size_t)row * dst_stride, src + (size_t)row * src_stride, row_bytes);
   IOSurfaceUnlock(io, kIOSurfaceLockReadOnly, NULL);
   vkr_mtl_iosurface_release_ref((void *)io); /* -1 */
   return 1;
}

#endif /* __APPLE__ */
