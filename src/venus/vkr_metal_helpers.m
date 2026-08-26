/*
 * Copyright 2026 Lucas Amaral
 * SPDX-License-Identifier: MIT
 */

#ifdef __APPLE__

#include "vkr_common.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mach/mach.h>
#include <pthread.h>
#include <servers/bootstrap.h>

#include "util/anon_file.h"
#include "vulkan/vulkan_metal.h"

#include "vkr_budget.h"
#include "vkr_metal_helpers.h"

/* Defined in virgl_resource.c; forward-declared to keep the resource layer's headers out of
 * this Objective-C TU. */
void virgl_resource_forget_iosurface(uint32_t iosurface_id);

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

/* +1/-1 balance for every host-side reference we take on an IOSurface. The budget ledger
 * answers "did we allocate and release it"; these answer the question that outlived it —
 * "is something ELSE still holding the storage after our release". The 2026-08-07 storm
 * charged 5912 surfaces, credited all but 3, and still held 52 GiB, so a holder exists
 * that the ledger cannot see. Each pair is a site that takes a retain; a difference that
 * grows with frames names the holder without any guessing. */
static _Atomic uint64_t g_limina_n_ios_alloc;   /* IOSurfaceCreate in our helpers */
static _Atomic uint64_t g_limina_n_ios_free;    /* vkr_mtl_iosurface_free */
static _Atomic uint64_t g_limina_n_tex_make;    /* newTextureWithDescriptor:iosurface: +1 */
static _Atomic uint64_t g_limina_n_tex_release; /* vkr_mtl_texture_release */
static _Atomic uint64_t g_limina_n_reg_insert;  /* registry retain */
static _Atomic uint64_t g_limina_n_reg_remove;  /* registry release */
static _Atomic uint64_t g_limina_n_lookup;      /* lookup +1 handed to a caller */
static _Atomic uint64_t g_limina_n_unref;       /* vkr_mtl_iosurface_release_ref */
static _Atomic uint64_t g_limina_n_publish_ok;  /* Mach send right handed to supervisor */
static _Atomic uint64_t g_limina_n_publish_err; /* send failed, right dropped locally */

/* limina: DEALLOC sentinels.
 *
 * The counters above answer "did our release calls balance", and the 2026-08-07 storm showed
 * that is the wrong question: 613 surfaces created, 608 released by us, all 613 still resident.
 * A balanced +1/-1 tally cannot see a ref taken by someone else, because only the object's own
 * death proves the last ref went away. So attach an associated object to every surface and
 * texture we create — the runtime releases associated objects when the host object is
 * deallocated — and count the sentinel's -dealloc. created-vs-deallocated then reads directly as
 * "did this object actually die", per class of object, with no guessing about who holds it.
 */
#define LIMINA_SENTINEL_BUILD_TAG "sentinel-1"

enum {
   LIMINA_SENTINEL_IOSURFACE,  /* IOSurfaceCreate in our helpers */
   LIMINA_SENTINEL_VKR_TEX,    /* the MTLTexture vkr_mtl_iosurface_alloc builds internally */
   LIMINA_SENTINEL_IMPORT_TEX, /* vkr_mtl_texture_from_iosurface (the vrend import) */
   LIMINA_SENTINEL_SELFTEST,   /* proves the mechanism itself fires; never a production object */
   LIMINA_SENTINEL_KINDS,
};

static _Atomic uint64_t g_limina_n_dealloc[LIMINA_SENTINEL_KINDS];

@interface LiminaDeallocSentinel : NSObject
@property(nonatomic) int kind;
@end

@implementation LiminaDeallocSentinel
- (void)dealloc
{
   atomic_fetch_add(&g_limina_n_dealloc[_kind], 1);
   [super dealloc];
}
@end

static const char g_limina_sentinel_key; /* address only */

static void
limina_sentinel_attach(id obj, int kind)
{
   if (!obj)
      return;
   LiminaDeallocSentinel *s = [[LiminaDeallocSentinel alloc] init];
   s.kind = kind;
   objc_setAssociatedObject(obj, &g_limina_sentinel_key, s,
                            OBJC_ASSOCIATION_RETAIN_NONATOMIC);
   [s release]; /* the association holds the only ref; it dies with obj */
}

/* Run once, at the first surface we create. An instrument that silently never fires reads
 * exactly like "nothing leaked", so prove the mechanism on a surface whose death we control
 * before trusting a single production number — and print a build tag, so a stale dylib is
 * visible in the log rather than inferred. */
static void
limina_sentinel_selftest(void)
{
   uint64_t before = atomic_load(&g_limina_n_dealloc[LIMINA_SENTINEL_SELFTEST]);
   @autoreleasepool {
      NSDictionary *props = @{
         (id)kIOSurfaceWidth : @16,
         (id)kIOSurfaceHeight : @16,
         (id)kIOSurfaceBytesPerElement : @4,
      };
      IOSurfaceRef io = IOSurfaceCreate((__bridge CFDictionaryRef)props);
      if (io) {
         limina_sentinel_attach((id)io, LIMINA_SENTINEL_SELFTEST);
         CFRelease(io);
      }
   }
   uint64_t after = atomic_load(&g_limina_n_dealloc[LIMINA_SENTINEL_SELFTEST]);
   fprintf(stderr,
           "[LIMINA-SENTINEL] vkr " LIMINA_SENTINEL_BUILD_TAG
           " armed; IOSurface dealloc sentinel self-test: %s\n",
           after > before ? "OK" : "FAILED (dealloc counts below are meaningless)");
}

static void
limina_sentinel_once(void)
{
   static pthread_once_t once = PTHREAD_ONCE_INIT;
   pthread_once(&once, limina_sentinel_selftest);
}

void
vkr_mtl_refcount_census(char *buf, unsigned long len)
{
   uint64_t a = atomic_load(&g_limina_n_ios_alloc), f = atomic_load(&g_limina_n_ios_free);
   uint64_t tm = atomic_load(&g_limina_n_tex_make),
            tr = atomic_load(&g_limina_n_tex_release);
   uint64_t ri = atomic_load(&g_limina_n_reg_insert),
            rr = atomic_load(&g_limina_n_reg_remove);
   uint64_t lk = atomic_load(&g_limina_n_lookup), ur = atomic_load(&g_limina_n_unref);
   uint64_t po = atomic_load(&g_limina_n_publish_ok),
            pe = atomic_load(&g_limina_n_publish_err);
   uint64_t dio = atomic_load(&g_limina_n_dealloc[LIMINA_SENTINEL_IOSURFACE]),
            dvt = atomic_load(&g_limina_n_dealloc[LIMINA_SENTINEL_VKR_TEX]),
            dit = atomic_load(&g_limina_n_dealloc[LIMINA_SENTINEL_IMPORT_TEX]);
   snprintf(buf, len,
            "iosurface %llu/%llu (+%lld) texture %llu/%llu (+%lld) registry %llu/%llu "
            "(+%lld) lookup %llu/%llu (+%lld) publish %llu ok %llu err | DEALLOC "
            "iosurface %llu (alive %lld) vkr-tex %llu (alive %lld) import-tex %llu",
            (unsigned long long)a, (unsigned long long)f, (long long)(a - f),
            (unsigned long long)tm, (unsigned long long)tr, (long long)(tm - tr),
            (unsigned long long)ri, (unsigned long long)rr, (long long)(ri - rr),
            (unsigned long long)lk, (unsigned long long)ur, (long long)(lk - ur),
            (unsigned long long)po, (unsigned long long)pe, (unsigned long long)dio,
            (long long)(a - dio), (unsigned long long)dvt, (long long)(a - dvt),
            (unsigned long long)dit);
}

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
   atomic_fetch_add(&g_limina_n_reg_insert, 1);
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
      atomic_fetch_add(&g_limina_n_reg_remove, 1);
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
      atomic_fetch_add(&g_limina_n_publish_err, 1);
   } else {
      atomic_fetch_add(&g_limina_n_publish_ok, 1);
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
   shm->limina_budget_ctx = vkr_budget_charge(aligned_size, "shm carrier");
   if (vkr_fd_trace())
      vkr_log_error("[FDTRACE] shm_alloc fd=%d size=%zu", shm_fd, aligned_size);
   return shm;
}

void
vkr_mtl_shm_free(struct vkr_mtl_shm *shm)
{
   if (!shm)
      return;
   vkr_budget_credit(shm->limina_budget_ctx, shm->shm_size);
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
   surf->limina_budget_ctx = vkr_budget_charge(surf->alloc_size, "IOSurface");
   atomic_fetch_add(&g_limina_n_ios_alloc, 1);
   limina_sentinel_once();
   limina_sentinel_attach((id)io, LIMINA_SENTINEL_IOSURFACE);
   limina_sentinel_attach(tex, LIMINA_SENTINEL_VKR_TEX);

   /* Scoped surfaces are non-global: register id->IOSurfaceRef so vkr_mtl_iosurface_lookup can
    * still resolve them, and hand the Mach port to the supervisor for present. */
   if (scope) {
      limina_registry_insert(surf->id, io);
      limina_publish_surface(surf->id, io);
   }
   return surf;
}

/* Row pitch a Metal LINEAR texture over these bytes will use — i.e. exactly what
 * KosmicKrisp computes in kk_image_layout.c (align(width * bpe, minimumLinearTexture-
 * AlignmentForPixelFormat)). Returns 0 for formats we can't map, meaning "let IOSurface
 * pick".
 *
 * Why this exists (spikes/vrend-stride-2026-08-14): left to itself IOSurface aligns rows
 * to 256 B, while KK computes 16 B for BGRA8. A GL client's buffer is allocated here, but
 * the compositor imports it through venus as a LINEAR VkImage, and KK then addresses it at
 * ITS pitch — so every row of every GL window slipped (256-16)/4 px sideways. Measured
 * 16.000 px/row at width 1968, 8.000 at 1976, matching (align256 - align16)/4 exactly.
 *
 * Deliberately querying Metal rather than hardcoding 16: the 256-vs-16 gap is settled at
 * the source, so the number can never drift out of step with KK's own computation. */
static uint32_t
limina_metal_linear_pitch(uint32_t iosurface_pixel_format,
                          uint32_t width,
                          uint32_t bytes_per_element)
{
   MTLPixelFormat mtl_format;
   switch (iosurface_pixel_format) {
   case 'BGRA':
      mtl_format = MTLPixelFormatBGRA8Unorm;
      break;
   case 'RGBA':
      mtl_format = MTLPixelFormatRGBA8Unorm;
      break;
   default:
      return 0;
   }

   /* Cached: this is on the resource-create path. */
   static id<MTLDevice> device;
   static dispatch_once_t once;
   dispatch_once(&once, ^{
      device = MTLCreateSystemDefaultDevice();
   });
   if (!device)
      return 0;

   NSUInteger align = [device minimumLinearTextureAlignmentForPixelFormat:mtl_format];
   if (!align)
      return 0;

   uint64_t row = (uint64_t)width * bytes_per_element;
   return (uint32_t)(((row + align - 1) / align) * align);
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

   /* Force the pitch a Metal linear texture over these bytes will use, so an importer
    * that lays the surface out as a LINEAR image addresses the same rows we allocated. */
   uint32_t want_row = limina_metal_linear_pitch(iosurface_pixel_format, width,
                                                 bytes_per_element);

   IOSurfaceRef io = NULL;
   @autoreleasepool {
      NSMutableDictionary *props = [@{
         (id)kIOSurfaceWidth : @(width),
         (id)kIOSurfaceHeight : @(height),
         (id)kIOSurfaceBytesPerElement : @(bytes_per_element),
         (id)kIOSurfacePixelFormat : @(iosurface_pixel_format),
         (id)kIOSurfaceIsGlobal : @(mark_global),
      } mutableCopy];
      if (want_row)
         props[(id)kIOSurfaceBytesPerRow] = @(want_row);
      io = IOSurfaceCreate((__bridge CFDictionaryRef)props);
   }
   if (!io)
      return NULL;

   /* Unlike the venus scanout path, which drops zero-copy when IOSurface overrides the
    * pitch, there is no fallback here: refusing the surface would take VIRGL_BIND_SHARED
    * down with it and re-break Vulkan-compositor imports outright. Keep it and be loud —
    * an override means importers shear again, and this line is the only warning. */
   if (want_row && (uint32_t)IOSurfaceGetBytesPerRow(io) != want_row) {
      fprintf(stderr,
              "[KK-STRIDE] IOSurface overrode the row pitch (asked %u, got %zu) for %ux%u "
              "— linear importers will shear\n",
              want_row, IOSurfaceGetBytesPerRow(io), width, height);
   }

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
   surf->limina_budget_ctx = vkr_budget_charge(surf->alloc_size, "IOSurface");
   atomic_fetch_add(&g_limina_n_ios_alloc, 1);
   limina_sentinel_once();
   limina_sentinel_attach((id)io, LIMINA_SENTINEL_IOSURFACE);

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
   vkr_budget_credit(surf->limina_budget_ctx, surf->alloc_size);
   atomic_fetch_add(&g_limina_n_ios_free, 1);

   /* LIMINA_SURF_REFTRACE=N: dump the CF retain count of the surface and its texture on the
    * last line where we can still legally touch them. Our own refs are one each, so anything
    * above that names a holder we did not put there. A count is a LEAD, not a verdict (the
    * runtime takes transient refs); the dealloc sentinels above are the verdict. Budgeted,
    * because 613 frees a run drowns the log. */
   {
      static _Atomic int traced;
      static _Atomic int budget = -1;
      int b = atomic_load(&budget);
      if (b < 0) {
         const char *e = getenv("LIMINA_SURF_REFTRACE");
         b = (e && *e) ? (int)strtol(e, NULL, 10) : 0;
         atomic_store(&budget, b);
      }
      if (b > 0 && atomic_fetch_add(&traced, 1) < b)
         fprintf(stderr,
                 "[SURF-REF] id=%u retain: surface=%ld texture=%ld (ours: 1 each)\n",
                 surf->id, surf->io_surface ? CFGetRetainCount(surf->io_surface) : -1,
                 surf->mtl_texture ? CFGetRetainCount(surf->mtl_texture) : -1);
   }
   /* Drop the registry's retain before ours (no-op if the surface was global). */
   limina_registry_remove(surf->id);
   /* An IOSurface id is reusable the moment its surface dies, so every cached copy of this id
    * must die with it -- a resource still naming it would hand the VMM a stranger's surface to
    * present or, worse, to release. (Declared in virgl_resource.h; forward-declared here so this
    * TU does not pull the resource layer's headers in.) */
   virgl_resource_forget_iosurface(surf->id);
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
   atomic_fetch_add(&g_limina_n_lookup, 1);
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
   if (tex) {
      atomic_fetch_add(&g_limina_n_tex_make, 1);
      limina_sentinel_attach(tex, LIMINA_SENTINEL_IMPORT_TEX);
   }
   return (void *)tex; /* +1 from newTextureWithDescriptor (ARC-retained by the cast) */
}

void
vkr_mtl_texture_release(void *mtl_texture)
{
   if (mtl_texture) {
      CFRelease(mtl_texture);
      atomic_fetch_add(&g_limina_n_tex_release, 1);
   }
}

void
vkr_mtl_iosurface_release_ref(void *io_surface)
{
   if (io_surface) {
      CFRelease(io_surface);
      atomic_fetch_add(&g_limina_n_unref, 1);
   }
}

/* limina: re-hand an already-registered scanout surface to the supervisor.
 *
 * The supervisor holds our published surfaces in a bounded store and evicts at its cap. A
 * non-global surface it has dropped is unrecoverable from its side — IOSurfaceLookup fails by
 * design and only the creator can mint a Mach port — so a guest that presents an evicted id had
 * every later frame silently skipped, freezing the display permanently for that id
 * (spikes/scanout-blob-freeze/RESULTS.md). This lets the supervisor ask for it back, which makes
 * its cap a memory parameter rather than a correctness one.
 *
 * The reply is an ordinary publish on the SAME Mach port as every other publish and release, so
 * it inherits that queue's ordering: an id cannot recycle until its surface dies, a release is
 * enqueued before the unref that kills it, and one port is one FIFO. Answering over any other
 * channel would reintroduce the two-queue hazard that ordering closed.
 *
 * Returns 1 if the id was registered and a publish was attempted, 0 if we do not have it.
 * Thread-safe: the registry is mutex-guarded and mach_msg is not thread-affine. */
int
limina_republish_surface(uint32_t id)
{
   IOSurfaceRef io = limina_registry_lookup(id); /* +1 */
   if (!io)
      return 0;
   limina_publish_surface(id, io);
   vkr_mtl_iosurface_release_ref((void *)io); /* -1 */
   return 1;
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
