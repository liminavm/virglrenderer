// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
// Copyright © 2026 Gustavo Noronha Silva
//
// Replay a captured vrend command stream through libvirglrenderer on the HOST, with no VM.
//
// This is harness layer 2 (docs/rust-rewrite.md): a VM-free vehicle that drives only the public
// ABI, so the same corpus runs against the C renderer and against virglrs and the outputs are
// comparable. A replay costs seconds where a boot costs minutes, which is what makes a rewrite
// testable at subagent speed.
//
// INPUT is a dump from vrend's in-memory tracer (LIMINA_VREND_TRACE, src/vrend/vrend_trace.[ch]):
// submit boundaries, every decoded command with its full payload, the bytes each guest->host
// transfer carried, and an ordered resource create/blob/unref log in a never-evicted side store.
// The replayer needs no blob or iov machinery of its own — it hands every resource a plain
// malloc'd iov and memcpys the recorded bytes in before the batch that reads them.
//
// ORACLE. Each scored resource is read back through transfer_read_iov and reported as an FNV-1a
// hash plus an ink count; REPLAY_DUMP_DIR additionally writes the raw pixels and a manifest.
// The hash line is the golden: record it from the C build, diff it against virglrs.
//
//   vrend-replay <dump> [--ctx N[,N...]] [--loops N] [--nodraw] [--nofeed] [--readback RES] [--sweep]
//
//   --ctx N[,N]   which virgl contexts to replay (default: the one with the most records).
//                 A list, because one workload can span contexts that only make sense
//                 together -- a video player draws in one and decodes in another, and
//                 either alone replays half the operation.
//   --loops N     replay the captured stream N times (default 1)
//   --nofeed      positive control for blob content: land the recorded bytes in the backing
//                 store as usual, but never carry them into the texture. What inks a blob's
//                 window then is the renderer reading the guest's pages for itself, which is
//                 the whole of the import path -- and with the feed on, a renderer that reads
//                 nothing scores exactly like one that reads correctly.
//   --nodraw      positive control: drop every DRAW_VBO. Score it and diff against the ordinary
//                 score -- the resources that lose their ink are the ones drawing reaches, and an
//                 empty diff means the oracle is measuring nothing.
//   --readback R  score only this resource (default: every colour offscreen, at its unref)
//   --sweep       score every colour offscreen at its unref -- the default
//   --no-zero-new leave a new resource's contents undefined instead of zeroing them, which is
//                 how to see what an unwritten resource was reading. Goldens are recorded with
//                 the zeroing ON, so a score line is the renderer's answer and not its allocator's.
//   --sweep-w W   restrict --sweep to targets of this width
//   --score F     write the score to F
//   --expect F    compare the score against F and exit non-zero on any difference
//   --rebuild-score F   write the rebuild report -- entries in and out per context, every
//                 entry the rebuild could not use, and the contents account (how many levels
//                 were captured, how many were all zeros, and every resource that did not come
//                 back byte for byte) -- to F
//   --rebuild-expect F  compare that report against F. Without it, a rebuild that loses an entry
//                 fails; with it, the pinned losses are the ones this corpus is known to make
//   --caps F      write the classic capsets (VIRGL and VIRGL2) the renderer fills, one field a
//                 line, to F and exit. Record it from the C, diff it against virglrs: the guest's
//                 driver configures itself from nothing else.
//
// Env: REPLAY_DUMP_DIR (raw BGRA + manifest.txt), REPLAY_DUMP_W, REPLAY_NO_UNREF.
//
// Run it through vrend-replay.sh, which supplies the KosmicKrisp/zink environment the worker uses.
//
/* USE_VIDEO lives behind this guard in the header, and limina passes it, so the replay has to
 * see it too. */
#define VIRGL_RENDERER_UNSTABLE_APIS
#include "virglrenderer.h"
#include "virgl_hw.h"

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <IOSurface/IOSurface.h>
#include <string.h>
#include <sys/uio.h>

#include "vrend-replay-formats.h"

#define TRACE_MAGIC 0x4c4d5654u

enum { T_SUBMIT = 1, T_CMD = 2, T_DRAW_FB = 3, T_TRANSFER = 4, T_FENCE = 5, T_RETIRE = 6,
       T_PAD = 7, T_XFERDATA = 9, T_BLOBDATA = 10 };
enum { RES_CREATE = 0, RES_BLOB = 1, RES_UNREF = 2 };

#define VIRGL_CCMD_CREATE_OBJECT        1
#define VIRGL_CCMD_SET_FRAMEBUFFER_STATE 5
#define VIRGL_CCMD_SET_VERTEX_BUFFERS    6
#define VIRGL_CCMD_DRAW_VBO              8
#define VIRGL_CCMD_COPY_TRANSFER3D       45
/* dword index of the SOURCE resource handle inside a COPY_TRANSFER3D payload */
#define VIRGL_COPY_TRANSFER3D_SRC_RES_HANDLE 12
/* dword index of the source OFFSET, which is the offset the payload record carries */
#define VIRGL_COPY_TRANSFER3D_SRC_RES_OFFSET 13
/* dword index of the DESTINATION handle, which is what the payload record is keyed by */
#define VIRGL_COPY_TRANSFER3D_DST_RES_HANDLE 1
#define VIRGL_OBJECT_SURFACE             8

struct rec_hdr {
   uint32_t total_len;
   uint8_t  type, cmd;
   uint16_t ctx_id;
   uint64_t seq, mono_ns;
   uint32_t payload_len, aux_count;
};

struct res_ev {
   uint64_t seq;
   uint32_t kind, handle, target, format, bind;
   uint32_t width, height, depth, array_size, last_level, nr_samples, flags;
};

/* One host-side backing store per replayed resource. The capture records transfer CONTENTS, so
 * the replayer never reconstructs blob or iov machinery: it hands every resource a plain malloc'd
 * iov and memcpys the recorded bytes in before the batch that reads them. */
struct backing {
   uint32_t handle;
   uint8_t *mem;
   size_t   size;
   struct iovec iov;
   bool     live;
   /* A blob is registered UNTYPED and carries no format or extent of its own: the stream's
    * PIPE_RESOURCE_SET_TYPE is what says which. These three are that command's answer, kept so
    * the readback can ask for the resource at the geometry the guest gave it. `typed` is the
    * one place that says whether the answer arrived -- a blob nothing ever typed has no texture
    * to read back and is counted rather than scored. */
   bool     is_blob;
   bool     typed;
   uint32_t t_format, t_width, t_height, t_stride;
   /* Content landed since the last write into the texture. A blob's pixels are not the
    * texture's: the bytes go into the backing store, and something has to carry them across. */
   bool     dirty;
   /* Set when this blob's geometry has been refused once. The geometry is fixed the moment the
    * blob is typed, so the refusal is permanent: without this the retry fires again at every
    * subsequent content record and says the same thing once per frame. */
   bool     declined;
};

/* An array of POINTERS, never of structs. vrend stores the `struct iovec *` it is handed and
 * dereferences it for the resource's whole life, so a backing's iovec must never move. Holding
 * the backings by value in a realloc'd array left every previously attached resource pointing
 * into freed memory the moment the array grew past its capacity -- which is why a run would
 * sometimes sail through and sometimes report `src iov_len=0`, poison the context on one failed
 * copy, and then fail every submit and readback after it in silence. */
static struct backing **backings;
static uint32_t backing_n, backing_cap;

static struct backing *backing_find(uint32_t handle)
{
   for (uint32_t i = 0; i < backing_n; i++)
      if (backings[i]->live && backings[i]->handle == handle)
         return backings[i];
   return NULL;
}

static struct backing *backing_add(uint32_t handle, size_t size)
{
   if (backing_n == backing_cap) {
      backing_cap = backing_cap ? backing_cap * 2 : 256;
      backings = realloc(backings, backing_cap * sizeof *backings);
      if (!backings) { fprintf(stderr, "OOM\n"); exit(2); }
   }
   struct backing *b = calloc(1, sizeof *b);
   if (!b) { fprintf(stderr, "OOM\n"); exit(2); }
   backings[backing_n++] = b;
   b->handle = handle;
   b->size = size ? size : 4096;
   b->mem = calloc(1, b->size);
   if (!b->mem) { fprintf(stderr, "OOM\n"); exit(2); }
   b->iov.iov_base = b->mem;
   b->iov.iov_len = b->size;
   b->live = true;
   return b;
}

static void write_fence(void *cookie, uint32_t fence) { (void)cookie; (void)fence; }

static struct virgl_renderer_callbacks cbs = {
   .version = VIRGL_RENDERER_CALLBACKS_VERSION,
   .write_fence = write_fence,
};

/* A generous over-estimate of a resource's byte size. The capture records the create arguments
 * but not the host's chosen layout, and an iov that is too small makes vrend reject transfers
 * with ILLEGAL_RESOURCE -- which would read as a replay fault rather than a sizing mistake. */
static size_t res_bytes(const struct res_ev *r)
{
   uint64_t w = r->width ? r->width : 1;
   uint64_t h = r->height ? r->height : 1;
   uint64_t d = r->depth ? r->depth : 1;
   uint64_t a = r->array_size ? r->array_size : 1;
   if (r->kind == RES_BLOB)
      return (size_t)(((uint64_t)r->height << 32) | r->width);   /* size was split across two words */
   /* target 0 is PIPE_BUFFER: width is already bytes. Otherwise assume at most 16 bytes/texel
    * and add the mip tail. */
   if (r->target == 0)
      return (size_t)w;
   return (size_t)(w * h * d * a * 16u + 65536u);
}



/* Score at the resource's UNREF, not after the whole stream.
 *
 * The obvious alternative -- suppress the unref so the resource survives to the end -- is not
 * neutral: virgl handles are REUSED, so keeping one alive makes a later create collide and fail,
 * and everything downstream of that resource silently stops rendering. That is how the first
 * scoring attempt read "no ink" from a card the bug is known to spare. The unref is the last
 * moment the resource is both complete and still alive, so read there and let the stream run on
 * exactly as captured. */
static uint32_t readback_res;
static int scored, scored_ink;
/* How many created resources came back IOSurface-backed. A count, never the ids: an id is
 * host-private, recycled the instant its surface dies, and free to change across a snapshot
 * restore. The count is the part a port owes us -- backing the wrong set of resources is
 * silent everywhere else, and it decides whether the present path is zero-copy at all. */
static uint32_t iosurf_backed;
/* Summed across loops, for the --smoke verdict; the per-loop counters stay per-loop so the
 * counters line still says which pass a divergence appeared in. */
static uint32_t made_total, failed_total;

/* IOSurface-backed resources are scored at the END of the stream, not at their unref, because
 * they are the ones a capture never unrefs: the scanout outlives every frame in it. That is the
 * mirror image of why the colour offscreens are scored AT their unref -- each is read at the last
 * moment it is both complete and still alive, and for these two kinds of resource that moment is
 * at opposite ends of the run. */
static struct { uint32_t handle, w, h; } *iosurf_res;
static uint32_t iosurf_n, iosurf_cap;

static void iosurf_remember(uint32_t handle, uint32_t w, uint32_t h)
{
   if (iosurf_n == iosurf_cap) {
      iosurf_cap = iosurf_cap ? iosurf_cap * 2 : 8;
      iosurf_res = realloc(iosurf_res, iosurf_cap * sizeof *iosurf_res);
      if (!iosurf_res) { fprintf(stderr, "OOM\n"); exit(2); }
   }
   iosurf_res[iosurf_n].handle = handle;
   iosurf_res[iosurf_n].w = w ? w : 1;
   iosurf_res[iosurf_n].h = h ? h : 1;
   iosurf_n++;
}
static int want_ctx = -1;
/* The contexts to replay. One is the common case; more than one exists because a workload can
 * split across contexts that only make sense together -- a video player draws in one and decodes
 * in another, and either alone replays half an operation. `want_ctx` is the first of them, and is
 * what the single-context library calls use. */
enum { MAX_CTX = 8 };
static int ctx_list[MAX_CTX];
static int n_ctx;

static bool ctx_wanted(uint16_t id)
{
   for (int i = 0; i < n_ctx; i++)
      if ((uint16_t)ctx_list[i] == id) return true;
   return false;
}
static const char *score_path, *expect_path, *caps_path;
static const char *rb_score_path, *rb_expect_path;
static bool zero_new = true;

/* The score, accumulated in stream order. Two implementations that render the same pixels in a
 * different order are not the same implementation, so the ORDER is part of what is pinned.
 * Counters live in their own buffer only so the finished score can put them FIRST: a score that
 * diverges should say whether the stream was applied the same way before it says the pixels are. */
static char *score_buf;
static size_t score_len, score_cap;
static char *count_buf;
static size_t count_len, count_cap;

static void buf_addf(char **buf, size_t *len, size_t *cap, const char *fmt, va_list ap)
{
   char line[512];
   int n = vsnprintf(line, sizeof line, fmt, ap);
   if (n < 0 || (size_t)n >= sizeof line)
      return;
   if (*len + (size_t)n + 1 > *cap) {
      *cap = *cap ? *cap * 2 : 4096;
      while (*len + (size_t)n + 1 > *cap) *cap *= 2;
      *buf = realloc(*buf, *cap);
      if (!*buf) { perror("score"); exit(2); }
   }
   memcpy(*buf + *len, line, (size_t)n);
   *len += (size_t)n;
   (*buf)[*len] = 0;
}

static void score_addf(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   buf_addf(&score_buf, &score_len, &score_cap, fmt, ap);
   va_end(ap);
}

static void count_addf(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   buf_addf(&count_buf, &count_len, &count_cap, fmt, ap);
   va_end(ap);
}

/* The rebuild report: what each context's journal rebuilt into, and every entry the rebuild could
 * not use. Its own buffer and not the score's, because the two answer different questions -- the
 * score is about the pixels this stream drew and the report is about whether they could be drawn
 * again after a resume -- and a corpus that legitimately loses an entry must not have to rewrite
 * its pixels to say so. */
static char *rb_buf;
static size_t rb_len, rb_cap;

static void rb_addf(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   buf_addf(&rb_buf, &rb_len, &rb_cap, fmt, ap);
   va_end(ap);
}

/* Line by line, positional: the score is ordered, so a line that moved is as much a difference as
 * a line that changed. */
static void diff_lines(const char *a, const char *b)
{
   while (*a || *b) {
      const char *ae = strchr(a, '\n'), *be = strchr(b, '\n');
      size_t an = ae ? (size_t)(ae - a) : strlen(a);
      size_t bn = be ? (size_t)(be - b) : strlen(b);
      if (an != bn || memcmp(a, b, an)) {
         if (an) fprintf(stderr, "  - %.*s\n", (int)an, a);
         if (bn) fprintf(stderr, "  + %.*s\n", (int)bn, b);
      }
      a = ae ? ae + 1 : a + an;
      b = be ? be + 1 : b + bn;
   }
}


/* The payload records name the COPY_TRANSFER3D's DESTINATION, and the copy path never reads the
 * destination's iov -- it reads the SOURCE's. vrend captures the bytes inside
 * vrend_renderer_transfer_write_iov, which the copy reaches as
 * transfer_write_iov(dst_res, src_res->iov, ...), so the payload is keyed by dst while the
 * offset and the bytes both belong to src. Seeding the destination therefore puts every
 * copy-fed upload in a buffer nothing reads and leaves the real source at zeros, which is
 * indistinguishable from a draw whose vertices are legitimately blank.
 *
 * A record carries no source handle, but it does not need one: command records are emitted
 * AFTER their dispatch, so the next CMD in the same context is the command that produced this
 * payload. Reading the source handle out of it needs no recapture. */
static int copy_src_of_next_cmd(const uint8_t *blob, size_t flen, size_t p,
                                int want_ctx, uint32_t want_dst, uint64_t want_off,
                                uint32_t *src_handle, uint32_t *mismatches)
{
   for (; p + sizeof(struct rec_hdr) <= flen; ) {
      struct rec_hdr h;
      memcpy(&h, blob + p, sizeof h);
      if (h.total_len < sizeof h || p + h.total_len > flen) return 0;
      if (h.ctx_id == (uint16_t)want_ctx && h.type == T_CMD) {
         if (h.cmd != VIRGL_CCMD_COPY_TRANSFER3D) return 0;
         const uint32_t *dw = (const uint32_t *)(blob + p + sizeof h + (size_t)h.aux_count * 4);
         if (h.payload_len / 4 <= VIRGL_COPY_TRANSFER3D_SRC_RES_OFFSET) return 0;
         /* Prove this copy is the producer rather than assuming it. The record is keyed by the
          * copy's destination and carries the copy's source offset, so both must agree. Without
          * the check an API-path transfer -- which has no following command of its own -- would
          * silently adopt the next unrelated copy and land its bytes in a stranger's buffer, a
          * drift that grows over a run and would be easy to mistake for the fault under study. */
         if (dw[VIRGL_COPY_TRANSFER3D_DST_RES_HANDLE] != want_dst
             || dw[VIRGL_COPY_TRANSFER3D_SRC_RES_OFFSET] != (uint32_t)want_off) {
            (*mismatches)++;
            return 0;
         }
         *src_handle = dw[VIRGL_COPY_TRANSFER3D_SRC_RES_HANDLE];
         return 1;
      }
      p += h.total_len;
   }
   return 0;
}

/* The IOSurface leg. The readback above reads a resource's TEXTURE; this reads the display
 * surface that texture renders into, which is what the present actually shows. On this stack they
 * are not the same path -- the scanout is an EGL_IOSURFACE_LIMINA EGLImage, so the surface IS the
 * framebuffer's storage -- and a port can get one right while getting the other wrong. For venus
 * there is no choice at all: a scanout blob has no CPU transfer_read, and the surface is the only
 * place its pixels exist.
 *
 * The id is deliberately NOT in the score. IOSurface ids are host-private, recycled the instant a
 * surface dies, and free to change across a snapshot restore -- pinning one would pin a number no
 * implementation owes us. What is pinned is that the resource IS backed, and what it contains.
 *
 * sync first, and only here: the blit-and-wait is a CLASSIC vrend operation (the VMM calls it on
 * RESOURCE_FLUSH for ctx 0 only). A venus blob renders into its surface directly and must never
 * be synced. */
/* A planar surface, read plane by plane, because nothing else can read it at all.
 *
 * `read_iosurface` copies a surface out as BGRA, which is what a scanout is. A composite decode
 * target is not: it is two 4:2:0 planes, and asking for BGRA gets -22 on both renderers -- which
 * agrees, and measures nothing. Every corpus that decodes into the composite shape therefore
 * scored its decode targets as `read-failed` and gated only what surrounds them.
 *
 * The id is used to find the surface and is still not scored: ids are host-private and recycled,
 * so pinning one pins a number no implementation owes us. Nor is the plane's PITCH scored -- it
 * is the allocator's, Metal-aligned and free to differ -- so only the tight rows are hashed, the
 * picture and none of the padding. The plane's own extent and element size are facts about the
 * format and the resource, so those are pinned.
 *
 * Returns false if this is not a planar surface, and the caller falls back to the BGRA read.
 */
static bool score_iosurface_planes(uint32_t handle, int sr)
{
   uint32_t id = 0;
   if (virgl_renderer_resource_get_iosurface_id(handle, &id) != 0 || !id)
      return false;
   IOSurfaceRef surf = IOSurfaceLookup((IOSurfaceID)id);
   if (!surf)
      return false;
   size_t planes = IOSurfaceGetPlaneCount(surf);
   if (planes < 2) {
      CFRelease(surf);
      return false;
   }
   if (IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
      CFRelease(surf);
      score_addf("iosurface res=%u planes=%zu sync=%d lock-failed\n", handle, planes, sr);
      return true;
   }
   for (size_t p = 0; p < planes; p++) {
      size_t pw = IOSurfaceGetWidthOfPlane(surf, p);
      size_t ph = IOSurfaceGetHeightOfPlane(surf, p);
      size_t bpe = IOSurfaceGetBytesPerElementOfPlane(surf, p);
      size_t pitch = IOSurfaceGetBytesPerRowOfPlane(surf, p);
      const uint8_t *base = IOSurfaceGetBaseAddressOfPlane(surf, p);
      size_t row = pw * bpe;
      if (!base || !row || row > pitch) {
         score_addf("iosurface res=%u plane=%zu unreadable\n", handle, p);
         continue;
      }
      uint64_t hash = 1469598103934665603ull;
      size_t ink = 0;
      for (size_t y = 0; y < ph; y++) {
         const uint8_t *r = base + y * pitch;
         for (size_t i = 0; i < row; i++) {
            hash ^= r[i];
            hash *= 1099511628211ull;
            if (r[i]) ink++;
         }
      }
      score_addf("iosurface res=%u plane=%zu %zux%zu bpe=%zu sync=%d hash=%016llx ink=%zu/%zu\n",
                 handle, p, pw, ph, bpe, sr, (unsigned long long)hash, ink, row * ph);
      scored_ink |= ink != 0;
   }
   IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
   CFRelease(surf);
   return true;
}

static void score_iosurface(uint32_t handle, uint32_t w, uint32_t h)
{
   size_t need = (size_t)w * h * 4;
   int sr = virgl_renderer_resource_sync_iosurface(handle);
   if (score_iosurface_planes(handle, sr))
      return;
   uint8_t *sp = calloc(1, need);
   if (!sp) { fprintf(stderr, "OOM\n"); exit(2); }

   /* dst_stride is BYTES, not pixels. Passing the pixel width here is a known bug shape: the
    * image comes out quarter-width, tiled four across and squashed four down. */
   int ir = virgl_renderer_resource_read_iosurface(handle, sp, w * 4, h);
   if (ir == 0) {
      size_t ink = 0;
      for (size_t i = 0; i < need; i += 4)
         if (sp[i] | sp[i + 1] | sp[i + 2] | sp[i + 3]) ink++;
      uint64_t hash = 1469598103934665603ull;
      for (size_t i = 0; i < need; i++) { hash ^= sp[i]; hash *= 1099511628211ull; }
      score_addf("iosurface res=%u %ux%u sync=%d hash=%016llx ink=%zu/%zu\n",
                 handle, w, h, sr, (unsigned long long)hash, ink, need / 4);
      scored_ink |= ink != 0;
      /* The surface's pixels, for the same reason the texture readback dumps: a hash says
       * two frames differ, and only the frame says where. */
      const char *dir = getenv("REPLAY_DUMP_DIR");
      const char *only = getenv("REPLAY_DUMP_W");
      if (dir && (!only || w == (uint32_t)atoi(only))) {
         char fn[512];
         snprintf(fn, sizeof fn, "%s/iosurface-res%u_%ux%u.rgba", dir, handle, w, h);
         FILE *f = fopen(fn, "wb");
         if (f) { fwrite(sp, 1, need, f); fclose(f); }
      }
   } else {
      score_addf("iosurface res=%u %ux%u sync=%d read-failed=%d\n", handle, w, h, sr, ir);
   }
   free(sp);
}

/* Define what the score will read, before anything else can leave undefined bytes there.
 *
 * A texture's contents are undefined until something writes them -- glTexStorage2D and
 * glTexImage2D with NULL both say so -- and this driver does not zero them, so an unwritten
 * resource reads back whatever the last tenant of that GPU memory left. Hashing that grades the
 * allocator, not the renderer: two implementations that agree on every pixel they actually draw
 * still differ on every resource neither of them ever wrote, and one that agrees today agrees by
 * luck. This is the tool that tells the two apart. A score line that moves under --zero-new was
 * never the renderer's answer.
 *
 * It zeroes through exactly the call shape score_resource() reads with: same box, same
 * 4-bytes-per-pixel stride, same level, so precisely the region the score observes is defined,
 * and anything the renderer genuinely writes overwrites it.
 *
 * Measured before it became the default: on vrend, blit, sampled and surface it moves not one
 * line, so it disturbs nothing any renderer actually draws. On the VP9 corpus it moves 224
 * resources -- decode target planes the guest allocates and never decodes into -- and with it on
 * the two renderers agree on every one of them, where before they disagreed on 22. --no-zero-new
 * turns it off, which is how you look at what was there instead.
 *
 * A format that refuses the transfer -- depth/stencil, compressed -- is left alone: a refusal
 * here is not a failure of the run.
 */
/* The four planar YUV formats, which must NOT be zeroed -- see zero_resource. */
static bool format_is_planar_yuv(uint32_t format)
{
   return format == 163 || format == 165 || format == 166 || format == 167;
}

/* A tight w x h image's row stride and total size, from the format and nothing else.
 *
 * This used to be `w * h * 4` in two places, under the belief -- written into the comment -- that
 * four bytes per texel "is what every format this scores actually is". A browser corpus disproved
 * it with an R32G32B32A32_FLOAT render target: the transfer then offers a stride a quarter of the
 * minimum, vrend refuses it, the refusal latches the context's in_error, and every later submit in
 * the corpus is dropped. The damage reads as a renderer that stopped drawing, which is the
 * expensive kind of harness bug.
 *
 * false means this tree's format table does not describe the format -- compressed, subsampled, or
 * simply unknown. The caller must then leave the resource alone. It must NOT fall back to a
 * guess, because a guess is exactly what this replaces. */
static bool format_geometry(uint32_t format, uint32_t w, uint32_t h, uint32_t *stride, size_t *size)
{
   if (format > VREND_REPLAY_MAX_FORMAT)
      return false;
   const struct vrend_replay_format *f = &vrend_replay_formats[format];
   if (!f->block_bytes || f->block_w != 1 || f->block_h != 1)
      return false;
   *stride = w * f->block_bytes;
   *size = (size_t)*stride * h;
   return true;
}

static void zero_resource(uint32_t handle, uint32_t format, uint32_t width, uint32_t height,
                          int ctx)
{
   uint32_t w = width ? width : 1, h = height ? height : 1;
   uint32_t stride;
   size_t need;
   if (!format_geometry(format, w, h, &stride, &need))
      return;
   uint8_t *zeros = calloc(1, need);
   if (!zeros) return;
   struct iovec ziov = { .iov_base = zeros, .iov_len = need };
   struct virgl_box box = { .x = 0, .y = 0, .z = 0, .w = w, .h = h, .d = 1 };
   (void)virgl_renderer_transfer_write_iov(handle, (uint32_t)ctx, 0, stride, 0, &box,
                                           0, &ziov, 1);
   free(zeros);
}

/* Carry a blob's recorded bytes from its backing store into its texture.
 *
 * Both legs need this and for opposite reasons, which is why it is one unconditional path and
 * never a per-renderer branch. The C re-reads the backing at every sampling batch on its own
 * (vrend_resource_refresh_guest_pixels), so a write here is idempotent -- it asserts the same
 * bytes the refresh is about to assert. virglrs has no such refresh, so this write is the only
 * way its texture ever holds anything but the zeros the upgrade left. Feeding the backing alone
 * would score content on one leg and zeros on the other.
 *
 * The guest's own stride is used, not a packed row: this corpus declares 4096 for a 500-wide
 * 8-byte format whose packed row is 4000, and the difference is exactly the class of bug the
 * content fixture exists to catch. */
static uint32_t blob_feed(int ctx)
{
   uint32_t fed = 0;
   for (uint32_t i = 0; i < backing_n; i++) {
      struct backing *b = backings[i];
      if (!b->is_blob || !b->typed || !b->dirty || !b->live || b->declined)
         continue;
      uint32_t w = b->t_width ? b->t_width : 1, h = b->t_height ? b->t_height : 1;
      uint32_t packed;
      size_t need;
      /* A refusal is a result, not an absence of one. Both declines below are permanent for the
       * blob and leave it reading ink=0, so the score has to say which blob and why -- a `fed`
       * count short of `records` says only that something went unfed. */
      if (!format_geometry(b->t_format, w, h, &packed, &need)) {
         score_addf("blob res=%u %ux%u fmt=%u declined=block-size-unknown\n",
                    b->handle, w, h, b->t_format);
         b->declined = true;
         continue;
      }
      const uint32_t stride = b->t_stride ? b->t_stride : packed;
      /* The guest's layout must fit in the pages it declared, or the write walks off the end of
       * the backing. A capture that says otherwise is a capture to fix, not to clamp. */
      if ((size_t)stride * h > b->size) {
         score_addf("blob res=%u %ux%u stride=%u declined=layout-exceeds-backing size=%zu\n",
                    b->handle, w, h, stride, b->size);
         fprintf(stderr, "blob %u: stride %u x %u rows exceeds its %zu-byte backing\n",
                 b->handle, stride, h, b->size);
         b->declined = true;
         continue;
      }
      struct iovec biov = { .iov_base = b->mem, .iov_len = b->size };
      struct virgl_box box = { .x = 0, .y = 0, .z = 0, .w = w, .h = h, .d = 1 };
      if (!virgl_renderer_transfer_write_iov(b->handle, (uint32_t)ctx, 0, stride, 0, &box,
                                             0, &biov, 1))
         fed++;
      b->dirty = false;
   }
   return fed;
}

static void score_resource(const struct res_ev *ev)
{
   /* A planar resource is scored through its IOSurface, plane by plane, and asking for it as one
    * RGBA texture is asking for the thing that has no answer: the renderer refuses, and the C
    * refuses by poisoning the context, which drops every later submit in the corpus -- 893 of
    * them on the composite leg. The sweep must not manufacture the guest-hostile request it
    * exists to observe the absence of. */
   if (format_is_planar_yuv(ev->format))
      return;
   uint32_t w = ev->width ? ev->width : 1, h = ev->height ? ev->height : 1;
   /* Ask for the resource at ITS bytes per texel, not at four. Asking at four for a 16-byte
    * format offers a stride a quarter of the minimum, which vrend refuses -- and that refusal
    * used to be recorded as `failed=22`, a renderer's answer to a question the sweep had asked
    * wrong. A format this tree cannot size is declined openly instead, because a request the
    * sweep knows it cannot phrase is not a result about the renderer. */
   uint32_t stride;
   size_t need;
   if (!format_geometry(ev->format, w, h, &stride, &need)) {
      score_addf("readback res=%u %ux%u fmt=%u declined=block-size-unknown\n",
                 ev->handle, w, h, ev->format);
      return;
   }
   const uint32_t texel = stride / w;
   uint8_t *px = calloc(1, need);
   struct iovec riov = { .iov_base = px, .iov_len = need };
   struct virgl_box box = { .x = 0, .y = 0, .z = 0, .w = w, .h = h, .d = 1 };

   /* No force_ctx_0 here. The context path does its own switch -- transfer_read_iov with a
    * non-zero ctx_id reaches vrend_renderer_transfer_internal, which calls vrend_hw_switch_context
    * on the replayed context and leaves it current -- so forcing ctx0 first only added a switch
    * away and back. */
   int rr = virgl_renderer_transfer_read_iov(ev->handle, (uint32_t)want_ctx, 0, stride, 0,
                                             &box, 0, &riov, 1);
   if (rr) {
      /* A refusal is a result, not an absence of one. Dropping it here meant a renderer that
       * started refusing what the reference serves scored byte-identical -- the one class of
       * divergence the sweep could not see, and one that a hardening change makes real rather
       * than hypothetical. The errno is the whole content: what it refused, and with which
       * answer, is what a guest would have been told. */
      score_addf("readback res=%u %ux%u failed=%d\n", ev->handle, w, h, rr);
      fprintf(stderr, "readback of res %u failed: %d\n", ev->handle, rr);
      free(px);
      return;
   }

   /* Ink is counted per TEXEL, so the denominator is the texel count whatever the format's
    * width -- stepping four bytes at a time counted a 16-byte texel four times. */
   size_t ink = 0;
   for (size_t i = 0; i < need; i += texel) {
      uint8_t any = 0;
      for (uint32_t b = 0; b < texel; b++) any |= px[i + b];
      if (any) ink++;
   }

   /* FNV-1a over the readback. This line IS the golden: a content hash compares two
    * implementations without carrying megabytes of reference pixels in the tree, and it is
    * stable across runs in a way an ink count is not (ink says only "something rendered"). */
   uint64_t hash = 1469598103934665603ull;
   for (size_t i = 0; i < need; i++) { hash ^= px[i]; hash *= 1099511628211ull; }

   score_addf("res=%u %ux%u hash=%016llx ink=%zu/%zu\n",
              ev->handle, w, h, (unsigned long long)hash, ink, need / texel);

   /* An ink COUNT cannot say which offscreen is the header and which is the body, and that
    * mapping is what any verdict about "the title is lost" rests on. Dump the pixels and look. */
   const char *dir = getenv("REPLAY_DUMP_DIR");
   const char *only = getenv("REPLAY_DUMP_W");
   if (dir && (!only || w == (uint32_t)atoi(only))) {
      char fn[512];
      snprintf(fn, sizeof fn, "%s/res%u_%ux%u.rgba", dir, ev->handle, w, h);
      FILE *f = fopen(fn, "wb");
      if (f) { fwrite(px, 1, need, f); fclose(f); }
      /* Appended in stream order, so the manifest records the SEQUENCE of readbacks as well as
       * their contents -- two implementations that render the same pixels in a different order
       * are not the same implementation. */
      snprintf(fn, sizeof fn, "%s/manifest.txt", dir);
      f = fopen(fn, "a");
      if (f) {
         fprintf(f, "res=%u %ux%u hash=%016llx ink=%zu/%zu\n",
                 ev->handle, w, h, (unsigned long long)hash, ink, need / 4);
         fclose(f);
      }
   }
   scored = 1;
   /* Any ink anywhere clears the floor, so |= and never =. Plain assignment made the
    * verdict the LAST resource's ink, and the sweep ends on whatever the resource log
    * ends on -- a corpus that rendered a whole desktop failed the floor because its
    * final unref happened to be an empty scratch texture. */
   scored_ink |= ink != 0;
   free(px);
}

/* The classic capsets, field by field. A word each for the masks and arrays, in hex, so a diff
 * names the field that moved and the bit that moved in it. */
static void caps_words(FILE *f, const char *name, const void *p, size_t bytes)
{
   const uint32_t *w = p;
   fprintf(f, "%s =", name);
   for (size_t i = 0; i < bytes / 4; i++) fprintf(f, " %08x", w[i]);
   fprintf(f, "\n");
}

/* virtgpu's VIRTGPU_DRM_CAPSET_VIRGL and _VIRGL2: the ids a guest names the sets by. */
enum { CAPSET_VIRGL = 1, CAPSET_VIRGL2 = 2 };

static int dump_caps(const char *path)
{
   FILE *f = fopen(path, "w");
   if (!f) { perror(path); return 2; }
   for (uint32_t set = CAPSET_VIRGL; set <= CAPSET_VIRGL2; set++) {
      uint32_t max_ver = 0, max_size = 0;
      virgl_renderer_get_cap_set(set, &max_ver, &max_size);
      fprintf(f, "set %u: max_ver=%u max_size=%u\n", set, max_ver, max_size);
      if (!max_size) continue;
      union virgl_caps caps;
      memset(&caps, 0xa5, sizeof(caps));   /* so an unfilled byte shows */
      virgl_renderer_fill_caps(set, max_ver, &caps);
      const struct virgl_caps_v1 *v1 = &caps.v1;
#define U(field) fprintf(f, "  %s = %u\n", #field, (unsigned)v1->field)
#define W(field) caps_words(f, "  " #field, &v1->field, sizeof(v1->field))
      U(max_version); W(sampler); W(render); W(depthstencil); W(vertexbuffer); W(bset);
      U(glsl_level); U(max_texture_array_layers); U(max_streamout_buffers);
      U(max_dual_source_render_targets); U(max_render_targets); U(max_samples); U(prim_mask);
      U(max_tbo_size); U(max_uniform_blocks); U(max_viewports); U(max_texture_gather_components);
#undef U
#undef W
      if (set != CAPSET_VIRGL2) continue;
      const struct virgl_caps_v2 *v2 = &caps.v2;
#define U(field) fprintf(f, "  %s = %u\n", #field, (unsigned)v2->field)
#define I(field) fprintf(f, "  %s = %d\n", #field, (int)v2->field)
#define F(field) fprintf(f, "  %s = %g\n", #field, (double)v2->field)
#define W(field) caps_words(f, "  " #field, &v2->field, sizeof(v2->field))
      F(min_aliased_point_size); F(max_aliased_point_size); F(min_smooth_point_size);
      F(max_smooth_point_size); F(min_aliased_line_width); F(max_aliased_line_width);
      F(min_smooth_line_width); F(max_smooth_line_width); F(max_texture_lod_bias);
      U(max_geom_output_vertices); U(max_geom_total_output_components); U(max_vertex_outputs);
      U(max_vertex_attribs); U(max_shader_patch_varyings); I(min_texel_offset); I(max_texel_offset);
      I(min_texture_gather_offset); I(max_texture_gather_offset); U(texture_buffer_offset_alignment);
      U(uniform_buffer_offset_alignment); U(shader_buffer_offset_alignment); W(capability_bits);
      W(sample_locations); U(max_vertex_attrib_stride); U(max_shader_buffer_frag_compute);
      U(max_shader_buffer_other_stages); U(max_shader_image_frag_compute);
      U(max_shader_image_other_stages); U(max_image_samples); U(max_compute_work_group_invocations);
      U(max_compute_shared_memory_size); W(max_compute_grid_size); W(max_compute_block_size);
      U(max_texture_2d_size); U(max_texture_3d_size); U(max_texture_cube_size);
      U(max_combined_shader_buffers); W(max_atomic_counters); W(max_atomic_counter_buffers);
      U(max_combined_atomic_counters); U(max_combined_atomic_counter_buffers);
      U(host_feature_check_version); W(supported_readback_formats); W(scanout); W(capability_bits_v2);
      U(max_video_memory); fprintf(f, "  renderer = %.64s\n", v2->renderer); F(max_anisotropy);
      U(max_texture_samplers); W(supported_multisample_formats); W(max_const_buffer_size);
      U(num_video_caps); W(video_caps); U(max_uniform_block_size); U(max_tcs_outputs);
      U(max_tes_outputs); W(max_shader_storage_blocks);
#undef U
#undef I
#undef F
#undef W
   }
   fclose(f);
   printf("replay: caps written to %s\n", path);
   return 0;
}

/* A cursor over a "VRJ1" journal export, refusing anything that does not fit in what it was
 * given. The replayer reads a blob the renderer under test produced, so a malformed one is a
 * result to report, never something to walk off the end of. */
struct vrj {
   const uint8_t *p;
   size_t left;
   bool bad;
};

static uint32_t vrj_u32(struct vrj *c)
{
   if (c->left < 4) { c->bad = true; return 0; }
   uint32_t v;
   memcpy(&v, c->p, 4);
   c->p += 4;
   c->left -= 4;
   return v;
}

/* One entry of a journal, pointing into the journal it was parsed from: the bytes from its kind
 * dword to the end of its last chunk, which is everything two journals have to agree on. The seq
 * pair in front of it is left out because the rebuild renumbers it -- the rebuilt context records
 * from 1 while the original numbers from wherever its guest got to. */
struct vrj_ent {
   const uint8_t *raw;
   size_t len;
};

/* Split a journal into its entries, or NULL if it is not one. */
static struct vrj_ent *vrj_entries(const void *j, uint64_t len, uint32_t *n_out, uint32_t *ver)
{
   struct vrj c = { j, (size_t)len, false };
   uint32_t magic = vrj_u32(&c);
   *ver = vrj_u32(&c);
   uint32_t n = vrj_u32(&c);
   (void)vrj_u32(&c);                                 /* reserved */
   if (magic != 0x314a5256u || c.bad)
      return NULL;
   struct vrj_ent *e = calloc(n ? n : 1, sizeof *e);
   if (!e) { perror("rebuild"); exit(2); }
   for (uint32_t i = 0; i < n; i++) {
      (void)vrj_u32(&c); (void)vrj_u32(&c);           /* seq, renumbered by the rebuild */
      const uint8_t *start = c.p;
      (void)vrj_u32(&c); (void)vrj_u32(&c);           /* kind, sub */
      uint32_t nc = vrj_u32(&c);
      for (uint32_t k = 0; k < nc; k++) {
         uint32_t dwords = vrj_u32(&c);
         for (uint32_t d = 0; d < dwords; d++) (void)vrj_u32(&c);
      }
      if (c.bad) { free(e); return NULL; }
      e[i].raw = start;
      e[i].len = (size_t)(c.p - start);
   }
   *n_out = n;
   return e;
}

/* Name a dropped entry in the report: its kind, its sub-context, its size, and its leading
 * dwords, which is where the object handle and the resource it names live. The prefix and not
 * the whole entry, because a shader create runs to hundreds of dwords and the pin has to fit on
 * a line -- the size guards what the prefix does not reach. */
static void rb_drop_line(const struct vrj_ent *e)
{
   struct vrj c = { e->raw, e->len, false };
   uint32_t kind = vrj_u32(&c), sub = vrj_u32(&c), nc = vrj_u32(&c);
   uint32_t total = 0;
   for (uint32_t k = 0; k < nc; k++) {
      uint32_t dwords = vrj_u32(&c);
      total += dwords;
      for (uint32_t d = 0; d < dwords; d++) (void)vrj_u32(&c);
   }
   rb_addf("  drop kind=%u sub=%u chunks=%u dwords=%u:", kind, sub, nc, total);
   struct vrj w = { e->raw, e->len, false };
   (void)vrj_u32(&w); (void)vrj_u32(&w); (void)vrj_u32(&w);
   uint32_t first = nc ? vrj_u32(&w) : 0;
   uint32_t show = first < 8 ? first : 8;
   for (uint32_t d = 0; d < show && !w.bad; d++)
      rb_addf(" %08x", vrj_u32(&w));
   rb_addf("\n");
}

/* -------- the classic contents fixed point -------- *
 *
 * The journal gate above proves the rebuilt context has the same OBJECTS. It says nothing about
 * what is in them: a rebuilt world is structurally sound and blank, which is the gray-blocks
 * state a restore is supposed to avoid. This is the other half -- export a context's resource
 * contents, put them into the rebuilt context, export that one, and require the two blobs to
 * agree entry for entry.
 *
 * Blob layout (`VRCC`): u32 magic, u32 version, then per entry u32 res_id, level, w, h, d, size,
 * followed by `size` bytes.
 */

struct vrcc_ent {
   uint32_t res_id, level, w, h, d, size;
   const uint8_t *bytes;
};

/* Split a content blob into its entries, or NULL if it is not one. */
static struct vrcc_ent *vrcc_entries(const void *b, uint64_t len, uint32_t *n_out)
{
   const uint8_t *p = b, *end = p + len;
   uint32_t magic, version;
   if (len < 8)
      return NULL;
   memcpy(&magic, p, 4);
   memcpy(&version, p + 4, 4);
   if (magic != 0x43435256u || version != 1u)
      return NULL;
   p += 8;
   uint32_t cap = 16, n = 0;
   struct vrcc_ent *e = calloc(cap, sizeof *e);
   if (!e) { perror("content"); exit(2); }
   while (p + 24 <= end) {
      uint32_t h[6];
      memcpy(h, p, 24);
      p += 24;
      if ((uint64_t)(end - p) < h[5]) { free(e); return NULL; }
      if (n == cap) {
         cap *= 2;
         struct vrcc_ent *ne = realloc(e, cap * sizeof *e);
         if (!ne) { perror("content"); exit(2); }
         e = ne;
      }
      e[n].res_id = h[0];
      e[n].level = h[1];
      e[n].w = h[2];
      e[n].h = h[3];
      e[n].d = h[4];
      e[n].size = h[5];
      e[n].bytes = p;
      n++;
      p += h[5];
   }
   if (p != end) { free(e); return NULL; }
   *n_out = n;
   return e;
}

static bool vrcc_all_zero(const struct vrcc_ent *e)
{
   for (uint32_t i = 0; i < e->size; i++)
      if (e->bytes[i])
         return false;
   return true;
}

/* A copy of a content blob with every entry's bytes inverted: same resources, same layout,
 * content nothing could have produced.
 *
 * Without it the fixed point is vacuous. A classic resource is global and the rebuilt context is
 * attached to the same ones, so exporting from the rebuilt context re-reads the very storage the
 * source read -- and a restore that wrote nothing at all would still compare equal. Scrubbing
 * first means the bytes that come back can only have come from the restore. */
static void *vrcc_scrub(const void *b, uint64_t len)
{
   uint8_t *c = malloc((size_t)len);
   if (!c) { perror("content"); exit(2); }
   memcpy(c, b, (size_t)len);
   const uint8_t *end = c + len;
   uint8_t *p = c + 8;
   while (p + 24 <= end) {
      uint32_t sz;
      memcpy(&sz, p + 20, 4);
      p += 24;
      if ((uint64_t)(end - p) < sz) break;
      for (uint32_t i = 0; i < sz; i++)
         p[i] = (uint8_t)~p[i];
      p += sz;
   }
   return c;
}

/* Whether two content blobs hold the same bytes for the entries they share.
 *
 * `scored` says whether anything was actually compared: a blob with no entries, or none with any
 * bytes, shares nothing with anything and reads as "the same" -- which is the answer that would
 * make the scrub check below accuse an empty capture of writing nothing. */
static bool vrcc_same(const void *a, uint64_t a_len, const void *b, uint64_t b_len, bool *scored)
{
   uint32_t n_a = 0, n_b = 0;
   struct vrcc_ent *ea = vrcc_entries(a, a_len, &n_a);
   struct vrcc_ent *eb = vrcc_entries(b, b_len, &n_b);
   bool same = ea && eb;
   *scored = false;
   for (uint32_t i = 0; ea && i < n_a; i++)
      if (ea[i].size)
         *scored = true;
   for (uint32_t i = 0; same && i < n_a; i++)
      for (uint32_t j = 0; j < n_b; j++)
         if (eb[j].res_id == ea[i].res_id && eb[j].level == ea[i].level) {
            if (eb[j].size != ea[i].size || memcmp(eb[j].bytes, ea[i].bytes, ea[i].size))
               same = false;
            break;
         }
   free(ea); free(eb);
   return same;
}

/* The fixed point itself. Every entry the source exported must come back from the rebuilt
 * context, at the same res_id and level, byte for byte. An entry that did not come back is a
 * resource the restore did not reach -- reported, and the caller decides against the pin. */
static bool contents_agree(const void *a, uint64_t a_len, const void *b, uint64_t b_len,
                           uint32_t src, uint32_t dst, uint32_t *missing, uint32_t *deviations,
                           const struct res_ev *res, uint32_t n_res)
{
   uint32_t n_a = 0, n_b = 0;
   struct vrcc_ent *ea = vrcc_entries(a, a_len, &n_a);
   struct vrcc_ent *eb = vrcc_entries(b, b_len, &n_b);
   if (!ea || !eb) {
      fprintf(stderr, "content: ctx %u: not a pair of content blobs\n", src);
      free(ea); free(eb);
      return false;
   }

   /* How many entries carry nothing but zeros. Not a failure -- a scratch buffer the guest never
    * wrote is legitimately zero -- but a run where EVERY entry is zero is a gate comparing
    * nothing to nothing, and that is worth saying out loud rather than reading as a pass. */
   uint32_t zeros = 0, bytes = 0;
   for (uint32_t i = 0; i < n_a; i++) {
      zeros += vrcc_all_zero(&ea[i]) ? 1 : 0;
      bytes += ea[i].size;
   }

   bool ok = true;
   uint32_t differ = 0;
   for (uint32_t i = 0; i < n_a; i++) {
      const struct vrcc_ent *m = NULL;
      for (uint32_t j = 0; j < n_b; j++)
         if (eb[j].res_id == ea[i].res_id && eb[j].level == ea[i].level) { m = &eb[j]; break; }
      if (!m) {
         (*missing)++;
         continue;
      }
      if (m->size != ea[i].size || memcmp(m->bytes, ea[i].bytes, ea[i].size)) {
         /* Where and how much, not just that: a whole-buffer difference, a tail, and a
          * channel-order swap are three different bugs and the line has to tell them apart. */
         uint32_t first = 0, differing = 0;
         if (m->size == ea[i].size) {
            while (first < ea[i].size && m->bytes[first] == ea[i].bytes[first])
               first++;
            for (uint32_t k = 0; k < ea[i].size; k++)
               differing += m->bytes[k] != ea[i].bytes[k];
         }
         const struct res_ev *born = NULL;
         for (uint32_t k = 0; k < n_res; k++)
            if (res[k].handle == ea[i].res_id && res[k].kind != RES_UNREF)
               born = &res[k];
         rb_addf("  content res=%u level=%u %ux%ux%u fmt=%u bind=%#x target=%u differs "
                 "(%u vs %u bytes, %u from offset %u)\n", ea[i].res_id, ea[i].level, ea[i].w,
                 ea[i].h, ea[i].d, born ? born->format : 0u, born ? born->bind : 0u,
                 born ? born->target : 0u, ea[i].size, m->size, differing, first);
         differ++;
      }
   }
   /* The line goes into the pinned report, not just onto stderr: a resource that does not
    * survive the round trip is a fact about this corpus on this host -- a depth buffer whose
    * low byte does not come back is one -- and a fact that is only true of some resources is
    * pinned, the way a dropped journal entry is, rather than being argued about per run. */
   rb_addf("ctx %u: content %u entries, %u all-zero, %u missing, %u differ\n", src, n_a, zeros,
           *missing, differ);
   fprintf(stderr, "content: ctx %u -> %u: %u entries (%u bytes, %u all-zero), %u missing, "
           "%u different\n", src, dst, n_a, bytes, zeros, *missing, differ);
   *deviations += *missing + differ;

   free(ea); free(eb);
   return ok;
}

/* Whether the rebuilt journal describes the same world, and what it lost on the way.
 *
 * The rebuilt journal must be a SUBSEQUENCE of the source. A rebuild is allowed to lose an entry
 * -- a create over a resource the guest has since destroyed is the case a real capture reaches,
 * and dropping it is what the renderer owes a guest that did that -- but never to invent one or
 * to reorder what it kept. So the walk is greedy: on a match both advance, on a mismatch the
 * source entry is recorded as dropped and only the source advances. An entry whose CONTENT
 * changed reads as a drop and takes every later entry with it, ending with the rebuilt journal
 * outrunning the source, which fails here and is not something a pin can accept.
 *
 * Whether the drops themselves are acceptable is not decided here: they go into the report, and
 * the pin in `--rebuild-expect` is what says which ones this corpus is known to make. With no
 * pin, a drop is a failure. */
static bool journals_agree(const void *a, uint64_t a_len, const void *b, uint64_t b_len,
                           uint32_t ctx, uint32_t *dropped)
{
   uint32_t n_a = 0, n_b = 0, ver_a = 0, ver_b = 0;
   struct vrj_ent *ea = vrj_entries(a, a_len, &n_a, &ver_a);
   struct vrj_ent *eb = vrj_entries(b, b_len, &n_b, &ver_b);
   if (!ea || !eb || ver_a != ver_b) {
      fprintf(stderr, "rebuild: ctx %u: not a pair of journals\n", ctx);
      free(ea); free(eb);
      return false;
   }

   rb_addf("ctx %u: %u in, %u out\n", ctx, n_a, n_b);
   bool ok = true;
   uint32_t i = 0, j = 0;
   while (i < n_a && j < n_b) {
      if (ea[i].len == eb[j].len && !memcmp(ea[i].raw, eb[j].raw, ea[i].len)) {
         i++; j++;
         continue;
      }
      rb_drop_line(&ea[i]);
      (*dropped)++;
      i++;
   }
   if (j < n_b) {
      fprintf(stderr, "rebuild: ctx %u: the rebuild kept %u entries the journal does not have\n",
              ctx, n_b - j);
      ok = false;
   }
   for (; i < n_a; i++) {
      rb_drop_line(&ea[i]);
      (*dropped)++;
   }

   free(ea); free(eb);
   return ok;
}

int main(int argc, char **argv)
{
   const char *path = NULL;
   int loops = 1;
   bool nodraw = false;
   bool nofeed = false;
   /* Exercise only what a skeleton owes: init, context create, resource create. No submits, no
    * transfers, no scoring. This is P1's gate -- a renderer that gets through it has a working
    * ABI, resource table and context table, which is all a skeleton claims. */
   bool smoke = false;
   /* --rebuild: after the stream, rebuild each context from its own journal and check the
    * rebuild describes the same world. Off by default: it makes contexts and GL objects the
    * corpus never asked for, and every pinned score was recorded without it. */
   bool rebuild = false;
   unsigned rebuild_failed = 0;
   uint32_t rebuild_dropped = 0;
   unsigned content_failed = 0;
   uint32_t content_deviations = 0;
   bool sweep = false;
   uint64_t draws_from = 0;
   /* --until stops the stream after a sequence number and scores what is on the surfaces THEN.
    * The end-of-run score sees only the last frame each scanout received; a frame that differs
    * between two arms may have gone wrong many frames earlier, and this is how the point where
    * they part is bisected. */
   uint64_t until = 0;
   uint32_t sweep_w = 0;

   bool no_unref = getenv("REPLAY_NO_UNREF") != NULL;
   uint32_t watch = getenv("REPLAY_WATCH") ? (uint32_t)atoi(getenv("REPLAY_WATCH")) : 0;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
         for (const char *c = argv[++i]; *c; ) {
            if (n_ctx == MAX_CTX) { fprintf(stderr, "--ctx takes at most %d\n", MAX_CTX); return 2; }
            ctx_list[n_ctx++] = atoi(c);
            while (*c && *c != ',') c++;
            if (*c == ',') c++;
         }
         if (!n_ctx) { fprintf(stderr, "--ctx wants at least one context\n"); return 2; }
         want_ctx = ctx_list[0];
      }
      else if (!strcmp(argv[i], "--sweep")) sweep = true;
      else if (!strcmp(argv[i], "--sweep-w") && i + 1 < argc)
         sweep_w = (uint32_t)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--draws-from") && i + 1 < argc)
         draws_from = strtoull(argv[++i], NULL, 10);
      else if (!strcmp(argv[i], "--until") && i + 1 < argc)
         until = strtoull(argv[++i], NULL, 10);
      else if (!strcmp(argv[i], "--loops") && i + 1 < argc) loops = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--nodraw")) nodraw = true;
      else if (!strcmp(argv[i], "--nofeed")) nofeed = true;
      else if (!strcmp(argv[i], "--smoke")) smoke = true;
      else if (!strcmp(argv[i], "--readback") && i + 1 < argc) readback_res = (uint32_t)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--no-zero-new")) zero_new = false;
      else if (!strcmp(argv[i], "--rebuild")) rebuild = true;
      else if (!strcmp(argv[i], "--rebuild-score") && i + 1 < argc) rb_score_path = argv[++i];
      else if (!strcmp(argv[i], "--rebuild-expect") && i + 1 < argc) rb_expect_path = argv[++i];
      else if (!strcmp(argv[i], "--score") && i + 1 < argc) score_path = argv[++i];
      else if (!strcmp(argv[i], "--expect") && i + 1 < argc) expect_path = argv[++i];
      else if (!strcmp(argv[i], "--caps") && i + 1 < argc) caps_path = argv[++i];
      else if (argv[i][0] != '-') path = argv[i];
   }
   if (!path) { fprintf(stderr, "usage: vrend-replay <dump> [--ctx N[,N...]] [--loops N] [--nodraw] [--draws-from SEQ] [--until SEQ]\n"); return 2; }

   FILE *f = fopen(path, "rb");
   if (!f) { perror(path); return 2; }
   fseek(f, 0, SEEK_END);
   long flen = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint8_t *blob = malloc((size_t)flen);
   if (!blob || fread(blob, 1, (size_t)flen, f) != (size_t)flen) { fprintf(stderr, "short read\n"); return 2; }
   fclose(f);

   uint32_t head[16];
   memcpy(head, blob, sizeof head);
   if (head[0] != TRACE_MAGIC) { fprintf(stderr, "not a vrend trace dump\n"); return 2; }
   if (head[1] < 2) {
      fprintf(stderr, "trace is version %u: it has no resource log, so it cannot be replayed.\n"
                      "Recapture with a tracer built from this tree.\n", head[1]);
      return 2;
   }
   if (head[13]) fprintf(stderr, "WARNING: resource log overflowed; this trace is NOT replayable\n");

   uint32_t res_n = head[12];
   struct res_ev *res = (struct res_ev *)(blob + 64);
   size_t off = 64 + (size_t)res_n * sizeof(struct res_ev);
   printf("replay: %u resource events, ring %u MB, %u evicted\n", res_n, head[2], head[6]);
   if (head[6]) fprintf(stderr, "WARNING: %u records evicted; the window does not reach the start\n", head[6]);

   /* Pick the busiest context if not told. */
   if (want_ctx < 0) {
      uint32_t count[65536];
      memset(count, 0, sizeof count);
      for (size_t p = off; p + sizeof(struct rec_hdr) <= (size_t)flen; ) {
         struct rec_hdr h;
         memcpy(&h, blob + p, sizeof h);
         if (h.total_len < sizeof h || p + h.total_len > (size_t)flen) break;
         if (h.type == T_CMD) count[h.ctx_id]++;
         p += h.total_len;
      }
      uint32_t best = 0;
      for (uint32_t i = 0; i < 65536; i++) if (count[i] > count[best]) best = i;
      want_ctx = (int)best;
      ctx_list[0] = want_ctx;
      n_ctx = 1;
      printf("replay: no --ctx given, picking ctx %d (%u commands)\n", want_ctx, count[best]);
   }

   /* USE_VIDEO is in the flag word limina passes, and it is load-bearing here rather than
    * decorative: virgl_video_init is what calls VTRegisterSupplementalVideoDecoderIfAvailable,
    * and VP9 and AV1 arrive as supplemental decoders, so without it every
    * VTDecompressionSessionCreate in a replay fails with kVTCouldNotFindVideoDecoderErr and a
    * decode corpus scores its targets empty. */
   int flags = VIRGL_RENDERER_USE_EGL | VIRGL_RENDERER_USE_SURFACELESS | VIRGL_RENDERER_USE_GLES
             | VIRGL_RENDERER_USE_VIDEO;
   /* The cookie must be non-NULL: virglrenderer rejects the vrend path outright with "invalid
    * renderer vrend callbacks" when it is null, whatever the callbacks contain. It is opaque to
    * the library and only handed back to our callbacks, so any live address will do. */
   static int cookie;
   int ret = virgl_renderer_init(&cookie, flags, &cbs);
   if (ret) { fprintf(stderr, "virgl_renderer_init failed: %d\n", ret); return 2; }
   printf("replay: virglrenderer initialised (flags 0x%x)\n", flags);

   if (caps_path)
      return dump_caps(caps_path);

   const char *name = "limina-replay";
   for (int i = 0; i < n_ctx; i++) {
      ret = virgl_renderer_context_create((uint32_t)ctx_list[i], (uint32_t)strlen(name), name);
      if (ret) { fprintf(stderr, "context_create %d failed: %d\n", ctx_list[i], ret); return 2; }
   }

   /* One probe, because a refusal's ANSWER is part of the ABI and no corpus contains a request
    * that is refused with an errno rather than with -1. This is not the manufactured request the
    * sweep must not make: it names a handle nothing created, so it touches no resource, poisons
    * no context and drops no submit -- it asks the one question a recorded guest never asks, and
    * both implementations have to give the same answer. virglrenderer answers a transfer with a
    * POSITIVE errno, which is the opposite of most of its ABI, and a port that returns -22 here
    * reads to a VMM as a different failure entirely. */
   {
      uint8_t probe[4] = { 0 };
      struct iovec piov = { .iov_base = probe, .iov_len = sizeof probe };
      struct virgl_box pbox = { .x = 0, .y = 0, .z = 0, .w = 1, .h = 1, .d = 1 };
      int pr = virgl_renderer_transfer_read_iov(0xfffffff0u, (uint32_t)want_ctx, 0, 4, 0,
                                                &pbox, 0, &piov, 1);
      score_addf("probe transfer-of-unknown-resource answered=%d\n", pr);
   }

   /* Score every colour offscreen at its unref unless told to narrow. The old default picked ONE
    * resource with a glyph-pipeline heuristic inherited from the debugging spike this grew out of,
    * and bailed outright when it matched nothing -- which is what it does on a corpus that draws
    * no glyphs. A heuristic scores a different set on every corpus, which is the exact drift the
    * fixture exists to prevent. The set is defended by the pinned score now, not guessed. */
   if (!readback_res)
      sweep = true;

   for (int loop = 0; loop < loops; loop++) {
      uint32_t next_res = 0;
      /* Batch assembly: CMD records between two SUBMITs form one submit_cmd call. */
      uint32_t *batch = NULL;
      size_t batch_dw = 0, batch_cap = 0;
      /* A batch belongs to the context whose records built it, which with more than one
       * replayed context is not always the first one named. */
      int batch_ctx = want_ctx;
      uint32_t submits = 0, cmds = 0, xfers = 0, dropped = 0, copy_fed = 0, copy_bad = 0;
      uint32_t blobdata = 0, blobs_fed = 0;
      uint32_t made = 0, failed = 0, unrefs = 0;
      bool batch_watch = false;

      for (size_t p = off; p + sizeof(struct rec_hdr) <= (size_t)flen; ) {
         struct rec_hdr h;
         memcpy(&h, blob + p, sizeof h);
         if (h.total_len < sizeof h || p + h.total_len > (size_t)flen) break;
         const uint32_t *aux = (const uint32_t *)(blob + p + sizeof h);
         const uint8_t *pay = blob + p + sizeof h + (size_t)h.aux_count * 4;

         /* Blob content is keyed by RESOURCE, not by context: the bytes belong to the memory
          * a blob exports, and the recorder reads them at a sampler bind with no context in
          * hand. Filtering them by ctx dropped every one of them -- they carry ctx 0, which no
          * corpus selects -- and the blobs replayed as the zeros this record exists to replace. */
         if (h.type != T_BLOBDATA && !ctx_wanted(h.ctx_id)) { p += h.total_len; continue; }
         if (until && h.seq > until) break;

         switch (h.type) {
         case T_SUBMIT:
            if (smoke) { batch_dw = 0; break; }
            if (watch && batch_watch)
               fprintf(stderr, "[watch] submitting batch with a TRANSFER3D for res=%u at seq %llu\n",
                       watch, (unsigned long long)h.seq);
            batch_watch = false;
            if (batch_dw) {
               if (virgl_renderer_submit_cmd(batch, batch_ctx, (int)batch_dw))
                  dropped++;
               submits++;
               batch_dw = 0;
            }
            batch_ctx = h.ctx_id;
            break;
         case T_CMD: {
            if (smoke) break;
            size_t dw = h.payload_len / 4;
            if (nodraw && h.cmd == VIRGL_CCMD_DRAW_VBO) break;   /* the positive control */
            /* --draws-from is the bounded form of --nodraw, and it asks the one question the
             * replay was built for: is the fault ACCUMULATED? Every resource, transfer and state
             * command still runs, so a later card is set up exactly as captured -- only the
             * earlier cards' rasterisation is removed. If that card then renders its header, the
             * damage is carried by work done before it rather than by its own stream. */
            if (draws_from && h.cmd == VIRGL_CCMD_DRAW_VBO && h.seq < draws_from) break;
            /* Two contexts' commands never share a submit. The capture interleaves them, and a
             * batch handed to the wrong context is rejected wholesale as "Illegal resource" --
             * which reads exactly like the resource bug this replay exists to find. */
            if (batch_dw && h.ctx_id != (uint16_t)batch_ctx) {
               if (virgl_renderer_submit_cmd(batch, batch_ctx, (int)batch_dw))
                  dropped++;
               submits++;
               batch_dw = 0;
            }
            batch_ctx = h.ctx_id;
            if (batch_dw + dw > batch_cap) {
               batch_cap = (batch_dw + dw) * 2;
               batch = realloc(batch, batch_cap * 4);
               if (!batch) { fprintf(stderr, "OOM\n"); return 2; }
            }
            memcpy(batch + batch_dw, pay, dw * 4);
            /* CCMD 43 is TRANSFER3D; its payload dword 1 is the resource handle. */
            if (watch && h.cmd == 43 && dw > 1 && ((const uint32_t *)pay)[1] == watch)
               batch_watch = true;
            /* CCMD 49 is PIPE_RESOURCE_SET_TYPE, the only command that says what a blob is.
             * Read it here rather than reconstructing a type at the create: the stream carries
             * the guest's own description, and the create record carries none. Recorded even
             * though vrend is about to be handed the same command -- this is the copy the
             * readback reads, and the renderer's is the copy that types the texture. */
            if (h.cmd == 49 && dw > 5) {
               const uint32_t *f = (const uint32_t *)pay;
               struct backing *tb = backing_find(f[1]);
               if (tb && tb->is_blob && !tb->typed) {
                  tb->typed = true;
                  tb->t_format = f[2];
                  tb->t_width = f[4];
                  tb->t_height = f[5];
                  /* Plane 0's stride, at dword 9. The guest's rows are not necessarily packed
                   * -- this corpus declares 4096 for a 500-wide 8-byte format whose packed row
                   * is 4000 -- and a write that assumed packed would shear every frame. */
                  tb->t_stride = dw > 9 ? f[9] : 0;
               }
            }
            batch_dw += dw;
            cmds++;
            break;
         }
         case T_BLOBDATA: {
            /* A blob's pixels are written GPU-side by a Vulkan client and never travel as a
             * transfer, so the recorder reads them where vrend does and they arrive here. Land
             * them in the backing store at the blob's own offset 0, exactly as the guest laid
             * them out; blob_feed below is what carries them into the texture. */
            uint32_t handle = h.aux_count > 0 ? aux[0] : 0;
            struct backing *b = backing_find(handle);
            if (b) {
               /* The record carries the whole SHM CARRIER, which is page-rounded and so is
                * routinely larger than the blob it backs -- 2 MiB behind a 2,048,000-byte
                * window here. Both start at offset 0, so the blob's own size is the honest
                * amount to take. Requiring the record to FIT dropped all 24 of them in silence,
                * which is why only the surplus is discarded. */
               if (h.payload_len < b->size) {
                  /* The other direction is not a short read to fill in: the recorder writes a
                   * record whole or refuses it whole, so a payload smaller than the blob is a
                   * corrupt capture. Taking the prefix and counting it landed would report
                   * content the replay does not have, and score it green on both legs. */
                  score_addf("blob res=%u content=%u/%zu declined=short-record\n",
                             b->handle, h.payload_len, b->size);
                  fprintf(stderr, "blob %u: content record is %u bytes, blob is %zu\n",
                          b->handle, h.payload_len, b->size);
                  break;
               }
               memcpy(b->mem, pay, b->size);
               b->dirty = true;
               blobdata++;
               /* The recorder reads a blob at the sampler bind INSIDE a batch, so this record
                * sits among the commands of the batch that samples it. Hand that batch to vrend
                * first: the pending commands may carry the SET_TYPE that makes the texture
                * exist, and the write must land before the draw that reads it -- on the leg
                * with no refresh of its own, that ordering is the whole content. */
               if (batch_dw) {
                  if (virgl_renderer_submit_cmd(batch, batch_ctx, (int)batch_dw))
                     dropped++;
                  submits++;
                  batch_dw = 0;
               }
               if (!nofeed)
                  blobs_fed += blob_feed(batch_ctx ? batch_ctx : want_ctx);
            }
            break;
         }
         case T_XFERDATA: {
            /* Land the recorded bytes in the resource's backing store BEFORE the batch that
             * reads them -- which is why this is applied at its recorded position in the stream
             * and not hoisted. */
            uint32_t handle = h.aux_count > 0 ? aux[0] : 0;
            uint64_t xoff = h.aux_count > 2 ? ((uint64_t)aux[2] << 32 | aux[1]) : 0;
            uint32_t src = 0;
            if (copy_src_of_next_cmd(blob, (size_t)flen, p + h.total_len, h.ctx_id,
                                     handle, xoff, &src, &copy_bad)) {
               handle = src;                 /* see copy_src_of_next_cmd */
               copy_fed++;
            }
            struct backing *b = backing_find(handle);
            if (b && xoff + h.payload_len <= b->size) {
               memcpy(b->mem + xoff, pay, h.payload_len);
               xfers++;
            }
            break;
         }
         default:
            break;
         }

         /* Resource events are applied AFTER the record, not before it. A SUBMIT record marks
          * the START of a batch in the capture, so the commands that follow it belong to it --
          * which means the buffered batch above must be handed to vrend before any create or
          * UNREF carrying this record's sequence is applied. Doing it the other way round runs
          * a batch after the unref of a resource it references, and vrend rejects the whole
          * batch with "Illegal resource" -- a defect of the replay that reads exactly like a
          * capture too incomplete to replay. */
         while (next_res < res_n && res[next_res].seq <= h.seq) {
            struct res_ev *r = &res[next_res++];
            if (r->kind == RES_UNREF) {
               /* --sweep scores EVERY colour offscreen at its unref. One run, and it answers the
                * question a single-resource verdict cannot: whether the replay renders at all.
                * A verdict of "lost" is only worth reading once some sibling comes back "present". */
               if (sweep && r->handle != readback_res) {
                  const struct res_ev *born = NULL;
                  for (uint32_t k = 0; k < res_n; k++)
                     if (res[k].handle == r->handle && res[k].kind == RES_CREATE
                         && res[k].seq < r->seq)
                        born = &res[k];
                  /* target 2 is a 2D texture; format 20 is D24S8, which carries no ink. */
                  /* --sweep-w narrows the readbacks to one width. Needed for the llvmpipe arm:
                   * reading every offscreen back trips an assert inside llvmpipe on resources
                   * that have nothing to do with the cards, and aborts the run before it reaches
                   * them. Scoring only the width under study keeps the A/B possible. */
                  if (born && born->target == 2 && born->format != 20 && born->width > 8
                      && (!sweep_w || born->width == sweep_w))
                     score_resource(born);
               }
               if (r->handle == readback_res) {
                  const struct res_ev *born = NULL;
                  for (uint32_t k = 0; k < res_n; k++)
                     if (res[k].handle == readback_res && res[k].kind != RES_UNREF
                         && res[k].seq < r->seq)
                        born = &res[k];
                  if (born) score_resource(born);
               }
               {
                  /* A typed blob is scored at its unref, like every other resource, and from
                   * the geometry SET_TYPE gave it rather than the create record's -- the record
                   * carries blob_mem and a size, which describe pages and not an image. An
                   * untyped one has no texture to read and is counted at the end instead. */
                  struct backing *tb = backing_find(r->handle);
                  if (tb && tb->is_blob && tb->typed) {
                     const struct res_ev bev = {
                        .handle = r->handle, .target = 2, .format = tb->t_format,
                        .width = tb->t_width, .height = tb->t_height,
                     };
                     score_resource(&bev);
                  }
               }
               if (no_unref) continue;
               struct backing *b = backing_find(r->handle);
               if (b) { virgl_renderer_resource_unref(r->handle); b->live = false; free(b->mem); b->mem = NULL; unrefs++; }
               continue;
            }
            struct backing *b = backing_add(r->handle, res_bytes(r));
            if (r->kind == RES_BLOB) {
               /* A blob is created as a blob, and deliberately WITHOUT a type. The stream says
                * what it is -- PIPE_RESOURCE_SET_TYPE, later, from the context that samples it --
                * and typing it here instead threw that description away: set_type returns early
                * on an already-typed resource, so pre-creating one as a buffer made the command
                * a no-op on both legs and left a sampled 500x500 R16G16B16X16_FLOAT texture
                * standing in as a linear R8 vertex buffer.
                *
                * blob_mem is GUEST, not the recorded HOST3D. HOST3D routes through ctx->get_blob,
                * which pops a resource parked by PIPE_RESOURCE_CREATE -- an opcode that appears
                * in no capture, so live these blobs came from venus and no classic context can
                * serve one. GUEST asks for the one thing the replayer can honestly supply: pages.
                * The C reads blob_flags only on the HOST3D path, so the recorded flags pass
                * through unexamined and need no masking.
                *
                * The iov goes in HERE rather than through attach_iov below, which is the reverse
                * of every other resource: the GUEST path requires iov_size >= size at create and
                * virgl_resource_create_from_iov takes the pages directly, while attach_iov would
                * then refuse with EINVAL for an iov already set. */
               b->is_blob = true;
               struct virgl_renderer_resource_create_blob_args a = {
                  .res_handle = r->handle, .ctx_id = r->flags,
                  .blob_mem = VIRGL_RENDERER_BLOB_MEM_GUEST, .blob_flags = r->bind,
                  .blob_id = ((uint64_t)r->array_size << 32) | r->depth,
                  .size = b->size, .iovecs = &b->iov, .num_iovs = 1,
               };
               int cr = virgl_renderer_resource_create_blob(&a);
               if (cr) {
                  if (failed < 5)
                     fprintf(stderr, "create BLOB res=%u size=%zu failed: %d\n", r->handle, b->size, cr);
                  b->live = false; failed++; continue;
               }
               made++;
            } else {
               struct virgl_renderer_resource_create_args a = {
                  .handle = r->handle, .target = r->target, .format = r->format, .bind = r->bind,
                  .width = r->width, .height = r->height, .depth = r->depth,
                  .array_size = r->array_size, .nr_samples = r->nr_samples,
                  .last_level = r->last_level, .flags = r->flags,
               };
               int cr = virgl_renderer_resource_create(&a, NULL, 0);
               if (cr) {
                  if (failed < 5)
                     fprintf(stderr, "create res=%u target=%u fmt=%u bind=0x%x %ux%ux%u array=%u "
                             "levels=%u samples=%u iov=%zu failed: %d\n",
                             r->handle, r->target, r->format, r->bind, r->width, r->height,
                             r->depth, r->array_size, r->last_level, r->nr_samples, b->size, cr);
                  b->live = false; failed++; continue;
               }
               made++;
               {
                  uint32_t id = 0;
                  if (virgl_renderer_resource_get_iosurface_id(r->handle, &id) == 0 && id) {
                     iosurf_backed++;
                     iosurf_remember(r->handle, r->width, r->height);
                  }
               }
            }
            if (watch && r->handle == watch) {
               /* set_priv/get_priv round-trip is a registration probe: both go through
                * virgl_resource_lookup, so a NULL read back means the handle is not registered
                * even though create returned success. */
               virgl_renderer_resource_set_priv(r->handle, (void *)(uintptr_t)0xf00d);
               void *pv = virgl_renderer_resource_get_priv(r->handle);
               fprintf(stderr, "[watch] create res=%u at res-seq %llu, applied at record seq %llu, "
                       "registered=%s\n",
                       r->handle, (unsigned long long)r->seq, (unsigned long long)h.seq,
                       pv ? "yes" : "NO");
            }
            /* The iov MUST arrive here and not through create. create stores it on the VIRGL
             * resource and stops -- its own signature marks the parameter UNUSED -- while vrend's
             * resource, the one every transfer actually reads, is fed only by this call. Worse,
             * the two are mutually exclusive: virgl_resource_attach_iov refuses with EINVAL when
             * an iov is already set, so passing it to create silently BLOCKS the attach that
             * reaches vrend. The resource then creates, registers and attaches cleanly, and every
             * TRANSFER3D touching it fails check_transfer_iovec -- reported as the very same
             * "Illegal resource" as a handle the context has never heard of. */
            if (r->kind != RES_BLOB)
               virgl_renderer_resource_attach_iov((int)r->handle, &b->iov, 1);
            for (int i = 0; i < n_ctx; i++)
               virgl_renderer_ctx_attach_resource(ctx_list[i], (int)r->handle);
            /* After the attach, which is what gives vrend the resource a transfer can reach. */
            /* Not a planar YUV one. Zeroing writes at four bytes per texel, which is what
             * every format this scores actually is -- but a planar resource cannot take a
             * transfer at all: vrend guards the upload with gallium's blocksize for the format
             * (1 for NV12) and then performs it with the format's GL triple, which
             * vrend_formats.c registers as GL_RGBA/GL_UNSIGNED_BYTE, four bytes. The guard is
             * computed at a quarter of what the upload reads, so GL walks off the end of the
             * iov whatever size it is given. On a 2560x1440 NV12 target that is a segfault in
             * util_copy_rect; at 64x64 it merely reads 48 KB of somebody else's heap. Skipping
             * leaves such a resource undefined, which costs nothing: nothing reads its texture,
             * and the sweep scores it through its IOSurface. */
            if (zero_new && r->kind != RES_BLOB && !format_is_planar_yuv(r->format))
               zero_resource(r->handle, r->format, r->width, r->height, want_ctx);
         }

         p += h.total_len;
      }

      if (!smoke && batch_dw && virgl_renderer_submit_cmd(batch, batch_ctx, (int)batch_dw))
         dropped++;
      free(batch);
      count_addf("loop %d created %u failed %u unrefs %u submits %u cmds %u xfers %u "
                 "copy-fed %u copy-unmatched %u submit-errors %u iosurface-backed %u\n",
                 loop, made, failed, unrefs, submits, cmds, xfers, copy_fed, copy_bad, dropped,
                 iosurf_backed);
      /* Its own line, and only for a corpus that carries blob content: appending two fields to
       * the counter line above would move every pinned score in the tree. `records` is what the
       * capture holds, `fed` what reached a texture, and the two differing is the signal --
       * bytes recorded for a blob nothing typed land nowhere and say so. */
      if (blobdata)
         count_addf("loop %d blob-content records %u fed %u\n", loop, blobdata, blobs_fed);
      made_total += made;
      failed_total += failed;
   }

   /* A smoke score is a strict subset of a real one, so it is never written or compared: pinning
    * it would replace a golden with a weaker one that still passes, which is the failure a fixture
    * exists to prevent. The verdict is whether every create landed. */
   if (smoke) {
      printf("smoke: created %u failed %u contexts 1 iosurface-backed %u -- "
             "init, contexts and resources %s\n",
             made_total, failed_total, iosurf_backed, failed_total ? "FAILED" : "OK");
      return failed_total ? 1 : 0;
   }

   for (uint32_t i = 0; i < iosurf_n; i++)
      score_iosurface(iosurf_res[i].handle, iosurf_res[i].w, iosurf_res[i].h);

   /* A typed blob the stream never unrefs is scored HERE, at the end, from its texture -- the
    * same second chance the --readback resource gets below. Scoring only at the unref read three
    * of this corpus's six sampled windows and passed over the other three in silence, because a
    * capture stops where it stops and a resource still alive at that point has no unref to hang
    * a readback on. Creation order, so the sequence is the corpus's and not the allocator's. */
   for (uint32_t i = 0; i < backing_n; i++) {
      const struct backing *b = backings[i];
      if (!b->is_blob || !b->typed || !b->live)
         continue;
      const struct res_ev bev = {
         .handle = b->handle, .target = 2, .format = b->t_format,
         .width = b->t_width, .height = b->t_height,
      };
      score_resource(&bev);
   }

   /* A --readback resource that is still alive at the end of the stream is read back THEN, from
    * its texture. For a scanout this is the second leg of the IOSurface question: the surface
    * line above reads the display storage, this reads the texture rendered into it, and the two
    * disagreeing says which side a fault is on. */
   if (readback_res) {
      const struct res_ev *born = NULL;
      bool alive = false;
      for (uint32_t k = 0; k < res_n; k++) {
         if (res[k].handle != readback_res)
            continue;
         if (res[k].kind == RES_UNREF)
            alive = false;
         else { born = &res[k]; alive = true; }
      }
      if (born && alive)
         score_resource(born);
   }

   /* --rebuild: the gate for the snapshot journal. Export what a context retained, feed it into
    * a context that has never seen the stream, and export that one. The comparison is
    * structural, not byte-for-byte: the rebuilt context re-records from seq 1 as it replays, so
    * every position is renumbered, and what has to match is the steps and their order.
    *
    * What this catches is a journal that cannot rebuild what it came from -- an entry that fails
    * to replay, an order that puts a bind before its object, a serializer that loses a shader's
    * later chunks. What it cannot catch is a durable command the recorder never learned to keep:
    * that is absent from both journals and both agree about it. Only pixels answer that, so this
    * is the floor and not the ceiling. */
   if (rebuild) {
      for (int i = 0; i < n_ctx; i++) {
         uint32_t src = (uint32_t)ctx_list[i];
         void *a = NULL; uint64_t a_len = 0;
         int rc = virgl_renderer_limina_journal_export(src, &a, &a_len);
         if (rc) {
            fprintf(stderr, "rebuild: ctx %u exported nothing (%d)\n", src, rc);
            rebuild_failed++;
            continue;
         }
         uint32_t dst = src + 1000;
         const char *rname = "limina-rebuild";
         if (virgl_renderer_context_create(dst, (uint32_t)strlen(rname), rname)) {
            fprintf(stderr, "rebuild: context_create %u failed\n", dst);
            rebuild_failed++;
            free(a);
            continue;
         }
         /* The journal names resources; a context that is not attached to them cannot rebuild a
          * sampler view or a surface over one, and every such create would drop. */
         for (uint32_t r = 0; r < backing_n; r++)
            if (backings[r]->live)
               virgl_renderer_ctx_attach_resource((int)dst, (int)backings[r]->handle);

         virgl_renderer_limina_replay_begin(dst);
         int rr = virgl_renderer_limina_journal_restore(dst, a, a_len);
         if (rr) {
            fprintf(stderr, "rebuild: ctx %u refused its own journal (%d)\n", src, rr);
            rebuild_failed++;
         }
         virgl_renderer_limina_journal_replay_upto(dst, UINT64_MAX);
         virgl_renderer_limina_replay_end(dst);

         /* The contents half, in the VMM's own order: the classic pixels go in after the wire
          * replay has re-created the objects to hold them (virtio_gpu's restore does the same,
          * after the per-context replay loop and before the scanout flips). */
         void *ca = NULL; uint64_t ca_len = 0;
         int crc = virgl_renderer_limina_classic_content_export(src, &ca, &ca_len);
         if (crc) {
            fprintf(stderr, "content: ctx %u exported nothing (%d)\n", src, crc);
            content_failed++;
         } else {
            /* Scrub, then restore, then read back: the bytes that come back can only have come
             * from the restore. A scrub that reads back unchanged says the restore wrote
             * nothing, which is the way this gate would otherwise pass while doing nothing. */
            void *scrub = vrcc_scrub(ca, ca_len);
            int cw = virgl_renderer_limina_classic_content_restore(dst, scrub, ca_len);
            void *cs = NULL; uint64_t cs_len = 0;
            int cy = cw ? cw : virgl_renderer_limina_classic_content_export(dst, &cs, &cs_len);
            bool scored = false;
            if (!cy && vrcc_same(ca, ca_len, cs, cs_len, &scored) && scored) {
               fprintf(stderr, "content: ctx %u -> %u: the scrub read back unchanged -- "
                       "the restore wrote nothing\n", src, dst);
               content_failed++;
            }
            free(cs);
            free(scrub);

            int cr = cw ? cw : virgl_renderer_limina_classic_content_restore(dst, ca, ca_len);
            void *cb = NULL; uint64_t cb_len = 0;
            int cx = cr ? cr : virgl_renderer_limina_classic_content_export(dst, &cb, &cb_len);
            if (cx) {
               fprintf(stderr, "content: ctx %u -> %u restore/export failed (%d)\n", src, dst, cx);
               content_failed++;
            } else {
               uint32_t missing = 0;
               if (!contents_agree(ca, ca_len, cb, cb_len, src, dst, &missing, &content_deviations,
                                   res, res_n))
                  content_failed++;
            }
            free(cb);
            free(ca);
         }

         void *b = NULL; uint64_t b_len = 0;
         if (virgl_renderer_limina_journal_export(dst, &b, &b_len)) {
            fprintf(stderr, "rebuild: ctx %u rebuilt into nothing\n", src);
            rebuild_failed++;
            free(a);
            continue;
         }
         uint32_t dropped = 0;
         if (!journals_agree(a, a_len, b, b_len, src, &dropped))
            rebuild_failed++;
         else if (dropped)
            fprintf(stderr, "rebuild: ctx %u -> %u rebuilt without %u entr%s the report names\n",
                    src, dst, dropped, dropped == 1 ? "y" : "ies");
         else
            fprintf(stderr, "rebuild: ctx %u -> %u rebuilt identically (%llu bytes)\n",
                    src, dst, (unsigned long long)a_len);
         rebuild_dropped += dropped;
         free(a);
         free(b);
      }
   }

   /* The journal census, on stderr and never in the score: it is a fact about what the recorder
    * retained, not about the pixels, and a number that moves whenever the recorder changes must
    * not be able to rewrite every pinned score in the tree. */
   virgl_renderer_limina_dump_state();

   /* A blob nothing typed is a real result -- it says the stream never described storage the
    * guest went on to use -- so it is counted rather than passed over in silence. Emitted only
    * for a corpus that HAS blobs: an unconditional line would rewrite every pinned score in the
    * tree to say zero about a thing it does not contain. */
   {
      uint32_t blobs = 0, untyped = 0;
      for (uint32_t i = 0; i < backing_n; i++)
         if (backings[i]->is_blob) { blobs++; if (!backings[i]->typed) untyped++; }
      if (blobs)
         score_addf("blobs %u untyped %u\n", blobs, untyped);
   }

   if (!scored)
      fprintf(stderr, "nothing was read back: no scored resource was unref'd in the trace\n");

   /* The score: counters, then one readback line per scored resource in stream order -- the same
    * shape the venus replayer emits, so both layers are compared with plain diff. */
   size_t total = count_len + score_len;
   char *text = malloc(total + 1);
   if (!text) { perror("score"); return 2; }
   memcpy(text, count_buf ? count_buf : "", count_len);
   memcpy(text + count_len, score_buf ? score_buf : "", score_len);
   text[total] = 0;
   fputs(text, stdout);

   /* Ink somewhere is the floor -- from ANY of the three readbacks, texture, IOSurface or
    * plane: a run that reads back nothing but zeros rendered nothing, and
    * every hash it reports is the hash of an empty buffer. --nodraw is NOT the inverse of that.
    * It used to fail on any ink at all, which was right when one render target was scored and is
    * wrong now that the sweep scores every offscreen: a texture filled by a transfer has ink with
    * no draw involved, and 5 of this corpus's 310 do. The control is the DIFF between the two
    * pinned scores -- the resources that lose their ink when draws are dropped are exactly the
    * ones drawing reaches, and an empty diff means the oracle is measuring nothing.
    *
    * The floor only applies when there is no golden. Against --expect the match IS the verdict:
    * the golden carries its own ink lines, so a corpus whose expected score is legitimately
    * inkless must still pass. */
   int ok = expect_path ? 1 : scored_ink;

   if (score_path) {
      FILE *sf = fopen(score_path, "wb");
      if (!sf || fwrite(text, 1, total, sf) != total) {
         fprintf(stderr, "writing %s: %s\n", score_path, strerror(errno));
         ok = 0;
      } else {
         fprintf(stderr, "score written to %s\n", score_path);
      }
      if (sf) fclose(sf);
   }

   if (expect_path) {
      FILE *ef = fopen(expect_path, "rb");
      if (!ef) {
         fprintf(stderr, "reading %s: %s\n", expect_path, strerror(errno));
         ok = 0;
      } else {
         fseek(ef, 0, SEEK_END);
         long elen = ftell(ef);
         fseek(ef, 0, SEEK_SET);
         char *want = malloc((size_t)elen + 1);
         if (!want || fread(want, 1, (size_t)elen, ef) != (size_t)elen) {
            fprintf(stderr, "short read of %s\n", expect_path);
            ok = 0;
         } else {
            want[elen] = 0;
            if ((size_t)elen == total && !memcmp(want, text, total)) {
               fprintf(stderr, "score matches %s\n", expect_path);
            } else {
               fprintf(stderr, "SCORE DIFFERS from %s:\n", expect_path);
               diff_lines(want, text);
               /* A hash says two implementations disagree and nothing about how. The pixels
                * behind it are gone by now -- they are read, hashed and freed one resource at a
                * time, and keeping every readback against the chance of a mismatch would hold a
                * corpus's worth of images to print one line. So name the way to get them, once,
                * at the only moment anyone wants them. */
               if (!getenv("REPLAY_DUMP_DIR"))
                  fprintf(stderr,
                          "  (a hash names no pixels: re-run with REPLAY_DUMP_DIR=<dir> to write "
                          "the readbacks, then look at them -- rgba2png.py for the 8-bit BGRA "
                          "offscreens, half2png.py for a half-float one such as a blob window)\n");
               ok = 0;
            }
         }
         free(want);
         fclose(ef);
      }
   }

   /* A rebuild that did not describe the same world fails the run, whatever the score said: the
    * score is about the pixels this stream drew, and the rebuild is about whether they could be
    * drawn again after a resume. */
   /* A content mismatch fails the run for the same reason a rebuild failure does, and is
    * reported after it: a journal that did not rebuild the objects is the cause, and contents
    * that could not be put into objects that are not there is its symptom. */
   if (rebuild_failed) {
      fprintf(stderr, "REBUILD FAILED for %u context(s)\n", rebuild_failed);
      ok = 0;
   }
   if (content_failed) {
      fprintf(stderr, "CONTENT FAILED for %u context(s)\n", content_failed);
      ok = 0;
   }
   /* A resource the rebuild dropped has nowhere for its content to go, so a missing entry is the
    * journal drop's shadow and is pinned by the same report. Without a pin it is a failure, the
    * same way an unpinned journal drop is. */
   /* A resource that did not come back, or came back different, is in the rebuild report and is
    * a failure unless that report is pinned -- the same rule the journal's dropped entries
    * follow, for the same reason: what is acceptable is a property of the corpus, not of a run. */
   if (content_deviations && !rb_expect_path) {
      fprintf(stderr, "CONTENT DEVIATED for %u resource(s) and nothing pins them "
              "(see --rebuild-expect)\n", content_deviations);
      ok = 0;
   }

   if (rb_score_path && rebuild) {
      FILE *rf = fopen(rb_score_path, "wb");
      if (!rf || fwrite(rb_buf ? rb_buf : "", 1, rb_len, rf) != rb_len) {
         fprintf(stderr, "writing %s: %s\n", rb_score_path, strerror(errno));
         ok = 0;
      } else {
         fprintf(stderr, "rebuild report written to %s\n", rb_score_path);
      }
      if (rf) fclose(rf);
   }

   /* What a corpus is known to lose, and why a lost entry is not automatically a fault. A guest
    * that destroys a resource under an object of its own leaves a create the rebuild cannot
    * replay; the object is already unusable, because binding it faults on the resource lookup
    * whether or not a rebuild happened. So the drop is the right behaviour and the pin is what
    * says WHICH drops this corpus makes -- a different entry going missing is still a diff.
    *
    * With no pin, a drop is a failure. That is the resting state, and every corpus but the one
    * with two guest processes sharing buffers stays in it. */
   if (rebuild && rb_expect_path) {
      FILE *ef = fopen(rb_expect_path, "rb");
      if (!ef) {
         fprintf(stderr, "reading %s: %s\n", rb_expect_path, strerror(errno));
         ok = 0;
      } else {
         fseek(ef, 0, SEEK_END);
         long elen = ftell(ef);
         fseek(ef, 0, SEEK_SET);
         char *want = malloc((size_t)elen + 1);
         if (!want || fread(want, 1, (size_t)elen, ef) != (size_t)elen) {
            fprintf(stderr, "short read of %s\n", rb_expect_path);
            ok = 0;
         } else {
            want[elen] = 0;
            if ((size_t)elen == rb_len && !memcmp(want, rb_buf ? rb_buf : "", rb_len)) {
               fprintf(stderr, "rebuild matches %s\n", rb_expect_path);
            } else {
               fprintf(stderr, "REBUILD DIFFERS from %s:\n", rb_expect_path);
               diff_lines(want, rb_buf ? rb_buf : "");
               ok = 0;
            }
         }
         free(want);
         fclose(ef);
      }
   } else if (rebuild_dropped) {
      fprintf(stderr,
              "REBUILD DROPPED %u entr%s and nothing pins them (see --rebuild-expect)\n",
              rebuild_dropped, rebuild_dropped == 1 ? "y" : "ies");
      ok = 0;
   }

   free(rb_buf);

   free(text);
   return ok ? 0 : 1;
}
