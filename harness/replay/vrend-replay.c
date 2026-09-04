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
//   vrend-replay <dump> [--ctx N] [--loops N] [--nodraw] [--readback RES] [--sweep]
//
//   --ctx N       which virgl context to replay (default: the one with the most records)
//   --loops N     replay the captured stream N times (default 1)
//   --nodraw      positive control: drop every DRAW_VBO. Score it and diff against the ordinary
//                 score -- the resources that lose their ink are the ones drawing reaches, and an
//                 empty diff means the oracle is measuring nothing.
//   --readback R  score only this resource (default: every colour offscreen, at its unref)
//   --sweep       score every colour offscreen at its unref -- the default
//   --sweep-w W   restrict --sweep to targets of this width
//   --score F     write the score to F
//   --expect F    compare the score against F and exit non-zero on any difference
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
#include <string.h>
#include <sys/uio.h>

#define TRACE_MAGIC 0x4c4d5654u

enum { T_SUBMIT = 1, T_CMD = 2, T_DRAW_FB = 3, T_TRANSFER = 4, T_FENCE = 5, T_RETIRE = 6,
       T_PAD = 7, T_XFERDATA = 9 };
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
static const char *score_path, *expect_path, *caps_path;

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
static void score_iosurface(uint32_t handle, uint32_t w, uint32_t h)
{
   size_t need = (size_t)w * h * 4;
   int sr = virgl_renderer_resource_sync_iosurface(handle);
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

static void score_resource(const struct res_ev *ev)
{
   uint32_t w = ev->width ? ev->width : 1, h = ev->height ? ev->height : 1;
   size_t need = (size_t)w * h * 4;
   uint8_t *px = calloc(1, need);
   struct iovec riov = { .iov_base = px, .iov_len = need };
   struct virgl_box box = { .x = 0, .y = 0, .z = 0, .w = w, .h = h, .d = 1 };

   /* No force_ctx_0 here. The context path does its own switch -- transfer_read_iov with a
    * non-zero ctx_id reaches vrend_renderer_transfer_internal, which calls vrend_hw_switch_context
    * on the replayed context and leaves it current -- so forcing ctx0 first only added a switch
    * away and back. */
   int rr = virgl_renderer_transfer_read_iov(ev->handle, (uint32_t)want_ctx, 0, w * 4, 0,
                                             &box, 0, &riov, 1);
   if (rr) {
      fprintf(stderr, "readback of res %u failed: %d\n", ev->handle, rr);
      free(px);
      return;
   }

   size_t ink = 0;
   for (size_t i = 0; i < need; i += 4)
      if (px[i] | px[i + 1] | px[i + 2] | px[i + 3]) ink++;

   /* FNV-1a over the readback. This line IS the golden: a content hash compares two
    * implementations without carrying megabytes of reference pixels in the tree, and it is
    * stable across runs in a way an ink count is not (ink says only "something rendered"). */
   uint64_t hash = 1469598103934665603ull;
   for (size_t i = 0; i < need; i++) { hash ^= px[i]; hash *= 1099511628211ull; }

   score_addf("res=%u %ux%u hash=%016llx ink=%zu/%zu\n",
              ev->handle, w, h, (unsigned long long)hash, ink, need / 4);

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
   scored_ink = ink != 0;
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

int main(int argc, char **argv)
{
   const char *path = NULL;
   int loops = 1;
   bool nodraw = false;
   /* Exercise only what a skeleton owes: init, context create, resource create. No submits, no
    * transfers, no scoring. This is P1's gate -- a renderer that gets through it has a working
    * ABI, resource table and context table, which is all a skeleton claims. */
   bool smoke = false;
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
      if (!strcmp(argv[i], "--ctx") && i + 1 < argc) want_ctx = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--sweep")) sweep = true;
      else if (!strcmp(argv[i], "--sweep-w") && i + 1 < argc)
         sweep_w = (uint32_t)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--draws-from") && i + 1 < argc)
         draws_from = strtoull(argv[++i], NULL, 10);
      else if (!strcmp(argv[i], "--until") && i + 1 < argc)
         until = strtoull(argv[++i], NULL, 10);
      else if (!strcmp(argv[i], "--loops") && i + 1 < argc) loops = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--nodraw")) nodraw = true;
      else if (!strcmp(argv[i], "--smoke")) smoke = true;
      else if (!strcmp(argv[i], "--readback") && i + 1 < argc) readback_res = (uint32_t)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--score") && i + 1 < argc) score_path = argv[++i];
      else if (!strcmp(argv[i], "--expect") && i + 1 < argc) expect_path = argv[++i];
      else if (!strcmp(argv[i], "--caps") && i + 1 < argc) caps_path = argv[++i];
      else if (argv[i][0] != '-') path = argv[i];
   }
   if (!path) { fprintf(stderr, "usage: vrend-replay <dump> [--ctx N] [--loops N] [--nodraw] [--draws-from SEQ] [--until SEQ]\n"); return 2; }

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
   ret = virgl_renderer_context_create((uint32_t)want_ctx, (uint32_t)strlen(name), name);
   if (ret) { fprintf(stderr, "context_create failed: %d\n", ret); return 2; }

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
      uint32_t submits = 0, cmds = 0, xfers = 0, dropped = 0, copy_fed = 0, copy_bad = 0;
      uint32_t made = 0, failed = 0, unrefs = 0;
      bool batch_watch = false;

      for (size_t p = off; p + sizeof(struct rec_hdr) <= (size_t)flen; ) {
         struct rec_hdr h;
         memcpy(&h, blob + p, sizeof h);
         if (h.total_len < sizeof h || p + h.total_len > (size_t)flen) break;
         const uint32_t *aux = (const uint32_t *)(blob + p + sizeof h);
         const uint8_t *pay = blob + p + sizeof h + (size_t)h.aux_count * 4;

         if (h.ctx_id != (uint16_t)want_ctx) { p += h.total_len; continue; }
         if (until && h.seq > until) break;

         switch (h.type) {
         case T_SUBMIT:
            if (smoke) { batch_dw = 0; break; }
            if (watch && batch_watch)
               fprintf(stderr, "[watch] submitting batch with a TRANSFER3D for res=%u at seq %llu\n",
                       watch, (unsigned long long)h.seq);
            batch_watch = false;
            if (batch_dw) {
               if (virgl_renderer_submit_cmd(batch, want_ctx, (int)batch_dw))
                  dropped++;
               submits++;
               batch_dw = 0;
            }
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
            if (batch_dw + dw > batch_cap) {
               batch_cap = (batch_dw + dw) * 2;
               batch = realloc(batch, batch_cap * 4);
               if (!batch) { fprintf(stderr, "OOM\n"); return 2; }
            }
            memcpy(batch + batch_dw, pay, dw * 4);
            /* CCMD 43 is TRANSFER3D; its payload dword 1 is the resource handle. */
            if (watch && h.cmd == 43 && dw > 1 && ((const uint32_t *)pay)[1] == watch)
               batch_watch = true;
            batch_dw += dw;
            cmds++;
            break;
         }
         case T_XFERDATA: {
            /* Land the recorded bytes in the resource's backing store BEFORE the batch that
             * reads them -- which is why this is applied at its recorded position in the stream
             * and not hoisted. */
            uint32_t handle = h.aux_count > 0 ? aux[0] : 0;
            uint64_t xoff = h.aux_count > 2 ? ((uint64_t)aux[2] << 32 | aux[1]) : 0;
            uint32_t src = 0;
            if (copy_src_of_next_cmd(blob, (size_t)flen, p + h.total_len, want_ctx,
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
               if (no_unref) continue;
               struct backing *b = backing_find(r->handle);
               if (b) { virgl_renderer_resource_unref(r->handle); b->live = false; free(b->mem); b->mem = NULL; unrefs++; }
               continue;
            }
            struct backing *b = backing_add(r->handle, res_bytes(r));
            if (r->kind == RES_BLOB) {
               /* A guest-memory blob is just shared pages; the command stream reads it through
                * its iov exactly like any other resource, so a plain resource with a backing
                * store is a faithful stand-in and needs no get_blob plumbing. */
               struct virgl_renderer_resource_create_args a = {
                  .handle = r->handle, .target = 0, .format = 64 /* R8_UNORM */,
                  .bind = 0x10 /* VIRGL_BIND_VERTEX_BUFFER */, .width = (uint32_t)b->size,
                  .height = 1, .depth = 1, .array_size = 1, .nr_samples = 0, .last_level = 0, .flags = 0,
               };
               int cr = virgl_renderer_resource_create(&a, NULL, 0);
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
            virgl_renderer_resource_attach_iov((int)r->handle, &b->iov, 1);
            virgl_renderer_ctx_attach_resource(want_ctx, (int)r->handle);
         }

         p += h.total_len;
      }

      if (!smoke && batch_dw && virgl_renderer_submit_cmd(batch, want_ctx, (int)batch_dw))
         dropped++;
      free(batch);
      count_addf("loop %d created %u failed %u unrefs %u submits %u cmds %u xfers %u "
                 "copy-fed %u copy-unmatched %u submit-errors %u iosurface-backed %u\n",
                 loop, made, failed, unrefs, submits, cmds, xfers, copy_fed, copy_bad, dropped,
                 iosurf_backed);
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

   /* Ink somewhere is the floor: a run that reads back nothing but zeros rendered nothing, and
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
               ok = 0;
            }
         }
         free(want);
         fclose(ef);
      }
   }

   free(text);
   return ok ? 0 : 1;
}
