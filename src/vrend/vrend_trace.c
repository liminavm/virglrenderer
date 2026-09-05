/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * See vrend_trace.h for why this buffers in memory instead of writing as it goes.
 */

#include "vrend_trace.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define TRACE_MAGIC   0x4c4d5654u   /* "LMVT" */
#define TRACE_VERSION 2
/* Creates are never evicted, so the log grows rather than windowing. It starts here and doubles;
 * the ceiling is only there so a runaway guest cannot eat the host, and reaching it is loud.
 *
 * A fixed cap was wrong about which guests exist. A stock desktop fits in 32768 creates and an
 * enhanced one does not -- it overflows during boot, long before the workload starts -- so the
 * cap silently decided that the tier this renderer is being ported for could not be captured at
 * all. A create is rare next to a draw, so growing on one costs nothing the ring's own budget
 * does not already dwarf. */
#define TRACE_RES_INITIAL 32768u
#define TRACE_RES_MAX     4194304u
#define MAX_AUX       8

struct trace_state {
   uint8_t *buf;
   size_t   cap;
   /* A circular FIFO of variable-length records. head is where the next record goes, tail is
    * the oldest live record; used tracks the bytes between them. Records are never split
    * across the end of the buffer -- a PAD record fills the tail end instead -- so a reader
    * can always walk forward from tail by total_len. */
   size_t   head, tail, used;
   uint64_t seq;
   uint64_t evicted;        /* records dropped to make room; a nonzero count means the window
                             * no longer reaches back to the start of the session */
   uint64_t base_realtime_ns;
   uint64_t base_mono_ns;
   char     out_path[512];
   char     fifo_path[512];
};

static struct trace_state tr;
/* The resource create/destroy log. Deliberately NOT in the ring: the resources a replay most
 * needs are created once at client startup, so in a FIFO they are the first thing evicted once
 * transfer payloads inflate the stream. */
static struct vrend_trace_res *tr_res;
static uint32_t tr_res_n, tr_res_cap;
static bool tr_res_full;
static bool tr_on;
#define TR_BLOB_SEEN 64
static struct { uint32_t handle; uint64_t hash; uint32_t kept, dropped; bool used; }
   tr_blob_seen[TR_BLOB_SEEN];

/* How many distinct frames to keep per blob. A client that animates changes its window every
 * frame, so dedup alone bounds nothing: vkcube at 2 MB a frame fills a 512 MB ring in seconds
 * and FIFO eviction then eats the command records the corpus is actually made of. A handful of
 * real frames is all the fixture needs -- it scores each blob once, and what it is testing is
 * that both renderers read one set of bytes the same way, not that they can be fed many.
 * LIMINA_VREND_TRACE_BLOB_MAX overrides; 0 disables blob content capture entirely. */
static uint32_t tr_blob_max = 4;

static bool tr_inited;
static atomic_int tr_dump_req;
/* Records do NOT all arrive on the decode thread: fences are created there, but they RETIRE on
 * vrend's fence-poll thread. The first version of this file assumed a single writer and produced
 * a trace with a duplicated sequence number -- two records claiming the same seq, which is the
 * visible tip of a torn append. An uncontended pthread mutex is a few tens of nanoseconds, which
 * is nothing next to the file I/O this design already refuses to do in the hot path. */
static pthread_mutex_t tr_lock = PTHREAD_MUTEX_INITIALIZER;

static uint64_t mono_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t real_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_REALTIME, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void trace_dump_locked(void);

/* Blocks on a FIFO so that asking for a dump costs the render thread nothing until it happens.
 * A signal would have been simpler, but the worker's signal set is already crowded (TERM, HUP,
 * USR1 snapshot, USR2 suspend, TSTP, WINCH) and stealing one would be a trap for a later
 * reader. */
static void *trace_fifo_thread(void *arg)
{
   (void)arg;
   for (;;) {
      int fd = open(tr.fifo_path, O_RDONLY);
      if (fd < 0) {
         struct timespec s = { 1, 0 };
         nanosleep(&s, NULL);
         continue;
      }
      for (;;) {
         char c;
         ssize_t n = read(fd, &c, 1);
         if (n <= 0)
            break;
         /* Dump straight from this thread. The first version only set a flag for the render
          * thread to notice at its next submit, which meant a dump requested while the guest
          * was idle simply never happened -- and looked exactly like a broken tracer. Now that
          * the ring is mutex-protected there is no reason to wait for guest activity. */
         trace_dump_locked();
      }
      close(fd);
   }
   return NULL;
}

void vrend_trace_init(void)
{
   const char *env, *p;
   size_t mb;
   pthread_t th;

   if (tr_inited)
      return;
   tr_inited = true;

   env = getenv("LIMINA_VREND_TRACE");
   mb = env ? (size_t)strtoul(env, NULL, 10) : 0;
   if (!mb) {
      fprintf(stderr, "[LIMINA-TRACE] vrend command tracer off "
                      "(LIMINA_VREND_TRACE=<MB> to arm)\n");
      fflush(stderr);
      return;
   }

   tr.cap = mb * 1024u * 1024u;
   tr.buf = malloc(tr.cap);
   if (!tr.buf) {
      fprintf(stderr, "[LIMINA-TRACE] could not allocate %zu MB ring; tracer off\n", mb);
      fflush(stderr);
      return;
   }
   /* Touch it now: a first-touch page fault inside the hot path would be a timing
    * perturbation of exactly the kind this design exists to avoid. */
   memset(tr.buf, 0, tr.cap);
   tr_res_cap = TRACE_RES_INITIAL;
   tr_res = calloc(tr_res_cap, sizeof *tr_res);
   if (!tr_res) {
      free(tr.buf);
      tr.buf = NULL;
      return;
   }

   env = getenv("LIMINA_VREND_TRACE_BLOB_MAX");
   if (env)
      tr_blob_max = (uint32_t)strtoul(env, NULL, 10);
   fprintf(stderr, "[LIMINA-TRACE] blob content: up to %u frame(s) per blob\n", tr_blob_max);

   p = getenv("LIMINA_VREND_TRACE_OUT");
   snprintf(tr.out_path, sizeof tr.out_path, "%s",
            p ? p : "/tmp/limina-vrend-trace.bin");
   p = getenv("LIMINA_VREND_TRACE_FIFO");
   snprintf(tr.fifo_path, sizeof tr.fifo_path, "%s",
            p ? p : "/tmp/limina-vrend-trace.fifo");

   unlink(tr.fifo_path);
   if (mkfifo(tr.fifo_path, 0600) != 0 && errno != EEXIST) {
      fprintf(stderr, "[LIMINA-TRACE] mkfifo %s failed (%s); dump on exit only\n",
              tr.fifo_path, strerror(errno));
   } else if (pthread_create(&th, NULL, trace_fifo_thread, NULL) == 0) {
      pthread_detach(th);
   }

   tr.base_mono_ns = mono_ns();
   tr.base_realtime_ns = real_ns();
   tr_on = true;

   atexit(trace_dump_locked);

   fprintf(stderr, "[LIMINA-TRACE] vrend command tracer ARMED: %zu MB ring, "
                   "dump on `echo x > %s` -> %s\n", mb, tr.fifo_path, tr.out_path);
   fflush(stderr);
}

/* Arms on first ask, not on first submit. Resources are created on the CONTROL path, and the
 * guest KMS driver makes its scanout and cursor resources at boot -- before any 3D client
 * submits a command. Arming only from the decode path left those creates unlogged while the
 * stream referenced them forever, which makes a trace look complete and replay as if the
 * resources had never existed. */
bool vrend_trace_enabled(void)
{
   if (!tr_inited)
      vrend_trace_init();
   return tr_on;
}

static void evict_one(void)
{
   struct vrend_trace_rec r;
   memcpy(&r, tr.buf + tr.tail, sizeof r);
   tr.tail += r.total_len;
   if (tr.tail >= tr.cap)
      tr.tail = 0;
   tr.used -= r.total_len;
   if (r.type != VREND_TRACE_PAD)
      tr.evicted++;
}

/* Append one record. Allocation-free and syscall-free; the only cost beyond the memcpy is a
 * clock_gettime, which is a vDSO read. */
static void trace_put(uint8_t type, uint8_t cmd, uint32_t ctx_id,
                      const uint32_t *aux, uint32_t aux_count,
                      const void *payload, uint32_t payload_len)
{
   struct vrend_trace_rec r;
   size_t need, tail_room;
   uint8_t *dst;

   if (!tr_on)
      return;
   if (aux_count > MAX_AUX)
      aux_count = MAX_AUX;

   pthread_mutex_lock(&tr_lock);

   need = sizeof r + (size_t)aux_count * 4u + payload_len;
   need = (need + 7u) & ~(size_t)7u;
   if (need > tr.cap / 2) {
      /* absurdly large single record; refuse rather than thrash the ring */
      pthread_mutex_unlock(&tr_lock);
      return;
   }

   /* Never split a record across the end: pad the tail end and wrap. A sliver smaller than a
    * header would be unwalkable -- there would be nowhere to put the total_len that gets a
    * reader past it -- so absorb any such remainder into this record instead. That keeps the
    * invariant every other branch relies on: at entry, tail_room is always >= a full header. */
   tail_room = tr.cap - tr.head;
   if (tail_room >= need && tail_room - need < sizeof r)
      need = tail_room;
   if (tail_room < need) {
      while (tr.used + tail_room > tr.cap)
         evict_one();
      memset(&r, 0, sizeof r);
      r.total_len = (uint32_t)tail_room;
      r.type = VREND_TRACE_PAD;
      memcpy(tr.buf + tr.head, &r, sizeof r);
      tr.used += tail_room;
      tr.head = 0;
   }

   while (tr.used + need > tr.cap)
      evict_one();

   memset(&r, 0, sizeof r);
   r.total_len = (uint32_t)need;
   r.type = type;
   r.cmd = cmd;
   r.ctx_id = (uint16_t)ctx_id;
   r.seq = tr.seq++;
   r.mono_ns = mono_ns();
   r.payload_len = payload_len;
   r.aux_count = aux_count;

   dst = tr.buf + tr.head;
   memcpy(dst, &r, sizeof r);
   dst += sizeof r;
   if (aux_count) {
      memcpy(dst, aux, (size_t)aux_count * 4u);
      dst += (size_t)aux_count * 4u;
   }
   if (payload_len)
      memcpy(dst, payload, payload_len);

   tr.head += need;
   if (tr.head >= tr.cap)
      tr.head = 0;
   tr.used += need;

   pthread_mutex_unlock(&tr_lock);
}

void vrend_trace_submit(uint32_t ctx_id, size_t bytes)
{
   uint32_t aux[1];
   if (!tr_on) return;
   aux[0] = (uint32_t)bytes;
   trace_put(VREND_TRACE_SUBMIT, 0, ctx_id, aux, 1, NULL, 0);
}

void vrend_trace_cmd(uint32_t ctx_id, uint32_t cmd, const uint32_t *buf, uint32_t dwords)
{
   if (!tr_on) return;
   trace_put(VREND_TRACE_CMD, (uint8_t)cmd, ctx_id, NULL, 0, buf, dwords * 4u);
}

void vrend_trace_draw_fb(uint32_t ctx_id, uint32_t width, uint32_t height,
                         uint32_t nr_cbufs, uint32_t gl_id)
{
   uint32_t aux[4];
   if (!tr_on) return;
   aux[0] = width; aux[1] = height; aux[2] = nr_cbufs; aux[3] = gl_id;
   trace_put(VREND_TRACE_DRAW_FB, 0, ctx_id, aux, 4, NULL, 0);
}

void vrend_trace_transfer(uint32_t ctx_id, uint32_t res_handle, int mode, uint32_t level,
                          uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                          uint32_t stride, uint64_t offset)
{
   uint32_t aux[8];
   if (!tr_on) return;
   aux[0] = res_handle; aux[1] = (uint32_t)mode; aux[2] = level; aux[3] = x;
   aux[4] = y; aux[5] = w; aux[6] = h; aux[7] = stride;
   trace_put(VREND_TRACE_TRANSFER, 0, ctx_id, aux, 8, &offset, sizeof offset);
}

void vrend_trace_res_event(struct vrend_trace_res *res)
{
   if (!tr_on)
      return;
   pthread_mutex_lock(&tr_lock);
   if (tr_res_n >= tr_res_cap) {
      uint32_t want = tr_res_cap * 2u;
      struct vrend_trace_res *bigger =
         want > TRACE_RES_MAX ? NULL : realloc(tr_res, (size_t)want * sizeof *tr_res);
      if (bigger) {
         memset(bigger + tr_res_cap, 0, (size_t)(want - tr_res_cap) * sizeof *bigger);
         tr_res = bigger;
         tr_res_cap = want;
      }
   }
   if (tr_res_n >= tr_res_cap) {
      /* Loud once. Silently dropping creates would produce a trace that looks complete and
       * replays into a context missing resources -- a failure that reads as a renderer bug. */
      if (!tr_res_full) {
         tr_res_full = true;
         fprintf(stderr, "[LIMINA-TRACE] resource log full at %u entries; trace is NOT replayable\n",
                 tr_res_cap);
         fflush(stderr);
      }
      pthread_mutex_unlock(&tr_lock);
      return;
   }
   res->seq = tr.seq;
   tr_res[tr_res_n++] = *res;
   pthread_mutex_unlock(&tr_lock);
}

void vrend_trace_transfer_data(uint32_t ctx_id, uint32_t res_handle, uint64_t offset,
                               const void *data, uint32_t len)
{
   uint32_t aux[3];
   if (!tr_on || !data || !len)
      return;
   aux[0] = res_handle;
   aux[1] = (uint32_t)(offset & 0xffffffffu);
   aux[2] = (uint32_t)(offset >> 32);
   trace_put(VREND_TRACE_XFERDATA, 0, ctx_id, aux, 3, data, len);
}

/* The last blob bytes recorded per handle, as a 64-bit FNV-1a. A handle table rather than a
 * per-resource field because the tracer is deliberately reachable from vrend without owning
 * anything of vrend's: it records, and holds no resource state. */


void vrend_trace_blob_data(uint32_t ctx_id, uint32_t res_handle,
                           const void *data, uint32_t len)
{
   uint32_t aux[1];
   uint64_t hash = 1469598103934665603ull;
   const uint8_t *p = data;
   uint32_t slot;

   if (!tr_on || !data || !len || !tr_blob_max)
      return;

   for (uint32_t i = 0; i < len; i++) {
      hash ^= p[i];
      hash *= 1099511628211ull;
   }

   /* Open addressing with a linear probe, and no eviction: a full table stops deduping rather
    * than recording a handle under another's hash, which would drop a frame that really did
    * change. It stops CAPPING too -- past the 64th blob every changed frame is recorded, and
    * the dropped tally below stays silent about it because nothing was dropped. The ring's
    * `evicted` count is the tell in that regime. */
   slot = res_handle % TR_BLOB_SEEN;
   for (uint32_t i = 0; i < TR_BLOB_SEEN; i++) {
      uint32_t k = (slot + i) % TR_BLOB_SEEN;
      if (tr_blob_seen[k].used && tr_blob_seen[k].handle == res_handle) {
         if (tr_blob_seen[k].hash == hash)
            return;
         tr_blob_seen[k].hash = hash;
         if (tr_blob_seen[k].kept >= tr_blob_max) {
            tr_blob_seen[k].dropped++;
            return;
         }
         tr_blob_seen[k].kept++;
         break;
      }
      if (!tr_blob_seen[k].used) {
         tr_blob_seen[k].used = true;
         tr_blob_seen[k].handle = res_handle;
         tr_blob_seen[k].hash = hash;
         tr_blob_seen[k].kept = 1;
         break;
      }
   }

   aux[0] = res_handle;
   trace_put(VREND_TRACE_BLOBDATA, 0, ctx_id, aux, 1, data, len);
}

void vrend_trace_fence(uint32_t ctx_id, uint32_t flags, uint64_t fence_id)
{
   uint32_t aux[1];
   if (!tr_on) return;
   aux[0] = flags;
   trace_put(VREND_TRACE_FENCE, 0, ctx_id, aux, 1, &fence_id, sizeof fence_id);
}

void vrend_trace_retire_fence(uint32_t ctx_id, uint64_t fence_id)
{
   if (!tr_on) return;
   trace_put(VREND_TRACE_RETIRE, 0, ctx_id, NULL, 0, &fence_id, sizeof fence_id);
}

static void trace_dump_locked(void)
{
   FILE *f;
   uint32_t hdr[16];
   size_t pos, left;

   if (!tr_on)
      return;

   pthread_mutex_lock(&tr_lock);

   f = fopen(tr.out_path, "wb");
   if (!f) {
      fprintf(stderr, "[LIMINA-TRACE] cannot open %s (%s)\n", tr.out_path, strerror(errno));
      fflush(stderr);
      pthread_mutex_unlock(&tr_lock);
      return;
   }

   memset(hdr, 0, sizeof hdr);
   hdr[0] = TRACE_MAGIC;
   hdr[1] = TRACE_VERSION;
   hdr[2] = (uint32_t)(tr.cap / (1024u * 1024u));
   hdr[3] = (uint32_t)(tr.used);
   hdr[4] = (uint32_t)(tr.seq & 0xffffffffu);
   hdr[5] = (uint32_t)(tr.seq >> 32);
   hdr[6] = (uint32_t)(tr.evicted & 0xffffffffu);
   hdr[7] = (uint32_t)(tr.evicted >> 32);
   /* Both clocks at init, so records (monotonic only) can be lined up against the worker log. */
   hdr[8]  = (uint32_t)(tr.base_mono_ns & 0xffffffffu);
   hdr[9]  = (uint32_t)(tr.base_mono_ns >> 32);
   hdr[10] = (uint32_t)(tr.base_realtime_ns & 0xffffffffu);
   hdr[11] = (uint32_t)(tr.base_realtime_ns >> 32);
   hdr[12] = tr_res_n;
   hdr[13] = tr_res_full ? 1u : 0u;
   fwrite(hdr, sizeof hdr, 1, f);
   /* The resource log sits between the header and the ring, so a reader can build every
    * resource before it walks a single command. */
   if (tr_res_n)
      fwrite(tr_res, sizeof *tr_res, tr_res_n, f);

   /* Walk from the oldest live record forward, wrapping once. */
   pos = tr.tail;
   left = tr.used;
   while (left > 0) {
      size_t chunk = tr.cap - pos;
      if (chunk > left)
         chunk = left;
      fwrite(tr.buf + pos, 1, chunk, f);
      left -= chunk;
      pos += chunk;
      if (pos >= tr.cap)
         pos = 0;
   }
   fclose(f);

   fprintf(stderr, "[LIMINA-TRACE] dumped %zu bytes, %llu records, %llu evicted -> %s\n",
           tr.used, (unsigned long long)tr.seq, (unsigned long long)tr.evicted, tr.out_path);

   /* A corpus short of content looks exactly like a client that drew fewer frames, so the cap
    * has to say when it bit. Counting it and never printing it left the two indistinguishable. */
   {
      uint32_t dropped = 0, capped = 0;
      for (uint32_t i = 0; i < TR_BLOB_SEEN; i++) {
         if (tr_blob_seen[i].used && tr_blob_seen[i].dropped) {
            dropped += tr_blob_seen[i].dropped;
            capped++;
         }
      }
      if (dropped)
         fprintf(stderr, "[LIMINA-TRACE] blob content: %u changed frame(s) dropped past the "
                         "cap of %u across %u blob(s); raise LIMINA_VREND_TRACE_BLOB_MAX to "
                         "keep more\n", dropped, tr_blob_max, capped);
   }
   fflush(stderr);

   pthread_mutex_unlock(&tr_lock);
}

void vrend_trace_maybe_dump(void)
{
   /* Retained as the hook the decode loop calls; the FIFO thread now dumps directly, so this
    * only services a request raised by some other means. */
   if (!tr_on)
      return;
   if (atomic_load_explicit(&tr_dump_req, memory_order_relaxed) == 0)
      return;
   atomic_store(&tr_dump_req, 0);
   trace_dump_locked();
}
