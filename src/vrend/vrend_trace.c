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
/* Creates are never evicted, so this is a hard cap rather than a window. Overflow is loud. */
#define TRACE_MAX_RES 32768u
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
static uint32_t tr_res_n;
static bool tr_res_full;
static bool tr_on;
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
   tr_res = calloc(TRACE_MAX_RES, sizeof *tr_res);
   if (!tr_res) {
      free(tr.buf);
      tr.buf = NULL;
      return;
   }

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

bool vrend_trace_enabled(void) { return tr_on; }

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
   if (tr_res_n >= TRACE_MAX_RES) {
      /* Loud once. Silently dropping creates would produce a trace that looks complete and
       * replays into a context missing resources -- a failure that reads as a renderer bug. */
      if (!tr_res_full) {
         tr_res_full = true;
         fprintf(stderr, "[LIMINA-TRACE] resource log full at %u entries; trace is NOT replayable\n",
                 TRACE_MAX_RES);
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
