/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 *
 * Full venus ring-stream recorder. Rationale, format and replay contract are in
 * vkr_record.h; this file is the mechanics.
 */

#include "vkr_record.h"

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

#include "vkr_context.h"
#include "vkr_journal.h"

#define VKR_RECORD_MAX_CTX 64

/* One context's prologue: the journal export taken when its first command was
 * recorded. Owned here, freed at teardown. */
struct vkr_record_ctx {
   uint32_t ctx_id;
   uint32_t generation;
   bool live;
   void *journal_blob;
   size_t journal_size;
};

static struct {
   uint8_t *buf;
   size_t cap;
   size_t used;
   uint64_t seq;
   uint64_t records;
   uint32_t flags;
   struct vkr_record_ctx ctxs[VKR_RECORD_MAX_CTX];
   uint32_t nctx;
   uint32_t next_generation;
   char out_path[512];
   char fifo_path[512];
} rec;

static bool rec_on;
static bool rec_inited;
static pthread_mutex_t rec_lock = PTHREAD_MUTEX_INITIALIZER;

/* Guards init against the several dispatch threads that can reach the tee at
 * once; the work itself is done exactly once. */
static pthread_once_t rec_once = PTHREAD_ONCE_INIT;

static void
rec_dump_locked(void);

/* Blocks on a FIFO, mirroring the vrend tracer: asking for a dump costs the
 * dispatch threads nothing until it happens, and the worker's signal set is
 * already crowded enough that stealing one would be a trap for a later reader. */
static void *
rec_fifo_thread(void *arg)
{
   (void)arg;
   for (;;) {
      int fd = open(rec.fifo_path, O_RDONLY);
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
         pthread_mutex_lock(&rec_lock);
         rec_dump_locked();
         pthread_mutex_unlock(&rec_lock);
      }
      close(fd);
   }
   return NULL;
}

static void
rec_init_once(void)
{
   const char *env = getenv("LIMINA_VKR_RECORD");
   const size_t mb = env ? (size_t)strtoul(env, NULL, 10) : 0;
   pthread_t th;

   rec_inited = true;
   if (!mb)
      return;

   /* The prologue is a journal export, so there is no recorder without a
    * journal. Refusing loudly beats arming and producing a corpus whose stream
    * starts from state nothing describes. */
   if (!vkr_journal_enabled()) {
      fprintf(stderr, "[LIMINA-VKR-RECORD] refusing to arm: the journal is off "
                      "(VKR_JOURNAL=0) and it supplies the prologue\n");
      fflush(stderr);
      return;
   }

   rec.cap = mb * 1024u * 1024u;
   rec.buf = malloc(rec.cap);
   if (!rec.buf) {
      fprintf(stderr, "[LIMINA-VKR-RECORD] could not allocate %zu MB; recorder off\n", mb);
      fflush(stderr);
      return;
   }
   /* Touch it now: a first-touch fault inside the dispatch path would be a
    * timing perturbation charged to whatever command happened to be running. */
   memset(rec.buf, 0, rec.cap);

   const char *p = getenv("LIMINA_VKR_RECORD_OUT");
   snprintf(rec.out_path, sizeof rec.out_path, "%s", p ? p : "/tmp/limina-vkr-record.bin");
   p = getenv("LIMINA_VKR_RECORD_FIFO");
   snprintf(rec.fifo_path, sizeof rec.fifo_path, "%s", p ? p : "/tmp/limina-vkr-record.fifo");

   unlink(rec.fifo_path);
   if (mkfifo(rec.fifo_path, 0600) != 0 && errno != EEXIST) {
      fprintf(stderr, "[LIMINA-VKR-RECORD] mkfifo %s failed (%s); no dump trigger\n",
              rec.fifo_path, strerror(errno));
   } else if (pthread_create(&th, NULL, rec_fifo_thread, NULL) == 0) {
      pthread_detach(th);
   }

   rec_on = true;
   fprintf(stderr, "[LIMINA-VKR-RECORD] armed: %zu MB -> %s (dump: echo x > %s)\n", mb,
           rec.out_path, rec.fifo_path);
   fflush(stderr);
}

void
vkr_record_init(void)
{
   pthread_once(&rec_once, rec_init_once);
}

bool
vkr_record_enabled(void)
{
   return rec_on;
}

/* --- append --- */

static inline size_t
rec_align4(size_t n)
{
   return (n + 3u) & ~(size_t)3u;
}

/* Find (or adopt) the per-context slot, capturing its prologue on first sight.
 * Called with rec_lock held. Returns NULL when the context table is full, which
 * stops recording rather than silently dropping a context's commands. */
static struct vkr_record_ctx *
rec_ctx_locked(struct vkr_context *ctx)
{
   const uint32_t ctx_id = ctx->ctx_id;

   for (uint32_t i = 0; i < rec.nctx; i++)
      if (rec.ctxs[i].live && rec.ctxs[i].ctx_id == ctx_id)
         return &rec.ctxs[i];

   if (rec.nctx == VKR_RECORD_MAX_CTX)
      return NULL;

   struct vkr_record_ctx *rc = &rec.ctxs[rec.nctx];
   rc->ctx_id = ctx_id;
   /* Monotonic over adopted contexts, never reset. The guest reuses context ids, so this is the
    * half of the identity that makes a record attributable to one context rather than to a number
    * two contexts happened to share. */
   rc->generation = rec.next_generation++;
   rc->live = true;

   /* The prologue is the state this context's stream starts from, so it must be
    * taken BEFORE the command that triggered this — which it is: the journal
    * does not learn about that command until post_dispatch pushes it, after the
    * recorder tee has run.
    *
    * vkr_journal_export quiesces the journal thread first, so pending messages
    * from other threads' earlier commands are applied. What it cannot see is a
    * RECORDING block still buffered on some thread's TLS batch; arming at
    * process start (the intended use) makes that empty, and arming mid-run
    * therefore gives a best-effort prologue rather than an exact one. */
   if (!vkr_journal_export(ctx->journal, &rc->journal_blob, &rc->journal_size)) {
      rc->journal_blob = NULL;
      rc->journal_size = 0;
   }

   rec.nctx++;
   return rc;
}

void
vkr_record_dispatch(struct vkr_context *ctx,
                    uint64_t ring_id,
                    uint32_t cmd_type,
                    const void *data,
                    size_t size,
                    bool fatal)
{
   if (!rec_on || !ctx)
      return;

   pthread_mutex_lock(&rec_lock);

   /* Already stopped: a truncated corpus is a valid prefix, and appending past
    * the stop would make it neither prefix nor window. */
   if (rec.flags) {
      pthread_mutex_unlock(&rec_lock);
      return;
   }

   if (fatal) {
      /* The command did not take effect, so it must not appear in the stream —
       * and everything after it happened in a context the guest had already
       * broken. Stop, and say why in the header. */
      rec.flags |= VKR_RECORD_FLAG_TRUNC_FATAL;
      pthread_mutex_unlock(&rec_lock);
      return;
   }

   if (!size || !data) {
      pthread_mutex_unlock(&rec_lock);
      return;
   }

   struct vkr_record_ctx *rc = rec_ctx_locked(ctx);
   if (!rc) {
      rec.flags |= VKR_RECORD_FLAG_TRUNC_FULL;
      pthread_mutex_unlock(&rec_lock);
      return;
   }

   const size_t hdr = sizeof(uint64_t) * 2 + sizeof(uint32_t) * 4;
   const size_t need = hdr + rec_align4(size);
   if (rec.used + need > rec.cap) {
      rec.flags |= VKR_RECORD_FLAG_TRUNC_FULL;
      pthread_mutex_unlock(&rec_lock);
      return;
   }

   /* The sequence number is assigned here, inside the same critical section
    * that appends the bytes. Handing out sequence numbers outside the lock lets
    * two threads append in inverted order, and a replayer that trusts stream
    * order would then replay a serialization that never happened. */
   uint8_t *p = rec.buf + rec.used;
   const uint64_t seq = rec.seq++;
   const uint32_t ctx_id = rc->ctx_id;
   const uint32_t generation = rc->generation;
   const uint32_t sz = (uint32_t)size;

   memcpy(p, &seq, sizeof seq);
   p += sizeof seq;
   memcpy(p, &ring_id, sizeof ring_id);
   p += sizeof ring_id;
   memcpy(p, &ctx_id, sizeof ctx_id);
   p += sizeof ctx_id;
   memcpy(p, &generation, sizeof generation);
   p += sizeof generation;
   memcpy(p, &cmd_type, sizeof cmd_type);
   p += sizeof cmd_type;
   memcpy(p, &sz, sizeof sz);
   p += sizeof sz;
   memcpy(p, data, size);
   memset(p + size, 0, rec_align4(size) - size);

   rec.used += need;
   rec.records++;

   pthread_mutex_unlock(&rec_lock);
}

void
vkr_record_context_gone(uint32_t ctx_id)
{
   if (!rec_on)
      return;

   pthread_mutex_lock(&rec_lock);
   for (uint32_t i = 0; i < rec.nctx; i++) {
      if (rec.ctxs[i].live && rec.ctxs[i].ctx_id == ctx_id) {
         /* Keep the prologue: the stream still references this context, and a
          * corpus whose prologue vanished when the guest tore the context down
          * would be unreplayable from its own beginning. Only stop adopting new
          * commands under the id, which the guest may reuse. */
         rec.ctxs[i].live = false;
         break;
      }
   }
   pthread_mutex_unlock(&rec_lock);
}

/* --- dump --- */

static bool
rec_write_all(FILE *f, const void *data, size_t size)
{
   return size == 0 || fwrite(data, 1, size, f) == size;
}

static void
rec_dump_locked(void)
{
   if (!rec_on)
      return;

   FILE *f = fopen(rec.out_path, "wb");
   if (!f) {
      fprintf(stderr, "[LIMINA-VKR-RECORD] cannot write %s (%s)\n", rec.out_path,
              strerror(errno));
      fflush(stderr);
      return;
   }

   uint64_t prologue_bytes = 0;
   uint32_t ctx_count = 0;
   for (uint32_t i = 0; i < rec.nctx; i++) {
      if (!rec.ctxs[i].journal_blob)
         continue;
      ctx_count++;
      prologue_bytes +=
         sizeof(uint32_t) * 2 + sizeof(uint64_t) + rec_align4(rec.ctxs[i].journal_size);
   }

   const uint32_t magic = VKR_RECORD_MAGIC;
   const uint32_t version = VKR_RECORD_VERSION;
   const uint64_t stream_bytes = rec.used;
   bool ok = rec_write_all(f, &magic, sizeof magic) &&
             rec_write_all(f, &version, sizeof version) &&
             rec_write_all(f, &rec.flags, sizeof rec.flags) &&
             rec_write_all(f, &ctx_count, sizeof ctx_count) &&
             rec_write_all(f, &rec.records, sizeof rec.records) &&
             rec_write_all(f, &prologue_bytes, sizeof prologue_bytes) &&
             rec_write_all(f, &stream_bytes, sizeof stream_bytes);

   static const uint8_t zeros[4] = { 0 };
   for (uint32_t i = 0; ok && i < rec.nctx; i++) {
      struct vkr_record_ctx *rc = &rec.ctxs[i];
      if (!rc->journal_blob)
         continue;
      const uint64_t size = rc->journal_size;
      ok = rec_write_all(f, &rc->ctx_id, sizeof rc->ctx_id) &&
           rec_write_all(f, &rc->generation, sizeof rc->generation) &&
           rec_write_all(f, &size, sizeof size) &&
           rec_write_all(f, rc->journal_blob, rc->journal_size) &&
           rec_write_all(f, zeros, rec_align4(rc->journal_size) - rc->journal_size);
   }

   if (ok)
      ok = rec_write_all(f, rec.buf, rec.used);

   fclose(f);

   fprintf(stderr,
           "[LIMINA-VKR-RECORD] %s: %llu records, %zu stream bytes, %u context "
           "prologues (%llu bytes)%s%s%s\n",
           ok ? "wrote" : "FAILED writing", (unsigned long long)rec.records, rec.used,
           ctx_count, (unsigned long long)prologue_bytes,
           rec.flags ? " [TRUNCATED:" : "",
           (rec.flags & VKR_RECORD_FLAG_TRUNC_FULL) ? " cap" : "",
           (rec.flags & VKR_RECORD_FLAG_TRUNC_FATAL) ? " fatal]" : (rec.flags ? "]" : ""));
   fflush(stderr);
}
