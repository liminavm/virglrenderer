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
   /* The prologue is taken from a struct vkr_context, which the control-path tees do not have.
    * A context first seen through a control event is therefore adopted without one, and the flag
    * keeps its first dispatch from skipping the capture. */
   bool prologue_taken;
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

/* Find (or adopt) the per-context slot. Called with rec_lock held. Returns NULL when the context
 * table is full, which stops recording rather than silently dropping a context's commands. */
static struct vkr_record_ctx *
rec_ctx_locked(uint32_t ctx_id)
{
   for (uint32_t i = 0; i < rec.nctx; i++)
      if (rec.ctxs[i].live && rec.ctxs[i].ctx_id == ctx_id)
         return &rec.ctxs[i];

   if (rec.nctx == VKR_RECORD_MAX_CTX)
      return NULL;

   struct vkr_record_ctx *rc = &rec.ctxs[rec.nctx++];
   rc->ctx_id = ctx_id;
   /* Monotonic over adopted contexts, never reset. The guest reuses context ids, so this is the
    * half of the identity that makes a record attributable to one context rather than to a number
    * two contexts happened to share. */
   rc->generation = rec.next_generation++;
   rc->live = true;
   return rc;
}

/* Capture the prologue: the state this context's stream starts from. It must be taken BEFORE the
 * command that triggered it -- which it is: the journal does not learn about that command until
 * post_dispatch pushes it, after the recorder tee has run.
 *
 * vkr_journal_export quiesces the journal thread first, so pending messages from other threads'
 * earlier commands are applied. What it cannot see is a RECORDING block still buffered on some
 * thread's TLS batch; arming at process start (the intended use) makes that empty, and arming
 * mid-run therefore gives a best-effort prologue rather than an exact one. */
static void
rec_take_prologue_locked(struct vkr_record_ctx *rc, struct vkr_context *ctx)
{
   if (rc->prologue_taken)
      return;
   rc->prologue_taken = true;
   if (!vkr_journal_export(ctx->journal, &rc->journal_blob, &rc->journal_size)) {
      rc->journal_blob = NULL;
      rc->journal_size = 0;
   }
}

/* The one append. Called with rec_lock held; returns false having set a truncation flag when the
 * record does not fit, so every caller stops rather than writing a stream with a hole in it. */
static bool
rec_append_locked(uint32_t kind,
                  uint64_t ring_id,
                  uint32_t ctx_id,
                  uint32_t generation,
                  uint32_t op,
                  const void *data,
                  size_t size)
{
   const size_t hdr = sizeof(uint64_t) * 2 + sizeof(uint32_t) * 6;
   const size_t need = hdr + rec_align4(size);
   if (rec.used + need > rec.cap) {
      rec.flags |= VKR_RECORD_FLAG_TRUNC_FULL;
      return false;
   }

   /* The sequence number is assigned here, inside the same critical section that appends the
    * bytes. Handing out sequence numbers outside the lock lets two threads append in inverted
    * order, and a replayer that trusts stream order would then replay a serialization that never
    * happened. */
   uint8_t *p = rec.buf + rec.used;
   const uint64_t seq = rec.seq++;
   const uint32_t sz = (uint32_t)size;
   const uint32_t reserved = 0;

   memcpy(p, &seq, sizeof seq);
   p += sizeof seq;
   memcpy(p, &ring_id, sizeof ring_id);
   p += sizeof ring_id;
   memcpy(p, &ctx_id, sizeof ctx_id);
   p += sizeof ctx_id;
   memcpy(p, &generation, sizeof generation);
   p += sizeof generation;
   memcpy(p, &kind, sizeof kind);
   p += sizeof kind;
   memcpy(p, &op, sizeof op);
   p += sizeof op;
   memcpy(p, &sz, sizeof sz);
   p += sizeof sz;
   memcpy(p, &reserved, sizeof reserved);
   p += sizeof reserved;
   if (size)
      memcpy(p, data, size);
   memset(p + size, 0, rec_align4(size) - size);

   rec.used += need;
   rec.records++;
   return true;
}

/* Shared prologue of every control-path tee: take the lock, refuse if recording has stopped, and
 * resolve the owning context. `ctx_id` 0 means the event names no context. Returns false with the
 * lock RELEASED when there is nothing to record. */
static bool
rec_ctl_begin(uint32_t ctx_id, uint32_t *out_ctx_id, uint32_t *out_generation)
{
   if (!rec_on)
      return false;

   pthread_mutex_lock(&rec_lock);
   if (rec.flags) {
      pthread_mutex_unlock(&rec_lock);
      return false;
   }

   *out_ctx_id = 0;
   *out_generation = 0;
   if (ctx_id) {
      struct vkr_record_ctx *rc = rec_ctx_locked(ctx_id);
      if (!rc) {
         rec.flags |= VKR_RECORD_FLAG_TRUNC_FULL;
         pthread_mutex_unlock(&rec_lock);
         return false;
      }
      *out_ctx_id = rc->ctx_id;
      *out_generation = rc->generation;
   }
   return true;
}

static void
rec_ctl_end(uint32_t op, uint32_t ctx_id, uint32_t generation, const void *payload, size_t size)
{
   rec_append_locked(VKR_RECORD_KIND_CTL, 0, ctx_id, generation, op, payload, size);
   pthread_mutex_unlock(&rec_lock);
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

   struct vkr_record_ctx *rc = rec_ctx_locked(ctx->ctx_id);
   if (!rc) {
      rec.flags |= VKR_RECORD_FLAG_TRUNC_FULL;
      pthread_mutex_unlock(&rec_lock);
      return;
   }
   rec_take_prologue_locked(rc, ctx);

   rec_append_locked(VKR_RECORD_KIND_CMD, ring_id, rc->ctx_id, rc->generation, cmd_type, data,
                     size);

   pthread_mutex_unlock(&rec_lock);
}

/* --- control path --- */

void
vkr_record_ctx_create(uint32_t ctx_id, uint32_t context_init, const char *name, uint32_t nlen)
{
   uint32_t id, gen;
   if (!rec_ctl_begin(ctx_id, &id, &gen))
      return;

   /* A context adopted here is brand new, so its prologue is empty by construction and taking one
    * later would wrongly describe state the stream itself builds. */
   for (uint32_t i = 0; i < rec.nctx; i++)
      if (rec.ctxs[i].live && rec.ctxs[i].ctx_id == ctx_id)
         rec.ctxs[i].prologue_taken = true;

   if (nlen > 256)
      nlen = 256;
   uint8_t payload[16 + 256];
   const uint32_t words[4] = { ctx_id, context_init, nlen, 0 };
   memcpy(payload, words, sizeof words);
   if (nlen && name)
      memcpy(payload + sizeof words, name, nlen);
   rec_ctl_end(VKR_RECORD_CTL_CTX_CREATE, id, gen, payload, sizeof words + nlen);
}

void
vkr_record_ctx_destroy(uint32_t ctx_id)
{
   uint32_t id, gen;
   if (!rec_ctl_begin(ctx_id, &id, &gen))
      return;

   /* Retire the slot here rather than waiting for vkr_journal_destroy: the guest may create a new
    * context under the same id immediately, and it must be adopted as a new generation. The
    * prologue stays -- the stream up to this point still references it. */
   for (uint32_t i = 0; i < rec.nctx; i++)
      if (rec.ctxs[i].live && rec.ctxs[i].ctx_id == ctx_id)
         rec.ctxs[i].live = false;

   const uint32_t payload[2] = { ctx_id, 0 };
   rec_ctl_end(VKR_RECORD_CTL_CTX_DESTROY, id, gen, payload, sizeof payload);
}

void
vkr_record_create_blob(uint32_t res_handle,
                       uint32_t ctx_id,
                       uint32_t blob_mem,
                       uint32_t blob_flags,
                       uint64_t blob_id,
                       uint64_t size,
                       uint32_t num_iovs)
{
   uint32_t id, gen;
   if (!rec_ctl_begin(ctx_id, &id, &gen))
      return;
   struct {
      uint32_t res_handle, ctx_id, blob_mem, blob_flags;
      uint64_t blob_id, size;
      uint32_t num_iovs, pad;
   } payload = { res_handle, ctx_id, blob_mem, blob_flags, blob_id, size, num_iovs, 0 };
   rec_ctl_end(VKR_RECORD_CTL_CREATE_BLOB, id, gen, &payload, sizeof payload);
}

void
vkr_record_import_blob(uint32_t res_handle, uint32_t fd_type, uint64_t size)
{
   uint32_t id, gen;
   if (!rec_ctl_begin(0, &id, &gen))
      return;
   struct {
      uint32_t res_handle, fd_type;
      uint64_t size;
   } payload = { res_handle, fd_type, size };
   rec_ctl_end(VKR_RECORD_CTL_IMPORT_BLOB, id, gen, &payload, sizeof payload);
}

static void
rec_ctl_ctx_res_pair(uint32_t op, uint32_t ctx_id, uint32_t res_handle)
{
   uint32_t id, gen;
   if (!rec_ctl_begin(ctx_id, &id, &gen))
      return;
   const uint32_t payload[2] = { ctx_id, res_handle };
   rec_ctl_end(op, id, gen, payload, sizeof payload);
}

void
vkr_record_attach_resource(uint32_t ctx_id, uint32_t res_handle)
{
   rec_ctl_ctx_res_pair(VKR_RECORD_CTL_ATTACH_RESOURCE, ctx_id, res_handle);
}

void
vkr_record_detach_resource(uint32_t ctx_id, uint32_t res_handle)
{
   rec_ctl_ctx_res_pair(VKR_RECORD_CTL_DETACH_RESOURCE, ctx_id, res_handle);
}

void
vkr_record_resource_unref(uint32_t res_handle)
{
   uint32_t id, gen;
   if (!rec_ctl_begin(0, &id, &gen))
      return;
   const uint32_t payload[2] = { res_handle, 0 };
   rec_ctl_end(VKR_RECORD_CTL_RESOURCE_UNREF, id, gen, payload, sizeof payload);
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
