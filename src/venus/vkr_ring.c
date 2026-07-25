/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_ring.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

#include "vn_protocol_renderer_dispatches.h"

#include "vkr_context.h"

static inline void *
get_resource_pointer(const struct vkr_resource *res, size_t offset)
{
   assert(offset < res->size);
   return res->u.data + offset;
}

static void
vkr_ring_init_extra(struct vkr_ring *ring, const struct vkr_ring_layout *layout)
{
   struct vkr_ring_extra *extra = &ring->extra;

   extra->offset = layout->extra.begin;
   extra->region = vkr_region_make_relative(&layout->extra);
}

static void
vkr_ring_init_buffer(struct vkr_ring *ring, const struct vkr_ring_layout *layout)
{
   struct vkr_ring_buffer *buf = &ring->buffer;

   buf->size = vkr_region_size(&layout->buffer);
   assert(util_is_power_of_two_nonzero(buf->size));
   buf->mask = buf->size - 1;
   buf->cur = 0;
   buf->data = get_resource_pointer(layout->resource, layout->buffer.begin);
}

static bool
vkr_ring_init_control(struct vkr_ring *ring,
                      const struct vkr_ring_layout *layout,
                      bool replaying)
{
   struct vkr_ring_control *ctrl = &ring->control;

   ctrl->head = get_resource_pointer(layout->resource, layout->head.begin);
   ctrl->tail = get_resource_pointer(layout->resource, layout->tail.begin);
   ctrl->status = get_resource_pointer(layout->resource, layout->status.begin);

   /* we will manage head and status, and we expect them to be 0 initially —
    * except at snapshot-replay (limina), where the control words hold the
    * restored pre-suspend cursors and must be preserved, not rejected */
   if (!replaying && (*ctrl->head || *ctrl->status))
      return false;

   return true;
}

static void
vkr_ring_store_head(struct vkr_ring *ring, uint32_t ring_head)
{
   /* the renderer is expected to load the head with memory_order_acquire,
    * forming a release-acquire ordering
    */
   atomic_store_explicit(ring->control.head, ring_head, memory_order_release);
}

static uint32_t
vkr_ring_load_tail(const struct vkr_ring *ring)
{
   /* the driver is expected to store the tail with memory_order_release,
    * forming a release-acquire ordering
    */
   return atomic_load_explicit(ring->control.tail, memory_order_acquire);
}

/* limina (idle-wakeups): seq_cst tail load for the idle/notify handshake ONLY.
 *
 * The handshake is a store-buffer (SB) litmus: the host stores the IDLE status bit then
 * loads the tail; the guest (mesa vn_ring_submit) stores the tail then loads the status
 * (seq_cst) and emits vkNotifyRingMESA iff it observes IDLE. A lost wakeup needs BOTH the
 * host to miss the guest's tail AND the guest to miss the host's IDLE — which seq_cst on the
 * host store (vkr_ring_set_status_bits) AND this load forbids: whichever seq_cst op the total
 * order puts second observes the first (the guest's tail store is release but sequenced-before
 * its seq_cst status load, so it inherits the ordering). The plain load_tail above is
 * memory_order_acquire — correct and cheaper for the producer/consumer *data* path, but on
 * weakly-ordered Apple Silicon an acquire load (LDAPR) can reorder ahead of the prior seq_cst
 * IDLE store, reopening the SB race. That race — not #28 blob coherency, which is a GPU-write
 * SLC artifact, whereas the IDLE bit is a host-CPU write to CPU-coherent memory — is the real
 * cause of the missed-notify #30 hang the 2 ms poll used to paper over. Use seq_cst (LDAR) here
 * so the handshake is race-free and the idle wait can block indefinitely (0 idle wakeups). */
static uint32_t
vkr_ring_load_tail_seqcst(const struct vkr_ring *ring)
{
   return atomic_load_explicit(ring->control.tail, memory_order_seq_cst);
}

static void
vkr_ring_unset_status_bits(struct vkr_ring *ring, uint32_t mask)
{
   atomic_fetch_and_explicit(ring->control.status, ~mask, memory_order_seq_cst);
}

static void
vkr_ring_read_buffer(struct vkr_ring *ring, void *data, uint32_t size)
{
   struct vkr_ring_buffer *buf = &ring->buffer;

   const size_t offset = buf->cur & buf->mask;
   assert(size <= buf->size);
   if (offset + size <= buf->size) {
      memcpy(data, buf->data + offset, size);
   } else {
      const size_t s = buf->size - offset;
      memcpy(data, buf->data + offset, s);
      memcpy((uint8_t *)data + s, buf->data, size - s);
   }

   /* advance cur */
   buf->cur += size;
}

static inline void
vkr_ring_init_dispatch(struct vkr_ring *ring, struct vkr_context *ctx)
{
   ring->dispatch = ctx->dispatch;
   ring->dispatch.encoder = (struct vn_cs_encoder *)&ring->encoder;
   ring->dispatch.decoder = (struct vn_cs_decoder *)&ring->decoder;
}

struct vkr_ring *
vkr_ring_create(const struct vkr_ring_layout *layout,
                struct vkr_context *ctx,
                uint64_t idle_timeout)
{
   struct vkr_ring *ring = calloc(1, sizeof(*ring));
   if (!ring)
      return NULL;

   ring->resource = layout->resource;

   if (!vkr_ring_init_control(ring, layout, ctx->replaying))
      goto err_init_control;

   vkr_ring_init_buffer(ring, layout);
   vkr_ring_init_extra(ring, layout);

   /* limina snapshot-replay: resume the read cursor at the restored head (the
    * quiesce drained the ring, so head == tail; the thread then simply waits
    * for the resumed guest's next submission) */
   if (ctx->replaying) {
      ring->buffer.cur = *ring->control.head;
      vkr_log("limina ring replay-create ctx %u: head=%u tail=%u status=0x%x cur=%u",
              ctx->ctx_id, *ring->control.head, *ring->control.tail,
              *ring->control.status, ring->buffer.cur);
   }

   ring->cmd = malloc(ring->buffer.size);
   if (!ring->cmd)
      goto err_cmd_malloc;

   if (vkr_cs_decoder_init(&ring->decoder, ctx))
      goto err_cs_decoder_init;

   if (vkr_cs_encoder_init(&ring->encoder, &ctx->cs_fatal_error))
      goto err_cs_encoder_init;

   vkr_ring_init_dispatch(ring, ctx);

   ring->idle_timeout = idle_timeout;

   if (mtx_init(&ring->mutex, mtx_plain) != thrd_success)
      goto err_mtx_init;

   if (cnd_init(&ring->cond) != thrd_success)
      goto err_cond_init;

   list_inithead(&ring->limina_barriers);

   return ring;

err_cond_init:
   mtx_destroy(&ring->mutex);
err_mtx_init:
   vkr_cs_encoder_fini(&ring->encoder);
err_cs_encoder_init:
   vkr_cs_decoder_fini(&ring->decoder);
err_cs_decoder_init:
   free(ring->cmd);
err_cmd_malloc:
err_init_control:
   free(ring);
   return NULL;
}

void
vkr_ring_destroy(struct vkr_ring *ring)
{
   list_del(&ring->head);

   assert(!ring->started);
   vkr_cs_decoder_fini(&ring->decoder);
   vkr_cs_encoder_fini(&ring->encoder);
   mtx_destroy(&ring->mutex);
   cnd_destroy(&ring->cond);
   free(ring->cmd);
   free(ring);
}

static uint64_t
vkr_ring_now(void)
{
   const uint64_t ns_per_sec = 1000000000llu;
   struct timespec now;
   if (clock_gettime(CLOCK_MONOTONIC, &now))
      return 0;
   return ns_per_sec * now.tv_sec + now.tv_nsec;
}

/* limina wake-trace (LIMINA_WAKE_TRACE=1): ring-thread wake accounting, ~5s cadence to
 * stderr — how many timed poll sleeps the relax ladder burns, how often the poll window
 * actually catches new work vs. the ring parking and resuming via doorbell. Used for
 * host-wakeup attribution (limina docs/perf/overhead-inventory.md). Single ring thread
 * per context; plain statics are fine for a diagnostic aggregate across rings. */
static struct {
   int enabled; /* -1 unknown, 0 off, 1 on */
   uint64_t last_report_ns;
   uint64_t sleeps;
   uint64_t poll_resumes;
   uint64_t parks;
   uint64_t park_resumes;
} vkr_ring_wake_trace = { .enabled = -1 };

/* limina wake-chain profile (LIMINA_RING_WAKE_PROFILE=1).
 *
 * A submit that lands on a parked ring has to walk: guest ioctl -> VM exit -> libkrun's gpu
 * worker -> vkr_ring_notify's cnd_signal -> the ring thread waking out of cnd_wait -> decode.
 * Guest-side measurement (gnome-shell-rs venus-cost.md §9.4) shows the first submit after an
 * idle gap costing ~1 ms against ~0.05 ms back-to-back, growing with the length of the gap,
 * and identical whether the guest thread spins or sleeps — i.e. the cost is host-side. This
 * splits that interval into the two hops we can see from here:
 *
 *   signal->resume : cnd_signal to the ring thread actually running. Pure thread wake latency.
 *   resume->decode : ring thread running to starting on the command.
 *
 * Cross-tabbed by how long the ring had been parked, because "grows with the gap" is the claim
 * under test — an effect that is flat across buckets is not the one we are looking for.
 *
 * If both hops are small the cost is upstream of us (VM exit, or libkrun's gpu worker being
 * slow to wake), and the next instrumentation goes there. */
#define VKR_WAKE_BUCKETS 4
static const char *const vkr_wake_bucket_name[VKR_WAKE_BUCKETS] = { "<1ms", "1-4ms", "4-16ms",
                                                                    ">=16ms" };
static struct {
   int enabled; /* -1 unknown, 0 off, 1 on */
   uint64_t last_report_ns;
   struct {
      uint64_t count;
      uint64_t signal_sum, signal_max;
      uint64_t decode_sum, decode_max;
      uint64_t lost_signal; /* woke with no recorded cnd_signal (spurious/racy) */
   } b[VKR_WAKE_BUCKETS];
} vkr_ring_wake_profile = { .enabled = -1 };

static bool
vkr_ring_wake_profile_enabled(void)
{
   if (vkr_ring_wake_profile.enabled < 0) {
      /* Value-aware, not merely present: the supervisor turns this on by default for the
       * instrumented build, so "=0" has to be able to turn it back off. */
      const char *v = getenv("LIMINA_RING_WAKE_PROFILE");
      vkr_ring_wake_profile.enabled = v && strcmp(v, "0") != 0;
   }
   return vkr_ring_wake_profile.enabled > 0;
}

static void
vkr_ring_wake_profile_add(uint64_t idle_ns, uint64_t signal_ns, uint64_t decode_ns, bool lost)
{
   unsigned i = idle_ns < 1000000ull      ? 0
                : idle_ns < 4000000ull    ? 1
                : idle_ns < 16000000ull   ? 2
                                          : 3;
   vkr_ring_wake_profile.b[i].count++;
   if (lost) {
      vkr_ring_wake_profile.b[i].lost_signal++;
   } else {
      vkr_ring_wake_profile.b[i].signal_sum += signal_ns;
      if (signal_ns > vkr_ring_wake_profile.b[i].signal_max)
         vkr_ring_wake_profile.b[i].signal_max = signal_ns;
   }
   vkr_ring_wake_profile.b[i].decode_sum += decode_ns;
   if (decode_ns > vkr_ring_wake_profile.b[i].decode_max)
      vkr_ring_wake_profile.b[i].decode_max = decode_ns;

   const uint64_t now = vkr_ring_now();
   if (!vkr_ring_wake_profile.last_report_ns)
      vkr_ring_wake_profile.last_report_ns = now;
   if (now - vkr_ring_wake_profile.last_report_ns < 5000000000ull)
      return;

   for (unsigned k = 0; k < VKR_WAKE_BUCKETS; k++) {
      const uint64_t n = vkr_ring_wake_profile.b[k].count;
      if (!n)
         continue;
      const uint64_t sn = n - vkr_ring_wake_profile.b[k].lost_signal;
      fprintf(stderr,
              "[RINGWAKE idle %-6s] n=%-5llu signal->resume avg %6.3f ms max %6.3f ms | "
              "resume->decode avg %6.3f ms max %6.3f ms | lost_signal=%llu\n",
              vkr_wake_bucket_name[k], (unsigned long long)n,
              sn ? (double)vkr_ring_wake_profile.b[k].signal_sum / sn / 1e6 : 0.0,
              (double)vkr_ring_wake_profile.b[k].signal_max / 1e6,
              (double)vkr_ring_wake_profile.b[k].decode_sum / n / 1e6,
              (double)vkr_ring_wake_profile.b[k].decode_max / 1e6,
              (unsigned long long)vkr_ring_wake_profile.b[k].lost_signal);
   }
   memset(vkr_ring_wake_profile.b, 0, sizeof(vkr_ring_wake_profile.b));
   vkr_ring_wake_profile.last_report_ns = now;
}

static void
vkr_ring_wake_trace_add(uint64_t *counter)
{
   if (vkr_ring_wake_trace.enabled < 0)
      vkr_ring_wake_trace.enabled = getenv("LIMINA_WAKE_TRACE") != NULL;
   if (!vkr_ring_wake_trace.enabled)
      return;
   (*counter)++;
   const uint64_t now = vkr_ring_now();
   if (!vkr_ring_wake_trace.last_report_ns)
      vkr_ring_wake_trace.last_report_ns = now;
   const uint64_t elapsed = now - vkr_ring_wake_trace.last_report_ns;
   if (elapsed >= 5000000000ull) {
      const double secs = (double)elapsed / 1e9;
      fprintf(stderr,
              "[WAKETRACE vkr_ring] poll_sleeps=%.0f/s poll_resumes=%.0f/s parks=%.0f/s "
              "park_resumes=%.0f/s\n",
              (double)vkr_ring_wake_trace.sleeps / secs,
              (double)vkr_ring_wake_trace.poll_resumes / secs,
              (double)vkr_ring_wake_trace.parks / secs,
              (double)vkr_ring_wake_trace.park_resumes / secs);
      vkr_ring_wake_trace.sleeps = 0;
      vkr_ring_wake_trace.poll_resumes = 0;
      vkr_ring_wake_trace.parks = 0;
      vkr_ring_wake_trace.park_resumes = 0;
      vkr_ring_wake_trace.last_report_ns = now;
   }
}

static void
vkr_ring_relax(uint32_t *iter, uint32_t warm_rungs)
{
   /* limina (host-wakeup trim, adaptive plateau — measured A/B, task #38/#39; plateau DEPTH
    * made adaptive, task #42):
    *
    * An idle ring is polled on a two-phase backoff. The first 16 spins are a cheap
    * thrd_yield (no timed wakeup). Then a RESPONSIVE PLATEAU: short sleeps ramping
    * 10 -> 40 us and holding at 40 us for `warm_rungs` rungs. Only once the ring has stayed
    * idle PAST the plateau does it fall back to a DEEP-IDLE sleep of 640 us.
    *
    * Why two phases: relax_iter resets to 0 on every processed command, so an
    * actively-fed ring (a game / uncapped GL, whose inter-submit gaps are a few
    * hundred us) never leaves the plateau -> ~40 us worst-case pickup latency ->
    * full throughput. A quiet desktop (idle gaps of milliseconds+) crosses the
    * plateau once and then sleeps at 640 us. Measured on vkmark (uncapped,
    * submit-latency-bound): a flat 640 us cap scored 1193 at 11k wakeups/s and a flat
    * 40 us cap scored ~2340 but would sleep ~25k/s at idle; this plateau scores ~2340
    * at ~18k under load while holding low at true idle. (Upstream slept once per
    * ITERATION, ~15.5k wakeups/s — the original regression this whole path fixes.)
    *
    * PLATEAU DEPTH is now caller-supplied (`warm_rungs`) and adapts to the ring's recent
    * inter-flush cadence (see vkr_ring_thread): a bursting ring keeps the full warm plateau
    * (WARM_RUNGS_MAX) for low latency; a ring in a demonstrably SPARSE regime (recent gaps
    * long, i.e. it keeps walking to the idle_timeout park) uses a MINIMAL plateau so it
    * reaches the deep-idle sleep in a few rungs instead of ~16, cutting the "plateau-walk to
    * an inevitable park" poll-sleeps. This never parks earlier than idle_timeout, so the
    * guest notify-rate-limit handshake is untouched — it only makes the mandatory pre-park
    * poll window coarser when a flush is unlikely to land in it. (task #42: measured on
    * clean-fullscreen blobs the plateau-walk was ~3-4k/s of the wakeups; a purely idle/
    * parking ring like the fullscreen compositor burnt ~1.9k/s walking a plateau it never
    * needed.) */
   const uint32_t busy_wait_order = 4;
   const uint32_t base_sleep_us = 10;
   const uint32_t warm_sleep_us = 40;  /* responsive plateau: low pickup latency for active rings */
   const uint32_t idle_sleep_us = 640; /* deep-idle backoff: few wakeups on a quiet desktop */

   (*iter)++;
   if (*iter < (1u << busy_wait_order)) {
      thrd_yield();
      return;
   }

   vkr_ring_wake_trace_add(&vkr_ring_wake_trace.sleeps);

   const uint32_t rung = *iter - (1u << busy_wait_order);
   const uint32_t us = (rung < warm_rungs)
                          ? MIN2(base_sleep_us << MIN2(2 * rung, 16), warm_sleep_us)
                          : idle_sleep_us;
   const struct timespec ts = {
      .tv_sec = us / 1000000,
      .tv_nsec = (us % 1000000) * 1000,
   };
#ifdef __APPLE__
   /* macOS does not implement clock_nanosleep; a unified path is TBD */
   nanosleep(&ts, NULL);
#else
   clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, NULL);
#endif
}

/* task #42: choose the warm-plateau depth for the UPCOMING idle period from a TIME-WEIGHTED
 * profile of the ring over a sliding window. Called once per drain with the drain time.
 *
 * The insight: the expensive "plateau-walk" happens on the inter-frame idle gap, which FOLLOWS
 * a burst — so immediate history (the just-ended gaps) can't predict it. But over a window the
 * REGIME is obvious: a vsync-capped app spends the BULK OF WALL-CLOCK idle in a long per-frame
 * gap, while a saturated, submit-latency-bound app (uncapped vkmark) spends it busy with only
 * sporadic short stalls. We classify by the FRACTION OF WALL-CLOCK spent in long (>= long_idle)
 * idle gaps — time-weighted, so a frame's one long idle dominates even when the app also emits
 * many sub-ms flushes/frame (firefox ~24); a gap-COUNT fraction would misread that as saturated,
 * and any single stray long gap could latch coarsening (the earlier 100 ms "saw one gap" bug).
 * In the CAPPED/SLACK regime (long-idle time >= coarsen_pct% of the window) we coarsen EVERY gap
 * — safe because the slack hides the <=640 us pickup latency (empirically: forced minimal plateau
 * held 60 fps). Otherwise keep the full responsive plateau. Requiring a SUSTAINED time fraction
 * also denies coarsening's own added latency any way to re-arm itself on a latency-bound ring.
 * Starts responsive (relax_coarsen=0). Never changes when the ring parks (idle_timeout), so the
 * guest notify handshake is untouched. Env-tunable for A/B. */
static uint32_t
vkr_ring_profile_warm_rungs(struct vkr_ring *ring, uint64_t drain_now)
{
   static int inited = 0;
   static uint32_t warm_max = 16, warm_min = 2;
   static uint64_t long_idle_ns = 2000000ull; /* 2 ms: a genuine per-cycle (vsync-like) idle gap */
   static uint64_t window_ns = 200000000ull;  /* 200 ms evaluation window */
   static uint32_t coarsen_pct = 50;          /* coarsen when >= this % of wall-clock is long-idle */
   if (!inited) {
      const char *e;
      if ((e = getenv("LIMINA_RELAX_WARM_MAX")))
         warm_max = (uint32_t)atoi(e);
      if ((e = getenv("LIMINA_RELAX_WARM_MIN")))
         warm_min = (uint32_t)atoi(e);
      if ((e = getenv("LIMINA_RELAX_LONG_IDLE_US")))
         long_idle_ns = (uint64_t)atoi(e) * 1000ull;
      if ((e = getenv("LIMINA_RELAX_WINDOW_MS")))
         window_ns = (uint64_t)atoi(e) * 1000000ull;
      if ((e = getenv("LIMINA_RELAX_COARSEN_PCT")))
         coarsen_pct = (uint32_t)atoi(e);
      inited = 1;
   }

   if (ring->relax_last_drain_ns) {
      uint64_t gap = drain_now - ring->relax_last_drain_ns;
      if (gap >= long_idle_ns)
         ring->relax_long_time_ns += gap;
   } else {
      /* first drain on this ring: anchor the window, don't count a bogus startup gap */
      ring->relax_window_start_ns = drain_now;
   }
   ring->relax_last_drain_ns = drain_now;

   uint64_t elapsed = drain_now - ring->relax_window_start_ns;
   if (elapsed >= window_ns) {
      ring->relax_coarsen =
         (ring->relax_long_time_ns * 100 >= elapsed * (uint64_t)coarsen_pct) ? 1u : 0u;
      ring->relax_window_start_ns = drain_now;
      ring->relax_long_time_ns = 0;
   }

   return ring->relax_coarsen ? warm_min : warm_max;
}

static bool
vkr_ring_submit_cmd(struct vkr_ring *ring,
                    const uint8_t *buffer,
                    size_t size,
                    uint32_t ring_head)
{
   struct vkr_cs_decoder *dec = &ring->decoder;
   if (vkr_cs_decoder_get_fatal(dec)) {
      vkr_log("ring_submit_cmd: early bail due to fatal decoder state");
      return false;
   }

   vkr_cs_decoder_set_buffer_stream(dec, buffer, size);

   while (vkr_cs_decoder_has_command(dec)) {
      vn_dispatch_command(&ring->dispatch);
      if (vkr_cs_decoder_get_fatal(dec)) {
         vkr_log("ring_submit_cmd: vn_dispatch_command failed");

         vkr_cs_decoder_reset(dec);
         return false;
      }

      /* update the ring head intra-cs to optimize ring space */
      const uint32_t cur_ring_head = ring_head + (dec->cur - buffer);
      vkr_ring_store_head(ring, cur_ring_head);
      vkr_context_on_ring_seqno_update(ring->dispatch.data, ring->id, cur_ring_head);
   }

   vkr_cs_decoder_reset(dec);
   return true;
}

/* limina (#8): fire present-fence barriers whose target the decode position has
 * passed (or all of them on teardown). Runs on the ring thread, except the
 * teardown sweep which runs wherever the thread exits. */
/* limina snapshot-replay: decode one journal-replayed command on this ring's
 * decoder before the ring thread starts — ring-scoped state (the reply command
 * stream set/seek) is per-decoder and must be re-established here, not on the
 * context decoder. Safe only pre-start: nothing else touches the decoder. */
bool
vkr_ring_replay_cmd(struct vkr_ring *ring, const void *buffer, size_t size)
{
   assert(!ring->started);

   struct vkr_cs_decoder *dec = &ring->decoder;
   if (vkr_cs_decoder_get_fatal(dec))
      return false;

   vkr_cs_decoder_set_buffer_stream(dec, buffer, size);
   while (vkr_cs_decoder_has_command(dec)) {
      vn_dispatch_command(&ring->dispatch);
      if (vkr_cs_decoder_get_fatal(dec)) {
         vkr_log("ring_replay_cmd: vn_dispatch_command failed");
         vkr_cs_decoder_reset(dec);
         return false;
      }
   }
   vkr_cs_decoder_reset(dec);
   return true;
}

static void
vkr_ring_check_limina_barriers(struct vkr_ring *ring, bool fire_all)
{
   if (!atomic_load_explicit(&ring->has_limina_barriers, memory_order_acquire))
      return;

   struct list_head fired;
   list_inithead(&fired);

   mtx_lock(&ring->mutex);
   list_for_each_entry_safe (struct vkr_limina_barrier, bar, &ring->limina_barriers, head) {
      if (fire_all || vkr_seqno_ge(ring->buffer.cur, bar->target)) {
         list_del(&bar->head);
         list_addtail(&bar->head, &fired);
      }
   }
   if (list_is_empty(&ring->limina_barriers))
      atomic_store_explicit(&ring->has_limina_barriers, false, memory_order_release);
   mtx_unlock(&ring->mutex);

   list_for_each_entry_safe (struct vkr_limina_barrier, bar, &fired, head) {
      vkr_limina_present_barrier_release(bar->pf);
      free(bar);
   }
}

void
vkr_ring_add_limina_barrier(struct vkr_ring *ring, struct vkr_present_fence *pf)
{
   struct vkr_limina_barrier *bar = malloc(sizeof(*bar));
   if (!bar || !ring->started) {
      /* fall back to "already passed": weaker ordering beats a stuck frame */
      free(bar);
      vkr_limina_present_barrier_release(pf);
      return;
   }

   bar->target = vkr_ring_load_tail(ring);
   bar->pf = pf;

   mtx_lock(&ring->mutex);
   list_addtail(&bar->head, &ring->limina_barriers);
   atomic_store_explicit(&ring->has_limina_barriers, true, memory_order_release);
   mtx_unlock(&ring->mutex);

   /* wake an idle ring thread so a quiescent ring fires the barrier promptly */
   vkr_ring_notify(ring);
}

static int
vkr_ring_thread(void *arg)
{
   struct vkr_ring *ring = arg;
   struct vkr_context *ctx = ring->dispatch.data;
   char thread_name[16];

   snprintf(thread_name, ARRAY_SIZE(thread_name), "vkr-ring-%d", ctx->ctx_id);
   u_thread_setname(thread_name);
   if (ring->prio_valid && setpriority(PRIO_PROCESS, 0, ring->prio)) {
#ifdef DEBUG
      /* Currently venus doesn't forward the CAP_SYS_NICE request upon forking, so
       * requesting a high priority outside of the normal range would be at the best
       * effort. Thus only enable logging on debug build.
       */
      vkr_log("failed to set ring thread priority to %d (%d)", ring->prio, errno);
#endif
   }

   uint64_t last_submit = vkr_ring_now();
   uint32_t relax_iter = 0;
   uint32_t warm_rungs = 16; /* task #42: plateau depth for the current idle period; the profile
                              * classifier updates it per drain. Starts responsive. */
   int ret = 0;
   while (ring->started) {
      /* limina (#8): everything decoded so far has fully executed — fire any
       * present-fence barriers the decode position has passed. */
      vkr_ring_check_limina_barriers(ring, false);

      bool wait = false;
      if (vkr_ring_now() >= last_submit + ring->idle_timeout) {
         ring->pending_notify = false;
         vkr_ring_set_status_bits(ring, VK_RING_STATUS_IDLE_BIT_MESA);
         /* seq_cst so this load is SC-ordered with the IDLE store above — closes the
          * store-buffer race against the guest's store-tail/load-status notify handshake
          * (see vkr_ring_load_tail_seqcst). */
         wait = ring->buffer.cur == vkr_ring_load_tail_seqcst(ring);
         if (!wait)
            vkr_ring_unset_status_bits(ring, VK_RING_STATUS_IDLE_BIT_MESA);
      }

      if (wait) {
         TRACE_SCOPE("ring idle");

         vkr_ring_wake_trace_add(&vkr_ring_wake_trace.parks);
         mtx_lock(&ring->mutex);
         if (vkr_ring_wake_profile_enabled()) {
            ring->wake_park_ns = vkr_ring_now();
            ring->wake_signal_ns = 0;
         }
         while (ring->started && !ring->pending_notify) {
            /* limina (idle-wakeups): block indefinitely — the seq_cst IDLE-check handshake
             * (vkr_ring_load_tail_seqcst) guarantees the guest observes the IDLE bit and
             * emits vkNotifyRingMESA, so a quiescent ring parks here with 0 host wakeups
             * (matches upstream). This reverts the #30 2 ms poll, which existed only to
             * survive a "missed notify" that was really the store-buffer race the seq_cst
             * load now closes — not the #28 blob coherency gap it was attributed to. */
            ret = cnd_wait(&ring->cond, &ring->mutex);
            if (ret != thrd_success) {
               vkr_log("%s: ring idle cnd_wait has failed(%d)", __func__, ret);
               ret = -EINVAL;
               goto out;
            }
         }
         vkr_ring_unset_status_bits(ring, VK_RING_STATUS_IDLE_BIT_MESA);
         /* Still under the mutex: wake_signal_ns is written there too. */
         if (vkr_ring_wake_profile_enabled())
            ring->wake_resume_ns = vkr_ring_now();
         mtx_unlock(&ring->mutex);

         if (!ring->started)
            break;

         vkr_ring_wake_trace_add(&vkr_ring_wake_trace.park_resumes);
         last_submit = vkr_ring_now();
         relax_iter = 0;
      }

      const uint32_t cmd_size = vkr_ring_load_tail(ring) - ring->buffer.cur;
      if (cmd_size) {
         if (relax_iter >= (1u << 4))
            vkr_ring_wake_trace_add(&vkr_ring_wake_trace.poll_resumes);

         /* task #42: update the longer-period profile and pick the plateau depth for the next
          * idle period (capped/slack ring => coarse, saturated/latency-bound => responsive). */
         warm_rungs = vkr_ring_profile_warm_rungs(ring, vkr_ring_now());

         if (cmd_size > ring->buffer.size) {
            vkr_log("%s: cmd_size(%u) > ring->buffer.size(%u)", __func__, cmd_size,
                    ring->buffer.size);
            ret = -EINVAL;
            break;
         }

         /* limina wake-chain profile: this is the first command decoded after a park, so
          * close out the sample here — the interval that matters ends when real work starts,
          * not when the thread merely became runnable. One sample per park; wake_resume_ns is
          * cleared so a busy ring's subsequent drains do not count. */
         if (vkr_ring_wake_profile_enabled() && ring->wake_resume_ns) {
            const uint64_t resume = ring->wake_resume_ns;
            /* A stamp outside [park, resume] did not belong to the wake we are closing out:
             * either there was none (0), it predates the park, or vkr_ring_notify raced in
             * after the thread was already running. The last case used to fall through to an
             * unsigned `resume - signal` and print a 1.8e13 ms max. */
            const bool lost = ring->wake_signal_ns < ring->wake_park_ns ||
                              ring->wake_signal_ns > resume;
            vkr_ring_wake_profile_add(resume - ring->wake_park_ns,
                                      lost ? 0 : resume - ring->wake_signal_ns,
                                      vkr_ring_now() - resume, lost);
            ring->wake_resume_ns = 0;
         }

         const uint32_t ring_head = ring->buffer.cur;
         vkr_ring_read_buffer(ring, ring->cmd, cmd_size);

         if (!vkr_ring_submit_cmd(ring, ring->cmd, cmd_size, ring_head)) {
            ret = -EINVAL;
            break;
         }

         last_submit = vkr_ring_now();
         relax_iter = 0;
      } else {
         /* Get the active wait_ring seqno first to ensure ordering. */
         uint32_t wait_ring_seqno = 0;
         if (vkr_context_get_wait_ring_seqno(ctx, ring->id, &wait_ring_seqno)) {
            /* Error out if the latest ring cmd is unable to signal the virtqueue that is
             * currently waiting for this ring. This happens when the driver emits invalid
             * asynchronous ring wait cmds.
             */
            const uint32_t ring_tail = vkr_ring_load_tail(ring);
            if (unlikely(!vkr_seqno_ge(ring_tail, wait_ring_seqno))) {
               vkr_log("%s: ring seqno(%u) unable to reach wait seqno(%u)", __func__,
                       ring_tail, wait_ring_seqno);
               ret = -EINVAL;
               break;
            }
         }

         vkr_ring_relax(&relax_iter, warm_rungs);
      }
   }

out:
   /* limina (#8): fire every remaining barrier so present fences can't dangle
    * past ring teardown (the frames present immediately — stale beats stuck). */
   vkr_ring_check_limina_barriers(ring, true);

   if (ret < 0) {
      vkr_ring_set_status_bits(ring, VK_RING_STATUS_FATAL_BIT_MESA);
      vkr_context_on_ring_fatal(ctx);
   }

   return ret;
}

void
vkr_ring_start(struct vkr_ring *ring)
{
   int ret;

   assert(!ring->started);
   ring->started = true;
   ret = thrd_create(&ring->thread, vkr_ring_thread, ring);
   if (ret != thrd_success)
      ring->started = false;
}

bool
vkr_ring_stop(struct vkr_ring *ring)
{
   mtx_lock(&ring->mutex);
   if (thrd_equal(ring->thread, thrd_current())) {
      mtx_unlock(&ring->mutex);
      return false;
   }
   assert(ring->started);
   ring->started = false;
   cnd_signal(&ring->cond);
   mtx_unlock(&ring->mutex);

   thrd_join(ring->thread, NULL);

   return true;
}

void
vkr_ring_notify(struct vkr_ring *ring)
{
   mtx_lock(&ring->mutex);
   ring->pending_notify = true;
   /* limina wake-chain profile: stamp the signal so the ring thread can price its own wake.
    * Under the mutex, so it cannot race the parked thread's read. */
   if (vkr_ring_wake_profile_enabled())
      ring->wake_signal_ns = vkr_ring_now();
   cnd_signal(&ring->cond);
   mtx_unlock(&ring->mutex);

   {
      TRACE_SCOPE("ring notify done");
   }
}

bool
vkr_ring_write_extra(struct vkr_ring *ring, size_t offset, uint32_t val)
{
   struct vkr_ring_extra *extra = &ring->extra;

   if (unlikely(extra->cached_offset != offset || !extra->cached_data)) {
      const struct vkr_region access = VKR_REGION_INIT(offset, sizeof(val));
      if (!vkr_region_is_valid(&access) || !vkr_region_is_within(&access, &extra->region))
         return false;

      /* Mesa always sets offset to 0 and the cache hit rate will be 100% */
      extra->cached_offset = offset;
      extra->cached_data = get_resource_pointer(ring->resource, extra->offset + offset);
   }

   atomic_store_explicit(extra->cached_data, val, memory_order_release);

   {
      TRACE_SCOPE("ring extra done");
   }

   return true;
}

void
vkr_ring_submit_virtqueue_seqno(struct vkr_ring *ring, uint64_t seqno)
{
   mtx_lock(&ring->mutex);
   ring->virtqueue_seqno = seqno;

   /* There are 3 cases:
    * 1. ring is not waiting on the cond thus no-op
    * 2. ring is idle and then wakes up earlier
    * 3. ring is waiting for roundtrip and then checks seqno again
    */
   cnd_signal(&ring->cond);
   mtx_unlock(&ring->mutex);

   {
      TRACE_SCOPE("submit vq seqno done");
   }
}

bool
vkr_ring_wait_virtqueue_seqno(struct vkr_ring *ring, uint64_t seqno)
{
   TRACE_FUNC();

   bool ok = true;

   mtx_lock(&ring->mutex);
   while (ok && ring->started && ring->virtqueue_seqno < seqno)
      ok = cnd_wait(&ring->cond, &ring->mutex) == thrd_success;
   mtx_unlock(&ring->mutex);

   return ok;
}
