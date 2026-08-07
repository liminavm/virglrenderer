/*
 * Copyright 2026 limina contributors
 * SPDX-License-Identifier: MIT
 */

#include "vkr_budget.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "vkr_common.h"
#include "vkr_metal_helpers.h"

/* Contexts tracked individually; the rest fold into a shared overflow slot. A desktop
 * session runs one venus context per GPU client, so 64 covers a real workload with room
 * to spare, and the fold keeps a runaway context count from growing this table. */
#define VKR_BUDGET_MAX_SLOTS 64

/* Exact-size buckets per context. The diagnostic value comes from sizes being EXACT — a
 * leak with a single call site shows up as one bucket with a large live count, which is
 * how the 2026-08-06 backdrop leak was identified (767 x 33,177,600 B = 3840x2160x4).
 * Rounding sizes would have destroyed that signal, so buckets are keyed by exact size and
 * one-off sizes are evicted instead. */
#define VKR_BUDGET_MAX_BUCKETS 24

/* Buckets to print per context in a report. */
#define VKR_BUDGET_REPORT_BUCKETS 4

struct vkr_budget_bucket {
   uint64_t size;
   uint64_t live;  /* charges outstanding at this exact size */
   uint64_t total; /* charges ever made — live vs total separates churn from a leak */
   const char *what;
};

struct vkr_budget_slot {
   bool used;
   uint32_t ctx_id;
   char name[32];
   uint64_t live;
   uint64_t peak;
   uint64_t untracked_live; /* charges whose bucket was evicted */
   /* Cumulative: every charge ever made by this context, never decremented. LIVE says what
    * is held; these say what passed through. The difference is the whole diagnosis when the
    * process footprint disagrees with the ledger — a context that charged a thousand
    * surfaces and holds two has freed them, so anything still resident is retained
    * downstream of our release; a context that never charged them at all means the
    * allocator responsible is simply not instrumented. Those are different searches. */
   uint64_t lifetime_charges;
   uint64_t lifetime_bytes;
   struct vkr_budget_bucket buckets[VKR_BUDGET_MAX_BUCKETS];
};

static pthread_mutex_t vkr_budget_lock = PTHREAD_MUTEX_INITIALIZER;
static struct vkr_budget_slot vkr_budget_slots[VKR_BUDGET_MAX_SLOTS];
static uint64_t vkr_budget_total_live;
static uint64_t vkr_budget_total_peak;

/* Cap in bytes; 0 = accounting only. Read once from the environment. */
static uint64_t vkr_budget_cap;
static bool vkr_budget_cap_read;

/* Soft mode: refuse without killing the context. Only useful paired with a guest running
 * VN_PERF=no_async_mem_alloc — see vkr_budget_kills_context(). */
static bool vkr_budget_soft;

/* Periodic census interval in seconds; 0 = off. The host half of a guest-vs-host leak
 * hunt: pair it with the guest's own per-allocation-site census and diff the two series.
 * vmmap is NOT a substitute — it did not move measurably for a 2 GiB live set of venus
 * images during the 2026-08-07 churn probes, which is precisely why this exists. */
static unsigned vkr_budget_census_secs;
static uint64_t vkr_budget_last_census_ns;

static uint64_t
vkr_budget_now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Watermark hysteresis: report once on the way up, re-arm on the way back down, so a
 * session hovering at the line does not spam the log. */
static bool vkr_budget_warned;

/* The context whose commands this thread is currently dispatching. Allocations made deep
 * inside a dispatch (image create -> IOSurface) have no ctx argument to thread through. */
static __thread uint32_t vkr_budget_tls_ctx_id;
static __thread const char *vkr_budget_tls_name;

static void
vkr_budget_init_locked(void)
{
   if (vkr_budget_cap_read)
      return;
   vkr_budget_cap_read = true;

   /* Read the two modifiers BEFORE the cap, so they stay independent of it: the census in
    * particular is most useful with no cap set at all (measure first, bound later). */
   const char *soft = getenv("LIMINA_GPU_MEM_BUDGET_SOFT");
   vkr_budget_soft = soft && soft[0] == '1';

   const char *census = getenv("LIMINA_GPU_MEM_BUDGET_CENSUS");
   if (census && *census) {
      vkr_budget_census_secs = (unsigned)strtoul(census, NULL, 10);
      if (vkr_budget_census_secs) {
         vkr_budget_last_census_ns = vkr_budget_now_ns();
         vkr_log_error("limina GPU budget: census every %us", vkr_budget_census_secs);
      }
   }

   const char *env = getenv("LIMINA_GPU_MEM_BUDGET_MIB");
   if (!env || !*env)
      return;

   char *end = NULL;
   unsigned long long mib = strtoull(env, &end, 10);
   if (end == env || (end && *end)) {
      vkr_log_error("limina GPU budget: ignoring malformed LIMINA_GPU_MEM_BUDGET_MIB=\"%s\"",
                    env);
      return;
   }
   vkr_budget_cap = (uint64_t)mib * 1024u * 1024u;
   /* ERROR level deliberately: one line, once, and it is the context for any refusal that
    * follows — a log showing a killed context without it leaves you guessing what the cap
    * was. It also makes the env plumbing verifiable without provoking a refusal. */
   if (vkr_budget_cap)
      vkr_log_error("limina GPU budget: cap %llu MiB%s", mib,
                    vkr_budget_soft ? " (SOFT — refuse but do not kill the context; only"
                                      " meaningful with VN_PERF=no_async_mem_alloc in the"
                                      " guest)"
                                    : "");
   else
      vkr_log_error("limina GPU budget: no cap (LIMINA_GPU_MEM_BUDGET_MIB=0) — "
                    "accounting only");
}

bool
vkr_budget_kills_context(void)
{
   bool kills;
   pthread_mutex_lock(&vkr_budget_lock);
   vkr_budget_init_locked();
   kills = !vkr_budget_soft;
   pthread_mutex_unlock(&vkr_budget_lock);
   return kills;
}

/* The last slot is RESERVED as the overflow bucket and never assigned to a real context.
 * It has to be reserved rather than borrowed: charge and credit both find their slot by
 * context id, so if overflow folded into some other context's slot, credits would look up
 * the overflowed id, find nothing, and be dropped — the ledger would ratchet upward and
 * eventually refuse everything. Overflow keeps its own id so credits find their way home;
 * only the attribution is lost, never the arithmetic. */
#define VKR_BUDGET_OVERFLOW_CTX UINT32_MAX

/* The shared bucket for classic (vrend) resources — see vkr_budget_set_vrend(). Like the
 * overflow slot it uses an id no venus context can have, so its credits find it and
 * vkr_budget_forget_context() never sweeps it. */
#define VKR_BUDGET_VREND_CTX (UINT32_MAX - 1)

/* Find the slot for ctx_id, creating it if there is room. Never returns NULL. */
static struct vkr_budget_slot *
vkr_budget_slot_locked(uint32_t ctx_id, const char *name)
{
   struct vkr_budget_slot *free_slot = NULL;
   for (unsigned i = 0; i < VKR_BUDGET_MAX_SLOTS; i++) {
      struct vkr_budget_slot *s = &vkr_budget_slots[i];
      if (s->used && s->ctx_id == ctx_id)
         return s;
      if (!s->used && !free_slot && i < VKR_BUDGET_MAX_SLOTS - 1)
         free_slot = s;
   }

   if (!free_slot) {
      struct vkr_budget_slot *overflow = &vkr_budget_slots[VKR_BUDGET_MAX_SLOTS - 1];
      if (!overflow->used) {
         overflow->used = true;
         overflow->ctx_id = VKR_BUDGET_OVERFLOW_CTX;
         snprintf(overflow->name, sizeof(overflow->name), "overflow");
      }
      return overflow;
   }

   free_slot->used = true;
   free_slot->ctx_id = ctx_id;
   snprintf(free_slot->name, sizeof(free_slot->name), "%s", name ? name : "?");
   return free_slot;
}

/* Bucket for an exact size. When the table is full, evict a dead bucket (live == 0)
 * first — those are pure history. If every bucket is live, evict the smallest live
 * footprint, since the whole point of the histogram is to surface the biggest one. */
static struct vkr_budget_bucket *
vkr_budget_bucket_locked(struct vkr_budget_slot *slot, uint64_t size, const char *what)
{
   struct vkr_budget_bucket *victim = NULL;
   for (unsigned i = 0; i < VKR_BUDGET_MAX_BUCKETS; i++) {
      struct vkr_budget_bucket *b = &slot->buckets[i];
      if (b->size == size)
         return b;
      if (!b->size) {
         b->size = size;
         b->what = what;
         return b;
      }
      if (!victim || b->live * b->size < victim->live * victim->size)
         victim = b;
   }

   if (victim->live) {
      /* Losing a live bucket would silently unbalance its future credits; park those
       * bytes in untracked_live so the slot total stays honest. */
      slot->untracked_live += victim->live * victim->size;
   }
   memset(victim, 0, sizeof(*victim));
   victim->size = size;
   victim->what = what;
   return victim;
}

static void
vkr_budget_fmt_size(uint64_t bytes, char *out, size_t out_size)
{
   if (bytes >= 1024ull * 1024 * 1024)
      snprintf(out, out_size, "%.1f GiB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
   else if (bytes >= 1024ull * 1024)
      snprintf(out, out_size, "%.1f MiB", (double)bytes / (1024.0 * 1024.0));
   else
      snprintf(out, out_size, "%" PRIu64 " B", bytes);
}

static void
vkr_budget_report_locked(const char *reason)
{
   char live[32], cap[32];
   vkr_budget_fmt_size(vkr_budget_total_live, live, sizeof(live));
   if (vkr_budget_cap) {
      vkr_budget_fmt_size(vkr_budget_cap, cap, sizeof(cap));
      vkr_log_error("limina GPU budget: %s — %s live of %s cap (%u%%)", reason, live, cap,
                    (unsigned)((vkr_budget_total_live * 100) / vkr_budget_cap));
   } else {
      vkr_log_error("limina GPU budget: %s — %s live (no cap)", reason, live);
   }

   /* The ledger says what we allocated and released. It cannot say whether some OTHER
    * reference is keeping the storage alive past our release — which is exactly the shape
    * of the 2026-08-07 storm (5912 charged, 3 live, 52 GiB held). Print the +1/-1 balance
    * of every retain site next to it, so the two can be read together. */
   {
      char refs[320];
      vkr_mtl_refcount_census(refs, sizeof(refs));
      if (refs[0])
         vkr_log_error("limina GPU budget:   refs — %s", refs);
   }

   for (unsigned i = 0; i < VKR_BUDGET_MAX_SLOTS; i++) {
      struct vkr_budget_slot *s = &vkr_budget_slots[i];
      /* A context that churned hard and holds nothing is REPORTABLE, not boring: it is the
       * signature of memory we charged, credited, and yet the process still has. Filtering
       * on live alone would hide exactly that case. */
      if (!s->used || (!s->live && !s->lifetime_charges))
         continue;

      /* Selection sort of the top buckets, ranked by live footprint but falling back to
       * lifetime volume once the live ones run out — same reason as above. The histogram is
       * tiny and this only runs when reporting. */
      char line[320];
      size_t at = 0;
      bool taken[VKR_BUDGET_MAX_BUCKETS] = { false };
      for (unsigned n = 0; n < VKR_BUDGET_REPORT_BUCKETS; n++) {
         int best = -1;
         for (unsigned b = 0; b < VKR_BUDGET_MAX_BUCKETS; b++) {
            const struct vkr_budget_bucket *cand = &s->buckets[b];
            if (taken[b] || !cand->size || !cand->total)
               continue;
            if (best < 0)
               best = (int)b;
            else {
               const struct vkr_budget_bucket *cur = &s->buckets[best];
               /* Any live bucket outranks any dead one; ties broken by footprint. */
               if ((cand->live > 0) != (cur->live > 0))
                  best = cand->live ? (int)b : best;
               else if (cand->live ? (cand->live * cand->size > cur->live * cur->size)
                                   : (cand->total * cand->size > cur->total * cur->size))
                  best = (int)b;
            }
         }
         if (best < 0)
            break;
         taken[best] = true;

         char each[32];
         vkr_budget_fmt_size(s->buckets[best].size, each, sizeof(each));
         int n_written =
            snprintf(line + at, sizeof(line) - at, "%s%" PRIu64 " x %s (%s, %" PRIu64 " ever)",
                     at ? ", " : "", s->buckets[best].live, each,
                     s->buckets[best].what ? s->buckets[best].what : "?",
                     s->buckets[best].total);
         if (n_written < 0 || (size_t)n_written >= sizeof(line) - at)
            break;
         at += (size_t)n_written;
      }

      char slot_live[32], slot_ever[32];
      vkr_budget_fmt_size(s->live, slot_live, sizeof(slot_live));
      vkr_budget_fmt_size(s->lifetime_bytes, slot_ever, sizeof(slot_ever));
      vkr_log_error("limina GPU budget:   ctx %u [%s]: %s live, %s over %" PRIu64
                    " charges — %s",
                    s->ctx_id, s->name, slot_live, slot_ever, s->lifetime_charges,
                    at ? line : "(no per-size detail)");
   }
}

void
vkr_budget_init(void)
{
   pthread_mutex_lock(&vkr_budget_lock);
   vkr_budget_init_locked();
   pthread_mutex_unlock(&vkr_budget_lock);
}

void
vkr_budget_set_context(uint32_t ctx_id, const char *debug_name)
{
   vkr_budget_tls_ctx_id = ctx_id;
   vkr_budget_tls_name = debug_name;
}

uint32_t
vkr_budget_current_ctx(void)
{
   return vkr_budget_tls_ctx_id;
}

void
vkr_budget_set_vrend(void)
{
   /* Safe to leave set: every venus dispatch re-binds the thread at its own entry, so this
    * can never bleed into a venus allocation. */
   vkr_budget_tls_ctx_id = VKR_BUDGET_VREND_CTX;
   vkr_budget_tls_name = "vrend";
}

bool
vkr_budget_admit(uint64_t size)
{
   bool ok;
   pthread_mutex_lock(&vkr_budget_lock);
   vkr_budget_init_locked();
   ok = !vkr_budget_cap || vkr_budget_total_live + size <= vkr_budget_cap;
   pthread_mutex_unlock(&vkr_budget_lock);
   return ok;
}

void
vkr_budget_refused(uint64_t size, const char *what)
{
   char want[32];
   vkr_budget_fmt_size(size, want, sizeof(want));

   pthread_mutex_lock(&vkr_budget_lock);
   /* Spell out that this is OUR cap and that WE are killing the context. The guest sees a
    * lost device or an aborted process, which is also what a dozen unrelated venus
    * transport failures look like — without this line, a deliberate refusal is
    * indistinguishable from a transport bug (limina-vulkan-oom-lies). */
   vkr_log_error("limina GPU budget: REFUSING a %s %s for ctx %u [%s] and killing this "
                 "context deliberately. This is limina's host-memory cap "
                 "(LIMINA_GPU_MEM_BUDGET_MIB), NOT a driver or transport failure — this "
                 "guest client is holding more host GPU memory than the cap allows. Only "
                 "this context dies; the VM and its other clients keep running.",
                 want, what, vkr_budget_tls_ctx_id,
                 vkr_budget_tls_name ? vkr_budget_tls_name : "?");
   vkr_budget_report_locked("at refusal");
   pthread_mutex_unlock(&vkr_budget_lock);
}

uint32_t
vkr_budget_charge(uint64_t size, const char *what)
{
   if (!size)
      return 0;

   uint32_t ctx_id;
   bool report = false;

   pthread_mutex_lock(&vkr_budget_lock);
   vkr_budget_init_locked();

   struct vkr_budget_slot *slot =
      vkr_budget_slot_locked(vkr_budget_tls_ctx_id, vkr_budget_tls_name);
   /* The SLOT's id, not the thread's: an overflowed charge lives in the overflow slot and
    * its credit must land there too. */
   ctx_id = slot->ctx_id;
   struct vkr_budget_bucket *bucket = vkr_budget_bucket_locked(slot, size, what);
   bucket->live++;
   bucket->total++;
   slot->live += size;
   slot->lifetime_charges++;
   slot->lifetime_bytes += size;
   if (slot->live > slot->peak)
      slot->peak = slot->live;

   vkr_budget_total_live += size;
   if (vkr_budget_total_live > vkr_budget_total_peak)
      vkr_budget_total_peak = vkr_budget_total_live;

   /* Warn once at 80% of the cap, re-arming below 70%. The breakdown printed here is the
    * whole point: it names the leaking allocation while the VM is still healthy. */
   if (vkr_budget_cap) {
      if (!vkr_budget_warned && vkr_budget_total_live * 10 >= vkr_budget_cap * 8) {
         vkr_budget_warned = true;
         report = true;
      } else if (vkr_budget_warned && vkr_budget_total_live * 10 < vkr_budget_cap * 7) {
         vkr_budget_warned = false;
      }
   }
   if (report)
      vkr_budget_report_locked("80% watermark crossed");

   /* Census on the allocation path rather than from a timer thread: it needs no thread,
    * and a workload that has stopped allocating has nothing new to report anyway. */
   if (vkr_budget_census_secs) {
      const uint64_t now = vkr_budget_now_ns();
      if (now - vkr_budget_last_census_ns >=
          (uint64_t)vkr_budget_census_secs * 1000000000ull) {
         vkr_budget_last_census_ns = now;
         vkr_budget_report_locked("census");
      }
   }
   pthread_mutex_unlock(&vkr_budget_lock);

   return ctx_id;
}

void
vkr_budget_credit(uint32_t ctx_id, uint64_t size)
{
   if (!size)
      return;

   pthread_mutex_lock(&vkr_budget_lock);
   struct vkr_budget_slot *slot = NULL;
   for (unsigned i = 0; i < VKR_BUDGET_MAX_SLOTS; i++) {
      if (vkr_budget_slots[i].used && vkr_budget_slots[i].ctx_id == ctx_id) {
         slot = &vkr_budget_slots[i];
         break;
      }
   }

   /* A credit with no slot means the context was already forgotten, which zeroed its
    * charges — dropping it here keeps the total from going negative. */
   if (slot) {
      struct vkr_budget_bucket *bucket = NULL;
      for (unsigned b = 0; b < VKR_BUDGET_MAX_BUCKETS; b++) {
         if (slot->buckets[b].size == size) {
            bucket = &slot->buckets[b];
            break;
         }
      }
      if (bucket && bucket->live)
         bucket->live--;
      else if (slot->untracked_live >= size)
         slot->untracked_live -= size;

      slot->live = slot->live >= size ? slot->live - size : 0;
      vkr_budget_total_live =
         vkr_budget_total_live >= size ? vkr_budget_total_live - size : 0;
   }
   pthread_mutex_unlock(&vkr_budget_lock);
}

void
vkr_budget_forget_context(uint32_t ctx_id)
{
   if (ctx_id == VKR_BUDGET_OVERFLOW_CTX)
      return; /* shared bucket: it outlives every context that folded into it */

   pthread_mutex_lock(&vkr_budget_lock);
   for (unsigned i = 0; i < VKR_BUDGET_MAX_SLOTS; i++) {
      struct vkr_budget_slot *s = &vkr_budget_slots[i];
      if (!s->used || s->ctx_id != ctx_id)
         continue;

      /* Everything this context allocated should have been released by teardown. A
       * residual is either a missed credit here or a genuine leak in our release path —
       * and this is the cheapest place in the stack to see the difference. */
      if (s->live) {
         char residual[32];
         vkr_budget_fmt_size(s->live, residual, sizeof(residual));
         vkr_log_error("limina GPU budget: ctx %u [%s] destroyed with %s still charged — "
                       "host GPU memory was not released at teardown",
                       s->ctx_id, s->name, residual);
         vkr_budget_total_live =
            vkr_budget_total_live >= s->live ? vkr_budget_total_live - s->live : 0;
      }

      /* Log the lifetime totals for EVERY teardown, not just the ones with a residual.
       * The census is a sampler: a context that is born and dies between two ticks is
       * otherwise invisible, and short-lived clients are exactly what one reaches for
       * when reproducing a churn bug. This line is also the leak signature worth having
       * in production logs — "destroyed after N GiB of lifetime charges" says what a
       * client cost us even when it gave every byte back. */
      if (s->lifetime_charges) {
         char lifetime[32], peak[32];
         vkr_budget_fmt_size(s->lifetime_bytes, lifetime, sizeof(lifetime));
         vkr_budget_fmt_size(s->peak, peak, sizeof(peak));
         vkr_log_error("limina GPU budget: ctx %u [%s] destroyed — lifetime %" PRIu64
                       " charges totalling %s, peak %s",
                       s->ctx_id, s->name, s->lifetime_charges, lifetime, peak);
      }
      memset(s, 0, sizeof(*s));
      break;
   }
   pthread_mutex_unlock(&vkr_budget_lock);
}

uint64_t
vkr_budget_live(void)
{
   pthread_mutex_lock(&vkr_budget_lock);
   uint64_t live = vkr_budget_total_live;
   pthread_mutex_unlock(&vkr_budget_lock);
   return live;
}

void
vkr_budget_report(const char *reason)
{
   pthread_mutex_lock(&vkr_budget_lock);
   vkr_budget_report_locked(reason);
   pthread_mutex_unlock(&vkr_budget_lock);
}
