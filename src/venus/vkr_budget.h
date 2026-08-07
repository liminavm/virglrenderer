/*
 * Copyright 2026 limina contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef VKR_BUDGET_H
#define VKR_BUDGET_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Host-memory budget for guest-driven GPU allocations.
 *
 * Every allocation venus makes on the guest's behalf lands in the renderer PROCESS's
 * address space, but the guest's own memory accounting cannot see it: a guest that leaks
 * VkDeviceMemory grows the host worker, not itself. On macOS the end state is that the
 * kernel picks the worker as the largest compressed process and SIGKILLs it — the whole
 * VM dies, with no guest-side backtrace and no crash report, at a point unrelated to the
 * allocation that caused it. (2026-08-06: a Vulkan compositor re-allocated a 4K backdrop
 * texture instead of reusing it, ~51 GB/hour, and jetsam took the VM at 142 GB.)
 *
 * This ledger makes that failure mode legible and bounded:
 *
 *   - ACCOUNTING (always on) charges each host allocator call site and keeps a per-context
 *     live total plus an exact-size histogram. A leak with one repeated call site shows up
 *     as "767 x 31.6 MiB live" in one log line, which names the allocation immediately.
 *     It is also the guest-vs-host discriminator: if this ledger stays flat while the
 *     process footprint climbs, the leak is in OUR release path, not the guest's.
 *
 *   - ENFORCEMENT (opt-in, policy set by limina) refuses an allocation past a cap and
 *     deliberately kills the offending context. One guest client loses its GPU context;
 *     the VM and every other client survive.
 *
 * Configured by LIMINA_GPU_MEM_BUDGET_MIB, read once:
 *   unset       -> accounting only, no cap (a bare virglrenderer is unaffected)
 *   0           -> accounting only, no cap (explicitly disabled)
 *   N           -> cap at N MiB
 *
 * WHY ENFORCEMENT KILLS THE CONTEXT RATHER THAN RETURNING AN ERROR. It would be far nicer
 * to hand the guest VK_ERROR_OUT_OF_DEVICE_MEMORY and let it cope at the call site, and
 * that is what an early version of this tried. It cannot work: venus allocates memory
 * ASYNCHRONOUSLY. `vn_device_memory_alloc_simple` (mesa, src/virtio/vulkan) submits
 * vkAllocateMemory with `vn_submit_vkAllocateMemory` and returns VK_SUCCESS as soon as the
 * command is on the ring, without waiting for us; the later `vn_device_memory_wait_alloc`
 * only waits on a ring seqno and never reads a VkResult back. Our `args->ret` is discarded.
 * (`VN_PERF=no_async_mem_alloc` makes it synchronous, but that is a guest-side, per-process
 * opt-in with a round-trip cost, so it cannot be the mechanism.)
 *
 * So the only refusal a host can make stick is to stop the context. That is not a
 * consolation prize — it is strictly better than the status quo, because a host allocation
 * failure ALREADY kills the context today: the guest keeps a handle to a VkDeviceMemory
 * that was never created, and the next command touching it fails object lookup and poisons
 * the ring (this is the "ghost VkDeviceMemory" the neighbouring code warns about — a 2026-07-11
 * dogfood session lost firefox and kitty that way). Doing it deliberately and immediately
 * closes the window where the guest can touch a ghost, and puts the reason in the log.
 *
 * `args->ret` is still set: it is correct under VN_PERF=no_async_mem_alloc and costs nothing.
 *
 * The client's failure will surface guest-side as a lost device / aborted process, which is
 * also what a dozen unrelated venus transport failures look like (see limina-vulkan-oom-lies).
 * A refusal therefore ALWAYS logs a line that names itself and the culprit, immediately
 * before the ring-FATAL line. Never diagnose a venus context death from the guest symptom
 * alone — read the worker log at that timestamp.
 *
 * KNOWN LIMIT — attribution on the vrend path. `vkr_budget_set_context` is called from the
 * venus dispatch entry points only. vrend has its own dispatch, so an IOSurface allocated
 * for a classic vrend resource is charged to whatever context that thread last dispatched
 * for (or to 0). Totals stay exact — charge and credit hit the same ledger — but per-context
 * blame is unreliable for vrend-allocated surfaces. Bind the TLS at the vrend entry too if
 * that ever matters.
 */

/* Read the configuration and announce it. Called from vkr_renderer_init — which the render
 * server runs when the guest creates its FIRST venus context, not at worker startup — so
 * the cap is on record from the moment venus is in use and well before any refusal. A
 * session that kills a context needs the cap in the same log to be interpretable.
 * Idempotent; the accessors below also self-initialize. */
void
vkr_budget_init(void);

/* Bind the calling thread to a venus context, so allocations made deeper in the call stack
 * (image create -> IOSurface, memory alloc -> shm carrier) attribute to the right guest
 * client. Called from the command-dispatch entry points; ctx_id 0 = unattributed. */
void
vkr_budget_set_context(uint32_t ctx_id, const char *debug_name);

/* True if a `size`-byte allocation fits under the cap. Does not charge. Always true when
 * no cap is configured. On false the caller must log via vkr_budget_refused() AND kill the
 * context — see the header comment for why an error return alone does nothing. */
bool
vkr_budget_admit(uint64_t size);

/* Log the refusal of a `size`-byte allocation, with the full per-context breakdown. */
void
vkr_budget_refused(uint64_t size, const char *what);

/* Whether a refusal should kill the context. True by default — see the header comment for
 * why an error return alone does nothing on venus.
 *
 * LIMINA_GPU_MEM_BUDGET_SOFT=1 turns it off, which is ONLY useful paired with a guest
 * running `VN_PERF=no_async_mem_alloc`: that makes the guest's vkAllocateMemory
 * synchronous, so it actually receives VK_ERROR_OUT_OF_DEVICE_MEMORY and can report it at
 * the call site with a backtrace — the debugging shape you want when hunting WHERE a guest
 * client allocates, rather than merely stopping it. Without that guest-side setting, soft
 * mode is worse than useless: the guest ignores the refusal, keeps a ghost VkDeviceMemory,
 * and poisons its ring on the next use anyway — the same context death, with the reason
 * now several commands in the past. */
bool
vkr_budget_kills_context(void);

/* Charge `size` bytes to the current context. `what` names the host allocator ("device
 * memory", "IOSurface", ...) and is retained by pointer — pass a literal. Returns the
 * context id charged, which the caller must hand back to vkr_budget_credit() at free
 * (the object usually outlives the thread binding). */
uint32_t
vkr_budget_charge(uint64_t size, const char *what);

/* Release a charge previously made by vkr_budget_charge(). */
void
vkr_budget_credit(uint32_t ctx_id, uint64_t size);

/* Drop a context's slot. Logs if the context tore down with charges still live — that
 * residual is a leak in our release path, and this is the cheapest place to see it. */
void
vkr_budget_forget_context(uint32_t ctx_id);

/* Total live bytes charged across all contexts. */
uint64_t
vkr_budget_live(void);

/* Log the per-context breakdown (live total + top exact-size buckets). `reason` is
 * prefixed to the line. */
void
vkr_budget_report(const char *reason);

#endif /* VKR_BUDGET_H */
