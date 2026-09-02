// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! What the guest has made this process hold, and the cap on it.
//!
//! Every allocation venus makes on the guest's behalf lands in the *renderer's* address space,
//! where the guest's own accounting cannot see it: a guest that leaks `VkDeviceMemory` grows the
//! host worker, not itself. On macOS that ends with the kernel picking the worker as the largest
//! compressed process and killing it -- the whole VM, no guest backtrace, no crash report, at a
//! moment unrelated to the allocation that caused it. Measured 2026-08-06: a Vulkan compositor
//! re-allocated a 4K backdrop instead of reusing it, ~51 GB/hour, and jetsam took the VM at 142 GB.
//!
//! Accounting is always on and enforcement is opt-in, because the two answer different questions.
//! The ledger alone says *which* allocation is growing -- one repeated call site reads as
//! "767 x 31.6 MiB device memory" in a single line -- and it is also the only thing that separates
//! a guest leak from ours: a flat ledger under a climbing process footprint means the leak is in
//! this renderer's release path.
//!
//! **A refusal has to kill the context, because nothing else reaches the guest.** venus allocates
//! asynchronously: mesa's `vn_device_memory_alloc_simple` submits `vkAllocateMemory` and returns
//! `VK_SUCCESS` as soon as the command is on the ring, and `vn_device_memory_wait_alloc` waits on
//! a ring seqno without ever reading a `VkResult` back. The answer is discarded. Stopping the
//! context is not the consolation prize either: a host allocation failure already kills it today,
//! by leaving the guest a handle to memory that was never created and poisoning the ring on the
//! next command that names it. Doing it deliberately closes the window and puts the reason in the
//! log. `ret` is still set, because it is correct under `VN_PERF=no_async_mem_alloc` and costs
//! nothing.
//!
//! **A charge is a value, and crediting it is dropping it.** [`Charge`] lives in the record of
//! whatever it paid for, so every path that retires that record -- a free, a device teardown, a
//! context destroy -- credits the ledger without knowing the ledger exists. There is no release
//! call site to forget, which is the whole reason this is not a pair of `charge`/`credit`
//! functions: a ledger with manual credits has one more place to be wrong every time a new
//! destroy path is added.
//!
//! **Who is charged is structural, not ambient.** The C binds the billing context to the calling
//! *thread*, so an allocation made deeper in the stack attributes to whoever that thread served
//! last -- which is why it also needs a pseudo-context for vrend. Here the context is an argument,
//! because the caller always has one, and there is nothing for a second caller to get wrong.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::ids::CtxId;

/// How the cap is configured, and what it is called.
const CAP_ENV: &str = "LIMINA_GPU_MEM_BUDGET_MIB";
/// Whether a refusal stops at the error, instead of stopping the context.
const SOFT_ENV: &str = "LIMINA_GPU_MEM_BUDGET_SOFT";

/// The live host memory this renderer holds on guests' behalf, and the cap on it.
pub struct Budget {
    /// Bytes, or `None` for accounting with no enforcement -- which is what a bare
    /// virglrenderer, and every replay, runs as.
    cap: Option<u64>,
    /// Whether a refusal leaves the context alive with an error it will not read.
    ///
    /// Only useful paired with a guest running `VN_PERF=no_async_mem_alloc`, which makes its
    /// `vkAllocateMemory` synchronous so the error actually arrives somewhere with a backtrace.
    /// Without that, soft mode is worse than the default: the guest ignores the refusal, keeps a
    /// handle to memory that does not exist, and poisons its ring on the next use anyway -- the
    /// same context death, with the reason several commands in the past.
    soft: bool,
    ledger: Mutex<BTreeMap<CtxId, PerCtx>>,
}

/// One context's live allocations, by what they are and how big.
///
/// Only the histogram. A running total beside it would be a second answer to one question, and
/// the two would disagree the first time a path updated one of them -- so the totals below are
/// derived from this every time they are asked for. There are a handful of contexts and a handful
/// of distinct sizes; the cost is not worth a number that can drift.
#[derive(Default)]
struct PerCtx {
    live: BTreeMap<(&'static str, u64), u32>,
}

impl PerCtx {
    fn bytes(&self) -> u64 {
        self.live.iter().map(|((_, size), n)| size * u64::from(*n)).sum()
    }

    /// The biggest buckets first, which is what names a leak.
    fn worst(&self) -> Vec<(&'static str, u64, u32)> {
        let mut rows: Vec<_> = self.live.iter().map(|((w, size), n)| (*w, *size, *n)).collect();
        rows.sort_by_key(|(_, size, n)| std::cmp::Reverse(size * u64::from(*n)));
        rows
    }
}

/// A charge against the ledger, which credits itself when it is dropped.
///
/// Held by the record of whatever it paid for. Never constructed except by
/// [`Budget::try_charge`], so a charge that exists was admitted, and a charge that is gone has
/// been credited -- there is no third state and nothing to keep in step.
pub struct Charge {
    budget: Arc<Budget>,
    ctx: CtxId,
    what: &'static str,
    size: u64,
}

impl Charge {
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl std::fmt::Debug for Charge {
    /// Never the ledger behind it: a charge is interesting for what it is, and a `Debug` that
    /// walked to the budget would print every other context's business beside it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Charge(ctx {}, {} {})", self.ctx.get(), mib(self.size), self.what)
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        let mut ledger = self.budget.ledger.lock().expect("the budget ledger");
        let Some(per) = ledger.get_mut(&self.ctx) else {
            return;
        };
        if let Some(n) = per.live.get_mut(&(self.what, self.size)) {
            *n -= 1;
            if *n == 0 {
                per.live.remove(&(self.what, self.size));
            }
        }
    }
}

/// One context's key to the ledger.
///
/// Charges are made through this and never against the [`Budget`] directly, which is what makes
/// attribution structural: there is no context argument for a caller to pass wrongly, because the
/// only handle a context can reach is its own. It is also what retires the context's slot --
/// dropping the account is the destroy, so no teardown path has to remember to say so.
pub struct Account {
    budget: Arc<Budget>,
    ctx: CtxId,
}

impl Account {
    pub fn open(budget: &Arc<Budget>, ctx: CtxId) -> Account {
        Account { budget: Arc::clone(budget), ctx }
    }

    /// Take `size` bytes for this context, or say why not. See [`Budget::try_charge`].
    pub fn try_charge(&self, what: &'static str, size: u64) -> Result<Charge, Refused> {
        self.budget.try_charge(self.ctx, what, size)
    }

    /// Whether a refusal should stop this context. See the module doc for why it must, by default.
    pub fn kills_context(&self) -> bool {
        self.budget.kills_context()
    }

    /// What this context holds. The cap is on the total, not on this -- see
    /// [`Budget::live`] for the number that is actually enforced.
    pub fn live(&self) -> u64 {
        self.budget.live_for(self.ctx)
    }

    /// Say what was refused and what the process is holding, which is the whole reason the
    /// histogram exists: one repeated call site reads as a single line naming its size and count.
    pub fn report_refusal(&self, refused: Refused) {
        eprintln!(
            "[virglrs] ctx {}: refused {} -- {} live of {}, and this renderer will not go over",
            self.ctx.get(),
            mib(refused.wanted),
            mib(refused.live),
            mib(refused.cap),
        );
        self.budget.report("what the host is holding");
    }

    /// An account on a ledger of its own, for a test with no renderer around it.
    ///
    /// Tests never read the environment: the harness runs them in one process in parallel, so a
    /// test that set `LIMINA_GPU_MEM_BUDGET_MIB` would be setting it for every other test at once.
    #[cfg(test)]
    pub fn for_test(cap: Option<u64>) -> Account {
        Account::open(&Budget::with_cap(cap, false), CtxId::new(1).expect("1 is not zero"))
    }
}

impl Drop for Account {
    fn drop(&mut self) {
        self.budget.retire(self.ctx);
    }
}

/// A refusal, carrying what it takes to say so.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Refused {
    pub wanted: u64,
    pub live: u64,
    pub cap: u64,
}

impl Budget {
    /// The budget limina configured, read once.
    ///
    /// Unset or zero is accounting with no cap, so a bare virglrenderer and every replay are
    /// unaffected. A value that is not a number is a misconfiguration worth saying out loud
    /// rather than silently running uncapped -- but not worth refusing to start over.
    pub fn from_env() -> Arc<Budget> {
        let cap = match std::env::var(CAP_ENV) {
            Err(_) => None,
            Ok(v) => match v.trim().parse::<u64>() {
                Ok(0) => None,
                Ok(mib) => Some(mib * 1024 * 1024),
                Err(_) => {
                    eprintln!("[virglrs] {CAP_ENV}={v:?} is not a number of MiB; no cap");
                    None
                }
            },
        };
        let soft = std::env::var(SOFT_ENV).is_ok_and(|v| v == "1");
        if let Some(bytes) = cap {
            eprintln!(
                "[virglrs] gpu memory budget: {} MiB, refusal {}",
                bytes / (1024 * 1024),
                if soft { "returns an error the guest will not read" } else { "stops the context" }
            );
        }
        Budget::with_cap(cap, soft)
    }

    /// A budget with the cap given rather than the one configured. The tests' way in: reading the
    /// environment from a test would make every other test in the process depend on it.
    pub fn with_cap(cap: Option<u64>, soft: bool) -> Arc<Budget> {
        Arc::new(Budget { cap, soft, ledger: Mutex::new(BTreeMap::new()) })
    }

    /// Whether a refusal should stop the context. See the module doc for why it must, by default.
    pub fn kills_context(&self) -> bool {
        !self.soft
    }

    /// Take `size` bytes for `ctx`, or say why not.
    ///
    /// Admission and charge are one step on purpose. As two -- ask, then take -- they are two
    /// values that must agree, and ring threads allocate concurrently, so between the two
    /// another context can take the room this one was told it had.
    pub fn try_charge(
        self: &Arc<Self>,
        ctx: CtxId,
        what: &'static str,
        size: u64,
    ) -> Result<Charge, Refused> {
        let mut ledger = self.ledger.lock().expect("the budget ledger");
        let live: u64 = ledger.values().map(PerCtx::bytes).sum();
        if let Some(cap) = self.cap
            && live.saturating_add(size) > cap
        {
            return Err(Refused { wanted: size, live, cap });
        }
        *ledger.entry(ctx).or_default().live.entry((what, size)).or_insert(0) += 1;
        Ok(Charge { budget: Arc::clone(self), ctx, what, size })
    }

    /// Total live bytes across every context.
    pub fn live(&self) -> u64 {
        self.ledger.lock().expect("the budget ledger").values().map(PerCtx::bytes).sum()
    }

    /// What one context holds.
    pub fn live_for(&self, ctx: CtxId) -> u64 {
        self.ledger.lock().expect("the budget ledger").get(&ctx).map_or(0, PerCtx::bytes)
    }

    /// Drop a context's slot, because the context is gone.
    ///
    /// A guest reuses context ids, so a slot left behind bills a dead context's bytes to whoever
    /// takes its number next. Anything still live here is a [`Charge`] this renderer failed to
    /// drop -- our leak, not the guest's, since a guest cannot hold one open. It is logged and
    /// not asserted: the process is being torn down around it, and taking the VM with us to
    /// report a leak is worse than reporting it.
    pub fn retire(&self, ctx: CtxId) {
        let Some(per) = self.ledger.lock().expect("the budget ledger").remove(&ctx) else {
            return;
        };
        let residual = per.bytes();
        if residual != 0 {
            eprintln!(
                "[virglrs] ctx {}: destroyed holding {} still charged -- a charge this \
                 renderer did not drop, not the guest's doing",
                ctx.get(),
                mib(residual)
            );
            for (what, size, n) in per.worst() {
                eprintln!("[virglrs]   {n} x {} {what}", mib(size));
            }
        }
    }

    /// Say what everything holds, biggest first. What makes a leak name itself.
    pub fn report(&self, reason: &str) {
        let ledger = self.ledger.lock().expect("the budget ledger");
        let live: u64 = ledger.values().map(PerCtx::bytes).sum();
        let cap = self.cap.map_or(String::from("no cap"), mib);
        eprintln!("[virglrs] {reason}: {} live of {cap}", mib(live));
        for (ctx, per) in ledger.iter() {
            eprintln!("[virglrs]   ctx {}: {}", ctx.get(), mib(per.bytes()));
            for (what, size, n) in per.worst() {
                eprintln!("[virglrs]     {n} x {} {what}", mib(size));
            }
        }
    }
}

/// Bytes as a human reads them. A budget line is read while something is on fire.
fn mib(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes < 1024 * 1024 {
        format!("{bytes} B")
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(n: u32) -> CtxId {
        CtxId::new(n).expect("not zero")
    }

    /// A charge is credited by being dropped, and by nothing else. That is the whole design: the
    /// record that owns the allocation owns the charge, so every path that retires the record --
    /// including ones not written yet -- credits the ledger without knowing it exists.
    #[test]
    fn a_charge_is_credited_by_going_out_of_scope() {
        let budget = Budget::with_cap(None, false);
        assert_eq!(budget.live(), 0);

        let a = budget.try_charge(ctx(1), "device memory", 4096).expect("no cap");
        let b = budget.try_charge(ctx(1), "device memory", 4096).expect("no cap");
        assert_eq!(budget.live(), 8192, "two of the same size are two, not one");

        drop(a);
        assert_eq!(budget.live(), 4096);
        drop(b);
        assert_eq!(budget.live(), 0, "and the bucket is gone, not left at zero");
    }

    /// The cap is on the process, not on a context: the host OS kills the worker for the total.
    /// So one context's allocations have to count against another's headroom.
    #[test]
    fn the_cap_is_what_every_context_holds_together() {
        let budget = Budget::with_cap(Some(1000), false);
        let held = budget.try_charge(ctx(1), "device memory", 800).expect("fits");

        let refused = budget.try_charge(ctx(2), "device memory", 300).expect_err("does not fit");
        assert_eq!(refused, Refused { wanted: 300, live: 800, cap: 1000 });

        // Bound, not dropped on the spot: a charge that goes out of scope at the end of the
        // statement that made it is credited before the next line reads the ledger.
        let _exact = budget
            .try_charge(ctx(2), "device memory", 200)
            .expect("exactly the room left is room enough");

        drop(held);
        let _reused = budget
            .try_charge(ctx(1), "device memory", 800)
            .expect("freeing reopens the headroom it was using");
        assert_eq!(budget.live(), 1000, "and the ledger is at the cap, not over it");
    }

    /// A guest reuses context ids. A slot left behind after a destroy bills a dead context's
    /// bytes to whoever takes its number next -- which is exactly the shape the venus corpus has,
    /// running three generations of ctx 8.
    #[test]
    fn retiring_a_context_takes_its_bytes_with_it() {
        let budget = Budget::with_cap(Some(1000), false);
        let charge = budget.try_charge(ctx(8), "device memory", 600).expect("fits");

        budget.retire(ctx(8));
        assert_eq!(budget.live_for(ctx(8)), 0, "the slot is gone with the context");
        let _next = budget
            .try_charge(ctx(8), "device memory", 900)
            .expect("the next generation of ctx 8 gets the room, not the debt");

        // The leaked charge still credits when it goes, against a slot that no longer exists.
        // Crediting must not resurrect it -- a slot conjured here would outlive every context
        // and never be retired by anything.
        drop(charge);
        assert_eq!(budget.live_for(ctx(8)), 900, "the survivor's own charge is untouched");
    }

    /// Accounting is always on and enforcement is not. Without a cap nothing is ever refused --
    /// which is what a bare virglrenderer and every replay run as, and why the corpora do not
    /// move when this lands.
    #[test]
    fn without_a_cap_nothing_is_refused_and_everything_is_counted() {
        let budget = Budget::with_cap(None, false);
        let _huge = budget.try_charge(ctx(1), "device memory", u64::MAX / 2).expect("no cap");
        let _more = budget.try_charge(ctx(1), "IOSurface", u64::MAX / 2).expect("still no cap");
        assert_eq!(budget.live(), u64::MAX - 1, "counted, both of them");
        assert!(budget.kills_context(), "and a refusal would still stop the context");
    }
}
