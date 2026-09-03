// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The ring monitor: telling the guest this renderer is still scheduled.
//!
//! # What the bit means, and what happens without it
//!
//! A guest that creates a ring may chain a `VkRingMonitorInfoMESA` onto the create info, naming
//! the longest it will tolerate going without hearing from us. From then on the host must
//! periodically set `VK_RING_STATUS_ALIVE_BIT_MESA` in that ring's status word.
//!
//! It is not a liveness report about the ring. The guest's `vn_relax` clears the bit when it
//! starts waiting on anything and re-reads it a few seconds later; if it is still clear, mesa
//! calls `abort()` -- on the whole guest process, not the waiting thread. So the bit answers "is
//! the renderer scheduled at all", and a renderer that never stamps does not hang a guest, it
//! kills one. Any single wait longer than the tolerance is fatal, and a cold shader cache crosses
//! it routinely.
//!
//! # Why this has its own thread and its own lock
//!
//! Because the answer must stay true while the renderer is busy. A stamp that could be delayed by
//! a long dispatch would report the opposite of the fact it exists to report, so nothing here ever
//! touches the lock a dispatch holds: the registry below is a leaf, held only for the few
//! microseconds of a stamping pass.
//!
//! The C reaches its rings through the same mutex its dispatch uses, and pays for that with a
//! diagnostic that charges time lost blocked on the lock. That charge is not portable to a
//! renderer whose dispatch lock is the whole context; the registry is separate instead.
//!
//! # Oversampling
//!
//! Stamping at exactly the requested period is compliant and does not work. The period is an upper
//! bound, mesa asks for 3.0s and re-checks about 3.48s later, and one late wakeup aborts the
//! guest -- the C tree records 489ms of lateness on a near-idle host despite pinning the thread's
//! QoS class. Stamping at a third of the request widens the tolerated lateness to seconds for
//! every guest, stock mesa included, and costs a few extra atomic stores a second.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::ids::CtxId;

use super::proto::types::VkRingStatusFlagBitsMESA;
use super::ring::RingControl;

/// The ALIVE bit, as the guest reads it: "the renderer is still being scheduled".
const STATUS_ALIVE: u32 = VkRingStatusFlagBitsMESA::VK_RING_STATUS_ALIVE_BIT_MESA.0 as u32;

/// Above this, stamp at a third of what was asked. Below it, at exactly what was asked -- a guest
/// asking for a period this short has left no room to divide.
const OVERSAMPLE_ABOVE: Duration = Duration::from_millis(300);

/// What the thread and its owner share. A leaf lock and a condvar, and nothing else.
struct Shared {
    /// The rings to stamp, as `Weak` handles. Never keyed by ring id: a ring's entry is live
    /// exactly while its ring is, so liveness is the identity, and there is no id here to have
    /// gone stale against the one the context holds.
    rings: Mutex<Vec<Weak<RingControl>>>,
    /// The shortest period any monitored ring asked for, and the condvar the thread sleeps on.
    ///
    /// One value: the cadence the thread wakes at and the lateness it complains about are both
    /// derived from this at the point of use, so they cannot drift apart.
    period: Mutex<Duration>,
    wake: Condvar,
    running: AtomicBool,
}

impl Shared {
    /// How often to stamp, given what the guest asked to hear from us.
    fn cadence(period: Duration) -> Duration {
        if period > OVERSAMPLE_ABOVE { period / 3 } else { period }
    }
}

/// One context's monitor thread.
///
/// Started lazily, by the first ring that asks to be monitored, and kept running after the last
/// such ring dies -- an empty registry costs one wakeup per period and nothing else, and this
/// matches the C, which also never stops a monitor short of context teardown.
pub struct Monitor {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Monitor {
    /// Start monitoring, at the period a ring's create info asked for.
    ///
    /// `period_us` is the guest's number and is never zero -- a zero period is a guest error, and
    /// is refused by the handler before it gets here.
    pub fn start(ctx: CtxId, period_us: u32) -> Monitor {
        assert!(period_us > 0, "a zero reporting period is refused at the boundary, not here");
        let shared = Arc::new(Shared {
            rings: Mutex::new(Vec::new()),
            period: Mutex::new(Duration::from_micros(period_us as u64)),
            wake: Condvar::new(),
            running: AtomicBool::new(true),
        });
        let thread = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name(format!("virglrs-ringmon-{}", ctx))
                .spawn(move || run(ctx, &shared))
                .expect("the host can start a ring monitor thread")
        };
        Monitor { shared, thread: Some(thread) }
    }

    /// Stamp this ring from now on, and hear from us at least this often.
    ///
    /// The registry takes a `Weak`: the ring's own `Arc` is the only thing keeping its status word
    /// reachable, so destroying the ring un-registers it with no destroy path to remember.
    pub fn watch(&self, status: &Arc<RingControl>, period_us: u32) {
        self.shared
            .rings
            .lock()
            .expect("the monitor registry is a leaf lock")
            .push(Arc::downgrade(status));
        self.shorten(period_us);
    }

    /// How many rings are still registered. Tests only: liveness is the identity here, so this is
    /// the only way to ask whether an expired entry has actually been dropped.
    #[cfg(test)]
    pub fn watched(&self) -> usize {
        self.shared.rings.lock().expect("the monitor registry is a leaf lock").len()
    }

    /// Lower the period if this ring wants to hear from us more often than the current one.
    ///
    /// Only ever lower, like the C: the period is a promise to every monitored ring at once, and
    /// raising it would break the promise made to the ones that asked for less.
    pub fn shorten(&self, period_us: u32) {
        let want = Duration::from_micros(period_us as u64);
        let mut period = self.shared.period.lock().expect("the monitor period is a leaf lock");
        if want < *period {
            *period = want;
            // Without this the thread is asleep until the end of a wait it began under the old,
            // longer contract -- the first stamp under the new one would be late by construction.
            self.shared.wake.notify_one();
        }
    }
}

impl Drop for Monitor {
    /// Stop the thread and wait for it.
    ///
    /// A join, unlike the ring threads': this thread takes no lock any caller could be holding and
    /// never reaches a context, so there is no path on which a monitor joins itself.
    fn drop(&mut self) {
        {
            let _held = self.shared.period.lock().expect("the monitor period is a leaf lock");
            self.shared.running.store(false, Ordering::Release);
            self.shared.wake.notify_one();
        }
        if let Some(t) = self.thread.take() {
            t.join()
                .expect("the monitor loop does not panic, and this crate aborts if anything does");
        }
    }
}

/// The loop: stamp every live ring, drop the dead ones, sleep until the next cadence.
fn run(ctx: CtxId, shared: &Shared) {
    let mut last: Option<Instant> = None;
    while shared.running.load(Ordering::Acquire) {
        let period = *shared.period.lock().expect("the monitor period is a leaf lock");

        {
            let mut rings = shared.rings.lock().expect("the monitor registry is a leaf lock");
            // Measured here, after the wake *and* after the lock, so anything that delayed the
            // stamp is charged to it rather than to the sleep that preceded it.
            let now = Instant::now();
            if let Some(prev) = last
                && now.duration_since(prev) > period
            {
                eprintln!(
                    "[virglrs] ringmon {ctx}: ALIVE stamp late: {}ms since the last one, guest \
                     tolerates ~{}ms -- the guest's venus watchdog may abort the guest process",
                    now.duration_since(prev).as_millis(),
                    period.as_millis(),
                );
            }
            last = Some(now);
            // Dropping the expired entries here rather than at any ring's destroy is the whole
            // point of holding `Weak`s: there is no destroy path left that can forget to do it.
            rings.retain(|w| match w.upgrade() {
                Some(status) => {
                    status.set_bits(STATUS_ALIVE);
                    true
                }
                None => false,
            });
        }

        let mut period_guard = shared.period.lock().expect("the monitor period is a leaf lock");
        while shared.running.load(Ordering::Acquire) {
            let (g, timeout) = shared
                .wake
                .wait_timeout(period_guard, Shared::cadence(period))
                .expect("the monitor period is a leaf lock");
            period_guard = g;
            // A shortened period wakes us early on purpose: go stamp under the new contract
            // rather than finishing a sleep the old one paid for.
            if timeout.timed_out() || *period_guard < period {
                break;
            }
        }
    }

    // Torn down while the last stamp is already stale, which is exactly the case the in-loop
    // check can no longer report: a guest that aborted on an expired ALIVE bit is being cleaned
    // up right now, and a silent exit here is what makes that abort unattributable host-side.
    if let Some(prev) = last {
        let period = *shared.period.lock().expect("the monitor period is a leaf lock");
        let since = prev.elapsed();
        if since > period {
            eprintln!(
                "[virglrs] ringmon {ctx}: exiting with a stale ALIVE stamp: {}ms since the last \
                 one, guest tolerates ~{}ms -- if the guest aborted on an expired ring status, \
                 this was why",
                since.as_millis(),
                period.as_millis(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest_mem::{self, GuestMap};
    use crate::ids::ResourceHandle;
    use crate::venus::proto::types::VkRingCreateInfoMESA;
    use crate::venus::ring::{Ring, ShmResources};

    fn ctx() -> CtxId {
        CtxId::new(3).expect("3 is a context id")
    }
    const RES: ResourceHandle = ResourceHandle::new(449).unwrap();
    const AT: usize = 8;

    struct OneShm(Arc<GuestMap>);
    impl ShmResources for OneShm {
        fn shm(&self, handle: ResourceHandle) -> Option<Arc<GuestMap>> {
            (handle == RES).then(|| Arc::clone(&self.0))
        }
    }

    /// A real ring's control words, reached the way a ring hands them out. Built through
    /// `Ring::create` rather than assembled here, so the offsets under test are the ones a
    /// validated layout produces.
    fn word() -> (Arc<GuestMap>, Arc<RingControl>) {
        let (fd, map) = guest_mem::anonymous_shm(0x24000, "virglrs-ringmon").expect("shm");
        drop(fd);
        let map = Arc::new(map);
        let info = VkRingCreateInfoMESA {
            resourceId: RES.get(),
            offset: 0,
            size: 0x200c4,
            idleTimeout: 1_000_000,
            headOffset: 0,
            tailOffset: 4,
            statusOffset: AT,
            bufferOffset: 0xc0,
            bufferSize: 0x20000,
            extraOffset: 0x200c0,
            extraSize: 4,
            ..Default::default()
        };
        let ring = Ring::create(&OneShm(Arc::clone(&map)), &info, false).expect("a layout we take");
        (map, Arc::clone(&ring.control))
    }

    /// Wait for something the monitor thread must do, and fail rather than hang if it does not.
    fn until(what: &str, mut pred: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if pred() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("timed out waiting for {what}");
    }

    fn alive(map: &GuestMap) -> bool {
        map.load_u32(AT).expect("the word is inside the mapping") & STATUS_ALIVE != 0
    }

    /// The whole job. Without this stamp the guest's `vn_relax` aborts the guest *process* a few
    /// seconds into any wait, so a monitor that never lands a bit is worse than no monitor at
    /// all: the guest asked, and was told it would hear from us.
    #[test]
    fn a_watched_ring_gets_the_alive_bit() {
        let (map, status) = word();
        let m = Monitor::start(ctx(), 1_000);
        m.watch(&status, 1_000);
        until("the alive bit", || alive(&map));
    }

    /// A destroyed ring stops being stamped because its status word stops existing, not because
    /// anyone remembered to say so. That is the whole reason the registry holds `Weak`s: a purge
    /// at the destroy site is a purge some future destroy path forgets, and the stamp would then
    /// be writing into a mapping the ring no longer holds a share of.
    #[test]
    fn a_dropped_ring_leaves_the_registry_without_being_told() {
        let (_map, status) = word();
        let m = Monitor::start(ctx(), 1_000);
        m.watch(&status, 1_000);
        assert_eq!(m.watched(), 1, "registered");
        drop(status);
        until("the expired entry to be dropped", || m.watched() == 0);
    }

    /// A ring asking for a shorter period is a promise that starts now, not at the end of the
    /// sleep the thread began under the old one. Without the wake the first stamp under the new
    /// contract is late by construction -- by up to the whole of the old period.
    #[test]
    fn a_shorter_period_wakes_the_sleeping_thread() {
        let (map, status) = word();
        // Long enough that a thread which failed to wake would blow the deadline in `until`.
        let m = Monitor::start(ctx(), 30_000_000);
        m.watch(&status, 30_000_000);
        // Clear what the first pass stamped, so what is measured is a stamp after the change.
        until("the first stamp", || alive(&map));
        status.unset_bits(STATUS_ALIVE);

        m.shorten(1_000);
        until("a stamp under the new period", || alive(&map));
    }

    /// Dropping the monitor joins its thread. A detached one would go on stamping the status
    /// words of a context being torn down, and would keep the process alive past the last one.
    #[test]
    fn dropping_the_monitor_stops_the_thread() {
        let (map, status) = word();
        let m = Monitor::start(ctx(), 1_000);
        m.watch(&status, 1_000);
        until("the first stamp", || alive(&map));
        drop(m);

        status.unset_bits(STATUS_ALIVE);
        std::thread::sleep(Duration::from_millis(50));
        assert!(!alive(&map), "a joined monitor is not still stamping");
    }
}
