// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The ring loop: one thread per ring, draining what the guest wrote.
//!
//! A ring exists so the guest can hand work over without a VM exit. It writes commands into shared
//! memory and advances a tail; a host thread notices and runs them. Nothing here talks to the VMM,
//! which is the whole point -- the doorbell is a fallback for a ring that has gone to sleep.
//!
//! # Why this module owns almost nothing
//!
//! The C runs `vn_dispatch_command` on the ring thread against the context every other thread is
//! also using, with no lock over the object table or the driver. That is where the data races this
//! rewrite exists to remove actually live, and it cannot be ported.
//!
//! So the thread here does not dispatch. It owns exactly what it can own alone -- the ring's
//! memory, its position in the buffer, its reply slot -- copies each batch out of guest memory, and
//! hands the bytes to a [`Dispatch`] seam that takes whatever lock the renderer needs. The seam is
//! a trait for two reasons: this module never learns what a context is, and the loop is testable
//! without a Vulkan driver anywhere near it.
//!
//! # The rule the whole design rests on
//!
//! **The ring thread never blocks on a lock.** [`Dispatch::try_dispatch`] may answer
//! [`Verdict::Busy`], which is not an error -- it is one more turn of the backoff ladder the loop
//! was going to take anyway.
//!
//! That is what makes stopping safe. `vkDestroyRingMESA` runs *inside* a dispatch, holding the very
//! lock the thread would want, and then has to join this thread. If the thread could block on that
//! lock the two would deadlock. Because it cannot, a stopping thread is always somewhere it can
//! reach its own stop check -- spinning, sleeping, copying, or parked on its own condvar -- and the
//! join completes. Nothing in this loop may acquire a lock by waiting for it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::ids::RingId;

use super::proto::types::VkRingStatusFlagBitsMESA;
use super::ring::{ReplyStream, Ring};

/// The IDLE bit, as the guest reads it: "this ring is asleep, ring the doorbell".
const STATUS_IDLE: u32 = VkRingStatusFlagBitsMESA::VK_RING_STATUS_IDLE_BIT_MESA.0 as u32;
/// The FATAL bit: "this ring is dead, stop feeding it".
const STATUS_FATAL: u32 = VkRingStatusFlagBitsMESA::VK_RING_STATUS_FATAL_BIT_MESA.0 as u32;

/// How a batch handed to the seam turned out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Dispatched. The bytes are consumed and the position may advance.
    Ran,
    /// The lock was busy and nothing happened. The batch is untouched and must be offered again --
    /// this is a normal outcome, not a failure.
    Busy,
    /// The context is poisoned. The ring is finished.
    Poisoned,
}

/// Running one batch, wherever the state to run it against actually lives.
///
/// Implemented once for real by the renderer, and by a recorder in tests. `Sync` because every
/// ring thread of a context shares one of these.
pub trait Dispatch: Send + Sync {
    /// Try to run `buf` as commands that arrived on `ring`, answering into `reply`.
    ///
    /// Must never block. Whatever lock this needs is taken with a try, and a failure to get it is
    /// [`Verdict::Busy`] -- see the module docs for why that is load-bearing rather than lazy.
    fn try_dispatch(&self, ring: RingId, reply: &mut Option<ReplyStream>, buf: &[u8]) -> Verdict;
}

/// Where a sleeping ring waits, and how it is woken.
///
/// A leaf lock: nothing else is ever acquired while this is held, which is what keeps it out of
/// any cycle. `notified` is inside the mutex rather than beside it because a condvar needs the
/// predicate and the wait to be atomic -- a flag checked outside would let a notify land in the
/// gap between the check and the wait, and the ring would sleep through its own doorbell.
#[derive(Default)]
struct Park {
    notified: Mutex<bool>,
    wake: Condvar,
}

/// A running ring, from the outside.
///
/// The body is gone -- the thread owns it -- and comes back from [`RingThread::stop`]. That is the
/// state machine made structural: while a ring is running there is no `Ring` for anyone else to
/// reach, so nothing can read a position or a reply slot the thread is busy changing.
pub struct RingThread {
    id: RingId,
    park: Arc<Park>,
    started: Arc<AtomicBool>,
    thread: Option<JoinHandle<Ring>>,
}

impl RingThread {
    /// Which ring this is.
    pub fn id(&self) -> RingId {
        self.id
    }

    /// The guest rang the doorbell: wake the ring if it was asleep.
    ///
    /// Setting the flag under the mutex is what makes this reliable. A parked thread holds the
    /// mutex across its predicate check and its wait, so this either lands before the check (the
    /// thread sees it and never sleeps) or after the wait has begun (the signal reaches it).
    pub fn notify(&self) {
        let mut notified = self.park.notified.lock().expect("the park lock is never poisoned");
        *notified = true;
        self.park.wake.notify_one();
    }

    /// Stop the ring and take its body back.
    ///
    /// Safe to call while holding whatever lock [`Dispatch`] wants, which is the only reason a
    /// destroy handler can call it at all. See the module docs.
    pub fn stop(mut self) -> Ring {
        // Under the park mutex, for the same reason `notify` is: a thread about to sleep must not
        // miss this between checking `started` and waiting.
        {
            let _held = self.park.notified.lock().expect("the park lock is never poisoned");
            self.started.store(false, Ordering::Release);
            self.park.wake.notify_one();
        }
        self.thread
            .take()
            .expect("a RingThread holds its handle until stop takes it, and stop consumes self")
            .join()
            .expect("the ring loop does not panic, and this crate aborts if anything does")
    }
}

/// The backoff an idle ring walks, in the shape the C settled on.
///
/// Sixteen cheap yields, then short sleeps ramping to 40us and holding there, then a deep 640us.
/// The plateau is the point: `iter` resets to zero on every batch, so a ring being actively fed
/// never leaves it and keeps ~40us pickup latency, while a genuinely quiet one crosses it once and
/// then sleeps cheaply until the idle timeout parks it for good.
///
/// The C makes the plateau's depth adaptive from a profile of the ring's cadence. That is a
/// measured optimisation on top of this shape, and it is not ported here: a fixed plateau is the
/// behaviour to beat before there is anything to measure against.
fn relax(iter: &mut u32) {
    const SPINS: u32 = 16;
    const RAMP: [u64; 3] = [10, 20, 30];
    const WARM_RUNGS: u32 = 16;
    const DEEP_US: u64 = 640;

    let i = *iter;
    *iter = i.saturating_add(1);
    if i < SPINS {
        std::thread::yield_now();
        return;
    }
    let rung = i - SPINS;
    let us = match RAMP.get(rung as usize) {
        Some(&us) => us,
        None if rung < RAMP.len() as u32 + WARM_RUNGS => 40,
        None => DEEP_US,
    };
    std::thread::sleep(Duration::from_micros(us));
}

/// Start a thread draining `ring`.
///
/// `fatal` is the context's poison, shared rather than copied: a ring whose stream goes bad takes
/// the whole context down with it, which is what the C's `vkr_context_on_ring_fatal` does. The
/// thread can set it without waiting for anything, which it must be able to do -- it has nowhere
/// to report to and no lock it is allowed to block on.
pub fn spawn(
    id: RingId,
    ring: Ring,
    dispatch: Arc<dyn Dispatch>,
    fatal: Arc<AtomicBool>,
) -> RingThread {
    let park = Arc::new(Park::default());
    let started = Arc::new(AtomicBool::new(true));

    let thread = {
        let park = Arc::clone(&park);
        let started = Arc::clone(&started);
        std::thread::Builder::new()
            .name(format!("virglrs-ring-{}", id.0))
            .spawn(move || run(id, ring, &park, &started, dispatch.as_ref(), &fatal))
            .expect("the host can start a ring thread")
    };

    RingThread { id, park, started, thread: Some(thread) }
}

/// The loop itself. Returns the ring body so a stop can hand it back.
fn run(
    id: RingId,
    mut ring: Ring,
    park: &Park,
    started: &AtomicBool,
    dispatch: &dyn Dispatch,
    fatal: &AtomicBool,
) -> Ring {
    // Where we have read up to. Free-running, like the guest's tail: both wrap at 32 bits and only
    // their difference means anything, which is why every comparison below is a wrapping one.
    let mut cur: u32 = 0;
    // Bytes copied out but not yet dispatched. Held across iterations so a `Busy` verdict costs a
    // retry and not a re-read -- and, more importantly, so the position never advances past work
    // that has not run.
    let mut pending: Vec<u8> = Vec::new();
    let mut last_work = Instant::now();
    let mut iter = 0u32;

    while started.load(Ordering::Acquire) {
        if pending.is_empty() && last_work.elapsed() >= ring.idle_timeout {
            if park_if_quiet(&ring, park, started, &mut cur) {
                break;
            }
            last_work = Instant::now();
            iter = 0;
        }

        if pending.is_empty() {
            let available = ring.tail().wrapping_sub(cur);
            if available > 0 && !ring.read_batch(cur, available, &mut pending) {
                // The guest says it wrote more than its own ring holds, so it has already run over
                // whatever was in there. Nothing recovers from that: the ring dies and takes the
                // context with it, which is what the C does too.
                eprintln!(
                    "[virglrs] {id}: guest wrote {available} bytes into a {}-byte ring",
                    ring.layout.buffer.size()
                );
                die(&ring, fatal);
                break;
            }
        }

        if pending.is_empty() {
            relax(&mut iter);
            continue;
        }

        // The reply slot is lent to the batch, exactly as a context lends its own. The thread
        // owns the body, so this is the one place that slot can be reached at all.
        match dispatch.try_dispatch(id, &mut ring.reply, &pending) {
            Verdict::Ran => {
                // Only now does the position move. Until the bytes have actually run, the ring
                // still says they are unread -- which is what a `Busy` answer depends on.
                cur = cur.wrapping_add(pending.len() as u32);
                ring.set_head(cur);
                pending.clear();
                last_work = Instant::now();
                iter = 0;
            }
            Verdict::Busy => relax(&mut iter),
            Verdict::Poisoned => {
                ring.set_status_bits(STATUS_FATAL);
                break;
            }
        }
    }

    ring
}

/// Tell the guest the ring is dead, and poison the context with it.
fn die(ring: &Ring, fatal: &AtomicBool) {
    ring.set_status_bits(STATUS_FATAL);
    fatal.store(true, Ordering::Release);
}

/// Announce the ring is going to sleep and, if nothing arrived while we said so, sleep.
///
/// Returns whether the ring was stopped while parked.
///
/// The order here is the protocol, not a style choice. The host sets IDLE and *then* loads the
/// tail; the guest stores the tail and *then* loads the status. Both loads are sequentially
/// consistent, so at least one side sees the other's store -- either the host notices the work and
/// does not sleep, or the guest notices the IDLE bit and rings the doorbell. Weaken either load to
/// acquire and both can miss, leaving a ring asleep on work that is already there with nobody left
/// to wake it. The C's `vkr_ring_load_tail_seqcst` carries the same reasoning, and records that the
/// 2ms poll it replaced existed only to survive this race.
fn park_if_quiet(ring: &Ring, park: &Park, started: &AtomicBool, cur: &mut u32) -> bool {
    let mut notified = park.notified.lock().expect("the park lock is never poisoned");
    *notified = false;
    ring.set_status_bits(STATUS_IDLE);

    if *cur == ring.tail_seqcst() {
        while started.load(Ordering::Acquire) && !*notified {
            notified = park.wake.wait(notified).expect("the park lock is never poisoned");
        }
    }

    ring.unset_status_bits(STATUS_IDLE);
    !started.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest_mem::GuestMap;
    use crate::ids::ResourceHandle;
    use crate::venus::proto::types::VkRingCreateInfoMESA;
    use crate::venus::ring::ShmResources;

    const RES: ResourceHandle = ResourceHandle(449);
    const BUF_AT: usize = 0xc0;
    const BUF_SIZE: usize = 0x20000;

    struct OneShm(Arc<GuestMap>);
    impl ShmResources for OneShm {
        fn shm(&self, handle: ResourceHandle) -> Option<Arc<GuestMap>> {
            (handle == RES).then(|| Arc::clone(&self.0))
        }
    }

    /// A ring in freshly minted memory, with the idle timeout the caller wants to exercise.
    fn ring(idle: Duration) -> (Arc<GuestMap>, Ring) {
        let (fd, map) =
            crate::guest_mem::anonymous_shm(0x24000, "virglrs-ringthread").expect("shm");
        drop(fd);
        let map = Arc::new(map);
        let info = VkRingCreateInfoMESA {
            resourceId: RES.0,
            offset: 0,
            size: 0x200c4,
            idleTimeout: idle.as_nanos() as u64,
            headOffset: 0,
            tailOffset: 4,
            statusOffset: 8,
            bufferOffset: BUF_AT,
            bufferSize: BUF_SIZE,
            extraOffset: 0x200c0,
            extraSize: 4,
            ..Default::default()
        };
        let table = OneShm(Arc::clone(&map));
        let r = Ring::create(&table, &info).expect("a layout we accept");
        (map, r)
    }

    fn head(map: &GuestMap) -> u32 {
        map.load_u32(0).expect("head")
    }
    fn status(map: &GuestMap) -> u32 {
        map.load_u32(8).expect("status")
    }
    /// Publish `bytes` at free-running position `at` and advance the tail, as a guest would.
    fn guest_writes(map: &GuestMap, at: u32, bytes: &[u8]) {
        let offset = (at as usize) & (BUF_SIZE - 1);
        let to_end = BUF_SIZE - offset;
        if bytes.len() <= to_end {
            assert!(map.copy_in(BUF_AT + offset, bytes));
        } else {
            assert!(map.copy_in(BUF_AT + offset, &bytes[..to_end]));
            assert!(map.copy_in(BUF_AT, &bytes[to_end..]));
        }
        assert!(map.store_u32(4, at.wrapping_add(bytes.len() as u32)));
    }

    /// Wait for something the ring thread must do, and fail rather than hang if it does not.
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

    /// What the seam should answer, and what it saw.
    #[derive(Default)]
    struct Recorder {
        batches: Mutex<Vec<Vec<u8>>>,
        /// How many times to answer `Busy` before running anything.
        busy_first: Mutex<u32>,
        poison: bool,
    }

    impl Dispatch for Recorder {
        fn try_dispatch(&self, _: RingId, _: &mut Option<ReplyStream>, buf: &[u8]) -> Verdict {
            if self.poison {
                return Verdict::Poisoned;
            }
            {
                let mut left = self.busy_first.lock().unwrap();
                if *left > 0 {
                    *left -= 1;
                    return Verdict::Busy;
                }
            }
            self.batches.lock().unwrap().push(buf.to_vec());
            Verdict::Ran
        }
    }

    fn spawn_with(rec: Arc<Recorder>, r: Ring) -> (RingThread, Arc<AtomicBool>) {
        let fatal = Arc::new(AtomicBool::new(false));
        let t = spawn(RingId(7), r, rec, Arc::clone(&fatal));
        (t, fatal)
    }

    /// The loop's whole job: what the guest put in the buffer reaches the seam, and the head moves
    /// only once it has.
    #[test]
    fn a_ring_thread_runs_what_the_guest_wrote() {
        let (map, r) = ring(Duration::from_secs(3600));
        let rec = Arc::new(Recorder::default());
        let (t, fatal) = spawn_with(Arc::clone(&rec), r);

        guest_writes(&map, 0, b"hello ring");
        until("the batch to be dispatched", || !rec.batches.lock().unwrap().is_empty());

        assert_eq!(rec.batches.lock().unwrap().as_slice(), &[b"hello ring".to_vec()]);
        until("the head to catch up", || head(&map) == 10);
        assert!(!fatal.load(Ordering::Acquire), "nothing went wrong");
        t.stop();
    }

    /// Stopping hands the body back, which is what lets a destroyed ring release its share of the
    /// resource and a replayed one go back to being idle.
    #[test]
    fn a_stopped_ring_gives_its_body_back() {
        let (map, r) = ring(Duration::from_secs(3600));
        let want = r.layout;
        let (t, _) = spawn_with(Arc::new(Recorder::default()), r);
        guest_writes(&map, 0, b"x");
        let back = t.stop();
        assert_eq!(back.layout, want, "the same ring came back");
    }

    /// A parked ring is woken by the doorbell. Without this a quiet ring never runs again.
    #[test]
    fn a_parked_ring_wakes_on_notify() {
        // Park almost immediately, so the test does not depend on how fast the machine is.
        let (map, r) = ring(Duration::from_millis(1));
        let rec = Arc::new(Recorder::default());
        let (t, _) = spawn_with(Arc::clone(&rec), r);

        until("the ring to announce it is idle", || status(&map) & STATUS_IDLE != 0);

        // Write without letting the loop see it on its own: it is parked on the condvar, so only
        // the doorbell can start it again.
        guest_writes(&map, 0, b"wake up");
        t.notify();

        until("the woken ring to dispatch", || !rec.batches.lock().unwrap().is_empty());
        assert_eq!(rec.batches.lock().unwrap().as_slice(), &[b"wake up".to_vec()]);
        t.stop();
    }

    /// The deadlock witness. A ring asleep on its condvar must still be stoppable -- and stop is
    /// called by a destroy handler that is itself holding the lock the seam wants, so a thread
    /// that could block on that lock would never join.
    #[test]
    fn a_parked_ring_can_still_be_stopped() {
        let (map, r) = ring(Duration::from_millis(1));
        let (t, _) = spawn_with(Arc::new(Recorder::default()), r);
        until("the ring to park", || status(&map) & STATUS_IDLE != 0);
        t.stop();
        assert_eq!(status(&map) & STATUS_IDLE, 0, "a stopped ring is not left claiming to be idle");
    }

    /// A busy lock loses nothing. The batch is offered again, exactly once, and the head does not
    /// move until it has actually run -- the property that lets the thread refuse to block.
    #[test]
    fn a_busy_dispatch_offers_the_same_batch_again() {
        let (map, r) = ring(Duration::from_secs(3600));
        let rec = Arc::new(Recorder { busy_first: Mutex::new(3), ..Default::default() });
        let (t, _) = spawn_with(Arc::clone(&rec), r);

        guest_writes(&map, 0, b"retry me");
        until("the batch to get through", || !rec.batches.lock().unwrap().is_empty());

        assert_eq!(
            rec.batches.lock().unwrap().as_slice(),
            &[b"retry me".to_vec()],
            "delivered once despite three refusals"
        );
        until("the head to catch up", || head(&map) == 8);
        t.stop();
    }

    /// A guest claiming to have written more than the ring holds has overrun its own buffer.
    /// There is nothing left to read, so the ring dies and takes the context with it.
    #[test]
    fn a_batch_larger_than_the_ring_poisons_the_context() {
        let (map, r) = ring(Duration::from_secs(3600));
        let rec = Arc::new(Recorder::default());
        let (t, fatal) = spawn_with(Arc::clone(&rec), r);

        // One byte more than the buffer can hold.
        assert!(map.store_u32(4, BUF_SIZE as u32 + 1));

        until("the context to be poisoned", || fatal.load(Ordering::Acquire));
        until("the guest to be told", || status(&map) & STATUS_FATAL != 0);
        assert!(rec.batches.lock().unwrap().is_empty(), "nothing was dispatched from it");
        t.stop();
    }

    /// The buffer is a circle. A batch that starts near the end and runs off it must arrive whole
    /// and in order -- reading it as one flat range would deliver the tail of the ring followed by
    /// whatever came before the start.
    #[test]
    fn a_batch_that_wraps_the_end_of_the_buffer_arrives_in_order() {
        let (map, r) = ring(Duration::from_secs(3600));
        let rec = Arc::new(Recorder::default());
        let (t, _) = spawn_with(Arc::clone(&rec), r);

        // Walk the position to six bytes from the end the only way there is: by feeding the ring
        // that much and letting it consume it.
        let filler = vec![0xa5u8; BUF_SIZE - 6];
        guest_writes(&map, 0, &filler);
        until("the filler to be consumed", || head(&map) == (BUF_SIZE - 6) as u32);

        let start = (BUF_SIZE - 6) as u32;
        let payload: Vec<u8> = (0u8..16).collect();
        guest_writes(&map, start, &payload);

        until("the wrapped batch", || rec.batches.lock().unwrap().len() == 2);
        assert_eq!(
            rec.batches.lock().unwrap()[1],
            payload,
            "the ten bytes before the end, then the six after the start"
        );
        t.stop();
    }
}
