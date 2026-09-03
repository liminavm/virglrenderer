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

use crate::ids::{CtxId, RingId};

use super::proto::types::VkRingStatusFlagBitsMESA;
use super::ring::{ReplyStream, Ring, RingControl};

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
    /// The batch stopped at a `vkWaitVirtqueueSeqnoMESA`. The first `consumed` bytes ran; the
    /// wait command and everything after it have not, and must be offered again once the context
    /// has published `seqno`.
    ///
    /// The wait command is *not* consumed, so re-offering re-decodes it and its handler decides
    /// again. That is what keeps a reply-carrying wait honest: the generated wrapper encodes the
    /// answer on the pass that proceeds, which is after the wait was satisfied, exactly as a
    /// handler that had blocked would have. It costs the command being decoded twice.
    Wait { consumed: usize, seqno: u64 },
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

/// Everything a sleeping ring waits on, and how it is woken.
///
/// One mutex and one condvar serve all four reasons a ring sleeps -- the idle park, the doorbell,
/// the stop, and a `vkWaitVirtqueueSeqnoMESA` -- because they are one question asked three ways:
/// "has anything changed that I was waiting for". The C reaches the same shape from the same
/// pressure, serving idle, roundtrip and stop from a single cond with predicate re-checks.
///
/// A leaf lock: nothing else is ever acquired while this is held, which is what keeps it out of
/// any cycle, and what makes it safe for the *context* thread to take it -- which it does, to
/// publish a virtqueue seqno and to ask whether a ring is blocked on one.
///
/// The state is inside the mutex rather than beside it because a condvar needs the predicate and
/// the wait to be atomic -- a flag checked outside would let a wake land in the gap between the
/// check and the wait, and the ring would sleep through its own doorbell.
#[derive(Default)]
struct Park {
    state: Mutex<ParkState>,
    wake: Condvar,
}

/// What a parked ring is waiting for, and what it has been told.
#[derive(Default)]
struct ParkState {
    /// The doorbell rang. Cleared by whoever consumes it.
    notified: bool,
    /// The highest virtqueue seqno the *context* has published for this ring.
    ///
    /// State, not an edge: a `vkSubmitVirtqueueSeqnoMESA` that arrives before the ring ever waits
    /// must still satisfy that later wait, so this is stored and compared, never signalled and
    /// forgotten. It only ever rises.
    vq_seqno: u64,
    /// The virtqueue seqno this ring is asleep waiting for, if it is.
    ///
    /// Published so a `vkWaitRingSeqnoMESA` on the context's own stream can see that this ring
    /// cannot advance until *the asking thread* publishes that seqno -- which it cannot, because
    /// it is the thread doing the asking. Without it the two waits deadlock the whole device.
    blocked_on_vq: Option<u64>,
}

impl Park {
    /// The virtqueue seqno this ring is asleep on *and cannot reach*, if it is in that state.
    ///
    /// One answer under one lock rather than two questions asked separately. Asked apart, the
    /// ring can wake between them and the caller reads a stall that has already resolved -- and
    /// this caller's response to a stall is to poison a context, which is not a verdict to reach
    /// on two values that were never true at the same moment.
    fn stalled_on(&self) -> Option<u64> {
        let state = self.state.lock().expect("the park lock is never poisoned");
        state.blocked_on_vq.filter(|&want| state.vq_seqno < want)
    }
}

/// Wrap-aware "has `a` reached `b`", for the free-running byte positions in a ring.
///
/// The head and tail count commands forever and wrap at 32 bits, so only their *difference* means
/// anything: `a >= b` as plain integers calls a head that has just wrapped past a seqno that has
/// not "behind", and the wait it is deciding never ends. This is the C's `vkr_seqno_ge`.
pub fn seqno_ge(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) >= 0
}

/// A `vkWaitRingSeqnoMESA` in the one form it can safely be waited on: with nothing locked.
///
/// Assembled while the context is held and waited on after it is released, because the thread it
/// is waiting for needs that very lock to make the progress being waited for. Every field is an
/// `Arc` to something the ring thread also holds, so once this exists it needs the renderer for
/// nothing at all.
#[must_use = "a waiter that is never waited on is a submission that never finished"]
pub struct RingWaiter {
    ctx: CtxId,
    id: RingId,
    seqno: u32,
    control: Arc<RingControl>,
    park: Arc<Park>,
    wait_ring: Arc<WaitRing>,
    fatal: Arc<AtomicBool>,
}

/// How long a ring wait may block before the log names it, in ms.
///
/// `LIMINA_RING_WAIT_WARN_MS` lowers it, so the slow path -- otherwise at the mercy of host load
/// -- can be exercised on purpose. Inherited from the C along with the diagnostic itself.
fn wait_warn() -> Duration {
    let ms = std::env::var("LIMINA_RING_WAIT_WARN_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .unwrap_or(500);
    Duration::from_millis(ms)
}

impl RingWaiter {
    /// Sleep until the ring's head reaches the seqno. `false` means the context is poisoned.
    ///
    /// Two of the three ways out are refusals, and both are deadlocks caught rather than waited
    /// through. They are checked before the first sleep and again on every wake, because the
    /// state that makes them true can arrive either side of the sleep beginning.
    pub fn wait(self) -> bool {
        let warn = wait_warn();
        let mut logged = false;
        loop {
            if self.fatal.load(Ordering::Acquire) {
                return false;
            }
            if seqno_ge(self.control.head(), self.seqno) {
                return true;
            }

            // The ring is asleep on a virtqueue seqno nobody has published. The only command that
            // publishes one arrives on *this* stream, and this stream is here -- so the ring
            // cannot advance, this wait cannot end, and neither can be rescued by waiting longer.
            // Left to sleep, the two hold the virtio-gpu control queue between them, which is one
            // queue for the whole device: every other context's submissions, every scanout flush
            // and every fence would stop with them. The C's guard runs only in the ring thread's
            // idle branch and never sees this pair at all.
            if let Some(want) = self.park.stalled_on() {
                eprintln!(
                    "[virglrs] ctx {}: {} waits for ring seqno {} while {} sleeps for virtqueue                      seqno {}, which only this stream can publish -- neither can proceed",
                    self.ctx, self.id, self.seqno, self.id, want,
                );
                self.die();
                return false;
            }

            // The ring has consumed everything the guest wrote and is still short of the seqno
            // asked for, so no head this ring can reach will ever satisfy it: the guest asked to
            // be told about bytes it never sent. This is the C's guard, asked from the waiting
            // side rather than from the ring's idle branch -- the head advancing to meet the tail
            // is itself a wake, so the re-check that sees this always happens.
            let (head, tail) = (self.control.head(), self.control.tail());
            if head == tail && !seqno_ge(tail, self.seqno) {
                eprintln!(
                    "[virglrs] ctx {}: {} is drained at {head} and cannot reach ring seqno {}",
                    self.ctx, self.id, self.seqno,
                );
                self.die();
                return false;
            }

            if self.wait_ring.wait(warn) && !logged {
                // A timeout is the diagnostic firing, never a failure: the wait goes on. Latched
                // to one line, because this dispatch runs per exported frame sync fd -- hundreds
                // of times a second on a busy compositor -- and an unconditional log here is a
                // frame stutter.
                eprintln!(
                    "[virglrs] ctx {}: {} ring-seqno wait stuck >{}ms: want {} head {} tail {}                      status {:#x}",
                    self.ctx,
                    self.id,
                    warn.as_millis(),
                    self.seqno,
                    self.control.head(),
                    self.control.tail(),
                    self.control.status(),
                );
                logged = true;
            }
        }
    }

    /// Kill the ring and the context with it. The guest is told through the status word, because
    /// a wait that ends this way has no reply to carry the news.
    fn die(&self) {
        self.control.set_bits(STATUS_FATAL);
        self.fatal.store(true, Ordering::Release);
        self.wait_ring.changed();
    }
}

/// A running ring, from the outside.
///
/// The body is gone -- the thread owns it -- and comes back from [`RingThread::stop`]. That is the
/// state machine made structural: while a ring is running there is no `Ring` for anyone else to
/// reach, so nothing can read a position or a reply slot the thread is busy changing.
pub struct RingThread {
    id: RingId,
    park: Arc<Park>,
    /// The ring's control words. Held here because a running ring has no body anyone else can
    /// reach, and three of those words are still other threads' business -- see [`RingControl`].
    control: Arc<RingControl>,
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
        let mut state = self.park.state.lock().expect("the park lock is never poisoned");
        state.notified = true;
        self.park.wake.notify_one();
    }

    /// This ring's control words, for the callers that must reach them while it is running.
    pub fn control(&self) -> &Arc<RingControl> {
        &self.control
    }

    /// The context published a virtqueue seqno for this ring.
    ///
    /// Only ever raises it. A guest that submits seqnos out of order is describing a past it has
    /// already passed, and lowering the value would un-satisfy a wait this ring may already have
    /// been released from.
    pub fn submit_virtqueue_seqno(&self, seqno: u64) {
        let mut state = self.park.state.lock().expect("the park lock is never poisoned");
        state.vq_seqno = state.vq_seqno.max(seqno);
        self.park.wake.notify_one();
    }

    /// The virtqueue seqno this ring is asleep on, if it is asleep on one.
    ///
    /// The question a `vkWaitRingSeqnoMESA` asks before it sleeps: a ring blocked on a seqno the
    /// asking thread has not published is a ring that thread can never unblock, because
    /// publishing is the asking thread's own job and it is here instead.
    pub fn blocked_on(&self) -> Option<u64> {
        self.park.state.lock().expect("the park lock is never poisoned").blocked_on_vq
    }

    /// The highest virtqueue seqno published for this ring so far.
    pub fn virtqueue_seqno(&self) -> u64 {
        self.park.state.lock().expect("the park lock is never poisoned").vq_seqno
    }

    /// A wait on this ring's head, in a form that holds nothing of the renderer.
    pub fn waiter(
        &self,
        ctx: CtxId,
        seqno: u32,
        wait_ring: Arc<WaitRing>,
        fatal: Arc<AtomicBool>,
    ) -> RingWaiter {
        RingWaiter {
            ctx,
            id: self.id,
            seqno,
            control: Arc::clone(&self.control),
            park: Arc::clone(&self.park),
            wait_ring,
            fatal,
        }
    }

    /// Stop the ring and take its body back.
    ///
    /// Safe to call while holding whatever lock [`Dispatch`] wants, which is the only reason a
    /// destroy handler can call it at all. See the module docs.
    pub fn stop(mut self) -> Ring {
        // Under the park mutex, for the same reason `notify` is: a thread about to sleep must not
        // miss this between checking `started` and waiting.
        {
            let _held = self.park.state.lock().expect("the park lock is never poisoned");
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

impl Drop for RingThread {
    /// Tell the thread to stop, and do not wait for it.
    ///
    /// Never a join, because this can run *on the ring thread itself*: a dispatch upgrades its
    /// weak handle on the context, and if the owner dropped the context meanwhile, the last strong
    /// reference dies here, on the thread that would be joining itself. An orderly shutdown goes
    /// through [`RingThread::stop`], which joins on a thread that is provably not the caller;
    /// this is the backstop for every other path, and a detached ring loop terminates on its own
    /// -- it can no longer reach a context, so its next dispatch is `Verdict::Poisoned`.
    fn drop(&mut self) {
        let _held = self.park.state.lock().expect("the park lock is never poisoned");
        self.started.store(false, Ordering::Release);
        self.park.wake.notify_one();
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

/// Where a `vkWaitRingSeqnoMESA` sleeps, and what wakes it.
///
/// The waiter is on the context's own stream, so it is not this module's caller -- but every input
/// to its predicate is produced here, on a ring thread, which must never block on a context lock
/// to report one. So the wake lives in a leaf object both sides hold an `Arc` of, exactly as the
/// C's `ctx->wait_ring` does.
///
/// It carries no seqno and no head. Those are read from [`RingControl`] by whoever is waiting: the
/// head in guest memory is the one value, and a copy published beside it would be a second.
#[derive(Default)]
pub struct WaitRing {
    /// Held across the waiter's predicate check and its sleep, so a wake cannot land in between.
    changed: Mutex<()>,
    wake: Condvar,
}

impl WaitRing {
    /// Something a ring-seqno waiter's predicate depends on has changed: a head advanced, a ring
    /// went to sleep on a virtqueue seqno, or a ring died.
    ///
    /// Broadcast rather than signalled: one context may have several ring waits outstanding on
    /// different threads once the Rust API has more than one caller, and waking the wrong one
    /// costs a predicate re-check while waking none costs a hang.
    pub fn changed(&self) {
        let _held = self.changed.lock().expect("the wait-ring lock is never poisoned");
        self.wake.notify_all();
    }

    /// Sleep until [`Self::changed`] fires or `timeout` elapses. Returns whether it timed out.
    ///
    /// The caller re-checks its own predicate around this; nothing is decided here. A timeout is
    /// not a failure -- it is the diagnostic firing, and the wait goes on. (The C has to say this
    /// at length because its C11 shim maps `ETIMEDOUT` to `thrd_busy` rather than `thrd_timeout`,
    /// and testing the wrong one turned every slow wait into a poisoned context. `wait_timeout`
    /// has no such trap: the timeout is a distinct value, not an error.)
    #[must_use]
    pub fn wait(&self, timeout: Duration) -> bool {
        let held = self.changed.lock().expect("the wait-ring lock is never poisoned");
        let (_g, r) =
            self.wake.wait_timeout(held, timeout).expect("the wait-ring lock is never poisoned");
        r.timed_out()
    }
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
    wait_ring: Arc<WaitRing>,
) -> RingThread {
    // The seqno the ring was carrying while it was idle comes along. A
    // `vkSubmitVirtqueueSeqnoMESA` may legitimately arrive in the same batch that created the
    // ring, before there was a thread to tell -- and it is state, not an edge, so dropping it
    // here would strand the first wait rather than merely delay it.
    let park = Arc::new(Park {
        state: Mutex::new(ParkState { vq_seqno: ring.virtqueue_seqno, ..ParkState::default() }),
        wake: Condvar::new(),
    });
    let control = Arc::clone(&ring.control);
    let started = Arc::new(AtomicBool::new(true));

    let thread = {
        let park = Arc::clone(&park);
        let started = Arc::clone(&started);
        std::thread::Builder::new()
            .name(format!("virglrs-ring-{}", id.0))
            .spawn(move || run(id, ring, &park, &started, dispatch.as_ref(), &fatal, &wait_ring))
            .expect("the host can start a ring thread")
    };

    RingThread { id, park, control, started, thread: Some(thread) }
}

/// The loop itself. Returns the ring body so a stop can hand it back.
fn run(
    id: RingId,
    mut ring: Ring,
    park: &Park,
    started: &AtomicBool,
    dispatch: &dyn Dispatch,
    fatal: &AtomicBool,
    wait_ring: &WaitRing,
) -> Ring {
    // Where we have read up to. Free-running, like the guest's tail: both wrap at 32 bits and only
    // their difference means anything, which is why every comparison below is a wrapping one.
    //
    // From the ring rather than from zero: a ring restored from a snapshot resumes at the head the
    // guest was quiesced at. See `Ring::create`.
    let mut cur: u32 = ring.cur;
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
                die(&ring, fatal, wait_ring);
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
                // A `vkWaitRingSeqnoMESA` is waiting on exactly this number. It re-reads the head
                // itself; what it cannot do is know when to look.
                wait_ring.changed();
                pending.clear();
                last_work = Instant::now();
                iter = 0;
            }
            Verdict::Busy => relax(&mut iter),
            Verdict::Wait { consumed, seqno } => {
                // The prefix ran and is published before the sleep, so a `vkWaitRingSeqnoMESA`
                // on the context's stream sees exactly the work that actually completed -- and
                // so the wait command itself stays unconsumed, to be decoded again on the way
                // out. `pending` keeps it and everything after it.
                cur = cur.wrapping_add(consumed as u32);
                ring.set_head(cur);
                pending.drain(..consumed);
                wait_ring.changed();
                if wait_virtqueue_seqno(park, started, wait_ring, seqno) {
                    break;
                }
                last_work = Instant::now();
                iter = 0;
            }
            Verdict::Poisoned => {
                ring.set_status_bits(STATUS_FATAL);
                break;
            }
        }
    }

    // Whatever this ring was blocked on, it is not blocked on it now. Left set, it would tell a
    // later `vkWaitRingSeqnoMESA` that a thread which no longer exists is about to advance.
    park.state.lock().expect("the park lock is never poisoned").blocked_on_vq = None;
    wait_ring.changed();
    ring
}

/// Sleep until the context publishes `seqno` for this ring, or the ring is stopped.
///
/// Returns whether it was stopped. Nothing here can fail: an unreachable seqno is not this
/// thread's to diagnose, because from in here it is indistinguishable from one that has not
/// arrived yet. The thread that *can* tell the difference is a `vkWaitRingSeqnoMESA` on the
/// context's own stream, which is why `blocked_on_vq` is published before the sleep and the
/// waiter is woken -- see [`WaitRing`].
fn wait_virtqueue_seqno(
    park: &Park,
    started: &AtomicBool,
    wait_ring: &WaitRing,
    seqno: u64,
) -> bool {
    let mut state = park.state.lock().expect("the park lock is never poisoned");
    // Set before the wake below and before the first sleep, so a waiter that arrives at any point
    // from here on reads the true answer rather than a stale `None`.
    state.blocked_on_vq = Some(seqno);
    drop(state);
    wait_ring.changed();

    let mut state = park.state.lock().expect("the park lock is never poisoned");
    while started.load(Ordering::Acquire) && state.vq_seqno < seqno {
        state = park.wake.wait(state).expect("the park lock is never poisoned");
    }
    state.blocked_on_vq = None;
    drop(state);
    // The ring is moving again, and a waiter that refused to sleep on the strength of
    // `blocked_on_vq` must get the chance to re-check now that it is clear.
    wait_ring.changed();
    !started.load(Ordering::Acquire)
}

/// Tell the guest the ring is dead, and poison the context with it.
fn die(ring: &Ring, fatal: &AtomicBool, wait_ring: &WaitRing) {
    ring.set_status_bits(STATUS_FATAL);
    fatal.store(true, Ordering::Release);
    // A `vkWaitRingSeqnoMESA` on this ring is waiting for a head that will never move again. Its
    // predicate reads the poison, but only when something wakes it to look -- this is the C's
    // `vkr_context_on_ring_fatal` signalling `wait_ring`, and for the same reason.
    wait_ring.changed();
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
    let mut state = park.state.lock().expect("the park lock is never poisoned");
    state.notified = false;
    ring.set_status_bits(STATUS_IDLE);

    if *cur == ring.tail_seqcst() {
        while started.load(Ordering::Acquire) && !state.notified {
            state = park.wake.wait(state).expect("the park lock is never poisoned");
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

    const RES: ResourceHandle = ResourceHandle::new(449).unwrap();
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
            resourceId: RES.get(),
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
        let r = Ring::create(&table, &info, false).expect("a layout we accept");
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

    fn spawn_with(rec: Arc<Recorder>, r: Ring) -> (RingThread, Arc<AtomicBool>, Arc<WaitRing>) {
        let fatal = Arc::new(AtomicBool::new(false));
        let wait_ring = Arc::new(WaitRing::default());
        let t = spawn(RingId(7), r, rec, Arc::clone(&fatal), Arc::clone(&wait_ring));
        (t, fatal, wait_ring)
    }

    /// The loop's whole job: what the guest put in the buffer reaches the seam, and the head moves
    /// only once it has.
    #[test]
    fn a_ring_thread_runs_what_the_guest_wrote() {
        let (map, r) = ring(Duration::from_secs(3600));
        let rec = Arc::new(Recorder::default());
        let (t, fatal, _wr) = spawn_with(Arc::clone(&rec), r);

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
        let (t, _, _wr) = spawn_with(Arc::new(Recorder::default()), r);
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
        let (t, _, _wr) = spawn_with(Arc::clone(&rec), r);

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
        let (t, _, _wr) = spawn_with(Arc::new(Recorder::default()), r);
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
        let (t, _, _wr) = spawn_with(Arc::clone(&rec), r);

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
        let (t, fatal, _wr) = spawn_with(Arc::clone(&rec), r);

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
        let (t, _, _wr) = spawn_with(Arc::clone(&rec), r);

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
