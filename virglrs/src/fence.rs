// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Asynchronous fence retirement.
//!
//! `RENDER_SERVER | THREAD_SYNC | ASYNC_FENCE_CB` is a behavioural contract, not a hint: venus
//! fences must retire on a thread that is NOT the one submitting, or the guest deadlocks inside
//! `vkQueueWaitIdle` waiting for a fence whose retirement is queued behind its own call. So even
//! though a skeleton has no GPU work to wait on, it retires through the thread -- otherwise P2
//! would inherit a synchronous path that looks fine until a guest waits on it.
//!
//! Ordering is per (context, ring), which is the only order the guest can observe: it reads a
//! ring's seqno, so fences on one ring must retire in the order they were created. Across rings
//! there is no order to preserve and none is imposed.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use crate::ids::{ClientFenceId, CtxId, FenceId, RingIdx};

/// Where a retired fence goes.
///
/// Implemented by whoever drives the renderer -- the C shim wraps the VMM's callback table in one
/// of these, and a Rust caller writes its own. It is the renderer's only way to tell anyone that
/// work has completed.
///
/// **Called from the renderer's retirement thread**, never from the thread that submitted: that
/// is what `THREAD_SYNC | ASYNC_FENCE_CB` buys, and a guest waiting in `vkQueueWaitIdle`
/// deadlocks without it. Two things follow for an implementor. Blocking here stalls every later
/// fence, on every context, because one thread delivers them all. And a panic here aborts the
/// process, since the crate builds `panic = "abort"` -- so a sink handles its own errors rather
/// than unwrapping.
pub trait FenceSink: Send {
    /// A fence on one context's ring has retired. Delivered in creation order within a ring; no
    /// order is imposed across rings, because the guest can observe none.
    fn context_fence(&mut self, ctx: CtxId, ring: RingIdx, fence: FenceId);

    /// A fence on the legacy global path has retired.
    fn global_fence(&mut self, fence: ClientFenceId);
}

enum Job {
    /// A context fence: retires through `write_context_fence` with its ring and id.
    Context(CtxId, RingIdx, FenceId),
    /// A legacy global fence: retires through `global_fence` with the client's own id.
    Global(ClientFenceId),
    Stop,
}

struct Queue {
    jobs: VecDeque<Job>,
    stopped: bool,
}

pub struct Retirement {
    q: Arc<(Mutex<Queue>, Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Retirement {
    pub fn start(sink: Box<dyn FenceSink>) -> Retirement {
        let q =
            Arc::new((Mutex::new(Queue { jobs: VecDeque::new(), stopped: false }), Condvar::new()));
        let qt = Arc::clone(&q);
        let thread = std::thread::Builder::new()
            .name("virglrs-fence".into())
            .spawn(move || run(sink, qt))
            .expect("spawning the fence retirement thread");
        Retirement { q, thread: Some(thread) }
    }

    pub fn retire_context(&self, ctx: CtxId, ring: RingIdx, fence: FenceId) {
        self.push(Job::Context(ctx, ring, fence));
    }

    pub fn retire_global(&self, fence: ClientFenceId) {
        self.push(Job::Global(fence));
    }

    fn push(&self, job: Job) {
        let (m, cv) = &*self.q;
        let mut g = m.lock().expect("the fence queue lock is never held across a panic");
        if g.stopped {
            return;
        }
        g.jobs.push_back(job);
        cv.notify_one();
    }
}

impl Drop for Retirement {
    fn drop(&mut self) {
        self.push(Job::Stop);
        if let Some(t) = self.thread.take() {
            // A fence the VMM is still waiting on must not be dropped on the floor: the thread
            // drains what is queued before it stops, and cleanup waits for that to finish.
            let _ = t.join();
        }
    }
}

fn run(mut sink: Box<dyn FenceSink>, q: Arc<(Mutex<Queue>, Condvar)>) {
    let (m, cv) = &*q;
    loop {
        let job = {
            let mut g = m.lock().expect("the fence queue lock is never held across a panic");
            loop {
                match g.jobs.pop_front() {
                    Some(j) => break j,
                    None => {
                        g = cv.wait(g).expect("the fence queue lock is never held across a panic")
                    }
                }
            }
        };
        match job {
            Job::Context(ctx, ring, fence) => sink.context_fence(ctx, ring, fence),
            Job::Global(id) => sink.global_fence(id),
            Job::Stop => {
                m.lock().expect("the fence queue lock is never held across a panic").stopped = true;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{Sender, channel};

    /// A sink that reports everything it is handed, in the order it arrives.
    struct Recorder(Sender<(u32, u32, u64)>);

    impl FenceSink for Recorder {
        fn context_fence(&mut self, ctx: CtxId, ring: RingIdx, fence: FenceId) {
            let _ = self.0.send((ctx.get(), ring.0, fence.0));
        }

        fn global_fence(&mut self, fence: ClientFenceId) {
            let _ = self.0.send((0, 0, fence.0 as u64));
        }
    }

    /// Dropping the renderer must not drop fences the VMM is still waiting on.
    ///
    /// The queue is asynchronous, so at drop there is almost always work outstanding -- a guest
    /// blocked in `vkQueueWaitIdle` on one of those fences waits forever if it is discarded. This
    /// was a comment on `Drop` until the sink became a trait; a C function pointer could not be
    /// written in a test, so nothing checked it.
    ///
    /// Ordering rides along: a ring's fences are the one order a guest can observe, because it
    /// reads that ring's seqno.
    #[test]
    fn every_queued_fence_is_delivered_in_ring_order_before_the_queue_stops() {
        const N: u64 = 500;
        let (tx, rx) = channel();
        let r = Retirement::start(Box::new(Recorder(tx)));

        let ctx = CtxId::new(7).unwrap();
        for i in 1..=N {
            r.retire_context(ctx, RingIdx(1), FenceId(i));
        }
        r.retire_global(ClientFenceId(99));
        drop(r);

        // Drained without blocking, on purpose: a blocking read would wait for the sender to be
        // dropped and so would pass whether or not `drop` waited for anything. What is under test
        // is that delivery has ALREADY happened by the time `drop` returns.
        let got: Vec<_> = rx.try_iter().collect();
        assert_eq!(got.len() as u64, N + 1, "a fence was dropped on the floor");
        for (i, (c, ring, fence)) in got.iter().take(N as usize).enumerate() {
            assert_eq!((*c, *ring, *fence), (7, 1, i as u64 + 1), "a ring retired out of order");
        }
        assert_eq!(got[N as usize], (0, 0, 99));
    }
}
