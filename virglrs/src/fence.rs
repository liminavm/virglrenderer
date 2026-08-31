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
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};

use crate::abi::Callbacks;
use crate::ids::{CtxId, FenceId, RingIdx};

/// The VMM's cookie and callback table, as handed to `virgl_renderer_init`.
///
/// The pointers are the VMM's, opaque to us, and are only ever handed back to it. They cross to
/// the retirement thread, which is why this exists rather than passing the raw pointers around.
struct Sink {
    cookie: *mut c_void,
    write_fence: Option<extern "C" fn(*mut c_void, u32)>,
    write_context_fence: Option<extern "C" fn(*mut c_void, u32, u32, u64)>,
}

// SAFETY: `cookie` is an opaque token the VMM gave us and never dereferenced here -- it is only
// passed back through the callbacks. The function pointers are `extern "C" fn`, which are `Send`
// on their own. The VMM's contract for ASYNC_FENCE_CB is that these are callable from a renderer
// thread; that is the whole point of the flag.
unsafe impl Send for Sink {}

enum Job {
    /// A context fence: retires through `write_context_fence` with its ring and id.
    Context(CtxId, RingIdx, FenceId),
    /// A legacy global fence: retires through `write_fence` with the client's own id.
    Global(u32),
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
    pub fn start(cookie: *mut c_void, cb: &Callbacks) -> Retirement {
        let sink = Sink {
            cookie,
            write_fence: cb.write_fence,
            write_context_fence: cb.write_context_fence,
        };
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

    pub fn retire_global(&self, client_fence_id: u32) {
        self.push(Job::Global(client_fence_id));
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

fn run(sink: Sink, q: Arc<(Mutex<Queue>, Condvar)>) {
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
            Job::Context(ctx, ring, fence) => {
                if let Some(f) = sink.write_context_fence {
                    f(sink.cookie, ctx.0, ring.0, fence.0);
                }
            }
            Job::Global(id) => {
                if let Some(f) = sink.write_fence {
                    f(sink.cookie, id);
                }
            }
            Job::Stop => {
                m.lock().expect("the fence queue lock is never held across a panic").stopped = true;
                return;
            }
        }
    }
}
