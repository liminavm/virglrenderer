// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The venus renderer root.
//!
//! The C runs vkr behind the render-server proxy (`server/render_state.c`), which is a lock and a
//! forward to `vkr_renderer_*` in the same process. None of that survives here: vkr is an object
//! the `Renderer` owns, reached through it, with no file-scope state and no proxy hop. The C's
//! `vkr_renderer.h` is the interface this mirrors -- that header, not the proxy, is the design.
//!
//! # Lock order
//!
//! Once rings run on their own threads, three locks exist and every path takes them in this order:
//!
//! 1. the resource table (`Renderer::resources`, a read-write lock),
//! 2. one context (`Vkr::contexts`, a mutex each -- so two contexts' rings never wait on each other),
//! 3. the unimplemented-command census (`Vkr::todo`).
//!
//! The park mutex inside a `RingThread` is a leaf: nothing is taken while it is held.
//!
//! A caller arriving through the ABI blocks at each step, because it must run the command it was
//! given. A ring thread never blocks on any of them -- it tries, and a miss is `Verdict::Busy`,
//! retried on the next turn of its loop. That asymmetry is what makes stopping a ring safe:
//! `vkDestroyRingMESA` runs inside a dispatch that already holds the context lock and then joins
//! the thread, so a thread that could block on that lock would deadlock with its own destroy.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock, Weak};

use crate::config::Config;
use crate::ids::{CtxId, RingId};

use super::context::{Context, Unimplemented};
use super::ring::{ReplyStream, Ring, ShmResources};
use super::ring_thread::{self, Dispatch, Verdict};
use crate::vulkan::Global;

/// Why a venus call could not be served. Both are the caller's mistake, not ours: a context that
/// was never created for venus, or one whose stream we already stopped trusting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    NoContext,
    Poisoned,
}

/// Everything venus owns. Present exactly when the renderer was initialized to serve venus, which
/// is what makes the capset honest: it is advertised because this exists, not because a flag was
/// passed.
pub struct Vkr {
    /// What the renderer was configured to be. The capset reports part of it straight back to
    /// the guest, which is why it is kept rather than consumed at startup.
    pub config: Config,
    /// One lock per context, not one over the table: a context is exactly the unit a ring thread
    /// needs exclusively, so two guests' rings dispatch at the same time. The `Arc` is what lets a
    /// ring thread hold a claim on its own context without holding the renderer.
    contexts: BTreeMap<CtxId, Arc<Mutex<Context>>>,
    /// The commands this build does not serve yet, counted across every context. Kept on the root
    /// because it answers a question about the build, not about a guest.
    pub todo: Arc<Mutex<Unimplemented>>,
    /// The entry points that exist before any instance does. One per renderer rather than one per
    /// context: they are the loader's, identical for every guest, and immutable once resolved.
    global: Arc<Global>,
    /// The renderer's resource table, shared rather than borrowed per call: a ring thread has to
    /// reach it on its own schedule, long after the call that created the ring returned.
    ///
    /// Held as the trait, not the table, so this module still never learns what a `Resource` is.
    resources: SharedResources,
}

/// The resource table as everything outside the renderer sees it: shared, read-mostly, and known
/// only by the one question venus asks of it.
pub type SharedResources = Arc<RwLock<dyn ShmResources + Send + Sync>>;

/// What a ring thread needs to run a batch, and nothing else.
///
/// It holds a *weak* claim on its context. A strong one would be a cycle -- context owns the
/// thread owns the dispatch owns the context -- and a cycle here means `context_destroy` frees
/// nothing and `Drop for Context`, which is the actual teardown, never runs. A failed upgrade is
/// a context that is gone, which is a ring that should stop, so it reports `Poisoned`.
struct RingDispatch {
    ctx: Weak<Mutex<Context>>,
    resources: SharedResources,
    todo: Arc<Mutex<Unimplemented>>,
    global: Arc<Global>,
}

impl Dispatch for RingDispatch {
    /// Take the three locks in the renderer's order, or give up and say so.
    ///
    /// Every one of them is a `try`. A ring thread that blocked on the context lock would deadlock
    /// against its own `vkDestroyRingMESA`, which runs inside a dispatch that holds it -- and one
    /// that blocked on the census would deadlock against the same batch, which holds that too.
    /// `Busy` costs a retry on the next turn of the loop and nothing else, because the loop keeps
    /// the batch and does not advance its position until the batch has actually run.
    fn try_dispatch(&self, ring: RingId, reply: &mut Option<ReplyStream>, buf: &[u8]) -> Verdict {
        let Some(ctx) = self.ctx.upgrade() else {
            return Verdict::Poisoned;
        };
        let Ok(resources) = self.resources.try_read() else {
            return Verdict::Busy;
        };
        let Ok(mut ctx) = ctx.try_lock() else {
            return Verdict::Busy;
        };
        let Ok(mut todo) = self.todo.try_lock() else {
            return Verdict::Busy;
        };
        if ctx.dispatch_ring(ring, reply, buf, &mut todo, &self.global, &*resources) {
            Verdict::Ran
        } else {
            Verdict::Poisoned
        }
    }
}

impl Vkr {
    pub fn new(config: Config, resources: SharedResources) -> Vkr {
        Vkr {
            config,
            contexts: BTreeMap::new(),
            todo: Arc::new(Mutex::new(Unimplemented::default())),
            global: Arc::new(crate::vulkan::global()),
            resources,
        }
    }

    pub fn context_create(&mut self, id: CtxId) {
        self.contexts.insert(id, Arc::new(Mutex::new(Context::new(id))));
    }

    /// Tear a context down. Every host handle it still holds dies with it -- a guest that leaks is
    /// not a guest that gets to keep host memory after it is gone.
    pub fn context_destroy(&mut self, id: CtxId) {
        // Stop the rings before letting go of the context, and do it in that order deliberately.
        // A ring thread upgrades a weak claim on this context for the length of one dispatch, so
        // a thread still running when the last strong reference goes could be holding the last
        // one itself -- and the teardown would run on the ring thread, joining itself. After the
        // joins there is no thread left to hold a claim, so the drop below runs here.
        //
        // Joining while holding the context lock is safe for the reason the whole design rests
        // on: a ring thread never blocks on that lock. One mid-dispatch already holds it and
        // finishes; one about to try gets `Busy`, backs off, and sees that it was stopped.
        if let Some(ctx) = self.contexts.get(&id) {
            ctx.lock().expect("a context lock is never poisoned").stop_rings();
        }
        // Dropping it is the teardown -- see `impl Drop for Context`. A context usually dies
        // mid-workload with the guest still holding everything it made, so this is the destroy
        // that actually runs, not a tidy-up after the guest's own.
        self.contexts.remove(&id);
    }

    /// Look at one context, for as long as the closure runs and no longer.
    ///
    /// Scoped rather than returning a `&Context`, because the reference is only sound while the
    /// lock is held and a signature that hands one out cannot say that.
    pub fn with_context<R>(&self, id: CtxId, f: impl FnOnce(&Context) -> R) -> Option<R> {
        let ctx = self.contexts.get(&id)?;
        Some(f(&ctx.lock().expect("a context lock is never poisoned")))
    }

    /// The same, for the paths that change a context rather than read one.
    pub fn with_context_mut<R>(&self, id: CtxId, f: impl FnOnce(&mut Context) -> R) -> Option<R> {
        let ctx = self.contexts.get(&id)?;
        Some(f(&mut ctx.lock().expect("a context lock is never poisoned")))
    }

    /// Start every ring a context has created but not yet begun reading.
    ///
    /// The one place a ring thread is spawned, for both of the C's paths: a live create, promoted
    /// once its own batch has finished, and a replayed one, promoted at `replay_end`. Doing it
    /// here rather than in the create handler is what keeps a handler free of the renderer's
    /// locks -- and it is why a ring cannot start reading in the middle of the batch that made it,
    /// which would let it race the rest of its own creation.
    fn promote(&mut self, id: CtxId) {
        let Some(arc) = self.contexts.get(&id) else {
            return;
        };
        let weak = Arc::downgrade(arc);
        let (resources, todo, global) =
            (Arc::clone(&self.resources), Arc::clone(&self.todo), Arc::clone(&self.global));
        let mut ctx = arc.lock().expect("a context lock is never poisoned");
        // The C's `if (!ctx->replaying) vkr_ring_start(ring)`, asked at the one place that starts
        // a ring. A journal is still being fed to these rings; they start at `replay_end`.
        if ctx.replaying() {
            return;
        }
        let fatal = ctx.fatal_flag();
        ctx.start_idle_rings(|ring_id, ring: Ring| {
            let dispatch = RingDispatch {
                ctx: Weak::clone(&weak),
                resources: Arc::clone(&resources),
                todo: Arc::clone(&todo),
                global: Arc::clone(&global),
            };
            ring_thread::spawn(ring_id, ring, Arc::new(dispatch), Arc::clone(&fatal))
        });
    }

    /// Run a submission on one context.
    ///
    /// A batch that succeeds is followed by promotion, so a ring the guest just created starts
    /// reading the moment the batch that created it is over -- and not before. Replay does not
    /// promote: its rings wait for `replay_end`, which is the C's `ctx->replaying` check moved to
    /// the place that knows the answer.
    pub fn submit(&mut self, id: CtxId, buf: &[u8]) -> Result<(), Error> {
        self.on_context(id, |ctx, todo, global, resources| {
            ctx.submit(buf, todo, global, resources)
        })?;
        self.promote(id);
        Ok(())
    }

    /// Feed one journal entry to a ring's stream. Replay only, so nothing is promoted here.
    pub fn submit_ring(&mut self, id: CtxId, ring: RingId, buf: &[u8]) -> Result<(), Error> {
        self.on_context(id, |ctx, todo, global, resources| {
            ctx.submit_ring(ring, buf, todo, global, resources)
        })
    }

    /// Run one submission against a locked context and the census behind it.
    ///
    /// The two locks are taken here, in the order this module documents, so that no caller picks
    /// its own. `false` from the closure is a poisoned stream, which is the only way a submission
    /// fails once the context has been found.
    fn on_context(
        &mut self,
        id: CtxId,
        f: impl FnOnce(&mut Context, &mut Unimplemented, &Global, &dyn ShmResources) -> bool,
    ) -> Result<(), Error> {
        let ctx = self.contexts.get(&id).ok_or(Error::NoContext)?;
        let resources = self.resources.read().expect("the resource lock is never poisoned");
        let mut ctx = ctx.lock().expect("a context lock is never poisoned");
        let mut todo = self.todo.lock().expect("the census lock is never poisoned");
        if f(&mut ctx, &mut todo, &self.global, &*resources) {
            Ok(())
        } else {
            Err(Error::Poisoned)
        }
    }

    pub fn replay_begin(&mut self, id: CtxId) -> Result<(), Error> {
        self.with_context_mut(id, Context::replay_begin).ok_or(Error::NoContext)?;
        Ok(())
    }

    /// Leave replay mode, and start every ring the journal built.
    ///
    /// The order matters and matches the C: the rings start first, then `replaying` clears. A ring
    /// promoted here resumes at the head its snapshot restored, not at the start of its buffer.
    pub fn replay_end(&mut self, id: CtxId) -> Result<(), Error> {
        self.with_context_mut(id, Context::replay_end).ok_or(Error::NoContext)?;
        // After the flag clears, not before: promotion refuses to start a ring while the context
        // is still replaying, which is the single check both paths go through.
        self.promote(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest_mem::GuestMap;
    use crate::ids::ResourceHandle;
    use crate::venus::proto::types::{VkFlags, VkRingCreateInfoMESA};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    const RES: ResourceHandle = ResourceHandle::new(449).unwrap();
    const BUF_AT: usize = 0xc0;
    const BUF_SIZE: usize = 0x20000;
    fn ctx_id() -> CtxId {
        CtxId::new(1).expect("1 is not zero")
    }

    /// A resource table with exactly one mapped shm resource, which is all a ring needs.
    struct OneShm(Arc<GuestMap>);

    impl ShmResources for OneShm {
        fn shm(&self, handle: ResourceHandle) -> Option<Arc<GuestMap>> {
            (handle == RES).then(|| Arc::clone(&self.0))
        }
    }

    /// A renderer with one venus context and the memory a ring lives in.
    fn vkr() -> (Vkr, Arc<GuestMap>) {
        let (fd, map) =
            crate::guest_mem::anonymous_shm(0x24000, "virglrs-vkrtest").expect("minted");
        drop(fd);
        let map = Arc::new(map);
        let table: SharedResources = Arc::new(RwLock::new(OneShm(Arc::clone(&map))));
        let mut v = Vkr::new(Config::default(), table);
        v.context_create(ctx_id());
        (v, map)
    }

    fn ring_info() -> VkRingCreateInfoMESA {
        VkRingCreateInfoMESA {
            resourceId: RES.get(),
            offset: 0,
            size: 0x200c4,
            headOffset: 0,
            tailOffset: 4,
            statusOffset: 8,
            bufferOffset: BUF_AT,
            bufferSize: BUF_SIZE,
            extraOffset: 0x200c0,
            extraSize: 4,
            // Park almost immediately, so a test that wants a parked ring gets one fast.
            idleTimeout: 1_000_000,
            ..Default::default()
        }
    }

    fn wire_create_ring(ring: u64, info: &VkRingCreateInfoMESA) -> Vec<u8> {
        use crate::venus::proto::serialize::{
            vn_encode_vkCreateRingMESA_args, vn_sizeof_vkCreateRingMESA_args,
        };
        use crate::venus::proto::types::vn_command_vkCreateRingMESA as Args;

        let args = Args { ring, pCreateInfo: Some(info), ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkCreateRingMESA_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkCreateRingMESA_args(&mut enc, VkFlags(0), &args);
        buf
    }

    /// A command that is legal on a ring's own stream, which is what the ring loop must be fed.
    ///
    /// Most transport commands are the context's business and a ring is refused them -- correctly,
    /// and the refusal poisons -- so a batch of those would test the refusal, not the loop.
    fn wire_ring_work() -> Vec<u8> {
        use crate::venus::proto::serialize::{
            vn_encode_vkSetReplyCommandStreamMESA_args, vn_sizeof_vkSetReplyCommandStreamMESA_args,
        };
        use crate::venus::proto::types::{
            VkCommandStreamDescriptionMESA, vn_command_vkSetReplyCommandStreamMESA as Args,
        };

        // A window in the same resource, clear of the ring's own regions.
        let stream =
            VkCommandStreamDescriptionMESA { resourceId: RES.get(), offset: 0x21000, size: 0x1000 };
        let args = Args { pStream: Some(&stream), ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkSetReplyCommandStreamMESA_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkSetReplyCommandStreamMESA_args(&mut enc, VkFlags(0), &args);
        buf
    }

    fn wire_destroy_ring(ring: u64) -> Vec<u8> {
        use crate::venus::proto::serialize::{
            vn_encode_vkDestroyRingMESA_args, vn_sizeof_vkDestroyRingMESA_args,
        };
        use crate::venus::proto::types::vn_command_vkDestroyRingMESA as Args;

        let args = Args { ring, ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkDestroyRingMESA_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkDestroyRingMESA_args(&mut enc, VkFlags(0), &args);
        buf
    }

    /// Wait for something a ring thread does, or fail rather than hang the suite.
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

    fn head(map: &GuestMap) -> u32 {
        map.load_u32(0).expect("the head is inside the mapping")
    }

    /// Put bytes in the ring's buffer and tell the guest side we did.
    fn guest_writes(map: &GuestMap, bytes: &[u8]) {
        assert!(map.copy_in(BUF_AT, bytes), "the buffer is inside the mapping");
        assert!(map.store_u32(4, bytes.len() as u32), "the tail is inside the mapping");
    }

    /// The capstone: a guest creates a ring, writes to it, and the host reads it -- through the
    /// real renderer, the real locks and a real thread, with nothing standing in for the seam.
    ///
    /// Every other ring test stops at a fake dispatcher or a locked context. This is the only one
    /// that says the pieces fit together, which is the thing a corpus of replays cannot tell us:
    /// replay never starts a ring until `replay_end`, so it never has a guest and a ring thread
    /// running at once.
    #[test]
    fn a_live_ring_reads_what_the_guest_writes() {
        let (mut v, map) = vkr();
        v.submit(ctx_id(), &wire_create_ring(7, &ring_info())).expect("the ring was created");

        guest_writes(&map, &wire_ring_work());
        until("the ring thread to consume the batch", || head(&map) != 0);

        v.context_destroy(ctx_id());
    }

    /// A ring the guest destroys stops, and its thread is gone by the time the destroy returns.
    ///
    /// The deadlock witness for the lock order. `vkDestroyRingMESA` runs inside a dispatch that
    /// already holds the context lock, and then joins the ring's thread. A thread that blocked on
    /// that lock rather than backing off would still be waiting for it, and the join would wait
    /// for the thread: this test hangs rather than fails. Nothing at module level can reach it --
    /// it needs the real `Dispatch` and the real locks.
    ///
    /// The guest keeps writing throughout, because a parked ring never takes the context lock at
    /// all: without work in flight the destroy joins a sleeping thread and proves nothing.
    #[test]
    fn destroying_a_running_ring_from_a_submission_returns() {
        let (mut v, map) = vkr();
        let feed = AtomicBool::new(true);

        // A scoped thread, so the writer borrows the mapping instead of taking a share of it: the
        // count below has to mean "a ring thread still holds this", and a share belonging to the
        // test's own scaffolding -- appearing and disappearing as that thread starts and finishes
        // -- would drown the signal.
        std::thread::scope(|s| {
            s.spawn(|| {
                let work = wire_ring_work();
                let mut at = 0usize;
                // Up to the end of the buffer and no further: a writer that wrapped would have to
                // split a command across the end, and this test is not about that. One pass is
                // thousands of batches, far more than the destroy needs to land inside.
                while feed.load(Ordering::Acquire) && at + work.len() <= BUF_SIZE {
                    assert!(map.copy_in(BUF_AT + at, &work), "the buffer is inside the mapping");
                    at += work.len();
                    assert!(map.store_u32(4, at as u32), "the tail is inside the mapping");
                    std::thread::yield_now();
                }
            });

            // Measured before a ring holds anything: the test's share and the resource table's.
            let before = Arc::strong_count(&map);
            v.submit(ctx_id(), &wire_create_ring(7, &ring_info())).expect("the ring was created");
            until("the ring thread to start consuming", || head(&map) != 0);

            v.submit(ctx_id(), &wire_destroy_ring(7)).expect("the ring was destroyed");
            // Synchronous, not eventual: the ring's share of the mapping is gone the moment the
            // destroy returns, which it can only be if the thread was joined rather than merely
            // told to stop.
            assert_eq!(
                Arc::strong_count(&map),
                before,
                "the destroy returned with the ring's thread still holding its buffer"
            );
            feed.store(false, Ordering::Release);
        });

        v.context_destroy(ctx_id());
    }

    /// Tearing down a context with a ring still running returns, and does not join a thread on
    /// itself.
    ///
    /// A ring thread upgrades a weak claim on its context for each dispatch. If teardown simply
    /// dropped the context, that upgrade could leave the ring thread holding the last reference --
    /// and `Drop for Context` would run there, joining the thread it was running on. `stop_rings`
    /// before the drop is what orders this; the assertion is that the call returns at all.
    #[test]
    fn destroying_a_context_stops_its_rings() {
        let (mut v, map) = vkr();
        // The closure's share, taken before the baseline so the baseline counts it.
        let seen = Arc::clone(&map);
        let before = Arc::strong_count(&map);

        v.submit(ctx_id(), &wire_create_ring(7, &ring_info())).expect("the ring was created");
        guest_writes(&map, &wire_ring_work());
        until("the ring thread to start consuming", || head(&map) != 0);

        // On another thread so that a teardown which deadlocks fails this test in five seconds
        // rather than hanging the suite.
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        std::thread::spawn(move || {
            v.context_destroy(ctx_id());
            // Checked while the renderer is still alive, so the count is over the same shares the
            // baseline was: this closure's, the test's, and the resource table's -- and none
            // belonging to a ring thread.
            assert_eq!(
                Arc::strong_count(&seen),
                before,
                "the teardown returned with a ring thread still holding its buffer"
            );
            flag.store(true, Ordering::Release);
        });
        until("the context teardown to return", || done.load(Ordering::Acquire));
    }

    /// A ring restored from a snapshot resumes at the head it was quiesced at.
    ///
    /// The witness for the read cursor, and the reason it lives on the `Ring` rather than starting
    /// at zero inside the loop. A snapshot is taken with the ring drained -- head equals tail, both
    /// somewhere in the middle of the buffer -- and the restored host must agree that those bytes
    /// are already answered for. A thread starting from zero instead sees the whole prefix as new
    /// work, dispatches whatever the buffer happens to hold, and poisons the context on it.
    ///
    /// The corpus cannot catch this: its rings are minted fresh and zeroed, so the two behaviours
    /// only diverge on a ring whose control words arrive non-zero.
    #[test]
    fn a_restored_ring_resumes_at_the_head_it_was_quiesced_at() {
        const QUIESCED_AT: u32 = 0x1000;

        let (mut v, map) = vkr();
        v.replay_begin(ctx_id()).expect("replay began");
        // The state a snapshot restores: drained, and not at the start of the buffer.
        assert!(map.store_u32(0, QUIESCED_AT), "the head is inside the mapping");
        assert!(map.store_u32(4, QUIESCED_AT), "the tail is inside the mapping");

        v.submit(ctx_id(), &wire_create_ring(7, &ring_info())).expect("the ring was created");
        v.replay_end(ctx_id()).expect("replay ended");

        // Nothing new has arrived, so a ring that resumed correctly has nothing to do.
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(head(&map), QUIESCED_AT, "the ring re-read bytes it had already answered for");
        assert!(
            !v.with_context(ctx_id(), Context::fatal).expect("the context is here"),
            "the ring dispatched the buffer's contents as if they were new commands"
        );

        v.context_destroy(ctx_id());
    }

    /// Replay builds rings idle and starts them at `replay_end`, never before.
    ///
    /// This is the C's `if (!ctx->replaying) vkr_ring_start(ring)` plus the loop that starts the
    /// deferred ones, and it matters for the case the corpus is made of: a snapshot's journal is
    /// fed to a ring's own stream, which is only legal while nothing else is reading it.
    #[test]
    fn a_replayed_ring_does_not_start_until_replay_ends() {
        let (mut v, map) = vkr();
        v.replay_begin(ctx_id()).expect("replay began");
        v.submit(ctx_id(), &wire_create_ring(7, &ring_info())).expect("the ring was created");

        // Still idle: the journal may yet address this ring, and a thread reading it meanwhile
        // would be a second reader of the same buffer.
        guest_writes(&map, &wire_ring_work());
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(head(&map), 0, "a replayed ring reads nothing before replay ends");

        v.replay_end(ctx_id()).expect("replay ended");
        until("the promoted ring to consume the batch", || head(&map) != 0);

        v.context_destroy(ctx_id());
    }
}
