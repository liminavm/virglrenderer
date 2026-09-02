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
use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::ids::{CtxId, RingId};

use super::context::{Context, Unimplemented};
use super::ring::ShmResources;
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
    global: Global,
}

impl Vkr {
    pub fn new(config: Config) -> Vkr {
        Vkr {
            config,
            contexts: BTreeMap::new(),
            todo: Arc::new(Mutex::new(Unimplemented::default())),
            global: crate::vulkan::global(),
        }
    }

    pub fn context_create(&mut self, id: CtxId) {
        self.contexts.insert(id, Arc::new(Mutex::new(Context::new(id))));
    }

    /// Tear a context down. Every host handle it still holds dies with it -- a guest that leaks is
    /// not a guest that gets to keep host memory after it is gone.
    pub fn context_destroy(&mut self, id: CtxId) {
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

    /// Run a submission on one context.
    pub fn submit(
        &mut self,
        id: CtxId,
        buf: &[u8],
        resources: &dyn ShmResources,
    ) -> Result<(), Error> {
        self.on_context(id, |ctx, todo, global| ctx.submit(buf, todo, global, resources))
    }

    pub fn submit_ring(
        &mut self,
        id: CtxId,
        ring: RingId,
        buf: &[u8],
        resources: &dyn ShmResources,
    ) -> Result<(), Error> {
        self.on_context(id, |ctx, todo, global| ctx.submit_ring(ring, buf, todo, global, resources))
    }

    /// Run one submission against a locked context and the census behind it.
    ///
    /// The two locks are taken here, in the order this module documents, so that no caller picks
    /// its own. `false` from the closure is a poisoned stream, which is the only way a submission
    /// fails once the context has been found.
    fn on_context(
        &mut self,
        id: CtxId,
        f: impl FnOnce(&mut Context, &mut Unimplemented, &Global) -> bool,
    ) -> Result<(), Error> {
        let ctx = self.contexts.get(&id).ok_or(Error::NoContext)?;
        let mut ctx = ctx.lock().expect("a context lock is never poisoned");
        let mut todo = self.todo.lock().expect("the census lock is never poisoned");
        if f(&mut ctx, &mut todo, &self.global) { Ok(()) } else { Err(Error::Poisoned) }
    }

    pub fn replay_begin(&mut self, id: CtxId) -> Result<(), Error> {
        self.with_context_mut(id, Context::replay_begin).ok_or(Error::NoContext)?;
        Ok(())
    }

    pub fn replay_end(&mut self, id: CtxId) -> Result<(), Error> {
        self.with_context_mut(id, Context::replay_end).ok_or(Error::NoContext)?;
        Ok(())
    }
}
