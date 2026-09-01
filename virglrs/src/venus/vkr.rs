// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The venus renderer root.
//!
//! The C runs vkr behind the render-server proxy (`server/render_state.c`), which is a lock and a
//! forward to `vkr_renderer_*` in the same process. None of that survives here: vkr is an object
//! the `Renderer` owns, reached through it, with no file-scope state and no proxy hop. The C's
//! `vkr_renderer.h` is the interface this mirrors -- that header, not the proxy, is the design.

use std::collections::BTreeMap;

use crate::config::Config;
use crate::ids::{CtxId, RingIdx};

use super::context::{Context, Unimplemented};
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
    contexts: BTreeMap<CtxId, Context>,
    /// The commands this build does not serve yet, counted across every context. Kept on the root
    /// because it answers a question about the build, not about a guest.
    pub todo: Unimplemented,
    /// The entry points that exist before any instance does. One per renderer rather than one per
    /// context: they are the loader's, identical for every guest, and immutable once resolved.
    global: Global,
}

impl Vkr {
    pub fn new(config: Config) -> Vkr {
        Vkr {
            config,
            contexts: BTreeMap::new(),
            todo: Unimplemented::default(),
            global: crate::vulkan::global(),
        }
    }

    pub fn context_create(&mut self, id: CtxId) {
        self.contexts.insert(id, Context::new(id));
    }

    /// Tear a context down. Every host handle it still holds dies with it -- a guest that leaks is
    /// not a guest that gets to keep host memory after it is gone.
    pub fn context_destroy(&mut self, id: CtxId) {
        let Some(mut ctx) = self.contexts.remove(&id) else {
            return;
        };
        // A context usually dies mid-workload, with the guest still holding everything it made,
        // so this is the destroy that actually runs -- not a tidy-up after the guest's own.
        // The table is emptied first, and it is what says which device each object hangs off: the
        // driver is about to destroy the devices, and afterwards there is nothing left to destroy
        // anything on.
        let doomed = ctx.objects().borrow_mut().take_all();
        ctx.driver_mut().teardown(&doomed);
    }

    pub fn context(&self, id: CtxId) -> Option<&Context> {
        self.contexts.get(&id)
    }

    /// Run a submission on one context.
    pub fn submit(&mut self, id: CtxId, buf: &[u8]) -> Result<(), Error> {
        let ctx = self.contexts.get_mut(&id).ok_or(Error::NoContext)?;
        if ctx.submit(buf, &mut self.todo, &self.global) { Ok(()) } else { Err(Error::Poisoned) }
    }

    pub fn submit_ring(&mut self, id: CtxId, ring: RingIdx, buf: &[u8]) -> Result<(), Error> {
        let ctx = self.contexts.get_mut(&id).ok_or(Error::NoContext)?;
        if ctx.submit_ring(ring, buf, &mut self.todo, &self.global) {
            Ok(())
        } else {
            Err(Error::Poisoned)
        }
    }

    pub fn replay_begin(&mut self, id: CtxId) -> Result<(), Error> {
        self.contexts.get_mut(&id).ok_or(Error::NoContext)?.replay_begin();
        Ok(())
    }

    pub fn replay_end(&mut self, id: CtxId) -> Result<(), Error> {
        self.contexts.get_mut(&id).ok_or(Error::NoContext)?.replay_end();
        Ok(())
    }
}
