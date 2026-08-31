// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! One guest's venus context: its object table, its poison, and the loop that drains a submission.
//!
//! A submission is a stream of commands, not one command. The loop mirrors
//! `vkr_context_submit_cmd`: clear the per-command poison, dispatch, and stop the whole batch the
//! moment the stream itself becomes untrustworthy. The distinction is the point -- a lost command
//! costs the guest one operation, a poisoned context costs it the ring, and only the second one
//! ends the loop.

use bumpalo::Bump;
use std::cell::Cell;

use crate::ids::{CtxId, RingIdx};

use super::cs::Decoder;
use super::objects::Shared;
use super::proto::serialize::{Commands, vn_command_name, vn_dispatch_command};
use super::proto::types::{VkCommandTypeEXT, VkFlags};

/// `VK_COMMAND_GENERATE_REPLY_BIT_EXT`: the guest wants an answer to this command.
const GENERATE_REPLY: u32 = 0x1;

pub struct Context {
    pub id: CtxId,
    /// The hard poison. It outlives any one command and any one submission: once the stream cannot
    /// be trusted, nothing later in it can be either.
    fatal: Cell<bool>,
    objects: Shared,
    /// Replay mode. The journal's entries are fed straight to the dispatcher with their reply flag
    /// stripped, so rings are never started and no reply is ever encoded.
    replay: bool,
    /// Commands dispatched, and how many of those reached a handler this build does not have.
    /// The second number is what says how far a corpus actually got.
    pub dispatched: u64,
    pub unhandled: u64,
}

impl Context {
    pub fn new(id: CtxId) -> Context {
        Context {
            id,
            fatal: Cell::new(false),
            objects: Shared::new(),
            replay: false,
            dispatched: 0,
            unhandled: 0,
        }
    }

    pub fn fatal(&self) -> bool {
        self.fatal.get()
    }

    pub fn objects(&self) -> &Shared {
        &self.objects
    }

    /// Enter replay mode: the journal is about to be fed in, so nothing may answer it.
    pub fn replay_begin(&mut self) {
        self.replay = true;
    }

    /// Leave replay mode. In the C this is also where deferred rings start; there are no rings to
    /// start until a ring loop exists.
    pub fn replay_end(&mut self) {
        self.replay = false;
    }

    /// Drain one submission, dispatching every command in it.
    ///
    /// Returns false when the context was poisoned -- by this batch or by an earlier one. The C
    /// bails early on an already-fatal context for the same reason: a stream we stopped trusting
    /// does not become trustworthy because the guest sent more of it.
    pub fn submit(&mut self, buf: &[u8], h: &mut dyn Commands) -> bool {
        if self.fatal.get() {
            return false;
        }

        // One arena for the batch. Every temporary a command decodes into lives until the batch
        // ends, which is the same bargain the C makes with its temp pool -- and the decoder's own
        // cap, not this arena, is what stops a guest from asking for all of memory.
        let temp = Bump::new();
        let mut dec = Decoder::new(buf, &temp, &self.objects, &self.fatal);

        while dec.has_command() {
            dec.clear_soft_fatal();

            let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
            let flags = dec.decode_scalar::<VkFlags>();
            if dec.hard_fatal() {
                // The header itself was short: there is no command here to lose.
                break;
            }

            // A command that wants an answer has nowhere to be answered into: the reply buffer is
            // the ring's, and there is no ring loop yet. Poisoning is the honest response --
            // dispatching and dropping the reply would leave the guest waiting on a reply that was
            // never written, which is a hang rather than an error. Replay never takes this branch:
            // the journal's entries have had their reply flag stripped already.
            if flags.0 & GENERATE_REPLY != 0 && !self.replay {
                self.unhandled += 1;
                dec.set_fatal();
                break;
            }

            if vn_dispatch_command(&mut dec, None, cmd, h).is_none() {
                // A command type this protocol does not define. We cannot even skip it: its length
                // is only knowable by decoding it.
                dec.set_fatal();
            }
            self.dispatched += 1;

            if self.fatal.get() {
                return false;
            }
        }
        true
    }

    /// A ring-scoped submission. The ring the command belongs to is recorded but not yet acted on:
    /// the replay feed hands commands straight to the dispatcher, which is what makes a VM-free
    /// replay possible, and a real ring loop is what will need the index.
    pub fn submit_ring(&mut self, _ring: RingIdx, buf: &[u8], h: &mut dyn Commands) -> bool {
        self.submit(buf, h)
    }
}

/// A renderer that serves no commands yet: every one of them lands on `unsupported`.
///
/// It exists so the plumbing -- contexts, routing, the decode loop -- can be exercised over a whole
/// corpus before a single Vulkan call is made. What it counts is the shape of the work left.
#[derive(Default)]
pub struct Unimplemented {
    pub seen: std::collections::BTreeMap<i32, u64>,
}

impl Commands for Unimplemented {
    fn unsupported(&mut self, cmd: VkCommandTypeEXT) {
        *self.seen.entry(cmd.0).or_default() += 1;
    }
}

impl Unimplemented {
    /// The commands a corpus asked for, most-used first -- the order to implement them in.
    pub fn by_frequency(&self) -> Vec<(&'static str, u64)> {
        let mut v: Vec<_> = self
            .seen
            .iter()
            .map(|(c, n)| (vn_command_name(VkCommandTypeEXT(*c)).unwrap_or("?"), *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }
}
