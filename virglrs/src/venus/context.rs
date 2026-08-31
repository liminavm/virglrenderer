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
use super::cs::ObjectId;
use super::objects::Shared;
use super::proto::serialize::{Commands, vn_command_name, vn_dispatch_command};
use super::proto::types::{VkCommandTypeEXT, VkFlags, VkObjectType};

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

    /// Poison the context, naming the command that did it -- once. A ring the guest can no longer
    /// use looks the same from inside the guest whatever caused it, so the command type is the
    /// only thing that tells a bug report from a hostile stream apart.
    fn poison(&self, dec: &Decoder<'_>, cmd: VkCommandTypeEXT, why: &str) {
        if !self.fatal.get() {
            let name = vn_command_name(cmd)
                .map(str::to_string)
                .unwrap_or_else(|| format!("command type {}", cmd.0));
            eprintln!("[virglrs] ctx {}: {name} {why}, {} bytes in", self.id.0, dec.pos());
        }
        dec.set_fatal();
    }

    /// Drain one submission, dispatching every command in it.
    ///
    /// Returns false when the context was poisoned -- by this batch or by an earlier one. The C
    /// bails early on an already-fatal context for the same reason: a stream we stopped trusting
    /// does not become trustworthy because the guest sent more of it.
    pub fn submit(&mut self, buf: &[u8], todo: &mut Unimplemented) -> bool {
        if self.fatal.get() {
            return false;
        }

        // One arena for the batch. Every temporary a command decodes into lives until the batch
        // ends, which is the same bargain the C makes with its temp pool -- and the decoder's own
        // cap, not this arena, is what stops a guest from asking for all of memory.
        let temp = Bump::new();
        let mut dec = Decoder::new(buf, &temp, &self.objects, &self.fatal);
        let mut h = Handlers { objects: &self.objects, todo, bad_id: false };

        while dec.has_command() {
            dec.clear_soft_fatal();

            let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
            let flags = dec.decode_scalar::<VkFlags>();
            if dec.hard_fatal() {
                // The header itself was short: there is no command here to lose.
                eprintln!(
                    "[virglrs] ctx {}: submission ends mid-header, {} bytes into {}",
                    self.id.0,
                    dec.pos(),
                    buf.len()
                );
                break;
            }

            // A command that wants an answer has nowhere to be answered into: the reply buffer is
            // the ring's, and there is no ring loop yet. Poisoning is the honest response --
            // dispatching and dropping the reply would leave the guest waiting on a reply that was
            // never written, which is a hang rather than an error. Replay never takes this branch:
            // the journal's entries have had their reply flag stripped already.
            if flags.0 & GENERATE_REPLY != 0 && !self.replay {
                self.unhandled += 1;
                self.poison(&dec, cmd, "wants a reply, and there is no ring to answer into");
                break;
            }

            if vn_dispatch_command(&mut dec, None, cmd, &mut h).is_none() {
                // A command type this protocol does not define. We cannot even skip it: its length
                // is only knowable by decoding it.
                self.poison(&dec, cmd, "is not a command type this protocol defines");
                break;
            }
            self.dispatched += 1;
            // A guest that named id zero, or reused one that is still live, is naming objects
            // it cannot have. The handler had no decoder to say so; this is where it lands.
            if h.bad_id {
                self.poison(&dec, cmd, "named an object it cannot have");
            }

            if self.fatal.get() {
                // The decoder poisoned itself inside the command: a malformed argument, or a
                // shape the generator has no decoder for. Either way the command is what a
                // reader needs, because without it a gap reaches a user as a hung guest.
                self.poison(&dec, cmd, "did not decode");
                break;
            }
        }
        // Every exit from the loop is one place, so a branch that poisons and breaks cannot report
        // success on the way out.
        !self.fatal.get()
    }

    /// A ring-scoped submission. The ring the command belongs to is recorded but not yet acted on:
    /// the replay feed hands commands straight to the dispatcher, which is what makes a VM-free
    /// replay possible, and a real ring loop is what will need the index.
    pub fn submit_ring(&mut self, _ring: RingIdx, buf: &[u8], todo: &mut Unimplemented) -> bool {
        self.submit(buf, todo)
    }
}

/// The commands a build does not serve yet, counted.
#[derive(Default)]
pub struct Unimplemented {
    pub seen: std::collections::BTreeMap<i32, u64>,
}

/// What a command reaches: the object table it registers into, and the tally of what this build
/// cannot do yet.
///
/// Every command method is left at its generated default, so each lands on `unsupported` and is
/// counted. What is *not* left to a default is the object bookkeeping: a create registers the id
/// the guest chose, a destroy forgets it, and that is enough for the whole corpus to decode --
/// every later command that names an object finds it.
///
/// **The host handle is the guest id, and only because nothing here has a real one.** Registering
/// them equal is what lets a destroy work at all: the generated `_lookup` decode replaces the id
/// in the arguments with the host handle before the handler sees it, so the destroy extraction
/// reads back a host handle and calls it an id. When real handles arrive that identity breaks, and
/// with it this shortcut -- the destroy path needs the guest id carried alongside, which is a
/// change to what the generated argument structs hold.
pub struct Handlers<'a> {
    objects: &'a Shared,
    todo: &'a mut Unimplemented,
    /// A guest that reused a live id or named id zero. It cannot be reported from here -- the
    /// handler has no decoder -- so the loop reads it back and poisons.
    bad_id: bool,
}

impl Commands for Handlers<'_> {
    fn unsupported(&mut self, cmd: VkCommandTypeEXT) {
        *self.todo.seen.entry(cmd.0).or_default() += 1;
    }

    fn object_created(&mut self, ty: VkObjectType, id: ObjectId) {
        if self.objects.borrow_mut().add(id, ty.0, id.0).is_err() {
            self.bad_id = true;
        }
    }

    fn object_destroyed(&mut self, _ty: VkObjectType, id: ObjectId) {
        self.objects.borrow_mut().remove(id);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn header(cmd: VkCommandTypeEXT, flags: u32) -> Vec<u8> {
        let mut w = (cmd.0 as u32).to_le_bytes().to_vec();
        w.extend_from_slice(&flags.to_le_bytes());
        w
    }

    /// A shape the generator has no decoder for poisons the ring, and the log line that says so
    /// has to be able to name the command. Without the name a gap reaches a user as a hung guest
    /// with nothing to report; `vn_command_name` returning `None` here would be that silently.
    #[test]
    fn an_undecodable_command_poisons_the_context_by_name() {
        let cmd = VkCommandTypeEXT::VK_COMMAND_TYPE_vkGetPipelineCacheData_EXT;
        assert_eq!(vn_command_name(cmd), Some("vkGetPipelineCacheData"));

        let mut ctx = Context::new(CtxId(1));
        ctx.replay_begin();
        let mut todo = Unimplemented::default();
        assert!(!ctx.submit(&header(cmd, 0), &mut todo), "a stubbed decoder must poison");
        assert!(ctx.fatal());

        // The poison outlives the batch: a stream we stopped trusting stays untrusted.
        assert!(!ctx.submit(&header(cmd, 0), &mut todo));
    }

    /// A command that wants an answer has nowhere to be answered into, so it poisons -- but only
    /// outside replay, where the journal's replies have already been stripped.
    #[test]
    fn a_reply_request_poisons_only_outside_replay() {
        let cmd = VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyInstance_EXT;
        let mut todo = Unimplemented::default();

        let mut ctx = Context::new(CtxId(1));
        let w = header(cmd, GENERATE_REPLY);
        let mut full = w.clone();
        full.extend_from_slice(&1u64.to_le_bytes()); // instance id
        full.extend_from_slice(&0u64.to_le_bytes()); // no allocator
        assert!(!ctx.submit(&full, &mut todo));
        assert_eq!(ctx.unhandled, 1);

        // In replay the flag is stripped, so the command reaches the dispatcher instead of the
        // poison. It still names an instance nothing created, which poisons for its own reason --
        // what separates the two paths is whether the command was dispatched at all.
        let mut ctx = Context::new(CtxId(1));
        ctx.replay_begin();
        assert!(!ctx.submit(&full, &mut todo));
        assert_eq!(ctx.dispatched, 1);
        assert_eq!(ctx.unhandled, 0);
    }
}
