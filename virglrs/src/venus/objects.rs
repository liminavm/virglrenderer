// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The per-context object table: guest object ids to host Vulkan handles.
//!
//! The guest chooses every id and sends it on the wire, so this table is the only thing standing
//! between a guest and a host handle it invented. Every id it names is checked here, against both
//! existence and Vulkan object type -- a guest does not get to create a buffer and then use the id
//! as a device.
//!
//! One table per context, and a context owns exactly one `VkInstance`, so every object in it hangs
//! off one instance tree. Nothing is shared between contexts: two guests may pick the same id and
//! never see each other's objects.
//!
//! **A guest id and a host handle are never the same member.** A command that creates an object
//! carries the id the guest picked, and the reply encoder sends that same field back -- a handler
//! that overwrote it with the host handle would leak a host pointer into the guest. Input handles
//! go the other way: the generated `_lookup` decode replaces the id with the host handle in place,
//! which is what makes an argument struct callable by the driver with no conversion, and is safe
//! because a reply never re-encodes an input member.
//!
//! Each direction is therefore missing one half of the pairing by the time this table is written,
//! and the generator supplies it as a shadow member alongside: `handle_<name>` is where the driver
//! writes a created object's handle, and `id_<name>` is the guest id the decoder kept as the
//! lookup overwrote it. The two arrive together at `object_created` and `object_destroyed`.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use super::cs::{Lookup, ObjectId, Objects};

/// One live object: the host handle, and the Vulkan type the guest must name it by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Object {
    /// `VkObjectType`, as a bare i32 because that is what the generated decode passes.
    pub ty: i32,
    /// The host handle, whatever Vulkan gave us for it.
    pub handle: u64,
}

/// Why an insert was refused. Each is a guest that broke the protocol, not a host mistake.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AddError {
    /// Id zero is not an object. The C asserts on it; here the guest picked it, so it is rejected.
    ZeroId,
    /// The guest reused a live id. Accepting it would orphan the object already under it.
    Duplicate,
}

#[derive(Default)]
pub struct Table {
    live: BTreeMap<ObjectId, Object>,
    /// Ids whose creation the host refused. A guest pipelines: it sends the create and the commands
    /// using it without waiting for an answer, so those later commands are already in flight when
    /// the create fails. Remembering the id turns each of them into one lost command instead of a
    /// poisoned ring -- the guest's own error handling then unwinds, as it would on real hardware.
    ghosts: BTreeSet<ObjectId>,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    /// Register a host handle under the id the guest chose.
    pub fn add(&mut self, id: ObjectId, ty: i32, handle: u64) -> Result<(), AddError> {
        if id.0 == 0 {
            return Err(AddError::ZeroId);
        }
        if self.live.contains_key(&id) {
            return Err(AddError::Duplicate);
        }
        // An id the host once refused can be created for real later: the guest is free to reuse it
        // once it has seen the failure, and a stale ghost would silently swallow the new object's
        // commands.
        self.ghosts.remove(&id);
        self.live.insert(id, Object { ty, handle });
        Ok(())
    }

    /// Record that the host refused to create this id. See [`Table::ghosts`].
    pub fn add_ghost(&mut self, id: ObjectId) {
        if id.0 != 0 {
            self.ghosts.insert(id);
        }
    }

    /// Forget an object. Returns what was there, so the caller can destroy the host handle.
    pub fn remove(&mut self, id: ObjectId) -> Option<Object> {
        self.live.remove(&id)
    }

    pub fn get(&self, id: ObjectId) -> Option<&Object> {
        self.live.get(&id)
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// Every live object, for the teardown that destroys their host handles.
    ///
    /// Ordered teardown -- instance, then physical devices, then devices, then their objects, the
    /// way the C walks its intrusive lists -- arrives with the real handles that need destroying.
    /// There is nothing to order while a handle is a number.
    pub fn drain(&mut self) -> impl Iterator<Item = (ObjectId, Object)> {
        self.ghosts.clear();
        core::mem::take(&mut self.live).into_iter()
    }
}

impl Objects for Table {
    fn lookup(&self, id: ObjectId, ty: i32) -> Lookup {
        match self.live.get(&id) {
            Some(o) if o.ty == ty => Lookup::Found(o.handle),
            // A live id named by the wrong type is a guest reinterpreting one object as another.
            // That is not a race it can lose; it is a protocol violation, and the ring stops.
            Some(_) => Lookup::Missing,
            None if self.ghosts.contains(&id) => Lookup::Ghost,
            None => Lookup::Missing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUFFER: i32 = 9;
    const IMAGE: i32 = 10;

    #[test]
    fn a_registered_object_resolves_only_under_its_own_type() {
        let mut t = Table::new();
        t.add(ObjectId(7), BUFFER, 0xdead_beef).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Found(0xdead_beef));
        // The whole point of the table: an id is not a capability for every object type.
        assert_eq!(t.lookup(ObjectId(7), IMAGE), Lookup::Missing);
    }

    #[test]
    fn an_id_the_guest_invented_stops_the_ring() {
        let t = Table::new();
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Missing);
    }

    /// A ghost is the one miss that is not the guest's fault: it pipelined commands behind a create
    /// the host refused, and they were already in flight when it failed.
    #[test]
    fn a_refused_creation_becomes_a_ghost_and_then_stops_being_one() {
        let mut t = Table::new();
        t.add_ghost(ObjectId(7));
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Ghost);

        t.add(ObjectId(7), BUFFER, 1).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Found(1));
    }

    #[test]
    fn an_id_may_not_be_zero_or_reused_while_it_is_live() {
        let mut t = Table::new();
        assert_eq!(t.add(ObjectId(0), BUFFER, 1), Err(AddError::ZeroId));
        t.add(ObjectId(7), BUFFER, 1).unwrap();
        assert_eq!(t.add(ObjectId(7), IMAGE, 2), Err(AddError::Duplicate));
        // Still the original: a refused insert must not have disturbed it.
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Found(1));

        t.remove(ObjectId(7)).unwrap();
        t.add(ObjectId(7), IMAGE, 2).unwrap();
        assert_eq!(t.lookup(ObjectId(7), IMAGE), Lookup::Found(2));
    }

    #[test]
    fn draining_hands_back_every_handle_that_needs_destroying() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, 10).unwrap();
        t.add(ObjectId(2), IMAGE, 20).unwrap();
        let mut got: Vec<_> = t.drain().map(|(id, o)| (id.0, o.handle)).collect();
        got.sort_unstable();
        assert_eq!(got, [(1, 10), (2, 20)]);
        assert!(t.is_empty());
    }
}

/// The table as the decoder and the handlers both see it.
///
/// The decoder holds it while it reads a command, and the handler that command reaches writes to
/// it -- so it cannot be an exclusive borrow on either side. Decoding finishes before the handler
/// runs, so the two never overlap; the cell is what lets the compiler stop caring, and it will
/// panic loudly rather than quietly if that stops being true.
#[derive(Default)]
pub struct Shared(RefCell<Table>);

impl Shared {
    pub fn new() -> Shared {
        Shared::default()
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, Table> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, Table> {
        self.0.borrow_mut()
    }
}

impl Objects for Shared {
    fn lookup(&self, id: ObjectId, ty: i32) -> Lookup {
        self.0.borrow().lookup(id, ty)
    }
}
