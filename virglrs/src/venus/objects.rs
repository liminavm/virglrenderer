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
use std::collections::BTreeMap;

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

/// What one id names. An id names exactly one of these, which is the point of the enum: a live
/// object and a refused creation are mutually exclusive states, and keeping them in two containers
/// meant nothing stopped an id from being written into both.
///
/// It happened along one real path -- a guest naming a live id in a create the driver refused
/// ghosted an object that was still there. Nothing went wrong while both held, because the lookup
/// consulted the live side first; the damage was deferred to the destroy, after which the stale
/// ghost went on answering for an id that named nothing, turning every later command naming it into
/// a silent drop where the ring should have stopped.
enum Slot {
    Live(Object),
    /// The host refused to create this id. A guest pipelines: it sends the create and the commands
    /// using it without waiting for an answer, so those later commands are already in flight when
    /// the create fails. Remembering the id turns each of them into one lost command instead of a
    /// poisoned ring -- the guest's own error handling then unwinds, as it would on real hardware.
    Ghost,
}

impl Slot {
    fn live(&self) -> Option<&Object> {
        match self {
            Slot::Live(o) => Some(o),
            Slot::Ghost => None,
        }
    }

    fn into_live(self) -> Option<Object> {
        match self {
            Slot::Live(o) => Some(o),
            Slot::Ghost => None,
        }
    }
}

#[derive(Default)]
pub struct Table {
    slots: BTreeMap<ObjectId, Slot>,
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
        if self.get(id).is_some() {
            return Err(AddError::Duplicate);
        }
        // An id the host once refused can be created for real later: the guest is free to reuse it
        // once it has seen the failure, and the slot it lands in is the one the ghost occupied --
        // a stale ghost left beside it would silently swallow the new object's commands.
        self.slots.insert(id, Slot::Live(Object { ty, handle }));
        Ok(())
    }

    /// Whether the host refused this id, so a later registration must not resurrect it.
    pub fn is_ghost(&self, id: ObjectId) -> bool {
        matches!(self.slots.get(&id), Some(Slot::Ghost))
    }

    /// Record that the host refused to create this id. See [`Slot::Ghost`].
    ///
    /// An id that already names a live object keeps it. A guest may name a live id in a create,
    /// and when the driver refuses that create nothing has changed -- the object still there is
    /// the truth, and a ghost written over it would outlive the object's own destroy.
    pub fn add_ghost(&mut self, id: ObjectId) {
        if id.0 == 0 || self.get(id).is_some() {
            return;
        }
        self.slots.insert(id, Slot::Ghost);
    }

    /// Forget an object. Returns what was there, so the caller can destroy the host handle.
    ///
    /// Only a live object is taken out. A ghost outlives the destroy that names it: the guest
    /// pipelined that destroy behind the create that failed, and every command in between is still
    /// in flight behind it.
    pub fn remove(&mut self, id: ObjectId) -> Option<Object> {
        match self.slots.get(&id)? {
            Slot::Ghost => None,
            Slot::Live(_) => self.slots.remove(&id).and_then(Slot::into_live),
        }
    }

    pub fn get(&self, id: ObjectId) -> Option<&Object> {
        self.slots.get(&id).and_then(Slot::live)
    }

    pub fn len(&self) -> usize {
        self.slots.values().filter_map(Slot::live).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every live object, for the teardown that destroys their host handles.
    ///
    /// Ordered teardown -- instance, then physical devices, then devices, then their objects, the
    /// way the C walks its intrusive lists -- arrives with the real handles that need destroying.
    /// There is nothing to order while a handle is a number.
    pub fn drain(&mut self) -> impl Iterator<Item = (ObjectId, Object)> {
        core::mem::take(&mut self.slots)
            .into_iter()
            .filter_map(|(id, slot)| slot.into_live().map(|o| (id, o)))
    }
}

impl Objects for Table {
    fn lookup(&self, id: ObjectId, ty: i32) -> Lookup {
        match self.slots.get(&id) {
            Some(Slot::Live(o)) if o.ty == ty => Lookup::Found(o.handle),
            // A live id named by the wrong type is a guest reinterpreting one object as another.
            // That is not a race it can lose; it is a protocol violation, and the ring stops.
            Some(Slot::Live(_)) => Lookup::Missing,
            Some(Slot::Ghost) => Lookup::Ghost,
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

    /// The state the two-container shape allowed and this one cannot represent.
    ///
    /// Neither corpus does this -- measured, over every context in both: no id was ever added
    /// while ghosted, and no live id was ever also a ghost. Which is exactly why it needs a test:
    /// a sequence the recordings do not contain is a sequence only the types can rule out.
    #[test]
    fn a_refused_create_never_ghosts_an_id_that_already_names_something() {
        let mut t = Table::new();
        t.add(ObjectId(7), BUFFER, 1).unwrap();

        // A guest naming a live id in a create, and a driver that refuses that create. Nothing
        // changed, so the object still there is the truth.
        t.add_ghost(ObjectId(7));
        assert!(!t.is_ghost(ObjectId(7)));
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Found(1));

        // And after the guest destroys it the id names nothing at all. A ghost surviving here is
        // the whole bug: every later command naming the id would be swallowed as one lost command
        // when the ring should have stopped.
        t.remove(ObjectId(7)).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Missing);
    }

    /// The other direction, which must keep working: a ghost is not something a destroy clears.
    #[test]
    fn a_ghost_outlives_the_destroy_the_guest_pipelined_behind_it() {
        let mut t = Table::new();
        t.add_ghost(ObjectId(7));
        // The guest sent create, use, use, destroy without waiting for the create's answer. The
        // destroy names nothing, and the uses still in flight behind it are still the guest's to
        // unwind from.
        assert!(t.remove(ObjectId(7)).is_none());
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Ghost);
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
