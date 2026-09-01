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
    /// What this object was created under, or `None` for the instance, which is the root.
    ///
    /// A key and not an id: an id is what the *guest* calls the parent, and the guest is free to
    /// destroy a parent and name a fresh object by the same id. The key stops pointing at anything
    /// the moment the parent dies, so a new parent cannot inherit a dead one's children.
    parent: Option<Key>,
}

/// Where an object lives in the [`Arena`], and which occupant of that place it is.
///
/// The generation is the whole mechanism. A key names a slot *and* the object that was in it when
/// the key was made, so a key to something destroyed resolves to nothing even after the slot has
/// been handed to an unrelated object. That is what makes a stale reference fail on its own,
/// rather than by someone remembering to go and delete it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Key {
    index: u32,
    generation: u32,
}

/// One place in the arena, occupied or not.
struct Entry {
    /// Bumped every time the slot is emptied, which is what invalidates every key to it.
    generation: u32,
    object: Option<Object>,
}

/// The one owner of every live object.
///
/// Everything else -- the guest's ids here, and in time the driver's own maps -- holds [`Key`]s
/// into this and nothing else. There is exactly one copy of an object's handle, so there is no
/// second copy to forget to update.
#[derive(Default)]
struct Arena {
    entries: Vec<Entry>,
    /// Slots emptied by a removal, to be handed out again.
    free: Vec<u32>,
}

impl Arena {
    fn insert(&mut self, object: Object) -> Key {
        if let Some(index) = self.free.pop() {
            let e = &mut self.entries[index as usize];
            e.object = Some(object);
            return Key { index, generation: e.generation };
        }
        let index = u32::try_from(self.entries.len()).expect("far more objects than a guest makes");
        self.entries.push(Entry { generation: 0, object: Some(object) });
        Key { index, generation: 0 }
    }

    fn get(&self, key: Key) -> Option<&Object> {
        let e = self.entries.get(key.index as usize)?;
        if e.generation != key.generation {
            return None;
        }
        e.object.as_ref()
    }

    fn remove(&mut self, key: Key) -> Option<Object> {
        let e = self.entries.get_mut(key.index as usize)?;
        if e.generation != key.generation {
            return None;
        }
        let object = e.object.take()?;
        // Every key to this slot, including the one just used, now names an occupant that is gone.
        e.generation = e.generation.wrapping_add(1);
        self.free.push(key.index);
        Some(object)
    }

    /// Remove an object and everything created under it, however deep.
    ///
    /// Vulkan destroys a device's objects with the device and a pool's contents with the pool,
    /// naming none of them. Walking the arena is what makes that a fact about the one place
    /// objects live, rather than a child list kept beside them that a destroy path has to
    /// remember to visit -- the list that goes stale is the bug this replaces.
    fn remove_tree(&mut self, key: Key) -> Option<Object> {
        let root = self.remove(key)?;
        let mut doomed = vec![key];
        while let Some(parent) = doomed.pop() {
            let children: Vec<Key> = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.object.as_ref().is_some_and(|o| o.parent == Some(parent)))
                .map(|(i, e)| Key { index: i as u32, generation: e.generation })
                .collect();
            for child in children {
                self.remove(child);
                doomed.push(child);
            }
        }
        Some(root)
    }

    fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.object.is_some()).count()
    }
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
    Live(Key),
    /// The host refused to create this id. A guest pipelines: it sends the create and the commands
    /// using it without waiting for an answer, so those later commands are already in flight when
    /// the create fails. Remembering the id turns each of them into one lost command instead of a
    /// poisoned ring -- the guest's own error handling then unwinds, as it would on real hardware.
    Ghost,
}

impl Slot {
    fn key(&self) -> Option<Key> {
        match self {
            Slot::Live(k) => Some(*k),
            Slot::Ghost => None,
        }
    }
}

#[derive(Default)]
pub struct Table {
    /// What the guest calls each object. Entries here are allowed to go stale: a key whose object
    /// the arena has dropped resolves to nothing, so a parent's destroy does not have to come back
    /// and tidy this map. That is the point of the whole shape -- the bookkeeping that could be
    /// forgotten no longer exists.
    slots: BTreeMap<ObjectId, Slot>,
    arena: Arena,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    /// Register a host handle under the id the guest chose, beneath the object that owns it.
    ///
    /// `owner` is the guest's id for the parent -- the device for most objects, the physical
    /// device for a device, `None` for the instance. An owner that no longer resolves leaves the
    /// object parentless rather than refusing it: the id came off the wire, so it is the guest's
    /// to get wrong, and the create it belongs to has already happened.
    pub fn add(
        &mut self,
        id: ObjectId,
        ty: i32,
        handle: u64,
        owner: Option<ObjectId>,
    ) -> Result<(), AddError> {
        if id.0 == 0 {
            return Err(AddError::ZeroId);
        }
        if self.get(id).is_some() {
            return Err(AddError::Duplicate);
        }
        let parent = owner.and_then(|o| self.slots.get(&o)).and_then(Slot::key);
        let key = self.arena.insert(Object { ty, handle, parent });
        // An id the host once refused can be created for real later, and so can one whose object
        // died with its parent: both leave an entry here that resolves to nothing, and both are
        // overwritten rather than left beside the new object to swallow its commands.
        self.slots.insert(id, Slot::Live(key));
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
        let key = self.slots.get(&id)?.key()?;
        // Everything created under it goes at the same moment, because Vulkan has just destroyed
        // it all without naming any of it. Their ids stay in `slots` and stop resolving, which is
        // the cheap half of the trade: nothing has to walk back here and delete them.
        let object = self.arena.remove_tree(key)?;
        self.slots.remove(&id);
        Some(object)
    }

    pub fn get(&self, id: ObjectId) -> Option<&Object> {
        self.arena.get(self.slots.get(&id)?.key()?)
    }

    pub fn len(&self) -> usize {
        self.arena.len()
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
        let slots = core::mem::take(&mut self.slots);
        let mut arena = core::mem::take(&mut self.arena);
        // Taken through `slots` rather than straight off the arena because the caller is owed the
        // guest's name for each, and stale entries drop out on their own: an object the arena no
        // longer holds was destroyed with its parent and has nothing left to destroy.
        slots
            .into_iter()
            .filter_map(move |(id, slot)| slot.key().and_then(|k| arena.remove(k)).map(|o| (id, o)))
    }
}

impl Objects for Table {
    fn lookup(&self, id: ObjectId, ty: i32) -> Lookup {
        match self.slots.get(&id) {
            // A key the arena no longer honours is an object that died with its parent. The id is
            // as good as one the guest invented, and is answered the same way.
            Some(Slot::Live(key)) => match self.arena.get(*key) {
                Some(o) if o.ty == ty => Lookup::Found(o.handle),
                // A live id named by the wrong type is a guest reinterpreting one object as
                // another. That is not a race it can lose; it is a protocol violation, and the
                // ring stops.
                Some(_) => Lookup::Missing,
                None => Lookup::Missing,
            },
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
        t.add(ObjectId(7), BUFFER, 0xdead_beef, None).unwrap();
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

        t.add(ObjectId(7), BUFFER, 1, None).unwrap();
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
        t.add(ObjectId(7), BUFFER, 1, None).unwrap();

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
        assert_eq!(t.add(ObjectId(0), BUFFER, 1, None), Err(AddError::ZeroId));
        t.add(ObjectId(7), BUFFER, 1, None).unwrap();
        assert_eq!(t.add(ObjectId(7), IMAGE, 2, None), Err(AddError::Duplicate));
        // Still the original: a refused insert must not have disturbed it.
        assert_eq!(t.lookup(ObjectId(7), BUFFER), Lookup::Found(1));

        t.remove(ObjectId(7)).unwrap();
        t.add(ObjectId(7), IMAGE, 2, None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), IMAGE), Lookup::Found(2));
    }

    /// The reason parentage is a key and not the guest's id for the parent.
    ///
    /// A slot freed by one object is handed to the next, so an object's place in the arena says
    /// nothing about which object it is. The generation beside it does, and it is what stops a
    /// freshly created parent from inheriting a dead one's orphans.
    #[test]
    fn a_reused_arena_slot_does_not_answer_to_the_dead_objects_key() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, 10, None).unwrap();
        t.remove(ObjectId(1)).unwrap();

        // Same place, different occupant -- and the id that named the first one is not fooled.
        t.add(ObjectId(2), IMAGE, 20, None).unwrap();
        assert_eq!(t.lookup(ObjectId(2), IMAGE), Lookup::Found(20));
        assert_eq!(t.lookup(ObjectId(1), BUFFER), Lookup::Missing);
    }

    /// Destroying a parent destroys what hangs off it, however deep -- Vulkan does exactly this
    /// and names none of it.
    #[test]
    fn destroying_a_parent_takes_every_generation_below_it() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, 10, None).unwrap();
        t.add(ObjectId(2), IMAGE, 20, Some(ObjectId(1))).unwrap();
        t.add(ObjectId(3), IMAGE, 30, Some(ObjectId(2))).unwrap();
        // A sibling tree that must survive: a cascade that took everything would pass a test that
        // only looked down one branch.
        t.add(ObjectId(4), BUFFER, 40, None).unwrap();
        t.add(ObjectId(5), IMAGE, 50, Some(ObjectId(4))).unwrap();

        t.remove(ObjectId(1)).unwrap();
        for id in [1, 2, 3] {
            assert!(t.get(ObjectId(id)).is_none(), "id {id} died with the root above it");
        }
        assert_eq!(t.lookup(ObjectId(4), BUFFER), Lookup::Found(40));
        assert_eq!(t.lookup(ObjectId(5), IMAGE), Lookup::Found(50));
        assert_eq!(t.len(), 2);
    }

    /// The half that makes the cascade affordable: an id orphaned by its parent's destroy is left
    /// where it is, and stops resolving on its own. Nothing walks back to delete it, which is why
    /// there is nothing to forget to do -- and the id is still free to name something new.
    #[test]
    fn an_id_orphaned_by_its_parents_destroy_is_reusable_without_being_cleaned_up() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, 10, None).unwrap();
        t.add(ObjectId(2), IMAGE, 20, Some(ObjectId(1))).unwrap();
        t.remove(ObjectId(1)).unwrap();

        assert_eq!(t.lookup(ObjectId(2), IMAGE), Lookup::Missing);
        // A destroy the guest pipelined behind the device's finds nothing left to destroy, and
        // must not report one -- the host handle is already gone.
        assert!(t.remove(ObjectId(2)).is_none());
        // And the id is not burned: the guest may name a new object by it.
        t.add(ObjectId(2), BUFFER, 21, None).unwrap();
        assert_eq!(t.lookup(ObjectId(2), BUFFER), Lookup::Found(21));
    }

    /// An owner the guest names that no longer resolves is the guest's mistake, not a reason to
    /// refuse a create the host has already served. The object lands parentless.
    #[test]
    fn a_create_under_an_owner_that_is_already_gone_still_registers() {
        let mut t = Table::new();
        t.add(ObjectId(9), BUFFER, 90, Some(ObjectId(1))).unwrap();
        assert_eq!(t.lookup(ObjectId(9), BUFFER), Lookup::Found(90));
    }

    #[test]
    fn draining_hands_back_every_handle_that_needs_destroying() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, 10, None).unwrap();
        t.add(ObjectId(2), IMAGE, 20, None).unwrap();
        // One of them hangs off the other, so a drain that walked the tree would hand back only
        // the root. Teardown wants every live handle: Vulkan is being told about each one.
        t.add(ObjectId(3), IMAGE, 30, Some(ObjectId(1))).unwrap();
        let mut got: Vec<_> = t.drain().map(|(id, o)| (id.0, o.handle)).collect();
        got.sort_unstable();
        assert_eq!(got, [(1, 10), (2, 20), (3, 30)]);
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
