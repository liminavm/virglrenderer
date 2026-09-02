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

use super::cs::{HostHandle, Lookup, ObjectId, Objects};
use super::proto::types::VkObjectType;

/// One live object: the host handle, and the Vulkan type the guest must name it by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Object {
    /// The id the guest gave it.
    ///
    /// The arena is reached by [`Key`], but a cascade takes objects out that no command ever
    /// named, and their ids are what the rest of the context files them under. Carried on the
    /// object rather than looked back up because there is no key-to-id direction to look up:
    /// `slots` only goes the other way, and scanning it could not tell an id this cascade just
    /// orphaned from one orphaned by an earlier destroy. Written once, in [`Table::add`], beside
    /// the `slots` entry it mirrors -- the pair `Pools` keeps for the same reason.
    pub id: ObjectId,
    /// What kind of object it is. Named rather than a bare i32 because a destroy has to switch on
    /// it to pick the right `vkDestroyX`, and a match on an integer is a match nobody can check.
    pub ty: VkObjectType,
    /// The host handle, whatever Vulkan gave us for it.
    pub handle: HostHandle,
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
    index: usize,
    /// Wide on purpose. A slot is reused every time the object in it is destroyed, so this counts
    /// create/destroy pairs on one slot -- a number a guest chooses. At 32 bits a guest could cycle
    /// a slot back to a generation an orphaned id still holds and have that dead id name a live
    /// object again; the cycles cost it nothing but wire commands. At 64 there is no rate at which
    /// that finishes.
    generation: u64,
}

/// One place in the arena, occupied or not.
struct Entry {
    /// Bumped every time the slot is emptied, which is what invalidates every key to it.
    generation: u64,
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
    free: Vec<usize>,
}

impl Arena {
    fn insert(&mut self, object: Object) -> Key {
        if let Some(index) = self.free.pop() {
            let e = &mut self.entries[index];
            e.object = Some(object);
            return Key { index, generation: e.generation };
        }
        // Indexed by `usize` rather than a narrower type so there is no count to check here: a
        // guest cannot reach a length the machine could not hold the entries for anyway, and the
        // allocation is what fails if it tries.
        let index = self.entries.len();
        self.entries.push(Entry { generation: 0, object: Some(object) });
        Key { index, generation: 0 }
    }

    fn get(&self, key: Key) -> Option<&Object> {
        let e = self.entries.get(key.index)?;
        if e.generation != key.generation {
            return None;
        }
        e.object.as_ref()
    }

    fn remove(&mut self, key: Key) -> Option<Object> {
        let e = self.entries.get_mut(key.index)?;
        if e.generation != key.generation {
            return None;
        }
        let object = e.object.take()?;
        // Every key to this slot, including the one just used, now names an occupant that is gone.
        // Not a wrapping add: coming back around is the one thing that would make a stale key live
        // again, so it is left to overflow loudly rather than quietly, and 64 bits puts it out of
        // reach of anything a guest can send.
        e.generation += 1;
        self.free.push(key.index);
        Some(object)
    }

    /// Remove an object and everything created under it, however deep.
    ///
    /// Vulkan destroys a device's objects with the device and a pool's contents with the pool,
    /// naming none of them. Walking the arena is what makes that a fact about the one place
    /// objects live, rather than a child list kept beside them that a destroy path has to
    /// remember to visit -- the list that goes stale is the bug this replaces.
    fn take_tree(&mut self, key: Key) -> Vec<Doomed> {
        let Some(root) = self.remove(key) else {
            return Vec::new();
        };
        // The device an object has to be destroyed *on* is not its parent -- a fence's parent is
        // the device, but a framebuffer three levels under an instance still needs the device
        // handle from further up. So the walk carries it down: passing a device sets it for
        // everything below, and everything else inherits what it was handed.
        let under = |o: &Object, inherited| match o.ty {
            VkObjectType::VK_OBJECT_TYPE_DEVICE => Some(o.handle),
            _ => inherited,
        };
        let mut taken =
            vec![Doomed { id: root.id, ty: root.ty, handle: root.handle, device: None }];
        let mut walk = vec![(key, under(&root, None))];
        while let Some((parent, device)) = walk.pop() {
            let children: Vec<Key> = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.object.as_ref().is_some_and(|o| o.parent == Some(parent)))
                .map(|(i, e)| Key { index: i, generation: e.generation })
                .collect();
            for child in children {
                if let Some(o) = self.remove(child) {
                    walk.push((child, under(&o, device)));
                    taken.push(Doomed { id: o.id, ty: o.ty, handle: o.handle, device });
                }
            }
        }
        taken
    }

    fn remove_tree(&mut self, key: Key) -> Option<Object> {
        let object = self.get(key).copied()?;
        // Dropping the descent's result would strand host handles anywhere else; here it cannot,
        // because the only caller is `Table::remove`, which serves a leaf destroy the guest asked
        // for by name. A parent's destroy never comes this way -- it goes through
        // `Table::take_tree`, whose `#[must_use]` hands the children to the driver.
        let _ = self.take_tree(key);
        Some(object)
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

/// One object on its way out, and the device whose entry points can destroy it.
///
/// Not an [`Object`]: an object in the table knows its parent, which for anything below a device
/// is not the device itself. What a destroy needs is the `VkDevice` to call *on*, and that is
/// worked out once by the walk that takes the tree apart rather than by each caller guessing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Doomed {
    /// The name the rest of the context knows it by, for the records keyed on the guest's id
    /// rather than on a host handle -- the memory census is the one that matters.
    pub id: ObjectId,
    pub ty: VkObjectType,
    pub handle: HostHandle,
    /// `None` for the instance, its physical devices, and the devices themselves -- none of which
    /// is destroyed by a device's entry points.
    pub device: Option<HostHandle>,
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
        ty: VkObjectType,
        handle: HostHandle,
        owner: Option<ObjectId>,
    ) -> Result<(), AddError> {
        if id.0 == 0 {
            return Err(AddError::ZeroId);
        }
        if self.get(id).is_some() {
            return Err(AddError::Duplicate);
        }
        let parent = owner.and_then(|o| self.slots.get(&o)).and_then(Slot::key);
        let key = self.arena.insert(Object { id, ty, handle, parent });
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

    /// Take an object and everything under it, root first, and hand back every one.
    ///
    /// [`Table::remove`] is the same cascade with the descendants dropped, which is right for a
    /// leaf -- its handler destroyed the one host handle there was. A parent's children were never
    /// named by any command, so nobody else has destroyed them and nobody else can: they arrive
    /// here or they leak. That is why this returns them and why the result may not be discarded.
    #[must_use = "these host handles are still alive, and this is the only place that names them"]
    pub fn take_tree(&mut self, id: ObjectId) -> Vec<Doomed> {
        let Some(key) = self.slots.get(&id).and_then(Slot::key) else {
            return Vec::new();
        };
        let doomed = self.arena.take_tree(key);
        self.slots.remove(&id);
        doomed
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

    /// The host `VkDevice` an id was created under, or `None` if nothing above it is a device.
    ///
    /// The same ancestry [`Arena::take_tree`] carries down as it destroys, asked without
    /// destroying anything -- so a caller that needs to know which device an object lives on has
    /// one answer to consult rather than a map of its own to keep in step.
    pub fn device_of(&self, id: ObjectId) -> Option<HostHandle> {
        let mut at = self.slots.get(&id)?.key()?;
        loop {
            let o = self.arena.get(at)?;
            if o.ty == VkObjectType::VK_OBJECT_TYPE_DEVICE {
                return Some(o.handle);
            }
            at = o.parent?;
        }
    }

    /// The id the guest gave a host handle, which is the reply direction of every lookup here.
    ///
    /// A handful of queries hand back handles *inside* a struct the driver filled --
    /// `vkEnumeratePhysicalDeviceGroups` is the one that reaches a guest -- and those are host
    /// handles that must never leave this process. There is no shadow to put them in, because the
    /// generator can only shadow a member and these are buried in an out-struct's fixed array,
    /// so the swap is the handler's and this is what it swaps through.
    ///
    /// A scan rather than a second map, deliberately. A reverse index would be a second container
    /// holding a fact this one already holds, and every destroy path would owe it an entry it
    /// could be forgotten at -- which is the shape of bug this table was rebuilt to remove. What
    /// it costs is a walk over live objects for a query a guest asks once or twice at startup.
    ///
    /// `ty` is part of the question, not a check on the answer: handles are only unique within a
    /// type, and Vulkan is free to give a physical device and a buffer the same number.
    pub fn id_of_handle(&self, ty: VkObjectType, handle: HostHandle) -> Option<ObjectId> {
        self.arena
            .entries
            .iter()
            .filter_map(|e| e.object.as_ref())
            .find(|o| o.ty == ty && o.handle == handle)
            .map(|o| o.id)
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

    /// Empty the table, handing back every live object with the device it must be destroyed on.
    ///
    /// The teardown that actually runs. A context usually dies mid-workload with the guest still
    /// holding everything it made -- it sent no destroy for any of it and never will -- so this is
    /// not a tidy-up after the guest's own teardown but the only one there is.
    ///
    /// Taken root by root rather than slot by slot, because descending from a root is what works
    /// out which device each object belongs to, and destroying one on the wrong device is worse
    /// than leaking it.
    #[must_use = "these host handles are still alive, and this is the only place that names them"]
    pub fn take_all(&mut self) -> Vec<Doomed> {
        let roots: Vec<ObjectId> = self
            .slots
            .iter()
            .filter(|(_, slot)| {
                slot.key().and_then(|k| self.arena.get(k)).is_some_and(|o| o.parent.is_none())
            })
            .map(|(id, _)| *id)
            .collect();
        let mut doomed: Vec<Doomed> = roots.into_iter().flat_map(|id| self.take_tree(id)).collect();
        // An object whose owner was already gone when it was created has no root above it, so no
        // descent reaches it. Swept here instead, with no device to destroy it on -- which is the
        // truth about it, not an omission.
        for (_, slot) in core::mem::take(&mut self.slots) {
            if let Some(o) = slot.key().and_then(|k| self.arena.remove(k)) {
                doomed.push(Doomed { id: o.id, ty: o.ty, handle: o.handle, device: None });
            }
        }
        doomed
    }
}

impl Objects for Table {
    fn lookup(&self, id: ObjectId, ty: i32) -> Lookup {
        match self.slots.get(&id) {
            // A key the arena no longer honours is an object that died with its parent. The id is
            // as good as one the guest invented, and is answered the same way.
            Some(Slot::Live(key)) => match self.arena.get(*key) {
                // Compared as the integer the generated decode passes: the wire's number is the
                // guest's to choose, so it is not turned into a `VkObjectType` before it has been
                // matched against one that came from us.
                Some(o) if o.ty.0 == ty => Lookup::Found(o.handle),
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

    const BUFFER: VkObjectType = VkObjectType::VK_OBJECT_TYPE_BUFFER;
    const IMAGE: VkObjectType = VkObjectType::VK_OBJECT_TYPE_IMAGE;
    const DEVICE: VkObjectType = VkObjectType::VK_OBJECT_TYPE_DEVICE;

    /// The two names of one object are two types, and the table keeps both.
    ///
    /// The pair was always here -- [`ObjectId`] has said so in its own doc since it was written --
    /// but only the guest's half had a type. The host's was a bare `u64` from the table through
    /// the driver's maps to the `Handle` trait, whose `raw()` returned the guest id at some call
    /// sites and the host handle at others, and reading a slot the wrong way compiled.
    ///
    /// A handle newtype still holds one word and both readings still see it. What they cannot do
    /// is reach a sink meant for the other: `add_ghost` takes an id, `from_host` takes a handle,
    /// and the three sabotages that swap them no longer build.
    #[test]
    fn an_object_is_known_by_two_names_of_two_types() {
        use super::super::cs::Handle;
        use super::super::proto::types::VkBuffer;

        let slot = VkBuffer(0x1234);
        assert_eq!(slot.host(), HostHandle(0x1234));
        assert_eq!(slot.guest_id(), ObjectId(0x1234));
        assert_eq!(VkBuffer::from_host(HostHandle(0x1234)), slot);
        assert_eq!(VkBuffer::null(), VkBuffer(0));

        // What the table records is the host's name, under the guest's.
        let mut t = Table::new();
        t.add(ObjectId(7), BUFFER, HostHandle(0xfeed), None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle(0xfeed)));
        assert_eq!(t.id_of_handle(BUFFER, HostHandle(0xfeed)), Some(ObjectId(7)));
    }

    #[test]
    fn a_registered_object_resolves_only_under_its_own_type() {
        let mut t = Table::new();
        t.add(ObjectId(7), BUFFER, HostHandle(0xdead_beef), None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle(0xdead_beef)));
        // The whole point of the table: an id is not a capability for every object type.
        assert_eq!(t.lookup(ObjectId(7), IMAGE.0), Lookup::Missing);
    }

    #[test]
    fn an_id_the_guest_invented_stops_the_ring() {
        let t = Table::new();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Missing);
    }

    /// A ghost is the one miss that is not the guest's fault: it pipelined commands behind a create
    /// the host refused, and they were already in flight when it failed.
    #[test]
    fn a_refused_creation_becomes_a_ghost_and_then_stops_being_one() {
        let mut t = Table::new();
        t.add_ghost(ObjectId(7));
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Ghost);

        t.add(ObjectId(7), BUFFER, HostHandle(1), None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle(1)));
    }

    /// The state the two-container shape allowed and this one cannot represent.
    ///
    /// Neither corpus does this -- measured, over every context in both: no id was ever added
    /// while ghosted, and no live id was ever also a ghost. Which is exactly why it needs a test:
    /// a sequence the recordings do not contain is a sequence only the types can rule out.
    #[test]
    fn a_refused_create_never_ghosts_an_id_that_already_names_something() {
        let mut t = Table::new();
        t.add(ObjectId(7), BUFFER, HostHandle(1), None).unwrap();

        // A guest naming a live id in a create, and a driver that refuses that create. Nothing
        // changed, so the object still there is the truth.
        t.add_ghost(ObjectId(7));
        assert!(!t.is_ghost(ObjectId(7)));
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle(1)));

        // And after the guest destroys it the id names nothing at all. A ghost surviving here is
        // the whole bug: every later command naming the id would be swallowed as one lost command
        // when the ring should have stopped.
        t.remove(ObjectId(7)).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Missing);
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
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Ghost);
    }

    #[test]
    fn an_id_may_not_be_zero_or_reused_while_it_is_live() {
        let mut t = Table::new();
        assert_eq!(t.add(ObjectId(0), BUFFER, HostHandle(1), None), Err(AddError::ZeroId));
        t.add(ObjectId(7), BUFFER, HostHandle(1), None).unwrap();
        assert_eq!(t.add(ObjectId(7), IMAGE, HostHandle(2), None), Err(AddError::Duplicate));
        // Still the original: a refused insert must not have disturbed it.
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle(1)));

        t.remove(ObjectId(7)).unwrap();
        t.add(ObjectId(7), IMAGE, HostHandle(2), None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), IMAGE.0), Lookup::Found(HostHandle(2)));
    }

    /// What the generation is actually for.
    ///
    /// An id the guest destroys by name loses its entry here, so it would answer `Missing`
    /// whatever the arena did. The ids that keep their entry are the ones a *cascade* orphaned --
    /// and the slots that cascade freed are the first ones handed to the next create. Without the
    /// generation beside the index, an orphan's key would come to name whatever moved in.
    #[test]
    fn an_orphans_stale_key_does_not_come_to_name_whatever_reuses_its_slot() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle(20), Some(ObjectId(1))).unwrap();
        // Takes the child with it, and hands both their places back to be used again. Id 2 keeps
        // its entry: nothing walked back here to delete it, which is the whole trade.
        t.remove(ObjectId(1)).unwrap();

        let newcomer = t.add(ObjectId(3), IMAGE, HostHandle(30), None);
        assert_eq!(newcomer, Ok(()));
        assert_eq!(t.lookup(ObjectId(3), IMAGE.0), Lookup::Found(HostHandle(30)));
        // Same slot, same object type, different occupant. The generation is the only thing
        // separating them, and a guest naming id 2 must not be handed 30.
        assert_eq!(t.lookup(ObjectId(2), IMAGE.0), Lookup::Missing);
    }

    /// Destroying a parent destroys what hangs off it, however deep -- Vulkan does exactly this
    /// and names none of it.
    #[test]
    fn destroying_a_parent_takes_every_generation_below_it() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle(20), Some(ObjectId(1))).unwrap();
        t.add(ObjectId(3), IMAGE, HostHandle(30), Some(ObjectId(2))).unwrap();
        // A sibling tree that must survive: a cascade that took everything would pass a test that
        // only looked down one branch.
        t.add(ObjectId(4), BUFFER, HostHandle(40), None).unwrap();
        t.add(ObjectId(5), IMAGE, HostHandle(50), Some(ObjectId(4))).unwrap();

        t.remove(ObjectId(1)).unwrap();
        for id in [1, 2, 3] {
            assert!(t.get(ObjectId(id)).is_none(), "id {id} died with the root above it");
        }
        assert_eq!(t.lookup(ObjectId(4), BUFFER.0), Lookup::Found(HostHandle(40)));
        assert_eq!(t.lookup(ObjectId(5), IMAGE.0), Lookup::Found(HostHandle(50)));
        assert_eq!(t.len(), 2);
    }

    /// The half that makes the cascade affordable: an id orphaned by its parent's destroy is left
    /// where it is, and stops resolving on its own. Nothing walks back to delete it, which is why
    /// there is nothing to forget to do -- and the id is still free to name something new.
    #[test]
    fn an_id_orphaned_by_its_parents_destroy_is_reusable_without_being_cleaned_up() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle(20), Some(ObjectId(1))).unwrap();
        t.remove(ObjectId(1)).unwrap();

        assert_eq!(t.lookup(ObjectId(2), IMAGE.0), Lookup::Missing);
        // A destroy the guest pipelined behind the device's finds nothing left to destroy, and
        // must not report one -- the host handle is already gone.
        assert!(t.remove(ObjectId(2)).is_none());
        // And the id is not burned: the guest may name a new object by it.
        t.add(ObjectId(2), BUFFER, HostHandle(21), None).unwrap();
        assert_eq!(t.lookup(ObjectId(2), BUFFER.0), Lookup::Found(HostHandle(21)));
    }

    /// A slot cycled many times still does not hand a stale key back its object.
    ///
    /// The generation is a counter a guest advances for free: one create and one destroy move it
    /// on by one, and it can do that for as long as it likes. What must never happen is the
    /// counter coming back around to a value some orphaned id is still holding, which is why it is
    /// 64 bits wide rather than 32. This walks a slot far enough to show the mechanism is the
    /// counter and not luck; the width is what puts the wrap out of reach.
    #[test]
    fn a_slot_cycled_over_and_over_never_hands_an_orphan_its_object_back() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle(20), Some(ObjectId(1))).unwrap();
        t.remove(ObjectId(1)).unwrap();
        // Id 2 is now an orphan holding a key to a slot that is back in circulation.

        for round in 0..500u64 {
            t.add(ObjectId(3), IMAGE, HostHandle(1000 + round), None).unwrap();
            assert_eq!(
                t.lookup(ObjectId(2), IMAGE.0),
                Lookup::Missing,
                "the orphan resolved again on round {round}"
            );
            t.remove(ObjectId(3)).unwrap();
        }
    }

    /// An owner the guest names that no longer resolves is the guest's mistake, not a reason to
    /// refuse a create the host has already served. The object lands parentless.
    #[test]
    fn a_create_under_an_owner_that_is_already_gone_still_registers() {
        let mut t = Table::new();
        t.add(ObjectId(9), BUFFER, HostHandle(90), Some(ObjectId(1))).unwrap();
        assert_eq!(t.lookup(ObjectId(9), BUFFER.0), Lookup::Found(HostHandle(90)));
    }

    #[test]
    fn draining_hands_back_every_handle_that_needs_destroying() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle(20), None).unwrap();
        // One of them hangs off the other, so a drain that walked the tree would hand back only
        // the root. Teardown wants every live handle: Vulkan is being told about each one.
        t.add(ObjectId(3), IMAGE, HostHandle(30), Some(ObjectId(1))).unwrap();
        let mut got: Vec<_> = t.take_all().into_iter().map(|d| (d.handle, d.device)).collect();
        got.sort_unstable();
        // Every live handle, and each with the device it has to be destroyed on: 30 hangs off the
        // buffer, which is not a device, so it inherits the nothing above it.
        assert_eq!(got, [(HostHandle(10), None), (HostHandle(20), None), (HostHandle(30), None)]);
        assert!(t.is_empty());
    }

    /// Teardown reaches an object several levels below the device that owns it, and the device it
    /// names has to be the one it was made on -- destroying an image on the wrong device is worse
    /// than leaking it. The device is worked out by descending, so the depth is the test: the
    /// image here is a grandchild, and nothing above the device is a device at all.
    #[test]
    fn a_drained_object_names_the_device_it_was_made_on_however_deep_it_sits() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle(10), None).unwrap(); // stands in for the instance
        t.add(ObjectId(2), DEVICE, HostHandle(20), Some(ObjectId(1))).unwrap();
        t.add(ObjectId(3), IMAGE, HostHandle(30), Some(ObjectId(2))).unwrap();
        t.add(ObjectId(4), IMAGE, HostHandle(40), Some(ObjectId(3))).unwrap();
        let mut got: Vec<_> = t.take_all().into_iter().map(|d| (d.handle, d.device)).collect();
        got.sort_unstable();
        // The root and the device itself are destroyed as themselves, not on a device; everything
        // under the device, at any depth, carries the device's handle down with it.
        assert_eq!(
            got,
            [
                (HostHandle(10), None),
                (HostHandle(20), None),
                (HostHandle(30), Some(HostHandle(20))),
                (HostHandle(40), Some(HostHandle(20))),
            ]
        );
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
