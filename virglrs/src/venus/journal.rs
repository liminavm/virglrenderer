// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The venus snapshot journal: what a context has to be told again to be itself.
//!
//! A context's host-side world is not a thing that can be copied — it is Vulkan objects inside a
//! driver we do not own. What *can* be copied is the guest's own words for building it, so the
//! journal keeps the wire bytes of every command that still describes live state, and a restore
//! replays them through the ordinary decoder. Guest-assigned object ids make that id-faithful:
//! the rebuilt world answers to the same names the guest still holds.
//!
//! **An entry lives exactly as long as what it is about.** Every entry names the objects it
//! describes by [`ObjectKey`], and a key to a destroyed object stops resolving on its own — so a
//! destroy prunes nothing, a cascading destroy prunes nothing, and there is no destroy site that
//! can be forgotten. This is where the C's journal spends most of its size: an explicit prune on
//! removal, a pin/unpin protocol so a blob's memory can outlive its free, and a `noted_multi_key`
//! counter for the times its "prune when ANY key dies" rule is a guess. None of that has anything
//! to do here.
//!
//! **Except in one direction, which is not liveness but reachability.** A create may name objects
//! it does not own: a pipeline names its layout and its shader modules, and the guest may destroy
//! a shader module the moment the pipeline exists. The pipeline's create still has to replay, so
//! the module's create has to survive its object. Entries therefore record what they *referenced*
//! as well as what they are about, and the export takes the closure: a retained entry drags in the
//! creates of everything it named, alive or not. The C reaches the same place by pinning at
//! create time and unpinning at destroy; taking the closure once, at export, needs no protocol and
//! cannot leak a pin.

use std::collections::{BTreeMap, BTreeSet};

use super::objects::ObjectKey;

/// Where a command sat in the stream. The VMM fences its own rebuilding against these.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Seq(pub u64);

impl Seq {
    fn advance(&mut self) -> Seq {
        self.0 += 1;
        *self
    }
}

impl std::fmt::Display for Seq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What an entry is about, which is what decides when it stops being true.
#[derive(Clone, Debug)]
enum About {
    /// It created these. It is true while any of them lives — a batch allocation whose ids died
    /// one at a time is still the command that made the survivors.
    Created(Vec<ObjectKey>),
    /// It is part of this command buffer's recording. True while the buffer lives, and dropped
    /// wholesale when the buffer is begun or reset, which is what discards a recording.
    Recording(ObjectKey),
    /// It mutated these and owns none of them — a bind, a descriptor-set update. True only while
    /// every one of them lives: a write into a descriptor set that is gone describes nothing.
    Mutated(Vec<ObjectKey>),
    /// It belongs to a ring rather than to any object, and is true until the ring is gone.
    Ring,
}

#[derive(Clone, Debug)]
struct Entry {
    seq: Seq,
    cmd_type: u32,
    /// The ring this must replay on, or 0 for the context's own decoder. The one classification
    /// the wire format still carries, because it is the only one replay acts on.
    ring_key: u64,
    wire: Vec<u8>,
    about: About,
    /// Objects the command named but does not own. Only creates need dragging in, so this holds
    /// the referenced keys and the export resolves them against the creates it knows.
    refs: Vec<ObjectKey>,
}

/// A latest-wins slot: state a later command replaces rather than adds to.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RingSlot {
    ring: u64,
    cmd_type: u32,
}

/// One context's journal.
#[derive(Default)]
pub struct Journal {
    seq: Seq,
    /// Append-ordered. Entries are never removed on destroy — they stop being exported, which is
    /// the same thing said in the one place that can see all of them at once.
    entries: Vec<Entry>,
    /// Ring state a later command supersedes: reply-stream set/seek, per ring and command.
    ring_state: BTreeMap<RingSlot, Entry>,
    /// How many commands went by without being retained. A fact about the recorder, reported by
    /// the census and never inferred from a count of what survived.
    transient: u64,
}

impl Journal {
    pub fn new() -> Journal {
        Journal::default()
    }

    /// The watermark, for the VMM's cross-layer fence.
    pub fn seq(&self) -> Seq {
        self.seq
    }

    pub fn transient(&self) -> u64 {
        self.transient
    }

    /// Note a command that built nothing durable.
    pub fn skip(&mut self) {
        self.seq.advance();
        self.transient += 1;
    }

    /// Retain a command as the record of what it created.
    pub fn created(
        &mut self,
        cmd_type: u32,
        wire: &[u8],
        keys: Vec<ObjectKey>,
        refs: Vec<ObjectKey>,
    ) {
        self.push(cmd_type, 0, wire, About::Created(keys), refs);
    }

    /// Retain a command as part of a command buffer's recording.
    ///
    /// `resets` is `vkBeginCommandBuffer` and `vkResetCommandBuffer`, which discard whatever was
    /// recorded before them — the one prune the keys cannot do, because the buffer outlives it.
    pub fn recorded(&mut self, cmd_type: u32, wire: &[u8], buffer: ObjectKey, resets: bool) {
        if resets {
            self.entries.retain(|e| !matches!(e.about, About::Recording(b) if b == buffer));
        }
        self.push(cmd_type, 0, wire, About::Recording(buffer), Vec::new());
    }

    /// Retain a command that wrote into objects it does not own.
    pub fn mutated(
        &mut self,
        cmd_type: u32,
        wire: &[u8],
        keys: Vec<ObjectKey>,
        refs: Vec<ObjectKey>,
    ) {
        self.push(cmd_type, 0, wire, About::Mutated(keys), refs);
    }

    /// Retain a ring-scoped command, replayed on that ring's own decoder.
    pub fn ring(&mut self, cmd_type: u32, wire: &[u8], ring: u64) {
        self.push(cmd_type, ring, wire, About::Ring, Vec::new());
    }

    /// Retain ring state that a later command of the same kind replaces.
    pub fn ring_latest(&mut self, cmd_type: u32, wire: &[u8], ring: u64) {
        let seq = self.seq.advance();
        let entry = Entry {
            seq,
            cmd_type,
            ring_key: ring,
            wire: wire.to_vec(),
            about: About::Ring,
            refs: Vec::new(),
        };
        self.ring_state.insert(RingSlot { ring, cmd_type }, entry);
    }

    /// Forget everything a ring owned. Called when the ring is destroyed: unlike an object, a ring
    /// has no key whose generation can answer for it.
    pub fn ring_gone(&mut self, ring: u64) {
        self.entries.retain(|e| e.ring_key != ring);
        self.ring_state.retain(|slot, _| slot.ring != ring);
    }

    fn push(
        &mut self,
        cmd_type: u32,
        ring_key: u64,
        wire: &[u8],
        about: About,
        refs: Vec<ObjectKey>,
    ) {
        let seq = self.seq.advance();
        self.entries.push(Entry { seq, cmd_type, ring_key, wire: wire.to_vec(), about, refs });
    }
}

/// Whether an entry still describes something that exists.
///
/// Taken as a closure rather than a table reference so the journal never borrows the object table:
/// the table is inside a `RefCell` the handlers hold across a dispatch, and a journal that reached
/// into it would deadlock exactly when a command is being recorded.
pub trait Live {
    fn holds(&self, key: ObjectKey) -> bool;
}

impl Journal {
    /// The entries a restore would replay, in seq order.
    ///
    /// Two passes, because reachability is not liveness: first every entry still true of a live
    /// object, then the creates those entries referenced — which may name objects that are gone,
    /// and must replay anyway or the entry that needs them cannot.
    fn retained(&self, live: &dyn Live) -> Vec<&Entry> {
        let alive = |keys: &[ObjectKey]| keys.iter().any(|k| live.holds(*k));
        let all_alive = |keys: &[ObjectKey]| keys.iter().all(|k| live.holds(*k));

        // Which entry created a given key, so a reference can be resolved to the command that
        // would rebuild it.
        let mut creator: BTreeMap<ObjectKey, usize> = BTreeMap::new();
        for (i, e) in self.entries.iter().enumerate() {
            if let About::Created(keys) = &e.about {
                for k in keys {
                    creator.insert(*k, i);
                }
            }
        }

        let mut keep: BTreeSet<usize> = BTreeSet::new();
        let mut queue: Vec<usize> = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            let true_still = match &e.about {
                About::Created(keys) => alive(keys),
                About::Recording(b) => live.holds(*b),
                About::Mutated(keys) => all_alive(keys),
                About::Ring => true,
            };
            if true_still && keep.insert(i) {
                queue.push(i);
            }
        }
        // The closure: everything a kept entry named has to be creatable, transitively.
        while let Some(i) = queue.pop() {
            for r in &self.entries[i].refs {
                if let Some(&c) = creator.get(r)
                    && keep.insert(c)
                {
                    queue.push(c);
                }
            }
        }

        let mut out: Vec<&Entry> = keep.iter().map(|i| &self.entries[*i]).collect();
        out.extend(self.ring_state.values());
        out.sort_by_key(|e| e.seq);
        out
    }
}

// ---- the wire format -------------------------------------------------------------------------
//
// 'VKJR', which the harness's pinned corpora carry as per-context prologues and which the C
// recorded. It is not ours to change: those fixtures can only be re-recorded by the C reference,
// and they are what score the venus replay path.
//
//   u32 magic 'VKJR', u32 version=1, u32 entry_count, u32 reserved
//   per entry: u64 seq, u32 cmd_type, u8 klass, u8 pad[3], u64 ring_key, u32 size, bytes (4-aligned)

const MAGIC: u32 = 0x524a_4b56;
const VERSION: u32 = 1;
/// The only klass replay acts on: this entry belongs to a ring's decoder. The C spells eight more
/// and libkrun sorted its drops by them; nothing reads those now, and a classification nobody acts
/// on is a claim waiting to be believed. Rings are told apart by `ring_key` either way.
const KLASS_RING_STREAM: u8 = 8;

impl Journal {
    /// Serialize what a restore would need. `None` when there is nothing to say.
    pub fn export(&self, live: &dyn Live) -> Option<Vec<u8>> {
        let entries = self.retained(live);
        if entries.is_empty() {
            return None;
        }
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for e in entries {
            out.extend_from_slice(&e.seq.0.to_le_bytes());
            out.extend_from_slice(&e.cmd_type.to_le_bytes());
            out.push(if e.ring_key != 0 { KLASS_RING_STREAM } else { 0 });
            out.extend_from_slice(&[0; 3]);
            out.extend_from_slice(&e.ring_key.to_le_bytes());
            out.extend_from_slice(&(e.wire.len() as u32).to_le_bytes());
            out.extend_from_slice(&e.wire);
            out.resize(out.len() + (4 - (e.wire.len() % 4)) % 4, 0);
        }
        Some(out)
    }
}

/// One entry as it comes back out of a snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    pub seq: Seq,
    pub ring_key: u64,
    pub wire: Vec<u8>,
}

struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        let end = self.at.checked_add(n).ok_or("length overflows the blob")?;
        let out = self.data.get(self.at..end).ok_or("blob ends mid-entry")?;
        self.at = end;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, &'static str> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes")))
    }
    fn u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes")))
    }
}

/// Read a journal a snapshot carried.
///
/// Total: every length is checked against what is actually there. The blob has been through a
/// file since we wrote it, and a guest's memory before that, so it is untrusted in exactly the
/// sense the tenets mean — a malformed one is refused with a reason, never a short read.
pub fn parse(data: &[u8]) -> Result<Vec<Parsed>, &'static str> {
    let mut c = Cursor { data, at: 0 };
    if c.u32()? != MAGIC {
        return Err("not a venus journal");
    }
    if c.u32()? != VERSION {
        return Err("journal version this build does not read");
    }
    let count = c.u32()?;
    let _reserved = c.u32()?;
    let mut out = Vec::new();
    for _ in 0..count {
        let seq = Seq(c.u64()?);
        let _cmd_type = c.u32()?;
        let _klass_and_pad = c.u32()?;
        let ring_key = c.u64()?;
        let size = c.u32()? as usize;
        let wire = c.take(size)?.to_vec();
        c.take((4 - (size % 4)) % 4)?;
        out.push(Parsed { seq, ring_key, wire });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Live` that answers from a set, so retention can be tested without a driver.
    struct Some_(Vec<ObjectKey>);
    impl Live for Some_ {
        fn holds(&self, key: ObjectKey) -> bool {
            self.0.contains(&key)
        }
    }

    fn keys(n: usize) -> Vec<ObjectKey> {
        let mut t = super::super::objects::Table::new();
        let mut out = Vec::new();
        for i in 1..=n {
            let id = super::super::cs::ObjectId(i as u64);
            let ty = super::super::proto::types::VkObjectType::VK_OBJECT_TYPE_BUFFER;
            t.add(id, ty, super::super::cs::HostHandle(i as u64 + 100), None).expect("add");
            out.push(t.key_of(id).expect("just added"));
        }
        out
    }

    #[test]
    fn an_entry_goes_when_what_it_describes_does() {
        let k = keys(2);
        let mut j = Journal::new();
        j.created(1, &[1, 2, 3, 4], vec![k[0]], Vec::new());
        j.created(2, &[5, 6, 7, 8], vec![k[1]], Vec::new());
        assert_eq!(j.retained(&Some_(vec![k[0], k[1]])).len(), 2);
        assert_eq!(j.retained(&Some_(vec![k[1]])).len(), 1);
        assert_eq!(j.retained(&Some_(vec![])).len(), 0);
    }

    #[test]
    fn a_create_drags_in_what_it_referenced_even_when_that_is_gone() {
        let k = keys(2);
        let mut j = Journal::new();
        // A shader module, then a pipeline naming it. The guest destroys the module.
        j.created(1, &[1; 4], vec![k[0]], Vec::new());
        j.created(2, &[2; 4], vec![k[1]], vec![k[0]]);
        let out = j.retained(&Some_(vec![k[1]]));
        assert_eq!(out.len(), 2, "the module's create has to replay for the pipeline's to");
        assert_eq!(out[0].seq, Seq(1), "and it has to replay first");
    }

    #[test]
    fn beginning_a_buffer_discards_what_it_had_recorded() {
        let k = keys(1);
        let mut j = Journal::new();
        j.recorded(10, &[1; 4], k[0], true);
        j.recorded(11, &[2; 4], k[0], false);
        j.recorded(10, &[3; 4], k[0], true);
        let out = j.retained(&Some_(vec![k[0]]));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].wire, vec![3; 4]);
    }

    #[test]
    fn ring_state_is_latest_wins_and_dies_with_its_ring() {
        let mut j = Journal::new();
        j.ring_latest(7, &[1; 4], 0xaa);
        j.ring_latest(7, &[2; 4], 0xaa);
        j.ring_latest(7, &[3; 4], 0xbb);
        assert_eq!(j.retained(&Some_(vec![])).len(), 2);
        j.ring_gone(0xaa);
        let out = j.retained(&Some_(vec![]));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].ring_key, 0xbb);
    }

    #[test]
    fn a_journal_round_trips_through_its_own_format() {
        let k = keys(1);
        let mut j = Journal::new();
        j.created(1, &[1, 2, 3, 4, 5, 6], vec![k[0]], Vec::new());
        j.ring(2, &[9; 8], 0xdead_beef);
        let blob = j.export(&Some_(vec![k[0]])).expect("something to say");
        let back = parse(&blob).expect("parses");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].wire, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(back[0].ring_key, 0);
        assert_eq!(back[1].ring_key, 0xdead_beef);
    }

    #[test]
    fn a_length_the_blob_does_not_have_is_refused() {
        let k = keys(1);
        let mut j = Journal::new();
        j.created(1, &[1, 2, 3, 4, 5, 6], vec![k[0]], Vec::new());
        let blob = j.export(&Some_(vec![k[0]])).expect("something to say");
        for n in 0..blob.len() {
            assert!(parse(&blob[..n]).is_err(), "a blob cut at {n} must be refused");
        }
        assert!(parse(&blob).is_ok());
    }

    #[test]
    fn an_empty_journal_says_nothing_rather_than_an_empty_blob() {
        let j = Journal::new();
        assert!(j.export(&Some_(vec![])).is_none());
    }
}
