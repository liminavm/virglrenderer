//! What a context would have to be told again to be rebuilt.
//!
//! A snapshot restores a guest's memory but not the renderer's: the GL objects a compositor built
//! over its lifetime live only here, and classic virgl reports nothing to the guest, so a context
//! that comes back empty is invisible from inside the VM -- the clients render into the void and
//! every process-survival signal stays green. The journal is what makes the world rebuildable.
//!
//! **The wire is retained, not re-derived.** The obvious alternative is to walk the decoded state
//! at export time and emit the commands that would rebuild it. That dies on shaders:
//! [`ShaderText::Whole`](super::context::ShaderText) holds parsed IR and the source text is gone
//! the moment it completes, so the one object class that dominates the bill of materials cannot be
//! re-serialized at all. Retaining the dwords the guest sent is the only thing that works for
//! every class, so it is what every class does -- one mechanism, not two.
//!
//! **The record belongs to the thing it created.** The C keeps a per-context log keyed by handle
//! and prunes it at each destroy site. That is a second table holding one fact, and it fails the
//! way those always fail: the destroy path nobody remembered leaves an entry naming an object that
//! is gone. Here a create's dwords live *inside* the object's own entry, so `DESTROY_OBJECT` drops
//! the record with the object and there is no pruning to forget. The same applies one level up:
//! a sub-context's latest-wins state goes with the sub-context, and a blob-defining resource
//! create goes with the resource.

use super::pipe::ShaderStage;
use super::proto::{Cmd, Command};

/// A position in one context's journal.
///
/// Deliberately three things at once, because they are the same thing: the order a rebuild must
/// replay in, the guarantee that an object's create precedes every command naming it, and the
/// watermark the VMM fences its own blob creates against. Splitting them would be three counters
/// that have to agree.
///
/// Starts at one so that zero reads as "nothing retained yet", which is what a context with no
/// journal answers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Seq(pub u64);

impl Seq {
    /// The next position, for the command being retained now.
    pub fn advance(&mut self) -> Seq {
        self.0 += 1;
        *self
    }
}

impl std::fmt::Display for Seq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The commands that built one thing, as the guest sent them.
///
/// `chunks` is a list because a create is not always one command: a shader's text arrives split
/// across as many `CREATE_OBJECT`s as it takes, and a rebuild needs all of them, in order. Every
/// other class has exactly one.
///
/// The seq is the *first* chunk's, so that an object created early and completed late still sorts
/// before whatever was bound against it in between.
pub struct Retained {
    pub seq: Seq,
    pub chunks: Vec<Vec<u32>>,
}

impl Retained {
    /// Retain a command, at the position it was accepted.
    pub fn new(seq: Seq, wire: &[u32]) -> Retained {
        Retained { seq, chunks: vec![wire.to_vec()] }
    }

    /// Retain a continuation of the same create -- another chunk of a shader's text.
    pub fn extend(&mut self, wire: &[u32]) {
        self.chunks.push(wire.to_vec());
    }

    /// Dwords retained, for the census.
    pub fn dwords(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }
}

/// Which slot of a sub-context's current state a command sets.
///
/// Creates alone rebuild nothing usable. Gallium re-emits a bind or a set only when it *changes*,
/// so a client that bound its shader once and has drawn ever since never sends that bind again --
/// replaying only the creates would leave a world of objects with nothing bound. What has to be
/// kept is the last command per slot, and the slot is what stops two commands of the same kind
/// overwriting each other: a stage, a start slot, an index, an object type.
///
/// Taken from the decoded command rather than from dword offsets, so the discriminator is the
/// field the protocol names -- not a position that has to be kept in step with the encoder by
/// hand.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct StateKey {
    cmd: Cmd,
    slot: (u32, u32),
}

/// The state slot this command sets, or `None` if it leaves nothing that has to be rebuilt.
///
/// `None` covers three kinds of command, and they are all deliberate. The transient ones --
/// draws, clears, blits, barriers, queries, transfers -- describe work, not state, and the client
/// re-issues them every frame. The structural ones -- creates, destroys, sub-context lifecycle --
/// are retained somewhere that owns them instead. And `LINK_SHADER` is neither: it assembles a
/// program early that a draw would otherwise assemble on demand, restoring the previous binds
/// when it is done, so a rebuild that omits it is slower for one frame and identical after. The C
/// design retains it; there is nothing in it to retain.
pub fn state_key(cmd: &Command<'_>) -> Option<StateKey> {
    let stage_slot = |s: &ShaderStage, n: &u32| (s.index() as u32, *n);
    let (cmd, slot) = match cmd {
        // Per object type: one bind of each kind is current at a time.
        Command::BindObject { kind, .. } => (Cmd::BindObject, (kind.wire(), 0)),
        // Per stage.
        Command::BindShader { stage, .. } => (Cmd::BindShader, (stage.index() as u32, 0)),
        // Per stage and the first slot the set writes.
        Command::BindSamplerStates { stage, start_slot, .. } => {
            (Cmd::BindSamplerStates, stage_slot(stage, start_slot))
        }
        Command::SetSamplerViews { stage, start_slot, .. } => {
            (Cmd::SetSamplerViews, stage_slot(stage, start_slot))
        }
        Command::SetConstantBuffer { stage, index, .. } => {
            (Cmd::SetConstantBuffer, stage_slot(stage, index))
        }
        Command::SetUniformBuffer { stage, index, .. } => {
            (Cmd::SetUniformBuffer, stage_slot(stage, index))
        }
        Command::SetShaderBuffers { stage, start_slot, .. } => {
            (Cmd::SetShaderBuffers, stage_slot(stage, start_slot))
        }
        Command::SetShaderImages { stage, start_slot, .. } => {
            (Cmd::SetShaderImages, stage_slot(stage, start_slot))
        }
        // Per first slot.
        Command::SetAtomicBuffers { start_slot, .. } => (Cmd::SetAtomicBuffers, (*start_slot, 0)),
        Command::SetViewportState { start_slot, .. } => (Cmd::SetViewportState, (*start_slot, 0)),
        Command::SetScissorState { start_slot, .. } => (Cmd::SetScissorState, (*start_slot, 0)),
        // Per tweak.
        Command::SetTweaks { id, .. } => (Cmd::SetTweaks, (*id, 0)),
        // One of each per sub-context.
        Command::SetFramebufferState { .. } => (Cmd::SetFramebufferState, (0, 0)),
        Command::SetFramebufferStateNoAttach { .. } => (Cmd::SetFramebufferStateNoAttach, (0, 0)),
        Command::SetVertexBuffers(_) => (Cmd::SetVertexBuffers, (0, 0)),
        Command::SetIndexBuffer(_) => (Cmd::SetIndexBuffer, (0, 0)),
        Command::SetStencilRef { .. } => (Cmd::SetStencilRef, (0, 0)),
        Command::SetBlendColor(_) => (Cmd::SetBlendColor, (0, 0)),
        Command::SetClipState(_) => (Cmd::SetClipState, (0, 0)),
        Command::SetSampleMask(_) => (Cmd::SetSampleMask, (0, 0)),
        Command::SetMinSamples(_) => (Cmd::SetMinSamples, (0, 0)),
        Command::SetPolygonStipple(_) => (Cmd::SetPolygonStipple, (0, 0)),
        Command::SetStreamoutTargets { .. } => (Cmd::SetStreamoutTargets, (0, 0)),
        Command::SetTessState(_) => (Cmd::SetTessState, (0, 0)),
        Command::SetRenderCondition { .. } => (Cmd::SetRenderCondition, (0, 0)),
        _ => return None,
    };
    Some(StateKey { cmd, slot })
}

/// What one context has retained: creates, current-state slots, and the dwords behind them.
///
/// The point of counting is that the answer must be bounded by the world that is *live*, not by
/// how long the guest has been running: destroys prune by dropping, and state overwrites in
/// place, so a desktop left alone for a day must not have a larger journal than one just seated.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Census {
    pub creates: usize,
    pub slots: usize,
    pub dwords: usize,
}

impl Census {
    pub fn add(&mut self, at: &Retained, create: bool) {
        if create {
            self.creates += 1;
        } else {
            self.slots += 1;
        }
        self.dwords += at.dwords();
    }
}

impl std::ops::AddAssign for Census {
    fn add_assign(&mut self, o: Census) {
        self.creates += o.creates;
        self.slots += o.slots;
        self.dwords += o.dwords;
    }
}

impl std::fmt::Display for Census {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} creates + {} state slots, {} KiB of wire",
            self.creates,
            self.slots,
            self.dwords * 4 / 1024
        )
    }
}

/// One thing a rebuild has to do, and where it sits in the order.
///
/// The sub-context is data here rather than a `SET_SUB_CTX` command in the stream because both
/// ends of this journal are this renderer: nothing outside reads it, so the switch is applied at
/// replay from the field instead of being encoded, decoded and re-applied.
pub enum Step<'a> {
    /// Make this sub-context. Sub-context 0 is never emitted -- a fresh context has it.
    CreateSub(u32),
    /// Feed these dwords with that sub-context current.
    Feed { sub: u32, chunks: &'a [Vec<u32>] },
}

/// One entry of a context's export, before it is written out.
pub struct Entry<'a> {
    pub seq: Seq,
    pub step: Step<'a>,
}

/// Order a context's retained commands into the one sequence that rebuilds it.
///
/// Sorting is global across sub-contexts, not grouped by them. Grouping reads more naturally and
/// is what the C's design proposes, but it reorders commands against the seq the VMM fences its
/// own control-queue replay on, so a blob create waiting on wire position N can be fed entries
/// from a different sub-context that were never what it was waiting for. One order, and it is the
/// order the commands were accepted in -- which is also what makes a create always precede the
/// binds that name it.
pub fn order<'a>(entries: impl Iterator<Item = Entry<'a>>) -> Vec<Entry<'a>> {
    let mut all: Vec<Entry<'a>> = entries.collect();
    all.sort_by_key(|e| e.seq);
    all
}

/// Magic for a classic journal export: "VRJ1", little-endian.
///
/// Its own format rather than the venus journal's `VKJR`, because nothing outside this renderer
/// reads it. `VKJR` carries a `klass` taxonomy and a `ring_key` that exist so the VMM can route
/// entries it has parsed; the VMM no longer parses this one, so neither field has anything to
/// say, and inventing values for them would be describing the C's design rather than ours.
const MAGIC: u32 = 0x314a_5256;
const VERSION: u32 = 1;

/// Write a context's ordered journal as the bytes the VMM will hand back.
pub fn serialize(entries: &[Entry<'_>]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut w = |v: u32| out.extend_from_slice(&v.to_le_bytes());
    w(MAGIC);
    w(VERSION);
    w(entries.len() as u32);
    w(0);
    for e in entries {
        out.extend_from_slice(&e.seq.0.to_le_bytes());
        match &e.step {
            Step::CreateSub(id) => {
                out.extend_from_slice(&0u32.to_le_bytes());
                out.extend_from_slice(&id.to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            Step::Feed { sub, chunks } => {
                out.extend_from_slice(&1u32.to_le_bytes());
                out.extend_from_slice(&sub.to_le_bytes());
                out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
                for c in chunks.iter() {
                    out.extend_from_slice(&(c.len() as u32).to_le_bytes());
                    for d in c {
                        out.extend_from_slice(&d.to_le_bytes());
                    }
                }
            }
        }
    }
    out
}

/// A parsed entry, owning its dwords: the bytes it came from are the VMM's and do not outlive the
/// call that handed them over.
pub struct Parsed {
    pub seq: Seq,
    pub sub: u32,
    /// Empty for "create this sub-context", which has nothing to feed.
    pub chunks: Vec<Vec<u32>>,
}

/// Read back what [`serialize`] wrote.
///
/// Every length is checked against what is actually there. The blob has been round-tripped
/// through a snapshot file and a VMM, so however much this renderer wrote it, by the time it
/// comes back it is input -- and input is not trusted just because we recognise the magic.
pub fn parse(bytes: &[u8]) -> Result<Vec<Parsed>, &'static str> {
    let mut at = 0usize;
    let u32_at = |at: &mut usize| -> Result<u32, &'static str> {
        let end = at.checked_add(4).ok_or("truncated")?;
        let b = bytes.get(*at..end).ok_or("truncated")?;
        *at = end;
        Ok(u32::from_le_bytes(b.try_into().expect("four bytes")))
    };
    if u32_at(&mut at)? != MAGIC {
        return Err("not a vrend journal");
    }
    if u32_at(&mut at)? != VERSION {
        return Err("a vrend journal from another version");
    }
    let count = u32_at(&mut at)? as usize;
    let _reserved = u32_at(&mut at)?;
    let mut out = Vec::with_capacity(count.min(1 << 16));
    for _ in 0..count {
        let lo = u32_at(&mut at)? as u64;
        let hi = u32_at(&mut at)? as u64;
        let seq = Seq(lo | (hi << 32));
        let kind = u32_at(&mut at)?;
        let sub = u32_at(&mut at)?;
        let n = u32_at(&mut at)? as usize;
        if kind > 1 {
            return Err("an entry of no known kind");
        }
        let mut chunks = Vec::with_capacity(n.min(1 << 12));
        for _ in 0..n {
            let len = u32_at(&mut at)? as usize;
            let mut c = Vec::with_capacity(len.min(1 << 16));
            for _ in 0..len {
                c.push(u32_at(&mut at)?);
            }
            chunks.push(c);
        }
        out.push(Parsed { seq, sub, chunks });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(seq: u64, sub: u32, chunks: &[Vec<u32>]) -> Entry<'_> {
        Entry { seq: Seq(seq), step: Step::Feed { sub, chunks } }
    }

    #[test]
    fn a_journal_round_trips() {
        let one = vec![vec![0x1234_5678, 9]];
        let long = vec![vec![1, 2, 3], vec![4], vec![]];
        let entries = vec![
            Entry { seq: Seq(1), step: Step::CreateSub(3) },
            feed(2, 0, &one),
            feed(7, 3, &long),
        ];
        let back = parse(&serialize(&entries)).expect("what we just wrote");
        assert_eq!(back.len(), 3);
        assert_eq!(back[0].seq, Seq(1));
        assert_eq!(back[0].sub, 3);
        assert!(back[0].chunks.is_empty(), "a sub-context create feeds nothing");
        assert_eq!(back[1].chunks, one);
        assert_eq!(back[2].seq, Seq(7));
        assert_eq!(back[2].sub, 3);
        assert_eq!(back[2].chunks, long, "a shader's chunks keep their order and their count");
    }

    #[test]
    fn a_seq_past_four_billion_survives() {
        // The counter is 64-bit and a long-lived context will pass 2^32; a journal that wrapped
        // there would replay creates after the binds that name them.
        let entries = vec![Entry { seq: Seq(0x1_0000_0007), step: Step::CreateSub(1) }];
        assert_eq!(parse(&serialize(&entries)).unwrap()[0].seq, Seq(0x1_0000_0007));
    }

    #[test]
    fn nothing_retained_is_still_a_journal() {
        let back = parse(&serialize(&[])).expect("an empty journal is a journal");
        assert!(back.is_empty());
    }

    #[test]
    fn a_blob_that_is_not_ours_is_refused() {
        assert!(parse(&[]).is_err(), "no header at all");
        assert!(parse(&[0; 16]).is_err(), "the wrong magic");
        let mut wrong_version = serialize(&[]);
        wrong_version[4] = 2;
        assert!(parse(&wrong_version).is_err());
    }

    #[test]
    fn a_length_the_blob_does_not_have_is_refused() {
        // The count says one entry and the bytes stop short: this has been through a snapshot
        // file and a VMM, so it is input, and input is not trusted for recognising the magic.
        let chunks = vec![vec![1, 2, 3]];
        let full = serialize(&[feed(1, 0, &chunks)]);
        for cut in 1..full.len() {
            assert!(parse(&full[..cut]).is_err(), "truncated at {cut} was accepted");
        }
    }

    #[test]
    fn the_order_is_by_seq_across_sub_contexts() {
        // Grouping per sub-context would put 9 before 4 here; the fence the VMM feeds against is
        // a seq, so the order has to be global.
        let a = vec![vec![0xaa]];
        let b = vec![vec![0xbb]];
        let c = vec![vec![0xcc]];
        let got = order(vec![feed(9, 1, &a), feed(4, 2, &b), feed(6, 1, &c)].into_iter());
        let seqs: Vec<u64> = got.iter().map(|e| e.seq.0).collect();
        assert_eq!(seqs, vec![4, 6, 9]);
    }
}
