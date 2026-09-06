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
