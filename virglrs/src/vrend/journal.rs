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
