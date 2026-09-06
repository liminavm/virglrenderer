// SPDX-License-Identifier: MIT
//! A venus context's sync objects, as one blob.
//!
//! The journal rebuilds a context's fences and semaphores; this says what state they were in.
//! Without it a resumed guest waits on a fence that will never signal, because the submit that
//! would have signalled it died with the renderer.
//!
//! **What is recorded is what the guest asked for, not what the GPU had got round to.** Work still
//! in flight at a snapshot is lost whatever we observe: the journal replays object *creates* and
//! never submits, the recorded command buffers are gone, and so a rebuilt context has no way to
//! represent "this fence has a submit pending" -- there is nothing left to complete it. The one
//! self-consistent world to come back to is therefore the world as if everything the guest had
//! already submitted had completed, and that is knowable from what the guest sent, with no device
//! query that could block and no idle wait that could fail. A snapshot cannot refuse.
//!
//! The layout is the C reference's, entry for entry, so a VM suspended under one renderer can be
//! resumed under the other and one harness parser scores both: `u32` magic, `u32` count, then per
//! entry `u64` id, `u32` kind, `u32` signalled, `u64` value.

use crate::venus::cs::ObjectId;

/// 'LZYN', little-endian -- `VKR_SYNC_BLOB_MAGIC` in the C.
const MAGIC: u32 = 0x4e59_5a4c;

/// Bytes per entry: id, kind, signalled, value.
const ENTRY: usize = 24;

/// What one sync object was, in the only three shapes a restore can act on.
///
/// The numbers are the C's: it writes 0 for a fence and 2 for a timeline semaphore, and records no
/// binaries at all. 1 is this tree's addition -- a binary semaphore carries no value, but whether
/// it is *there* is what the fixed point compares, and a blob that leaves it out cannot be
/// compared against one that would have signalled it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Kind {
    Fence = 0,
    Binary = 1,
    Timeline = 2,
}

impl Kind {
    fn of(raw: u32) -> Option<Kind> {
        match raw {
            0 => Some(Kind::Fence),
            1 => Some(Kind::Binary),
            2 => Some(Kind::Timeline),
            _ => None,
        }
    }
}

/// One sync object's captured state.
///
/// `signalled` is a fence's, and a binary semaphore's, which is always signalled -- see the module
/// docs for why that is the honest answer and not an optimism. `value` is a timeline's counter and
/// zero for everything else.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Captured {
    pub id: ObjectId,
    pub kind: Kind,
    pub signalled: bool,
    pub value: u64,
}

/// Why a blob is not one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Malformed {
    /// The magic is not ours.
    NotASyncBlob,
    /// The count and the bytes behind it disagree.
    Truncated,
    /// A kind no version of this format has used.
    UnknownKind,
}

impl std::fmt::Display for Malformed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Malformed::NotASyncBlob => write!(f, "not a sync blob"),
            Malformed::Truncated => write!(f, "the entry count and the bytes disagree"),
            Malformed::UnknownKind => {
                write!(f, "a sync object of a kind this build has no name for")
            }
        }
    }
}

/// Write a capture out, in id order.
///
/// Sorted because the object arena has no order worth exporting and the fixed-point gate compares
/// two blobs entry for entry: an order that came from a hash walk would make the comparison depend
/// on where objects happened to land.
pub fn encode(mut of: Vec<Captured>) -> Vec<u8> {
    of.sort();
    let mut out = Vec::with_capacity(8 + of.len() * ENTRY);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&(of.len() as u32).to_le_bytes());
    for e in &of {
        out.extend_from_slice(&e.id.0.to_le_bytes());
        out.extend_from_slice(&(e.kind as u32).to_le_bytes());
        out.extend_from_slice(&u32::from(e.signalled).to_le_bytes());
        out.extend_from_slice(&e.value.to_le_bytes());
    }
    out
}

/// Read a capture back.
///
/// A blob out of a VMM's store is as untrusted as anything else that crosses the ABI: the count is
/// checked against the bytes rather than believed, and a kind this build cannot act on is refused
/// rather than skipped -- skipping it would restore a world missing one object's state and report
/// that it had restored everything.
pub fn decode(blob: &[u8]) -> Result<Vec<Captured>, Malformed> {
    let u32_at = |at: usize| u32::from_le_bytes(blob[at..at + 4].try_into().expect("four bytes"));
    let u64_at = |at: usize| u64::from_le_bytes(blob[at..at + 8].try_into().expect("eight bytes"));
    if blob.len() < 8 || u32_at(0) != MAGIC {
        return Err(Malformed::NotASyncBlob);
    }
    let count = u32_at(4) as usize;
    if blob.len() - 8 != count * ENTRY {
        return Err(Malformed::Truncated);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = 8 + i * ENTRY;
        out.push(Captured {
            id: ObjectId(u64_at(at)),
            kind: Kind::of(u32_at(at + 8)).ok_or(Malformed::UnknownKind)?,
            signalled: u32_at(at + 12) != 0,
            value: u64_at(at + 16),
        });
    }
    Ok(out)
}

/// What a restore did, one entry at a time.
///
/// A count per outcome and the ids behind the ones that went wrong. Never a bucket: the C sorts
/// its own drops into classes and calls the low ones benign, which is exactly how a restore with
/// holes reads like a clean one until the guest uses what is missing.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Account {
    /// Entries whose object was already in the captured state, and needed nothing.
    pub agreed: u32,
    /// Entries this restore moved into the captured state.
    pub applied: u32,
    /// Entries naming an object the rebuilt context does not have. The journal is allowed to drop
    /// a create whose object died around the snapshot, and this is that drop seen from here.
    pub dropped: Vec<ObjectId>,
    /// Entries whose object is there and could not be moved: no queue on its device, or the
    /// driver refused. Content this restore did not put back.
    pub failed: Vec<ObjectId>,
}

impl Account {
    /// Whether everything the blob named came back. What the ABI's single integer answers with.
    pub fn whole(&self) -> bool {
        self.dropped.is_empty() && self.failed.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(id: u64, kind: Kind, signalled: bool, value: u64) -> Captured {
        Captured { id: ObjectId(id), kind, signalled, value }
    }

    #[test]
    fn a_capture_decodes_back_into_what_went_into_it() {
        let of = vec![
            e(9, Kind::Timeline, false, 0x1234_5678_9abc),
            e(3, Kind::Fence, true, 0),
            e(5, Kind::Binary, true, 0),
        ];
        let blob = encode(of.clone());
        let back = decode(&blob).expect("what we wrote decodes");
        // Sorted, which is what the gate compares: same set, canonical order.
        let mut want = of;
        want.sort();
        assert_eq!(back, want);
    }

    #[test]
    fn an_empty_capture_is_still_a_blob() {
        let blob = encode(Vec::new());
        assert_eq!(blob.len(), 8);
        assert_eq!(decode(&blob), Ok(Vec::new()));
    }

    #[test]
    fn a_blob_that_is_not_one_is_refused() {
        assert_eq!(decode(&[]), Err(Malformed::NotASyncBlob));
        assert_eq!(decode(&[0; 8]), Err(Malformed::NotASyncBlob));

        let blob = encode(vec![e(3, Kind::Fence, true, 0)]);
        // A count that outruns the bytes behind it, and bytes that outrun the count: both are a
        // store that lost track of its own blob, and neither may be read as far as it goes.
        assert_eq!(decode(&blob[..blob.len() - 1]), Err(Malformed::Truncated));
        let mut extra = blob.clone();
        extra.push(0);
        assert_eq!(decode(&extra), Err(Malformed::Truncated));

        // A kind from a format this build does not have. Refused whole, not skipped: an object
        // whose state was dropped is a restore that did not happen and said it had.
        let mut wrong = blob.clone();
        wrong[8 + 8] = 7;
        assert_eq!(decode(&wrong), Err(Malformed::UnknownKind));
    }

    #[test]
    fn an_account_is_whole_only_when_nothing_was_lost() {
        let mut a = Account { agreed: 3, applied: 2, ..Default::default() };
        assert!(a.whole());
        a.dropped.push(ObjectId(7));
        assert!(!a.whole());
    }
}
