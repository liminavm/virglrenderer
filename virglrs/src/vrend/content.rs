// SPDX-License-Identifier: MIT
//! A classic context's resource contents, as one blob.
//!
//! The journal restores the *structure* of a classic context; this restores what is in it. A
//! texture the guest uploaded once at startup -- an icon atlas, a glyph cache, a corner mask --
//! has its only copy in the host GL object, which dies with the renderer: the guest will never
//! send it again, and a world rebuilt without it is structurally sound and shows grey blocks.
//! That is the same gap the venus side closes by capturing a `VkDeviceMemory`'s bytes.
//!
//! This module is the format and nothing else: no GL, no resource table, no transfers. What to
//! capture and how to read it back is [`Renderer`](crate::renderer::Renderer)'s, which is where
//! the attachment that decides "this context's resources" already lives.
//!
//! The layout is the C's, entry for entry, because the harness has one parser for both and it
//! should score both: `u32` magic, `u32` version, then per entry `u32` resource, level, width,
//! height, depth, size, followed by `size` bytes.

use super::proto::Box3;
use crate::ids::ResourceHandle;

/// 'VRCC', little-endian.
const MAGIC: u32 = 0x4343_5256;
const VERSION: u32 = 1;

/// Six dwords of header in front of every entry's bytes.
const HEADER: usize = 24;

/// What a capture or a restore did, in resources.
///
/// It is a return value and not a log line because the caller has to be able to act on it -- and
/// because the C's own comment says the opposite choice was a mistake there: its restore returns
/// success whatever happens, so a VMM counting return codes reads every lost resource as a
/// success and the warning it writes instead is the only account anyone ever gets.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Account {
    /// Levels captured, or put back.
    pub entries: u32,
    /// Levels a readback or an upload refused. Content this snapshot does not carry.
    pub skipped: u32,
    /// Resources left out on purpose: a multisample render target the compositor regenerates, a
    /// resource whose only storage is the guest's pages and is already in the VMM's RAM dump.
    /// Every context has a few, so this is not a loss and does not warn.
    pub excluded: u32,
    /// Restore only: an entry naming a resource this context cannot reach. The journal is allowed
    /// to drop a create whose resource died around the snapshot, and this is that drop's shadow.
    pub dropped: u32,
}

/// Why a blob is not one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Malformed {
    /// The magic or the version is not ours.
    NotAContentBlob,
    /// An entry runs past the end of the blob.
    Truncated,
}

impl std::fmt::Display for Malformed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Malformed::NotAContentBlob => write!(f, "not a classic content blob"),
            Malformed::Truncated => write!(f, "an entry runs past the end of the blob"),
        }
    }
}

/// A blob under construction.
pub struct Capture {
    bytes: Vec<u8>,
    account: Account,
}

impl Capture {
    pub fn new() -> Capture {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        Capture { bytes, account: Account::default() }
    }

    /// One level of one resource, at the region it was read from.
    ///
    /// The region travels with the bytes rather than beside them: a restore that recomputed the
    /// box from the resource would be trusting that the resource is still the size it was, and
    /// the whole point of this blob is that it is read back into a world that was rebuilt.
    pub fn push(&mut self, res: ResourceHandle, level: u32, region: Box3, bytes: &[u8]) {
        for v in [res.get(), level, region.width as u32, region.height as u32, region.depth as u32]
        {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.bytes.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        self.bytes.extend_from_slice(bytes);
        self.account.entries += 1;
    }

    /// A level that could not be read back. Counted, never written as zeros: bytes nobody
    /// produced would restore over content the guest may still have.
    pub fn skipped(&mut self) {
        self.account.skipped += 1;
    }

    /// A resource deliberately left out.
    pub fn excluded(&mut self) {
        self.account.excluded += 1;
    }

    pub fn account(&self) -> Account {
        self.account
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl Default for Capture {
    fn default() -> Capture {
        Capture::new()
    }
}

/// One captured level, borrowed from the blob it was parsed out of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Entry<'a> {
    pub res: ResourceHandle,
    pub level: u32,
    pub region: Box3,
    pub bytes: &'a [u8],
}

/// Split a blob into its entries.
///
/// A blob from a VMM's store is as untrusted as anything else that crosses the ABI: every length
/// is checked against what is left rather than believed.
pub fn entries(blob: &[u8]) -> Result<Vec<Entry<'_>>, Malformed> {
    let u32_at = |at: usize| -> u32 {
        u32::from_le_bytes([blob[at], blob[at + 1], blob[at + 2], blob[at + 3]])
    };
    if blob.len() < 8 || u32_at(0) != MAGIC || u32_at(4) != VERSION {
        return Err(Malformed::NotAContentBlob);
    }
    let mut out = Vec::new();
    let mut at = 8;
    while at < blob.len() {
        if blob.len() - at < HEADER {
            return Err(Malformed::Truncated);
        }
        let h: Vec<u32> = (0..6).map(|i| u32_at(at + i * 4)).collect();
        at += HEADER;
        let size = h[5] as usize;
        if blob.len() - at < size {
            return Err(Malformed::Truncated);
        }
        let Some(res) = ResourceHandle::new(h[0]) else {
            return Err(Malformed::NotAContentBlob);
        };
        out.push(Entry {
            res,
            level: h[1],
            region: Box3 {
                x: 0,
                y: 0,
                z: 0,
                width: h[2] as i32,
                height: h[3] as i32,
                depth: h[4] as i32,
            },
            bytes: &blob[at..at + size],
        });
        at += size;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(n: u32) -> ResourceHandle {
        ResourceHandle::new(n).expect("a resource handle is not zero")
    }

    fn region(w: i32, h: i32, d: i32) -> Box3 {
        Box3 { x: 0, y: 0, z: 0, width: w, height: h, depth: d }
    }

    #[test]
    fn a_capture_parses_back_into_what_went_into_it() {
        let mut c = Capture::new();
        c.push(res(7), 0, region(4, 2, 1), &[1, 2, 3, 4, 5, 6, 7, 8]);
        c.push(res(7), 1, region(2, 1, 1), &[9, 10]);
        c.push(res(9), 0, region(1, 1, 1), &[]);
        c.skipped();
        c.excluded();
        assert_eq!(c.account(), Account { entries: 3, skipped: 1, excluded: 1, dropped: 0 });

        let blob = c.into_bytes();
        let e = entries(&blob).expect("what we wrote parses");
        assert_eq!(e.len(), 3);
        assert_eq!(
            e[0],
            Entry {
                res: res(7),
                level: 0,
                region: region(4, 2, 1),
                bytes: &[1, 2, 3, 4, 5, 6, 7, 8]
            }
        );
        assert_eq!(e[1].level, 1);
        assert_eq!(e[1].bytes, &[9, 10]);
        // A level with no bytes is still a level: it says the resource was reached and held
        // nothing, which is not what a missing entry says.
        assert_eq!(e[2].res, res(9));
        assert!(e[2].bytes.is_empty());
    }

    /// An empty world is a valid blob with no entries -- distinct from no blob at all, which is
    /// what a context this renderer does not serve answers.
    #[test]
    fn an_empty_capture_is_still_a_blob() {
        let blob = Capture::new().into_bytes();
        assert_eq!(blob.len(), 8);
        assert_eq!(entries(&blob).expect("an empty blob is a blob"), vec![]);
    }

    #[test]
    fn a_blob_that_is_not_one_is_refused() {
        assert_eq!(entries(&[]), Err(Malformed::NotAContentBlob));
        assert_eq!(entries(&[0; 8]), Err(Malformed::NotAContentBlob));
        // Ours, but a version nothing here wrote.
        let mut wrong = MAGIC.to_le_bytes().to_vec();
        wrong.extend_from_slice(&2u32.to_le_bytes());
        assert_eq!(entries(&wrong), Err(Malformed::NotAContentBlob));

        let mut c = Capture::new();
        c.push(res(7), 0, region(4, 1, 1), &[1, 2, 3, 4]);
        let blob = c.into_bytes();
        // A length that outruns the bytes behind it, which is the shape a truncated store has.
        assert_eq!(entries(&blob[..blob.len() - 1]), Err(Malformed::Truncated));
        assert_eq!(entries(&blob[..HEADER]), Err(Malformed::Truncated));
    }
}
