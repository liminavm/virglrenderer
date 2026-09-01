// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The ring: where the guest writes commands and the host reads them.
//!
//! This module is the boundary that decides whether a guest's description of a ring is one we are
//! willing to touch, and it is deliberately pure -- no mapping, no pointers, no unsafe. Every
//! number here arrives from a guest that may be lying, and all of them are checked before anything
//! maps a byte.
//!
//! A ring lives inside one shm resource. The guest names five regions within it -- three 32-bit
//! control words and two byte ranges -- and the host has to be sure they are inside the resource,
//! aligned, and do not overlap, because the whole point of the control words is that host and
//! guest write different ones. A `buffer` that overlapped `head` would let a guest's command
//! stream rewrite the host's own progress counter.

use super::proto::types::VkRingCreateInfoMESA;

/// The largest ring buffer we will accept, from the C's `VKR_RING_BUFFER_MAX_SIZE`.
///
/// Two reasons, both still true here: commands are copied out of the ring before they are decoded,
/// so the buffer bounds a host-side copy, and the head and tail are 32-bit, so a ring at or above
/// `u32::MAX` could not be indexed by them anyway.
pub const RING_BUFFER_MAX_SIZE: usize = 16 * 1024 * 1024;

/// Which part of the layout a complaint is about. Named rather than described, so a rejection says
/// what the guest got wrong and a test can assert on it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Part {
    /// The window within the resource that the other five must live inside.
    Whole,
    Head,
    Tail,
    Status,
    Buffer,
    Extra,
}

/// Why a guest's ring layout was refused. Every one of these is the guest's mistake, so a ring
/// that fails to parse poisons its context and never reaches a mapping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayoutError {
    /// An offset plus its size does not fit in a `usize`. The C spells this `vkr_region_is_valid`
    /// and reaches it only because C's `+` wraps; here the addition simply cannot be done, which
    /// is why there is no "invalid region" state to check for afterwards.
    Overflow(Part),
    /// A region that is not inside the window the guest itself declared.
    OutOfBounds(Part),
    /// A region whose begin or end is not 32-bit aligned. The control words are read atomically,
    /// and an unaligned atomic is not one.
    Misaligned(Part),
    /// Two regions sharing a byte. Which two is worth saying: the pair is the bug.
    Overlap(Part, Part),
    /// A buffer that is not a non-zero power of two, or is larger than [`RING_BUFFER_MAX_SIZE`].
    /// The size doubles as the mask for the free-running offset, which is what makes it a power
    /// of two rather than merely a tidy number.
    BufferSize(usize),
}

/// A half-open byte range, `begin..end`.
///
/// There is no invalid `Region`: [`Region::new`] is the only way to make one and it refuses the
/// overflow that would produce `end < begin`. That is the difference from the C's
/// `struct vkr_region`, which is constructed by a macro that can overflow and then asks
/// `vkr_region_is_valid` afterwards -- a check every new caller has to remember.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Region {
    begin: usize,
    end: usize,
}

impl Region {
    /// A region of `size` bytes at `begin`, or `None` if that range does not fit in a `usize`.
    pub fn new(begin: usize, size: usize) -> Option<Region> {
        Some(Region { begin, end: begin.checked_add(size)? })
    }

    pub fn begin(&self) -> usize {
        self.begin
    }

    pub fn size(&self) -> usize {
        self.end - self.begin
    }

    /// Whether both ends sit on an `align`-byte boundary.
    pub fn is_aligned(&self, align: usize) -> bool {
        debug_assert!(align.is_power_of_two(), "alignment must be a non-zero power of two");
        (self.begin | self.end) & (align - 1) == 0
    }

    /// Whether the two share no byte. An empty region is disjoint from everything, including one
    /// it sits inside -- which is deliberate, because `extra` is allowed to be empty.
    pub fn is_disjoint(&self, other: &Region) -> bool {
        self.begin >= other.end || self.end <= other.begin
    }

    /// Whether this region lies entirely within `other`.
    pub fn is_within(&self, other: &Region) -> bool {
        self.begin >= other.begin && self.end <= other.end
    }
}

/// A guest's ring layout, once it has been found habitable.
///
/// Holding this is the evidence that the checks in [`RingLayout::parse`] passed. Nothing
/// constructs one otherwise, so a later stage cannot be handed an unvalidated layout by mistake.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RingLayout {
    /// The window inside the resource that the guest gave the ring. Kept because every offset
    /// below is absolute within the resource, and the mapping needs to know what it may touch.
    pub whole: Region,
    /// Written by the host as commands are consumed, read by the guest.
    pub head: Region,
    /// Written by the guest as commands are added, read by the host.
    pub tail: Region,
    /// Written by the host to report a status change to the guest.
    pub status: Region,
    /// The command stream itself.
    pub buffer: Region,
    /// Guest words the host may be asked to write, addressed by offset.
    pub extra: Region,
}

impl RingLayout {
    /// Check a guest's description of a ring against the resource it claims to live in.
    ///
    /// `resource_size` is the size of the shm resource the guest named, which the caller has
    /// already resolved -- this function never looks anything up, so what it decides depends only
    /// on its arguments and can be tested exhaustively.
    pub fn parse(
        resource_size: usize,
        info: &VkRingCreateInfoMESA,
    ) -> Result<RingLayout, LayoutError> {
        let region = |part: Part, offset: usize, size: usize| {
            // The five regions are placed relative to the window, so their offsets are added to
            // it. An overflow here is the guest naming an offset near the top of the address
            // space, not a resource that is genuinely that large.
            let begin = info.offset.checked_add(offset).ok_or(LayoutError::Overflow(part))?;
            Region::new(begin, size).ok_or(LayoutError::Overflow(part))
        };

        let whole =
            Region::new(info.offset, info.size).ok_or(LayoutError::Overflow(Part::Whole))?;

        // The resource is the only bound the guest did not choose, so it is checked first: every
        // other check below is against a window that the guest supplied, and a window outside the
        // resource would make all of them meaningless.
        let resource = Region::new(0, resource_size).ok_or(LayoutError::Overflow(Part::Whole))?;
        if !whole.is_within(&resource) {
            return Err(LayoutError::OutOfBounds(Part::Whole));
        }

        const WORD: usize = size_of::<u32>();
        let parts = [
            (Part::Head, region(Part::Head, info.headOffset, WORD)?),
            (Part::Tail, region(Part::Tail, info.tailOffset, WORD)?),
            (Part::Status, region(Part::Status, info.statusOffset, WORD)?),
            (Part::Buffer, region(Part::Buffer, info.bufferOffset, info.bufferSize)?),
            (Part::Extra, region(Part::Extra, info.extraOffset, info.extraSize)?),
        ];

        for (part, r) in &parts {
            if !r.is_within(&whole) {
                return Err(LayoutError::OutOfBounds(*part));
            }
            if !r.is_aligned(WORD) {
                return Err(LayoutError::Misaligned(*part));
            }
        }

        // Every pair, not every neighbour: the guest chose all five offsets independently, so
        // there is no ordering among them to exploit.
        for (i, (part, r)) in parts.iter().enumerate() {
            for (other_part, other) in &parts[i + 1..] {
                if !r.is_disjoint(other) {
                    return Err(LayoutError::Overlap(*part, *other_part));
                }
            }
        }

        let buffer = parts[3].1;
        let size = buffer.size();
        if size == 0 || !size.is_power_of_two() || size > RING_BUFFER_MAX_SIZE {
            return Err(LayoutError::BufferSize(size));
        }

        Ok(RingLayout {
            whole,
            head: parts[0].1,
            tail: parts[1].1,
            status: parts[2].1,
            buffer,
            extra: parts[4].1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout every case below starts from and breaks in exactly one way. The offsets are
    /// deliberately not in a tidy ascending run, because a guest's need not be.
    fn good() -> VkRingCreateInfoMESA {
        VkRingCreateInfoMESA {
            offset: 0x1000,
            size: 0x2000,
            headOffset: 0,
            tailOffset: 4,
            statusOffset: 8,
            bufferOffset: 0x100,
            bufferSize: 0x1000,
            extraOffset: 0x1100,
            extraSize: 0x40,
            ..Default::default()
        }
    }

    const RES: usize = 0x4000;
    const BIG_RES: usize = 0x8000_0000;

    /// One way to misdescribe a ring: what it is, the resource it is described against, the break
    /// applied to a good layout, and the refusal that break must draw.
    type Row = (&'static str, usize, fn(&mut VkRingCreateInfoMESA), LayoutError);

    /// The same shape as `good`, sized for a buffer too large to sit in `RES`. The control words
    /// and `extra` go below the buffer so that growing it cannot collide with them -- otherwise a
    /// size case would be refused for overlapping and never reach the bound being tested.
    fn big(buffer_size: usize) -> VkRingCreateInfoMESA {
        VkRingCreateInfoMESA {
            offset: 0x1000,
            size: 0x1000 + buffer_size,
            headOffset: 0,
            tailOffset: 4,
            statusOffset: 8,
            extraOffset: 0x10,
            extraSize: 0x40,
            bufferOffset: 0x1000,
            bufferSize: buffer_size,
            ..Default::default()
        }
    }

    #[test]
    fn a_layout_that_fits_is_taken_at_its_word() {
        let l = RingLayout::parse(RES, &good()).expect("this layout is habitable");
        assert_eq!(l.head.begin(), 0x1000, "offsets are absolute within the resource");
        assert_eq!(l.buffer.begin(), 0x1100);
        assert_eq!(l.buffer.size(), 0x1000);
        assert_eq!(l.extra.size(), 0x40);
        assert_eq!(l.whole.size(), 0x2000);
    }

    /// Each row breaks the good layout in one way and names the refusal it must draw. A row that
    /// stopped being refused would be a hole in the boundary, which is the only thing standing
    /// between a hostile guest and a mapping.
    #[test]
    fn every_way_a_guest_can_misdescribe_a_ring_is_refused() {
        let cases: [Row; 12] = [
            (
                "a window past the end of the resource",
                RES,
                |i| i.size = 0x4000,
                LayoutError::OutOfBounds(Part::Whole),
            ),
            (
                "a window whose own extent overflows",
                RES,
                |i| i.size = usize::MAX,
                LayoutError::Overflow(Part::Whole),
            ),
            (
                "a control word past the end of the window",
                RES,
                |i| i.headOffset = 0x2000,
                LayoutError::OutOfBounds(Part::Head),
            ),
            (
                "a control word whose offset overflows",
                RES,
                |i| i.tailOffset = usize::MAX,
                LayoutError::Overflow(Part::Tail),
            ),
            (
                "a buffer running past the end of the window",
                RES,
                |i| i.bufferSize = 0x4000,
                LayoutError::OutOfBounds(Part::Buffer),
            ),
            (
                "an unaligned control word",
                RES,
                |i| i.statusOffset = 9,
                LayoutError::Misaligned(Part::Status),
            ),
            (
                "an unaligned buffer",
                RES,
                |i| i.bufferOffset = 0x102,
                LayoutError::Misaligned(Part::Buffer),
            ),
            (
                "the head and the tail on the same word",
                RES,
                |i| i.tailOffset = 0,
                LayoutError::Overlap(Part::Head, Part::Tail),
            ),
            (
                "a buffer swallowing the host's own progress counter",
                RES,
                |i| {
                    i.bufferOffset = 0;
                    i.bufferSize = 0x1000;
                },
                LayoutError::Overlap(Part::Head, Part::Buffer),
            ),
            (
                "a buffer that is not a power of two",
                RES,
                |i| i.bufferSize = 0x900,
                LayoutError::BufferSize(0x900),
            ),
            ("an empty buffer", RES, |i| i.bufferSize = 0, LayoutError::BufferSize(0)),
            (
                "a buffer larger than we will copy out of",
                BIG_RES,
                |i| *i = big(RING_BUFFER_MAX_SIZE * 2),
                LayoutError::BufferSize(RING_BUFFER_MAX_SIZE * 2),
            ),
        ];

        for (what, res, break_it, want) in cases {
            let mut info = good();
            break_it(&mut info);
            let got = RingLayout::parse(res, &info);
            assert_eq!(got, Err(want), "{what}");
        }
    }

    /// `extra` is the one region a guest may legitimately leave empty, and an empty region is
    /// disjoint from everything -- including whatever it sits on top of. Checked because the
    /// overlap loop would otherwise be free to reject it, and the C explicitly does not.
    #[test]
    fn an_empty_extra_region_is_allowed_to_sit_anywhere() {
        let mut info = good();
        info.extraSize = 0;
        info.extraOffset = 0x100; // exactly on top of the buffer, were it not empty
        RingLayout::parse(RES, &info).expect("an empty region overlaps nothing");
    }

    /// The buffer bound is a real refusal, not a clamp: a guest asking for more must be told no,
    /// because silently serving a smaller ring than it believes it has is how a guest ends up
    /// writing past what the host will read.
    #[test]
    fn the_largest_buffer_we_accept_is_accepted_and_the_next_one_up_is_not() {
        let info = big(RING_BUFFER_MAX_SIZE);
        let l = RingLayout::parse(BIG_RES, &info).expect("the bound itself is allowed");
        assert_eq!(l.buffer.size(), RING_BUFFER_MAX_SIZE);

        let over = big(RING_BUFFER_MAX_SIZE * 2);
        assert!(RingLayout::parse(BIG_RES, &over).is_err(), "twice the bound is refused");
    }
}
