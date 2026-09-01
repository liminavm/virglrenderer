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

use std::sync::Arc;

use super::proto::types::VkRingCreateInfoMESA;
use crate::guest_mem::GuestMap;
use crate::ids::ResourceHandle;

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

/// Why a guest's ring could not be created. Both are the guest's mistake, so either poisons the
/// context: naming a resource that is not there or is not host-addressable, or describing a layout
/// inside it that we will not touch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RingError {
    NoResource(ResourceHandle),
    Layout(LayoutError),
}

/// The only thing venus needs from the renderer's resource table.
///
/// A trait rather than the table itself, so this module never learns what a `Resource` is: what a
/// ring wants is a share of some host-addressable memory, and everything else about a resource --
/// its iovs, its attachments, the VMM's private token -- is none of its business. It also makes
/// the ring handlers testable without a renderer, which is the practical half of the same point.
pub trait ShmResources {
    /// A share of the mapping behind a resource, or `None` if the guest named one that does not
    /// exist or is not host-addressable. The two are deliberately one answer here: from the ring's
    /// side both mean "no memory", and the caller that can tell them apart says so in its log.
    fn shm(&self, handle: ResourceHandle) -> Option<Arc<GuestMap>>;
}

/// A ring the guest created and the host has agreed to read from.
///
/// It holds a share of the mapping rather than a way to find one. That is what makes it outlive
/// the resource handle it was created from: the guest may unref the resource, and the ring goes on
/// working until the ring itself is destroyed. The C keeps a bare `const struct vkr_resource *`
/// here and depends on the guest destroying things in the right order.
pub struct Ring {
    /// Where the ring's parts sit inside the resource. Absolute offsets, already validated.
    pub layout: RingLayout,
    /// The memory those offsets are into.
    pub map: Arc<GuestMap>,
}

impl Ring {
    /// Take a guest's ring description and, if it holds up, the memory to go with it.
    pub fn create(
        resources: &dyn ShmResources,
        info: &VkRingCreateInfoMESA,
    ) -> Result<Ring, RingError> {
        let handle = ResourceHandle(info.resourceId);
        // The resource is resolved before the layout is parsed, because the layout is checked
        // against the resource's real size -- checking it against the size the guest claimed
        // would be checking the guest against itself.
        let map = resources.shm(handle).ok_or(RingError::NoResource(handle))?;
        let layout = RingLayout::parse(map.len(), info).map_err(RingError::Layout)?;
        Ok(Ring { layout, map })
    }

    /// How far the guest says it has written.
    pub fn tail(&self) -> u32 {
        self.map.load_u32(self.layout.tail.begin()).expect("a validated tail is inside the mapping")
    }

    /// How far the host has read. Written by us, read by the guest.
    pub fn set_head(&self, head: u32) {
        assert!(
            self.map.store_u32(self.layout.head.begin(), head),
            "a validated head is inside the mapping"
        );
    }

    /// Tell the guest something about the ring changed.
    pub fn set_status_bits(&self, bits: u32) {
        assert!(
            self.map.fetch_or_u32(self.layout.status.begin(), bits),
            "a validated status word is inside the mapping"
        );
    }

    /// Write one guest-named word in the `extra` region.
    ///
    /// The offset is relative to `extra` and arrives from the guest at write time, with no layout
    /// left to have checked it -- so unlike the three above this one can legitimately fail, and
    /// says so rather than asserting.
    #[must_use]
    pub fn write_extra(&self, offset: usize, value: u32) -> bool {
        let Some(at) = self.layout.extra.begin().checked_add(offset) else { return false };
        // Inside `extra`, not merely inside the mapping: the rest of the resource is not the
        // guest's to have us write through this door.
        let Some(end) = at.checked_add(size_of::<u32>()) else { return false };
        if end > self.layout.extra.begin() + self.layout.extra.size() {
            return false;
        }
        self.map.store_u32(at, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::{AsFd, OwnedFd};

    /// A stand-in resource table holding one mapping under one handle. Enough to drive every ring
    /// path, and it exists because `ShmResources` was made a trait precisely so these tests need
    /// no renderer.
    struct OneResource(ResourceHandle, Arc<GuestMap>);

    impl ShmResources for OneResource {
        fn shm(&self, handle: ResourceHandle) -> Option<Arc<GuestMap>> {
            (handle == self.0).then(|| Arc::clone(&self.1))
        }
    }

    fn shm_fd(len: usize) -> OwnedFd {
        let mut file = std::env::temp_dir();
        file.push(format!(
            "virglrs-ring-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file)
            .expect("a temp file to back the mapping");
        f.set_len(len as u64).expect("sized");
        std::fs::remove_file(&file).expect("unlinked; the descriptor keeps it alive");
        OwnedFd::from(f)
    }

    const HANDLE: ResourceHandle = ResourceHandle(7);

    fn table(len: usize) -> OneResource {
        let fd = shm_fd(len);
        OneResource(HANDLE, Arc::new(GuestMap::shm(fd.as_fd(), len).expect("mapped")))
    }

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

    /// A ring created from a resource keeps working after the guest unrefs it. The resource table
    /// is what a `resource_unref` empties, so dropping the whole table is exactly that event.
    #[test]
    fn a_ring_outlives_the_resource_it_was_created_from() {
        let t = table(0x4000);
        let mut info = good();
        info.resourceId = HANDLE.0;
        let ring = Ring::create(&t, &info).expect("created");

        ring.set_head(0x1234);
        drop(t); // the guest unrefs the resource while the ring is still alive

        assert_eq!(ring.map.load_u32(ring.layout.head.begin()), Some(0x1234));
        ring.set_head(0x5678);
        assert_eq!(
            ring.map.load_u32(ring.layout.head.begin()),
            Some(0x5678),
            "still ours to write"
        );
    }

    #[test]
    fn a_resource_that_is_not_there_is_refused_by_name() {
        let t = table(0x4000);
        let mut info = good();
        info.resourceId = HANDLE.0 + 1;
        assert_eq!(
            Ring::create(&t, &info).err(),
            Some(RingError::NoResource(ResourceHandle(HANDLE.0 + 1))),
            "the refusal says which resource, because that is the bug"
        );
    }

    /// The layout is checked against the mapping's real size, not the size the guest claimed --
    /// otherwise the check is the guest marking its own work.
    #[test]
    fn the_layout_is_checked_against_the_resource_and_not_the_guest_s_word() {
        let t = table(0x2000);
        let mut info = good();
        info.resourceId = HANDLE.0;
        info.offset = 0x1000;
        info.size = 0x2000; // would fit a 0x4000 resource; this one is 0x2000
        assert_eq!(
            Ring::create(&t, &info).err(),
            Some(RingError::Layout(LayoutError::OutOfBounds(Part::Whole)))
        );
    }

    #[test]
    fn the_head_and_tail_are_the_words_the_layout_named() {
        let t = table(0x4000);
        let mut info = good();
        info.resourceId = HANDLE.0;
        let ring = Ring::create(&t, &info).expect("created");

        // The guest writes the tail; the host reads it.
        assert!(t.1.store_u32(ring.layout.tail.begin(), 42));
        assert_eq!(ring.tail(), 42);

        // The host writes the head; nothing else moved.
        ring.set_head(9);
        assert_eq!(t.1.load_u32(ring.layout.head.begin()), Some(9));
        assert_eq!(t.1.load_u32(ring.layout.tail.begin()), Some(42), "the tail is not ours");

        ring.set_status_bits(0b10);
        ring.set_status_bits(0b01);
        assert_eq!(t.1.load_u32(ring.layout.status.begin()), Some(0b11), "bits accumulate");
    }

    /// `extra` offsets arrive from the guest at write time, with no layout left to have checked
    /// them. The door has to be exactly the size of the room: being inside the mapping is not
    /// enough, or the guest could write through it into the buffer or the control words.
    #[test]
    fn an_extra_write_cannot_reach_outside_the_extra_region() {
        let t = table(0x4000);
        let mut info = good();
        info.resourceId = HANDLE.0;
        let ring = Ring::create(&t, &info).expect("created");
        assert_eq!(ring.layout.extra.size(), 0x40);

        assert!(ring.write_extra(0, 1), "the first word of extra");
        assert!(ring.write_extra(0x3c, 2), "the last word of extra");
        assert!(!ring.write_extra(0x40, 3), "one word past the end of extra, still in the mapping");
        assert!(!ring.write_extra(usize::MAX, 4), "an offset that would overflow");

        // The offset that makes wrapping worth a guest's while: chosen so that `extra.begin()`
        // plus it wraps to exactly the head's address. It is aligned and it lands inside the
        // region check, so every guard except the overflow one waves it through -- which is why
        // the overflow guard is a refusal and not an `as` cast.
        let wrap_to_head = usize::MAX - (ring.layout.extra.begin() - ring.layout.head.begin()) + 1;
        assert!(!ring.write_extra(wrap_to_head, 0xbad), "an offset that wraps onto the head");

        // The head sits below `extra` and is inside the same mapping; no extra offset may reach it.
        ring.set_head(0xfeed);
        for off in [0usize, 4, 0x3c, 0x40, 0x1000, wrap_to_head] {
            let _ = ring.write_extra(off, 0xbad);
        }
        assert_eq!(
            t.1.load_u32(ring.layout.head.begin()),
            Some(0xfeed),
            "no extra write reached the host's own progress counter"
        );
    }
}
