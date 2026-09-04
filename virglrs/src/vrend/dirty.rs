// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Which slots of one kind have changed since a draw last bound them.

/// A per-slot dirty mask `N` slots wide, where `N` is the cap on the slots it tracks. The width
/// and the cap are one number: a mask written as a bare `u32` beside a `if slot < 32` at each
/// site is two, and the site that forgets the test marks a bit no binder ever reads.
///
/// Marking a slot the mask cannot hold is a host bug -- the decoder has already refused a slot
/// the guest may not use -- so it asserts. The one place the decoder deliberately admits more
/// than the mask carries is a sampler view, and that call site says so.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dirty<const N: usize>(u32);

impl<const N: usize> Default for Dirty<N> {
    fn default() -> Self {
        Self::none()
    }
}

impl<const N: usize> Dirty<N> {
    /// Read by [`Dirty::mark`], so a mask wider than its bits fails the build.
    const WIDTH_FITS: () = assert!(N <= u32::BITS as usize, "a dirty mask is at most 32 slots");

    /// The bits a slot may occupy.
    const OCCUPIED: u32 = if N == u32::BITS as usize { u32::MAX } else { (1 << N) - 1 };

    pub const fn none() -> Self {
        Self(0)
    }

    /// Only `slot`, as a reset that leaves one slot to rebind.
    pub fn just(slot: u32) -> Self {
        Self(Self::bit(slot))
    }

    /// Every slot, as a new program takes them: nothing bound for it is still bound.
    pub const fn all() -> Self {
        Self(Self::OCCUPIED)
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn contains(self, slot: u32) -> bool {
        self.0 & Self::bit(slot) != 0
    }

    pub fn mark(&mut self, slot: u32) {
        let () = Self::WIDTH_FITS;
        assert!((slot as usize) < N, "slot {slot} is past the {N} this mask tracks");
        self.0 |= Self::bit(slot);
    }

    pub fn unmark(&mut self, slot: u32) {
        self.0 &= !Self::bit(slot);
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }

    /// The slots dirty in both, which is how a bind narrows "changed" to "changed and used".
    pub fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    fn bit(slot: u32) -> u32 {
        assert!((slot as usize) < N, "slot {slot} is past the {N} this mask tracks");
        1 << slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_holds_only_the_slots_it_tracks() {
        assert_eq!(Dirty::<16>::all().0, 0xffff);
        assert_eq!(Dirty::<32>::all().0, u32::MAX);
        assert!(Dirty::<16>::all().contains(15));
    }

    #[test]
    fn marking_and_unmarking_one_slot_leaves_the_rest() {
        let mut d = Dirty::<32>::none();
        assert!(d.is_empty());
        d.mark(3);
        d.mark(7);
        assert!(d.contains(3) && d.contains(7) && !d.contains(4));
        d.unmark(3);
        assert!(!d.contains(3) && d.contains(7));
        d.clear();
        assert!(d.is_empty());
    }

    #[test]
    fn intersect_keeps_what_both_hold() {
        let mut a = Dirty::<32>::none();
        a.mark(1);
        a.mark(2);
        let mut b = Dirty::<32>::none();
        b.mark(2);
        b.mark(3);
        let both = a.intersect(b);
        assert!(both.contains(2) && !both.contains(1) && !both.contains(3));
    }

    #[test]
    #[should_panic(expected = "past the 16")]
    fn marking_past_the_cap_is_a_host_bug() {
        Dirty::<16>::none().mark(16);
    }
}
