// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Guest memory the host can reach.
//!
//! This is one of the named unsafe modules (see CLAUDE.md). It exists so that everything else can
//! be safe: it owns the mapping, it owns the bounds check, and it hands out no way to hold a
//! reference into guest memory.
//!
//! # Why there is no `&[u8]` here
//!
//! The obvious API -- deref to a slice -- cannot be written soundly. A Rust reference asserts that
//! the memory behind it is not concurrently modified, and this memory is shared with a guest that
//! modifies it whenever it likes, hostile or merely racy. A `&[u8]` over it is a data race by
//! construction, and the compiler is entitled to optimise on an assumption that is false.
//!
//! So the entire surface is copies and atomics, by value:
//!
//! * [`GuestMap::load_u32`] / [`GuestMap::store_u32`] / [`GuestMap::fetch_or_u32`] for the control
//!   words, which host and guest genuinely share and which are therefore atomic on both sides.
//! * [`GuestMap::copy_out`] / [`GuestMap::copy_in`] for byte ranges, which are copied to or from
//!   host memory and then worked on there.
//!
//! The C reaches the same shape from the other direction: `vkr_ring.h` notes that commands are
//! read out of the ring into a temporary buffer before they are dispatched. Here that is not a
//! note, it is the only thing the type permits.
//!
//! # What is and is not guaranteed
//!
//! Memory safety is guaranteed unconditionally: every access is bounds-checked against the length
//! that was mapped, so no offset a guest can name reaches outside the mapping.
//!
//! Freedom from *torn reads* is not, and cannot be. A guest that rewrites a command while the host
//! copies it gets a copy that is half old and half new. That is a protocol violation, not a
//! soundness problem -- the bytes still land in a host buffer of known size, and the decoder that
//! reads them already treats every byte as hostile.

use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

/// A shared mapping of a guest-visible buffer.
///
/// Dropping it unmaps. Nothing else does, and nothing hands out a pointer that could outlive it.
pub struct GuestMap {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: `GuestMap` is a pointer and a length, and every method that touches the memory does so
// atomically or by copying bytes it owns on the host side. There is no interior `&mut` and no
// cached derived pointer, so moving one between threads or sharing it does not create an alias
// that was not already there -- the guest is a concurrent writer regardless of what the host does.
unsafe impl Send for GuestMap {}
// SAFETY: as above; `&GuestMap` grants only atomic access and copies.
unsafe impl Sync for GuestMap {}

impl GuestMap {
    /// Map `len` bytes of an shm descriptor shared with the guest.
    ///
    /// The descriptor is borrowed, not consumed: the resource that owns it goes on owning it, and
    /// the mapping stays valid after the descriptor is closed, which is what POSIX guarantees and
    /// what lets a ring outlive the resource handle it was created from.
    pub fn shm(fd: BorrowedFd<'_>, len: usize) -> io::Result<GuestMap> {
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot map an empty resource",
            ));
        }
        // SAFETY: a null hint lets the kernel choose the address; `len` is non-zero; the
        // descriptor is live for the duration of this call because it is borrowed. The flags are
        // the C's, from `vkr_context.c`: shared, so the guest sees our writes and we see its.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: mmap returned something other than MAP_FAILED, so it is a valid mapping of
        // `len` bytes and is never null.
        Ok(GuestMap { ptr: unsafe { NonNull::new_unchecked(ptr.cast::<u8>()) }, len })
    }

    /// Map `len` bytes of fresh anonymous memory, rounded up to whole pages.
    ///
    /// This is the host minting storage for a Vulkan allocation the guest means to share: the
    /// driver is handed these pages by host-pointer import instead of memory of its own, so that
    /// the pages -- and not a mapping the driver lends -- are what a resource can hold a share of.
    /// No descriptor, because nothing needs one: the share travels inside this process as an
    /// `Arc`, and on this platform nothing re-imports a buffer by descriptor across contexts.
    ///
    /// Page-rounded because a host-pointer import requires both the pointer and the size to be
    /// multiples of the driver's import alignment, and a whole number of pages satisfies every
    /// alignment a driver reports.
    pub fn anonymous(len: usize) -> io::Result<GuestMap> {
        if len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "cannot map zero bytes"));
        }
        let len = page_round(len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "size overflows when paged")
        })?;
        // SAFETY: a null hint lets the kernel choose the address; `len` is non-zero and
        // page-rounded; an anonymous private mapping names no descriptor.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: mmap returned something other than MAP_FAILED, so it is a valid mapping of
        // `len` bytes and is never null.
        Ok(GuestMap { ptr: unsafe { NonNull::new_unchecked(ptr.cast::<u8>()) }, len })
    }

    /// Where the mapping starts in this process.
    ///
    /// An address, not a pointer: this is for handing to a VMM that will publish it to a guest,
    /// and nothing in this process may dereference it without going through the accessors above.
    pub fn host_addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The address of a naturally aligned 32-bit word at `at`, or `None` if it does not lie
    /// wholly inside the mapping or is misaligned.
    ///
    /// Both conditions have to be checked here rather than trusted from the layout, because the
    /// guest also names `extra` offsets at write time, long after any layout was validated.
    fn word(&self, at: usize) -> Option<*const AtomicU32> {
        const WORD: usize = size_of::<u32>();
        if !at.is_multiple_of(WORD) || at.checked_add(WORD)? > self.len {
            return None;
        }
        // SAFETY: `at + 4 <= len`, so the offset stays inside the mapping, and `at` is a multiple
        // of 4 while the mapping itself is page-aligned -- so the result is aligned for AtomicU32.
        Some(unsafe { self.ptr.as_ptr().add(at) }.cast::<AtomicU32>())
    }

    /// Read a control word the guest may be writing concurrently.
    pub fn load_u32(&self, at: usize) -> Option<u32> {
        let p = self.word(at)?;
        // SAFETY: `word` returned a pointer that is in bounds and aligned, and the mapping
        // outlives this borrow. The access is atomic, which is what makes a concurrent guest
        // write defined rather than a race.
        Some(unsafe { &*p }.load(Ordering::Acquire))
    }

    /// Write a control word the guest may be reading concurrently.
    ///
    /// Returns whether the offset named one. A `false` is always the guest's mistake -- the host's
    /// own control words came from a validated layout.
    #[must_use]
    pub fn store_u32(&self, at: usize, value: u32) -> bool {
        let Some(p) = self.word(at) else { return false };
        // SAFETY: as `load_u32`.
        unsafe { &*p }.store(value, Ordering::Release);
        true
    }

    /// Set bits in a control word without disturbing the others.
    ///
    /// Sequentially consistent, matching the C's `vkr_ring_set_status_bits`: the status word is
    /// how the host tells the guest something changed, and it is ordered against everything.
    #[must_use]
    pub fn fetch_or_u32(&self, at: usize, bits: u32) -> bool {
        let Some(p) = self.word(at) else { return false };
        // SAFETY: as `load_u32`.
        unsafe { &*p }.fetch_or(bits, Ordering::SeqCst);
        true
    }

    /// Clear bits in a control word without disturbing the others.
    ///
    /// The counterpart to [`GuestMap::fetch_or_u32`] and sequentially consistent for the same
    /// reason: the C's `vkr_ring_unset_status_bits` clears the IDLE bit as the ring leaves its
    /// park, and the guest is reading that word to decide whether it must ring the doorbell.
    #[must_use]
    pub fn fetch_and_u32(&self, at: usize, keep: u32) -> bool {
        let Some(p) = self.word(at) else { return false };
        // SAFETY: as `load_u32`.
        unsafe { &*p }.fetch_and(keep, Ordering::SeqCst);
        true
    }

    /// Read a control word with sequential consistency rather than acquire.
    ///
    /// This exists for exactly one caller: the ring's park handshake. A parking ring stores the
    /// IDLE bit and then loads the tail to check nothing arrived in between; the guest stores the
    /// tail and then loads the status to decide whether to ring the doorbell. If either load may
    /// be reordered before the other side's store, both can miss, and the ring sleeps on work that
    /// is already there with no one left to wake it. Acquire does not prevent that -- only a pair
    /// of sequentially consistent operations orders the two stores against the two loads.
    ///
    /// The C names the same requirement in `vkr_ring_load_tail_seqcst`, whose comment records that
    /// the 2 ms poll it replaced existed only to survive this race.
    pub fn load_u32_seqcst(&self, at: usize) -> Option<u32> {
        let p = self.word(at)?;
        // SAFETY: as `load_u32`.
        Some(unsafe { &*p }.load(Ordering::SeqCst))
    }

    /// Copy bytes out of guest memory into a host buffer.
    ///
    /// Returns whether the range lay inside the mapping. The copy may tear if the guest writes the
    /// same bytes at the same time; see the module docs for why that is a protocol problem and not
    /// a safety one.
    #[must_use]
    pub fn copy_out(&self, at: usize, dst: &mut [u8]) -> bool {
        if !self.holds(at, dst.len()) {
            return false;
        }
        // SAFETY: `holds` proved `at + dst.len() <= self.len`, so the source range is inside the
        // mapping. `dst` is a live host slice of exactly that length, and the two cannot overlap
        // because one is a mapping this type owns and the other is a borrow the caller brought.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr().add(at), dst.as_mut_ptr(), dst.len())
        };
        true
    }

    /// Copy bytes from a host buffer into guest memory.
    #[must_use]
    pub fn copy_in(&self, at: usize, src: &[u8]) -> bool {
        if !self.holds(at, src.len()) {
            return false;
        }
        // SAFETY: as `copy_out`, with the direction reversed. Writing is sound for the same reason
        // reading is: the destination is inside a mapping this type owns for its whole life.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr().add(at), src.len())
        };
        true
    }

    /// Whether `len` bytes at `at` lie wholly inside the mapping. A zero-length range at the very
    /// end is inside it, which matters because an empty `extra` region is legal.
    fn holds(&self, at: usize, len: usize) -> bool {
        at.checked_add(len).is_some_and(|end| end <= self.len)
    }
}

/// A classic resource's guest pages: the scatter list the VMM attached, as one byte-addressed
/// span.
///
/// The VMM owns the pages and the list describes them; this type holds neither. What it adds is
/// the walk -- an offset into the concatenation resolved to a page and a position -- and the
/// bounds check, so that no offset a guest names reaches outside the pages the VMM described.
/// The C's walker asserts on an offset past the end (`iov.c`), a host abort reachable from a
/// guest value; here it is a `false`.
///
/// Copies only, for the reason [`GuestMap`] gives: the guest writes these pages whenever it
/// likes, so a reference into them cannot be sound.
pub struct Iov<'a>(&'a [crate::abi::GuestIov]);

impl<'a> Iov<'a> {
    pub fn new(entries: &'a [crate::abi::GuestIov]) -> Iov<'a> {
        Iov(entries)
    }

    /// The total bytes the list describes.
    pub fn len(&self) -> u64 {
        self.0.iter().map(|e| e.len as u64).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|e| e.len == 0)
    }

    /// Whether the entries are the same pages, in the same order, as `other`'s.
    pub fn same_pages(&self, other: &Iov<'_>) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().zip(other.0).all(|(a, b)| a.base == b.base && a.len == b.len)
    }

    /// Walk `len` bytes from `at`, handing each contiguous piece to `f` as a host pointer and a
    /// length. `false`, with nothing visited, if the range is not wholly inside the list.
    fn walk(&self, at: u64, len: usize, mut f: impl FnMut(*mut u8, usize, usize)) -> bool {
        let Some(end) = at.checked_add(len as u64) else {
            return false;
        };
        if end > self.len() {
            return false;
        }
        let mut skip = at;
        let mut done = 0usize;
        for e in self.0 {
            if done == len {
                break;
            }
            let elen = e.len as u64;
            if skip >= elen {
                skip -= elen;
                continue;
            }
            let start = skip as usize;
            let take = (e.len - start).min(len - done);
            // The entry's base is the VMM's host address for the page; `start` is inside it.
            f(e.base.0.cast::<u8>().wrapping_add(start), done, take);
            done += take;
            skip = 0;
        }
        done == len
    }

    /// Copy bytes out of the guest pages into `dst`. Returns whether the range was inside them.
    #[must_use]
    pub fn copy_out(&self, at: u64, dst: &mut [u8]) -> bool {
        self.walk(at, dst.len(), |src, into, n| {
            // SAFETY: the VMM's contract for an attached iov is that every entry addresses `len`
            // bytes of live guest memory until it detaches the list, and this renderer holds no
            // list past a detach. `walk` proved the piece is inside its entry; `dst` is a host
            // slice the caller owns, so the two cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr().add(into), n) };
        })
    }

    /// Copy `src` into the guest pages at `at`. Returns whether the range was inside them.
    #[must_use]
    pub fn copy_in(&self, at: u64, src: &[u8]) -> bool {
        self.walk(at, src.len(), |dst, from, n| {
            // SAFETY: as `copy_out`, with the direction reversed; the VMM maps the pages writable
            // because a transfer from the host is what they are for.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(from), dst, n) };
        })
    }
}

/// The host's page size, which is what a mapping's length has to be a multiple of.
pub fn page_size() -> usize {
    // SAFETY: a plain sysconf query with no pointers involved.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(n > 0, "the host must report a page size");
    n as usize
}

/// Round a length up to a whole number of pages, or `None` if that overflows.
///
/// The VMM maps a resource with `MAP_FIXED`, which requires a page-aligned size, so the rounded
/// length is the resource's real size -- not a detail of the allocation. Everything that checks an
/// offset against the resource has to check it against this, or it refuses layouts that fit.
pub fn page_round(len: usize) -> Option<usize> {
    let page = page_size();
    len.checked_add(page - 1).map(|n| n & !(page - 1))
}

/// Create an anonymous shared file of `len` bytes and map it.
///
/// This is the host minting memory for the guest rather than receiving it -- the C's
/// `os_create_anonymous_file` followed by the `mmap` in `vkr_context_create_resource_from_shm`.
/// The descriptor comes back with the mapping because the VMM will eventually want it: it is what
/// the guest maps on its side, and closing it here would make that impossible to offer later.
///
/// The file never appears in a directory anyone can open: on Linux it has no name at all, and on
/// macOS its name is unlinked before this returns.
pub fn anonymous_shm(len: usize, debug_name: &str) -> io::Result<(OwnedFd, GuestMap)> {
    let len = page_round(len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "size overflows when paged"))?;
    let fd = anonymous_fd(debug_name)?;
    // The file starts empty; this is what gives it the length the mapping needs.
    // SAFETY: `fd` is a live descriptor this function just created and still owns.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let map = GuestMap::shm(fd.as_fd(), len)?;
    Ok((fd, map))
}

/// A descriptor for a file with no directory entry, by whatever route the host offers.
fn anonymous_fd(debug_name: &str) -> io::Result<OwnedFd> {
    #[cfg(target_os = "linux")]
    {
        let name = std::ffi::CString::new(debug_name)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "name has a NUL in it"))?;
        // SAFETY: `name` is a live NUL-terminated string for the duration of the call.
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: memfd_create returned a fresh descriptor that nothing else owns.
        return Ok(unsafe { OwnedFd::from_raw_fd(fd) });
    }
    #[cfg(not(target_os = "linux"))]
    {
        // macOS has no memfd, so this is shm_open under a name that is unlinked immediately. The
        // name only has to survive until then, but it still has to be unique, because two workers
        // racing on the same one would share memory that is supposed to be private.
        //
        // shm_open rejects O_CLOEXEC with EINVAL here, so close-on-exec is set afterwards.
        for attempt in 0..32u32 {
            let name = format!("/{}-{}-{}\0", debug_name, std::process::id(), attempt);
            // SAFETY: `name` is NUL-terminated above and live for the duration of the call.
            let fd = unsafe {
                libc::shm_open(
                    name.as_ptr().cast::<libc::c_char>(),
                    libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                    0o600 as libc::c_uint,
                )
            };
            if fd >= 0 {
                // SAFETY: shm_open returned a fresh descriptor that nothing else owns; taking
                // ownership here is what closes it on every path out.
                let owned = unsafe { OwnedFd::from_raw_fd(fd) };
                // SAFETY: `owned` is live; both calls take the descriptor and no pointers.
                unsafe {
                    libc::fcntl(owned.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
                    libc::shm_unlink(name.as_ptr().cast::<libc::c_char>());
                }
                return Ok(owned);
            }
            let err = io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EEXIST) {
                return Err(err);
            }
        }
        Err(io::Error::new(io::ErrorKind::AlreadyExists, "no unused shm name in 32 tries"))
    }
}

impl Drop for GuestMap {
    fn drop(&mut self) {
        // SAFETY: this pointer and length came from the `mmap` in `shm` and have not changed
        // since; nothing else unmaps them, because nothing else holds them.
        let rc = unsafe { libc::munmap(self.ptr.as_ptr().cast::<libc::c_void>(), self.len) };
        assert_eq!(
            rc,
            0,
            "munmap of a mapping we made must succeed: {}",
            io::Error::last_os_error()
        );
    }
}

impl std::fmt::Debug for GuestMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately not the address: it is noise that changes every run, and printing guest
        // memory's location into a log is not something a debug impl should decide to do.
        f.debug_struct("GuestMap").field("len", &self.len).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::{AsFd, OwnedFd};

    /// An anonymous shared mapping standing in for a guest resource. It is the same kind of object
    /// -- an shm descriptor -- so the mapping path under test is the real one.
    fn shm_fd(len: usize) -> OwnedFd {
        let mut file = std::env::temp_dir();
        file.push(format!(
            "virglrs-guestmap-test-{}-{:?}",
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

    #[test]
    fn bytes_written_to_the_mapping_come_back() {
        let fd = shm_fd(0x1000);
        let m = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped");
        assert_eq!(m.len(), 0x1000);

        assert!(m.copy_in(0x40, b"venus"), "an in-bounds write is accepted");
        let mut out = [0u8; 5];
        assert!(m.copy_out(0x40, &mut out), "and reads back");
        assert_eq!(&out, b"venus");
    }

    /// The whole reason this type exists: an offset the guest chose can never reach outside what
    /// was mapped, whatever it is.
    #[test]
    fn no_offset_a_guest_can_name_reaches_outside_the_mapping() {
        let fd = shm_fd(0x1000);
        let m = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped");
        let mut buf = [0u8; 8];

        assert!(!m.copy_out(0x1000, &mut buf), "starting at the end reads nothing");
        assert!(!m.copy_out(0xffc, &mut buf), "a range that straddles the end is refused whole");
        assert!(!m.copy_out(usize::MAX, &mut buf), "an offset that would overflow is refused");
        assert!(!m.copy_in(usize::MAX - 4, b"nope"), "and so is the same on the way in");
        assert!(!m.store_u32(0x1000, 1), "a word starting at the end is outside it");
        assert!(m.load_u32(0x1000).is_none(), "and cannot be read either");

        // The tightest case there is, and the one a loosened bound survives: a *single* byte at
        // exactly the end. Every range above is wide enough that an off-by-one still refuses it.
        let mut one = [0u8; 1];
        assert!(!m.copy_out(0xfff + 1, &mut one), "one byte starting at the end is outside it");
        assert!(!m.copy_in(0x1000, b"x"), "and cannot be written there either");
        assert!(m.copy_out(0xfff, &mut one), "the byte before it is the last one inside");

        assert!(m.copy_out(0xff8, &mut buf), "the last eight bytes are inside, and readable");
        assert!(m.copy_out(0x1000, &mut []), "an empty range at the very end is inside it");
    }

    /// The control words are read atomically, so an unaligned one is not a control word at all.
    /// The layout check rejects these too, but `extra` offsets arrive at write time with no layout
    /// left to check them against, so the mapping refuses them itself.
    #[test]
    fn a_misaligned_control_word_is_refused_rather_than_read() {
        let fd = shm_fd(0x1000);
        let m = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped");

        for at in [1, 2, 3, 0x101] {
            assert!(m.load_u32(at).is_none(), "offset {at} is not 32-bit aligned");
            assert!(!m.store_u32(at, 7), "and cannot be written either");
            assert!(!m.fetch_or_u32(at, 7), "nor have bits set in it");
        }
        assert!(m.store_u32(0x100, 7), "the aligned word next to them is fine");
        assert_eq!(m.load_u32(0x100), Some(7));
    }

    #[test]
    fn status_bits_accumulate_instead_of_replacing_each_other() {
        let fd = shm_fd(0x1000);
        let m = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped");

        assert!(m.fetch_or_u32(0x20, 0b001));
        assert!(m.fetch_or_u32(0x20, 0b100));
        assert_eq!(m.load_u32(0x20), Some(0b101), "the earlier bit survived the later one");
    }

    /// Two mappings of one descriptor see each other's writes -- which is the property the whole
    /// ring depends on, and the one a private mapping would silently not have.
    #[test]
    fn the_mapping_is_shared_and_not_a_private_copy() {
        let fd = shm_fd(0x1000);
        let a = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped once");
        let b = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped twice");

        assert!(a.store_u32(0x80, 0xdecafbad));
        assert_eq!(b.load_u32(0x80), Some(0xdecafbad), "the other mapping sees it");

        assert!(b.copy_in(0x200, b"shared"));
        let mut out = [0u8; 6];
        assert!(a.copy_out(0x200, &mut out));
        assert_eq!(&out, b"shared");
    }

    /// The mapping outliving the descriptor is what lets a ring outlive the resource handle it was
    /// created from, so it is asserted rather than assumed.
    #[test]
    fn the_mapping_survives_the_descriptor_being_closed() {
        let m = {
            let fd = shm_fd(0x1000);
            GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped")
        };
        assert!(m.store_u32(0, 0x1234), "still writable with the descriptor long gone");
        assert_eq!(m.load_u32(0), Some(0x1234));
    }

    #[test]
    fn a_minted_mapping_is_page_rounded_and_usable() {
        let page = page_size();
        // The size the venus corpus asks for a ring resource, which is not a whole page.
        let (fd, m) = anonymous_shm(0x24000 - 1, "virglrs-test").expect("minted");
        assert_eq!(m.len() % page, 0, "the length is a whole number of pages");
        assert!(m.len() >= 0x24000 - 1, "and is at least what was asked for");
        assert!(m.len() < 0x24000 - 1 + page, "without rounding further than one page");

        assert!(m.copy_in(0, b"ring"), "the memory is there and writable");
        let mut out = [0u8; 4];
        assert!(m.copy_out(0, &mut out));
        assert_eq!(&out, b"ring");

        // The descriptor comes back with it, because the VMM will need it to map its own side.
        let second = GuestMap::shm(fd.as_fd(), m.len()).expect("the descriptor still names it");
        assert_eq!(second.load_u32(0), m.load_u32(0), "and it is the same memory");
    }

    /// Every minting must be its own memory. On macOS the name is reused across attempts only on
    /// EEXIST, and a bug there would hand two resources the same bytes.
    #[test]
    fn two_minted_mappings_are_not_the_same_memory() {
        let (_a_fd, a) = anonymous_shm(0x1000, "virglrs-test").expect("one");
        let (_b_fd, b) = anonymous_shm(0x1000, "virglrs-test").expect("two");
        assert!(a.store_u32(0, 0xaaaa_aaaa));
        assert!(b.store_u32(0, 0xbbbb_bbbb));
        assert_eq!(a.load_u32(0), Some(0xaaaa_aaaa), "the second did not overwrite the first");
        assert_eq!(b.load_u32(0), Some(0xbbbb_bbbb));
    }

    #[test]
    fn a_page_round_that_would_overflow_is_refused() {
        assert_eq!(page_round(0), Some(0));
        assert_eq!(page_round(1), Some(page_size()));
        assert_eq!(page_round(page_size()), Some(page_size()), "an exact page is left alone");
        assert_eq!(page_round(usize::MAX), None, "rather than wrapping to something small");
    }

    #[test]
    fn an_empty_resource_is_refused_rather_than_mapped() {
        let fd = shm_fd(0x1000);
        assert!(GuestMap::shm(fd.as_fd(), 0).is_err(), "mmap of zero bytes is an error, not a map");
    }
}
