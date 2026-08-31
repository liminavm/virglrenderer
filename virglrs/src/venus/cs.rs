// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The venus command stream: the trust boundary.
//!
//! Every byte the guest sends the Vulkan renderer arrives here. The generated decoder in
//! [`super::proto`] is written against this module and nothing else, which is what makes the
//! generated code safe Rust: reads are bounds-checked against a slice, allocations come from a
//! capped arena, and a malformed stream sets a poison flag instead of panicking.
//!
//! Poison, not panic, is the rule at this boundary. An `assert!` is for a host invariant we got
//! wrong; a guest that sends a truncated command, an impossible array size or an unknown object id
//! is not that. The C side calls this `fatal` and the name is kept, but it is fatal to the ring,
//! never to the process.
//!
//! Two flavours of poison, mirroring `src/venus/vkr_cs.h`:
//!
//! * **hard** -- the stream is unusable. It is shared with the ring loop, which stops.
//! * **soft** -- this one command named an object the host never created (a ghost). The command is
//!   dropped, the ring keeps going. Generated code cannot tell the two apart, and must not: both
//!   mean "do not call the handler, do not encode a reply".

use std::cell::Cell;

use bumpalo::Bump;

/// Cap on the temp arena, matching `VKR_CS_DECODER_TEMP_POOL_MAX_SIZE`. It is not a resource limit
/// so much as an overflow guard: the guest encodes the array sizes, and a plausible one is dozens
/// of megabytes (`vkGetPipelineCacheData`), not a gigabyte.
pub const TEMP_POOL_MAX: usize = 1024 * 1024 * 1024;

/// Round a wire size up to the stream's 4-byte granularity.
#[inline]
pub const fn align4(size: usize) -> usize {
    (size + 3) & !3
}

/// The object id the guest uses to name a Vulkan object. Distinct from the host handle it resolves
/// to: the guest never sees a host handle, and confusing the two is the bug this newtype exists to
/// make unrepresentable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct ObjectId(pub u64);

/// Resolves guest object ids to host objects. The decoder holds one by reference; what sits behind
/// it is vkr's object table in the renderer and an identity map in the round-trip test.
/// A pointer stored in an arena array -- an array of strings is an array of these.
///
/// It exists because the standard library gives a raw pointer no `Default`, and the arena needs
/// one to hand back a fresh element. `repr(transparent)` is the point: `[Ptr]` and `[*const T]`
/// have the same layout, so the decoded array can be handed to Vulkan as itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Ptr(pub *const core::ffi::c_void);

impl Default for Ptr {
    fn default() -> Self {
        Ptr(core::ptr::null())
    }
}

pub trait Objects {
    /// The host handle for `id`, or `None` if the host never created it. `ty` is the
    /// `VkObjectType` the wire claims, and a mismatch is a miss -- the guest does not get to
    /// reinterpret one object as another.
    fn lookup(&self, id: ObjectId, ty: i32) -> Option<u64>;
}

/// An object table that resolves every id to itself. Used by the wire round-trip, where the point
/// is to reproduce the guest's bytes and no host object exists.
pub struct IdentityObjects;

impl Objects for IdentityObjects {
    fn lookup(&self, id: ObjectId, _ty: i32) -> Option<u64> {
        Some(id.0)
    }
}

/// Reads a command stream.
///
/// The lifetime `'a` ties the decoded structures to both the guest bytes and the arena they were
/// allocated from, so a decoded command cannot outlive either. `temp` is a shared reference on
/// purpose: allocating must not borrow the decoder, because the generated code allocates an array
/// and keeps reading in the same expression.
pub struct Decoder<'a> {
    buf: &'a [u8],
    pos: usize,
    temp: &'a Bump,
    temp_used: Cell<usize>,
    objects: &'a dyn Objects,
    /// Shared with the ring loop, which stops when it is set.
    hard: &'a Cell<bool>,
    /// Per-command: this command named a ghost and must be dropped.
    soft: Cell<bool>,
}

impl<'a> Decoder<'a> {
    pub fn new(
        buf: &'a [u8],
        temp: &'a Bump,
        objects: &'a dyn Objects,
        hard: &'a Cell<bool>,
    ) -> Self {
        Decoder {
            buf,
            pos: 0,
            temp,
            temp_used: Cell::new(0),
            objects,
            hard,
            soft: Cell::new(false),
        }
    }

    /// Bytes consumed so far. The recorder stamps a command as the slice between the position
    /// before and after its dispatch, so this is what the round-trip compares against.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn has_command(&self) -> bool {
        self.pos < self.buf.len()
    }

    /// Poison the stream. Takes `&self` because the generated code reaches it from paths that only
    /// hold a shared borrow, exactly as the C does.
    pub fn set_fatal(&self) {
        self.hard.set(true);
    }

    /// Poison this command only -- it named an object the host never created.
    pub fn set_soft_fatal(&self) {
        self.soft.set(true);
    }

    /// What the generated dispatch wrappers ask: should this command be skipped? Both flavours say
    /// yes, and that identity is the containment mechanism.
    pub fn fatal(&self) -> bool {
        self.hard.get() || self.soft.get()
    }

    /// What the ring loop asks: is the stream itself unusable?
    pub fn hard_fatal(&self) -> bool {
        self.hard.get()
    }

    /// End of a command: the soft poison does not outlive it.
    pub fn clear_soft_fatal(&self) {
        self.soft.set(false);
    }

    /// The next `n` bytes without consuming them, or `None` (having poisoned the stream) if the
    /// stream is shorter than that.
    pub fn peek_bytes(&self, n: usize) -> Option<&'a [u8]> {
        match self.buf.get(self.pos..self.pos + n) {
            Some(b) => Some(b),
            None => {
                self.set_fatal();
                None
            }
        }
    }

    /// Consume `advance` bytes of stream and hand back the first `n` of them. `advance` is the wire
    /// size including padding; `n` is the payload. Poisons and yields `None` on a short stream.
    pub fn read_bytes(&mut self, advance: usize, n: usize) -> Option<&'a [u8]> {
        debug_assert!(n <= advance);
        let b = self.peek_bytes(advance)?;
        self.pos += advance;
        Some(&b[..n])
    }

    /// Allocate zeroed storage for one `T`, or `None` (poisoning) if the arena is exhausted.
    pub fn alloc_temp<T: Default>(&self) -> Option<&'a mut T> {
        self.charge(size_of::<T>())?;
        Some(self.temp.alloc(T::default()))
    }

    /// Allocate a zeroed array of `count` `T`, or `None` (poisoning) if the size is implausible or
    /// the arena is exhausted. A zero count still yields an empty slice, never `None`: an empty
    /// array is a thing the guest is allowed to send.
    pub fn alloc_temp_array<T: Default + Clone>(&self, count: usize) -> Option<&'a mut [T]> {
        let Some(bytes) = size_of::<T>().checked_mul(count) else {
            self.set_fatal();
            return None;
        };
        self.charge(bytes)?;
        if count == 0 {
            // The arena's cursor for a zero-byte request is wherever the last allocation left it,
            // which need not be aligned for `T`. An empty array is a thing the guest sends often
            // enough that this is the common path, not a corner.
            return Some(&mut []);
        }
        Some(self.temp.alloc_slice_fill_with(count, |_| T::default()))
    }

    fn charge(&self, bytes: usize) -> Option<()> {
        let used = self.temp_used.get().saturating_add(bytes);
        if used > TEMP_POOL_MAX {
            self.set_fatal();
            return None;
        }
        self.temp_used.set(used);
        Some(())
    }

    /// Resolve a guest object id, poisoning softly on a miss. The soft flavour is deliberate: a
    /// guest whose own error handling left it naming a dead object must lose the command, not the
    /// ring.
    pub fn lookup_object(&self, id: ObjectId, ty: i32) -> u64 {
        match self.objects.lookup(id, ty) {
            Some(handle) => handle,
            None => {
                self.set_soft_fatal();
                0
            }
        }
    }
}

/// What the guest's protocol supports.
///
/// Encoding a `pNext` chain has to skip structs the guest's venus protocol does not know, or the
/// reply is unreadable to it. The generated encoder asks this; vkr answers from what the guest
/// negotiated at `vkSetReplyCommandStreamMESA` time.
pub trait Protocol {
    fn has_extension(&self, number: u32) -> bool;
    fn has_api_version(&self, version: u32) -> bool;
}

/// A protocol that supports everything.
///
/// Correct for the wire round trip specifically: re-encoding a chain the guest itself sent cannot
/// need to skip any of it. Not correct for replies, which is why vkr answers with the real one.
pub struct AllOfIt;

impl Protocol for AllOfIt {
    fn has_extension(&self, _number: u32) -> bool {
        true
    }
    fn has_api_version(&self, _version: u32) -> bool {
        true
    }
}

/// Writes a reply stream.
///
/// Unlike the decoder this side is trusted -- the bytes come from us -- so the only failure it can
/// have is running out of room in the guest's reply buffer, which poisons the ring for the same
/// reason a short read does: the guest gave us a buffer that cannot hold the answer.
pub struct Encoder<'a> {
    buf: &'a mut [u8],
    pos: usize,
    fatal: bool,
    protocol: &'a dyn Protocol,
    /// Offsets this encoder filled as padding rather than payload, once someone asks for them.
    ///
    /// Only the differential test does. A guest's encoder leaves its padding uninitialised, so no
    /// faithful reproduction of its wire can match those bytes -- and this renderer will not copy
    /// them, because padding a reply with whatever the host buffer held is how host memory leaks
    /// into a guest. Knowing exactly which bytes those are is what keeps the comparison exact
    /// instead of merely lenient.
    padding: Option<Vec<core::ops::Range<usize>>>,
}

impl<'a> Encoder<'a> {
    pub fn new(buf: &'a mut [u8], protocol: &'a dyn Protocol) -> Self {
        Encoder { buf, pos: 0, fatal: false, protocol, padding: None }
    }

    /// Start recording padded byte ranges. See the `padding` field.
    pub fn record_padding(&mut self) {
        self.padding = Some(Vec::new());
    }

    pub fn padding(&self) -> &[core::ops::Range<usize>] {
        self.padding.as_deref().unwrap_or(&[])
    }

    fn note_padding(&mut self, from: usize, to: usize) {
        if from < to
            && let Some(p) = &mut self.padding
        {
            p.push(from..to);
        }
    }

    pub fn protocol(&self) -> &dyn Protocol {
        self.protocol
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn fatal(&self) -> bool {
        self.fatal
    }

    /// The bytes written so far.
    pub fn written(&self) -> &[u8] {
        &self.buf[..self.pos]
    }

    /// Write `val` and advance `advance` bytes, zero-filling the padding. `advance` is the wire
    /// size; `val` is the payload.
    pub fn write(&mut self, advance: usize, val: &[u8]) {
        debug_assert!(val.len() <= advance);
        let Some(dst) = self.buf.get_mut(self.pos..self.pos + advance) else {
            self.fatal = true;
            return;
        };
        dst[..val.len()].copy_from_slice(val);
        dst[val.len()..].fill(0);
        self.note_padding(self.pos + val.len(), self.pos + advance);
        self.pos += advance;
    }
}

/// Scalars on the wire.
///
/// Every scalar occupies a multiple of four bytes -- a `u8` costs four, a `u64` costs eight -- and
/// arrays of sub-word scalars are packed and then padded as a whole. `vn_sizeof_*` in the C
/// generator is this rule; implementing it once here keeps the generated code free of it.
pub trait Scalar: Copy + Default {
    fn from_le_bytes(b: &[u8]) -> Self;
    fn write_le(self, out: &mut [u8]);
}

macro_rules! scalar {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            fn from_le_bytes(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b.try_into().expect("caller sized the slice"))
            }
            fn write_le(self, out: &mut [u8]) {
                out.copy_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}

// `usize` is here because vk.xml has `size_t` members. It is eight bytes on every target
// this renderer supports, which the wire assumes.
scalar!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);

/// Wire size of a single scalar: padded up to a word.
#[inline]
pub const fn sizeof_scalar<T: Scalar>() -> usize {
    if size_of::<T>() >= 4 { size_of::<T>() } else { 4 }
}

/// Wire size of a packed scalar array, padded as a whole.
#[inline]
pub fn sizeof_scalar_array<T: Scalar>(count: usize) -> usize {
    align4(size_of::<T>() * count)
}

impl<'a> Decoder<'a> {
    pub fn decode_scalar<T: Scalar>(&mut self) -> T {
        match self.read_bytes(sizeof_scalar::<T>(), size_of::<T>()) {
            Some(b) => T::from_le_bytes(b),
            None => T::default(),
        }
    }

    pub fn peek_scalar<T: Scalar>(&self) -> T {
        match self.peek_bytes(sizeof_scalar::<T>()) {
            Some(b) => T::from_le_bytes(&b[..size_of::<T>()]),
            None => T::default(),
        }
    }

    /// Fill `out` from the stream. On a short stream `out` is left zeroed and the stream poisoned,
    /// so the caller never reads uninitialised data.
    #[allow(clippy::chunks_exact_to_as_chunks)] // the chunk size is generic, not const
    pub fn decode_scalar_array<T: Scalar>(&mut self, out: &mut [T]) {
        let packed = size_of_val(out);
        let Some(b) = self.read_bytes(align4(packed), packed) else {
            out.fill(T::default());
            return;
        };
        for (dst, src) in out.iter_mut().zip(b.chunks_exact(size_of::<T>())) {
            *dst = T::from_le_bytes(src);
        }
    }
}

impl<'a> Encoder<'a> {
    pub fn encode_scalar<T: Scalar>(&mut self, val: T) {
        let mut b = [0u8; 8];
        val.write_le(&mut b[..size_of::<T>()]);
        self.write(sizeof_scalar::<T>(), &b[..size_of::<T>()]);
    }

    #[allow(clippy::chunks_exact_to_as_chunks)] // the chunk size is generic, not const
    pub fn encode_scalar_array<T: Scalar>(&mut self, vals: &[T]) {
        let packed = size_of_val(vals);
        let Some(dst) = self.buf.get_mut(self.pos..self.pos + align4(packed)) else {
            self.fatal = true;
            return;
        };
        for (chunk, v) in dst.chunks_exact_mut(size_of::<T>()).zip(vals) {
            v.write_le(chunk);
        }
        dst[packed..].fill(0);
        self.note_padding(self.pos + packed, self.pos + align4(packed));
        self.pos += align4(packed);
    }
}

/// A pointer on the wire is an array size of one or zero: present or not. The pointee follows only
/// if present, so it costs the same eight bytes an element count does.
impl<'a> Decoder<'a> {
    pub fn decode_simple_pointer(&mut self) -> bool {
        self.decode_array_size_unchecked() != 0
    }

    /// An array's element count, as the guest claims it. Nothing here validates it against what the
    /// command said the count should be -- that is `decode_array_size`'s job.
    pub fn peek_array_size(&self) -> u64 {
        self.peek_scalar::<u64>()
    }

    pub fn decode_array_size_unchecked(&mut self) -> u64 {
        self.decode_scalar::<u64>()
    }

    /// An array's element count, checked against the count the command already gave us. A guest
    /// that disagrees with itself is poisoned rather than trusted: this is the check that keeps an
    /// oversized array from being allocated and filled.
    pub fn decode_array_size(&mut self, expected: u64) -> u64 {
        let size = self.decode_scalar::<u64>();
        if size != expected {
            self.set_fatal();
            return 0;
        }
        size
    }

    /// `size` bytes of opaque payload, borrowed from the stream rather than copied. The renderer
    /// only ever reads a blob, and the stream outlives the command that named it.
    pub fn decode_blob(&mut self, size: usize) -> Option<&'a [u8]> {
        self.read_bytes(align4(size), size)
    }

    /// `size` bytes of string, copied into the arena and forced NUL-terminated. The copy is what
    /// buys the terminator: a guest that sends an unterminated string gets it truncated here
    /// rather than run off the end of the stream in whatever host call receives it.
    pub fn decode_c_string(&mut self, size: usize) -> Option<&'a mut [u8]> {
        if size == 0 {
            self.set_fatal();
            return None;
        }
        let bytes = self.read_bytes(align4(size), size)?;
        let out = self.alloc_temp_array::<u8>(size)?;
        out.copy_from_slice(bytes);
        out[size - 1] = 0;
        Some(out)
    }
}

/// The wire size of `size` bytes of opaque payload: padded to a word, with no count of its own.
pub const fn sizeof_blob(size: usize) -> usize {
    align4(size)
}

/// The length a NUL-terminated string occupies on the wire, terminator included.
///
/// # Safety
/// `p` must point at a NUL-terminated string that outlives the call.
pub unsafe fn c_string_len(p: *const core::ffi::c_char) -> usize {
    let mut n = 0;
    // SAFETY: the caller promises a terminator, so the walk stops inside the allocation.
    while unsafe { *p.add(n) } != 0 {
        n += 1;
    }
    n + 1
}

impl<'a> Encoder<'a> {
    pub fn encode_simple_pointer(&mut self, present: bool) -> bool {
        self.encode_array_size(present as u64);
        present
    }

    pub fn encode_array_size(&mut self, count: u64) {
        self.encode_scalar::<u64>(count);
    }

    /// Opaque payload, padded to a word. The count, when the wire carries one, is the caller's.
    pub fn encode_blob(&mut self, val: &[u8]) {
        self.write(align4(val.len()), val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_are_word_padded_on_the_wire() {
        assert_eq!(sizeof_scalar::<u8>(), 4);
        assert_eq!(sizeof_scalar::<u16>(), 4);
        assert_eq!(sizeof_scalar::<u32>(), 4);
        assert_eq!(sizeof_scalar::<u64>(), 8);
        assert_eq!(sizeof_scalar_array::<u8>(5), 8);
        assert_eq!(sizeof_scalar_array::<u32>(3), 12);
    }

    #[test]
    fn a_short_stream_poisons_instead_of_panicking() {
        let temp = Bump::new();
        let hard = Cell::new(false);
        let buf = [1u8, 0, 0, 0];
        let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        assert_eq!(dec.decode_scalar::<u32>(), 1);
        assert!(!dec.fatal());
        assert_eq!(dec.decode_scalar::<u32>(), 0);
        assert!(dec.hard_fatal());
    }

    #[test]
    fn an_array_size_the_guest_disagrees_with_poisons() {
        let temp = Bump::new();
        let hard = Cell::new(false);
        let buf = 7u64.to_le_bytes();
        let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        assert_eq!(dec.decode_array_size(3), 0);
        assert!(dec.hard_fatal());
    }

    #[test]
    fn an_implausible_array_size_is_refused_before_it_is_allocated() {
        let temp = Bump::new();
        let hard = Cell::new(false);
        let buf = [0u8; 0];
        let dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        assert!(dec.alloc_temp_array::<u32>(usize::MAX / 2).is_none());
        assert!(dec.hard_fatal());
    }

    #[test]
    fn a_ghost_object_poisons_the_command_and_not_the_ring() {
        struct Empty;
        impl Objects for Empty {
            fn lookup(&self, _id: ObjectId, _ty: i32) -> Option<u64> {
                None
            }
        }
        let temp = Bump::new();
        let hard = Cell::new(false);
        let buf = [0u8; 0];
        let dec = Decoder::new(&buf, &temp, &Empty, &hard);
        assert_eq!(dec.lookup_object(ObjectId(42), 0), 0);
        assert!(dec.fatal());
        assert!(!dec.hard_fatal());
        dec.clear_soft_fatal();
        assert!(!dec.fatal());
    }

    #[test]
    fn scalar_round_trip_reproduces_the_wire_including_padding() {
        let mut buf = [0xaau8; 16];
        let mut enc = Encoder::new(&mut buf, &AllOfIt);
        enc.encode_scalar::<u8>(0x5a);
        enc.encode_scalar::<u64>(0x0102_0304_0506_0708);
        enc.encode_scalar_array::<u8>(&[1, 2, 3]);
        assert_eq!(enc.pos(), 16);
        assert_eq!(&buf[..4], &[0x5a, 0, 0, 0]);
        assert_eq!(&buf[12..], &[1, 2, 3, 0]);
    }
}
