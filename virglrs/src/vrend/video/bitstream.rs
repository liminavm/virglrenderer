// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Bit-level reading and writing for the parameter sets the VideoToolbox backend synthesizes.
//!
//! Nothing here knows about a codec: the H.264 and H.265 serializers own their syntax, this owns
//! the bits underneath it -- the same Exp-Golomb elements over the same RBSP escaping rules.

/// Whether a writer inserts emulation-prevention bytes.
///
/// An RBSP may not contain `00 00 00`, `00 00 01`, `00 00 02` or `00 00 03`, because a decoder
/// scanning for start codes could not tell the payload from the framing. `Rbsp` inserts the `03`
/// that keeps the payload distinguishable; `Raw` is for the boxes that are never start-code framed
/// and so must be handed on exactly as written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Escape {
    Rbsp,
    Raw,
}

/// A bit writer over a growing buffer.
///
/// The buffer grows, so there is no overflow to report and no capacity for a caller to have got
/// wrong -- a fixed buffer with a flag beside it is two values that must agree, and one of them is
/// a C-ism the port does not owe anything to.
pub struct Writer {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
    /// Consecutive zero bytes emitted, for the escaping rule above.
    zeros: u32,
    escape: Escape,
}

impl Writer {
    pub fn new(escape: Escape) -> Self {
        Self { out: Vec::new(), acc: 0, nbits: 0, zeros: 0, escape }
    }

    /// The bytes written so far. Bits still in the accumulator are not among them; a caller that
    /// wants them flushed ends with [`Writer::rbsp_trailing`].
    pub fn finish(self) -> Vec<u8> {
        debug_assert_eq!(self.nbits, 0, "a partial byte was left in the accumulator");
        self.out
    }

    /// How many bytes are written. Used by the serializers to fill in a length they wrote earlier.
    pub fn bytes_written(&self) -> usize {
        self.out.len()
    }

    /// Emit a byte with no escaping and no effect on the zero run -- for framing a payload rather
    /// than writing one: a start code, a NAL header, a box tag.
    pub fn raw_byte(&mut self, b: u8) {
        self.out.push(b);
    }

    /// Emit one payload byte, escaping it where the RBSP rules require.
    fn byte(&mut self, b: u8) {
        if self.escape == Escape::Rbsp && self.zeros >= 2 && b <= 0x03 {
            self.out.push(0x03);
            self.zeros = 0;
        }
        self.out.push(b);
        self.zeros = if b == 0 { self.zeros + 1 } else { 0 };
    }

    /// `u(n)` / `f(n)`: `n` bits of `v`, most significant first.
    pub fn u(&mut self, n: u32, v: u32) {
        assert!(n <= 32, "u({n}) is wider than the value it writes");
        for i in 0..n {
            self.acc = (self.acc << 1) | ((v >> (n - 1 - i)) & 1);
            self.nbits += 1;
            if self.nbits == 8 {
                let b = self.acc as u8;
                self.byte(b);
                self.acc = 0;
                self.nbits = 0;
            }
        }
    }

    pub fn flag(&mut self, v: bool) {
        self.u(1, u32::from(v));
    }

    /// `ue(v)`: Exp-Golomb. `v` is written as a prefix of N zeros, a 1, then N bits.
    pub fn ue(&mut self, v: u32) {
        // 64-bit: v == u32::MAX would carry past 32 bits.
        let x = u64::from(v) + 1;
        let n = 63 - x.leading_zeros();

        self.u(n, 0);
        self.u(1, 1);
        if n != 0 {
            self.u(n, (x & ((1u64 << n) - 1)) as u32);
        }
    }

    /// `se(v)`: signed Exp-Golomb, mapped 0, 1, -1, 2, -2, ...
    pub fn se(&mut self, v: i32) {
        let code = if v <= 0 { -2 * i64::from(v) } else { 2 * i64::from(v) - 1 };
        self.ue(code as u32);
    }

    /// `rbsp_trailing_bits()`: a 1 bit, then zeros to the byte boundary.
    pub fn rbsp_trailing(&mut self) {
        self.flag(true);
        while self.nbits != 0 {
            self.flag(false);
        }
    }
}

/// A bit reader over an RBSP, stripping emulation-prevention bytes as it goes.
///
/// Every read returns `None` at truncation rather than a sentinel, so a caller cannot read past
/// the end by forgetting to check -- there is no value to mistake for one.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    bit: u32,
    zeros: u32,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0, bit: 0, zeros: 0 }
    }

    pub fn bit(&mut self) -> Option<u32> {
        if self.pos >= self.buf.len() {
            return None;
        }

        // An emulation-prevention 0x03 after two zero bytes is not part of the RBSP.
        if self.bit == 0 && self.zeros >= 2 && self.buf[self.pos] == 0x03 {
            self.pos += 1;
            self.zeros = 0;
            if self.pos >= self.buf.len() {
                return None;
            }
        }

        let byte = self.buf[self.pos];
        let v = u32::from((byte >> (7 - self.bit)) & 1);
        self.bit += 1;
        if self.bit == 8 {
            self.zeros = if byte == 0 { self.zeros + 1 } else { 0 };
            self.bit = 0;
            self.pos += 1;
        }
        Some(v)
    }

    /// `u(n)`: `n` bits, most significant first.
    pub fn u(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Some(v)
    }

    /// `ue(v)`: Exp-Golomb.
    pub fn ue(&mut self) -> Option<u32> {
        let mut n = 0;
        while self.bit()? == 0 {
            n += 1;
            if n > 32 {
                return None;
            }
        }

        let mut v = 1u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Some(v.wrapping_sub(1))
    }

    /// `se(v)`: signed Exp-Golomb, mapped 0, 1, -1, 2, -2, ...
    pub fn se(&mut self) -> Option<i32> {
        // Wrapping, not saturating: an Exp-Golomb prefix of 32 zeros can hand back a codeNum
        // no signed value maps to. The C wraps there, and a stream that reaches it is already
        // malformed -- what matters is that both sides read the same nonsense and move on.
        let k = self.ue()?;
        Some(if k & 1 != 0 {
            (k.wrapping_add(1) / 2) as i32
        } else {
            ((k / 2) as i32).wrapping_neg()
        })
    }
}

/// One NAL unit, and the stream from where it starts.
///
/// The two views begin at the same byte and are both wanted: framing a NAL needs to know where it
/// ends, and parsing a slice header does not -- emulation prevention keeps a start code out of a
/// payload, so within a well-formed stream reading to either bound reads the same bytes.
pub struct Nal<'a> {
    /// The unit itself, ending where the next start code begins.
    pub unit: &'a [u8],
    /// The unit and everything after it, to the end of the stream.
    pub onward: &'a [u8],
}

/// The NAL units of an Annex-B stream, in order.
///
/// A NAL runs from just past its start code to the byte before the next one, trailing zeros
/// included -- the framing does not say where the payload's own zeros stop, so neither does this.
/// Empty units are skipped: back-to-back start codes frame nothing.
pub struct NalUnits<'a> {
    rest: &'a [u8],
}

impl<'a> NalUnits<'a> {
    /// The units of `annexb`, or `None` if it holds no start code at all and so is not Annex-B.
    ///
    /// The distinction is the caller's to act on: a stream that frames nothing is empty, and one
    /// that is not framed at all is a stream we were handed in a format we did not expect. Guessing
    /// between them is how a decoder ends up playing garbage.
    pub fn new(annexb: &'a [u8]) -> Option<Self> {
        let (_, after) = split_at_start_code(annexb)?;
        Some(Self { rest: after })
    }
}

impl<'a> Iterator for NalUnits<'a> {
    type Item = Nal<'a>;

    fn next(&mut self) -> Option<Nal<'a>> {
        loop {
            if self.rest.is_empty() {
                return None;
            }
            let onward = self.rest;
            let (unit, after) = match split_at_start_code(onward) {
                Some((before, after)) => (before, after),
                None => (onward, &onward[onward.len()..]),
            };
            self.rest = after;
            if !unit.is_empty() {
                return Some(Nal { unit, onward });
            }
        }
    }
}

/// Split at the first start code: everything before it, and everything after it.
///
/// `None` when there is none, in which case nothing is consumed.
fn split_at_start_code(buf: &[u8]) -> Option<(&[u8], &[u8])> {
    for i in 0..buf.len().saturating_sub(2) {
        if buf[i] != 0 || buf[i + 1] != 0 {
            continue;
        }
        if buf[i + 2] == 1 {
            return Some((&buf[..i], &buf[i + 3..]));
        }
        if i + 3 < buf.len() && buf[i + 2] == 0 && buf[i + 3] == 1 {
            return Some((&buf[..i], &buf[i + 4..]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_writes_the_codes_the_spec_tabulates() {
        // H.264 9.1: 1, 010, 011, 00100, 00101, 00110, 00111, 0001000, ...
        let mut w = Writer::new(Escape::Rbsp);
        for v in 0..8 {
            w.ue(v);
        }
        w.rbsp_trailing();
        let bytes = w.finish();

        let mut r = Reader::new(&bytes);
        for v in 0..8 {
            assert_eq!(r.ue(), Some(v));
        }
    }

    #[test]
    fn signed_exp_golomb_alternates_around_zero() {
        for (v, code) in [(0, 0), (1, 1), (-1, 2), (2, 3), (-2, 4), (3, 5)] {
            let mut w = Writer::new(Escape::Rbsp);
            w.se(v);
            w.rbsp_trailing();
            let bytes = w.finish();
            assert_eq!(Reader::new(&bytes).ue(), Some(code), "se({v})");
            assert_eq!(Reader::new(&bytes).se(), Some(v));
        }
    }

    #[test]
    fn a_payload_that_looks_like_a_start_code_is_escaped_and_unescaped() {
        let mut w = Writer::new(Escape::Rbsp);
        for b in [0u8, 0, 1, 0, 0, 2] {
            w.u(8, u32::from(b));
        }
        w.rbsp_trailing();
        let bytes = w.finish();
        assert_eq!(&bytes[..8], &[0, 0, 3, 1, 0, 0, 3, 2]);

        let mut r = Reader::new(&bytes);
        for b in [0u32, 0, 1, 0, 0, 2] {
            assert_eq!(r.u(8), Some(b));
        }
    }

    #[test]
    fn a_raw_byte_carries_no_escaping_and_starts_no_zero_run() {
        let mut w = Writer::new(Escape::Rbsp);
        w.raw_byte(0);
        w.raw_byte(0);
        w.u(8, 1);
        w.rbsp_trailing();
        assert_eq!(&w.finish()[..3], &[0, 0, 1]);
    }

    #[test]
    fn reading_past_the_end_is_none_rather_than_a_value() {
        let mut r = Reader::new(&[0x80]);
        assert_eq!(r.u(8), Some(0x80));
        assert_eq!(r.bit(), None);
        assert_eq!(r.ue(), None);
        assert_eq!(r.se(), None);
    }

    #[test]
    fn a_units_onward_view_reaches_the_end_of_the_stream() {
        let stream = [0, 0, 1, 0x67, 0xaa, 0, 0, 1, 0x68];
        let first = NalUnits::new(&stream).unwrap().next().unwrap();
        assert_eq!(first.unit, &[0x67, 0xaa]);
        assert_eq!(first.onward, &stream[3..]);
    }

    #[test]
    fn a_nal_runs_to_the_next_start_code_trailing_zeros_and_all() {
        // The scan takes the earliest start code, so the four-byte form claims its own leading
        // zero and a fifth zero before it stays in the NAL. Trailing zeros belong to the payload
        // wherever the framing does not need them.
        let stream = [0, 0, 1, 0x67, 0xaa, 0, 0, 0, 0, 1, 0x68, 0xbb];
        let nals: Vec<_> = NalUnits::new(&stream).unwrap().map(|n| n.unit).collect();
        assert_eq!(nals, vec![&[0x67u8, 0xaa, 0][..], &[0x68, 0xbb][..]]);

        let stream = [0, 0, 0, 1, 0x67, 0xaa, 0, 0, 1, 0x68];
        let nals: Vec<_> = NalUnits::new(&stream).unwrap().map(|n| n.unit).collect();
        assert_eq!(nals, vec![&[0x67u8, 0xaa][..], &[0x68][..]]);
    }

    #[test]
    fn back_to_back_start_codes_frame_nothing_and_are_skipped() {
        let stream = [0, 0, 1, 0, 0, 1, 0x65];
        let nals: Vec<_> = NalUnits::new(&stream).unwrap().map(|n| n.unit).collect();
        assert_eq!(nals, vec![&[0x65u8][..]]);
    }

    #[test]
    fn a_stream_with_no_start_code_is_not_annex_b_at_all() {
        assert!(NalUnits::new(&[0x67, 0xaa, 0xbb]).is_none());
        assert!(NalUnits::new(&[]).is_none());
        // Framing nothing is still framing: that is a stream, and it is empty.
        assert_eq!(NalUnits::new(&[0, 0, 1]).unwrap().count(), 0);
    }
}

/// Diff the writer and reader against the C they were ported from, byte for byte.
///
/// The C is the only reference these have: there is no conformance vector for a synthesized
/// parameter set, only the bytes that have played. So agreement is the standard, and the way to
/// check it is to drive both sides from one script and compare what comes out.
#[cfg(all(test, feature = "video-oracle"))]
mod oracle {
    use super::*;

    const OP_RAW: u32 = 0;
    const OP_U: u32 = 1;
    const OP_FLAG: u32 = 2;
    const OP_UE: u32 = 3;
    const OP_SE: u32 = 4;
    const OP_TRAILING: u32 = 5;

    /// One step of a writer script, laid out as `tests/oracle/video_oracle.c` reads it.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Op {
        tag: u32,
        n: u32,
        value: i64,
    }

    unsafe extern "C" {
        fn virgl_oracle_bs_run(
            ops: *const Op,
            n_ops: usize,
            escape: i32,
            out: *mut u8,
            out_cap: usize,
        ) -> isize;
        fn virgl_oracle_br_run(
            buf: *const u8,
            len: usize,
            tags: *const u32,
            widths: *const u32,
            n_ops: usize,
            out: *mut i64,
        ) -> usize;
    }

    /// Run a script through the C writer.
    fn c_write(ops: &[Op], escape: Escape) -> Vec<u8> {
        let mut out = vec![0u8; 64 * 1024];
        // SAFETY: `ops` and `out` are live slices whose lengths are passed beside them, which is
        // the shape the C entry point takes; it writes at most `out_cap` bytes and reads exactly
        // `n_ops` elements. The capacity is far past what these scripts can emit, so the overflow
        // return -- which the growing Rust writer has no equivalent for -- cannot fire.
        let n = unsafe {
            virgl_oracle_bs_run(
                ops.as_ptr(),
                ops.len(),
                i32::from(escape == Escape::Rbsp),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert!(n >= 0, "the C writer overflowed a buffer the script cannot fill");
        out.truncate(n as usize);
        out
    }

    /// Run the same script through the Rust writer.
    fn rust_write(ops: &[Op], escape: Escape) -> Vec<u8> {
        let mut w = Writer::new(escape);
        for op in ops {
            match op.tag {
                OP_RAW => w.raw_byte(op.value as u8),
                OP_U => w.u(op.n, op.value as u32),
                OP_FLAG => w.flag(op.value != 0),
                OP_UE => w.ue(op.value as u32),
                OP_SE => w.se(op.value as i32),
                OP_TRAILING => w.rbsp_trailing(),
                other => panic!("unknown op {other}"),
            }
        }
        // The C reports `pos`, the bytes actually emitted; a partial accumulator is not among them
        // on either side, and every script ends byte-aligned so that no test turns on it.
        w.finish()
    }

    /// A small deterministic source, so a failure names a script that can be re-run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// A script that ends byte-aligned, so both sides have emitted every bit they were given.
    fn script(rng: &mut Rng, len: usize) -> Vec<Op> {
        let mut ops = Vec::with_capacity(len + 1);
        for _ in 0..len {
            ops.push(match rng.below(5) {
                0 => Op { tag: OP_RAW, n: 0, value: rng.below(256) as i64 },
                1 => {
                    let n = rng.below(33) as u32;
                    // A value wider than its field is the caller's bug, not a case to diff.
                    let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
                    Op { tag: OP_U, n, value: i64::from(rng.next() as u32 & mask) }
                }
                2 => Op { tag: OP_FLAG, n: 0, value: rng.below(2) as i64 },
                // u32::MAX is excluded: the C shifts by 32 to write it, which is undefined there.
                3 => Op { tag: OP_UE, n: 0, value: rng.below(u64::from(u32::MAX)) as i64 },
                _ => Op { tag: OP_SE, n: 0, value: i64::from(rng.next() as i32 / 2) },
            });
        }
        ops.push(Op { tag: OP_TRAILING, n: 0, value: 0 });
        ops
    }

    #[test]
    fn the_writer_emits_the_bytes_the_c_emits() {
        let mut rng = Rng(0x5eed);
        for escape in [Escape::Rbsp, Escape::Raw] {
            for len in 0..200 {
                let ops = script(&mut rng, len);
                assert_eq!(
                    rust_write(&ops, escape),
                    c_write(&ops, escape),
                    "{escape:?}, {len} ops"
                );
            }
        }
    }

    #[test]
    fn the_writer_escapes_a_deliberate_flood_of_zeros_the_way_the_c_does() {
        // Random bytes almost never produce a `00 00 0x`; this is the case the escaping exists
        // for, so it gets a script that is nothing else.
        let mut rng = Rng(0x2e40_5eed);
        let mut ops = Vec::new();
        for _ in 0..400 {
            ops.push(Op { tag: OP_U, n: 8, value: rng.below(5) as i64 });
        }
        ops.push(Op { tag: OP_TRAILING, n: 0, value: 0 });
        assert_eq!(rust_write(&ops, Escape::Rbsp), c_write(&ops, Escape::Rbsp));
        assert_eq!(rust_write(&ops, Escape::Raw), c_write(&ops, Escape::Raw));
    }

    #[test]
    fn the_reader_returns_what_the_c_reader_returns_and_stops_where_it_stops() {
        let mut rng = Rng(0xdec0de);
        for len in 0..200 {
            // Write a script, then read it back with a different, arbitrary shape -- including
            // shapes that run off the end, which is where the two have to agree on stopping.
            let bytes = rust_write(&script(&mut rng, len), Escape::Rbsp);

            let n_ops = 1 + rng.below(40) as usize;
            let mut tags = Vec::with_capacity(n_ops);
            let mut widths = Vec::with_capacity(n_ops);
            for _ in 0..n_ops {
                match rng.below(3) {
                    0 => {
                        tags.push(OP_U);
                        widths.push(rng.below(33) as u32);
                    }
                    1 => {
                        tags.push(OP_UE);
                        widths.push(0);
                    }
                    _ => {
                        tags.push(OP_SE);
                        widths.push(0);
                    }
                }
            }

            let mut c_out = vec![0i64; n_ops];
            // SAFETY: every pointer is a live slice's, each with its length passed beside it as
            // the C entry point takes them; `out` has `n_ops` elements, which is the most it
            // writes. The C reads only, and the borrows outlive the call.
            let c_n = unsafe {
                virgl_oracle_br_run(
                    bytes.as_ptr(),
                    bytes.len(),
                    tags.as_ptr(),
                    widths.as_ptr(),
                    n_ops,
                    c_out.as_mut_ptr(),
                )
            };

            let mut r = Reader::new(&bytes);
            let mut rust_out = Vec::new();
            for (&tag, &width) in tags.iter().zip(widths.iter()) {
                let v = match tag {
                    OP_U => r.u(width).map(i64::from),
                    OP_UE => r.ue().map(i64::from),
                    _ => r.se().map(i64::from),
                };
                match v {
                    Some(v) => rust_out.push(v),
                    None => break,
                }
            }

            assert_eq!(rust_out.len(), c_n, "stopped at a different op, {len} ops written");
            assert_eq!(&rust_out[..], &c_out[..c_n], "{len} ops written");
        }
    }
}
