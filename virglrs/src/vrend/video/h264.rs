// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! H.264 framing and parameter-set synthesis for the VideoToolbox backend.
//!
//! Written against ITU-T H.264: 7.3.2.1.1 for the SPS, 7.3.2.2 for the PPS, 7.3.3 for the slice
//! header, 7.4.1.1 for emulation prevention.

use super::bitstream::{NalUnits, Reader};

/// NAL unit types that carry a slice header (7.4.1, `nal_unit_type`).
const NAL_SLICE_NON_IDR: u8 = 1;
const NAL_SLICE_IDR: u8 = 5;

/// Rewrite an Annex-B stream as AVCC: each NAL prefixed by its big-endian 32-bit length.
///
/// `None` when the input carries no start code at all, which means it is not Annex-B and we were
/// handed something other than what we expected. There is nothing to guess at there: an AVCC
/// length read off a stream that was never framed decodes as garbage of an arbitrary size.
///
/// A stream that frames nothing is a different answer -- an empty rewrite, which is what it is.
pub fn annexb_to_avcc(annexb: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(annexb.len());
    for nal in NalUnits::new(annexb)? {
        let len = u32::try_from(nal.unit.len()).expect("a NAL longer than 4 GiB");
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nal.unit);
    }
    Some(out)
}

/// The `pic_parameter_set_id` the guest's slices reference, read out of the first slice header.
///
/// Nothing on the wire says which id the guest chose, and a PPS bearing the wrong one is simply
/// not found -- so it is parsed back rather than assumed. It is the third `ue(v)` in the slice
/// header (7.3.3), after `first_mb_in_slice` and `slice_type`.
///
/// The header is read from the whole remaining stream rather than from the one NAL. Emulation
/// prevention keeps a start code out of a payload, so for any stream that is not already
/// malformed the two bounds hold the same bytes.
pub fn slice_pps_id(annexb: &[u8]) -> Option<u32> {
    for nal in NalUnits::new(annexb)? {
        // Other types -- SEI, AUD, parameter sets -- do not carry the header we want.
        let kind = nal.unit[0] & 0x1f;
        if kind != NAL_SLICE_NON_IDR && kind != NAL_SLICE_IDR {
            continue;
        }

        let mut r = Reader::new(&nal.onward[1..]);
        let _first_mb_in_slice = r.ue()?;
        let _slice_type = r.ue()?;
        let pps_id = r.ue()?;

        // 7.4.2.2: pic_parameter_set_id is 0..255.
        return (pps_id <= 255).then_some(pps_id);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rewrite_replaces_each_start_code_with_the_length_that_follows_it() {
        let stream = [0, 0, 0, 1, 0x67, 0xaa, 0xbb, 0, 0, 1, 0x68, 0xcc];
        assert_eq!(
            annexb_to_avcc(&stream).unwrap(),
            vec![0, 0, 0, 3, 0x67, 0xaa, 0xbb, 0, 0, 0, 2, 0x68, 0xcc]
        );
    }

    #[test]
    fn a_stream_that_was_never_framed_is_refused_rather_than_guessed_at() {
        assert!(annexb_to_avcc(&[0x67, 0xaa, 0xbb]).is_none());
        assert!(slice_pps_id(&[0x65, 0x88, 0x84]).is_none());
        // Framed, but framing nothing: an empty rewrite, not a refusal.
        assert_eq!(annexb_to_avcc(&[0, 0, 1]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn the_pps_id_is_read_past_the_headers_that_do_not_carry_one() {
        // An SEI (type 6), then an IDR slice (type 5) whose header is
        // first_mb_in_slice = 0 (1), slice_type = 7 (0001000), pic_parameter_set_id = 2 (00011).
        let mut header =
            super::super::bitstream::Writer::new(super::super::bitstream::Escape::Rbsp);
        header.raw_byte(0x65);
        header.ue(0);
        header.ue(7);
        header.ue(2);
        header.rbsp_trailing();

        let mut stream = vec![0, 0, 1, 0x06, 0xff, 0x80, 0, 0, 1];
        stream.extend_from_slice(&header.finish());
        assert_eq!(slice_pps_id(&stream), Some(2));
    }

    #[test]
    fn an_id_outside_the_range_the_spec_allows_is_refused() {
        let mut header =
            super::super::bitstream::Writer::new(super::super::bitstream::Escape::Rbsp);
        header.raw_byte(0x65);
        header.ue(0);
        header.ue(7);
        header.ue(256);
        header.rbsp_trailing();

        let mut stream = vec![0, 0, 1];
        stream.extend_from_slice(&header.finish());
        assert_eq!(slice_pps_id(&stream), None);
    }
}

/// Diff the framing against the C it was ported from.
#[cfg(all(test, feature = "video-oracle"))]
mod oracle {
    use super::*;

    unsafe extern "C" {
        fn virgl_h264_slice_pps_id(annexb: *const u8, len: usize, out_id: *mut u32) -> i32;
        fn virgl_h264_annexb_to_avcc(
            input: *const u8,
            in_len: usize,
            out: *mut u8,
            out_cap: usize,
        ) -> isize;
    }

    fn c_annexb_to_avcc(annexb: &[u8]) -> Option<Vec<u8>> {
        // Four bytes of prefix per NAL, and a NAL is at least one byte past a three-byte start
        // code, so the rewrite is never longer than twice the input plus a prefix.
        let mut out = vec![0u8; 2 * annexb.len() + 8];
        // SAFETY: both slices are live, each with its length passed beside it as the C entry point
        // takes them; it writes at most `out_cap` bytes and reads at most `in_len`.
        let n = unsafe {
            virgl_h264_annexb_to_avcc(annexb.as_ptr(), annexb.len(), out.as_mut_ptr(), out.len())
        };
        if n < 0 {
            return None;
        }
        out.truncate(n as usize);
        Some(out)
    }

    fn c_slice_pps_id(annexb: &[u8]) -> Option<u32> {
        let mut id = 0u32;
        // SAFETY: `annexb` is a live slice with its length beside it, and `id` is a live u32 the C
        // writes only on success. It reads only.
        let rc = unsafe { virgl_h264_slice_pps_id(annexb.as_ptr(), annexb.len(), &mut id) };
        (rc == 0).then_some(id)
    }

    /// A deterministic source, so a failure names a stream that can be re-run.
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

    /// A stream built out of the bytes the framing turns on -- zeros, ones, start codes and slice
    /// headers -- so that the walk is actually exercised. Uniform random bytes contain a start
    /// code about once in sixteen million and would diff an empty walk over and over.
    fn stream(rng: &mut Rng, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            match rng.below(8) {
                0..=2 => out.push(0),
                3 => out.push(1),
                4 => out.extend_from_slice(&[0, 0, 1]),
                5 => out.extend_from_slice(&[0, 0, 0, 1]),
                // A NAL header, weighted towards the slice types the pps_id walk looks for.
                6 => out.push(0x60 | (rng.below(8) as u8)),
                _ => out.push(rng.next() as u8),
            }
        }
        out.truncate(len);
        out
    }

    #[test]
    fn the_rewrite_is_the_one_the_c_writes() {
        let mut rng = Rng(0xa7cc);
        for len in 0..600 {
            let s = stream(&mut rng, len);
            assert_eq!(annexb_to_avcc(&s), c_annexb_to_avcc(&s), "{s:02x?}");
        }
    }

    #[test]
    fn the_pps_id_is_the_one_the_c_finds() {
        let mut rng = Rng(0x9955);
        for len in 0..600 {
            let s = stream(&mut rng, len);
            assert_eq!(slice_pps_id(&s), c_slice_pps_id(&s), "{s:02x?}");
        }
    }

    #[test]
    fn a_real_slice_header_reads_the_same_id_on_both_sides() {
        use super::super::bitstream::{Escape, Writer};

        for id in [0u32, 1, 7, 63, 255] {
            let mut w = Writer::new(Escape::Rbsp);
            w.raw_byte(0x65);
            w.ue(0);
            w.ue(7);
            w.ue(id);
            w.rbsp_trailing();

            let mut s = vec![0, 0, 0, 1];
            s.extend_from_slice(&w.finish());
            assert_eq!(slice_pps_id(&s), Some(id));
            assert_eq!(c_slice_pps_id(&s), Some(id));
        }
    }
}
