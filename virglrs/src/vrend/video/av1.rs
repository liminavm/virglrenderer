// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Rebuild AV1 OBUs from a parsed picture descriptor.
//!
//! A decoder that takes whole compressed frames -- VideoToolbox, and any other bitstream-in
//! decoder -- cannot use the VA-API-shaped picture the virgl protocol carries: the frame header
//! the encoder wrote was consumed and discarded by the guest's own parser long before the
//! descriptor was built. This turns the descriptor back into conformant bitstream.
//!
//! Written against the AV1 specification: 5.5 for the sequence header, 5.9 for the frame header.

use std::fmt;

use super::bitstream::{Escape, Writer};

/// OBU types (5.3.1).
const OBU_SEQUENCE_HEADER: u32 = 1;

/// Where a C bit-field sits in the descriptor: the storage unit that contains it, and the field's
/// position inside that unit.
///
/// Discovered by asking the compiler rather than by counting: the oracle sets each field to all
/// ones in a zeroed struct and reports which bits moved, and the differential compares the answer
/// to what is written here. Bit-field packing is ABI, not something a reader should assume.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Bits {
    pub at: usize,
    pub bytes: usize,
    pub shift: u32,
    pub width: u32,
}

/// Where each field sits in `struct virgl_av1_picture_desc`.
mod at {
    use super::Bits;

    pub const PROFILE: usize = 332;
    pub const ORDER_HINT_BITS_MINUS_1: usize = 333;
    pub const BIT_DEPTH_IDX: usize = 334;
    pub const FRAME_WIDTH: usize = 344;
    pub const FRAME_HEIGHT: usize = 346;
    pub const MAX_WIDTH: usize = 348;
    pub const MAX_HEIGHT: usize = 350;
    pub const SEQ_USE_128X128_SUPERBLOCK: Bits = Bits { at: 336, bytes: 4, shift: 0, width: 1 };
    pub const SEQ_ENABLE_FILTER_INTRA: Bits = Bits { at: 336, bytes: 4, shift: 1, width: 1 };
    pub const SEQ_ENABLE_INTRA_EDGE_FILTER: Bits = Bits { at: 336, bytes: 4, shift: 2, width: 1 };
    pub const SEQ_ENABLE_INTERINTRA_COMPOUND: Bits = Bits { at: 336, bytes: 4, shift: 3, width: 1 };
    pub const SEQ_ENABLE_MASKED_COMPOUND: Bits = Bits { at: 336, bytes: 4, shift: 4, width: 1 };
    pub const SEQ_ENABLE_DUAL_FILTER: Bits = Bits { at: 336, bytes: 4, shift: 5, width: 1 };
    pub const SEQ_ENABLE_ORDER_HINT: Bits = Bits { at: 336, bytes: 4, shift: 6, width: 1 };
    pub const SEQ_ENABLE_JNT_COMP: Bits = Bits { at: 336, bytes: 4, shift: 7, width: 1 };
    pub const SEQ_ENABLE_CDEF: Bits = Bits { at: 336, bytes: 4, shift: 8, width: 1 };
    pub const SEQ_MONO_CHROME: Bits = Bits { at: 336, bytes: 4, shift: 9, width: 1 };
    pub const SEQ_REF_FRAME_MVS: Bits = Bits { at: 336, bytes: 4, shift: 10, width: 1 };
    pub const SEQ_FILM_GRAIN_PARAMS_PRESENT: Bits = Bits { at: 336, bytes: 4, shift: 11, width: 1 };
}

/// Reads a descriptor's bytes. Total: a short descriptor reads as zeros, which is what the C does
/// and what the protocol allows -- the resource carries whatever the guest's driver wrote.
#[derive(Clone, Copy)]
struct Fields<'a>(&'a [u8]);

impl Fields<'_> {
    fn byte(&self, at: usize) -> u8 {
        self.0.get(at).copied().unwrap_or(0)
    }

    fn short(&self, at: usize) -> u16 {
        u16::from_le_bytes([self.byte(at), self.byte(at + 1)])
    }

    /// One bit-field, out of the storage unit that contains it.
    fn bits(&self, f: Bits) -> u32 {
        let mut unit = 0u64;
        for i in 0..f.bytes {
            unit |= u64::from(self.byte(f.at + i)) << (8 * i);
        }
        let mask = if f.width >= 32 { u32::MAX } else { (1u32 << f.width) - 1 };
        ((unit >> f.shift) as u32) & mask
    }

    fn flag(&self, f: Bits) -> bool {
        self.bits(f) != 0
    }
}

/// AV1's own syntax elements, over the shared bit writer.
///
/// The bits underneath are the same; what differs is the vocabulary above them -- and that AV1
/// never escapes, so every writer here is built with [`Escape::Raw`].
trait Av1Syntax {
    fn leb128(&mut self, v: u64);
}

impl Av1Syntax for Writer {
    fn leb128(&mut self, mut v: u64) {
        loop {
            let mut b = (v & 0x7f) as u32;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
            }
            self.u(8, b);
            if v == 0 {
                return;
            }
        }
    }
}

/// `FloorLog2`, which the spec defines as 0 for 0.
fn floor_log2(x: u32) -> u32 {
    if x == 0 { 0 } else { 31 - x.leading_zeros() }
}

/// An OBU: a header byte, a leb128 payload length, then the payload.
///
/// `obu_has_size_field` is always set. A decoder handed a temporal unit has to walk it, and the
/// sizes are the only framing that survives being concatenated.
fn emit_obu(w: &mut Writer, kind: u32, payload: &[u8]) {
    w.u(1, 0); // obu_forbidden_bit
    w.u(4, kind);
    w.u(1, 0); // obu_extension_flag
    w.u(1, 1); // obu_has_size_field
    w.u(1, 0); // obu_reserved_1bit
    w.leb128(payload.len() as u64);
    for &b in payload {
        w.u(8, u32::from(b));
    }
}

/// Why a descriptor could not be turned back into bitstream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unsupported {
    /// A profile other than 0 (Main). Only Main is in the capset, and only Main's `color_config`
    /// is written: profile 1 reads no `chroma_sample_position` and profile 2 at twelve bits reads
    /// a subsampling flag first, so writing Main's shape for either desynchronises the sequence
    /// header at its last field and everything the decoder does afterwards is wrong.
    Profile(u8),
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::Profile(p) => write!(f, "AV1 profile {p} is not supported (Main only)"),
        }
    }
}

/// What the sequence header says, derived from the picture descriptor.
///
/// Every frame repeats it, which is legal and saves tracking when a new one is owed. The fields
/// that gate symbol decoding *inside* the tiles are the reason this is derived rather than
/// invented: a permissive 1 for `enable_filter_intra` or `enable_cdef` silently corrupts every
/// tile, and the descriptor carries all of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SeqParams {
    pub profile: u8,
    pub level_idx: u8,
    /// 0 when order hints are off.
    pub order_hint_bits: u8,
    /// 8, 10 or 12.
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub use_128x128_superblock: bool,
    pub enable_order_hint: bool,
    pub enable_ref_frame_mvs: bool,
    pub film_grain_params_present: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_dual_filter: bool,
    pub enable_jnt_comp: bool,
    pub enable_cdef: bool,
    pub max_width: u16,
    pub max_height: u16,
    /// As coded, plus one: the field written is `width_bits - 1`.
    pub width_bits: u8,
    pub height_bits: u8,
}

/// The seq_level_idx limits (A.3), lowest first: pixels, width, height, and the level's index.
///
/// The level is not in the descriptor and does not affect decoding, only conformance signalling.
/// The lowest one whose limits cover the frame is picked, so a decoder that does enforce them is
/// not handed something it must reject. Levels are `(major - 2) * 4 + minor`.
const LEVELS: [(u64, u32, u32, u8); 8] = [
    (147_456, 2048, 1152, 0),      // 2.0
    (278_784, 2816, 1584, 1),      // 2.1
    (665_856, 4352, 2448, 4),      // 3.0
    (1_065_024, 5504, 3096, 5),    // 3.1
    (2_359_296, 6144, 3456, 8),    // 4.0
    (2_359_296, 6144, 3456, 9),    // 4.1
    (8_912_896, 8192, 4352, 12),   // 5.0
    (35_651_584, 16384, 8704, 16), // 6.0
];

/// 6.3 -- beyond this nothing we can decode exists anyway.
const LEVEL_MAX: u8 = 19;

fn pick_level(w: u32, h: u32) -> u8 {
    for &(pixels, max_w, max_h, idx) in &LEVELS {
        if w <= max_w && h <= max_h && u64::from(w) * u64::from(h) <= pixels {
            return idx;
        }
    }
    LEVEL_MAX
}

impl SeqParams {
    /// Derive the sequence parameters from a descriptor the guest wrote.
    pub fn read(blob: &[u8]) -> SeqParams {
        let d = Fields(blob);
        let max_width = match d.short(at::MAX_WIDTH) {
            0 => d.short(at::FRAME_WIDTH),
            w => w,
        };
        let max_height = match d.short(at::MAX_HEIGHT) {
            0 => d.short(at::FRAME_HEIGHT),
            h => h,
        };
        let enable_order_hint = d.flag(at::SEQ_ENABLE_ORDER_HINT);

        SeqParams {
            profile: d.byte(at::PROFILE),
            level_idx: pick_level(u32::from(max_width), u32::from(max_height)),
            order_hint_bits: if enable_order_hint {
                d.byte(at::ORDER_HINT_BITS_MINUS_1) + 1
            } else {
                0
            },
            bit_depth: match d.byte(at::BIT_DEPTH_IDX) {
                1 => 10,
                2 => 12,
                _ => 8,
            },
            mono_chrome: d.flag(at::SEQ_MONO_CHROME),
            use_128x128_superblock: d.flag(at::SEQ_USE_128X128_SUPERBLOCK),
            enable_order_hint,
            enable_ref_frame_mvs: d.flag(at::SEQ_REF_FRAME_MVS),
            film_grain_params_present: d.flag(at::SEQ_FILM_GRAIN_PARAMS_PRESENT),
            enable_filter_intra: d.flag(at::SEQ_ENABLE_FILTER_INTRA),
            enable_intra_edge_filter: d.flag(at::SEQ_ENABLE_INTRA_EDGE_FILTER),
            enable_interintra_compound: d.flag(at::SEQ_ENABLE_INTERINTRA_COMPOUND),
            enable_masked_compound: d.flag(at::SEQ_ENABLE_MASKED_COMPOUND),
            enable_dual_filter: d.flag(at::SEQ_ENABLE_DUAL_FILTER),
            enable_jnt_comp: d.flag(at::SEQ_ENABLE_JNT_COMP),
            enable_cdef: d.flag(at::SEQ_ENABLE_CDEF),
            max_width,
            max_height,
            width_bits: floor_log2(if max_width != 0 { u32::from(max_width) - 1 } else { 1 }) as u8
                + 1,
            height_bits: floor_log2(if max_height != 0 { u32::from(max_height) - 1 } else { 1 })
                as u8
                + 1,
        }
    }

    /// 5.5.2 `color_config`.
    ///
    /// The descriptor carries `matrix_coefficients` but not primaries or transfer, and none of the
    /// three changes a decoded sample -- they are display metadata -- so the description is marked
    /// absent, which defaults all three to unspecified. `separate_uv_delta_q` is set because the
    /// descriptor really does carry independent U and V quantizer deltas, and there would
    /// otherwise be no way to express them.
    ///
    /// Profile 0 only, which [`SeqParams::sequence_header`] has established: 4:2:0, so both
    /// subsampling flags are inferred and neither is written.
    fn write_color_config(&self, w: &mut Writer) {
        w.flag(self.bit_depth > 8); // high_bitdepth
        w.flag(self.mono_chrome);
        w.flag(false); // color_description_present_flag
        w.flag(false); // color_range: studio swing
        if self.mono_chrome {
            return;
        }
        w.u(2, 0); // chroma_sample_position: unknown
        w.flag(true); // separate_uv_delta_q
    }

    /// 5.5.1 `sequence_header_obu`, payload only.
    fn write(&self, w: &mut Writer) {
        w.u(3, u32::from(self.profile));
        w.flag(false); // still_picture
        w.flag(false); // reduced_still_picture_header
        w.flag(false); // timing_info_present_flag
        w.flag(false); // initial_display_delay_present_flag
        w.u(5, 0); // operating_points_cnt_minus_1
        w.u(12, 0); // operating_point_idc[0]
        w.u(5, u32::from(self.level_idx));
        if self.level_idx > 7 {
            w.flag(false); // seq_tier[0]
        }

        w.u(4, u32::from(self.width_bits) - 1);
        w.u(4, u32::from(self.height_bits) - 1);
        w.u(u32::from(self.width_bits), u32::from(self.max_width).saturating_sub(1));
        w.u(u32::from(self.height_bits), u32::from(self.max_height).saturating_sub(1));

        w.flag(false); // frame_id_numbers_present_flag
        w.flag(self.use_128x128_superblock);
        w.flag(self.enable_filter_intra);
        w.flag(self.enable_intra_edge_filter);

        w.flag(self.enable_interintra_compound);
        w.flag(self.enable_masked_compound);
        // Not in the descriptor, and safe to enable: it only decides whether the frame header
        // carries allow_warped_motion, which is then written from the descriptor. Reader and
        // writer stay in step either way.
        w.flag(true); // enable_warped_motion
        w.flag(self.enable_dual_filter);
        w.flag(self.enable_order_hint);
        if self.enable_order_hint {
            w.flag(self.enable_jnt_comp);
            w.flag(self.enable_ref_frame_mvs);
        }
        // SELECT for both, so each frame states its own choice and nothing is inherited from a
        // sequence-level decision the descriptor does not record.
        w.flag(true); // seq_choose_screen_content_tools
        w.flag(true); // seq_choose_integer_mv
        if self.enable_order_hint {
            w.u(3, u32::from(self.order_hint_bits) - 1);
        }

        // enable_superres and enable_restoration are likewise absent from the descriptor and only
        // gate header fields written from it. enable_cdef is carried, and matters.
        w.flag(true); // enable_superres
        w.flag(self.enable_cdef);
        w.flag(true); // enable_restoration

        self.write_color_config(w);

        w.flag(self.film_grain_params_present);
    }

    /// The sequence header OBU payload, with its trailing bits.
    pub fn sequence_header(&self) -> Result<Vec<u8>, Unsupported> {
        if self.profile != 0 {
            return Err(Unsupported::Profile(self.profile));
        }
        let mut w = Writer::new(Escape::Raw);
        self.write(&mut w);
        w.rbsp_trailing();
        Ok(w.finish())
    }

    /// The `av1C` configuration record a container-shaped decoder wants.
    ///
    /// Unlike VP9's `vpcC`, which is six scalars, this embeds real bitstream: four bytes of
    /// configuration record followed by the sequence header OBU itself.
    pub fn av1c(&self) -> Result<Vec<u8>, Unsupported> {
        let header = self.sequence_header()?;
        let mut w = Writer::new(Escape::Raw);

        w.u(1, 1); // marker
        w.u(7, 1); // version
        w.u(3, u32::from(self.profile));
        w.u(5, u32::from(self.level_idx));
        w.u(1, 0); // seq_tier_0
        w.u(1, u32::from(self.bit_depth > 8)); // high_bitdepth
        w.u(1, u32::from(self.bit_depth == 12)); // twelve_bit
        w.u(1, u32::from(self.mono_chrome));
        w.u(1, 1); // chroma_subsampling_x: 4:2:0
        w.u(1, 1); // chroma_subsampling_y
        w.u(2, 0); // chroma_sample_position: unknown
        w.u(3, 0); // reserved
        w.u(1, 0); // initial_presentation_delay_present
        w.u(4, 0); // initial_presentation_delay_minus_one

        emit_obu(&mut w, OBU_SEQUENCE_HEADER, &header);
        Ok(w.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leb128_grows_a_byte_every_seven_bits() {
        for (v, want) in [
            (0u64, vec![0x00]),
            (1, vec![0x01]),
            (127, vec![0x7f]),
            (128, vec![0x80, 0x01]),
            (300, vec![0xac, 0x02]),
            (16383, vec![0xff, 0x7f]),
            (16384, vec![0x80, 0x80, 0x01]),
        ] {
            let mut w = Writer::new(Escape::Raw);
            w.leb128(v);
            assert_eq!(w.finish(), want, "leb128({v})");
        }
    }

    #[test]
    fn the_level_is_the_lowest_one_that_covers_the_frame() {
        assert_eq!(pick_level(1920, 1080), 8); // 4.0
        assert_eq!(pick_level(1280, 720), 5); // 3.1: 720p is past 3.0's pixel limit
        assert_eq!(pick_level(640, 480), 4); // 3.0: 480p is past 2.1's pixel limit
        assert_eq!(pick_level(512, 288), 0); // 2.0, exactly at its limit
        assert_eq!(pick_level(3840, 2160), 12); // 5.0
        assert_eq!(pick_level(32768, 16384), LEVEL_MAX);
    }

    #[test]
    fn a_profile_other_than_main_is_refused_rather_than_written_as_main() {
        let mut seq = SeqParams::read(&[]);
        seq.profile = 1;
        assert_eq!(seq.sequence_header(), Err(Unsupported::Profile(1)));
        assert_eq!(seq.av1c(), Err(Unsupported::Profile(1)));
    }

    #[test]
    fn an_av1c_record_carries_the_sequence_header_after_its_four_bytes() {
        let seq = SeqParams::read(&[]);
        let record = seq.av1c().unwrap();
        let header = seq.sequence_header().unwrap();

        assert_eq!(record[0], 0x81, "marker and version");
        // Four bytes of record, then an OBU header byte and a one-byte length for a header this
        // short.
        assert_eq!(&record[6..], &header[..]);
        assert_eq!(usize::from(record[5]), header.len());
    }
}

/// Diff the layout and the sequence header against the C they were ported from.
#[cfg(all(test, feature = "video-oracle"))]
mod oracle {
    use super::*;

    unsafe extern "C" {
        fn virgl_oracle_av1_desc_bytes() -> usize;
        fn virgl_oracle_av1_layout(out: *mut u32, cap: usize) -> usize;
        fn virgl_oracle_av1_desc_fill(out: *mut u8, seed: u64);
        fn virgl_oracle_av1_build_av1c(desc: *const u8, out: *mut u8, out_size: usize) -> isize;
    }

    /// Every field the reader knows about, in the order the C shim reports them. A plain member
    /// carries only an offset; a bit-field carries the storage unit and its position in it.
    const LAYOUT: &[Bits] = &[
        plain(at::PROFILE),
        plain(at::ORDER_HINT_BITS_MINUS_1),
        plain(at::BIT_DEPTH_IDX),
        plain(at::FRAME_WIDTH),
        plain(at::FRAME_HEIGHT),
        plain(at::MAX_WIDTH),
        plain(at::MAX_HEIGHT),
        at::SEQ_USE_128X128_SUPERBLOCK,
        at::SEQ_ENABLE_FILTER_INTRA,
        at::SEQ_ENABLE_INTRA_EDGE_FILTER,
        at::SEQ_ENABLE_INTERINTRA_COMPOUND,
        at::SEQ_ENABLE_MASKED_COMPOUND,
        at::SEQ_ENABLE_DUAL_FILTER,
        at::SEQ_ENABLE_ORDER_HINT,
        at::SEQ_ENABLE_JNT_COMP,
        at::SEQ_ENABLE_CDEF,
        at::SEQ_MONO_CHROME,
        at::SEQ_REF_FRAME_MVS,
        at::SEQ_FILM_GRAIN_PARAMS_PRESENT,
    ];

    const fn plain(at: usize) -> Bits {
        Bits { at, bytes: 0, shift: 0, width: 0 }
    }

    #[test]
    fn every_field_is_read_from_where_the_c_struct_puts_it() {
        let mut raw = vec![0u32; LAYOUT.len() * 4];
        // SAFETY: `raw` is a live slice with its capacity passed beside it; the C writes four
        // words per field and returns 0 rather than overrunning.
        let n = unsafe { virgl_oracle_av1_layout(raw.as_mut_ptr(), raw.len()) };
        assert_eq!(n, LAYOUT.len(), "the C reports a different number of fields");

        let theirs: Vec<Bits> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| Bits { at: c[0] as usize, bytes: c[1] as usize, shift: c[2], width: c[3] })
            .collect();
        assert_eq!(&theirs[..], LAYOUT);
    }

    /// A descriptor, aligned as the C struct needs it and long enough to be one.
    struct Descriptor(Vec<u64>);

    impl Descriptor {
        fn new(seed: u64) -> Descriptor {
            // SAFETY: the C reports the size of its own struct, which is what it then fills.
            let bytes = unsafe { virgl_oracle_av1_desc_bytes() };
            let mut backing = vec![0u64; bytes.div_ceil(8)];
            // SAFETY: the buffer is `bytes` long, rounded up, and 8-byte aligned by its element
            // type, which is at least what a struct of 32-bit members needs.
            unsafe { virgl_oracle_av1_desc_fill(backing.as_mut_ptr().cast(), seed) };
            Descriptor(backing)
        }

        fn bytes(&self) -> &[u8] {
            // SAFETY: any initialized `u64` is a valid sequence of bytes, and the slice borrows
            // the same allocation for the same lifetime.
            unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), self.0.len() * 8) }
        }

        fn c_av1c(&self) -> Option<Vec<u8>> {
            let mut out = vec![0u8; 4096];
            // SAFETY: the descriptor is at least the size the C reported for its struct and is
            // aligned for it; `out` is far larger than a sequence header, and its capacity is
            // passed beside it.
            let n = unsafe {
                virgl_oracle_av1_build_av1c(self.0.as_ptr().cast(), out.as_mut_ptr(), out.len())
            };
            if n < 0 {
                return None;
            }
            out.truncate(n as usize);
            Some(out)
        }
    }

    #[test]
    fn the_av1c_record_is_the_bytes_the_c_writes() {
        for seed in 0..512u64 {
            let desc = Descriptor::new(seed);
            let ours = SeqParams::read(desc.bytes()).av1c().unwrap();
            assert_eq!(Some(ours), desc.c_av1c(), "seed {seed}");
        }
    }

    #[test]
    fn a_profile_other_than_main_diverges_deliberately() {
        // The C writes Main's color_config for every profile: profile 1 reads no
        // chroma_sample_position, so those two bits shift separate_uv_delta_q and the sequence
        // header's last field is misread. The Rust refuses instead. The assertion that the C
        // still emits something is the point -- it fails if the C is ever fixed, rather than
        // leaving this note to go quietly stale.
        for seed in 0..8u64 {
            let desc = Descriptor::new(seed);
            let mut seq = SeqParams::read(desc.bytes());
            assert_eq!(seq.profile, 0);
            seq.profile = 1;
            assert_eq!(seq.av1c(), Err(Unsupported::Profile(1)));
            assert!(desc.c_av1c().is_some(), "the C refused profile 0 after all");
        }
    }
}
