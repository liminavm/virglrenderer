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
const OBU_TEMPORAL_DELIMITER: u32 = 2;
const OBU_FRAME: u32 = 6;

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
    pub const REF: usize = 264;
    pub const REF_FRAME_IDX: usize = 352;
    pub const PRIMARY_REF_FRAME: usize = 359;
    pub const ORDER_HINT: usize = 360;
    pub const SEG_FEATURE_DATA: usize = 368;
    pub const SEG_FEATURE_MASK: usize = 496;
    pub const GRAIN_SEED: usize = 508;
    pub const NUM_Y_POINTS: usize = 510;
    pub const POINT_Y_VALUE: usize = 511;
    pub const POINT_Y_SCALING: usize = 525;
    pub const NUM_CB_POINTS: usize = 539;
    pub const POINT_CB_VALUE: usize = 540;
    pub const POINT_CB_SCALING: usize = 550;
    pub const NUM_CR_POINTS: usize = 560;
    pub const POINT_CR_VALUE: usize = 561;
    pub const POINT_CR_SCALING: usize = 571;
    pub const AR_COEFFS_Y: usize = 581;
    pub const AR_COEFFS_CB: usize = 605;
    pub const AR_COEFFS_CR: usize = 630;
    pub const CB_MULT: usize = 655;
    pub const CB_LUMA_MULT: usize = 656;
    pub const CB_OFFSET: usize = 658;
    pub const CR_MULT: usize = 660;
    pub const CR_LUMA_MULT: usize = 661;
    pub const CR_OFFSET: usize = 662;
    pub const TILE_COLS: usize = 664;
    pub const TILE_ROWS: usize = 665;
    pub const WIDTH_IN_SBS: usize = 1188;
    pub const HEIGHT_IN_SBS: usize = 1316;
    pub const CONTEXT_UPDATE_TILE_ID: usize = 1444;
    pub const SUPERRES_SCALE_DENOMINATOR: usize = 1452;
    pub const INTERP_FILTER: usize = 1453;
    pub const FILTER_LEVEL: usize = 1454;
    pub const FILTER_LEVEL_U: usize = 1456;
    pub const FILTER_LEVEL_V: usize = 1457;
    pub const REF_DELTAS: usize = 1459;
    pub const MODE_DELTAS: usize = 1467;
    pub const BASE_QINDEX: usize = 1469;
    pub const Y_DC_DELTA_Q: usize = 1470;
    pub const U_DC_DELTA_Q: usize = 1471;
    pub const U_AC_DELTA_Q: usize = 1472;
    pub const V_DC_DELTA_Q: usize = 1473;
    pub const V_AC_DELTA_Q: usize = 1474;
    pub const CDEF_DAMPING_MINUS_3: usize = 1484;
    pub const CDEF_BITS: usize = 1485;
    pub const CDEF_Y_STRENGTHS: usize = 1486;
    pub const CDEF_UV_STRENGTHS: usize = 1494;
    pub const WM_WMTYPE: usize = 1512;
    pub const WM_WMMAT: usize = 1520;
    pub const WM_NEXT_WMTYPE: usize = 1552;
    pub const SLICE_DATA_SIZE: usize = 1800;
    pub const SLICE_DATA_OFFSET: usize = 2824;
    pub const SLICE_COUNT: usize = 5128;

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
    pub const SEG_ENABLED: Bits = Bits { at: 364, bytes: 4, shift: 0, width: 1 };
    pub const SEG_UPDATE_MAP: Bits = Bits { at: 364, bytes: 4, shift: 1, width: 1 };
    pub const SEG_TEMPORAL_UPDATE: Bits = Bits { at: 364, bytes: 4, shift: 3, width: 1 };
    pub const FG_APPLY_GRAIN: Bits = Bits { at: 504, bytes: 4, shift: 0, width: 1 };
    pub const FG_CHROMA_SCALING_FROM_LUMA: Bits = Bits { at: 504, bytes: 4, shift: 1, width: 1 };
    pub const FG_GRAIN_SCALING_MINUS_8: Bits = Bits { at: 504, bytes: 4, shift: 2, width: 2 };
    pub const FG_AR_COEFF_LAG: Bits = Bits { at: 504, bytes: 4, shift: 4, width: 2 };
    pub const FG_AR_COEFF_SHIFT_MINUS_6: Bits = Bits { at: 504, bytes: 4, shift: 6, width: 2 };
    pub const FG_GRAIN_SCALE_SHIFT: Bits = Bits { at: 504, bytes: 4, shift: 8, width: 2 };
    pub const FG_OVERLAP_FLAG: Bits = Bits { at: 504, bytes: 4, shift: 10, width: 1 };
    pub const FG_CLIP_TO_RESTRICTED_RANGE: Bits = Bits { at: 504, bytes: 4, shift: 11, width: 1 };
    pub const PIC_FRAME_TYPE: Bits = Bits { at: 1448, bytes: 4, shift: 0, width: 2 };
    pub const PIC_SHOW_FRAME: Bits = Bits { at: 1448, bytes: 4, shift: 2, width: 1 };
    pub const PIC_SHOWABLE_FRAME: Bits = Bits { at: 1448, bytes: 4, shift: 3, width: 1 };
    pub const PIC_ERROR_RESILIENT_MODE: Bits = Bits { at: 1448, bytes: 4, shift: 4, width: 1 };
    pub const PIC_DISABLE_CDF_UPDATE: Bits = Bits { at: 1448, bytes: 4, shift: 5, width: 1 };
    pub const PIC_ALLOW_SCREEN_CONTENT_TOOLS: Bits =
        Bits { at: 1448, bytes: 4, shift: 6, width: 1 };
    pub const PIC_FORCE_INTEGER_MV: Bits = Bits { at: 1448, bytes: 4, shift: 7, width: 1 };
    pub const PIC_ALLOW_INTRABC: Bits = Bits { at: 1448, bytes: 4, shift: 8, width: 1 };
    pub const PIC_USE_SUPERRES: Bits = Bits { at: 1448, bytes: 4, shift: 9, width: 1 };
    pub const PIC_ALLOW_HIGH_PRECISION_MV: Bits = Bits { at: 1448, bytes: 4, shift: 10, width: 1 };
    pub const PIC_IS_MOTION_MODE_SWITCHABLE: Bits =
        Bits { at: 1448, bytes: 4, shift: 11, width: 1 };
    pub const PIC_USE_REF_FRAME_MVS: Bits = Bits { at: 1448, bytes: 4, shift: 12, width: 1 };
    pub const PIC_DISABLE_FRAME_END_UPDATE_CDF: Bits =
        Bits { at: 1448, bytes: 4, shift: 13, width: 1 };
    pub const PIC_UNIFORM_TILE_SPACING_FLAG: Bits =
        Bits { at: 1448, bytes: 4, shift: 14, width: 1 };
    pub const PIC_ALLOW_WARPED_MOTION: Bits = Bits { at: 1448, bytes: 4, shift: 15, width: 1 };
    pub const LF_SHARPNESS_LEVEL: Bits = Bits { at: 1458, bytes: 1, shift: 0, width: 3 };
    pub const LF_MODE_REF_DELTA_ENABLED: Bits = Bits { at: 1458, bytes: 1, shift: 3, width: 1 };
    pub const QM_USING_QMATRIX: Bits = Bits { at: 1476, bytes: 2, shift: 0, width: 1 };
    pub const QM_QM_Y: Bits = Bits { at: 1476, bytes: 2, shift: 1, width: 4 };
    pub const QM_QM_U: Bits = Bits { at: 1476, bytes: 2, shift: 5, width: 4 };
    pub const QM_QM_V: Bits = Bits { at: 1476, bytes: 2, shift: 9, width: 4 };
    pub const MC_DELTA_Q_PRESENT_FLAG: Bits = Bits { at: 1480, bytes: 4, shift: 0, width: 1 };
    pub const MC_LOG2_DELTA_Q_RES: Bits = Bits { at: 1480, bytes: 4, shift: 1, width: 2 };
    pub const MC_DELTA_LF_PRESENT_FLAG: Bits = Bits { at: 1480, bytes: 4, shift: 3, width: 1 };
    pub const MC_LOG2_DELTA_LF_RES: Bits = Bits { at: 1480, bytes: 4, shift: 4, width: 2 };
    pub const MC_DELTA_LF_MULTI: Bits = Bits { at: 1480, bytes: 4, shift: 6, width: 1 };
    pub const MC_TX_MODE: Bits = Bits { at: 1480, bytes: 4, shift: 7, width: 2 };
    pub const MC_REFERENCE_SELECT: Bits = Bits { at: 1480, bytes: 4, shift: 9, width: 1 };
    pub const MC_REDUCED_TX_SET_USED: Bits = Bits { at: 1480, bytes: 4, shift: 10, width: 1 };
    pub const MC_SKIP_MODE_PRESENT: Bits = Bits { at: 1480, bytes: 4, shift: 11, width: 1 };
    pub const LR_YFRAME_RESTORATION_TYPE: Bits = Bits { at: 1502, bytes: 2, shift: 0, width: 2 };
    pub const LR_CBFRAME_RESTORATION_TYPE: Bits = Bits { at: 1502, bytes: 2, shift: 2, width: 2 };
    pub const LR_CRFRAME_RESTORATION_TYPE: Bits = Bits { at: 1502, bytes: 2, shift: 4, width: 2 };
    pub const LR_LR_UNIT_SHIFT: Bits = Bits { at: 1502, bytes: 2, shift: 6, width: 2 };
    pub const LR_LR_UV_SHIFT: Bits = Bits { at: 1502, bytes: 2, shift: 8, width: 1 };

    /// One `wm[]` entry to the next, from the offsets of the first two.
    pub const WM_STRIDE: usize = WM_NEXT_WMTYPE - WM_WMTYPE;
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
    fn su(&mut self, n: u32, v: i32);
    fn ns(&mut self, n: u32, v: u32);
    fn increment(&mut self, low: u32, high: u32, v: u32);
    fn leb128(&mut self, v: u64);
    fn le(&mut self, bytes: u32, v: u64);
}

impl Av1Syntax for Writer {
    /// `su(n)`: `n` bits total, two's complement.
    fn su(&mut self, n: u32, v: i32) {
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.u(n, (v as u32) & mask);
    }

    /// `ns(n)`: non-symmetric unsigned, for values in `[0, n)`.
    fn ns(&mut self, n: u32, v: u32) {
        if n <= 1 {
            return;
        }
        let width = floor_log2(n) + 1;
        let m = (1u32 << width) - n;
        if v < m {
            self.u(width - 1, v);
        } else {
            let val = v + m;
            self.u(width - 1, val >> 1);
            self.u(1, val & 1);
        }
    }

    /// Unary, terminated by a zero unless the ceiling is reached.
    fn increment(&mut self, low: u32, high: u32, v: u32) {
        for _ in low..v {
            self.u(1, 1);
        }
        if v < high {
            self.u(1, 0);
        }
    }

    /// `le(n)`: `n` bytes, least significant first.
    fn le(&mut self, bytes: u32, v: u64) {
        for i in 0..bytes {
            self.u(8, ((v >> (i * 8)) & 0xff) as u32);
        }
    }

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
    /// Neither the frame size nor the sequence maximum says how big the picture is.
    NoGeometry,
    /// `cdef_bits` past the two bits the syntax allows, which would index past the eight strengths
    /// the wire carries.
    CdefBits(u8),
    /// More tiles than the slice arrays hold. The C reads `slice_count` entries out of two hundred
    /// and fifty-six, which for a large count runs past the descriptor entirely.
    SliceCount(usize),
    /// More film grain scaling points than the wire has room for: fourteen for luma, ten each for
    /// the chroma planes.
    FilmGrainPoints(usize),
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::Profile(p) => write!(f, "AV1 profile {p} is not supported (Main only)"),
            Unsupported::NoGeometry => write!(f, "the descriptor carries no picture size"),
            Unsupported::CdefBits(n) => write!(f, "cdef_bits {n} is out of range"),
            Unsupported::SliceCount(n) => {
                write!(f, "{n} tiles is more than the wire carries offsets for")
            }
            Unsupported::FilmGrainPoints(n) => {
                write!(f, "{n} film grain scaling points is more than the wire carries")
            }
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

// ------------------------------------------------------------------ frame header

/// Frame types (6.8.2).
const FRAME_KEY: u8 = 0;
const FRAME_INTER: u8 = 1;
const FRAME_INTRA_ONLY: u8 = 2;
const FRAME_SWITCH: u8 = 3;

/// `PRIMARY_REF_NONE`: the frame inherits nothing.
const PRIMARY_REF_NONE: u8 = 7;

const NUM_REF_FRAMES: usize = 8;
const REFS_PER_FRAME: usize = 7;
const WARP_PARAMS: usize = 6;

const WARPEDMODEL_PREC_BITS: u32 = 16;
const GM_ABS_TRANS_BITS: u32 = 12;
const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
const GM_ABS_ALPHA_BITS: u32 = 12;
const GM_ALPHA_PREC_BITS: u32 = 15;
const GM_TRANS_PREC_BITS: u32 = 6;
const GM_TRANS_ONLY_PREC_BITS: u32 = 3;

const MAX_TILE_WIDTH: i32 = 4096;
const MAX_TILE_AREA: i32 = 4096 * 2304;
const MAX_TILE_COLS: i32 = 64;
const MAX_TILE_ROWS: i32 = 64;

/// How many bytes each tile's size field takes. Ours to choose; four keeps any tile a decoder will
/// meet representable without a size survey.
const TILE_SIZE_BYTES: u32 = 4;

/// The default warp model: identity.
const DEFAULT_WARP: [i32; WARP_PARAMS] =
    [0, 0, 1 << WARPEDMODEL_PREC_BITS, 0, 0, 1 << WARPEDMODEL_PREC_BITS];

/// How many film grain scaling points each plane's arrays have room for.
const MAX_Y_POINTS: usize = 14;
const MAX_UV_POINTS: usize = 10;
/// How many tiles the slice arrays have room for.
const MAX_SLICES: usize = 256;
/// `cdef_bits` is two bits wide, so at most eight strengths, which is the array's length.
const MAX_CDEF_BITS: u8 = 3;

/// One reference's global motion model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Warp {
    pub wmtype: u32,
    pub wmmat: [i32; WARP_PARAMS],
}

/// One tile's slice of the payload the guest sent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Slice {
    pub offset: u32,
    pub size: u32,
}

/// A scaling point: a value and the scaling at it.
type Point = (u8, u8);

/// 5.9.30 film grain parameters, as the descriptor carries them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilmGrain {
    pub apply_grain: bool,
    pub chroma_scaling_from_luma: bool,
    pub grain_scaling_minus_8: u32,
    pub ar_coeff_lag: u32,
    pub ar_coeff_shift_minus_6: u32,
    pub grain_scale_shift: u32,
    pub overlap: bool,
    pub clip_to_restricted_range: bool,
    pub grain_seed: u16,
    /// Each plane's points, as many as the count said: the count is the vector's length.
    pub y: Vec<Point>,
    pub cb: Vec<Point>,
    pub cr: Vec<Point>,
    pub ar_coeffs_y: [i8; 24],
    pub ar_coeffs_cb: [i8; 25],
    pub ar_coeffs_cr: [i8; 25],
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
}

/// The parts of `struct virgl_av1_picture_desc` a temporal unit is written out of.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FrameDesc {
    pub seq: SeqParams,
    /// The guest's reference map: the surface occupying each of *its* slots, before this frame.
    pub ref_map: [u32; NUM_REF_FRAMES],
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    pub primary_ref_frame: u8,
    pub order_hint: u8,

    pub frame_type: u8,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub allow_intrabc: bool,
    pub use_superres: bool,
    pub allow_high_precision_mv: bool,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub disable_frame_end_update_cdf: bool,
    pub uniform_tile_spacing: bool,
    pub allow_warped_motion: bool,

    pub frame_width: u16,
    pub frame_height: u16,
    pub superres_scale_denominator: u8,
    pub interp_filter: u8,

    pub seg_enabled: bool,
    pub seg_update_map: bool,
    pub seg_temporal_update: bool,
    pub seg_feature_mask: [u8; 8],
    pub seg_feature_data: [[i16; 8]; 8],

    pub base_qindex: u8,
    pub y_dc_delta_q: i8,
    pub u_dc_delta_q: i8,
    pub u_ac_delta_q: i8,
    pub v_dc_delta_q: i8,
    pub v_ac_delta_q: i8,
    pub using_qmatrix: bool,
    pub qm_y: u32,
    pub qm_u: u32,
    pub qm_v: u32,

    pub filter_level: [u8; 2],
    pub filter_level_u: u8,
    pub filter_level_v: u8,
    pub sharpness_level: u32,
    pub mode_ref_delta_enabled: bool,
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],

    pub cdef_damping_minus_3: u8,
    pub cdef_bits: u8,
    pub cdef_y_strengths: [u8; 8],
    pub cdef_uv_strengths: [u8; 8],

    pub lr_y_type: u32,
    pub lr_cb_type: u32,
    pub lr_cr_type: u32,
    pub lr_unit_shift: u32,
    pub lr_uv_shift: u32,

    pub tile_cols: u8,
    pub tile_rows: u8,
    pub width_in_sbs: [u16; 64],
    pub height_in_sbs: [u16; 64],
    pub context_update_tile_id: u16,

    pub delta_q_present: bool,
    pub log2_delta_q_res: u32,
    pub delta_lf_present: bool,
    pub log2_delta_lf_res: u32,
    pub delta_lf_multi: bool,
    pub tx_mode: u32,
    pub reference_select: bool,
    pub reduced_tx_set_used: bool,
    pub skip_mode_present: bool,

    pub wm: [Warp; REFS_PER_FRAME],
    pub film_grain: FilmGrain,
    /// One entry per tile the guest sent; the count is the vector's length.
    pub slices: Vec<Slice>,
}

impl FrameDesc {
    /// Read a descriptor the guest wrote.
    ///
    /// Every count that indexes a fixed array is reconciled with it here, and a count the array
    /// cannot answer for is refused by name rather than clamped. The C reads past each of them --
    /// `num_y_points` out of a fourteen-entry array, `1 << cdef_bits` strengths out of eight,
    /// `slice_count` tiles out of two hundred and fifty-six -- and the last of those runs past the
    /// descriptor entirely.
    pub fn read(blob: &[u8]) -> Result<FrameDesc, Unsupported> {
        let d = Fields(blob);
        let byte = |at: usize| d.byte(at);
        let signed = |at: usize| d.byte(at) as i8;
        let short = |at: usize| d.short(at);
        let word =
            |at: usize| u32::from_le_bytes([byte(at), byte(at + 1), byte(at + 2), byte(at + 3)]);
        let sword = |at: usize| word(at) as i32;
        let points = |count: usize, value: usize, scaling: usize| -> Vec<Point> {
            (0..count).map(|i| (byte(value + i), byte(scaling + i))).collect()
        };

        let seq = SeqParams::read(blob);
        if seq.max_width == 0 || seq.max_height == 0 {
            return Err(Unsupported::NoGeometry);
        }

        let cdef_bits = byte(at::CDEF_BITS);
        if cdef_bits > MAX_CDEF_BITS {
            return Err(Unsupported::CdefBits(cdef_bits));
        }
        let slice_count = usize::from(short(at::SLICE_COUNT));
        if slice_count > MAX_SLICES {
            return Err(Unsupported::SliceCount(slice_count));
        }

        let grain = |n: usize, max: usize| -> Result<usize, Unsupported> {
            if n > max { Err(Unsupported::FilmGrainPoints(n)) } else { Ok(n) }
        };
        let num_y = grain(usize::from(byte(at::NUM_Y_POINTS)), MAX_Y_POINTS)?;
        let num_cb = grain(usize::from(byte(at::NUM_CB_POINTS)), MAX_UV_POINTS)?;
        let num_cr = grain(usize::from(byte(at::NUM_CR_POINTS)), MAX_UV_POINTS)?;

        Ok(FrameDesc {
            seq,
            ref_map: std::array::from_fn(|i| word(at::REF + 4 * i)),
            ref_frame_idx: std::array::from_fn(|i| byte(at::REF_FRAME_IDX + i)),
            primary_ref_frame: byte(at::PRIMARY_REF_FRAME),
            order_hint: byte(at::ORDER_HINT),

            frame_type: d.bits(at::PIC_FRAME_TYPE) as u8,
            show_frame: d.flag(at::PIC_SHOW_FRAME),
            showable_frame: d.flag(at::PIC_SHOWABLE_FRAME),
            error_resilient_mode: d.flag(at::PIC_ERROR_RESILIENT_MODE),
            disable_cdf_update: d.flag(at::PIC_DISABLE_CDF_UPDATE),
            allow_screen_content_tools: d.flag(at::PIC_ALLOW_SCREEN_CONTENT_TOOLS),
            force_integer_mv: d.flag(at::PIC_FORCE_INTEGER_MV),
            allow_intrabc: d.flag(at::PIC_ALLOW_INTRABC),
            use_superres: d.flag(at::PIC_USE_SUPERRES),
            allow_high_precision_mv: d.flag(at::PIC_ALLOW_HIGH_PRECISION_MV),
            is_motion_mode_switchable: d.flag(at::PIC_IS_MOTION_MODE_SWITCHABLE),
            use_ref_frame_mvs: d.flag(at::PIC_USE_REF_FRAME_MVS),
            disable_frame_end_update_cdf: d.flag(at::PIC_DISABLE_FRAME_END_UPDATE_CDF),
            uniform_tile_spacing: d.flag(at::PIC_UNIFORM_TILE_SPACING_FLAG),
            allow_warped_motion: d.flag(at::PIC_ALLOW_WARPED_MOTION),

            frame_width: short(at::FRAME_WIDTH),
            frame_height: short(at::FRAME_HEIGHT),
            superres_scale_denominator: byte(at::SUPERRES_SCALE_DENOMINATOR),
            interp_filter: byte(at::INTERP_FILTER),

            seg_enabled: d.flag(at::SEG_ENABLED),
            seg_update_map: d.flag(at::SEG_UPDATE_MAP),
            seg_temporal_update: d.flag(at::SEG_TEMPORAL_UPDATE),
            seg_feature_mask: std::array::from_fn(|i| byte(at::SEG_FEATURE_MASK + i)),
            seg_feature_data: std::array::from_fn(|i| {
                std::array::from_fn(|j| short(at::SEG_FEATURE_DATA + 2 * (8 * i + j)) as i16)
            }),

            base_qindex: byte(at::BASE_QINDEX),
            y_dc_delta_q: signed(at::Y_DC_DELTA_Q),
            u_dc_delta_q: signed(at::U_DC_DELTA_Q),
            u_ac_delta_q: signed(at::U_AC_DELTA_Q),
            v_dc_delta_q: signed(at::V_DC_DELTA_Q),
            v_ac_delta_q: signed(at::V_AC_DELTA_Q),
            using_qmatrix: d.flag(at::QM_USING_QMATRIX),
            qm_y: d.bits(at::QM_QM_Y),
            qm_u: d.bits(at::QM_QM_U),
            qm_v: d.bits(at::QM_QM_V),

            filter_level: [byte(at::FILTER_LEVEL), byte(at::FILTER_LEVEL + 1)],
            filter_level_u: byte(at::FILTER_LEVEL_U),
            filter_level_v: byte(at::FILTER_LEVEL_V),
            sharpness_level: d.bits(at::LF_SHARPNESS_LEVEL),
            mode_ref_delta_enabled: d.flag(at::LF_MODE_REF_DELTA_ENABLED),
            ref_deltas: std::array::from_fn(|i| signed(at::REF_DELTAS + i)),
            mode_deltas: std::array::from_fn(|i| signed(at::MODE_DELTAS + i)),

            cdef_damping_minus_3: byte(at::CDEF_DAMPING_MINUS_3),
            cdef_bits,
            cdef_y_strengths: std::array::from_fn(|i| byte(at::CDEF_Y_STRENGTHS + i)),
            cdef_uv_strengths: std::array::from_fn(|i| byte(at::CDEF_UV_STRENGTHS + i)),

            lr_y_type: d.bits(at::LR_YFRAME_RESTORATION_TYPE),
            lr_cb_type: d.bits(at::LR_CBFRAME_RESTORATION_TYPE),
            lr_cr_type: d.bits(at::LR_CRFRAME_RESTORATION_TYPE),
            lr_unit_shift: d.bits(at::LR_LR_UNIT_SHIFT),
            lr_uv_shift: d.bits(at::LR_LR_UV_SHIFT),

            tile_cols: byte(at::TILE_COLS),
            tile_rows: byte(at::TILE_ROWS),
            width_in_sbs: std::array::from_fn(|i| short(at::WIDTH_IN_SBS + 2 * i)),
            height_in_sbs: std::array::from_fn(|i| short(at::HEIGHT_IN_SBS + 2 * i)),
            context_update_tile_id: short(at::CONTEXT_UPDATE_TILE_ID),

            delta_q_present: d.flag(at::MC_DELTA_Q_PRESENT_FLAG),
            log2_delta_q_res: d.bits(at::MC_LOG2_DELTA_Q_RES),
            delta_lf_present: d.flag(at::MC_DELTA_LF_PRESENT_FLAG),
            log2_delta_lf_res: d.bits(at::MC_LOG2_DELTA_LF_RES),
            delta_lf_multi: d.flag(at::MC_DELTA_LF_MULTI),
            tx_mode: d.bits(at::MC_TX_MODE),
            reference_select: d.flag(at::MC_REFERENCE_SELECT),
            reduced_tx_set_used: d.flag(at::MC_REDUCED_TX_SET_USED),
            skip_mode_present: d.flag(at::MC_SKIP_MODE_PRESENT),

            wm: std::array::from_fn(|r| Warp {
                wmtype: word(at::WM_WMTYPE + at::WM_STRIDE * r),
                wmmat: std::array::from_fn(|j| sword(at::WM_WMMAT + at::WM_STRIDE * r + 4 * j)),
            }),

            film_grain: FilmGrain {
                apply_grain: d.flag(at::FG_APPLY_GRAIN),
                chroma_scaling_from_luma: d.flag(at::FG_CHROMA_SCALING_FROM_LUMA),
                grain_scaling_minus_8: d.bits(at::FG_GRAIN_SCALING_MINUS_8),
                ar_coeff_lag: d.bits(at::FG_AR_COEFF_LAG),
                ar_coeff_shift_minus_6: d.bits(at::FG_AR_COEFF_SHIFT_MINUS_6),
                grain_scale_shift: d.bits(at::FG_GRAIN_SCALE_SHIFT),
                overlap: d.flag(at::FG_OVERLAP_FLAG),
                clip_to_restricted_range: d.flag(at::FG_CLIP_TO_RESTRICTED_RANGE),
                grain_seed: short(at::GRAIN_SEED),
                y: points(num_y, at::POINT_Y_VALUE, at::POINT_Y_SCALING),
                cb: points(num_cb, at::POINT_CB_VALUE, at::POINT_CB_SCALING),
                cr: points(num_cr, at::POINT_CR_VALUE, at::POINT_CR_SCALING),
                ar_coeffs_y: std::array::from_fn(|i| signed(at::AR_COEFFS_Y + i)),
                ar_coeffs_cb: std::array::from_fn(|i| signed(at::AR_COEFFS_CB + i)),
                ar_coeffs_cr: std::array::from_fn(|i| signed(at::AR_COEFFS_CR + i)),
                cb_mult: byte(at::CB_MULT),
                cb_luma_mult: byte(at::CB_LUMA_MULT),
                cb_offset: short(at::CB_OFFSET),
                cr_mult: byte(at::CR_MULT),
                cr_luma_mult: byte(at::CR_LUMA_MULT),
                cr_offset: short(at::CR_OFFSET),
            },

            slices: (0..slice_count)
                .map(|i| Slice {
                    offset: word(at::SLICE_DATA_OFFSET + 4 * i),
                    size: word(at::SLICE_DATA_SIZE + 4 * i),
                })
                .collect(),
        })
    }
}

/// What one frame's header needs beyond the descriptor: the values the spec infers rather than
/// codes, and the reference slots this writer chose.
struct FrameCtx {
    frame_is_intra: bool,
    coded_lossless: bool,
    all_lossless: bool,
    num_planes: usize,
    /// What the frame header codes.
    upscaled_width: u32,
    /// After the superres downscale.
    frame_width: u32,
    frame_height: u32,
    show_frame: bool,
    error_resilient: bool,
    primary_ref_frame: u8,
    tile_cols: u32,
    tile_rows: u32,
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    /// Ours, not the guest's -- see [`ObuState`].
    refresh: u8,
    /// This frame's references in our own slot numbering.
    our_ref_idx: [u8; REFS_PER_FRAME],
}

impl FrameCtx {
    fn new(d: &FrameDesc) -> FrameCtx {
        let s = &d.seq;
        let frame_is_intra = d.frame_type == FRAME_KEY || d.frame_type == FRAME_INTRA_ONLY;
        let upscaled_width = match d.frame_width {
            0 => u32::from(s.max_width),
            w => u32::from(w),
        };

        // error_resilient_mode is inferred, not coded, for switch frames and shown key frames;
        // the descriptor's copy is authoritative everywhere else.
        let inferred_resilient =
            d.frame_type == FRAME_SWITCH || (d.frame_type == FRAME_KEY && d.show_frame);
        let error_resilient = inferred_resilient || d.error_resilient_mode;

        FrameCtx {
            frame_is_intra,
            coded_lossless: false,
            all_lossless: false,
            num_planes: if s.mono_chrome { 1 } else { 3 },
            upscaled_width,
            frame_width: upscaled_width,
            frame_height: match d.frame_height {
                0 => u32::from(s.max_height),
                h => u32::from(h),
            },
            show_frame: d.show_frame,
            error_resilient,
            // Likewise primary_ref_frame: intra and error-resilient frames inherit nothing.
            primary_ref_frame: if frame_is_intra || error_resilient {
                PRIMARY_REF_NONE
            } else {
                d.primary_ref_frame & 7
            },
            tile_cols: 0,
            tile_rows: 0,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            refresh: 0,
            our_ref_idx: [0; REFS_PER_FRAME],
        }
    }

    /// 7.12.2.
    ///
    /// Derived rather than read because it gates whether the loop filter, CDEF and loop
    /// restoration sections appear in the bitstream at all -- not merely what they say.
    fn derive_lossless(&mut self, d: &FrameDesc) {
        self.coded_lossless = true;
        for i in 0..8 {
            let mut qindex = i32::from(d.base_qindex);
            // Segment feature 0 is SEG_LVL_ALT_Q.
            if d.seg_enabled && d.seg_feature_mask[i] & 1 != 0 {
                qindex += i32::from(d.seg_feature_data[i][0]);
            }
            qindex = qindex.clamp(0, 255);

            if qindex != 0
                || d.y_dc_delta_q != 0
                || d.u_ac_delta_q != 0
                || d.u_dc_delta_q != 0
                || d.v_ac_delta_q != 0
                || d.v_dc_delta_q != 0
            {
                self.coded_lossless = false;
                break;
            }
        }
        self.all_lossless = self.coded_lossless && self.frame_width == self.upscaled_width;
    }
}

fn tile_log2(blk_size: i32, target: i32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

fn relative_dist(order_hint_bits: u32, a: i32, b: i32) -> i32 {
    if order_hint_bits == 0 {
        return 0;
    }
    let diff = a - b;
    let m = 1 << (order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

/// `recenter(r, v)` (5.9.26).
fn recenter(r: u32, v: u32) -> u32 {
    if v > 2 * r {
        v
    } else if v >= r {
        (v - r) * 2
    } else {
        (r - v) * 2 - 1
    }
}

/// 5.9.27, written rather than read.
///
/// Values in `[mk, mk + a)` terminate with a zero bit and a `b2`-bit remainder; anything larger
/// costs a one bit and moves the window up.
fn write_subexp(w: &mut Writer, v: u32, num_syms: u32) {
    let mut i = 0u32;
    let mut mk = 0u32;
    const K: u32 = 3;

    loop {
        let b2 = if i != 0 { K + i - 1 } else { K };
        let a = 1u32 << b2;

        if num_syms <= mk + 3 * a {
            w.ns(num_syms - mk, v - mk);
            return;
        }
        if v >= mk + a {
            w.u(1, 1);
            i += 1;
            mk += a;
        } else {
            w.u(1, 0);
            w.u(b2, v - mk);
            return;
        }
    }
}

fn write_unsigned_subexp_with_ref(w: &mut Writer, v: u32, mx: u32, r: u32) {
    if (r << 1) <= mx {
        write_subexp(w, recenter(r, v), mx);
    } else {
        write_subexp(w, recenter(mx - 1 - r, mx - 1 - v), mx);
    }
}

fn write_signed_subexp_with_ref(w: &mut Writer, v: i32, low: i32, high: i32, r: i32) {
    write_unsigned_subexp_with_ref(w, (v - low) as u32, (high - low) as u32, (r - low) as u32);
}

/// The bitstream's `lr_type` is not the restoration-type enum: it is remapped through
/// `Remap_Lr_Type = { NONE, SWITCHABLE, WIENER, SGRPROJ }`, while the descriptor carries the enum
/// itself. Inverting that is the whole of this, and getting it wrong silently swaps two filters.
fn lr_type_to_coded(kind: u32) -> u32 {
    match kind {
        0 => 0, // NONE
        1 => 2, // WIENER
        2 => 3, // SGRPROJ
        _ => 1, // SWITCHABLE
    }
}

impl FrameDesc {
    /// 5.9.8 `superres_params`, which also settles the downscaled width.
    fn write_superres_params(&self, w: &mut Writer, c: &mut FrameCtx) {
        let mut denom = 8u32; // SUPERRES_NUM
        w.flag(self.use_superres);
        if self.use_superres {
            denom = u32::from(self.superres_scale_denominator).max(9);
            w.u(3, denom - 9); // coded_denom, from SUPERRES_DENOM_MIN
        }
        c.frame_width = (c.upscaled_width * 8 + denom / 2) / denom;
    }

    fn write_frame_size(&self, w: &mut Writer, c: &mut FrameCtx, size_override: bool) {
        if size_override {
            w.u(u32::from(self.seq.width_bits), c.upscaled_width - 1);
            w.u(u32::from(self.seq.height_bits), c.frame_height - 1);
        }
        self.write_superres_params(w, c);
    }

    /// 5.9.15.
    ///
    /// The descriptor hands us explicit tile boundaries, so the non-uniform form is always
    /// expressible; the uniform form is used when the guest says the spacing was uniform, because
    /// it is both shorter and what the original stream said.
    fn write_tile_info(&self, w: &mut Writer, c: &mut FrameCtx) {
        let mi_cols = 2 * ((c.frame_width as i32 + 7) >> 3);
        let mi_rows = 2 * ((c.frame_height as i32 + 7) >> 3);
        let sb_shift = if self.seq.use_128x128_superblock { 5 } else { 4 };
        let sb_size = sb_shift + 2;
        let sb_cols =
            if self.seq.use_128x128_superblock { (mi_cols + 31) >> 5 } else { (mi_cols + 15) >> 4 };
        let sb_rows =
            if self.seq.use_128x128_superblock { (mi_rows + 31) >> 5 } else { (mi_rows + 15) >> 4 };
        let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
        let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
        let min_log2_tiles_area = tile_log2(max_tile_area_sb, sb_rows * sb_cols);
        let min_log2_tiles = min_log2_tile_cols.max(min_log2_tiles_area);

        c.tile_cols = u32::from(self.tile_cols).max(1);
        c.tile_rows = u32::from(self.tile_rows).max(1);

        w.flag(self.uniform_tile_spacing);

        if self.uniform_tile_spacing {
            let cols_log2 =
                tile_log2(1, c.tile_cols as i32).clamp(min_log2_tile_cols, max_log2_tile_cols);
            w.increment(min_log2_tile_cols, max_log2_tile_cols, cols_log2);

            let tile_width_sb = (sb_cols + (1 << cols_log2) - 1) >> cols_log2;
            c.tile_cols = ((sb_cols + tile_width_sb - 1) / tile_width_sb) as u32;
            c.tile_cols_log2 = cols_log2;

            let min_log2_tile_rows = min_log2_tiles.saturating_sub(cols_log2);
            let rows_log2 =
                tile_log2(1, c.tile_rows as i32).clamp(min_log2_tile_rows, max_log2_tile_rows);
            w.increment(min_log2_tile_rows, max_log2_tile_rows, rows_log2);

            let tile_height_sb = (sb_rows + (1 << rows_log2) - 1) >> rows_log2;
            c.tile_rows = ((sb_rows + tile_height_sb - 1) / tile_height_sb) as u32;
            c.tile_rows_log2 = rows_log2;
        } else {
            let mut start_sb = 0;
            let mut widest = 0;
            let mut i = 0;
            while start_sb < sb_cols && i < MAX_TILE_COLS {
                let remaining = sb_cols - start_sb;
                let max_width = remaining.min(max_tile_width_sb);
                let declared = self.width_in_sbs[i as usize];
                let size_sb = if i < c.tile_cols as i32 && declared != 0 {
                    i32::from(declared).min(max_width)
                } else {
                    max_width
                };
                w.ns(max_width as u32, (size_sb - 1) as u32);
                widest = widest.max(size_sb);
                start_sb += size_sb;
                i += 1;
            }
            c.tile_cols = i as u32;
            c.tile_cols_log2 = tile_log2(1, i);

            max_tile_area_sb = if min_log2_tiles > 0 {
                (sb_rows * sb_cols) >> (min_log2_tiles + 1)
            } else {
                sb_rows * sb_cols
            };
            let max_tile_height_sb = if widest != 0 { max_tile_area_sb / widest } else { 1 }.max(1);

            start_sb = 0;
            i = 0;
            while start_sb < sb_rows && i < MAX_TILE_ROWS {
                let remaining = sb_rows - start_sb;
                let max_height = remaining.min(max_tile_height_sb);
                let declared = self.height_in_sbs[i as usize];
                let size_sb = if i < c.tile_rows as i32 && declared != 0 {
                    i32::from(declared).min(max_height)
                } else {
                    max_height
                };
                w.ns(max_height as u32, (size_sb - 1) as u32);
                start_sb += size_sb;
                i += 1;
            }
            c.tile_rows = i as u32;
            c.tile_rows_log2 = tile_log2(1, i);
        }

        if c.tile_cols_log2 > 0 || c.tile_rows_log2 > 0 {
            w.u(c.tile_cols_log2 + c.tile_rows_log2, u32::from(self.context_update_tile_id));
            w.u(2, TILE_SIZE_BYTES - 1);
        }
    }

    /// 5.9.13 `read_delta_q`, written.
    fn write_delta_q(w: &mut Writer, v: i8) {
        w.flag(v != 0);
        if v != 0 {
            w.su(1 + 6, i32::from(v));
        }
    }

    /// 5.9.12.
    fn write_quantization_params(&self, w: &mut Writer, c: &FrameCtx) {
        w.u(8, u32::from(self.base_qindex));
        FrameDesc::write_delta_q(w, self.y_dc_delta_q);

        if c.num_planes > 1 {
            // separate_uv_delta_q was set in the sequence header, so diff_uv_delta is present.
            let diff_uv =
                self.u_dc_delta_q != self.v_dc_delta_q || self.u_ac_delta_q != self.v_ac_delta_q;

            w.flag(diff_uv);
            FrameDesc::write_delta_q(w, self.u_dc_delta_q);
            FrameDesc::write_delta_q(w, self.u_ac_delta_q);
            if diff_uv {
                FrameDesc::write_delta_q(w, self.v_dc_delta_q);
                FrameDesc::write_delta_q(w, self.v_ac_delta_q);
            }
        }

        w.flag(self.using_qmatrix);
        if self.using_qmatrix {
            w.u(4, self.qm_y);
            w.u(4, self.qm_u);
            w.u(4, self.qm_v); // separate_uv_delta_q is on
        }
    }

    /// 5.9.14.
    ///
    /// Feature values are written outright rather than inherited: `update_data` is forced on,
    /// which is always legal, and removes any need to know what the reference held.
    fn write_segmentation_params(&self, w: &mut Writer, c: &FrameCtx) {
        const BITS: [u32; 8] = [8, 6, 6, 6, 6, 3, 0, 0];
        const SIGNED: [bool; 8] = [true, true, true, true, true, false, false, false];

        w.flag(self.seg_enabled);
        if !self.seg_enabled {
            return;
        }

        if c.primary_ref_frame != PRIMARY_REF_NONE {
            w.flag(self.seg_update_map);
            if self.seg_update_map {
                w.flag(self.seg_temporal_update);
            }
            w.flag(true); // update_data
        }

        for i in 0..8 {
            for j in 0..8 {
                let on = (self.seg_feature_mask[i] >> j) & 1 != 0;
                w.flag(on);
                if on && BITS[j] != 0 {
                    let v = i32::from(self.seg_feature_data[i][j]);
                    if SIGNED[j] {
                        w.su(1 + BITS[j], v);
                    } else {
                        w.u(BITS[j], v as u32);
                    }
                }
            }
        }
    }

    /// 5.9.11 `loop_filter_params`.
    fn write_loop_filter_params(&self, w: &mut Writer, c: &FrameCtx) {
        if c.coded_lossless || self.allow_intrabc {
            return;
        }

        w.u(6, u32::from(self.filter_level[0]));
        w.u(6, u32::from(self.filter_level[1]));
        if c.num_planes > 1 && (self.filter_level[0] != 0 || self.filter_level[1] != 0) {
            w.u(6, u32::from(self.filter_level_u));
            w.u(6, u32::from(self.filter_level_v));
        }
        w.u(3, self.sharpness_level);

        w.flag(self.mode_ref_delta_enabled);
        if !self.mode_ref_delta_enabled {
            return;
        }

        // Force the update on and write every delta. The descriptor carries resolved values but no
        // per-index update flags, and writing them all is both legal and independent of what the
        // reference frame held.
        w.flag(true); // loop_filter_delta_update
        for &delta in &self.ref_deltas {
            w.flag(true);
            w.su(1 + 6, i32::from(delta));
        }
        for &delta in &self.mode_deltas {
            w.flag(true);
            w.su(1 + 6, i32::from(delta));
        }
    }

    /// 5.9.19 `cdef_params`.
    fn write_cdef_params(&self, w: &mut Writer, c: &FrameCtx) {
        if c.coded_lossless || self.allow_intrabc || !self.seq.enable_cdef {
            return;
        }

        w.u(2, u32::from(self.cdef_damping_minus_3));
        w.u(2, u32::from(self.cdef_bits));
        for i in 0..(1usize << self.cdef_bits) {
            // VA packs each strength as (primary << 2) | secondary.
            w.u(4, u32::from(self.cdef_y_strengths[i] >> 2));
            w.u(2, u32::from(self.cdef_y_strengths[i] & 3));
            if c.num_planes > 1 {
                w.u(4, u32::from(self.cdef_uv_strengths[i] >> 2));
                w.u(2, u32::from(self.cdef_uv_strengths[i] & 3));
            }
        }
    }

    /// 5.9.20 `lr_params`.
    fn write_lr_params(&self, w: &mut Writer, c: &FrameCtx) {
        if c.all_lossless || self.allow_intrabc {
            return;
        }

        let types = [self.lr_y_type, self.lr_cb_type, self.lr_cr_type];
        let mut uses_lr = false;
        let mut uses_chroma_lr = false;

        for (i, &kind) in types.iter().take(c.num_planes).enumerate() {
            w.u(2, lr_type_to_coded(kind));
            if kind != 0 {
                uses_lr = true;
                if i > 0 {
                    uses_chroma_lr = true;
                }
            }
        }

        if uses_lr {
            if self.seq.use_128x128_superblock {
                w.increment(1, 2, self.lr_unit_shift);
            } else {
                w.increment(0, 2, self.lr_unit_shift);
            }
            // Profile 0 is 4:2:0, so both subsampling flags are set.
            if !self.seq.mono_chrome && uses_chroma_lr {
                w.u(1, self.lr_uv_shift);
            }
        }
    }

    /// One warp parameter, 5.9.25 `global_param`, written rather than read.
    fn write_gm_param(
        &self,
        w: &mut Writer,
        prev: &[i32; WARP_PARAMS],
        ref_idx: usize,
        kind: u32,
        idx: usize,
    ) {
        let high_precision = u32::from(!self.allow_high_precision_mv);
        let (abs_bits, prec_bits) = if idx < 2 {
            if kind == 1 {
                // TRANSLATION
                (GM_ABS_TRANS_ONLY_BITS - high_precision, GM_TRANS_ONLY_PREC_BITS - high_precision)
            } else {
                (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
            }
        } else {
            (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
        };

        let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
        let round = if idx % 3 == 2 { 1 << WARPEDMODEL_PREC_BITS } else { 0 };
        let sub = if idx % 3 == 2 { 1 << prec_bits } else { 0 };
        let mx = 1i32 << abs_bits;

        // 5.9.25: `sub` belongs to the reference alone. The decoder recovers the parameter as
        // (x << precDiff) + round, so the value coded is x = (param - round) >> precDiff, with
        // nothing subtracted; subtracting `sub` from it too sends every diagonal term of a
        // rotzoom or affine model to the range floor -- a scale of about 0.875 where the encoder
        // meant 1.0 -- and the warped prediction smears.
        let r = ((prev[idx] >> prec_diff) - sub).clamp(-mx, mx);
        let v = ((self.wm[ref_idx].wmmat[idx] - round) >> prec_diff).clamp(-mx, mx);

        write_signed_subexp_with_ref(w, v, -mx, mx + 1, r);
    }

    /// 5.9.24 `global_motion_params`.
    ///
    /// The only part of the frame header coded *relative* to a reference, which is why
    /// [`ObuState`] keeps saved warp parameters at all. Note the emission order: the two-by-two
    /// block (indices 2..5) precedes the translation pair (0, 1).
    fn write_global_motion_params(&self, w: &mut Writer, c: &FrameCtx, state: &ObuState) {
        if c.frame_is_intra {
            return;
        }

        for r in 0..REFS_PER_FRAME {
            let kind = self.wm[r].wmtype;
            let prev = if c.primary_ref_frame == PRIMARY_REF_NONE {
                &DEFAULT_WARP
            } else {
                &state.saved_gm[usize::from(c.our_ref_idx[usize::from(c.primary_ref_frame)] & 7)][r]
            };

            w.flag(kind != 0); // is_global
            if kind != 0 {
                w.flag(kind == 2); // is_rot_zoom
                if kind != 2 {
                    w.flag(kind == 1); // is_translation
                }
            }

            if kind >= 2 {
                self.write_gm_param(w, prev, r, kind, 2);
                self.write_gm_param(w, prev, r, kind, 3);
                if kind == 3 {
                    self.write_gm_param(w, prev, r, kind, 4);
                    self.write_gm_param(w, prev, r, kind, 5);
                }
            }
            if kind >= 1 {
                self.write_gm_param(w, prev, r, kind, 0);
                self.write_gm_param(w, prev, r, kind, 1);
            }
        }
    }

    /// 5.9.30.
    ///
    /// `update_grain` is forced on and every parameter written outright: the descriptor carries
    /// resolved grain parameters but neither `update_grain` nor the reference index it would
    /// otherwise point at, so inheriting is not expressible while writing always is.
    fn write_film_grain_params(&self, w: &mut Writer, c: &FrameCtx) {
        let g = &self.film_grain;

        if !self.seq.film_grain_params_present || (!c.show_frame && !self.showable_frame) {
            return;
        }

        w.flag(g.apply_grain);
        if !g.apply_grain {
            return;
        }

        w.u(16, u32::from(g.grain_seed));
        if self.frame_type == FRAME_INTER {
            w.flag(true); // update_grain
        }

        w.u(4, g.y.len() as u32);
        for &(value, scaling) in &g.y {
            w.u(8, u32::from(value));
            w.u(8, u32::from(scaling));
        }

        let mut chroma_from_luma = g.chroma_scaling_from_luma;
        if !self.seq.mono_chrome {
            w.flag(chroma_from_luma);
        } else {
            chroma_from_luma = false;
        }

        // Profile 0 is 4:2:0, so an achromatic luma-less frame codes no chroma points.
        let (num_cb, num_cr) = if self.seq.mono_chrome || chroma_from_luma || g.y.is_empty() {
            (0, 0)
        } else {
            w.u(4, g.cb.len() as u32);
            for &(value, scaling) in &g.cb {
                w.u(8, u32::from(value));
                w.u(8, u32::from(scaling));
            }
            w.u(4, g.cr.len() as u32);
            for &(value, scaling) in &g.cr {
                w.u(8, u32::from(value));
                w.u(8, u32::from(scaling));
            }
            (g.cb.len(), g.cr.len())
        };

        w.u(2, g.grain_scaling_minus_8);
        w.u(2, g.ar_coeff_lag);

        let num_pos_luma = 2 * g.ar_coeff_lag as usize * (g.ar_coeff_lag as usize + 1);
        let num_pos_chroma = if g.y.is_empty() {
            num_pos_luma
        } else {
            for &coeff in &g.ar_coeffs_y[..num_pos_luma] {
                w.u(8, (i32::from(coeff) + 128) as u32);
            }
            num_pos_luma + 1
        };

        if chroma_from_luma || num_cb != 0 {
            for &coeff in &g.ar_coeffs_cb[..num_pos_chroma] {
                w.u(8, (i32::from(coeff) + 128) as u32);
            }
        }
        if chroma_from_luma || num_cr != 0 {
            for &coeff in &g.ar_coeffs_cr[..num_pos_chroma] {
                w.u(8, (i32::from(coeff) + 128) as u32);
            }
        }

        w.u(2, g.ar_coeff_shift_minus_6);
        w.u(2, g.grain_scale_shift);

        if num_cb != 0 {
            w.u(8, u32::from(g.cb_mult));
            w.u(8, u32::from(g.cb_luma_mult));
            w.u(9, u32::from(g.cb_offset));
        }
        if num_cr != 0 {
            w.u(8, u32::from(g.cr_mult));
            w.u(8, u32::from(g.cr_luma_mult));
            w.u(9, u32::from(g.cr_offset));
        }

        w.flag(g.overlap);
        w.flag(g.clip_to_restricted_range);
    }

    /// 5.9.22 `skip_mode_params`.
    ///
    /// Only the *presence* of the bit is at stake, but getting that wrong shifts every bit after
    /// it, so the reference search is reproduced exactly.
    fn skip_mode_allowed(&self, c: &FrameCtx, state: &ObuState) -> bool {
        let bits = u32::from(self.seq.order_hint_bits);
        let order_hint = i32::from(self.order_hint);

        if c.frame_is_intra || !self.reference_select || !self.seq.enable_order_hint {
            return false;
        }

        let hint_of =
            |i: usize| i32::from(state.saved_order_hint[usize::from(c.our_ref_idx[i] & 7)]);

        let mut forward: Option<(usize, i32)> = None;
        let mut backward: Option<(usize, i32)> = None;
        for i in 0..REFS_PER_FRAME {
            let hint = hint_of(i);
            let dist = relative_dist(bits, hint, order_hint);
            if dist < 0 {
                if forward.is_none_or(|(_, h)| relative_dist(bits, hint, h) > 0) {
                    forward = Some((i, hint));
                }
            } else if dist > 0 && backward.is_none_or(|(_, h)| relative_dist(bits, hint, h) < 0) {
                backward = Some((i, hint));
            }
        }

        let Some((_, forward_hint)) = forward else { return false };
        if backward.is_some() {
            return true;
        }
        (0..REFS_PER_FRAME).any(|i| relative_dist(bits, hint_of(i), forward_hint) < 0)
    }

    /// 5.9.2 `uncompressed_header`.
    fn write_uncompressed_header(&self, w: &mut Writer, c: &mut FrameCtx, state: &ObuState) {
        let s = &self.seq;
        let size_override =
            c.upscaled_width != u32::from(s.max_width) || c.frame_height != u32::from(s.max_height);
        // Inferred, not what the descriptor holds: a switch frame and a shown key frame refresh
        // every slot by definition, and the *inferred* value is what the reader tests. Using the
        // descriptor's raw field here made the ref_order_hint loop below fire on a key frame and
        // emit 56 bits nobody reads, which desynchronised the rest of the header.
        let refresh = c.refresh;
        let all_frames_refreshed = refresh == 0xff;
        let refresh_is_inferred =
            self.frame_type == FRAME_SWITCH || (self.frame_type == FRAME_KEY && c.show_frame);

        w.flag(false); // show_existing_frame
        w.u(2, u32::from(self.frame_type));
        w.flag(c.show_frame);
        if !c.show_frame {
            w.flag(self.showable_frame);
        }

        if !refresh_is_inferred {
            w.flag(c.error_resilient);
        }

        w.flag(self.disable_cdf_update);

        // seq_choose_screen_content_tools was SELECT, so both are stated per frame.
        w.flag(self.allow_screen_content_tools);
        if self.allow_screen_content_tools {
            w.flag(self.force_integer_mv);
        }

        // frame_id_numbers_present_flag is off, so no current_frame_id.

        if self.frame_type != FRAME_SWITCH {
            w.flag(size_override);
        }

        if s.order_hint_bits != 0 {
            w.u(u32::from(s.order_hint_bits), u32::from(self.order_hint));
        }

        if !(c.frame_is_intra || c.error_resilient) {
            w.u(3, u32::from(c.primary_ref_frame));
        }

        if !refresh_is_inferred {
            w.u(8, u32::from(refresh));
        }

        if (!c.frame_is_intra || !all_frames_refreshed) && s.enable_order_hint && c.error_resilient
        {
            for hint in state.saved_order_hint {
                w.u(u32::from(s.order_hint_bits), u32::from(hint));
            }
        }

        if c.frame_is_intra {
            self.write_frame_size(w, c, size_override);
            w.flag(false); // render_and_frame_size_different
            if self.allow_screen_content_tools && c.upscaled_width == c.frame_width {
                w.flag(self.allow_intrabc);
            }
        } else {
            if s.enable_order_hint {
                w.flag(false); // frame_refs_short_signaling
            }

            for idx in c.our_ref_idx {
                w.u(3, u32::from(idx));
            }

            if size_override && !c.error_resilient {
                // frame_size_with_refs: decline every reference-derived size and state it.
                for _ in 0..REFS_PER_FRAME {
                    w.flag(false); // found_ref
                }
            }
            self.write_frame_size(w, c, size_override);
            w.flag(false); // render_and_frame_size_different

            if !self.force_integer_mv {
                w.flag(self.allow_high_precision_mv);
            }

            // interpolation_filter: 4 is SWITCHABLE in the descriptor's enum.
            if self.interp_filter == 4 {
                w.flag(true);
            } else {
                w.flag(false);
                w.u(2, u32::from(self.interp_filter));
            }

            w.flag(self.is_motion_mode_switchable);

            if !(c.error_resilient || !s.enable_ref_frame_mvs) {
                w.flag(self.use_ref_frame_mvs);
            }
        }

        if !self.disable_cdf_update {
            w.flag(self.disable_frame_end_update_cdf);
        }

        self.write_tile_info(w, c);
        self.write_quantization_params(w, c);
        self.write_segmentation_params(w, c);

        // delta_q_params / delta_lf_params
        if self.base_qindex > 0 {
            w.flag(self.delta_q_present);
        }
        if self.delta_q_present {
            w.u(2, self.log2_delta_q_res);
            if !self.allow_intrabc {
                w.flag(self.delta_lf_present);
            }
            if self.delta_lf_present {
                w.u(2, self.log2_delta_lf_res);
                w.flag(self.delta_lf_multi);
            }
        }

        c.derive_lossless(self);

        self.write_loop_filter_params(w, c);
        self.write_cdef_params(w, c);
        self.write_lr_params(w, c);

        // read_tx_mode
        if !c.coded_lossless {
            w.increment(1, 2, self.tx_mode);
        }

        // frame_reference_mode
        if !c.frame_is_intra {
            w.flag(self.reference_select);
        }

        if self.skip_mode_allowed(c, state) {
            w.flag(self.skip_mode_present);
        }

        if !(c.frame_is_intra || c.error_resilient) {
            w.flag(self.allow_warped_motion);
        }

        w.flag(self.reduced_tx_set_used);

        self.write_global_motion_params(w, c, state);
        self.write_film_grain_params(w, c);
    }

    /// 5.11.1 `tile_group_obu`, with every tile in one group.
    ///
    /// Each tile but the last is preceded by its size; the last takes whatever remains, which is
    /// why it needs none.
    fn write_tile_group(&self, w: &mut Writer, c: &FrameCtx, tiles: &[u8]) {
        if c.tile_cols * c.tile_rows > 1 {
            w.flag(false); // tile_start_and_end_present_flag
        }
        w.align();

        for (i, slice) in self.slices.iter().enumerate() {
            let start = slice.offset as usize;
            let end = start + slice.size as usize;
            if start > tiles.len() || end > tiles.len() {
                continue;
            }

            if i + 1 < self.slices.len() {
                w.le(TILE_SIZE_BYTES, u64::from(slice.size.wrapping_sub(1)));
            }
            for &b in &tiles[start..end] {
                w.u(8, u32::from(b));
            }
        }
    }
}

// ------------------------------------------------------------------ the model

/// A temporal unit to submit, and what to do with the picture it decodes to.
///
/// The two travel together because they are one fact about one sample: a re-emitted frame must
/// still be decoded -- later frames reference it -- but its picture must not be written anywhere,
/// having been delivered a submission earlier into a target the guest may since have recycled.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Unit {
    pub bytes: Vec<u8>,
    pub discard: bool,
}

/// What a held frame still owes.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Owed {
    /// A hidden frame owes both its picture and its slot, and can wait for both: nothing reads its
    /// target, and nothing needs its pixels until a later `show_existing_frame`.
    PictureAndSlot,
    /// A shown frame owes its picture *now* -- the guest reads that target as soon as the
    /// command-stream fence signals -- so it went out immediately storing nothing, and only its
    /// slot is owed. Its re-emission decodes to a picture that must be discarded.
    SlotOnly,
}

/// A frame kept across one submission, because which slot the guest stored it in is only visible
/// in the *next* frame's `ref[]`, and guessing evicts pictures later frames still reference.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Held {
    desc: FrameDesc,
    tiles: Vec<u8>,
    owes: Owed,
}

/// A frame was built while one was still held.
///
/// Two temporal units in one buffer reach the decoder as one sample and lose a picture, which is
/// the very failure the hold exists to prevent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StillHolding;

/// Everything the frame-header *syntax* is coded relative to, kept across frames.
///
/// Not a decoded-picture buffer: the decoder keeps its own. Two things live here and nothing else.
///
/// The saved warp parameters, because `global_motion_param` codes each value as a subexp delta
/// against the primary reference's saved parameters, and the descriptor carries only reconstructed
/// values -- everything else a frame inherits is written absolutely behind an inherit flag and can
/// simply be re-emitted.
///
/// And the reference slots, because `refresh_frame_flags` is not on the wire. VA-API does not
/// carry it -- mesa writes a constant 1 -- because a VA driver never needs it: the application
/// hands it the whole reference list per frame and manages the DPB itself. A bitstream writer does
/// need it, so slots are assigned here and the guest's `ref_frame_idx` remapped onto them.
pub struct ObuState {
    saved_gm: [[[i32; WARP_PARAMS]; REFS_PER_FRAME]; NUM_REF_FRAMES],
    /// Not needed to *write* a value, but to decide whether a bit is present at all:
    /// `skip_mode_params` searches the references for a forward and a backward one, and only
    /// writes `skip_mode_present` when that search succeeds. Getting this wrong desynchronises the
    /// bit position of everything after it.
    saved_order_hint: [u8; NUM_REF_FRAMES],
    /// The surface we placed in each of our slots.
    slot_surface: [u32; NUM_REF_FRAMES],
    held: Option<Held>,
    /// The model has been advanced onto the current descriptor. A frame's tile data may arrive
    /// over several `decode_bitstream` calls, each carrying the same descriptor, so the advance
    /// has to be idempotent within a frame.
    advanced: bool,
    /// The guest's reference map as of the previous submission. What a frame stored is the
    /// difference between that map and the next one.
    prev_ref: [u32; NUM_REF_FRAMES],
    /// The slot the previous frame was given, while the surface that landed in it is still
    /// unidentified.
    pending_slot: Option<u8>,
}

impl Default for ObuState {
    fn default() -> ObuState {
        ObuState::new()
    }
}

impl ObuState {
    /// The state implied by "no frames decoded yet": every slot empty, every saved model the
    /// identity, every saved order hint zero -- which is what a decoder starting from a key frame
    /// also assumes.
    pub fn new() -> ObuState {
        ObuState {
            saved_gm: [[DEFAULT_WARP; REFS_PER_FRAME]; NUM_REF_FRAMES],
            saved_order_hint: [0; NUM_REF_FRAMES],
            slot_surface: [0; NUM_REF_FRAMES],
            held: None,
            advanced: false,
            prev_ref: [0; NUM_REF_FRAMES],
            pending_slot: None,
        }
    }

    /// Discard a held frame without emitting it.
    ///
    /// For codec teardown: a frame the decoder never output cannot be read afterwards, so decoding
    /// it there would be work nobody collects.
    pub fn drop_held(&mut self) {
        self.held = None;
    }

    /// Map the guest's `ref_frame_idx`, which indexes its reference map, onto our own slots.
    fn resolve_refs(&self, d: &FrameDesc) -> [u8; REFS_PER_FRAME] {
        std::array::from_fn(|j| {
            let want = d.ref_map[usize::from(d.ref_frame_idx[j] & 7)];
            if want == 0 {
                return 0;
            }
            self.slot_surface.iter().position(|&s| s == want).unwrap_or(0) as u8
        })
    }

    /// The reference update of 7.20, restricted to what the *writer* has to remember.
    fn update(&mut self, d: &FrameDesc, refresh: u8) {
        for i in 0..NUM_REF_FRAMES {
            if refresh & (1 << i) == 0 {
                continue;
            }
            self.saved_order_hint[i] = d.order_hint;
            for r in 0..REFS_PER_FRAME {
                self.saved_gm[i][r] = d.wm[r].wmmat;
            }
        }
    }

    /// Build one temporal unit -- temporal delimiter, sequence header, frame header and tile
    /// group -- for the frame the descriptor describes.
    ///
    /// Each frame gets its own temporal delimiter deliberately: a stream's natural framing bundles
    /// a no-show frame with the frame that displays it into one unit, and a decoder then returns a
    /// single picture for the pair, while the protocol above submits one *frame* at a time and
    /// expects one picture back. Repeating the sequence header in every unit is legal and avoids
    /// tracking when a new one is owed.
    fn emit_frame(&mut self, d: &FrameDesc, tiles: &[u8], refresh: u8) -> Vec<u8> {
        let mut ctx = FrameCtx::new(d);
        ctx.refresh = refresh;
        ctx.our_ref_idx = self.resolve_refs(d);

        let mut w = Writer::new(Escape::Raw);
        emit_obu(&mut w, OBU_TEMPORAL_DELIMITER, &[]);

        let mut header = Writer::new(Escape::Raw);
        d.seq.write(&mut header);
        header.rbsp_trailing();
        emit_obu(&mut w, OBU_SEQUENCE_HEADER, &header.finish());

        // One OBU_FRAME rather than a separate header and tile group. Both are legal, but this is
        // the form every real stream uses, so it is the path decoders actually exercise -- and
        // inside it the frame header is byte-aligned rather than terminated with trailing bits,
        // which is the one syntactic difference.
        let mut frame = Writer::new(Escape::Raw);
        d.write_uncompressed_header(&mut frame, &mut ctx, self);
        frame.align();
        d.write_tile_group(&mut frame, &ctx, tiles);
        emit_obu(&mut w, OBU_FRAME, &frame.finish());

        self.update(d, refresh);
        w.finish()
    }

    /// The picture the frame between two reference maps produced: the surface `now` lists that
    /// `before` did not.
    ///
    /// Exactly one decode separates them, so a surface that has appeared can only be its output,
    /// and a guest that stored nothing leaves the map unchanged.
    ///
    /// A set difference against the previous map, not against our own slots and not per-slot.
    /// Surface ids are recycled -- a capture cycles through a handful of them -- so a freshly
    /// reused id reads as one we already hold and teaches us nothing, while a per-slot diff misses
    /// a frame that lands in a slot whose contents we had seen elsewhere. The guest never reuses
    /// an id still in its own live map, which is what makes the set difference exact.
    fn new_surface(before: &[u32; NUM_REF_FRAMES], now: &[u32; NUM_REF_FRAMES]) -> u32 {
        now.iter().copied().find(|&h| h != 0 && !before.contains(&h)).unwrap_or(0)
    }

    /// Drop pictures the guest no longer lists.
    ///
    /// Nothing can reference them again, and leaving them behind lets a recycled id match a slot
    /// holding a picture that is long gone.
    fn prune_slots(&mut self, live: &[u32; NUM_REF_FRAMES]) {
        for slot in &mut self.slot_surface {
            if *slot != 0 && !live.contains(slot) {
                *slot = 0;
            }
        }
    }

    /// A slot we may overwrite: one holding nothing, a second copy of a picture we keep elsewhere,
    /// or one holding a picture the guest no longer lists. `None` when all eight are needed.
    fn dead_slot(&self, live: &[u32; NUM_REF_FRAMES]) -> Option<usize> {
        if let Some(i) = self.slot_surface.iter().position(|&s| s == 0) {
            return Some(i);
        }
        for i in 0..NUM_REF_FRAMES {
            if self.slot_surface[..i].contains(&self.slot_surface[i]) {
                return Some(i);
            }
        }
        (0..NUM_REF_FRAMES).find(|&i| !live.contains(&self.slot_surface[i]))
    }

    /// Emit the held frame.
    ///
    /// `live` is the guest reference map from the descriptor that follows it, which is exactly
    /// where the guest's own choice of slot becomes visible: if the held picture is in it, the
    /// guest stored it and so must we; if it is absent, the guest kept nothing, and a frame it
    /// never stored can never be referenced. `None` at the end of a stream, where nothing follows
    /// to reveal anything and nothing that follows can reference it either.
    fn emit_held(&mut self, live: Option<&[u32; NUM_REF_FRAMES]>) -> Option<Unit> {
        let held = self.held.take()?;

        let surface = live.map_or(0, |l| ObuState::new_surface(&self.prev_ref, l));
        let mut refresh = 0;
        let mut slot = None;
        if surface != 0 {
            slot = live.and_then(|l| self.dead_slot(l));
            if let Some(i) = slot {
                refresh = 1u8 << i;
            }
        }

        let mut desc = held.desc;
        let mut discard = false;

        // A frame whose picture already went out owes only the DPB. If the guest stored nothing
        // there is nothing left to owe, and re-decoding it would be work no later frame collects.
        // Otherwise it is re-emitted hidden: the same picture, kept rather than shown. It stays
        // showable so that a stream displaying it later with show_existing_frame is still legal,
        // and so the film grain parameters are written exactly as they were the first time.
        if held.owes == Owed::SlotOnly {
            if surface == 0 {
                return None;
            }
            desc.show_frame = false;
            desc.showable_frame = true;
            discard = true;
        }

        let bytes = self.emit_frame(&desc, &held.tiles, refresh);
        if let Some(i) = slot {
            self.slot_surface[i] = surface;
        }
        Some(Unit { bytes, discard })
    }

    /// Advance the model onto this descriptor: emit the held frame, learn where the previous
    /// frame's picture landed, and forget pictures the guest has dropped.
    ///
    /// Separate from building the frame's own unit because the two must not share a buffer: a
    /// decoder is handed one sample per temporal unit, and two units in one sample loses a
    /// picture. It also happens at a different time -- the descriptor is known at the guest's
    /// first `decode_bitstream`, while the frame's own unit cannot be built until the frame ends,
    /// since its tile data may arrive over several calls.
    ///
    /// Call it as soon as the descriptor is known, so the held picture reaches its target as early
    /// as possible: a stream may display a hidden frame just one decode later, leaving no margin.
    /// Idempotent within a frame.
    pub fn flush_held(&mut self, d: &FrameDesc) -> Option<Unit> {
        if self.advanced {
            return None;
        }

        // The held frame goes first: decode order is preserved, and this descriptor's ref[] is
        // what makes its refresh exact.
        let unit = self.emit_held(Some(&d.ref_map));

        // Learn where a frame we emitted immediately ended up. Its slot was chosen when it was
        // written; only the surface that landed there was still unknown. This must happen before
        // anything asks for a free slot, or a just-stored slot still reading as empty is handed
        // out again and the picture in it is lost.
        if let Some(slot) = self.pending_slot.take() {
            let surface = ObuState::new_surface(&self.prev_ref, &d.ref_map);
            if surface != 0 {
                self.slot_surface[usize::from(slot)] = surface;
            }
        }

        self.prune_slots(&d.ref_map);
        self.prev_ref = d.ref_map;
        self.advanced = true;
        unit
    }

    /// Build this frame's temporal unit.
    ///
    /// `None` when a hidden frame is being held instead: nothing goes out this submission, and the
    /// unit arrives on the next one. `Err` when a frame is still held and was not flushed first.
    pub fn build_temporal_unit(
        &mut self,
        d: &FrameDesc,
        tiles: &[u8],
    ) -> Result<Option<Vec<u8>>, StillHolding> {
        // Refuse before advancing: this unit would be the second in one buffer.
        if self.held.is_some() {
            return Err(StillHolding);
        }
        if !self.advanced {
            let unit = self.flush_held(d);
            assert!(unit.is_none(), "nothing was held, so the advance emitted nothing");
        }

        // A key frame refreshes everything, by inference rather than by choice, and resets the
        // model with it. Its own surface is learned on the next submission, like any frame emitted
        // immediately.
        if d.frame_type == FRAME_SWITCH || (d.frame_type == FRAME_KEY && d.show_frame) {
            self.slot_surface = [0; NUM_REF_FRAMES];
            let bytes = self.emit_frame(d, tiles, 0xff);
            self.pending_slot = Some(0);
            self.advanced = false;
            return Ok(Some(bytes));
        }

        // A frame waits one submission so its refresh can be exact -- but only when it has to.
        // While a slot is free the frame is stored at once, evicting nothing, and whose picture
        // landed there is learned a submission later like any other. It is only at the wall, where
        // storing this frame means dropping a live one, that which slot the guest chose has to be
        // waited for. Holding only there keeps most frames on the immediate path and shrinks the
        // window in which a held picture has not reached its target.
        let Some(slot) = self.dead_slot(&d.ref_map) else {
            self.advanced = false;

            if !d.show_frame {
                // Nothing reads a hidden frame's target, so it owes nothing early.
                self.held = Some(Held {
                    desc: d.clone(),
                    tiles: tiles.to_vec(),
                    owes: Owed::PictureAndSlot,
                });
                return Ok(None);
            }

            // A shown frame's pixels cannot wait for the slot: the guest reads that target as soon
            // as the fence signals, and what it finds there otherwise is whatever the surface last
            // held -- an older picture, or nothing at all. Only the *bitstream* needs the slot, so
            // the two are separated. The frame goes out now storing nothing, which is always legal
            // and settles no eviction, and the copy that claims the slot follows on the next
            // submission, ahead of the frame that might reference it.
            let bytes = self.emit_frame(d, tiles, 0);
            self.held = Some(Held { desc: d.clone(), tiles: tiles.to_vec(), owes: Owed::SlotOnly });
            return Ok(Some(bytes));
        };

        // A slot is free, so the frame is emitted now into it; which picture landed there is
        // learned from the next descriptor, like any frame on the immediate path.
        let bytes = self.emit_frame(d, tiles, 1u8 << slot);
        self.slot_surface[slot] = 0; // the surface is learned on the next submission
        self.pending_slot = Some(slot as u8);
        self.advanced = false;
        Ok(Some(bytes))
    }

    /// Emit whatever is still held, once the last frame has been submitted.
    ///
    /// Nothing follows to reveal where the guest stored it, and nothing that follows can reference
    /// it either, so it is written storing nothing.
    pub fn flush_temporal_unit(&mut self) -> Option<Unit> {
        self.emit_held(None)
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

    // ------------------------------------------------------------ the model, as a sequence

    unsafe extern "C" {
        fn virgl_oracle_av1_full_layout(out: *mut u32, cap: usize) -> usize;
        fn virgl_oracle_av1_state_new() -> *mut std::ffi::c_void;
        fn virgl_oracle_av1_state_free(state: *mut std::ffi::c_void);
        fn virgl_oracle_av1_flush_held(
            state: *mut std::ffi::c_void,
            desc: *const u8,
            out: *mut u8,
            cap: usize,
            discard: *mut i32,
        ) -> isize;
        fn virgl_oracle_av1_build(
            state: *mut std::ffi::c_void,
            desc: *const u8,
            tiles: *const u8,
            tiles_size: usize,
            out: *mut u8,
            cap: usize,
        ) -> isize;
        fn virgl_oracle_av1_flush_unit(
            state: *mut std::ffi::c_void,
            out: *mut u8,
            cap: usize,
        ) -> isize;
        fn virgl_oracle_av1_drop_held(state: *mut std::ffi::c_void);
        fn virgl_oracle_av1_guest_new(seed: u64) -> *mut std::ffi::c_void;
        fn virgl_oracle_av1_guest_free(guest: *mut std::ffi::c_void);
        fn virgl_oracle_av1_guest_break(
            guest: *mut std::ffi::c_void,
            desc: *mut u8,
            guard: i32,
        ) -> usize;
    }

    /// Every field the reader knows about, in the order the C shim reports them.
    const FULL_LAYOUT: &[Bits] = &[
        plain(at::PROFILE),
        plain(at::ORDER_HINT_BITS_MINUS_1),
        plain(at::BIT_DEPTH_IDX),
        plain(at::FRAME_WIDTH),
        plain(at::FRAME_HEIGHT),
        plain(at::MAX_WIDTH),
        plain(at::MAX_HEIGHT),
        plain(at::REF),
        plain(at::REF_FRAME_IDX),
        plain(at::PRIMARY_REF_FRAME),
        plain(at::ORDER_HINT),
        plain(at::SEG_FEATURE_DATA),
        plain(at::SEG_FEATURE_MASK),
        plain(at::GRAIN_SEED),
        plain(at::NUM_Y_POINTS),
        plain(at::POINT_Y_VALUE),
        plain(at::POINT_Y_SCALING),
        plain(at::NUM_CB_POINTS),
        plain(at::POINT_CB_VALUE),
        plain(at::POINT_CB_SCALING),
        plain(at::NUM_CR_POINTS),
        plain(at::POINT_CR_VALUE),
        plain(at::POINT_CR_SCALING),
        plain(at::AR_COEFFS_Y),
        plain(at::AR_COEFFS_CB),
        plain(at::AR_COEFFS_CR),
        plain(at::CB_MULT),
        plain(at::CB_LUMA_MULT),
        plain(at::CB_OFFSET),
        plain(at::CR_MULT),
        plain(at::CR_LUMA_MULT),
        plain(at::CR_OFFSET),
        plain(at::TILE_COLS),
        plain(at::TILE_ROWS),
        plain(at::WIDTH_IN_SBS),
        plain(at::HEIGHT_IN_SBS),
        plain(at::CONTEXT_UPDATE_TILE_ID),
        plain(at::SUPERRES_SCALE_DENOMINATOR),
        plain(at::INTERP_FILTER),
        plain(at::FILTER_LEVEL),
        plain(at::FILTER_LEVEL_U),
        plain(at::FILTER_LEVEL_V),
        plain(at::REF_DELTAS),
        plain(at::MODE_DELTAS),
        plain(at::BASE_QINDEX),
        plain(at::Y_DC_DELTA_Q),
        plain(at::U_DC_DELTA_Q),
        plain(at::U_AC_DELTA_Q),
        plain(at::V_DC_DELTA_Q),
        plain(at::V_AC_DELTA_Q),
        plain(at::CDEF_DAMPING_MINUS_3),
        plain(at::CDEF_BITS),
        plain(at::CDEF_Y_STRENGTHS),
        plain(at::CDEF_UV_STRENGTHS),
        plain(at::WM_WMTYPE),
        plain(at::WM_WMMAT),
        plain(at::WM_NEXT_WMTYPE),
        plain(at::SLICE_DATA_SIZE),
        plain(at::SLICE_DATA_OFFSET),
        plain(at::SLICE_COUNT),
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
        at::SEG_ENABLED,
        at::SEG_UPDATE_MAP,
        at::SEG_TEMPORAL_UPDATE,
        at::FG_APPLY_GRAIN,
        at::FG_CHROMA_SCALING_FROM_LUMA,
        at::FG_GRAIN_SCALING_MINUS_8,
        at::FG_AR_COEFF_LAG,
        at::FG_AR_COEFF_SHIFT_MINUS_6,
        at::FG_GRAIN_SCALE_SHIFT,
        at::FG_OVERLAP_FLAG,
        at::FG_CLIP_TO_RESTRICTED_RANGE,
        at::PIC_FRAME_TYPE,
        at::PIC_SHOW_FRAME,
        at::PIC_SHOWABLE_FRAME,
        at::PIC_ERROR_RESILIENT_MODE,
        at::PIC_DISABLE_CDF_UPDATE,
        at::PIC_ALLOW_SCREEN_CONTENT_TOOLS,
        at::PIC_FORCE_INTEGER_MV,
        at::PIC_ALLOW_INTRABC,
        at::PIC_USE_SUPERRES,
        at::PIC_ALLOW_HIGH_PRECISION_MV,
        at::PIC_IS_MOTION_MODE_SWITCHABLE,
        at::PIC_USE_REF_FRAME_MVS,
        at::PIC_DISABLE_FRAME_END_UPDATE_CDF,
        at::PIC_UNIFORM_TILE_SPACING_FLAG,
        at::PIC_ALLOW_WARPED_MOTION,
        at::LF_SHARPNESS_LEVEL,
        at::LF_MODE_REF_DELTA_ENABLED,
        at::QM_USING_QMATRIX,
        at::QM_QM_Y,
        at::QM_QM_U,
        at::QM_QM_V,
        at::MC_DELTA_Q_PRESENT_FLAG,
        at::MC_LOG2_DELTA_Q_RES,
        at::MC_DELTA_LF_PRESENT_FLAG,
        at::MC_LOG2_DELTA_LF_RES,
        at::MC_DELTA_LF_MULTI,
        at::MC_TX_MODE,
        at::MC_REFERENCE_SELECT,
        at::MC_REDUCED_TX_SET_USED,
        at::MC_SKIP_MODE_PRESENT,
        at::LR_YFRAME_RESTORATION_TYPE,
        at::LR_CBFRAME_RESTORATION_TYPE,
        at::LR_CRFRAME_RESTORATION_TYPE,
        at::LR_LR_UNIT_SHIFT,
        at::LR_LR_UV_SHIFT,
    ];

    #[test]
    fn every_frame_header_field_is_read_from_where_the_c_struct_puts_it() {
        let mut raw = vec![0u32; FULL_LAYOUT.len() * 4];
        // SAFETY: `raw` is a live slice with its capacity passed beside it; the C writes four
        // words per field and returns 0 rather than overrunning.
        let n = unsafe { virgl_oracle_av1_full_layout(raw.as_mut_ptr(), raw.len()) };
        assert_eq!(n, FULL_LAYOUT.len(), "the C reports a different number of fields");

        let theirs: Vec<Bits> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| Bits { at: c[0] as usize, bytes: c[1] as usize, shift: c[2], width: c[3] })
            .collect();
        assert_eq!(&theirs[..], FULL_LAYOUT);
    }

    /// The C model, owned so that its held tiles are freed however a test ends.
    struct CState(*mut std::ffi::c_void);

    impl Drop for CState {
        fn drop(&mut self) {
            // SAFETY: the pointer came from the matching allocator and is freed exactly once.
            unsafe { virgl_oracle_av1_state_free(self.0) };
        }
    }

    /// The simulated guest, likewise.
    struct CGuest(*mut std::ffi::c_void);

    impl Drop for CGuest {
        fn drop(&mut self) {
            // SAFETY: as above.
            unsafe { virgl_oracle_av1_guest_free(self.0) };
        }
    }

    /// Room for any unit these streams produce: a few hundred bytes of headers over a tile
    /// payload of at most a few hundred.
    const UNIT_CAP: usize = 64 * 1024;

    impl CState {
        fn new() -> CState {
            // SAFETY: the allocation is checked below and freed by `Drop`.
            let p = unsafe { virgl_oracle_av1_state_new() };
            assert!(!p.is_null(), "the C model could not be allocated");
            CState(p)
        }

        fn flush_held(&self, desc: &Descriptor) -> Option<Unit> {
            let mut out = vec![0u8; UNIT_CAP];
            let mut discard = 0i32;
            // SAFETY: the descriptor is the C's own struct size and alignment; `out` carries its
            // capacity, and `discard` is a live int the C always writes.
            let n = unsafe {
                virgl_oracle_av1_flush_held(
                    self.0,
                    desc.0.as_ptr().cast(),
                    out.as_mut_ptr(),
                    out.len(),
                    &mut discard,
                )
            };
            assert!(n >= 0, "the C model refused a flush");
            if n == 0 {
                return None;
            }
            out.truncate(n as usize);
            Some(Unit { bytes: out, discard: discard != 0 })
        }

        fn build(&self, desc: &Descriptor, tiles: &[u8]) -> Result<Option<Vec<u8>>, StillHolding> {
            let mut out = vec![0u8; UNIT_CAP];
            // SAFETY: as above; `tiles` is a live slice with its length beside it.
            let n = unsafe {
                virgl_oracle_av1_build(
                    self.0,
                    desc.0.as_ptr().cast(),
                    tiles.as_ptr(),
                    tiles.len(),
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            if n < 0 {
                return Err(StillHolding);
            }
            if n == 0 {
                return Ok(None);
            }
            out.truncate(n as usize);
            Ok(Some(out))
        }

        fn flush_unit(&self) -> Option<Unit> {
            let mut out = vec![0u8; UNIT_CAP];
            // SAFETY: `out` carries its capacity, and the C writes at most that much.
            let n = unsafe { virgl_oracle_av1_flush_unit(self.0, out.as_mut_ptr(), out.len()) };
            assert!(n >= 0, "the C model refused a final flush");
            if n == 0 {
                return None;
            }
            out.truncate(n as usize);
            // The final flush never re-emits a delivered picture, so there is nothing to discard.
            Some(Unit { bytes: out, discard: false })
        }

        fn drop_held(&self) {
            // SAFETY: the pointer is live for the call.
            unsafe { virgl_oracle_av1_drop_held(self.0) };
        }
    }

    impl CGuest {
        fn new(seed: u64) -> CGuest {
            // SAFETY: the allocation is checked below and freed by `Drop`.
            let p = unsafe { virgl_oracle_av1_guest_new(seed) };
            assert!(!p.is_null(), "the simulated guest could not be allocated");
            CGuest(p)
        }

        /// The next descriptor, and the tile payload that goes with it.
        fn next(&self) -> (Descriptor, Vec<u8>) {
            self.produce(0)
        }

        /// The same, with one count pushed past the array it indexes.
        fn broken(&self, guard: i32) -> (Descriptor, Vec<u8>) {
            self.produce(guard)
        }

        fn produce(&self, guard: i32) -> (Descriptor, Vec<u8>) {
            // SAFETY: the C reports the size of its own struct, which is what the guest fills.
            let bytes = unsafe { virgl_oracle_av1_desc_bytes() };
            let mut backing = vec![0u64; bytes.div_ceil(8)];
            // SAFETY: the buffer is `bytes` long, rounded up, and 8-byte aligned by its element
            // type, which is at least what a struct of 32-bit members needs.
            let tiles_size =
                unsafe { virgl_oracle_av1_guest_break(self.0, backing.as_mut_ptr().cast(), guard) };
            // The payload's content is ours; only its length has to match what the descriptor's
            // slice offsets describe.
            let tiles = (0..tiles_size).map(|i| (i as u8).wrapping_mul(37)).collect();
            (Descriptor(backing), tiles)
        }
    }

    /// How many times each hard path was taken, so a run that never reached one fails rather than
    /// passing vacuously.
    #[derive(Default)]
    struct Coverage {
        held_flushed: u32,
        discarded: u32,
        held_hidden: u32,
        still_holding: u32,
        non_default_warp: u32,
    }

    /// Drive both models through one stream, comparing every observable at every step.
    fn play(seed: u64, frames: usize, cover: &mut Coverage) {
        let guest = CGuest::new(seed);
        let theirs = CState::new();
        let mut ours = ObuState::new();

        for frame in 0..frames {
            let (desc, tiles) = guest.next();
            let parsed = FrameDesc::read(desc.bytes()).expect("the guest sends a legal descriptor");
            if parsed.wm.iter().any(|w| w.wmtype != 0) {
                cover.non_default_warp += 1;
            }

            // A frame's tile data may arrive over several calls, each carrying the same
            // descriptor, so the advance is asked for more than once on purpose.
            for call in 0..2 {
                let mine = ours.flush_held(&parsed);
                let c = theirs.flush_held(&desc);
                assert_eq!(mine, c, "seed {seed}, frame {frame}, flush {call}");
                if call == 0 && mine.is_some() {
                    cover.held_flushed += 1;
                    if mine.as_ref().is_some_and(|u| u.discard) {
                        cover.discarded += 1;
                    }
                }
            }

            let mine = ours.build_temporal_unit(&parsed, &tiles);
            let c = theirs.build(&desc, &tiles);
            assert_eq!(mine, c, "seed {seed}, frame {frame}, build");
            match &mine {
                Ok(None) => cover.held_hidden += 1,
                Err(StillHolding) => cover.still_holding += 1,
                _ => {}
            }

            // A caller that builds twice without flushing is refused on both sides -- two units in
            // one sample lose a picture, which is the failure the hold exists to prevent.
            if frame % 7 == 3 {
                let mine = ours.build_temporal_unit(&parsed, &tiles);
                let c = theirs.build(&desc, &tiles);
                assert_eq!(mine, c, "seed {seed}, frame {frame}, second build");
                if mine.is_err() {
                    cover.still_holding += 1;
                }
            }
        }

        assert_eq!(ours.flush_temporal_unit(), theirs.flush_unit(), "seed {seed}, final flush");
    }

    #[test]
    fn the_model_emits_the_same_stream_the_c_does() {
        let mut cover = Coverage::default();
        for seed in 0..24u64 {
            play(seed, 60, &mut cover);
        }

        // Every path the hold exists for has to be reached, or the differential only proves the
        // easy one.
        assert!(cover.held_flushed > 20, "only {} held frames flushed", cover.held_flushed);
        assert!(cover.discarded > 5, "only {} re-emissions discarded", cover.discarded);
        assert!(cover.held_hidden > 5, "only {} hidden frames held", cover.held_hidden);
        assert!(cover.still_holding > 5, "only {} builds refused", cover.still_holding);
        assert!(cover.non_default_warp > 100, "only {} coded warps", cover.non_default_warp);
    }

    #[test]
    fn a_count_past_its_array_is_refused_rather_than_read_past() {
        // Each of these is a place the C indexes past the end of what the wire carries:
        // `1 << cdef_bits` strengths out of eight, `slice_count` tiles out of two hundred and
        // fifty-six -- which for a large count runs past the descriptor entirely -- and the film
        // grain point counts out of fourteen and ten. It emits the bytes it read; the Rust
        // reconciles each count with its array where the descriptor is read, and refuses.
        //
        // The assertions that the C still emits are the point: they fail if it is ever fixed,
        // rather than leaving these notes to go quietly stale.
        let expected = [
            Unsupported::CdefBits(4),
            Unsupported::SliceCount(300),
            Unsupported::FilmGrainPoints(20),
            Unsupported::FilmGrainPoints(16),
            Unsupported::FilmGrainPoints(16),
            Unsupported::NoGeometry,
        ];

        for (i, want) in expected.iter().enumerate() {
            let guest = CGuest::new(7);
            let theirs = CState::new();
            let (desc, tiles) = guest.broken(i as i32 + 1);

            assert_eq!(FrameDesc::read(desc.bytes()), Err(*want));
            // A zero picture size is the one the C refuses too, by dividing nothing by nothing;
            // the rest it serializes out of memory that is not the field it thinks it is.
            if *want != Unsupported::NoGeometry {
                assert!(
                    theirs.build(&desc, &tiles).is_ok_and(|u| u.is_some()),
                    "the C refused guard {i} after all"
                );
            }
        }
    }

    #[test]
    fn dropping_a_held_frame_leaves_both_models_agreeing() {
        let guest = CGuest::new(99);
        let theirs = CState::new();
        let mut ours = ObuState::new();

        for _ in 0..40 {
            let (desc, tiles) = guest.next();
            let parsed = FrameDesc::read(desc.bytes()).unwrap();
            assert_eq!(ours.flush_held(&parsed), theirs.flush_held(&desc));
            assert_eq!(ours.build_temporal_unit(&parsed, &tiles), theirs.build(&desc, &tiles));
            ours.drop_held();
            theirs.drop_held();
        }
        assert_eq!(ours.flush_temporal_unit(), theirs.flush_unit());
    }
}
