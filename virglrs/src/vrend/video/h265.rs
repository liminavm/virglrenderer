// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! HEVC parameter-set synthesis for the VideoToolbox backend.
//!
//! The wire carries a hardware decoder's view of a stream, not a bitstream writer's, so three
//! things are invented rather than copied: the whole VPS, the tier and level, and the
//! `short_term_ref_pic_set` structures. The first two are free -- nothing downstream reads more of
//! a VPS than its `profile_tier_level`, and the level is ours to over-declare. The sets are not:
//! they are absent from VA-API by design, and a slice that indexes one cannot be served.
//! [`slice_inspect`] exists to catch exactly that, which is the condition under which the empty
//! sets written here are sound.

use std::fmt;

use super::Profile;
use super::bitstream::{Escape, NalUnits, Reader, Writer};

/// NAL unit types (Table 7-1).
const NAL_BLA_W_LP: u8 = 16;
const NAL_IDR_W_RADL: u8 = 19;
const NAL_IDR_N_LP: u8 = 20;
const NAL_RSV_IRAP23: u8 = 23;
const NAL_VPS: u8 = 32;
const NAL_SPS: u8 = 33;
const NAL_PPS: u8 = 34;
/// Above this a NAL carries no slice (Table 7-1: 32 and up are non-VCL).
const NAL_LAST_VCL: u8 = 31;

/// Level 5.1. Not on the wire, deliberately generous, and fixed forever.
///
/// It must cover anything the guest can hand us, and it must never change: a level change is the
/// one format-description delta a live decompression session refuses outright, so a level derived
/// from the stream would turn a resolution change into a lost reference picture buffer.
const LEVEL_IDC: u8 = 153;

/// HEVC's default 8x8 scaling list, intra (Table 7-5/7-6). Used for the 8x8, 16x16 and 32x32
/// sizes; matrix ids 0..2 are intra, 3..5 inter.
const DEFAULT_INTRA_8X8: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20,
    21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];

/// The same, inter.
const DEFAULT_INTER_8X8: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20,
    20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];

/// The number of tile columns and rows the wire has room to describe.
///
/// The counts are bytes and the arrays are not: a count past the end of its array is a guest
/// asking us to read whatever follows it in the descriptor. There is nothing there to read.
const MAX_TILE_COLUMNS: usize = 20;
const MAX_TILE_ROWS: usize = 22;

/// The HEVC profiles this build serves.
///
/// One, and it is a type rather than a constant so that adding Main 10 to the capset cannot land
/// without also landing the `profile_idc` and bit depth that go with it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HevcProfile {
    Main,
}

impl HevcProfile {
    /// The HEVC profile a wire profile names, or `None` if it names another codec.
    pub fn of(profile: Profile) -> Option<HevcProfile> {
        matches!(profile, Profile::HevcMain).then_some(HevcProfile::Main)
    }

    /// `general_profile_idc` (A.3).
    fn idc(self) -> u8 {
        match self {
            HevcProfile::Main => 1,
        }
    }
}

/// Why a descriptor could not be serialized.
///
/// Each of these decodes to subtly wrong pixels if guessed at, which is the failure that costs
/// days -- so they are refused rather than approximated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unsupported {
    /// The codec object was created with no extent to size the conformance window from.
    NoGeometry,
    /// Custom scaling lists. Only the enable flag is emitted, with
    /// `sps_scaling_list_data_present_flag` clear, which selects the defaults -- exact when the
    /// stream did the same and silently wrong when it carried its own. Which of the two it is is
    /// not on the wire: mesa hands over the effective lists either way.
    CustomScalingLists,
    SeparateColourPlanes,
    /// `chroma_format_idc`, which must be 1 (4:2:0).
    Chroma(u8),
    /// SPS long-term reference pictures. Their contents are missing from the wire exactly as the
    /// short-term sets are. Slice-carried long-term references are self-contained and fine.
    LongTermRefPicsInSps(u8),
    /// The coded size is smaller than the display size, so there is no conformance window that
    /// could crop one to the other.
    CodedSmallerThanDisplay,
    /// More tile columns or rows than the wire carries widths for.
    TileCount,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::NoGeometry => write!(f, "the codec object has no picture size"),
            Unsupported::CustomScalingLists => write!(
                f,
                "the stream carries custom scaling lists, whose scan order on the VA-API wire is \
                 not established; refusing rather than dequantizing wrong"
            ),
            Unsupported::SeparateColourPlanes => {
                write!(f, "separate colour planes are not supported")
            }
            Unsupported::Chroma(idc) => {
                write!(f, "chroma_format_idc {idc} is not supported (4:2:0 only)")
            }
            Unsupported::LongTermRefPicsInSps(n) => write!(
                f,
                "the stream declares {n} long term ref pics in the SPS, whose contents the wire \
                 does not carry"
            ),
            Unsupported::CodedSmallerThanDisplay => {
                write!(f, "the coded size is smaller than the display size")
            }
            Unsupported::TileCount => {
                write!(f, "more tiles than the wire carries column widths or row heights for")
            }
        }
    }
}

/// Why a slice means the stream must not be decoded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SliceRefusal {
    /// The slice header ends before it has said what we need from it.
    Truncated,
    /// `slice_pic_parameter_set_id` outside 0..63 (7.4.7.1).
    PpsId(u32),
    /// The slice indexes one of the SPS short-term sets, which were written empty because their
    /// contents are not on the wire. Decoding would silently use the wrong reference pictures.
    /// Nothing about this is visible before the first inter-predicted slice, so the refusal
    /// necessarily lands one frame into playback.
    SpsRefPicSet,
    /// The slice predicts its own set from an SPS set, which is the same problem.
    PredictedRefPicSet,
}

impl fmt::Display for SliceRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SliceRefusal::Truncated => write!(f, "the slice header is truncated"),
            SliceRefusal::PpsId(id) => write!(f, "slice_pic_parameter_set_id {id} is out of range"),
            SliceRefusal::SpsRefPicSet => write!(
                f,
                "the slice indexes an SPS short term ref pic set, whose contents are not on the \
                 VA-API wire"
            ),
            SliceRefusal::PredictedRefPicSet => write!(
                f,
                "the slice predicts its ref pic set from an SPS set, whose contents the wire does \
                 not carry"
            ),
        }
    }
}

/// How the picture is divided into tiles, when it is.
///
/// The spacing decides what the PPS carries, so it decides what this holds: uniform spacing writes
/// only the counts and never reads the wire's size arrays, and explicit spacing writes one size
/// per tile, which makes each count the length of its own vector rather than a second value to
/// keep in step with it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Tiles {
    Uniform {
        columns_minus1: u32,
        rows_minus1: u32,
    },
    Explicit {
        column_widths_minus1: Vec<u16>,
        row_heights_minus1: Vec<u16>,
    },
    /// More columns or rows than the wire carries sizes for. A state the guest can put us in, so
    /// it is one the type admits -- see [`Unsupported::TileCount`].
    Overrun,
}

/// The three sets VideoToolbox builds a format description from, as raw NAL payloads: no start
/// codes, emulation prevention applied.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParameterSets {
    pub vps: Vec<u8>,
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
}

/// Where each field sits in `struct virgl_h265_picture_desc`, measured with `offsetof` rather than
/// counted by hand, and checked against it by the video oracle.
mod at {
    pub const SPS_PIC_WIDTH_IN_LUMA_SAMPLES: usize = 264;
    pub const SPS_PIC_HEIGHT_IN_LUMA_SAMPLES: usize = 268;
    pub const SPS_CHROMA_FORMAT_IDC: usize = 272;
    pub const SPS_SEPARATE_COLOUR_PLANE_FLAG: usize = 273;
    pub const SPS_BIT_DEPTH_LUMA_MINUS8: usize = 274;
    pub const SPS_BIT_DEPTH_CHROMA_MINUS8: usize = 275;
    pub const SPS_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: usize = 276;
    pub const SPS_SPS_MAX_DEC_PIC_BUFFERING_MINUS1: usize = 277;
    pub const SPS_LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3: usize = 278;
    pub const SPS_LOG2_DIFF_MAX_MIN_LUMA_CODING_BLOCK_SIZE: usize = 279;
    pub const SPS_LOG2_MIN_TRANSFORM_BLOCK_SIZE_MINUS2: usize = 280;
    pub const SPS_LOG2_DIFF_MAX_MIN_TRANSFORM_BLOCK_SIZE: usize = 281;
    pub const SPS_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER: usize = 282;
    pub const SPS_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA: usize = 283;
    pub const SPS_SCALINGLIST4X4: usize = 284;
    pub const SPS_SCALINGLIST8X8: usize = 380;
    pub const SPS_SCALINGLIST16X16: usize = 764;
    pub const SPS_SCALINGLIST32X32: usize = 1148;
    pub const SPS_SCALINGLISTDCCOEFF16X16: usize = 1276;
    pub const SPS_SCALINGLISTDCCOEFF32X32: usize = 1282;
    pub const SPS_SCALING_LIST_ENABLED_FLAG: usize = 1284;
    pub const SPS_AMP_ENABLED_FLAG: usize = 1285;
    pub const SPS_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG: usize = 1286;
    pub const SPS_PCM_ENABLED_FLAG: usize = 1287;
    pub const SPS_PCM_SAMPLE_BIT_DEPTH_LUMA_MINUS1: usize = 1288;
    pub const SPS_PCM_SAMPLE_BIT_DEPTH_CHROMA_MINUS1: usize = 1289;
    pub const SPS_LOG2_MIN_PCM_LUMA_CODING_BLOCK_SIZE_MINUS3: usize = 1290;
    pub const SPS_LOG2_DIFF_MAX_MIN_PCM_LUMA_CODING_BLOCK_SIZE: usize = 1291;
    pub const SPS_PCM_LOOP_FILTER_DISABLED_FLAG: usize = 1292;
    pub const SPS_NUM_SHORT_TERM_REF_PIC_SETS: usize = 1293;
    pub const SPS_LONG_TERM_REF_PICS_PRESENT_FLAG: usize = 1294;
    pub const SPS_NUM_LONG_TERM_REF_PICS_SPS: usize = 1295;
    pub const SPS_SPS_TEMPORAL_MVP_ENABLED_FLAG: usize = 1296;
    pub const SPS_STRONG_INTRA_SMOOTHING_ENABLED_FLAG: usize = 1297;
    pub const PPS_DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG: usize = 1300;
    pub const PPS_OUTPUT_FLAG_PRESENT_FLAG: usize = 1301;
    pub const PPS_NUM_EXTRA_SLICE_HEADER_BITS: usize = 1302;
    pub const PPS_SIGN_DATA_HIDING_ENABLED_FLAG: usize = 1303;
    pub const PPS_CABAC_INIT_PRESENT_FLAG: usize = 1304;
    pub const PPS_NUM_REF_IDX_L0_DEFAULT_ACTIVE_MINUS1: usize = 1305;
    pub const PPS_NUM_REF_IDX_L1_DEFAULT_ACTIVE_MINUS1: usize = 1306;
    pub const PPS_INIT_QP_MINUS26: usize = 1307;
    pub const PPS_CONSTRAINED_INTRA_PRED_FLAG: usize = 1308;
    pub const PPS_TRANSFORM_SKIP_ENABLED_FLAG: usize = 1309;
    pub const PPS_CU_QP_DELTA_ENABLED_FLAG: usize = 1310;
    pub const PPS_DIFF_CU_QP_DELTA_DEPTH: usize = 1311;
    pub const PPS_PPS_CB_QP_OFFSET: usize = 1312;
    pub const PPS_PPS_CR_QP_OFFSET: usize = 1313;
    pub const PPS_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG: usize = 1314;
    pub const PPS_WEIGHTED_PRED_FLAG: usize = 1315;
    pub const PPS_WEIGHTED_BIPRED_FLAG: usize = 1316;
    pub const PPS_TRANSQUANT_BYPASS_ENABLED_FLAG: usize = 1317;
    pub const PPS_TILES_ENABLED_FLAG: usize = 1318;
    pub const PPS_ENTROPY_CODING_SYNC_ENABLED_FLAG: usize = 1319;
    pub const PPS_COLUMN_WIDTH_MINUS1: usize = 1320;
    pub const PPS_ROW_HEIGHT_MINUS1: usize = 1360;
    pub const PPS_NUM_TILE_COLUMNS_MINUS1: usize = 1404;
    pub const PPS_NUM_TILE_ROWS_MINUS1: usize = 1405;
    pub const PPS_UNIFORM_SPACING_FLAG: usize = 1406;
    pub const PPS_LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG: usize = 1407;
    pub const PPS_PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: usize = 1408;
    pub const PPS_DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG: usize = 1409;
    pub const PPS_DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG: usize = 1410;
    pub const PPS_PPS_DEBLOCKING_FILTER_DISABLED_FLAG: usize = 1411;
    pub const PPS_PPS_BETA_OFFSET_DIV2: usize = 1412;
    pub const PPS_PPS_TC_OFFSET_DIV2: usize = 1413;
    pub const PPS_LISTS_MODIFICATION_PRESENT_FLAG: usize = 1414;
    pub const PPS_LOG2_PARALLEL_MERGE_LEVEL_MINUS2: usize = 1415;
    pub const PPS_SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG: usize = 1418;
}

/// The parts of `struct virgl_h265_picture_desc` the three sets are written out of.
///
/// Each field reduced to what it means: the six scaling list arrays become the one bit anything
/// asks of them, and the tile sizes become vectors exactly as long as the counts that describe
/// them, so no later reader has to be trusted to keep a count and its array in step.
pub struct PictureDesc {
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub chroma_format_idc: u8,
    pub separate_colour_plane: bool,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub sps_max_dec_pic_buffering_minus1: u8,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_transform_block_size_minus2: u8,
    pub log2_diff_max_min_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    /// Whether the effective lists the guest sent are HEVC's own defaults. See
    /// [`Unsupported::CustomScalingLists`].
    pub scaling_lists_are_default: bool,
    pub scaling_list_enabled: bool,
    pub amp_enabled: bool,
    pub sample_adaptive_offset_enabled: bool,
    pub pcm_enabled: bool,
    pub pcm_sample_bit_depth_luma_minus1: u8,
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub pcm_loop_filter_disabled: bool,
    pub num_short_term_ref_pic_sets: u8,
    pub long_term_ref_pics_present: bool,
    pub num_long_term_ref_pics_sps: u8,
    pub sps_temporal_mvp_enabled: bool,
    pub strong_intra_smoothing_enabled: bool,

    pub dependent_slice_segments_enabled: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub init_qp_minus26: i8,
    pub constrained_intra_pred: bool,
    pub transform_skip_enabled: bool,
    pub cu_qp_delta_enabled: bool,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub pps_slice_chroma_qp_offsets_present: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass_enabled: bool,
    pub entropy_coding_sync_enabled: bool,
    /// `None` when the picture is one tile, which is what `tiles_enabled_flag` clear means.
    pub tiles: Option<Tiles>,
    pub loop_filter_across_tiles_enabled: bool,
    pub pps_loop_filter_across_slices_enabled: bool,
    pub deblocking_filter_control_present: bool,
    pub deblocking_filter_override_enabled: bool,
    pub pps_deblocking_filter_disabled: bool,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub lists_modification_present: bool,
    pub log2_parallel_merge_level_minus2: u8,
    pub slice_segment_header_extension_present: bool,
}

impl PictureDesc {
    /// Read a descriptor the guest wrote.
    ///
    /// Total on purpose: a short descriptor reads as zeros rather than failing, which is what the
    /// C does and what the protocol allows -- the resource carries whatever the guest's driver
    /// wrote, and its size is the guest's choice.
    pub fn read(blob: &[u8]) -> PictureDesc {
        let byte = |at: usize| blob.get(at).copied().unwrap_or(0);
        let flag = |at: usize| byte(at) != 0;
        let signed = |at: usize| byte(at) as i8;
        let short = |at: usize| u16::from_le_bytes([byte(at), byte(at + 1)]);
        let word =
            |at: usize| u32::from_le_bytes([byte(at), byte(at + 1), byte(at + 2), byte(at + 3)]);
        // A count past the end of its array is refused rather than read past: the bytes after it
        // belong to another field, and averaging the two is how a decoder ends up tiled wrong.
        let sizes = |at: usize, count: usize, room: usize| {
            (count <= room).then(|| (0..count).map(|i| short(at + 2 * i)).collect())
        };

        PictureDesc {
            pic_width_in_luma_samples: word(at::SPS_PIC_WIDTH_IN_LUMA_SAMPLES),
            pic_height_in_luma_samples: word(at::SPS_PIC_HEIGHT_IN_LUMA_SAMPLES),
            chroma_format_idc: byte(at::SPS_CHROMA_FORMAT_IDC),
            separate_colour_plane: flag(at::SPS_SEPARATE_COLOUR_PLANE_FLAG),
            bit_depth_luma_minus8: byte(at::SPS_BIT_DEPTH_LUMA_MINUS8),
            bit_depth_chroma_minus8: byte(at::SPS_BIT_DEPTH_CHROMA_MINUS8),
            log2_max_pic_order_cnt_lsb_minus4: byte(at::SPS_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4),
            sps_max_dec_pic_buffering_minus1: byte(at::SPS_SPS_MAX_DEC_PIC_BUFFERING_MINUS1),
            log2_min_luma_coding_block_size_minus3: byte(
                at::SPS_LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3,
            ),
            log2_diff_max_min_luma_coding_block_size: byte(
                at::SPS_LOG2_DIFF_MAX_MIN_LUMA_CODING_BLOCK_SIZE,
            ),
            log2_min_transform_block_size_minus2: byte(
                at::SPS_LOG2_MIN_TRANSFORM_BLOCK_SIZE_MINUS2,
            ),
            log2_diff_max_min_transform_block_size: byte(
                at::SPS_LOG2_DIFF_MAX_MIN_TRANSFORM_BLOCK_SIZE,
            ),
            max_transform_hierarchy_depth_inter: byte(at::SPS_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER),
            max_transform_hierarchy_depth_intra: byte(at::SPS_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA),
            scaling_lists_are_default: lists_are_default(&byte),
            scaling_list_enabled: flag(at::SPS_SCALING_LIST_ENABLED_FLAG),
            amp_enabled: flag(at::SPS_AMP_ENABLED_FLAG),
            sample_adaptive_offset_enabled: flag(at::SPS_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG),
            pcm_enabled: flag(at::SPS_PCM_ENABLED_FLAG),
            pcm_sample_bit_depth_luma_minus1: byte(at::SPS_PCM_SAMPLE_BIT_DEPTH_LUMA_MINUS1),
            pcm_sample_bit_depth_chroma_minus1: byte(at::SPS_PCM_SAMPLE_BIT_DEPTH_CHROMA_MINUS1),
            log2_min_pcm_luma_coding_block_size_minus3: byte(
                at::SPS_LOG2_MIN_PCM_LUMA_CODING_BLOCK_SIZE_MINUS3,
            ),
            log2_diff_max_min_pcm_luma_coding_block_size: byte(
                at::SPS_LOG2_DIFF_MAX_MIN_PCM_LUMA_CODING_BLOCK_SIZE,
            ),
            pcm_loop_filter_disabled: flag(at::SPS_PCM_LOOP_FILTER_DISABLED_FLAG),
            num_short_term_ref_pic_sets: byte(at::SPS_NUM_SHORT_TERM_REF_PIC_SETS),
            long_term_ref_pics_present: flag(at::SPS_LONG_TERM_REF_PICS_PRESENT_FLAG),
            num_long_term_ref_pics_sps: byte(at::SPS_NUM_LONG_TERM_REF_PICS_SPS),
            sps_temporal_mvp_enabled: flag(at::SPS_SPS_TEMPORAL_MVP_ENABLED_FLAG),
            strong_intra_smoothing_enabled: flag(at::SPS_STRONG_INTRA_SMOOTHING_ENABLED_FLAG),

            dependent_slice_segments_enabled: flag(at::PPS_DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG),
            output_flag_present: flag(at::PPS_OUTPUT_FLAG_PRESENT_FLAG),
            num_extra_slice_header_bits: byte(at::PPS_NUM_EXTRA_SLICE_HEADER_BITS),
            sign_data_hiding_enabled: flag(at::PPS_SIGN_DATA_HIDING_ENABLED_FLAG),
            cabac_init_present: flag(at::PPS_CABAC_INIT_PRESENT_FLAG),
            num_ref_idx_l0_default_active_minus1: byte(
                at::PPS_NUM_REF_IDX_L0_DEFAULT_ACTIVE_MINUS1,
            ),
            num_ref_idx_l1_default_active_minus1: byte(
                at::PPS_NUM_REF_IDX_L1_DEFAULT_ACTIVE_MINUS1,
            ),
            init_qp_minus26: signed(at::PPS_INIT_QP_MINUS26),
            constrained_intra_pred: flag(at::PPS_CONSTRAINED_INTRA_PRED_FLAG),
            transform_skip_enabled: flag(at::PPS_TRANSFORM_SKIP_ENABLED_FLAG),
            cu_qp_delta_enabled: flag(at::PPS_CU_QP_DELTA_ENABLED_FLAG),
            diff_cu_qp_delta_depth: byte(at::PPS_DIFF_CU_QP_DELTA_DEPTH),
            pps_cb_qp_offset: signed(at::PPS_PPS_CB_QP_OFFSET),
            pps_cr_qp_offset: signed(at::PPS_PPS_CR_QP_OFFSET),
            pps_slice_chroma_qp_offsets_present: flag(
                at::PPS_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG,
            ),
            weighted_pred: flag(at::PPS_WEIGHTED_PRED_FLAG),
            weighted_bipred: flag(at::PPS_WEIGHTED_BIPRED_FLAG),
            transquant_bypass_enabled: flag(at::PPS_TRANSQUANT_BYPASS_ENABLED_FLAG),
            entropy_coding_sync_enabled: flag(at::PPS_ENTROPY_CODING_SYNC_ENABLED_FLAG),
            tiles: flag(at::PPS_TILES_ENABLED_FLAG).then(|| {
                let columns = byte(at::PPS_NUM_TILE_COLUMNS_MINUS1) as usize;
                let rows = byte(at::PPS_NUM_TILE_ROWS_MINUS1) as usize;
                if flag(at::PPS_UNIFORM_SPACING_FLAG) {
                    return Tiles::Uniform {
                        columns_minus1: columns as u32,
                        rows_minus1: rows as u32,
                    };
                }
                match (
                    sizes(at::PPS_COLUMN_WIDTH_MINUS1, columns, MAX_TILE_COLUMNS),
                    sizes(at::PPS_ROW_HEIGHT_MINUS1, rows, MAX_TILE_ROWS),
                ) {
                    (Some(column_widths_minus1), Some(row_heights_minus1)) => {
                        Tiles::Explicit { column_widths_minus1, row_heights_minus1 }
                    }
                    _ => Tiles::Overrun,
                }
            }),
            loop_filter_across_tiles_enabled: flag(at::PPS_LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG),
            pps_loop_filter_across_slices_enabled: flag(
                at::PPS_PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
            ),
            deblocking_filter_control_present: flag(at::PPS_DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG),
            deblocking_filter_override_enabled: flag(
                at::PPS_DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG,
            ),
            pps_deblocking_filter_disabled: flag(at::PPS_PPS_DEBLOCKING_FILTER_DISABLED_FLAG),
            pps_beta_offset_div2: signed(at::PPS_PPS_BETA_OFFSET_DIV2),
            pps_tc_offset_div2: signed(at::PPS_PPS_TC_OFFSET_DIV2),
            lists_modification_present: flag(at::PPS_LISTS_MODIFICATION_PRESENT_FLAG),
            log2_parallel_merge_level_minus2: byte(at::PPS_LOG2_PARALLEL_MERGE_LEVEL_MINUS2),
            slice_segment_header_extension_present: flag(
                at::PPS_SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG,
            ),
        }
    }

    /// Write the VPS, SPS and PPS a session for this picture is built around.
    ///
    /// `width` and `height` are the *display* size: the coded size is on the wire, and the
    /// difference between them becomes the conformance window.
    pub fn parameter_sets(
        &self,
        width: u32,
        height: u32,
        profile: HevcProfile,
    ) -> Result<ParameterSets, Unsupported> {
        if width == 0 || height == 0 {
            return Err(Unsupported::NoGeometry);
        }
        if self.scaling_list_enabled && !self.scaling_lists_are_default {
            return Err(Unsupported::CustomScalingLists);
        }
        if self.separate_colour_plane {
            return Err(Unsupported::SeparateColourPlanes);
        }
        if self.chroma_format_idc != 1 {
            return Err(Unsupported::Chroma(self.chroma_format_idc));
        }
        if self.num_long_term_ref_pics_sps != 0 {
            return Err(Unsupported::LongTermRefPicsInSps(self.num_long_term_ref_pics_sps));
        }
        if self.pic_width_in_luma_samples < width || self.pic_height_in_luma_samples < height {
            return Err(Unsupported::CodedSmallerThanDisplay);
        }
        if self.tiles == Some(Tiles::Overrun) {
            return Err(Unsupported::TileCount);
        }

        Ok(ParameterSets {
            vps: self.write_vps(profile),
            sps: self.write_sps(width, height, profile),
            pps: self.write_pps(),
        })
    }

    /// A whole VPS from nothing (7.3.2.1).
    ///
    /// Nothing downstream reads more of it than the `profile_tier_level` and the buffering values,
    /// both of which have SPS twins on the wire.
    fn write_vps(&self, profile: HevcProfile) -> Vec<u8> {
        let mut w = Writer::new(Escape::Rbsp);
        write_nal_header(&mut w, NAL_VPS);

        w.u(4, 0); // vps_video_parameter_set_id
        w.flag(true); // vps_base_layer_internal_flag
        w.flag(true); // vps_base_layer_available_flag
        w.u(6, 0); // vps_max_layers_minus1
        w.u(3, 0); // vps_max_sub_layers_minus1
        w.flag(true); // vps_temporal_id_nesting_flag
        w.u(16, 0xffff); // vps_reserved_0xffff_16bits

        write_ptl(&mut w, profile);

        w.flag(true); // vps_sub_layer_ordering_info_present_flag
        w.ue(u32::from(self.sps_max_dec_pic_buffering_minus1));
        w.ue(u32::from(self.sps_max_dec_pic_buffering_minus1)); // vps_max_num_reorder_pics
        w.ue(0); // vps_max_latency_increase_plus1: no limit

        w.u(6, 0); // vps_max_layer_id
        w.ue(0); // vps_num_layer_sets_minus1
        w.flag(false); // vps_timing_info_present_flag
        w.flag(false); // vps_extension_flag
        w.rbsp_trailing();
        w.finish()
    }

    /// 7.3.2.2.
    fn write_sps(&self, width: u32, height: u32, profile: HevcProfile) -> Vec<u8> {
        let mut w = Writer::new(Escape::Rbsp);
        write_nal_header(&mut w, NAL_SPS);

        w.u(4, 0); // sps_video_parameter_set_id
        w.u(3, 0); // sps_max_sub_layers_minus1
        w.flag(true); // sps_temporal_id_nesting_flag

        write_ptl(&mut w, profile);

        w.ue(0); // sps_seq_parameter_set_id
        w.ue(u32::from(self.chroma_format_idc));
        if self.chroma_format_idc == 3 {
            w.flag(self.separate_colour_plane);
        }

        w.ue(self.pic_width_in_luma_samples);
        w.ue(self.pic_height_in_luma_samples);

        // The coded size is on the wire; the display size is the codec object's. Their difference
        // is the conformance window, in chroma units. Unlike H.264 the geometry needs no
        // derivation -- only the crop does.
        let sub_w = if self.chroma_format_idc == 3 { 1 } else { 2 };
        let sub_h = if self.chroma_format_idc == 1 { 2 } else { 1 };
        if self.pic_width_in_luma_samples > width || self.pic_height_in_luma_samples > height {
            w.flag(true); // conformance_window_flag
            w.ue(0); // conf_win_left_offset
            w.ue((self.pic_width_in_luma_samples - width) / sub_w);
            w.ue(0); // conf_win_top_offset
            w.ue((self.pic_height_in_luma_samples - height) / sub_h);
        } else {
            w.flag(false);
        }

        w.ue(u32::from(self.bit_depth_luma_minus8));
        w.ue(u32::from(self.bit_depth_chroma_minus8));
        w.ue(u32::from(self.log2_max_pic_order_cnt_lsb_minus4));

        w.flag(true); // sps_sub_layer_ordering_info_present_flag
        w.ue(u32::from(self.sps_max_dec_pic_buffering_minus1));
        w.ue(u32::from(self.sps_max_dec_pic_buffering_minus1)); // sps_max_num_reorder_pics
        w.ue(0); // sps_max_latency_increase_plus1

        w.ue(u32::from(self.log2_min_luma_coding_block_size_minus3));
        w.ue(u32::from(self.log2_diff_max_min_luma_coding_block_size));
        w.ue(u32::from(self.log2_min_transform_block_size_minus2));
        w.ue(u32::from(self.log2_diff_max_min_transform_block_size));
        w.ue(u32::from(self.max_transform_hierarchy_depth_inter));
        w.ue(u32::from(self.max_transform_hierarchy_depth_intra));

        w.flag(self.scaling_list_enabled);
        if self.scaling_list_enabled {
            // sps_scaling_list_data_present_flag: the defaults, which the refusal above has
            // confirmed are what the stream carried.
            w.flag(false);
        }
        w.flag(self.amp_enabled);
        w.flag(self.sample_adaptive_offset_enabled);

        w.flag(self.pcm_enabled);
        if self.pcm_enabled {
            w.u(4, u32::from(self.pcm_sample_bit_depth_luma_minus1));
            w.u(4, u32::from(self.pcm_sample_bit_depth_chroma_minus1));
            w.ue(u32::from(self.log2_min_pcm_luma_coding_block_size_minus3));
            w.ue(u32::from(self.log2_diff_max_min_pcm_luma_coding_block_size));
            w.flag(self.pcm_loop_filter_disabled);
        }

        // The sets themselves are not on the wire and cannot be. Only the count matters to a slice
        // header, which reads an index whose width derives from it; the contents are read only by
        // a slice that indexes one, and `slice_inspect` refuses those. `st_ref_pic_set(i)` carries
        // inter_ref_pic_set_prediction_flag for every i != 0 -- omitting it desyncs the parse of
        // this SPS.
        w.ue(u32::from(self.num_short_term_ref_pic_sets));
        for i in 0..u32::from(self.num_short_term_ref_pic_sets) {
            if i != 0 {
                w.flag(false); // inter_ref_pic_set_prediction_flag
            }
            w.ue(0); // num_negative_pics
            w.ue(0); // num_positive_pics
        }

        w.flag(self.long_term_ref_pics_present);
        if self.long_term_ref_pics_present {
            w.ue(0); // num_long_term_ref_pics_sps; refused above if nonzero
        }

        w.flag(self.sps_temporal_mvp_enabled);
        w.flag(self.strong_intra_smoothing_enabled);
        w.flag(false); // vui_parameters_present_flag
        w.flag(false); // sps_extension_present_flag
        w.rbsp_trailing();
        w.finish()
    }

    /// 7.3.2.3.
    fn write_pps(&self) -> Vec<u8> {
        let mut w = Writer::new(Escape::Rbsp);
        write_nal_header(&mut w, NAL_PPS);

        w.ue(0); // pps_pic_parameter_set_id
        w.ue(0); // pps_seq_parameter_set_id

        w.flag(self.dependent_slice_segments_enabled);
        w.flag(self.output_flag_present);
        w.u(3, u32::from(self.num_extra_slice_header_bits));
        w.flag(self.sign_data_hiding_enabled);
        w.flag(self.cabac_init_present);

        w.ue(u32::from(self.num_ref_idx_l0_default_active_minus1));
        w.ue(u32::from(self.num_ref_idx_l1_default_active_minus1));
        w.se(i32::from(self.init_qp_minus26));

        w.flag(self.constrained_intra_pred);
        w.flag(self.transform_skip_enabled);
        w.flag(self.cu_qp_delta_enabled);
        if self.cu_qp_delta_enabled {
            w.ue(u32::from(self.diff_cu_qp_delta_depth));
        }

        w.se(i32::from(self.pps_cb_qp_offset));
        w.se(i32::from(self.pps_cr_qp_offset));
        w.flag(self.pps_slice_chroma_qp_offsets_present);
        w.flag(self.weighted_pred);
        w.flag(self.weighted_bipred);
        w.flag(self.transquant_bypass_enabled);
        w.flag(self.tiles.is_some());
        w.flag(self.entropy_coding_sync_enabled);

        if let Some(tiles) = &self.tiles {
            match tiles {
                Tiles::Uniform { columns_minus1, rows_minus1 } => {
                    w.ue(*columns_minus1);
                    w.ue(*rows_minus1);
                    w.flag(true);
                }
                Tiles::Explicit { column_widths_minus1, row_heights_minus1 } => {
                    w.ue(column_widths_minus1.len() as u32);
                    w.ue(row_heights_minus1.len() as u32);
                    w.flag(false);
                    for &width in column_widths_minus1 {
                        w.ue(u32::from(width));
                    }
                    for &height in row_heights_minus1 {
                        w.ue(u32::from(height));
                    }
                }
                // Refused by `parameter_sets`, which is the only caller.
                Tiles::Overrun => unreachable!("a tile overrun reached the writer"),
            }
            w.flag(self.loop_filter_across_tiles_enabled);
        }

        w.flag(self.pps_loop_filter_across_slices_enabled);
        w.flag(self.deblocking_filter_control_present);
        if self.deblocking_filter_control_present {
            w.flag(self.deblocking_filter_override_enabled);
            w.flag(self.pps_deblocking_filter_disabled);
            if !self.pps_deblocking_filter_disabled {
                w.se(i32::from(self.pps_beta_offset_div2));
                w.se(i32::from(self.pps_tc_offset_div2));
            }
        }

        w.flag(false); // pps_scaling_list_data_present_flag
        w.flag(self.lists_modification_present);
        w.ue(u32::from(self.log2_parallel_merge_level_minus2));
        w.flag(self.slice_segment_header_extension_present);
        w.flag(false); // pps_extension_present_flag
        w.rbsp_trailing();
        w.finish()
    }

    /// Parse the first independent slice segment header far enough to learn its
    /// `pic_parameter_set_id`, and to establish that the stream does not depend on reference
    /// picture sets we cannot reproduce.
    ///
    /// `Ok(None)` means no slice header has arrived in this submission yet, which is not an error:
    /// a guest may send parameter sets and SEI ahead of the first slice.
    pub fn slice_inspect(&self, annexb: &[u8]) -> Result<Option<u32>, SliceRefusal> {
        let Some(nals) = NalUnits::new(annexb) else {
            return Ok(None);
        };

        for nal in nals {
            // Two bytes of NAL header, and at least one of payload.
            if nal.onward.len() < 3 {
                break;
            }

            let kind = (nal.onward[0] >> 1) & 0x3f;
            if kind > NAL_LAST_VCL {
                continue;
            }

            // The two-byte NAL header is not part of the RBSP.
            let mut r = Reader::new(&nal.onward[2..]);
            let bit = |r: &mut Reader<'_>| r.bit().ok_or(SliceRefusal::Truncated);

            if bit(&mut r)? == 0 {
                // A dependent or later slice segment. Its header needs the CTB address width to
                // parse, and the picture's first slice has already answered every question we
                // have, so there is nothing to learn here.
                continue;
            }

            if (NAL_BLA_W_LP..=NAL_RSV_IRAP23).contains(&kind) {
                bit(&mut r)?; // no_output_of_prior_pics_flag
            }

            let pps_id = r.ue().ok_or(SliceRefusal::Truncated)?;
            if pps_id > 63 {
                return Err(SliceRefusal::PpsId(pps_id));
            }

            // An IDR carries no reference picture set, so there is nothing left to check.
            if kind == NAL_IDR_W_RADL || kind == NAL_IDR_N_LP {
                return Ok(Some(pps_id));
            }

            for _ in 0..self.num_extra_slice_header_bits {
                bit(&mut r)?;
            }
            r.ue().ok_or(SliceRefusal::Truncated)?; // slice_type
            if self.output_flag_present {
                bit(&mut r)?; // pic_output_flag
            }
            if self.separate_colour_plane {
                r.u(2).ok_or(SliceRefusal::Truncated)?; // colour_plane_id
            }
            // slice_pic_order_cnt_lsb
            r.u(u32::from(self.log2_max_pic_order_cnt_lsb_minus4) + 4)
                .ok_or(SliceRefusal::Truncated)?;

            if bit(&mut r)? != 0 {
                return Err(SliceRefusal::SpsRefPicSet);
            }
            if self.num_short_term_ref_pic_sets != 0 && bit(&mut r)? != 0 {
                return Err(SliceRefusal::PredictedRefPicSet);
            }
            return Ok(Some(pps_id));
        }

        Ok(None)
    }
}

/// `nal_unit_header()` (7.3.1.2).
fn write_nal_header(w: &mut Writer, kind: u8) {
    w.flag(false); // forbidden_zero_bit
    w.u(6, u32::from(kind));
    w.u(6, 0); // nuh_layer_id
    w.u(3, 1); // nuh_temporal_id_plus1
}

/// `profile_tier_level(1, 0)` -- 96 bits, no sub-layers (7.3.3).
fn write_ptl(w: &mut Writer, profile: HevcProfile) {
    let idc = profile.idc();

    w.u(2, 0); // general_profile_space
    w.flag(false); // general_tier_flag: main tier
    w.u(5, u32::from(idc)); // general_profile_idc

    // general_profile_compatibility_flag[32]: only our own profile's bit.
    for i in 0..32 {
        w.flag(i == u32::from(idc));
    }

    w.flag(true); // general_progressive_source_flag
    w.flag(false); // general_interlaced_source_flag
    w.flag(true); // general_non_packed_constraint_flag
    w.flag(true); // general_frame_only_constraint_flag

    // general_reserved_zero_43bits, then general_inbld_flag/reserved.
    w.u(22, 0);
    w.u(21, 0);
    w.flag(false);

    w.u(8, u32::from(LEVEL_IDC));
}

/// Whether the effective scaling lists the guest sent are HEVC's own defaults.
///
/// The comparison is on the sorted values, because the scan order VA-API delivers these in is not
/// established -- the same reason the H.264 serializer refuses custom matrices -- and a multiset
/// comparison does not depend on it. A custom list that is a permutation of the default would slip
/// through; nothing plausible produces one.
fn lists_are_default(byte: &impl Fn(usize) -> u8) -> bool {
    let list = |at: usize| -> [u8; 64] { std::array::from_fn(|i| byte(at + i)) };
    let sorted = |mut l: [u8; 64]| {
        l.sort_unstable();
        l
    };
    let intra = sorted(DEFAULT_INTRA_8X8);
    let inter = sorted(DEFAULT_INTER_8X8);

    if (0..6 * 16).any(|i| byte(at::SPS_SCALINGLIST4X4 + i) != 16) {
        return false;
    }
    if (0..6).any(|i| byte(at::SPS_SCALINGLISTDCCOEFF16X16 + i) != 16) {
        return false;
    }
    if (0..2).any(|i| byte(at::SPS_SCALINGLISTDCCOEFF32X32 + i) != 16) {
        return false;
    }

    for m in 0..6 {
        let want = if m < 3 { intra } else { inter };
        if sorted(list(at::SPS_SCALINGLIST8X8 + 64 * m)) != want {
            return false;
        }
        if sorted(list(at::SPS_SCALINGLIST16X16 + 64 * m)) != want {
            return false;
        }
        if m < 2 {
            let want = if m == 0 { intra } else { inter };
            if sorted(list(at::SPS_SCALINGLIST32X32 + 64 * m)) != want {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_that_is_one_tile_carries_no_tile_syntax() {
        let mut desc = PictureDesc::read(&[]);
        desc.pic_width_in_luma_samples = 64;
        desc.pic_height_in_luma_samples = 64;
        desc.chroma_format_idc = 1;
        assert!(desc.tiles.is_none());
        assert!(desc.parameter_sets(64, 64, HevcProfile::Main).is_ok());
    }

    #[test]
    fn more_tiles_than_the_wire_carries_sizes_for_is_refused_rather_than_read_past() {
        let mut desc = PictureDesc::read(&[]);
        desc.pic_width_in_luma_samples = 64;
        desc.pic_height_in_luma_samples = 64;
        desc.chroma_format_idc = 1;
        desc.tiles = Some(Tiles::Overrun);
        assert_eq!(desc.parameter_sets(64, 64, HevcProfile::Main), Err(Unsupported::TileCount));
    }

    #[test]
    fn a_submission_with_no_slice_header_yet_is_not_an_error() {
        let desc = PictureDesc::read(&[]);
        // Not Annex-B at all, and a stream of nothing but a VPS: neither is a refusal.
        assert_eq!(desc.slice_inspect(&[0xde, 0xad]), Ok(None));
        assert_eq!(desc.slice_inspect(&[0, 0, 1, 0x40, 0x01, 0x0c]), Ok(None));
    }

    #[test]
    fn an_idr_answers_the_only_question_asked_of_it() {
        let desc = PictureDesc::read(&[]);
        let mut w = Writer::new(Escape::Rbsp);
        write_nal_header(&mut w, NAL_IDR_W_RADL);
        w.flag(true); // first_slice_segment_in_pic_flag
        w.flag(false); // no_output_of_prior_pics_flag
        w.ue(3); // slice_pic_parameter_set_id
        w.rbsp_trailing();

        let mut stream = vec![0, 0, 0, 1];
        stream.extend_from_slice(&w.finish());
        assert_eq!(desc.slice_inspect(&stream), Ok(Some(3)));
    }
}

/// Diff the three sets and the slice inspection against the C they were ported from.
#[cfg(all(test, feature = "video-oracle"))]
mod oracle {
    use super::*;

    unsafe extern "C" {
        fn virgl_oracle_h265_offsets(out: *mut usize, cap: usize) -> usize;
        fn virgl_oracle_h265_desc_bytes() -> usize;
        fn virgl_oracle_h265_desc_fill(out: *mut u8, seed: u64, break_guard: i32);
        fn virgl_oracle_h265_build(
            desc: *const u8,
            width: u32,
            height: u32,
            profile: u32,
            vps: *mut u8,
            vps_len: *mut usize,
            sps: *mut u8,
            sps_len: *mut usize,
            pps: *mut u8,
            pps_len: *mut usize,
        ) -> i32;
        fn virgl_oracle_h265_slice_inspect(
            annexb: *const u8,
            len: usize,
            desc: *const u8,
            out_id: *mut u32,
        ) -> i32;
    }

    /// The offsets the Rust reader uses, in the order the C shim reports them.
    const OFFSETS: &[usize] = &[
        at::SPS_PIC_WIDTH_IN_LUMA_SAMPLES,
        at::SPS_PIC_HEIGHT_IN_LUMA_SAMPLES,
        at::SPS_CHROMA_FORMAT_IDC,
        at::SPS_SEPARATE_COLOUR_PLANE_FLAG,
        at::SPS_BIT_DEPTH_LUMA_MINUS8,
        at::SPS_BIT_DEPTH_CHROMA_MINUS8,
        at::SPS_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4,
        at::SPS_SPS_MAX_DEC_PIC_BUFFERING_MINUS1,
        at::SPS_LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3,
        at::SPS_LOG2_DIFF_MAX_MIN_LUMA_CODING_BLOCK_SIZE,
        at::SPS_LOG2_MIN_TRANSFORM_BLOCK_SIZE_MINUS2,
        at::SPS_LOG2_DIFF_MAX_MIN_TRANSFORM_BLOCK_SIZE,
        at::SPS_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER,
        at::SPS_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA,
        at::SPS_SCALINGLIST4X4,
        at::SPS_SCALINGLIST8X8,
        at::SPS_SCALINGLIST16X16,
        at::SPS_SCALINGLIST32X32,
        at::SPS_SCALINGLISTDCCOEFF16X16,
        at::SPS_SCALINGLISTDCCOEFF32X32,
        at::SPS_SCALING_LIST_ENABLED_FLAG,
        at::SPS_AMP_ENABLED_FLAG,
        at::SPS_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG,
        at::SPS_PCM_ENABLED_FLAG,
        at::SPS_PCM_SAMPLE_BIT_DEPTH_LUMA_MINUS1,
        at::SPS_PCM_SAMPLE_BIT_DEPTH_CHROMA_MINUS1,
        at::SPS_LOG2_MIN_PCM_LUMA_CODING_BLOCK_SIZE_MINUS3,
        at::SPS_LOG2_DIFF_MAX_MIN_PCM_LUMA_CODING_BLOCK_SIZE,
        at::SPS_PCM_LOOP_FILTER_DISABLED_FLAG,
        at::SPS_NUM_SHORT_TERM_REF_PIC_SETS,
        at::SPS_LONG_TERM_REF_PICS_PRESENT_FLAG,
        at::SPS_NUM_LONG_TERM_REF_PICS_SPS,
        at::SPS_SPS_TEMPORAL_MVP_ENABLED_FLAG,
        at::SPS_STRONG_INTRA_SMOOTHING_ENABLED_FLAG,
        at::PPS_DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG,
        at::PPS_OUTPUT_FLAG_PRESENT_FLAG,
        at::PPS_NUM_EXTRA_SLICE_HEADER_BITS,
        at::PPS_SIGN_DATA_HIDING_ENABLED_FLAG,
        at::PPS_CABAC_INIT_PRESENT_FLAG,
        at::PPS_NUM_REF_IDX_L0_DEFAULT_ACTIVE_MINUS1,
        at::PPS_NUM_REF_IDX_L1_DEFAULT_ACTIVE_MINUS1,
        at::PPS_INIT_QP_MINUS26,
        at::PPS_CONSTRAINED_INTRA_PRED_FLAG,
        at::PPS_TRANSFORM_SKIP_ENABLED_FLAG,
        at::PPS_CU_QP_DELTA_ENABLED_FLAG,
        at::PPS_DIFF_CU_QP_DELTA_DEPTH,
        at::PPS_PPS_CB_QP_OFFSET,
        at::PPS_PPS_CR_QP_OFFSET,
        at::PPS_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG,
        at::PPS_WEIGHTED_PRED_FLAG,
        at::PPS_WEIGHTED_BIPRED_FLAG,
        at::PPS_TRANSQUANT_BYPASS_ENABLED_FLAG,
        at::PPS_TILES_ENABLED_FLAG,
        at::PPS_ENTROPY_CODING_SYNC_ENABLED_FLAG,
        at::PPS_COLUMN_WIDTH_MINUS1,
        at::PPS_ROW_HEIGHT_MINUS1,
        at::PPS_NUM_TILE_COLUMNS_MINUS1,
        at::PPS_NUM_TILE_ROWS_MINUS1,
        at::PPS_UNIFORM_SPACING_FLAG,
        at::PPS_LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG,
        at::PPS_PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
        at::PPS_DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG,
        at::PPS_DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG,
        at::PPS_PPS_DEBLOCKING_FILTER_DISABLED_FLAG,
        at::PPS_PPS_BETA_OFFSET_DIV2,
        at::PPS_PPS_TC_OFFSET_DIV2,
        at::PPS_LISTS_MODIFICATION_PRESENT_FLAG,
        at::PPS_LOG2_PARALLEL_MERGE_LEVEL_MINUS2,
        at::PPS_SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG,
    ];

    #[test]
    fn every_field_is_read_from_where_the_c_struct_puts_it() {
        let mut c = vec![0usize; OFFSETS.len()];
        // SAFETY: `c` is a live slice with its length passed beside it; the C writes exactly that
        // many elements, or none and returns 0 if it has more to report than the buffer holds.
        let n = unsafe { virgl_oracle_h265_offsets(c.as_mut_ptr(), c.len()) };
        assert_eq!(n, OFFSETS.len(), "the C reports a different number of fields");
        assert_eq!(&c[..], OFFSETS);
    }

    /// A descriptor, aligned as the C struct needs it and long enough to be one.
    struct Descriptor(Vec<u64>);

    impl Descriptor {
        fn new(seed: u64, break_guard: i32) -> Descriptor {
            // SAFETY: the C reports the size of its own struct, which is what it then fills.
            let bytes = unsafe { virgl_oracle_h265_desc_bytes() };
            let mut backing = vec![0u64; bytes.div_ceil(8)];
            // SAFETY: the buffer is `bytes` long, rounded up, and 8-byte aligned by its element
            // type, which is at least what a struct of 32-bit members needs.
            unsafe { virgl_oracle_h265_desc_fill(backing.as_mut_ptr().cast(), seed, break_guard) };
            Descriptor(backing)
        }

        fn bytes(&self) -> &[u8] {
            // SAFETY: any initialized `u64` is a valid sequence of bytes, and the slice borrows
            // the same allocation for the same lifetime.
            unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), self.0.len() * 8) }
        }

        fn c_parameter_sets(&self, width: u32, height: u32) -> Option<ParameterSets> {
            // VIRGL_H265_PS_MAX, the C's own fixed capacity for each set.
            let (mut vps, mut sps, mut pps) = (vec![0u8; 512], vec![0u8; 512], vec![0u8; 512]);
            let (mut vn, mut sn, mut pn) = (0usize, 0usize, 0usize);
            // SAFETY: the descriptor is at least the size the C reported for its struct and is
            // aligned for it; the three output buffers are `VIRGL_H265_PS_MAX`, which is the
            // capacity the C copies out of, and the three lengths are live.
            let rc = unsafe {
                virgl_oracle_h265_build(
                    self.0.as_ptr().cast(),
                    width,
                    height,
                    Profile::HevcMain as u32,
                    vps.as_mut_ptr(),
                    &mut vn,
                    sps.as_mut_ptr(),
                    &mut sn,
                    pps.as_mut_ptr(),
                    &mut pn,
                )
            };
            if rc != 0 {
                return None;
            }
            vps.truncate(vn);
            sps.truncate(sn);
            pps.truncate(pn);
            Some(ParameterSets { vps, sps, pps })
        }

        /// What the C makes of a submission: `Ok(Some(id))`, `Ok(None)` for its 1, `Err` for -1.
        fn c_slice_inspect(&self, annexb: &[u8]) -> Result<Option<u32>, ()> {
            let mut id = 0u32;
            // SAFETY: `annexb` is a live slice with its length beside it, the descriptor is the
            // C's own struct size and alignment, and `id` is a live u32 the C writes only on
            // success. It reads only.
            let rc = unsafe {
                virgl_oracle_h265_slice_inspect(
                    annexb.as_ptr(),
                    annexb.len(),
                    self.0.as_ptr().cast(),
                    &mut id,
                )
            };
            match rc {
                0 => Ok(Some(id)),
                1 => Ok(None),
                _ => Err(()),
            }
        }
    }

    /// A deterministic source, so a failure names a case that can be re-run.
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

    #[test]
    fn the_parameter_sets_are_the_bytes_the_c_writes() {
        // Display sizes at and inside the coded size the generator picks, so the conformance
        // window is exercised both present and absent.
        let sizes = [(1920u32, 1080u32), (1920, 1088), (1912, 1080), (1280, 720)];

        let mut written = 0;
        let mut refused = 0;
        for seed in 0..96u64 {
            // 7 is the tile overrun, where the two deliberately disagree; it has its own test.
            for guard in [0, 1, 2, 3, 4, 5, 6, 8, 9] {
                let desc = Descriptor::new(seed, guard);
                let rust = PictureDesc::read(desc.bytes());

                for (i, &(w, h)) in sizes.iter().enumerate() {
                    let ours = rust.parameter_sets(w, h, HevcProfile::Main).ok();
                    let theirs = desc.c_parameter_sets(w, h);
                    assert_eq!(ours, theirs, "seed {seed}, guard {guard}, {w}x{h}, size {i}");
                    if ours.is_some() {
                        written += 1;
                    } else {
                        refused += 1;
                    }
                }
            }
        }

        // Both halves have to be reached, or the differential is only proving one of them.
        assert!(written > 300, "only {written} descriptors produced bytes");
        assert!(refused > 500, "only {refused} descriptors were refused");
    }

    #[test]
    fn a_tile_count_past_the_wires_arrays_is_refused_rather_than_read_past() {
        // The C reads `num_tile_columns_minus1` column widths out of a twenty-entry array, so a
        // guest that asks for forty gets forty, forty-widths' worth of whatever follows the array
        // serialized into its PPS. The Rust refuses instead: a count and the array it indexes are
        // one value, reconciled where the descriptor is read, and a count the array cannot answer
        // for is not a number to clamp.
        for seed in 0..8u64 {
            let desc = Descriptor::new(seed, 7);
            let rust = PictureDesc::read(desc.bytes());
            assert_eq!(rust.tiles, Some(Tiles::Overrun));
            assert_eq!(
                rust.parameter_sets(1920, 1080, HevcProfile::Main),
                Err(Unsupported::TileCount)
            );
            assert!(desc.c_parameter_sets(1920, 1080).is_some(), "the C refused it after all");
        }
    }

    #[test]
    fn a_missing_extent_is_refused_on_both_sides() {
        let desc = Descriptor::new(3, 0);
        let rust = PictureDesc::read(desc.bytes());
        for (w, h) in [(0u32, 0u32), (0, 1080), (1920, 0)] {
            assert_eq!(rust.parameter_sets(w, h, HevcProfile::Main), Err(Unsupported::NoGeometry));
            assert!(desc.c_parameter_sets(w, h).is_none());
        }
    }

    /// A stream built out of the bytes the walk turns on: start codes, HEVC's two-byte NAL
    /// headers weighted towards the VCL types, and payload that is mostly slice-header bits.
    fn stream(rng: &mut Rng, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            match rng.below(10) {
                0..=2 => out.push(0),
                3 => out.push(1),
                4 => out.extend_from_slice(&[0, 0, 1]),
                5 => out.extend_from_slice(&[0, 0, 0, 1]),
                6..=7 => {
                    // nal_unit_type in the low VCL range or around the IRAP types, layer 0,
                    // temporal id 1.
                    let kind = rng.below(36) as u8;
                    out.extend_from_slice(&[kind << 1, 0x01]);
                }
                _ => out.push(rng.next() as u8),
            }
        }
        out.truncate(len);
        out
    }

    #[test]
    fn the_slice_inspection_reaches_the_same_verdict_as_the_c() {
        let mut rng = Rng(0x511ce);
        let mut answered = 0;
        let mut refused = 0;
        for seed in 0..8u64 {
            let desc = Descriptor::new(seed, 0);
            let rust = PictureDesc::read(desc.bytes());
            for len in 0..300 {
                let s = stream(&mut rng, len);
                let ours = rust.slice_inspect(&s).map_err(|_| ());
                let theirs = desc.c_slice_inspect(&s);
                assert_eq!(ours, theirs, "seed {seed}, {s:02x?}");
                match ours {
                    Ok(Some(_)) => answered += 1,
                    Err(()) => refused += 1,
                    Ok(None) => {}
                }
            }
        }
        assert!(answered > 50, "only {answered} submissions carried a slice header");
        assert!(refused > 50, "only {refused} submissions were refused");
    }

    #[test]
    fn a_real_slice_header_reaches_the_same_verdict_on_both_sides() {
        let desc = Descriptor::new(11, 0);
        let rust = PictureDesc::read(desc.bytes());

        for id in [0u32, 1, 63] {
            let mut w = Writer::new(Escape::Rbsp);
            write_nal_header(&mut w, NAL_IDR_N_LP);
            w.flag(true); // first_slice_segment_in_pic_flag
            w.flag(false); // no_output_of_prior_pics_flag
            w.ue(id);
            w.rbsp_trailing();

            let mut s = vec![0, 0, 0, 1];
            s.extend_from_slice(&w.finish());
            assert_eq!(rust.slice_inspect(&s), Ok(Some(id)));
            assert_eq!(desc.c_slice_inspect(&s), Ok(Some(id)));
        }

        // A trailing picture that indexes an SPS short term ref pic set: refused, one frame in.
        let mut w = Writer::new(Escape::Rbsp);
        write_nal_header(&mut w, 1); // TRAIL_R
        w.flag(true); // first_slice_segment_in_pic_flag
        w.ue(0); // slice_pic_parameter_set_id
        for _ in 0..rust.num_extra_slice_header_bits {
            w.flag(false);
        }
        w.ue(1); // slice_type
        if rust.output_flag_present {
            w.flag(true);
        }
        w.u(u32::from(rust.log2_max_pic_order_cnt_lsb_minus4) + 4, 0);
        w.flag(true); // short_term_ref_pic_set_sps_flag
        w.rbsp_trailing();

        let mut s = vec![0, 0, 0, 1];
        s.extend_from_slice(&w.finish());
        assert_eq!(rust.slice_inspect(&s), Err(SliceRefusal::SpsRefPicSet));
        assert_eq!(desc.c_slice_inspect(&s), Err(()));
    }
}
