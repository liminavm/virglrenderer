// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! H.264 framing and parameter-set synthesis for the VideoToolbox backend.
//!
//! Written against ITU-T H.264: 7.3.2.1.1 for the SPS, 7.3.2.2 for the PPS, 7.3.3 for the slice
//! header, 7.4.1.1 for emulation prevention.

use std::fmt;

use super::Profile;
use super::bitstream::{Escape, NalUnits, Reader, Writer};

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

/// The H.264 profiles this build serves, and the only thing an SPS needs from one.
///
/// Three, not the wire's four: Baseline and Constrained Baseline write the same `profile_idc`,
/// and a constraint flag is the only thing that distinguishes them -- which this SPS clears,
/// because a constraint flag only ever narrows a profile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum H264Profile {
    Baseline,
    Main,
    High,
}

impl H264Profile {
    /// The H.264 profile a wire profile names, or `None` if it names another codec.
    pub fn of(profile: Profile) -> Option<H264Profile> {
        Some(match profile {
            Profile::H264Baseline | Profile::H264ConstrainedBaseline => H264Profile::Baseline,
            Profile::H264Main => H264Profile::Main,
            Profile::H264High => H264Profile::High,
            _ => return None,
        })
    }

    /// `profile_idc` (A.2).
    fn idc(self) -> u8 {
        match self {
            H264Profile::Baseline => 66,
            H264Profile::Main => 77,
            H264Profile::High => 100,
        }
    }

    /// Whether the SPS carries the chroma format and bit depth block (7.3.2.1.1).
    ///
    /// Present for the High family only. Emitting it for a Baseline or Main SPS would not merely
    /// be redundant: the decoder reads the next field as `log2_max_frame_num_minus4` and
    /// everything after it shifts.
    fn has_chroma_block(self) -> bool {
        matches!(self, H264Profile::High)
    }
}

/// Why a descriptor could not be serialized.
///
/// Each is a stream this build refuses rather than writing an approximation of -- a parameter set
/// that is nearly right decodes into mush, or worse, decodes cleanly and wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unsupported {
    /// The codec object was created with no extent to size the picture from.
    NoGeometry,
    FieldCoding,
    /// `chroma_format_idc`, which must be 1 (4:2:0). `separate_colour_plane_flag` implies 4:4:4
    /// and arrives here as 3.
    Chroma(u8),
    BitDepth,
    /// `frame_mbs_only_flag` clear. The height derivation counts map units as whole-frame
    /// macroblock rows, which only holds for a progressive stream.
    FieldCapable,
    /// Slice groups (FMO). The slice-group map syntax that follows a non-zero count is not on the
    /// wire, so a stream using it cannot be described.
    SliceGroups,
    /// A custom scaling matrix. The wire lists come straight from VA-API's `VAIQMatrixBufferH264`,
    /// and the scan order they are in has not been verified against a real stream carrying a
    /// non-flat matrix. Emitting them in the wrong order is invisible -- the stream decodes, with
    /// subtly wrong dequantization -- which is exactly the failure this must not produce.
    ScalingMatrix,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::NoGeometry => write!(f, "the codec object has no picture size"),
            Unsupported::FieldCoding => write!(f, "field coding is not supported"),
            Unsupported::Chroma(idc) => {
                write!(f, "chroma_format_idc {idc} is not supported (4:2:0 only)")
            }
            Unsupported::BitDepth => write!(f, "a bit depth over 8 is not supported"),
            Unsupported::FieldCapable => {
                write!(f, "field-capable streams (frame_mbs_only_flag 0) are not supported")
            }
            Unsupported::SliceGroups => write!(f, "slice groups (FMO) are not supported"),
            Unsupported::ScalingMatrix => write!(f, "custom scaling matrices are not supported"),
        }
    }
}

/// The parameter sets VideoToolbox builds a format description from.
///
/// `CMVideoFormatDescriptionCreateFromH264ParameterSets` wants real SPS and PPS *bytes*. The guest
/// sends the parsed semantic content instead -- the gallium hardware-decoder shape -- so the bytes
/// are written here.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParameterSets {
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
}

/// The parts of `struct virgl_h264_picture_desc` a parameter set is written out of.
///
/// Only the fields the SPS and PPS need, and each reduced to what it means: the scaling lists
/// become the one bit anything asks of them, and `offset_for_ref_frame` becomes exactly as long as
/// the count that describes it, so the two cannot be handed on disagreeing.
pub struct PictureDesc {
    pub level_idc: u8,
    pub chroma_format_idc: u8,
    pub separate_colour_plane: bool,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_frame_num_minus4: u8,
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub delta_pic_order_always_zero: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    /// One entry per `num_ref_frames_in_pic_order_cnt_cycle`; the count is its length.
    pub offset_for_ref_frame: Vec<i32>,
    pub frame_mbs_only: bool,
    pub direct_8x8_inference: bool,

    pub entropy_coding_mode: bool,
    pub bottom_field_pic_order_in_frame_present: bool,
    pub num_slice_groups_minus1: u8,
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    pub second_chroma_qp_index_offset: i8,
    /// Whether the guest sent a scaling matrix that is neither absent nor flat. A flat list
    /// decodes identically to no list at all, and an all-zero array means the guest sent no
    /// IQMatrix at all -- neither needs signalling, and neither is refused.
    pub scaling_matrix: bool,

    pub field_pic: bool,
    /// The DPB size, and a trap: `struct virgl_h264_sps` has a `max_num_ref_frames` and it is dead
    /// on the decode path. mesa's VA frontend puts VA-API's `num_ref_frames` in the picture
    /// descriptor, and only the encoder frontend writes the SPS one. Reading the SPS field yields
    /// 0, which VideoToolbox honours: it sizes the DPB at zero, drops every reference, and rejects
    /// the third frame onward while the first two decode fine.
    pub num_ref_frames: u8,
    /// The same trap in the PPS: `num_ref_idx_l{0,1}_default_active_minus1` are on the wire and
    /// dead. mesa fills the per-slice values into the picture descriptor instead. Using the
    /// per-slice value as the PPS default is right in both directions -- a slice that overrides it
    /// carries `num_ref_idx_active_override_flag` in its own header, which passes through
    /// untouched and still wins.
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
}

/// Where each field sits in `struct virgl_h264_picture_desc`, measured with `offsetof` rather than
/// counted by hand, and checked against it by the video oracle.
mod at {
    pub const LEVEL_IDC: usize = 264;
    pub const CHROMA_FORMAT_IDC: usize = 265;
    pub const SEPARATE_COLOUR_PLANE_FLAG: usize = 266;
    pub const BIT_DEPTH_LUMA_MINUS8: usize = 267;
    pub const BIT_DEPTH_CHROMA_MINUS8: usize = 268;
    pub const LOG2_MAX_FRAME_NUM_MINUS4: usize = 750;
    pub const PIC_ORDER_CNT_TYPE: usize = 751;
    pub const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: usize = 752;
    pub const DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG: usize = 753;
    pub const OFFSET_FOR_NON_REF_PIC: usize = 756;
    pub const OFFSET_FOR_TOP_TO_BOTTOM_FIELD: usize = 760;
    pub const OFFSET_FOR_REF_FRAME: usize = 764;
    pub const NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE: usize = 1788;
    pub const FRAME_MBS_ONLY_FLAG: usize = 1790;
    pub const DIRECT_8X8_INFERENCE_FLAG: usize = 1792;

    pub const ENTROPY_CODING_MODE_FLAG: usize = 1796;
    pub const BOTTOM_FIELD_PIC_ORDER_IN_FRAME_PRESENT_FLAG: usize = 1797;
    pub const NUM_SLICE_GROUPS_MINUS1: usize = 1798;
    pub const WEIGHTED_PRED_FLAG: usize = 1803;
    pub const WEIGHTED_BIPRED_IDC: usize = 1804;
    pub const PIC_INIT_QP_MINUS26: usize = 1805;
    pub const PIC_INIT_QS_MINUS26: usize = 1806;
    pub const CHROMA_QP_INDEX_OFFSET: usize = 1807;
    pub const DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG: usize = 1808;
    pub const CONSTRAINED_INTRA_PRED_FLAG: usize = 1809;
    pub const REDUNDANT_PIC_CNT_PRESENT_FLAG: usize = 1810;
    pub const TRANSFORM_8X8_MODE_FLAG: usize = 1811;
    pub const PPS_SCALING_LIST_4X4: usize = 1812;
    pub const PPS_SCALING_LIST_8X8: usize = 1908;
    pub const SECOND_CHROMA_QP_INDEX_OFFSET: usize = 2292;

    pub const FIELD_PIC_FLAG: usize = 2300;
    pub const NUM_REF_IDX_L0_ACTIVE_MINUS1: usize = 2302;
    pub const NUM_REF_IDX_L1_ACTIVE_MINUS1: usize = 2303;
    pub const NUM_REF_FRAMES: usize = 2621;

    /// The 4x4 lists are six of sixteen bytes. All six are meaningful.
    pub const SCALING_4X4_BYTES: usize = 6 * 16;
    /// Of the six 8x8 lists, only two are meaningful for 4:2:0 -- mesa copies exactly `2 * 64`
    /// bytes from VA-API and leaves the rest of the `[6][64]` array untouched, so reading all six
    /// would be reading whatever happened to be in the struct.
    pub const SCALING_8X8_BYTES: usize = 2 * 64;
}

impl PictureDesc {
    /// Read a descriptor the guest wrote.
    ///
    /// Total on purpose: a short descriptor reads as zeros rather than failing, which is what the
    /// C does and what the protocol allows -- the resource carries whatever the guest's driver
    /// wrote, and its size is the guest's choice. A stream that needs a field the guest did not
    /// send is refused later, by name, rather than here by length.
    pub fn read(blob: &[u8]) -> PictureDesc {
        let byte = |at: usize| blob.get(at).copied().unwrap_or(0);
        let flag = |at: usize| byte(at) != 0;
        let signed = |at: usize| byte(at) as i8;
        let word =
            |at: usize| i32::from_le_bytes([byte(at), byte(at + 1), byte(at + 2), byte(at + 3)]);
        // A flat list decodes identically to no list at all, and an all-zero array is a guest that
        // sent no IQMatrix. Neither is a matrix to serialize.
        let is_matrix = |at: usize, n: usize| {
            let mut zero = true;
            let mut flat = true;
            for i in 0..n {
                zero &= byte(at + i) == 0;
                flat &= byte(at + i) == 16;
            }
            !zero && !flat
        };

        let cycle = byte(at::NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE) as usize;

        PictureDesc {
            level_idc: byte(at::LEVEL_IDC),
            chroma_format_idc: byte(at::CHROMA_FORMAT_IDC),
            separate_colour_plane: flag(at::SEPARATE_COLOUR_PLANE_FLAG),
            bit_depth_luma_minus8: byte(at::BIT_DEPTH_LUMA_MINUS8),
            bit_depth_chroma_minus8: byte(at::BIT_DEPTH_CHROMA_MINUS8),
            log2_max_frame_num_minus4: byte(at::LOG2_MAX_FRAME_NUM_MINUS4),
            pic_order_cnt_type: byte(at::PIC_ORDER_CNT_TYPE),
            log2_max_pic_order_cnt_lsb_minus4: byte(at::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4),
            delta_pic_order_always_zero: flag(at::DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG),
            offset_for_non_ref_pic: word(at::OFFSET_FOR_NON_REF_PIC),
            offset_for_top_to_bottom_field: word(at::OFFSET_FOR_TOP_TO_BOTTOM_FIELD),
            offset_for_ref_frame: (0..cycle)
                .map(|i| word(at::OFFSET_FOR_REF_FRAME + 4 * i))
                .collect(),
            frame_mbs_only: flag(at::FRAME_MBS_ONLY_FLAG),
            direct_8x8_inference: flag(at::DIRECT_8X8_INFERENCE_FLAG),

            entropy_coding_mode: flag(at::ENTROPY_CODING_MODE_FLAG),
            bottom_field_pic_order_in_frame_present: flag(
                at::BOTTOM_FIELD_PIC_ORDER_IN_FRAME_PRESENT_FLAG,
            ),
            num_slice_groups_minus1: byte(at::NUM_SLICE_GROUPS_MINUS1),
            weighted_pred: flag(at::WEIGHTED_PRED_FLAG),
            weighted_bipred_idc: byte(at::WEIGHTED_BIPRED_IDC),
            pic_init_qp_minus26: signed(at::PIC_INIT_QP_MINUS26),
            pic_init_qs_minus26: signed(at::PIC_INIT_QS_MINUS26),
            chroma_qp_index_offset: signed(at::CHROMA_QP_INDEX_OFFSET),
            deblocking_filter_control_present: flag(at::DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG),
            constrained_intra_pred: flag(at::CONSTRAINED_INTRA_PRED_FLAG),
            redundant_pic_cnt_present: flag(at::REDUNDANT_PIC_CNT_PRESENT_FLAG),
            transform_8x8_mode: flag(at::TRANSFORM_8X8_MODE_FLAG),
            second_chroma_qp_index_offset: signed(at::SECOND_CHROMA_QP_INDEX_OFFSET),
            scaling_matrix: is_matrix(at::PPS_SCALING_LIST_4X4, at::SCALING_4X4_BYTES)
                || is_matrix(at::PPS_SCALING_LIST_8X8, at::SCALING_8X8_BYTES),

            field_pic: flag(at::FIELD_PIC_FLAG),
            num_ref_frames: byte(at::NUM_REF_FRAMES),
            num_ref_idx_l0_active_minus1: byte(at::NUM_REF_IDX_L0_ACTIVE_MINUS1),
            num_ref_idx_l1_active_minus1: byte(at::NUM_REF_IDX_L1_ACTIVE_MINUS1),
        }
    }

    /// Write the SPS and PPS a session for this picture is built around.
    ///
    /// `width` and `height` are the codec object's display size: there is no `pic_width_in_mbs` on
    /// the wire, so the macroblock counts are rounded up from it and the remainder cropped away.
    /// Without the cropping a stream whose width is not a multiple of 16 decodes at the padded
    /// size.
    ///
    /// `pps_id` is the one the guest's slices reference, read back by [`slice_pps_id`]: nothing on
    /// the wire says which the guest chose, and a PPS bearing the wrong one is not found.
    pub fn parameter_sets(
        &self,
        width: u32,
        height: u32,
        profile: H264Profile,
        pps_id: u32,
    ) -> Result<ParameterSets, Unsupported> {
        if width == 0 || height == 0 {
            return Err(Unsupported::NoGeometry);
        }
        if self.field_pic {
            return Err(Unsupported::FieldCoding);
        }
        if self.chroma_format_idc != 1 || self.separate_colour_plane {
            return Err(Unsupported::Chroma(if self.separate_colour_plane {
                3
            } else {
                self.chroma_format_idc
            }));
        }
        if self.bit_depth_luma_minus8 != 0 || self.bit_depth_chroma_minus8 != 0 {
            return Err(Unsupported::BitDepth);
        }
        if !self.frame_mbs_only {
            return Err(Unsupported::FieldCapable);
        }
        if self.num_slice_groups_minus1 != 0 {
            return Err(Unsupported::SliceGroups);
        }
        if self.scaling_matrix {
            return Err(Unsupported::ScalingMatrix);
        }

        // seq_parameter_set_id is 0 throughout: the PPS below is the only thing that refers to it,
        // and nothing in the guest's slice data does.
        const SPS_ID: u32 = 0;

        Ok(ParameterSets {
            sps: self.write_sps(width, height, profile, SPS_ID),
            pps: self.write_pps(pps_id, SPS_ID),
        })
    }

    /// 7.3.2.1.1.
    fn write_sps(&self, width: u32, height: u32, profile: H264Profile, sps_id: u32) -> Vec<u8> {
        let mut w = Writer::new(Escape::Rbsp);

        // NAL header: forbidden_zero = 0, nal_ref_idc = 3, nal_unit_type = 7 (SPS).
        w.raw_byte(0x67);

        w.u(8, u32::from(profile.idc()));
        // constraint_set0..5 and two reserved bits, all clear. They only ever narrow a profile,
        // so clearing them is always safe.
        w.u(8, 0);
        w.u(8, u32::from(self.level_idc));
        w.ue(sps_id);

        if profile.has_chroma_block() {
            w.ue(u32::from(self.chroma_format_idc));
            if self.chroma_format_idc == 3 {
                w.flag(self.separate_colour_plane);
            }
            w.ue(u32::from(self.bit_depth_luma_minus8));
            w.ue(u32::from(self.bit_depth_chroma_minus8));
            // qpprime_y_zero_transform_bypass_flag: lossless coding, not on the wire.
            w.flag(false);
            // seq_scaling_matrix_present_flag -- see PictureDesc::scaling_matrix.
            w.flag(false);
        }

        w.ue(u32::from(self.log2_max_frame_num_minus4));
        w.ue(u32::from(self.pic_order_cnt_type));
        if self.pic_order_cnt_type == 0 {
            w.ue(u32::from(self.log2_max_pic_order_cnt_lsb_minus4));
        } else if self.pic_order_cnt_type == 1 {
            w.flag(self.delta_pic_order_always_zero);
            w.se(self.offset_for_non_ref_pic);
            w.se(self.offset_for_top_to_bottom_field);
            w.ue(self.offset_for_ref_frame.len() as u32);
            for &offset in &self.offset_for_ref_frame {
                w.se(offset);
            }
        }

        // Clamped to at least 1: a stream with inter prediction needs somewhere to keep the
        // picture it predicts from, and 0 is what "the guest did not say" looks like.
        w.ue(u32::from(self.num_ref_frames.max(1)));
        w.flag(false); // gaps_in_frame_num_value_allowed_flag

        // Geometry. Chroma units for 4:2:0 are 2 luma samples horizontally, and 2 vertically for
        // a frame-only stream, which the refusal above has made this one.
        let mbs_w = width.div_ceil(16);
        let map_h = height.div_ceil(16);
        let crop_r = (mbs_w * 16 - width) / 2;
        let crop_b = (map_h * 16 - height) / 2;

        w.ue(mbs_w - 1);
        w.ue(map_h - 1);
        w.flag(true); // frame_mbs_only_flag: progressive only, enforced above
        w.flag(self.direct_8x8_inference);

        if crop_r != 0 || crop_b != 0 {
            w.flag(true);
            w.ue(0); // left
            w.ue(crop_r);
            w.ue(0); // top
            w.ue(crop_b);
        } else {
            w.flag(false);
        }

        // vui_parameters_present_flag: VUI carries timing and colour metadata the guest already
        // applies itself.
        w.flag(false);
        w.rbsp_trailing();
        w.finish()
    }

    /// 7.3.2.2.
    fn write_pps(&self, pps_id: u32, sps_id: u32) -> Vec<u8> {
        let mut w = Writer::new(Escape::Rbsp);

        // NAL header: nal_ref_idc = 3, nal_unit_type = 8 (PPS).
        w.raw_byte(0x68);

        w.ue(pps_id);
        w.ue(sps_id);
        w.flag(self.entropy_coding_mode);
        w.flag(self.bottom_field_pic_order_in_frame_present);
        // num_slice_groups_minus1: a non-zero count is refused above, so this is always 0.
        w.ue(0);
        w.ue(u32::from(self.num_ref_idx_l0_active_minus1));
        w.ue(u32::from(self.num_ref_idx_l1_active_minus1));
        w.flag(self.weighted_pred);
        w.u(2, u32::from(self.weighted_bipred_idc));
        w.se(i32::from(self.pic_init_qp_minus26));
        w.se(i32::from(self.pic_init_qs_minus26));
        w.se(i32::from(self.chroma_qp_index_offset));
        w.flag(self.deblocking_filter_control_present);
        w.flag(self.constrained_intra_pred);
        w.flag(self.redundant_pic_cnt_present);

        // The optional tail. transform_8x8_mode_flag is a real decoding parameter and is on the
        // wire, so it must be emitted when set -- a High-profile stream that uses 8x8 transforms
        // decodes into mush without it. Reaching it means writing the whole trailing group.
        if self.transform_8x8_mode || self.second_chroma_qp_index_offset != 0 {
            w.flag(self.transform_8x8_mode);
            // pic_scaling_matrix_present_flag -- see PictureDesc::scaling_matrix.
            w.flag(false);
            w.se(i32::from(self.second_chroma_qp_index_offset));
        }

        w.rbsp_trailing();
        w.finish()
    }
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

    // ------------------------------------------------------------ parameter sets

    unsafe extern "C" {
        fn virgl_oracle_h264_offsets(out: *mut usize, cap: usize) -> usize;
        fn virgl_oracle_h264_desc_bytes() -> usize;
        fn virgl_oracle_h264_desc_fill(out: *mut u8, seed: u64, break_guard: i32);
        #[allow(clippy::too_many_arguments)]
        fn virgl_oracle_h264_build(
            desc: *const u8,
            width: u32,
            height: u32,
            profile: u32,
            pps_id: u32,
            sps: *mut u8,
            sps_len: *mut usize,
            pps: *mut u8,
            pps_len: *mut usize,
        ) -> i32;
    }

    /// The offsets the Rust reader uses, in the order the C shim reports them.
    const OFFSETS: &[usize] = &[
        at::LEVEL_IDC,
        at::CHROMA_FORMAT_IDC,
        at::SEPARATE_COLOUR_PLANE_FLAG,
        at::BIT_DEPTH_LUMA_MINUS8,
        at::BIT_DEPTH_CHROMA_MINUS8,
        at::LOG2_MAX_FRAME_NUM_MINUS4,
        at::PIC_ORDER_CNT_TYPE,
        at::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4,
        at::DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG,
        at::OFFSET_FOR_NON_REF_PIC,
        at::OFFSET_FOR_TOP_TO_BOTTOM_FIELD,
        at::OFFSET_FOR_REF_FRAME,
        at::NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE,
        at::FRAME_MBS_ONLY_FLAG,
        at::DIRECT_8X8_INFERENCE_FLAG,
        at::ENTROPY_CODING_MODE_FLAG,
        at::BOTTOM_FIELD_PIC_ORDER_IN_FRAME_PRESENT_FLAG,
        at::NUM_SLICE_GROUPS_MINUS1,
        at::WEIGHTED_PRED_FLAG,
        at::WEIGHTED_BIPRED_IDC,
        at::PIC_INIT_QP_MINUS26,
        at::PIC_INIT_QS_MINUS26,
        at::CHROMA_QP_INDEX_OFFSET,
        at::DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG,
        at::CONSTRAINED_INTRA_PRED_FLAG,
        at::REDUNDANT_PIC_CNT_PRESENT_FLAG,
        at::TRANSFORM_8X8_MODE_FLAG,
        at::PPS_SCALING_LIST_4X4,
        at::PPS_SCALING_LIST_8X8,
        at::SECOND_CHROMA_QP_INDEX_OFFSET,
        at::FIELD_PIC_FLAG,
        at::NUM_REF_IDX_L0_ACTIVE_MINUS1,
        at::NUM_REF_IDX_L1_ACTIVE_MINUS1,
        at::NUM_REF_FRAMES,
    ];

    #[test]
    fn every_field_is_read_from_where_the_c_struct_puts_it() {
        let mut c = vec![0usize; OFFSETS.len()];
        // SAFETY: `c` is a live slice with its length passed beside it; the C writes exactly that
        // many elements, or none and returns 0 if it has more to report than the buffer holds.
        let n = unsafe { virgl_oracle_h264_offsets(c.as_mut_ptr(), c.len()) };
        assert_eq!(n, OFFSETS.len(), "the C reports a different number of fields");
        assert_eq!(&c[..], OFFSETS);
    }

    /// A descriptor, aligned as the C struct needs it and long enough to be one.
    ///
    /// `u64` backing rather than `u8`: the struct has 32-bit members, and casting an under-aligned
    /// buffer to it is undefined however well it would happen to work here.
    struct Descriptor(Vec<u64>);

    impl Descriptor {
        fn new(seed: u64, break_guard: i32) -> Descriptor {
            // SAFETY: the C reports the size of its own struct, which is what it then fills.
            let bytes = unsafe { virgl_oracle_h264_desc_bytes() };
            let mut backing = vec![0u64; bytes.div_ceil(8)];
            // SAFETY: the buffer is `bytes` long, rounded up, and 8-byte aligned by its element
            // type, which is at least what a struct of 32-bit members needs.
            unsafe { virgl_oracle_h264_desc_fill(backing.as_mut_ptr().cast(), seed, break_guard) };
            Descriptor(backing)
        }

        fn bytes(&self) -> &[u8] {
            // SAFETY: any initialized `u64` is a valid sequence of bytes, and the slice borrows
            // the same allocation for the same lifetime.
            unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), self.0.len() * 8) }
        }

        /// What the C makes of it: the two sets, or `None` for a refusal.
        fn c_parameter_sets(
            &self,
            width: u32,
            height: u32,
            profile: Profile,
            pps_id: u32,
        ) -> Option<(Vec<u8>, Vec<u8>)> {
            // VIRGL_H264_PS_MAX, the C's own fixed capacity for each set.
            let mut sps = vec![0u8; 512];
            let mut pps = vec![0u8; 512];
            let (mut sps_len, mut pps_len) = (0usize, 0usize);
            // SAFETY: the descriptor is at least the size the C reported for its struct and is
            // aligned for it; both output buffers are `VIRGL_H264_PS_MAX`, which is the capacity
            // the C copies out of, and both lengths are live.
            let rc = unsafe {
                virgl_oracle_h264_build(
                    self.0.as_ptr().cast(),
                    width,
                    height,
                    profile as u32,
                    pps_id,
                    sps.as_mut_ptr(),
                    &mut sps_len,
                    pps.as_mut_ptr(),
                    &mut pps_len,
                )
            };
            if rc != 0 {
                return None;
            }
            sps.truncate(sps_len);
            pps.truncate(pps_len);
            Some((sps, pps))
        }
    }

    #[test]
    fn the_parameter_sets_are_the_bytes_the_c_writes() {
        let profiles = [
            Profile::H264Baseline,
            Profile::H264ConstrainedBaseline,
            Profile::H264Main,
            Profile::H264High,
        ];
        // Sizes on and off a macroblock boundary, so the cropping is exercised both ways.
        let sizes = [(1920u32, 1080u32), (1280, 720), (640, 482), (17, 17), (16, 16)];

        let mut agreed = 0;
        let mut refused = 0;
        for seed in 0..64u64 {
            // 0 breaks nothing; 1..=10 break one guard each, which is diffed as a refusal.
            for guard in 0..=10 {
                let desc = Descriptor::new(seed, guard);
                let rust = PictureDesc::read(desc.bytes());

                for (i, &profile) in profiles.iter().enumerate() {
                    let (w, h) = sizes[(seed as usize + i) % sizes.len()];
                    let pps_id = (seed as u32 + i as u32) % 256;

                    let ours = rust
                        .parameter_sets(w, h, H264Profile::of(profile).unwrap(), pps_id)
                        .map(|s| (s.sps, s.pps))
                        .ok();
                    let theirs = desc.c_parameter_sets(w, h, profile, pps_id);

                    assert_eq!(ours, theirs, "seed {seed}, guard {guard}, {profile:?}, {w}x{h}");
                    if ours.is_some() {
                        agreed += 1;
                    } else {
                        refused += 1;
                    }
                }
            }
        }

        // Both halves have to be reached, or the differential is only proving one of them.
        assert!(agreed > 500, "only {agreed} descriptors produced bytes");
        assert!(refused > 500, "only {refused} descriptors were refused");
    }

    #[test]
    fn a_missing_extent_is_refused_on_both_sides() {
        let desc = Descriptor::new(1, 0);
        let rust = PictureDesc::read(desc.bytes());
        for (w, h) in [(0u32, 0u32), (0, 1080), (1920, 0)] {
            assert_eq!(
                rust.parameter_sets(w, h, H264Profile::High, 0),
                Err(Unsupported::NoGeometry)
            );
            assert!(desc.c_parameter_sets(w, h, Profile::H264High, 0).is_none());
        }
    }

    #[test]
    fn a_cycle_the_cs_fixed_buffer_could_not_hold_is_written_anyway() {
        // 256 entries of signed Exp-Golomb overflow VIRGL_H264_PS_MAX, and the C answers that by
        // refusing a stream it otherwise supports. The Rust writer grows, so this is a deliberate
        // divergence rather than an oversight: the SPS is valid, and a decoder is handed a pointer
        // and a length, not a 512-byte box.
        let desc = Descriptor::new(7, 0);
        let mut rust = PictureDesc::read(desc.bytes());
        rust.pic_order_cnt_type = 1;
        rust.offset_for_ref_frame = (0..256).map(|i| i32::MIN / 2 + i).collect();

        let sets = rust.parameter_sets(1920, 1080, H264Profile::High, 0).unwrap();
        assert!(sets.sps.len() > 512, "an SPS of {} bytes", sets.sps.len());
    }
}
