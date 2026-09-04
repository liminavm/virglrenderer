// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! VideoToolbox: the only hardware video decoder a macOS host has.
//!
//! There is no VA-API here, and no DRM node to open. A guest that asks virgl for hardware decode
//! is asking for this, and the whole of the host side is a `VTDecompressionSession` fed access
//! units and answering with `CVImageBuffer`s.
//!
//! One of the named unsafe modules (CLAUDE.md). The unsafe is foreign calls into VideoToolbox,
//! CoreMedia and CoreFoundation, and every safe type here exists to make a foreign object's
//! lifetime something the compiler tracks rather than something a reviewer checks.
//!
//! **VP9 and AV1 do not exist until they are registered.** They ship as *supplemental* decoders:
//! `VTIsHardwareDecodeSupported` answers "no" for both, and `VTDecompressionSessionCreate` fails
//! `kVTCouldNotFindVideoDecoderErr`, until `VTRegisterSupplementalVideoDecoderIfAvailable` has
//! been called for them. That is a process-global side effect with no undo and no query, which is
//! exactly the shape of thing that gets asked in the wrong order once and then answers wrongly
//! forever. So there is no way to ask this module a question without registering first:
//! [`Support`] is the only thing that answers, and [`Support::probe`] is the only thing that
//! makes one.

use std::sync::OnceLock;

// ------------------------------------------------------------------ foreign

/// `CMVideoCodecType`: a FourCC naming a compression format.
type CmVideoCodecType = u32;

/// `Boolean`, CoreFoundation's byte-wide bool.
type CfBoolean = u8;

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    fn VTRegisterSupplementalVideoDecoderIfAvailable(codec_type: CmVideoCodecType);
    fn VTIsHardwareDecodeSupported(codec_type: CmVideoCodecType) -> CfBoolean;
}

// ------------------------------------------------------------------ codecs

/// A compression format this build knows how to name to VideoToolbox.
///
/// The discriminant is the FourCC CoreMedia numbers it by, written out rather than composed from
/// characters: these are the literal `kCMVideoCodecType_*` values, and a diff against Apple's
/// header should be a diff, not an arithmetic exercise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Codec {
    /// `kCMVideoCodecType_H264`, `'avc1'`.
    H264 = 0x6176_6331,
    /// `kCMVideoCodecType_HEVC`, `'hvc1'`.
    Hevc = 0x6876_6331,
    /// `kCMVideoCodecType_VP9`, `'vp09'`. Supplemental.
    Vp9 = 0x7670_3039,
    /// `kCMVideoCodecType_AV1`, `'av01'`. Supplemental, and M3-or-later silicon.
    Av1 = 0x6176_3031,
}

impl Codec {
    /// Every codec, in the order [`Support`] stores them. The array is the length of the set, so
    /// adding a variant without widening the storage will not compile.
    pub const ALL: [Codec; 4] = [Codec::H264, Codec::Hevc, Codec::Vp9, Codec::Av1];

    /// What this codec is called in a log line. Not the FourCC: a reader wants to know whether
    /// their stream will decode, and `vp09` is not how anyone spells that question.
    pub fn name(self) -> &'static str {
        match self {
            Codec::H264 => "h264",
            Codec::Hevc => "hevc",
            Codec::Vp9 => "vp9",
            Codec::Av1 => "av1",
        }
    }

    /// Where this codec's answer sits in [`Support`].
    fn slot(self) -> usize {
        match self {
            Codec::H264 => 0,
            Codec::Hevc => 1,
            Codec::Vp9 => 2,
            Codec::Av1 => 3,
        }
    }
}

// ------------------------------------------------------------------ support

/// What this host decodes in hardware.
///
/// Answers are taken once, at probe, and carried: `VTIsHardwareDecodeSupported` is a question
/// about silicon, which does not change under a running process, and asking it per frame would
/// only widen the window in which a caller could ask it before registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Support {
    hardware: [bool; Codec::ALL.len()],
}

impl Support {
    /// Register the supplemental decoders, then ask about each codec.
    ///
    /// Registration happens once per process however many times this is called: the calls are
    /// documented as idempotent, but a `OnceLock` says so in a way that survives someone deciding
    /// to bring up a second renderer.
    pub fn probe() -> Support {
        static REGISTERED: OnceLock<()> = OnceLock::new();
        REGISTERED.get_or_init(|| {
            for codec in [Codec::Vp9, Codec::Av1] {
                // SAFETY: a plain foreign call taking a FourCC by value. It has no failure mode
                // to report -- a host with no such decoder registers nothing and the query below
                // keeps answering no.
                unsafe { VTRegisterSupplementalVideoDecoderIfAvailable(codec as CmVideoCodecType) };
            }
        });
        let mut hardware = [false; Codec::ALL.len()];
        for codec in Codec::ALL {
            // SAFETY: a plain foreign call taking a FourCC by value and returning a Boolean.
            hardware[codec.slot()] =
                unsafe { VTIsHardwareDecodeSupported(codec as CmVideoCodecType) } != 0;
        }
        Support { hardware }
    }

    /// Whether this host has silicon for `codec`.
    pub fn decodes(&self, codec: Codec) -> bool {
        self.hardware[codec.slot()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The FourCCs are the load-bearing numbers in this file: get one wrong and VideoToolbox
    /// answers about a codec nobody asked for. Spell them the other way and compare.
    #[test]
    fn each_codec_is_the_fourcc_coremedia_names_it_by() {
        for (codec, fourcc) in [
            (Codec::H264, b"avc1"),
            (Codec::Hevc, b"hvc1"),
            (Codec::Vp9, b"vp09"),
            (Codec::Av1, b"av01"),
        ] {
            assert_eq!(codec as u32, u32::from_be_bytes(*fourcc), "{codec:?}");
        }
    }

    /// Every codec has its own slot, so one's answer can never be read as another's.
    #[test]
    fn no_two_codecs_share_a_slot() {
        let mut slots: Vec<usize> = Codec::ALL.iter().map(|c| c.slot()).collect();
        slots.sort_unstable();
        assert_eq!(slots, (0..Codec::ALL.len()).collect::<Vec<_>>());
    }

    /// H.264 needs no registration to be visible and VP9 does, so the pair separates "probing
    /// works" from "probing registered first". A regression that dropped the registration would
    /// leave H.264 answering yes and VP9 answering no, which is exactly the state that pinned a
    /// caps fixture claiming this host has no VP9 decoder.
    ///
    /// Both are facts about this machine's silicon, so both are also the reason the video gate
    /// is not vacuous: a host answering no to VP9 advertises nothing and decodes nothing.
    #[test]
    fn probing_registers_the_supplemental_decoders_first() {
        let support = Support::probe();
        assert!(support.decodes(Codec::H264), "no H.264 silicon");
        assert!(
            support.decodes(Codec::Vp9),
            "no VP9 -- silicon, or a probe that skipped registration"
        );
    }
}
