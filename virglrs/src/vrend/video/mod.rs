// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Hardware video decode: the codecs and decode targets a context owns, and the frames they
//! decode.
//!
//! Safe throughout. The VideoToolbox calls live in [`crate::videotoolbox`]; what is here is the
//! protocol's own vocabulary -- profiles, targets, the accumulate-then-decode frame -- and the
//! copy of a decoded picture into the textures the guest samples.
//!
//! **Everything a frame needs is resolved when it is named, not when it is used.** A decode
//! target holds a share of each plane's texture, taken at CREATE_VIDEO_BUFFER; a codec holds a
//! share of the target it is mid-frame on. The C holds resource *handles* and looks them up at
//! delivery, which is a lookup the guest can empty -- and it needs a file-scope table of every
//! live codec so that destroying a buffer can go and null out the pointers to it. Neither exists
//! here: a share keeps what names it alive, and a destroy is a table drop.

pub mod av1;
pub mod bitstream;
pub mod h264;
pub mod h265;

use std::collections::BTreeMap;
use std::collections::btree_map::Entry as MapEntry;
use std::sync::Arc;

use super::formats::GlFormat;
use super::gl::Gl;
use super::gl::gles::GL_TEXTURE_2D;
use super::proto::{Format, VideoBufferHandle, VideoCodecHandle};
use super::resource::Texture;
use crate::videotoolbox::{self, Configuration, PixelFormat, Session, SessionKey};

/// `enum pipe_video_profile`, as virglrenderer numbers it.
///
/// The numbers are written out because virglrenderer carries its own copy of mesa's enum and the
/// two have drifted apart before -- a VP9 profile that landed on mesa's JPEG_BASELINE, so a host
/// advertising VP9 decode made the guest publish a JPEG decoder. Silent, and only visible as a
/// number. Here a drift is a diff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Profile {
    H264Baseline = 7,
    H264ConstrainedBaseline = 8,
    H264Main = 9,
    H264High = 11,
    HevcMain = 15,
    Vp9Profile0 = 24,
    Av1Main = 26,
}

impl Profile {
    /// The profile a wire number names, or `None` for one this build does not serve.
    ///
    /// Refusing here rather than carrying the number onward is the point: every profile that
    /// gets past this is one the backend has a path for, so nothing downstream needs a default
    /// arm that quietly decodes the wrong thing.
    pub fn from_wire(raw: u32) -> Option<Profile> {
        Some(match raw {
            7 => Profile::H264Baseline,
            8 => Profile::H264ConstrainedBaseline,
            9 => Profile::H264Main,
            11 => Profile::H264High,
            15 => Profile::HevcMain,
            24 => Profile::Vp9Profile0,
            26 => Profile::Av1Main,
            _ => return None,
        })
    }

    /// Which VideoToolbox codec decodes this profile.
    pub fn codec(self) -> videotoolbox::Codec {
        match self {
            Profile::H264Baseline
            | Profile::H264ConstrainedBaseline
            | Profile::H264Main
            | Profile::H264High => videotoolbox::Codec::H264,
            Profile::HevcMain => videotoolbox::Codec::Hevc,
            Profile::Vp9Profile0 => videotoolbox::Codec::Vp9,
            Profile::Av1Main => videotoolbox::Codec::Av1,
        }
    }

    /// The `max_level` this build advertises for the profile.
    ///
    /// Zero where the codec has no level worth declaring. For the two that do it is the level
    /// the serializer commits to, and it only sizes the guest's own allocations: VideoToolbox
    /// parses the real stream regardless.
    pub fn max_level(self) -> u32 {
        match self {
            // 5.2, above 4K.
            Profile::H264Baseline
            | Profile::H264ConstrainedBaseline
            | Profile::H264Main
            | Profile::H264High => 52,
            // 5.1, matching what the HEVC serializer declares.
            Profile::HevcMain => 153,
            Profile::Vp9Profile0 | Profile::Av1Main => 0,
        }
    }
}

/// `enum pipe_video_entrypoint`. Only decode is served; `fill_caps` advertises no other, so
/// anything else is a guest asking for something it was never offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entrypoint {
    Bitstream,
    Idct,
    Mc,
    Encode,
}

impl Entrypoint {
    pub fn from_wire(raw: u32) -> Option<Entrypoint> {
        Some(match raw {
            1 => Entrypoint::Bitstream,
            2 => Entrypoint::Idct,
            3 => Entrypoint::Mc,
            4 => Entrypoint::Encode,
            _ => return None,
        })
    }
}

/// The planar layout a decode target holds, as a `virgl_formats` number.
///
/// Only the four 8-bit 4:2:0 layouts, because they are the four a CoreVideo pixel buffer can be
/// asked for. The guest picks one when it allocates the target and does not always pick what we
/// advertise as preferred, so the session is built around whatever it chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetFormat {
    /// `VIRGL_FORMAT_NV12`: Y, then interleaved CbCr.
    Nv12,
    /// `VIRGL_FORMAT_NV21`: Y, then interleaved CrCb.
    Nv21,
    /// `VIRGL_FORMAT_IYUV`, aka I420: Y, Cb, Cr.
    Iyuv,
    /// `VIRGL_FORMAT_YV12`: Y, Cr, Cb -- I420 with the chroma planes the other way round.
    Yv12,
}

impl TargetFormat {
    pub fn from_wire(raw: u32) -> Option<TargetFormat> {
        Some(match raw {
            163 => TargetFormat::Yv12,
            165 => TargetFormat::Iyuv,
            166 => TargetFormat::Nv12,
            167 => TargetFormat::Nv21,
            _ => return None,
        })
    }

    /// The CoreVideo layout to decode into, or `None` for a target CoreVideo cannot produce.
    ///
    /// Asking VideoToolbox for the layout the target already has costs nothing, while converting
    /// afterwards costs a pass over every pixel. YV12 differs from I420 only in plane order, so
    /// it shares the planar output and swaps on delivery.
    ///
    /// NV21 has no CoreVideo layout: its chroma is swapped *within* the interleaved plane, which
    /// no output format produces and no plane reordering can repair. A guest that allocates one
    /// is refused rather than handed NV12 bytes with the colours exchanged.
    fn pixels(self) -> Option<PixelFormat> {
        Some(match self {
            TargetFormat::Nv12 => PixelFormat::BiPlanar420,
            TargetFormat::Iyuv | TargetFormat::Yv12 => PixelFormat::Planar420,
            TargetFormat::Nv21 => return None,
        })
    }

    /// Which of the picture's planes fills the target's plane `index`.
    ///
    /// The identity except for YV12, whose chroma planes are the other way round from
    /// CoreVideo's. Returning the mapping rather than swapping in place keeps the target's own
    /// plane order the thing every caller indexes by.
    fn source_plane(self, index: usize, plane_count: usize) -> usize {
        match self {
            TargetFormat::Yv12 if plane_count == 3 && index > 0 => 3 - index,
            _ => index,
        }
    }
}

/// What a VP9 picture descriptor says about the frame, as far as this backend reads it.
///
/// Six fields out of a 528-byte descriptor. VideoToolbox keeps its own reference-picture buffer
/// and parses the real bitstream, so the reference list, the segmentation probabilities and the
/// loop-filter deltas are all decoded by the hardware from the bytes the guest also sent -- the
/// descriptor is consulted only for what the *container* has to declare before the bitstream can
/// be handed over.
///
/// That is also why nothing here rewrites the descriptor. The C translates every `ref[i]` from a
/// guest buffer handle into a host buffer id before passing it on, and then the VideoToolbox
/// backend reads none of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vp9Frame {
    /// `frame_type == 0`: a key frame, which re-seeds every reference slot. An intra-only frame
    /// is not one -- it does not refresh them all.
    pub key: bool,
    pub profile: u8,
    pub bit_depth: u8,
    /// The vpcC encoding, not the stream's two flags: 1 is 4:2:0, 3 is 4:4:4.
    pub subsampling: u8,
    pub width: u32,
    pub height: u32,
}

impl Vp9Frame {
    /// Where each field sits in `struct virgl_vp9_picture_desc`, measured with `offsetof` rather
    /// than counted by hand.
    const FRAME_WIDTH: usize = 328;
    const FRAME_HEIGHT: usize = 330;
    const PIC_FIELDS: usize = 332;
    const PROFILE: usize = 354;
    const BIT_DEPTH: usize = 355;

    /// How much of a VP9 picture descriptor is read: through `bit_depth`, the last field this
    /// backend consults.
    const DESCRIPTOR_BYTES: usize = Vp9Frame::BIT_DEPTH + 1;

    /// `pic_fields` bit positions, measured the same way.
    const SUBSAMPLING_X: u32 = 1 << 0;
    const SUBSAMPLING_Y: u32 = 1 << 1;
    const FRAME_TYPE: u32 = 1 << 2;

    /// Read a descriptor the guest wrote.
    ///
    /// Total on purpose: a short descriptor reads as zeros rather than failing, which is what the
    /// C does and what the protocol allows -- the resource carries whatever the guest's driver
    /// wrote and its size is the guest's choice. A field that reads zero falls back to the
    /// codec's own creation arguments, which is the only place a missing extent can come from.
    pub fn read(blob: &[u8], codec_width: u32, codec_height: u32) -> Vp9Frame {
        let byte = |at: usize| blob.get(at).copied().unwrap_or(0);
        let short = |at: usize| u16::from_le_bytes([byte(at), byte(at + 1)]) as u32;
        let word =
            |at: usize| u32::from_le_bytes([byte(at), byte(at + 1), byte(at + 2), byte(at + 3)]);

        let fields = word(Vp9Frame::PIC_FIELDS);
        let width = short(Vp9Frame::FRAME_WIDTH);
        let height = short(Vp9Frame::FRAME_HEIGHT);
        Vp9Frame {
            key: fields & Vp9Frame::FRAME_TYPE == 0,
            profile: byte(Vp9Frame::PROFILE),
            // A descriptor that declares no depth means the only one profile 0 has.
            bit_depth: match byte(Vp9Frame::BIT_DEPTH) {
                0 => 8,
                depth => depth,
            },
            subsampling: if fields & Vp9Frame::SUBSAMPLING_X != 0
                && fields & Vp9Frame::SUBSAMPLING_Y != 0
            {
                1
            } else {
                3
            },
            width: if width == 0 { codec_width } else { width },
            height: if height == 0 { codec_height } else { height },
        }
    }

    /// The codec configuration record a session for this frame is built around.
    fn configuration(&self) -> Configuration {
        Configuration::vp9(self.profile, self.bit_depth, self.subsampling)
    }
}

/// One plane of a decode target: where the decoded plane is copied to.
///
/// Everything needed to make that copy is captured when the target is created, so delivery never
/// consults the resource table. That is not an optimisation -- the guest may free a plane's
/// resource while a frame is in flight, and a lookup at that moment finds nothing while a share
/// is always there.
pub struct Plane {
    texture: Arc<Texture>,
    /// The GL triple the texture's storage was made with. Derived from the resource's own
    /// format, never from the plane's size: an R8 luma plane and an RG8 chroma plane are the
    /// same bytes at different widths, and a guessed format uploads them silently wrong.
    gl: GlFormat,
    /// `util_format_get_blocksize` for that format: bytes per pixel, for turning the decoder's
    /// row pitch into a row length in pixels.
    block_bytes: u32,
    width: u32,
    height: u32,
}

impl Plane {
    pub fn new(
        texture: Arc<Texture>,
        gl: GlFormat,
        block_bytes: u32,
        width: u32,
        height: u32,
    ) -> Plane {
        Plane { texture, gl, block_bytes, width, height }
    }
}

/// The layout a decode target was allocated in.
///
/// A format this build cannot decode into is kept rather than refused, because creating such a
/// target is not itself a problem -- only decoding into one is, and the guest allocates targets
/// it never uses. The number is carried so the refusal, when it comes, can name it.
///
/// The two live in one value rather than as a `TargetFormat` beside the number it came from:
/// a target has one layout, and a pair could be handed on disagreeing about what it is.
pub enum Layout {
    /// A layout CoreVideo can produce.
    Served(TargetFormat),
    /// A `pipe_format` number naming no layout this build decodes into.
    Unserved(u32),
}

/// A decode target: the picture the guest handed us to decode into.
pub struct Buffer {
    pub format: Layout,
    pub width: u32,
    pub height: u32,
    planes: Vec<Plane>,
}

impl Buffer {
    /// Copy a decoded picture into this target's planes.
    ///
    /// Returns how many planes were written, which is the smaller of what the picture has and
    /// what the target has -- a target with fewer planes than the picture is the guest's own
    /// choice of layout, not an error.
    fn deliver(&self, gl: &Gl, layout: TargetFormat, picture: &videotoolbox::Locked<'_>) -> usize {
        let count = picture.plane_count();
        let mut written = 0;
        for (index, target) in self.planes.iter().enumerate().take(count) {
            let Some(source) = picture.plane(layout.source_plane(index, count)) else {
                continue;
            };

            // Clamp to what the SOURCE holds. The target is the aligned allocation while the
            // plane holds exactly the rows the picture has, so uploading the target's extent
            // reads past the mapping -- which is a fault here rather than wrong pixels, because
            // the source is a slice. The width bound is the padded row, not the picture's
            // width: the decoder's pitch is what the upload strides by.
            let row_pixels = source.pitch as u32 / target.block_bytes.max(1);
            let w = target.width.min(row_pixels);
            let h = target.height.min(source.height);
            if w == 0 || h == 0 {
                continue;
            }

            gl.bind_texture(GL_TEXTURE_2D, Some(target.texture.name));
            let ok = gl.tex_sub_image_2d_padded(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                w as i32,
                h as i32,
                target.gl.glformat,
                target.gl.gltype,
                source.bytes,
                row_pixels as i32,
            );
            // The source is a slice sized by CoreVideo and the rectangle is clamped to it just
            // above, so a refusal here is this function's own arithmetic being wrong -- a host
            // bug, and one that would otherwise show as a target holding the previous frame.
            assert!(ok, "a decoded plane clamped to its own extent does not fit it");
            written += 1;
        }
        gl.bind_texture(GL_TEXTURE_2D, None);
        written
    }
}

/// Whether a codec has reference pictures yet.
///
/// A codec starts with none, so it can decode nothing correctly before the first key frame:
/// inter frames handed to VideoToolbox against an empty reference buffer do not fail, they
/// produce quietly wrong pixels. The case that matters is a codec re-created mid-stream by a
/// snapshot restore, where the guest keeps sending inter frames and every one of them has to be
/// dropped until the stream's next key frame.
///
/// The three things the C keeps separately -- a flag, a count and a frozen target -- are one
/// value here because they only mean anything together: a drop count is noise while seeded, and
/// a freeze source outliving the wait is a picture nothing will ever copy again.
enum Gate {
    /// Reference pictures are established. Every frame decodes.
    Seeded,
    /// Waiting for a key frame. `freeze` is the picture the guest keeps seeing meanwhile.
    AwaitingKey {
        dropped: u32,
        /// The first dropped frame's target, copied into every later one.
        ///
        /// A dropped frame leaves its target holding whatever it held before, and a player
        /// presenting a pool of such targets shows pictures from before the restore, back and
        /// forth. Holding one still picture instead is the least confusing thing available.
        ///
        /// This is a share, so the guest destroying the buffer is not a dangling pointer that
        /// something has to go and clear -- which is exactly what the C's file-scope table of
        /// live codecs exists to do.
        freeze: Option<Arc<Buffer>>,
    },
}

impl Gate {
    /// Whether a frame of this key-ness decodes, updating the gate.
    fn admits(&mut self, key: bool, codec: VideoCodecHandle) -> bool {
        match self {
            Gate::Seeded => true,
            Gate::AwaitingKey { dropped, .. } if key => {
                if *dropped > 0 {
                    eprintln!(
                        "[virglrs] video codec {codec}: re-seeded by a key frame after dropping \
                         {dropped} inter frames"
                    );
                }
                *self = Gate::Seeded;
                true
            }
            Gate::AwaitingKey { dropped, .. } => {
                if *dropped == 0 {
                    eprintln!(
                        "[virglrs] video codec {codec}: no reference pictures yet; dropping \
                         inter frames until the stream's next key frame"
                    );
                }
                *dropped += 1;
                false
            }
        }
    }
}

/// What the descriptors so far said about the open frame.
///
/// One variant per codec because the codecs disagree about where each answer comes from -- VP9
/// reads its extent and key-ness out of the descriptor, H.264 reads key-ness out of the
/// bitstream and takes its extent from the codec -- and the three answers the rest of the frame
/// needs are asked for by name rather than each caller knowing which codec it has.
#[derive(Clone)]
enum Shape {
    Vp9(Vp9Frame),
    /// H.264: the parameter sets the last descriptor was written into, and whether an IDR slice
    /// has turned up in the access unit so far.
    ///
    /// The sets are kept rather than the descriptor they came from because they *are* the
    /// session's configuration -- keeping the descriptor would leave two places holding one
    /// fact, and the one the session was built from would be the derived copy.
    H264 {
        sets: h264::ParameterSets,
        key: bool,
        width: u32,
        height: u32,
    },
    /// HEVC: the same, with three sets. Key-ness comes from the descriptor here rather than
    /// from the bitstream -- `IDRPicFlag` and `RAPPicFlag` are on the wire, and H.264 has no
    /// equivalent.
    Hevc {
        sets: h265::ParameterSets,
        key: bool,
        width: u32,
        height: u32,
    },
    /// AV1: the frame's own descriptor, because the serializer writes the whole bitstream out of
    /// it, and the `av1C` box the session is configured by.
    ///
    /// Boxed: the descriptor is a kilobyte, and every open frame would otherwise carry that much
    /// whatever its codec.
    Av1 {
        desc: Box<av1::FrameDesc>,
        config: Vec<u8>,
        key: bool,
        width: u32,
        height: u32,
    },
}

impl Shape {
    /// Whether this frame re-seeds the reference pictures.
    fn key(&self) -> bool {
        match self {
            Shape::Vp9(frame) => frame.key,
            Shape::H264 { key, .. } | Shape::Hevc { key, .. } | Shape::Av1 { key, .. } => *key,
        }
    }

    /// The extent the decoded picture is expected to come back at.
    fn extent(&self) -> (u32, u32) {
        match self {
            Shape::Vp9(frame) => (frame.width, frame.height),
            Shape::H264 { width, height, .. }
            | Shape::Hevc { width, height, .. }
            | Shape::Av1 { width, height, .. } => (*width, *height),
        }
    }

    /// The codec configuration record a session for this frame is built around.
    fn configuration(&self) -> Configuration {
        match self {
            Shape::Vp9(frame) => frame.configuration(),
            Shape::H264 { sets, .. } => Configuration::h264(sets.sps.clone(), sets.pps.clone()),
            Shape::Hevc { sets, .. } => {
                Configuration::hevc(sets.vps.clone(), sets.sps.clone(), sets.pps.clone())
            }
            Shape::Av1 { config, .. } => Configuration::av1c(config.clone()),
        }
    }

    /// Re-frame the accumulated access unit into what VideoToolbox takes.
    ///
    /// VP9 is handed over as it arrives. H.264 and HEVC arrive Annex-B -- mesa's frontend
    /// prepends a start code per slice -- and VideoToolbox accepts only length-prefixed NALs,
    /// so the framing is rewritten and nothing else: the emulation-prevention bytes inside each
    /// NAL stay exactly as the encoder wrote them.
    ///
    /// One rewrite for both, because NAL framing is the one thing the two codecs did not change
    /// between them -- which is why the C reaches for its H.264 function here too.
    fn access_unit(&self, bitstream: Vec<u8>) -> Option<Vec<u8>> {
        match self {
            Shape::Vp9(_) => Some(bitstream),
            Shape::H264 { .. } | Shape::Hevc { .. } => h264::annexb_to_avcc(&bitstream),
            // AV1 never arrives here: what the guest sends is tile data, and the temporal unit
            // around it is synthesized rather than re-framed.
            Shape::Av1 { .. } => unreachable!("an AV1 unit is built by the serializer"),
        }
    }
}

/// A frame between its BEGIN_FRAME and its END_FRAME.
///
/// The target lives here rather than being looked up again at each command: BEGIN_FRAME, every
/// DECODE_BITSTREAM and END_FRAME all name it, and three lookups of one handle are three chances
/// to disagree. END_FRAME's handle is checked against this one instead.
enum Frame {
    /// No frame is open. DECODE_BITSTREAM and END_FRAME here are the guest out of sequence.
    Idle,
    Open {
        handle: VideoBufferHandle,
        target: Arc<Buffer>,
        /// The bitstream so far. The guest may split one picture across several calls, so a
        /// DECODE_BITSTREAM only accumulates; the decode itself is END_FRAME, mirroring
        /// `vaEndPicture`.
        bitstream: Vec<u8>,
        /// What the descriptors so far said about the frame. `None` until a DECODE_BITSTREAM
        /// produces one: an END_FRAME that arrives without one is the frame a snapshot cut in
        /// half, whose slices reached the codec that was saved -- or, for H.264, a frame whose
        /// slice headers never arrived to say which parameter set they reference.
        shape: Option<Shape>,
    },
}

impl Frame {
    /// The open frame's accumulator and shape, if it is open on that target.
    ///
    /// The target is checked here rather than at each caller: BEGIN_FRAME, every
    /// DECODE_BITSTREAM and END_FRAME all name it, and three checks of one handle are three
    /// chances to disagree.
    fn open_on(
        &mut self,
        target: VideoBufferHandle,
    ) -> Result<(&mut Vec<u8>, &mut Option<Shape>), Refusal> {
        let Frame::Open { handle, bitstream, shape, .. } = self else {
            return Err(Refusal::OutOfSequence("decode with no frame open"));
        };
        if *handle != target {
            return Err(Refusal::OutOfSequence("decode into a target the frame was not begun on"));
        }
        Ok((bitstream, shape))
    }
}

/// What the AV1 serializer needs kept between frames.
///
/// Only AV1 has one: the other codecs hand the guest's own bitstream over and keep nothing but a
/// session.
struct Av1 {
    obu: av1::ObuState<Owed>,
}

/// What a held AV1 frame still owes the guest, kept by the model alongside the frame itself.
///
/// The model decides what is held and when its picture goes out, so it is the model that carries
/// this: a record kept beside it here would be a second container for one fact, and every path
/// that drops a hold would become a place to remember to clear it.
struct Owed {
    /// The shape the held frame was built from. Needed whenever the unit is emitted, whether or
    /// not its picture is collected, because it names the session the bytes are decoded by.
    shape: Shape,
    /// Where the picture goes.
    ///
    /// A share, because by the time the frame goes out the guest is several frames on and may
    /// have destroyed the buffer. `None` once the picture has gone out: a shown frame held only
    /// for its reference slot decodes to a picture nothing collects.
    target: Option<Arc<Buffer>>,
}

impl av1::Carried for Owed {
    fn picture_delivered(&mut self) {
        self.target = None;
    }
}

/// One decoder the guest created.
pub struct Codec {
    pub profile: Profile,
    /// The extent the codec was created for, which is the fallback for a descriptor that
    /// declares none.
    width: u32,
    height: u32,
    gate: Gate,
    frame: Frame,
    /// The live decompression session, rebuilt when the frame's shape changes.
    ///
    /// `None` before the first frame: the session is keyed on the shape of the frame it will
    /// decode, and that arrives with the descriptor rather than with the creation arguments --
    /// a VP9 stream may change resolution or bit depth at a key frame.
    session: Option<Session>,
    /// The AV1 serializer's state. `None` for every other profile, which needs none.
    av1: Option<Av1>,
}

impl Codec {
    /// Decode one unit and put the picture it produced where it belongs.
    ///
    /// `target` is `None` for a unit decoded for its reference value alone: an AV1 frame
    /// re-emitted to claim its reference slot, whose picture went out a submission earlier into
    /// a target the guest may since have recycled.
    fn submit(
        &mut self,
        gl: &Gl,
        handle: VideoCodecHandle,
        shape: &Shape,
        unit: &[u8],
        target: Option<&Arc<Buffer>>,
    ) -> Result<(), Refusal> {
        let unserved = || Refusal::Unsupported("no CoreVideo layout for that decode target");
        let destination = match target {
            Some(buffer) => {
                let Layout::Served(layout) = buffer.format else {
                    return Err(unserved());
                };
                Some((buffer, layout, layout.pixels().ok_or_else(unserved)?))
            }
            None => None,
        };
        // A unit with no target expresses no opinion about the pixel layout, so the session
        // keeps the one it has: rebuilding it around a default would tear a live session down
        // mid-stream on any layout but NV12.
        let pixels = match destination {
            Some((_, _, pixels)) => pixels,
            None => self.session.as_ref().map_or(PixelFormat::BiPlanar420, Session::pixels),
        };

        let (width, height) = shape.extent();
        let key = SessionKey { width, height, pixels, config: shape.configuration() };
        // Rebuilt only when the frame's shape actually changes: a rebuild takes the reference
        // pictures with it, and every frame after one that did not need it then predicts from
        // an empty buffer -- which decodes "successfully" and looks like slightly wrong colour.
        if !self.session.as_ref().is_some_and(|s| s.serves(&key)) && !self.adopt(&key, handle) {
            let profile = self.profile;
            self.session = Some(Session::create(key).map_err(|status| {
                // The probe advertised this codec, so a host that now says it has no such
                // decoder is contradicting itself and every later frame will fail the same way.
                assert!(
                    !status.is_no_such_decoder(),
                    "VideoToolbox advertised {profile:?} and then had no decoder for it",
                );
                eprintln!("[virglrs] video codec {handle}: no decode session ({status:?})");
                Refusal::HostRefusedFrame
            })?);
        }
        let session = self.session.as_mut().expect("a session was just built or kept");

        let picture = match session.decode(unit) {
            Ok(picture) => picture,
            Err(why) => {
                eprintln!("[virglrs] video codec {handle}: the host decoded no picture ({why:?})");
                return Err(Refusal::HostRefusedFrame);
            }
        };
        // The picture comes back at its coded width. A host returning some other width has
        // returned something that is not this frame, and delivering it puts visibly wrong
        // content on screen with nothing anywhere reporting a problem.
        if picture.width() != width {
            eprintln!(
                "[virglrs] video codec {handle}: the host returned a {}-wide picture for a frame \
                 that declares {}; refusing it",
                picture.width(),
                width,
            );
            return Err(Refusal::HostRefusedFrame);
        }
        let Some((buffer, layout, _)) = destination else {
            return Ok(());
        };
        let Some(locked) = picture.lock() else {
            eprintln!("[virglrs] video codec {handle}: the decoded picture could not be mapped");
            return Err(Refusal::HostRefusedFrame);
        };
        buffer.deliver(gl, layout, &locked);
        Ok(())
    }

    /// DECODE_BITSTREAM for AV1.
    ///
    /// Nothing the guest sends is a bitstream: VA-API hands over a parsed frame header and the
    /// tile data, and the whole temporal unit around it is written from the descriptor. The
    /// descriptor is also what settles the *previous* frame's reference slot -- which slot the
    /// guest chose is visible only in the next frame's `ref[]` -- so a held frame goes out here.
    fn decode_av1(
        &mut self,
        gl: &Gl,
        handle: VideoCodecHandle,
        target: VideoBufferHandle,
        descriptor: &[u8],
        bitstream: &[u8],
    ) -> Result<(), Refusal> {
        let desc = match av1::FrameDesc::read(descriptor) {
            Ok(desc) => desc,
            Err(why) => {
                eprintln!("[virglrs] video codec {handle}: AV1 frame refused ({why})");
                return Err(Refusal::HostRefusedFrame);
            }
        };
        // VideoToolbox returns super-resolution frames wrongly and there is no software decoder
        // here to fall back to, so the frame is refused rather than delivered wrong.
        if desc.use_superres {
            eprintln!(
                "[virglrs] video codec {handle}: this host does not return super-resolution \
                 frames correctly"
            );
            return Err(Refusal::HostRefusedFrame);
        }
        let config = match av1::SeqParams::read(descriptor).and_then(|seq| seq.av1c()) {
            Ok(config) => config,
            Err(why) => {
                eprintln!("[virglrs] video codec {handle}: no AV1 configuration record ({why})");
                return Err(Refusal::HostRefusedFrame);
            }
        };

        // The held frame first, under its own shape: decode order is preserved, and it is this
        // descriptor's reference map that makes its refresh exact.
        let av1 = self.av1.as_mut().expect("an AV1 codec has its serializer");
        if let Some((unit, owed)) = av1.obu.flush_held(&desc) {
            self.submit(gl, handle, &owed.shape, &unit.bytes, owed.target.as_ref())?;
        }

        let (accumulated, shape) = self.frame.open_on(target)?;
        accumulated.extend_from_slice(bitstream);
        let width = if desc.frame_width == 0 { self.width } else { u32::from(desc.frame_width) };
        let height =
            if desc.frame_height == 0 { self.height } else { u32::from(desc.frame_height) };
        *shape = Some(Shape::Av1 {
            key: desc.starts_dpb(),
            desc: Box::new(desc),
            config,
            width,
            height,
        });
        Ok(())
    }

    /// END_FRAME for AV1: build the frame's temporal unit, or hold it.
    fn end_av1_frame(
        &mut self,
        gl: &Gl,
        handle: VideoCodecHandle,
        shape: &Shape,
        tiles: &[u8],
        buffer: Arc<Buffer>,
    ) -> Result<(), Refusal> {
        let Shape::Av1 { desc, .. } = shape else {
            unreachable!("an AV1 codec's frames carry an AV1 shape");
        };
        let av1 = self.av1.as_mut().expect("an AV1 codec has its serializer");
        let owed = Owed { shape: shape.clone(), target: Some(Arc::clone(&buffer)) };
        match av1.obu.build_temporal_unit(desc, tiles, owed) {
            // Nothing emitted: the serializer is holding this frame until the next descriptor
            // says which slot the guest stored it in. Its target is held with it.
            Ok(None) => Ok(()),
            Ok(Some(bytes)) => self.submit(gl, handle, shape, &bytes, Some(&buffer)),
            // A frame was built while one was still held: two temporal units would reach the
            // decoder as one sample and lose a picture, which is what the hold exists to
            // prevent. It cannot happen -- every descriptor flushes first -- but a broken model
            // must not become a lost picture.
            Err(av1::StillHolding) => {
                eprintln!("[virglrs] video codec {handle}: an AV1 frame is still held");
                Err(Refusal::HostRefusedFrame)
            }
        }
    }

    /// Try to carry the live session across a change in the frame's shape.
    ///
    /// **H.264's parameter sets are not constant across a stream, and tearing the session down
    /// when they change is not survivable.** `num_ref_idx_lX_active_minus1` reaches us as the
    /// effective *per-slice* count, so a slice that overrides the PPS default changes the PPS
    /// written for it by a byte or two mid-GOP. Keying the session on those bytes rebuilds the
    /// decompression session there and takes the reference pictures with it: every frame after
    /// the first override predicts from an empty buffer, which decodes "successfully" and puts
    /// quietly wrong pixels on screen.
    ///
    /// So the parameter sets drive the format description, and the session is asked whether it
    /// will take the new one. Falling through to a rebuild stays correct, just lossy -- and says
    /// so, because a stream that does it every frame is worth knowing about.
    fn adopt(&mut self, key: &SessionKey, handle: VideoCodecHandle) -> bool {
        let Some(live) = self.session.as_mut() else {
            return false;
        };
        if live.adopt(key) {
            return true;
        }
        if key.config.is_parameter_sets() {
            eprintln!(
                "[virglrs] video codec {handle}: the parameter sets changed in a way the live \
                 session would not take; its reference pictures are lost across the rebuild"
            );
        }
        false
    }
}

/// Why a video command could not be served.
///
/// Distinct from the renderer's `Fault` so this module owes nothing to the command decoder; the
/// caller maps it. Every variant is a guest error: a host that cannot decode a frame it agreed
/// to reports [`Refusal::HostRefusedFrame`], which the caller logs rather than faults.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// A profile, entrypoint or format this build does not serve.
    Unsupported(&'static str),
    /// An extent of zero, or a codec handle already in use for something else.
    Malformed(&'static str),
    /// A codec or buffer handle this context never created.
    ///
    /// The C reports success for these and logs, because the protocol has no way to say "your
    /// codec is gone". It is refused here because reporting success for work not done is how a
    /// snapshot restore that lost every codec came to spend 905 decodes writing nothing while
    /// the guest showed green -- and the only witness was a log line nobody was reading.
    NoSuchObject(&'static str),
    /// DECODE_BITSTREAM or END_FRAME with no frame open, or END_FRAME naming a different target
    /// than the frame was begun on.
    OutOfSequence(&'static str),
    /// The host would not decode a frame it accepted the bytes of. Not the guest's fault, and
    /// not fatal to the context: the frame is lost and the stream continues.
    HostRefusedFrame,
}

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refusal::Unsupported(what)
            | Refusal::Malformed(what)
            | Refusal::NoSuchObject(what)
            | Refusal::OutOfSequence(what) => f.write_str(what),
            Refusal::HostRefusedFrame => f.write_str("the host would not decode the frame"),
        }
    }
}

/// How much of a picture descriptor is read: the longest prefix any served codec asks for.
///
/// One length for every codec rather than one per profile, because the read happens where the
/// guest's resource is -- before the codec is looked up -- and a length that has to be paired
/// with a profile is a pair. Every `read` below is total over a shorter blob, so a codec handed
/// the union's full extent takes only its own prefix out of it.
///
/// The prefixes are short because the hardware parses the real bitstream: what the descriptor is
/// consulted for is only what the *container* has to declare before the bytes can be handed over.
pub const DESCRIPTOR_BYTES: usize = {
    let mut most = Vp9Frame::DESCRIPTOR_BYTES;
    if h264::DESCRIPTOR_BYTES > most {
        most = h264::DESCRIPTOR_BYTES;
    }
    if h265::DESCRIPTOR_BYTES > most {
        most = h265::DESCRIPTOR_BYTES;
    }
    if av1::DESCRIPTOR_BYTES > most {
        most = av1::DESCRIPTOR_BYTES;
    }
    most
};

// A leg that forgets to widen the read gets a short descriptor and reads its own fields as
// zeros -- so the build fails instead.
const _: () = {
    assert!(DESCRIPTOR_BYTES >= Vp9Frame::DESCRIPTOR_BYTES);
    assert!(DESCRIPTOR_BYTES >= h264::DESCRIPTOR_BYTES);
    assert!(DESCRIPTOR_BYTES >= h265::DESCRIPTOR_BYTES);
    assert!(DESCRIPTOR_BYTES >= av1::DESCRIPTOR_BYTES);
};

/// How many planes a guest lays a format out in when it hands the whole picture over as one
/// resource: two for the interleaved-chroma layouts, three for the fully planar ones, and one
/// for everything that is not a planar YUV layout at all.
pub fn guest_planes(format: Format) -> u32 {
    match TargetFormat::from_wire(format.wire()) {
        Some(TargetFormat::Nv12 | TargetFormat::Nv21) => 2,
        Some(TargetFormat::Iyuv | TargetFormat::Yv12) => 3,
        None => 1,
    }
}

/// Whether this build can back a composite planar decode target -- one resource holding every
/// plane -- in `format`.
///
/// Nothing, yet. Backing one needs a two-plane IOSurface and the plane views laid over it, and
/// this build has neither; the stock per-plane shape, one resource per plane, is what it serves.
///
/// The capset has to say so rather than leave it to a refusal at create, because **the sampler
/// bitmask is the guest's permission to take the shape**. By the time the host is asked, the
/// kernel has already handed the guest its handle, so a refusal never reaches it: it attaches
/// backing and builds plane views on a resource that does not exist, and its context is poisoned
/// for the rest of its life. A format advertised here that create then refuses is not a
/// degraded guest, it is a dead one.
pub fn composite_target_backable(_format: Format) -> bool {
    false
}

/// The profiles this host advertises decode for, in the order the capset lists them.
///
/// Two conditions, and both are necessary. The host must have the silicon, and this build must
/// have the leg: a profile advertised without a decode path is a guest choosing hardware decode
/// and having its context poisoned by the first frame, which is strictly worse for it than never
/// having been offered the choice. So the list grows as the legs land, and the capset and the
/// handler read it from here rather than each keeping their own idea of what is served.
pub fn advertised(support: Option<&videotoolbox::Support>) -> Vec<Profile> {
    let Some(support) = support else {
        return Vec::new();
    };
    // The order is the capset's, and it is the C's: VP9, then H.264 narrowest profile first,
    // then AV1, then HEVC.
    //
    // Only the H.264 profiles the serializer can write a parameter set for are offered. High
    // covers Baseline and Main streams too -- a decoder for High decodes both -- and offering
    // High10/422/444 would promise bit depths and chroma formats it refuses.
    [
        Profile::Vp9Profile0,
        Profile::H264ConstrainedBaseline,
        Profile::H264Baseline,
        Profile::H264Main,
        Profile::H264High,
        Profile::Av1Main,
        Profile::HevcMain,
    ]
    .into_iter()
    .filter(|profile| support.decodes(profile.codec()))
    .collect()
}

/// The codecs and decode targets one context owns.
#[derive(Default)]
pub struct Video {
    codecs: BTreeMap<VideoCodecHandle, Codec>,
    /// Shared because a codec mid-frame holds the target it is decoding into, and a frozen
    /// target outlives the guest's own reference to it.
    buffers: BTreeMap<VideoBufferHandle, Arc<Buffer>>,
}

impl Video {
    /// CREATE_VIDEO_CODEC. Re-creating a live handle is a no-op, as it is in the C.
    pub fn create_codec(
        &mut self,
        handle: VideoCodecHandle,
        profile: u32,
        entrypoint: u32,
        width: u32,
        height: u32,
        support: Option<&videotoolbox::Support>,
    ) -> Result<(), Refusal> {
        let MapEntry::Vacant(slot) = self.codecs.entry(handle) else {
            return Ok(());
        };
        let Some(profile) = Profile::from_wire(profile) else {
            eprintln!("[virglrs] video codec {handle}: profile {profile} is not served");
            return Err(Refusal::Unsupported("a video profile this build does not serve"));
        };
        match Entrypoint::from_wire(entrypoint) {
            Some(Entrypoint::Bitstream) => {}
            // Encode is not implemented and `fill_caps` advertises no entrypoint for it, so a
            // guest asking is asking for something it was never offered.
            _ => return Err(Refusal::Unsupported("video decode is the only entrypoint served")),
        }
        if width == 0 || height == 0 {
            return Err(Refusal::Malformed("a video codec with no extent"));
        }
        // The advertisement is what the guest chose this profile from, so a profile that is not
        // in it is a guest ignoring the capset rather than a host that changed its mind.
        if !advertised(support).contains(&profile) {
            eprintln!("[virglrs] video codec {handle}: {profile:?} is not advertised by this host");
            return Err(Refusal::Unsupported("that profile is not advertised"));
        }
        slot.insert(Codec {
            profile,
            width,
            height,
            // Nothing has been decoded, so there are no reference pictures.
            gate: Gate::AwaitingKey { dropped: 0, freeze: None },
            frame: Frame::Idle,
            session: None,
            av1: (profile == Profile::Av1Main).then(|| Av1 { obu: av1::ObuState::new() }),
        });
        Ok(())
    }

    /// DESTROY_VIDEO_CODEC. Destroying a handle that is not there is not an error: the guest is
    /// allowed to tear down what it never built.
    pub fn destroy_codec(&mut self, handle: VideoCodecHandle) {
        self.codecs.remove(&handle);
    }

    /// CREATE_VIDEO_BUFFER. The planes are already resolved to shares by the caller, which is
    /// the only place that can reach the resource table.
    ///
    /// A handle that is already in use is *replaced*, where the C keeps what it has. The guest
    /// does re-create a live handle over different plane resources -- the VP9 corpus does it nine
    /// times -- and keeping the old planes means the host writes pictures into resources the
    /// guest stopped believing were the target. It has not bitten yet only because the one
    /// buffer that does it is never decoded into.
    pub fn create_buffer(
        &mut self,
        handle: VideoBufferHandle,
        format: u32,
        width: u32,
        height: u32,
        planes: Vec<Plane>,
    ) -> Result<(), Refusal> {
        // Not refused: the C takes any format here and only asks for a CoreVideo layout when a
        // frame is decoded into the target, and a guest that allocates a target it never decodes
        // into must not lose its context over it.
        let layout = match TargetFormat::from_wire(format) {
            Some(served) => Layout::Served(served),
            None => {
                eprintln!(
                    "[virglrs] video buffer {handle}: format {format} names no layout this build \
                     decodes into; a frame targeting it will be refused"
                );
                Layout::Unserved(format)
            }
        };
        if width == 0 || height == 0 {
            return Err(Refusal::Malformed("a decode target with no extent"));
        }
        if planes.is_empty() {
            return Err(Refusal::Malformed("a decode target with no planes"));
        }
        self.buffers.insert(handle, Arc::new(Buffer { format: layout, width, height, planes }));
        Ok(())
    }

    /// DESTROY_VIDEO_BUFFER.
    ///
    /// Only the guest's own reference goes away. A codec mid-frame on this target, or frozen on
    /// it, holds a share and keeps it alive until it lets go -- which is the whole reason the C
    /// needs a file-scope registry of live codecs to walk here.
    pub fn destroy_buffer(&mut self, handle: VideoBufferHandle) {
        self.buffers.remove(&handle);
    }

    /// BEGIN_FRAME: open a frame on a codec, against a target.
    pub fn begin_frame(
        &mut self,
        codec: VideoCodecHandle,
        target: VideoBufferHandle,
    ) -> Result<(), Refusal> {
        let buffer = Arc::clone(
            self.buffers.get(&target).ok_or(Refusal::NoSuchObject("no such decode target"))?,
        );
        let codec = self.codec_mut(codec)?;
        codec.frame =
            Frame::Open { handle: target, target: buffer, bitstream: Vec::new(), shape: None };
        Ok(())
    }

    /// DECODE_BITSTREAM: accumulate one part of the open frame's picture.
    ///
    /// The bitstream arrives as a slice, not as a resource handle and a length: the two are
    /// reconciled by the caller, which is the only layer that knows how much of that resource
    /// the guest actually attached.
    pub fn decode_bitstream(
        &mut self,
        gl: &Gl,
        codec: VideoCodecHandle,
        target: VideoBufferHandle,
        descriptor: &[u8],
        bitstream: &[u8],
    ) -> Result<(), Refusal> {
        let handle = codec;
        let codec = self.codec_mut(codec)?;
        let (profile, width, height) = (codec.profile, codec.width, codec.height);
        // AV1 is the odd one and takes the whole call: its descriptor settles the *previous*
        // frame's reference slot, so a frame the serializer is holding goes out here rather than
        // at END_FRAME -- and a stream may display a hidden frame just one decode later, which
        // leaves no margin.
        if profile == Profile::Av1Main {
            return codec.decode_av1(gl, handle, target, descriptor, bitstream);
        }

        let (accumulated, shape) = codec.frame.open_on(target)?;
        // Accumulated first: H.264 reads the shape back out of the slice headers, so the answer
        // depends on the bytes this very call carried.
        accumulated.extend_from_slice(bitstream);

        match profile {
            Profile::Vp9Profile0 => {
                *shape = Some(Shape::Vp9(Vp9Frame::read(descriptor, width, height)))
            }
            Profile::H264Baseline
            | Profile::H264ConstrainedBaseline
            | Profile::H264Main
            | Profile::H264High => {
                let h264_profile = h264::H264Profile::of(profile).expect("an H.264 profile");
                // Nothing on the wire says which `pic_parameter_set_id` the guest's slices
                // reference, so it is read back out of them -- and until one arrives there is
                // nothing to guess with: a PPS bearing an id the slices do not use is simply not
                // found, and the frame decodes as nothing. A call that carried only a fragment
                // leaves the shape alone and waits.
                let Some(pps_id) = h264::slice_pps_id(accumulated) else {
                    return Ok(());
                };
                let desc = h264::PictureDesc::read(descriptor);
                let sets = match desc.parameter_sets(width, height, h264_profile, pps_id) {
                    Ok(sets) => sets,
                    // A stream this build's serializer has no parameter set for. The guest chose
                    // H.264 from a capset that advertises progressive 8-bit 4:2:0 only, so this
                    // is a stream outside what it was offered -- but one bad frame is not one bad
                    // context, and the next may be inside it again.
                    Err(why) => {
                        eprintln!("[virglrs] video codec {handle}: no H.264 parameter set ({why})");
                        return Err(Refusal::HostRefusedFrame);
                    }
                };
                // Sticky across the calls that make up one access unit: an IDR seen in an earlier
                // fragment is still an IDR in this frame.
                let key = shape.as_ref().is_some_and(Shape::key) || h264::has_idr(accumulated);
                *shape = Some(Shape::H264 { sets, key, width, height });
            }
            Profile::HevcMain => {
                let hevc_profile = h265::HevcProfile::of(profile).expect("an HEVC profile");
                let desc = h265::PictureDesc::read(descriptor);
                // The inspection is not only for the id: it establishes that the stream does not
                // depend on reference picture sets declared in the SPS, which are absent from the
                // wire and are therefore written empty. A stream that does depend on them is
                // refused here rather than decoded into quietly wrong pixels.
                match desc.slice_inspect(accumulated) {
                    // No slice header yet: this call carried only a fragment.
                    Ok(None) => return Ok(()),
                    Ok(Some(_id)) => {}
                    Err(why) => {
                        eprintln!("[virglrs] video codec {handle}: HEVC slice refused ({why})");
                        return Err(Refusal::HostRefusedFrame);
                    }
                }
                let sets = match desc.parameter_sets(width, height, hevc_profile) {
                    Ok(sets) => sets,
                    Err(why) => {
                        eprintln!("[virglrs] video codec {handle}: no HEVC parameter set ({why})");
                        return Err(Refusal::HostRefusedFrame);
                    }
                };
                let key = shape.as_ref().is_some_and(Shape::key) || desc.key;
                *shape = Some(Shape::Hevc { sets, key, width, height });
            }
            // Handled above, before the frame was even reached.
            Profile::Av1Main => unreachable!("AV1 takes the whole call"),
        }
        Ok(())
    }

    fn codec_mut(&mut self, handle: VideoCodecHandle) -> Result<&mut Codec, Refusal> {
        self.codecs.get_mut(&handle).ok_or(Refusal::NoSuchObject("no such video codec"))
    }
}

impl Video {
    /// END_FRAME: decode the accumulated picture and copy it into the target.
    ///
    /// This is where the frame happens. Everything before it only accumulated.
    pub fn end_frame(
        &mut self,
        gl: &Gl,
        handle: VideoCodecHandle,
        target: VideoBufferHandle,
    ) -> Result<(), Refusal> {
        let codec = self.codec_mut(handle)?;
        let Frame::Open { handle: began_on, target: buffer, bitstream, shape } =
            std::mem::replace(&mut codec.frame, Frame::Idle)
        else {
            return Err(Refusal::OutOfSequence("end of a frame that was never begun"));
        };
        if began_on != target {
            return Err(Refusal::OutOfSequence("end of a frame on a different target"));
        }

        // No bitstream is the frame a snapshot cut in half: its slices reached the codec that
        // was saved and its END_FRAME reached the one that was restored. Nothing to decode, and
        // the same stale target as a dropped frame's.
        let (Some(shape), false) = (shape, bitstream.is_empty()) else {
            codec.gate.freeze(&buffer, gl);
            return Ok(());
        };
        if !codec.gate.admits(shape.key(), handle) {
            codec.gate.freeze(&buffer, gl);
            return Ok(());
        }

        // AV1's unit is synthesized from the descriptor rather than re-framed from what the
        // guest sent, and may be held rather than submitted at all.
        if let Shape::Av1 { .. } = shape {
            return codec.end_av1_frame(gl, handle, &shape, &bitstream, buffer);
        }
        let Some(unit) = shape.access_unit(bitstream) else {
            eprintln!("[virglrs] video codec {handle}: the access unit is not Annex-B framed");
            return Err(Refusal::HostRefusedFrame);
        };
        codec.submit(gl, handle, &shape, &unit, Some(&buffer))
    }
}

impl Gate {
    /// Make a dropped frame's target show the picture the guest last saw.
    ///
    /// The first target dropped while waiting becomes the frozen picture; every later one is
    /// given a copy of it. Nothing to do once seeded.
    fn freeze(&mut self, target: &Arc<Buffer>, _gl: &Gl) {
        let Gate::AwaitingKey { freeze, .. } = self else {
            return;
        };
        match freeze {
            None => *freeze = Some(Arc::clone(target)),
            Some(source) if Arc::ptr_eq(source, target) => {}
            Some(_) => {
                // Copying one target's planes into another's needs a blit this module does not
                // have yet; until then a dropped frame keeps whatever its target held, which is
                // what the C did before the freeze was added.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The descriptor offsets are the load-bearing numbers in this file: read one wrong and the
    /// session is built for a frame nobody sent. They were measured with `offsetof` against
    /// `struct virgl_vp9_picture_desc`, so the test builds a descriptor the same way -- by
    /// planting each field at its offset -- and checks it reads back.
    #[test]
    fn the_descriptor_fields_are_where_offsetof_put_them() {
        let mut blob = vec![0u8; DESCRIPTOR_BYTES];
        blob[Vp9Frame::FRAME_WIDTH..][..2].copy_from_slice(&640u16.to_le_bytes());
        blob[Vp9Frame::FRAME_HEIGHT..][..2].copy_from_slice(&480u16.to_le_bytes());
        // subsampling_x and subsampling_y set, frame_type clear: a 4:2:0 key frame.
        blob[Vp9Frame::PIC_FIELDS..][..4].copy_from_slice(&0b011u32.to_le_bytes());
        blob[Vp9Frame::PROFILE] = 2;
        blob[Vp9Frame::BIT_DEPTH] = 10;

        let frame = Vp9Frame::read(&blob, 1, 1);
        assert_eq!(
            frame,
            Vp9Frame {
                key: true,
                profile: 2,
                bit_depth: 10,
                subsampling: 1,
                width: 640,
                height: 480
            }
        );

        // frame_type set is an inter frame, and it is the bit the keyframe gate turns on.
        blob[Vp9Frame::PIC_FIELDS] = 0b111;
        assert!(!Vp9Frame::read(&blob, 1, 1).key);
    }

    /// A guest need not fill the prefix, and a descriptor shorter than the fields we read must
    /// not panic -- it reads as zeros, and zero means "take it from the codec".
    #[test]
    fn a_short_descriptor_falls_back_to_the_codec() {
        for len in [0, 1, Vp9Frame::PIC_FIELDS, DESCRIPTOR_BYTES - 1] {
            let frame = Vp9Frame::read(&vec![0u8; len], 352, 240);
            assert_eq!(frame.width, 352, "len {len}");
            assert_eq!(frame.height, 240, "len {len}");
            assert_eq!(frame.bit_depth, 8, "len {len}: profile 0 has only one depth");
            assert!(frame.key, "len {len}: frame_type zero is a key frame");
        }
    }

    /// The two halves of the composite-target promise have to move together. A format that
    /// reports more than one plane is one the capset offers only if this build can back it, so
    /// making one backable without the path behind it is what this test is here to catch.
    #[test]
    fn no_planar_layout_is_offered_before_it_can_be_backed() {
        let planar = [163, 165, 166, 167];
        for raw in planar {
            let format = Format::from_wire(raw).expect("a planar format is on the wire");
            assert!(guest_planes(format) > 1, "format {raw} is planar");
            assert!(
                !composite_target_backable(format),
                "format {raw} is offered as a composite target, so this build must back one"
            );
        }
        // Everything else is one plane, and so is offered on its own merits.
        for raw in [1, 64, 65, 134, 314] {
            let format = Format::from_wire(raw).expect("on the wire");
            assert_eq!(guest_planes(format), 1, "format {raw}");
        }
    }

    /// A profile is advertised only where the silicon and the leg agree, and a host that was
    /// never asked for video advertises nothing at all.
    ///
    /// Written against the rule rather than against a list, because the list is a function of
    /// the silicon under the test and a machine without VP9 or H.264 is not a failing build.
    #[test]
    fn nothing_is_advertised_without_a_probe() {
        assert!(advertised(None).is_empty());
        let support = crate::videotoolbox::Support::probe();
        let list = advertised(Some(&support));

        assert!(list.iter().all(|profile| support.decodes(profile.codec())));
        for profile in
            [Profile::Vp9Profile0, Profile::H264Main, Profile::HevcMain, Profile::Av1Main]
        {
            assert_eq!(list.contains(&profile), support.decodes(profile.codec()));
        }
    }

    /// A frame's shape answers for its own codec: VP9 is handed on as it arrived, H.264 is
    /// re-framed, and each reports its own extent and key-ness.
    #[test]
    fn a_shape_answers_for_its_own_codec() {
        let vp9 = Shape::Vp9(Vp9Frame {
            key: true,
            profile: 0,
            bit_depth: 8,
            subsampling: 1,
            width: 320,
            height: 240,
        });
        assert!(vp9.key());
        assert_eq!(vp9.extent(), (320, 240));
        assert_eq!(vp9.access_unit(vec![1, 2, 3]), Some(vec![1, 2, 3]));

        let h264 = Shape::H264 {
            sets: h264::ParameterSets { sps: vec![0x67, 0x42], pps: vec![0x68, 0xce] },
            key: false,
            width: 176,
            height: 144,
        };
        assert!(!h264.key());
        assert_eq!(h264.extent(), (176, 144));
        // Annex-B in, AVCC out: one four-byte length in place of the start code.
        assert_eq!(
            h264.access_unit(vec![0, 0, 0, 1, 0x65, 0xaa]),
            Some(vec![0, 0, 0, 2, 0x65, 0xaa])
        );
        // Not Annex-B at all: nothing to guess at, and the caller refuses the frame.
        assert_eq!(h264.access_unit(vec![0x65, 0xaa]), None);

        // HEVC re-frames the same way, and reports key-ness the descriptor gave it.
        let hevc = Shape::Hevc {
            sets: h265::ParameterSets { vps: vec![0x40], sps: vec![0x42], pps: vec![0x44] },
            key: true,
            width: 1280,
            height: 720,
        };
        assert!(hevc.key());
        assert_eq!(hevc.extent(), (1280, 720));
        assert_eq!(
            hevc.access_unit(vec![0, 0, 1, 0x26, 0x01, 0xaf]),
            Some(vec![0, 0, 0, 3, 0x26, 0x01, 0xaf])
        );

        // AV1's configuration is an av1C box, and its extent comes from the descriptor the
        // serializer writes the whole unit out of.
        let blob = av1::test_descriptor(640, 360);
        let desc = av1::FrameDesc::read(&blob).expect("a Main frame");
        let av1 = Shape::Av1 {
            key: desc.starts_dpb(),
            desc: Box::new(desc),
            config: av1::SeqParams::read(&blob)
                .and_then(|seq| seq.av1c())
                .expect("a Main sequence header"),
            width: 640,
            height: 360,
        };
        assert!(!av1.key());
        assert_eq!(av1.extent(), (640, 360));
        assert!(matches!(av1.configuration(), Configuration::Av1c(_)));
    }
}
