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
        /// What the last descriptor said about the frame. `None` until the first
        /// DECODE_BITSTREAM: an END_FRAME that arrives without one is the frame a snapshot cut
        /// in half, whose slices reached the codec that was saved.
        shape: Option<Vp9Frame>,
    },
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

/// How much of a picture descriptor is read.
///
/// The union on the wire is 5132 bytes and the VP9 arm is 528, but the fields this backend
/// consults end at `bit_depth`. Reading the prefix that holds them is the whole of it -- the
/// reference list, the segmentation probabilities and the loop-filter deltas are decoded by the
/// hardware from the bitstream the guest also sent.
pub const DESCRIPTOR_BYTES: usize = Vp9Frame::BIT_DEPTH + 1;

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
    [Profile::Vp9Profile0].into_iter().filter(|profile| support.decodes(profile.codec())).collect()
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
        codec: VideoCodecHandle,
        target: VideoBufferHandle,
        descriptor: &[u8],
        bitstream: &[u8],
    ) -> Result<(), Refusal> {
        let codec = self.codec_mut(codec)?;
        let (width, height) = (codec.width, codec.height);
        let Frame::Open { handle, bitstream: accumulated, shape, .. } = &mut codec.frame else {
            return Err(Refusal::OutOfSequence("decode with no frame open"));
        };
        if *handle != target {
            return Err(Refusal::OutOfSequence("decode into a target the frame was not begun on"));
        }
        *shape = Some(Vp9Frame::read(descriptor, width, height));
        accumulated.extend_from_slice(bitstream);
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
        if !codec.gate.admits(shape.key, handle) {
            codec.gate.freeze(&buffer, gl);
            return Ok(());
        }

        let Layout::Served(layout) = buffer.format else {
            return Err(Refusal::Unsupported("no CoreVideo layout for that decode target"));
        };
        let Some(pixels) = layout.pixels() else {
            return Err(Refusal::Unsupported("no CoreVideo layout for that decode target"));
        };
        let key = SessionKey {
            width: shape.width,
            height: shape.height,
            pixels,
            config: shape.configuration(),
        };
        // Rebuilt only when the frame's shape actually changes: a rebuild takes the reference
        // pictures with it, and every frame after one that did not need it then predicts from
        // an empty buffer -- which decodes "successfully" and looks like slightly wrong colour.
        if !codec.session.as_ref().is_some_and(|s| s.serves(&key)) {
            codec.session = Some(Session::create(key).map_err(|status| {
                // The probe advertised this codec, so a host that now says it has no such
                // decoder is contradicting itself and every later frame will fail the same way.
                assert!(
                    !status.is_no_such_decoder(),
                    "VideoToolbox advertised {:?} and then had no decoder for it",
                    codec.profile,
                );
                eprintln!("[virglrs] video codec {handle}: no decode session ({status:?})");
                Refusal::HostRefusedFrame
            })?);
        }
        let session = codec.session.as_mut().expect("a session was just built or kept");

        let picture = match session.decode(&bitstream) {
            Ok(picture) => picture,
            Err(why) => {
                eprintln!("[virglrs] video codec {handle}: the host decoded no picture ({why:?})");
                return Err(Refusal::HostRefusedFrame);
            }
        };
        // The picture comes back at its coded width. A host returning some other width has
        // returned something that is not this frame, and delivering it puts visibly wrong
        // content on screen with nothing anywhere reporting a problem.
        if picture.width() != shape.width {
            eprintln!(
                "[virglrs] video codec {handle}: the host returned a {}-wide picture for a frame \
                 that declares {}; refusing it",
                picture.width(),
                shape.width,
            );
            return Err(Refusal::HostRefusedFrame);
        }
        let Some(locked) = picture.lock() else {
            eprintln!("[virglrs] video codec {handle}: the decoded picture could not be mapped");
            return Err(Refusal::HostRefusedFrame);
        };
        buffer.deliver(gl, layout, &locked);
        Ok(())
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
    #[test]
    fn nothing_is_advertised_without_a_probe() {
        assert!(advertised(None).is_empty());
        let support = crate::videotoolbox::Support::probe();
        assert_eq!(advertised(Some(&support)), vec![Profile::Vp9Profile0]);
    }
}
