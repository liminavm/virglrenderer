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

use std::ffi::{CStr, c_char, c_void};
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

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

// ------------------------------------------------------------------ CoreFoundation

/// An opaque CoreFoundation object. Every foreign handle below is one, distinguished by the
/// functions that accept it, exactly as they are in C.
#[repr(C)]
struct CfType {
    _private: [u8; 0],
}

type CfTypeRef = *const CfType;

/// `CFIndex`, and the several typedefs that are one: `CMItemCount`, `CFNumberType`.
type CfIndex = isize;

/// `kCFNumberSInt32Type`. The only number this module makes is a pixel format.
const CF_NUMBER_SINT32: CfIndex = 3;

/// `kCFStringEncodingASCII`. Every string this module makes is a four-letter atom key.
const CF_STRING_ASCII: u32 = 0x0600;

/// `OSStatus`. `noErr` is zero and everything else is a failure worth printing.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Status(i32);

impl Status {
    fn ok(self) -> Result<(), Status> {
        if self.0 == 0 { Ok(()) } else { Err(self) }
    }

    /// `kVTCouldNotFindVideoDecoderErr`. The one status worth telling apart: it means the codec
    /// this session asked for is not present, which contradicts the probe that advertised it.
    pub fn is_no_such_decoder(self) -> bool {
        self.0 == -12906
    }
}

impl core::fmt::Debug for Status {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "OSStatus {}", self.0)
    }
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFTypeDictionaryKeyCallBacks: CfType;
    static kCFTypeDictionaryValueCallBacks: CfType;
    /// The allocator that frees nothing: a block buffer built with it borrows our bytes.
    static kCFAllocatorNull: CfTypeRef;

    fn CFRelease(cf: CfTypeRef);
    fn CFRetain(cf: CfTypeRef) -> CfTypeRef;
    fn CFDataCreate(allocator: CfTypeRef, bytes: *const u8, length: CfIndex) -> CfTypeRef;
    fn CFNumberCreate(allocator: CfTypeRef, ty: CfIndex, value: *const c_void) -> CfTypeRef;
    fn CFStringCreateWithCString(
        allocator: CfTypeRef,
        cstr: *const c_char,
        encoding: u32,
    ) -> CfTypeRef;
    fn CFDictionaryCreateMutable(
        allocator: CfTypeRef,
        capacity: CfIndex,
        key_callbacks: *const CfType,
        value_callbacks: *const CfType,
    ) -> CfTypeRef;
    fn CFDictionarySetValue(dict: CfTypeRef, key: CfTypeRef, value: CfTypeRef);
}

/// A CoreFoundation object this module owns a reference to, released on drop.
///
/// Every `CFRelease` in this module is this drop and nothing else. The rule the C files have to
/// keep by hand -- release exactly what you created or retained, on every path out including the
/// error ones -- is the whole of what goes wrong in a refcounted C API, and it is what this type
/// takes away.
struct Owned(NonNull<CfType>);

impl Owned {
    /// Take ownership of a reference a Create function returned, or `None` if it returned null.
    ///
    /// # Safety
    /// `raw` must be a reference this caller owns: from a `Create`/`Copy` function, or one it
    /// has already retained. Wrapping a borrowed reference would over-release it.
    unsafe fn from_created(raw: CfTypeRef) -> Option<Owned> {
        NonNull::new(raw.cast_mut()).map(Owned)
    }

    /// Take a new reference to an object someone else owns -- a callback's argument, which is
    /// only guaranteed for the duration of the call.
    ///
    /// # Safety
    /// `raw` must be a live CoreFoundation object.
    unsafe fn retaining(raw: CfTypeRef) -> Option<Owned> {
        // SAFETY: the caller guarantees the object is live, and CFRetain on a live object is
        // always valid. The reference it returns is the one this Owned releases.
        NonNull::new(unsafe { CFRetain(raw) }.cast_mut()).map(Owned)
    }

    fn as_ref(&self) -> CfTypeRef {
        self.0.as_ptr()
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: this type is only ever constructed around a reference it owns, and nothing
        // else releases it -- there is no way to copy an `Owned`, only to move it.
        unsafe { CFRelease(self.0.as_ptr()) };
    }
}

/// A four-letter atom key, as a `CFString`. Panics on failure: the input is a literal in this
/// file, so a null return is a broken host rather than anything a guest can reach.
fn cf_string(s: &CStr) -> Owned {
    // SAFETY: `s` is a NUL-terminated C string by construction, and ASCII is a valid encoding.
    let raw = unsafe { CFStringCreateWithCString(std::ptr::null(), s.as_ptr(), CF_STRING_ASCII) };
    // SAFETY: CFStringCreateWithCString returns a reference the caller owns.
    unsafe { Owned::from_created(raw) }.expect("CoreFoundation cannot make a four-letter string")
}

// ------------------------------------------------------------------ CoreMedia, CoreVideo

/// `CMTime`. Nothing here schedules anything -- the decode is synchronous and every sample is
/// submitted untimed -- so this exists only to give the output callback the right shape.
#[repr(C)]
#[derive(Clone, Copy)]
struct CmTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

/// `VTDecompressionOutputCallbackRecord`.
#[repr(C)]
struct OutputCallbackRecord {
    callback:
        unsafe extern "C" fn(*mut c_void, *mut c_void, Status, u32, CfTypeRef, CmTime, CmTime),
    refcon: *mut c_void,
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    static kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms: CfTypeRef;

    fn CMVideoFormatDescriptionCreate(
        allocator: CfTypeRef,
        codec_type: CmVideoCodecType,
        width: i32,
        height: i32,
        extensions: CfTypeRef,
        out: *mut CfTypeRef,
    ) -> Status;
    fn CMBlockBufferCreateWithMemoryBlock(
        structure_allocator: CfTypeRef,
        memory_block: *mut c_void,
        block_length: usize,
        block_allocator: CfTypeRef,
        custom_block_source: *const c_void,
        offset_to_data: usize,
        data_length: usize,
        flags: u32,
        out: *mut CfTypeRef,
    ) -> Status;
    fn CMSampleBufferCreate(
        allocator: CfTypeRef,
        data_buffer: CfTypeRef,
        data_ready: CfBoolean,
        make_data_ready_callback: *const c_void,
        make_data_ready_refcon: *mut c_void,
        format_description: CfTypeRef,
        num_samples: CfIndex,
        num_timing_entries: CfIndex,
        timing_array: *const c_void,
        num_size_entries: CfIndex,
        size_array: *const usize,
        out: *mut CfTypeRef,
    ) -> Status;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: CfTypeRef;
    static kCVPixelBufferIOSurfacePropertiesKey: CfTypeRef;

    fn CVPixelBufferLockBaseAddress(pixbuf: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pixbuf: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferGetWidth(pixbuf: CfTypeRef) -> usize;
    fn CVPixelBufferGetHeight(pixbuf: CfTypeRef) -> usize;
    fn CVPixelBufferGetPlaneCount(pixbuf: CfTypeRef) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(pixbuf: CfTypeRef, plane: usize) -> *const u8;
    fn CVPixelBufferGetBytesPerRowOfPlane(pixbuf: CfTypeRef, plane: usize) -> usize;
    fn CVPixelBufferGetWidthOfPlane(pixbuf: CfTypeRef, plane: usize) -> usize;
    fn CVPixelBufferGetHeightOfPlane(pixbuf: CfTypeRef, plane: usize) -> usize;
}

/// `kCVPixelBufferLock_ReadOnly`. The picture is the decoder's; nothing here writes to it.
const LOCK_READ_ONLY: u64 = 1;

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    fn VTDecompressionSessionCreate(
        allocator: CfTypeRef,
        format_description: CfTypeRef,
        decoder_specification: CfTypeRef,
        destination_image_attributes: CfTypeRef,
        output_callback: *const OutputCallbackRecord,
        out: *mut CfTypeRef,
    ) -> Status;
    fn VTDecompressionSessionInvalidate(session: CfTypeRef);
    fn VTDecompressionSessionDecodeFrame(
        session: CfTypeRef,
        sample_buffer: CfTypeRef,
        decode_flags: u32,
        source_frame_refcon: *mut c_void,
        info_flags_out: *mut u32,
    ) -> Status;
}

// ------------------------------------------------------------------ pictures

/// The CoreVideo layout a session decodes into.
///
/// The guest picks this when it allocates the target and does not always pick what we advertise:
/// ffmpeg's VA-API path allocates three-plane I420 for decode targets while asking for NV12
/// elsewhere. VideoToolbox produces either, so asking for whatever the target already is costs
/// nothing and a conversion afterwards costs a pass over every pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PixelFormat {
    /// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`, `'420v'`: two planes, Y then CbCr.
    BiPlanar420 = 0x3432_3076,
    /// `kCVPixelFormatType_420YpCbCr8Planar`, `'y420'`: three planes, Y, Cb, Cr.
    Planar420 = 0x7934_3230,
}

/// A decoded picture, held by reference.
///
/// The pixels are not reachable from here: [`Picture::lock`] is the only way in, and what it
/// returns cannot outlive the lock. That is the whole point of the type -- the bytes are a
/// mapping that is only valid while locked, and a raw base address does not say so.
pub struct Picture(Owned);

impl Picture {
    /// The picture's own dimensions, which are the *coded* ones and need not be the frame's.
    ///
    /// Worth checking against what the stream declared: this host returns AV1 super-resolution
    /// frames at the coded width holding the wrong pixels, and the width is how that shows.
    pub fn width(&self) -> u32 {
        // SAFETY: a live CVPixelBuffer, which is what this type holds a reference to.
        unsafe { CVPixelBufferGetWidth(self.0.as_ref()) as u32 }
    }

    pub fn height(&self) -> u32 {
        // SAFETY: as above.
        unsafe { CVPixelBufferGetHeight(self.0.as_ref()) as u32 }
    }

    /// Map the picture's planes for reading. `None` if CoreVideo refuses the lock.
    pub fn lock(&self) -> Option<Locked<'_>> {
        // SAFETY: a live CVPixelBuffer. Read-only is not an optimisation: locking for write
        // would invalidate the decoder's own copy.
        let status = unsafe { CVPixelBufferLockBaseAddress(self.0.as_ref(), LOCK_READ_ONLY) };
        (status == 0).then_some(Locked(self))
    }
}

/// A picture whose planes are mapped, for as long as this value lives.
pub struct Locked<'a>(&'a Picture);

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        // SAFETY: this value exists only where the matching lock succeeded, and it cannot be
        // copied, so the unlock happens exactly once.
        unsafe { CVPixelBufferUnlockBaseAddress((self.0).0.as_ref(), LOCK_READ_ONLY) };
    }
}

impl Locked<'_> {
    /// How many planes the picture has: two for NV12, three for I420.
    pub fn plane_count(&self) -> usize {
        // SAFETY: a live, locked CVPixelBuffer.
        unsafe { CVPixelBufferGetPlaneCount((self.0).0.as_ref()) }
    }

    /// One plane, or `None` past the end.
    ///
    /// The bytes come back as a slice sized from CoreVideo's own answers, which is the fix for a
    /// whole class of bug the C had to patch by hand: every reader there sized its copy from the
    /// *resource*, which is the aligned allocation, while the plane holds exactly the rows the
    /// picture has. Copying the resource's height out of a shorter plane reads off the end of
    /// the mapping and faults once the pool's slack runs out -- intermittently, which is how it
    /// reached a dogfood build. A slice cannot be read past, so the caller clamps to it or does
    /// not compile.
    pub fn plane(&self, index: usize) -> Option<Plane<'_>> {
        if index >= self.plane_count() {
            return None;
        }
        let raw = (self.0).0.as_ref();
        // SAFETY: a live, locked CVPixelBuffer and an index it has that many planes for. The
        // base address of a locked plane is valid for `pitch * height` bytes, which is exactly
        // the extent CoreVideo reports and exactly the slice built from it; it stays valid for
        // the lifetime of the lock, which is this borrow.
        unsafe {
            let base = CVPixelBufferGetBaseAddressOfPlane(raw, index);
            let pitch = CVPixelBufferGetBytesPerRowOfPlane(raw, index);
            let height = CVPixelBufferGetHeightOfPlane(raw, index);
            let width = CVPixelBufferGetWidthOfPlane(raw, index);
            if base.is_null() {
                return None;
            }
            Some(Plane {
                width: width as u32,
                height: height as u32,
                pitch,
                bytes: std::slice::from_raw_parts(base, pitch * height),
            })
        }
    }
}

/// One mapped plane of a decoded picture.
///
/// `width` is in pixels and `pitch` in bytes, and neither can be derived from the other: the
/// decoder pads rows, and a plane's pixels are one byte each for luma and two for interleaved
/// chroma. `bytes` is the whole mapping, `pitch * height` long.
pub struct Plane<'a> {
    pub width: u32,
    pub height: u32,
    pub pitch: usize,
    pub bytes: &'a [u8],
}

impl Plane<'_> {
    /// Row `y`, as far as the pitch goes. `None` past the last row.
    pub fn row(&self, y: u32) -> Option<&[u8]> {
        self.bytes.get(y as usize * self.pitch..(y as usize + 1) * self.pitch)
    }
}

// ------------------------------------------------------------------ sessions

/// The codec configuration record a format description is built around.
///
/// VP9 needs a `vpcC` box; the codecs that follow bring their own shapes, which is why this is an
/// enum rather than a byte slice with a codec beside it -- the atom key and the bytes are one
/// fact, and a pair could be handed over mismatched.
#[derive(Clone, PartialEq, Eq)]
pub enum Configuration {
    /// A VP9 codec configuration record, the twelve bytes of a `vpcC` FullBox.
    Vpcc([u8; 12]),
}

impl Configuration {
    /// The VP9 record for a frame of this shape.
    ///
    /// `subsampling` is the vpcC encoding, not the stream's two flags: 1 is 4:2:0 with colocated
    /// chroma, 3 is 4:4:4. The level is declared at the maximum because a level constrains the
    /// *stream* and declaring a higher one than the stream uses refuses nothing.
    pub fn vp9(profile: u8, bit_depth: u8, subsampling: u8) -> Configuration {
        /// VP9 level 6.1, as the vpcC spells it. Declared rather than derived: see above.
        const LEVEL_MAX: u8 = 61;
        let mut box_ = [0u8; 12];
        box_[0] = 1; // version; [1..3] are flags and stay zero
        box_[4] = profile;
        box_[5] = LEVEL_MAX;
        box_[6] = (bit_depth << 4) | (subsampling << 1); // low bit: studio range
        box_[7] = 2; // colourPrimaries: unspecified
        box_[8] = 2; // transferCharacteristics: unspecified
        box_[9] = 2; // matrixCoefficients: unspecified
        // [10..12]: codecInitializationDataSize = 0
        Configuration::Vpcc(box_)
    }

    fn codec(&self) -> Codec {
        match self {
            Configuration::Vpcc(_) => Codec::Vp9,
        }
    }

    /// The extension atom key this record is filed under in the format description.
    fn atom_key(&self) -> &'static CStr {
        match self {
            Configuration::Vpcc(_) => c"vpcC",
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Configuration::Vpcc(b) => b,
        }
    }
}

/// Everything a session is a function of.
///
/// The session is rebuilt when any of this changes, so the fields travel together as one value
/// rather than as six the rebuild test has to remember to compare -- a comparison that grows a
/// field and forgets one is exactly how a live session survives a change it should not have.
#[derive(Clone, PartialEq, Eq)]
pub struct SessionKey {
    pub width: u32,
    pub height: u32,
    pub pixels: PixelFormat,
    pub config: Configuration,
}

/// A live `VTDecompressionSession` and the format description it was built from.
///
/// The session owns its key, so "does this session serve that frame" is one comparison against a
/// value the session cannot disagree with.
pub struct Session {
    session: Owned,
    format: Owned,
    key: SessionKey,
    /// Where the output callback leaves the picture it was handed.
    ///
    /// Boxed because the callback is given its address: a `Session` moves when it is returned
    /// from `create` and stored, and the heap allocation does not move with it. Locked because
    /// the callback runs on one of VideoToolbox's threads while `decode` blocks on this one --
    /// synchronous in ordering, not in threading.
    parked: Box<Mutex<Option<Picture>>>,
}

/// Where VideoToolbox leaves a decoded picture.
///
/// It cannot be delivered from here. The callback runs on a VideoToolbox thread with no EGL
/// context current, so a GL call made from it is dropped with "called without a rendering
/// context" and the guest reads an untouched texture. Parking it and delivering after
/// `DecodeFrame` returns puts the copy back on the thread that holds the context.
///
/// # Safety
/// `refcon` is the address of the `Mutex` inside a live [`Session`]; VideoToolbox hands back
/// exactly what the callback record was given. The session outlives every callback: it is
/// invalidated before the box is dropped, and invalidation waits for the decoder to finish.
unsafe extern "C" fn parked_output(
    refcon: *mut c_void,
    _frame_refcon: *mut c_void,
    status: Status,
    _info_flags: u32,
    image: CfTypeRef,
    _pts: CmTime,
    _duration: CmTime,
) {
    // SAFETY: as above -- the address of the parked slot of the session that registered this
    // callback, which is alive for the whole of the decode that invokes it.
    let parked = unsafe { &*refcon.cast::<Mutex<Option<Picture>>>() };
    if status.0 != 0 || image.is_null() {
        // VideoToolbox emits a picture even for frames the stream never shows, so a missing one
        // is a real failure rather than a hidden alt-ref. The caller reports it: it knows which
        // frame this was, and this thread has nowhere to say so.
        return;
    }
    // SAFETY: `image` is a live CVImageBuffer for the duration of this call, so retaining it is
    // valid and is what makes it outlive the call.
    let picture = unsafe { Owned::retaining(image) }.map(Picture);
    // A poisoned lock would mean a panic inside the callback, and panics abort in this build.
    *parked.lock().expect("the parked-picture lock is never held across a panic") = picture;
}

/// Why a decode produced no picture.
#[derive(Debug)]
pub enum DecodeError {
    /// VideoToolbox refused the sample.
    Rejected(Status),
    /// The sample was accepted and no picture came back. Not a guest error: the guest's bytes
    /// were accepted, and the decoder then produced nothing.
    NoPicture,
    /// A CoreMedia object could not be built around the bitstream.
    NoSample(Status),
}

impl Session {
    /// Build a session for frames of exactly this shape.
    ///
    /// Nothing is cached here: the caller holds the session and compares [`Session::serves`]
    /// before reaching for a new one, because the caller is what knows when the old one's
    /// reference pictures may be thrown away.
    pub fn create(key: SessionKey) -> Result<Session, Status> {
        let config = &key.config;
        // SAFETY: every call below is a CoreFoundation constructor over bytes and objects this
        // function owns; each result is wrapped in `Owned` at once, so every path out of here --
        // including the `?`s -- releases exactly what it created.
        let format = unsafe {
            let data = Owned::from_created(CFDataCreate(
                std::ptr::null(),
                config.bytes().as_ptr(),
                config.bytes().len() as CfIndex,
            ))
            .ok_or(Status(-1))?;
            let atoms = Owned::from_created(CFDictionaryCreateMutable(
                std::ptr::null(),
                1,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            ))
            .ok_or(Status(-1))?;
            CFDictionarySetValue(
                atoms.as_ref(),
                cf_string(config.atom_key()).as_ref(),
                data.as_ref(),
            );

            let extensions = Owned::from_created(CFDictionaryCreateMutable(
                std::ptr::null(),
                1,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            ))
            .ok_or(Status(-1))?;
            CFDictionarySetValue(
                extensions.as_ref(),
                kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
                atoms.as_ref(),
            );

            let mut format = std::ptr::null();
            CMVideoFormatDescriptionCreate(
                std::ptr::null(),
                config.codec() as CmVideoCodecType,
                key.width as i32,
                key.height as i32,
                extensions.as_ref(),
                &mut format,
            )
            .ok()?;
            Owned::from_created(format).ok_or(Status(-1))?
        };

        // The target's own layout, with an IOSurface behind it: the planes map straight onto the
        // guest's resources, and the surface is what a future zero-copy import would need.
        // SAFETY: as above.
        let attributes = unsafe {
            let attrs = Owned::from_created(CFDictionaryCreateMutable(
                std::ptr::null(),
                2,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            ))
            .ok_or(Status(-1))?;
            let pixels = key.pixels as u32 as i32;
            let number = Owned::from_created(CFNumberCreate(
                std::ptr::null(),
                CF_NUMBER_SINT32,
                (&raw const pixels).cast(),
            ))
            .ok_or(Status(-1))?;
            CFDictionarySetValue(attrs.as_ref(), kCVPixelBufferPixelFormatTypeKey, number.as_ref());
            let empty = Owned::from_created(CFDictionaryCreateMutable(
                std::ptr::null(),
                0,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            ))
            .ok_or(Status(-1))?;
            CFDictionarySetValue(
                attrs.as_ref(),
                kCVPixelBufferIOSurfacePropertiesKey,
                empty.as_ref(),
            );
            attrs
        };

        let parked: Box<Mutex<Option<Picture>>> = Box::new(Mutex::new(None));
        let callback = OutputCallbackRecord {
            callback: parked_output,
            refcon: (&raw const *parked).cast_mut().cast(),
        };
        let mut session = std::ptr::null();
        // SAFETY: a format description and an attribute dictionary this function owns, and a
        // callback record whose refcon is the boxed slot moved into the Session below -- so the
        // address stays valid for exactly as long as VideoToolbox may use it.
        unsafe {
            VTDecompressionSessionCreate(
                std::ptr::null(),
                format.as_ref(),
                std::ptr::null(),
                attributes.as_ref(),
                &raw const callback,
                &mut session,
            )
            .ok()?;
        }
        // SAFETY: VTDecompressionSessionCreate returns a reference the caller owns.
        let session = unsafe { Owned::from_created(session) }.ok_or(Status(-1))?;
        Ok(Session { session, format, key, parked })
    }

    /// Whether this session decodes frames of that shape, or a new one is needed.
    pub fn serves(&self, key: &SessionKey) -> bool {
        self.key == *key
    }

    /// Decode one access unit, and return the picture it produced.
    ///
    /// Synchronous: `kVTDecodeFrame_EnableAsynchronousDecompression` is not passed, so the
    /// output callback has run by the time this returns and exactly one picture is outstanding.
    pub fn decode(&mut self, unit: &[u8]) -> Result<Picture, DecodeError> {
        // SAFETY: `kCFAllocatorNull` is the block allocator, so the block buffer borrows `unit`
        // rather than taking it -- sound because the decode is synchronous and `unit` outlives
        // this call. Both objects are wrapped in `Owned` at once and released on every path.
        let sample = unsafe {
            let mut block = std::ptr::null();
            CMBlockBufferCreateWithMemoryBlock(
                std::ptr::null(),
                unit.as_ptr().cast_mut().cast(),
                unit.len(),
                kCFAllocatorNull,
                std::ptr::null(),
                0,
                unit.len(),
                0,
                &mut block,
            )
            .ok()
            .map_err(DecodeError::NoSample)?;
            let block = Owned::from_created(block).ok_or(DecodeError::NoSample(Status(-1)))?;

            let mut sample = std::ptr::null();
            let size = unit.len();
            CMSampleBufferCreate(
                std::ptr::null(),
                block.as_ref(),
                1,
                std::ptr::null(),
                std::ptr::null_mut(),
                self.format.as_ref(),
                1,
                0,
                std::ptr::null(),
                1,
                &raw const size,
                &mut sample,
            )
            .ok()
            .map_err(DecodeError::NoSample)?;
            Owned::from_created(sample).ok_or(DecodeError::NoSample(Status(-1)))?
        };

        let mut info = 0u32;
        // SAFETY: a live session and a sample buffer this function owns. The callback runs
        // before this returns and parks its picture in the slot the session registered.
        let status = unsafe {
            VTDecompressionSessionDecodeFrame(
                self.session.as_ref(),
                sample.as_ref(),
                0,
                std::ptr::null_mut(),
                &mut info,
            )
        };
        let parked = self.parked.lock().expect("the decode thread never panics").take();
        status.ok().map_err(DecodeError::Rejected)?;
        parked.ok_or(DecodeError::NoPicture)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: a live session this value owns. Invalidation must happen before the boxed slot
        // the callback writes into goes away, which is what this ordering buys: the box is
        // dropped after this body runs.
        unsafe { VTDecompressionSessionInvalidate(self.session.as_ref()) };
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;

    /// The vpcC bytes reach VideoToolbox as the description of the stream, so a wrong one is a
    /// decoder configured for a stream nobody sent. Spelled out here against the C's
    /// `build_vpcc`, byte for byte, for a profile-0 8-bit 4:2:0 frame.
    #[test]
    fn the_vp9_configuration_record_is_the_box_videotoolbox_expects() {
        let Configuration::Vpcc(box_) = Configuration::vp9(0, 8, 1);
        assert_eq!(
            box_,
            [
                1, 0, 0, 0,    // version 1, no flags
                0,    // profile 0
                61,   // level 6.1
                0x82, // bit_depth 8 << 4 | subsampling 1 << 1 | studio range
                2, 2, 2, // colour primaries, transfer, matrix: all unspecified
                0, 0, // no codec initialization data
            ]
        );
    }

    /// A 4:4:4 12-bit frame packs the same byte differently, which is the only field in the
    /// record that is not a constant or a straight copy.
    #[test]
    fn bit_depth_and_subsampling_share_a_byte() {
        let Configuration::Vpcc(box_) = Configuration::vp9(2, 12, 3);
        assert_eq!(box_[4], 2);
        assert_eq!(box_[6], (12 << 4) | (3 << 1));
    }

    /// The end of the risky half: registration, a format description built from the record
    /// above, and a session VideoToolbox accepts. Everything after this is bytes in and pixels
    /// out, which the replay score measures; this is the part that fails by returning an
    /// OSStatus nobody reads.
    #[test]
    fn a_vp9_session_comes_up_on_this_host() {
        assert!(Support::probe().decodes(Codec::Vp9), "no VP9 silicon; nothing below can pass");
        let key = SessionKey {
            width: 352,
            height: 288,
            pixels: PixelFormat::BiPlanar420,
            config: Configuration::vp9(0, 8, 1),
        };
        let session = Session::create(key.clone()).expect("VideoToolbox builds a VP9 session");
        assert!(session.serves(&key));
        assert!(
            !session.serves(&SessionKey { width: 640, ..key }),
            "a reshape needs a new session"
        );
    }
}
