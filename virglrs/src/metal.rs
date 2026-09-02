// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! IOSurfaces: the only currency a macOS host has for handing pixels to another process.
//!
//! There is no dma-buf here. A scanout blob's storage is an IOSurface, the supervisor composites
//! from it without a copy, and a guest's `VkImage` is imported *over* it -- which is why this
//! module mints them rather than asking Vulkan for one. It has to: KosmicKrisp supports
//! `VK_EXT_external_memory_metal`, so it will import an IOSurface, but not `VK_EXT_metal_objects`,
//! so no `VkImage` can produce one. The surface is the host's to create and the guest's to render
//! into.
//!
//! One of the named unsafe modules (CLAUDE.md). The unsafe here is entirely foreign calls into
//! IOSurface and CoreFoundation, and the invariant it all rests on is refcounting: this module
//! owns exactly the references it took, and [`Surface`] is what makes owning them a type rather
//! than a discipline.
//!
//! **An id is worth nothing after its surface dies.** Ids are recycled immediately, so a stored
//! one names a stranger's surface as soon as ours is freed, and releasing a stranger's surface
//! frees storage its owner cannot re-mint. Nothing here stores an id: [`Surface::id`] asks the
//! live surface every time, and there is no way to name a surface except by holding one.
//!
//! Metal is not here yet, and slice by slice may never need to be. The venus scanout path takes
//! its row pitch from the driver's own `VkSubresourceLayout::rowPitch` rather than from
//! `minimumLinearTextureAlignmentForPixelFormat:`, so nothing on this path sends an Objective-C
//! message. When something does, it belongs in this module and nowhere else.

use std::ffi::c_void;
use std::ptr::NonNull;

use crate::ids::SurfaceId;

// ------------------------------------------------------------------ foreign

/// An opaque CoreFoundation object. Every type below is one; they are distinguished by the
/// functions that accept them, exactly as they are in C.
#[repr(C)]
struct CfType {
    _private: [u8; 0],
}

type CfTypeRef = *const CfType;

/// `CFIndex`, and `CFNumberType` which is one.
type CfIndex = isize;

/// `kCFNumberSInt32Type`. The only number this module makes: every IOSurface property it sets is
/// a 32-bit count, and giving them all one type keeps the pointer-to-value cast below honest.
const CF_NUMBER_SINT32: CfIndex = 3;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFBooleanTrue: CfTypeRef;
    static kCFTypeDictionaryKeyCallBacks: CfType;
    static kCFTypeDictionaryValueCallBacks: CfType;

    fn CFRelease(cf: CfTypeRef);
    fn CFNumberCreate(allocator: CfTypeRef, ty: CfIndex, value: *const c_void) -> CfTypeRef;
    fn CFDictionaryCreate(
        allocator: CfTypeRef,
        keys: *const CfTypeRef,
        values: *const CfTypeRef,
        count: CfIndex,
        key_callbacks: *const CfType,
        value_callbacks: *const CfType,
    ) -> CfTypeRef;
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    static kIOSurfaceWidth: CfTypeRef;
    static kIOSurfaceHeight: CfTypeRef;
    static kIOSurfaceBytesPerElement: CfTypeRef;
    static kIOSurfaceBytesPerRow: CfTypeRef;
    static kIOSurfacePixelFormat: CfTypeRef;
    static kIOSurfaceIsGlobal: CfTypeRef;

    fn IOSurfaceCreate(properties: CfTypeRef) -> CfTypeRef;
    fn IOSurfaceGetID(surface: CfTypeRef) -> u32;
    fn IOSurfaceGetBaseAddress(surface: CfTypeRef) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: CfTypeRef) -> usize;
    fn IOSurfaceGetBytesPerRow(surface: CfTypeRef) -> usize;
    fn IOSurfaceLock(surface: CfTypeRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: CfTypeRef, options: u32, seed: *mut u32) -> i32;
}

/// `kIOSurfaceLockReadOnly`. Read-only is not an optimisation here: locking for write would
/// invalidate the GPU's copy of pixels the guest is still rendering into.
const LOCK_READ_ONLY: u32 = 1;

// ------------------------------------------------------------------- pixels

/// A pixel format a surface can be minted in.
///
/// An enum rather than a newtype over IOSurface's FourCC, so that the two things this module must
/// know about a format -- its code and its pixel size -- are derived from one variant instead of
/// travelling as a pair a caller could mismatch. A format we do not handle is then a build error
/// at the call site rather than a surface laid out to the wrong stride.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    /// 32-bit BGRA, which is what every scanout on this platform is.
    Bgra,
}

impl PixelFormat {
    /// The FourCC IOSurface names it by.
    fn fourcc(self) -> u32 {
        match self {
            PixelFormat::Bgra => u32::from_be_bytes(*b"BGRA"),
        }
    }

    /// How many bytes one pixel takes.
    fn bytes_per_element(self) -> u32 {
        match self {
            PixelFormat::Bgra => 4,
        }
    }
}

// ------------------------------------------------------------------ surface

/// A live IOSurface, and the one way to name one.
///
/// Owns a single reference, released on drop. Everything about the surface is asked of the
/// surface -- see the module docs on why nothing is cached here, ids least of all.
pub struct Surface {
    surface: NonNull<CfType>,
}

// SAFETY: an `IOSurfaceRef` is a CoreFoundation object whose accessors are read-only queries of
// immutable creation-time properties, and whose refcount is atomic. This type hands out no
// interior pointers except `host_addr`, which is an address rather than a reference and which the
// caller may only use through a mapping it establishes itself.
unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    /// Mint a scanout surface whose rows are laid out exactly as `bytes_per_row` says.
    ///
    /// The pitch is the caller's because only the caller knows it: for a venus scanout it is the
    /// driver's own `VkSubresourceLayout::rowPitch` for the linear image whose memory these bytes
    /// will back. IOSurface may refuse to lay the surface out that way and pick its own -- so the
    /// surface reports what it *did*, in [`Self::bytes_per_row`], and the caller compares. This
    /// module does not decide what a mismatch means: for a zero-copy import it is fatal, and for
    /// a surface that will be copied into it is not, and neither call site is here.
    pub fn scanout(
        width: u32,
        height: u32,
        format: PixelFormat,
        bytes_per_row: u32,
    ) -> Result<Surface, SurfaceError> {
        // A surface with no pixels has no storage, and IOSurface's own refusal of one arrives as
        // a null with no reason attached. Name it here instead.
        if width == 0 || height == 0 {
            return Err(SurfaceError::ZeroExtent);
        }
        if bytes_per_row == 0 {
            return Err(SurfaceError::NoPitch);
        }

        // SAFETY: the statics are the framework's exported property keys, and every value is a
        // `CFNumber` this call makes and releases below. `CFDictionaryCreate` copies the array and
        // retains what it holds, so the numbers are ours to release the moment it returns.
        let surface = unsafe {
            let numbers = [
                Number::new(width),
                Number::new(height),
                Number::new(format.bytes_per_element()),
                Number::new(bytes_per_row),
                Number::new(format.fourcc()),
            ];
            if numbers.iter().any(|n| n.0.is_null()) {
                return Err(SurfaceError::Refused);
            }
            let keys = [
                kIOSurfaceWidth,
                kIOSurfaceHeight,
                kIOSurfaceBytesPerElement,
                kIOSurfaceBytesPerRow,
                kIOSurfacePixelFormat,
                kIOSurfaceIsGlobal,
            ];
            let values = [
                numbers[0].0,
                numbers[1].0,
                numbers[2].0,
                numbers[3].0,
                numbers[4].0,
                // Global, because a global id is the only way another process can reach this
                // surface and this crate has no supervisor transport yet. A global surface is
                // resolvable by any process on the machine, which is a real widening -- it
                // narrows again when the Mach-port handoff lands, and not before.
                kCFBooleanTrue,
            ];
            let properties = CFDictionaryCreate(
                std::ptr::null(),
                keys.as_ptr(),
                values.as_ptr(),
                keys.len() as CfIndex,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            );
            if properties.is_null() {
                return Err(SurfaceError::Refused);
            }
            let surface = IOSurfaceCreate(properties);
            CFRelease(properties);
            surface
        };

        // SAFETY: `IOSurfaceCreate` returns a +1 reference or null, and `Surface` takes that one
        // reference -- there is no second owner and no second release.
        NonNull::new(surface.cast_mut())
            .map(|surface| Surface { surface })
            .ok_or(SurfaceError::Refused)
    }

    /// The global id another process looks this surface up by.
    ///
    /// Asked of the surface, never stored. See the module docs: an id outliving its surface is
    /// the bug this whole module is shaped to prevent.
    pub fn id(&self) -> SurfaceId {
        // SAFETY: we hold a reference to the surface for the duration of this call.
        SurfaceId(unsafe { IOSurfaceGetID(self.as_ref()) })
    }

    /// How the surface actually laid its rows out, which is not necessarily what was asked for.
    pub fn bytes_per_row(&self) -> u32 {
        // SAFETY: as above. A row pitch is bounded by the surface's own allocation.
        let bytes = unsafe { IOSurfaceGetBytesPerRow(self.as_ref()) };
        u32::try_from(bytes).expect("an IOSurface row pitch does not exceed a u32")
    }

    /// How many bytes of storage it has, which is what bounds any read of it.
    pub fn alloc_size(&self) -> u64 {
        // SAFETY: as above.
        unsafe { IOSurfaceGetAllocSize(self.as_ref()) as u64 }
    }

    /// Where its pixels live in this process.
    ///
    /// An address rather than a pointer, for the reason [`crate::guest_mem::GuestMap::host_addr`]
    /// gives: this is for handing to something that will map or import it, and nothing in safe
    /// Rust may dereference it.
    pub fn host_addr(&self) -> usize {
        // SAFETY: as above. The surface is not locked, which is correct for the scanout path: the
        // GPU writes these bytes and the address is what an importer binds, not what we read.
        unsafe { IOSurfaceGetBaseAddress(self.as_ref()) as usize }
    }

    /// Copy the surface's bytes out, returning how many landed in `dst`.
    ///
    /// This is how a scanout allocation is read back at all: its storage *is* the surface, and
    /// `vkMapMemory` refuses memory the host imported rather than allocated. A short buffer is
    /// the caller's business and not an error -- the census caps what it reads -- so the count
    /// comes back rather than being inferred from `dst`.
    ///
    /// Locked for the duration. The lock is what makes the GPU's writes visible to this process;
    /// reading the base address without it returns whatever the CPU's view last held, which on a
    /// surface being actively rendered into is neither the old frame nor the new one.
    pub fn read_into(&self, dst: &mut [u8]) -> usize {
        let n = dst.len().min(self.alloc_size() as usize);
        if n == 0 {
            return 0;
        }
        // SAFETY: `IOSurfaceLock` takes the surface we hold a reference to; a null seed is
        // documented as "do not report the seed" rather than as an out parameter we must supply.
        // A failed lock leaves nothing locked, so there is nothing to unlock and nothing to read.
        if unsafe { IOSurfaceLock(self.as_ref(), LOCK_READ_ONLY, core::ptr::null_mut()) } != 0 {
            return 0;
        }
        // SAFETY: the lock is held, so the base address addresses `alloc_size` readable bytes
        // that no one else is writing through the CPU's view; `n` is bounded by that above and by
        // `dst`'s own length. The regions cannot overlap -- `dst` is the caller's memory and this
        // is the surface's.
        unsafe {
            core::ptr::copy_nonoverlapping(
                IOSurfaceGetBaseAddress(self.as_ref()).cast::<u8>(),
                dst.as_mut_ptr(),
                n,
            );
            // Balanced against the lock above, with the same options, as IOSurface requires.
            IOSurfaceUnlock(self.as_ref(), LOCK_READ_ONLY, core::ptr::null_mut());
        }
        n
    }

    fn as_ref(&self) -> CfTypeRef {
        self.surface.as_ptr()
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: the one reference `scanout` took, released once. `self` is gone after this, so
        // nothing can reach the surface through it again.
        unsafe { CFRelease(self.as_ref()) };
    }
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("id", &self.id())
            .field("bytes_per_row", &self.bytes_per_row())
            .field("alloc_size", &self.alloc_size())
            .finish()
    }
}

/// Why a surface could not be minted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceError {
    /// Zero width or height. A surface with no pixels has no storage to hand anyone.
    ZeroExtent,
    /// No row pitch. The caller is the only one who knows how the importer will read these rows,
    /// so letting IOSurface pick would produce a surface nothing can bind.
    NoPitch,
    /// IOSurface would not create it. It gives no reason; the usual one is an extent or a pitch
    /// the kernel will not back.
    Refused,
}

/// A `CFNumber` that releases itself, so the properties dictionary can be built without a leak on
/// every early return between the first number and the last.
struct Number(CfTypeRef);

impl Number {
    /// # Safety
    /// The caller must keep this alive until the dictionary that retains it has been created.
    unsafe fn new(value: u32) -> Number {
        let value = value as i32;
        // SAFETY: `CF_NUMBER_SINT32` matches the `i32` whose address is passed, and CoreFoundation
        // copies the value rather than borrowing it.
        Number(unsafe {
            CFNumberCreate(std::ptr::null(), CF_NUMBER_SINT32, (&raw const value).cast())
        })
    }
}

impl Drop for Number {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the +1 from `CFNumberCreate`, released once.
            unsafe { CFRelease(self.0) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Resolving an id to a surface is the *importer's* half of this path and arrives with it.
    // It is declared here because the test below is about what another process can reach, and
    // asking the system is the only way to make that claim rather than assume it.
    #[link(name = "IOSurface", kind = "framework")]
    unsafe extern "C" {
        fn IOSurfaceLookup(id: u32) -> CfTypeRef;
    }

    /// Held across every mint in this file. Ids recycle immediately -- the fact the module is
    /// shaped around -- so a surface minted on another test thread can be handed the id the
    /// lifetime test below has just dropped, and that test would then see its own dead id
    /// resolve. The hazard is real and the serialization is the fix; a retry would only be
    /// hiding it.
    static MINT: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Look a surface up the way another process would, and release what the lookup returns.
    ///
    /// Null is the answer for an id naming nothing -- which is the whole point of the test below.
    fn looks_up(id: SurfaceId) -> bool {
        // SAFETY: `IOSurfaceLookup` takes a bare id and returns either null or a +1 reference,
        // which is released immediately. Nothing outlives this function.
        unsafe {
            let found = IOSurfaceLookup(id.0);
            if found.is_null() {
                return false;
            }
            CFRelease(found);
            true
        }
    }

    /// The first rule this module exists to enforce: an id is worth nothing after its surface
    /// dies. Ids are recycled immediately, so anything that cached one would go on resolving --
    /// to a stranger's surface, whose storage it could then release out from under its owner.
    ///
    /// Written as a lookup rather than an assertion about our own state, because the claim is
    /// about what the *rest of the machine* can still reach.
    #[test]
    fn a_surface_is_reachable_by_id_only_while_it_is_alive() {
        let _mint = MINT.lock().expect("the mint lock is never poisoned");
        let surface = Surface::scanout(64, 32, PixelFormat::Bgra, 64 * 4).expect("minted");
        let id = surface.id();
        assert!(id.0 != 0, "a global surface has an id");
        assert!(looks_up(id), "another process can reach it while we hold it");

        drop(surface);
        assert!(!looks_up(id), "and cannot the moment we let go");
    }

    /// The pitch is the caller's, because the importer's layout is the caller's. A surface that
    /// quietly chose its own would put every row of the guest's image at the wrong offset -- the
    /// shear this parameter exists to prevent -- so what it chose is reported, not assumed.
    #[test]
    fn a_surface_lays_its_rows_out_where_it_was_asked_to() {
        // Deliberately padded well past `width * 4`: a driver's `rowPitch` is aligned up, and a
        // surface that ignored the request would land on the tight pitch instead.
        const WIDTH: u32 = 1968;
        const HEIGHT: u32 = 64;
        const PITCH: u32 = 1968 * 4 + 256;

        let _mint = MINT.lock().expect("the mint lock is never poisoned");
        let surface = Surface::scanout(WIDTH, HEIGHT, PixelFormat::Bgra, PITCH).expect("minted");
        assert_eq!(surface.bytes_per_row(), PITCH, "the rows are where the importer will look");
        assert!(
            surface.alloc_size() >= u64::from(PITCH) * u64::from(HEIGHT),
            "and there is storage behind all of them"
        );
        assert!(surface.host_addr() != 0, "which is addressable");
    }

    /// A scanout's bytes are read back through the surface because they are not anywhere else:
    /// once the allocation is a host-pointer import of these pages, `vkMapMemory` has nothing to
    /// hand back. Written through the base address here because in the real path the writer is a
    /// GPU this test does not have.
    #[test]
    fn a_surface_hands_back_the_bytes_it_holds() {
        const PITCH: u32 = 256;
        const HEIGHT: u32 = 4;

        let _mint = MINT.lock().expect("the mint lock is never poisoned");
        let surface = Surface::scanout(64, HEIGHT, PixelFormat::Bgra, PITCH).expect("minted");

        // SAFETY: the surface is alive and this is its own storage, sized by `alloc_size`; the
        // slice is dropped before anything else touches the surface.
        let pixels = unsafe {
            std::slice::from_raw_parts_mut(
                surface.host_addr() as *mut u8,
                surface.alloc_size() as usize,
            )
        };
        pixels.fill(0);
        pixels[0] = 0xf0;
        pixels[(PITCH * (HEIGHT - 1)) as usize] = 0x0f;

        let mut out = vec![0u8; surface.alloc_size() as usize];
        assert_eq!(surface.read_into(&mut out), out.len(), "all of it");
        assert_eq!(out[0], 0xf0, "the first row");
        assert_eq!(out[(PITCH * (HEIGHT - 1)) as usize], 0x0f, "and the last");

        // A short buffer takes a prefix rather than failing: the census caps every read.
        let mut short = [0u8; 8];
        assert_eq!(surface.read_into(&mut short), 8);
        assert_eq!(short[0], 0xf0);

        assert_eq!(surface.read_into(&mut []), 0, "and asking for nothing reads nothing");
    }

    /// Every refusal is a caller's mistake, and each says which -- a null with no reason is what
    /// this boundary exists to translate.
    #[test]
    fn a_surface_that_cannot_exist_is_refused_by_name() {
        let bgra = PixelFormat::Bgra;
        let refused = |w, h, pitch| Surface::scanout(w, h, bgra, pitch).expect_err("refused");
        assert_eq!(refused(0, 32, 256), SurfaceError::ZeroExtent);
        assert_eq!(refused(64, 0, 256), SurfaceError::ZeroExtent);
        assert_eq!(refused(64, 32, 0), SurfaceError::NoPitch);
    }
}
