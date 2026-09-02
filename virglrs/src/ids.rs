// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Newtypes over the ABI's `uint32_t` soup.
//!
//! virgl passes resource handles, context ids, fence ids, ring indices and blob ids as bare
//! `uint32_t`/`uint64_t`, all mutually interchangeable and none of them checked. Every C bug of
//! the shape "passed a context id where a resource handle was wanted" compiles clean. Here they
//! are distinct types, so that class of bug is a build error.
//!
//! Conversion is deliberately explicit and one-directional at the boundary: the FFI shim wraps a
//! raw value on the way in and unwraps on the way out, and nothing between them can confuse two.

use std::num::NonZeroU32;

macro_rules! id {
    ($(#[$m:meta])* $name:ident($inner:ty)) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

id!(
    /// A fence within a context's ring. Monotonic per (context, ring), never globally.
    FenceId(u64)
);
id!(
    /// A fence on the legacy global path, named by the VMM rather than by a guest ring. Its own
    /// namespace: it shares nothing with [`FenceId`], which is per (context, ring).
    ClientFenceId(u32)
);
id!(
    /// A ring within a context. Ring 0 is the context's own command stream.
    ///
    /// This is a *timeline index*, which is what fences are keyed by. It is not the object id the
    /// guest gives a ring on the wire -- that is [`RingId`], and the two are different concepts
    /// that happen to share a word. Conflating them is what produced the truncation this pair of
    /// types replaces.
    RingIdx(u32)
);
id!(
    /// The object id a guest gives a ring in `vkCreateRingMESA`.
    ///
    /// A venus object id, 64 bits, from the same space as every other object the guest names.
    /// Distinct from [`RingIdx`]: this one identifies *which ring object*, that one identifies
    /// which fence timeline.
    RingId(u64)
);
id!(
    /// A global IOSurface id, which another process resolves to the surface itself.
    ///
    /// Live only as long as the surface, and recycled the instant it dies -- so this names a
    /// surface only in the hand of something that is *also* holding one. See `crate::metal`.
    SurfaceId(u32)
);
id!(
    /// Identifies the host-side object a blob resource exports. Meaningful only to the renderer
    /// that minted it.
    BlobId(u64)
);

/// A rendering context.
///
/// Guest-chosen and reused: an id freed by one destroy names something else after the next
/// create. Never zero -- `NonZeroU32` rather than a checked constructor over a `u32`, so that the
/// invariant is a property the compiler knows and `Option<CtxId>` costs no more than a `u32`.
///
/// The value originates in the guest: its kernel picks a context id when a process opens the DRM
/// node and sends it in the virtio-gpu header. So it arrives as an untrusted integer, and
/// [`CtxId::new`] is where that integer is parsed -- the same place the caller already handles a
/// header it could not make sense of.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct CtxId(NonZeroU32);

impl CtxId {
    pub fn new(raw: u32) -> Option<CtxId> {
        NonZeroU32::new(raw).map(CtxId)
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

impl std::fmt::Display for CtxId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A resource in the global resource table.
///
/// Guest-chosen and REUSED: a handle freed by one unref names something else after the next
/// create. Never zero -- `NonZeroU32` rather than a checked constructor over a `u32`, for the
/// reason [`CtxId`] gives, and because the alternative was a zero check at one call site with
/// every other lookup left to miss quietly.
///
/// Like a context id it arrives as an untrusted integer, from the VMM at the C ABI or from a
/// guest's own `vkCreateRingMESA`, and [`ResourceHandle::new`] is where that integer is parsed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct ResourceHandle(NonZeroU32);

impl ResourceHandle {
    pub const fn new(raw: u32) -> Option<ResourceHandle> {
        match NonZeroU32::new(raw) {
            Some(n) => Some(ResourceHandle(n)),
            None => None,
        }
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl std::fmt::Display for ResourceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
