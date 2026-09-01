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
    /// A resource in the global resource table. Guest-chosen, and REUSED: a handle freed by one
    /// unref names something else after the next create.
    ResourceHandle(u32)
);
id!(
    /// A fence within a context's ring. Monotonic per (context, ring), never globally.
    FenceId(u64)
);
id!(
    /// A ring within a context. Ring 0 is the context's own command stream.
    RingIdx(u32)
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
