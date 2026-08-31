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
    /// A rendering context. Guest-chosen and reused the same way. Context 0 is not a context:
    /// it is the C ABI's implicit global, and here it names no entry.
    CtxId(u32)
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

impl CtxId {
    /// Whether this names a real context rather than the ABI's implicit global.
    pub fn is_real(self) -> bool {
        self.0 != 0
    }
}
