// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! How the renderer and its contexts were configured.

/// What the caller asked this renderer to be.
///
/// The C ABI says this in a bitmask of eleven `virgl_renderer_init` flags, of which exactly three
/// mean anything here; the rest select a winsys this build does not use. So the Rust API asks for
/// the three, by name, and the shim does the decoding -- a caller should not have to know which
/// bit is which, nor that one of them is spelled inside out.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Config {
    /// Serve venus.
    pub venus: bool,
    /// Serve vrend, the classic virgl renderer. Spelled positively; the ABI spells it `NO_VIRGL`.
    pub vrend: bool,
    /// The VMM cannot inject memory pages, so a blob must come from the guest's own heap. The
    /// guest reads this back out of the venus capset and allocates accordingly.
    pub guest_vram: bool,
}

/// The renderer a context bound when it was created.
///
/// virtio-gpu calls this the context's capset id. The ABI carries it in the low byte of a flag
/// word whose remaining bits are unused, which is a detail the shim absorbs -- here it is the
/// choice itself, and a context has exactly one for its whole life.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CapsetId {
    /// Classic virgl, the original 3D protocol. Served by vrend, which arrives in P3.
    Virgl,
    /// virgl 2. Likewise vrend's.
    Virgl2,
    /// Vulkan, over the venus wire.
    Venus,
    /// A capset id this build has no name for.
    ///
    /// Not a rejection: a guest binding a renderer we do not serve still gets its context, and
    /// learns on its first submission. Refusing at creation time would fail a real desktop, which
    /// binds virgl2 contexts long before it binds a venus one.
    Unknown(u8),
}
