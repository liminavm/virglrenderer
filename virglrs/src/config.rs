// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! How the renderer was configured.

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
