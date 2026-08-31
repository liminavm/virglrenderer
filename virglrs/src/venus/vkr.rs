// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The venus renderer root.
//!
//! The C runs vkr behind the render-server proxy (`server/render_state.c`), which is a lock and a
//! forward to `vkr_renderer_*` in the same process. None of that survives here: vkr is an object
//! the `Renderer` owns, reached through it, with no file-scope state and no proxy hop. The C's
//! `vkr_renderer.h` is the interface this mirrors -- that header, not the proxy, is the design.

/// Everything venus owns. Present exactly when the renderer was initialized to serve venus, which
/// is what makes the capset honest: it is advertised because this exists, not because a flag was
/// passed.
pub struct Vkr {
    /// The flags the renderer was initialized with. The capset reports some of them straight back
    /// to the guest, which is why they are kept rather than consumed at startup.
    pub flags: i32,
}

impl Vkr {
    pub fn new(flags: i32) -> Vkr {
        Vkr { flags }
    }
}
