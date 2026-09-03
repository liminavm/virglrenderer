// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! GLES, as the driver exports it.
//!
//! One of the named unsafe modules (CLAUDE.md). The table is generated from the Khronos registry
//! by `gl-gen`, for the reason `vulkan.rs` gives: a binding transcribed by hand can disagree with
//! the driver about a parameter, and the disagreement is a stack smash rather than a compile
//! error. Every entry point is resolved through `eglGetProcAddress` ([`super::egl`]), which is
//! what makes the transmute in `Gles::load` sound: EGL promises the address of the function of
//! that name, and the registry gives that name its signature.
//!
//! Nothing here calls GL. This module is the vocabulary; the callers -- resources, blits,
//! shaders -- are safe Rust that reaches the driver through the table's accessors, each of which
//! hands back an `unsafe extern "C" fn` whose call site carries the `SAFETY:` for its arguments.

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod types {
    include!(concat!(env!("OUT_DIR"), "/gl/types.rs"));
}

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod gles {
    include!(concat!(env!("OUT_DIR"), "/gl/gles.rs"));
}

pub use gles::Gles;
pub use types::*;
