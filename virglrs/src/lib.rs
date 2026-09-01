// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! virglrs -- the Rust virglrenderer.
//!
//! The plan is `docs/rust-rewrite.md`; the test floor is `harness/README.md`. This crate builds a
//! `cdylib` exporting the same C ABI as `libvirglrenderer.1.dylib`, so it is swapped in by pointing
//! `VIRGL_PREFIX` at its prefix.
//!
//! What is real here: the ABI types, the resource table, the context table, fence tracking and
//! asynchronous fence retirement. What is not: both renderers. Every entry point belonging to a
//! later phase returns `-ENOTSUP` rather than a plausible success -- see `ffi::todo_phase`.

pub mod abi;
pub mod config;
pub mod fence;
pub mod ffi;
pub mod ids;
pub mod renderer;
pub mod venus;
pub mod vulkan;
