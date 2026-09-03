// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! vrend -- the classic virgl renderer: gallium state and TGSI shaders over the wire, GL on the
//! host.
//!
//! `proto` is the protocol as types, `decode` the boundary that parses the guest's dwords into
//! them and refuses the rest, `encode` the way back -- which exists so that a recorded stream
//! decoded and re-encoded is a differential test against the guest's own encoder, with no C dump
//! to diff against. `pipe` is gallium's vocabulary, the enums those types are made of.
//!
//! `egl` and `gl` are the host side: the winsys and the driver's entry points, the two named
//! unsafe modules of this renderer (CLAUDE.md). Everything above them is safe Rust.

pub mod decode;
pub mod egl;
pub mod encode;
pub mod gl;
pub mod pipe;
pub mod proto;
