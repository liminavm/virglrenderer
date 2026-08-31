// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The generated venus protocol.
//!
//! Emitted into `OUT_DIR` by `build.rs` from `virglrs/venus-gen/`, which forks venus-protocol's
//! generator to produce Rust instead of C. Nothing here is written by hand.
//!
//! Vulkan's names are kept verbatim, `sType` and all: the generated Rust has to stay diffable
//! against the generated C when a wire question comes up, and a renaming layer would cost that
//! for nothing.

/// Vulkan's types, as venus serializes them -- generated from the vk.xml the subproject pins,
/// because that vk.xml is what defines the wire format.
// `Default` is emitted uniformly rather than derived where a derive would do: the generator
// does not get to be clever about which of 554 structs happens to qualify.
#[allow(clippy::derivable_impls)]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]
pub mod types {
    include!(concat!(env!("OUT_DIR"), "/venus/types.rs"));
}
