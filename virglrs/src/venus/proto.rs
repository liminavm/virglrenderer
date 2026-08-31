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

/// The wire itself: `vn_sizeof_*`, `vn_encode_*`, `vn_decode_*` for every type venus serializes.
///
/// Generated code is uniform, not minimal: every `vn_sizeof_*` takes the protocol whether its type
/// consults it, every pointer read is wrapped in `unsafe` whether that particular one needed it,
/// and a walk that always returns from its match still carries the loop around it. Special-casing
/// each of those in the emitter would buy tidier output at the cost of a generator no one can
/// follow, so the cosmetic lints are allowed here and nowhere else.
#[allow(unreachable_code, unused_assignments, unused_mut, unused_unsafe, unused_variables)]
#[allow(
    clippy::needless_return,
    clippy::needless_borrow,
    clippy::unnecessary_cast,
    clippy::comparison_to_empty,
    clippy::len_zero,
    clippy::single_match
)]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]
pub mod serialize {
    include!(concat!(env!("OUT_DIR"), "/venus/serialize.rs"));
}
