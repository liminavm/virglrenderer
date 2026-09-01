// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The venus capset: what the guest reads before it sends a single command.
//!
//! The guest's mesa driver fetches this over virtio-gpu and decides from it which structs it may
//! put on the wire. Getting it wrong does not produce an error -- it produces a stream this
//! renderer parses into different structs than the guest encoded, which is the worst failure mode
//! the protocol has. Everything version-shaped in it therefore comes from the generated
//! [`info`](crate::venus::proto::info) table rather than from a constant written here.

use bytemuck::{Pod, Zeroable};

use crate::config::Config;
use crate::venus::proto::info;

/// The number of `u32`s in the capset's extension mask. Vulkan numbers extensions from 1, and the
/// mask covers the first 1023 of them.
const MASK_WORDS: usize = 32;

/// `struct virgl_renderer_capset_venus`. The layout is the guest's, not ours: it is read straight
/// out of a buffer the VMM sized from `get_cap_set`, so every field is where the C put it.
///
/// `repr(C)` here is a wire format and not a concession to the shim. The bytes go to the *guest*,
/// whose mesa driver reads them back by offset, so they are owed to whoever hands them over --
/// rutabaga through the Rust API just as much as `fill_caps` through the C one.
///
/// `Pod` is what turns those bytes into a safe operation, and it is a build-time claim rather than
/// a comment: the derive refuses to compile a struct with any padding in it, which is the whole of
/// what a cast to `&[u8]` needs to be sound. Every field is a `u32` or an array of them, so there
/// is none -- and the day someone adds a `u16`, the build says so instead of the guest misparsing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Capset {
    pub wire_format_version: u32,
    pub vk_xml_version: u32,
    pub vk_ext_command_serialization_spec_version: u32,
    pub vk_mesa_venus_protocol_spec_version: u32,
    /// Set on the render-server configuration, which is the only one mesa still supports for
    /// blob id 0.
    pub supports_blob_id_0: u32,
    /// Extension number N lives in bit `N % 32` of word `N / 32`. Bit 0 of word 0 is not an
    /// extension: it says the mask is meaningful at all, and a guest that finds it clear assumes
    /// the renderer supports everything.
    pub vk_extension_mask1: [u32; MASK_WORDS],
    /// Blocking waits may pass through from the guest. A single-threaded renderer cannot afford
    /// them -- this one is not single-threaded, and the guest driver carries the consequences.
    pub allow_vk_wait_syncs: u32,
    /// Each `VkQueue` may bind an unused `ring_idx` and get its own fence timeline. Ring 0 is
    /// reserved for CPU fences the renderer signals on consumption.
    pub supports_multiple_timelines: u32,
    /// The VMM cannot inject memory pages, so blobs must come from the guest's own heap.
    pub use_guest_vram: u32,
}

impl Capset {
    /// Build the capset this renderer advertises, for the configuration it was asked for.
    pub fn new(config: Config) -> Capset {
        let mut c = Capset {
            wire_format_version: info::WIRE_FORMAT_VERSION,
            vk_xml_version: info::VK_XML_VERSION,
            vk_ext_command_serialization_spec_version: info::spec_version(
                "VK_EXT_command_serialization",
            ),
            vk_mesa_venus_protocol_spec_version: info::spec_version("VK_MESA_venus_protocol"),
            supports_blob_id_0: 1,
            vk_extension_mask1: [0; MASK_WORDS],
            allow_vk_wait_syncs: 1,
            supports_multiple_timelines: 1,
            use_guest_vram: u32::from(config.guest_vram),
        };
        info::extension_mask(&mut c.vk_extension_mask1);
        // Bit 0 is the "mask is meaningful" flag, and no extension may claim it -- extension
        // numbers start at 1, so a set bit here would mean the table is corrupt.
        assert_eq!(c.vk_extension_mask1[0] & 1, 0, "extension 0 is not an extension");
        c.vk_extension_mask1[0] |= 1;
        c
    }

    /// The bytes the guest reads this out of, for whoever is handing them over.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// What `virgl_renderer_get_cap_set` reports for venus: version 0, and the struct's size.
///
/// The version is 0 because venus never versioned its capset -- it grew fields and let the size
/// say what is present.
pub const VERSION: u32 = 0;

pub fn size() -> u32 {
    core::mem::size_of::<Capset>() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guest reads this struct by offset out of a buffer the VMM sized from `get_cap_set`, so
    /// a layout that drifts from the C's is a silent misparse on the guest side.
    #[test]
    fn the_capset_has_the_layout_the_guest_reads() {
        // Ground truth: `cc` on `src/venus_hw.h`.
        assert_eq!(core::mem::size_of::<Capset>(), 160);
        assert_eq!(core::mem::align_of::<Capset>(), 4);
        let c = Capset::default();
        let base = (&c as *const Capset).addr();
        let at = |p: *const u32| p.addr() - base;
        assert_eq!(at(&c.wire_format_version), 0);
        assert_eq!(at(&c.vk_xml_version), 4);
        assert_eq!(at(&c.vk_ext_command_serialization_spec_version), 8);
        assert_eq!(at(&c.vk_mesa_venus_protocol_spec_version), 12);
        assert_eq!(at(&c.supports_blob_id_0), 16);
        assert_eq!(at(c.vk_extension_mask1.as_ptr()), 20);
        assert_eq!(at(&c.allow_vk_wait_syncs), 148);
        assert_eq!(at(&c.supports_multiple_timelines), 152);
        assert_eq!(at(&c.use_guest_vram), 156);
        assert_eq!(size() as usize, core::mem::size_of::<Capset>());

        let c = Capset::new(Config::default());
        assert_eq!(c.as_bytes().len(), core::mem::size_of::<Capset>());
        // The first word of the image is the first field, which is what fixes the field order.
        assert_eq!(&c.as_bytes()[..4], &info::WIRE_FORMAT_VERSION.to_ne_bytes());
    }

    #[test]
    fn the_mask_is_marked_meaningful_and_carries_venus() {
        let c = Capset::new(Config::default());
        assert_eq!(c.vk_extension_mask1[0] & 1, 1, "guest ignores a mask without bit 0");

        let (_, number, version) = info::extension("VK_MESA_venus_protocol").unwrap();
        assert_ne!(c.vk_extension_mask1[(number / 32) as usize] & (1 << (number % 32)), 0);
        assert_eq!(c.vk_mesa_venus_protocol_spec_version, *version);
    }

    /// The guest allocates from its own heap or not on the strength of this one word, so it has
    /// to follow the configuration rather than a default. Which bit sets the configuration is the
    /// shim's business, and is checked there.
    #[test]
    fn guest_vram_is_reported_exactly_as_it_was_configured() {
        assert_eq!(Capset::new(Config::default()).use_guest_vram, 0);
        let on = Config { guest_vram: true, ..Config::default() };
        assert_eq!(Capset::new(on).use_guest_vram, 1);
    }
}
