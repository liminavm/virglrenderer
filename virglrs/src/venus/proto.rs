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
    clippy::if_same_then_else,
    clippy::len_zero,
    clippy::single_match
)]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]
pub mod serialize {
    include!(concat!(env!("OUT_DIR"), "/venus/serialize.rs"));
}

/// What this build tells a guest it speaks: the wire format version, the vk.xml it was generated
/// from, and the extension table the venus capset's bitmask is built out of.
pub mod info {
    include!(concat!(env!("OUT_DIR"), "/venus/info.rs"));
}

#[cfg(test)]
mod tests {
    use super::serialize::*;
    use super::types::*;
    use crate::venus::cs::{AllOfIt, Decoder, Encoder, IdentityObjects};
    use bumpalo::Bump;
    use std::cell::Cell;

    /// Decode a struct from wire bytes and encode it straight back. The gate P2 rests on, in
    /// miniature: the guest's own encoder wrote these bytes, so reproducing them exactly is a
    /// diff against the C implementation with nothing to compare by hand.
    fn round_trip<T: Default>(
        wire: &[u8],
        decode: fn(&mut Decoder<'_>, &mut T),
        sizeof: fn(&dyn crate::venus::cs::Protocol, &T) -> usize,
        encode: fn(&mut Encoder<'_>, &T),
    ) {
        let temp = Bump::new();
        let hard = Cell::new(false);
        let mut dec = Decoder::new(wire, &temp, &IdentityObjects, &hard);
        let mut val = T::default();
        decode(&mut dec, &mut val);
        assert!(!dec.fatal(), "decode poisoned the stream");
        assert_eq!(dec.pos(), wire.len(), "decode did not consume the command");

        let size = sizeof(&AllOfIt, &val);
        let mut buf = vec![0u8; size];
        let mut enc = Encoder::new(&mut buf, &AllOfIt);
        encode(&mut enc, &val);
        assert!(!enc.fatal(), "encode overran the buffer it sized itself");
        assert_eq!(enc.written(), wire);
    }

    fn wire(words: &[&[u8]]) -> Vec<u8> {
        words.concat()
    }

    #[test]
    fn a_struct_with_strings_reproduces_the_wire() {
        let w = wire(&[
            &0u32.to_le_bytes(),           // sType = VK_STRUCTURE_TYPE_APPLICATION_INFO
            &0u64.to_le_bytes(),           // pNext: absent
            &6u64.to_le_bytes(),           // pApplicationName: six bytes, terminator included
            b"limin\0\0\0",                // padded to eight
            &0x0001_0002u32.to_le_bytes(), // applicationVersion
            &0u64.to_le_bytes(),           // pEngineName: absent
            &0u32.to_le_bytes(),           // engineVersion
            &0x0040_3000u32.to_le_bytes(), // apiVersion
        ]);
        round_trip::<VkApplicationInfo>(
            &w,
            vn_decode_VkApplicationInfo_temp,
            vn_sizeof_VkApplicationInfo,
            vn_encode_VkApplicationInfo,
        );
    }

    #[test]
    fn a_union_reproduces_the_wire_through_the_tag_venus_pins() {
        let w = wire(&[
            &0u32.to_le_bytes(), // tag: VkClearValue.color
            &2u32.to_le_bytes(), // tag: VkClearColorValue.uint32
            &4u64.to_le_bytes(), // four elements
            &1u32.to_le_bytes(),
            &2u32.to_le_bytes(),
            &3u32.to_le_bytes(),
            &4u32.to_le_bytes(),
        ]);
        round_trip::<VkClearValue>(
            &w,
            vn_decode_VkClearValue_temp,
            vn_sizeof_VkClearValue,
            vn_encode_VkClearValue,
        );
    }

    /// An array of strings is an array of pointers, and the arena element has to be one pointer
    /// wide. Allocating it a character wide compiles and then truncates every pointer it stores.
    #[test]
    fn an_array_of_strings_reproduces_the_wire() {
        let w = wire(&[
            &1u32.to_le_bytes(), // sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
            &0u64.to_le_bytes(), // pNext: absent
            &0u32.to_le_bytes(), // flags
            &0u64.to_le_bytes(), // pApplicationInfo: absent
            &0u32.to_le_bytes(), // enabledLayerCount
            &0u64.to_le_bytes(), // ppEnabledLayerNames: absent
            &2u32.to_le_bytes(), // enabledExtensionCount
            &2u64.to_le_bytes(), // ppEnabledExtensionNames: two of them
            &4u64.to_le_bytes(),
            b"one\0",
            &4u64.to_le_bytes(),
            b"two\0",
        ]);
        round_trip::<VkInstanceCreateInfo>(
            &w,
            vn_decode_VkInstanceCreateInfo_temp,
            vn_sizeof_VkInstanceCreateInfo,
            vn_encode_VkInstanceCreateInfo,
        );
    }

    /// A renderer implements the commands it serves and inherits `unsupported` for the rest.
    /// The dispatcher has to reach the override for an implemented command and the default for
    /// everything else, or vkr would grow six hundred stubs to say nothing.
    #[test]
    fn dispatch_reaches_the_override_and_defaults_to_unsupported() {
        #[derive(Default)]
        struct Only {
            saw_create: bool,
            unsupported: Vec<VkCommandTypeEXT>,
        }
        impl Commands for Only {
            fn unsupported(&mut self, cmd: VkCommandTypeEXT) {
                self.unsupported.push(cmd);
            }
            fn vkCreateInstance(&mut self, _args: &mut vn_command_vkCreateInstance) {
                self.saw_create = true;
            }
        }

        let temp = Bump::new();
        let hard = Cell::new(false);
        let mut h = Only::default();

        // vkDestroyInstance: an instance id and an absent allocator.
        let w = wire(&[&1u64.to_le_bytes(), &0u64.to_le_bytes()]);
        let mut dec = Decoder::new(&w, &temp, &IdentityObjects, &hard);
        let hit = vn_dispatch_command(
            &mut dec,
            None,
            VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyInstance_EXT,
            &mut h,
        );
        assert_eq!(hit, Some(()));
        assert!(!dec.fatal());
        assert!(!h.saw_create);
        assert_eq!(h.unsupported, [VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyInstance_EXT]);

        // A command type no protocol defines is a stream we cannot follow.
        let mut dec = Decoder::new(&[], &temp, &IdentityObjects, &hard);
        assert_eq!(
            vn_dispatch_command(&mut dec, None, VkCommandTypeEXT(0x7fff_ffff), &mut h),
            None
        );
    }

    /// The capset hands the guest a bitmask indexed by extension number, and the guest reads it
    /// to decide what it may send. A table that disagrees with what the serializer can actually
    /// decode is a protocol mismatch that shows up as a corrupt stream, not as an error.
    #[test]
    fn the_extension_table_is_searchable_and_masks_by_number() {
        use super::info;

        assert!(info::EXTENSIONS.windows(2).all(|w| w[0].0 < w[1].0), "table must be sorted");

        let venus = info::extension("VK_MESA_venus_protocol").expect("venus is serializable");
        assert_eq!(venus.2, info::spec_version("VK_MESA_venus_protocol"));
        assert_ne!(venus.2, 0);
        assert_eq!(info::spec_version("VK_NOT_A_REAL_EXTENSION"), 0);

        let mut mask = vec![0u32; (info::MAX_EXTENSION_NUMBER / 32 + 1) as usize];
        info::extension_mask(&mut mask);
        let n = venus.1;
        assert_ne!(mask[(n / 32) as usize] & (1 << (n % 32)), 0, "venus must be in the mask");
    }

    #[test]
    fn a_blob_is_reproduced_with_its_padding() {
        let w = wire(&[
            &0u32.to_le_bytes(), // mapEntryCount
            &0u64.to_le_bytes(), // pMapEntries: absent
            &5u64.to_le_bytes(), // dataSize
            &5u64.to_le_bytes(), // pData: five bytes
            b"\x01\x02\x03\x04\x05\0\0\0",
        ]);
        round_trip::<VkSpecializationInfo>(
            &w,
            vn_decode_VkSpecializationInfo_temp,
            vn_sizeof_VkSpecializationInfo,
            vn_encode_VkSpecializationInfo,
        );
    }
}
