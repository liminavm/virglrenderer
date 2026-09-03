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

/// Deterministic contents for a reply's output members, and the entry point the reply oracle
/// drives. Test scaffolding, behind the feature that builds the C encoder it exists to diff
/// against -- a renderer never plants values in its own replies.
#[cfg(feature = "reply-oracle")]
#[allow(non_camel_case_types, non_snake_case, dead_code)]
pub mod fill {
    include!(concat!(env!("OUT_DIR"), "/venus/fill.rs"));
}

/// What the compiler made of the generated structs, for the C oracle to be held to. Behind the
/// same feature as the reply differential, which is the code that depends on the two agreeing.
#[cfg(feature = "reply-oracle")]
#[allow(non_camel_case_types, non_snake_case, dead_code)]
pub mod layout {
    include!(concat!(env!("OUT_DIR"), "/venus/layout.rs"));
}

/// Every generated struct laid out the same way a C compiler lays out venus-protocol's own.
///
/// The reply oracle casts a pointer to one of our `vn_command_*` straight into venus-protocol's
/// encoder, and the driver is handed `Vk*` structs we filled. Both are memory contracts, and the
/// generator can satisfy them in source -- same members, same order -- while a compiler quietly
/// disagrees about padding, alignment, or how wide a pointer member is. That last one is the
/// live risk: a member that becomes a slice reference grows a second word, and every member
/// after it moves.
#[cfg(all(test, feature = "reply-oracle"))]
mod layout_parity {
    use super::layout;

    unsafe extern "C" {
        static vn_layout_types: [CType; 0];
        static vn_layout_members: [CMember; 0];
        static vn_layout_type_count: usize;
        static vn_layout_member_count: usize;
    }

    #[repr(C)]
    struct CType {
        size: u32,
        align: u32,
    }

    #[repr(C)]
    struct CMember {
        offset: u32,
        size: u32,
    }

    /// The tables are index-matched, so a length difference means the two halves came from
    /// different generator runs and no row-by-row comparison below would mean anything.
    #[test]
    fn the_two_tables_describe_the_same_types_in_the_same_order() {
        // SAFETY: both counts are `const size_t` the same generator run emitted beside the
        // tables they measure.
        let (types, members) = unsafe { (vn_layout_type_count, vn_layout_member_count) };
        assert_eq!(types, layout::TYPES.len());
        assert_eq!(members, layout::MEMBERS.len());
    }

    #[test]
    fn every_generated_struct_has_the_layout_a_c_compiler_gives_it() {
        // SAFETY: the arrays are declared `[T; 0]` because C sizes them and Rust cannot; every
        // index below is bounded by the count the same generator run emitted for them, which
        // `the_two_tables_describe_the_same_types_in_the_same_order` pins to the Rust length.
        let (c_types, c_members) = unsafe {
            (
                core::slice::from_raw_parts(vn_layout_types.as_ptr(), vn_layout_type_count),
                core::slice::from_raw_parts(vn_layout_members.as_ptr(), vn_layout_member_count),
            )
        };

        let mut bad = Vec::new();
        for (rs, c) in layout::TYPES.iter().zip(c_types) {
            // A shadowed struct is allowed to be the larger of the two, and nothing else is:
            // every member C knows about still has to be where C puts it, which the member rows
            // below are what actually check.
            let size_ok =
                if rs.shadowed { rs.size >= c.size as usize } else { rs.size == c.size as usize };
            if !size_ok || rs.align != c.align as usize {
                bad.push(format!(
                    "{}: C is {} bytes aligned {}, we are {} aligned {}",
                    rs.name, c.size, c.align, rs.size, rs.align
                ));
            }
        }
        for (rs, c) in layout::MEMBERS.iter().zip(c_members) {
            if rs.offset != c.offset as usize || rs.size != c.size as usize {
                bad.push(format!(
                    "{}.{}: C is {} bytes at offset {}, we are {} at {}",
                    rs.ty, rs.name, c.size, c.offset, rs.size, rs.offset
                ));
            }
        }
        assert!(bad.is_empty(), "{} members disagree:\n{}", bad.len(), bad.join("\n"));
    }
}

/// A test per command that its array accessors hand back the array the wire carried.
///
/// Inside this module because that is where the array members are visible: they are
/// `pub(in crate::venus::proto)` so nothing outside can read a pointer without its count, and the
/// witness has to read exactly that to say the accessor named the right member.
#[cfg(test)]
#[allow(non_snake_case)]
mod witness {
    include!(concat!(env!("OUT_DIR"), "/venus/witness.rs"));
}

#[cfg(test)]
mod tests {
    use super::serialize::*;
    use super::types::*;
    use crate::venus::cs::{AllOfIt, Decoder, Encoder, IdentityObjects};
    use bumpalo::Bump;
    use std::sync::atomic::AtomicBool;

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
        let hard = AtomicBool::new(false);
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
        let hard = AtomicBool::new(false);
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

    /// An output blob is room the guest offers and bytes the host sends back, and the two calls
    /// that shape it look nothing alike on the wire.
    ///
    /// The count call sends a size pointer whose value is whatever the guest had lying around --
    /// GTK's is an uninitialised `size_t` -- and a zero count for the blob; the decoder must
    /// carry the value and allocate nothing. The data call sends the room as the count, and the
    /// decoder must hand the handler exactly that much arena to write into. Either call that the
    /// decoder drops on the floor reaches the guest as no reply at all, which the driver reports
    /// as `VK_ERROR_OUT_OF_HOST_MEMORY` and a client then treats as a malloc of the garbage size.
    #[test]
    fn an_out_blob_is_room_the_guest_offers_and_bytes_the_host_returns() {
        const CMD: VkCommandTypeEXT = VkCommandTypeEXT::VK_COMMAND_TYPE_vkGetPipelineCacheData_EXT;
        const GARBAGE: usize = 0xdead_beef_dead_beef;

        #[derive(Default)]
        struct Cache {
            saw: Vec<(bool, usize)>,
            reply: Vec<u8>,
        }
        impl Commands for Cache {
            fn unsupported(&mut self, cmd: VkCommandTypeEXT) {
                panic!("{cmd:?} was refused");
            }
            fn vkGetPipelineCacheData(&mut self, args: &mut vn_command_vkGetPipelineCacheData) {
                let offered = *args.pDataSize_mut().expect("the size pointer was sent");
                self.saw.push((args.has_pData(), offered));
                args.ret = VkResult::VK_SUCCESS;
                match args.pData_mut() {
                    // The count call: say how much there is.
                    None => *args.pDataSize_mut().unwrap() = 5,
                    // The data call: fill what fits and say how much that was.
                    Some(room) => {
                        let n = room.len().min(5);
                        room[..n].copy_from_slice(&b"cache"[..n]);
                        *args.pDataSize_mut().unwrap() = n;
                    }
                }
                let size = vn_sizeof_vkGetPipelineCacheData_reply(&AllOfIt, args);
                self.reply = vec![0u8; size];
                let mut enc = Encoder::new(&mut self.reply, &AllOfIt);
                vn_encode_vkGetPipelineCacheData_reply(&mut enc, args);
                assert!(!enc.fatal());
            }
        }

        fn call(size: usize, room: u64) -> Vec<u8> {
            wire(&[
                &1u64.to_le_bytes(), // device
                &2u64.to_le_bytes(), // pipelineCache
                &1u64.to_le_bytes(), // pDataSize: present
                &size.to_le_bytes(), // its value, which is only meaningful with pData
                &room.to_le_bytes(), // pData: the room offered, and no bytes
            ])
        }
        fn reply(ret: i32, size: usize, blob: &[&[u8]]) -> Vec<u8> {
            let mut w = wire(&[
                &(CMD.0).to_le_bytes(),
                &ret.to_le_bytes(),
                &1u64.to_le_bytes(),
                &size.to_le_bytes(),
            ]);
            w.extend(wire(blob));
            w
        }

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut h = Cache::default();

        // The count call, as GTK sends it: the size is garbage and the blob is absent.
        let w = call(GARBAGE, 0);
        let mut dec = Decoder::new(&w, &temp, &IdentityObjects, &hard);
        assert_eq!(vn_dispatch_command(&mut dec, None, CMD, &mut h), Some(()));
        assert!(!dec.fatal(), "a garbage size beside an absent blob is what every count call is");
        assert_eq!(dec.pos(), w.len());
        assert_eq!(h.saw, [(false, GARBAGE)]);
        assert_eq!(h.reply, reply(0, 5, &[&0u64.to_le_bytes()]));

        // The data call: the room is the count, and the blob comes back padded to the word.
        let w = call(8, 8);
        let mut dec = Decoder::new(&w, &temp, &IdentityObjects, &hard);
        assert_eq!(vn_dispatch_command(&mut dec, None, CMD, &mut h), Some(()));
        assert!(!dec.fatal());
        assert_eq!(dec.pos(), w.len());
        assert_eq!(h.saw, [(false, GARBAGE), (true, 8)]);
        assert_eq!(h.reply, reply(0, 5, &[&5u64.to_le_bytes(), b"cache\0\0\0"]));

        // A guest that disagrees with itself about the room is not served.
        let w = call(8, 4);
        let mut dec = Decoder::new(&w, &temp, &IdentityObjects, &hard);
        assert_eq!(vn_dispatch_command(&mut dec, None, CMD, &mut h), Some(()));
        assert!(dec.fatal(), "the count and the length member are one value");
        assert_eq!(h.saw.len(), 2, "a poisoned decode reaches no handler");

        // Nor is one that offers more room than the arena will ever hold.
        let hard = AtomicBool::new(false);
        let huge = crate::venus::cs::TEMP_POOL_MAX + 1;
        let w = call(huge, huge as u64);
        let mut dec = Decoder::new(&w, &temp, &IdentityObjects, &hard);
        assert_eq!(vn_dispatch_command(&mut dec, None, CMD, &mut h), Some(()));
        assert!(dec.fatal(), "the room is bounded at the trust boundary, not by malloc");
        assert_eq!(h.saw.len(), 2);

        // And the round trip reproduces both calls byte for byte, garbage included.
        for w in [call(GARBAGE, 0), call(8, 8)] {
            let hard = AtomicBool::new(false);
            let mut dec = Decoder::new(&w, &temp, &IdentityObjects, &hard);
            let mut buf = vec![0u8; w.len() + 8];
            let mut enc = Encoder::new(&mut buf, &AllOfIt);
            let size =
                vn_round_trip_args(&mut dec, &mut enc, CMD, VkFlags(0)).expect("a defined command");
            assert!(!dec.fatal());
            assert_eq!(size, w.len() + 8, "the header the caller consumed is the encoder\'s");
            assert_eq!(enc.written()[8..], w[..]);
        }
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
