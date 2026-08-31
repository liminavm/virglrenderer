// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The Vulkan driver, reached through the Khronos loader.
//!
//! One of the named unsafe modules (CLAUDE.md): every call into a driver is an FFI call, and this
//! is where they are wrapped so the rest of the renderer is safe Rust.
//!
//! **The loader is mandatory, and it is linked rather than dlopened.** KosmicKrisp is a pure ICD:
//! it exports `vk_icdGetInstanceProcAddr` and `vk_icdNegotiateLoaderICDInterfaceVersion` and
//! nothing else, so there is no `vkGetInstanceProcAddr` to find in it and no way to reach it
//! except through the loader. Linking rather than dlopening is limina's constraint, not a
//! preference: the worker is codesigned, the hardened runtime strips `DYLD_*` from its
//! environment, and `/opt/homebrew/lib` is not on dyld's default path -- so a bare-name dlopen
//! finds nothing and venus enumerates zero GPUs. Which driver runs is then the loader's ICD
//! selection, which is per-process and finer-grained than any build flag.
//!
//! **The tables are generated, not written.** `proc` comes out of the same vk.xml the serializer
//! does, so a command's arguments go from the wire into the driver with no conversion and no
//! second set of signatures to keep in step. That is what `ash` would have cost: a parallel type
//! universe, an unchecked assumption that its layouts match ours, and no answer at all for MESA's
//! venus-private commands.

use core::ffi::{CStr, c_char};

/// The driver's entry points, three tables deep, generated from vk.xml.
#[allow(non_camel_case_types, non_snake_case, dead_code)]
pub mod proc {
    include!(concat!(env!("OUT_DIR"), "/venus/proc.rs"));
}

pub use proc::{Device, Global, Instance, ProcAddr};

use crate::venus::proto::types::{VkDevice, VkInstance};

// The loader, linked. `vkGetInstanceProcAddr` is the one symbol Vulkan guarantees a loader
// exports and the root of every table below it.
#[link(name = "vulkan")]
unsafe extern "C" {
    fn vkGetInstanceProcAddr(instance: VkInstance, name: *const c_char) -> Option<ProcAddr>;
}

/// Resolve a name against an instance, or against the loader itself when the instance is null.
///
/// # Safety
///
/// This is the loader's own entry point; the only requirement is that `name` is a valid C string,
/// which the caller supplies as a `&CStr`.
fn instance_proc(instance: VkInstance, name: &CStr) -> Option<ProcAddr> {
    // SAFETY: `name` is a `&CStr`, so it is NUL-terminated and live for the call. A null instance
    // is what the spec requires for a global command, and the loader is defined for it.
    unsafe { vkGetInstanceProcAddr(instance, name.as_ptr()) }
}

/// The commands that exist before any instance does: `vkCreateInstance` and the two
/// `vkEnumerateInstance*` queries.
pub fn global() -> Global {
    // SAFETY: every name comes from vk.xml as the name of the command whose signature it is
    // transmuted to, and `vkGetInstanceProcAddr(VK_NULL_HANDLE, ..)` is the spec's own way to
    // reach a global command.
    unsafe { Global::load(&mut |name| instance_proc(VkInstance(0), name)) }
}

/// The instance-level commands, including every `vkGetPhysicalDevice*` query.
pub fn instance(instance: VkInstance) -> Instance {
    assert!(instance.0 != 0, "an instance table needs an instance");
    // SAFETY: as `global`, with a real instance -- which is what makes the physical-device
    // commands resolvable at all.
    unsafe { Instance::load(&mut |name| instance_proc(instance, name)) }
}

/// The device-level commands, resolved through the device so they skip the loader's dispatch
/// trampoline -- which is the whole reason Vulkan has a second proc-addr call.
pub fn device(inst: &Instance, device: VkDevice) -> Device {
    assert!(device.0 != 0, "a device table needs a device");
    let get_device_proc_addr = inst.vkGetDeviceProcAddr();
    // SAFETY: `get_device_proc_addr` came from the loader under its own name, `device` is a
    // handle the driver returned, and each name is the command whose signature it becomes.
    unsafe { Device::load(&mut |name| get_device_proc_addr(device, name.as_ptr())) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loader has to be there and answer for the commands that exist before an instance does.
    /// It needs no ICD and no GPU: these three are the loader's own, and if this fails the link
    /// is wrong rather than the driver.
    #[test]
    fn the_loader_answers_for_the_global_commands() {
        let g = global();
        assert!(g.has_vkCreateInstance());
        assert!(g.has_vkEnumerateInstanceVersion());
        let (got, total) = g.loaded();
        assert_eq!(got, total, "the loader must answer for every global command");
    }
}
