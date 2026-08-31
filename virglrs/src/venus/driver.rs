// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! One context's live driver state, and the safe calls the handlers reach it through.
//!
//! Part of the Vulkan binding surface (CLAUDE.md): every `unsafe` here is one FFI call into the
//! driver, wrapped so a handler in `context.rs` is safe Rust. What it wraps is the state the
//! object table cannot hold -- the object table keys host handles by guest id, but a device also
//! owns a proc table, and a proc table is not a `u64`.
//!
//! A context owns at most one `VkInstance`, so its instance table is one slot. Devices are a map:
//! Vulkan permits several per instance, and keying them by the guest's id is the same lookup every
//! other object gets.

use std::collections::{BTreeMap, BTreeSet};

use super::proto::types::{
    VkAllocationCallbacks, VkDevice, VkDeviceCreateInfo, VkDeviceQueueInfo2, VkExtensionProperties,
    VkInstance, VkInstanceCreateInfo, VkPhysicalDevice, VkQueue, VkResult,
};
use crate::vulkan::{self, Device as DeviceFns, Global, Instance as InstanceFns};

/// The extensions the guest asks for that this renderer *emulates* rather than forwards.
///
/// A venus guest believes it is on Linux: it asks for dma-buf and fd-based external memory because
/// that is how zero-copy works there. On macOS the driver has neither, and forwarding one fails the
/// whole `vkCreateDevice` with `VK_ERROR_EXTENSION_NOT_PRESENT`. They are dropped here and provided
/// by the Metal path instead.
///
/// **Only these two.** An unsupported extension that this renderer does *not* emulate is left in
/// the list on purpose: dropping it would let the guest enable a feature nothing implements and
/// then use it, which is a wrong picture instead of a clean failure. If a driver does support them
/// natively they are re-added by `HOST_EXTENSIONS`, so this is a filter and not a ban.
const EMULATED_ON_THE_HOST: [&str; 2] =
    ["VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf"];

/// The extensions the *renderer* needs on a device, added to whatever the guest asked for, each
/// only if the driver has it.
///
/// The guest never asks for these -- they are how the host side does its half of the work: Metal
/// interop for the IOSurface path, and the fd-based handles when a driver really does have them.
const HOST_EXTENSIONS: [&str; 6] = [
    "VK_EXT_external_memory_metal",
    "VK_EXT_metal_objects",
    "VK_KHR_portability_subset",
    "VK_KHR_external_memory_fd",
    "VK_EXT_external_memory_dma_buf",
    "VK_KHR_external_fence_fd",
];

/// The driver objects one context has stood up.
#[derive(Default)]
pub struct Driver {
    instance: Option<InstanceFns>,
    /// Keyed by *host* handle, not guest id. Every handler that needs a device's entry points
    /// reaches them through the `VkDevice` the lookup already resolved for it; only a destroy
    /// carries the guest id, and it does not need the table to find one.
    devices: BTreeMap<u64, DeviceFns>,
    /// What each physical device supports, by name, keyed by host handle.
    ///
    /// A set of names rather than the C's one bool per extension: the question asked of it is
    /// always "does this driver have <name>", and a hand-maintained struct of booleans is a list
    /// that has to be extended every time a new name matters.
    physical_device_exts: BTreeMap<u64, BTreeSet<String>>,
}

// Every pointer these take is one the decoder allocated in the batch arena and handed to a
// handler; the arena outlives the whole submission, so none can dangle for the length of a call.
// That invariant belongs to the decoder and is stated here rather than in each signature on
// purpose: making these `unsafe fn` would push an `unsafe` block into every one of the several
// hundred handlers `context.rs` will grow, which is the opposite of keeping unsafe in a named
// module. This is that module (CLAUDE.md); the handlers stay safe Rust.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl Driver {
    pub fn new() -> Driver {
        Driver::default()
    }

    /// The instance table, or None when this context has not created an instance.
    ///
    /// A guest command that names an instance cannot reach a handler without the object table
    /// having resolved that instance first, so in practice a handler that needs this has one --
    /// but the guest chooses the order, so it is an `Option` and never an assert.
    pub fn instance(&self) -> Option<&InstanceFns> {
        self.instance.as_ref()
    }

    pub fn device(&self, device: VkDevice) -> Option<&DeviceFns> {
        self.devices.get(&device.0)
    }

    /// Every device this context still holds, for a teardown that has to destroy them.
    pub fn take_devices(&mut self) -> BTreeMap<u64, DeviceFns> {
        core::mem::take(&mut self.devices)
    }

    pub fn take_instance(&mut self) -> Option<InstanceFns> {
        self.instance.take()
    }

    /// Create the context's instance, and load the entry points that hang off it.
    ///
    /// Returns the host handle, or the driver's error. A `VK_NULL_HANDLE` on success would be a
    /// driver breaking its own contract, so it is an assert rather than a rejection: nothing the
    /// guest sent can cause it.
    pub fn create_instance(
        &mut self,
        global: &Global,
        info: *const VkInstanceCreateInfo,
        alloc: *const VkAllocationCallbacks,
    ) -> Result<VkInstance, VkResult> {
        let mut out = VkInstance(0);
        // SAFETY: `info` and `alloc` are the decoder's arena allocations, live for this call, and
        // `out` is a local. The guest cannot make them dangle: the arena outlives the batch.
        let r = unsafe { (global.vkCreateInstance())(info, alloc, &mut out) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        assert!(out.0 != 0, "vkCreateInstance succeeded and returned a null instance");
        self.instance = Some(vulkan::instance(out));
        Ok(out)
    }

    /// Record what a physical device supports, so device creation can be filtered against it.
    ///
    /// Asked once per physical device, when the guest first enumerates them -- the answer does not
    /// change for the life of the instance.
    pub fn learn_extensions(&mut self, pd: VkPhysicalDevice) {
        if self.physical_device_exts.contains_key(&pd.0) {
            return;
        }
        let Some(inst) = self.instance.as_ref() else {
            return;
        };
        let mut n = 0u32;
        // SAFETY: the count query with a null array is the spec's own first call.
        let r = unsafe {
            (inst.vkEnumerateDeviceExtensionProperties())(
                pd,
                core::ptr::null(),
                &mut n,
                core::ptr::null_mut(),
            )
        };
        if r != VkResult::VK_SUCCESS {
            return;
        }
        let mut props = vec![VkExtensionProperties::default(); n as usize];
        // SAFETY: `props` has room for `n`, which is what the count query just said.
        let r = unsafe {
            (inst.vkEnumerateDeviceExtensionProperties())(
                pd,
                core::ptr::null(),
                &mut n,
                props.as_mut_ptr(),
            )
        };
        if r != VkResult::VK_SUCCESS && r != VkResult::VK_INCOMPLETE {
            return;
        }
        props.truncate(n as usize);
        let names = props
            .iter()
            .map(|p| {
                p.extensionName.iter().take_while(|c| **c != 0).map(|c| *c as u8 as char).collect()
            })
            .collect();
        self.physical_device_exts.insert(pd.0, names);
    }

    fn supports(&self, pd: VkPhysicalDevice, name: &str) -> bool {
        self.physical_device_exts.get(&pd.0).is_some_and(|s| s.contains(name))
    }

    /// The extension list to create a device with: the guest's, minus what this renderer emulates,
    /// plus what the renderer needs and the driver has.
    ///
    /// Returns the names as owned strings; the caller turns them into the NUL-terminated array
    /// Vulkan wants and keeps it alive across the call.
    fn device_extensions(&self, pd: VkPhysicalDevice, guest: &[&str]) -> Vec<String> {
        let mut out: Vec<String> = guest
            .iter()
            .filter(|name| {
                // Dropped whether or not the driver has it: if it does, `HOST_EXTENSIONS` puts it
                // back, and doing it in one place is what stops it appearing twice.
                !EMULATED_ON_THE_HOST.contains(name)
            })
            .map(|n| n.to_string())
            .collect();
        for name in HOST_EXTENSIONS {
            if self.supports(pd, name) && !out.iter().any(|h| h == name) {
                out.push(name.to_string());
            }
        }
        out
    }

    /// Create a device under this context's instance, and load its entry points.
    pub fn create_device(
        &mut self,
        pd: VkPhysicalDevice,
        info: *const VkDeviceCreateInfo,
        alloc: *const VkAllocationCallbacks,
    ) -> Result<VkDevice, VkResult> {
        if info.is_null() {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        if self.instance.is_none() {
            // A device on an instance this context never created. The guest named an instance the
            // object table resolved, so this cannot happen without a host bug -- but it is the
            // guest's ordering that would expose it, so it is rejected rather than asserted.
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }

        // SAFETY: non-null, and the decoder allocated it in the arena for this batch.
        let guest_info = unsafe { *info };
        let guest = read_names(
            guest_info.ppEnabledExtensionNames,
            guest_info.enabledExtensionCount as usize,
        );
        let wanted = self.device_extensions(pd, &guest.iter().map(|s| &**s).collect::<Vec<_>>());

        // The list Vulkan reads has to be NUL-terminated pointers, and both it and the strings it
        // points at have to outlive the call -- so they are locals here and not a borrow of
        // anything the guest owns. The create-info is a copy for the same reason: the decoder's
        // struct is the guest's request, and rewriting it in place would edit the record.
        let cstrings: Vec<std::ffi::CString> =
            wanted.iter().filter_map(|n| std::ffi::CString::new(n.as_str()).ok()).collect();
        let ptrs: Vec<*const std::ffi::c_char> = cstrings.iter().map(|c| c.as_ptr()).collect();
        let mut info = guest_info;
        info.enabledExtensionCount = ptrs.len() as u32;
        info.ppEnabledExtensionNames = ptrs.as_ptr();

        let inst = self.instance.as_ref().expect("checked above");
        let mut out = VkDevice(0);
        // SAFETY: `pd` is a handle this instance returned; `info` and everything it points at are
        // live for the call, including the extension array built just above.
        let r = unsafe { (inst.vkCreateDevice())(pd, &info, alloc, &mut out) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        assert!(out.0 != 0, "vkCreateDevice succeeded and returned a null device");
        self.devices.insert(out.0, vulkan::device(inst, out));
        Ok(out)
    }

    /// The instance's physical devices, into a caller-owned slice.
    ///
    /// Vulkan's two-call idiom collapses here: the guest already asked the count, so the slice is
    /// exactly as long as it said and the driver is asked once.
    pub fn physical_devices(
        &self,
        instance: VkInstance,
        out: &mut [VkPhysicalDevice],
    ) -> Result<u32, VkResult> {
        let Some(inst) = self.instance.as_ref() else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        let mut n = out.len() as u32;
        // SAFETY: `out` is a live slice of `n` elements, which is what `n` is initialised to.
        let r = unsafe { (inst.vkEnumeratePhysicalDevices())(instance, &mut n, out.as_mut_ptr()) };
        // VK_INCOMPLETE means the driver had more than the guest asked for, which is the guest's
        // business rather than an error: it gets the count it sized for.
        if r != VkResult::VK_SUCCESS && r != VkResult::VK_INCOMPLETE {
            return Err(r);
        }
        Ok(n)
    }

    /// How many physical devices the instance has, asked with a null array.
    pub fn physical_device_count(&self, instance: VkInstance) -> Result<u32, VkResult> {
        let Some(inst) = self.instance.as_ref() else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        let mut n = 0u32;
        // SAFETY: the count query with a null array is the spec's own first call.
        let r =
            unsafe { (inst.vkEnumeratePhysicalDevices())(instance, &mut n, core::ptr::null_mut()) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        Ok(n)
    }

    /// A device's queue, which is owned by the device and never created or destroyed.
    pub fn device_queue(
        &self,
        device: VkDevice,
        info: *const VkDeviceQueueInfo2,
    ) -> Option<VkQueue> {
        let d = self.devices.get(&device.0)?;
        let mut out = VkQueue(0);
        // SAFETY: `device` is a handle this table was loaded from and `info` is an arena
        // allocation live for the call.
        unsafe { (d.vkGetDeviceQueue2())(device, info, &mut out) };
        (out.0 != 0).then_some(out)
    }

    /// Destroy a device and forget its entry points.
    pub fn destroy_device(&mut self, device: VkDevice) {
        let Some(d) = self.devices.remove(&device.0) else {
            return;
        };
        // SAFETY: a handle this context created, destroyed once -- `remove` is what makes it once.
        unsafe { (d.vkDestroyDevice())(device, core::ptr::null()) };
    }
}

/// Read a Vulkan `const char *const *` array into owned strings.
///
/// Owned rather than borrowed because the list is rebuilt before it is used, and a `&str` into the
/// decoder's arena would tie the rebuilt list's lifetime to the guest's.
fn read_names(names: *const *const std::ffi::c_char, count: usize) -> Vec<String> {
    if names.is_null() {
        return Vec::new();
    }
    (0..count)
        .filter_map(|i| {
            // SAFETY: the decoder allocated this array with `count` elements, each a pointer to a
            // NUL-terminated string it decoded into the same arena.
            let p = unsafe { *names.add(i) };
            (!p.is_null()).then(|| unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into())
        })
        .collect()
}

/// Destroy an instance with a table that has already been taken out of its `Driver`.
///
/// Free-standing because teardown runs after the context has given up its state, and because
/// taking the table by value is what says the instance cannot be destroyed twice.
pub fn destroy_instance(inst: &InstanceFns, instance: VkInstance) {
    // SAFETY: a handle this context created, and the table was loaded from it.
    unsafe { (inst.vkDestroyInstance())(instance, core::ptr::null()) };
}
