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

use super::cs::{Handle, ObjectId};
use super::proto::types::{
    VkAllocationCallbacks, VkBaseInStructure, VkCopyDescriptorSet, VkDevice, VkDeviceCreateInfo,
    VkDeviceMemory, VkDeviceQueueInfo2, VkDeviceSize, VkExtensionProperties, VkFlags, VkInstance,
    VkInstanceCreateInfo, VkMemoryAllocateInfo, VkMemoryPropertyFlagBits, VkMemoryPropertyFlags,
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties, VkQueue, VkResult, VkStructureType,
    VkWriteDescriptorSet,
};
use crate::vulkan::{self, Device as DeviceFns, Global, Instance as InstanceFns};

/// `VK_WHOLE_SIZE`: map an allocation from an offset to its end.
const VK_WHOLE_SIZE: VkDeviceSize = VkDeviceSize(!0);

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
    /// The instance's own handle, so a teardown with no command behind it can still destroy it.
    instance_handle: u64,
    /// Keyed by *host* handle, not guest id. Every handler that needs a device's entry points
    /// reaches them through the `VkDevice` the lookup already resolved for it; only a destroy
    /// carries the guest id, and it does not need the table to find one.
    devices: BTreeMap<u64, DeviceState>,
    /// What each physical device supports, by name, keyed by host handle.
    ///
    /// A set of names rather than the C's one bool per extension: the question asked of it is
    /// always "does this driver have <name>", and a hand-maintained struct of booleans is a list
    /// that has to be extended every time a new name matters.
    physical_device_exts: BTreeMap<u64, BTreeSet<String>>,
    /// Live device memory, keyed by the *guest's* id -- because that is the name the census
    /// reports and the VMM reads back by. See [`Memory`].
    memory: BTreeMap<u64, Memory>,
    /// Live command and descriptor pools, by host handle, each with the objects allocated from it.
    ///
    /// Destroying a pool destroys everything in it, and nothing in the guest's stream says so --
    /// so without this the object table would keep resolving ids whose driver objects are gone,
    /// and the next command naming one would hand the driver a freed handle. See
    /// [`Driver::pool_child`].
    pools: BTreeMap<u64, BTreeSet<u64>>,
    /// Every pool-allocated object, by host handle, pointing back at its pool. The reverse of
    /// `pools`, because the question a command asks is "is this buffer still alive", and answering
    /// it by searching every pool would be a scan on the hottest path there is.
    pool_children: BTreeMap<u64, u64>,
}

/// One live `VkDevice`: its entry points, and what its allocations need to know.
struct DeviceState {
    fns: DeviceFns,
    /// The property flags of each memory type, indexed by `memoryTypeIndex`. Read once at device
    /// creation because it never changes, and because an allocation must not pay an instance
    /// round trip to learn whether it is host-visible.
    memory_types: Vec<VkMemoryPropertyFlags>,
}

/// One live `VkDeviceMemory`.
///
/// Tracked apart from the object table because the census needs two things the table does not
/// hold: the size the guest asked for, and which device owns the allocation -- a device cannot be
/// destroyed while its memory is live, so a teardown has to walk from one to the other.
pub struct Memory {
    /// The host `VkDevice` that owns it.
    device: u64,
    handle: u64,
    /// `allocationSize` as the guest asked for it. The driver may have rounded up; the census
    /// reports the guest's number because that is what the guest will read back.
    size: u64,
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
        self.devices.get(&device.0).map(|d| &d.fns)
    }

    /// Destroy everything this context still holds, devices before the instance.
    ///
    /// Both the guest's `vkDestroyInstance` and the context's own teardown come here. A guest is
    /// under no obligation to have destroyed its devices first, and a context can go away
    /// mid-workload having sent no destroy at all -- which is the common case, because a guest
    /// that is still drawing when its VM stops never unwinds. Emptying the maps is what makes
    /// each destroy happen once.
    pub fn teardown(&mut self) {
        for (handle, d) in core::mem::take(&mut self.devices) {
            // A device cannot be destroyed while its memory is live -- Vulkan calls that an
            // application error, and the guest is under no obligation to have avoided it.
            self.free_device_memory(&d.fns, handle);
            // SAFETY: a handle this context created, and the table was loaded from it.
            unsafe { (d.fns.vkDestroyDevice())(VkDevice(handle), core::ptr::null()) };
        }
        if let Some(inst) = self.instance.take() {
            let handle = VkInstance(core::mem::take(&mut self.instance_handle));
            // SAFETY: as above.
            unsafe { (inst.vkDestroyInstance())(handle, core::ptr::null()) };
        }
        self.physical_device_exts.clear();
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
        // One instance per context, as `objects` describes: a second would orphan the first's
        // devices and leak it, and no guest has a reason to ask.
        if self.instance.is_some() {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        let mut out = VkInstance(0);
        // SAFETY: `info` and `alloc` are the decoder's arena allocations, live for this call, and
        // `out` is a local. The guest cannot make them dangle: the arena outlives the batch.
        let r = unsafe { (global.vkCreateInstance())(info, alloc, &mut out) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        assert!(out.0 != 0, "vkCreateInstance succeeded and returned a null instance");
        self.instance = Some(vulkan::instance(out));
        self.instance_handle = out.0;
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
        let mut props = VkPhysicalDeviceMemoryProperties::default();
        // SAFETY: `pd` is a handle this instance returned, and `props` is a local.
        unsafe { (inst.vkGetPhysicalDeviceMemoryProperties())(pd, &mut props) };
        let memory_types = props.memoryTypes[..props.memoryTypeCount.min(32) as usize]
            .iter()
            .map(|t| t.propertyFlags)
            .collect();
        self.devices.insert(out.0, DeviceState { fns: vulkan::device(inst, out), memory_types });
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
        unsafe { (d.fns.vkGetDeviceQueue2())(device, info, &mut out) };
        (out.0 != 0).then_some(out)
    }

    /// Destroy a device and forget its entry points.
    pub fn destroy_device(&mut self, device: VkDevice) {
        let Some(d) = self.devices.remove(&device.0) else {
            return;
        };
        self.free_device_memory(&d.fns, device.0);
        // SAFETY: a handle this context created, destroyed once -- `remove` is what makes it once.
        unsafe { (d.fns.vkDestroyDevice())(device, core::ptr::null()) };
    }

    // ------------------------------------------------------------------- simple objects

    /// Create an object whose whole host action is one `vkCreateX(device, info, alloc, out)`.
    ///
    /// The entry point arrives as a closure over the device's proc table rather than as a name,
    /// because that is the one part of the call that differs between the twenty objects shaped
    /// like this -- and taking it this way is what keeps the `unsafe` here instead of at each of
    /// the twenty call sites.
    ///
    /// The device is looked up in *this* table and never taken from the caller on trust. An id the
    /// object table resolved is not evidence the driver still has the device: a guest that
    /// destroys a device and then creates against it gets a rejection, not a call on a dead
    /// handle.
    pub fn create_object<T: Handle, I>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(
            &DeviceFns,
        ) -> unsafe extern "C" fn(
            VkDevice,
            *const I,
            *const VkAllocationCallbacks,
            *mut T,
        ) -> VkResult,
        info: *const I,
        alloc: *const VkAllocationCallbacks,
    ) -> Result<u64, VkResult> {
        let Some(d) = self.devices.get(&device.0) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        let mut out = T::from_raw(0);
        // SAFETY: `device` is a handle in this table, `info` and `alloc` are the decoder's arena
        // allocations live for this call, and `out` is a local.
        let r = unsafe { proc(&d.fns)(device, info, alloc, &mut out) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        let handle = out.raw();
        assert!(handle != 0, "a create succeeded and returned a null handle");
        Ok(handle)
    }

    /// Destroy an object created by [`Driver::create_object`].
    ///
    /// A destroy naming a device this table does not have is a no-op: the device is already gone,
    /// and everything it owned went with it.
    pub fn destroy_object<T: Handle>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, T, *const VkAllocationCallbacks),
        object: T,
        alloc: *const VkAllocationCallbacks,
    ) {
        let Some(d) = self.devices.get(&device.0) else {
            return;
        };
        if object.raw() == 0 {
            // Vulkan makes destroying a null handle a legal no-op, and guests rely on it.
            return;
        }
        // SAFETY: `device` and `object` are handles this context created, and the generated
        // lifecycle hook removes the id from the object table exactly once, so this runs once.
        unsafe { proc(&d.fns)(device, object, alloc) };
    }

    /// Allocate a run of objects from a pool: `vkAllocateX(device, info, out)`.
    ///
    /// Unlike an enumeration there is no short answer to handle -- Vulkan fills every element or
    /// none -- so the caller's only two cases are the whole array and nothing.
    ///
    /// `out` is the shadow array the decoder allocated, so the driver writes host handles straight
    /// into the place the generated lifecycle hook will read them from.
    pub fn allocate_objects<T: Handle, I>(
        &mut self,
        device: VkDevice,
        pool: u64,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, *const I, *mut T) -> VkResult,
        info: *const I,
        out: *mut T,
        count: usize,
    ) -> Result<(), VkResult> {
        let Some(d) = self.devices.get(&device.0) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        if info.is_null() || out.is_null() {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        // The pool is re-checked here for the same reason the device is: the guest may have
        // destroyed it, and an id the object table still resolves is not a live driver object.
        if !self.pools.contains_key(&pool) {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        // SAFETY: `device` is a handle in this table; `info` is an arena allocation live for the
        // call, and `out` is the arena array the decoder sized from the count inside `info`.
        let r = unsafe { proc(&d.fns)(device, info, out) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        for i in 0..count {
            // SAFETY: `i` is inside the array the decoder sized to `count`, and the driver filled
            // it -- Vulkan fills every element of a pool allocation or none.
            let child = unsafe { *out.add(i) }.raw();
            if child != 0 {
                self.pools.entry(pool).or_default().insert(child);
                self.pool_children.insert(child, pool);
            }
        }
        Ok(())
    }

    /// Whether a pool-allocated object is still live -- its pool undestroyed and it unfreed.
    ///
    /// The check every command that *uses* a command buffer owes before handing it to the driver:
    /// the object table resolves an id to a handle, but only this says the handle still names
    /// something.
    pub fn pool_child(&self, handle: u64) -> bool {
        self.pool_children.contains_key(&handle)
    }

    /// Free a run of objects back to the pool they came from.
    pub fn free_objects<T: Handle, P: Handle>(
        &mut self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, P, u32, *const T),
        pool: P,
        count: u32,
        objects: *const T,
    ) {
        let Some(d) = self.devices.get(&device.0) else {
            return;
        };
        if count == 0 || objects.is_null() || !self.pools.contains_key(&pool.raw()) {
            return;
        }
        // SAFETY: handles this context allocated, in the arena array the decoder sized to `count`.
        // The generated lifecycle hook removes the ids from the object table exactly once.
        unsafe { proc(&d.fns)(device, pool, count, objects) };
        let children = self.pools.entry(pool.raw()).or_default();
        for i in 0..count as usize {
            // SAFETY: as above.
            let child = unsafe { *objects.add(i) }.raw();
            children.remove(&child);
            self.pool_children.remove(&child);
        }
    }

    /// Create a pool, and start tracking what will be allocated from it.
    pub fn create_pool<T: Handle, I>(
        &mut self,
        device: VkDevice,
        proc: impl FnOnce(
            &DeviceFns,
        ) -> unsafe extern "C" fn(
            VkDevice,
            *const I,
            *const VkAllocationCallbacks,
            *mut T,
        ) -> VkResult,
        info: *const I,
        alloc: *const VkAllocationCallbacks,
    ) -> Result<u64, VkResult> {
        let handle = self.create_object(device, proc, info, alloc)?;
        self.pools.insert(handle, BTreeSet::new());
        Ok(handle)
    }

    /// Destroy a pool, and with it everything allocated from it.
    ///
    /// Vulkan frees a pool's objects when the pool goes, without a command per object -- so this
    /// is the only place their handles stop being live, and forgetting them here is what keeps a
    /// later command from reaching the driver with one.
    pub fn destroy_pool<T: Handle>(
        &mut self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, T, *const VkAllocationCallbacks),
        pool: T,
        alloc: *const VkAllocationCallbacks,
    ) {
        for child in self.pools.remove(&pool.raw()).unwrap_or_default() {
            self.pool_children.remove(&child);
        }
        self.destroy_object(device, proc, pool, alloc);
    }

    // ------------------------------------------------------------ binding and updating
    //
    // Commands that change an existing object rather than create one. They register nothing and
    // return no handle, so nothing here touches the object table -- but a submit that draws from
    // a buffer with no memory bound, or through a descriptor set that was never written, is
    // undefined behaviour just as surely as one naming a handle that does not exist. Serving
    // these is what makes a later `vkQueueSubmit` safe to pass through.

    /// Bind memory to a run of buffers or images: `vkBindXMemory2(device, count, infos)`.
    ///
    /// Both `vkBindBufferMemory2` and `vkBindImageMemory2` have exactly this shape, which is why
    /// the entry point arrives as a closure -- the info type is the only thing that differs, and
    /// it is a type parameter.
    pub fn bind_memory<I>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, u32, *const I) -> VkResult,
        count: u32,
        infos: *const I,
    ) -> VkResult {
        let Some(d) = self.devices.get(&device.0) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // Binding nothing is legal and the guest sends it. The count is the caller's, already
        // reconciled against the array it came with, so a zero here means an absent array and
        // never a pointer this must not read.
        if count == 0 {
            return VkResult::VK_SUCCESS;
        }
        // SAFETY: `device` is a handle in this table, and `infos` is the decoder's arena array,
        // sized to `count` and live for this call.
        unsafe { proc(&d.fns)(device, count, infos) }
    }

    /// Write and copy descriptors: `vkUpdateDescriptorSets`.
    ///
    /// Alone among the commands here it returns nothing -- Vulkan gives it no failure to report,
    /// because everything it could refuse is a validation error the guest was required not to
    /// commit. So a device this table does not have is a silent no-op, the same as a destroy.
    pub fn update_descriptor_sets(
        &self,
        device: VkDevice,
        writes: (u32, *const VkWriteDescriptorSet),
        copies: (u32, *const VkCopyDescriptorSet),
    ) {
        let Some(d) = self.devices.get(&device.0) else {
            return;
        };
        // Both counts are the caller's, each already reconciled against the array it came with.
        let ((nw, pw), (nc, pc)) = (writes, copies);
        if nw == 0 && nc == 0 {
            return;
        }
        // SAFETY: `device` is a handle in this table, and both arrays are the decoder's arena
        // allocations, sized to their counts and live for this call.
        unsafe { (d.fns.vkUpdateDescriptorSets())(device, nw, pw, nc, pc) };
    }

    // ------------------------------------------------------------------- device memory

    /// Allocate device memory and remember it, so the census can find it by the guest's id.
    ///
    /// `id` is the guest's, and it is what goes in the table: the census reports guest ids and the
    /// VMM reads back by them. The host handle it returns is the one the guest's shadow gets.
    pub fn allocate_memory(
        &mut self,
        device: VkDevice,
        id: u64,
        info: *const VkMemoryAllocateInfo,
        alloc: *const VkAllocationCallbacks,
    ) -> Result<VkDeviceMemory, VkResult> {
        if info.is_null() {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        let Some(d) = self.devices.get(&device.0) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        // A copy, not an edit in place: the decoder's struct is the guest's request, and the
        // round trip re-encodes it. The `pNext` chain is carried over untouched.
        // SAFETY: non-null, and the decoder allocated it in the arena for this batch.
        let mut info = unsafe { *info };
        info.allocationSize = VkDeviceSize(pad_for_blob(
            info.allocationSize.0,
            d.memory_types.get(info.memoryTypeIndex as usize).copied(),
            imports_a_resource(info.pNext),
        ));

        let mut out = VkDeviceMemory(0);
        // SAFETY: `info` is a local whose chain the decoder owns for the batch, `alloc` is another
        // of its arena allocations, and `out` is a local.
        let r = unsafe { (d.fns.vkAllocateMemory())(device, &info, alloc, &mut out) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        assert!(out.0 != 0, "vkAllocateMemory succeeded and returned a null handle");
        let size = info.allocationSize.0;
        self.memory.insert(id, Memory { device: device.0, handle: out.0, size });
        Ok(out)
    }

    /// Free device memory the guest named by id.
    pub fn free_memory(&mut self, id: u64) {
        let Some(mem) = self.memory.remove(&id) else {
            return;
        };
        let Some(d) = self.devices.get(&mem.device) else {
            return;
        };
        // SAFETY: a handle this context allocated, freed once -- `remove` is what makes it once.
        unsafe {
            (d.fns.vkFreeMemory())(
                VkDevice(mem.device),
                VkDeviceMemory(mem.handle),
                core::ptr::null(),
            )
        };
    }

    /// Free every allocation belonging to one device, on the way to destroying it.
    fn free_device_memory(&mut self, d: &DeviceFns, device: u64) {
        let mine: Vec<u64> =
            self.memory.iter().filter(|(_, m)| m.device == device).map(|(id, _)| *id).collect();
        for id in mine {
            let mem = self.memory.remove(&id).expect("just collected from this map");
            // SAFETY: a handle this context allocated on `d`'s device, freed once.
            unsafe {
                (d.vkFreeMemory())(VkDevice(device), VkDeviceMemory(mem.handle), core::ptr::null())
            };
        }
    }

    /// Every live allocation, for the memory census.
    ///
    /// The C also skips memory it has exported as a blob and memory imported from another
    /// context's storage -- in both cases the bytes are captured where they actually live, not
    /// here. Neither flag can be set yet: both are decided by the blob path, which does not exist,
    /// so nothing is skipped and the count reads high against the C by exactly the blobs a corpus
    /// exported.
    pub fn memory_census(&self) -> Vec<Allocation> {
        self.memory.iter().map(|(id, m)| Allocation { id: ObjectId(*id), size: m.size }).collect()
    }

    /// Copy an allocation's contents out through a host mapping, returning how many bytes landed.
    ///
    /// Short buffers are the caller's business, not an error: the census reports whole sizes and
    /// the VMM caps what it reads, so a prefix is the normal request -- which is why the count
    /// comes back rather than being inferred from the buffer's length.
    pub fn memory_read(&self, id: u64, buf: &mut [u8]) -> Result<usize, MemoryError> {
        let Some(mem) = self.memory.get(&id) else {
            return Err(MemoryError::NoSuchAllocation);
        };
        let Some(d) = self.devices.get(&mem.device) else {
            return Err(MemoryError::NoSuchAllocation);
        };
        let device = VkDevice(mem.device);
        let handle = VkDeviceMemory(mem.handle);
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: a device and an allocation this context made, and `ptr` is a local.
        let r = unsafe {
            (d.fns.vkMapMemory())(
                device,
                handle,
                VkDeviceSize(0),
                VK_WHOLE_SIZE,
                VkFlags(0),
                &mut ptr,
            )
        };
        if r != VkResult::VK_SUCCESS || ptr.is_null() {
            return Err(MemoryError::NotMappable);
        }
        let n = buf.len().min(mem.size as usize);
        // SAFETY: the driver mapped at least `mem.size` bytes at `ptr`, which is what `n` is
        // clamped to, and `buf` is a live slice of at least `n`. The two cannot overlap: one is
        // the driver's mapping and the other the caller's.
        unsafe { core::ptr::copy_nonoverlapping(ptr.cast::<u8>(), buf.as_mut_ptr(), n) };
        // SAFETY: the mapping this call just made, unmapped once.
        unsafe { (d.fns.vkUnmapMemory())(device, handle) };
        Ok(n)
    }
}

/// One live allocation, as the census reports it.
///
/// Two bare `u64`s side by side is how the ABI carries this, and exactly the confusion the
/// newtypes exist to prevent -- so they are named here and paired only at the shim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Allocation {
    /// The id the guest gave the `VkDeviceMemory`, which is the name the VMM reads it back by.
    pub id: ObjectId,
    /// Its size in bytes, padded to the blob the guest may map it as -- see [`pad_for_blob`].
    pub size: u64,
}

/// Why an allocation could not be read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryError {
    /// Nothing is allocated under that id in this context.
    NoSuchAllocation,
    /// The allocation exists, but the driver would not map it. Device-local memory is not the
    /// driver's to hand out, and a scanout backed by an IOSurface lives in the surface rather
    /// than in the allocation -- in both cases the bytes are reachable, but by the blob path.
    NotMappable,
}

/// Round a host-visible allocation up to the size of the blob the guest may create from it.
///
/// A guest that maps memory does it by exporting the allocation as a virtio-gpu blob, and a blob
/// is sized in 64 KiB units. An allocation smaller than its own blob leaves the guest with a
/// mapping that runs off the end of what the driver actually reserved, which is a host
/// out-of-bounds read on the guest's say-so.
///
/// Only host-visible memory, because only host-visible memory is ever mapped; and never an import,
/// which aliases bytes that already exist at a size the exporter fixed.
fn pad_for_blob(size: u64, flags: Option<VkMemoryPropertyFlags>, imported: bool) -> u64 {
    /// Blobs are counted in 64 KiB units.
    const BLOB_ALIGN: u64 = 64 * 1024;
    const HOST_VISIBLE: VkMemoryPropertyFlagBits =
        VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

    let Some(flags) = flags else {
        // A memory type index the device never reported. The driver will reject it, and padding a
        // size for an allocation that is about to fail would only obscure the failure.
        return size;
    };
    if imported || flags.0 & HOST_VISIBLE.0 as u32 == 0 {
        return size;
    }
    // Checked, because `size` is the guest's own number and nothing has bounded it yet. A size with
    // no next multiple goes to the driver as it stands: the driver refuses it, `plant` ghosts the
    // id, and the guest gets a clean failure -- which is what padding it into a wrap would have
    // taken away, by handing the driver a small allocation the guest believes is enormous.
    size.checked_next_multiple_of(BLOB_ALIGN).unwrap_or(size)
}

/// Whether an allocation's `pNext` chain imports another context's storage.
///
/// Walked rather than asked of the guest, because the chain is where the guest put it.
fn imports_a_resource(mut node: *const core::ffi::c_void) -> bool {
    while !node.is_null() {
        // SAFETY: every link is a struct the decoder allocated in the batch arena, and every one
        // of them begins with the `sType`/`pNext` header `VkBaseInStructure` names.
        let base = unsafe { &*node.cast::<VkBaseInStructure>() };
        if base.sType == VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA {
            return true;
        }
        node = base.pNext.cast();
    }
    false
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

#[cfg(test)]
mod tests {
    use super::super::proto::types::VkCommandPool;
    use super::*;

    const HOST_VISIBLE: VkMemoryPropertyFlags =
        VkFlags(VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.0 as u32);
    const DEVICE_LOCAL: VkMemoryPropertyFlags =
        VkFlags(VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.0 as u32);

    /// Destroying a pool destroys everything in it, and the guest sends no command per object --
    /// so if the pool's contents stay tracked here, a later command naming one gets past the
    /// re-check and reaches the driver with a handle Vulkan already freed.
    #[test]
    fn a_pool_forgets_its_contents_when_it_is_destroyed() {
        let mut d = Driver::default();
        d.pools.insert(7, BTreeSet::from([11, 12]));
        d.pool_children.insert(11, 7);
        d.pool_children.insert(12, 7);
        assert!(d.pool_child(11) && d.pool_child(12));

        // No device is registered, so the driver call itself is skipped -- the bookkeeping is
        // what is under test, and it has to happen either way.
        d.destroy_pool(
            VkDevice(0),
            |f| f.vkDestroyCommandPool(),
            VkCommandPool(7),
            core::ptr::null(),
        );
        assert!(!d.pool_child(11), "a buffer outlived the pool it came from");
        assert!(!d.pool_child(12), "a buffer outlived the pool it came from");
        assert!(!d.pools.contains_key(&7));
    }

    /// The census reports the padded size, so this rule is directly what a score compares.
    #[test]
    fn only_a_mappable_allocation_is_padded_to_its_blob() {
        // Host-visible memory is what the guest maps, and it maps it through a 64 KiB blob.
        assert_eq!(pad_for_blob(1, Some(HOST_VISIBLE), false), 65536);
        assert_eq!(pad_for_blob(245760, Some(HOST_VISIBLE), false), 262144);
        assert_eq!(pad_for_blob(4096000, Some(HOST_VISIBLE), false), 4128768);
        // A size that is already a whole number of blobs is left where it is.
        assert_eq!(pad_for_blob(67108864, Some(HOST_VISIBLE), false), 67108864);

        // Device-local memory is never mapped, so padding it would only waste it.
        assert_eq!(pad_for_blob(4096, Some(DEVICE_LOCAL), false), 4096);
        // An import aliases bytes the exporter sized; growing the request would run off them.
        assert_eq!(pad_for_blob(4096, Some(HOST_VISIBLE), true), 4096);
        // A memory type the device never reported: let the driver reject it as it is.
        assert_eq!(pad_for_blob(4096, None, false), 4096);
        // A size with no next multiple. The guest chooses this number, so rounding it must not
        // wrap it to zero and hand the driver a tiny allocation the guest thinks is enormous.
        assert_eq!(pad_for_blob(u64::MAX, Some(HOST_VISIBLE), false), u64::MAX);
    }

    /// A zero allocation is legal to ask for and must not become a blob-sized one.
    #[test]
    fn a_zero_allocation_stays_zero() {
        assert_eq!(pad_for_blob(0, Some(HOST_VISIBLE), false), 0);
    }
}
