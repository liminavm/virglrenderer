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
use super::objects::Doomed;
use super::proto::types::{
    VkAllocationCallbacks, VkBaseInStructure, VkBool32, VkBuffer, VkBufferCopy, VkBufferImageCopy,
    VkBufferMemoryBarrier, VkBufferView, VkCommandBuffer, VkCommandBufferBeginInfo,
    VkCommandBufferResetFlags, VkCommandPool, VkCopyDescriptorSet, VkDependencyFlags,
    VkDescriptorPool, VkDescriptorSet, VkDescriptorSetLayout, VkDescriptorUpdateTemplate, VkDevice,
    VkDeviceCreateInfo, VkDeviceMemory, VkDeviceQueueInfo2, VkDeviceSize, VkEvent,
    VkExtensionProperties, VkExternalSemaphoreHandleTypeFlagBits, VkFence, VkFlags, VkFramebuffer,
    VkImage, VkImageLayout, VkImageMemoryBarrier, VkImageView, VkImportSemaphoreFdInfoKHR,
    VkInstance, VkInstanceCreateInfo, VkMemoryAllocateInfo, VkMemoryBarrier,
    VkMemoryPropertyFlagBits, VkMemoryPropertyFlags, VkObjectType, VkPhysicalDevice,
    VkPhysicalDeviceMemoryProperties, VkPipeline, VkPipelineBindPoint, VkPipelineCache,
    VkPipelineLayout, VkPipelineStageFlags, VkQueryPool, VkQueue, VkRect2D, VkRenderPass,
    VkRenderPassBeginInfo, VkResult, VkSampler, VkSamplerYcbcrConversion, VkSemaphore,
    VkSemaphoreGetFdInfoKHR, VkSemaphoreImportFlagBits, VkShaderModule, VkStructureType,
    VkSubmitInfo, VkSubpassContents, VkViewport, VkWriteDescriptorSet,
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

/// The pools a context has open, and the objects allocated from each.
///
/// Destroying a pool destroys everything in it, and nothing in the guest's stream says so -- so
/// without this the object table would keep resolving ids whose driver objects are gone, and the
/// next command naming one would hand the driver a freed handle.
///
/// Two maps that are exact inverses: a pool to its objects, and each object back to its pool. The
/// reverse direction earns its keep because closing a pool has to find each child again to forget
/// it, and searching every pool for each would be a scan.
///
/// They live behind a type because an inverse maintained at each mutation site is an invariant
/// four call sites have to remember, and the one that forgets leaves a record vouching for a
/// handle the driver has already freed. See CLAUDE.md, "two values that must agree are one value".
///
/// Each child is carried as *both* of its names -- the host handle this table is keyed by, and
/// the guest id the object table is keyed by. Destroying a pool destroys its objects without a
/// command per object, and the guest ids have to come out of the object table at that moment;
/// this is the only place that still knows what they were.
#[derive(Default)]
struct Pools {
    open: BTreeMap<u64, Pool>,
    /// Every pool-allocated object, by host handle, pointing back at its pool.
    owner: BTreeMap<u64, u64>,
}

/// One live pool: the device that owns it, and what has been allocated from it.
struct Pool {
    /// Recorded so a destroyed device can take its pools with it. Vulkan destroys them for us and
    /// says nothing, and a host handle the driver is free to reuse must stop being vouched for the
    /// moment that happens.
    device: u64,
    /// Host handle to the guest id it was allocated under.
    children: BTreeMap<u64, ObjectId>,
}

impl Pools {
    fn open(&mut self, device: u64, pool: u64) {
        self.open.insert(pool, Pool { device, children: BTreeMap::new() });
    }

    fn is_open(&self, pool: u64) -> bool {
        self.open.contains_key(&pool)
    }

    /// The device that owns the pool a handle came from.
    ///
    /// A `vkCmd*` carries only its command buffer: Vulkan does not repeat the device, because a
    /// command buffer already knows its own. Here it does not, so the pool is the way back.
    fn device_of(&self, handle: u64) -> Option<u64> {
        self.open.get(self.owner.get(&handle)?).map(|p| p.device)
    }

    /// Record objects freshly allocated from a pool. Both directions, or neither.
    fn adopt(&mut self, pool: u64, children: impl IntoIterator<Item = (u64, ObjectId)>) {
        let Some(p) = self.open.get_mut(&pool) else {
            return;
        };
        for (handle, id) in children {
            p.children.insert(handle, id);
            self.owner.insert(handle, pool);
        }
    }

    /// Forget objects freed back to their pool. The pool each belongs to is looked up rather than
    /// passed in, so a caller cannot name the wrong one.
    fn release(&mut self, children: impl IntoIterator<Item = u64>) {
        for child in children {
            if let Some(pool) = self.owner.remove(&child)
                && let Some(p) = self.open.get_mut(&pool)
            {
                p.children.remove(&child);
            }
        }
    }

    /// Forget a pool and everything in it, handing back the guest ids that just stopped naming
    /// anything -- the caller owes the object table their removal.
    fn close(&mut self, pool: u64) -> Vec<ObjectId> {
        let children = self.open.remove(&pool).map(|p| p.children).unwrap_or_default();
        for handle in children.keys() {
            self.owner.remove(handle);
        }
        children.into_values().collect()
    }

    /// Forget every pool a device owned, because destroying the device destroyed them.
    fn close_device(&mut self, device: u64) -> Vec<ObjectId> {
        let doomed: Vec<u64> =
            self.open.iter().filter(|(_, p)| p.device == device).map(|(h, _)| *h).collect();
        doomed.into_iter().flat_map(|pool| self.close(pool)).collect()
    }
}

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
    /// Live command and descriptor pools, and what was allocated from each. See [`Pools`].
    pools: Pools,
    /// The device each queue belongs to, by host handle.
    ///
    /// A queue is never created and never destroyed -- `vkGetDeviceQueue2` hands back one the
    /// device already owns -- but `vkQueueSubmit` carries only the queue, so this is the way back
    /// to the entry points. The same problem [`Pools::device_of`] solves for a command buffer, and
    /// kept apart from it because a queue owns nothing and takes nothing with it when it goes.
    queues: BTreeMap<u64, u64>,
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
/// A borrowed argument as the pointer the entry point below it wants.
///
/// This is the whole of the conversion, and it lives here because here is where the C ABI starts.
/// Above it a missing argument is `None`; below it, null -- and nothing in between has to know
/// that Vulkan spells absence with a pointer value.
fn ptr<T>(r: Option<&T>) -> *const T {
    r.map_or(core::ptr::null(), |r| r as *const T)
}

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
    pub fn teardown(&mut self, doomed: &[Doomed]) {
        // The same order a guest's own `vkDestroyDevice` gets: wait idle, then every object the
        // device owns, then the device. `empty_device` picks out the ones that are its.
        for handle in self.devices.keys().copied().collect::<Vec<_>>() {
            self.empty_device(VkDevice(handle), doomed);
        }
        for (handle, d) in core::mem::take(&mut self.devices) {
            // A device cannot be destroyed while its memory is live -- Vulkan calls that an
            // application error, and the guest is under no obligation to have avoided it.
            self.free_device_memory(&d.fns, handle);
            self.pools.close_device(handle);
            self.queues.retain(|_, owner| *owner != handle);
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
        info: Option<&VkInstanceCreateInfo>,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<VkInstance, VkResult> {
        // One instance per context, as `objects` describes: a second would orphan the first's
        // devices and leak it, and no guest has a reason to ask.
        if self.instance.is_some() {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        let mut out = VkInstance(0);
        // SAFETY: `info` and `alloc` are the decoder's arena allocations, live for this call, and
        // `out` is a local. The guest cannot make them dangle: the arena outlives the batch.
        let r = unsafe { (global.vkCreateInstance())(ptr(info), ptr(alloc), &mut out) };
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
        info: Option<&VkDeviceCreateInfo>,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<VkDevice, VkResult> {
        let Some(info) = info else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        if self.instance.is_none() {
            // A device on an instance this context never created. The guest named an instance the
            // object table resolved, so this cannot happen without a host bug -- but it is the
            // guest's ordering that would expose it, so it is rejected rather than asserted.
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }

        let guest_info = *info;
        // The extension list's size was checked against this count as it decoded, so the pair
        // cannot arrive split; a guest that asked for none simply gets an empty list.
        // SAFETY: a `Vk*` struct keeps C's pointers, so this pair is reconciled here rather than
        // by a generated accessor. The decoder allocated the list from the batch arena, sized to
        // this count, and the arena outlives this call.
        let guest = read_names(
            unsafe {
                super::cs::wire_array(
                    guest_info.enabledExtensionCount as usize,
                    guest_info.ppEnabledExtensionNames,
                )
            }
            .unwrap_or_default(),
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
        let r = unsafe { (inst.vkCreateDevice())(pd, &info, ptr(alloc), &mut out) };
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
        &mut self,
        device: VkDevice,
        info: Option<&VkDeviceQueueInfo2>,
    ) -> Option<VkQueue> {
        let d = self.devices.get(&device.0)?;
        let mut out = VkQueue(0);
        // SAFETY: `device` is a handle this table was loaded from and `info` is an arena
        // allocation live for the call.
        unsafe { (d.fns.vkGetDeviceQueue2())(device, ptr(info), &mut out) };
        if out.0 == 0 {
            return None;
        }
        // Asking twice for the same queue is how a guest works, not a mistake: Vulkan hands back
        // the same handle each time, and the answer recorded here is the same both times.
        self.queues.insert(out.0, device.0);
        Some(out)
    }

    /// Destroy a device and forget its entry points.
    ///
    /// Returns the guest ids of everything its pools held. Vulkan destroys a device's pools with
    /// it, and their objects with them, without a command naming any of them -- so the caller
    /// owes the object table their removal, or it goes on resolving ids to handles the driver has
    /// freed.
    /// Destroy one object the guest never named, on a device that is still alive.
    ///
    /// Vulkan does *not* free a device's objects when the device goes: it is undefined to destroy
    /// a device that still owns any. The guest is under no obligation to have tidied up -- it may
    /// send `vkDestroyDevice` with a hundred live fences behind it, and a VM that stops mid-frame
    /// sends nothing at all -- so this is where the tidying happens, and it has to happen before
    /// the device does.
    ///
    /// The match is total on purpose. Six kinds are skipped and each says why; a wildcard arm
    /// would let the next object type Vulkan adds leak silently, which is exactly how the pool
    /// children were lost the first time.
    fn destroy_tracked(fns: &DeviceFns, device: VkDevice, o: &Doomed) {
        use VkObjectType as T;
        let h = o.handle;
        let n = core::ptr::null();
        // SAFETY (all arms): `h` is a handle this context created on `device`, taken out of the
        // object table by this call so it is destroyed exactly once, and the entry point comes
        // from that device's own proc table.
        unsafe {
            match o.ty {
                T::VK_OBJECT_TYPE_SEMAPHORE => {
                    (fns.vkDestroySemaphore())(device, VkSemaphore(h), n)
                }
                T::VK_OBJECT_TYPE_FENCE => (fns.vkDestroyFence())(device, VkFence(h), n),
                T::VK_OBJECT_TYPE_BUFFER => (fns.vkDestroyBuffer())(device, VkBuffer(h), n),
                T::VK_OBJECT_TYPE_IMAGE => (fns.vkDestroyImage())(device, VkImage(h), n),
                T::VK_OBJECT_TYPE_EVENT => (fns.vkDestroyEvent())(device, VkEvent(h), n),
                T::VK_OBJECT_TYPE_QUERY_POOL => {
                    (fns.vkDestroyQueryPool())(device, VkQueryPool(h), n)
                }
                T::VK_OBJECT_TYPE_BUFFER_VIEW => {
                    (fns.vkDestroyBufferView())(device, VkBufferView(h), n)
                }
                T::VK_OBJECT_TYPE_IMAGE_VIEW => {
                    (fns.vkDestroyImageView())(device, VkImageView(h), n)
                }
                T::VK_OBJECT_TYPE_SHADER_MODULE => {
                    (fns.vkDestroyShaderModule())(device, VkShaderModule(h), n)
                }
                T::VK_OBJECT_TYPE_PIPELINE_CACHE => {
                    (fns.vkDestroyPipelineCache())(device, VkPipelineCache(h), n)
                }
                T::VK_OBJECT_TYPE_PIPELINE_LAYOUT => {
                    (fns.vkDestroyPipelineLayout())(device, VkPipelineLayout(h), n)
                }
                T::VK_OBJECT_TYPE_RENDER_PASS => {
                    (fns.vkDestroyRenderPass())(device, VkRenderPass(h), n)
                }
                T::VK_OBJECT_TYPE_PIPELINE => (fns.vkDestroyPipeline())(device, VkPipeline(h), n),
                T::VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT => {
                    (fns.vkDestroyDescriptorSetLayout())(device, VkDescriptorSetLayout(h), n)
                }
                T::VK_OBJECT_TYPE_SAMPLER => (fns.vkDestroySampler())(device, VkSampler(h), n),
                T::VK_OBJECT_TYPE_FRAMEBUFFER => {
                    (fns.vkDestroyFramebuffer())(device, VkFramebuffer(h), n)
                }
                T::VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION => {
                    (fns.vkDestroySamplerYcbcrConversion())(device, VkSamplerYcbcrConversion(h), n)
                }
                T::VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE => (fns
                    .vkDestroyDescriptorUpdateTemplate())(
                    device,
                    VkDescriptorUpdateTemplate(h),
                    n,
                ),
                // Destroying a pool frees everything allocated from it, which is why the two kinds
                // below it are skipped rather than walked.
                T::VK_OBJECT_TYPE_COMMAND_POOL => {
                    (fns.vkDestroyCommandPool())(device, VkCommandPool(h), n)
                }
                T::VK_OBJECT_TYPE_DESCRIPTOR_POOL => {
                    (fns.vkDestroyDescriptorPool())(device, VkDescriptorPool(h), n)
                }
                // Freed with the pool they came from, one line above.
                T::VK_OBJECT_TYPE_COMMAND_BUFFER | T::VK_OBJECT_TYPE_DESCRIPTOR_SET => {}
                // Freed by `free_device_memory`, which the census keeps its own record for and
                // which runs on this same teardown. Freeing it here as well would free it twice.
                T::VK_OBJECT_TYPE_DEVICE_MEMORY => {}
                // A queue is handed out by the device and dies with it; there is no destroy call.
                T::VK_OBJECT_TYPE_QUEUE => {}
                // Not device objects: the instance and its physical devices outlive this, and the
                // device itself is destroyed by the caller once its contents are gone.
                T::VK_OBJECT_TYPE_INSTANCE
                | T::VK_OBJECT_TYPE_PHYSICAL_DEVICE
                | T::VK_OBJECT_TYPE_DEVICE => {}
                // Anything else is an object this renderer never created, so there is no handle
                // here to leak -- but it is also a shape nobody has looked at, so it is logged
                // rather than passed over in silence.
                other => {
                    eprintln!("[virglrs] no destroy for VkObjectType {}, leaking {h:#x}", other.0)
                }
            }
        }
    }

    /// Everything a device owns, torn down in the order Vulkan requires, before the device itself.
    fn empty_device(&mut self, device: VkDevice, doomed: &[Doomed]) {
        let Some(d) = self.devices.get(&device.0) else {
            return;
        };
        // Nothing may be destroyed while the device is still working on it, and the guest is not
        // required to have waited. The C waits here too.
        // SAFETY: a device this context created and has not yet destroyed.
        let r = unsafe { (d.fns.vkDeviceWaitIdle())(device) };
        if r != VkResult::VK_SUCCESS {
            eprintln!("[virglrs] vkDeviceWaitIdle before teardown: VkResult {}", r.0);
        }
        // Filtered here rather than by the caller, so an object can only ever be destroyed on the
        // device the table says it belongs to.
        for o in doomed.iter().filter(|o| o.device == Some(device.0)) {
            Self::destroy_tracked(&d.fns, device, o);
        }
    }

    pub fn destroy_device(&mut self, device: VkDevice, doomed: &[Doomed]) -> Vec<ObjectId> {
        // Its pools go with it: Vulkan destroys them and says nothing, and a handle the driver may
        // now reuse must stop being vouched for at the same moment. Ahead of the lookup, not
        // behind it -- a device this table has already forgotten must still not leave records
        // behind that vouch for its objects.
        // Ahead of everything else, and while the device is still in the map: its objects have to
        // be destroyed before it is, and `empty_device` needs the entry points to do it.
        self.empty_device(device, doomed);
        let orphans = self.pools.close_device(device.0);
        self.queues.retain(|_, owner| *owner != device.0);
        let Some(d) = self.devices.remove(&device.0) else {
            return orphans;
        };
        self.free_device_memory(&d.fns, device.0);
        // SAFETY: a handle this context created, destroyed once -- `remove` is what makes it once.
        unsafe { (d.fns.vkDestroyDevice())(device, core::ptr::null()) };
        orphans
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
        info: Option<&I>,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<u64, VkResult> {
        let Some(d) = self.devices.get(&device.0) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        let mut out = T::from_raw(0);
        // SAFETY: `device` is a handle in this table, `info` and `alloc` are the decoder's arena
        // allocations live for this call, and `out` is a local.
        let r = unsafe { proc(&d.fns)(device, ptr(info), ptr(alloc), &mut out) };
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
        alloc: Option<&VkAllocationCallbacks>,
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
        unsafe { proc(&d.fns)(device, object, ptr(alloc)) };
    }

    /// Allocate a run of objects from a pool: `vkAllocateX(device, info, out)`.
    ///
    /// Unlike an enumeration there is no short answer to handle -- Vulkan fills every element or
    /// none -- so the caller's only two cases are the whole array and nothing.
    ///
    /// `out` is the shadow array the decoder allocated, so the driver writes host handles straight
    /// into the place the generated lifecycle hook will read them from. `ids` is the same run of
    /// objects under the names the guest gave them, which the pool records alongside so that
    /// destroying it can take them out of the object table.
    pub fn allocate_objects<T: Handle, I>(
        &mut self,
        device: VkDevice,
        pool: u64,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, *const I, *mut T) -> VkResult,
        info: Option<&I>,
        out: &mut [T],
        ids: &[ObjectId],
    ) -> Result<(), VkResult> {
        let Some(d) = self.devices.get(&device.0) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        if info.is_none() || out.is_empty() {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        // The pool is re-checked here for the same reason the device is: the guest may have
        // destroyed it, and an id the object table still resolves is not a live driver object.
        if !self.pools.is_open(pool) {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        // SAFETY: `device` is a handle in this table; `info` is an arena allocation live for the
        // call, and `out` is the arena array the decoder sized from the count inside `info`.
        let r = unsafe { proc(&d.fns)(device, ptr(info), out.as_mut_ptr()) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        // Vulkan fills every element of a pool allocation or none, so the whole slice is real.
        // Both names of each object are recorded together; see `Pools`.
        self.pools.adopt(
            pool,
            out.iter().map(|h| h.raw()).zip(ids.iter().copied()).filter(|(h, _)| *h != 0),
        );
        Ok(())
    }

    /// Create a run of pipelines: `vkCreateXPipelines(device, cache, count, infos, alloc, out)`.
    ///
    /// The one create in the protocol that can half-succeed. On a failure Vulkan still writes a
    /// handle for every pipeline it did build, `VK_NULL_HANDLE` for each it did not, and returns
    /// the first error -- so the array can come back part real. The guest is never told which
    /// half: the whole run is refused and every id it named is ghosted, because a partial answer
    /// is one the venus reply has no way to express.
    ///
    /// Which leaves the survivors owned by nobody, so they are destroyed here. The C zeroes the
    /// array and walks away, leaking them until the device goes; there is nothing to be faithful
    /// to in that.
    pub fn create_pipelines<I>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(
            &DeviceFns,
        ) -> unsafe extern "C" fn(
            VkDevice,
            VkPipelineCache,
            u32,
            *const I,
            *const VkAllocationCallbacks,
            *mut VkPipeline,
        ) -> VkResult,
        cache: VkPipelineCache,
        infos: &[I],
        alloc: Option<&VkAllocationCallbacks>,
        out: &mut [VkPipeline],
    ) -> Result<(), VkResult> {
        let Some(d) = self.devices.get(&device.0) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        // One handle comes back per create-info, so the decoder sized both from the same count.
        // A mismatch is this renderer having got it wrong, not the guest -- so it asserts.
        assert_eq!(infos.len(), out.len(), "a pipeline run needs one handle slot per create-info");
        if infos.is_empty() {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        // SAFETY: `device` is a handle in this table, `alloc` is an arena allocation live for the
        // call, and both counts Vulkan is given are the slices' own lengths.
        let r = unsafe {
            proc(&d.fns)(
                device,
                cache,
                infos.len() as u32,
                infos.as_ptr(),
                ptr(alloc),
                out.as_mut_ptr(),
            )
        };
        // A positive result is not a failure: `VK_PIPELINE_COMPILE_REQUIRED` says the driver
        // declined to compile early, and every handle is real.
        if r.0 >= VkResult::VK_SUCCESS.0 {
            return Ok(());
        }
        for survivor in out.iter_mut() {
            if survivor.raw() == 0 {
                continue;
            }
            // SAFETY: a handle this call just produced, destroyed once -- the slice is walked once
            // and the guest never learns the handle, so nothing else can name it.
            unsafe { (d.fns.vkDestroyPipeline())(device, *survivor, ptr(alloc)) };
            // The guest's reply must not carry a handle that is now gone.
            *survivor = VkPipeline(0);
        }
        Err(r)
    }

    /// Whether a pool-allocated object is still live -- its pool undestroyed and it unfreed.
    ///
    /// Free a run of objects back to the pool they came from.
    pub fn free_objects<T: Handle, P: Handle>(
        &mut self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, P, u32, *const T),
        pool: P,
        objects: &[T],
    ) {
        let Some(d) = self.devices.get(&device.0) else {
            return;
        };
        if objects.is_empty() || !self.pools.is_open(pool.raw()) {
            return;
        }
        // SAFETY: handles this context allocated, and the count Vulkan is given is the slice's own
        // length. The generated lifecycle hook removes the ids from the object table exactly once.
        unsafe { proc(&d.fns)(device, pool, objects.len() as u32, objects.as_ptr()) };
        self.pools.release(objects.iter().map(|h| h.raw()));
    }

    /// Register a device with a hand-built proc table, as `create_device` would have.
    ///
    /// Test scaffolding, and the other half of `Device::plant_*`: together they let a test watch
    /// what a handler hands the driver, which is the boundary nothing else in the harness can
    /// see. See `plant_pool` for why the real path is out of reach.
    #[cfg(test)]
    pub(super) fn plant_device(&mut self, handle: u64, fns: DeviceFns) {
        self.devices.insert(handle, DeviceState { fns, memory_types: Vec::new() });
    }

    /// Stand a pool up with contents already in it, as a run of allocations would have left it.
    ///
    /// Test scaffolding. Reaching the real path needs a live device and a driver that answers,
    /// which is the one thing a unit test has no way to arrange -- and what the tests want to ask
    /// about is what happens to those contents afterwards.
    #[cfg(test)]
    pub(super) fn plant_pool(&mut self, device: u64, pool: u64, children: &[(u64, ObjectId)]) {
        self.pools.open(device, pool);
        self.pools.adopt(pool, children.iter().copied());
    }

    /// Point a queue at a device, as `device_queue` would have.
    ///
    /// Test scaffolding. The real path needs a driver that answers `vkGetDeviceQueue2`, and what
    /// the tests want to ask about is what a submit does once the answer is in.
    #[cfg(test)]
    pub(super) fn plant_queue(&mut self, device: u64, queue: u64) {
        self.queues.insert(queue, device);
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
        info: Option<&I>,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<u64, VkResult> {
        let handle = self.create_object(device, proc, info, alloc)?;
        self.pools.open(device.0, handle);
        Ok(handle)
    }

    /// Destroy a pool, and with it everything allocated from it.
    ///
    /// Vulkan frees a pool's objects when the pool goes, without a command per object -- so this
    /// is the only place their handles stop being live. Their guest ids come back for the caller
    /// to take out of the object table, which is what keeps a later command from resolving one
    /// and reaching the driver with a freed handle.
    pub fn destroy_pool<T: Handle>(
        &mut self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, T, *const VkAllocationCallbacks),
        pool: T,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Vec<ObjectId> {
        let orphans = self.pools.close(pool.raw());
        self.destroy_object(device, proc, pool, alloc);
        orphans
    }

    // -------------------------------------------------------------------- recording
    //
    // A `vkCmd*` records into a command buffer and answers nothing: Vulkan defers every error it
    // could report to the submit. So none of these return a result, and what they can still fail
    // at is finding the device -- see `recorder`, which is why they return `Option<()>` rather
    // than nothing at all.
    //
    // Each is written out rather than folded into a closure-taking helper. The entry points differ
    // in arity, not just in name, and the whole reason these live here is that the `unsafe extern`
    // call belongs in the Vulkan binding module: a helper generic enough to cover all of them
    // would take the call itself from the caller, which is the one thing it must not do.

    /// The entry points of the device that owns a command buffer.
    ///
    /// `None` is a command buffer this context has no pool record for. That is a guest naming
    /// one it does not have -- the object table is what usually stops it, and this is the second
    /// answer for the case where the two disagree -- so it is a rejection and never an assert.
    fn recorder(&self, cb: VkCommandBuffer) -> Option<&DeviceFns> {
        self.devices.get(&self.pools.device_of(cb.0)?).map(|d| &d.fns)
    }

    /// `vkBeginCommandBuffer`. The one recording command with a result, because it is the one
    /// that can run the pool out of memory before anything has been recorded.
    pub fn begin_command_buffer(
        &self,
        cb: VkCommandBuffer,
        info: Option<&VkCommandBufferBeginInfo>,
    ) -> Option<VkResult> {
        let d = self.recorder(cb)?;
        // SAFETY: a command buffer this context allocated, and `info` is an arena allocation
        // live for the call. The same holds for every call in this section.
        Some(unsafe { (d.vkBeginCommandBuffer())(cb, ptr(info)) })
    }

    pub fn end_command_buffer(&self, cb: VkCommandBuffer) -> Option<VkResult> {
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        Some(unsafe { (d.vkEndCommandBuffer())(cb) })
    }

    pub fn reset_command_buffer(
        &self,
        cb: VkCommandBuffer,
        flags: VkCommandBufferResetFlags,
    ) -> Option<VkResult> {
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        Some(unsafe { (d.vkResetCommandBuffer())(cb, flags) })
    }

    /// `vkCmdPipelineBarrier`, whose three arrays are independent of each other -- a barrier may
    /// name memory, buffers, images, or any mix of them.
    // Vulkan's own signature, one argument per parameter. Folding the three count-and-array pairs
    // into a struct would be a second definition of what vk.xml already says.
    #[allow(clippy::too_many_arguments)]
    pub fn cmd_pipeline_barrier(
        &self,
        cb: VkCommandBuffer,
        src: VkPipelineStageFlags,
        dst: VkPipelineStageFlags,
        dependency: VkDependencyFlags,
        memory: &[VkMemoryBarrier],
        buffers: &[VkBufferMemoryBarrier],
        images: &[VkImageMemoryBarrier],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; every count is its own slice's length.
        unsafe {
            (d.vkCmdPipelineBarrier())(
                cb,
                src,
                dst,
                dependency,
                memory.len() as u32,
                memory.as_ptr(),
                buffers.len() as u32,
                buffers.as_ptr(),
                images.len() as u32,
                images.as_ptr(),
            )
        };
        Some(())
    }

    pub fn cmd_begin_render_pass(
        &self,
        cb: VkCommandBuffer,
        begin: Option<&VkRenderPassBeginInfo>,
        contents: VkSubpassContents,
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        unsafe { (d.vkCmdBeginRenderPass())(cb, ptr(begin), contents) };
        Some(())
    }

    pub fn cmd_end_render_pass(&self, cb: VkCommandBuffer) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        unsafe { (d.vkCmdEndRenderPass())(cb) };
        Some(())
    }

    pub fn cmd_bind_pipeline(
        &self,
        cb: VkCommandBuffer,
        bind_point: VkPipelineBindPoint,
        pipeline: VkPipeline,
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        unsafe { (d.vkCmdBindPipeline())(cb, bind_point, pipeline) };
        Some(())
    }

    pub fn cmd_bind_descriptor_sets(
        &self,
        cb: VkCommandBuffer,
        bind_point: VkPipelineBindPoint,
        layout: VkPipelineLayout,
        first_set: u32,
        sets: &[VkDescriptorSet],
        dynamic_offsets: &[u32],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; both counts are their own slice's length.
        unsafe {
            (d.vkCmdBindDescriptorSets())(
                cb,
                bind_point,
                layout,
                first_set,
                sets.len() as u32,
                sets.as_ptr(),
                dynamic_offsets.len() as u32,
                dynamic_offsets.as_ptr(),
            )
        };
        Some(())
    }

    pub fn cmd_draw(
        &self,
        cb: VkCommandBuffer,
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        unsafe { (d.vkCmdDraw())(cb, vertices, instances, first_vertex, first_instance) };
        Some(())
    }

    pub fn cmd_set_viewport(
        &self,
        cb: VkCommandBuffer,
        first: u32,
        viewports: &[VkViewport],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the slice's own length.
        unsafe { (d.vkCmdSetViewport())(cb, first, viewports.len() as u32, viewports.as_ptr()) };
        Some(())
    }

    pub fn cmd_set_scissor(
        &self,
        cb: VkCommandBuffer,
        first: u32,
        scissors: &[VkRect2D],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the slice's own length.
        unsafe { (d.vkCmdSetScissor())(cb, first, scissors.len() as u32, scissors.as_ptr()) };
        Some(())
    }

    /// `vkCmdBindVertexBuffers`, whose one count governs two arrays.
    ///
    /// That they are the same length is a host invariant, not a guest one: both come from the
    /// same generated accessor, which lengths them from the same member. A guest cannot make them
    /// differ, so a difference here would be ours.
    pub fn cmd_bind_vertex_buffers(
        &self,
        cb: VkCommandBuffer,
        first: u32,
        buffers: &[VkBuffer],
        offsets: &[VkDeviceSize],
    ) -> Option<()> {
        assert_eq!(buffers.len(), offsets.len(), "one count governs both arrays");
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the length both slices share.
        unsafe {
            (d.vkCmdBindVertexBuffers())(
                cb,
                first,
                buffers.len() as u32,
                buffers.as_ptr(),
                offsets.as_ptr(),
            )
        };
        Some(())
    }

    pub fn cmd_fill_buffer(
        &self,
        cb: VkCommandBuffer,
        buffer: VkBuffer,
        offset: VkDeviceSize,
        size: VkDeviceSize,
        data: u32,
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        unsafe { (d.vkCmdFillBuffer())(cb, buffer, offset, size, data) };
        Some(())
    }

    pub fn cmd_copy_buffer(
        &self,
        cb: VkCommandBuffer,
        src: VkBuffer,
        dst: VkBuffer,
        regions: &[VkBufferCopy],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the slice's own length.
        unsafe { (d.vkCmdCopyBuffer())(cb, src, dst, regions.len() as u32, regions.as_ptr()) };
        Some(())
    }

    pub fn cmd_copy_buffer_to_image(
        &self,
        cb: VkCommandBuffer,
        src: VkBuffer,
        dst: VkImage,
        layout: VkImageLayout,
        regions: &[VkBufferImageCopy],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the slice's own length.
        unsafe {
            (d.vkCmdCopyBufferToImage())(
                cb,
                src,
                dst,
                layout,
                regions.len() as u32,
                regions.as_ptr(),
            )
        };
        Some(())
    }

    // ----------------------------------------------------------------------- sync
    //
    // Submitting work and waiting for it. `vkWaitForFences` blocks the caller for as long as the
    // guest asked -- up to forever -- and is passed through with the guest's timeout intact: a
    // clamp would answer `VK_TIMEOUT` for a fence that had not timed out, which is a lie the guest
    // cannot tell from the truth. What keeps that honest is that this is the guest's own thread's
    // work, not ours to finish early.

    /// The entry points of the device a queue belongs to.
    ///
    /// `None` is a queue this context never retrieved -- a guest naming one it does not have. The
    /// object table is what usually stops that; this is the second answer for when the two
    /// disagree, so it is a rejection and never an assert. See [`Driver::recorder`].
    fn submitter(&self, queue: VkQueue) -> Option<&DeviceFns> {
        self.devices.get(self.queues.get(&queue.0)?).map(|d| &d.fns)
    }

    /// `vkQueueSubmit`. Every handle inside a `VkSubmitInfo` -- the wait and signal semaphores,
    /// the command buffers -- was resolved by the decoder as it read them, so what arrives here is
    /// already the driver's own.
    pub fn queue_submit(
        &self,
        queue: VkQueue,
        submits: &[VkSubmitInfo],
        fence: VkFence,
    ) -> Option<VkResult> {
        let d = self.submitter(queue)?;
        // Submitting no work to signal a fence is a normal thing for a guest to do, and Vulkan
        // takes a null array for it -- so the slice's own pointer is passed either way.
        // SAFETY: a queue this context retrieved, and `submits` is an arena allocation live for
        // the call whose count is its own length.
        Some(unsafe { (d.vkQueueSubmit())(queue, submits.len() as u32, submits.as_ptr(), fence) })
    }

    /// `vkResetFences`.
    pub fn reset_fences(&self, device: VkDevice, fences: &[VkFence]) -> VkResult {
        let Some(d) = self.devices.get(&device.0) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // SAFETY: `device` is a handle in this table, and the count Vulkan wants is the slice's
        // own length. The same holds for the wait below.
        unsafe { (d.fns.vkResetFences())(device, fences.len() as u32, fences.as_ptr()) }
    }

    /// `vkWaitForFences`. Blocks for up to `timeout` nanoseconds, as the guest asked.
    pub fn wait_for_fences(
        &self,
        device: VkDevice,
        fences: &[VkFence],
        wait_all: VkBool32,
        timeout: u64,
    ) -> VkResult {
        let Some(d) = self.devices.get(&device.0) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // SAFETY: as above.
        unsafe {
            (d.fns.vkWaitForFences())(
                device,
                fences.len() as u32,
                fences.as_ptr(),
                wait_all,
                timeout,
            )
        }
    }

    /// `vkWaitSemaphoreResourceMESA`: export the semaphore's payload to a sync fd, then drop it.
    ///
    /// The export is the entire operation. Exporting a `SYNC_FD` payload is what resolves the
    /// semaphore's pending signal into something outside Vulkan, and the guest asked for that and
    /// nothing else -- so the descriptor is closed the moment it exists, as the C does.
    pub fn export_semaphore_sync_fd(
        &self,
        device: VkDevice,
        semaphore: VkSemaphore,
    ) -> Result<VkResult, NoSyncFd> {
        let d = self.sync_fd_device(device, |f| f.has_vkGetSemaphoreFdKHR())?;
        let info = VkSemaphoreGetFdInfoKHR {
            sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            pNext: core::ptr::null(),
            semaphore,
            handleType:
                VkExternalSemaphoreHandleTypeFlagBits::VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        };
        // KosmicKrisp answers `VK_SUCCESS` with no descriptor at all -- measured over the venus
        // corpus, 71568 exports and not one non-negative fd. Handled rather than asserted: what
        // the guest asked for is that the payload move, and a driver is free to have moved it
        // somewhere that is not a file.
        let mut fd: core::ffi::c_int = -1;
        // SAFETY: `device` is a handle in this table; `info` and `fd` are ours and outlive the
        // call.
        let r = unsafe { (d.vkGetSemaphoreFdKHR())(device, &info, &mut fd) };
        if fd >= 0 {
            // SAFETY: a descriptor this call just produced and nothing else holds. Closing it is
            // the whole point -- the payload has already moved.
            unsafe { libc::close(fd) };
        }
        Ok(r)
    }

    /// `vkImportSemaphoreResourceMESA` for resource id 0: import an already-signaled payload.
    ///
    /// A `SYNC_FD` import of `-1` is Vulkan's spelling of "this semaphore is signaled now", which
    /// is what a guest's window system uses to hand itself an image it never waited for. The
    /// import is temporary, so it is undone by the first wait that consumes it.
    pub fn import_signaled_semaphore(
        &self,
        device: VkDevice,
        semaphore: VkSemaphore,
    ) -> Result<VkResult, NoSyncFd> {
        let d = self.sync_fd_device(device, |f| f.has_vkImportSemaphoreFdKHR())?;
        let info = VkImportSemaphoreFdInfoKHR {
            sType: VkStructureType::VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
            pNext: core::ptr::null(),
            semaphore,
            flags: VkFlags(VkSemaphoreImportFlagBits::VK_SEMAPHORE_IMPORT_TEMPORARY_BIT.0 as u32),
            handleType:
                VkExternalSemaphoreHandleTypeFlagBits::VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
            fd: -1,
        };
        // SAFETY: `device` is a handle in this table and `info` is ours for the call.
        Ok(unsafe { (d.vkImportSemaphoreFdKHR())(device, &info) })
    }

    /// The entry points for a sync-fd operation, or why there are none.
    ///
    /// Both callers reach the driver through an extension the guest asked for by name, so the
    /// proc table's own `.expect()` is not the right failure: a driver that does not export the
    /// one being asked for is a host we cannot serve on, not a host invariant we broke. Each
    /// caller names the entry point it is about to use and no other -- a driver that exports half
    /// of `VK_KHR_external_semaphore_fd` can still serve the half it has.
    fn sync_fd_device(
        &self,
        device: VkDevice,
        exports: impl FnOnce(&DeviceFns) -> bool,
    ) -> Result<&DeviceFns, NoSyncFd> {
        let d = self.devices.get(&device.0).ok_or(NoSyncFd::NoDevice)?;
        if !exports(&d.fns) {
            return Err(NoSyncFd::Unsupported);
        }
        Ok(&d.fns)
    }

    // ------------------------------------------------------------ binding and updating
    //
    // Commands that change an existing object rather than create one. They register nothing and
    // return no handle, so nothing here touches the object table -- but a submit that draws from
    // a buffer with no memory bound, or through a descriptor set that was never written, is
    // undefined behaviour just as surely as one naming a handle that does not exist. Serving
    // these is what makes a later `vkQueueSubmit` safe to pass through.

    /// Bind memory to a run of buffers or images: `vkBindXMemory2`.
    ///
    /// Both `vkBindBufferMemory2` and `vkBindImageMemory2` have exactly this shape, which is why
    /// the entry point arrives as a closure -- the info type is the only thing that differs, and
    /// it is a type parameter.
    pub fn bind_memory<I>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, u32, *const I) -> VkResult,
        infos: &[I],
    ) -> VkResult {
        let Some(d) = self.devices.get(&device.0) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // Binding nothing is legal and the guest sends it.
        if infos.is_empty() {
            return VkResult::VK_SUCCESS;
        }
        // SAFETY: `device` is a handle in this table, and the count and array Vulkan wants are
        // the slice's own -- which is the whole reason the pair is not carried this far.
        unsafe { proc(&d.fns)(device, infos.len() as u32, infos.as_ptr()) }
    }

    /// Write and copy descriptors: `vkUpdateDescriptorSets`.
    ///
    /// Alone among the commands here it returns nothing -- Vulkan gives it no failure to report,
    /// because everything it could refuse is a validation error the guest was required not to
    /// commit. So a device this table does not have is a silent no-op, the same as a destroy.
    pub fn update_descriptor_sets(
        &self,
        device: VkDevice,
        writes: &[VkWriteDescriptorSet],
        copies: &[VkCopyDescriptorSet],
    ) {
        let Some(d) = self.devices.get(&device.0) else {
            return;
        };
        if writes.is_empty() && copies.is_empty() {
            return;
        }
        // SAFETY: `device` is a handle in this table, and each count is its own slice's length.
        unsafe {
            (d.fns.vkUpdateDescriptorSets())(
                device,
                writes.len() as u32,
                writes.as_ptr(),
                copies.len() as u32,
                copies.as_ptr(),
            )
        };
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
        info: Option<&VkMemoryAllocateInfo>,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<VkDeviceMemory, VkResult> {
        let Some(info) = info else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        let Some(d) = self.devices.get(&device.0) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        // A copy, not an edit in place: the decoder's struct is the guest's request, and the
        // round trip re-encodes it. The `pNext` chain is carried over untouched.
        let mut info = *info;
        info.allocationSize = VkDeviceSize(pad_for_blob(
            info.allocationSize.0,
            d.memory_types.get(info.memoryTypeIndex as usize).copied(),
            imports_a_resource(info.pNext),
        ));

        let mut out = VkDeviceMemory(0);
        // SAFETY: `info` is a local whose chain the decoder owns for the batch, `alloc` is another
        // of its arena allocations, and `out` is a local.
        let r = unsafe { (d.fns.vkAllocateMemory())(device, &info, ptr(alloc), &mut out) };
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

/// Why a sync-fd operation was not attempted. Neither is a host invariant: one is a guest naming
/// a device it does not have, the other a driver this build cannot do venus sync on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NoSyncFd {
    /// No device behind the handle.
    NoDevice,
    /// The driver exports neither half of `VK_KHR_external_semaphore_fd`. There is nothing to
    /// substitute: a semaphore whose payload cannot be moved is one the guest's next submit waits
    /// on forever, so saying so is better than pretending it worked.
    Unsupported,
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
fn read_names(names: &[*const std::ffi::c_char]) -> Vec<String> {
    names
        .iter()
        .filter(|p| !p.is_null())
        // SAFETY: each is a pointer to a NUL-terminated string the decoder wrote into the arena,
        // which outlives this call.
        .map(|p| unsafe { std::ffi::CStr::from_ptr(*p) }.to_string_lossy().into())
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
    /// so the guest ids of its contents have to come back here, for the caller to take out of
    /// the object table. Left there, the table goes on resolving one to a handle Vulkan has
    /// freed and the next command naming it hands that handle back to the driver.
    #[test]
    fn a_destroyed_pool_hands_back_the_ids_of_everything_in_it() {
        const DEVICE: u64 = 3;

        let mut d = Driver::default();
        d.pools.open(DEVICE, 7);
        d.pools.adopt(7, [(11, ObjectId(110)), (12, ObjectId(120))]);

        // No device is registered, so the driver call itself is skipped -- the bookkeeping is
        // what is under test, and it has to happen either way.
        let mut orphans =
            d.destroy_pool(VkDevice(DEVICE), |f| f.vkDestroyCommandPool(), VkCommandPool(7), None);
        orphans.sort_unstable_by_key(|i| i.0);
        assert_eq!(orphans, [ObjectId(110), ObjectId(120)], "every id in the pool, and no other");
        assert!(!d.pools.is_open(7));

        // And a second destroy of the same pool has nothing left to hand back: the ids must not
        // be removed from the object table twice, because the guest may have reused them.
        assert!(
            d.destroy_pool(VkDevice(DEVICE), |f| f.vkDestroyCommandPool(), VkCommandPool(7), None)
                .is_empty()
        );
    }

    /// A destroyed device takes its pools with it.
    ///
    /// Vulkan destroys a device's pools for it and says nothing, and the guest sends no command
    /// per pool -- so the ids of every command buffer in them have to come back from here too.
    /// `vkCmd*` carries only a command buffer and has no device to re-check: the object table
    /// no longer knowing the id is the only thing standing between a recycled handle and the
    /// driver.
    #[test]
    fn a_device_takes_its_pools_and_their_ids_with_it() {
        let mut d = Driver::default();
        d.pools.open(3, 7);
        d.pools.adopt(7, [(11, ObjectId(110)), (12, ObjectId(120))]);
        d.pools.open(4, 8);
        d.pools.adopt(8, [(21, ObjectId(210))]);

        let mut orphans = d.destroy_device(VkDevice(3), &[]);
        orphans.sort_unstable_by_key(|i| i.0);

        assert_eq!(orphans, [ObjectId(110), ObjectId(120)], "its pools' ids, and no others");
        assert!(!d.pools.is_open(7), "a pool outlived its device");
        assert!(d.pools.is_open(8), "another device's pool must be untouched");
        assert!(
            d.destroy_device(VkDevice(4), &[]) == [ObjectId(210)],
            "another device's pool must still hold its own"
        );
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
