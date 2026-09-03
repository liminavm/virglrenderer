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

use super::budget::{Account, Charge};
use super::cs::{Handle, ObjectId, PoolOf, TypedHandle};
use super::objects::Doomed;
use super::proto::types::{
    VkAllocationCallbacks, VkBaseInStructure, VkBaseOutStructure, VkBool32, VkBuffer, VkBufferCopy,
    VkBufferImageCopy, VkBufferMemoryBarrier, VkBufferView, VkClearAttachment, VkClearColorValue,
    VkClearRect, VkCommandBuffer, VkCommandBufferBeginInfo, VkCommandBufferResetFlags,
    VkCommandPool, VkCopyDescriptorSet, VkDependencyFlags, VkDescriptorPool, VkDescriptorSet,
    VkDescriptorSetLayout, VkDescriptorUpdateTemplate, VkDevice, VkDeviceCreateInfo,
    VkDeviceMemory, VkDeviceQueueInfo2, VkDeviceSize, VkEvent, VkExtensionProperties,
    VkExternalMemoryHandleTypeFlagBits, VkExternalSemaphoreHandleTypeFlagBits, VkFence, VkFilter,
    VkFormat, VkFramebuffer, VkImage, VkImageAspectFlagBits, VkImageAspectFlags, VkImageBlit,
    VkImageCreateFlags, VkImageCreateInfo, VkImageFormatProperties, VkImageLayout,
    VkImageMemoryBarrier, VkImageSubresource, VkImageSubresourceRange, VkImageTiling, VkImageType,
    VkImageUsageFlags, VkImageView, VkImportMemoryHostPointerInfoEXT,
    VkImportMemoryResourceInfoMESA, VkImportSemaphoreFdInfoKHR, VkInstance, VkInstanceCreateInfo,
    VkMemoryAllocateInfo, VkMemoryBarrier, VkMemoryDedicatedAllocateInfo, VkMemoryMapFlags,
    VkMemoryPropertyFlagBits, VkMemoryPropertyFlags, VkMemoryResourceAllocationSizePropertiesMESA,
    VkObjectType, VkPhysicalDevice, VkPhysicalDeviceMemoryProperties, VkPipeline,
    VkPipelineBindPoint, VkPipelineCache, VkPipelineLayout, VkPipelineStageFlags, VkQueryPool,
    VkQueue, VkRect2D, VkRenderPass, VkRenderPassBeginInfo, VkResult, VkRingMonitorInfoMESA,
    VkSampleCountFlagBits, VkSampler, VkSamplerYcbcrConversion, VkSemaphore,
    VkSemaphoreGetFdInfoKHR, VkSemaphoreImportFlagBits, VkShaderModule, VkShaderStageFlags,
    VkStructureType, VkSubmitInfo, VkSubpassContents, VkSubresourceLayout, VkViewport,
    VkWriteDescriptorSet,
};
use super::ring::ResourceBytes;
use crate::ids::ResourceHandle;
use crate::ids::SurfaceId;
use crate::metal::{PixelFormat, Surface};
use crate::vulkan::{self, Device as DeviceFns, Global, Instance as InstanceFns};

/// The memory properties this renderer decides anything by: whether the host can address it at
/// all, and whether its own caching of it is write-back -- which is what a guest needs to know to
/// map it.
const HOST_VISIBLE_BIT: u32 =
    VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.0 as u32;
const HOST_COHERENT_BIT: u32 =
    VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT.0 as u32;
const HOST_CACHED_BIT: u32 = VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_CACHED_BIT.0 as u32;

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
    open: BTreeMap<TypedHandle, Pool>,
    /// Every pool-allocated object, by host handle, pointing back at its pool.
    owner: BTreeMap<TypedHandle, TypedHandle>,
}

/// One live pool: the device that owns it, and what has been allocated from it.
struct Pool {
    /// Recorded so a destroyed device can take its pools with it. Vulkan destroys them for us and
    /// says nothing, and a host handle the driver is free to reuse must stop being vouched for the
    /// moment that happens.
    device: VkDevice,
    /// Host handle to the guest id it was allocated under.
    children: BTreeMap<TypedHandle, ObjectId>,
}

impl Pools {
    fn open<P: PoolOf>(&mut self, device: VkDevice, pool: P) {
        self.open.insert(TypedHandle::of(pool), Pool { device, children: BTreeMap::new() });
    }

    fn is_open<P: PoolOf>(&self, pool: P) -> bool {
        self.open.contains_key(&TypedHandle::of(pool))
    }

    /// The device that owns the pool a handle came from.
    ///
    /// A `vkCmd*` carries only its command buffer: Vulkan does not repeat the device, because a
    /// command buffer already knows its own. Here it does not, so the pool is the way back.
    fn device_of<T: Handle>(&self, handle: T) -> Option<VkDevice> {
        self.open.get(self.owner.get(&TypedHandle::of(handle))?).map(|p| p.device)
    }

    /// Record objects freshly allocated from a pool. Both directions, or neither.
    fn adopt<P: PoolOf>(
        &mut self,
        pool: P,
        children: impl IntoIterator<Item = (P::Child, ObjectId)>,
    ) {
        let pool = TypedHandle::of(pool);
        let Some(p) = self.open.get_mut(&pool) else {
            return;
        };
        for (handle, id) in children {
            p.children.insert(TypedHandle::of(handle), id);
            self.owner.insert(TypedHandle::of(handle), pool);
        }
    }

    /// Forget objects freed back to their pool. The pool each belongs to is looked up rather than
    /// passed in, so a caller cannot name the wrong one.
    fn release<T: Handle>(&mut self, children: impl IntoIterator<Item = T>) {
        for child in children {
            let child = TypedHandle::of(child);
            if let Some(pool) = self.owner.remove(&child)
                && let Some(p) = self.open.get_mut(&pool)
            {
                p.children.remove(&child);
            }
        }
    }

    /// Forget a pool and everything in it, handing back the guest ids that just stopped naming
    /// anything -- the caller owes the object table their removal.
    fn close<P: Handle>(&mut self, pool: P) -> Vec<ObjectId> {
        let children =
            self.open.remove(&TypedHandle::of(pool)).map(|p| p.children).unwrap_or_default();
        for handle in children.keys() {
            self.owner.remove(handle);
        }
        children.into_values().collect()
    }

    /// Forget every pool a device owned, because destroying the device destroyed them.
    fn close_device(&mut self, device: VkDevice) -> Vec<ObjectId> {
        let doomed: Vec<TypedHandle> =
            self.open.iter().filter(|(_, p)| p.device == device).map(|(h, _)| *h).collect();
        // Already a `TypedHandle`, so it goes to the map directly rather than back through
        // `close`, which exists to build one from a caller's typed handle.
        doomed
            .into_iter()
            .flat_map(|pool| {
                let children = self.open.remove(&pool).map(|p| p.children).unwrap_or_default();
                for handle in children.keys() {
                    self.owner.remove(handle);
                }
                children.into_values()
            })
            .collect()
    }
}

/// Why no memory was allocated.
///
/// Two refusals that are not the same thing. A driver's is the guest's own affair -- it asked for
/// memory the host does not have, and unwinding from that is something it does on hardware too. A
/// budget refusal is this renderer declining to serve a request it could have served, which the
/// guest is given no way to find out about (see [`crate::venus::budget`]) and so cannot recover
/// from; the context stops instead, at the command that caused it rather than several later.
#[derive(Debug)]
pub enum NoMemory {
    Driver(VkResult),
    OverBudget { stop: bool },
}

impl NoMemory {
    /// What the guest is told, on the chance that it is one of the rare configurations that reads
    /// the answer back.
    pub fn ret(&self) -> VkResult {
        match self {
            NoMemory::Driver(r) => *r,
            NoMemory::OverBudget { .. } => VkResult::VK_ERROR_OUT_OF_DEVICE_MEMORY,
        }
    }
}

/// The driver objects one context has stood up.
pub struct Driver {
    /// This context's key to the host memory ledger -- see [`crate::venus::budget`]. Held here
    /// rather than passed to the calls that allocate, because the record of what was allocated
    /// lives here too, and a charge is credited by that record going away.
    account: Account,
    instance: Option<InstanceFns>,
    /// The instance's own handle, so a teardown with no command behind it can still destroy it.
    instance_handle: VkInstance,
    /// Keyed by *host* handle, not guest id. Every handler that needs a device's entry points
    /// reaches them through the `VkDevice` the lookup already resolved for it; only a destroy
    /// carries the guest id, and it does not need the table to find one.
    devices: BTreeMap<VkDevice, DeviceState>,
    /// What each physical device supports, by name, keyed by host handle.
    ///
    /// A set of names rather than the C's one bool per extension: the question asked of it is
    /// always "does this driver have <name>", and a hand-maintained struct of booleans is a list
    /// that has to be extended every time a new name matters.
    physical_device_exts: BTreeMap<VkPhysicalDevice, BTreeSet<String>>,
    /// How big each live allocation is, by the guest's id.
    ///
    /// Only the size. The handle and the owning device are the object table's to know, and this
    /// map holding its own copy of them was a second answer to "where does this allocation live"
    /// -- which is why `vkFreeMemory` had to be skipped in the destroy cascade to avoid freeing
    /// twice. The size is a fact nothing else has: the driver may round an allocation up, and the
    /// census reports the number the guest asked for, because that is what the guest reads back.
    memory: BTreeMap<ObjectId, Allocated>,
    /// What each live image was created as, by host handle.
    ///
    /// Only the facts Vulkan will not give back: a scanout surface has to be minted at the
    /// image's own width, height and format, and nothing can be asked for those after the create.
    /// The row pitch is *not* here -- that is queried from the live image, because the driver
    /// decides it and the driver is the only honest source.
    ///
    /// Keyed by host handle because that is what a dedicated allocation names, and the object
    /// table has no handle-to-id direction to reach an id through. A handle Vulkan has recycled
    /// could therefore find its predecessor's record; what stops that mattering is that the
    /// record is only ever used beside a live layout query, and the two disagreeing is what
    /// [`Driver::scanout_surface`] refuses on. A stale record cannot produce a wrong surface, only
    /// no surface.
    images: BTreeMap<VkImage, ImageFacts>,
    /// Live command and descriptor pools, and what was allocated from each. See [`Pools`].
    pools: Pools,
    /// The device each queue belongs to, by host handle.
    ///
    /// A queue is never created and never destroyed -- `vkGetDeviceQueue2` hands back one the
    /// device already owns -- but `vkQueueSubmit` carries only the queue, so this is the way back
    /// to the entry points. The same problem [`Pools::device_of`] solves for a command buffer, and
    /// kept apart from it because a queue owns nothing and takes nothing with it when it goes.
    queues: BTreeMap<VkQueue, VkDevice>,
}

/// The last stand: a driver may not be dropped while it still owes Vulkan a destroy.
///
/// Every one of these maps is a host handle nothing else in the process names, so a `Driver` that
/// reaches its destructor with any of them occupied has leaked whatever is in them -- and it can
/// only have got there through host code, never through anything a guest sent, which is what makes
/// it an assert rather than a rejection (CLAUDE.md). In normal operation it cannot fire: the
/// `Context` that owns this tears it down on its own drop, which runs first.
impl Drop for Driver {
    fn drop(&mut self) {
        // Not while something else is already being reported. A test that fails with planted
        // state still in the maps drops this on the way out, and asserting there replaces the
        // assertion that actually failed with this one -- and, because the crate aborts rather
        // than unwinds, takes every test after it down unrun. The invariant is about normal
        // operation, and during a panic there is no normal operation left to protect.
        if std::thread::panicking() {
            return;
        }
        assert!(
            self.owes_nothing(),
            "a Driver was dropped still holding host handles: {} device(s), {} allocation(s), \
             {} pool(s), {} queue(s), instance {}",
            self.devices.len(),
            self.memory.len(),
            self.pools.open.len(),
            self.queues.len(),
            if self.instance.is_some() { "live" } else { "gone" },
        );
    }
}

/// One live `VkDevice`: its entry points, and what its allocations need to know.
struct DeviceState {
    fns: DeviceFns,
    /// The property flags of each memory type, indexed by `memoryTypeIndex`. Read once at device
    /// creation because it never changes, and because an allocation must not pay an instance
    /// round trip to learn whether it is host-visible.
    memory_types: Vec<VkMemoryPropertyFlags>,
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

/// One extension name and version as Vulkan's own struct, or `None` for a name this build's
/// vk.xml does not know.
///
/// A version of zero is that "does not know": there is no honest properties entry to hand back for
/// an extension whose spec version we cannot state, and inventing one would advertise it.
fn extension_properties(name: &str) -> Option<VkExtensionProperties> {
    let version = crate::venus::proto::info::spec_version(name);
    if version == 0 {
        return None;
    }
    let mut out = VkExtensionProperties { specVersion: version, ..Default::default() };
    // The callers' names are either this build's own string literals or read back out of a
    // 256-byte array by `learn_extensions`, so a name too long to fit cannot reach here.
    assert!(name.len() < out.extensionName.len(), "extension name fits Vulkan's array");
    for (slot, b) in out.extensionName.iter_mut().zip(name.bytes()) {
        *slot = b as core::ffi::c_char;
    }
    Some(out)
}

/// The instance extensions this renderer speaks, which is not what the host loader has.
///
/// `vkEnumerateInstanceExtensionProperties` is the venus handshake rather than a Vulkan query: the
/// guest is asking what the *renderer* understands on the wire, and the answer is the two protocol
/// extensions this build serializes -- the same pair the capset states. Forwarding the host's list
/// would answer a question the guest did not ask, and one it cannot use: it never talks to that
/// loader.
pub fn renderer_extensions() -> Vec<VkExtensionProperties> {
    ["VK_EXT_command_serialization", "VK_MESA_venus_protocol"]
        .into_iter()
        .filter_map(extension_properties)
        .collect()
}

/// An enumeration's out-array as the (count, pointer) pair Vulkan wants, and the room to check the
/// answer against.
///
/// The pair is rebuilt here and nowhere else, which is the point (CLAUDE.md): above this line the
/// array is one slice, and its length is both what the driver is told it has room for and what the
/// count it writes back is measured against. `None` is the guest's count query -- no array and no
/// room, and no bound either: answering it *is* writing a count larger than the zero it was given.
fn split<T>(out: Option<&mut [T]>) -> (u32, Option<usize>, *mut T) {
    match out {
        Some(s) => (s.len() as u32, Some(s.len()), s.as_mut_ptr()),
        None => (0, None, core::ptr::null_mut()),
    }
}

/// The count a driver wrote back, against the room it was given.
///
/// A host invariant, so it asserts (CLAUDE.md): the guest cannot arrange this, and a driver that
/// claims to have filled more than it was handed would have the reply encoder walk off the end of
/// the arena the array lives in. Only the fill call has a bound -- see [`split`].
fn fits(n: u32, room: Option<usize>) {
    if let Some(room) = room {
        assert!(n as usize <= room, "the driver enumerated more than the room it was given");
    }
}

impl Driver {
    pub fn new(account: Account) -> Driver {
        Driver {
            account,
            instance: None,
            instance_handle: VkInstance(0),
            devices: BTreeMap::new(),
            physical_device_exts: BTreeMap::new(),
            memory: BTreeMap::new(),
            images: BTreeMap::new(),
            pools: Pools::default(),
            queues: BTreeMap::new(),
        }
    }

    /// The instance table, or None when this context has not created an instance.
    ///
    /// A guest command that names an instance cannot reach a handler without the object table
    /// having resolved that instance first, so in practice a handler that needs this has one --
    /// but the guest chooses the order, so it is an `Option` and never an assert.
    /// Whether every host handle this ever held has been given back.
    ///
    /// The condition the bomb above checks, named once so the witness that a teardown really
    /// empties these maps asserts the same thing the drop does rather than a restatement of it.
    pub(super) fn owes_nothing(&self) -> bool {
        self.instance.is_none()
            && self.devices.is_empty()
            && self.memory.is_empty()
            && self.pools.open.is_empty()
            && self.pools.owner.is_empty()
            && self.queues.is_empty()
    }

    pub fn instance(&self) -> Option<&InstanceFns> {
        self.instance.as_ref()
    }

    pub fn device(&self, device: VkDevice) -> Option<&DeviceFns> {
        self.devices.get(&device).map(|d| &d.fns)
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
            self.empty_device(handle, doomed);
        }
        for (handle, d) in core::mem::take(&mut self.devices) {
            self.pools.close_device(handle);
            self.queues.retain(|_, owner| *owner != handle);
            // SAFETY: a handle this context created, and the table was loaded from it.
            unsafe { (d.fns.vkDestroyDevice())(handle, core::ptr::null()) };
        }
        // A fallback, not the path that retires the census: `empty_device` does that, per device,
        // as it frees. What can be left here is an allocation the table could name no device for,
        // which was never freed above because there was no device to free it on. Nothing a guest
        // sends reaches this today -- an allocate is refused before it is recorded unless its
        // device resolved -- so it is said out loud and the record dropped, rather than left to
        // abort a teardown that is already unwinding.
        if !self.memory.is_empty() {
            eprintln!(
                "[virglrs] teardown: {} allocation(s) with no device to free them on",
                self.memory.len()
            );
            self.memory.clear();
        }
        if let Some(inst) = self.instance.take() {
            let handle = core::mem::take(&mut self.instance_handle);
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
        info: &VkInstanceCreateInfo,
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
        let r = unsafe { (global.vkCreateInstance())(info, ptr(alloc), &mut out) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        assert!(out.0 != 0, "vkCreateInstance succeeded and returned a null instance");
        self.instance = Some(vulkan::instance(out));
        self.instance_handle = out;
        Ok(out)
    }

    /// Record what a physical device supports, so device creation can be filtered against it.
    ///
    /// Asked once per physical device that answers, when the guest first enumerates them -- the
    /// answer does not change for the life of the instance.
    ///
    /// A driver that refuses the question is reported rather than recorded as an empty answer:
    /// nothing downstream can tell "supports no extensions" from "was never successfully asked",
    /// and the second one silently strips from `vkCreateDevice` every extension the guest asked
    /// for and the hardware has.
    pub fn learn_extensions(&mut self, pd: VkPhysicalDevice) -> Result<(), VkResult> {
        if self.physical_device_exts.contains_key(&pd) {
            return Ok(());
        }
        let Some(inst) = self.instance.as_ref() else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
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
            return Err(r);
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
            return Err(r);
        }
        props.truncate(n as usize);
        let names = props
            .iter()
            .map(|p| {
                p.extensionName.iter().take_while(|c| **c != 0).map(|c| *c as u8 as char).collect()
            })
            .collect();
        self.physical_device_exts.insert(pd, names);
        Ok(())
    }

    /// What the guest is told a physical device supports.
    ///
    /// Deliberately not what the driver supports, and derived from that list rather than stored
    /// beside it -- the two are different facts and only one of them is a fact about hardware.
    /// An extension this build cannot *serialize* must not be advertised: the guest would enable
    /// it, send one of its structs, and the decoder would poison the ring for asking. Three of
    /// `HOST_EXTENSIONS` are exactly that case -- Metal interop and the portability subset are
    /// how the host does its half of the work, and no venus guest has any business seeing them.
    ///
    /// The spec version is this protocol's, not the driver's, for the same reason: it is the
    /// version whose structs the decoder knows.
    ///
    /// The driver is never asked again. `learn_extensions` asked once and the answer does not
    /// change for the life of the instance -- and a device it could not ask is not enumerated to
    /// the guest at all, so an empty answer here is a device the guest was never given.
    pub fn advertised_extensions(&self, pd: VkPhysicalDevice) -> Vec<VkExtensionProperties> {
        let Some(names) = self.physical_device_exts.get(&pd) else {
            return Vec::new();
        };
        let mut out: Vec<VkExtensionProperties> =
            names.iter().filter_map(|name| extension_properties(name)).collect();

        // The two the Metal path provides, advertised even though the driver has neither.
        //
        // A venus guest reads this list to decide what external memory it has, and the two
        // answers are not independent: mesa sets its renderer handle type only under
        // `VK_EXT_external_memory_dma_buf` (`vn_physical_device.c`), so advertising the fd
        // extension alone leaves it zero and the guest concludes the renderer exposes no
        // external memory at all. A compositor then finds no dma-buf, falls back to a dumb
        // buffer, and never exports a scanout -- which is a black screen behind a process that
        // looks entirely healthy. Both, or neither.
        //
        // They go in only when the driver has the Metal interop that emulates them and lacks the
        // real thing, which is the condition under which the emulation is both needed and
        // available. `EMULATED_ON_THE_HOST` is the other half of the same decision: advertised
        // here, and stripped from the list the device is created with, because the driver would
        // fail `vkCreateDevice` outright for an extension it does not have.
        if self.supports(pd, "VK_EXT_external_memory_metal")
            && !self.supports(pd, "VK_KHR_external_memory_fd")
        {
            out.extend(EMULATED_ON_THE_HOST.iter().filter_map(|n| extension_properties(n)));
        }
        out
    }

    fn supports(&self, pd: VkPhysicalDevice, name: &str) -> bool {
        self.physical_device_exts.get(&pd).is_some_and(|s| s.contains(name))
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
        info: &VkDeviceCreateInfo,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<VkDevice, VkResult> {
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
        self.devices.insert(out, DeviceState { fns: vulkan::device(inst, out), memory_types });
        Ok(out)
    }

    // ------------------------------------------------------------------- queries
    //
    // Vulkan's `vkGet*` queries are one shape wearing several arities: find a table, call one
    // entry point, let it fill a struct the guest supplied. The struct is the guest's because the
    // decoder allocated it from the arena at the layout a C compiler agrees with -- so the driver
    // writes into the very memory the reply encoder will read back, chained `pNext` structs and
    // all, and nothing has to be copied between two spellings of the same fact.
    //
    // What varies is only the signature, so these are generic over it: the caller names the entry
    // point and the types follow. `R` is the entry point's own return, which is `()` for the
    // queries that cannot fail and `VkResult` for the ones that can -- one set of primitives for
    // both. `Err` is this renderer unable to *ask* (no instance, no such entry point); `Ok(r)` is
    // the driver's own answer, whatever it was.
    //
    // A missing entry point is deliberately not a panic here. The accessors that panic are for
    // commands this build advertises, where absence is our bug; a query is a path the guest
    // steers, and one guest asking for an extension this driver lacks must not take the worker
    // down with it.

    /// The memory types a host pointer can be imported into, as Vulkan's own bitmask.
    ///
    /// Answered off the list taken once at [`Driver::create_device`] rather than by asking the
    /// driver again: the properties cannot change under us, so a second reading would be a second
    /// copy of one fact (CLAUDE.md), and a query that took an instance round trip per call would
    /// pay it on a path a compositor walks every frame.
    ///
    /// This has to agree with what the allocation path actually does. A resource the guest can
    /// reach through a host pointer is imported as `HOST_ALLOCATION_BIT_EXT`, which accepts the
    /// host-visible types and no others -- so reporting anything wider hands the guest a memory
    /// type the very next `vkAllocateMemory` will refuse, and anything narrower hands it zero
    /// types for a buffer that binds perfectly well.
    pub fn host_visible_memory_types(&self, device: VkDevice) -> Result<u32, VkResult> {
        let d = self.devices.get(&device).ok_or(VkResult::VK_ERROR_DEVICE_LOST)?;
        let mut bits = 0u32;
        for (i, flags) in d.memory_types.iter().enumerate() {
            if flags.0 & HOST_VISIBLE_BIT != 0 {
                bits |= 1 << i;
            }
        }
        Ok(bits)
    }

    /// A physical-device query with nothing between the handle and the answer.
    pub fn pd_query<T, R>(
        &self,
        pd: VkPhysicalDevice,
        out: &mut T,
        pick: impl FnOnce(&InstanceFns) -> Option<unsafe extern "C" fn(VkPhysicalDevice, *mut T) -> R>,
    ) -> Result<R, VkResult> {
        let f = self
            .instance
            .as_ref()
            .and_then(pick)
            .ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: `pd` is a handle this instance returned, and `out` is a live exclusive
        // borrow for the length of the call.
        Ok(unsafe { f(pd, out) })
    }

    /// A physical-device query that names what it is asking about by value.
    pub fn pd_query_arg<A, T, R>(
        &self,
        pd: VkPhysicalDevice,
        a: A,
        out: &mut T,
        pick: impl FnOnce(
            &InstanceFns,
        ) -> Option<unsafe extern "C" fn(VkPhysicalDevice, A, *mut T) -> R>,
    ) -> Result<R, VkResult> {
        let f = self
            .instance
            .as_ref()
            .and_then(pick)
            .ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: as `pd_query`; `a` is a plain value the guest sent.
        Ok(unsafe { f(pd, a, out) })
    }

    /// A physical-device query that names what it is asking about with a struct.
    ///
    /// `info` is a borrow rather than an `Option`, for the reason [`Driver::dev_ask_info`] gives:
    /// vk.xml marks the info struct of every command routed through this family required, so
    /// there is no null for the helper to forward. A guest that sends one anyway is a guest
    /// asking a question it did not state, and the handler decides that before it gets here.
    pub fn pd_query_info<I, T, R>(
        &self,
        pd: VkPhysicalDevice,
        info: &I,
        out: &mut T,
        pick: impl FnOnce(
            &InstanceFns,
        )
            -> Option<unsafe extern "C" fn(VkPhysicalDevice, *const I, *mut T) -> R>,
    ) -> Result<R, VkResult> {
        let f = self
            .instance
            .as_ref()
            .and_then(pick)
            .ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: as `pd_query`; `info` borrows an arena struct live for the call.
        Ok(unsafe { f(pd, info, out) })
    }

    /// A physical-device query whose request is six loose scalars: only
    /// `vkGetPhysicalDeviceImageFormatProperties`, and it is spelled out rather than made generic
    /// because six interchangeable-looking values in a row is precisely what a type parameter
    /// would stop catching. The `2` spelling next door gathers the same six into a struct.
    ///
    /// `Ok(VK_ERROR_FORMAT_NOT_SUPPORTED)` is the driver's ordinary answer to a format probe, not
    /// a failure to ask -- guests loop over formats expecting it. Only `Err` means this renderer
    /// could not put the question, and only that is a refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn image_format_properties(
        &self,
        pd: VkPhysicalDevice,
        format: VkFormat,
        ty: VkImageType,
        tiling: VkImageTiling,
        usage: VkImageUsageFlags,
        flags: VkImageCreateFlags,
        out: &mut VkImageFormatProperties,
        pick: impl FnOnce(
            &InstanceFns,
        ) -> Option<
            unsafe extern "C" fn(
                VkPhysicalDevice,
                VkFormat,
                VkImageType,
                VkImageTiling,
                VkImageUsageFlags,
                VkImageCreateFlags,
                *mut VkImageFormatProperties,
            ) -> VkResult,
        >,
    ) -> Result<VkResult, VkResult> {
        let f = self
            .instance
            .as_ref()
            .and_then(pick)
            .ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: as `pd_query`; the six between the handle and the answer are plain scalars the
        // guest sent, passed in the order Vulkan declares them.
        Ok(unsafe { f(pd, format, ty, tiling, usage, flags, out) })
    }

    /// What one device in a group may do with another's memory:
    /// `vkGetDeviceGroupPeerMemoryFeatures`.
    ///
    /// Three `u32`s in a row, and two of them are device indices that mean opposite things. Named
    /// parameters are the only thing standing between "what may the local device do with the
    /// remote one's memory" and its mirror image, which is a different answer and compiles.
    pub fn peer_memory_features<T, R>(
        &self,
        device: VkDevice,
        heap_index: u32,
        local_device: u32,
        remote_device: u32,
        out: &mut T,
        pick: impl FnOnce(
            &DeviceFns,
        ) -> Option<unsafe extern "C" fn(VkDevice, u32, u32, u32, *mut T) -> R>,
    ) -> Result<R, VkResult> {
        let d = self.devices.get(&device).ok_or(VkResult::VK_ERROR_DEVICE_LOST)?;
        let f = pick(&d.fns).ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: as `dev_query_info`; the three indices are plain scalars off the wire.
        Ok(unsafe { f(device, heap_index, local_device, remote_device, out) })
    }

    /// A device query whose answer is the entry point's own return value, with no out-parameter
    /// at all: the three address queries.
    ///
    /// The only query family where a refusal cannot be softened. There is no struct to leave
    /// unfilled and no `VkResult` to carry an error, so a driver this renderer could not ask
    /// leaves nothing but the zero the reply would encode -- and zero is a null address the guest
    /// would hand straight to the GPU. `Err` here has to reach the guest as a stopped ring.
    /// `info` is a borrow rather than an `Option`, unlike the query helpers above: all three of
    /// these commands require their struct, so there is no null for this helper to forward. The
    /// handler decides what an absent one means before it gets here.
    pub fn dev_ask_info<I, R>(
        &self,
        device: VkDevice,
        info: &I,
        pick: impl FnOnce(&DeviceFns) -> Option<unsafe extern "C" fn(VkDevice, *const I) -> R>,
    ) -> Result<R, VkResult> {
        let d = self.devices.get(&device).ok_or(VkResult::VK_ERROR_DEVICE_LOST)?;
        let f = pick(&d.fns).ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: as `dev_query_info`; `info` borrows an arena struct live for the call.
        Ok(unsafe { f(device, info) })
    }

    /// A device query that names what it is asking about with a struct. `info` is a borrow for
    /// the reason [`Driver::pd_query_info`] gives.
    pub fn dev_query_info<I, T, R>(
        &self,
        device: VkDevice,
        info: &I,
        out: &mut T,
        pick: impl FnOnce(&DeviceFns) -> Option<unsafe extern "C" fn(VkDevice, *const I, *mut T) -> R>,
    ) -> Result<R, VkResult> {
        let d = self.devices.get(&device).ok_or(VkResult::VK_ERROR_DEVICE_LOST)?;
        let f = pick(&d.fns).ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: `device` is a handle this table was loaded from, `info` borrows an arena struct
        // live for the call, and `out` is a live exclusive borrow.
        Ok(unsafe { f(device, info, out) })
    }

    /// A device query about one of the device's own objects.
    pub fn dev_query_arg<A, T, R>(
        &self,
        device: VkDevice,
        a: A,
        out: &mut T,
        pick: impl FnOnce(&DeviceFns) -> Option<unsafe extern "C" fn(VkDevice, A, *mut T) -> R>,
    ) -> Result<R, VkResult> {
        let d = self.devices.get(&device).ok_or(VkResult::VK_ERROR_DEVICE_LOST)?;
        let f = pick(&d.fns).ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: as `dev_query_info`; `a` is a handle the guest named, already resolved.
        Ok(unsafe { f(device, a, out) })
    }

    /// A device query about one of its objects, narrowed by a struct. `info` is a borrow for the
    /// reason [`Driver::pd_query_info`] gives.
    pub fn dev_query_arg_info<A, I, T, R>(
        &self,
        device: VkDevice,
        a: A,
        info: &I,
        out: &mut T,
        pick: impl FnOnce(
            &DeviceFns,
        ) -> Option<unsafe extern "C" fn(VkDevice, A, *const I, *mut T) -> R>,
    ) -> Result<R, VkResult> {
        let d = self.devices.get(&device).ok_or(VkResult::VK_ERROR_DEVICE_LOST)?;
        let f = pick(&d.fns).ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        // SAFETY: as `dev_query_info`.
        Ok(unsafe { f(device, a, info, out) })
    }

    /// An enumeration, in whichever of Vulkan's two calls the guest asked for.
    ///
    /// Generic over what is being enumerated *for* -- a physical device's queue families, an
    /// instance's device groups -- because the two-call shape is the same and only the handle in
    /// front of it differs.
    ///
    /// `out` is `None` for the count query -- the guest asking how many there are, with no array
    /// behind it -- and `Some` for the fill, where the slice's own length is what the driver is
    /// told it has room for. The count comes back either way, and the caller writes it where the
    /// guest can read it.
    ///
    /// `VK_INCOMPLETE` is the driver having more than the guest asked for. That is the guest's
    /// business rather than an error: it sized the array and it gets what fits.
    pub fn enumerate_into<H, T, R>(
        &self,
        h: H,
        out: Option<&mut [T]>,
        pick: impl FnOnce(&InstanceFns) -> Option<unsafe extern "C" fn(H, *mut u32, *mut T) -> R>,
    ) -> Result<(u32, R), VkResult> {
        let f = self
            .instance
            .as_ref()
            .and_then(pick)
            .ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        let (mut n, room, array) = split(out);
        // SAFETY: `h` is a handle this instance returned, and `n` is initialised to the length of
        // the array `array` points at -- the pair the caller handed us as one slice.
        let r = unsafe { f(h, &mut n, array) };
        fits(n, room);
        Ok((n, r))
    }

    /// The same, narrowed by a struct: `vkGetPhysicalDeviceSparseImageFormatProperties2`.
    ///
    /// `info` is a borrow rather than an `Option` for the reason [`Driver::dev_ask_info`] gives:
    /// the command requires its struct, so there is no null for this helper to forward, and what
    /// an absent one means is the handler's to decide before it gets here.
    pub fn enumerate_info_into<H, I, T, R>(
        &self,
        h: H,
        info: &I,
        out: Option<&mut [T]>,
        pick: impl FnOnce(
            &InstanceFns,
        ) -> Option<unsafe extern "C" fn(H, *const I, *mut u32, *mut T) -> R>,
    ) -> Result<(u32, R), VkResult> {
        let f = self
            .instance
            .as_ref()
            .and_then(pick)
            .ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        let (mut n, room, array) = split(out);
        // SAFETY: as `enumerate_into`; `info` borrows an arena struct live for the call.
        let r = unsafe { f(h, info, &mut n, array) };
        fits(n, room);
        Ok((n, r))
    }

    /// `vkGetPhysicalDeviceSparseImageFormatProperties`, whose request is six loose scalars.
    ///
    /// Spelled out rather than generic, for the reason [`Driver::image_format_properties`] gives:
    /// a row of interchangeable-looking values is exactly what a type parameter would stop the
    /// compiler catching. The 1.0 form of a query whose `2` neighbour above takes a struct.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_format_properties<T>(
        &self,
        pd: VkPhysicalDevice,
        format: VkFormat,
        ty: VkImageType,
        samples: VkSampleCountFlagBits,
        usage: VkImageUsageFlags,
        tiling: VkImageTiling,
        out: Option<&mut [T]>,
        pick: impl FnOnce(
            &InstanceFns,
        ) -> Option<
            unsafe extern "C" fn(
                VkPhysicalDevice,
                VkFormat,
                VkImageType,
                VkSampleCountFlagBits,
                VkImageUsageFlags,
                VkImageTiling,
                *mut u32,
                *mut T,
            ),
        >,
    ) -> Result<u32, VkResult> {
        let f = self
            .instance
            .as_ref()
            .and_then(pick)
            .ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        let (mut n, room, array) = split(out);
        // SAFETY: as `enumerate_into`; the six scalars are plain values off the wire.
        unsafe { f(pd, format, ty, samples, usage, tiling, &mut n, array) };
        fits(n, room);
        Ok(n)
    }

    /// An enumeration about one of a device's own objects: `vkGetImageSparseMemoryRequirements`.
    pub fn dev_enumerate_arg<A, T, R>(
        &self,
        device: VkDevice,
        a: A,
        out: Option<&mut [T]>,
        pick: impl FnOnce(
            &DeviceFns,
        ) -> Option<unsafe extern "C" fn(VkDevice, A, *mut u32, *mut T) -> R>,
    ) -> Result<(u32, R), VkResult> {
        let d = self.devices.get(&device).ok_or(VkResult::VK_ERROR_DEVICE_LOST)?;
        let f = pick(&d.fns).ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        let (mut n, room, array) = split(out);
        // SAFETY: as `enumerate_into`; `a` is a handle the guest named, already resolved.
        let r = unsafe { f(device, a, &mut n, array) };
        fits(n, room);
        Ok((n, r))
    }

    /// The same, narrowed by a struct rather than a handle: the `2` forms of the above.
    pub fn dev_enumerate_info<I, T, R>(
        &self,
        device: VkDevice,
        info: &I,
        out: Option<&mut [T]>,
        pick: impl FnOnce(
            &DeviceFns,
        )
            -> Option<unsafe extern "C" fn(VkDevice, *const I, *mut u32, *mut T) -> R>,
    ) -> Result<(u32, R), VkResult> {
        let d = self.devices.get(&device).ok_or(VkResult::VK_ERROR_DEVICE_LOST)?;
        let f = pick(&d.fns).ok_or(VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)?;
        let (mut n, room, array) = split(out);
        // SAFETY: as `enumerate_into`; `info` borrows an arena struct live for the call.
        let r = unsafe { f(device, info, &mut n, array) };
        fits(n, room);
        Ok((n, r))
    }

    /// The loader's own version, before any instance exists.
    ///
    /// The one query with no handle in it at all: the guest may ask it before `vkCreateInstance`,
    /// so it is answered off the global table rather than this driver's state. Named rather than
    /// generic because it has exactly one shape and one caller, and a seventh primitive to carry
    /// it would be more machinery than the call.
    pub fn instance_version(&self, global: &Global) -> Result<u32, VkResult> {
        let f = global
            .try_vkEnumerateInstanceVersion()
            .ok_or(VkResult::VK_ERROR_INCOMPATIBLE_DRIVER)?;
        let mut version = 0u32;
        // SAFETY: `version` is a live local for the length of the call, which is the whole of
        // what this entry point touches.
        let r = unsafe { f(&mut version) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        Ok(version)
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
    pub fn device_queue(&mut self, device: VkDevice, info: &VkDeviceQueueInfo2) -> Option<VkQueue> {
        let d = self.devices.get(&device)?;
        let mut out = VkQueue(0);
        // SAFETY: `device` is a handle this table was loaded from and `info` is an arena
        // allocation live for the call.
        unsafe { (d.fns.vkGetDeviceQueue2())(device, info, &mut out) };
        if out.0 == 0 {
            return None;
        }
        // Asking twice for the same queue is how a guest works, not a mistake: Vulkan hands back
        // the same handle each time, and the answer recorded here is the same both times.
        self.queues.insert(out, device);
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
                    (fns.vkDestroySemaphore())(device, VkSemaphore::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_FENCE => (fns.vkDestroyFence())(device, VkFence::from_host(h), n),
                T::VK_OBJECT_TYPE_BUFFER => {
                    (fns.vkDestroyBuffer())(device, VkBuffer::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_IMAGE => (fns.vkDestroyImage())(device, VkImage::from_host(h), n),
                T::VK_OBJECT_TYPE_EVENT => (fns.vkDestroyEvent())(device, VkEvent::from_host(h), n),
                T::VK_OBJECT_TYPE_QUERY_POOL => {
                    (fns.vkDestroyQueryPool())(device, VkQueryPool::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_BUFFER_VIEW => {
                    (fns.vkDestroyBufferView())(device, VkBufferView::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_IMAGE_VIEW => {
                    (fns.vkDestroyImageView())(device, VkImageView::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_SHADER_MODULE => {
                    (fns.vkDestroyShaderModule())(device, VkShaderModule::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_PIPELINE_CACHE => {
                    (fns.vkDestroyPipelineCache())(device, VkPipelineCache::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_PIPELINE_LAYOUT => {
                    (fns.vkDestroyPipelineLayout())(device, VkPipelineLayout::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_RENDER_PASS => {
                    (fns.vkDestroyRenderPass())(device, VkRenderPass::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_PIPELINE => {
                    (fns.vkDestroyPipeline())(device, VkPipeline::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT => (fns.vkDestroyDescriptorSetLayout())(
                    device,
                    VkDescriptorSetLayout::from_host(h),
                    n,
                ),
                T::VK_OBJECT_TYPE_SAMPLER => {
                    (fns.vkDestroySampler())(device, VkSampler::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_FRAMEBUFFER => {
                    (fns.vkDestroyFramebuffer())(device, VkFramebuffer::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION => (fns
                    .vkDestroySamplerYcbcrConversion())(
                    device,
                    VkSamplerYcbcrConversion::from_host(h),
                    n,
                ),
                T::VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE => (fns
                    .vkDestroyDescriptorUpdateTemplate())(
                    device,
                    VkDescriptorUpdateTemplate::from_host(h),
                    n,
                ),
                // Destroying a pool frees everything allocated from it, which is why the two kinds
                // below it are skipped rather than walked.
                T::VK_OBJECT_TYPE_COMMAND_POOL => {
                    (fns.vkDestroyCommandPool())(device, VkCommandPool::from_host(h), n)
                }
                T::VK_OBJECT_TYPE_DESCRIPTOR_POOL => {
                    (fns.vkDestroyDescriptorPool())(device, VkDescriptorPool::from_host(h), n)
                }
                // Freed with the pool they came from, one line above.
                T::VK_OBJECT_TYPE_COMMAND_BUFFER | T::VK_OBJECT_TYPE_DESCRIPTOR_SET => {}
                // Last of all, and `empty_device` is what puts it last: anything bound to an
                // allocation has to be destroyed before the allocation is freed.
                T::VK_OBJECT_TYPE_DEVICE_MEMORY => {
                    (fns.vkFreeMemory())(device, VkDeviceMemory::from_host(h), n)
                }
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
                    eprintln!(
                        "[virglrs] no destroy for VkObjectType {}, leaking {:#x}",
                        other.0, h.0
                    )
                }
            }
        }
    }

    /// Everything a device owns, torn down in the order Vulkan requires, before the device itself.
    fn empty_device(&mut self, device: VkDevice, doomed: &[Doomed]) {
        let Some(d) = self.devices.get(&device) else {
            return;
        };
        // Nothing may be destroyed while the device is still working on it, and the guest is not
        // required to have waited. The C waits here too.
        // SAFETY: a device this context created and has not yet destroyed.
        let r = unsafe { (d.fns.vkDeviceWaitIdle())(device) };
        if r != VkResult::VK_SUCCESS {
            eprintln!("[virglrs] vkDeviceWaitIdle before teardown: VkResult {}", r.0);
        }
        // Two passes, because freeing an allocation while a buffer or an image is still bound to
        // it is undefined. Every other kind first, then the memory underneath them -- an ordering
        // Vulkan requires and the arena's walk order cannot be relied on to produce.
        //
        // Filtered by device here rather than by the caller, so an object can only ever be
        // destroyed on the device the table says it belongs to.
        let mine = || doomed.iter().filter(|o| o.device == Some(device));
        let is_memory = |o: &Doomed| o.ty == VkObjectType::VK_OBJECT_TYPE_DEVICE_MEMORY;
        for o in mine().filter(|o| !is_memory(o)) {
            Self::destroy_tracked(&d.fns, device, o);
        }
        for o in mine().filter(|o| is_memory(o)) {
            Self::destroy_tracked(&d.fns, device, o);
        }
        // The other place an image dies -- the guest left it live and the teardown took it. Its
        // record goes with it, here rather than in `destroy_tracked`, which holds the device's
        // entry points borrowed out of `self` and so cannot reach the map.
        let images: Vec<VkImage> = doomed
            .iter()
            .filter(|o| o.device == Some(device) && o.ty == VkObjectType::VK_OBJECT_TYPE_IMAGE)
            .map(|o| VkImage::from_host(o.handle))
            .collect();
        for image in images {
            self.forget_image(image);
        }
        // The census records the freed allocations by the guest's id, and they have just stopped
        // being live. Done after the borrow above rather than beside each free.
        let freed: Vec<ObjectId> = mine().filter(|o| is_memory(o)).map(|o| o.id).collect();
        for id in freed {
            self.memory.remove(&id);
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
        let orphans = self.pools.close_device(device);
        self.queues.retain(|_, owner| *owner != device);
        let Some(d) = self.devices.remove(&device) else {
            return orphans;
        };
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
        info: &I,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<T, VkResult> {
        let Some(d) = self.devices.get(&device) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        let mut out = T::null();
        // SAFETY: `device` is a handle in this table, `info` and `alloc` are the decoder's arena
        // allocations live for this call, and `out` is a local.
        let r = unsafe { proc(&d.fns)(device, info, ptr(alloc), &mut out) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        assert!(out.host().0 != 0, "a create succeeded and returned a null handle");
        Ok(out)
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
        let Some(d) = self.devices.get(&device) else {
            return;
        };
        if object.host().0 == 0 {
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
    pub fn allocate_objects<P: PoolOf, I>(
        &mut self,
        device: VkDevice,
        pool: P,
        proc: impl FnOnce(
            &DeviceFns,
        )
            -> unsafe extern "C" fn(VkDevice, *const I, *mut P::Child) -> VkResult,
        info: &I,
        out: &mut [P::Child],
        ids: &[ObjectId],
    ) -> Result<(), VkResult> {
        let Some(d) = self.devices.get(&device) else {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        };
        if out.is_empty() {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        // The pool is re-checked here for the same reason the device is: the guest may have
        // destroyed it, and an id the object table still resolves is not a live driver object.
        if !self.pools.is_open(pool) {
            return Err(VkResult::VK_ERROR_INITIALIZATION_FAILED);
        }
        // SAFETY: `device` is a handle in this table; `info` is an arena allocation live for the
        // call, and `out` is the arena array the decoder sized from the count inside `info`.
        let r = unsafe { proc(&d.fns)(device, info, out.as_mut_ptr()) };
        if r != VkResult::VK_SUCCESS {
            return Err(r);
        }
        // Vulkan fills every element of a pool allocation or none, so the whole slice is real.
        // Both names of each object are recorded together; see `Pools`.
        self.pools.adopt(
            pool,
            out.iter().copied().zip(ids.iter().copied()).filter(|(h, _)| h.host().0 != 0),
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
        let Some(d) = self.devices.get(&device) else {
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
            if survivor.host().0 == 0 {
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
    pub fn free_objects<P: PoolOf>(
        &mut self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, P, u32, *const P::Child),
        pool: P,
        objects: &[P::Child],
    ) {
        let Some(d) = self.devices.get(&device) else {
            return;
        };
        if objects.is_empty() || !self.pools.is_open(pool) {
            return;
        }
        // SAFETY: handles this context allocated, and the count Vulkan is given is the slice's own
        // length. The generated lifecycle hook removes the ids from the object table exactly once.
        unsafe { proc(&d.fns)(device, pool, objects.len() as u32, objects.as_ptr()) };
        self.pools.release(objects.iter().copied());
    }

    /// Register a device with a hand-built proc table, as `create_device` would have.
    ///
    /// Test scaffolding, and the other half of `Device::plant_*`: together they let a test watch
    /// what a handler hands the driver, which is the boundary nothing else in the harness can
    /// see. See `plant_pool` for why the real path is out of reach.
    #[cfg(test)]
    pub(super) fn plant_device(&mut self, handle: VkDevice, fns: DeviceFns) {
        self.devices.insert(handle, DeviceState { fns, memory_types: Vec::new() });
    }

    /// Give a planted device the memory types `vkCreateDevice` would have read off the driver.
    /// Test scaffolding, separate from `plant_device` because most tests never look at them.
    #[cfg(test)]
    pub(super) fn plant_memory_types(&mut self, handle: VkDevice, types: &[VkMemoryPropertyFlags]) {
        let d = self.devices.get_mut(&handle).expect("a planted device");
        d.memory_types = types.to_vec();
    }

    /// Record what a physical device supports, as `learn_extensions` would have off a real
    /// driver. Test scaffolding: the real path needs an instance and a loader.
    #[cfg(test)]
    pub(super) fn plant_extensions(&mut self, pd: VkPhysicalDevice, names: &[&str]) {
        self.physical_device_exts.insert(pd, names.iter().map(|n| n.to_string()).collect());
    }

    /// Stand an instance table up with no loader behind it, so an instance-level query has
    /// somewhere to be watched. The device half of this is `plant_device`.
    #[cfg(test)]
    pub(super) fn plant_instance(&mut self, fns: InstanceFns) {
        self.instance = Some(fns);
    }

    /// Plant a live allocation from a memory type with the given properties.
    #[cfg(test)]
    pub(super) fn plant_allocation_of(&mut self, id: ObjectId, size: u64, props: u32) {
        // Charged like a real one, so a test's ledger says what a guest's would.
        let charge = self.account.try_charge("device memory", size).ok();
        self.memory.insert(
            id,
            Allocated {
                size,
                backing: Backing::Driver,
                props: VkMemoryPropertyFlags(props as _),
                exported: None,
                charge,
            },
        );
    }

    /// Plant an allocation that aliases another context's storage -- host-visible like any
    /// import, so that what keeps it out of the census is the aliasing and nothing else.
    #[cfg(test)]
    pub(super) fn plant_imported_allocation(&mut self, id: ObjectId, size: u64) {
        self.plant_allocation(id, size);
        self.memory.get_mut(&id).expect("just planted").backing = Backing::Imported;
    }

    /// Plant an allocation backed by a real IOSurface, which is the only way to get one: a
    /// surface cannot be faked, and every claim about a scanout is a claim about what the system
    /// did with it.
    #[cfg(test)]
    pub(super) fn plant_scanout_allocation(&mut self, id: ObjectId, surface: Surface) {
        let size = surface.alloc_size();
        self.plant_allocation(id, size);
        self.memory.get_mut(&id).expect("just planted").backing = Backing::Scanout(surface);
    }

    /// Plant a live allocation the host can address, cache and see coherently -- what MoltenVK's
    /// host-visible types are, and what every test that reads or exports one needs.
    #[cfg(test)]
    pub(super) fn plant_allocation(&mut self, id: ObjectId, size: u64) {
        self.plant_allocation_of(id, size, HOST_VISIBLE_BIT | HOST_COHERENT_BIT | HOST_CACHED_BIT);
    }

    /// The same, for memory the host cannot address -- the one case an export must refuse before
    /// it ever reaches the driver.
    #[cfg(test)]
    pub(super) fn plant_device_local_allocation(&mut self, id: ObjectId, size: u64) {
        self.plant_allocation_of(id, size, 0);
    }

    /// Drop planted state without destroying it, for a test that stood a driver up by hand.
    ///
    /// Nothing planted came from Vulkan, so there is nothing to leak and nothing to call a destroy
    /// on -- and a planted table has only the few entry points its own test needed, so a real
    /// teardown would abort on the first one it did not plant. A test that plants and forgets this
    /// fails loudly on the drop rather than quietly, which is the point of the bomb.
    #[cfg(test)]
    pub(super) fn abandon_planted(&mut self) {
        self.instance = None;
        self.devices.clear();
        self.memory.clear();
        self.pools = Pools::default();
        self.queues.clear();
        self.physical_device_exts.clear();
    }

    /// Stand a pool up with contents already in it, as a run of allocations would have left it.
    ///
    /// Test scaffolding. Reaching the real path needs a live device and a driver that answers,
    /// which is the one thing a unit test has no way to arrange -- and what the tests want to ask
    /// about is what happens to those contents afterwards.
    #[cfg(test)]
    pub(super) fn plant_pool<P: PoolOf>(
        &mut self,
        device: VkDevice,
        pool: P,
        children: &[(P::Child, ObjectId)],
    ) {
        self.pools.open(device, pool);
        self.pools.adopt(pool, children.iter().copied());
    }

    /// The guest id a pool has filed a host handle under, if it holds it at all.
    ///
    /// Test scaffolding, beside [`Driver::plant_pool`] because it is the same seam read the other
    /// way. What a run of pool allocations has to get right is the *pairing* -- a run recorded
    /// shifted by one resolves every id to a live object that belongs to another, which nothing
    /// afterwards can detect -- and a pairing is only observable by asking about both halves.
    #[cfg(test)]
    pub(super) fn pool_child_id<P: PoolOf>(&self, pool: P, child: P::Child) -> Option<ObjectId> {
        let p = self.pools.open.get(&TypedHandle::of(pool))?;
        p.children.get(&TypedHandle::of(child)).copied()
    }

    /// Point a queue at a device, as `device_queue` would have.
    ///
    /// Test scaffolding. The real path needs a driver that answers `vkGetDeviceQueue2`, and what
    /// the tests want to ask about is what a submit does once the answer is in.
    #[cfg(test)]
    pub(super) fn plant_queue(&mut self, device: VkDevice, queue: VkQueue) {
        self.queues.insert(queue, device);
    }

    /// Create a pool, and start tracking what will be allocated from it.
    pub fn create_pool<T: PoolOf, I>(
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
        info: &I,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<T, VkResult> {
        let handle = self.create_object(device, proc, info, alloc)?;
        self.pools.open(device, handle);
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
        let orphans = self.pools.close(pool);
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
        self.devices.get(&self.pools.device_of(cb)?).map(|d| &d.fns)
    }

    /// `vkBeginCommandBuffer`. The one recording command with a result, because it is the one
    /// that can run the pool out of memory before anything has been recorded.
    pub fn begin_command_buffer(
        &self,
        cb: VkCommandBuffer,
        info: &VkCommandBufferBeginInfo,
    ) -> Option<VkResult> {
        let d = self.recorder(cb)?;
        // SAFETY: a command buffer this context allocated, and `info` is an arena allocation
        // live for the call. The same holds for every call in this section.
        Some(unsafe { (d.vkBeginCommandBuffer())(cb, info) })
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
        begin: &VkRenderPassBeginInfo,
        contents: VkSubpassContents,
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        unsafe { (d.vkCmdBeginRenderPass())(cb, begin, contents) };
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

    /// The mirror of [`Self::cmd_copy_buffer_to_image`]: the image is the source, so it is the
    /// image that carries the layout and the buffer that does not.
    pub fn cmd_copy_image_to_buffer(
        &self,
        cb: VkCommandBuffer,
        src: VkImage,
        layout: VkImageLayout,
        dst: VkBuffer,
        regions: &[VkBufferImageCopy],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the slice's own length.
        unsafe {
            (d.vkCmdCopyImageToBuffer())(
                cb,
                src,
                layout,
                dst,
                regions.len() as u32,
                regions.as_ptr(),
            )
        };
        Some(())
    }

    // Vulkan's own signature: the two images each carry a layout, and the filter is a
    // parameter of the blit rather than of a region.
    #[allow(clippy::too_many_arguments)]
    pub fn cmd_blit_image(
        &self,
        cb: VkCommandBuffer,
        src: VkImage,
        src_layout: VkImageLayout,
        dst: VkImage,
        dst_layout: VkImageLayout,
        regions: &[VkImageBlit],
        filter: VkFilter,
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the slice's own length.
        unsafe {
            (d.vkCmdBlitImage())(
                cb,
                src,
                src_layout,
                dst,
                dst_layout,
                regions.len() as u32,
                regions.as_ptr(),
                filter,
            )
        };
        Some(())
    }

    pub fn cmd_clear_color_image(
        &self,
        cb: VkCommandBuffer,
        image: VkImage,
        layout: VkImageLayout,
        color: &VkClearColorValue,
        ranges: &[VkImageSubresourceRange],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the slice's own length, and the colour is a reference.
        unsafe {
            (d.vkCmdClearColorImage())(
                cb,
                image,
                layout,
                color,
                ranges.len() as u32,
                ranges.as_ptr(),
            )
        };
        Some(())
    }

    /// `vkCmdClearAttachments`, whose two arrays are counted separately and mean different things.
    ///
    /// Every attachment is cleared over every rect, so the two are a product, not a pair: neither
    /// count governs the other and neither may be derived from the other.
    pub fn cmd_clear_attachments(
        &self,
        cb: VkCommandBuffer,
        attachments: &[VkClearAttachment],
        rects: &[VkClearRect],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; each count is its own slice's length.
        unsafe {
            (d.vkCmdClearAttachments())(
                cb,
                attachments.len() as u32,
                attachments.as_ptr(),
                rects.len() as u32,
                rects.as_ptr(),
            )
        };
        Some(())
    }

    /// `vkCmdPushConstants`, whose `size` is the length of the bytes and nothing else.
    ///
    /// The guest sends both, and the decoder has already lengthened the slice from the guest's
    /// `size`. Taking the slice and re-deriving the count from it is what keeps a later edit from
    /// reintroducing a size that disagrees with the buffer it measures.
    pub fn cmd_push_constants(
        &self,
        cb: VkCommandBuffer,
        layout: VkPipelineLayout,
        stages: VkShaderStageFlags,
        offset: u32,
        values: &[u8],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the size is the slice's own length in bytes.
        unsafe {
            (d.vkCmdPushConstants())(
                cb,
                layout,
                stages,
                offset,
                values.len() as u32,
                values.as_ptr().cast(),
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
        self.devices.get(self.queues.get(&queue)?).map(|d| &d.fns)
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
        let Some(d) = self.devices.get(&device) else {
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
        let Some(d) = self.devices.get(&device) else {
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
            flags: VkSemaphoreImportFlagBits::VK_SEMAPHORE_IMPORT_TEMPORARY_BIT.into(),
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
        let d = self.devices.get(&device).ok_or(NoSyncFd::NoDevice)?;
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

    /// A device entry point that takes the device and nothing else: `vkDeviceWaitIdle`.
    ///
    /// A device this table does not have answers with an error rather than a success. Every helper
    /// in this group does the same, and it is the point of them: the alternative is reporting that
    /// something happened to a device we never had, which the guest cannot tell from the truth.
    pub fn device_op(
        &self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice) -> VkResult,
    ) -> VkResult {
        let Some(d) = self.devices.get(&device) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // SAFETY: `device` is a handle in this table. The same holds for every call below.
        unsafe { proc(&d.fns)(device) }
    }

    /// A queue entry point that takes the queue and nothing else: `vkQueueWaitIdle`.
    ///
    /// `None` is a queue this context never retrieved, which the caller turns into a rejection --
    /// the same answer `vkQueueSubmit` gives, and for the same reason. See [`Driver::submitter`].
    pub fn queue_op(
        &self,
        queue: VkQueue,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkQueue) -> VkResult,
    ) -> Option<VkResult> {
        let d = self.submitter(queue)?;
        // SAFETY: a queue this context retrieved, on the device that produced it.
        Some(unsafe { proc(d)(queue) })
    }

    /// A device entry point that names one object and nothing else: the event and fence states.
    pub fn object_op<T: Handle>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, T) -> VkResult,
        target: T,
    ) -> VkResult {
        let Some(d) = self.devices.get(&device) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // SAFETY: `device` is a handle in this table, and `target` is one the decoder resolved
        // through the object table before this handler ran.
        unsafe { proc(&d.fns)(device, target) }
    }

    /// A device entry point that names one object and a flags word: the pool resets.
    pub fn object_flags_op<T: Handle, F: Copy>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, T, F) -> VkResult,
        target: T,
        flags: F,
    ) -> VkResult {
        let Some(d) = self.devices.get(&device) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // SAFETY: as above; `flags` is a plain scalar off the wire.
        unsafe { proc(&d.fns)(device, target, flags) }
    }

    /// Bind memory to a single buffer or image: `vkBindBufferMemory`, `vkBindImageMemory`.
    ///
    /// The pre-1.1 spelling of what [`Driver::counted_op`] serves for the `2` forms. A guest that
    /// has both still sends this one, so it is served rather than translated: rewriting it into
    /// the `2` form would be this renderer inventing a `VkBindBufferMemoryInfo` the guest never
    /// wrote, and any difference between the two would be ours and invisible.
    pub fn bind_one<T: Handle>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(
            &DeviceFns,
        )
            -> unsafe extern "C" fn(VkDevice, T, VkDeviceMemory, VkDeviceSize) -> VkResult,
        target: T,
        memory: VkDeviceMemory,
        offset: VkDeviceSize,
    ) -> VkResult {
        let Some(d) = self.devices.get(&device) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // SAFETY: as above; `memory` and `offset` are the guest's own, resolved and scalar.
        unsafe { proc(&d.fns)(device, target, memory, offset) }
    }

    /// A device entry point whose arguments are one counted array and nothing else.
    ///
    /// `vkBindBufferMemory2`, `vkBindImageMemory2`, `vkFlushMappedMemoryRanges` and
    /// `vkInvalidateMappedMemoryRanges` are all exactly this shape, which is why the entry point
    /// arrives as a closure -- the element type is the only thing that differs, and it is a type
    /// parameter. An empty array is legal for every one of them and the guest sends it.
    ///
    /// The count is never carried here: it is the slice's own length, rebuilt at the call. That is
    /// the whole point of reconciling the wire's pair at the decoder -- see CLAUDE.md, "two values
    /// that must agree are one value".
    pub fn counted_op<I>(
        &self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, u32, *const I) -> VkResult,
        infos: &[I],
    ) -> VkResult {
        let Some(d) = self.devices.get(&device) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        // An empty array is legal and the guest sends it.
        if infos.is_empty() {
            return VkResult::VK_SUCCESS;
        }
        // SAFETY: `device` is a handle in this table, and the count and array Vulkan wants are
        // the slice's own -- which is the whole reason the pair is not carried this far.
        unsafe { proc(&d.fns)(device, infos.len() as u32, infos.as_ptr()) }
    }

    /// A device entry point whose whole request is one struct: `vkSignalSemaphore`.
    ///
    /// An *operation*, not a query, and the distinction decides what an error means. There is no
    /// out-parameter here -- the reply is a bare `VkResult` -- so a refusal is a complete answer
    /// and goes back as one. The query family next door ([`Driver::dev_query_info`] and friends)
    /// cannot do that, because its reply carries a struct the driver never filled, which is why it
    /// rejects instead. Pick the family by whether the guest is owed a value, not by taste.
    ///
    /// The entry point is optional here where it is mandatory in the groups above: the timeline
    /// semaphore calls are Vulkan 1.2, and a driver without them exports none of the three. So
    /// `try_` rather than the panicking accessor -- which commands the guest may send is the
    /// guest's reading of what we advertise, and being wrong about that must cost it a command
    /// rather than the process.
    pub fn dev_op_info<I>(
        &self,
        device: VkDevice,
        info: &I,
        pick: impl FnOnce(&DeviceFns) -> Option<unsafe extern "C" fn(VkDevice, *const I) -> VkResult>,
    ) -> VkResult {
        let Some(d) = self.devices.get(&device) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        let Some(f) = pick(&d.fns) else {
            return VkResult::VK_ERROR_EXTENSION_NOT_PRESENT;
        };
        // SAFETY: `device` is a handle in this table, and `info` borrows an arena struct the
        // decoder filled and outlives the call. Any counted array inside it was reconciled against
        // the array actually sent before the struct was handed on, so the pair Vulkan reads out of
        // it agrees with itself.
        unsafe { f(device, info) }
    }

    /// The same, with the guest's own timeout beside it: `vkWaitSemaphores`.
    ///
    /// The timeout is passed exactly as it arrived, `UINT64_MAX` included, and nothing here
    /// shortens it. A clamp would come back `VK_TIMEOUT` from a wait that did not time out, which
    /// the guest cannot tell from a real one -- it would loop, and the loop would be ours. Blocking
    /// this thread for as long as the guest asked is the same bargain [`Driver::wait_for_fences`]
    /// already makes, and it is the guest's own thread being spent. A host that needs a bound on
    /// how long a wait may sit here wants it in the replayer, never in the renderer.
    pub fn dev_op_info_timeout<I>(
        &self,
        device: VkDevice,
        info: &I,
        timeout: u64,
        pick: impl FnOnce(
            &DeviceFns,
        ) -> Option<unsafe extern "C" fn(VkDevice, *const I, u64) -> VkResult>,
    ) -> VkResult {
        let Some(d) = self.devices.get(&device) else {
            return VkResult::VK_ERROR_INITIALIZATION_FAILED;
        };
        let Some(f) = pick(&d.fns) else {
            return VkResult::VK_ERROR_EXTENSION_NOT_PRESENT;
        };
        // SAFETY: as `dev_op_info`; `timeout` is a plain scalar off the wire.
        unsafe { f(device, info, timeout) }
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
        let Some(d) = self.devices.get(&device) else {
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
        id: ObjectId,
        info: &VkMemoryAllocateInfo,
        alloc: Option<&VkAllocationCallbacks>,
        resource_bytes: &dyn Fn(ResourceHandle) -> Option<ResourceBytes>,
    ) -> Result<VkDeviceMemory, NoMemory> {
        let Some(d) = self.devices.get(&device) else {
            return Err(NoMemory::Driver(VkResult::VK_ERROR_INITIALIZATION_FAILED));
        };
        // A copy, not an edit in place: the decoder's struct is the guest's request, and the
        // round trip re-encodes it. The `pNext` chain is carried over untouched.
        let mut info = *info;
        // Resolved once, both of them. The padding, the export and the census all answer from
        // the memory type's properties and from whether this aliases someone else's storage; a
        // second lookup is a second chance for two answers to disagree about one allocation.
        let props = d.memory_types.get(info.memoryTypeIndex as usize).copied();
        let import = imported_resource(info.pNext);
        info.allocationSize =
            VkDeviceSize(pad_for_blob(info.allocationSize.0, props, import.is_some()));

        // Whichever of the two this allocation is, it comes out as one host address the driver is
        // handed instead of memory of its own. They are mutually exclusive by construction: an
        // import names storage that exists, and a scanout is storage being made.
        //
        // An import that resolves to nothing falls through to an ordinary allocation, which is
        // what this did before anything resolved at all: the guest gets memory, and the storage
        // it meant to reach stays where it is. That is wrong for the guest -- it renders into a
        // buffer nobody presents -- but it is the driver's own behaviour for a `pNext` link it
        // does not recognise, and inventing a refusal here would fail allocations the C serves.
        let alias = import.and_then(|r| self.span(&resource_bytes(r)?));
        let surface = if import.is_some() { None } else { self.scanout_surface(device, &info) };
        let mut host_pointer = VkImportMemoryHostPointerInfoEXT {
            sType: VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
            pNext: info.pNext,
            handleType: VkExternalMemoryHandleTypeFlagBits::
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            pHostPointer: core::ptr::null_mut(),
        };
        // What the census reports and what the guest maps: the guest's own figure, padded. A
        // surface's page-rounded extent is a fact about how IOSurface rounds, and telling the
        // guest that number would be answering a question it did not ask.
        let size = info.allocationSize.0;
        // The pages back exactly this much, whoever owns them. The guest's figure is its own
        // image's size, and a request larger than the backing would let the driver address past
        // the end of it -- the one place a guest's arithmetic could reach outside the host's.
        if let Some(span) = surface.as_ref().map(|s| (s.host_addr(), s.alloc_size())).or(alias) {
            host_pointer.pHostPointer = span.0 as *mut core::ffi::c_void;
            // Prepended, not spliced in: the guest's chain is the decoder's arena and the round
            // trip re-encodes it, so it is read here and never rewritten.
            info.pNext = (&raw const host_pointer).cast();
            info.allocationSize = VkDeviceSize(info.allocationSize.0.min(span.1));
        }

        // What the bytes are decides both how they are freed and what they cost, so it is settled
        // once, here, and read twice.
        let backing = match (import, surface) {
            (Some(_), _) => Backing::Imported,
            (None, Some(s)) => Backing::Scanout(s),
            (None, None) => Backing::Driver,
        };
        // Charged before the driver is asked, so a refusal costs no host memory -- and credited
        // by `charge` going out of scope if the driver then refuses. A scanout is charged at the
        // surface's own extent, because the surface is the commitment; the allocation importing
        // its pages commits nothing further. An import commits nothing at all.
        let charge = match &backing {
            Backing::Driver => Some(self.account.try_charge("device memory", size)),
            Backing::Scanout(s) => Some(self.account.try_charge("IOSurface", s.alloc_size())),
            Backing::Imported => None,
        };
        let charge = match charge.transpose() {
            Ok(c) => c,
            Err(refused) => {
                self.account.report_refusal(refused);
                return Err(NoMemory::OverBudget { stop: self.account.kills_context() });
            }
        };

        // `d` was borrowed before the surface was minted, which needed `&mut self`.
        let d = self.devices.get(&device).expect("the device was here a moment ago");
        let mut out = VkDeviceMemory(0);
        // SAFETY: `info` is a local whose chain the decoder owns for the batch, extended with a
        // local that outlives this call, `alloc` is another arena allocation, and `out` is a local.
        let r = unsafe { (d.fns.vkAllocateMemory())(device, &info, ptr(alloc), &mut out) };
        if r != VkResult::VK_SUCCESS {
            return Err(NoMemory::Driver(r));
        }
        assert!(out.0 != 0, "vkAllocateMemory succeeded and returned a null handle");
        // A type index the device does not have is one `vkAllocateMemory` would have refused, so
        // the fallback describes memory that cannot exist -- and describes it as addressable by
        // nothing, which is the safe reading.
        let props = props.unwrap_or(VkMemoryPropertyFlags(0));
        self.memory.insert(id, Allocated { size, backing, props, exported: None, charge });
        Ok(out)
    }

    /// The IOSurface an allocation is backed by, asked of the surface itself.
    ///
    /// Never stored: an id is a name the system recycles the moment the surface it named is
    /// released, so a remembered one is a claim about a stranger's surface. Reaching it through
    /// the record that owns the surface is what makes "the surface is gone" and "there is no id"
    /// the same answer.
    pub fn memory_surface_id(&self, id: ObjectId) -> Option<SurfaceId> {
        self.memory.get(&id)?.surface().map(Surface::id)
    }

    /// Copy an allocation's presented pixels out, for an allocation that is a scanout.
    ///
    /// The surface itself never leaves this module: an `IOSurfaceRef` handed outward is a
    /// lifetime no one can see, which is the whole reason `metal.rs` owns them. What leaves is
    /// the bytes and how many rows of them there were.
    ///
    /// `None` for an allocation that is not a scanout -- there is no surface to read, which is a
    /// different answer from a surface that read nothing.
    pub fn memory_read_surface(
        &self,
        id: ObjectId,
        dst: &mut [u8],
        stride: usize,
        height: u32,
    ) -> Option<u32> {
        Some(self.memory.get(&id)?.surface()?.read_rows(dst, stride, height))
    }

    /// Where a resource's bytes are, as the one pair a caller can do anything with.
    ///
    /// The only place either shape of [`ResourceBytes`] becomes an address and a length, so the
    /// property query that says a resource is importable and the allocation that imports it get
    /// the same answer or no answer at all -- they cannot get two.
    pub fn span(&self, bytes: &ResourceBytes) -> Option<(usize, u64)> {
        match bytes {
            ResourceBytes::Host(map) => Some((map.host_addr(), map.len() as u64)),
            ResourceBytes::Allocation(published) => self.aliased_span(published.memory),
        }
    }

    /// Where an allocation this context already owns lives, for a second allocation that names
    /// it: the host address and how far it runs.
    ///
    /// One value, because an address and the length it is good for are only meaningful together
    /// -- the caller clamps the guest's figure to the second before handing the driver the first.
    ///
    /// `None` for storage there is no address for: an allocation the driver keeps to itself, or
    /// one that is itself an alias. Only a scanout has an address before anyone asks; ordinary
    /// memory has one once it has been published, and the guest publishes before it imports,
    /// because the resource it names is the blob that publishing made.
    fn aliased_span(&self, id: ObjectId) -> Option<(usize, u64)> {
        let record = self.memory.get(&id)?;
        match &record.backing {
            Backing::Scanout(s) => Some((s.host_addr(), s.alloc_size())),
            Backing::Driver => Some((record.exported?, record.size)),
            Backing::Imported => None,
        }
    }

    /// Mint the IOSurface a scanout allocation lives in, if this allocation is one.
    ///
    /// A scanout is recognised by shape, not by a flag: the guest exports the memory to the
    /// outside world (`VkExportMemoryAllocateInfo`) and dedicates it to one image
    /// (`VkMemoryDedicatedAllocateInfo`). That is what a window buffer is, and nothing else in a
    /// venus stream looks like it.
    ///
    /// `None` at every step that cannot be answered honestly -- a format no IOSurface has, a
    /// driver that will not report a layout, a pitch the surface would not take. Every one of
    /// those leaves an ordinary allocation, which renders correctly and merely cannot be
    /// composited without a copy. A surface whose rows sit somewhere other than where the driver
    /// will write them is worse than no surface: it displays, and it displays sheared.
    fn scanout_surface(
        &mut self,
        device: VkDevice,
        info: &VkMemoryAllocateInfo,
    ) -> Option<Surface> {
        if !exports_memory(info.pNext) {
            return None;
        }
        let image = dedicated_image(info.pNext)?;
        let facts = *self.images.get(&image)?;
        let format = pixel_format(facts.format)?;

        let d = self.devices.get(&device)?;
        let layout = {
            let query = d.fns.try_vkGetImageSubresourceLayout()?;
            let subresource = VkImageSubresource {
                aspectMask: VkImageAspectFlags(
                    VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT.0 as u32,
                ),
                mipLevel: 0,
                arrayLayer: 0,
            };
            let mut layout = VkSubresourceLayout::default();
            // SAFETY: a device and an image this context created, and both structs are locals.
            unsafe { query(device, image, &subresource, &mut layout) };
            layout
        };
        let pitch = u32::try_from(layout.rowPitch.0).ok()?;
        if pitch == 0 {
            return None;
        }
        // The record says how wide the guest asked for; the live query says how the driver laid
        // it out. A record left behind by an image whose handle has since been recycled will not
        // describe this image, and this is where that shows: the rows would not add up.
        if layout.size.0 != u64::from(pitch) * u64::from(facts.height) {
            return None;
        }

        let surface = Surface::scanout(facts.width, facts.height, format, pitch).ok()?;
        // IOSurface may lay the rows out its own way. The allocation is about to be a
        // host-pointer import of these pages, so a pitch that is not the driver's is a surface
        // whose every row is at the wrong offset.
        if surface.bytes_per_row() != pitch {
            return None;
        }
        Some(surface)
    }

    /// Record what an image was created as, for a scanout allocation that has to match it.
    ///
    /// Kept because Vulkan will not answer for an image's extent or format after the fact, and
    /// forgotten in [`Self::forget_image`].
    pub fn note_image(&mut self, image: VkImage, info: &VkImageCreateInfo) {
        self.images.insert(
            image,
            ImageFacts {
                width: info.extent.width,
                height: info.extent.height,
                format: info.format,
            },
        );
    }

    /// Drop an image's record. Called from the two places Vulkan destroys an image: the guest's
    /// own `vkDestroyImage`, and the teardown that empties a device the guest left full.
    pub fn forget_image(&mut self, image: VkImage) {
        self.images.remove(&image);
    }

    /// Free device memory the guest named.
    ///
    /// The device and the handle come from the command the guest sent, resolved by the object
    /// table like every other input handle -- not from a record of this driver's own. The id is
    /// only the census entry to retire.
    pub fn free_memory(&mut self, device: VkDevice, memory: VkDeviceMemory, id: ObjectId) {
        let was = self.memory.remove(&id);
        let Some(d) = self.devices.get(&device) else {
            return;
        };
        // An exported allocation is still mapped -- the export handed the VMM that address and
        // left the mapping standing. `vkFreeMemory` would drop it implicitly, but the record that
        // owns the mapping is being retired here, so releasing it here is what keeps the two the
        // same act: nothing is left holding an address after the thing it named is gone.
        //
        // Only a driver mapping. A published scanout's address is its surface's, which the
        // surface owns and this record's drop releases; unmapping it would be undoing something
        // `vkMapMemory` never did.
        if was
            .as_ref()
            .is_some_and(|a| a.exported.is_some() && matches!(a.backing, Backing::Driver))
        {
            // SAFETY: the mapping this driver made in `memory_export` and has not released, on the
            // device that owns it. The record is out of the map, so it cannot be unmapped twice.
            unsafe { (d.fns.vkUnmapMemory())(device, memory) };
        }
        // SAFETY: a device and an allocation this context made; the object table took the id out
        // before this call, so the same handle cannot arrive twice.
        unsafe { (d.fns.vkFreeMemory())(device, memory, core::ptr::null()) };
    }

    /// Every live allocation the census is responsible for.
    ///
    /// Two kinds are not. Exported memory's bytes are the blob's, and the VMM captures them where
    /// they live rather than reading them a second time through here. Imported memory's bytes are
    /// the *exporter's*, censused where they are owned -- reading them here would report one
    /// buffer under two ids and read it through a mapping of memory this context does not own.
    ///
    /// Both are the same rule: the census reports storage once, at whoever owns it.
    pub fn memory_census(&self) -> Vec<Allocation> {
        self.memory
            .iter()
            .filter(|(_, a)| a.censused())
            .map(|(id, a)| Allocation { id: *id, size: a.size })
            .collect()
    }

    /// Map an allocation for the VMM to publish into the guest, and mark it exported.
    ///
    /// The address outlives this call, which is the whole point: the VMM maps it into the guest
    /// and reads and writes it for as long as the resource lives. It is *not* handed out again --
    /// exporting twice would give two resources one storage, and the second holder would have no
    /// way to know. The mapping is released when the allocation is freed, by the record that owns
    /// it, so an address for freed memory cannot be produced.
    ///
    /// Every refusal here is the guest's error, not ours: it names memory it never allocated,
    /// exports the same memory twice, asks for a blob larger than the allocation behind it, or
    /// asks to map memory the host was never able to address. None of them may stop the worker.
    ///
    /// The device and the handle are the caller's to resolve through the object table, exactly as
    /// [`Self::memory_read`] takes them, and for the same reason: this map does not keep a second
    /// copy of what the table already knows.
    pub fn memory_export(
        &mut self,
        device: VkDevice,
        handle: VkDeviceMemory,
        id: ObjectId,
        blob_size: u64,
    ) -> Result<Exported, ExportError> {
        let Some(record) = self.memory.get(&id) else {
            return Err(ExportError::NoSuchAllocation);
        };
        if record.exported.is_some() {
            return Err(ExportError::AlreadyExported);
        }
        if let Some(surface) = record.surface() {
            // A scanout is published as the surface it already is. There is nothing to map: the
            // pages are the surface's, the driver imported them, and the address is the one the
            // compositor will read the same bytes through.
            if blob_size > surface.alloc_size() {
                return Err(ExportError::LargerThanAllocation);
            }
            let addr = surface.host_addr();
            let write_back = record.write_back();
            self.memory.get_mut(&id).expect("the record was here a moment ago").exported =
                Some(addr);
            return Ok(Exported { addr, write_back });
        }
        if !record.host_visible() {
            return Err(ExportError::NotHostVisible);
        }
        // The VMM publishes the *blob's* size from this address, not the allocation's, so a blob
        // larger than what was reserved would put host memory past the end of the allocation into
        // the guest. `pad_for_blob` sizes an allocation up so this does not normally happen;
        // refuse rather than over-map on the guest's say-so if it ever does.
        if blob_size > record.size {
            return Err(ExportError::LargerThanAllocation);
        }
        let Some(d) = self.devices.get(&device) else {
            return Err(ExportError::NoSuchAllocation);
        };
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: a device and an allocation this context made, and `ptr` is a local. The mapping
        // is deliberately left standing -- see this function's contract.
        let r = unsafe {
            (d.fns.vkMapMemory())(
                device,
                handle,
                VkDeviceSize(0),
                VK_WHOLE_SIZE,
                VkMemoryMapFlags(0),
                &mut ptr,
            )
        };
        if r != VkResult::VK_SUCCESS || ptr.is_null() {
            return Err(ExportError::NotMappable);
        }
        let addr = ptr as usize;
        // Written back only now: until the map succeeds there is nothing to mark, and a mark
        // without an address is the disagreement `exported` exists to make impossible.
        let record = self.memory.get_mut(&id).expect("the record was here a moment ago");
        record.exported = Some(addr);
        Ok(Exported { addr, write_back: record.write_back() })
    }

    /// Where an allocation was exported to, if it has been.
    ///
    /// The VMM asks for this again after the create -- the C caches it on the resource, which is
    /// how a mapping outlives the memory it points into. Resolved through the live record instead,
    /// so memory the guest has freed has no address to give.
    pub fn memory_exported_at(&self, id: ObjectId) -> Option<Exported> {
        let a = self.memory.get(&id)?;
        Some(Exported { addr: a.exported?, write_back: a.write_back() })
    }

    /// Copy an allocation's contents out through a host mapping, returning how many bytes landed.
    ///
    /// Short buffers are the caller's business, not an error: the census reports whole sizes and
    /// the VMM caps what it reads, so a prefix is the normal request -- which is why the count
    /// comes back rather than being inferred from the buffer's length.
    /// The device and the handle are the caller's to resolve through the object table, which owns
    /// both; the size is this map's, and is the one thing the table does not know.
    pub fn memory_read(
        &self,
        device: VkDevice,
        handle: VkDeviceMemory,
        id: ObjectId,
        buf: &mut [u8],
    ) -> Result<usize, MemoryError> {
        let Some(record) = self.memory.get(&id) else {
            return Err(MemoryError::NoSuchAllocation);
        };
        // A scanout's bytes are the surface's, and only the surface can hand them over
        // coherently -- `vkMapMemory` has nothing to map, because the driver imported these pages
        // rather than allocating them.
        if let Some(surface) = record.surface() {
            return Ok(surface.read_into(buf));
        }
        let size = record.size;
        let Some(d) = self.devices.get(&device) else {
            return Err(MemoryError::NoSuchAllocation);
        };
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: a device and an allocation this context made, and `ptr` is a local.
        let r = unsafe {
            (d.fns.vkMapMemory())(
                device,
                handle,
                VkDeviceSize(0),
                VK_WHOLE_SIZE,
                VkMemoryMapFlags(0),
                &mut ptr,
            )
        };
        if r != VkResult::VK_SUCCESS || ptr.is_null() {
            return Err(MemoryError::NotMappable);
        }
        let n = buf.len().min(size as usize);
        // SAFETY: the driver mapped at least `mem.size` bytes at `ptr`, which is what `n` is
        // clamped to, and `buf` is a live slice of at least `n`. The two cannot overlap: one is
        // the driver's mapping and the other the caller's.
        unsafe { core::ptr::copy_nonoverlapping(ptr.cast::<u8>(), buf.as_mut_ptr(), n) };
        // SAFETY: the mapping this call just made, unmapped once.
        unsafe { (d.fns.vkUnmapMemory())(device, handle) };
        Ok(n)
    }
}

/// One live allocation, as this driver holds it.
///
/// Not what the census reports -- see [`Allocation`]. This is the record the driver keeps for its
/// own purposes, and it is where an export's mapping lives.
struct Allocated {
    /// Its size, padded to the blob the guest may map it as -- see [`pad_for_blob`].
    size: u64,
    /// What the bytes actually are. See [`Backing`].
    backing: Backing,
    /// The properties of the memory type it was allocated from.
    ///
    /// The flags themselves rather than the questions asked of them: an export must refuse memory
    /// the host cannot address, and the VMM must be told how the guest may cache it. Two answers
    /// derived from one recorded fact cannot drift apart the way two recorded booleans can.
    props: VkMemoryPropertyFlags,
    /// The host address this allocation was published to the VMM at, if it has been.
    ///
    /// `Some` *is* the export mark: one value, not a flag beside an address that could disagree
    /// with it. Where the address came from is [`Backing`]'s to say, and that is what decides
    /// whether freeing owes an unmap -- so nothing has to guess.
    ///
    /// The record owns whatever the address names, so retiring it on [`Driver::free_memory`] is
    /// the same act as making the address unreachable: there is no second place to purge, and
    /// nothing can hand the VMM a pointer into memory the guest has freed.
    exported: Option<usize>,
    /// What this allocation cost the host, held so that retiring the record credits it back.
    ///
    /// Never read, and that is the design: the charge is a value whose only job is to be dropped,
    /// so there is no release call for a future destroy path to forget.
    ///
    /// `None` for an import, which costs nothing: its bytes are the exporter's, charged where
    /// they were made. The same rule as [`Allocated::censused`], for the same reason -- storage
    /// is accounted once, at whoever owns it.
    #[expect(
        dead_code,
        reason = "held for its Drop -- crediting the ledger is this field going away"
    )]
    charge: Option<Charge>,
}

/// What an allocation's bytes are, which decides how it is published, read and freed.
///
/// One value rather than a bool per question. "Is it an import", "does it have a surface" and
/// "does freeing owe an unmap" are three readings of one fact, and as three fields two of them
/// could disagree about the same allocation.
enum Backing {
    /// Memory the driver allocated for this guest. Publishing it maps it, so freeing a published
    /// one owes the unmap.
    Driver,
    /// An IOSurface this renderer minted: the allocation is a host-pointer import of the
    /// surface's own pages, so the memory *is* the surface.
    ///
    /// Publishing hands out the surface's base address, which the surface owns -- there is
    /// nothing to unmap, and dropping this record is what releases it. Reading goes through
    /// [`crate::metal::Surface::read_into`], because a surface read without its lock sees
    /// whatever the CPU's view last held rather than what the GPU wrote.
    Scanout(Surface),
    /// Storage another context owns, which this allocation only aliases.
    ///
    /// A guest imports when one context has to reach what another rendered -- a compositor
    /// sampling a client's window. The census must not report it: one buffer under two ids, read
    /// through a mapping this context has no claim to.
    Imported,
}

/// What an image was created as, for a scanout surface that has to match it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ImageFacts {
    width: u32,
    height: u32,
    format: VkFormat,
}

impl Allocated {
    /// The surface behind it, for the one backing that has one.
    fn surface(&self) -> Option<&Surface> {
        match &self.backing {
            Backing::Scanout(s) => Some(s),
            Backing::Driver | Backing::Imported => None,
        }
    }

    /// Whether the census reports it.
    ///
    /// Storage is reported once, at whoever owns it. An import owns none. A published ordinary
    /// allocation is the VMM's to read through the blob it published as. A scanout is reported
    /// even when published, because the address the VMM got is a surface, and a surface read
    /// without its lock is not a read of what the GPU wrote -- this is the only place that can
    /// take that lock.
    fn censused(&self) -> bool {
        match self.backing {
            Backing::Driver => self.exported.is_none(),
            Backing::Scanout(_) => true,
            Backing::Imported => false,
        }
    }

    /// Whether the host can address it -- what an export needs before it may map anything.
    fn host_visible(&self) -> bool {
        self.props.0 & HOST_VISIBLE_BIT != 0
    }

    /// Whether the host reads and writes it through a write-back cache that stays coherent
    /// without explicit flushes. That is the one arrangement a guest may map cached; anything
    /// else it must map write-combining, or see writes the host has not published.
    fn write_back(&self) -> bool {
        self.props.0 & (HOST_COHERENT_BIT | HOST_CACHED_BIT)
            == (HOST_COHERENT_BIT | HOST_CACHED_BIT)
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

/// An allocation the guest published as a blob, as the VMM has to see it: one address, and how
/// the host's own caching of it constrains the guest's.
///
/// The two travel together because they are answered from the same record at the same moment. A
/// VMM that fetched the address and then asked separately how to cache it could be told about a
/// different allocation, or about one that had since been freed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Exported {
    /// Where it lives in this process.
    pub addr: usize,
    /// See [`Allocated::write_back`].
    pub write_back: bool,
}

/// Why an allocation could not be exported as a blob. Every one of these is the guest's doing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportError {
    /// No allocation under that id in this context.
    NoSuchAllocation,
    /// Already exported. A memory backs one blob: two resources sharing one storage is a bug
    /// neither of them could detect.
    AlreadyExported,
    /// The host cannot address this memory, so there is nothing to publish into the guest.
    NotHostVisible,
    /// The blob is bigger than the allocation behind it, and mapping it would publish whatever
    /// follows the allocation in this process.
    LargerThanAllocation,
    /// The driver refused to map memory it said was host-visible.
    NotMappable,
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

/// A `pNext` struct nameable by its own `sType`.
///
/// The constant is on the type rather than beside it because the two must agree: a caller that
/// passes the tag separately from the type it wants back is one typo away from reading a
/// `VkMemoryResourceAllocationSizePropertiesMESA` out of whatever struct happened to carry a
/// different tag. One value, on the type that is that value (CLAUDE.md).
///
/// # Safety
///
/// The implementor must be the exact struct venus-protocol decodes for `TYPE`, laid out as
/// `repr(C)` with the `sType`/`pNext` header first -- that is what makes the cast in
/// [`chained_mut`] sound.
pub unsafe trait OutStruct {
    const TYPE: VkStructureType;
}

/// The struct the guest chained onto an out-parameter, or `None` if it chained none.
///
/// A guest asking a query for more than the base struct says so by hanging a second struct off the
/// answer's `pNext`, and this is how a handler reaches it: as a borrow, so `context.rs` stays free
/// of unsafe. Safe to call for the reason stated at the top of this section -- every pointer these
/// take is one the decoder allocated in the batch arena, which outlives the whole submission.
///
/// It takes the `pNext` field itself rather than the struct that holds it, which is what makes the
/// borrow honest: the returned reference lives exactly as long as the exclusive borrow of the
/// chain it was found in.
/// The same claim for a struct the guest chained onto an *in*-parameter.
///
/// Separate from [`OutStruct`] because the two are read through different pointers and a type is
/// rarely both -- an in-struct is a request the guest wrote, an out-struct is an answer we fill.
///
/// # Safety
///
/// As [`OutStruct`]: implementing this asserts the tag names exactly this `repr(C)` type.
pub unsafe trait InStruct {
    const TYPE: VkStructureType;
}

/// The struct the guest chained onto a request, or `None` if it chained none.
///
/// The read-only mirror of [`chained_mut`], and it exists for the same reason: a handler that
/// wants what the guest hung off a `pNext` gets a borrow, so `context.rs` stays free of unsafe.
/// It takes the `pNext` field itself rather than the struct holding it, so the returned reference
/// lives exactly as long as the borrow of the chain it was found in.
pub fn chained<T: InStruct>(head: &*const core::ffi::c_void) -> Option<&T> {
    let mut node = (*head).cast::<VkBaseInStructure>();
    while !node.is_null() {
        // SAFETY: every link is a struct the decoder allocated in the batch arena, and every one
        // of them begins with the `sType`/`pNext` header `VkBaseInStructure` names.
        let base = unsafe { &*node };
        if base.sType == T::TYPE {
            // SAFETY: the tag says this node is a `T`, and `InStruct` is unsafe to implement
            // precisely so that claim is the implementor's to uphold.
            return Some(unsafe { &*node.cast::<T>() });
        }
        node = base.pNext;
    }
    None
}

// SAFETY: this is the struct venus-protocol decodes for that tag, generated `repr(C)` from the
// same vk.xml with Vulkan's `sType`/`pNext` header first.
unsafe impl InStruct for VkRingMonitorInfoMESA {
    const TYPE: VkStructureType = VkStructureType::VK_STRUCTURE_TYPE_RING_MONITOR_INFO_MESA;
}

pub fn chained_mut<T: OutStruct>(head: &mut *mut core::ffi::c_void) -> Option<&mut T> {
    let mut node = (*head).cast::<VkBaseOutStructure>();
    while !node.is_null() {
        // SAFETY: every link is a struct the decoder allocated in the batch arena, and every one
        // of them begins with the `sType`/`pNext` header `VkBaseOutStructure` names.
        let base = unsafe { &mut *node };
        if base.sType == T::TYPE {
            // SAFETY: the tag says this node is a `T`, and `OutStruct` is unsafe to implement
            // precisely so that claim is the implementor's to uphold.
            return Some(unsafe { &mut *node.cast::<T>() });
        }
        node = base.pNext;
    }
    None
}

// SAFETY: this is the struct venus-protocol decodes for that tag, generated `repr(C)` from the
// same vk.xml with Vulkan's `sType`/`pNext` header first.
unsafe impl OutStruct for VkMemoryResourceAllocationSizePropertiesMESA {
    const TYPE: VkStructureType =
        VkStructureType::VK_STRUCTURE_TYPE_MEMORY_RESOURCE_ALLOCATION_SIZE_PROPERTIES_MESA;
}

/// Whether an allocation's `pNext` chain says the memory is for the world outside this guest.
fn exports_memory(node: *const core::ffi::c_void) -> bool {
    chain_has(node, VkStructureType::VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO)
}

/// The image an allocation is dedicated to, if it is dedicated to one.
///
/// A dedicated allocation backs exactly one image, which is what makes it the image whose layout
/// a scanout surface must match. `VK_NULL_HANDLE` is the legal way to say "a buffer, not an
/// image", and reads as no image rather than as image zero.
fn dedicated_image(mut node: *const core::ffi::c_void) -> Option<VkImage> {
    while !node.is_null() {
        // SAFETY: every link is a struct the decoder allocated in the batch arena, and every one
        // of them begins with the `sType`/`pNext` header `VkBaseInStructure` names.
        let base = unsafe { &*node.cast::<VkBaseInStructure>() };
        if base.sType == VkStructureType::VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO {
            // SAFETY: the tag says this link is a `VkMemoryDedicatedAllocateInfo`.
            let ded = unsafe { &*node.cast::<VkMemoryDedicatedAllocateInfo>() };
            return (ded.image.0 != 0).then_some(ded.image);
        }
        node = base.pNext.cast();
    }
    None
}

/// The IOSurface format a Vulkan format is, for the formats a scanout can be.
///
/// `None` is not a failure: it is a format no IOSurface has, and an image in one is simply not a
/// window buffer. The four 8-bit spellings a compositor presents in, and no more -- guessing at
/// the rest would mint surfaces whose bytes mean something other than what they say. sRGB and
/// UNORM are the same bytes under different reading rules, which is the image view's business
/// and not the surface's.
fn pixel_format(format: VkFormat) -> Option<PixelFormat> {
    match format {
        VkFormat::VK_FORMAT_B8G8R8A8_UNORM | VkFormat::VK_FORMAT_B8G8R8A8_SRGB => {
            Some(PixelFormat::Bgra)
        }
        VkFormat::VK_FORMAT_R8G8B8A8_UNORM | VkFormat::VK_FORMAT_R8G8B8A8_SRGB => {
            Some(PixelFormat::Rgba)
        }
        _ => None,
    }
}

/// Whether a `pNext` chain carries a link of this type.
fn chain_has(mut node: *const core::ffi::c_void, ty: VkStructureType) -> bool {
    while !node.is_null() {
        // SAFETY: as `imports_a_resource`.
        let base = unsafe { &*node.cast::<VkBaseInStructure>() };
        if base.sType == ty {
            return true;
        }
        node = base.pNext.cast();
    }
    false
}

/// The resource an allocation's `pNext` chain names, when it is aliasing storage rather than
/// asking for some.
///
/// Walked rather than asked of the guest, because the chain is where the guest put it. Resource
/// zero is no resource, the same way a null handle is no image.
fn imported_resource(mut node: *const core::ffi::c_void) -> Option<ResourceHandle> {
    while !node.is_null() {
        // SAFETY: every link is a struct the decoder allocated in the batch arena, and every one
        // of them begins with the `sType`/`pNext` header `VkBaseInStructure` names.
        let base = unsafe { &*node.cast::<VkBaseInStructure>() };
        if base.sType == VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA {
            // SAFETY: the tag says this link is a `VkImportMemoryResourceInfoMESA`.
            let import = unsafe { &*node.cast::<VkImportMemoryResourceInfoMESA>() };
            return ResourceHandle::new(import.resourceId);
        }
        node = base.pNext.cast();
    }
    None
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
    use super::super::cs::HostHandle;
    use super::super::proto::types::{VkCommandBuffer, VkCommandPool};
    use super::*;

    /// The name of an advertised extension, as the guest reads it back.
    fn names(props: &[VkExtensionProperties]) -> Vec<String> {
        props
            .iter()
            .map(|p| {
                p.extensionName.iter().take_while(|c| **c != 0).map(|c| *c as u8 as char).collect()
            })
            .collect()
    }

    /// The two extensions the Metal path emulates are advertised together, and only where the
    /// emulation exists.
    ///
    /// Together is the whole of it. Mesa derives its renderer handle type from
    /// `VK_EXT_external_memory_dma_buf` alone, so a guest told only about the fd extension
    /// concludes there is no external memory at all -- and a compositor then exports no scanout
    /// and presents nothing, with every process still reporting itself healthy. Measured against
    /// KosmicKrisp: with neither advertised the synoik guest never leaves the boot console.
    #[test]
    fn the_extensions_the_metal_path_emulates_are_advertised_in_pairs() {
        const METAL: VkPhysicalDevice = VkPhysicalDevice(1);
        const NATIVE: VkPhysicalDevice = VkPhysicalDevice(2);
        const NEITHER: VkPhysicalDevice = VkPhysicalDevice(3);

        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_extensions(METAL, &["VK_EXT_external_memory_metal"]);
        // A driver with the real thing needs no emulation, and must not be told about it twice.
        driver.plant_extensions(
            NATIVE,
            &["VK_EXT_external_memory_metal", "VK_KHR_external_memory_fd"],
        );
        driver.plant_extensions(NEITHER, &["VK_KHR_external_fence_fd"]);

        let advertised = names(&driver.advertised_extensions(METAL));
        for want in EMULATED_ON_THE_HOST {
            assert_eq!(
                advertised.iter().filter(|n| *n == want).count(),
                1,
                "{want} is what the Metal path provides, and the guest is told so exactly once",
            );
        }

        let native = names(&driver.advertised_extensions(NATIVE));
        assert_eq!(
            native.iter().filter(|n| *n == "VK_KHR_external_memory_fd").count(),
            1,
            "a driver that has it natively is not also injected with it",
        );

        let neither = names(&driver.advertised_extensions(NEITHER));
        for want in EMULATED_ON_THE_HOST {
            assert!(
                !neither.contains(&want.to_string()),
                "{want} is not claimed where nothing emulates it",
            );
        }
    }

    /// A guest chains what it wants onto an answer's `pNext`, in whatever order it likes, and
    /// the struct a handler is after is rarely the first link. A walk that stops at the head
    /// finds it exactly when the guest happened to put it there, which is not a contract.
    #[test]
    fn a_chained_struct_is_found_wherever_the_guest_hung_it() {
        use super::super::proto::types::{VkBaseOutStructure, VkStructureType};

        let mut want = VkMemoryResourceAllocationSizePropertiesMESA {
            sType:
                VkStructureType::VK_STRUCTURE_TYPE_MEMORY_RESOURCE_ALLOCATION_SIZE_PROPERTIES_MESA,
            ..Default::default()
        };
        // Two links the walk has to step over first, one of them carrying a tag that is not ours.
        let mut second = VkBaseOutStructure {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_RESOURCE_PROPERTIES_MESA,
            pNext: (&mut want) as *mut _ as *mut VkBaseOutStructure,
        };
        let mut first = VkBaseOutStructure {
            sType: VkStructureType::VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: (&mut second) as *mut _,
        };
        let mut head = (&mut first) as *mut _ as *mut core::ffi::c_void;

        let found = chained_mut::<VkMemoryResourceAllocationSizePropertiesMESA>(&mut head)
            .expect("the third link is still on the chain");
        found.allocationSize = 0x1234;
        assert_eq!(want.allocationSize, 0x1234, "the borrow writes into the guest's own struct");

        // And a chain without it says so, rather than handing back the nearest thing.
        first.pNext = core::ptr::null_mut();
        let mut head = (&mut first) as *mut _ as *mut core::ffi::c_void;
        assert!(chained_mut::<VkMemoryResourceAllocationSizePropertiesMESA>(&mut head).is_none());
    }

    const HOST_VISIBLE: VkMemoryPropertyFlags = VkMemoryPropertyFlags(
        VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.0 as u32,
    );
    const DEVICE_LOCAL: VkMemoryPropertyFlags = VkMemoryPropertyFlags(
        VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.0 as u32,
    );

    /// An allocation is freed on the way out, after everything bound to it, and the census forgets
    /// it at the same moment.
    ///
    /// Two facts about one allocation used to live in two places: the object table held its handle
    /// under the guest's id, and this driver held a second copy of that handle beside the size.
    /// The cascade had to skip `VK_OBJECT_TYPE_DEVICE_MEMORY` entirely to avoid freeing both, and
    /// An export publishes memory to the VMM, and the mark it leaves is the mapping itself.
    ///
    /// Every refusal here is a guest's doing, so each has to be an answer rather than an abort:
    /// naming memory that does not exist, exporting the same memory twice, asking for a blob
    /// The census reports storage once, at whoever owns it -- and an import owns none. A guest
    /// imports so one context can reach what another rendered, and the second Vulkan handle onto
    /// those bytes is not a second buffer. Counting it would report the exporter's window twice
    /// and read it through a mapping this context has no claim to.
    ///
    /// Pinned against the corpus rather than against the C's source: `synoik` allocates exactly
    /// two imports, and they are exactly the two allocations the C's census omits and ours did
    /// not.
    #[test]
    fn the_census_does_not_report_storage_a_context_only_borrows() {
        const OWNED: ObjectId = ObjectId(20);
        const BORROWED: ObjectId = ObjectId(21);
        const SIZE: u64 = 4_096_000;

        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_allocation(OWNED, SIZE);
        driver.plant_imported_allocation(BORROWED, SIZE);

        let census = driver.memory_census();
        assert_eq!(census.len(), 1, "the borrowed one is the exporter's to report");
        assert_eq!(census[0].id, OWNED);
        assert_eq!(census[0].size, SIZE);

        driver.abandon_planted();
    }

    /// An import is the same storage under a second handle, so what it resolves to has to be an
    /// address that already exists -- and the length it is good for, in the same answer. The two
    /// are what the allocation is clamped against, and a length that could travel separately from
    /// its address is the pair this tree spells as one value.
    ///
    /// Three backings, three answers: a scanout has an address from the moment it is minted, a
    /// driver allocation has one only once it has been published, and an import has none of its
    /// own to lend.
    #[test]
    fn an_import_resolves_to_storage_that_already_exists() {
        let mut d = Driver::new(Account::for_test(None));

        let surface = Surface::scanout(64, 8, PixelFormat::Bgra, 256).expect("the system minted");
        let addr = surface.host_addr();
        let extent = surface.alloc_size();
        d.plant_scanout_allocation(ObjectId(66), surface);
        assert_eq!(
            d.aliased_span(ObjectId(66)),
            Some((addr, extent)),
            "a scanout lends the surface's own pages, and how far they run"
        );

        d.plant_allocation(ObjectId(70), 4096);
        assert_eq!(
            d.aliased_span(ObjectId(70)),
            None,
            "an allocation nobody has published has no address to lend"
        );

        d.plant_imported_allocation(ObjectId(71), 4096);
        assert_eq!(
            d.aliased_span(ObjectId(71)),
            None,
            "and an import lends nothing: the storage is not its to offer twice"
        );

        assert_eq!(d.aliased_span(ObjectId(999)), None, "nor does an id that names nothing");

        d.abandon_planted();
    }

    /// A `pNext` chain naming another allocation's resource is what makes an allocation an
    /// import, and the guest may hang it anywhere in the chain. Reading only the head would find
    /// it exactly when the guest happened to put it first, which is not a contract.
    ///
    /// Which resource, not whether: the number is what the renderer resolves to the storage the
    /// two allocations then share, so a walk that found the link and lost the id would leave the
    /// import aliasing nothing.
    #[test]
    fn an_import_yields_the_resource_it_names_wherever_the_guest_hung_it() {
        use super::super::proto::types::{
            VkBaseInStructure, VkImportMemoryResourceInfoMESA, VkStructureType,
        };

        let mut tail = VkImportMemoryResourceInfoMESA {
            sType: VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA,
            pNext: core::ptr::null(),
            resourceId: 7,
        };
        let mut head = VkBaseInStructure {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            pNext: (&raw const tail).cast(),
        };
        assert_eq!(
            imported_resource((&raw const head).cast()),
            ResourceHandle::new(7),
            "found past the head, with its resource"
        );

        head.pNext = core::ptr::null();
        assert_eq!(imported_resource((&raw const head).cast()), None, "and none when absent");

        // Resource zero is how the wire spells "no resource" -- it must not resolve to a handle,
        // because a handle is what the renderer would then go looking for.
        tail.resourceId = 0;
        head.pNext = (&raw const tail).cast();
        assert_eq!(imported_resource((&raw const head).cast()), None);
    }

    /// The budget's three answers, at the one call that asks it.
    ///
    /// A refusal must land *before* `vkAllocateMemory`, not after: refusing an allocation the host
    /// has already made costs the memory it was meant to save, which is the whole point of the
    /// cap. The counter is what proves it -- an implementation that allocated first and credited
    /// back would pass every assertion about the ledger and none about this.
    #[test]
    fn an_allocation_over_the_budget_never_reaches_the_driver() {
        use std::cell::Cell;

        const DEVICE: VkDevice = VkDevice(3);
        const CAP: u64 = 1000;
        const SIZE: u64 = 600;

        thread_local! {
            static ASKED: Cell<u32> = const { Cell::new(0) };
        }

        unsafe extern "C" fn allocate(
            _d: VkDevice,
            _info: *const VkMemoryAllocateInfo,
            _a: *const VkAllocationCallbacks,
            out: *mut VkDeviceMemory,
        ) -> VkResult {
            ASKED.with(|n| n.set(n.get() + 1));
            // SAFETY: the caller's local.
            unsafe { *out = VkDeviceMemory(0x9000) };
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn free(
            _d: VkDevice,
            _m: VkDeviceMemory,
            _a: *const VkAllocationCallbacks,
        ) {
        }

        let mut d = Driver::new(Account::for_test(Some(CAP)));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateMemory(allocate);
        fns.plant_vkFreeMemory(free);
        d.plant_device(DEVICE, fns);
        // Not host-visible, so nothing is padded and the ledger's numbers are the guest's own.
        d.plant_memory_types(DEVICE, &[VkMemoryPropertyFlags(0)]);

        let ask = |d: &mut Driver, id: u64| {
            let info = VkMemoryAllocateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                pNext: core::ptr::null(),
                allocationSize: VkDeviceSize(SIZE),
                memoryTypeIndex: 0,
            };
            d.allocate_memory(DEVICE, ObjectId(id), &info, None, &|_| None)
        };

        assert!(ask(&mut d, 1).is_ok(), "the first fits under the cap");
        assert_eq!(d.account.live(), SIZE);
        assert_eq!(ASKED.with(Cell::get), 1);

        match ask(&mut d, 2) {
            Err(NoMemory::OverBudget { stop: true }) => {}
            other => {
                panic!("the second is over the cap and stops the context, not {:?}", other.is_ok())
            }
        }
        assert_eq!(ASKED.with(Cell::get), 1, "and the driver was never asked for it");
        assert_eq!(d.account.live(), SIZE, "so the ledger is where it was");

        // Freeing is the only thing that credits, and it does so by retiring the record -- there
        // is no release call anywhere in `free_memory` for this to be testing instead.
        d.free_memory(DEVICE, VkDeviceMemory(0x9000), ObjectId(1));
        assert_eq!(d.account.live(), 0, "the room comes back with the allocation");
        assert!(ask(&mut d, 3).is_ok(), "and the next one fits again");
        assert_eq!(ASKED.with(Cell::get), 2);

        d.abandon_planted();
    }

    /// A driver refusal is not a budget refusal. The guest unwinds from the first the way it would
    /// on hardware, and the charge taken before the call has to go back -- which it does by going
    /// out of scope, so there is no error path that can forget it.
    #[test]
    fn a_driver_that_refuses_costs_the_budget_nothing() {
        const DEVICE: VkDevice = VkDevice(3);

        unsafe extern "C" fn refuse(
            _d: VkDevice,
            _info: *const VkMemoryAllocateInfo,
            _a: *const VkAllocationCallbacks,
            _out: *mut VkDeviceMemory,
        ) -> VkResult {
            VkResult::VK_ERROR_OUT_OF_DEVICE_MEMORY
        }

        let mut d = Driver::new(Account::for_test(Some(1000)));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateMemory(refuse);
        d.plant_device(DEVICE, fns);
        d.plant_memory_types(DEVICE, &[VkMemoryPropertyFlags(0)]);

        let info = VkMemoryAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: core::ptr::null(),
            allocationSize: VkDeviceSize(900),
            memoryTypeIndex: 0,
        };
        match d.allocate_memory(DEVICE, ObjectId(1), &info, None, &|_| None) {
            Err(NoMemory::Driver(r)) => {
                assert_eq!(r, VkResult::VK_ERROR_OUT_OF_DEVICE_MEMORY, "the driver's own answer")
            }
            _ => panic!("the driver refused, so this is not the budget's refusal"),
        }
        assert_eq!(d.account.live(), 0, "and nothing is left charged for memory that never was");

        d.abandon_planted();
    }

    /// A scanout is charged at the surface's own extent, not at the number in the request.
    ///
    /// The surface is the commitment: IOSurface rounds an allocation up to whole pages, and those
    /// pages are the host memory that is actually gone. The `VkDeviceMemory` on top of it is a
    /// host-pointer import of those same pages and commits nothing further -- so the guest's
    /// figure is the wrong number to bill, and it is the smaller one, which is the direction that
    /// lets a leak run past the cap.
    #[test]
    fn a_scanout_is_charged_for_the_pages_the_surface_took() {
        use super::super::proto::types::{
            VkExportMemoryAllocateInfo, VkExtent3D, VkExternalMemoryHandleTypeFlags, VkFormat,
            VkImageCreateInfo, VkMemoryDedicatedAllocateInfo,
        };

        const DEVICE: VkDevice = VkDevice(3);
        const IMAGE: VkImage = VkImage(0x4100);
        const W: u32 = 64;
        const H: u32 = 8;
        const PITCH: u64 = (W * 4) as u64;
        /// What the guest asks for: the rows, and nothing for the page the surface rounds to.
        const ASKED: u64 = PITCH * H as u64;

        unsafe extern "C" fn layout(
            _d: VkDevice,
            _i: VkImage,
            _s: *const VkImageSubresource,
            out: *mut VkSubresourceLayout,
        ) {
            // SAFETY: the caller's local.
            unsafe {
                *out = VkSubresourceLayout {
                    rowPitch: VkDeviceSize(PITCH),
                    size: VkDeviceSize(PITCH * H as u64),
                    ..Default::default()
                }
            };
        }
        unsafe extern "C" fn allocate(
            _d: VkDevice,
            _info: *const VkMemoryAllocateInfo,
            _a: *const VkAllocationCallbacks,
            out: *mut VkDeviceMemory,
        ) -> VkResult {
            // SAFETY: the caller's local.
            unsafe { *out = VkDeviceMemory(0x9000) };
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn free(
            _d: VkDevice,
            _m: VkDeviceMemory,
            _a: *const VkAllocationCallbacks,
        ) {
        }

        // What the system will hand back for this geometry, minted here so the assertion is
        // against the real rounding rather than a number this test made up.
        let extent = Surface::scanout(W, H, PixelFormat::Bgra, PITCH as u32)
            .expect("the system minted")
            .alloc_size();
        assert!(extent > ASKED, "the premise: IOSurface rounds up, so the two numbers differ");

        let mut d = Driver::new(Account::for_test(None));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateMemory(allocate);
        fns.plant_vkFreeMemory(free);
        fns.plant_vkGetImageSubresourceLayout(layout);
        d.plant_device(DEVICE, fns);
        d.plant_memory_types(DEVICE, &[VkMemoryPropertyFlags(0)]);
        d.note_image(
            IMAGE,
            &VkImageCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                format: VkFormat::VK_FORMAT_B8G8R8A8_UNORM,
                extent: VkExtent3D { width: W, height: H, depth: 1 },
                ..Default::default()
            },
        );

        // The shape a window buffer has: exported to the outside world, and dedicated to one
        // image. Nothing else in a venus stream looks like this.
        let dedicated = VkMemoryDedicatedAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            pNext: core::ptr::null(),
            image: IMAGE,
            buffer: VkBuffer(0),
        };
        let export = VkExportMemoryAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            pNext: (&raw const dedicated).cast(),
            handleTypes: VkExternalMemoryHandleTypeFlags(0),
        };
        let info = VkMemoryAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: (&raw const export).cast(),
            allocationSize: VkDeviceSize(ASKED),
            memoryTypeIndex: 0,
        };
        d.allocate_memory(DEVICE, ObjectId(1), &info, None, &|_| None).expect("no cap");
        assert!(d.memory_surface_id(ObjectId(1)).is_some(), "the premise: this minted a surface");
        assert_eq!(d.account.live(), extent, "charged for the pages, not for the rows");

        d.free_memory(DEVICE, VkDeviceMemory(0x9000), ObjectId(1));
        assert_eq!(d.account.live(), 0, "and the surface's pages come back with it");

        d.abandon_planted();
    }

    /// An import aliases bytes another allocation already paid for, so charging it would bill one
    /// buffer twice and refuse work the host has room for. The same rule as the census, which is
    /// why both read the one `Backing` rather than a flag each.
    #[test]
    fn an_import_is_not_charged_because_its_bytes_are_the_exporters() {
        use super::super::proto::types::VkImportMemoryResourceInfoMESA;

        const DEVICE: VkDevice = VkDevice(3);

        unsafe extern "C" fn allocate(
            _d: VkDevice,
            _info: *const VkMemoryAllocateInfo,
            _a: *const VkAllocationCallbacks,
            out: *mut VkDeviceMemory,
        ) -> VkResult {
            // SAFETY: the caller's local.
            unsafe { *out = VkDeviceMemory(0x9000) };
            VkResult::VK_SUCCESS
        }

        let mut d = Driver::new(Account::for_test(Some(1000)));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateMemory(allocate);
        d.plant_device(DEVICE, fns);
        d.plant_memory_types(DEVICE, &[VkMemoryPropertyFlags(0)]);

        // Nearly the whole cap is already spoken for, so an import that were charged at all
        // would be refused -- and one charged at its own size would be refused loudly.
        d.plant_allocation_of(ObjectId(1), 900, 0);
        assert_eq!(d.account.live(), 900);

        let import = VkImportMemoryResourceInfoMESA {
            sType: VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA,
            pNext: core::ptr::null(),
            resourceId: 7,
        };
        let info = VkMemoryAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: (&raw const import).cast(),
            allocationSize: VkDeviceSize(400),
            memoryTypeIndex: 0,
        };
        assert!(
            d.allocate_memory(DEVICE, ObjectId(2), &info, None, &|_| None).is_ok(),
            "an import is admitted with no room left, because it takes none"
        );
        assert_eq!(d.account.live(), 900, "and the ledger did not move");

        d.abandon_planted();
    }

    /// bigger than what backs it, or asking to map memory the host cannot address. The second of
    /// those is the one with teeth -- two resources over one storage is a state neither holder
    /// could detect afterwards.
    #[test]
    fn memory_is_published_once_and_leaves_the_census_when_it_is() {
        use std::cell::RefCell;

        const DEVICE: VkDevice = VkDevice(3);
        const MEM: ObjectId = ObjectId(12);
        const LOCAL: ObjectId = ObjectId(13);
        const SIZE: u64 = 128 * 1024;
        /// Any address will do; nothing dereferences it. Page-aligned so it reads like one.
        const ADDR: usize = 0x7000_0000;

        thread_local! {
            static MAPS: RefCell<u32> = const { RefCell::new(0) };
            static UNMAPS: RefCell<u32> = const { RefCell::new(0) };
        }

        unsafe extern "C" fn map(
            _d: VkDevice,
            _m: VkDeviceMemory,
            _o: VkDeviceSize,
            _s: VkDeviceSize,
            _f: VkMemoryMapFlags,
            out: *mut *mut core::ffi::c_void,
        ) -> VkResult {
            MAPS.with_borrow_mut(|n| *n += 1);
            // SAFETY: the caller passes a local of its own.
            unsafe { *out = ADDR as *mut core::ffi::c_void };
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn unmap(_d: VkDevice, _m: VkDeviceMemory) {
            UNMAPS.with_borrow_mut(|n| *n += 1);
        }
        unsafe extern "C" fn free(
            _d: VkDevice,
            _m: VkDeviceMemory,
            _a: *const VkAllocationCallbacks,
        ) {
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkMapMemory(map);
        fns.plant_vkUnmapMemory(unmap);
        fns.plant_vkFreeMemory(free);

        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(DEVICE, fns);
        driver.plant_allocation(MEM, SIZE);
        driver.plant_device_local_allocation(LOCAL, SIZE);
        let handle = VkDeviceMemory(0xd0);

        assert_eq!(driver.memory_census().len(), 2, "both are live and unexported");

        // Memory nobody allocated, refused before anything is mapped.
        assert_eq!(
            driver.memory_export(DEVICE, handle, ObjectId(999), SIZE),
            Err(ExportError::NoSuchAllocation)
        );
        // A blob bigger than the allocation would publish whatever follows it in this process.
        assert_eq!(
            driver.memory_export(DEVICE, handle, MEM, SIZE + 1),
            Err(ExportError::LargerThanAllocation)
        );
        // Memory the host cannot address has nothing to publish.
        assert_eq!(
            driver.memory_export(DEVICE, handle, LOCAL, SIZE),
            Err(ExportError::NotHostVisible)
        );
        MAPS.with_borrow(|n| assert_eq!(*n, 0, "not one of those reached the driver"));

        // The export itself. Coherent and cached on the host, so the guest may map it cached.
        let published = Exported { addr: ADDR, write_back: true };
        assert_eq!(driver.memory_export(DEVICE, handle, MEM, SIZE), Ok(published));
        MAPS.with_borrow(|n| assert_eq!(*n, 1));
        UNMAPS.with_borrow(|n| assert_eq!(*n, 0, "the mapping is the VMM's now and stays up"));
        assert_eq!(driver.memory_exported_at(MEM), Some(published), "asked again, not remembered");

        // Memory the host reaches through a cache it has to flush is memory the guest must not
        // map cached, and the answer comes from the type it was allocated from -- not from a
        // default that happens to be right for the driver we run on today.
        const UNCACHED: ObjectId = ObjectId(14);
        driver.plant_allocation_of(UNCACHED, SIZE, HOST_VISIBLE_BIT | HOST_COHERENT_BIT);
        assert_eq!(
            driver.memory_export(DEVICE, handle, UNCACHED, SIZE),
            Ok(Exported { addr: ADDR, write_back: false })
        );
        driver.free_memory(DEVICE, handle, UNCACHED);
        MAPS.with_borrow(|n| assert_eq!(*n, 2));
        UNMAPS.with_borrow(|n| assert_eq!(*n, 1));
        MAPS.with_borrow_mut(|n| *n = 1);
        UNMAPS.with_borrow_mut(|n| *n = 0);

        // The census stops reporting it: its bytes are the blob's, captured where they live.
        let census = driver.memory_census();
        assert_eq!(census.len(), 1, "the exported allocation is no longer the census's to read");
        assert_eq!(census[0].id, LOCAL);

        // And it cannot be published a second time.
        assert_eq!(
            driver.memory_export(DEVICE, handle, MEM, SIZE),
            Err(ExportError::AlreadyExported)
        );
        MAPS.with_borrow(|n| assert_eq!(*n, 1, "a refused export maps nothing"));

        // Freeing it releases the mapping the export left standing -- the record that owned the
        // address is gone, so this is the last moment it could be unmapped at all.
        driver.free_memory(DEVICE, handle, MEM);
        UNMAPS.with_borrow(|n| assert_eq!(*n, 1, "the export's mapping went with the allocation"));
        assert_eq!(driver.memory_exported_at(MEM), None, "and there is no address left to give");

        // The unexported one was never mapped, so freeing it must not unmap anything.
        driver.free_memory(DEVICE, handle, LOCAL);
        UNMAPS.with_borrow(|n| assert_eq!(*n, 1, "nothing unmaps memory that was never mapped"));

        driver.abandon_planted();
    }

    /// a skip arm that exists to work around a duplicated fact is the duplication still costing
    /// something. Now the handle comes from the table like every other, and the only thing left
    /// here is the size -- so the free happens in the cascade, and this is what pins its order.
    #[test]
    fn an_allocation_is_freed_after_what_was_bound_to_it_and_leaves_the_census() {
        use std::cell::RefCell;

        use super::super::proto::types::{VkAllocationCallbacks, VkBuffer};

        const DEVICE: VkDevice = VkDevice(3);
        const BUFFER: (u64, u64) = (11, 0xb0);
        const MEMORY: (u64, u64) = (12, 0xd0);

        thread_local! { static SAW: RefCell<Vec<(&'static str, u64)>> = const { RefCell::new(Vec::new()) }; }
        fn saw(what: &'static str, h: HostHandle) {
            SAW.with_borrow_mut(|s| s.push((what, h.0)));
        }

        unsafe extern "C" fn wait_idle(_d: VkDevice) -> VkResult {
            saw("wait", HostHandle(0));
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn buffer(_d: VkDevice, h: VkBuffer, _a: *const VkAllocationCallbacks) {
            saw("buffer", h.host());
        }
        unsafe extern "C" fn free(
            _d: VkDevice,
            h: VkDeviceMemory,
            _a: *const VkAllocationCallbacks,
        ) {
            saw("free", h.host());
        }
        unsafe extern "C" fn device(h: VkDevice, _a: *const VkAllocationCallbacks) {
            saw("device", h.host());
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkDeviceWaitIdle(wait_idle);
        fns.plant_vkDestroyBuffer(buffer);
        fns.plant_vkFreeMemory(free);
        fns.plant_vkDestroyDevice(device);

        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(DEVICE, fns);
        driver.plant_allocation(ObjectId(MEMORY.0), 4096);
        assert_eq!(driver.memory_census().len(), 1, "the allocation is live before the teardown");

        // The order the doomed list arrives in is the arena's, not Vulkan's: memory first here on
        // purpose, so a teardown that simply walked the list would free it under a live buffer.
        let doomed = [
            Doomed {
                id: ObjectId(MEMORY.0),
                ty: VkObjectType::VK_OBJECT_TYPE_DEVICE_MEMORY,
                handle: HostHandle(MEMORY.1),
                device: Some(DEVICE),
            },
            Doomed {
                id: ObjectId(BUFFER.0),
                ty: VkObjectType::VK_OBJECT_TYPE_BUFFER,
                handle: HostHandle(BUFFER.1),
                device: Some(DEVICE),
            },
            Doomed {
                id: ObjectId(DEVICE.0),
                ty: VkObjectType::VK_OBJECT_TYPE_DEVICE,
                handle: DEVICE.host(),
                device: None,
            },
        ];
        // The guest's own destroy, not the context teardown: teardown ends by dropping any
        // allocation it could not attribute to a device, and that fallback would hide whether the
        // cascade retired this one.
        assert!(driver.destroy_device(DEVICE, &doomed).is_empty(), "it owned no pools");

        SAW.with_borrow(|s| {
            assert_eq!(
                s.as_slice(),
                [("wait", 0), ("buffer", BUFFER.1), ("free", MEMORY.1), ("device", DEVICE.0)],
                "idle, then what is bound to the memory, then the memory, then the device"
            );
        });
        assert!(driver.memory_census().is_empty(), "and the census stops reporting it");
        assert!(driver.owes_nothing(), "a teardown that returns owes Vulkan nothing");
    }

    /// Destroying a pool destroys everything in it, and the guest sends no command per object --
    /// so the guest ids of its contents have to come back here, for the caller to take out of
    /// the object table. Left there, the table goes on resolving one to a handle Vulkan has
    /// freed and the next command naming it hands that handle back to the driver.
    #[test]
    fn a_destroyed_pool_hands_back_the_ids_of_everything_in_it() {
        const DEVICE: VkDevice = VkDevice(3);

        let mut d = Driver::new(Account::for_test(None));
        d.pools.open(DEVICE, VkCommandPool(7));
        d.pools.adopt(
            VkCommandPool(7),
            [(VkCommandBuffer(11), ObjectId(110)), (VkCommandBuffer(12), ObjectId(120))],
        );

        // No device is registered, so the driver call itself is skipped -- the bookkeeping is
        // what is under test, and it has to happen either way.
        let mut orphans =
            d.destroy_pool(DEVICE, |f| f.vkDestroyCommandPool(), VkCommandPool(7), None);
        orphans.sort_unstable_by_key(|i| i.0);
        assert_eq!(orphans, [ObjectId(110), ObjectId(120)], "every id in the pool, and no other");
        assert!(!d.pools.is_open(VkCommandPool(7)));

        // And a second destroy of the same pool has nothing left to hand back: the ids must not
        // be removed from the object table twice, because the guest may have reused them.
        assert!(
            d.destroy_pool(DEVICE, |f| f.vkDestroyCommandPool(), VkCommandPool(7), None).is_empty()
        );
    }

    /// Two pool kinds may share a handle value, and must not share a record.
    ///
    /// A command pool and a descriptor pool are separate Vulkan handle spaces, and a driver is
    /// free to hand out the same `u64` in each -- non-dispatchable handles routinely are small
    /// indices. Keyed by the bare handle the two collide: opening the second overwrites the
    /// first, and destroying either hands back the other's ids, which the caller then takes out
    /// of the object table while the objects they name are still live.
    #[test]
    fn two_pools_of_different_kinds_may_share_a_handle_value() {
        use super::super::proto::types::{VkDescriptorPool, VkDescriptorSet};

        const SHARED: u64 = 7;

        let mut d = Driver::new(Account::for_test(None));
        d.pools.open(VkDevice(3), VkCommandPool(SHARED));
        d.pools.adopt(VkCommandPool(SHARED), [(VkCommandBuffer(11), ObjectId(110))]);
        d.pools.open(VkDevice(3), VkDescriptorPool(SHARED));
        d.pools.adopt(VkDescriptorPool(SHARED), [(VkDescriptorSet(21), ObjectId(210))]);

        assert!(
            d.pools.is_open(VkCommandPool(SHARED)),
            "the command pool survived the second open"
        );
        assert!(d.pools.is_open(VkDescriptorPool(SHARED)));

        // Each destroy hands back its own contents and nothing of the other's.
        assert_eq!(d.pools.close(VkCommandPool(SHARED)), [ObjectId(110)]);
        assert!(d.pools.is_open(VkDescriptorPool(SHARED)), "the descriptor pool outlived it");
        assert_eq!(d.pools.close(VkDescriptorPool(SHARED)), [ObjectId(210)]);
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
        let mut d = Driver::new(Account::for_test(None));
        d.pools.open(VkDevice(3), VkCommandPool(7));
        d.pools.adopt(
            VkCommandPool(7),
            [(VkCommandBuffer(11), ObjectId(110)), (VkCommandBuffer(12), ObjectId(120))],
        );
        d.pools.open(VkDevice(4), VkCommandPool(8));
        d.pools.adopt(VkCommandPool(8), [(VkCommandBuffer(21), ObjectId(210))]);

        let mut orphans = d.destroy_device(VkDevice(3), &[]);
        orphans.sort_unstable_by_key(|i| i.0);

        assert_eq!(orphans, [ObjectId(110), ObjectId(120)], "its pools' ids, and no others");
        assert!(!d.pools.is_open(VkCommandPool(7)), "a pool outlived its device");
        assert!(d.pools.is_open(VkCommandPool(8)), "another device's pool must be untouched");
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
