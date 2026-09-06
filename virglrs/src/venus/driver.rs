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
use super::cs::{Handle, HostHandle, ObjectId, PoolOf, TypedHandle};
use super::objects::Doomed;
use super::proto::types::{
    VkAllocationCallbacks, VkBaseInStructure, VkBaseOutStructure, VkBool32, VkBuffer, VkBufferCopy,
    VkBufferImageCopy, VkBufferMemoryBarrier, VkBufferView, VkClearAttachment, VkClearColorValue,
    VkClearRect, VkCommandBuffer, VkCommandBufferBeginInfo, VkCommandBufferResetFlags,
    VkCommandPool, VkCopyDescriptorSet, VkDependencyFlags, VkDescriptorPool, VkDescriptorSet,
    VkDescriptorSetLayout, VkDescriptorUpdateTemplate, VkDevice, VkDeviceCreateInfo,
    VkDeviceMemory, VkDeviceQueueInfo2, VkDeviceSize, VkEvent, VkExportMemoryAllocateInfo,
    VkExtensionProperties, VkExternalMemoryHandleTypeFlagBits, VkExternalMemoryImageCreateInfo,
    VkExternalSemaphoreHandleTypeFlagBits, VkFence, VkFilter, VkFormat, VkFramebuffer, VkImage,
    VkImageAspectFlagBits, VkImageAspectFlags, VkImageBlit, VkImageCopy, VkImageCreateFlags,
    VkImageCreateInfo, VkImageFormatProperties, VkImageLayout, VkImageMemoryBarrier,
    VkImageSubresource, VkImageSubresourceRange, VkImageTiling, VkImageType, VkImageUsageFlagBits,
    VkImageUsageFlags, VkImageView, VkImportMemoryHostPointerInfoEXT,
    VkImportMemoryResourceInfoMESA, VkImportSemaphoreFdInfoKHR, VkInstance, VkInstanceCreateInfo,
    VkMemoryAllocateInfo, VkMemoryBarrier, VkMemoryDedicatedAllocateInfo, VkMemoryMapFlags,
    VkMemoryPropertyFlagBits, VkMemoryPropertyFlags, VkMemoryResourceAllocationSizePropertiesMESA,
    VkObjectType, VkPhysicalDevice, VkPhysicalDeviceMemoryProperties, VkPipeline,
    VkPipelineBindPoint, VkPipelineCache, VkPipelineLayout, VkPipelineStageFlagBits,
    VkPipelineStageFlags, VkQueryControlFlags, VkQueryPool, VkQueryPoolCreateInfo,
    VkQueryResultFlagBits, VkQueryResultFlags, VkQueryType, VkQueue, VkRect2D, VkRenderPass,
    VkRenderPassBeginInfo, VkResult, VkRingMonitorInfoMESA, VkSampleCountFlagBits, VkSampler,
    VkSamplerYcbcrConversion, VkSemaphore, VkSemaphoreGetFdInfoKHR, VkSemaphoreImportFlagBits,
    VkShaderModule, VkShaderStageFlags, VkStructureType, VkSubmitInfo, VkSubpassContents,
    VkSubresourceLayout, VkViewport, VkWriteDescriptorSet,
};
use std::sync::Arc;

use super::ring::ResourceBytes;
use crate::guest_mem::GuestMap;
use crate::ids::ResourceHandle;
use crate::ids::SurfaceId;
use crate::metal::{Held, PixelFormat, Surface};
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

    /// Whether every one of `children` was allocated from `pool`, and is still held by it.
    ///
    /// The one question a free has to ask before the driver is: Vulkan's free takes the pool and
    /// the objects as separate arguments, and a guest that pairs them wrongly is asking for
    /// undefined behaviour on the host. The table already knows the answer, per child.
    fn all_from<P: PoolOf>(&self, pool: P, children: &[P::Child]) -> bool {
        let pool = TypedHandle::of(pool);
        children.iter().all(|c| self.owner.get(&TypedHandle::of(*c)) == Some(&pool))
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

    /// Empty a pool without closing it: what a *reset* does, as against the destroy `close` serves.
    ///
    /// The pool stays open and keeps taking allocations; everything it handed out stops existing.
    fn recycle<P: Handle>(&mut self, pool: P) -> Vec<ObjectId> {
        let Some(p) = self.open.get_mut(&TypedHandle::of(pool)) else {
            return Vec::new();
        };
        let children = std::mem::take(&mut p.children);
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

/// Why a query command was not put to the driver.
///
/// Every command that names a query by index is measured against the pool's record first,
/// because the driver takes the index on the caller's word: on this host a reset writes host
/// memory at it, a begin reads a remap table at it, and the recorded ones address the pool's
/// buffer at it on the GPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueryRefused {
    /// The device named has no table here, or the command buffer no device behind it.
    NoDevice,
    /// The device exports no `vkResetQueryPool`, having advertised the feature that needs it.
    NoHostReset,
    /// The pool has no record here: created before this renderer kept one, or never by it.
    UnknownPool,
    /// The queries named run past the pool's end.
    OutOfPool,
    /// The results asked for run past the room offered.
    OutOfRoom,
    /// A pool whose result this renderer cannot size, so it cannot hold the read to the room.
    Unsized,
}

/// Why a run of pool objects was not freed. Neither reached the driver.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FreeRefused {
    /// The device named has no table here.
    NoDevice,
    /// At least one object in the run is not the pool's -- another pool's, or already freed.
    NotFromThisPool,
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
    /// What each query pool answers with, for the read-back that has to fit the room the guest
    /// offered. Keyed by host handle for the same reason as `images`, and kept honest the same
    /// way: the record dies at both places the pool does, so a recycled handle finds no record
    /// from a previous life.
    query_pools: BTreeMap<VkQueryPool, QueryFacts>,
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
            query_pools: BTreeMap::new(),
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
    /// A pipeline cache's serialised contents, in Vulkan's count-then-fill shape -- but counted
    /// in bytes, with a `size_t` where the enumerations have a `u32`, and on the device table.
    ///
    /// The guest saves this to disk between runs, so it is asked a few seconds after every new
    /// pipeline, on every client that has a cache at all.
    pub fn pipeline_cache_data(
        &self,
        device: VkDevice,
        cache: VkPipelineCache,
        out: Option<&mut [u8]>,
    ) -> Result<(usize, VkResult), VkResult> {
        let f = self
            .devices
            .get(&device)
            .and_then(|d| d.fns.try_vkGetPipelineCacheData())
            .ok_or(VkResult::VK_ERROR_INITIALIZATION_FAILED)?;
        let (mut n, room, data) = match out {
            Some(s) => (s.len(), Some(s.len()), s.as_mut_ptr().cast::<core::ffi::c_void>()),
            None => (0, None, core::ptr::null_mut()),
        };
        // SAFETY: a device and a cache this context made, and `n` is initialised to the length of
        // the buffer `data` points at -- the pair the caller handed us as one slice.
        let r = unsafe { f(device, cache, &mut n, data) };
        if let Some(room) = room {
            assert!(n <= room, "the driver wrote more cache data than the room it was given");
        }
        Ok((n, r))
    }

    // ---------------------------------------------------------------------- query pools
    //
    // A pool of GPU-side counters the guest reads back through `vkGetQueryPoolResults`, into a
    // buffer it sized itself. Vulkan makes fitting the results the caller's promise and reads
    // off the end undefined; here the caller is a guest, so the pool is recorded at create and
    // the read is measured against the record before the driver is handed the buffer.

    /// `vkCreateQueryPool`, and the record of what its queries will answer with.
    pub fn create_query_pool(
        &mut self,
        device: VkDevice,
        info: &VkQueryPoolCreateInfo,
        alloc: Option<&VkAllocationCallbacks>,
    ) -> Result<VkQueryPool, VkResult> {
        let pool = self.create_object(device, |d| d.vkCreateQueryPool(), info, alloc)?;
        self.query_pools.insert(pool, QueryFacts::of(info));
        Ok(pool)
    }

    /// The record of `pool`, for a command about to name its queries.
    fn query_facts(&self, pool: VkQueryPool) -> Result<&QueryFacts, QueryRefused> {
        self.query_pools.get(&pool).ok_or(QueryRefused::UnknownPool)
    }

    /// Drop a query pool's record. Called from the two places Vulkan destroys one: the guest's
    /// own `vkDestroyQueryPool`, and the teardown that empties a device the guest left full.
    pub fn forget_query_pool(&mut self, pool: VkQueryPool) {
        self.query_pools.remove(&pool);
    }

    /// `vkResetQueryPool`: the host-side reset, a Vulkan 1.2 entry point the guest sends only
    /// once it has enabled the feature the driver advertised for it. A device that advertised
    /// the feature and exports no entry point is the host contradicting itself, so that is a
    /// refusal rather than a no-op.
    pub fn reset_query_pool(
        &self,
        device: VkDevice,
        pool: VkQueryPool,
        first: u32,
        count: u32,
    ) -> Result<(), QueryRefused> {
        let d = self.devices.get(&device).ok_or(QueryRefused::NoDevice)?;
        let f = d.fns.try_vkResetQueryPool().ok_or(QueryRefused::NoHostReset)?;
        self.query_facts(pool)?.holds(first, count)?;
        // SAFETY: a device in this table, a pool recorded on it, and a range of queries the pool
        // holds -- which is what the driver writes host memory at.
        unsafe { f(device, pool, first, count) };
        Ok(())
    }

    /// `vkGetQueryPoolResults`: `count` results from `first` on, laid `stride` apart in `out`.
    ///
    /// The driver is handed the slice's own length as `dataSize`, after the queries named have
    /// been held to the pool and their results to the slice -- the two bounds Vulkan leaves to
    /// the caller's word, which is a guest's here. `VK_NOT_READY` is an answer, not an error:
    /// results not yet available, with `out` left as it was.
    #[allow(clippy::too_many_arguments)]
    pub fn query_pool_results(
        &self,
        device: VkDevice,
        pool: VkQueryPool,
        first: u32,
        count: u32,
        out: &mut [u8],
        stride: VkDeviceSize,
        flags: VkQueryResultFlags,
    ) -> Result<VkResult, QueryRefused> {
        let d = self.devices.get(&device).ok_or(QueryRefused::NoDevice)?;
        let facts = self.query_facts(pool)?;
        facts.holds(first, count)?;
        if facts.bytes_for(count, stride, flags)? > out.len() as u64 {
            return Err(QueryRefused::OutOfRoom);
        }
        // SAFETY: a device in this table and a pool recorded on it; `out` holds every byte the
        // driver may write, as just measured, and its length is what the driver is told.
        Ok(unsafe {
            (d.fns.vkGetQueryPoolResults())(
                device,
                pool,
                first,
                count,
                out.len(),
                out.as_mut_ptr().cast(),
                stride,
                flags,
            )
        })
    }

    /// Fold `srcs` into `dst`. The handles are the guest's names already resolved to the
    /// driver's, and the count Vulkan is given is the slice's own length.
    pub fn merge_pipeline_caches(
        &self,
        device: VkDevice,
        dst: VkPipelineCache,
        srcs: &[VkPipelineCache],
    ) -> Result<VkResult, VkResult> {
        let f = self
            .devices
            .get(&device)
            .and_then(|d| d.fns.try_vkMergePipelineCaches())
            .ok_or(VkResult::VK_ERROR_INITIALIZATION_FAILED)?;
        // SAFETY: handles this context made, and the count is the slice's own length.
        Ok(unsafe { f(device, dst, srcs.len() as u32, srcs.as_ptr()) })
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
        // The other place an image or a query pool dies -- the guest left it live and the
        // teardown took it. Its record goes with it, here rather than in `destroy_tracked`, which
        // holds the device's entry points borrowed out of `self` and so cannot reach the maps.
        let recorded: Vec<(VkObjectType, HostHandle)> =
            doomed.iter().filter(|o| o.device == Some(device)).map(|o| (o.ty, o.handle)).collect();
        for (ty, handle) in recorded {
            match ty {
                VkObjectType::VK_OBJECT_TYPE_IMAGE => self.forget_image(VkImage::from_host(handle)),
                VkObjectType::VK_OBJECT_TYPE_QUERY_POOL => {
                    self.forget_query_pool(VkQueryPool::from_host(handle));
                }
                _ => {}
            }
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

    /// Free a run of pool children back to the pool that allocated them, handing back whatever
    /// the driver's free returns.
    ///
    /// `R` is `()` for `vkFreeCommandBuffers` and `VkResult` for `vkFreeDescriptorSets` -- the
    /// one free in Vulkan with a return, which the spec says is always `VK_SUCCESS`. An empty run
    /// is the caller's to short-circuit: there is nothing here to check for it.
    ///
    /// Every object has to be `pool`'s, and that is checked before the driver is called rather
    /// than trusted: a guest pairing a pool with another pool's objects is asking for undefined
    /// behaviour on the host while the pool is open, and while it is closed would have the
    /// objects forgotten from the guest's table and not from the pool that still holds them --
    /// which hands them back at that pool's destroy, to remove whatever the guest has since
    /// named by the same ids.
    pub fn free_objects<P: PoolOf, R>(
        &mut self,
        device: VkDevice,
        proc: impl FnOnce(&DeviceFns) -> unsafe extern "C" fn(VkDevice, P, u32, *const P::Child) -> R,
        pool: P,
        objects: &[P::Child],
    ) -> Result<R, FreeRefused> {
        let d = self.devices.get(&device).ok_or(FreeRefused::NoDevice)?;
        if !self.pools.all_from(pool, objects) {
            return Err(FreeRefused::NotFromThisPool);
        }
        // SAFETY: handles this context allocated from this pool -- just checked -- and the count
        // Vulkan is given is the slice's own length. The generated lifecycle hook removes the
        // ids from the object table exactly once.
        let r = unsafe { proc(&d.fns)(device, pool, objects.len() as u32, objects.as_ptr()) };
        self.pools.release(objects.iter().copied());
        Ok(r)
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
        // The same fork the real allocate takes, so a planted allocation is the shape a guest's
        // would be: memory the host can address is minted pages this renderer owns, and only
        // memory it cannot is left to the driver. Charged like a real one either way, so a test's
        // ledger says what a guest's would.
        let backing = if props & HOST_VISIBLE_BIT != 0 {
            let len = size_for_pages(size).expect("a test size fits");
            let map = GuestMap::anonymous(len).expect("the host has pages");
            let charge = self
                .account
                .try_charge("exported pages", len as u64)
                .expect("a test ledger has no cap");
            Backing::Owned {
                storage: Storage::pages(map, charge, NoSurface::NotExported),
                published: false,
            }
        } else {
            let charge =
                self.account.try_charge("device memory", size).expect("a test ledger has no cap");
            Backing::Driver { charge }
        };
        self.memory
            .insert(id, Allocated { size, backing, props: VkMemoryPropertyFlags(props as _) });
    }

    /// Plant an allocation that aliases storage it resolved elsewhere -- host-visible like any
    /// import, so that what keeps it out of the census is the aliasing and nothing else.
    #[cfg(test)]
    pub(super) fn plant_imported_allocation(&mut self, id: ObjectId, size: u64, of: ResourceBytes) {
        // Uncharged, like a real one: the bytes are whoever's it resolved to.
        self.memory.insert(
            id,
            Allocated {
                size,
                backing: Backing::Imported(of),
                props: VkMemoryPropertyFlags(
                    (HOST_VISIBLE_BIT | HOST_COHERENT_BIT | HOST_CACHED_BIT) as _,
                ),
            },
        );
    }

    /// Plant an allocation backed by a real IOSurface, which is the only way to get one: a
    /// surface cannot be faked, and every claim about a scanout is a claim about what the system
    /// did with it.
    #[cfg(test)]
    pub(super) fn plant_scanout_allocation(&mut self, id: ObjectId, surface: Surface) {
        let charge = self
            .account
            .try_charge("IOSurface", surface.alloc_size())
            .expect("a test ledger has no cap");
        self.memory.insert(
            id,
            Allocated {
                size: surface.alloc_size(),
                backing: Backing::Owned {
                    storage: Storage::Texture(Arc::new(Charged { it: surface, charge })),
                    published: false,
                },
                props: VkMemoryPropertyFlags(
                    (HOST_VISIBLE_BIT | HOST_COHERENT_BIT | HOST_CACHED_BIT) as _,
                ),
            },
        );
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
        self.images.clear();
        self.query_pools.clear();
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

    /// Whether a pool is still open, for the test that a reset keeps it so where a destroy does
    /// not. Test scaffolding, beside [`Driver::plant_pool`] for the same reason.
    #[cfg(test)]
    pub(super) fn pool_is_open<P: PoolOf>(&self, pool: P) -> bool {
        self.pools.is_open(pool)
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

    /// Recycle a pool's allocations, handing back the guest ids that stopped naming anything.
    ///
    /// The pool itself survives, so unlike [`Driver::destroy_pool`] there is no object to destroy
    /// here -- Vulkan freed the children as part of the reset and names none of them.
    pub fn recycle_pool<T: Handle>(&mut self, pool: T) -> Vec<ObjectId> {
        self.pools.recycle(pool)
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

    /// The unscaled sibling of [`Self::cmd_blit_image`]: the regions name one extent, not a
    /// source rectangle and a destination one, so there is nothing for a filter to do.
    pub fn cmd_copy_image(
        &self,
        cb: VkCommandBuffer,
        src: VkImage,
        src_layout: VkImageLayout,
        dst: VkImage,
        dst_layout: VkImageLayout,
        regions: &[VkImageCopy],
    ) -> Option<()> {
        let d = self.recorder(cb)?;
        // SAFETY: as above; the count is the slice's own length.
        unsafe {
            (d.vkCmdCopyImage())(
                cb,
                src,
                src_layout,
                dst,
                dst_layout,
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

    // The query commands. Each names queries by index, and each is held to the pool's record
    // before the driver sees it: the driver addresses its own memory at that index -- host
    // memory for a begin, the pool's buffer on the GPU for the rest -- on the caller's word,
    // and the caller here is a guest. See [`QueryRefused`].

    /// The recorder for `cb` and the record for `pool`, for the query commands.
    fn query_recorder(
        &self,
        cb: VkCommandBuffer,
        pool: VkQueryPool,
    ) -> Result<(&DeviceFns, &QueryFacts), QueryRefused> {
        let d = self.recorder(cb).ok_or(QueryRefused::NoDevice)?;
        Ok((d, self.query_facts(pool)?))
    }

    pub fn cmd_begin_query(
        &self,
        cb: VkCommandBuffer,
        pool: VkQueryPool,
        query: u32,
        flags: VkQueryControlFlags,
    ) -> Result<(), QueryRefused> {
        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(query, 1)?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdBeginQuery())(cb, pool, query, flags) };
        Ok(())
    }

    pub fn cmd_end_query(
        &self,
        cb: VkCommandBuffer,
        pool: VkQueryPool,
        query: u32,
    ) -> Result<(), QueryRefused> {
        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(query, 1)?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdEndQuery())(cb, pool, query) };
        Ok(())
    }

    pub fn cmd_reset_query_pool(
        &self,
        cb: VkCommandBuffer,
        pool: VkQueryPool,
        first: u32,
        count: u32,
    ) -> Result<(), QueryRefused> {
        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(first, count)?;
        // SAFETY: as above, and a range of queries the pool holds.
        unsafe { (d.vkCmdResetQueryPool())(cb, pool, first, count) };
        Ok(())
    }

    pub fn cmd_write_timestamp(
        &self,
        cb: VkCommandBuffer,
        stage: VkPipelineStageFlagBits,
        pool: VkQueryPool,
        query: u32,
    ) -> Result<(), QueryRefused> {
        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(query, 1)?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdWriteTimestamp())(cb, stage, pool, query) };
        Ok(())
    }

    /// The GPU-side read-back: results land in a buffer of the guest's, on the device, where the
    /// bounds are the guest's own allocation's -- the same footing as every other `vkCmd*`
    /// that names a buffer. The queries read are the pool's to hold, as everywhere.
    #[allow(clippy::too_many_arguments)]
    pub fn cmd_copy_query_pool_results(
        &self,
        cb: VkCommandBuffer,
        pool: VkQueryPool,
        first: u32,
        count: u32,
        dst: VkBuffer,
        offset: VkDeviceSize,
        stride: VkDeviceSize,
        flags: VkQueryResultFlags,
    ) -> Result<(), QueryRefused> {
        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(first, count)?;
        // SAFETY: as above, and a range of queries the pool holds.
        unsafe {
            (d.vkCmdCopyQueryPoolResults())(cb, pool, first, count, dst, offset, stride, flags)
        };
        Ok(())
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

        // An import and a scanout are mutually exclusive by construction: an import names
        // storage that exists, and a scanout is storage being made.
        //
        // Held, not merely resolved: the driver is handed an address, and an address is good
        // only as long as what it points into. The importer's record keeps what it resolved --
        // a share, or the host's own mapping -- so the bytes outlive the exporter's record and
        // the resource both, for exactly as long as the driver may still reach them.
        let alias = import.flatten().and_then(resource_bytes);
        // An import that resolves to nothing is refused, never quietly turned into an allocation
        // of our own. The guest asked to alias storage it named; memory that aliases nothing is
        // a buffer nobody presents, and the bind that follows would reach a handle the host
        // never backed. Refusing puts the failure where the guest can see it, and ghosts the id
        // so the commands already in flight behind it are lost rather than fatal.
        if import.is_some() && alias.is_none() {
            return Err(NoMemory::Driver(VkResult::VK_ERROR_INVALID_EXTERNAL_HANDLE));
        }
        let alias_span = alias.as_ref().map(|b| self.span(b));
        let surface = if import.is_some() {
            Err(NoSurface::NotExported)
        } else {
            self.scanout_surface(device, &info)
        };
        // The third shape: memory the guest asked to be able to export, that is not a window
        // buffer. The driver would allocate it and later lend a mapping -- a pointer with the
        // device's lifetime, which no other context could be handed. So the pages are minted here
        // and the driver imports them, and the pages are what a resource can hold a share of.
        // Every allocation the host can address is minted here and imported, whether or not the
        // guest said it meant to export it. Waiting for the guest to say so was the bug: a guest
        // may export at any later moment, and an allocation that was not minted has only the
        // driver's own `vkMapMemory` pointer to publish -- an address the VMM maps into guest
        // physical memory and that `vkFreeMemory` then unmaps underneath it. Minting up front is
        // what makes the storage outlive the allocation: a blob holds a share, so the guest's
        // free retires the record and not the pages.
        //
        // There is deliberately no fallback to a driver allocation when the mint or the import
        // fails. A fallback is the borrowed-pointer lifetime coming back under another name, and
        // it would come back exactly on the hosts where it is least testable. Refusing is loud
        // and the guest's allocation fails; the alternative is quiet and the hypervisor's mapping
        // dangles.
        let pages = if import.is_none()
            && surface.is_err()
            && props.is_some_and(|p| p.0 & HOST_VISIBLE_BIT != 0)
        {
            Some(size_for_pages(info.allocationSize.0)?)
        } else {
            None
        };
        // What the census reports and what the guest maps: the guest's own figure, padded. A
        // surface's page-rounded extent is a fact about how IOSurface rounds, and telling the
        // guest that number would be answering a question it did not ask.
        let size = info.allocationSize.0;

        // What the bytes are decides both how they are freed and what they cost, so it is settled
        // once, here. Charged before anything is minted or the driver is asked, so a refusal
        // costs no host memory -- and credited by the charge going out of scope if the driver
        // then refuses. A scanout is charged at the surface's own extent, because the surface is
        // the commitment, and the charge goes into the storage rather than beside it: the storage
        // may outlive this allocation and this context, and the bytes are the host's for as long
        // as it does. An import commits nothing at all; an import that resolved to nothing is not
        // an import but the ordinary allocation it fell through to, and is charged as one.
        let backing = match (alias, surface, pages) {
            (Some(bytes), _, _) => Backing::Imported(bytes),
            (None, Ok(surface), _) => {
                let charge = self.admit("IOSurface", surface.alloc_size())?;
                Backing::Owned {
                    storage: Storage::Texture(Arc::new(Charged { it: surface, charge })),
                    published: false,
                }
            }
            (None, Err(why), Some(len)) => {
                let charge = self.admit("exported pages", len as u64)?;
                let map = match GuestMap::anonymous(len) {
                    Ok(map) => map,
                    Err(e) => {
                        eprintln!("[virglrs] cannot mint {len} bytes to export: {e}");
                        return Err(NoMemory::Driver(VkResult::VK_ERROR_OUT_OF_HOST_MEMORY));
                    }
                };
                Backing::Owned { storage: Storage::pages(map, charge, why), published: false }
            }
            (None, Err(_), None) => Backing::Driver { charge: self.admit("device memory", size)? },
        };

        // Whichever it is, it comes out as one host address the driver is handed instead of
        // memory of its own. The storage backs exactly this much, whoever owns
        // it: the guest's figure is its own image's size, and a request larger than the backing
        // would let the driver address past the end of it -- the one place a guest's arithmetic
        // could reach outside the host's.
        let span = match &backing {
            Backing::Owned { storage, .. } => Some(storage.span()),
            Backing::Imported(_) => alias_span,
            Backing::Driver { .. } => None,
        };
        let mut host_pointer = VkImportMemoryHostPointerInfoEXT {
            sType: VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
            pNext: info.pNext,
            handleType: VkExternalMemoryHandleTypeFlagBits::
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            pHostPointer: core::ptr::null_mut(),
        };
        if let Some(span) = span {
            host_pointer.pHostPointer = span.0 as *mut core::ffi::c_void;
            // Prepended, not spliced in: the guest's chain is the decoder's arena and the round
            // trip re-encodes it, so it is read here and never rewritten.
            info.pNext = (&raw const host_pointer).cast();
            info.allocationSize = VkDeviceSize(info.allocationSize.0.min(span.1));
        }

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
        self.memory.insert(id, Allocated { size, backing, props });
        Ok(out)
    }

    /// Take `size` bytes against the budget, or say why the allocation is refused.
    fn admit(&self, what: &'static str, size: u64) -> Result<Charge, NoMemory> {
        self.account.try_charge(what, size).map_err(|refused| {
            self.account.report_refusal(refused);
            NoMemory::OverBudget { stop: self.account.kills_context() }
        })
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
    /// the same answer -- they cannot get two.
    ///
    /// Total, and no longer a lookup. Both shapes are things the resource *holds*, so an address
    /// exists by construction; there is no id left to resolve through a table that could have
    /// emptied, and so no "importable" answer the import can go on to contradict. It stays a
    /// method rather than becoming a free function because the driver is what the caller has,
    /// and moving it would only relocate the call.
    pub fn span(&self, bytes: &ResourceBytes) -> (usize, u64) {
        match bytes {
            ResourceBytes::Host(map) => (map.host_addr(), map.len() as u64),
            // Resolved from the share itself. No table is consulted, so it answers the same for
            // the context that made the storage and for any other the guest attached it to.
            ResourceBytes::Shared(storage) => storage.span(),
        }
    }

    /// Mint the IOSurface a scanout allocation lives in, if this allocation is one.
    ///
    /// A scanout is recognised by shape, not by a flag: the guest exports the memory to the
    /// outside world (`VkExportMemoryAllocateInfo`) and dedicates it to one image
    /// (`VkMemoryDedicatedAllocateInfo`). That is what a window buffer is, and nothing else in a
    /// venus stream looks like it.
    ///
    /// A named refusal at every step that cannot be answered honestly -- a format no IOSurface
    /// has, a driver that will not report a layout, a pitch the surface would not take. Every one
    /// of those leaves an ordinary allocation, which renders correctly and merely cannot be
    /// composited without a copy; the reason travels with the pages, so a scanout that later
    /// asks them for a surface can say which question the image failed. A surface whose rows
    /// sit somewhere other than where the driver will write them is worse than no surface: it
    /// displays, and it displays sheared.
    fn scanout_surface(
        &mut self,
        device: VkDevice,
        info: &VkMemoryAllocateInfo,
    ) -> Result<Surface, NoSurface> {
        if !exports_memory(info.pNext) {
            return Err(NoSurface::NotExported);
        }
        let image = dedicated_image(info.pNext).ok_or(NoSurface::NotDedicated)?;
        let facts = *self.images.get(&image).ok_or(NoSurface::UnknownImage)?;
        let format = pixel_format(facts.format).ok_or(NoSurface::Format(facts.format))?;
        // Only a layout the CPU can address has rows to alias. An OPTIMAL image is opaque: the
        // driver keeps its storage in a private layout of its own choosing, renders there, and
        // would never write a byte into pages minted here. Asked for its row pitch it answers
        // with a number that describes nothing, so the pitch checks below are not this test --
        // they guard against a linear layout the driver reports wrongly, not against a tiled one.
        if !matches!(
            facts.tiling,
            VkImageTiling::VK_IMAGE_TILING_LINEAR
                | VkImageTiling::VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
        ) {
            return Err(NoSurface::Tiling(facts.tiling));
        }

        let d = self.devices.get(&device).ok_or(NoSurface::Layout)?;
        let layout = {
            let query = d.fns.try_vkGetImageSubresourceLayout().ok_or(NoSurface::Layout)?;
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
        let pitch =
            u32::try_from(layout.rowPitch.0).ok().filter(|p| *p != 0).ok_or(NoSurface::Layout)?;
        // The record says how wide the guest asked for; the live query says how the driver laid
        // it out. A record left behind by an image whose handle has since been recycled will not
        // describe this image, and this is where that shows: the rows would not add up.
        if layout.size.0 != u64::from(pitch) * u64::from(facts.height) {
            return Err(NoSurface::Layout);
        }

        let surface = Surface::scanout(facts.width, facts.height, format, pitch)
            .map_err(|_| NoSurface::Layout)?;
        // IOSurface may lay the rows out its own way. The allocation is about to be a
        // host-pointer import of these pages, so a pitch that is not the driver's is a surface
        // whose every row is at the wrong offset.
        if surface.bytes_per_row() != pitch {
            return Err(NoSurface::Layout);
        }
        Ok(surface)
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
                tiling: info.tiling,
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
        self.memory.remove(&id);
        // Dropping the record is the whole retirement. Nothing here holds a mapping the VMM was
        // handed: an exported allocation's bytes are storage this renderer minted, and a blob
        // over it holds a share, so the pages stand until the last holder goes. Freeing the
        // allocation retires the record and the charge with it, and leaves the guest's blob
        // working -- which is the arrangement the C reaches on Linux by keeping a dup'd fd.
        let Some(d) = self.devices.get(&device) else {
            return;
        };
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

    /// Publish an allocation as a blob: a share of the storage behind it, and the address the
    /// VMM maps into the guest.
    ///
    /// It touches Vulkan not at all, which is the change: every allocation the host can address
    /// is storage this renderer minted and the driver imported, so publishing hands out what
    /// already exists rather than asking the driver to map something. The share is what makes the
    /// bytes outlive the allocation, the context and this call -- the VMM reads and writes them
    /// for as long as the resource lives, and the guest is free to `vkFreeMemory` in the meantime.
    ///
    /// It is *not* handed out again: exporting twice would give two resources one storage, and
    /// the second holder would have no way to know.
    ///
    /// Every refusal here is the guest's error, not ours: it names memory it never allocated,
    /// exports the same memory twice, asks for a blob larger than the storage behind it, or asks
    /// to publish memory that has no host address to give. None of them may stop the worker.
    pub fn memory_export(
        &mut self,
        id: ObjectId,
        blob_size: u64,
    ) -> Result<(Exported, Storage), ExportError> {
        let Some(record) = self.memory.get(&id) else {
            return Err(ExportError::NoSuchAllocation);
        };
        if record.exported() {
            return Err(ExportError::AlreadyExported);
        }
        let storage = match &record.backing {
            Backing::Owned { storage, .. } => storage,
            // Memory the host cannot address. There is no pointer to publish and never was: the
            // mint at allocate is gated on host-visibility, so this arm is exactly the memory
            // that was left to the driver.
            Backing::Driver { .. } => return Err(ExportError::NotHostVisible),
            // Bytes another allocation owns. The exporter publishes them, once; a second export
            // through the alias would be two resources over one storage with no way for either
            // holder to learn of the other.
            Backing::Imported(_) => return Err(ExportError::NotMappable),
        };
        // The VMM publishes the *blob's* size from this address, not the storage's, so a blob
        // larger than what was minted would put host memory past the end of it into the guest.
        // `pad_for_blob` sizes an allocation up so this does not normally happen; refuse rather
        // than over-map on the guest's say-so if it ever does.
        let (addr, len) = storage.span();
        if blob_size > len {
            return Err(ExportError::LargerThanAllocation);
        }
        let write_back = record.write_back();
        let share = storage.clone();
        let record = self.memory.get_mut(&id).expect("the record was here a moment ago");
        let Backing::Owned { published, .. } = &mut record.backing else {
            unreachable!("the backing was owned a moment ago");
        };
        *published = true;
        Ok((Exported { addr, write_back }, share))
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
        // Minted pages are host memory this renderer owns, coherent like any anonymous page, and
        // there is nothing for `vkMapMemory` to map -- the driver imported them.
        if let Backing::Owned { storage: Storage::Linear(p), .. } = &record.backing {
            let n = buf.len().min(size as usize);
            assert!(p.it.map.copy_out(0, &mut buf[..n]), "the census reads within pages it minted");
            return Ok(n);
        }
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

    /// Copy bytes back into an allocation, returning how many landed.
    ///
    /// [`Self::memory_read`] backwards, arm for arm, because a restore has to reach the bytes by
    /// whichever route the capture read them by: a scanout through its surface, minted pages
    /// through the mapping this renderer owns, and anything else through `vkMapMemory`. A route
    /// the two sides disagreed about would put a capture back somewhere nothing samples.
    ///
    /// Length is the one place the two are not mirrors. A short read is ordinary -- the VMM caps
    /// what it keeps -- so a short write puts that prefix back and says so. A write *longer* than
    /// the allocation is not a caller being economical: it means the allocation this id names now
    /// is not the one the bytes were read from, and the refusal is the point.
    ///
    /// No flush after the mapped write, and none of this host's business: KosmicKrisp advertises
    /// exactly one memory type and it is `HOST_COHERENT`. A host with a non-coherent host-visible
    /// type would owe `vkFlushMappedMemoryRanges` here and `vkInvalidateMappedMemoryRanges` in
    /// the read above -- both, or the pair is worth nothing.
    pub fn memory_write(
        &self,
        device: VkDevice,
        handle: VkDeviceMemory,
        id: ObjectId,
        src: &[u8],
    ) -> Result<usize, MemoryError> {
        let Some(record) = self.memory.get(&id) else {
            return Err(MemoryError::NoSuchAllocation);
        };
        if src.len() as u64 > record.size {
            return Err(MemoryError::LargerThanAllocation);
        }
        if let Some(surface) = record.surface() {
            return Ok(surface.write_from(src));
        }
        if let Backing::Owned { storage: Storage::Linear(p), .. } = &record.backing {
            assert!(p.it.map.copy_in(0, src), "a write within pages this renderer minted");
            return Ok(src.len());
        }
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
        // SAFETY: the driver mapped at least `record.size` bytes at `ptr`, and `src` is no longer
        // than that -- checked above, which is what that check is for. The two cannot overlap:
        // one is the driver's mapping and the other the caller's.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), ptr.cast::<u8>(), src.len()) };
        // SAFETY: the mapping this call just made, unmapped once.
        unsafe { (d.fns.vkUnmapMemory())(device, handle) };
        Ok(src.len())
    }
}

/// One live allocation, as this driver holds it.
///
/// Not what the census reports -- see [`Allocation`]. This is the record the driver keeps for its
/// own purposes, and it is where an export's mapping lives.
struct Allocated {
    /// Its size, padded to the blob the guest may map it as -- see [`pad_for_blob`].
    size: u64,
    /// What the bytes actually are, what they cost, and whether they have been published. See
    /// [`Backing`].
    backing: Backing,
    /// The properties of the memory type it was allocated from.
    ///
    /// The flags themselves rather than the questions asked of them: an export must refuse memory
    /// the host cannot address, and the VMM must be told how the guest may cache it. Two answers
    /// derived from one recorded fact cannot drift apart the way two recorded booleans can.
    props: VkMemoryPropertyFlags,
}

/// What an allocation's bytes are, which decides how it is published, read, freed and charged.
///
/// One value rather than a bool per question. "Is it an import", "does it have a surface", "does
/// freeing owe an unmap", "has it been exported" and "what did it cost" are readings of one
/// fact, and as separate fields any two of them could disagree about the same allocation -- a
/// driver record with no charge, or an export mark on storage that was never mapped. Each arm
/// carries exactly the state its kind of bytes has.
enum Backing {
    /// Memory the driver allocated for this guest, and what it cost the host.
    ///
    /// Only memory the host cannot address reaches this arm: everything host-visible is minted
    /// and imported instead, so the renderer owns the pages. There is therefore no address to
    /// publish and nothing to unmap -- which is the point. An allocation whose bytes exist only
    /// where the driver put them cannot be handed to the VMM, and so cannot be handed to the VMM
    /// and then freed underneath it.
    ///
    /// The charge is never read, and that is the design: it is a value whose only job is to be
    /// dropped with the record, so there is no release call for a future destroy path to forget.
    Driver {
        #[expect(
            dead_code,
            reason = "held for its Drop -- crediting the ledger is this going away"
        )]
        charge: Charge,
    },
    /// Storage this renderer minted -- an IOSurface, or pages -- that the allocation is a
    /// host-pointer import of, so the memory *is* the storage. The storage carries its own
    /// charge, because a resource holding a share keeps it alive past this record and past the
    /// context, and the bytes are the host's to count for as long as anyone does.
    ///
    /// Publishing hands out the storage's own address -- there is nothing to unmap, and the last
    /// holder going is what releases it -- so the mark is all that `published` has to carry. A
    /// surface is read through [`crate::metal::Surface::read_into`], because a surface read
    /// without its lock sees whatever the CPU's view last held rather than what the GPU wrote.
    Owned { storage: Storage, published: bool },
    /// Storage another context owns, which this allocation only aliases -- held, so that the
    /// address the driver was handed stays good for as long as this allocation can use it.
    ///
    /// A guest imports when one context has to reach what another rendered -- a compositor
    /// sampling a client's window. The census must not report it: one buffer under two ids, read
    /// through a mapping this context has no claim to. What it holds is what the resource
    /// resolved to: a share keeps its storage alive, the host's mapping keeps its pages, and a
    /// published allocation's name keeps nothing -- that one is the same context's own memory,
    /// and lives or dies with it.
    Imported(
        #[expect(dead_code, reason = "held for what it keeps alive, never read")] ResourceBytes,
    ),
}

/// What a query pool was created as, for the commands that name its queries by index and the
/// read-back that has to fit its buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct QueryFacts {
    /// How many queries the pool holds.
    queries: u32,
    /// How many values one query's result is, before the availability and status words the
    /// read-back flags may append -- one for most kinds, one per counted statistic for a
    /// pipeline-statistics pool. `None` is a kind this renderer cannot size: a performance
    /// query's counters are negotiated through an extension, a video encode's feedback through
    /// its own. Such a pool is created and indexed like any other; only its read-back into
    /// guest-offered room is refused, since the room cannot be held to a size nobody knows.
    values: Option<u32>,
}

impl QueryFacts {
    /// The facts of the pool `info` creates. Total: whether the driver will make the pool is the
    /// driver's to decide, and a type this renderer has no size for is still a pool.
    fn of(info: &VkQueryPoolCreateInfo) -> Self {
        let values = match info.queryType {
            VkQueryType::VK_QUERY_TYPE_OCCLUSION
            | VkQueryType::VK_QUERY_TYPE_TIMESTAMP
            | VkQueryType::VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT
            | VkQueryType::VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT
            | VkQueryType::VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR
            | VkQueryType::VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR
            | VkQueryType::VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR
            | VkQueryType::VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR
            | VkQueryType::VK_QUERY_TYPE_MICROMAP_SERIALIZATION_SIZE_EXT
            | VkQueryType::VK_QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT => Some(1),
            VkQueryType::VK_QUERY_TYPE_PIPELINE_STATISTICS => {
                Some(info.pipelineStatistics.0.count_ones())
            }
            VkQueryType::VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT => Some(2),
            VkQueryType::VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR => Some(0),
            _ => None,
        };
        QueryFacts { queries: info.queryCount, values }
    }

    /// Whether queries `first..first + count` are all the pool's.
    fn holds(&self, first: u32, count: u32) -> Result<(), QueryRefused> {
        match first.checked_add(count) {
            Some(end) if end <= self.queries => Ok(()),
            _ => Err(QueryRefused::OutOfPool),
        }
    }

    /// How many bytes a read of `count` results laid `stride` apart needs, with `flags` saying how
    /// wide each value is and which words follow it. Arithmetic the guest's numbers overflow is
    /// a size no buffer holds, and is refused as one.
    fn bytes_for(
        &self,
        count: u32,
        stride: VkDeviceSize,
        flags: VkQueryResultFlags,
    ) -> Result<u64, QueryRefused> {
        let values = self.values.ok_or(QueryRefused::Unsized)?;
        let Some(last) = count.checked_sub(1) else {
            return Ok(0);
        };
        let has = |bit: VkQueryResultFlagBits| flags.0 & bit.0 as u32 != 0;
        let width: u64 = if has(VkQueryResultFlagBits::VK_QUERY_RESULT_64_BIT) { 8 } else { 4 };
        let words = u64::from(values)
            + u64::from(has(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_AVAILABILITY_BIT))
            + u64::from(has(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_STATUS_BIT_KHR));
        u64::from(last)
            .checked_mul(stride.0)
            .and_then(|n| n.checked_add(words.checked_mul(width)?))
            .ok_or(QueryRefused::OutOfRoom)
    }
}

/// What an image was created as, for a scanout surface that has to match it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ImageFacts {
    width: u32,
    height: u32,
    format: VkFormat,
    /// As the driver was told to lay it out -- after [`external_images_are_linear`], not before.
    tiling: VkImageTiling,
}

impl Allocated {
    /// The surface behind it, for the one backing that has one.
    fn surface(&self) -> Option<&Surface> {
        self.storage()?.surface().ok()
    }

    /// Whether it has been published to the VMM -- which a second export must refuse, because
    /// two resources over one storage is a state neither holder could detect afterwards.
    fn exported(&self) -> bool {
        match &self.backing {
            Backing::Driver { .. } | Backing::Imported(_) => false,
            Backing::Owned { published, .. } => *published,
        }
    }

    /// The storage behind it, for the backings that own theirs.
    ///
    /// A resource that must go on naming these bytes after the allocation, and after the context
    /// that made it, holds a clone of this. `None` is not a gap: an import's bytes are the
    /// exporter's to lend, and driver memory is memory the host cannot address, which has no
    /// address to lend in the first place.
    fn storage(&self) -> Option<&Storage> {
        match &self.backing {
            Backing::Owned { storage, .. } => Some(storage),
            Backing::Driver { .. } | Backing::Imported(_) => None,
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
        match &self.backing {
            Backing::Driver { .. } => true,
            Backing::Owned { storage: Storage::Linear(_), published } => !published,
            Backing::Owned { storage: Storage::Texture(_), .. } => true,
            Backing::Imported(_) => false,
        }
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

/// A share of the storage behind a published allocation, held by whoever needs those bytes.
///
/// A resource keeps one of these rather than the *name* of an allocation, because a name is only
/// meaningful in the context that chose it and stops resolving the moment that context is torn
/// down. A share resolves for anyone holding it, for as long as they hold it -- which is what a
/// compositor sampling a client's last frame after the client has exited requires.
///
/// Deliberately not a raw address: an address would outlive the storage the first time a guest
/// freed it, which is the lifetime bug the whole late-resolution scheme was built to avoid. This
/// keeps the storage *alive* instead of describing where it used to be.
#[derive(Clone)]
pub enum Storage {
    /// A share of an IOSurface. The pages are the surface's, and the surface outlives every
    /// Vulkan object that ever imported them -- it depends on no device, no instance and no
    /// object table, so nothing cascades from holding one.
    ///
    /// The share and not the surface, because the owner is not always venus: an allocation minted
    /// here lends a [`Charged`] one, so the budget is credited when the last holder lets go, and
    /// a classic resource lends the share its EGL image already holds. Either way what travels is
    /// the right to keep the surface alive, and the owner decides what that costs -- see
    /// [`Held`].
    Texture(Arc<dyn Held>),
    /// Pages this renderer minted for an allocation the guest meant to share, and handed the
    /// driver by host-pointer import. Plain memory with rows the CPU can address; the guest's
    /// fences are the only barrier over them, as they are for any host-visible allocation.
    Linear(Arc<Charged<Pages>>),
}

/// Minted pages, and why they are pages rather than a surface.
///
/// Every export that is not a recognised scanout ends up here, and a compositor can still be
/// handed a share of it and try to present from it. There is no surface to present, and the
/// reason is the one fact worth having at that moment -- which image, and which question it
/// failed -- so it is kept with the pages and said once, the first time the share is asked.
pub struct Pages {
    map: GuestMap,
    why: NoSurface,
    /// Whether the refusal has been said. A compositor asks every frame and the answer does not
    /// change, so it is said once per storage rather than sixty times a second.
    said: std::sync::atomic::AtomicBool,
}

/// Why an exported allocation was given pages and not a surface.
///
/// The steps of [`Driver::scanout_surface`], each named for the question it asks of the image,
/// because a black window is otherwise the only symptom and it says nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NoSurface {
    /// The guest did not ask to export it, so nothing outside the guest was ever going to see
    /// it. Never the reason behind minted pages, which are minted only for an export.
    NotExported,
    /// Not dedicated to one image: a buffer, or an image allocation the guest left undedicated.
    NotDedicated,
    /// Dedicated to an image this renderer has no record of.
    UnknownImage,
    /// A pixel format IOSurface has no equivalent of.
    Format(VkFormat),
    /// An opaque layout: the driver keeps its storage in a layout of its own, and would never
    /// write a byte into pages minted here.
    Tiling(VkImageTiling),
    /// The driver's layout is not rows a surface could alias -- no layout query, a zero pitch,
    /// rows that do not add up to the image, or a pitch IOSurface would not take.
    Layout,
}

impl core::fmt::Display for NoSurface {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NoSurface::NotExported => f.write_str("the guest never exported it"),
            NoSurface::NotDedicated => f.write_str("the memory is not dedicated to an image"),
            NoSurface::UnknownImage => f.write_str("the image it is dedicated to has no record"),
            NoSurface::Format(format) => write!(f, "IOSurface has no format for {format:?}"),
            NoSurface::Tiling(tiling) => write!(f, "the image's layout is opaque ({tiling:?})"),
            NoSurface::Layout => {
                f.write_str("the driver's row layout is not one a surface could alias")
            }
        }
    }
}

/// Storage this renderer minted, and what it cost -- one value, because they have one lifetime.
///
/// The charge lives with the storage rather than on the allocation record so that it is credited
/// when the *storage* goes, not when the allocation does. A resource holding a share keeps the
/// storage alive past the context that made it, and those bytes are still the host's to count; a
/// charge on the record would have been credited at the context's destroy while the memory stood.
pub struct Charged<T> {
    it: T,
    #[expect(dead_code, reason = "held for its Drop -- crediting the ledger is this going away")]
    charge: Charge,
}

impl crate::metal::Held for Charged<Surface> {
    fn surface(&self) -> &Surface {
        &self.it
    }
}

impl Storage {
    /// A share of this storage, as the keepalive an EGL image over its surface must hold.
    ///
    /// `None` when the storage is pages: there is no surface to image, and the reason is the
    /// storage's own to give. Handing out the `Charged` share rather than a fresh one over the
    /// same surface is what keeps this allocation's charge standing for as long as the image
    /// does -- see [`crate::metal::Held`].
    pub fn held(&self) -> Option<Arc<dyn crate::metal::Held>> {
        match self {
            Storage::Texture(t) => Some(Arc::clone(t)),
            Storage::Linear(_) => None,
        }
    }

    /// Storage a classic resource owns, as a share venus can import.
    ///
    /// The one route from the classic side into a venus context, and the reason there is no
    /// dma-buf here: a compositor rendering through Vulkan imports each client window as an
    /// IOSurface, and a GL client's window buffer is one an EGL image already holds a share of.
    /// Lending that share -- not minting a second one over the same surface, and not passing the
    /// surface's id -- is what makes the import outlive the classic context that created it.
    ///
    /// It carries no charge, because vrend keeps no ledger; see the plan on what that costs.
    pub fn lent(held: Arc<dyn Held>) -> Storage {
        Storage::Texture(held)
    }

    /// A share over a real surface, charged to `account`, for a test outside this module.
    #[cfg(test)]
    pub(crate) fn minted_for_test(surface: Surface, account: &Account) -> Storage {
        let charge = account.try_charge("IOSurface", surface.alloc_size()).expect("no cap");
        Storage::Texture(Arc::new(Charged { it: surface, charge }))
    }

    /// A share over pages this renderer minted, charged to `account`, for the same tests.
    #[cfg(test)]
    pub(crate) fn pages_for_test(len: usize, account: &Account) -> Storage {
        let map = GuestMap::anonymous(len).expect("the host has pages");
        let charge = account.try_charge("exported pages", map.len() as u64).expect("no cap");
        Storage::pages(map, charge, NoSurface::NotDedicated)
    }

    /// A share over minted pages, carrying why they are not a surface.
    fn pages(map: GuestMap, charge: Charge, why: NoSurface) -> Storage {
        let it = Pages { map, why, said: std::sync::atomic::AtomicBool::new(false) };
        Storage::Linear(Arc::new(Charged { it, charge }))
    }
}

/// Two shares are the same share when they name the same storage -- not when they describe
/// storage that happens to look alike. Identity is the question every caller is actually asking.
impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Storage::Texture(a), Storage::Texture(b)) => Arc::ptr_eq(a, b),
            (Storage::Linear(a), Storage::Linear(b)) => Arc::ptr_eq(a, b),
            (Storage::Texture(_), Storage::Linear(_))
            | (Storage::Linear(_), Storage::Texture(_)) => false,
        }
    }
}

impl Eq for Storage {}

/// The surface's id and nothing else. An `IOSurfaceRef` printed as a pointer would be a lifetime
/// nobody can see, which is the whole reason `metal.rs` owns these.
impl core::fmt::Debug for Storage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Storage::Texture(m) => f.debug_tuple("Texture").field(&m.surface().id()).finish(),
            Storage::Linear(p) => {
                f.debug_tuple("Linear").field(&p.it.map.len()).field(&p.it.why).finish()
            }
        }
    }
}

impl Storage {
    /// Where the bytes are and how far they run, as the one pair anything can act on.
    pub fn span(&self) -> (usize, u64) {
        match self {
            Storage::Texture(m) => (m.surface().host_addr(), m.surface().alloc_size()),
            Storage::Linear(p) => (p.it.map.host_addr(), p.it.map.len() as u64),
        }
    }

    /// The mapping behind storage that is pages, and nothing for storage that is a surface.
    ///
    /// The counterpart to [`Storage::surface`]: a surface is *adopted* whole and its bytes are
    /// never read out, so answering `None` for one is not a gap. Borrowed, never cloned -- the
    /// pages live as long as the share does and a caller holding them past it would outlive
    /// what it describes.
    pub fn mapping(&self) -> Option<&GuestMap> {
        match self {
            Storage::Texture(_) => None,
            Storage::Linear(p) => Some(&p.it.map),
        }
    }

    /// The surface, for storage that is one -- or why this storage is not. Pages have nothing
    /// to adopt and no presented pixels to read, and the one question -- "is this a surface" --
    /// is answered here once rather than per thing a caller wants from it.
    pub fn surface(&self) -> Result<&Surface, NoSurface> {
        match self {
            Storage::Texture(m) => Ok(m.surface()),
            Storage::Linear(p) => Err(p.it.why),
        }
    }

    /// Whether this is the first time the storage has been asked for a surface it does not
    /// have. `true` once per storage, so the caller says the reason exactly once; `false` for a
    /// surface, which has nothing to explain.
    pub fn first_refusal(&self) -> bool {
        match self {
            Storage::Texture(_) => false,
            Storage::Linear(p) => !p.it.said.swap(true, std::sync::atomic::Ordering::Relaxed),
        }
    }
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
    /// A write carries more bytes than the allocation holds. Only a restore writes, and a capture
    /// bigger than what it is being put back into means the journal rebuilt a different
    /// allocation than the one the bytes came from -- the world disagreeing, which is refused
    /// rather than clamped to whatever happens to fit.
    LargerThanAllocation,
}

/// How many bytes to mint for an allocation of `size`, as the host counts them.
///
/// A size this host cannot hold -- the guest's own number, unbounded -- is answered with the
/// driver's refusal rather than a mint that wraps; the driver would have refused it too.
fn size_for_pages(size: u64) -> Result<usize, NoMemory> {
    usize::try_from(size)
        .ok()
        .and_then(crate::guest_mem::page_round)
        .ok_or(NoMemory::Driver(VkResult::VK_ERROR_OUT_OF_HOST_MEMORY))
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

/// The create info the driver is handed for an image, which is the guest's unless the guest
/// means to share the image and left its tiling to the driver.
///
/// Tiling is decided here and nowhere later: an image's layout is fixed at create, and the
/// surface a shared image is presented from is minted at allocate, when it is too late to ask
/// for rows. An external-memory image the guest created `OPTIMAL` -- WSI with no modifier lists,
/// or any application exporting a tiled image itself -- would then get a surface the driver
/// never writes into, because a tiled plane over host-imported memory is given private storage
/// of the driver's own and rendered there. So an image with external handle types and no DRM
/// format modifier is made `LINEAR`. Nothing the guest could rely on is lost: an image shared
/// outside this device was never going to be read in a layout only this device understands.
///
/// The same images lose `INPUT_ATTACHMENT` usage. The driver promotes an input attachment to a
/// 2D-array texture, and a linear texture over a Metal buffer must be plain 2D -- otherwise
/// render passes drop every draw while clears still land, which from the screen is
/// indistinguishable from a dozen other faults. zink sets the bit speculatively, and a buffer
/// shared for scanout is never fetched from.
///
/// A copy, not an edit in place: the decoder's struct is the guest's request, and the round trip
/// re-encodes it. The `pNext` chain is carried over untouched.
pub fn external_images_are_linear(info: &VkImageCreateInfo) -> VkImageCreateInfo {
    let mut info = *info;
    if info.tiling != VkImageTiling::VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
        && has_external_handle_types(info.pNext)
    {
        info.tiling = VkImageTiling::VK_IMAGE_TILING_LINEAR;
        info.usage.0 &= !(VkImageUsageFlagBits::VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT.0 as u32);
    }
    info
}

/// A struct that may sit in a `pNext` chain, and the tag it carries there. What lets one walk
/// serve every question asked of a chain, with the tag and the type it vouches for written
/// beside each other once instead of at every call site.
trait Chained: Copy {
    const S_TYPE: VkStructureType;
}

impl Chained for VkExternalMemoryImageCreateInfo {
    const S_TYPE: VkStructureType =
        VkStructureType::VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
}
impl Chained for VkExportMemoryAllocateInfo {
    const S_TYPE: VkStructureType = VkStructureType::VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
}
impl Chained for VkMemoryDedicatedAllocateInfo {
    const S_TYPE: VkStructureType =
        VkStructureType::VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
}
impl Chained for VkImportMemoryResourceInfoMESA {
    const S_TYPE: VkStructureType =
        VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA;
}

/// The link of type `T` in a `pNext` chain, if there is one -- copied out, so nothing holds a
/// pointer into the chain past the walk.
///
/// The one place a chain is walked, and the one `unsafe` for it: every link is a struct the
/// decoder allocated in the batch arena, every one begins with the `sType`/`pNext` header
/// `VkBaseInStructure` names, and the tag says which struct a link is.
fn chain_find<T: Chained>(mut node: *const core::ffi::c_void) -> Option<T> {
    while !node.is_null() {
        // SAFETY: as above.
        let base = unsafe { &*node.cast::<VkBaseInStructure>() };
        if base.sType == T::S_TYPE {
            // SAFETY: the tag says this link is a `T`.
            return Some(unsafe { *node.cast::<T>() });
        }
        node = base.pNext.cast();
    }
    None
}

/// Whether an image's `pNext` chain says it is for the world outside this guest -- external
/// memory with at least one handle type. A link naming no handle types shares nothing.
fn has_external_handle_types(node: *const core::ffi::c_void) -> bool {
    chain_find::<VkExternalMemoryImageCreateInfo>(node).is_some_and(|e| e.handleTypes.0 != 0)
}

/// Whether an allocation's `pNext` chain says the memory is for the world outside this guest.
fn exports_memory(node: *const core::ffi::c_void) -> bool {
    chain_find::<VkExportMemoryAllocateInfo>(node).is_some()
}

/// The image an allocation is dedicated to, if it is dedicated to one.
///
/// A dedicated allocation backs exactly one image, which is what makes it the image whose layout
/// a scanout surface must match. `VK_NULL_HANDLE` is the legal way to say "a buffer, not an
/// image", and reads as no image rather than as image zero.
fn dedicated_image(node: *const core::ffi::c_void) -> Option<VkImage> {
    chain_find::<VkMemoryDedicatedAllocateInfo>(node)
        .and_then(|d| (d.image.0 != 0).then_some(d.image))
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

/// The resource an allocation's `pNext` chain names, when it is aliasing storage rather than
/// asking for some.
///
/// Walked rather than asked of the guest, because the chain is where the guest put it. The two
/// layers are one answer: the outer says the guest asked to import, the inner which resource it
/// named -- and `Some(None)` is an import naming resource zero, which is no resource the way a
/// null handle is no image. Both are answered here so that no caller can hold a presence and a
/// resolution that disagree.
fn imported_resource(node: *const core::ffi::c_void) -> Option<Option<ResourceHandle>> {
    chain_find::<VkImportMemoryResourceInfoMESA>(node).map(|i| ResourceHandle::new(i.resourceId))
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
        let lent = driver.memory.get(&OWNED).expect("planted").storage().expect("pages").clone();
        driver.plant_imported_allocation(BORROWED, SIZE, ResourceBytes::Shared(lent));

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
    /// The share a scanout lends carries the surface's charge with it, so the ledger counts the
    /// surface for as long as anyone holds the share -- not for as long as the allocation that
    /// minted it happens to stand.
    #[test]
    fn a_shared_surface_stays_charged_after_its_allocation_is_gone() {
        use super::super::budget::Budget;
        let budget = Budget::with_cap(None, false);
        let one = crate::ids::ContextId::new(1).expect("not zero");
        let mut d =
            Driver::new(Account::open(&budget, crate::venus::vkr::ContextKey::for_test(one)));

        let surface = Surface::scanout(64, 8, PixelFormat::Bgra, 256).expect("the system minted");
        let extent = surface.alloc_size();
        d.plant_scanout_allocation(ObjectId(66), surface);
        assert_eq!(budget.live(), extent, "minted, and charged to the context that minted it");
        assert_eq!(budget.live_for(one), extent);

        let share = d
            .memory
            .get(&ObjectId(66))
            .expect("planted")
            .storage()
            .expect("a scanout lends")
            .clone();
        assert_eq!(budget.live_for(one), extent, "shared, and still the minting context's");

        // The allocation goes -- a free, or the context's whole memory table at its destroy.
        drop(d.memory.remove(&ObjectId(66)));
        assert_eq!(budget.live(), extent, "the share is what keeps it counted now");
        assert_eq!(
            budget.live_for(one),
            extent,
            "attributed to the context for as long as it lives"
        );

        // The context goes: the surface outlives it, and the ledger knows that too.
        drop(d);
        assert_eq!(budget.live(), extent, "a context's destroy does not uncount what outlives it");
        assert_eq!(budget.live_for(one), 0);
        assert_eq!(budget.shared(), extent);

        drop(share);
        assert_eq!(budget.live(), 0, "and the last share going is what credits it");
    }

    /// A capture goes back in by the route it came out of, whichever backing that is.
    ///
    /// The three arms are three different pieces of memory -- a surface's pages, pages this
    /// renderer minted, and whatever the driver hands back from `vkMapMemory` -- and a restore
    /// that knew one route and not another would put a snapshot back somewhere nothing reads. So
    /// the property under test is the round trip and not the write: write, then read, for each.
    ///
    /// The refusal is the other half. A capture larger than the allocation it is going back into
    /// means the id names a different allocation than the one it was read from, and a write that
    /// clamped would report success for a restore it did not do.
    #[test]
    fn a_capture_goes_back_in_by_the_route_it_came_out_of() {
        use std::cell::RefCell;

        const DEVICE: VkDevice = VkDevice(3);
        const HANDLE: VkDeviceMemory = VkDeviceMemory(0x9000);

        thread_local! {
            /// What the driver would have allocated: the bytes `vkMapMemory` hands out.
            static DRIVER_MEMORY: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
        }
        unsafe extern "C" fn map(
            _d: VkDevice,
            _m: VkDeviceMemory,
            _o: VkDeviceSize,
            _s: VkDeviceSize,
            _f: VkMemoryMapFlags,
            out: *mut *mut core::ffi::c_void,
        ) -> VkResult {
            let at = DRIVER_MEMORY.with(|b| b.borrow_mut().as_mut_ptr());
            // SAFETY: the driver's out parameter, written once. The buffer behind `at` is a
            // thread-local that outlives every call in this test and is never resized after the
            // planting below, so the pointer stays good until it unmaps.
            unsafe { *out = at.cast() };
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn unmap(_d: VkDevice, _m: VkDeviceMemory) {}

        let mut d = Driver::new(Account::for_test(None));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkMapMemory(map);
        fns.plant_vkUnmapMemory(unmap);
        d.plant_device(DEVICE, fns);

        let surface = Surface::scanout(64, 8, PixelFormat::Bgra, 256).expect("the system minted");
        let extent = surface.alloc_size() as usize;
        d.plant_scanout_allocation(ObjectId(66), surface);
        d.plant_allocation(ObjectId(70), 4096);
        DRIVER_MEMORY.with(|b| *b.borrow_mut() = vec![0u8; 4096]);
        d.plant_device_local_allocation(ObjectId(72), 4096);

        let round_trip = |d: &Driver, id: u64, len: usize| {
            let src: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            assert_eq!(
                d.memory_write(DEVICE, HANDLE, ObjectId(id), &src),
                Ok(len),
                "the whole capture goes back into {id}"
            );
            let mut back = vec![0u8; len];
            assert_eq!(d.memory_read(DEVICE, HANDLE, ObjectId(id), &mut back), Ok(len));
            assert_eq!(back, src, "and reads back as what went in, for {id}");
        };
        round_trip(&d, 66, extent);
        round_trip(&d, 70, 4096);
        round_trip(&d, 72, 4096);

        // A prefix is ordinary -- the VMM caps what it keeps -- and lands at the front.
        assert_eq!(d.memory_write(DEVICE, HANDLE, ObjectId(70), &[0xab; 8]), Ok(8));
        let mut head = [0u8; 8];
        assert_eq!(d.memory_read(DEVICE, HANDLE, ObjectId(70), &mut head), Ok(8));
        assert_eq!(head, [0xab; 8], "a short write is a prefix, not a failure");

        assert_eq!(
            d.memory_write(DEVICE, HANDLE, ObjectId(70), &vec![0u8; 4096 * 4]),
            Err(MemoryError::LargerThanAllocation),
            "more bytes than the allocation holds is a different allocation, and is refused"
        );
        assert_eq!(
            d.memory_write(DEVICE, HANDLE, ObjectId(999), &[0u8; 4]),
            Err(MemoryError::NoSuchAllocation),
            "and an id nothing is allocated under is refused before any of that"
        );

        d.abandon_planted();
    }

    /// What each backing lends, now that every allocation the host can address owns its bytes.
    ///
    /// A scanout lends the surface. Ordinary host-visible memory lends its minted pages, from the
    /// moment it is allocated and not from the moment it is published -- which is the change, and
    /// the reason a blob can outlive the `vkFreeMemory` that retires the allocation. Memory the
    /// host cannot address lends nothing, because there is nothing of it to lend. And an import
    /// lends nothing *of its own*: the storage is the exporter's, offered once.
    #[test]
    fn every_allocation_the_host_can_address_owns_the_bytes_it_lends() {
        let mut d = Driver::new(Account::for_test(None));

        let surface = Surface::scanout(64, 8, PixelFormat::Bgra, 256).expect("the system minted");
        let addr = surface.host_addr();
        let extent = surface.alloc_size();
        d.plant_scanout_allocation(ObjectId(66), surface);

        d.plant_allocation(ObjectId(70), 4096);
        d.plant_device_local_allocation(ObjectId(72), 4096);

        let lent =
            |d: &Driver, id: u64| d.memory.get(&ObjectId(id)).expect("planted").storage().cloned();

        assert_eq!(
            lent(&d, 66).map(|s| s.span()),
            Some((addr, extent)),
            "a scanout lends the surface's own pages, consulting no table at all"
        );
        let pages = lent(&d, 70).expect("host-visible memory is minted pages");
        assert!(matches!(pages, Storage::Linear(_)));
        assert!(pages.span().1 >= 4096, "and they cover what the guest asked for");
        assert!(lent(&d, 72).is_none(), "memory the host cannot address has no bytes here to lend");

        let borrowed = ResourceBytes::Shared(pages.clone());
        d.plant_imported_allocation(ObjectId(71), 4096, borrowed);
        assert!(
            lent(&d, 71).is_none(),
            "and an import lends nothing of its own: the storage is not its to offer twice"
        );

        d.abandon_planted();
    }

    /// An import keeps what it resolved alive for as long as the driver may reach it.
    ///
    /// The driver is handed an address, and keeps it for the life of the importer's memory. The
    /// storage behind that address is the exporter's record's, or the resource's -- and a guest
    /// process can drop its handle to the resource while another still has the memory bound. A
    /// client exiting is the ordinary case: the compositor's next submit still reads the buffer.
    /// So the importer's record holds the share, and the pages go only when the last holder does.
    #[test]
    fn an_import_holds_the_storage_it_resolved() {
        use super::super::proto::types::VkImportMemoryResourceInfoMESA;

        const DEVICE: VkDevice = VkDevice(3);
        // A whole page, so the mint's rounding does not turn the figure into two numbers.
        const LEN: u64 = 16384;

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

        let mut d = Driver::new(Account::for_test(None));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateMemory(allocate);
        d.plant_device(DEVICE, fns);
        d.plant_memory_types(DEVICE, &[VkMemoryPropertyFlags(HOST_VISIBLE_BIT as _)]);

        let storage = Storage::pages_for_test(LEN as usize, &d.account);
        assert_eq!(d.account.live(), LEN, "the pages are charged from the mint");

        let import = VkImportMemoryResourceInfoMESA {
            sType: VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA,
            pNext: core::ptr::null(),
            resourceId: 7,
        };
        let info = VkMemoryAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: (&raw const import).cast(),
            allocationSize: VkDeviceSize(LEN),
            memoryTypeIndex: 0,
        };
        let resolve = |_| Some(ResourceBytes::Shared(storage.clone()));
        assert!(d.allocate_memory(DEVICE, ObjectId(80), &info, None, &resolve).is_ok());

        // The exporter's record and the resource both let go: the client is gone.
        drop(storage);
        assert_eq!(d.account.live(), LEN, "the import is what keeps the pages now");

        // The importer frees its memory: nothing holds them any more.
        drop(d.memory.remove(&ObjectId(80)));
        assert_eq!(d.account.live(), 0, "and the last holder going is what releases them");

        d.abandon_planted();
    }

    /// An import that resolves to nothing is refused, and costs the host neither memory nor a
    /// call to the driver.
    ///
    /// The resource is gone, or was never this context's, or the chain named resource zero. The
    /// tempting answer is to allocate ordinary memory and report success, and it is the wrong
    /// one twice over: the guest renders into a buffer nobody presents, and the bind that
    /// follows reaches a handle the host never backed -- which is fatal on the ring, not
    /// recoverable. The reference refuses these with the same code, so a guest that sees one is
    /// seeing what it would see on the C.
    #[test]
    fn an_import_that_resolves_to_nothing_is_refused() {
        use super::super::proto::types::VkImportMemoryResourceInfoMESA;
        use std::cell::Cell;

        const DEVICE: VkDevice = VkDevice(3);

        thread_local! {
            static ASKED: Cell<u32> = const { Cell::new(0) };
        }

        unsafe extern "C" fn allocate(
            _d: VkDevice,
            _info: *const VkMemoryAllocateInfo,
            _a: *const VkAllocationCallbacks,
            out: *mut VkDeviceMemory,
        ) -> VkResult {
            ASKED.with(|a| a.set(a.get() + 1));
            // SAFETY: the caller's local.
            unsafe { *out = VkDeviceMemory(0x9000) };
            VkResult::VK_SUCCESS
        }

        let mut d = Driver::new(Account::for_test(None));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateMemory(allocate);
        d.plant_device(DEVICE, fns);
        d.plant_memory_types(DEVICE, &[VkMemoryPropertyFlags(HOST_VISIBLE_BIT as _)]);

        // Two ways for a chain to name nothing, and one answer to both: a resource the resolver
        // cannot find, and the wire's spelling of no resource at all.
        for resource_id in [17, 0] {
            let import = VkImportMemoryResourceInfoMESA {
                sType: VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_RESOURCE_INFO_MESA,
                pNext: core::ptr::null(),
                resourceId: resource_id,
            };
            let info = VkMemoryAllocateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                pNext: (&raw const import).cast(),
                allocationSize: VkDeviceSize(16384),
                memoryTypeIndex: 0,
            };
            let refused = d.allocate_memory(DEVICE, ObjectId(80), &info, None, &|_| None);
            assert!(
                matches!(
                    refused,
                    Err(NoMemory::Driver(VkResult::VK_ERROR_INVALID_EXTERNAL_HANDLE))
                ),
                "res {resource_id} names no storage, so there is nothing to alias"
            );
        }

        assert_eq!(ASKED.with(Cell::get), 0, "and the driver was never asked to allocate");
        assert_eq!(d.account.live(), 0, "nor was a byte charged for memory nobody got");
        assert!(!d.memory.contains_key(&ObjectId(80)), "and no record was filed under the id");

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
            Some(ResourceHandle::new(7)),
            "found past the head, with its resource"
        );

        head.pNext = core::ptr::null();
        assert_eq!(imported_resource((&raw const head).cast()), None, "and none when absent");

        // Resource zero is how the wire spells "no resource" -- it must not resolve to a handle,
        // because a handle is what the renderer would then go looking for. The chain still asked
        // to import, though, which is why the outer answer stays `Some`: it names nothing, and
        // an import naming nothing is refused rather than allocated.
        tail.resourceId = 0;
        head.pNext = (&raw const tail).cast();
        assert_eq!(imported_resource((&raw const head).cast()), Some(None));
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

    /// An image the guest shares outside this device is created with rows the host can address,
    /// and only that image: the rule keys on external handle types, and leaves alone an image
    /// that already chose its layout by DRM format modifier.
    #[test]
    fn an_image_the_guest_shares_is_created_linear() {
        use super::super::proto::types::VkExternalMemoryHandleTypeFlags;
        const INPUT: u32 = VkImageUsageFlagBits::VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT.0 as u32;
        const SAMPLED: u32 = VkImageUsageFlagBits::VK_IMAGE_USAGE_SAMPLED_BIT.0 as u32;

        let external = VkExternalMemoryImageCreateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            pNext: core::ptr::null(),
            handleTypes: VkExternalMemoryHandleTypeFlags(1),
        };
        let none = VkExternalMemoryImageCreateInfo {
            handleTypes: VkExternalMemoryHandleTypeFlags(0),
            ..external
        };
        let image = |tiling, chain: *const core::ffi::c_void| VkImageCreateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            pNext: chain,
            tiling,
            usage: VkImageUsageFlags(SAMPLED | INPUT),
            ..Default::default()
        };

        let shared = external_images_are_linear(&image(
            VkImageTiling::VK_IMAGE_TILING_OPTIMAL,
            (&raw const external).cast(),
        ));
        assert_eq!(shared.tiling, VkImageTiling::VK_IMAGE_TILING_LINEAR, "made addressable");
        assert_eq!(shared.usage.0, SAMPLED, "and never an input attachment");
        assert_eq!(shared.pNext, (&raw const external).cast(), "the chain is the guest's");

        let untouched = |why: &str, info: VkImageCreateInfo| {
            let out = external_images_are_linear(&info);
            assert_eq!(out.tiling, info.tiling, "{why}");
            assert_eq!(out.usage, info.usage, "{why}");
        };
        untouched(
            "an image kept to this device may be as opaque as the driver likes",
            image(VkImageTiling::VK_IMAGE_TILING_OPTIMAL, core::ptr::null()),
        );
        untouched(
            "external memory naming no handle types shares nothing",
            image(VkImageTiling::VK_IMAGE_TILING_OPTIMAL, (&raw const none).cast()),
        );
        untouched(
            "an image that chose its layout by modifier has already answered the question",
            image(
                VkImageTiling::VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
                (&raw const external).cast(),
            ),
        );
    }

    /// Every allocation the host can address is backed by pages this renderer minted, handed to
    /// the driver by host-pointer import -- so that the pages, and not a mapping the driver
    /// lends, are what a resource holds a share of.
    ///
    /// Four things have to be true of it at once. The driver was handed our pages and nothing
    /// else. Publishing it maps nothing: the address is the pages', so `vkMapMemory` is never
    /// asked. The share carries the charge, so the pages stay counted for as long as any holder
    /// has them, whatever happened to the allocation. And it does not wait for the guest to say
    /// it means to export: an allocation with no `VkExportMemoryAllocateInfo` is minted the same
    /// way, because the guest may export it at any later moment and an allocation that was not
    /// minted has only the driver's mapping to publish.
    #[test]
    fn host_addressable_memory_is_backed_by_pages_this_renderer_minted() {
        use super::super::budget::Budget;
        use super::super::proto::types::{
            VkExportMemoryAllocateInfo, VkExternalMemoryHandleTypeFlags,
        };
        use std::cell::Cell;

        const DEVICE: VkDevice = VkDevice(3);
        const ASKED: u64 = 100_000;

        thread_local! {
            /// What the driver was handed: the imported pointer, and the size it was told.
            static GIVEN: Cell<(usize, u64)> = const { Cell::new((0, 0)) };
            static MAPPED: Cell<bool> = const { Cell::new(false) };
        }
        unsafe extern "C" fn allocate(
            _d: VkDevice,
            info: *const VkMemoryAllocateInfo,
            _a: *const VkAllocationCallbacks,
            out: *mut VkDeviceMemory,
        ) -> VkResult {
            // SAFETY: the caller's locals, and a chain the caller built for this call.
            unsafe {
                let info = &*info;
                let mut node = info.pNext;
                let mut ptr = 0usize;
                while !node.is_null() {
                    let base = &*node.cast::<VkBaseInStructure>();
                    if base.sType
                        == VkStructureType::VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT
                    {
                        ptr = (*node.cast::<VkImportMemoryHostPointerInfoEXT>()).pHostPointer
                            as usize;
                    }
                    node = base.pNext.cast();
                }
                GIVEN.with(|g| g.set((ptr, info.allocationSize.0)));
                *out = VkDeviceMemory(0x9000);
            }
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn map(
            _d: VkDevice,
            _m: VkDeviceMemory,
            _o: VkDeviceSize,
            _s: VkDeviceSize,
            _f: VkMemoryMapFlags,
            _out: *mut *mut core::ffi::c_void,
        ) -> VkResult {
            MAPPED.with(|m| m.set(true));
            VkResult::VK_ERROR_MEMORY_MAP_FAILED
        }
        unsafe extern "C" fn free(
            _d: VkDevice,
            _m: VkDeviceMemory,
            _a: *const VkAllocationCallbacks,
        ) {
        }

        let budget = Budget::with_cap(None, false);
        let one = crate::ids::ContextId::new(1).expect("not zero");
        let mut d =
            Driver::new(Account::open(&budget, crate::venus::vkr::ContextKey::for_test(one)));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateMemory(allocate);
        fns.plant_vkMapMemory(map);
        fns.plant_vkFreeMemory(free);
        d.plant_device(DEVICE, fns);
        d.plant_memory_types(
            DEVICE,
            &[VkMemoryPropertyFlags(HOST_VISIBLE_BIT | HOST_COHERENT_BIT | HOST_CACHED_BIT)],
        );

        let export = VkExportMemoryAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            pNext: core::ptr::null(),
            handleTypes: VkExternalMemoryHandleTypeFlags(1),
        };
        let info = VkMemoryAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: (&raw const export).cast(),
            allocationSize: VkDeviceSize(ASKED),
            memoryTypeIndex: 0,
        };
        d.allocate_memory(DEVICE, ObjectId(1), &info, None, &|_| None).expect("no cap");

        let (ptr, told) = GIVEN.with(Cell::get);
        let page = crate::guest_mem::page_size();
        assert_ne!(ptr, 0, "the driver was handed pages rather than asked for memory");
        assert_eq!(ptr % page, 0, "pages start on a page, which every import alignment divides");
        let padded = pad_for_blob(ASKED, Some(VkMemoryPropertyFlags(HOST_VISIBLE_BIT)), false);
        assert_eq!(told, padded, "and told the guest's padded figure, which the pages cover");
        let Backing::Owned { storage, .. } =
            &d.memory.get(&ObjectId(1)).expect("allocated").backing
        else {
            panic!("host-addressable memory is minted pages");
        };
        let span = storage.span();
        assert_eq!(span.0, ptr, "the address a later import aliases is the one the driver got");
        assert!(span.1 >= padded, "the pages cover everything the driver was told it has");
        assert_eq!(budget.live(), span.1, "charged for the pages, as 'exported pages'");

        // Publishing is a matter of saying where the pages are. Nothing is mapped.
        let (published, share) = d.memory_export(ObjectId(1), ASKED).expect("exports");
        assert_eq!(published.addr, ptr);
        assert!(published.write_back, "coherent and cached, as the type says");
        assert!(!MAPPED.with(Cell::get), "the driver was never asked to map what it imported");
        assert!(matches!(share, Storage::Linear(_)), "and the share is the pages");
        assert_eq!(
            share.surface().err(),
            Some(NoSurface::NotDedicated),
            "which know why they are not a surface: this export dedicated no image"
        );
        assert_eq!(share.span(), span, "resolving to exactly what the driver was handed");
        assert_eq!(budget.live_for(one), span.1, "shared, and still this context's while it lives");

        // The census reads the pages themselves -- coherent host memory -- not a driver mapping.
        let mut buf = vec![0u8; 16];
        assert_eq!(d.memory_read(DEVICE, VkDeviceMemory(0x9000), ObjectId(1), &mut buf), Ok(16));
        assert!(!MAPPED.with(Cell::get), "still never mapped");

        // The allocation goes; the share keeps the pages, and the ledger keeps counting them.
        d.free_memory(DEVICE, VkDeviceMemory(0x9000), ObjectId(1));
        assert_eq!(budget.live(), span.1, "the share is what keeps the pages counted now");
        drop(share);
        assert_eq!(budget.live(), 0, "and the last share going is what credits them");

        // And plain host-visible memory, with no export info at all, takes the same path. This
        // is the whole change: the gate used to be the guest's `VkExportMemoryAllocateInfo`, and
        // an allocation that had not asked was published as the driver's own `vkMapMemory`
        // pointer -- an address the VMM maps into guest physical memory and `vkFreeMemory` then
        // unmaps underneath it.
        let plain = VkMemoryAllocateInfo { pNext: core::ptr::null(), ..info };
        GIVEN.with(|g| g.set((0, 0)));
        d.allocate_memory(DEVICE, ObjectId(2), &plain, None, &|_| None).expect("no cap");
        assert_ne!(GIVEN.with(Cell::get).0, 0, "it was handed pages too");
        let (_, plain_share) = d.memory_export(ObjectId(2), ASKED).expect("it exports");
        d.free_memory(DEVICE, VkDeviceMemory(0x9000), ObjectId(2));
        assert!(
            matches!(plain_share, Storage::Linear(_)),
            "and its share outlives the free, which is what the hypervisor's mapping needs"
        );
        drop(plain_share);

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
                tiling: VkImageTiling::VK_IMAGE_TILING_LINEAR,
                ..Default::default()
            },
        );
        // The same image, opaque. The layout stub answers with the same plausible pitch for it,
        // which is exactly why the refusal below has to come from the tiling and not the pitch.
        const OPAQUE: VkImage = VkImage(0x4200);
        d.note_image(
            OPAQUE,
            &VkImageCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                format: VkFormat::VK_FORMAT_B8G8R8A8_UNORM,
                extent: VkExtent3D { width: W, height: H, depth: 1 },
                tiling: VkImageTiling::VK_IMAGE_TILING_OPTIMAL,
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

        let opaque_dedicated = VkMemoryDedicatedAllocateInfo { image: OPAQUE, ..dedicated };
        let opaque_export =
            VkExportMemoryAllocateInfo { pNext: (&raw const opaque_dedicated).cast(), ..export };
        let opaque = VkMemoryAllocateInfo { pNext: (&raw const opaque_export).cast(), ..info };
        d.allocate_memory(DEVICE, ObjectId(2), &opaque, None, &|_| None).expect("no cap");
        assert!(
            d.memory_surface_id(ObjectId(2)).is_none(),
            "an opaque image has no rows to alias, whatever pitch the driver quotes for it"
        );
        d.free_memory(DEVICE, VkDeviceMemory(0x9000), ObjectId(2));

        d.free_memory(DEVICE, VkDeviceMemory(0x9000), ObjectId(1));
        assert_eq!(d.account.live(), 0, "and the surface's pages come back with it");

        d.abandon_planted();
    }

    /// An import aliases bytes another allocation already paid for, so charging it would bill one
    /// buffer twice and refuse work the host has room for. The same rule as the census, which is
    /// why both read the one `Backing` rather than a flag each.
    ///
    /// Only an import that resolved, though. One naming a resource with no storage behind it is
    /// forwarded to the driver as the ordinary allocation it then is, and that is host memory
    /// like any other: charged, and refused when there is no room.
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
        let pages = Storage::pages_for_test(4096, &Account::for_test(None));
        let resolve = |_| Some(ResourceBytes::Shared(pages.clone()));
        assert!(
            d.allocate_memory(DEVICE, ObjectId(2), &info, None, &resolve).is_ok(),
            "an import is admitted with no room left, because it takes none"
        );
        assert_eq!(d.account.live(), 900, "and the ledger did not move");

        assert!(
            d.allocate_memory(DEVICE, ObjectId(3), &info, None, &|_| None).is_err(),
            "an import that resolved to nothing is an ordinary allocation, and there is no room"
        );
        assert_eq!(d.account.live(), 900, "a refusal costs nothing");

        d.abandon_planted();
    }

    /// bigger than what backs it, or asking to map memory the host cannot address. The second of
    /// those is the one with teeth -- two resources over one storage is a state neither holder
    /// could detect afterwards.
    #[test]
    fn memory_is_published_once_and_leaves_the_census_when_it_is() {
        const DEVICE: VkDevice = VkDevice(3);
        const MEM: ObjectId = ObjectId(12);
        const LOCAL: ObjectId = ObjectId(13);
        const SIZE: u64 = 128 * 1024;

        unsafe extern "C" fn free(
            _d: VkDevice,
            _m: VkDeviceMemory,
            _a: *const VkAllocationCallbacks,
        ) {
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkFreeMemory(free);

        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(DEVICE, fns);
        driver.plant_allocation(MEM, SIZE);
        driver.plant_device_local_allocation(LOCAL, SIZE);
        let handle = VkDeviceMemory(0xd0);

        assert_eq!(driver.memory_census().len(), 2, "both are live and unexported");

        // Memory nobody allocated.
        assert_eq!(driver.memory_export(ObjectId(999), SIZE), Err(ExportError::NoSuchAllocation));
        // A blob bigger than the storage would publish whatever follows it in this process.
        assert_eq!(driver.memory_export(MEM, SIZE + 1), Err(ExportError::LargerThanAllocation));
        // Memory the host cannot address has nothing to publish. It is the one kind still left
        // to the driver, precisely because there is no address to hand out.
        assert_eq!(driver.memory_export(LOCAL, SIZE), Err(ExportError::NotHostVisible));

        // The export itself. Coherent and cached on the host, so the guest may map it cached.
        let (published, share) = driver.memory_export(MEM, SIZE).expect("it publishes");
        assert!(published.write_back, "coherent and cached, as the type says");
        assert_eq!(share.span().0, published.addr, "the address is the share's own");
        assert_eq!(
            driver.memory_export(MEM, SIZE).err(),
            Some(ExportError::AlreadyExported),
            "the mark is the storage's: exporting twice would give two resources one storage"
        );

        // Memory the host reaches through a cache it has to flush is memory the guest must not
        // map cached, and the answer comes from the type it was allocated from -- not from a
        // default that happens to be right for the driver we run on today.
        const UNCACHED: ObjectId = ObjectId(14);
        driver.plant_allocation_of(UNCACHED, SIZE, HOST_VISIBLE_BIT | HOST_COHERENT_BIT);
        let (uncached, _) = driver.memory_export(UNCACHED, SIZE).expect("it publishes");
        assert!(!uncached.write_back);
        driver.free_memory(DEVICE, handle, UNCACHED);

        // The census stops reporting it: its bytes are the blob's, captured where they live.
        let census = driver.memory_census();
        assert_eq!(census.len(), 1, "the exported allocation is no longer the census's to read");
        assert_eq!(census[0].id, LOCAL);

        // Freeing it retires the record and nothing else. The share is what the VMM's mapping
        // stands on, and it stands: this is the sequence -- allocate, export as a blob, map into
        // the guest, free -- that used to leave the hypervisor pointing at unmapped host memory.
        driver.free_memory(DEVICE, handle, MEM);
        assert_eq!(
            driver.memory_export(MEM, SIZE),
            Err(ExportError::NoSuchAllocation),
            "and there is no allocation left to export"
        );
        assert_eq!(share.span().1, SIZE, "while the pages the guest is mapped over are still here");
        drop(share);

        driver.free_memory(DEVICE, handle, LOCAL);

        // A scanout publishes the same way, and the share it lends is the surface itself. The
        // storage travels, never a name in this context's table -- which is what lets a
        // compositor in another context import the blob after the exporter is gone.
        const SCAN: ObjectId = ObjectId(15);
        let surface = Surface::scanout(64, 8, PixelFormat::Bgra, 256).expect("the system minted");
        let (scan_addr, scan_extent) = (surface.host_addr(), surface.alloc_size());
        driver.plant_scanout_allocation(SCAN, surface);
        let (scan, scan_share) = driver.memory_export(SCAN, 1024).expect("a scanout publishes too");
        assert!(matches!(scan_share, Storage::Texture(_)), "the surface itself, not a copy of it");
        assert_eq!(
            scan_share.span(),
            (scan_addr, scan_extent),
            "and it is the surface's own pages"
        );
        assert_eq!(scan.addr, scan_addr, "which is the address the VMM was handed");

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

    /// Every command that names a query by index is held to the pool before the driver sees it.
    ///
    /// Vulkan makes the index the caller's promise and a miss undefined, and this host's driver
    /// takes the promise at its word: a host-side reset zeroes host memory at `first + i`, a
    /// begin reads a remap table at `query`, the recorded commands address the pool's buffer on
    /// the GPU, and a read-back writes `count` results `stride` apart into whatever it was
    /// handed. Here the caller is a guest, so all eight are measured against what the pool was
    /// created as, and no refusal reaches a planted entry point.
    #[test]
    fn every_query_index_is_held_to_the_pool() {
        use super::super::proto::types::{
            VkCommandPool, VkPipelineStageFlagBits, VkQueryControlFlags,
            VkQueryPipelineStatisticFlags, VkQueryPoolCreateInfo, VkQueryResultFlags, VkQueryType,
        };
        use std::cell::RefCell;

        const DEVICE: VkDevice = VkDevice(3);
        const CB: VkCommandBuffer = VkCommandBuffer(0x30);
        const POOL: VkQueryPool = VkQueryPool(0x50);
        const STATS: VkQueryPool = VkQueryPool(0x51);
        const PERF: VkQueryPool = VkQueryPool(0x52);

        // What the driver was asked, by entry point: first (or the query) and count.
        thread_local! {
            static ASKED: RefCell<Vec<(&'static str, u32, u32)>> = const { RefCell::new(Vec::new()) };
        }
        fn asked(what: &'static str, first: u32, count: u32) {
            ASKED.with_borrow_mut(|a| a.push((what, first, count)));
        }
        unsafe extern "C" fn create(
            _d: VkDevice,
            info: *const VkQueryPoolCreateInfo,
            _a: *const VkAllocationCallbacks,
            out: *mut VkQueryPool,
        ) -> VkResult {
            // SAFETY: the caller passes a struct and a local of its own.
            unsafe {
                *out = match (*info).queryType {
                    VkQueryType::VK_QUERY_TYPE_PIPELINE_STATISTICS => STATS,
                    VkQueryType::VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR => PERF,
                    _ => POOL,
                }
            };
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn reset(_d: VkDevice, _p: VkQueryPool, first: u32, count: u32) {
            asked("reset", first, count);
        }
        unsafe extern "C" fn destroy(
            _d: VkDevice,
            _p: VkQueryPool,
            _a: *const VkAllocationCallbacks,
        ) {
        }
        unsafe extern "C" fn wait_idle(_d: VkDevice) -> VkResult {
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn destroy_device(_d: VkDevice, _a: *const VkAllocationCallbacks) {}
        unsafe extern "C" fn results(
            _d: VkDevice,
            _p: VkQueryPool,
            first: u32,
            count: u32,
            size: usize,
            _data: *mut core::ffi::c_void,
            _stride: VkDeviceSize,
            _flags: VkQueryResultFlags,
        ) -> VkResult {
            asked("results", first, count);
            ASKED.with_borrow_mut(|a| a.push(("bytes", size as u32, 0)));
            VkResult::VK_NOT_READY
        }
        unsafe extern "C" fn begin(
            _cb: VkCommandBuffer,
            _p: VkQueryPool,
            query: u32,
            _f: VkQueryControlFlags,
        ) {
            asked("begin", query, 1);
        }
        unsafe extern "C" fn end(_cb: VkCommandBuffer, _p: VkQueryPool, query: u32) {
            asked("end", query, 1);
        }
        unsafe extern "C" fn cmd_reset(
            _cb: VkCommandBuffer,
            _p: VkQueryPool,
            first: u32,
            count: u32,
        ) {
            asked("cmd_reset", first, count);
        }
        unsafe extern "C" fn timestamp(
            _cb: VkCommandBuffer,
            _s: VkPipelineStageFlagBits,
            _p: VkQueryPool,
            query: u32,
        ) {
            asked("timestamp", query, 1);
        }
        unsafe extern "C" fn copy(
            _cb: VkCommandBuffer,
            _p: VkQueryPool,
            first: u32,
            count: u32,
            _b: VkBuffer,
            _o: VkDeviceSize,
            _s: VkDeviceSize,
            _f: VkQueryResultFlags,
        ) {
            asked("copy", first, count);
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkCreateQueryPool(create);
        fns.plant_vkResetQueryPool(reset);
        fns.plant_vkGetQueryPoolResults(results);
        fns.plant_vkCmdBeginQuery(begin);
        fns.plant_vkCmdEndQuery(end);
        fns.plant_vkCmdResetQueryPool(cmd_reset);
        fns.plant_vkCmdWriteTimestamp(timestamp);
        fns.plant_vkCmdCopyQueryPoolResults(copy);
        fns.plant_vkDestroyQueryPool(destroy);
        fns.plant_vkDeviceWaitIdle(wait_idle);
        fns.plant_vkDestroyDevice(destroy_device);
        let mut d = Driver::new(Account::for_test(None));
        d.plant_device(DEVICE, fns);
        d.plant_pool(DEVICE, VkCommandPool(0x20), &[(CB, ObjectId(9))]);

        let info = VkQueryPoolCreateInfo {
            queryType: VkQueryType::VK_QUERY_TYPE_TIMESTAMP,
            queryCount: 4,
            ..Default::default()
        };
        assert_eq!(d.create_query_pool(DEVICE, &info, None), Ok(POOL));

        const NONE: VkQueryResultFlags = VkQueryResultFlags(0);
        const WIDE: VkQueryResultFlags =
            VkQueryResultFlags(VkQueryResultFlagBits::VK_QUERY_RESULT_64_BIT.0 as u32);
        const WIDE_AVAIL: VkQueryResultFlags = VkQueryResultFlags(
            (VkQueryResultFlagBits::VK_QUERY_RESULT_64_BIT.0
                | VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_AVAILABILITY_BIT.0)
                as u32,
        );
        const WITH_STATUS: VkQueryResultFlags =
            VkQueryResultFlags(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_STATUS_BIT_KHR.0 as u32);
        const STAGE: VkPipelineStageFlagBits = VkPipelineStageFlagBits(1);
        const BUF: VkBuffer = VkBuffer(0x60);
        let mut buf = [0u8; 64];
        let out_of_pool: Result<(), QueryRefused> = Err(QueryRefused::OutOfPool);
        let read_out_of_pool: Result<VkResult, QueryRefused> = Err(QueryRefused::OutOfPool);

        // The whole pool, by every command that takes a range, and the last query by every one
        // that takes a single index: all reach the driver with the guest's own numbers.
        assert_eq!(d.reset_query_pool(DEVICE, POOL, 0, 4), Ok(()));
        assert_eq!(d.cmd_reset_query_pool(CB, POOL, 0, 4), Ok(()));
        assert_eq!(
            d.cmd_copy_query_pool_results(
                CB,
                POOL,
                0,
                4,
                BUF,
                VkDeviceSize(0),
                VkDeviceSize(4),
                NONE
            ),
            Ok(())
        );
        assert_eq!(d.cmd_begin_query(CB, POOL, 3, VkQueryControlFlags(0)), Ok(()));
        assert_eq!(d.cmd_end_query(CB, POOL, 3), Ok(()));
        assert_eq!(d.cmd_write_timestamp(CB, STAGE, POOL, 3), Ok(()));
        let r = d.query_pool_results(DEVICE, POOL, 0, 4, &mut buf[..16], VkDeviceSize(4), NONE);
        assert_eq!(r, Ok(VkResult::VK_NOT_READY), "the driver's answer, as it gave it");
        ASKED.with_borrow(|a| {
            assert_eq!(
                a.as_slice(),
                [
                    ("reset", 0, 4),
                    ("cmd_reset", 0, 4),
                    ("copy", 0, 4),
                    ("begin", 3, 1),
                    ("end", 3, 1),
                    ("timestamp", 3, 1),
                    ("results", 0, 4),
                    ("bytes", 16, 0),
                ],
                "each with its own arguments, and the read told the room's own length"
            );
        });
        ASKED.with_borrow_mut(Vec::clear);

        // One past the end, by every command. The one the C forwards blindly into the driver's
        // host memory is `reset`; the rest would address the pool's buffer past its end.
        assert_eq!(d.reset_query_pool(DEVICE, POOL, 1, 4), out_of_pool);
        assert_eq!(
            d.reset_query_pool(DEVICE, POOL, 0, 0x7fff_ffff),
            out_of_pool,
            "a scribble the length of the heap"
        );
        assert_eq!(d.cmd_reset_query_pool(CB, POOL, 4, 1), out_of_pool);
        assert_eq!(
            d.cmd_copy_query_pool_results(
                CB,
                POOL,
                3,
                2,
                BUF,
                VkDeviceSize(0),
                VkDeviceSize(4),
                NONE
            ),
            out_of_pool
        );
        assert_eq!(d.cmd_begin_query(CB, POOL, 4, VkQueryControlFlags(0)), out_of_pool);
        assert_eq!(d.cmd_end_query(CB, POOL, u32::MAX), out_of_pool);
        assert_eq!(d.cmd_write_timestamp(CB, STAGE, POOL, 4), out_of_pool);
        assert_eq!(
            d.query_pool_results(DEVICE, POOL, 3, 2, &mut buf, VkDeviceSize(4), NONE),
            read_out_of_pool
        );
        assert_eq!(
            d.query_pool_results(DEVICE, POOL, u32::MAX, 1, &mut buf, VkDeviceSize(4), NONE),
            read_out_of_pool,
            "and a first query that wraps is past it too"
        );
        // An empty range at the very end is within the pool; one past that is not.
        assert_eq!(d.cmd_reset_query_pool(CB, POOL, 4, 0), Ok(()));
        assert_eq!(d.cmd_reset_query_pool(CB, POOL, 5, 0), out_of_pool);
        ASKED.with_borrow(|a| {
            assert_eq!(a.as_slice(), [("cmd_reset", 4, 0)], "no refusal reached the driver")
        });
        ASKED.with_borrow_mut(Vec::clear);

        // The read-back is held to the room as well. Four 32-bit timestamps four apart need
        // sixteen bytes; fifteen is a miss.
        let r = d.query_pool_results(DEVICE, POOL, 0, 4, &mut buf[..15], VkDeviceSize(4), NONE);
        assert_eq!(r, Err(QueryRefused::OutOfRoom));
        // 64-bit results are twice as wide, and availability and status each add a word of the
        // same width.
        let r = d.query_pool_results(DEVICE, POOL, 0, 2, &mut buf[..16], VkDeviceSize(8), WIDE);
        assert_eq!(r, Ok(VkResult::VK_NOT_READY));
        let r =
            d.query_pool_results(DEVICE, POOL, 0, 2, &mut buf[..16], VkDeviceSize(8), WIDE_AVAIL);
        assert_eq!(
            r,
            Err(QueryRefused::OutOfRoom),
            "the last result's availability word does not fit"
        );
        let r =
            d.query_pool_results(DEVICE, POOL, 0, 2, &mut buf[..32], VkDeviceSize(16), WIDE_AVAIL);
        assert_eq!(r, Ok(VkResult::VK_NOT_READY), "laid sixteen apart in thirty-two, it does");
        let r =
            d.query_pool_results(DEVICE, POOL, 0, 1, &mut buf[..4], VkDeviceSize(4), WITH_STATUS);
        assert_eq!(r, Err(QueryRefused::OutOfRoom), "a status word is a second word");
        let r =
            d.query_pool_results(DEVICE, POOL, 0, 1, &mut buf[..8], VkDeviceSize(4), WITH_STATUS);
        assert_eq!(r, Ok(VkResult::VK_NOT_READY));
        // A stride the arithmetic cannot hold is a buffer nothing holds either.
        let r = d.query_pool_results(DEVICE, POOL, 0, 4, &mut buf, VkDeviceSize(u64::MAX), NONE);
        assert_eq!(r, Err(QueryRefused::OutOfRoom));
        // Nothing asked for needs no room at all.
        let r = d.query_pool_results(DEVICE, POOL, 4, 0, &mut buf[..0], VkDeviceSize(4), NONE);
        assert_eq!(r, Ok(VkResult::VK_NOT_READY));

        // A pipeline-statistics pool answers one value per statistic counted.
        let info = VkQueryPoolCreateInfo {
            queryType: VkQueryType::VK_QUERY_TYPE_PIPELINE_STATISTICS,
            queryCount: 1,
            pipelineStatistics: VkQueryPipelineStatisticFlags(0b1011),
            ..Default::default()
        };
        assert_eq!(d.create_query_pool(DEVICE, &info, None), Ok(STATS));
        let r = d.query_pool_results(DEVICE, STATS, 0, 1, &mut buf[..12], VkDeviceSize(12), NONE);
        assert_eq!(r, Ok(VkResult::VK_NOT_READY), "three statistics, three words");
        let r = d.query_pool_results(DEVICE, STATS, 0, 1, &mut buf[..8], VkDeviceSize(8), NONE);
        assert_eq!(r, Err(QueryRefused::OutOfRoom));

        // A kind whose result this renderer cannot size is still a pool the driver decides on,
        // and is still indexed; only its read-back into guest room is refused.
        let info = VkQueryPoolCreateInfo {
            queryType: VkQueryType::VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR,
            queryCount: 2,
            ..Default::default()
        };
        assert_eq!(
            d.create_query_pool(DEVICE, &info, None),
            Ok(PERF),
            "the driver's call, not ours"
        );
        assert_eq!(d.cmd_begin_query(CB, PERF, 1, VkQueryControlFlags(0)), Ok(()));
        assert_eq!(d.cmd_begin_query(CB, PERF, 2, VkQueryControlFlags(0)), out_of_pool);
        let r = d.query_pool_results(DEVICE, PERF, 0, 1, &mut buf, VkDeviceSize(64), NONE);
        assert_eq!(r, Err(QueryRefused::Unsized));
        // Whereas a kind this host advertises through an extension is sized like any other.
        let info = VkQueryPoolCreateInfo {
            queryType: VkQueryType::VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT,
            queryCount: 1,
            ..Default::default()
        };
        assert_eq!(QueryFacts::of(&info).values, Some(1), "one value: primitives generated");
        let info = VkQueryPoolCreateInfo {
            queryType: VkQueryType::VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT,
            ..Default::default()
        };
        assert_eq!(QueryFacts::of(&info).values, Some(2), "two: written, and needed");
        let info = VkQueryPoolCreateInfo {
            queryType: VkQueryType::VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR,
            ..Default::default()
        };
        assert_eq!(QueryFacts::of(&info).values, Some(0), "none: the status word is the result");

        // A pool this renderer has no record of is not sized and not indexed, so it is not
        // touched -- which is what a destroyed pool becomes: a recycled handle finds no record.
        let r = d.cmd_end_query(CB, VkQueryPool(0x99), 0);
        assert_eq!(r, Err(QueryRefused::UnknownPool));
        d.forget_query_pool(POOL);
        let r = d.query_pool_results(DEVICE, POOL, 0, 1, &mut buf, VkDeviceSize(4), NONE);
        assert_eq!(r, Err(QueryRefused::UnknownPool));
        assert_eq!(d.reset_query_pool(DEVICE, POOL, 0, 1), Err(QueryRefused::UnknownPool));

        // A command buffer with no device behind it, and a device with no table.
        assert_eq!(d.cmd_end_query(VkCommandBuffer(0x31), STATS, 0), Err(QueryRefused::NoDevice));
        assert_eq!(d.reset_query_pool(VkDevice(4), STATS, 0, 1), Err(QueryRefused::NoDevice));

        // The other place a pool dies: the guest left it live and the device's teardown took
        // it. Its record goes too, so the handle the driver may now reuse vouches for nothing.
        let doomed = [
            Doomed {
                id: ObjectId(41),
                ty: VkObjectType::VK_OBJECT_TYPE_QUERY_POOL,
                handle: STATS.host(),
                device: Some(DEVICE),
            },
            Doomed {
                id: ObjectId(42),
                ty: VkObjectType::VK_OBJECT_TYPE_QUERY_POOL,
                handle: PERF.host(),
                device: Some(DEVICE),
            },
        ];
        d.destroy_device(DEVICE, &doomed);
        assert!(d.query_pools.is_empty(), "no record outlives the pool it describes");

        ASKED.with_borrow(|a| {
            let reached: Vec<&str> = a.iter().map(|(w, _, _)| *w).collect();
            assert_eq!(
                reached,
                [
                    "results", "bytes", "results", "bytes", "results", "bytes", "results", "bytes",
                    "results", "bytes", "begin"
                ],
                "only what fit reached the driver, and no refusal did"
            );
        });
        d.abandon_planted();
    }
}
