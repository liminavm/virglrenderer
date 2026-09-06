// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! One guest's venus context: its object table, its poison, and the loop that drains a submission.
//!
//! A submission is a stream of commands, not one command. The loop mirrors
//! `vkr_context_submit_cmd`: clear the per-command poison, dispatch, and stop the whole batch the
//! moment the stream itself becomes untrustworthy. The distinction is the point -- a lost command
//! costs the guest one operation, a poisoned context costs it the ring, and only the second one
//! ends the loop.

use bumpalo::Bump;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::{ContextId, ResourceHandle, RingId};

use super::budget::{Account, Budget};
use super::cs::Handle;
use super::cs::{AllOfIt, Decoder, Dispatched, Encoder};
use super::cs::{Guest, HostHandle, ObjectId};
use super::driver::{self, Driver, ExportError, Exported, MemoryError, NoSyncFd};
use super::journal::{self, Journal, Seq};
use super::monitor::Monitor;
use super::objects::{ObjectKey, Shared};
use super::proto::serialize::{Commands, vn_command_name, vn_dispatch_command};
use super::proto::types::{
    VkCommandStreamDescriptionMESA, VkCommandTypeEXT, VkDeviceMemory, VkFlags,
    VkMemoryResourceAllocationSizePropertiesMESA, VkObjectType, VkPhysicalDevice, VkResult,
    VkRingCreateInfoMESA, VkRingMonitorInfoMESA, vn_command_vkAllocateCommandBuffers,
    vn_command_vkAllocateDescriptorSets, vn_command_vkAllocateMemory,
    vn_command_vkBeginCommandBuffer, vn_command_vkBindBufferMemory, vn_command_vkBindBufferMemory2,
    vn_command_vkBindImageMemory, vn_command_vkBindImageMemory2, vn_command_vkCmdBeginQuery,
    vn_command_vkCmdBeginRenderPass, vn_command_vkCmdBindDescriptorSets,
    vn_command_vkCmdBindPipeline, vn_command_vkCmdBindVertexBuffers, vn_command_vkCmdBlitImage,
    vn_command_vkCmdClearAttachments, vn_command_vkCmdClearColorImage, vn_command_vkCmdCopyBuffer,
    vn_command_vkCmdCopyBufferToImage, vn_command_vkCmdCopyImage,
    vn_command_vkCmdCopyImageToBuffer, vn_command_vkCmdCopyQueryPoolResults, vn_command_vkCmdDraw,
    vn_command_vkCmdEndQuery, vn_command_vkCmdEndRenderPass, vn_command_vkCmdFillBuffer,
    vn_command_vkCmdPipelineBarrier, vn_command_vkCmdPushConstants, vn_command_vkCmdResetQueryPool,
    vn_command_vkCmdSetScissor, vn_command_vkCmdSetViewport, vn_command_vkCmdWriteTimestamp,
    vn_command_vkCreateBuffer, vn_command_vkCreateCommandPool, vn_command_vkCreateDescriptorPool,
    vn_command_vkCreateDescriptorSetLayout, vn_command_vkCreateDevice, vn_command_vkCreateFence,
    vn_command_vkCreateFramebuffer, vn_command_vkCreateGraphicsPipelines, vn_command_vkCreateImage,
    vn_command_vkCreateImageView, vn_command_vkCreateInstance, vn_command_vkCreatePipelineCache,
    vn_command_vkCreatePipelineLayout, vn_command_vkCreateQueryPool, vn_command_vkCreateRenderPass,
    vn_command_vkCreateRingMESA, vn_command_vkCreateSampler,
    vn_command_vkCreateSamplerYcbcrConversion, vn_command_vkCreateSemaphore,
    vn_command_vkCreateShaderModule, vn_command_vkDestroyBuffer, vn_command_vkDestroyCommandPool,
    vn_command_vkDestroyDescriptorPool, vn_command_vkDestroyDescriptorSetLayout,
    vn_command_vkDestroyDevice, vn_command_vkDestroyFence, vn_command_vkDestroyFramebuffer,
    vn_command_vkDestroyImage, vn_command_vkDestroyImageView, vn_command_vkDestroyInstance,
    vn_command_vkDestroyPipeline, vn_command_vkDestroyPipelineCache,
    vn_command_vkDestroyPipelineLayout, vn_command_vkDestroyQueryPool,
    vn_command_vkDestroyRenderPass, vn_command_vkDestroyRingMESA, vn_command_vkDestroySampler,
    vn_command_vkDestroySamplerYcbcrConversion, vn_command_vkDestroySemaphore,
    vn_command_vkDestroyShaderModule, vn_command_vkDeviceWaitIdle, vn_command_vkEndCommandBuffer,
    vn_command_vkEnumerateDeviceExtensionProperties,
    vn_command_vkEnumerateInstanceExtensionProperties, vn_command_vkEnumerateInstanceVersion,
    vn_command_vkEnumeratePhysicalDeviceGroups, vn_command_vkEnumeratePhysicalDevices,
    vn_command_vkExecuteCommandStreamsMESA, vn_command_vkFlushMappedMemoryRanges,
    vn_command_vkFreeCommandBuffers, vn_command_vkFreeDescriptorSets, vn_command_vkFreeMemory,
    vn_command_vkGetBufferDeviceAddress, vn_command_vkGetBufferMemoryRequirements,
    vn_command_vkGetBufferMemoryRequirements2, vn_command_vkGetBufferOpaqueCaptureAddress,
    vn_command_vkGetDescriptorSetLayoutSupport, vn_command_vkGetDeviceBufferMemoryRequirements,
    vn_command_vkGetDeviceGroupPeerMemoryFeatures, vn_command_vkGetDeviceImageMemoryRequirements,
    vn_command_vkGetDeviceImageSparseMemoryRequirements,
    vn_command_vkGetDeviceImageSubresourceLayout, vn_command_vkGetDeviceMemoryCommitment,
    vn_command_vkGetDeviceMemoryOpaqueCaptureAddress, vn_command_vkGetDeviceQueue2,
    vn_command_vkGetEventStatus, vn_command_vkGetFenceStatus,
    vn_command_vkGetImageDrmFormatModifierPropertiesEXT, vn_command_vkGetImageMemoryRequirements,
    vn_command_vkGetImageMemoryRequirements2, vn_command_vkGetImageSparseMemoryRequirements,
    vn_command_vkGetImageSparseMemoryRequirements2, vn_command_vkGetImageSubresourceLayout,
    vn_command_vkGetImageSubresourceLayout2, vn_command_vkGetMemoryResourcePropertiesMESA,
    vn_command_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR,
    vn_command_vkGetPhysicalDeviceExternalBufferProperties,
    vn_command_vkGetPhysicalDeviceExternalFenceProperties,
    vn_command_vkGetPhysicalDeviceExternalSemaphoreProperties,
    vn_command_vkGetPhysicalDeviceFeatures, vn_command_vkGetPhysicalDeviceFeatures2,
    vn_command_vkGetPhysicalDeviceFormatProperties,
    vn_command_vkGetPhysicalDeviceFormatProperties2,
    vn_command_vkGetPhysicalDeviceImageFormatProperties,
    vn_command_vkGetPhysicalDeviceImageFormatProperties2,
    vn_command_vkGetPhysicalDeviceMemoryProperties,
    vn_command_vkGetPhysicalDeviceMemoryProperties2,
    vn_command_vkGetPhysicalDeviceMultisamplePropertiesEXT,
    vn_command_vkGetPhysicalDeviceProperties, vn_command_vkGetPhysicalDeviceProperties2,
    vn_command_vkGetPhysicalDeviceQueueFamilyProperties,
    vn_command_vkGetPhysicalDeviceQueueFamilyProperties2,
    vn_command_vkGetPhysicalDeviceSparseImageFormatProperties,
    vn_command_vkGetPhysicalDeviceSparseImageFormatProperties2,
    vn_command_vkGetPhysicalDeviceToolProperties, vn_command_vkGetPipelineCacheData,
    vn_command_vkGetQueryPoolResults, vn_command_vkGetRenderAreaGranularity,
    vn_command_vkGetRenderingAreaGranularity, vn_command_vkGetSemaphoreCounterValue,
    vn_command_vkImportSemaphoreResourceMESA, vn_command_vkInvalidateMappedMemoryRanges,
    vn_command_vkMergePipelineCaches, vn_command_vkNotifyRingMESA, vn_command_vkQueueSubmit,
    vn_command_vkQueueWaitIdle, vn_command_vkResetCommandBuffer, vn_command_vkResetCommandPool,
    vn_command_vkResetDescriptorPool, vn_command_vkResetEvent, vn_command_vkResetFences,
    vn_command_vkResetQueryPool, vn_command_vkSeekReplyCommandStreamMESA, vn_command_vkSetEvent,
    vn_command_vkSetReplyCommandStreamMESA, vn_command_vkSignalSemaphore,
    vn_command_vkSubmitVirtqueueSeqnoMESA, vn_command_vkUpdateDescriptorSets,
    vn_command_vkWaitForFences, vn_command_vkWaitRingSeqnoMESA,
    vn_command_vkWaitSemaphoreResourceMESA, vn_command_vkWaitSemaphores,
    vn_command_vkWaitVirtqueueSeqnoMESA, vn_command_vkWriteRingExtraMESA,
};
use super::ring::{
    ReplyStream, ReplyStreamError, ResourceBytes, Ring, RingControl, RingError, ShmResources,
};
use super::ring_thread::{RingThread, RingWaiter, WaitRing, seqno_ge};
use crate::vulkan::Global;

/// `VK_COMMAND_GENERATE_REPLY_BIT_EXT`: the guest wants an answer to this command.
const GENERATE_REPLY: u32 = 0x1;

pub struct Context {
    pub id: ContextId,
    /// The hard poison. It outlives any one command and any one submission: once the stream cannot
    /// be trusted, nothing later in it can be either.
    /// Shared rather than owned: a ring's thread poisons the context it belongs to when its
    /// stream goes bad, and it must be able to do that without waiting for a lock -- the C's
    /// `vkr_context_on_ring_fatal` is a plain `vkr_context_set_fatal` for the same reason. One
    /// flag with many keys to it, never a copy per holder.
    fatal: Arc<AtomicBool>,
    objects: Shared,
    /// The driver objects this context has stood up. Per context, because a context owns its
    /// instance tree and shares nothing with another guest.
    driver: Driver,
    /// Replay mode. The journal's entries are fed straight to the dispatcher with their reply flag
    /// stripped, so rings are never started and no reply is ever encoded.
    replay: bool,
    /// Commands dispatched, and how many of those reached a handler this build does not have.
    /// The second number is what says how far a corpus actually got.
    pub dispatched: u64,
    pub unhandled: u64,
    /// Commands a replay dropped because their objects could not be rebuilt. A restore that ends
    /// with this above zero did not rebuild the world it was given.
    pub replay_ghosted: u64,
    /// The rings this context has stood up, by the object id the guest named them with.
    ///
    /// One map, so a ring has exactly one owner. Destroying the context drops it, which is what
    /// releases the share each ring holds of its resource's mapping.
    rings: BTreeMap<RingId, RingSlot>,
    /// Where answers to commands that arrived on the context's own stream go. Each ring holds its
    /// own; this is the one for everything that did not come in on a ring.
    reply: Option<ReplyStream>,
    /// Where a `vkWaitRingSeqnoMESA` on this context's own stream sleeps.
    ///
    /// One per context, shared with every ring thread it owns, because every input to that wait's
    /// predicate is produced on a ring thread -- which must never block on this context's lock to
    /// report one. The C's `ctx->wait_ring` is the same object for the same reason.
    wait_ring: Arc<WaitRing>,
    /// The thread stamping ALIVE into the status word of every ring that asked to be monitored.
    ///
    /// `None` until a ring asks. Once started it runs until the context goes, even if every
    /// monitored ring is destroyed -- see [`Monitor`].
    monitor: Option<Monitor>,
    /// What this context would have to be told again to be itself. Written by the dispatch loop as
    /// commands go by; read only by an export.
    journal: Journal,
    /// Entries a restore handed over and the fence has not yet released.
    ///
    /// A queue rather than an index beside a vector: the two would be a count and the thing it
    /// counts, and a feed that resumed from the wrong one would replay a command twice. Popping
    /// from the front makes "what is left" and "where we are" the same fact.
    restoring: VecDeque<journal::Parsed>,
}

/// The object table, answering the journal's one question about it.
///
/// A wrapper rather than an `impl` on `Shared` so the borrow is taken here, for the length of one
/// export, and never while a dispatch is running -- the table lives in a `RefCell` the handlers
/// hold across a command.
struct LiveObjects<'a>(&'a Shared);

impl journal::Live for LiveObjects<'_> {
    fn holds(&self, key: ObjectKey) -> bool {
        self.0.borrow().holds(key)
    }
}

/// A context must be `Send`: each one is owned by whoever is driving it, and a ring's thread will
/// take that ownership through a lock of its own. It is emphatically not `Sync` -- the object table
/// is a `RefCell` and the poison a `Cell`, so exclusive access is the only access, which is exactly
/// what a `Mutex` hands out.
///
/// A compile-time check rather than an argument: an `Rc` or a raw pointer appearing anywhere in the
/// driver, the object table or a device's state would break this, and the compiler names the field
/// that did it.
const _: () = {
    const fn is_send<T: Send>() {}
    is_send::<Context>();
};

/// What a suspended batch is waiting for.
///
/// The two are never interchangeable and never both reachable from one stream: a virtqueue wait is
/// legal only on a ring's own stream and a ring wait only on the context's, and the handlers
/// refuse the other way round. They share a type because they share a mechanism -- a batch that
/// stops partway through and is offered again -- not because a caller ever has to tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wait {
    /// `vkWaitVirtqueueSeqnoMESA`: the ring this batch arrived on must sleep until the context
    /// publishes this seqno for it. Strictly increasing and never wrapped -- it is a guest-side
    /// counter, not a position in anything.
    Virtqueue(u64),
    /// `vkWaitRingSeqnoMESA`: the caller must sleep until `ring`'s head reaches `seqno`. A byte
    /// position in that ring's buffer, so every comparison against it is wrap-aware.
    Ring { ring: RingId, seqno: u32 },
}

/// How a submission ended.
///
/// `Waiting` is the reason this is not a `bool`. A batch can stop partway through, and the caller
/// has to know both that it must wait and how much of its buffer has already run -- passing the
/// remainder back in is the resume. Losing either half loses commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "a suspended batch that is not resumed loses every command after the wait"]
pub enum Submitted {
    /// The whole batch ran.
    Done,
    /// The stream is poisoned, by this batch or an earlier one.
    Poisoned,
    /// The batch stopped. The first `consumed` bytes ran and will not run again; the wait command
    /// itself was *not* consumed, so offering `buf[consumed..]` once `on` is satisfied re-decodes
    /// it and lets its handler answer for real.
    ///
    /// That the wait command re-runs is what keeps a reply-carrying wait honest: the answer is
    /// encoded on the pass that proceeds, which is after the wait completed -- the same order a
    /// handler that could block would have produced.
    Waiting { consumed: usize, on: Wait },
}

impl Submitted {
    /// Whether the batch ran to its end. A suspension is deliberately not "ran": the caller still
    /// owes it a resume, and a `bool` that said yes would be how the remainder gets dropped.
    pub fn ran(self) -> bool {
        matches!(self, Submitted::Done)
    }
}

/// A ring, in whichever of its two states it is in.
///
/// The states differ by who owns the ring body, and that is the whole point of making them a type:
/// while a ring is running there is no `Ring` here for anyone else to reach. It went into the
/// thread and only comes back out of [`RingThread::stop`]. Nothing has to remember not to touch a
/// running ring's buffer, because there is nothing to touch.
enum RingSlot {
    /// Created, not yet reading. Snapshot replay builds every ring this way and promotes them all
    /// at `replay_end`; a live create is promoted at the end of the batch that made it.
    Idle(Ring),
    /// Reading on its own thread, which owns the body and lends the reply slot to each dispatch.
    Running(RingThread),
}

impl RingSlot {
    /// The ring's control words, whichever state it is in.
    ///
    /// The one thing that can be asked of a ring without knowing who owns its body -- which is
    /// exactly why those words live in an `Arc` of their own. See [`RingControl`].
    fn control(&self) -> &Arc<RingControl> {
        match self {
            RingSlot::Idle(r) => &r.control,
            RingSlot::Running(t) => t.control(),
        }
    }
}

#[cfg(test)]
impl RingSlot {
    /// The body of a ring that has not been started. Tests only, and deliberately panicking on a
    /// running ring: there is no body here to look at once the thread owns it.
    fn idle(&self) -> &Ring {
        match self {
            RingSlot::Idle(r) => r,
            RingSlot::Running(_) => panic!("a running ring's body belongs to its thread"),
        }
    }
}

impl Context {
    pub fn new(id: ContextId, budget: &Arc<Budget>) -> Context {
        Context {
            id,
            fatal: Arc::new(AtomicBool::new(false)),
            objects: Shared::new(),
            driver: Driver::new(Account::open(budget, id)),
            replay: false,
            dispatched: 0,
            unhandled: 0,
            replay_ghosted: 0,
            rings: BTreeMap::new(),
            reply: None,
            wait_ring: Arc::new(WaitRing::default()),
            monitor: None,
            journal: Journal::new(),
            restoring: VecDeque::new(),
        }
    }

    pub fn fatal(&self) -> bool {
        self.fatal.load(Ordering::Acquire)
    }

    /// A share of the poison flag, for a ring thread that must be able to set it.
    ///
    /// One flag with many keys, never a copy per holder: a ring that dies takes its whole context
    /// with it, which is what the C's `vkr_context_on_ring_fatal` does. A thread can set it
    /// without waiting for anything -- which it must, having no lock it is allowed to block on.
    pub fn fatal_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.fatal)
    }

    pub fn objects(&self) -> &Shared {
        &self.objects
    }

    /// A `vkWaitRingSeqnoMESA` on `ring`, in a form the caller can wait on with nothing locked.
    ///
    /// `None` when there is no running ring under that id. An idle one is deliberately not
    /// enough: nothing advances an idle ring's head, so a wait on one would be a wait on a
    /// number that cannot arrive, and returning a waiter for it would hand the caller a hang
    /// instead of the refusal it is owed.
    pub fn ring_waiter(&self, ring: RingId, seqno: u32) -> Option<RingWaiter> {
        match self.rings.get(&ring)? {
            RingSlot::Running(t) => {
                Some(t.waiter(self.id, seqno, self.wait_ring(), self.fatal_flag()))
            }
            RingSlot::Idle(_) => None,
        }
    }

    /// A share of the place a ring-seqno wait sleeps, for a ring thread that has to wake it.
    ///
    /// One object with many keys, like the poison flag beside it: a ring thread reports a head
    /// advance, a block on a virtqueue seqno, or its own death into this without ever waiting for
    /// the context lock -- which it must be able to do, having no lock it may block on at all.
    pub fn wait_ring(&self) -> Arc<WaitRing> {
        Arc::clone(&self.wait_ring)
    }

    /// Enter replay mode: the journal is about to be fed in, so nothing may answer it.
    pub fn replay_begin(&mut self) {
        self.replay = true;
    }

    /// Leave replay mode. The rings the journal built are started by the caller straight after,
    /// which is the C's loop at the end of `vkr_renderer_replay_end`.
    pub fn replay_end(&mut self) {
        self.replay = false;
    }

    /// Whether this context is being rebuilt from a journal rather than driven by a guest.
    ///
    /// Read by the promotion path: a replayed ring must not start reading while the journal is
    /// still being fed to it. This is the C's `ctx->replaying`, consulted at the same decision.
    pub fn replaying(&self) -> bool {
        self.replay
    }

    /// Everything this context would have to be told again, as bytes for the VMM to store.
    ///
    /// `None` when there is nothing to say, which is not the same as an empty blob: a caller that
    /// stored zero bytes and later restored them would have rebuilt nothing and been told it
    /// succeeded.
    pub fn journal_export(&self) -> Option<Vec<u8>> {
        self.journal.export(&LiveObjects(&self.objects))
    }

    /// How far this context's journal has been written, for the VMM's cross-layer fence.
    pub fn journal_seq(&self) -> Seq {
        self.journal.seq()
    }

    /// What the recorder dropped, by command type. A fact about the recorder, for the census.
    pub fn journal_transient(&self) -> Vec<(&'static str, u64)> {
        self.journal
            .transient()
            .iter()
            .map(|(c, n)| (vn_command_name(VkCommandTypeEXT(*c as i32)).unwrap_or("?"), *n))
            .collect()
    }

    /// Hand this context the journal it will be rebuilt from, and say how many entries it holds.
    ///
    /// Stored, not fed. Nothing replays until [`Context::replay_upto`] says how far, because what
    /// the entries name is created on the VMM's side as its own rebuild walks on -- the fence is
    /// the whole reason these are two calls.
    pub fn journal_restore(&mut self, bytes: &[u8]) -> Result<usize, &'static str> {
        let entries = journal::parse(bytes)?;
        let n = entries.len();
        self.restoring = entries.into();
        Ok(n)
    }

    /// Feed every restored entry up to and including `upto`, in the order they were recorded.
    ///
    /// Entries are consumed from the front, so a second call resumes where the first stopped and
    /// nothing is replayed twice. A poisoned context stops the feed: the remaining entries name a
    /// world we no longer built, and running them would be building on a lie.
    pub fn replay_upto(
        &mut self,
        upto: Seq,
        todo: &mut Unimplemented,
        global: &Global,
        resources: &dyn ShmResources,
    ) -> bool {
        let ghosted_before = self.replay_ghosted;
        while let Some(entry) = self.restoring.front() {
            if entry.seq > upto {
                break;
            }
            let entry = self.restoring.pop_front().expect("just looked at it");
            let ok = match entry.ring_key {
                0 => matches!(self.submit(&entry.wire, todo, global, resources), Submitted::Done),
                // A ring the journal itself created earlier in this same feed: the create was
                // routed to the context's decoder precisely so it would exist by now.
                key => self.submit_ring(RingId(key), &entry.wire, todo, global, resources),
            };
            if !ok {
                eprintln!(
                    "[virglrs] ctx {}: journal entry {} did not replay; {} entries abandoned",
                    self.id.get(),
                    entry.seq,
                    self.restoring.len()
                );
                self.restoring.clear();
                return false;
            }
        }
        // Every entry was accepted, which is not the same as every entry having worked. A command
        // skipped because its object could not be rebuilt leaves a hole nothing above can see, and
        // reporting success here is how a black screen after a resume becomes a mystery.
        let lost = self.replay_ghosted - ghosted_before;
        if lost > 0 {
            eprintln!(
                "[virglrs] ctx {}: {lost} replayed commands named objects the rebuild could not \
                 produce; the restored context is not the one that was snapshotted",
                self.id.get()
            );
            return false;
        }
        true
    }

    /// Drain one submission, dispatching every command in it.
    ///
    /// Returns false when the context was poisoned -- by this batch or by an earlier one. The C
    /// bails early on an already-fatal context for the same reason: a stream we stopped trusting
    /// does not become trustworthy because the guest sent more of it.
    pub fn submit(
        &mut self,
        buf: &[u8],
        todo: &mut Unimplemented,
        global: &Global,
        resources: &dyn ShmResources,
    ) -> Submitted {
        // The context lends its own reply slot for the length of the batch. Taking it out and
        // putting it back is what keeps one owner: nothing else can reach it while it is lent.
        let mut reply = self.reply.take();
        let out = self.submit_on(None, &mut reply, buf, todo, global, resources);
        self.reply = reply;
        out
    }

    /// Drain one submission that arrived on `on` -- a ring, or the context's own stream when
    /// `None`. Which one it was is not bookkeeping: it decides which commands are legal at all.
    ///
    /// `reply` is the slot answers go into, lent by whoever owns the stream: the context lends its
    /// own, a replayed ring lends the one in its entry, and a running ring's thread will lend the
    /// one in the body it owns. Lending rather than looking it up is what lets the same loop serve
    /// all three without knowing where the stream lives -- and it means there is no lookup here
    /// that a destroyed ring could make wrong.
    fn submit_on(
        &mut self,
        on: Option<RingId>,
        reply: &mut Option<ReplyStream>,
        buf: &[u8],
        todo: &mut Unimplemented,
        global: &Global,
        resources: &dyn ShmResources,
    ) -> Submitted {
        if self.fatal.load(Ordering::Acquire) {
            return Submitted::Poisoned;
        }

        // Read out what the poison path needs before the handlers borrow the rest of the
        // context: they hold the driver mutably for as long as the loop runs.
        let id = self.id;
        let replay = self.replay;
        let fatal = &self.fatal;
        let mut counts = Counts::default();
        let mut h = Handlers {
            objects: &self.objects,
            todo,
            driver: &mut self.driver,
            global,
            ctx: id,
            reject: None,
            resources,
            rings: &mut self.rings,
            monitor: &mut self.monitor,
            wait: None,
            execute: None,
            current_ring: on,
            reply,
            replaying: replay,
            note: None,
            journal: &mut self.journal,
        };

        let suspended = run_batch(&mut h, buf, fatal, &mut counts, 0);

        self.dispatched += counts.dispatched;
        self.unhandled += counts.unhandled;
        self.replay_ghosted += counts.replay_ghosted;
        // Every exit from the loop is one place, so a branch that poisons and breaks cannot report
        // success on the way out. The poison check comes first: a batch that suspended *and* then
        // poisoned has nothing left to resume into.
        if self.fatal.load(Ordering::Acquire) {
            return Submitted::Poisoned;
        }
        match suspended {
            Some((consumed, on)) => Submitted::Waiting { consumed, on },
            None => Submitted::Done,
        }
    }

    /// A submission that arrived on one ring's stream.
    ///
    /// A ring nobody created is refused without poisoning the context. That is the C's behaviour
    /// and it is the right one: the caller named a stream that is not here, which says nothing
    /// about the streams that are -- and a guest that destroys a ring while a submission for it is
    /// still in flight would otherwise take down every other ring it owns.
    pub fn submit_ring(
        &mut self,
        ring: RingId,
        buf: &[u8],
        todo: &mut Unimplemented,
        global: &Global,
        resources: &dyn ShmResources,
    ) -> bool {
        match self.rings.get(&ring) {
            None => {
                eprintln!(
                    "[virglrs] ctx {}: submission for {ring}, which is not a ring here",
                    self.id.get()
                );
                return false;
            }
            // The journal is replayed strictly before anything is promoted, so a running ring
            // here would mean the caller fed a journal entry to a ring a guest is already
            // driving. The C asserts the same thing at `vkr_renderer_replay_ring_cmd`.
            Some(RingSlot::Running(_)) => {
                panic!("ctx {}: {ring} was replayed into after it started running", self.id.get())
            }
            Some(RingSlot::Idle(_)) => {}
        }
        // The ring lends its slot for the batch, exactly as the context lends its own above.
        // A running ring's thread owns the body and lends from there instead -- see `dispatch_ring`.
        let mut reply = match self.rings.get_mut(&ring) {
            Some(RingSlot::Idle(r)) => r.reply.take(),
            _ => None,
        };
        let out = self.submit_on(Some(ring), &mut reply, buf, todo, global, resources);
        if let Some(RingSlot::Idle(r)) = self.rings.get_mut(&ring) {
            r.reply = reply;
        }
        // A journal is a record of commands that already ran on a live guest, replayed into a
        // context whose rings have no threads yet. Nothing here can satisfy a wait and nothing is
        // waiting for one, so a journal carrying one is a journal that does not describe a run
        // this renderer can rebuild -- which is a fact about the journal, said once, here.
        match out {
            Submitted::Done => true,
            Submitted::Poisoned => false,
            Submitted::Waiting { .. } => {
                eprintln!(
                    "[virglrs] ctx {}: {ring} replayed a wait, which replay cannot serve",
                    self.id.get()
                );
                self.fatal.store(true, Ordering::Release);
                false
            }
        }
    }

    /// Run a batch a ring's own thread read, answering into the slot that thread owns.
    ///
    /// The counterpart of [`Context::submit_ring`] for a running ring: same loop, same context,
    /// but the reply stream is lent by the caller rather than found here -- because the caller is
    /// the thread that owns the ring body, and no entry in `rings` holds it while it runs.
    pub fn dispatch_ring(
        &mut self,
        ring: RingId,
        reply: &mut Option<ReplyStream>,
        buf: &[u8],
        todo: &mut Unimplemented,
        global: &Global,
        resources: &dyn ShmResources,
    ) -> Submitted {
        self.submit_on(Some(ring), reply, buf, todo, global, resources)
    }

    /// Start every ring that is not running yet, and say how many that was.
    ///
    /// One promotion point for both paths the C has: a live create, promoted at the end of the
    /// batch that made it, and a replayed one, promoted at `replay_end`. The C spells these as
    /// `if (!ctx->replaying) vkr_ring_start(ring)` in the handler plus a loop in
    /// `vkr_renderer_replay_end`; collapsing them into one function is what keeps the handler from
    /// needing anything the renderer root owns.
    ///
    /// `spawn` is a closure because a ring thread needs a claim on this context, and a context
    /// cannot hand out a claim on itself -- only the owner of the `Arc` can. See `Vkr::promote`.
    pub fn start_idle_rings(&mut self, mut spawn: impl FnMut(RingId, Ring) -> RingThread) -> usize {
        // The common case by far: every batch a guest sends comes through here, and almost none
        // of them create a ring. Answering that without allocating keeps promotion off the
        // submission path's conscience.
        if !self.rings.values().any(|slot| matches!(slot, RingSlot::Idle(_))) {
            return 0;
        }
        let idle: Vec<RingId> = self
            .rings
            .iter()
            .filter(|(_, slot)| matches!(slot, RingSlot::Idle(_)))
            .map(|(id, _)| *id)
            .collect();
        for id in &idle {
            let Some(RingSlot::Idle(ring)) = self.rings.remove(id) else {
                unreachable!("just filtered for idle rings, and nothing else runs meanwhile")
            };
            self.rings.insert(*id, RingSlot::Running(spawn(*id, ring)));
        }
        idle.len()
    }

    /// Stop every running ring and join its thread.
    ///
    /// Called before a context is dropped, so that the drop always runs on the caller's thread.
    /// A ring thread holds a weak claim on this context and upgrades it for the length of one
    /// dispatch; if a thread were still running when the last strong reference went, that upgrade
    /// could make the thread itself the last owner -- and the drop would join the thread it was
    /// running on. Joining here, while no dispatch can be in flight afterwards, removes that.
    pub fn stop_rings(&mut self) {
        for (_, slot) in std::mem::take(&mut self.rings) {
            if let RingSlot::Running(t) = slot {
                drop(t.stop());
            }
        }
    }

    /// The driver state, for the teardown that has to destroy what it holds.
    pub fn driver_mut(&mut self) -> &mut Driver {
        &mut self.driver
    }

    /// The driver state, for the census that has to read what it holds.
    pub fn driver(&self) -> &Driver {
        &self.driver
    }

    /// Copy one allocation's contents out, returning how many bytes landed in `buf`.
    ///
    /// Here rather than on [`Driver`] because it takes both halves and only this owns both: the
    /// table says what handle the id names and which device it lives on, and the driver knows how
    /// big the guest asked for it to be. Neither keeps a copy of the other's answer.
    pub fn memory_read(&self, id: ObjectId, buf: &mut [u8]) -> Result<usize, MemoryError> {
        let objects = self.objects.borrow();
        let handle = objects
            .get(id)
            .filter(|o| o.ty == VkObjectType::VK_OBJECT_TYPE_DEVICE_MEMORY)
            .map(|o| VkDeviceMemory::from_host(o.handle))
            .ok_or(MemoryError::NoSuchAllocation)?;
        let device = objects.device_of(id).ok_or(MemoryError::NoSuchAllocation)?;
        self.driver.memory_read(device, handle, id, buf)
    }

    /// Export one allocation as a blob, handing back the host address the VMM will publish.
    ///
    /// Here for the reason [`Self::memory_read`] gives: the table owns the handle and the device,
    /// the driver owns the size and the mapping, and neither holds a copy of the other's answer.
    pub fn memory_export(
        &mut self,
        id: ObjectId,
        blob_size: u64,
    ) -> Result<(Exported, Option<driver::Storage>, ObjectKey), ExportError> {
        let (handle, device, key) = {
            let objects = self.objects.borrow();
            let handle = objects
                .get(id)
                .filter(|o| o.ty == VkObjectType::VK_OBJECT_TYPE_DEVICE_MEMORY)
                .map(|o| VkDeviceMemory::from_host(o.handle))
                .ok_or(ExportError::NoSuchAllocation)?;
            (
                handle,
                objects.device_of(id).ok_or(ExportError::NoSuchAllocation)?,
                objects.key_of(id).ok_or(ExportError::NoSuchAllocation)?,
            )
        };
        let (exported, share) = self.driver.memory_export(device, handle, id, blob_size)?;
        Ok((exported, share, key))
    }

    /// Whether the object a key was taken for still stands. What a resource that borrowed an
    /// allocation's mapping asks before handing the address on: the id may name a fresh object
    /// by now, and the key will not.
    pub fn holds(&self, key: ObjectKey) -> bool {
        self.objects.borrow().holds(key)
    }
}

/// Everything this context stood up on the host goes when the context does.
///
/// A teardown a caller had to remember to call was a teardown three of its four paths skipped: the
/// guest's own context destroy called it, and a duplicate context id replacing a live context, a
/// renderer dropped with contexts still in it, and a panic on the way up did not -- each leaking
/// an instance, its devices and every handle under them. None of those paths is going to grow a
/// call; the drop is the one thing all four already do.
///
/// The table is emptied first, and it is what says which device each object hangs off. The driver
/// is about to destroy the devices, and afterwards there is nothing left to destroy anything on.
impl Drop for Context {
    fn drop(&mut self) {
        // The rings go with it. An orderly teardown has already called `stop_rings`, so this is
        // usually empty; anything still here is detached rather than joined, because a join in a
        // drop can be a thread joining itself. See `impl Drop for RingThread`.
        self.rings.clear();
        let doomed = self.objects.borrow_mut().take_all();
        self.driver.teardown(&doomed);
    }
}

/// What a ring's create info asked of the monitor: nothing, or a reporting period.
///
/// Three answers in two layers, because they are three different things: no chained monitor info
/// at all (`None` -- this ring is not monitored), a monitor info asking for zero (`Some(None)` --
/// a guest error), and a real period (`Some(Some(us))`). Collapsing the first two would turn a
/// malformed request into a silently unmonitored ring, and the guest would abort itself seconds
/// later with nothing said about why.
fn monitor_period(info: &VkRingCreateInfoMESA) -> Option<Option<u32>> {
    let m: &VkRingMonitorInfoMESA = driver::chained(&info.pNext)?;
    Some(Some(m.maxReportingPeriodMicroseconds).filter(|&us| us != 0))
}

/// Poison a context, naming the command that did it -- once.
///
/// A ring the guest can no longer use looks the same from inside the guest whatever caused it, so
/// the command type is the only thing that tells a bug report from a hostile stream apart.
///
/// Free-standing rather than a method because the dispatch loop has already lent the rest of the
/// context to the handlers by the time it needs this.
/// How many commands a batch dispatched, and how many reached no handler.
///
/// One counter travelling through the nested dispatch, so an executed stream's commands land in
/// the same census as the batch that asked for them.
#[derive(Default)]
struct Counts {
    dispatched: u64,
    unhandled: u64,
    /// Commands skipped during a *replay* because they named an object the host refused.
    ///
    /// Counted apart from `unhandled` because it means something entirely different. From a guest
    /// a ghost is containment working as designed -- one command lost instead of a poisoned ring,
    /// because the guest had already pipelined it behind a create the host refused. During a
    /// replay there is no guest and nothing was pipelined: the create that failed is one *we*
    /// replayed, so this is the rebuild failing to rebuild.
    replay_ghosted: u64,
}

/// Command streams a handler asked to have executed, and where each one's answers go.
///
/// Descriptors, not bytes. `streamCount` is bounded only by what fits in the batch, and each
/// descriptor may name a whole resource, so copying them all out up front would let a guest ask
/// the host to materialise far more than it ever mapped. The bytes are copied one stream at a
/// time, in `run_streams`, and peak cost is the largest single stream -- memory the guest has
/// already paid for.
struct Execute {
    streams: Vec<VkCommandStreamDescriptionMESA>,
    reply_positions: Option<Vec<usize>>,
}

/// Dispatch every command in one stream of wire bytes.
///
/// Returns where the batch stopped when a handler asked to be suspended, and `None` otherwise --
/// including when it was poisoned, which the caller reads from `fatal`.
///
/// `depth` is 0 for a submission and 1 for the streams a `vkExecuteCommandStreamsMESA` names.
/// Where the C saves and restores its one decoder's state around the nested run, this needs
/// nothing: each level builds its own decoder, arena and reply scratch, so the outer decode is
/// untouched by construction and cannot be left half-restored.
fn run_batch(
    h: &mut Handlers<'_>,
    buf: &[u8],
    fatal: &AtomicBool,
    counts: &mut Counts,
    depth: u32,
) -> Option<(usize, Wait)> {
    let id = h.ctx;
    let replay = h.replaying;
    let objects = h.objects;

    let temp = Bump::new();
    // One reply buffer for the batch, refilled per command. Host memory: an answer is built
    // here in full and only then offered to the guest, so a reply that turns out not to fit
    // never reaches it. See `ReplyStream::write`.
    let mut scratch: Vec<u8> = Vec::new();
    let proto = AllOfIt;
    let mut dec = Decoder::new(buf, &temp, objects, fatal);

    // Where this batch stopped, when it stopped at a wait. The position *before* the command
    // that asked, so the resume re-decodes it -- see `Submitted::Waiting`.
    let mut suspended = None;

    while dec.has_command() {
        let at = dec.pos();
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let flags = dec.decode_scalar::<VkFlags>();
        if dec.hard_fatal() {
            // The header itself was short: there is no command here to lose.
            eprintln!(
                "[virglrs] ctx {}: submission ends mid-header, {} bytes into {}",
                id.get(),
                dec.pos(),
                buf.len()
            );
            break;
        }

        // The guest is waiting for an answer to this one. Replay never is: the journal's
        // entries have had their reply flag stripped already, which is why a replayed stream
        // needs no reply buffer at all.
        let wants_reply = flags.0 & GENERATE_REPLY != 0 && !replay;

        // Asked before the command runs rather than after, which is where the C's
        // `vkr_cs_encoder_acquire` asks it too. Running the handler first would let a command
        // create an object and only then discover there is nowhere to report it -- state
        // changed on a path that ends in a poisoned context either way.
        if wants_reply && h.reply.is_none() {
            counts.unhandled += 1;
            poison(id, &dec, cmd, "wants a reply, and no reply stream was ever set");
            break;
        }

        let mut enc = Encoder::growing(&mut scratch, &proto);
        let verdict = vn_dispatch_command(&mut dec, wants_reply.then_some(&mut enc), cmd, &mut *h);
        // How much answer there is. Read here so the encoder's borrow of the scratch ends
        // before the commit below reads it back.
        let answer = enc.pos();
        counts.dispatched += 1;

        // The recorder's raw materials, drained before any branch below can leave the loop. A
        // command that ghosted or was rejected still resolved ids and may still have added
        // objects; leaving either behind would hand them to whichever command is recorded next,
        // which would then claim to have created or named something it never mentioned.
        let named = dec.take_resolved();
        let made = objects.borrow_mut().take_added();
        let note = h.note.take();

        match verdict {
            Dispatched::Served => {}
            // A malformed argument, or a shape the generator has no decoder for. Either way
            // the command is what a reader needs, because without it a gap reaches a user as
            // a hung guest.
            Dispatched::Undecodable => {
                poison(id, &dec, cmd, "did not decode");
                break;
            }
            // The wrapper skipped it as containment asks -- and containment is for commands
            // the guest is not waiting on. When it is waiting, a reply the host never wrote is
            // whatever the slot held before, shaped exactly like success, so the batch stops
            // instead.
            Dispatched::Ghosted(ghost) => {
                if wants_reply {
                    poison(
                        id,
                        &dec,
                        cmd,
                        &format!("wanted a reply, and names object {} the host refused", ghost.0),
                    );
                    break;
                }
                // Silent from a guest, never from a replay. A restore with holes in it looks
                // exactly like a clean one from the outside -- the commands were accepted, the
                // context is alive, and what is missing is only visible to the guest that will
                // shortly use it. Name the command and let the caller fail.
                if replay {
                    counts.replay_ghosted += 1;
                    eprintln!(
                        "[virglrs] ctx {}: replaying {} names object {}, whose create this \
                         replay could not rebuild -- the restored world is missing it",
                        id.get(),
                        vn_command_name(cmd).unwrap_or("an unnamed command"),
                        ghost.0
                    );
                }
                continue;
            }
            Dispatched::Undefined => {
                poison(id, &dec, cmd, "is not a command type this protocol defines");
                break;
            }
        }

        // The handler's own verdict: the command itself was unusable -- an id the guest cannot
        // have, a length that would send the driver off the end of what was decoded -- or no
        // handler exists for it. The handler has no decoder to say so with; this is where it
        // lands.
        if let Some(why) = h.reject.take() {
            poison(id, &dec, cmd, why);
            break;
        }

        // Poisoned from outside while the command ran -- a ring thread stopping this context.
        // The command itself was served, and its answer must not go over: a reply from a
        // context that has stopped is one the guest would act on.
        if fatal.load(Ordering::Acquire) {
            poison(id, &dec, cmd, "ran while the context was being poisoned elsewhere");
            break;
        }

        // The handler could not proceed and asked to be tried again later. Nothing is
        // committed: not the position, so the command decodes again from the same byte, and
        // not the answer, which would otherwise report a wait as finished before it was. The
        // command is counted twice in `dispatched` for the same reason -- a cosmetic cost of
        // the resume being a real re-dispatch rather than a resumption of one.
        if let Some(on) = h.wait.take() {
            // Not from inside an execute. A suspension unwinds to `ffi.rs`, which resumes the
            // *outer* batch from the position it was handed -- and that position names a byte
            // in the outer stream, not in the copied one this command came from. The C can
            // block here because it blocks in the handler; we cannot, so this is a deviation
            // and is logged as one. Mesa's execute streams carry recorded `vkCmd*` work and no
            // transport waits, which is why nothing real is expected to reach this line.
            if depth > 0 {
                poison(
                    id,
                    &dec,
                    cmd,
                    "suspends the batch, and a command stream being executed has nowhere to suspend to",
                );
                break;
            }
            suspended = Some((at, on));
            break;
        }

        // A handler asking for streams to be executed. Same message shape as `wait`, for the
        // same reason: the nested dispatch needs a decoder, and a handler has none.
        if let Some(exec) = h.execute.take() {
            if depth > 0 {
                poison(
                    id,
                    &dec,
                    cmd,
                    "executes command streams from inside a command stream it is already executing",
                );
                break;
            }
            run_streams(h, &exec, fatal, counts, depth);
            if fatal.load(Ordering::Acquire) {
                break;
            }
        }

        // The answer goes over only once the command is known to have worked. A poisoned
        // context has nothing to say, and a rejected command's half-built reply would be an
        // answer to a question we did not finish -- which the guest cannot tell apart from a
        // real one.
        if answer > 0 {
            let stream = h.reply.as_mut().expect("a reply had a stream before the command ran");
            if let Err(over) = stream.write(&scratch[..answer]) {
                poison(id, &dec, cmd, &format!("could not be answered: {over}"));
                break;
            }
        }

        // The command worked, in full, and nothing after this can undo it -- which is the only
        // point at which it is safe to record. Every branch above either breaks (the command is
        // not part of any world worth rebuilding) or, for a wait, leaves the position uncommitted
        // so the command is dispatched again; recording earlier would journal a command that never
        // happened, or the same command twice.
        record(h, &buf[at..dec.pos()], cmd, named, made, note);
    }
    suspended
}

/// Keep, or deliberately drop, one command that has just been served.
///
/// The recorder is not gated on `replaying`: a replayed command has to re-record, or a restored
/// context could never be snapshotted again -- and the harness's `--rebuild` gate is exactly the
/// claim that exporting, restoring and exporting again yields the same journal.
fn record(
    h: &mut Handlers<'_>,
    wire: &[u8],
    cmd: VkCommandTypeEXT,
    named: Vec<ObjectId>,
    made: Vec<ObjectKey>,
    note: Option<Note>,
) {
    // The wire spells a command type unsigned and the generated enum signs it. Reconciled once,
    // here, rather than at each call below: the journal's format is the one that leaves this
    // function, so this is the boundary that knows which reading is the truth.
    let cmd_type = cmd.0 as u32;

    // The envelope, not its contents. `vkExecuteCommandStreamsMESA` names streams whose commands
    // the nested `run_batch` records one by one as it dispatches them; keeping the envelope as
    // well would replay every one of them twice. This is the one command whose own bytes are
    // worth nothing because its effects are recorded elsewhere.
    if cmd == VkCommandTypeEXT::VK_COMMAND_TYPE_vkExecuteCommandStreamsMESA_EXT {
        h.journal.skip(cmd_type);
        return;
    }

    // A ring's own commands carry the reply flag the guest set, and a replay has no reply stream
    // to answer into. Stripping it here rather than at replay keeps the stored bytes honest about
    // what will be fed back: the loop's `wants_reply` reads a flag the journal has already
    // cleared, which is why a replayed stream needs no reply buffer at all.
    let wire = strip_reply_flag(wire);

    let key_of = |ids: &[ObjectId]| -> Vec<ObjectKey> {
        let t = h.objects.borrow();
        ids.iter().filter_map(|id| t.key_of(*id)).collect()
    };

    match note {
        Some(Note::RingCreated(ring)) => {
            h.journal.ring_created(cmd_type, &wire, ring);
            return;
        }
        Some(Note::RingGone(ring)) => {
            // The destroy is not kept: a ring that is gone has nothing to rebuild, and every
            // entry that belonged to it goes with it.
            h.journal.ring_gone(ring);
            h.journal.skip(cmd_type);
            return;
        }
        Some(Note::PoolReset(pool)) => {
            let key = {
                let t = h.objects.borrow();
                t.id_of_handle(VkObjectType::VK_OBJECT_TYPE_COMMAND_POOL, pool)
                    .and_then(|id| t.key_of(id))
            };
            if let Some(key) = key {
                h.journal.pool_reset(key);
            }
            // The reset itself rebuilds nothing: replaying the allocates that survive it produces
            // buffers already in the state a reset leaves them.
            h.journal.skip(cmd_type);
            return;
        }
        None => {}
    }

    // Everything a running ring's stream carries replays on that ring's decoder. A command that
    // arrived on the context's own stream replays there, which is `ring_key` 0.
    let route = h.current_ring.map_or(0, |r| r.0);

    // Where a ring's answers go: state a later command of the same kind replaces outright, rather
    // than adding to. Kept per ring and per command, so a set followed by a seek keeps both.
    if matches!(
        cmd,
        VkCommandTypeEXT::VK_COMMAND_TYPE_vkSetReplyCommandStreamMESA_EXT
            | VkCommandTypeEXT::VK_COMMAND_TYPE_vkSeekReplyCommandStreamMESA_EXT
    ) {
        h.journal.ring_latest(cmd_type, &wire, route);
        return;
    }

    // A create is not a list to keep in step with the generator: it is any command the object
    // table grew under, which the table itself just said. A command that made nothing is not one,
    // however it is spelled.
    if !made.is_empty() {
        h.journal.created(cmd_type, &wire, made, key_of(&named));
        return;
    }

    if let Some(class) = recording_class(cmd) {
        // The buffer is the first object the command named, which is where every one of these
        // spells it -- `vkCmd*` takes it as its first argument, and so do begin, end and reset.
        // Taken from what the decoder resolved rather than read off the wire again, so the
        // recorder and the dispatch cannot disagree about which buffer this was.
        let Some(&buffer) = named.first() else {
            h.journal.skip(cmd_type);
            return;
        };
        let Some(buffer) = h.objects.borrow().key_of(buffer) else {
            h.journal.skip(cmd_type);
            return;
        };
        let refs = key_of(named.get(1..).unwrap_or_default());
        h.journal.recorded(cmd_type, &wire, buffer, class == Recording::Resets, refs);
        return;
    }

    if mutates(cmd) {
        h.journal.mutated(cmd_type, &wire, key_of(&named), Vec::new());
        return;
    }

    // Everything else. Not a hole by assumption -- the census counts these by type, so what we
    // drop is a list someone can read rather than a number to be reassured by. A submission, a
    // wait, a query, a doorbell: state the guest reproduces itself, or transport that has already
    // happened.
    h.journal.skip(cmd_type);
}

/// Clear `VK_COMMAND_GENERATE_REPLY_BIT_EXT` from a command's stored bytes.
///
/// A replay has no reply stream, and the dispatch loop refuses a command that wants one where
/// there is none. Stripping at record time rather than at replay is what makes that refusal a real
/// check on the guest rather than something replay has to be excused from.
///
/// The flag is the second word of every command header, so a command whose bytes do not even reach
/// that far is not one we can keep -- and never one we recorded, since the loop breaks on a short
/// header before dispatching.
fn strip_reply_flag(wire: &[u8]) -> Vec<u8> {
    let mut out = wire.to_vec();
    if let Some(flags) = out.get_mut(4..8) {
        let cleared = u32::from_le_bytes(flags.try_into().expect("four bytes")) & !GENERATE_REPLY;
        flags.copy_from_slice(&cleared.to_le_bytes());
    }
    out
}

/// Whether a command is part of a command buffer's recording, and whether it discards what was
/// recorded before it.
#[derive(PartialEq, Eq)]
enum Recording {
    /// It adds to the recording.
    Adds,
    /// It starts the recording over: `vkBeginCommandBuffer` and `vkResetCommandBuffer`.
    Resets,
}

/// Classify by name rather than by a four-hundred-arm match.
///
/// Every command that records into a buffer is spelled `vkCmd*` -- that is a naming rule of the
/// Vulkan specification, not a coincidence of this generator, and it is why a new extension's
/// commands are recorded without this function being touched. The three that bracket a recording
/// are named individually because nothing in their spelling says so.
///
/// `vkFreeCommandBuffers` is deliberately absent: it destroys the buffers, which takes their keys
/// with them, and every recording naming one stops being true on its own.
fn recording_class(cmd: VkCommandTypeEXT) -> Option<Recording> {
    match cmd {
        VkCommandTypeEXT::VK_COMMAND_TYPE_vkBeginCommandBuffer_EXT
        | VkCommandTypeEXT::VK_COMMAND_TYPE_vkResetCommandBuffer_EXT => Some(Recording::Resets),
        VkCommandTypeEXT::VK_COMMAND_TYPE_vkEndCommandBuffer_EXT => Some(Recording::Adds),
        _ => vn_command_name(cmd).filter(|n| n.starts_with("vkCmd")).map(|_| Recording::Adds),
    }
}

/// Commands that write into objects they do not own, and so are true only while every object they
/// named is still there.
///
/// A short explicit list, unlike the recordings: there is no naming rule that picks these out, and
/// a wrong guess here is not a missing entry but a stale one -- an entry claiming to describe a
/// descriptor set that has been freed and reallocated.
fn mutates(cmd: VkCommandTypeEXT) -> bool {
    matches!(
        cmd,
        VkCommandTypeEXT::VK_COMMAND_TYPE_vkUpdateDescriptorSets_EXT
            | VkCommandTypeEXT::VK_COMMAND_TYPE_vkUpdateDescriptorSetWithTemplate_EXT
            | VkCommandTypeEXT::VK_COMMAND_TYPE_vkBindBufferMemory_EXT
            | VkCommandTypeEXT::VK_COMMAND_TYPE_vkBindBufferMemory2_EXT
            | VkCommandTypeEXT::VK_COMMAND_TYPE_vkBindImageMemory_EXT
            | VkCommandTypeEXT::VK_COMMAND_TYPE_vkBindImageMemory2_EXT
    )
}

/// Run the streams one `vkExecuteCommandStreamsMESA` named, seeking the reply stream per stream.
///
/// Poisons through `fatal` rather than returning a reason: every refusal here names a stream
/// index, which the one-line `poison` does not carry.
fn run_streams(
    h: &mut Handlers<'_>,
    exec: &Execute,
    fatal: &AtomicBool,
    counts: &mut Counts,
    depth: u32,
) {
    let id = h.ctx;
    for (i, s) in exec.streams.iter().enumerate() {
        // Before the empty-stream skip, exactly as in the C: a zero-sized stream is still a
        // position the guest asked its answers to resume from.
        if let Some(&pos) = exec.reply_positions.as_ref().map(|p| &p[i]) {
            let stream =
                h.reply.as_mut().expect("a reply position had a stream before the streams ran");
            if !stream.seek(pos) {
                eprintln!(
                    "[virglrs] ctx {}: vkExecuteCommandStreamsMESA: stream {i} asks its reply to \
                     resume at {pos}, which is outside the reply stream",
                    id.get(),
                );
                fatal.store(true, Ordering::Release);
                return;
            }
        }

        if s.size == 0 {
            continue;
        }

        let Some(map) = ResourceHandle::new(s.resourceId).and_then(|r| h.resources.shm(r)) else {
            eprintln!(
                "[virglrs] ctx {}: vkExecuteCommandStreamsMESA: stream {i} is in resource {}, \
                 which this context has no mapping for",
                id.get(),
                s.resourceId,
            );
            fatal.store(true, Ordering::Release);
            return;
        };

        // Bounds first, allocation second. The size is the guest's, and a size checked only
        // against nothing is a request for as much host memory as a u64 can name.
        let end = s.offset.checked_add(s.size);
        if end.is_none_or(|end| end > map.len()) {
            eprintln!(
                "[virglrs] ctx {}: vkExecuteCommandStreamsMESA: stream {i} asks for {} bytes at \
                 {} of resource {}, which is {} bytes",
                id.get(),
                s.size,
                s.offset,
                s.resourceId,
                map.len(),
            );
            fatal.store(true, Ordering::Release);
            return;
        }

        // Copied out rather than decoded in place. The mapping can be torn down under us -- the
        // resource is the guest's to detach -- and the `Arc` this holds is what keeps the pages
        // alive for exactly as long as the copy takes. Decoding straight from guest memory would
        // instead need that guarantee to hold for the whole nested dispatch, which runs handlers.
        let mut bytes = vec![0u8; s.size];
        assert!(
            map.copy_out(s.offset, &mut bytes),
            "a copy the bounds check above admitted did not fit",
        );

        let suspended = run_batch(h, &bytes, fatal, counts, depth + 1);
        assert!(suspended.is_none(), "a nested batch suspended, which its own depth guard refuses",);
        if fatal.load(Ordering::Acquire) {
            return;
        }
    }
}

/// Poison the context, saying which command and why. Every caller leaves the loop right after,
/// so this prints unconditionally: the flag may already be set -- the decoder sets the same one
/// when a command does not decode -- and that is the case whose reason is most worth reading.
fn poison(id: ContextId, dec: &Decoder<'_>, cmd: VkCommandTypeEXT, why: &str) {
    let name = vn_command_name(cmd)
        .map(str::to_string)
        .unwrap_or_else(|| format!("command type {}", cmd.0));
    eprintln!("[virglrs] ctx {id}: {name} {why}, {} bytes in", dec.pos());
    dec.set_fatal();
}

/// The commands a build does not serve yet, counted.
#[derive(Default)]
pub struct Unimplemented {
    pub seen: std::collections::BTreeMap<i32, u64>,
}

/// What a command reaches: the object table it registers into, and the tally of what this build
/// cannot do yet.
///
/// Every command method is left at its generated default, so each lands on `unsupported` and is
/// counted. What is *not* left to a default is the object bookkeeping: a create registers the id
/// the guest chose, a destroy forgets it, and that is enough for the whole corpus to decode --
/// every later command that names an object finds it.
///
/// **A command with no handler still registers its objects, and its handle is its own id.** That
/// is what lets the whole corpus decode while most of the protocol is unserved: every later command
/// that names the object finds it. A served command makes the handle real, and the two stop being
/// equal -- nothing depends on them being the same number, because the generator carries both
/// halves of the pairing separately. See `objects`.
pub struct Handlers<'a> {
    objects: &'a Shared,
    todo: &'a mut Unimplemented,
    /// The driver objects this context has stood up: its instance, and its devices.
    driver: &'a mut Driver,
    /// The entry points that exist before an instance does. Owned by the renderer root, because
    /// they are the same for every context.
    global: &'a Global,
    /// Why the handler refused the command, if it did. It cannot be reported from here -- the
    /// handler has no decoder -- so the loop reads it back and poisons with this as the reason.
    reject: Option<&'static str>,
    /// The renderer's resource table, for the commands that name guest memory.
    resources: &'a dyn ShmResources,
    /// Which context this is, for the resource questions whose answer is only meaningful within
    /// one -- a guest id names an allocation, and every context numbers its own.
    ctx: ContextId,
    /// The ring this batch arrived on, or `None` for the context's own stream. Several commands
    /// are legal on exactly one of the two, and a reply belongs to whichever it was.
    current_ring: Option<RingId>,
    /// Where this batch's answers go, lent by whoever owns the stream it arrived on.
    reply: &'a mut Option<ReplyStream>,
    /// The rings this context has stood up. Held mutably because creating one is a command.
    rings: &'a mut BTreeMap<RingId, RingSlot>,
    /// The context's ring monitor, started here by the first ring that asks for one.
    monitor: &'a mut Option<Monitor>,
    /// A handler asking to be suspended: it cannot proceed until something outside this context
    /// happens, and it must not sleep here.
    ///
    /// Read back and cleared by the loop, like `reject`. The reason it is a message rather than a
    /// blocking call is the lock: this loop runs with the context locked and, on the ABI path,
    /// with the renderer root locked behind that. A handler that slept would hold both, and the
    /// thread it is waiting for needs the first of them to make any progress at all.
    wait: Option<Wait>,
    /// A handler asking for command streams to be executed, for the same reason `wait` is a
    /// message: the nested dispatch needs a decoder, and a handler is handed none.
    execute: Option<Execute>,
    /// Whether this batch is a snapshot journal being replayed rather than a guest talking.
    ///
    /// A created ring reads it: replay restores head and status words the host would otherwise
    /// insist on owning, and resumes the read cursor from them.
    replaying: bool,
    /// What the recorder could not work out for itself, left by the handler that knows.
    ///
    /// A message, like `wait` and `execute`, and for a narrower version of the same reason: the
    /// loop can see what a command created (the object table counted it) and what it named (the
    /// decoder collected it), but a ring is not an object in that table and a pool reset names its
    /// pool among several handles the loop cannot tell apart. Rather than have the loop re-read
    /// arguments the handler already decoded -- a second opinion about a reconciled value -- the
    /// handler says.
    note: Option<Note>,
    /// Where a command that still describes live state is kept, so the context can be rebuilt.
    journal: &'a mut Journal,
}

/// What a handler tells the recorder about the command it just served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Note {
    /// It made this ring. Owned by the ring, and replayed on the context's own decoder, because
    /// when it replays the ring does not exist yet.
    RingCreated(u64),
    /// It destroyed this ring, and everything the ring owned goes with it.
    RingGone(u64),
    /// It reset this command pool, discarding every recording made from it without invalidating a
    /// single buffer -- so no key changes and only the journal can be told.
    ///
    /// The host handle, because that is what the decoded argument holds: a reset keeps no shadow
    /// of the guest's id, and picking the pool out of the ids the command resolved would mean
    /// knowing which argument position it sat in. The table answers by handle and type instead.
    PoolReset(HostHandle),
}

impl Handlers<'_> {
    /// The guest id a single out-handle carries, or None when the guest asked for no object.
    fn out_id<T: Handle>(&self, out: Option<&Guest<T>>) -> Option<ObjectId> {
        Some(out?.id())
    }

    /// Write what the driver produced into the shadow the generated hook will read, or ghost the
    /// id when it produced nothing.
    ///
    /// A guest pipelines: it sends a create and the commands using it without waiting for an
    /// answer, so those are already in flight when the create fails. A ghost turns each of them
    /// into one lost command instead of a poisoned ring -- see `objects::Slot::Ghost`. Leaving
    /// the shadow zero would instead register the id as its own handle, which is the unserved
    /// command's fiction and a lie for a served one.
    fn plant<T: Handle>(
        &mut self,
        what: &str,
        out: Option<&Guest<T>>,
        shadow: Option<&mut T>,
        host: Result<T, VkResult>,
    ) {
        let Some(id) = self.out_id(out) else {
            return;
        };
        match host {
            Ok(h) if h.host().0 != 0 => {
                if let Some(shadow) = shadow {
                    *shadow = h;
                }
            }
            Err(r) => {
                // The guest recovers from this on its own -- it sees the failure in the reply and
                // unwinds, exactly as it would on real hardware. What it cannot do is tell the
                // host operator why, and a driver that refuses a create is not something to pass
                // over in silence.
                eprintln!("[virglrs] {what} refused by the driver: VkResult {}", r.0);
                self.objects.borrow_mut().add_ghost(id);
            }
            Ok(_) => self.objects.borrow_mut().add_ghost(id),
        }
    }

    /// The verdict on an array the generated accessor could not reconcile.
    ///
    /// The reconciliation is the accessor's -- it is the only code that knows how long the arena
    /// allocation is. What is left is what a split pair *means*, and that is not the accessor's to
    /// decide: a count with no array behind it is a command that cannot be carried out, and
    /// refusing it is the only honest answer. Doing nothing and reporting success would leave the
    /// guest drawing through binds and writes that never happened.
    fn array<T>(&mut self, a: Option<T>) -> Option<T> {
        if a.is_none() {
            self.reject = Some("counted an array it did not send");
        }
        a
    }

    /// The verdict on a query whose answer has nowhere to go.
    ///
    /// A `vkGet*` is the guest handing over a struct and asking for it back filled. If it sent no
    /// struct there is no answer to give, and the only dishonest option is the quiet one: return,
    /// encode a reply built from a null, and let the guest read whatever it had there before as
    /// though the host had written it.
    fn fills<T>(&mut self, out: Option<T>) -> Option<T> {
        if out.is_none() {
            self.reject = Some("asked a query with no struct to answer into");
        }
        out
    }

    /// The verdict on a two-call enumeration that never asked how many there are.
    ///
    /// Vulkan's enumerations are asked twice -- once for the count, once with room for that many --
    /// and the count pointer is the only thing carried by both calls. Without it there is no first
    /// call to answer and no second call to bound: the guest has described neither question.
    ///
    /// A predicate rather than an `Option` like its four neighbours, because the count is read
    /// again *after* the array borrow ends and so cannot be held across it. The generator's
    /// `has_*` accessor is the only shape that survives that, and the wording belongs here rather
    /// than at the twelve handlers that ask.
    /// It owns the wording, not the control flow: each of the twelve still has to return on a
    /// `false`, and a sabotage that drops one `return` survives this. That is the residue of a
    /// predicate, and the reason `#[must_use]` is on it -- it closes the half a type can close.
    #[must_use]
    fn counted(&mut self, asked: bool) -> bool {
        if !asked {
            self.reject = Some("enumerated without asking for a count");
        }
        asked
    }

    /// The verdict on a query that did not say what it is asking about.
    ///
    /// The mirror of [`Vkr::fills`], and the same dishonesty from the other side. Every command
    /// routed through the info-carrying query helpers has its struct marked required in vk.xml,
    /// so an absent one is not a guest exercising an option -- it is a question with no subject.
    /// Answering it would mean inventing the subject, and the answer would go back looking like
    /// the host had been asked.
    ///
    /// The `Option` it takes is a fact about the wire, not about Vulkan: the decoder types every
    /// by-ref member this way because a guest can always send a null. Deciding what that null
    /// means is the handler's, and this is where the whole family decides it once.
    fn names<'w, I>(&mut self, info: Option<&'w I>) -> Option<&'w I> {
        if info.is_none() {
            self.reject = Some("asked a query without saying what it is about");
        }
        info
    }

    /// The verdict on a query this renderer could not put to the driver at all.
    ///
    /// Not the same as a query the driver answered badly, which is the driver's answer and goes
    /// back as it came. This is no instance, no such device, or an entry point this driver does
    /// not export -- and the struct is then still whatever the guest sent, so reporting anything
    /// other than a refusal would be reporting success for a call that never happened.
    fn asked<R>(&mut self, r: Result<R, VkResult>) -> Option<R> {
        if r.is_err() {
            self.reject = Some("asked a query this driver cannot answer");
        }
        r.ok()
    }

    /// The other honest reading of a split pair, for the arrays where it is the right one.
    ///
    /// Exactly the arrays vk.xml marks `noautovalidity` -- `vkFreeCommandBuffers`,
    /// `vkFreeDescriptorSets` -- where the decoder deliberately does not check the size, so the
    /// pair can genuinely arrive apart. Nothing else can be passed here: an array the decoder
    /// does check hands back a slice rather than an `Option`, because a split one poisons the
    /// stream and dispatch drops the command before a handler sees it. Freeing "three, list not supplied" identifies nothing to
    /// free, which is not the same as claiming work was done: there is no work to claim. Poisoning
    /// a ring over it would cost the guest everything to punish a request that asked for nothing.
    fn array_or_empty<'w, T>(&mut self, a: Option<&'w [T]>) -> &'w [T] {
        a.unwrap_or_default()
    }

    /// The verdict on a query command that names queries by index. Every refusal is the guest
    /// naming what it does not have -- a pool with no record, queries past the pool, results
    /// past the room -- or a device this table cannot reach, and none of them reached the
    /// driver, which would have trusted the index.
    fn queried<R>(&mut self, r: Result<R, driver::QueryRefused>) -> Option<R> {
        use driver::QueryRefused as Q;
        match r {
            Ok(r) => Some(r),
            Err(why) => {
                self.reject = Some(match why {
                    Q::NoDevice => "named a query pool on a device with no table here",
                    Q::NoHostReset => {
                        "reset a query pool from the host on a device that exports no reset"
                    }
                    Q::UnknownPool => "named a query pool this renderer has no record of",
                    Q::OutOfPool => "named queries past the end of the pool",
                    Q::OutOfRoom => "asked for query results past the room it offered",
                    Q::Unsized => "read results of a query kind this renderer cannot size",
                });
                None
            }
        }
    }

    /// The verdict on a free, for both frees. A run that was not the pool's is the guest naming
    /// objects it does not hold under that pool, and the command is refused for it; a device
    /// with no table here is the same. Neither reached the driver.
    fn freed<R>(&mut self, r: Result<R, driver::FreeRefused>) -> Option<R> {
        match r {
            Ok(r) => Some(r),
            Err(driver::FreeRefused::NotFromThisPool) => {
                self.reject = Some("frees objects the pool it names did not allocate");
                None
            }
            Err(driver::FreeRefused::NoDevice) => {
                self.reject = Some("frees from a pool on a device with no table here");
                None
            }
        }
    }

    /// Refuse every id in a run the guest sent.
    ///
    /// The generated lifecycle hook walks the whole array whatever the handler did with it, so
    /// every id it will reach needs a decision recorded against it. The ones a short answer did
    /// not fill, and all of them when the call failed outright, are refusals: left alone they
    /// would be registered as their own handles and the guest would hold objects that do not
    /// exist. A zero id is not one the guest can name, and [`Table::add_ghost`] drops it.
    /// Take a run of ids out of the object table because the thing that owned them is gone.
    ///
    /// Destroying a pool destroys its objects, and destroying a device destroys its pools -- both
    /// without a command naming any of the objects. Their entries have to go at that moment, or
    /// the table keeps resolving an id to a handle the driver has freed and the next command
    /// naming one hands it back to Vulkan. Afterwards the id resolves to nothing, and a guest
    /// that names it stops its own ring, which is what the C does too.
    fn forget(&mut self, orphans: Vec<ObjectId>) {
        let mut table = self.objects.borrow_mut();
        for id in orphans {
            table.remove(id);
        }
    }

    /// Every recording command's verdict on whether it reached the driver at all.
    ///
    /// The only way one can fail before the submit: the command buffer resolved in the object
    /// table but the driver has no pool record for it, so there is no device to record through.
    /// The two disagreeing is not something a guest can arrange, but it is also not something to
    /// record blindly past -- so the ring stops rather than the process.
    fn recorded(&mut self, done: Option<()>) {
        if done.is_none() {
            self.no_recorder();
        }
    }

    fn no_recorder(&mut self) {
        self.reject = Some("recorded into a command buffer with no device behind it");
    }

    /// The verdict on a sync-fd command. Neither carries a result the guest can read, so the ring
    /// is the only place a failure can be reported -- which is what the C does with them too.
    fn synced(&mut self, cmd: &str, done: Result<VkResult, NoSyncFd>) {
        match done {
            Ok(VkResult::VK_SUCCESS) => {}
            Ok(r) => {
                eprintln!("[virglrs] {cmd} refused by the driver: {r:?}");
                self.reject = Some("asked for a semaphore payload the driver would not move");
            }
            Err(NoSyncFd::NoDevice) => {
                self.reject = Some("moved a semaphore payload on a device it does not have");
            }
            Err(NoSyncFd::Unsupported) => {
                eprintln!("[virglrs] {cmd}: this driver exports no external semaphore fd");
                self.reject = Some("asked for venus sync on a driver that cannot do it");
            }
        }
    }

    fn ghost_ids<T: Handle>(&mut self, ids: &[Guest<T>]) {
        for id in ids {
            self.objects.borrow_mut().add_ghost(id.id());
        }
    }
}

/// A create whose whole host action is one `vkCreateX(device, info, alloc, out)`.
///
/// Twenty Vulkan objects have exactly this shape, and writing them out forty times would be forty
/// chances to transpose two arguments. The C generates the same bodies from a JSON list and still
/// hand-writes each dispatch function, which is the split copied here: the macro is the body, the
/// invocation list below is the dispatch, and an object that needs more than the body does not
/// appear in the list at all -- it gets a method written out in full.
///
/// The device is re-checked inside `Driver`, and a miss is a refusal rather than a silent return,
/// so the id becomes a ghost and the commands the guest already pipelined behind it are lost one
/// at a time instead of poisoning the ring.
macro_rules! simple_create {
    ($cmd:ident, $args:ty, $info:ident, $out:ident, $shadow:ident) => {
        fn $cmd(&mut self, args: &mut $args) {
            let Some(info) = self.names(args.$info) else { return };
            let host = self.driver.create_object(args.device, |d| d.$cmd(), info, args.pAllocator);
            args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
            self.plant(stringify!($cmd), args.$out(), args.$shadow(), host);
        }
    };
}

/// A pool, which is [`simple_create`] plus the record of what will be allocated from it.
///
/// Destroying a pool destroys its contents, so the driver has to know which objects a pool owns
/// to stop answering for them afterwards; see [`Driver::create_pool`].
macro_rules! pool_create {
    ($cmd:ident, $args:ty, $info:ident, $out:ident, $shadow:ident) => {
        fn $cmd(&mut self, args: &mut $args) {
            let Some(info) = self.names(args.$info) else { return };
            let host = self.driver.create_pool(args.device, |d| d.$cmd(), info, args.pAllocator);
            args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
            self.plant(stringify!($cmd), args.$out(), args.$shadow(), host);
        }
    };
}

/// The destroy half of [`pool_create`].
macro_rules! pool_destroy {
    ($cmd:ident, $args:ty, $target:ident) => {
        fn $cmd(&mut self, args: &mut $args) {
            let orphans =
                self.driver.destroy_pool(args.device, |d| d.$cmd(), args.$target, args.pAllocator);
            self.forget(orphans);
        }
    };
}

/// The destroy half of [`simple_create`]. The object table entry is removed by the generated
/// lifecycle hook, so all this owes is the driver call.
macro_rules! simple_destroy {
    ($cmd:ident, $args:ty, $target:ident) => {
        fn $cmd(&mut self, args: &mut $args) {
            self.driver.destroy_object(args.device, |d| d.$cmd(), args.$target, args.pAllocator);
        }
    };
}

/// The API version the guest is allowed to be told, whatever the driver claims.
///
/// Not a workaround, and not a number to copy: it is the highest Vulkan version *this build can
/// serialize*, which is a property of the pinned vk.xml the decoder was generated from and of
/// nothing else. So it is read off `info::VK_XML_VERSION` rather than written down, and a bump of
/// the pinned protocol moves it without anyone remembering to.
///
/// Telling the guest a higher version is telling it to use structs this renderer would then refuse
/// -- the guest enables 1.5, sends a 1.5 `pNext`, and the decoder poisons its ring for asking. The
/// patch level stays the driver's own: it identifies the implementation and carries no structs.
fn cap_api_version(version: u32) -> u32 {
    const MINOR: u32 = 12;
    const PATCH: u32 = 0xfff;
    let ceiling = crate::venus::proto::info::VK_XML_VERSION;
    if (version >> MINOR) > (ceiling >> MINOR) {
        (ceiling & !PATCH) | (version & PATCH)
    } else {
        version
    }
}

impl Commands for Handlers<'_> {
    /// No handler ran: the command is counted for the census and nothing else. There is no
    /// version of that which is safe to continue from. When the guest wanted a reply, the
    /// generator encoded one regardless, out of arguments no handler ever filled in, and a
    /// zeroed reply is shaped exactly like a successful one -- there is no field in which to
    /// say "we did not do this". When it wanted none, the guest is not waiting, but it does go
    /// on believing the host did the thing; the divergence surfaces later, somewhere that cannot
    /// name this command. Either way the context dies, saying which command it was, the way it
    /// does for any command a handler refuses. This is also what upstream does: its generated
    /// wrapper for a command with no handler sets fatal before decoding, whatever the reply flag
    /// says.
    fn unsupported(&mut self, cmd: VkCommandTypeEXT) {
        *self.todo.seen.entry(cmd.0).or_default() += 1;
        self.reject = Some("is not a command this build serves");
    }

    fn object_created(
        &mut self,
        ty: VkObjectType,
        id: ObjectId,
        host: HostHandle,
        owner: Option<ObjectId>,
    ) {
        // This hook runs for every create, served or not, and a zero handle means only that the
        // shadow was never written -- it cannot tell "no handler ran" from "the handler ran and
        // the driver refused". So a handler that already decided is not second-guessed here:
        // registered stands, and a ghost stands. Only an id with no decision behind it gets the
        // unserved command's fiction, where the id stands in for a handle so the rest of the
        // stream still decodes.
        {
            let objects = self.objects.borrow();
            if objects.get(id).is_some() || objects.is_ghost(id) {
                return;
            }
        }
        let handle = if host.0 == 0 { HostHandle(id.0) } else { host };
        if self.objects.borrow_mut().add(id, ty, handle, owner).is_err() {
            self.reject = Some("named an object it cannot have");
        }
    }

    fn object_destroyed(&mut self, _ty: VkObjectType, id: ObjectId) {
        self.objects.borrow_mut().remove(id);
    }

    // ------------------------------------------------------------- the instance tree
    //
    // The dependency spine, in the only order it can be built: nothing below reaches a driver
    // without the instance above it. The corpus asks for these once or twice each, at the very
    // bottom of the frequency table -- and every one of the hot commands is unreachable until
    // they are served.

    fn vkCreateInstance(&mut self, args: &mut vn_command_vkCreateInstance<'_>) {
        let Some(info) = self.names(args.pCreateInfo) else { return };
        let host = self.driver.create_instance(self.global, info, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateInstance", args.pInstance(), args.handle_pInstance_mut(), host);
    }

    fn vkDestroyInstance(&mut self, args: &mut vn_command_vkDestroyInstance<'_>) {
        // The whole tree comes out first, and while every device in it is still alive -- an object
        // cannot be destroyed after the device that owns it, and the guest is under no obligation
        // to have destroyed any of them itself.
        let doomed = self.objects.borrow_mut().take_tree(args.id_instance);
        // Every device under it dies next: Vulkan's teardown order is not advisory, and a guest
        // that skipped its own destroys does not get to leak them onto the host.
        self.driver.teardown(&doomed);
    }

    fn vkEnumeratePhysicalDevices(&mut self, args: &mut vn_command_vkEnumeratePhysicalDevices<'_>) {
        if !args.has_pPhysicalDeviceCount() {
            return;
        }
        // A null array is the guest asking how many there are. Answering it needs no ids, so
        // there is nothing to register and nothing to plant.
        if !args.has_pPhysicalDevices() {
            if let Ok(n) = self.driver.physical_device_count(args.instance)
                && let Some(count) = args.pPhysicalDeviceCount_mut()
            {
                *count = n;
            }
            return;
        }

        // Both arrays were sized from the same count, and the wire's own size was checked
        // against it as it decoded -- so one length governs the pair, and neither accessor
        // can hand back a slice of any other length.
        let Some(ids) = self.array(args.pPhysicalDevices()) else { return };
        // The shadow is borrowed from `args`, so everything else this needs off it is read first.
        // That is the borrow doing its job: while the driver is writing host handles into the
        // array, nothing else may be reading the struct that owns it.
        let instance = args.instance;
        let Some(out) = self.array(args.handle_pPhysicalDevices_mut()) else { return };
        let Ok(got) = self.driver.physical_devices(instance, out) else {
            args.ret = VkResult::VK_ERROR_INITIALIZATION_FAILED;
            self.ghost_ids(ids);
            return;
        };
        // What each one supports is asked once, here, because device creation is filtered against
        // it and there is no later point where the guest is guaranteed to have named them all.
        // A device whose driver refuses the question fails the whole enumeration: handing it over
        // unlearned would advertise nothing and then quietly create devices without the
        // extensions the guest asked for.
        let mut learned = Ok(());
        for pd in out.iter().take(got as usize) {
            learned = self.driver.learn_extensions(*pd);
            if learned.is_err() {
                break;
            }
        }
        if let Err(e) = learned {
            args.ret = e;
            self.ghost_ids(ids);
            return;
        }
        // The count goes back last, and has to: it lives in the same struct the shadow array was
        // borrowed from, so the two cannot be held at once. That is not the borrow checker being
        // awkward -- how many there are is not known until the array has been filled.
        if let Some(count) = args.pPhysicalDeviceCount_mut() {
            *count = got;
        }
        self.ghost_ids(&ids[got as usize..]);
    }

    fn vkCreateDevice(&mut self, args: &mut vn_command_vkCreateDevice<'_>) {
        let Some(info) = self.names(args.pCreateInfo) else { return };
        let host = self.driver.create_device(args.physicalDevice, info, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateDevice", args.pDevice(), args.handle_pDevice_mut(), host);
    }

    fn vkDestroyDevice(&mut self, args: &mut vn_command_vkDestroyDevice<'_>) {
        // Taken out before the driver call, because destroying them afterwards would be destroying
        // them on a device that no longer exists. Nothing else names these: the guest sent no
        // command for any of them, which is why the table hands them back rather than dropping
        // them, and why the result cannot be ignored.
        let doomed = self.objects.borrow_mut().take_tree(args.id_device);
        let orphans = self.driver.destroy_device(args.device, &doomed);
        self.forget(orphans);
    }

    // ------------------------------------------------------------------- device memory
    //
    // What the memory census reads back, and the first thing the guest does with a device.

    fn vkAllocateMemory(&mut self, args: &mut vn_command_vkAllocateMemory<'_>) {
        // The id has to be read before the allocation, because it is the key the driver files it
        // under -- and it is the guest's, chosen in the request, not anything the host picks.
        let Some(id) = self.out_id(args.pMemory()) else {
            return;
        };
        let Some(info) = self.names(args.pAllocateInfo) else { return };
        // Read out of `self` before the driver takes it mutably: both are plain copies of a
        // shared reference and an id, so the resolver borrows nothing the driver also wants.
        let (resources, ctx) = (self.resources, self.ctx);
        let host = self.driver.allocate_memory(args.device, id, info, args.pAllocator, &|handle| {
            resources.bytes(ctx, handle)
        });
        // A budget refusal stops the context, and it is this handler's to say so: the guest is
        // never going to read `ret`, which is the whole reason the budget module exists. A
        // driver refusal is left alone -- the guest unwinds from that the way it would on
        // hardware.
        let host = host.map_err(|e| {
            if let driver::NoMemory::OverBudget { stop: true } = e {
                self.reject = Some("the host memory budget refused this allocation");
            }
            e.ret()
        });
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkAllocateMemory", args.pMemory(), args.handle_pMemory_mut(), host);
    }

    fn vkFreeMemory(&mut self, args: &mut vn_command_vkFreeMemory<'_>) {
        // A null handle is a legal no-op in Vulkan, and the guest sends it: the lookup resolved
        // it to a null handle and the census has nothing under the id, so this needs no guard of
        // its own.
        self.driver.free_memory(args.device, args.memory, args.id_memory);
    }

    // --------------------------------------------------------------- the simple objects
    //
    // One `vkCreateX`/`vkDestroyX` pair each, with no host state beyond the object table. The
    // whole list is here rather than behind a loop in the generator so that adding one is a
    // visible line in a diff, and so that an object needing more than the pair cannot be added by
    // accident -- `vkCreateShaderModule` below is what that looks like.

    simple_create!(vkCreateFence, vn_command_vkCreateFence, pCreateInfo, pFence, handle_pFence_mut);
    simple_destroy!(vkDestroyFence, vn_command_vkDestroyFence, fence);

    simple_create!(
        vkCreateSemaphore,
        vn_command_vkCreateSemaphore,
        pCreateInfo,
        pSemaphore,
        handle_pSemaphore_mut
    );
    simple_destroy!(vkDestroySemaphore, vn_command_vkDestroySemaphore, semaphore);

    pool_create!(
        vkCreateCommandPool,
        vn_command_vkCreateCommandPool,
        pCreateInfo,
        pCommandPool,
        handle_pCommandPool_mut
    );
    pool_destroy!(vkDestroyCommandPool, vn_command_vkDestroyCommandPool, commandPool);

    simple_create!(
        vkCreateBuffer,
        vn_command_vkCreateBuffer,
        pCreateInfo,
        pBuffer,
        handle_pBuffer_mut
    );
    simple_destroy!(vkDestroyBuffer, vn_command_vkDestroyBuffer, buffer);

    /// Not [`simple_create`]: an image's extent and format cannot be asked for afterwards, and a
    /// scanout surface has to be minted at exactly them.
    fn vkCreateImage(&mut self, args: &mut vn_command_vkCreateImage<'_>) {
        let Some(info) = self.names(args.pCreateInfo) else { return };
        // An image the guest means to share is created with rows the host can address -- see
        // `external_images_are_linear`. The facts noted below are of the image the driver made.
        let info = driver::external_images_are_linear(info);
        let host =
            self.driver.create_object(args.device, |d| d.vkCreateImage(), &info, args.pAllocator);
        if let Ok(image) = host {
            self.driver.note_image(image, &info);
        }
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateImage", args.pImage(), args.handle_pImage_mut(), host);
    }

    /// Not [`simple_destroy`]: the record [`Self::vkCreateImage`] made goes with the image.
    fn vkDestroyImage(&mut self, args: &mut vn_command_vkDestroyImage<'_>) {
        self.driver.forget_image(args.image);
        self.driver.destroy_object(
            args.device,
            |d| d.vkDestroyImage(),
            args.image,
            args.pAllocator,
        );
    }

    simple_create!(
        vkCreateImageView,
        vn_command_vkCreateImageView,
        pCreateInfo,
        pView,
        handle_pView_mut
    );
    simple_destroy!(vkDestroyImageView, vn_command_vkDestroyImageView, imageView);

    simple_create!(
        vkCreateSampler,
        vn_command_vkCreateSampler,
        pCreateInfo,
        pSampler,
        handle_pSampler_mut
    );
    simple_destroy!(vkDestroySampler, vn_command_vkDestroySampler, sampler);

    simple_create!(
        vkCreateSamplerYcbcrConversion,
        vn_command_vkCreateSamplerYcbcrConversion,
        pCreateInfo,
        pYcbcrConversion,
        handle_pYcbcrConversion_mut
    );
    simple_destroy!(
        vkDestroySamplerYcbcrConversion,
        vn_command_vkDestroySamplerYcbcrConversion,
        ycbcrConversion
    );

    simple_create!(
        vkCreateRenderPass,
        vn_command_vkCreateRenderPass,
        pCreateInfo,
        pRenderPass,
        handle_pRenderPass_mut
    );
    simple_destroy!(vkDestroyRenderPass, vn_command_vkDestroyRenderPass, renderPass);

    simple_create!(
        vkCreateFramebuffer,
        vn_command_vkCreateFramebuffer,
        pCreateInfo,
        pFramebuffer,
        handle_pFramebuffer_mut
    );
    simple_destroy!(vkDestroyFramebuffer, vn_command_vkDestroyFramebuffer, framebuffer);

    simple_create!(
        vkCreateDescriptorSetLayout,
        vn_command_vkCreateDescriptorSetLayout,
        pCreateInfo,
        pSetLayout,
        handle_pSetLayout_mut
    );
    simple_destroy!(
        vkDestroyDescriptorSetLayout,
        vn_command_vkDestroyDescriptorSetLayout,
        descriptorSetLayout
    );

    pool_create!(
        vkCreateDescriptorPool,
        vn_command_vkCreateDescriptorPool,
        pCreateInfo,
        pDescriptorPool,
        handle_pDescriptorPool_mut
    );
    pool_destroy!(vkDestroyDescriptorPool, vn_command_vkDestroyDescriptorPool, descriptorPool);

    simple_create!(
        vkCreatePipelineLayout,
        vn_command_vkCreatePipelineLayout,
        pCreateInfo,
        pPipelineLayout,
        handle_pPipelineLayout_mut
    );
    simple_destroy!(vkDestroyPipelineLayout, vn_command_vkDestroyPipelineLayout, pipelineLayout);

    simple_create!(
        vkCreatePipelineCache,
        vn_command_vkCreatePipelineCache,
        pCreateInfo,
        pPipelineCache,
        handle_pPipelineCache_mut
    );
    simple_destroy!(vkDestroyPipelineCache, vn_command_vkDestroyPipelineCache, pipelineCache);

    /// Count-then-fill like the enumerations below, but counted in bytes and asked of the device.
    /// A guest that has a pipeline cache at all asks this a few seconds after every new pipeline,
    /// to save the cache to disk -- so a build that refuses it loses every such client on the
    /// first pipeline it builds, a little after the first frame it presents.
    fn vkGetPipelineCacheData(&mut self, args: &mut vn_command_vkGetPipelineCacheData<'_>) {
        let device = args.device;
        let cache = args.pipelineCache;
        if !self.counted(args.has_pDataSize()) {
            return;
        }
        if !args.has_pData() {
            let asked = self.driver.pipeline_cache_data(device, cache, None);
            match asked {
                Ok((n, ret)) => {
                    args.ret = ret;
                    if let Some(size) = args.pDataSize_mut() {
                        *size = n;
                    }
                }
                Err(e) => args.ret = e,
            }
            return;
        }
        let Some(out) = self.array(args.pData_mut()) else { return };
        let asked = self.driver.pipeline_cache_data(device, cache, Some(out));
        match asked {
            Ok((n, ret)) => {
                args.ret = ret;
                if let Some(size) = args.pDataSize_mut() {
                    *size = n;
                }
            }
            Err(e) => args.ret = e,
        }
    }

    /// Not [`simple_create`]: the pool is recorded, because every command naming its queries is
    /// measured against it -- see [`driver::QueryRefused`].
    fn vkCreateQueryPool(&mut self, args: &mut vn_command_vkCreateQueryPool<'_>) {
        let Some(info) = self.names(args.pCreateInfo) else { return };
        let host = self.driver.create_query_pool(args.device, info, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateQueryPool", args.pQueryPool(), args.handle_pQueryPool_mut(), host);
    }

    /// Not [`simple_destroy`]: the record [`Self::vkCreateQueryPool`] made goes with the pool.
    fn vkDestroyQueryPool(&mut self, args: &mut vn_command_vkDestroyQueryPool<'_>) {
        self.driver.forget_query_pool(args.queryPool);
        self.driver.destroy_object(
            args.device,
            |d| d.vkDestroyQueryPool(),
            args.queryPool,
            args.pAllocator,
        );
    }

    fn vkResetQueryPool(&mut self, args: &mut vn_command_vkResetQueryPool<'_>) {
        let done = self.driver.reset_query_pool(
            args.device,
            args.queryPool,
            args.firstQuery,
            args.queryCount,
        );
        self.queried(done);
    }

    /// The read-back. `pData` is room the guest offered, `dataSize` bytes of it, and the reply
    /// carries all of it back whatever the driver filled -- the wire's shape, not a choice. What
    /// is checked here is that the queries named and the results asked for fit: a pool has
    /// so many queries, and a result has a size the pool's record fixes.
    fn vkGetQueryPoolResults(&mut self, args: &mut vn_command_vkGetQueryPoolResults<'_>) {
        let device = args.device;
        let pool = args.queryPool;
        let (first, count, stride, flags) =
            (args.firstQuery, args.queryCount, args.stride, args.flags);
        let Some(out) = self.array(args.pData_mut()) else { return };
        let read = self.driver.query_pool_results(device, pool, first, count, out, stride, flags);
        // The one query command with a result, so a device with no table here is answered the
        // way every other device call answers it, rather than refused.
        if read == Err(driver::QueryRefused::NoDevice) {
            args.ret = VkResult::VK_ERROR_INITIALIZATION_FAILED;
            return;
        }
        if let Some(ret) = self.queried(read) {
            args.ret = ret;
        }
    }

    fn vkMergePipelineCaches(&mut self, args: &mut vn_command_vkMergePipelineCaches<'_>) {
        let srcs = args.pSrcCaches();
        match self.driver.merge_pipeline_caches(args.device, args.dstCache, srcs) {
            Ok(ret) | Err(ret) => args.ret = ret,
        }
    }

    /// The one simple object with a check in front of it.
    ///
    /// `codeSize` is a byte count, uniquely among Vulkan's typed arrays, and the wire carries
    /// `codeSize / 4` words -- so a `codeSize` that is not a multiple of four decodes into an
    /// allocation shorter than the number the driver is then handed, and the driver reads off the
    /// end of it. The guest chooses that number, which makes rejecting it the boundary's job.
    fn vkCreateShaderModule(&mut self, args: &mut vn_command_vkCreateShaderModule<'_>) {
        let Some(info) = self.names(args.pCreateInfo) else { return };
        if info.codeSize % 4 != 0 {
            self.reject = Some("gave a shader a code size that is not a whole number of words");
            return;
        }
        let host = self.driver.create_object(
            args.device,
            |d| d.vkCreateShaderModule(),
            info,
            args.pAllocator,
        );
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant(
            "vkCreateShaderModule",
            args.pShaderModule(),
            args.handle_pShaderModule_mut(),
            host,
        );
    }

    simple_destroy!(vkDestroyShaderModule, vn_command_vkDestroyShaderModule, shaderModule);

    // --------------------------------------------------------------- the pool objects
    //
    // Allocated in runs from a pool rather than one at a time, so the out-handles are an array and
    // a refusal has to ghost every id in it. Vulkan fills the whole array or none of it, so unlike
    // an enumeration there is no short answer between those two.

    fn vkAllocateCommandBuffers(&mut self, args: &mut vn_command_vkAllocateCommandBuffers<'_>) {
        // The count inside the create-info is what sized both arrays, and the wire's own size
        // was checked against it -- which is the count both accessors below read.
        let ids = args.pCommandBuffers();
        // Read before the shadow is borrowed: see `vkEnumeratePhysicalDevices`.
        let device = args.device;
        let Some(info) = self.names(args.pAllocateInfo) else { return };
        let pool = info.commandPool;
        // The pool records both names of every object it holds, so that destroying it can take
        // the guest's out of the object table. Built before the shadow is borrowed.
        let named: Vec<ObjectId> = ids.iter().map(|h| h.id()).collect();
        let out = args.handle_pCommandBuffers_mut();
        let host = self.driver.allocate_objects(
            device,
            pool,
            |d| d.vkAllocateCommandBuffers(),
            info,
            out,
            &named,
        );
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        if host.is_err() {
            eprintln!("[virglrs] vkAllocateCommandBuffers refused by the driver");
            self.ghost_ids(ids);
        }
    }

    fn vkFreeCommandBuffers(&mut self, args: &mut vn_command_vkFreeCommandBuffers<'_>) {
        let buffers = self.array_or_empty(args.pCommandBuffers());
        if buffers.is_empty() {
            return;
        }
        let r = self.driver.free_objects(
            args.device,
            |d| d.vkFreeCommandBuffers(),
            args.commandPool,
            buffers,
        );
        self.freed(r);
    }

    fn vkFreeDescriptorSets(&mut self, args: &mut vn_command_vkFreeDescriptorSets<'_>) {
        let sets = self.array_or_empty(args.pDescriptorSets());
        // The spec's answer is always success, and freeing nothing is not a failure either.
        args.ret = VkResult::VK_SUCCESS;
        if sets.is_empty() {
            return;
        }
        let r = self.driver.free_objects(
            args.device,
            |d| d.vkFreeDescriptorSets(),
            args.descriptorPool,
            sets,
        );
        if let Some(ret) = self.freed(r) {
            args.ret = ret;
        }
    }

    fn vkAllocateDescriptorSets(&mut self, args: &mut vn_command_vkAllocateDescriptorSets<'_>) {
        let ids = args.pDescriptorSets();
        // Read before the shadow is borrowed: see `vkEnumeratePhysicalDevices`.
        let device = args.device;
        let Some(info) = self.names(args.pAllocateInfo) else { return };
        let pool = info.descriptorPool;
        // The pool records both names of every object it holds, so that destroying it can take
        // the guest's out of the object table. Built before the shadow is borrowed.
        let named: Vec<ObjectId> = ids.iter().map(|h| h.id()).collect();
        let out = args.handle_pDescriptorSets_mut();
        let host = self.driver.allocate_objects(
            device,
            pool,
            |d| d.vkAllocateDescriptorSets(),
            info,
            out,
            &named,
        );
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        if host.is_err() {
            // Running a descriptor pool dry is a normal thing for a guest to do -- it is how a
            // guest discovers the pool is too small -- so this one is not logged as a surprise.
            self.ghost_ids(ids);
        }
    }

    fn vkGetDeviceQueue2(&mut self, args: &mut vn_command_vkGetDeviceQueue2<'_>) {
        // A queue is owned by its device and never created, so the guest's id is registered
        // against a handle the driver merely hands back.
        let Some(info) = self.names(args.pQueueInfo) else { return };
        let host = self.driver.device_queue(args.device, info);
        self.plant(
            "vkGetDeviceQueue2",
            args.pQueue(),
            args.handle_pQueue_mut(),
            host.ok_or(VkResult::VK_ERROR_INITIALIZATION_FAILED),
        );
    }

    fn vkEnumeratePhysicalDeviceGroups(
        &mut self,
        args: &mut vn_command_vkEnumeratePhysicalDeviceGroups<'_>,
    ) {
        // The one query that hands back handles *inside* the struct the driver filled. Every
        // other command names its objects in members the generator can shadow; these sit in a
        // fixed array inside an out-struct, where there is nowhere to put a shadow -- so the swap
        // back to the guest's own ids is this handler's, and it has to happen before the reply
        // encodes. Left alone, the guest is handed live host pointers.
        if !self.counted(args.has_pPhysicalDeviceGroupCount()) {
            return;
        }
        let instance = args.instance;
        if !args.has_pPhysicalDeviceGroupProperties() {
            let asked = self
                .driver
                .enumerate_into(instance, None, |i| i.try_vkEnumeratePhysicalDeviceGroups());
            match asked {
                Ok((n, ret)) => {
                    args.ret = ret;
                    if ret == VkResult::VK_SUCCESS
                        && let Some(count) = args.pPhysicalDeviceGroupCount_mut()
                    {
                        *count = n;
                    }
                }
                Err(e) => args.ret = e,
            }
            return;
        }

        let Some(out) = self.array(args.pPhysicalDeviceGroupProperties_mut()) else { return };
        let asked = self
            .driver
            .enumerate_into(instance, Some(out), |i| i.try_vkEnumeratePhysicalDeviceGroups());
        let (n, ret) = match asked {
            Ok(pair) => pair,
            Err(e) => {
                args.ret = e;
                return;
            }
        };
        if ret != VkResult::VK_SUCCESS && ret != VkResult::VK_INCOMPLETE {
            args.ret = ret;
            return;
        }

        // Swap every host handle for the id the guest gave it. A handle with no id behind it is
        // a guest that never enumerated its physical devices -- it cannot be told about a device
        // it has no name for, and inventing one would name something it never asked to exist.
        let table = self.objects.borrow();
        let mut unknown = false;
        for group in out.iter_mut().take(n as usize) {
            let live = (group.physicalDeviceCount as usize).min(group.physicalDevices.len());
            for pd in &mut group.physicalDevices[..live] {
                match table.id_of_handle(VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE, pd.host()) {
                    Some(id) => *pd = VkPhysicalDevice(id.0),
                    None => unknown = true,
                }
            }
        }
        drop(table);
        if unknown {
            // Refusing outright rather than half-swapping: a group carrying one real id and one
            // host pointer is worse than no answer, because the guest cannot tell them apart.
            args.ret = VkResult::VK_ERROR_INITIALIZATION_FAILED;
            return;
        }

        args.ret = ret;
        if let Some(count) = args.pPhysicalDeviceGroupCount_mut() {
            *count = n;
        }
    }

    fn vkEnumerateDeviceExtensionProperties(
        &mut self,
        args: &mut vn_command_vkEnumerateDeviceExtensionProperties<'_>,
    ) {
        // The one query answered without asking the driver at all. What the guest may be told is
        // not what the hardware has: it is what this build can serialize, which `Driver` derives.
        if args.has_pLayerName() {
            // A layer is host-side software the guest cannot see and this renderer does not load,
            // so naming one is not a request that can be honoured or a mistake to smooth over.
            self.reject = Some("named a layer, which no venus renderer has");
            return;
        }
        if !self.counted(args.has_pPropertyCount()) {
            return;
        }
        let advertised = self.driver.advertised_extensions(args.physicalDevice);
        if !args.has_pProperties() {
            if let Some(count) = args.pPropertyCount_mut() {
                *count = advertised.len() as u32;
            }
            args.ret = VkResult::VK_SUCCESS;
            return;
        }
        let Some(out) = self.array(args.pProperties_mut()) else { return };
        // Vulkan's own short-answer rule: the guest gets what it sized for, and being told there
        // were more is not an error it has to recover from.
        let n = out.len().min(advertised.len());
        out[..n].copy_from_slice(&advertised[..n]);
        args.ret =
            if n < advertised.len() { VkResult::VK_INCOMPLETE } else { VkResult::VK_SUCCESS };
        if let Some(count) = args.pPropertyCount_mut() {
            *count = n as u32;
        }
    }

    fn vkGetPhysicalDeviceQueueFamilyProperties2(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceQueueFamilyProperties2<'_>,
    ) {
        let pd = args.physicalDevice;
        if !self.counted(args.has_pQueueFamilyPropertyCount()) {
            return;
        }
        // A null array is the guest asking how many there are, which is the spec's own first call.
        if !args.has_pQueueFamilyProperties() {
            let asked = self
                .driver
                .enumerate_into(pd, None, |i| i.try_vkGetPhysicalDeviceQueueFamilyProperties2());
            if let Some((n, ())) = self.asked(asked)
                && let Some(count) = args.pQueueFamilyPropertyCount_mut()
            {
                *count = n;
            }
            return;
        }
        let Some(out) = self.array(args.pQueueFamilyProperties_mut()) else { return };
        let asked = self
            .driver
            .enumerate_into(pd, Some(out), |i| i.try_vkGetPhysicalDeviceQueueFamilyProperties2());
        // The count goes back last, and has to: it lives in the struct the array was borrowed
        // from, so the two cannot be held at once -- and how many there are is not known until
        // the array has been filled. See `vkEnumeratePhysicalDevices`.
        if let Some((n, ())) = self.asked(asked)
            && let Some(count) = args.pQueueFamilyPropertyCount_mut()
        {
            *count = n;
        }
    }

    // ------------------------------------------------------------------------ rings
    //
    // A ring is the guest's own command buffer in memory both sides can see. These two commands
    // arrive on the context's stream, never on a ring's, and they are what makes a ring exist.

    fn vkCreateRingMESA(&mut self, args: &mut vn_command_vkCreateRingMESA<'_>) {
        if self.current_ring.is_some() {
            self.reject = Some("created a ring from inside a ring's own stream");
            return;
        }
        let id = RingId(args.ring);
        let Some(info) = args.pCreateInfo else {
            self.reject = Some("asked to create a ring with no description of it");
            return;
        };

        // A ring id already in use is the guest contradicting itself. Refusing rather than
        // replacing matters: the entry that is already there owns a share of a mapping, and
        // dropping it silently would strand whatever is still reading from it.
        if self.rings.contains_key(&id) {
            self.reject = Some("created a ring under an id that is already a ring");
            return;
        }

        let ring = match Ring::create(self.resources, info, self.replaying) {
            Ok(r) => r,
            Err(RingError::NoResource(h)) => {
                eprintln!("[virglrs] vkCreateRingMESA: resource {h} is not a mapped shm resource");
                self.reject = Some("created a ring in a resource that has no host mapping");
                return;
            }
            Err(RingError::Layout(e)) => {
                eprintln!("[virglrs] vkCreateRingMESA: ring {id}: {e:?}");
                self.reject = Some("created a ring with a layout we will not touch");
                return;
            }

            Err(RingError::NotOurs { head, status }) => {
                eprintln!(
                    "[virglrs] vkCreateRingMESA: ring {id}: head={head} status={status:#x} before \
                     the host has written either"
                );
                self.reject = Some("created a ring another renderer is already driving");
                return;
            }
        };

        // A guest that wants to hear from us says so here, and says how often. Zero is not a
        // period the protocol leaves room to interpret -- it is a guest asking for a stamp every
        // no-time -- and it is refused rather than substituted, before the ring is registered, so
        // nothing has to be unwound.
        if let Some(want) = monitor_period(info) {
            let Some(period_us) = want else {
                self.reject = Some("asked to be monitored with a reporting period of zero");
                return;
            };
            match self.monitor {
                Some(m) => m.watch(&ring.control, period_us),
                None => {
                    let m = Monitor::start(self.ctx, period_us);
                    m.watch(&ring.control, period_us);
                    *self.monitor = Some(m);
                }
            }
        }

        // Registered idle, never started here. Promotion happens at the end of the batch, which is
        // where the caller knows whether it was replaying -- and doing it there rather than in the
        // handler is why a handler never needs to reach the renderer's locks. See `Vkr::promote`.
        self.rings.insert(id, RingSlot::Idle(ring));
        self.note = Some(Note::RingCreated(args.ring));
    }

    fn vkDestroyRingMESA(&mut self, args: &mut vn_command_vkDestroyRingMESA<'_>) {
        if self.current_ring.is_some() {
            self.reject = Some("destroyed a ring from inside a ring's own stream");
            return;
        }
        let id = RingId(args.ring);
        // Dropping the entry is the teardown: it releases this ring's share of the resource's
        // mapping, and the mapping goes when the last share does. A running ring is stopped
        // first, which joins its thread -- safe from here only because the refusal above means
        // this is never a ring's own thread asking, and because that thread never blocks on the
        // context lock this dispatch is holding.
        match self.rings.remove(&id) {
            None => self.reject = Some("destroyed a ring that was never created"),
            Some(RingSlot::Idle(_)) => self.note = Some(Note::RingGone(args.ring)),
            Some(RingSlot::Running(t)) => {
                drop(t.stop());
                self.note = Some(Note::RingGone(args.ring));
            }
        }
    }

    /// The guest rang a ring's doorbell.
    ///
    /// An unknown ring poisons, matching the C: unlike a submission for a missing ring, which the
    /// caller may simply have raced, a notify names a ring the guest believes it owns.
    fn vkNotifyRingMESA(&mut self, args: &mut vn_command_vkNotifyRingMESA<'_>) {
        if self.current_ring.is_some() {
            self.reject = Some("rang a ring's doorbell from inside a ring's own stream");
            return;
        }
        match self.rings.get(&RingId(args.ring)) {
            None => self.reject = Some("rang the doorbell of a ring that was never created"),
            // Not yet reading, so there is nothing to wake. Harmless to miss: promotion happens
            // at the end of this batch, and a fresh thread reads the tail before it can park.
            Some(RingSlot::Idle(_)) => {}
            Some(RingSlot::Running(t)) => t.notify(),
        }
    }

    /// Write one guest-named word into a ring's `extra` region.
    ///
    /// The whole of the command: the offset is checked against `extra` alone, not against the
    /// mapping, because the rest of the resource is not the guest's to reach through this door.
    fn vkWriteRingExtraMESA(&mut self, args: &mut vn_command_vkWriteRingExtraMESA<'_>) {
        if self.current_ring.is_some() {
            self.reject = Some("wrote a ring's extra word from inside a ring's own stream");
            return;
        }
        match self.rings.get(&RingId(args.ring)) {
            None => self.reject = Some("wrote the extra word of a ring that was never created"),
            Some(slot) => {
                if !slot.control().write_extra(args.offset, args.value) {
                    self.reject = Some("wrote outside the ring's extra region");
                }
            }
        }
    }

    /// Publish a virtqueue seqno for a ring, releasing any wait of that ring's that it satisfies.
    ///
    /// The counterpart of [`Self::vkWaitVirtqueueSeqnoMESA`], and the *only* producer of the value
    /// that one consumes -- which is why a ring blocked on a seqno this stream has not published
    /// can never be unblocked by this stream waiting instead. See [`RingWaiter`].
    ///
    /// An idle ring is a ring created earlier in this very batch: the seqno is stored in its body
    /// and carried into its thread's park state at promotion, because it is state and not an edge.
    fn vkSubmitVirtqueueSeqnoMESA(&mut self, args: &mut vn_command_vkSubmitVirtqueueSeqnoMESA<'_>) {
        if self.current_ring.is_some() {
            self.reject = Some("submitted a virtqueue seqno from inside a ring's own stream");
            return;
        }
        match self.rings.get_mut(&RingId(args.ring)) {
            None => {
                self.reject = Some("submitted a virtqueue seqno for a ring that was never created")
            }
            Some(RingSlot::Idle(r)) => r.virtqueue_seqno = r.virtqueue_seqno.max(args.seqno),
            Some(RingSlot::Running(t)) => t.submit_virtqueue_seqno(args.seqno),
        }
    }

    /// Block this ring until the context publishes a virtqueue seqno.
    ///
    /// Only legal on a ring's own stream -- it has no `ring` argument because the ring is the one
    /// it arrived on -- and it does not block here. The dispatch it is inside holds the context
    /// lock, and the thread that would satisfy this wait needs that same lock to run the command
    /// that satisfies it; sleeping here would be sleeping on a door held shut from this side. So
    /// the batch suspends instead and the ring's own loop does the waiting, where a stop can
    /// still reach it. See [`Submitted::Waiting`].
    fn vkWaitVirtqueueSeqnoMESA(&mut self, args: &mut vn_command_vkWaitVirtqueueSeqnoMESA<'_>) {
        let Some(id) = self.current_ring else {
            self.reject = Some("waited on a virtqueue seqno from the context's own stream");
            return;
        };
        // Already satisfied is the common case and costs nothing: the guest submits the seqno and
        // waits for it in that order far more often than it gets ahead of itself.
        let published = match self.rings.get(&id) {
            None => {
                self.reject = Some("waited on a virtqueue seqno for a ring that is not here");
                return;
            }
            Some(RingSlot::Idle(r)) => r.virtqueue_seqno,
            Some(RingSlot::Running(t)) => t.virtqueue_seqno(),
        };
        if published < args.seqno {
            self.wait = Some(Wait::Virtqueue(args.seqno));
        }
    }

    /// Block this stream until a ring's head reaches a position.
    ///
    /// Only legal on the context's own stream. Like its sibling it suspends rather than blocks,
    /// and for a sharper reason: this dispatch runs on the virtio-gpu control queue thread, which
    /// is one thread for the whole *device*. Sleeping here with the context locked would stop the
    /// ring being waited for, and sleeping at all would stop every other context's submissions,
    /// every scanout flush and every fence with it.
    ///
    /// The ring is woken first, as the C does: a parked ring must drain to a state the waiter's
    /// guards can judge, or a wait that can never be satisfied looks the same as one that has not
    /// been yet.
    fn vkWaitRingSeqnoMESA(&mut self, args: &mut vn_command_vkWaitRingSeqnoMESA<'_>) {
        if self.current_ring.is_some() {
            self.reject = Some("waited on a ring seqno from inside a ring's own stream");
            return;
        }
        // A ring seqno is a byte position in a 32-bit free-running counter, widened to fit the
        // wire's field. A guest naming a value that does not fit is describing a position its own
        // ring cannot hold; truncating it would build a wait on a number nobody asked for.
        let Ok(seqno) = u32::try_from(args.seqno) else {
            self.reject = Some("waited on a ring seqno too large to be a position in a ring");
            return;
        };
        match self.rings.get(&RingId(args.ring)) {
            None => self.reject = Some("waited on the seqno of a ring that was never created"),
            // Nothing advances an idle ring's head -- it has no thread yet, and promotion happens
            // only once this batch is over, which this command is inside. Suspending on it would
            // be suspending forever.
            Some(RingSlot::Idle(_)) => {
                self.reject = Some("waited on the seqno of a ring that is not reading yet")
            }
            Some(RingSlot::Running(t)) => {
                t.notify();
                if !seqno_ge(t.control().head(), seqno) {
                    self.wait = Some(Wait::Ring { ring: RingId(args.ring), seqno });
                }
            }
        }
    }

    /// Point the answers at a window of guest memory.
    ///
    /// The guest re-establishes this before each batch it expects replies for, which is why the
    /// corpus is thick with them and why setting one twice rewinds rather than being refused.
    ///
    /// This is the one transport command with no opinion about where it arrived: the C runs it on
    /// both the context dispatch and every ring's, writing into whichever encoder that dispatch
    /// owns. The routing below is that, made explicit.
    fn vkSetReplyCommandStreamMESA(
        &mut self,
        args: &mut vn_command_vkSetReplyCommandStreamMESA<'_>,
    ) {
        let Some(stream) = args.pStream else {
            self.reject = Some("set a reply stream without saying where it is");
            return;
        };

        let reply = match ReplyStream::set(self.resources, stream) {
            Ok(r) => r,
            Err(ReplyStreamError::NoResource(h)) => {
                eprintln!(
                    "[virglrs] vkSetReplyCommandStreamMESA: resource {h} is not a mapped shm resource"
                );
                self.reject = Some("set a reply stream in a resource that has no host mapping");
                return;
            }
            Err(e @ ReplyStreamError::OutOfRange { .. }) => {
                eprintln!("[virglrs] vkSetReplyCommandStreamMESA: {e:?}");
                self.reject = Some("set a reply stream that does not fit the resource holding it");
                return;
            }
        };

        // Whoever owns this stream lent us its slot, so there is nothing to route: the answer
        // lands where the question came from by construction. Looking the ring up here instead
        // would be a lookup that a destroyed ring could make wrong.
        *self.reply = Some(reply);
    }

    /// Put the next answer somewhere other than after the last one.
    ///
    /// The guest does this when it knows where in its own window a reply belongs -- it has already
    /// worked out the layout and is telling us, rather than asking us to append. A position past
    /// the end of the window is refused instead of clamped: clamping would write the answer
    /// somewhere the guest is not reading and report that as success, which is the shape of bug
    /// this renderer exists to stop making.
    fn vkSeekReplyCommandStreamMESA(
        &mut self,
        args: &mut vn_command_vkSeekReplyCommandStreamMESA<'_>,
    ) {
        let Some(reply) = self.reply.as_mut() else {
            self.reject = Some("seeked a reply stream that was never set");
            return;
        };
        if !reply.seek(args.position) {
            eprintln!(
                "[virglrs] vkSeekReplyCommandStreamMESA: {} is past a {}-byte window",
                args.position,
                reply.window().size()
            );
            self.reject = Some("seeked a reply stream past the end of its own window");
        }
    }

    /// Run the command streams the guest recorded elsewhere.
    ///
    /// This is how every recorded `vkCmd*` reaches us: mesa fills a resource with a stream and
    /// then names it here, rather than sending the commands inline. The handler only resolves and
    /// records what was asked; the loop runs it, because running a stream needs a decoder and a
    /// handler is handed none. See [`Execute`].
    ///
    /// `pDependencies` and `flags` are read by nobody, here or in the C: the streams named are
    /// executed in the order given, which is what a dependency between them could ask for anyway.
    fn vkExecuteCommandStreamsMESA(
        &mut self,
        args: &mut vn_command_vkExecuteCommandStreamsMESA<'_>,
    ) {
        if !args.has_pStreams() {
            self.reject = Some("executed command streams without saying which");
            return;
        }
        let streams = args.pStreams();
        if streams.is_empty() {
            self.reject = Some("executed no command streams at all");
            return;
        }

        // Reply positions without a reply stream is not an empty request: the guest has said where
        // each stream's answers belong, and there is no window they could belong in. Serving the
        // streams anyway would run them and drop every answer.
        let reply_positions = match args.pReplyPositions() {
            Some(_) if self.reply.is_none() => {
                self.reject =
                    Some("executed command streams with reply positions and no reply stream");
                return;
            }
            Some(p) => Some(p.to_vec()),
            None => None,
        };

        self.execute = Some(Execute { streams: streams.to_vec(), reply_positions });
    }

    // The queries that carry a `ret`. Where a command has a field designed to say "no", that is
    // the honest channel and refusal is not: a driver without `VK_EXT_image_drm_format_modifier`
    // is an answer the guest asked for and can act on, not a reason to take its ring down. The
    // void queries above have no such field, which is why refusal is all they have.

    fn vkEnumerateInstanceExtensionProperties(
        &mut self,
        args: &mut vn_command_vkEnumerateInstanceExtensionProperties<'_>,
    ) {
        // Answered without the driver, like its device-level twin and for the same reason: what
        // the guest may be told is not what the host loader has, it is what this build speaks on
        // the wire. See `driver::renderer_extensions`.
        //
        // `pLayerName` is ignored rather than refused, which is the one place this differs from
        // `vkEnumerateDeviceExtensionProperties` -- and the difference is the answer's, not a
        // policy's. There is no layer whose extensions this list belongs to: it is the renderer's
        // own, so narrowing it by a layer name narrows it to nothing the guest could have meant.
        if !self.counted(args.has_pPropertyCount()) {
            return;
        }
        let speaks = crate::venus::driver::renderer_extensions();
        if !args.has_pProperties() {
            if let Some(count) = args.pPropertyCount_mut() {
                *count = speaks.len() as u32;
            }
            args.ret = VkResult::VK_SUCCESS;
            return;
        }
        let Some(out) = self.array(args.pProperties_mut()) else { return };
        let n = out.len().min(speaks.len());
        out[..n].copy_from_slice(&speaks[..n]);
        args.ret = if n < speaks.len() { VkResult::VK_INCOMPLETE } else { VkResult::VK_SUCCESS };
        if let Some(count) = args.pPropertyCount_mut() {
            *count = n as u32;
        }
    }

    fn vkEnumerateInstanceVersion(&mut self, args: &mut vn_command_vkEnumerateInstanceVersion<'_>) {
        // No handle anywhere in it: the guest may ask before any instance exists, so this is the
        // one query answered off the global table.
        let asked = self.driver.instance_version(self.global);
        let Some(out) = self.fills(args.pApiVersion_mut()) else { return };
        match asked {
            Ok(v) => {
                *out = cap_api_version(v);
                args.ret = VkResult::VK_SUCCESS;
            }
            Err(e) => args.ret = e,
        }
    }

    fn vkGetPhysicalDeviceImageFormatProperties2(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceImageFormatProperties2<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(info) = self.names(args.pImageFormatInfo) else { return };
        let Some(out) = self.fills(args.pImageFormatProperties_mut()) else { return };
        args.ret = self
            .driver
            .pd_query_info(pd, info, out, |i| i.try_vkGetPhysicalDeviceImageFormatProperties2())
            .unwrap_or_else(|e| e);
    }

    fn vkGetImageDrmFormatModifierPropertiesEXT(
        &mut self,
        args: &mut vn_command_vkGetImageDrmFormatModifierPropertiesEXT<'_>,
    ) {
        let (device, image) = (args.device, args.image);
        let Some(out) = self.fills(args.pProperties_mut()) else { return };
        args.ret = self
            .driver
            .dev_query_arg(device, image, out, |d| d.try_vkGetImageDrmFormatModifierPropertiesEXT())
            .unwrap_or_else(|e| e);
    }

    fn vkGetPhysicalDeviceProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(out) = self.fills(args.pProperties_mut()) else { return };
        let r = self.driver.pd_query(pd, &mut *out, |i| i.try_vkGetPhysicalDeviceProperties());
        if r.is_ok() {
            out.apiVersion = cap_api_version(out.apiVersion);
        }
        self.asked(r);
    }

    fn vkGetPhysicalDeviceProperties2(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceProperties2<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(out) = self.fills(args.pProperties_mut()) else { return };
        let r = self.driver.pd_query(pd, &mut *out, |i| i.try_vkGetPhysicalDeviceProperties2());
        if r.is_ok() {
            out.properties.apiVersion = cap_api_version(out.properties.apiVersion);
        }
        self.asked(r);
    }

    // ------------------------------------------------------------------------ queries
    //
    // The guest hands over a struct and asks for it back filled. Every one of these is three
    // lines because the work is not here: the decoder allocated the struct from the arena at the
    // layout a C compiler agrees with, so the driver writes into the very memory the reply
    // encoder reads back -- chained `pNext` structs included -- and this file only has to decide
    // what an absent struct and an unanswerable query mean.

    fn vkGetPhysicalDeviceFeatures2(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceFeatures2<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(out) = self.fills(args.pFeatures_mut()) else { return };
        let r = self.driver.pd_query(pd, out, |i| i.try_vkGetPhysicalDeviceFeatures2());
        self.asked(r);
    }

    fn vkGetPhysicalDeviceMemoryProperties2(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceMemoryProperties2<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(out) = self.fills(args.pMemoryProperties_mut()) else { return };
        let r = self.driver.pd_query(pd, out, |i| i.try_vkGetPhysicalDeviceMemoryProperties2());
        self.asked(r);
    }

    fn vkGetPhysicalDeviceFormatProperties2(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceFormatProperties2<'_>,
    ) {
        let (pd, format) = (args.physicalDevice, args.format);
        let Some(out) = self.fills(args.pFormatProperties_mut()) else { return };
        let r = self
            .driver
            .pd_query_arg(pd, format, out, |i| i.try_vkGetPhysicalDeviceFormatProperties2());
        self.asked(r);
    }

    fn vkGetPhysicalDeviceExternalFenceProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceExternalFenceProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(info) = self.names(args.pExternalFenceInfo) else { return };
        let Some(out) = self.fills(args.pExternalFenceProperties_mut()) else { return };
        let r = self
            .driver
            .pd_query_info(pd, info, out, |i| i.try_vkGetPhysicalDeviceExternalFenceProperties());
        self.asked(r);
    }

    fn vkGetPhysicalDeviceExternalSemaphoreProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceExternalSemaphoreProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(info) = self.names(args.pExternalSemaphoreInfo) else { return };
        let Some(out) = self.fills(args.pExternalSemaphoreProperties_mut()) else { return };
        let r = self.driver.pd_query_info(pd, info, out, |i| {
            i.try_vkGetPhysicalDeviceExternalSemaphoreProperties()
        });
        self.asked(r);
    }

    fn vkGetImageMemoryRequirements2(
        &mut self,
        args: &mut vn_command_vkGetImageMemoryRequirements2<'_>,
    ) {
        let device = args.device;
        let Some(info) = self.names(args.pInfo) else { return };
        let Some(out) = self.fills(args.pMemoryRequirements_mut()) else { return };
        let r = self
            .driver
            .dev_query_info(device, info, out, |d| d.try_vkGetImageMemoryRequirements2());
        self.asked(r);
    }

    fn vkGetBufferMemoryRequirements2(
        &mut self,
        args: &mut vn_command_vkGetBufferMemoryRequirements2<'_>,
    ) {
        let device = args.device;
        let Some(info) = self.names(args.pInfo) else { return };
        let Some(out) = self.fills(args.pMemoryRequirements_mut()) else { return };
        let r = self
            .driver
            .dev_query_info(device, info, out, |d| d.try_vkGetBufferMemoryRequirements2());
        self.asked(r);
    }

    fn vkGetImageSubresourceLayout(
        &mut self,
        args: &mut vn_command_vkGetImageSubresourceLayout<'_>,
    ) {
        let (device, image) = (args.device, args.image);
        let Some(sub) = self.names(args.pSubresource) else { return };
        let Some(out) = self.fills(args.pLayout_mut()) else { return };
        let r = self
            .driver
            .dev_query_arg_info(device, image, sub, out, |d| d.try_vkGetImageSubresourceLayout());
        self.asked(r);
    }

    fn vkGetPhysicalDeviceFeatures(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceFeatures<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(out) = self.fills(args.pFeatures_mut()) else { return };
        let r = self.driver.pd_query(pd, out, |i| i.try_vkGetPhysicalDeviceFeatures());
        self.asked(r);
    }

    fn vkGetPhysicalDeviceMemoryProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceMemoryProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(out) = self.fills(args.pMemoryProperties_mut()) else { return };
        let r = self.driver.pd_query(pd, out, |i| i.try_vkGetPhysicalDeviceMemoryProperties());
        self.asked(r);
    }

    fn vkGetPhysicalDeviceFormatProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceFormatProperties<'_>,
    ) {
        let (pd, format) = (args.physicalDevice, args.format);
        let Some(out) = self.fills(args.pFormatProperties_mut()) else { return };
        let r = self
            .driver
            .pd_query_arg(pd, format, out, |i| i.try_vkGetPhysicalDeviceFormatProperties());
        self.asked(r);
    }

    fn vkGetPhysicalDeviceMultisamplePropertiesEXT(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceMultisamplePropertiesEXT<'_>,
    ) {
        let (pd, samples) = (args.physicalDevice, args.samples);
        let Some(out) = self.fills(args.pMultisampleProperties_mut()) else { return };
        let r = self.driver.pd_query_arg(pd, samples, out, |i| {
            i.try_vkGetPhysicalDeviceMultisamplePropertiesEXT()
        });
        self.asked(r);
    }

    fn vkGetPhysicalDeviceExternalBufferProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceExternalBufferProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(info) = self.names(args.pExternalBufferInfo) else { return };
        let Some(out) = self.fills(args.pExternalBufferProperties_mut()) else { return };
        let r = self
            .driver
            .pd_query_info(pd, info, out, |i| i.try_vkGetPhysicalDeviceExternalBufferProperties());
        self.asked(r);
    }

    fn vkGetBufferMemoryRequirements(
        &mut self,
        args: &mut vn_command_vkGetBufferMemoryRequirements<'_>,
    ) {
        let (device, buffer) = (args.device, args.buffer);
        let Some(out) = self.fills(args.pMemoryRequirements_mut()) else { return };
        let r = self
            .driver
            .dev_query_arg(device, buffer, out, |d| d.try_vkGetBufferMemoryRequirements());
        self.asked(r);
    }

    fn vkGetImageMemoryRequirements(
        &mut self,
        args: &mut vn_command_vkGetImageMemoryRequirements<'_>,
    ) {
        let (device, image) = (args.device, args.image);
        let Some(out) = self.fills(args.pMemoryRequirements_mut()) else { return };
        let r =
            self.driver.dev_query_arg(device, image, out, |d| d.try_vkGetImageMemoryRequirements());
        self.asked(r);
    }

    fn vkGetDeviceMemoryCommitment(
        &mut self,
        args: &mut vn_command_vkGetDeviceMemoryCommitment<'_>,
    ) {
        let (device, memory) = (args.device, args.memory);
        let Some(out) = self.fills(args.pCommittedMemoryInBytes_mut()) else { return };
        let r =
            self.driver.dev_query_arg(device, memory, out, |d| d.try_vkGetDeviceMemoryCommitment());
        self.asked(r);
    }

    fn vkGetRenderAreaGranularity(&mut self, args: &mut vn_command_vkGetRenderAreaGranularity<'_>) {
        let (device, pass) = (args.device, args.renderPass);
        let Some(out) = self.fills(args.pGranularity_mut()) else { return };
        let r =
            self.driver.dev_query_arg(device, pass, out, |d| d.try_vkGetRenderAreaGranularity());
        self.asked(r);
    }

    fn vkGetRenderingAreaGranularity(
        &mut self,
        args: &mut vn_command_vkGetRenderingAreaGranularity<'_>,
    ) {
        let device = args.device;
        let Some(info) = self.names(args.pRenderingAreaInfo) else { return };
        let Some(out) = self.fills(args.pGranularity_mut()) else { return };
        let r = self
            .driver
            .dev_query_info(device, info, out, |d| d.try_vkGetRenderingAreaGranularity());
        self.asked(r);
    }

    fn vkGetDeviceBufferMemoryRequirements(
        &mut self,
        args: &mut vn_command_vkGetDeviceBufferMemoryRequirements<'_>,
    ) {
        let device = args.device;
        let Some(info) = self.names(args.pInfo) else { return };
        let Some(out) = self.fills(args.pMemoryRequirements_mut()) else { return };
        let r = self
            .driver
            .dev_query_info(device, info, out, |d| d.try_vkGetDeviceBufferMemoryRequirements());
        self.asked(r);
    }

    fn vkGetDeviceImageMemoryRequirements(
        &mut self,
        args: &mut vn_command_vkGetDeviceImageMemoryRequirements<'_>,
    ) {
        let device = args.device;
        let Some(info) = self.names(args.pInfo) else { return };
        let Some(out) = self.fills(args.pMemoryRequirements_mut()) else { return };
        let r = self
            .driver
            .dev_query_info(device, info, out, |d| d.try_vkGetDeviceImageMemoryRequirements());
        self.asked(r);
    }

    fn vkGetDescriptorSetLayoutSupport(
        &mut self,
        args: &mut vn_command_vkGetDescriptorSetLayoutSupport<'_>,
    ) {
        let device = args.device;
        let Some(info) = self.names(args.pCreateInfo) else { return };
        let Some(out) = self.fills(args.pSupport_mut()) else { return };
        let r = self
            .driver
            .dev_query_info(device, info, out, |d| d.try_vkGetDescriptorSetLayoutSupport());
        self.asked(r);
    }

    fn vkGetDeviceImageSubresourceLayout(
        &mut self,
        args: &mut vn_command_vkGetDeviceImageSubresourceLayout<'_>,
    ) {
        let device = args.device;
        let Some(info) = self.names(args.pInfo) else { return };
        let Some(out) = self.fills(args.pLayout_mut()) else { return };
        let r = self
            .driver
            .dev_query_info(device, info, out, |d| d.try_vkGetDeviceImageSubresourceLayout());
        self.asked(r);
    }

    fn vkGetImageSubresourceLayout2(
        &mut self,
        args: &mut vn_command_vkGetImageSubresourceLayout2<'_>,
    ) {
        let (device, image) = (args.device, args.image);
        let Some(sub) = self.names(args.pSubresource) else { return };
        let Some(out) = self.fills(args.pLayout_mut()) else { return };
        let r = self
            .driver
            .dev_query_arg_info(device, image, sub, out, |d| d.try_vkGetImageSubresourceLayout2());
        self.asked(r);
    }

    /// Probe what the driver will do with one combination of format, type, tiling and usage.
    ///
    /// The one query in this section whose failure is an *answer*.
    /// `VK_ERROR_FORMAT_NOT_SUPPORTED` is how a driver says no to a probe, and a guest walks
    /// a table of formats collecting exactly that -- so it goes back in `ret` like any other
    /// result. What is refused is the other thing: this renderer unable to put the question at
    /// all, which leaves the properties struct as the guest sent it and no way to say so.
    fn vkGetPhysicalDeviceImageFormatProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceImageFormatProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        let (format, ty, tiling) = (args.format, args.r#type, args.tiling);
        let (usage, flags) = (args.usage, args.flags);
        let Some(out) = self.fills(args.pImageFormatProperties_mut()) else { return };
        let r =
            self.driver.image_format_properties(pd, format, ty, tiling, usage, flags, out, |i| {
                i.try_vkGetPhysicalDeviceImageFormatProperties()
            });
        if let Some(ret) = self.asked(r) {
            args.ret = ret;
        }
    }

    fn vkGetDeviceGroupPeerMemoryFeatures(
        &mut self,
        args: &mut vn_command_vkGetDeviceGroupPeerMemoryFeatures<'_>,
    ) {
        let device = args.device;
        let (heap, local, remote) = (args.heapIndex, args.localDeviceIndex, args.remoteDeviceIndex);
        let Some(out) = self.fills(args.pPeerMemoryFeatures_mut()) else { return };
        let r = self.driver.peer_memory_features(device, heap, local, remote, out, |d| {
            d.try_vkGetDeviceGroupPeerMemoryFeatures()
        });
        self.asked(r);
    }

    /// The three queries whose whole answer is the number they return.
    ///
    /// No out-struct, so nothing here can be half-filled -- and no `VkResult` either, so
    /// nothing can carry a refusal. A driver this renderer could not ask leaves `ret` at zero,
    /// and zero is a null address the guest would hand to the GPU. See
    /// [`Driver::dev_ask_info`]: the only honest thing left is to stop the ring.
    fn vkGetBufferDeviceAddress(&mut self, args: &mut vn_command_vkGetBufferDeviceAddress<'_>) {
        let Some(info) = self.names(args.pInfo) else { return };
        let r = self.driver.dev_ask_info(args.device, info, |d| d.try_vkGetBufferDeviceAddress());
        if let Some(ret) = self.asked(r) {
            args.ret = ret;
        }
    }

    fn vkGetBufferOpaqueCaptureAddress(
        &mut self,
        args: &mut vn_command_vkGetBufferOpaqueCaptureAddress<'_>,
    ) {
        let Some(info) = self.names(args.pInfo) else { return };
        let r = self
            .driver
            .dev_ask_info(args.device, info, |d| d.try_vkGetBufferOpaqueCaptureAddress());
        if let Some(ret) = self.asked(r) {
            args.ret = ret;
        }
    }

    fn vkGetDeviceMemoryOpaqueCaptureAddress(
        &mut self,
        args: &mut vn_command_vkGetDeviceMemoryOpaqueCaptureAddress<'_>,
    ) {
        let Some(info) = self.names(args.pInfo) else { return };
        let r = self
            .driver
            .dev_ask_info(args.device, info, |d| d.try_vkGetDeviceMemoryOpaqueCaptureAddress());
        if let Some(ret) = self.asked(r) {
            args.ret = ret;
        }
    }

    // ------------------------------------------------------ the MESA resource queries
    //
    // venus inventions rather than Vulkan, and the difference shows in what they name: a
    // virtio-gpu resource id, not a Vulkan handle. The object table cannot resolve one -- it keys
    // Vulkan objects -- so the id crosses into the resource table instead, and the two failures
    // that live on the other side of that crossing are not the same failure.

    fn vkGetMemoryResourcePropertiesMESA(
        &mut self,
        args: &mut vn_command_vkGetMemoryResourcePropertiesMESA<'_>,
    ) {
        let (device, named) = (args.device, args.resourceId);
        let Some(out) = self.fills(args.pMemoryResourceProperties_mut()) else { return };

        // A resource id that names nothing is answered, never refused, and this is the one place
        // in the query family where that is true. The C says why (vkr_device_memory.c:945): a
        // dead id is a reachable runtime state -- a fire-and-forget CREATE_BLOB that failed --
        // and it is the one thing here a guest can arrange from outside. Poisoning the ring for
        // it aborts the whole guest process, because mesa's `vn_relax` aborts on the fatal bit.
        //
        // `shm` folds "no such resource" and "not host-addressable" into one answer, and folding
        // them is right here: Vulkan has one error for both, and it is the same error.
        //
        // Zero is folded in with them: it cannot become a [`ResourceHandle`], and a guest that
        // sends one has named a resource that is not there by the shortest route.
        // `bytes` is the same resolution `vkAllocateMemory` performs on the resource the guest
        // will name next, and the span below is the same span the import will alias. Two answers
        // to one question is what the C's own comment records losing: a query that said a buffer
        // was importable, and an allocation that then refused it.
        let Some(bytes) =
            ResourceHandle::new(named).and_then(|r| self.resources.bytes(self.ctx, r))
        else {
            args.ret = VkResult::VK_ERROR_INVALID_EXTERNAL_HANDLE;
            return;
        };
        let Some(span) = self.driver.span(&bytes) else {
            args.ret = VkResult::VK_ERROR_INVALID_EXTERNAL_HANDLE;
            return;
        };

        // A device this renderer has no table for is the other kind of failure: not the guest's
        // state, ours. It goes back through the query family's own refusal rather than a second
        // copy of its wording -- there is one policy here, so there is one place that states it.
        let asked = self.driver.host_visible_memory_types(device);
        let Some(bits) = self.asked(asked) else { return };

        // The answer has to agree with what the allocation path will do with the same resource --
        // see `Driver::host_visible_memory_types`. A guest reads this, intersects it with the
        // image's own requirements, and hands the result straight back as a `memoryTypeIndex`.
        out.memoryTypeBits = bits;
        if let Some(size) =
            driver::chained_mut::<VkMemoryResourceAllocationSizePropertiesMESA>(&mut out.pNext)
        {
            size.allocationSize = match &bytes {
                ResourceBytes::Host(map) => map.len() as u64,
                // The span the import will alias, reported as the size the guest may allocate
                // over it -- literally the same number, so the query cannot promise an extent
                // the allocation then clamps away.
                ResourceBytes::Shared(_) => span.1,
                ResourceBytes::Allocation(published) => published.size,
            };
        }
        args.ret = VkResult::VK_SUCCESS;
    }

    // --------------------------------------------------------- the two-call enumerations
    //
    // Vulkan's count-then-fill idiom, eight commands of it. The guest calls once with a null
    // array to be told how many there are, then again with an array that size; both calls arrive
    // here as the same command, told apart by whether `pXProperties` was sent.
    //
    // Two shapes, and the difference is not cosmetic. The ones with a `ret` can say
    // `VK_INCOMPLETE` -- the driver had more than the guest sized for, which is the guest's
    // business rather than an error. The void ones have no field to say it in: the driver simply
    // writes what fits and lowers the count, and that lowered count is the whole of what the
    // guest is told. Do not look for an `args.ret` in those; there is nowhere to put one.
    //
    // The count goes back last everywhere, and has to: it lives in the struct the array was
    // borrowed from, so the two cannot be held at once. See `vkGetPhysicalDeviceQueueFamilyProperties2`.

    fn vkGetPhysicalDeviceQueueFamilyProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceQueueFamilyProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        if !self.counted(args.has_pQueueFamilyPropertyCount()) {
            return;
        }
        if !args.has_pQueueFamilyProperties() {
            let asked = self
                .driver
                .enumerate_into(pd, None, |i| i.try_vkGetPhysicalDeviceQueueFamilyProperties());
            if let Some((n, ())) = self.asked(asked)
                && let Some(count) = args.pQueueFamilyPropertyCount_mut()
            {
                *count = n;
            }
            return;
        }
        let Some(out) = self.array(args.pQueueFamilyProperties_mut()) else { return };
        let asked = self
            .driver
            .enumerate_into(pd, Some(out), |i| i.try_vkGetPhysicalDeviceQueueFamilyProperties());
        if let Some((n, ())) = self.asked(asked)
            && let Some(count) = args.pQueueFamilyPropertyCount_mut()
        {
            *count = n;
        }
    }

    fn vkGetPhysicalDeviceToolProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceToolProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        if !self.counted(args.has_pToolCount()) {
            return;
        }
        if !args.has_pToolProperties() {
            let asked =
                self.driver.enumerate_into(pd, None, |i| i.try_vkGetPhysicalDeviceToolProperties());
            match asked {
                Ok((n, ret)) => {
                    args.ret = ret;
                    if let Some(count) = args.pToolCount_mut() {
                        *count = n;
                    }
                }
                Err(e) => args.ret = e,
            }
            return;
        }
        let Some(out) = self.array(args.pToolProperties_mut()) else { return };
        let asked = self
            .driver
            .enumerate_into(pd, Some(out), |i| i.try_vkGetPhysicalDeviceToolProperties());
        match asked {
            Ok((n, ret)) => {
                args.ret = ret;
                if let Some(count) = args.pToolCount_mut() {
                    *count = n;
                }
            }
            Err(e) => args.ret = e,
        }
    }

    fn vkGetPhysicalDeviceCalibrateableTimeDomainsKHR(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR<'_>,
    ) {
        let pd = args.physicalDevice;
        if !self.counted(args.has_pTimeDomainCount()) {
            return;
        }
        if !args.has_pTimeDomains() {
            let asked = self.driver.enumerate_into(pd, None, |i| {
                i.try_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR()
            });
            match asked {
                Ok((n, ret)) => {
                    args.ret = ret;
                    if let Some(count) = args.pTimeDomainCount_mut() {
                        *count = n;
                    }
                }
                Err(e) => args.ret = e,
            }
            return;
        }
        let Some(out) = self.array(args.pTimeDomains_mut()) else { return };
        let asked = self.driver.enumerate_into(pd, Some(out), |i| {
            i.try_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR()
        });
        match asked {
            Ok((n, ret)) => {
                args.ret = ret;
                if let Some(count) = args.pTimeDomainCount_mut() {
                    *count = n;
                }
            }
            Err(e) => args.ret = e,
        }
    }

    fn vkGetPhysicalDeviceSparseImageFormatProperties(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceSparseImageFormatProperties<'_>,
    ) {
        let pd = args.physicalDevice;
        let (format, ty, samples) = (args.format, args.r#type, args.samples);
        let (usage, tiling) = (args.usage, args.tiling);
        if !self.counted(args.has_pPropertyCount()) {
            return;
        }
        let out = if args.has_pProperties() {
            match self.array(args.pProperties_mut()) {
                Some(out) => Some(out),
                None => return,
            }
        } else {
            None
        };
        let asked = self.driver.sparse_format_properties(
            pd,
            format,
            ty,
            samples,
            usage,
            tiling,
            out,
            |i| i.try_vkGetPhysicalDeviceSparseImageFormatProperties(),
        );
        if let Some(n) = self.asked(asked)
            && let Some(count) = args.pPropertyCount_mut()
        {
            *count = n;
        }
    }

    fn vkGetPhysicalDeviceSparseImageFormatProperties2(
        &mut self,
        args: &mut vn_command_vkGetPhysicalDeviceSparseImageFormatProperties2<'_>,
    ) {
        let pd = args.physicalDevice;
        let Some(info) = args.pFormatInfo else {
            self.reject = Some("asked which formats are sparse without naming one");
            return;
        };
        if !self.counted(args.has_pPropertyCount()) {
            return;
        }
        let out = if args.has_pProperties() {
            match self.array(args.pProperties_mut()) {
                Some(out) => Some(out),
                None => return,
            }
        } else {
            None
        };
        let asked = self.driver.enumerate_info_into(pd, info, out, |i| {
            i.try_vkGetPhysicalDeviceSparseImageFormatProperties2()
        });
        if let Some((n, ())) = self.asked(asked)
            && let Some(count) = args.pPropertyCount_mut()
        {
            *count = n;
        }
    }

    fn vkGetImageSparseMemoryRequirements(
        &mut self,
        args: &mut vn_command_vkGetImageSparseMemoryRequirements<'_>,
    ) {
        let (device, image) = (args.device, args.image);
        if !self.counted(args.has_pSparseMemoryRequirementCount()) {
            return;
        }
        let out = if args.has_pSparseMemoryRequirements() {
            match self.array(args.pSparseMemoryRequirements_mut()) {
                Some(out) => Some(out),
                None => return,
            }
        } else {
            None
        };
        let asked = self
            .driver
            .dev_enumerate_arg(device, image, out, |d| d.try_vkGetImageSparseMemoryRequirements());
        if let Some((n, ())) = self.asked(asked)
            && let Some(count) = args.pSparseMemoryRequirementCount_mut()
        {
            *count = n;
        }
    }

    fn vkGetImageSparseMemoryRequirements2(
        &mut self,
        args: &mut vn_command_vkGetImageSparseMemoryRequirements2<'_>,
    ) {
        let device = args.device;
        let Some(info) = args.pInfo else {
            self.reject = Some("asked an image's sparse requirements without naming the image");
            return;
        };
        if !self.counted(args.has_pSparseMemoryRequirementCount()) {
            return;
        }
        let out = if args.has_pSparseMemoryRequirements() {
            match self.array(args.pSparseMemoryRequirements_mut()) {
                Some(out) => Some(out),
                None => return,
            }
        } else {
            None
        };
        let asked = self
            .driver
            .dev_enumerate_info(device, info, out, |d| d.try_vkGetImageSparseMemoryRequirements2());
        if let Some((n, ())) = self.asked(asked)
            && let Some(count) = args.pSparseMemoryRequirementCount_mut()
        {
            *count = n;
        }
    }

    fn vkGetDeviceImageSparseMemoryRequirements(
        &mut self,
        args: &mut vn_command_vkGetDeviceImageSparseMemoryRequirements<'_>,
    ) {
        let device = args.device;
        let Some(info) = args.pInfo else {
            self.reject =
                Some("asked an unbuilt image's sparse requirements without describing it");
            return;
        };
        if !self.counted(args.has_pSparseMemoryRequirementCount()) {
            return;
        }
        let out = if args.has_pSparseMemoryRequirements() {
            match self.array(args.pSparseMemoryRequirements_mut()) {
                Some(out) => Some(out),
                None => return,
            }
        } else {
            None
        };
        let asked = self.driver.dev_enumerate_info(device, info, out, |d| {
            d.try_vkGetDeviceImageSparseMemoryRequirements()
        });
        if let Some((n, ())) = self.asked(asked)
            && let Some(count) = args.pSparseMemoryRequirementCount_mut()
        {
            *count = n;
        }
    }

    // ------------------------------------------------------------------------ pipelines
    //
    // Not a `simple_create`: one command makes a run of them, and it is the only create that can
    // come back part real. What that costs is in [`Driver::create_pipelines`]; what is left here
    // is the all-or-nothing the guest sees.

    fn vkCreateGraphicsPipelines(&mut self, args: &mut vn_command_vkCreateGraphicsPipelines<'_>) {
        let infos = args.pCreateInfos();
        let ids = args.pPipelines();
        // Read before the shadow is borrowed: see `vkEnumeratePhysicalDevices`.
        let (device, cache, alloc) = (args.device, args.pipelineCache, args.pAllocator);
        let out = args.handle_pPipelines_mut();
        let host = self.driver.create_pipelines(
            device,
            |d| d.vkCreateGraphicsPipelines(),
            cache,
            infos,
            alloc,
            out,
        );
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        if host.is_err() {
            // A pipeline the host refuses is a shader the guest cannot draw with, which is worth
            // saying out loud -- unlike a descriptor pool running dry, it is not something a
            // working guest does on purpose.
            eprintln!("[virglrs] vkCreateGraphicsPipelines refused by the driver");
            self.ghost_ids(ids);
        }
    }

    simple_destroy!(vkDestroyPipeline, vn_command_vkDestroyPipeline, pipeline);

    // -------------------------------------------------------------- binding and updating
    //
    // Nothing here creates or destroys an object, so nothing here plants a handle. They are served
    // because a later `vkQueueSubmit` is only safe to pass through once the objects it names are
    // fully built: an image with no memory bound and a descriptor set that was never written are
    // undefined behaviour at draw time, not errors the driver reports.

    fn vkBindBufferMemory2(&mut self, args: &mut vn_command_vkBindBufferMemory2<'_>) {
        let infos = args.pBindInfos();
        args.ret = self.driver.counted_op(args.device, |d| d.vkBindBufferMemory2(), infos);
    }

    fn vkBindImageMemory2(&mut self, args: &mut vn_command_vkBindImageMemory2<'_>) {
        let infos = args.pBindInfos();
        args.ret = self.driver.counted_op(args.device, |d| d.vkBindImageMemory2(), infos);
    }

    fn vkBindBufferMemory(&mut self, args: &mut vn_command_vkBindBufferMemory<'_>) {
        args.ret = self.driver.bind_one(
            args.device,
            |d| d.vkBindBufferMemory(),
            args.buffer,
            args.memory,
            args.memoryOffset,
        );
    }

    fn vkBindImageMemory(&mut self, args: &mut vn_command_vkBindImageMemory<'_>) {
        args.ret = self.driver.bind_one(
            args.device,
            |d| d.vkBindImageMemory(),
            args.image,
            args.memory,
            args.memoryOffset,
        );
    }

    /// Publish host writes to a mapped range, and pick up the driver's.
    ///
    /// Both are no-ops on coherent memory, which is what this renderer hands out today -- but a
    /// guest is entitled to call them and a guest that does is not wrong. Serving them costs a
    /// forwarded call; refusing them would stop a ring over a command that has nothing to fail.
    fn vkFlushMappedMemoryRanges(&mut self, args: &mut vn_command_vkFlushMappedMemoryRanges<'_>) {
        let ranges = args.pMemoryRanges();
        args.ret = self.driver.counted_op(args.device, |d| d.vkFlushMappedMemoryRanges(), ranges);
    }

    fn vkInvalidateMappedMemoryRanges(
        &mut self,
        args: &mut vn_command_vkInvalidateMappedMemoryRanges<'_>,
    ) {
        let ranges = args.pMemoryRanges();
        args.ret =
            self.driver.counted_op(args.device, |d| d.vkInvalidateMappedMemoryRanges(), ranges);
    }

    fn vkUpdateDescriptorSets(&mut self, args: &mut vn_command_vkUpdateDescriptorSets<'_>) {
        let writes = args.pDescriptorWrites();
        let copies = args.pDescriptorCopies();
        self.driver.update_descriptor_sets(args.device, writes, copies);
    }

    // ---------------------------------------------------------------------- recording
    //
    // What a frame is made of. Every one of these takes a command buffer and records into it,
    // and Vulkan defers what could go wrong to the submit -- so with the single exception of
    // `vkBeginCommandBuffer`, which can run the pool dry, none of them has an answer to give.
    //
    // None carries a device: a command buffer knows its own, and the driver walks back to it
    // through the pool. Nor does any of them check that the command buffer is still alive. The
    // object table is that check -- a destroyed pool takes its buffers' ids out of it, so a guest
    // naming one stops its own ring at the lookup, before a handler is reached.

    fn vkBeginCommandBuffer(&mut self, args: &mut vn_command_vkBeginCommandBuffer<'_>) {
        let Some(info) = self.names(args.pBeginInfo) else { return };
        let Some(ret) = self.driver.begin_command_buffer(args.commandBuffer, info) else {
            return self.no_recorder();
        };
        args.ret = ret;
    }

    fn vkEndCommandBuffer(&mut self, args: &mut vn_command_vkEndCommandBuffer<'_>) {
        let Some(ret) = self.driver.end_command_buffer(args.commandBuffer) else {
            return self.no_recorder();
        };
        args.ret = ret;
    }

    fn vkResetCommandBuffer(&mut self, args: &mut vn_command_vkResetCommandBuffer<'_>) {
        let Some(ret) = self.driver.reset_command_buffer(args.commandBuffer, args.flags) else {
            return self.no_recorder();
        };
        args.ret = ret;
    }

    fn vkCmdPipelineBarrier(&mut self, args: &mut vn_command_vkCmdPipelineBarrier<'_>) {
        // Three independent arrays, each optional: a barrier may name memory, buffers, images,
        // or any mix. An absent one is the guest barring nothing of that kind, not a violation.
        let memory = args.pMemoryBarriers();
        let buffers = args.pBufferMemoryBarriers();
        let images = args.pImageMemoryBarriers();
        let done = self.driver.cmd_pipeline_barrier(
            args.commandBuffer,
            args.srcStageMask,
            args.dstStageMask,
            args.dependencyFlags,
            memory,
            buffers,
            images,
        );
        self.recorded(done);
    }

    fn vkCmdBeginRenderPass(&mut self, args: &mut vn_command_vkCmdBeginRenderPass<'_>) {
        let Some(begin) = self.names(args.pRenderPassBegin) else { return };
        let done = self.driver.cmd_begin_render_pass(args.commandBuffer, begin, args.contents);
        self.recorded(done);
    }

    fn vkCmdEndRenderPass(&mut self, args: &mut vn_command_vkCmdEndRenderPass<'_>) {
        let done = self.driver.cmd_end_render_pass(args.commandBuffer);
        self.recorded(done);
    }

    fn vkCmdBindPipeline(&mut self, args: &mut vn_command_vkCmdBindPipeline<'_>) {
        let done = self.driver.cmd_bind_pipeline(
            args.commandBuffer,
            args.pipelineBindPoint,
            args.pipeline,
        );
        self.recorded(done);
    }

    fn vkCmdBindDescriptorSets(&mut self, args: &mut vn_command_vkCmdBindDescriptorSets<'_>) {
        // The sets are what the command is for, so counting some and sending none is a violation.
        // The dynamic offsets are their own array with their own count, and a pipeline layout
        // with no dynamic descriptors legitimately binds none.
        let sets = args.pDescriptorSets();
        let offsets = args.pDynamicOffsets();
        let done = self.driver.cmd_bind_descriptor_sets(
            args.commandBuffer,
            args.pipelineBindPoint,
            args.layout,
            args.firstSet,
            sets,
            offsets,
        );
        self.recorded(done);
    }

    fn vkCmdDraw(&mut self, args: &mut vn_command_vkCmdDraw<'_>) {
        let done = self.driver.cmd_draw(
            args.commandBuffer,
            args.vertexCount,
            args.instanceCount,
            args.firstVertex,
            args.firstInstance,
        );
        self.recorded(done);
    }

    fn vkCmdSetViewport(&mut self, args: &mut vn_command_vkCmdSetViewport<'_>) {
        let viewports = args.pViewports();
        let done = self.driver.cmd_set_viewport(args.commandBuffer, args.firstViewport, viewports);
        self.recorded(done);
    }

    fn vkCmdSetScissor(&mut self, args: &mut vn_command_vkCmdSetScissor<'_>) {
        let scissors = args.pScissors();
        let done = self.driver.cmd_set_scissor(args.commandBuffer, args.firstScissor, scissors);
        self.recorded(done);
    }

    fn vkCmdBindVertexBuffers(&mut self, args: &mut vn_command_vkCmdBindVertexBuffers<'_>) {
        // One count, two arrays. Both accessors read that same count, so the two slices are the
        // same length by construction -- the driver asserts it rather than trusting the pair.
        let buffers = args.pBuffers();
        let offsets = args.pOffsets();
        let done = self.driver.cmd_bind_vertex_buffers(
            args.commandBuffer,
            args.firstBinding,
            buffers,
            offsets,
        );
        self.recorded(done);
    }

    fn vkCmdFillBuffer(&mut self, args: &mut vn_command_vkCmdFillBuffer<'_>) {
        let done = self.driver.cmd_fill_buffer(
            args.commandBuffer,
            args.dstBuffer,
            args.dstOffset,
            args.size,
            args.data,
        );
        self.recorded(done);
    }

    fn vkCmdCopyBuffer(&mut self, args: &mut vn_command_vkCmdCopyBuffer<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_buffer(
            args.commandBuffer,
            args.srcBuffer,
            args.dstBuffer,
            regions,
        );
        self.recorded(done);
    }

    fn vkCmdCopyBufferToImage(&mut self, args: &mut vn_command_vkCmdCopyBufferToImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_buffer_to_image(
            args.commandBuffer,
            args.srcBuffer,
            args.dstImage,
            args.dstImageLayout,
            regions,
        );
        self.recorded(done);
    }

    fn vkCmdCopyImageToBuffer(&mut self, args: &mut vn_command_vkCmdCopyImageToBuffer<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_image_to_buffer(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstBuffer,
            regions,
        );
        self.recorded(done);
    }

    fn vkCmdCopyImage(&mut self, args: &mut vn_command_vkCmdCopyImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_image(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstImage,
            args.dstImageLayout,
            regions,
        );
        self.recorded(done);
    }

    fn vkCmdBlitImage(&mut self, args: &mut vn_command_vkCmdBlitImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_blit_image(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstImage,
            args.dstImageLayout,
            regions,
            args.filter,
        );
        self.recorded(done);
    }

    fn vkCmdClearColorImage(&mut self, args: &mut vn_command_vkCmdClearColorImage<'_>) {
        // The colour is what the clear is for: without it there is no defensible value to write,
        // and picking one would be inventing guest intent.
        let Some(color) = self.names(args.pColor) else { return };
        let ranges = args.pRanges();
        let done = self.driver.cmd_clear_color_image(
            args.commandBuffer,
            args.image,
            args.imageLayout,
            color,
            ranges,
        );
        self.recorded(done);
    }

    fn vkCmdClearAttachments(&mut self, args: &mut vn_command_vkCmdClearAttachments<'_>) {
        // Two counts over two arrays, cleared as a product. Either being empty clears nothing,
        // which is legal and is the guest's business rather than a violation.
        let attachments = args.pAttachments();
        let rects = args.pRects();
        let done = self.driver.cmd_clear_attachments(args.commandBuffer, attachments, rects);
        self.recorded(done);
    }

    fn vkCmdPushConstants(&mut self, args: &mut vn_command_vkCmdPushConstants<'_>) {
        // The bytes are the command. A null `pValues` with a non-zero `size` is the one shape the
        // decoder cannot resolve into a slice, and pushing whatever the layout last held would
        // hand the next draw constants the guest never sent.
        let Some(values) = args.pValues() else {
            self.reject = Some("pushed constants without saying what they are");
            return;
        };
        let done = self.driver.cmd_push_constants(
            args.commandBuffer,
            args.layout,
            args.stageFlags,
            args.offset,
            values,
        );
        self.recorded(done);
    }

    fn vkCmdBeginQuery(&mut self, args: &mut vn_command_vkCmdBeginQuery<'_>) {
        let done =
            self.driver.cmd_begin_query(args.commandBuffer, args.queryPool, args.query, args.flags);
        self.queried(done);
    }

    fn vkCmdEndQuery(&mut self, args: &mut vn_command_vkCmdEndQuery<'_>) {
        let done = self.driver.cmd_end_query(args.commandBuffer, args.queryPool, args.query);
        self.queried(done);
    }

    fn vkCmdResetQueryPool(&mut self, args: &mut vn_command_vkCmdResetQueryPool<'_>) {
        let done = self.driver.cmd_reset_query_pool(
            args.commandBuffer,
            args.queryPool,
            args.firstQuery,
            args.queryCount,
        );
        self.queried(done);
    }

    fn vkCmdWriteTimestamp(&mut self, args: &mut vn_command_vkCmdWriteTimestamp<'_>) {
        let done = self.driver.cmd_write_timestamp(
            args.commandBuffer,
            args.pipelineStage,
            args.queryPool,
            args.query,
        );
        self.queried(done);
    }

    fn vkCmdCopyQueryPoolResults(&mut self, args: &mut vn_command_vkCmdCopyQueryPoolResults<'_>) {
        let done = self.driver.cmd_copy_query_pool_results(
            args.commandBuffer,
            args.queryPool,
            args.firstQuery,
            args.queryCount,
            args.dstBuffer,
            args.dstOffset,
            args.stride,
            args.flags,
        );
        self.queried(done);
    }

    // --------------------------------------------------------------------------- sync
    //
    // Where a frame is handed to the GPU and waited for. Everything the recording section built
    // is inert until a submit names it, and everything after a submit is the guest asking whether
    // the work is done yet.

    fn vkQueueSubmit(&mut self, args: &mut vn_command_vkQueueSubmit<'_>) {
        // Submitting nothing is legal -- it is how a guest signals a fence with no work -- so the
        // empty slice goes through rather than being turned away.
        let submits = args.pSubmits();
        let Some(ret) = self.driver.queue_submit(args.queue, submits, args.fence) else {
            self.reject = Some("submitted to a queue with no device behind it");
            return;
        };
        args.ret = ret;
    }

    fn vkResetFences(&mut self, args: &mut vn_command_vkResetFences<'_>) {
        let fences = args.pFences();
        args.ret = self.driver.reset_fences(args.device, fences);
    }

    /// Blocks the caller for as long as the guest asked, up to forever. That is the guest's own
    /// thread being spent on the guest's own wait; answering early would be answering wrongly.
    fn vkWaitForFences(&mut self, args: &mut vn_command_vkWaitForFences<'_>) {
        let fences = args.pFences();
        args.ret = self.driver.wait_for_fences(args.device, fences, args.waitAll, args.timeout);
    }

    /// Wait for everything on a device, or on one queue, to finish.
    ///
    /// Blocking, like `vkWaitForFences` above: the guest's own thread is what is being spent, and
    /// answering before the driver is idle would be answering wrongly.
    fn vkDeviceWaitIdle(&mut self, args: &mut vn_command_vkDeviceWaitIdle<'_>) {
        args.ret = self.driver.device_op(args.device, |d| d.vkDeviceWaitIdle());
    }

    fn vkQueueWaitIdle(&mut self, args: &mut vn_command_vkQueueWaitIdle<'_>) {
        let Some(ret) = self.driver.queue_op(args.queue, |d| d.vkQueueWaitIdle()) else {
            self.reject = Some("waited on a queue with no device behind it");
            return;
        };
        args.ret = ret;
    }

    /// The event and fence states, and the two commands that set an event from the host side.
    ///
    /// `ret` here is the answer, not an error code: `VK_EVENT_SET`, `VK_EVENT_RESET`,
    /// `VK_NOT_READY` and `VK_SUCCESS` are all ordinary outcomes the guest is asking about. What
    /// makes forwarding them safe is that a device this renderer does not have answers
    /// `VK_ERROR_INITIALIZATION_FAILED` rather than any of those -- an error the guest can act on
    /// instead of a state it would believe.
    fn vkGetEventStatus(&mut self, args: &mut vn_command_vkGetEventStatus<'_>) {
        args.ret = self.driver.object_op(args.device, |d| d.vkGetEventStatus(), args.event);
    }

    fn vkSetEvent(&mut self, args: &mut vn_command_vkSetEvent<'_>) {
        args.ret = self.driver.object_op(args.device, |d| d.vkSetEvent(), args.event);
    }

    fn vkResetEvent(&mut self, args: &mut vn_command_vkResetEvent<'_>) {
        args.ret = self.driver.object_op(args.device, |d| d.vkResetEvent(), args.event);
    }

    fn vkGetFenceStatus(&mut self, args: &mut vn_command_vkGetFenceStatus<'_>) {
        args.ret = self.driver.object_op(args.device, |d| d.vkGetFenceStatus(), args.fence);
    }

    // ----------------------------------------------------------------- timeline semaphores
    //
    // A binary semaphore is only ever waited on inside a submit, so the host never sees one from
    // this side. A timeline semaphore has a counter the guest can read, raise and block on
    // directly, and these three are how it does that.
    //
    // All three are forwarded with nothing added. The handles inside `VkSemaphoreWaitInfo` and
    // `VkSemaphoreSignalInfo` are already host handles: the generated decoder resolves each one
    // through the object table as it reads it, so a guest naming a semaphore it does not own stops
    // its own ring before any handler is reached. `pSemaphores` and `pValues` are likewise already
    // reconciled against `semaphoreCount` there -- a guest that disagrees with itself about how
    // many it sent is fatal at the decode -- so there is no pair left here to check.

    /// Read a timeline semaphore's counter.
    ///
    /// A query: the answer is the `u64` the driver writes, so a driver that cannot be asked leaves
    /// the guest's own value in place and the ring stops rather than encode it back as an answer.
    fn vkGetSemaphoreCounterValue(&mut self, args: &mut vn_command_vkGetSemaphoreCounterValue<'_>) {
        let (device, semaphore) = (args.device, args.semaphore);
        let Some(out) = self.fills(args.pValue_mut()) else { return };
        let r = self
            .driver
            .dev_query_arg(device, semaphore, out, |d| d.try_vkGetSemaphoreCounterValue());
        if let Some(ret) = self.asked(r) {
            args.ret = ret;
        }
    }

    /// Raise a timeline semaphore's counter from the host side.
    fn vkSignalSemaphore(&mut self, args: &mut vn_command_vkSignalSemaphore<'_>) {
        let Some(info) = args.pSignalInfo else {
            self.reject = Some("signalled a semaphore it did not name");
            return;
        };
        args.ret = self.driver.dev_op_info(args.device, info, |d| d.try_vkSignalSemaphore());
    }

    /// Block until a set of timeline semaphores reaches the values the guest named.
    ///
    /// Blocking, like `vkWaitForFences` above, and for the same reason: the guest asked to wait
    /// and it is the guest's thread being spent. The timeout crosses untouched -- see
    /// [`Driver::dev_op_info_timeout`] for why shortening it would be worse than blocking.
    fn vkWaitSemaphores(&mut self, args: &mut vn_command_vkWaitSemaphores<'_>) {
        let Some(info) = args.pWaitInfo else {
            self.reject = Some("waited on semaphores it did not name");
            return;
        };
        args.ret = self
            .driver
            .dev_op_info_timeout(args.device, info, args.timeout, |d| d.try_vkWaitSemaphores());
    }

    /// Recycle everything a pool handed out, without destroying the pool or the objects.
    ///
    /// The object table is deliberately left alone. A reset does not free the command buffers or
    /// descriptor sets -- their handles stay valid and the guest goes on naming them -- so
    /// forgetting them here would poison the next command that did, over a reset that was legal.
    /// This is the difference between a reset and the destroy that `pool_destroy!` serves.
    fn vkResetCommandPool(&mut self, args: &mut vn_command_vkResetCommandPool<'_>) {
        args.ret = self.driver.object_flags_op(
            args.device,
            |d| d.vkResetCommandPool(),
            args.commandPool,
            args.flags,
        );
        // The journal's copy of the recycling. Every buffer this pool handed out keeps its handle
        // and its key, so nothing above can tell the recorder that their recordings are gone.
        if args.ret == VkResult::VK_SUCCESS {
            self.note = Some(Note::PoolReset(args.commandPool.host()));
        }
    }

    fn vkResetDescriptorPool(&mut self, args: &mut vn_command_vkResetDescriptorPool<'_>) {
        args.ret = self.driver.object_flags_op(
            args.device,
            |d| d.vkResetDescriptorPool(),
            args.descriptorPool,
            args.flags,
        );
    }

    fn vkWaitSemaphoreResourceMESA(
        &mut self,
        args: &mut vn_command_vkWaitSemaphoreResourceMESA<'_>,
    ) {
        let done = self.driver.export_semaphore_sync_fd(args.device, args.semaphore);
        self.synced("vkWaitSemaphoreResourceMESA", done);
    }

    fn vkImportSemaphoreResourceMESA(
        &mut self,
        args: &mut vn_command_vkImportSemaphoreResourceMESA<'_>,
    ) {
        let Some(info) = args.pImportSemaphoreResourceInfo else {
            self.reject = Some("imported a semaphore payload from no descriptor at all");
            return;
        };
        // The C asserts on this. Here it is the guest's own number, arriving over the wire, so an
        // assert would let a guest abort the process: only id 0 -- an already-signaled payload
        // with no resource behind it -- is a thing this serves, and anything else is rejected.
        if info.resourceId != 0 {
            self.reject = Some("imported a semaphore payload from a resource id");
            return;
        }
        let done = self.driver.import_signaled_semaphore(args.device, info.semaphore);
        self.synced("vkImportSemaphoreResourceMESA", done);
    }
}

impl Unimplemented {
    /// The commands a corpus asked for, most-used first -- the order to implement them in.
    pub fn by_frequency(&self) -> Vec<(&'static str, u64)> {
        let mut v: Vec<_> = self
            .seen
            .iter()
            .map(|(c, n)| (vn_command_name(VkCommandTypeEXT(*c)).unwrap_or("?"), *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }
}

#[cfg(test)]
mod tests {
    /// A resource table with nothing in it, for the tests that are not about rings. A ring
    /// handler reaching for a resource here gets the same answer a guest naming a bogus one does.
    struct NoResources;

    impl ShmResources for NoResources {
        fn shm(
            &self,
            _: crate::ids::ResourceHandle,
        ) -> Option<std::sync::Arc<crate::guest_mem::GuestMap>> {
            None
        }
    }

    static NO_RESOURCES: NoResources = NoResources;

    /// A resource table with one mapped shm resource in it, for the ring tests.
    struct OneShm(crate::ids::ResourceHandle, std::sync::Arc<crate::guest_mem::GuestMap>);

    impl ShmResources for OneShm {
        fn shm(
            &self,
            handle: crate::ids::ResourceHandle,
        ) -> Option<std::sync::Arc<crate::guest_mem::GuestMap>> {
            (handle == self.0).then(|| std::sync::Arc::clone(&self.1))
        }
    }

    const RING_RES: crate::ids::ResourceHandle = crate::ids::ResourceHandle::new(449).unwrap();

    /// A resource minted the way the renderer mints one for a ring, at the size the venus corpus
    /// actually asks for.
    fn ring_table() -> OneShm {
        let (fd, map) =
            crate::guest_mem::anonymous_shm(0x24000, "virglrs-ringtest").expect("minted");
        drop(fd);
        OneShm(RING_RES, std::sync::Arc::new(map))
    }

    /// The layout the venus corpus asks for, give or take: a window with three control words, a
    /// power-of-two buffer and a small extra region.
    fn ring_info() -> crate::venus::proto::types::VkRingCreateInfoMESA {
        crate::venus::proto::types::VkRingCreateInfoMESA {
            resourceId: RING_RES.get(),
            offset: 0,
            size: 0x200c4,
            headOffset: 0,
            tailOffset: 4,
            statusOffset: 8,
            bufferOffset: 0xc0,
            bufferSize: 0x20000,
            extraOffset: 0x200c0,
            extraSize: 4,
            ..Default::default()
        }
    }

    use super::super::proto::types::{
        VkDevice, VkImageAspectFlags, VkImageCreateFlags, VkImageUsageFlags, VkMemoryPropertyFlags,
        VkSemaphoreWaitFlags, VkToolPurposeFlags,
    };
    use super::*;

    fn header(cmd: VkCommandTypeEXT, flags: u32) -> Vec<u8> {
        let mut w = (cmd.0 as u32).to_le_bytes().to_vec();
        w.extend_from_slice(&flags.to_le_bytes());
        w
    }

    /// A command that does not decode -- here, one cut off after its header -- poisons the ring,
    /// and the log line that says so has to be able to name the command. Without the name a gap
    /// reaches a user as a hung guest with nothing to report; `vn_command_name` returning `None`
    /// here would be that silently.
    #[test]
    fn an_undecodable_command_poisons_the_context_by_name() {
        let cmd = VkCommandTypeEXT::VK_COMMAND_TYPE_vkGetPipelineCacheData_EXT;
        assert_eq!(vn_command_name(cmd), Some("vkGetPipelineCacheData"));

        let g = crate::vulkan::global();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();
        let mut todo = Unimplemented::default();
        assert!(
            !ctx.submit(&header(cmd, 0), &mut todo, &g, &NO_RESOURCES).ran(),
            "a command with no body must poison"
        );
        assert!(ctx.fatal());

        // The poison outlives the batch: a stream we stopped trusting stays untrusted.
        assert!(!ctx.submit(&header(cmd, 0), &mut todo, &g, &NO_RESOURCES).ran());
    }

    /// A command that wants an answer has nowhere to be answered into, so it poisons -- but only
    /// outside replay, where the journal's replies have already been stripped.
    #[test]
    fn a_reply_request_poisons_only_outside_replay() {
        let cmd = VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyInstance_EXT;
        let mut todo = Unimplemented::default();
        let g = crate::vulkan::global();

        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        let w = header(cmd, GENERATE_REPLY);
        let mut full = w.clone();
        full.extend_from_slice(&1u64.to_le_bytes()); // instance id
        full.extend_from_slice(&0u64.to_le_bytes()); // no allocator
        assert!(!ctx.submit(&full, &mut todo, &g, &NO_RESOURCES).ran());
        assert_eq!(ctx.unhandled, 1);

        // In replay the flag is stripped, so the command reaches the dispatcher instead of the
        // poison. It still names an instance nothing created, which poisons for its own reason --
        // what separates the two paths is whether the command was dispatched at all.
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();
        assert!(!ctx.submit(&full, &mut todo, &g, &NO_RESOURCES).ran());
        assert_eq!(ctx.dispatched, 1);
        assert_eq!(ctx.unhandled, 0);
    }

    /// The tee runs, and a command that built nothing durable is still counted.
    ///
    /// The narrowest possible check on the wiring: `seq` advances once per dispatched command
    /// whatever the classification, so a zero here means the recorder was never reached at all --
    /// which is a thing that can be true while every command succeeds and every score matches.
    #[test]
    fn a_dispatched_command_reaches_the_recorder() {
        let cmd = VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyInstance_EXT;
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();

        assert_eq!(ctx.journal_seq(), Seq(0), "nothing has gone by yet");

        let mut w = header(cmd, 0);
        w.extend_from_slice(&0u64.to_le_bytes()); // a null instance: legal, and destroys nothing
        w.extend_from_slice(&0u64.to_le_bytes()); // no allocator
        assert!(ctx.submit(&w, &mut todo, &g, &NO_RESOURCES).ran());

        assert_eq!(ctx.dispatched, 1);
        assert_eq!(ctx.journal_seq(), Seq(1), "the recorder saw the command the dispatcher did");
        // It created nothing, so it is retained as nothing -- and named, not merely counted.
        assert_eq!(ctx.journal_transient(), vec![("vkDestroyInstance", 1)]);
        assert!(ctx.journal_export().is_none(), "a journal of nothing exports nothing");
    }

    /// A window in the ring resource, for the reply-stream tests.
    fn reply_at(
        offset: usize,
        size: usize,
    ) -> super::super::proto::types::VkCommandStreamDescriptionMESA {
        super::super::proto::types::VkCommandStreamDescriptionMESA {
            resourceId: RING_RES.get(),
            offset,
            size,
        }
    }

    /// Build one command's bytes with the generator's own encoder.
    ///
    /// Hand-rolling the wire layout in a test would be a second implementation of it, and the one
    /// that drifts silently. This goes through the same encoder the reply oracle checks.
    fn wire_set_reply(
        stream: &super::super::proto::types::VkCommandStreamDescriptionMESA,
    ) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkSetReplyCommandStreamMESA_args, vn_sizeof_vkSetReplyCommandStreamMESA_args,
        };
        use super::super::proto::types::vn_command_vkSetReplyCommandStreamMESA as Args;

        let args = Args { pStream: Some(stream), ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkSetReplyCommandStreamMESA_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkSetReplyCommandStreamMESA_args(&mut enc, VkFlags(0), &args);
        buf
    }

    /// A seek command's bytes, with whatever header flags the caller wants.
    ///
    /// It is the cheapest command that both reaches a handler and produces a reply, which makes it
    /// the one to drive the reply path with: no driver, no objects, no instance to have created.
    fn wire_seek(position: usize, flags: u32) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkSeekReplyCommandStreamMESA_args,
            vn_sizeof_vkSeekReplyCommandStreamMESA_args,
        };
        use super::super::proto::types::vn_command_vkSeekReplyCommandStreamMESA as Args;

        let args = Args { position, ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkSeekReplyCommandStreamMESA_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkSeekReplyCommandStreamMESA_args(&mut enc, VkFlags(flags), &args);
        buf
    }

    /// An execute command's bytes, naming streams and where each one's answers belong.
    fn wire_execute(
        streams: &[super::super::proto::types::VkCommandStreamDescriptionMESA],
        positions: Option<&[usize]>,
    ) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkExecuteCommandStreamsMESA_args, vn_sizeof_vkExecuteCommandStreamsMESA_args,
        };
        use super::super::proto::types::vn_command_vkExecuteCommandStreamsMESA as Args;

        let mut args = Args::default();
        // Positions first: both planters set the one count the pair shares, and the streams are
        // what that count is about.
        if let Some(p) = positions {
            args.plant_pReplyPositions(p);
        }
        args.plant_pStreams(streams);
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkExecuteCommandStreamsMESA_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkExecuteCommandStreamsMESA_args(&mut enc, VkFlags(0), &args);
        buf
    }

    /// A stream descriptor for bytes already written into the ring resource.
    fn stream_at(
        offset: usize,
        size: usize,
    ) -> super::super::proto::types::VkCommandStreamDescriptionMESA {
        reply_at(offset, size)
    }

    /// The claim the whole command exists for: commands recorded in guest memory run, their
    /// answers reach the window the outer batch set, and the outer batch carries on afterwards.
    ///
    /// The last part is the one the C pays for with `vkr_cs_decoder_save_state`. We build a
    /// decoder per stream instead, so the outer decode cannot be disturbed -- and this is what
    /// says so.
    #[test]
    fn an_executed_stream_runs_and_the_outer_batch_carries_on() {
        const WINDOW: usize = 0x21000;
        const STREAM: usize = 0x22000;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let inner = wire_seek(0x10, GENERATE_REPLY);
        assert!(t.1.copy_in(STREAM, &inner));

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire_execute(&[stream_at(STREAM, inner.len())], None));
        batch.extend_from_slice(&wire_seek(0x30, GENERATE_REPLY));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran(), "a served batch does not poison");

        let mut got = [0u8; 4];
        assert!(t.1.copy_out(WINDOW + 0x10, &mut got));
        assert_eq!(got, seek_reply_bytes(), "the executed stream's command ran and answered");

        assert!(t.1.copy_out(WINDOW + 0x30, &mut got));
        assert_eq!(got, seek_reply_bytes(), "the command after the execute ran too");
    }

    /// A pipeline-cache count call naming `device`, with whatever header flags the caller wants.
    fn wire_cache_data(device: u64, flags: u32) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkGetPipelineCacheData_args, vn_sizeof_vkGetPipelineCacheData_args,
        };
        use super::super::proto::types::vn_command_vkGetPipelineCacheData as Args;

        let mut size = 0usize;
        let mut args = Args::default();
        args.device = VkDevice(device);
        args.plant_pDataSize(&mut size);
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkGetPipelineCacheData_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkGetPipelineCacheData_args(&mut enc, VkFlags(flags), &args);
        buf
    }

    /// A ghost absorbs the commands pipelined behind a refused create -- but only the ones the
    /// guest is not waiting on. One it is waiting on cannot be absorbed: skipping it leaves the
    /// reply slot holding whatever was there before, which the guest reads as an answer, so the
    /// context has to stop instead.
    #[test]
    fn a_ghost_absorbs_a_command_unless_the_guest_is_waiting_on_it() {
        const WINDOW: usize = 0x21000;
        const GHOST: u64 = 7;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();

        // Not waiting: the command is lost, the context lives.
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.objects.borrow_mut().add_ghost(ObjectId(GHOST));
        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire_cache_data(GHOST, 0));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran(), "absorbed, not poisoned");
        assert!(!ctx.fatal());

        // Waiting: nothing the host could write is an honest answer, so it writes none and stops.
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.objects.borrow_mut().add_ghost(ObjectId(GHOST));
        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire_cache_data(GHOST, GENERATE_REPLY));
        assert!(!ctx.submit(&batch, &mut todo, &g, &t).ran(), "a reply it cannot give poisons");
        assert!(ctx.fatal());
        let mut got = [0u8; 4];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, [0; 4], "and no reply was written for it");
    }

    /// A command that replies without moving the reply position, for the tests about where a
    /// reply lands. Every other cheap replying command is a seek, which is exactly the thing that
    /// would hide the position under test.
    fn wire_instance_version() -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkEnumerateInstanceVersion_args, vn_sizeof_vkEnumerateInstanceVersion_args,
        };
        use super::super::proto::types::vn_command_vkEnumerateInstanceVersion as Args;

        let mut out = 0u32;
        let mut args = Args::default();
        args.plant_pApiVersion(&mut out);
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkEnumerateInstanceVersion_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkEnumerateInstanceVersion_args(&mut enc, VkFlags(GENERATE_REPLY), &args);
        buf
    }

    /// Its reply's first word: the command type, like every reply's.
    fn instance_version_reply_bytes() -> [u8; 4] {
        (VkCommandTypeEXT::VK_COMMAND_TYPE_vkEnumerateInstanceVersion_EXT.0 as u32).to_le_bytes()
    }

    /// Each stream's answers start where the guest said they would.
    #[test]
    fn each_stream_answers_at_the_position_it_was_given() {
        const WINDOW: usize = 0x21000;
        const A: usize = 0x22000;
        const B: usize = 0x22800;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let inner = wire_instance_version();
        assert!(t.1.copy_in(A, &inner));
        assert!(t.1.copy_in(B, &inner));

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire_execute(
            &[stream_at(A, inner.len()), stream_at(B, inner.len())],
            Some(&[0x40, 0x80]),
        ));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran());

        let mut got = [0u8; 4];
        assert!(t.1.copy_out(WINDOW + 0x40, &mut got));
        assert_eq!(
            got,
            instance_version_reply_bytes(),
            "the first stream answered where it was told"
        );
        assert!(t.1.copy_out(WINDOW + 0x80, &mut got));
        assert_eq!(
            got,
            instance_version_reply_bytes(),
            "the second stream answered where it was told"
        );

        // Nothing at the top of the window: both answers moved, neither was also appended.
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, [0; 4], "a reply position moves the answer, it does not copy it");
    }

    /// A stream of no bytes is skipped -- but its reply position is honoured first, and a position
    /// outside the window is refused whether or not there was anything to run.
    ///
    /// Two claims in one batch because they are the same claim: the seek happens before the skip,
    /// which is only visible when the seek is the thing that fails.
    #[test]
    fn an_empty_stream_is_skipped_and_its_reply_position_is_still_checked() {
        const WINDOW: usize = 0x21000;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();

        // Nothing to run, in a resource that is not even mapped: the skip is what keeps this from
        // being an error at all.
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        let nowhere = super::super::proto::types::VkCommandStreamDescriptionMESA {
            resourceId: RING_RES.get() + 1,
            offset: 0,
            size: 0,
        };
        batch.extend_from_slice(&wire_execute(&[nowhere], Some(&[0x40])));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran(), "an empty stream is not an error");

        // The same empty stream, asked to answer past the end of the window.
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire_execute(&[nowhere], Some(&[0x101])));
        assert!(
            !ctx.submit(&batch, &mut todo, &g, &t).ran(),
            "a position past the window is refused"
        );
        assert!(ctx.fatal());
    }

    /// A stream that names memory outside its resource is refused, and so is one whose offset and
    /// size only fit because they wrapped.
    #[test]
    fn a_stream_outside_its_resource_is_refused() {
        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();

        for s in [
            stream_at(t.1.len() - 4, 8),
            stream_at(usize::MAX, 8),
            super::super::proto::types::VkCommandStreamDescriptionMESA {
                resourceId: RING_RES.get() + 1,
                offset: 0,
                size: 4,
            },
        ] {
            let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
            assert!(
                !ctx.submit(&wire_execute(&[s], None), &mut todo, &g, &t).ran(),
                "{} bytes at {} of resource {} is not a stream this resource holds",
                s.size,
                s.offset,
                s.resourceId,
            );
            assert!(ctx.fatal());
        }
    }

    /// An execute inside an execute is refused, rather than recursing as deep as the guest likes.
    ///
    /// Three levels, so the refusal has something to be told apart from: the innermost stream
    /// answers into the window, and its answer never appearing is what says the second level did
    /// not run. A nesting test that only checked for poison would pass on a build that recursed
    /// happily and then died of something else.
    #[test]
    fn an_execute_inside_an_executed_stream_is_refused() {
        const WINDOW: usize = 0x21000;
        const B: usize = 0x22000;
        const C: usize = 0x22800;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let innermost = wire_instance_version();
        assert!(t.1.copy_in(C, &innermost));
        let middle = wire_execute(&[stream_at(C, innermost.len())], None);
        assert!(t.1.copy_in(B, &middle));

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire_execute(&[stream_at(B, middle.len())], None));
        assert!(!ctx.submit(&batch, &mut todo, &g, &t).ran(), "nesting is refused");
        assert!(ctx.fatal());

        let mut got = [0u8; 4];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, [0; 4], "the stream the nested execute named never ran");
    }

    /// Naming no streams at all is refused, and so is asking for reply positions with no window
    /// for them to be positions in.
    #[test]
    fn an_execute_that_cannot_mean_anything_is_refused() {
        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();

        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        assert!(!ctx.submit(&wire_execute(&[], None), &mut todo, &g, &t).ran(), "no streams");
        assert!(ctx.fatal());

        // Positions, and no reply stream was ever set: the guest has said where every answer
        // belongs and there is nowhere any of them could belong.
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        let w = wire_execute(&[stream_at(0x22000, 4)], Some(&[0]));
        assert!(!ctx.submit(&w, &mut todo, &g, &t).ran(), "positions with no window");
        assert!(ctx.fatal());
    }

    /// What a reply to a seek looks like on the wire: the command type, and nothing else.
    fn seek_reply_bytes() -> [u8; 4] {
        (VkCommandTypeEXT::VK_COMMAND_TYPE_vkSeekReplyCommandStreamMESA_EXT.0 as u32).to_le_bytes()
    }

    /// The end-to-end claim of the whole reply path: a command that asks for an answer gets one,
    /// in the guest's own memory, at the offset the guest chose.
    ///
    /// Driven over the wire rather than by calling the handler, because the thing being tested is
    /// the dispatch loop's commit -- who writes, when, and to where -- and none of that is visible
    /// from inside a handler.
    #[test]
    fn a_command_that_asks_for_an_answer_gets_one_in_the_guests_window() {
        const WINDOW: usize = 0x21000;
        const AT: usize = 8;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        // Set the window, then seek inside it and ask for a reply. The seek is what moves the
        // answer off the top of the window, so finding it at `AT` proves the position was honoured
        // rather than that everything happens to land at zero.
        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire_seek(AT, GENERATE_REPLY));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran(), "a served batch does not poison");

        let mut got = [0u8; 4];
        assert!(t.1.copy_out(WINDOW + AT, &mut got));
        assert_eq!(got, seek_reply_bytes(), "the answer is in the window, at the seeked offset");

        // Nothing was written at the top of the window: the seek moved the write, it did not copy.
        let mut top = [0u8; 4];
        assert!(t.1.copy_out(WINDOW, &mut top));
        assert_eq!(top, [0; 4], "a seek leaves what it skipped alone");
    }

    /// A reply the guest left no room for kills the context, and leaves the window untouched.
    ///
    /// The second half is the part worth having. The C encoder writes members straight into guest
    /// memory and finds out it has overrun partway through, so a guest that is polling its window
    /// can see half an answer and act on it. Ours cannot write anything it has not first measured.
    #[test]
    fn a_reply_that_does_not_fit_writes_nothing_at_all() {
        const WINDOW: usize = 0x21000;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        // Two bytes of room for a four-byte answer.
        let mut batch = wire_set_reply(&reply_at(WINDOW, 2));
        batch.extend_from_slice(&wire_seek(0, GENERATE_REPLY));
        assert!(
            !ctx.submit(&batch, &mut todo, &g, &t).ran(),
            "an answer with nowhere to go poisons"
        );
        assert!(ctx.fatal());

        let mut got = [0u8; 4];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, [0; 4], "a refused reply is not a partial one");
    }

    /// A seek past the end of the window is refused rather than clamped, and a seek to exactly the
    /// end is not: that is a stream with no room left, which is a state, not a mistake.
    #[test]
    fn a_seek_outside_the_window_is_refused() {
        const WINDOW: usize = 0x21000;
        const SIZE: usize = 0x100;

        for (position, ok) in [(0usize, true), (SIZE, true), (SIZE + 1, false), (usize::MAX, false)]
        {
            let t = ring_table();
            let g = crate::vulkan::global();
            let mut todo = Unimplemented::default();
            let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

            let mut batch = wire_set_reply(&reply_at(WINDOW, SIZE));
            batch.extend_from_slice(&wire_seek(position, 0));
            assert_eq!(
                ctx.submit(&batch, &mut todo, &g, &t).ran(),
                ok,
                "seeking to {position:#x} in a {SIZE:#x}-byte window"
            );
        }
    }

    /// A command that asked for an answer and then failed does not get to leave one behind.
    ///
    /// The encoder has already run by the time the handler's verdict lands, so the bytes exist --
    /// what must not happen is their reaching a guest that would read them as a real answer to a
    /// command that never worked. The commit is downstream of the verdict for exactly this.
    #[test]
    fn a_rejected_command_leaves_no_answer_behind() {
        const WINDOW: usize = 0x21000;
        const SIZE: usize = 0x100;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        // Out of range, and asking for a reply: the seek fails and the answer must not land.
        let mut batch = wire_set_reply(&reply_at(WINDOW, SIZE));
        batch.extend_from_slice(&wire_seek(SIZE + 1, GENERATE_REPLY));
        assert!(!ctx.submit(&batch, &mut todo, &g, &t).ran(), "a rejected command poisons");

        let mut got = [0u8; 4];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, [0; 4], "a failed command's reply is not delivered");
    }

    /// Seeking a stream that was never set is the guest talking about something that is not there.
    #[test]
    fn a_seek_with_no_stream_is_refused() {
        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        assert!(!ctx.submit(&wire_seek(0, 0), &mut todo, &g, &t).ran(), "there is nothing to seek");
        assert!(ctx.fatal());
    }

    /// One command's bytes, built by the generator's own encoder for it.
    ///
    /// The sizeof and encode functions come in as paths because a macro cannot paste a command
    /// name into an identifier, and hand-rolling the layout instead would be a second
    /// implementation of the wire -- the one that drifts silently.
    macro_rules! wire {
        ($size:path, $encode:path, $args:expr, $flags:expr) => {{
            let args = $args;
            let proto = crate::venus::cs::AllOfIt;
            let mut buf = vec![0u8; $size(&proto, &args)];
            let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
            $encode(&mut enc, VkFlags($flags), &args);
            buf
        }};
    }

    /// A forwarded command whose device this context never created answers with an error.
    ///
    /// The half of the group's contract that holds when there is no device to forward to. What
    /// crosses to a device that *is* there is pinned separately, against a planted one -- see
    /// [`a_forwarded_command_hands_the_driver_the_guests_own_arguments`].
    ///
    /// Every command in this group answers with a `VkResult` the guest will act on, and for most
    /// of them zero is `VK_SUCCESS`; `vkGetEventStatus` and `vkGetFenceStatus` go further and
    /// report *state* in that same field, where zero means the event is set or the fence is
    /// signalled. So the failure mode is not a missing answer, it is a confident wrong one: a
    /// guest told its fence is ready waits for nothing and reads memory the GPU has not written.
    ///
    /// The driver's device table is what stands between those two outcomes. A device it does not
    /// have must produce `VK_ERROR_INITIALIZATION_FAILED` and never a state, which is what this
    /// checks -- once per command, because the guarantee belongs to each handler's own lookup and
    /// a handler that forgot it would be invisible in any other's test.
    #[test]
    fn a_forwarded_command_on_an_unknown_device_answers_with_an_error() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;

        const WINDOW: usize = 0x21000;
        // Both timeline structs are required, so they have to be real: a null one is fatal at the
        // decode and would never reach the device lookup this test is about.
        let wait_info = ty::VkSemaphoreWaitInfo {
            sType: ty::VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            ..Default::default()
        };
        let signal_info = ty::VkSemaphoreSignalInfo {
            sType: ty::VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            ..Default::default()
        };
        let batches: Vec<(&str, Vec<u8>)> = vec![
            (
                "vkDeviceWaitIdle",
                wire!(
                    ser::vn_sizeof_vkDeviceWaitIdle_args,
                    ser::vn_encode_vkDeviceWaitIdle_args,
                    ty::vn_command_vkDeviceWaitIdle::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkGetEventStatus",
                wire!(
                    ser::vn_sizeof_vkGetEventStatus_args,
                    ser::vn_encode_vkGetEventStatus_args,
                    ty::vn_command_vkGetEventStatus::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkSetEvent",
                wire!(
                    ser::vn_sizeof_vkSetEvent_args,
                    ser::vn_encode_vkSetEvent_args,
                    ty::vn_command_vkSetEvent::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkResetEvent",
                wire!(
                    ser::vn_sizeof_vkResetEvent_args,
                    ser::vn_encode_vkResetEvent_args,
                    ty::vn_command_vkResetEvent::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkGetFenceStatus",
                wire!(
                    ser::vn_sizeof_vkGetFenceStatus_args,
                    ser::vn_encode_vkGetFenceStatus_args,
                    ty::vn_command_vkGetFenceStatus::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkResetCommandPool",
                wire!(
                    ser::vn_sizeof_vkResetCommandPool_args,
                    ser::vn_encode_vkResetCommandPool_args,
                    ty::vn_command_vkResetCommandPool::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkResetDescriptorPool",
                wire!(
                    ser::vn_sizeof_vkResetDescriptorPool_args,
                    ser::vn_encode_vkResetDescriptorPool_args,
                    ty::vn_command_vkResetDescriptorPool::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkBindBufferMemory",
                wire!(
                    ser::vn_sizeof_vkBindBufferMemory_args,
                    ser::vn_encode_vkBindBufferMemory_args,
                    ty::vn_command_vkBindBufferMemory::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkBindImageMemory",
                wire!(
                    ser::vn_sizeof_vkBindImageMemory_args,
                    ser::vn_encode_vkBindImageMemory_args,
                    ty::vn_command_vkBindImageMemory::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkFlushMappedMemoryRanges",
                wire!(
                    ser::vn_sizeof_vkFlushMappedMemoryRanges_args,
                    ser::vn_encode_vkFlushMappedMemoryRanges_args,
                    ty::vn_command_vkFlushMappedMemoryRanges::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkInvalidateMappedMemoryRanges",
                wire!(
                    ser::vn_sizeof_vkInvalidateMappedMemoryRanges_args,
                    ser::vn_encode_vkInvalidateMappedMemoryRanges_args,
                    ty::vn_command_vkInvalidateMappedMemoryRanges::default(),
                    GENERATE_REPLY
                ),
            ),
            (
                "vkWaitSemaphores",
                wire!(
                    ser::vn_sizeof_vkWaitSemaphores_args,
                    ser::vn_encode_vkWaitSemaphores_args,
                    ty::vn_command_vkWaitSemaphores {
                        pWaitInfo: Some(&wait_info),
                        timeout: u64::MAX,
                        ..Default::default()
                    },
                    GENERATE_REPLY
                ),
            ),
            (
                "vkSignalSemaphore",
                wire!(
                    ser::vn_sizeof_vkSignalSemaphore_args,
                    ser::vn_encode_vkSignalSemaphore_args,
                    ty::vn_command_vkSignalSemaphore {
                        pSignalInfo: Some(&signal_info),
                        ..Default::default()
                    },
                    GENERATE_REPLY
                ),
            ),
        ];

        for (name, cmd) in batches {
            let t = ring_table();
            let g = crate::vulkan::global();
            let mut todo = Unimplemented::default();
            let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

            let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
            batch.extend_from_slice(&cmd);
            assert!(
                ctx.submit(&batch, &mut todo, &g, &t).ran(),
                "{name} is served, so it does not poison"
            );
            assert!(todo.seen.is_empty(), "{name} reached a handler, so it is off the census");

            let mut got = [0u8; 8];
            assert!(t.1.copy_out(WINDOW, &mut got));
            let ret = i32::from_le_bytes([got[4], got[5], got[6], got[7]]);
            assert_eq!(
                ret,
                VkResult::VK_ERROR_INITIALIZATION_FAILED.0,
                "{name} answered {ret} for a device this context never created"
            );
        }
    }

    /// Every argument the guest sent reaches the driver unchanged, and the driver's answer
    /// reaches the guest.
    ///
    /// The shape helpers are thin, and thin is exactly where a swapped argument survives review:
    /// `object_flags_op` and `bind_one` each take several values of interchangeable-looking type,
    /// and passing the memory where the buffer goes compiles. Nothing outside this test can see
    /// it -- replay reports a command as accounted for either way, and the reply oracle compares
    /// encoders rather than arguments.
    ///
    /// Each stub returns `VK_NOT_READY`, which no path here fabricates: zero is the success these
    /// wrappers must not invent and `VK_ERROR_INITIALIZATION_FAILED` is the answer for a device
    /// that is missing, so a sentinel in `ret` can only have come back from the driver.
    #[test]
    fn a_forwarded_command_hands_the_driver_the_guests_own_arguments() {
        use super::super::proto::types::{
            VkBuffer, VkCommandPool, VkCommandPoolResetFlags, VkDeviceMemory, VkDeviceSize,
            VkEvent, VkMappedMemoryRange,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const SENTINEL: VkResult = VkResult::VK_NOT_READY;

        #[derive(Default)]
        struct Saw {
            waited: Vec<u64>,
            events: Vec<(u64, u64)>,
            pools: Vec<(u64, u32)>,
            binds: Vec<(u64, u64, u64)>,
            ranges: Vec<u32>,
        }
        thread_local! {
            static SAW: RefCell<Saw> = RefCell::new(Saw::default());
        }

        unsafe extern "C" fn wait_idle(device: VkDevice) -> VkResult {
            SAW.with_borrow_mut(|s| s.waited.push(device.0));
            SENTINEL
        }
        unsafe extern "C" fn set_event(device: VkDevice, event: VkEvent) -> VkResult {
            SAW.with_borrow_mut(|s| s.events.push((device.0, event.0)));
            SENTINEL
        }
        unsafe extern "C" fn reset_pool(
            _device: VkDevice,
            pool: VkCommandPool,
            flags: VkCommandPoolResetFlags,
        ) -> VkResult {
            SAW.with_borrow_mut(|s| s.pools.push((pool.0, flags.0)));
            SENTINEL
        }
        unsafe extern "C" fn bind_buffer(
            _device: VkDevice,
            buffer: VkBuffer,
            memory: VkDeviceMemory,
            offset: VkDeviceSize,
        ) -> VkResult {
            SAW.with_borrow_mut(|s| s.binds.push((buffer.0, memory.0, offset.0)));
            SENTINEL
        }
        unsafe extern "C" fn flush(
            _device: VkDevice,
            count: u32,
            p: *const VkMappedMemoryRange,
        ) -> VkResult {
            assert!(!p.is_null(), "a non-empty array arrives with its pointer");
            SAW.with_borrow_mut(|s| s.ranges.push(count));
            SENTINEL
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkDeviceWaitIdle(wait_idle);
        fns.plant_vkSetEvent(set_event);
        fns.plant_vkResetCommandPool(reset_pool);
        fns.plant_vkBindBufferMemory(bind_buffer);
        fns.plant_vkFlushMappedMemoryRanges(flush);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let device = VkDevice(DEVICE);

        let mut args = vn_command_vkDeviceWaitIdle { device, ..Default::default() };
        h.vkDeviceWaitIdle(&mut args);
        assert_eq!(args.ret, SENTINEL, "the driver's answer, not one of ours");
        SAW.with_borrow(|s| assert_eq!(s.waited, [DEVICE]));

        let mut args =
            vn_command_vkSetEvent { device, event: VkEvent(0x111), ..Default::default() };
        h.vkSetEvent(&mut args);
        assert_eq!(args.ret, SENTINEL);
        SAW.with_borrow(|s| assert_eq!(s.events, [(DEVICE, 0x111)], "the event the guest named"));

        // A pool and a flags word: distinct values, because each is a bare integer to the compiler
        // and a wrapper that swapped them would still build.
        let mut args = vn_command_vkResetCommandPool {
            device,
            commandPool: VkCommandPool(0x222),
            flags: VkCommandPoolResetFlags(0x4),
            ..Default::default()
        };
        h.vkResetCommandPool(&mut args);
        assert_eq!(args.ret, SENTINEL);
        SAW.with_borrow(|s| assert_eq!(s.pools, [(0x222, 0x4)], "the pool, then its flags"));

        // Three interchangeable-looking values in a row -- the shape most worth pinning.
        let mut args = vn_command_vkBindBufferMemory {
            device,
            buffer: VkBuffer(0x333),
            memory: VkDeviceMemory(0x444),
            memoryOffset: VkDeviceSize(0x555),
            ..Default::default()
        };
        h.vkBindBufferMemory(&mut args);
        assert_eq!(args.ret, SENTINEL);
        SAW.with_borrow(|s| assert_eq!(s.binds, [(0x333, 0x444, 0x555)], "buffer, memory, offset"));

        // The count Vulkan is given is the slice's own length, and nothing else carries it.
        let ranges = [VkMappedMemoryRange::default(); 3];
        let mut args = vn_command_vkFlushMappedMemoryRanges::default();
        args.device = device;
        args.plant_pMemoryRanges(&ranges);
        h.vkFlushMappedMemoryRanges(&mut args);
        assert_eq!(args.ret, SENTINEL);
        SAW.with_borrow(|s| assert_eq!(s.ranges, [3], "all three ranges, counted once"));

        // Nothing here came from Vulkan, so there is nothing to destroy. See `abandon_planted`.
        h.driver.abandon_planted();
    }

    /// The driver's answer travels the whole way: through the handler, through the commit, into
    /// the guest's window.
    ///
    /// [`a_forwarded_command_hands_the_driver_the_guests_own_arguments`] stops at `args.ret`, and
    /// the reply witnesses stop at the encoder. This is the one that runs the full length --
    /// wire, decode, handler, driver, commit -- so that a result which is right in the handler and
    /// never reaches the guest cannot pass. `vkDeviceWaitIdle` drives it because it names no
    /// object, so nothing here depends on the object table being set up first.
    #[test]
    fn the_drivers_answer_reaches_the_guests_window() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;
        use super::super::proto::types::VkAllocationCallbacks;
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const WINDOW: usize = 0x21000;
        const SENTINEL: VkResult = VkResult::VK_NOT_READY;

        /// The guest's name for the device, which is not the host's -- so a handler that reached
        /// the driver with the id instead of the handle would be visible here.
        const GUEST_ID: u64 = 0x5001;

        thread_local! {
            static CALLS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
        }
        unsafe extern "C" fn wait_idle(device: VkDevice) -> VkResult {
            CALLS.with_borrow_mut(|c| c.push(device.0));
            SENTINEL
        }
        /// Teardown destroys what the context created, and the planted table owes it an entry.
        unsafe extern "C" fn destroy_device(_d: VkDevice, _a: *const VkAllocationCallbacks) {}

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkDeviceWaitIdle(wait_idle);
        fns.plant_vkDestroyDevice(destroy_device);
        ctx.driver.plant_device(VkDevice(DEVICE), fns);
        ctx.objects
            .borrow_mut()
            .add(ObjectId(GUEST_ID), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(DEVICE), None)
            .expect("a fresh id");

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkDeviceWaitIdle_args,
            ser::vn_encode_vkDeviceWaitIdle_args,
            ty::vn_command_vkDeviceWaitIdle { device: VkDevice(GUEST_ID), ..Default::default() },
            GENERATE_REPLY
        ));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran(), "a served command does not poison");
        CALLS.with_borrow(|c| {
            assert_eq!(*c, [DEVICE], "the driver was called, with the host handle not the guest id")
        });

        let mut got = [0u8; 8];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(
            u32::from_le_bytes([got[0], got[1], got[2], got[3]]),
            VkCommandTypeEXT::VK_COMMAND_TYPE_vkDeviceWaitIdle_EXT.0 as u32,
            "the reply names the command it answers"
        );
        assert_eq!(
            i32::from_le_bytes([got[4], got[5], got[6], got[7]]),
            SENTINEL.0,
            "the driver's own result, carried all the way into the guest's memory"
        );
    }

    /// The timeline trio crosses the wire, the object table and the driver without being
    /// rewritten on the way.
    ///
    /// These three are the first commands this renderer forwards whose *interesting* arguments
    /// are nested inside a struct rather than sitting at the top of the command. That moves the
    /// risk: the handles in `VkSemaphoreWaitInfo` are resolved one at a time by the generated
    /// struct decoder, deep under the handler, and `pSemaphores` and `pValues` are two arrays the
    /// guest pairs by position under one count. A resolution that never happened hands the driver
    /// guest ids; a pairing that slipped waits on the right semaphores for the wrong values and
    /// comes back looking like an ordinary timeout. Neither is visible to replay, which counts the
    /// command as accounted for either way, nor to the reply oracle, which compares encoders.
    ///
    /// So it runs the full length -- wire, decode, object table, handler, driver, commit -- and
    /// the arrays carry values that differ from the handles and from each other, so a transposed
    /// pair cannot alias a correct one. `timeout` is `UINT64_MAX` because the whole point of
    /// Option A is that it is not clamped: a renderer that shortened it would report a timeout the
    /// guest's own wait never had.
    #[test]
    fn the_timeline_trio_crosses_as_the_guest_sent_it() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;
        use super::super::proto::types::{
            VkAllocationCallbacks, VkSemaphore, VkSemaphoreSignalInfo, VkSemaphoreWaitInfo,
            VkStructureType,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const WINDOW: usize = 0x21000;
        const SENTINEL: VkResult = VkResult::VK_NOT_READY;
        const COUNTER: u64 = 0x1234_5678_9abc;

        // Guest ids on the left, host handles on the right, and no digit shared between a pair --
        // so an id that reached the driver unresolved is unmistakable in the recording.
        const GUEST_DEV: u64 = 0x5001;
        const GUEST_SEM_A: u64 = 0x5002;
        const GUEST_SEM_B: u64 = 0x5003;
        const HOST_SEM_A: u64 = 0x901;
        const HOST_SEM_B: u64 = 0x902;

        /// One `vkWaitSemaphores` as the driver saw it. Named fields rather than a tuple because
        /// four of the five are integers, and a mismatch has to read as which one moved.
        #[derive(Debug, PartialEq, Eq)]
        struct Wait {
            device: u64,
            flags: u32,
            semaphores: Vec<u64>,
            values: Vec<u64>,
            timeout: u64,
        }

        #[derive(Default)]
        struct Saw {
            waited: Vec<Wait>,
            signalled: Vec<(u64, u64, u64)>,
            counted: Vec<(u64, u64)>,
        }
        thread_local! {
            static SAW: RefCell<Saw> = RefCell::new(Saw::default());
        }

        unsafe extern "C" fn wait(
            device: VkDevice,
            info: *const VkSemaphoreWaitInfo,
            timeout: u64,
        ) -> VkResult {
            assert!(!info.is_null(), "a required struct arrives with its pointer");
            // SAFETY: this stub stands where the driver stands, and reads exactly what the driver
            // would: the struct the decoder allocated, and the two arrays it sized from the count
            // in it. Both outlive the call.
            let i = unsafe { &*info };
            // A count of zero is the one case where the decoder leaves both pointers null, so it
            // is read without touching them -- and reading it as empty rather than walking off a
            // null is what lets a struct that arrived empty fail the comparison below instead of
            // taking the process down with it.
            let n = i.semaphoreCount as usize;
            let (sems, vals): (Vec<u64>, Vec<u64>) = if n == 0 {
                (Vec::new(), Vec::new())
            } else {
                assert!(
                    !i.pSemaphores.is_null() && !i.pValues.is_null(),
                    "a counted pair of arrays arrives with both its pointers"
                );
                // SAFETY: as above, now that the count and the two pointers agree.
                unsafe {
                    (
                        core::slice::from_raw_parts(i.pSemaphores, n).iter().map(|s| s.0).collect(),
                        core::slice::from_raw_parts(i.pValues, n).to_vec(),
                    )
                }
            };
            let flags = i.flags.0;
            SAW.with_borrow_mut(|s| {
                s.waited.push(Wait {
                    device: device.0,
                    flags,
                    semaphores: sems,
                    values: vals,
                    timeout,
                })
            });
            SENTINEL
        }
        unsafe extern "C" fn signal(
            device: VkDevice,
            info: *const VkSemaphoreSignalInfo,
        ) -> VkResult {
            assert!(!info.is_null());
            // SAFETY: as `wait`.
            let i = unsafe { &*info };
            SAW.with_borrow_mut(|s| s.signalled.push((device.0, i.semaphore.0, i.value)));
            SENTINEL
        }
        unsafe extern "C" fn counter(
            device: VkDevice,
            semaphore: VkSemaphore,
            out: *mut u64,
        ) -> VkResult {
            assert!(!out.is_null(), "the handler refuses a query with nowhere to answer");
            // SAFETY: `out` is the single arena slot the decoder allocated for this out-parameter.
            unsafe { *out = COUNTER };
            SAW.with_borrow_mut(|s| s.counted.push((device.0, semaphore.0)));
            SENTINEL
        }
        /// Teardown drains the device before destroying it, and the planted table owes it both.
        unsafe extern "C" fn idle(_d: VkDevice) -> VkResult {
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn destroy_device(_d: VkDevice, _a: *const VkAllocationCallbacks) {}

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkDeviceWaitIdle(idle);
        fns.plant_vkWaitSemaphores(wait);
        fns.plant_vkSignalSemaphore(signal);
        fns.plant_vkGetSemaphoreCounterValue(counter);
        fns.plant_vkDestroyDevice(destroy_device);
        ctx.driver.plant_device(VkDevice(DEVICE), fns);
        {
            let mut table = ctx.objects.borrow_mut();
            for (id, host, ty) in [
                (GUEST_DEV, DEVICE, VkObjectType::VK_OBJECT_TYPE_DEVICE),
                (GUEST_SEM_A, HOST_SEM_A, VkObjectType::VK_OBJECT_TYPE_SEMAPHORE),
                (GUEST_SEM_B, HOST_SEM_B, VkObjectType::VK_OBJECT_TYPE_SEMAPHORE),
            ] {
                table.add(ObjectId(id), ty, HostHandle(host), None).expect("a fresh id");
            }
        }

        // The counter query goes first so its reply -- the only one carrying a value the driver
        // wrote -- sits at the front of the window, where no other reply's size can move it.
        let mut value = 0u64;
        let mut cv = ty::vn_command_vkGetSemaphoreCounterValue::default();
        cv.device = VkDevice(GUEST_DEV);
        cv.semaphore = VkSemaphore(GUEST_SEM_A);
        cv.plant_pValue(&mut value);

        let sems = [VkSemaphore(GUEST_SEM_A), VkSemaphore(GUEST_SEM_B)];
        let vals = [0x77u64, 0x99u64];
        let info = VkSemaphoreWaitInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            flags: VkSemaphoreWaitFlags(0x1),
            semaphoreCount: 2,
            pSemaphores: sems.as_ptr(),
            pValues: vals.as_ptr(),
            ..Default::default()
        };
        let signal_info = VkSemaphoreSignalInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            semaphore: VkSemaphore(GUEST_SEM_B),
            value: 0xabcd,
            ..Default::default()
        };

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkGetSemaphoreCounterValue_args,
            ser::vn_encode_vkGetSemaphoreCounterValue_args,
            cv,
            GENERATE_REPLY
        ));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkWaitSemaphores_args,
            ser::vn_encode_vkWaitSemaphores_args,
            ty::vn_command_vkWaitSemaphores {
                device: VkDevice(GUEST_DEV),
                pWaitInfo: Some(&info),
                timeout: u64::MAX,
                ..Default::default()
            },
            GENERATE_REPLY
        ));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkSignalSemaphore_args,
            ser::vn_encode_vkSignalSemaphore_args,
            ty::vn_command_vkSignalSemaphore {
                device: VkDevice(GUEST_DEV),
                pSignalInfo: Some(&signal_info),
                ..Default::default()
            },
            GENERATE_REPLY
        ));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran(), "three served commands do not poison");

        SAW.with_borrow(|s| {
            assert_eq!(
                s.waited,
                [Wait {
                    device: DEVICE,
                    flags: 0x1,
                    semaphores: vec![HOST_SEM_A, HOST_SEM_B],
                    values: vec![0x77, 0x99],
                    timeout: u64::MAX,
                }],
                "host handles in order, each still against its own value, and the timeout whole"
            );
            assert_eq!(
                s.signalled,
                [(DEVICE, HOST_SEM_B, 0xabcd)],
                "the semaphore nested in the signal struct is resolved too"
            );
            assert_eq!(s.counted, [(DEVICE, HOST_SEM_A)]);
        });

        // The counter query's reply: command type, result, the out-pointer's marker, the value.
        let mut got = [0u8; 24];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(
            u32::from_le_bytes(got[0..4].try_into().unwrap()),
            VkCommandTypeEXT::VK_COMMAND_TYPE_vkGetSemaphoreCounterValue_EXT.0 as u32,
            "the reply names the command it answers"
        );
        assert_eq!(
            i32::from_le_bytes(got[4..8].try_into().unwrap()),
            SENTINEL.0,
            "the driver's own result, not a success this renderer invented"
        );
        assert_eq!(
            u64::from_le_bytes(got[16..24].try_into().unwrap()),
            COUNTER,
            "the counter the driver wrote, carried into the guest's memory"
        );
    }

    /// Reading a counter off a device this context never created is refused, not answered.
    ///
    /// The line between the two families this slice straddles, drawn where it is easiest to cross
    /// by accident. `vkWaitSemaphores` next door answers an unknown device with
    /// `VK_ERROR_INITIALIZATION_FAILED`, and that is a complete answer because its whole reply is
    /// that result. This one's reply also carries a `u64` -- and on a failed ask nobody wrote it,
    /// so the guest would read back the value it sent, encoded by us as though the host had put it
    /// there. There is no result code that says "and ignore the number", so the ring stops.
    #[test]
    fn reading_a_counter_off_an_unknown_device_is_refused() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;
        use super::super::proto::types::VkSemaphore;

        const WINDOW: usize = 0x21000;

        // Ids the object table resolves, naming host handles the driver has never heard of. Both
        // have to be registered: an id the table does not hold stops the ring at the decode
        // instead, which would leave this test passing for a reason that has nothing to do with
        // the handler at all.
        const GUEST_DEV: u64 = 0x5001;
        const GUEST_SEM: u64 = 0x5002;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        {
            let mut table = ctx.objects.borrow_mut();
            for (id, host, ty) in [
                (GUEST_DEV, 3, VkObjectType::VK_OBJECT_TYPE_DEVICE),
                (GUEST_SEM, 0x901, VkObjectType::VK_OBJECT_TYPE_SEMAPHORE),
            ] {
                table.add(ObjectId(id), ty, HostHandle(host), None).expect("a fresh id");
            }
        }

        let mut value = 0u64;
        let mut cv = ty::vn_command_vkGetSemaphoreCounterValue::default();
        cv.device = VkDevice(GUEST_DEV);
        cv.semaphore = VkSemaphore(GUEST_SEM);
        cv.plant_pValue(&mut value);

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkGetSemaphoreCounterValue_args,
            ser::vn_encode_vkGetSemaphoreCounterValue_args,
            cv,
            GENERATE_REPLY
        ));
        assert!(!ctx.submit(&batch, &mut todo, &g, &t).ran(), "no device to ask");
        assert!(ctx.fatal());
    }

    /// What one command's reply looks like when the answer is the one given, built by the
    /// generator's own encoder for it.
    ///
    /// The counterpart of [`wire!`], and there for the same reason: a query's reply is a struct
    /// laid out field by field, and writing those bytes out by hand in a test would be a second
    /// implementation of the wire that drifts the moment vk.xml moves. What this pins is
    /// everything between the driver's write and the guest's memory -- that the handler asked, that
    /// the driver's answer landed in the struct the reply reads, and that the reply reached the
    /// window the ring named. The encoder itself is pinned by the reply oracle, not here.
    macro_rules! reply {
        ($size:path, $encode:path, $args:expr) => {{
            let args = $args;
            let proto = crate::venus::cs::AllOfIt;
            let mut buf = vec![0u8; $size(&proto, &args)];
            let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
            $encode(&mut enc, &args);
            buf
        }};
    }

    /// A query's answer travels the whole way: the driver writes a struct, and the guest reads
    /// that struct out of its own window.
    ///
    /// The output direction, which nothing else covers. The input direction is pinned by the
    /// timeline witness and by replay; a query's *answer* is seen by neither -- replay strips the
    /// reply flag, and the reply oracle compares encoders against the C's without ever asking
    /// whether a driver was called.
    ///
    /// `vkGetImageSubresourceLayout2` drives it because it is the maximal shape in this group: a
    /// device, an object handle, an in-struct and an out-struct. Every field asserted on carries a
    /// distinct non-zero value, so a reply that encoded only part of the struct cannot pass.
    #[test]
    fn a_querys_answer_reaches_the_guests_window() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;
        use super::super::proto::types::{
            VkAllocationCallbacks, VkDeviceSize, VkImage, VkImageSubresource, VkImageSubresource2,
            VkStructureType, VkSubresourceLayout, VkSubresourceLayout2,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const WINDOW: usize = 0x21000;
        const GUEST_DEV: u64 = 0x5001;
        const GUEST_IMG: u64 = 0x5002;
        const HOST_IMG: u64 = 0x811;

        /// The answer, with nothing zero in it and no two fields alike.
        fn answer() -> VkSubresourceLayout2 {
            VkSubresourceLayout2 {
                sType: VkStructureType::VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2,
                subresourceLayout: VkSubresourceLayout {
                    offset: VkDeviceSize(0x1000),
                    size: VkDeviceSize(0x2000),
                    rowPitch: VkDeviceSize(0x300),
                    arrayPitch: VkDeviceSize(0x40),
                    depthPitch: VkDeviceSize(0x5),
                },
                ..Default::default()
            }
        }

        /// One layout query as the driver saw it. Named fields rather than a tuple, so a
        /// mismatch reads as which value moved.
        #[derive(Debug, PartialEq, Eq)]
        struct Asked {
            device: u64,
            image: u64,
            aspect: u32,
            mip: u32,
            layer: u32,
        }

        thread_local! {
            static SAW: RefCell<Vec<Asked>> = const { RefCell::new(Vec::new()) };
        }
        unsafe extern "C" fn layout(
            device: VkDevice,
            image: VkImage,
            sub: *const VkImageSubresource2,
            out: *mut VkSubresourceLayout2,
        ) {
            assert!(!sub.is_null() && !out.is_null(), "both structs arrive with their pointers");
            // SAFETY: this stub stands where the driver stands: `sub` is the arena struct the
            // decoder filled and `out` the arena slot the reply will read back, both live for the
            // call.
            unsafe {
                let s = (*sub).imageSubresource;
                SAW.with_borrow_mut(|v| {
                    v.push(Asked {
                        device: device.0,
                        image: image.0,
                        aspect: s.aspectMask.0,
                        mip: s.mipLevel,
                        layer: s.arrayLayer,
                    })
                });
                *out = answer();
            }
        }
        unsafe extern "C" fn idle(_d: VkDevice) -> VkResult {
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn destroy_device(_d: VkDevice, _a: *const VkAllocationCallbacks) {}

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkGetImageSubresourceLayout2(layout);
        fns.plant_vkDeviceWaitIdle(idle);
        fns.plant_vkDestroyDevice(destroy_device);
        ctx.driver.plant_device(VkDevice(DEVICE), fns);
        {
            let mut table = ctx.objects.borrow_mut();
            for (id, host, ty) in [
                (GUEST_DEV, DEVICE, VkObjectType::VK_OBJECT_TYPE_DEVICE),
                (GUEST_IMG, HOST_IMG, VkObjectType::VK_OBJECT_TYPE_IMAGE),
            ] {
                table.add(ObjectId(id), ty, HostHandle(host), None).expect("a fresh id");
            }
        }

        let sub = VkImageSubresource2 {
            sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2,
            imageSubresource: VkImageSubresource {
                aspectMask: VkImageAspectFlags(0x2),
                mipLevel: 3,
                arrayLayer: 5,
            },
            ..Default::default()
        };
        let mut layout_out = VkSubresourceLayout2::default();
        let mut q = ty::vn_command_vkGetImageSubresourceLayout2::default();
        q.device = VkDevice(GUEST_DEV);
        q.image = VkImage(GUEST_IMG);
        q.pSubresource = Some(&sub);
        q.plant_pLayout(&mut layout_out);

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkGetImageSubresourceLayout2_args,
            ser::vn_encode_vkGetImageSubresourceLayout2_args,
            q,
            GENERATE_REPLY
        ));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran(), "a served query does not poison");
        SAW.with_borrow(|v| {
            assert_eq!(
                *v,
                [Asked { device: DEVICE, image: HOST_IMG, aspect: 0x2, mip: 3, layer: 5 }],
                "host handles, and the subresource the guest asked about"
            )
        });

        // What the guest should be looking at, built the only honest way: by handing the
        // generator's own reply encoder the answer the driver gave.
        let mut expect_out = answer();
        let mut expect = ty::vn_command_vkGetImageSubresourceLayout2::default();
        expect.plant_pLayout(&mut expect_out);
        let want = reply!(
            ser::vn_sizeof_vkGetImageSubresourceLayout2_reply,
            ser::vn_encode_vkGetImageSubresourceLayout2_reply,
            expect
        );
        let mut got = vec![0u8; want.len()];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, want, "the driver's whole answer, in the guest's memory");
    }

    /// The one query that names a virtio-gpu resource rather than a Vulkan object, and the
    /// promise it has to keep.
    ///
    /// The guest reads `memoryTypeBits`, intersects it with its image's own requirements, and
    /// hands the result back as the `memoryTypeIndex` of a `vkAllocateMemory` that imports this
    /// same resource. So this answer and that import are one statement made twice: report a type
    /// the import will refuse and the guest binds nothing, report none and it binds nothing
    /// either -- for a buffer that would have worked. Host-visible is what the import accepts,
    /// so host-visible is what this says.
    ///
    /// Driven over the wire so the chained size struct is checked where it actually matters: the
    /// reply encoder walks the guest's `pNext` for itself, so a handler that never found the
    /// struct hands back a chain with a zero in it and no error anywhere.
    #[test]
    fn a_resource_the_guest_can_reach_reports_the_memory_it_can_bind() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;
        use super::super::proto::types::{
            VkAllocationCallbacks, VkMemoryPropertyFlagBits, VkMemoryResourcePropertiesMESA,
            VkStructureType,
        };

        const DEVICE: u64 = 3;
        const GUEST_DEV: u64 = 0x5001;
        const WINDOW: usize = 0x21000;
        /// Host-visible on 1 and 3 and not on 0 and 2, so a handler that reports "all of them"
        /// or reads the wrong bit cannot land on the same mask by accident.
        const TYPES: [VkMemoryPropertyFlags; 4] = [
            VkMemoryPropertyFlags(
                VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.0 as u32,
            ),
            VkMemoryPropertyFlags(
                VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.0 as u32,
            ),
            VkMemoryPropertyFlags(
                VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.0 as u32,
            ),
            VkMemoryPropertyFlags(
                VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.0 as u32
                    | VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.0 as u32,
            ),
        ];
        const HOST_VISIBLE_MASK: u32 = 0b1010;

        unsafe extern "C" fn idle(_d: VkDevice) -> VkResult {
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn destroy_device(_d: VkDevice, _a: *const VkAllocationCallbacks) {}

        let t = ring_table();
        let mapped = t.1.len() as u64;
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkDeviceWaitIdle(idle);
        fns.plant_vkDestroyDevice(destroy_device);
        ctx.driver.plant_device(VkDevice(DEVICE), fns);
        ctx.driver.plant_memory_types(VkDevice(DEVICE), &TYPES);
        ctx.objects
            .borrow_mut()
            .add(ObjectId(GUEST_DEV), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(DEVICE), None)
            .expect("a fresh id");

        /// The out-struct as the guest chains it: the base, and the size struct behind it.
        fn asked() -> (VkMemoryResourcePropertiesMESA, VkMemoryResourceAllocationSizePropertiesMESA)
        {
            (
                VkMemoryResourcePropertiesMESA {
                    sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_RESOURCE_PROPERTIES_MESA,
                    ..Default::default()
                },
                VkMemoryResourceAllocationSizePropertiesMESA {
                    sType:
                        VkStructureType::VK_STRUCTURE_TYPE_MEMORY_RESOURCE_ALLOCATION_SIZE_PROPERTIES_MESA,
                    ..Default::default()
                },
            )
        }

        let (mut props, mut size) = asked();
        props.pNext = (&mut size) as *mut _ as *mut core::ffi::c_void;
        let mut q = ty::vn_command_vkGetMemoryResourcePropertiesMESA::default();
        q.device = VkDevice(GUEST_DEV);
        q.resourceId = RING_RES.get();
        q.plant_pMemoryResourceProperties(&mut props);

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x200));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkGetMemoryResourcePropertiesMESA_args,
            ser::vn_encode_vkGetMemoryResourcePropertiesMESA_args,
            q,
            GENERATE_REPLY
        ));
        assert!(
            ctx.submit(&batch, &mut todo, &g, &t).ran(),
            "a resource the guest owns is answered"
        );

        let (mut want_props, mut want_size) = asked();
        want_props.memoryTypeBits = HOST_VISIBLE_MASK;
        want_size.allocationSize = mapped;
        want_props.pNext = (&mut want_size) as *mut _ as *mut core::ffi::c_void;
        let mut expect = ty::vn_command_vkGetMemoryResourcePropertiesMESA::default();
        expect.ret = VkResult::VK_SUCCESS;
        expect.plant_pMemoryResourceProperties(&mut want_props);
        let want = reply!(
            ser::vn_sizeof_vkGetMemoryResourcePropertiesMESA_reply,
            ser::vn_encode_vkGetMemoryResourcePropertiesMESA_reply,
            expect
        );
        let mut got = vec![0u8; want.len()];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, want, "the host-visible types, and the mapping's own size behind them");

        // An id naming no resource is answered, not refused. The C is explicit about why
        // (vkr_device_memory.c:945): a dead id is a reachable runtime state, and taking the ring
        // down for it aborts the guest process.
        let (mut props, mut size) = asked();
        props.pNext = (&mut size) as *mut _ as *mut core::ffi::c_void;
        let mut q = ty::vn_command_vkGetMemoryResourcePropertiesMESA::default();
        q.device = VkDevice(GUEST_DEV);
        q.resourceId = RING_RES.get() + 1;
        q.plant_pMemoryResourceProperties(&mut props);
        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x200));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkGetMemoryResourcePropertiesMESA_args,
            ser::vn_encode_vkGetMemoryResourcePropertiesMESA_args,
            q,
            GENERATE_REPLY
        ));
        assert!(
            ctx.submit(&batch, &mut todo, &g, &t).ran(),
            "an id that names nothing keeps the ring alive"
        );

        let (mut want_props, mut want_size) = asked();
        want_props.pNext = (&mut want_size) as *mut _ as *mut core::ffi::c_void;
        let mut expect = ty::vn_command_vkGetMemoryResourcePropertiesMESA::default();
        expect.ret = VkResult::VK_ERROR_INVALID_EXTERNAL_HANDLE;
        expect.plant_pMemoryResourceProperties(&mut want_props);
        let want = reply!(
            ser::vn_sizeof_vkGetMemoryResourcePropertiesMESA_reply,
            ser::vn_encode_vkGetMemoryResourcePropertiesMESA_reply,
            expect
        );
        let mut got = vec![0u8; want.len()];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, want, "the refusal, and nothing written into the struct behind it");
    }

    /// The other two ways this query can fail, and why only one of them is the guest's.
    ///
    /// No struct to answer into, and a device this renderer has no table for, are both *us*
    /// unable to put the question -- the query family's refusal. An unknown resource id, tested
    /// above, is the guest's own state and gets an answer. Keeping the two apart is the whole
    /// point of the command being here rather than beside the ordinary queries.
    #[test]
    fn only_a_question_this_renderer_cannot_put_refuses() {
        use super::super::proto::types::{
            VkMemoryResourcePropertiesMESA, vn_command_vkGetMemoryResourcePropertiesMESA as Cmd,
        };

        const GUEST_DEV: u64 = 0x5001;

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let t = ring_table();

        macro_rules! run {
            ($args:expr) => {{
                let mut rings = BTreeMap::new();
                let mut ctx_reply = None;
                let mut monitor = None;
                let mut jrnl = Journal::new();
                let mut h = Handlers {
                    objects: &objects,
                    todo: &mut todo,
                    driver: &mut driver,
                    global: &global,
                    ctx: ContextId::new(1).expect("1 is not zero"),
                    reject: None,
                    resources: &t,
                    rings: &mut rings,
                    monitor: &mut monitor,
                    wait: None,
                    execute: None,
                    replaying: false,
                    current_ring: None,
                    reply: &mut ctx_reply,
                    note: None,
                    journal: &mut jrnl,
                };
                h.vkGetMemoryResourcePropertiesMESA($args);
                h.reject
            }};
        }

        let mut args = Cmd::default();
        args.device = VkDevice(GUEST_DEV);
        args.resourceId = RING_RES.get();
        assert!(run!(&mut args).is_some(), "a query with nowhere to answer is refused");

        // A live resource and nowhere to look up the device: the struct would go back untouched
        // with VK_SUCCESS on it, which is the fiction this family exists to refuse.
        let mut props = VkMemoryResourcePropertiesMESA::default();
        let mut args = Cmd::default();
        args.device = VkDevice(GUEST_DEV);
        args.resourceId = RING_RES.get();
        args.plant_pMemoryResourceProperties(&mut props);
        assert!(run!(&mut args).is_some(), "a device with no table behind it is refused");
        assert_eq!(props.memoryTypeBits, 0, "and nothing was written");
    }

    /// A resource the property query can resolve and the allocation could not alias.
    ///
    /// The two commands read one resolution now, so the only way they can still disagree is for
    /// the resolution to name an allocation the driver has no record of -- a memory freed between
    /// the query and its answer, say. That is a resource the guest must be told is not importable,
    /// because the import that follows would fall through to ordinary memory and hand it a buffer
    /// nobody presents.
    ///
    /// It is the guest's own state and not ours, so it is answered rather than refused: the ring
    /// stays alive, exactly as it does for a resource id that names nothing.
    #[test]
    fn an_allocation_the_import_could_not_alias_is_not_called_importable() {
        use super::super::proto::types::{
            VkMemoryResourcePropertiesMESA, vn_command_vkGetMemoryResourcePropertiesMESA as Cmd,
        };
        use super::super::ring::Published;

        /// A table whose one resource publishes an allocation that is not in any driver.
        struct GhostExport;
        impl ShmResources for GhostExport {
            fn shm(
                &self,
                _: crate::ids::ResourceHandle,
            ) -> Option<std::sync::Arc<crate::guest_mem::GuestMap>> {
                None
            }
            fn bytes(&self, _: ContextId, _: crate::ids::ResourceHandle) -> Option<ResourceBytes> {
                Some(ResourceBytes::Allocation(Published { memory: ObjectId(0x9999), size: 4096 }))
            }
        }

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let t = GhostExport;

        let mut props = VkMemoryResourcePropertiesMESA::default();
        let mut args = Cmd::default();
        args.device = VkDevice(0x5001);
        args.resourceId = 7;
        args.plant_pMemoryResourceProperties(&mut props);

        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &t,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetMemoryResourcePropertiesMESA(&mut args);

        assert!(h.reject.is_none(), "the guest's own state does not poison the ring");
        assert_eq!(
            args.ret,
            VkResult::VK_ERROR_INVALID_EXTERNAL_HANDLE,
            "a resource the import cannot alias is not importable",
        );
        assert_eq!(props.memoryTypeBits, 0, "and nothing was written");
    }

    /// An enumeration that never asked how many there are.
    ///
    /// The count pointer is the only thing Vulkan's two calls have in common: the first writes it,
    /// the second is bounded by it, and a guest that sends neither has described no question.
    /// Twelve handlers ask this, all through one helper -- the wording was inline at all twelve
    /// before, which is how a duplicated refusal got into this file once already.
    ///
    /// Both families are checked, because the refusal is the same either way and their answers are
    /// not: one carries a `VkResult` a refusal could have been written into, the other carries
    /// nothing at all.
    #[test]
    fn an_enumeration_with_no_count_is_refused_before_it_is_asked() {
        use super::super::proto::types::{
            vn_command_vkEnumerateDeviceExtensionProperties,
            vn_command_vkGetPhysicalDeviceQueueFamilyProperties,
        };

        const MISSING: &str = "enumerated without asking for a count";

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let t = ring_table();

        macro_rules! run {
            ($call:expr) => {{
                let mut rings = BTreeMap::new();
                let mut ctx_reply = None;
                let mut monitor = None;
                let mut jrnl = Journal::new();
                let mut h = Handlers {
                    objects: &objects,
                    todo: &mut todo,
                    driver: &mut driver,
                    global: &global,
                    ctx: ContextId::new(1).expect("1 is not zero"),
                    reject: None,
                    resources: &t,
                    rings: &mut rings,
                    monitor: &mut monitor,
                    wait: None,
                    execute: None,
                    replaying: false,
                    current_ring: None,
                    reply: &mut ctx_reply,
                    note: None,
                    journal: &mut jrnl,
                };
                #[allow(clippy::redundant_closure_call)]
                (|h: &mut Handlers| $call(h))(&mut h);
                h.reject
            }};
        }

        // The `ret`-carrying family. Nothing may be written into `ret`: a refused command has no
        // verdict, and VK_SUCCESS on an enumeration that never happened is the worst answer here.
        let mut args = vn_command_vkEnumerateDeviceExtensionProperties::default();
        assert!(!args.has_pPropertyCount());
        assert_eq!(
            run!(|h: &mut Handlers| h.vkEnumerateDeviceExtensionProperties(&mut args)),
            Some(MISSING)
        );
        assert_eq!(args.ret, VkResult::default(), "no verdict for a call never made");

        // The void family, which has no `ret` and so nothing but the ring to refuse with.
        let mut args = vn_command_vkGetPhysicalDeviceQueueFamilyProperties::default();
        assert_eq!(
            run!(|h: &mut Handlers| h.vkGetPhysicalDeviceQueueFamilyProperties(&mut args)),
            Some(MISSING)
        );
    }

    /// A bitmask is its own type, at the width the wire gives it.
    ///
    /// vk.xml declares all 87 of them as typedefs of `VkFlags` or `VkFlags64`, and they used to
    /// be emitted as exactly that: 87 aliases for two types, so an image's usage and its create
    /// flags were the same type and passing one for the other compiled. That is not a
    /// hypothetical -- it is a sabotage that survived a whole sweep by compiling, and the same
    /// sabotage no longer builds.
    ///
    /// What the alias did carry, and must not be lost with it, is the width: `VkFlags` is four
    /// bytes and `VkFlags64` is eight, and a mask encoded at the wrong one moves every member
    /// after it. `repr(transparent)` is what keeps that true, and this is what says so.
    #[test]
    fn a_bitmask_is_its_own_type_at_its_own_width() {
        use super::super::proto::types::{
            VkImageUsageFlagBits, VkPipelineStageFlagBits2, VkPipelineStageFlags2,
        };

        assert_eq!(size_of::<VkImageUsageFlags>(), 4, "a VkFlags mask");
        assert_eq!(size_of::<VkPipelineStageFlags2>(), 8, "a VkFlags64 mask");

        // A bit belongs to exactly one mask, and reaches it by name rather than by a cast the
        // reader has to check the width of.
        assert_eq!(
            VkImageUsageFlags::from(VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
            VkImageUsageFlags(1)
        );
        assert_eq!(
            VkPipelineStageFlags2::from(VkPipelineStageFlagBits2::VK_PIPELINE_STAGE_2_COPY_BIT),
            VkPipelineStageFlags2(1 << 32)
        );
    }

    /// The struct a command cannot be carried out without, when the guest did not send one.
    ///
    /// vk.xml marks the info struct of every command in these three families required, so a null
    /// is not the guest declining an option -- it is a request with its subject missing. Before
    /// this guard existed the null went straight through to the host driver in five of these
    /// entry points, which is a guest handing the loader a pointer Vulkan says cannot be one.
    ///
    /// The wording is what is asserted rather than merely that something was refused: each of
    /// these three would refuse for a *different* reason if the guard were gone -- no device
    /// table, no recorder, or a driver error planted in `ret` -- and only the wording tells the
    /// missing subject apart from those.
    #[test]
    fn a_required_struct_the_guest_left_out_stops_the_ring() {
        use super::super::proto::types::{
            VkCommandBuffer, VkMemoryRequirements2, vn_command_vkCmdBeginRenderPass,
            vn_command_vkCreateFence, vn_command_vkGetImageMemoryRequirements2,
        };

        const MISSING: &str = "asked a query without saying what it is about";

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let t = ring_table();

        macro_rules! run {
            ($call:expr) => {{
                let mut rings = BTreeMap::new();
                let mut ctx_reply = None;
                #[allow(unused_mut)]
                let mut monitor = None;
                let mut jrnl = Journal::new();
                let mut h = Handlers {
                    objects: &objects,
                    todo: &mut todo,
                    driver: &mut driver,
                    global: &global,
                    ctx: ContextId::new(1).expect("1 is not zero"),
                    reject: None,
                    resources: &t,
                    rings: &mut rings,
                    monitor: &mut monitor,
                    wait: None,
                    execute: None,
                    replaying: false,
                    current_ring: None,
                    reply: &mut ctx_reply,
                    note: None,
                    journal: &mut jrnl,
                };
                #[allow(clippy::redundant_closure_call)]
                (|h: &mut Handlers| $call(h))(&mut h);
                h.reject
            }};
        }

        // A query. The out-struct is present, so the only thing missing is what to ask about.
        let mut out = VkMemoryRequirements2::default();
        let mut args = vn_command_vkGetImageMemoryRequirements2::default();
        args.plant_pMemoryRequirements(&mut out);
        assert_eq!(
            run!(|h: &mut Handlers| h.vkGetImageMemoryRequirements2(&mut args)),
            Some(MISSING)
        );

        // A create. Nothing may be planted for the guest to hold afterwards.
        let mut args = vn_command_vkCreateFence::default();
        assert_eq!(run!(|h: &mut Handlers| h.vkCreateFence(&mut args)), Some(MISSING));
        assert_eq!(args.ret, VkResult::default(), "no verdict was invented for a call never made");

        // A recording command, which has no reply at all to carry a refusal.
        let mut args = vn_command_vkCmdBeginRenderPass {
            commandBuffer: VkCommandBuffer(0x9001),
            ..Default::default()
        };
        assert_eq!(run!(|h: &mut Handlers| h.vkCmdBeginRenderPass(&mut args)), Some(MISSING));
    }

    /// The two-call idiom, in the half of it that has no way to say "there were more".
    ///
    /// `vkGetPhysicalDeviceQueueFamilyProperties` returns nothing at all: where its `ret`-carrying
    /// neighbours answer `VK_INCOMPLETE`, this one has only the count, which the driver lowers to
    /// what it wrote. That lowered count is the whole of what the guest is told, so a handler that
    /// forwards the array and never writes the count back leaves the guest reading its own
    /// question as the answer.
    #[test]
    fn a_void_enumeration_tells_the_guest_only_what_fit() {
        use super::super::proto::types::{
            VkPhysicalDevice, VkQueueFamilyProperties,
            vn_command_vkGetPhysicalDeviceQueueFamilyProperties as Cmd,
        };

        const PD: VkPhysicalDevice = VkPhysicalDevice(0x711);
        /// More families than the guest will make room for, so a short answer is a real one.
        const FAMILIES: u32 = 3;

        unsafe extern "C" fn families(
            pd: VkPhysicalDevice,
            count: *mut u32,
            out: *mut VkQueueFamilyProperties,
        ) {
            assert_eq!(pd, PD, "the physical device the guest named");
            // SAFETY: the caller passed a live count, and an array of that length or null.
            let count = unsafe { &mut *count };
            if out.is_null() {
                *count = FAMILIES;
                return;
            }
            let room = (*count).min(FAMILIES);
            // SAFETY: `count` is the length the caller sized the array to, and `room` is at most
            // that.
            let out = unsafe { core::slice::from_raw_parts_mut(out, room as usize) };
            for (i, f) in out.iter_mut().enumerate() {
                f.queueCount = 100 + i as u32;
            }
            *count = room;
        }

        let objects = Shared::new();
        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkGetPhysicalDeviceQueueFamilyProperties(families);
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(fns);
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();

        macro_rules! run {
            ($args:expr) => {{
                let mut rings = BTreeMap::new();
                let mut ctx_reply = None;
                let mut monitor = None;
                let mut jrnl = Journal::new();
                let mut h = Handlers {
                    objects: &objects,
                    todo: &mut todo,
                    driver: &mut driver,
                    global: &global,
                    ctx: ContextId::new(1).expect("1 is not zero"),
                    reject: None,
                    resources: &NO_RESOURCES,
                    rings: &mut rings,
                    monitor: &mut monitor,
                    wait: None,
                    execute: None,
                    replaying: false,
                    current_ring: None,
                    reply: &mut ctx_reply,
                    note: None,
                    journal: &mut jrnl,
                };
                h.vkGetPhysicalDeviceQueueFamilyProperties($args);
                assert!(h.reject.is_none(), "a served enumeration is not a refusal");
            }};
        }

        // The spec's first call: no array, so the count comes back as the total there are.
        let mut n = 0u32;
        let mut args = Cmd::default();
        args.physicalDevice = PD;
        args.plant_pQueueFamilyPropertyCount(&mut n);
        run!(&mut args);
        assert_eq!(n, FAMILIES, "the count query is answered with how many there are");

        // The second call, sized short on purpose. The guest gets what it sized for, and the
        // count is lowered to say so -- there is no other field that could.
        let mut props = [VkQueueFamilyProperties::default(); 2];
        let mut n = 2u32;
        let mut args = Cmd::default();
        args.physicalDevice = PD;
        args.plant_pQueueFamilyPropertyCount(&mut n);
        args.plant_pQueueFamilyProperties(&mut props);
        run!(&mut args);
        assert_eq!(n, 2, "the count comes back as what was written, not what was asked for");
        assert_eq!(
            [props[0].queueCount, props[1].queueCount],
            [100, 101],
            "the families the driver wrote reach the guest's own array"
        );

        // And the same call sized long, which is where a dropped count write-back shows: the two
        // numbers agree in the short case above, so only a guest with room to spare can tell a
        // handler that wrote the count from one that left the guest's own question in place.
        let mut props = [VkQueueFamilyProperties::default(); 4];
        let mut n = 4u32;
        let mut args = Cmd::default();
        args.physicalDevice = PD;
        args.plant_pQueueFamilyPropertyCount(&mut n);
        args.plant_pQueueFamilyProperties(&mut props);
        run!(&mut args);
        assert_eq!(
            n, FAMILIES,
            "the count is the driver's total, never the room the guest offered"
        );
        assert_eq!(
            props.map(|p| p.queueCount),
            [100, 101, 102, 0],
            "three families written, and the slot past them left as the guest sent it"
        );

        driver.abandon_planted();
    }

    /// The other half of the idiom, where the driver *can* say there were more.
    ///
    /// `VK_INCOMPLETE` is not an error the guest has to recover from: it sized the array, and being
    /// told the answer was trimmed is what lets it size a bigger one. Passing it through as an
    /// answer is the whole contract, and the roomy call beside it pins the count write-back --
    /// with a guest count of four and a driver total of two, a dropped write-back reads back as
    /// four families that were never written.
    #[test]
    fn a_short_enumeration_is_incomplete_and_a_roomy_one_is_not() {
        use super::super::proto::types::{
            VkPhysicalDevice, VkPhysicalDeviceToolProperties,
            vn_command_vkGetPhysicalDeviceToolProperties as Cmd,
        };

        const PD: VkPhysicalDevice = VkPhysicalDevice(0x711);
        const TOOLS: u32 = 2;

        unsafe extern "C" fn tools(
            _pd: VkPhysicalDevice,
            count: *mut u32,
            out: *mut VkPhysicalDeviceToolProperties,
        ) -> VkResult {
            // SAFETY: the caller passed a live count, and an array of that length or null.
            let count = unsafe { &mut *count };
            if out.is_null() {
                *count = TOOLS;
                return VkResult::VK_SUCCESS;
            }
            let room = (*count).min(TOOLS);
            // SAFETY: `count` is the length the caller sized the array to.
            let out = unsafe { core::slice::from_raw_parts_mut(out, room as usize) };
            for (i, t) in out.iter_mut().enumerate() {
                t.purposes = VkToolPurposeFlags(1 << i);
            }
            *count = room;
            if room < TOOLS { VkResult::VK_INCOMPLETE } else { VkResult::VK_SUCCESS }
        }

        let objects = Shared::new();
        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkGetPhysicalDeviceToolProperties(tools);
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(fns);
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();

        macro_rules! run {
            ($args:expr) => {{
                let mut rings = BTreeMap::new();
                let mut ctx_reply = None;
                let mut monitor = None;
                let mut jrnl = Journal::new();
                let mut h = Handlers {
                    objects: &objects,
                    todo: &mut todo,
                    driver: &mut driver,
                    global: &global,
                    ctx: ContextId::new(1).expect("1 is not zero"),
                    reject: None,
                    resources: &NO_RESOURCES,
                    rings: &mut rings,
                    monitor: &mut monitor,
                    wait: None,
                    execute: None,
                    replaying: false,
                    current_ring: None,
                    reply: &mut ctx_reply,
                    note: None,
                    journal: &mut jrnl,
                };
                h.vkGetPhysicalDeviceToolProperties($args);
                assert!(h.reject.is_none(), "a short answer is an answer, not a refusal");
            }};
        }

        let mut props = [VkPhysicalDeviceToolProperties::default(); 1];
        let mut n = 1u32;
        let mut args = Cmd::default();
        args.physicalDevice = PD;
        args.plant_pToolCount(&mut n);
        args.plant_pToolProperties(&mut props);
        run!(&mut args);
        assert_eq!(
            args.ret,
            VkResult::VK_INCOMPLETE,
            "the driver had more than the guest sized for"
        );
        assert_eq!(n, 1, "and the count says how many of them arrived");

        let mut props = [VkPhysicalDeviceToolProperties::default(); 4];
        let mut n = 4u32;
        let mut args = Cmd::default();
        args.physicalDevice = PD;
        args.plant_pToolCount(&mut n);
        args.plant_pToolProperties(&mut props);
        run!(&mut args);
        assert_eq!(args.ret, VkResult::VK_SUCCESS, "room to spare is not a short answer");
        assert_eq!(n, TOOLS, "the count is the driver's total, never the room the guest offered");

        driver.abandon_planted();
    }

    /// The venus handshake, which is not a Vulkan query however much it looks like one.
    ///
    /// `vkEnumerateInstanceExtensionProperties` asks what the *renderer* understands on the wire,
    /// not what the host loader has installed -- the guest never talks to that loader, so its list
    /// would be an answer to a question nobody asked. The C answers it from a table of two and so
    /// do we, off the same spec versions the capset states.
    ///
    /// Driven over the wire because the count write-back is only observable there: it decides how
    /// many entries the reply encoder walks, so a handler that fills the array and forgets the
    /// count hands the guest two real names followed by two it never wrote.
    #[test]
    fn the_guest_is_told_the_two_extensions_this_renderer_speaks() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;
        use super::super::proto::types::VkExtensionProperties;

        const WINDOW: usize = 0x21000;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let speaks = crate::venus::driver::renderer_extensions();
        assert_eq!(speaks.len(), 2, "the two protocol extensions this build serializes");

        // Room for four, and there are two. The two numbers have to differ or a dropped
        // write-back is invisible: the reply would encode the guest's own four and match.
        let mut room = [VkExtensionProperties::default(); 4];
        let mut n = 4u32;
        let mut q = ty::vn_command_vkEnumerateInstanceExtensionProperties::default();
        q.plant_pPropertyCount(&mut n);
        q.plant_pProperties(&mut room);

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x1000));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkEnumerateInstanceExtensionProperties_args,
            ser::vn_encode_vkEnumerateInstanceExtensionProperties_args,
            q,
            GENERATE_REPLY
        ));
        assert!(ctx.submit(&batch, &mut todo, &g, &t).ran(), "the handshake does not poison");

        let mut want_props = speaks.clone();
        let mut want_n = 2u32;
        let mut expect = ty::vn_command_vkEnumerateInstanceExtensionProperties::default();
        expect.ret = VkResult::VK_SUCCESS;
        expect.plant_pPropertyCount(&mut want_n);
        expect.plant_pProperties(&mut want_props[..]);
        let want = reply!(
            ser::vn_sizeof_vkEnumerateInstanceExtensionProperties_reply,
            ser::vn_encode_vkEnumerateInstanceExtensionProperties_reply,
            expect
        );
        let mut got = vec![0u8; want.len()];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(got, want, "both names, and a count that says two rather than four");
    }

    /// A format probe the driver says no to is an answer, not a refusal.
    ///
    /// `VK_ERROR_FORMAT_NOT_SUPPORTED` is how a driver declines one combination of format, type,
    /// tiling and usage, and a guest walks a table of them collecting exactly that. Treating it as
    /// a failure to ask would stop the ring on a guest doing something completely ordinary. The
    /// line the handler draws is between the driver's own answer and this renderer being unable to
    /// put the question at all -- only the second is a refusal.
    ///
    /// It also pins the six loose scalars, which is the other reason this command is here: they
    /// are interchangeable to the compiler and passing `usage` where `tiling` goes builds.
    #[test]
    fn a_format_probe_the_driver_declines_is_still_an_answer() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;
        use super::super::proto::types::{
            VkAllocationCallbacks, VkFormat, VkImageFormatProperties, VkImageTiling, VkImageType,
            VkInstance, VkPhysicalDevice,
        };
        use std::cell::RefCell;

        const WINDOW: usize = 0x21000;
        const GUEST_PD: u64 = 0x5001;
        const HOST_PD: u64 = 0x711;
        const DECLINED: VkResult = VkResult::VK_ERROR_FORMAT_NOT_SUPPORTED;

        /// One probe as the driver saw it. The six are all bare integers to the compiler, so
        /// naming them here is what makes a swapped pair read as one.
        #[derive(Debug, PartialEq, Eq)]
        struct Probe {
            pd: u64,
            format: i32,
            ty: i32,
            tiling: i32,
            usage: u32,
            flags: u32,
        }

        thread_local! {
            static SAW: RefCell<Vec<Probe>> = const { RefCell::new(Vec::new()) };
        }
        #[allow(clippy::too_many_arguments)]
        unsafe extern "C" fn probe(
            pd: VkPhysicalDevice,
            format: VkFormat,
            ty: VkImageType,
            tiling: VkImageTiling,
            usage: VkImageUsageFlags,
            flags: VkImageCreateFlags,
            out: *mut VkImageFormatProperties,
        ) -> VkResult {
            assert!(!out.is_null(), "the handler refuses a probe with nowhere to answer");
            SAW.with_borrow_mut(|v| {
                v.push(Probe {
                    pd: pd.0,
                    format: format.0,
                    ty: ty.0,
                    tiling: tiling.0,
                    usage: usage.0,
                    flags: flags.0,
                })
            });
            DECLINED
        }
        unsafe extern "C" fn destroy_instance(_i: VkInstance, _a: *const VkAllocationCallbacks) {}

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkGetPhysicalDeviceImageFormatProperties(probe);
        fns.plant_vkDestroyInstance(destroy_instance);
        ctx.driver.plant_instance(fns);
        ctx.objects
            .borrow_mut()
            .add(
                ObjectId(GUEST_PD),
                VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE,
                HostHandle(HOST_PD),
                None,
            )
            .expect("a fresh id");

        // Six values, no two alike, so a pair passed in the wrong order cannot look right.
        let mut props = VkImageFormatProperties::default();
        let mut q = ty::vn_command_vkGetPhysicalDeviceImageFormatProperties::default();
        q.physicalDevice = VkPhysicalDevice(GUEST_PD);
        q.format = VkFormat(37);
        q.r#type = VkImageType(1);
        q.tiling = VkImageTiling(2);
        q.usage = VkImageUsageFlags(0x40);
        q.flags = VkImageCreateFlags(0x800);
        q.plant_pImageFormatProperties(&mut props);

        let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
        batch.extend_from_slice(&wire!(
            ser::vn_sizeof_vkGetPhysicalDeviceImageFormatProperties_args,
            ser::vn_encode_vkGetPhysicalDeviceImageFormatProperties_args,
            q,
            GENERATE_REPLY
        ));
        assert!(
            ctx.submit(&batch, &mut todo, &g, &t).ran(),
            "the driver answered, so the ring lives on"
        );
        SAW.with_borrow(|v| {
            assert_eq!(
                *v,
                [Probe { pd: HOST_PD, format: 37, ty: 1, tiling: 2, usage: 0x40, flags: 0x800 }],
                "the physical device, then format, type, tiling, usage, flags, in that order"
            )
        });

        let mut got = [0u8; 8];
        assert!(t.1.copy_out(WINDOW, &mut got));
        assert_eq!(
            i32::from_le_bytes([got[4], got[5], got[6], got[7]]),
            DECLINED.0,
            "the driver's refusal of this format, carried back as the answer it is"
        );
    }

    /// The query helpers hand the driver the guest's own arguments, and refuse rather than invent.
    ///
    /// Two things at once, because they are the same risk seen from either side. Every helper in
    /// this group takes a handle or a run of scalars that the compiler cannot tell apart, so a
    /// transposition builds; and every one of them can fail to ask at all, where the only wrong
    /// answer is a confident one -- an unfilled struct encoded as though the host had written it,
    /// or an address of zero the guest would hand to the GPU.
    #[test]
    fn a_query_forwards_what_it_was_given_and_refuses_what_it_cannot_ask() {
        use super::super::proto::types::{
            VkBuffer, VkBufferDeviceAddressInfo, VkDeviceAddress, VkImage, VkMemoryRequirements,
            VkPeerMemoryFeatureFlags,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const ADDRESS: u64 = 0xdead_0000_beef;

        #[derive(Default)]
        struct Saw {
            buffers: Vec<u64>,
            images: Vec<u64>,
            peers: Vec<(u32, u32, u32)>,
        }
        thread_local! {
            static SAW: RefCell<Saw> = RefCell::new(Saw::default());
        }

        unsafe extern "C" fn buffer_reqs(
            _d: VkDevice,
            buffer: VkBuffer,
            out: *mut VkMemoryRequirements,
        ) {
            assert!(!out.is_null());
            SAW.with_borrow_mut(|s| s.buffers.push(buffer.0));
        }
        unsafe extern "C" fn image_reqs(
            _d: VkDevice,
            image: VkImage,
            out: *mut VkMemoryRequirements,
        ) {
            assert!(!out.is_null());
            SAW.with_borrow_mut(|s| s.images.push(image.0));
        }
        unsafe extern "C" fn peers(
            _d: VkDevice,
            heap: u32,
            local: u32,
            remote: u32,
            out: *mut VkPeerMemoryFeatureFlags,
        ) {
            assert!(!out.is_null());
            SAW.with_borrow_mut(|s| s.peers.push((heap, local, remote)));
        }
        unsafe extern "C" fn address(
            _d: VkDevice,
            info: *const VkBufferDeviceAddressInfo,
        ) -> VkDeviceAddress {
            assert!(!info.is_null(), "a required struct arrives with its pointer");
            VkDeviceAddress(ADDRESS)
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkGetBufferMemoryRequirements(buffer_reqs);
        fns.plant_vkGetImageMemoryRequirements(image_reqs);
        fns.plant_vkGetDeviceGroupPeerMemoryFeatures(peers);
        fns.plant_vkGetBufferDeviceAddress(address);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let device = VkDevice(DEVICE);

        // Buffer and image requirements have identical shapes and adjacent entry points; each has
        // to reach its own.
        let mut reqs = VkMemoryRequirements::default();
        let mut args = vn_command_vkGetBufferMemoryRequirements::default();
        args.device = device;
        args.buffer = VkBuffer(0x111);
        args.plant_pMemoryRequirements(&mut reqs);
        h.vkGetBufferMemoryRequirements(&mut args);

        let mut reqs = VkMemoryRequirements::default();
        let mut args = vn_command_vkGetImageMemoryRequirements::default();
        args.device = device;
        args.image = VkImage(0x222);
        args.plant_pMemoryRequirements(&mut reqs);
        h.vkGetImageMemoryRequirements(&mut args);

        // Three `u32`s, two of which mean opposite things.
        let mut feats = VkPeerMemoryFeatureFlags::default();
        let mut args = vn_command_vkGetDeviceGroupPeerMemoryFeatures::default();
        args.device = device;
        args.heapIndex = 1;
        args.localDeviceIndex = 2;
        args.remoteDeviceIndex = 3;
        args.plant_pPeerMemoryFeatures(&mut feats);
        h.vkGetDeviceGroupPeerMemoryFeatures(&mut args);

        // The answer is the return value, and nothing else carries it.
        let info = VkBufferDeviceAddressInfo::default();
        let mut args = vn_command_vkGetBufferDeviceAddress {
            device,
            pInfo: Some(&info),
            ..Default::default()
        };
        h.vkGetBufferDeviceAddress(&mut args);
        assert_eq!(args.ret.0, ADDRESS, "the driver's address, not a zero of ours");

        assert!(h.reject.is_none(), "every one of those was answerable");
        SAW.with_borrow(|s| {
            assert_eq!(s.buffers, [0x111], "the buffer went to the buffer query");
            assert_eq!(s.images, [0x222], "and the image to the image one");
            assert_eq!(s.peers, [(1, 2, 3)], "heap, then local, then remote");
        });

        // A query with no struct to answer into: there is nowhere to put an answer, and encoding
        // the guest's own bytes back at it would be indistinguishable from one.
        let mut args = vn_command_vkGetBufferMemoryRequirements::default();
        args.device = device;
        h.vkGetBufferMemoryRequirements(&mut args);
        assert!(h.reject.take().is_some(), "a query with nowhere to answer");

        // A query this driver has no entry point for. `vkGetRenderAreaGranularity` was never
        // planted above, so the table has no opinion about it.
        let mut extent = super::super::proto::types::VkExtent2D::default();
        let mut args = vn_command_vkGetRenderAreaGranularity::default();
        args.device = device;
        args.plant_pGranularity(&mut extent);
        h.vkGetRenderAreaGranularity(&mut args);
        assert!(h.reject.take().is_some(), "a query this driver cannot answer");

        // The address queries have no field in which to say no, so a failed ask must reject too:
        // `ret` stays zero, and zero is a null address.
        let info = super::super::proto::types::VkDeviceMemoryOpaqueCaptureAddressInfo::default();
        let mut args = vn_command_vkGetDeviceMemoryOpaqueCaptureAddress {
            device,
            pInfo: Some(&info),
            ..Default::default()
        };
        h.vkGetDeviceMemoryOpaqueCaptureAddress(&mut args);
        assert!(h.reject.take().is_some(), "an address this driver cannot be asked for");
        assert_eq!(args.ret, 0, "and nothing was invented to fill it");

        // And an address query with no struct naming what to look up. The struct is required, so
        // a guest omitting it is already fatal at the decode -- but the handler is what stands
        // between a null and the driver, and it has to hold on its own.
        let mut args = vn_command_vkGetBufferDeviceAddress { device, ..Default::default() };
        h.vkGetBufferDeviceAddress(&mut args);
        assert!(h.reject.take().is_some(), "an address query naming nothing");
        assert_eq!(args.ret.0, 0, "and no address was invented for it");

        // Nothing here came from Vulkan, so there is nothing to destroy. See `abandon_planted`.
        h.driver.abandon_planted();
    }

    /// A wait or a signal that names no semaphores at all is refused, not forwarded.
    ///
    /// The struct is required, so a guest omitting it is already fatal at the decode -- but the
    /// handler is what stands between a null and the driver, and it has to hold on its own. There
    /// is no honest answer to give here: the reply is a bare `VkResult`, and every value it could
    /// carry says something happened to semaphores that were never named.
    #[test]
    fn a_timeline_command_with_no_struct_is_refused() {
        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let mut args = vn_command_vkWaitSemaphores { timeout: u64::MAX, ..Default::default() };
        h.vkWaitSemaphores(&mut args);
        assert!(h.reject.is_some(), "a wait with no wait info");
        h.reject = None;

        let mut args = vn_command_vkSignalSemaphore::default();
        h.vkSignalSemaphore(&mut args);
        assert!(h.reject.is_some(), "a signal with no signal info");
    }

    /// Waiting on a queue this context never retrieved is refused, not answered.
    ///
    /// The odd one out of the group, and deliberately so: a queue carries no device of its own, so
    /// there is no table to miss and no error that means "not yours". `vkQueueSubmit` already
    /// rejects for exactly this reason and this follows it, because the two must not disagree
    /// about what an unknown queue is.
    #[test]
    fn waiting_on_an_unknown_queue_is_refused() {
        use super::super::proto::serialize as ser;
        use super::super::proto::types as ty;

        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        let cmd = wire!(
            ser::vn_sizeof_vkQueueWaitIdle_args,
            ser::vn_encode_vkQueueWaitIdle_args,
            ty::vn_command_vkQueueWaitIdle::default(),
            0
        );
        assert!(!ctx.submit(&cmd, &mut todo, &g, &t).ran(), "a queue with no device behind it");
        assert!(ctx.fatal());
    }

    /// A command with no handler at all, with whatever header flags the caller wants.
    ///
    /// `vkGetDeferredOperationResultKHR` is the one to ask with on two counts. Its whole reply is
    /// the command type and a `VkResult`, and a `VkResult` of zero is `VK_SUCCESS` -- so a
    /// default-constructed answer to it is not obviously-wrong noise, it is the guest being told
    /// the operation finished. And it belongs to `VK_KHR_deferred_host_operations`, which this
    /// renderer cannot advertise, so it is one of the commands deliberately left unserved rather
    /// than one merely waiting its turn. Serving it would break this test, which is the point:
    /// whoever does has to come here and pick another still-unserved command.
    fn wire_unserved(flags: u32) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkGetDeferredOperationResultKHR_args,
            vn_sizeof_vkGetDeferredOperationResultKHR_args,
        };
        use super::super::proto::types::vn_command_vkGetDeferredOperationResultKHR as Args;

        // Null handles throughout: `VK_NULL_HANDLE` is an ordinary value on the wire, so this
        // decodes cleanly and reaches the trait default, which is the whole point here.
        let args = Args::default();
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkGetDeferredOperationResultKHR_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkGetDeferredOperationResultKHR_args(&mut enc, VkFlags(flags), &args);
        buf
    }

    /// A command this build does not serve is counted, and then stops the context.
    ///
    /// The generated dispatch encodes a reply for every command it decodes, run or not, out of the
    /// argument struct as it stands -- and for an unserved command that struct is still all zeros.
    /// Committing it would hand the guest a well-formed `VK_SUCCESS` for work no handler ever did,
    /// which is indistinguishable from a real answer and is the precise failure this renderer
    /// exists to stop making. The reply flag does not soften it: a guest that asked for no answer
    /// still goes on believing the host did the thing, and that divergence surfaces later,
    /// somewhere with no way left to name the command that caused it. So the loop covers both
    /// flags and expects the same verdict from each.
    #[test]
    fn an_unserved_command_stops_the_context_either_way() {
        const WINDOW: usize = 0x21000;

        for reply_wanted in [false, true] {
            let t = ring_table();
            let g = crate::vulkan::global();
            let mut todo = Unimplemented::default();
            let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

            let mut batch = wire_set_reply(&reply_at(WINDOW, 0x100));
            batch.extend_from_slice(&wire_unserved(if reply_wanted { GENERATE_REPLY } else { 0 }));

            assert!(
                !ctx.submit(&batch, &mut todo, &g, &t).ran(),
                "an unserved command, reply wanted: {reply_wanted}"
            );

            // Either way it is on the census: refusing to answer is not refusing to notice.
            assert_eq!(
                todo.seen
                    .get(&VkCommandTypeEXT::VK_COMMAND_TYPE_vkGetDeferredOperationResultKHR_EXT.0),
                Some(&1),
                "the command was counted"
            );

            // And either way the window is untouched -- there was never an answer to put in it.
            let mut got = [0u8; 8];
            assert!(t.1.copy_out(WINDOW, &mut got));
            assert_eq!(got, [0; 8], "nothing was invented to fill the guest's window");
        }
    }

    /// An answer goes back down the stream the question came up.
    ///
    /// The other reply witnesses all drive the context's own stream, so "the reply lands in the
    /// window belonging to whoever asked" is true of them by having only one window to land in.
    /// This one gives the context and a ring a window each and asks on the ring, which is the only
    /// arrangement where writing to the wrong one is possible at all.
    #[test]
    fn a_reply_lands_in_the_window_of_the_ring_that_asked() {
        const CONTEXT_WINDOW: usize = 0x21000;
        const RING_WINDOW: usize = 0x22000;

        let t = ring_table();
        let info = ring_info();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));

        assert!(
            ctx.submit(&wire_create_ring(7, &info), &mut todo, &g, &t).ran(),
            "ring 7 is accepted"
        );

        // Two streams, two windows, set down the stream each belongs to -- the ring's first, so
        // that a context-wide slot would have the context's window in it by the time the ring
        // asks. Setting the context's first instead makes the two orders indistinguishable: a
        // single shared slot would just carry the ring's window, and the answer would still land
        // in the right place for the wrong reason.
        assert!(ctx.submit_ring(
            RingId(7),
            &wire_set_reply(&reply_at(RING_WINDOW, 0x100)),
            &mut todo,
            &g,
            &t
        ));
        assert!(
            ctx.submit(&wire_set_reply(&reply_at(CONTEXT_WINDOW, 0x100)), &mut todo, &g, &t).ran()
        );

        // The question arrives on the ring, so the answer belongs in the ring's window.
        assert!(ctx.submit_ring(RingId(7), &wire_seek(0, GENERATE_REPLY), &mut todo, &g, &t));

        let mut got = [0u8; 4];
        assert!(t.1.copy_out(RING_WINDOW, &mut got));
        assert_eq!(
            got,
            seek_reply_bytes(),
            "the ring's question was answered in the ring's window"
        );

        let mut other = [0u8; 4];
        assert!(t.1.copy_out(CONTEXT_WINDOW, &mut other));
        assert_eq!(other, [0; 4], "the context's window is not where a ring's answers go");
    }

    fn wire_create_ring(
        ring: u64,
        info: &crate::venus::proto::types::VkRingCreateInfoMESA,
    ) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkCreateRingMESA_args, vn_sizeof_vkCreateRingMESA_args,
        };
        use super::super::proto::types::vn_command_vkCreateRingMESA as Args;

        let args = Args { ring, pCreateInfo: Some(info), ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; vn_sizeof_vkCreateRingMESA_args(&proto, &args)];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        vn_encode_vkCreateRingMESA_args(&mut enc, VkFlags(0), &args);
        buf
    }

    /// A ring create that asks to be monitored, at the period the caller names.
    ///
    /// The chain is built with a real `VkRingMonitorInfoMESA` rather than a hand-rolled wire
    /// blob, so the decoder's own `pNext` walk is what is being exercised -- an unknown chained
    /// struct poisons the stream, and a monitor that only worked against a synthetic chain would
    /// leave that undiscovered until a guest boot.
    fn wire_monitored_ring(ring: u64, period_us: u32) -> Vec<u8> {
        let monitor = VkRingMonitorInfoMESA {
            sType: crate::venus::proto::types::VkStructureType::VK_STRUCTURE_TYPE_RING_MONITOR_INFO_MESA,
            pNext: core::ptr::null(),
            maxReportingPeriodMicroseconds: period_us,
        };
        let info = VkRingCreateInfoMESA { pNext: (&raw const monitor).cast(), ..ring_info() };
        wire_create_ring(ring, &info)
    }

    /// A monitored ring is stamped, and the stamping outlives every command that set it up.
    ///
    /// The guest's `vn_relax` clears this bit at the start of any wait and, seconds later,
    /// aborts the *guest process* if it is still clear. So a renderer that decodes the request
    /// and does nothing with it is not a renderer that runs slowly -- it is one that kills the
    /// guest a few seconds into the first cold shader compile. Nothing in replay can see this:
    /// there is no ring buffer there and no thread, so this witness is the whole of the coverage.
    #[test]
    fn a_ring_that_asks_to_be_monitored_gets_its_alive_bit_stamped() {
        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();

        assert!(
            ctx.submit(&wire_monitored_ring(7, 1_000), &mut todo, &g, &t).ran(),
            "a monitor request is part of the protocol, not an unknown chained struct"
        );

        let status = ring_info().statusOffset;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let alive =
            crate::venus::proto::types::VkRingStatusFlagBitsMESA::VK_RING_STATUS_ALIVE_BIT_MESA.0
                as u32;
        while std::time::Instant::now() < deadline {
            if t.1.load_u32(status).expect("the status word is in the mapping") & alive != 0 {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("the monitored ring was never stamped alive");
    }

    /// A reporting period of zero is a guest asking to be told it is alive every no-time. There
    /// is no reading of that which the host can serve, and no default it may quietly substitute:
    /// picking one would leave the guest with a contract it never agreed to and no way to learn
    /// that. It is refused, and the context stops, before the ring is registered anywhere.
    #[test]
    fn a_reporting_period_of_zero_is_refused() {
        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();

        assert!(!ctx.submit(&wire_monitored_ring(7, 0), &mut todo, &g, &t).ran(), "refused");
        assert!(ctx.rings.is_empty(), "and the ring it came with was never registered");
    }

    /// One transport command, encoded with whatever arguments the caller wants.
    fn wire_transport(
        cmd: impl FnOnce(&mut crate::venus::cs::Encoder<'_>),
        size: usize,
    ) -> Vec<u8> {
        let proto = crate::venus::cs::AllOfIt;
        let mut buf = vec![0u8; size];
        let mut enc = crate::venus::cs::Encoder::new(&mut buf, &proto);
        cmd(&mut enc);
        buf
    }

    fn wire_submit_vq(ring: u64, seqno: u64) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkSubmitVirtqueueSeqnoMESA_args, vn_sizeof_vkSubmitVirtqueueSeqnoMESA_args,
        };
        use super::super::proto::types::vn_command_vkSubmitVirtqueueSeqnoMESA as Args;
        let args = Args { ring, seqno, ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        wire_transport(
            |e| vn_encode_vkSubmitVirtqueueSeqnoMESA_args(e, VkFlags(0), &args),
            vn_sizeof_vkSubmitVirtqueueSeqnoMESA_args(&proto, &args),
        )
    }

    fn wire_wait_vq(seqno: u64) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkWaitVirtqueueSeqnoMESA_args, vn_sizeof_vkWaitVirtqueueSeqnoMESA_args,
        };
        use super::super::proto::types::vn_command_vkWaitVirtqueueSeqnoMESA as Args;
        let args = Args { seqno, ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        wire_transport(
            |e| vn_encode_vkWaitVirtqueueSeqnoMESA_args(e, VkFlags(0), &args),
            vn_sizeof_vkWaitVirtqueueSeqnoMESA_args(&proto, &args),
        )
    }

    fn wire_wait_ring(ring: u64, seqno: u64) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkWaitRingSeqnoMESA_args, vn_sizeof_vkWaitRingSeqnoMESA_args,
        };
        use super::super::proto::types::vn_command_vkWaitRingSeqnoMESA as Args;
        let args = Args { ring, seqno, ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        wire_transport(
            |e| vn_encode_vkWaitRingSeqnoMESA_args(e, VkFlags(0), &args),
            vn_sizeof_vkWaitRingSeqnoMESA_args(&proto, &args),
        )
    }

    fn wire_write_extra(ring: u64, offset: usize, value: u32) -> Vec<u8> {
        use super::super::proto::serialize::{
            vn_encode_vkWriteRingExtraMESA_args, vn_sizeof_vkWriteRingExtraMESA_args,
        };
        use super::super::proto::types::vn_command_vkWriteRingExtraMESA as Args;
        let args = Args { ring, offset, value, ..Default::default() };
        let proto = crate::venus::cs::AllOfIt;
        wire_transport(
            |e| vn_encode_vkWriteRingExtraMESA_args(e, VkFlags(0), &args),
            vn_sizeof_vkWriteRingExtraMESA_args(&proto, &args),
        )
    }

    /// A context with one ring, idle, so a batch can be aimed at either stream.
    fn ctx_with_ring(t: &OneShm) -> Context {
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();
        assert!(
            ctx.submit(&wire_create_ring(7, &ring_info()), &mut todo, &g, t).ran(),
            "the ring was created"
        );
        ctx
    }

    /// Every transport command is legal on exactly one of the two streams, and the wrong one is
    /// refused rather than served somewhere it would mean something else.
    ///
    /// Not a tidiness check. `vkWaitVirtqueueSeqnoMESA` names no ring at all -- it means "the ring
    /// this arrived on" -- so on the context's own stream there is nothing for it to refer to, and
    /// a build that served it anyway would be inventing a ring. The other three name a ring and
    /// reach the context's ring table, which a ring's own dispatch is inside: serving them there
    /// is the reentrancy the C's `is_dispatched_from_vkr_context` checks exist to stop. Both
    /// directions, because a check that only fired one way would leave the other silent.
    #[test]
    fn a_transport_command_on_the_wrong_stream_is_refused() {
        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();

        // Ring-only, sent on the context's own stream. Poisoned specifically, not merely "did
        // not finish": a build that took the wrong stream's command and *suspended* on it would
        // also fail a `ran()` check, while having invented a ring for a command that names none.
        let mut ctx = ctx_with_ring(&t);
        assert_eq!(
            ctx.submit(&wire_wait_vq(1), &mut todo, &g, &t),
            Submitted::Poisoned,
            "a virtqueue wait names no ring, so the context's own stream cannot send it"
        );

        // Context-only, each sent on a ring's stream.
        for (what, batch) in [
            ("a virtqueue submit", wire_submit_vq(7, 1)),
            ("a ring-seqno wait", wire_wait_ring(7, 1)),
            ("a ring extra write", wire_write_extra(7, 0, 1)),
        ] {
            let mut ctx = ctx_with_ring(&t);
            assert!(
                !ctx.submit_ring(RingId(7), &batch, &mut todo, &g, &t),
                "{what} reaches the ring table, which a ring's own dispatch is inside"
            );
        }
    }

    /// A ring-seqno wait on a ring that is not reading is refused, not slept on.
    ///
    /// Nothing advances an idle ring's head -- promotion happens at the end of the batch, and this
    /// command is inside that batch. A build that suspended here would suspend forever, holding
    /// the virtio-gpu control queue for the whole device while it did.
    #[test]
    fn a_ring_seqno_wait_on_a_ring_that_is_not_reading_is_refused() {
        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = ctx_with_ring(&t);
        assert!(!ctx.submit(&wire_wait_ring(7, 1), &mut todo, &g, &t).ran(), "refused");
    }

    /// The `extra` region is a door of a fixed size, and the offset comes from the guest at write
    /// time with no layout left to have checked it. Everything past that region is the rest of a
    /// resource the guest does not get to reach through this command.
    #[test]
    fn a_ring_extra_write_stays_inside_the_extra_region() {
        let t = ring_table();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();

        let mut ctx = ctx_with_ring(&t);
        assert!(
            ctx.submit(&wire_write_extra(7, 0, 0xfeed), &mut todo, &g, &t).ran(),
            "the one word `extra` holds is the guest's to write"
        );
        assert_eq!(
            t.1.load_u32(ring_info().extraOffset).expect("inside the mapping"),
            0xfeed,
            "and it landed where the layout says extra begins"
        );

        for (what, offset) in [("one word past the end", 4), ("far past the end", 0x1000)] {
            let mut ctx = ctx_with_ring(&t);
            assert!(
                !ctx.submit(&wire_write_extra(7, offset, 1), &mut todo, &g, &t).ran(),
                "{what} is outside the region and is refused"
            );
        }
    }

    /// Two rings, two reply streams, and they stay two.
    ///
    /// This is the witness the per-stream slot owes, and it runs through the real submission path
    /// rather than poking a handler: which slot a batch writes into is decided by which slot the
    /// caller lends, so only `submit`/`submit_ring` exercise it at all. A single context-wide slot
    /// passes every gate the corpus has -- the replay score counts commands accounted for, and one
    /// slot accounts for them all -- while quietly answering ring A's caller into ring B's buffer.
    #[test]
    fn a_reply_stream_belongs_to_the_ring_it_was_set_on() {
        let t = ring_table();
        let info = ring_info();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();

        for ring in [7u64, 9] {
            assert!(
                ctx.submit(&wire_create_ring(ring, &info), &mut todo, &g, &t).ran(),
                "ring {ring} is one we accept"
            );
        }

        // Each ring's own stream sets its own window.
        for (ring, offset) in [(7u64, 0x21000usize), (9, 0x22000)] {
            let d = reply_at(offset, 0x100);
            assert!(
                ctx.submit_ring(RingId(ring), &wire_set_reply(&d), &mut todo, &g, &t),
                "ring {ring}'s window fits its resource"
            );
        }

        let window = |ring: u64| {
            ctx.rings[&RingId(ring)]
                .idle()
                .reply
                .as_ref()
                .expect("the ring was given a stream")
                .window()
        };
        assert_eq!(window(7).begin(), 0x21000, "ring 7 kept the window ring 7 set");
        assert_eq!(window(9).begin(), 0x22000, "ring 9 kept the window ring 9 set");
        assert!(
            ctx.reply.is_none(),
            "nothing arrived on the context's own stream, so it has no reply window"
        );
    }

    /// The same command on the context's own stream lands on the context and leaves the rings
    /// alone -- the other half of the routing.
    #[test]
    fn a_reply_stream_set_off_a_ring_belongs_to_the_context() {
        let t = ring_table();
        let info = ring_info();
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();

        assert!(ctx.submit(&wire_create_ring(7, &info), &mut todo, &g, &t).ran());

        let d = reply_at(0x21000, 0x100);
        assert!(ctx.submit(&wire_set_reply(&d), &mut todo, &g, &t).ran());

        assert_eq!(
            ctx.reply.as_ref().expect("the context was given a stream").window().begin(),
            0x21000
        );
        assert!(
            ctx.rings[&RingId(7)].idle().reply.is_none(),
            "the ring was not the one that asked"
        );
    }

    /// Setting again is how the guest rewinds: it re-establishes the stream before each batch it
    /// expects answers for, so a second set must move the window rather than be refused as a
    /// duplicate.
    #[test]
    fn setting_a_reply_stream_again_moves_it() {
        use super::super::proto::types::vn_command_vkSetReplyCommandStreamMESA as SetReply;

        let t = ring_table();
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(None));
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;

        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &t,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        for offset in [0x21000usize, 0x22000] {
            let d = reply_at(offset, 0x100);
            h.vkSetReplyCommandStreamMESA(&mut SetReply {
                pStream: Some(&d),
                ..Default::default()
            });
            assert_eq!(h.reject, None, "setting a stream at {offset:#x} is not a duplicate");
            assert_eq!(h.reply.as_ref().unwrap().window().begin(), offset);
            assert_eq!(h.reply.as_ref().unwrap().pos(), 0, "and it starts from the top");
        }
    }

    /// Every way a guest can misdescribe a reply stream.
    #[test]
    fn a_reply_stream_we_cannot_write_into_is_refused() {
        use super::super::proto::types::vn_command_vkSetReplyCommandStreamMESA as SetReply;

        let t = ring_table();
        let len = 0x24000usize; // what `ring_table` mints, and what the checks are against

        // (what the guest asked for, whether it should be taken)
        let cases: [(Option<(usize, usize)>, bool); 6] = [
            (Some((0, len)), true),             // the whole resource
            (Some((len - 4, 4)), true),         // right up to the end
            (Some((0, len + 1)), false),        // one byte more than there is
            (Some((len - 4, 8)), false),        // starts inside, ends outside
            (Some((usize::MAX, 0x100)), false), // an offset that overflows when the size is added
            (None, false),                      // no description at all
        ];

        for (asked, taken) in cases {
            let objects = Shared::new();
            let mut todo = Unimplemented::default();
            let global = crate::vulkan::global();
            let mut driver = Driver::new(Account::for_test(None));
            let mut rings = BTreeMap::new();
            let mut ctx_reply = None;
            let mut monitor = None;
            let mut jrnl = Journal::new();
            let mut h = Handlers {
                objects: &objects,
                todo: &mut todo,
                driver: &mut driver,
                global: &global,
                ctx: ContextId::new(1).expect("1 is not zero"),
                reject: None,
                resources: &t,
                rings: &mut rings,
                monitor: &mut monitor,
                wait: None,
                execute: None,
                replaying: false,
                current_ring: None,
                reply: &mut ctx_reply,
                note: None,
                journal: &mut jrnl,
            };
            let d = asked.map(|(o, s)| reply_at(o, s));
            h.vkSetReplyCommandStreamMESA(&mut SetReply {
                pStream: d.as_ref(),
                ..Default::default()
            });
            assert_eq!(
                h.reject.is_none(),
                taken,
                "{asked:x?} against a {len:#x}-byte resource: reject was {:?}",
                h.reject
            );
            assert_eq!(h.reply.is_some(), taken, "a refused stream leaves nothing behind");
        }
    }

    /// A reply stream in a resource with no host mapping is refused, the same way a ring in one is.
    #[test]
    fn a_reply_stream_in_a_resource_with_no_host_mapping_is_refused() {
        use super::super::proto::types::vn_command_vkSetReplyCommandStreamMESA as SetReply;

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(None));
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let d = reply_at(0, 0x100);
        h.vkSetReplyCommandStreamMESA(&mut SetReply { pStream: Some(&d), ..Default::default() });
        assert_eq!(h.reject, Some("set a reply stream in a resource that has no host mapping"));
        assert!(h.reply.is_none());
    }

    /// Creating or destroying a ring is the context's business. A ring asking to do it is the
    /// guest confusing its own streams, and the C makes it fatal for the same reason: a ring
    /// destroying itself mid-batch is a lifetime no one can reason about.
    #[test]
    fn a_ring_may_not_create_or_destroy_rings() {
        use super::super::proto::types::vn_command_vkCreateRingMESA as Create;
        use super::super::proto::types::vn_command_vkDestroyRingMESA as Destroy;

        let t = ring_table();
        let info = ring_info();
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(None));
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &t,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkCreateRingMESA(&mut Create { ring: 7, pCreateInfo: Some(&info), ..Default::default() });
        assert_eq!(h.reject, None, "on the context's stream, creating is fine");

        h.current_ring = Some(RingId(7));
        h.vkCreateRingMESA(&mut Create { ring: 8, pCreateInfo: Some(&info), ..Default::default() });
        assert_eq!(h.reject.take(), Some("created a ring from inside a ring's own stream"));
        h.vkDestroyRingMESA(&mut Destroy { ring: 7, ..Default::default() });
        assert_eq!(h.reject.take(), Some("destroyed a ring from inside a ring's own stream"));

        assert!(h.rings.contains_key(&RingId(7)), "the refusals changed nothing");
        assert_eq!(h.rings.len(), 1);
    }

    /// A submission for a ring nobody created fails that submission and nothing else. Poisoning
    /// here would let a guest take down every other ring it owns by racing a destroy against a
    /// submission still in flight.
    #[test]
    fn a_submission_for_a_ring_that_is_not_here_fails_without_poisoning() {
        let g = crate::vulkan::global();
        let mut todo = Unimplemented::default();
        let mut ctx = Context::new(ContextId::new(1).unwrap(), &Budget::with_cap(None, false));
        ctx.replay_begin();

        assert!(
            !ctx.submit_ring(RingId(7), &[], &mut todo, &g, &NO_RESOURCES),
            "there is no ring 7 to submit to"
        );
        assert!(!ctx.fatal(), "and the context is still usable");
        assert!(
            ctx.submit(&[], &mut todo, &g, &NO_RESOURCES).ran(),
            "so its own stream still works"
        );
    }

    /// The witness the RingId split owed: two rings whose ids differ only above bit 32 are two
    /// rings. Under the old `ring_id as u32` these collapsed into one, and the second create
    /// would have found the first already there.
    #[test]
    fn two_ring_ids_differing_only_above_bit_32_are_two_rings() {
        use super::super::proto::types::vn_command_vkCreateRingMESA as Create;

        let t = ring_table();
        let info = ring_info();
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(None));
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;

        let low: u64 = 0x0000_0001_dead_beef;
        let high: u64 = 0x0000_0002_dead_beef;
        assert_eq!(low as u32, high as u32, "the two ids are identical in their low word");

        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &t,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        for ring in [low, high] {
            let mut args = Create { ring, pCreateInfo: Some(&info), ..Default::default() };
            h.vkCreateRingMESA(&mut args);
            assert_eq!(h.reject, None, "ring {ring:#x} is a layout we accept");
        }
        assert_eq!(rings.len(), 2, "two ids, two rings");
        assert!(rings.contains_key(&RingId(low)) && rings.contains_key(&RingId(high)));
    }

    /// A ring id the guest is already using is the guest contradicting itself. Refusing rather
    /// than replacing is deliberate and stricter than the C, which appends the duplicate and lets
    /// `lookup_ring` silently return whichever it finds first -- so a destroy tears down one ring
    /// while the other keeps a share of the mapping that nothing can reach any more.
    #[test]
    fn a_ring_id_already_in_use_is_refused_and_the_first_ring_survives() {
        use super::super::proto::types::vn_command_vkCreateRingMESA as Create;
        use super::super::proto::types::vn_command_vkDestroyRingMESA as Destroy;

        let t = ring_table();
        let info = ring_info();
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(None));
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &t,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let mut first = Create { ring: 7, pCreateInfo: Some(&info), ..Default::default() };
        h.vkCreateRingMESA(&mut first);
        assert_eq!(h.reject, None);
        h.rings[&RingId(7)].idle().set_head(0x1234);

        let mut again = Create { ring: 7, pCreateInfo: Some(&info), ..Default::default() };
        h.vkCreateRingMESA(&mut again);
        assert!(h.reject.is_some(), "the second create under the same id is refused");
        assert_eq!(h.rings.len(), 1, "and did not add a second entry");
        assert_eq!(
            h.rings[&RingId(7)].idle().map.load_u32(0),
            Some(0x1234),
            "the ring that was already there is untouched"
        );

        // And destroying it is the only thing that removes it.
        h.reject = None;
        let mut gone = Destroy { ring: 7, ..Default::default() };
        h.vkDestroyRingMESA(&mut gone);
        assert_eq!(h.reject, None);
        assert!(h.rings.is_empty());

        let mut twice = Destroy { ring: 7, ..Default::default() };
        h.vkDestroyRingMESA(&mut twice);
        assert!(h.reject.is_some(), "destroying a ring that is not there is refused");
    }

    /// A budget refusal has to stop the context, and this handler is the only place that can say
    /// so. The driver knows it refused and the loop knows how to poison; between them is one line
    /// in `vkAllocateMemory`, and without it the guest is handed `VK_ERROR_OUT_OF_DEVICE_MEMORY`
    /// -- which venus never reads, because `vn_device_memory_alloc_simple` returns `VK_SUCCESS`
    /// as soon as the command is on the ring. The guest would carry on with a handle to memory
    /// that does not exist and poison its ring several commands later, on whatever touched it.
    ///
    /// The counter is here for the same reason as in the driver's own witness: a refusal that
    /// arrives after the allocation costs exactly the memory the cap exists to save.
    #[test]
    fn an_allocation_the_budget_refuses_stops_the_context() {
        use super::super::proto::types::{
            VkAllocationCallbacks, VkDeviceSize, VkMemoryAllocateInfo, VkMemoryPropertyFlags,
            VkStructureType, vn_command_vkAllocateMemory as Alloc,
        };
        use std::cell::Cell;

        const DEVICE: u64 = 3;
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

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(Some(CAP)));
        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateMemory(allocate);
        driver.plant_device(VkDevice(DEVICE), fns);
        // Not host-visible, so nothing is padded and the cap is measured in the guest's numbers.
        driver.plant_memory_types(VkDevice(DEVICE), &[VkMemoryPropertyFlags(0)]);
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let info = VkMemoryAllocateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: core::ptr::null(),
            allocationSize: VkDeviceSize(SIZE),
            memoryTypeIndex: 0,
        };

        let mut first_out = VkDeviceMemory(0x6001);
        let mut first = Alloc::default();
        first.device = VkDevice(DEVICE);
        first.pAllocateInfo = Some(&info);
        first.plant_pMemory(&mut first_out);
        h.vkAllocateMemory(&mut first);
        assert_eq!(h.reject, None, "the first fits under the cap");
        assert_eq!(first.ret, VkResult::VK_SUCCESS);
        assert_eq!(ASKED.with(Cell::get), 1);

        let mut second_out = VkDeviceMemory(0x6002);
        let mut second = Alloc::default();
        second.device = VkDevice(DEVICE);
        second.pAllocateInfo = Some(&info);
        second.plant_pMemory(&mut second_out);
        h.vkAllocateMemory(&mut second);
        assert!(
            h.reject.is_some(),
            "the second is over the cap, and the guest will never read the error it was given"
        );
        assert_eq!(
            second.ret,
            VkResult::VK_ERROR_OUT_OF_DEVICE_MEMORY,
            "which is still set, for the guest configured to wait for it"
        );
        assert_eq!(ASKED.with(Cell::get), 1, "and the driver was never asked");

        h.driver.abandon_planted();
    }

    /// A ring in a resource the host cannot address is refused, not quietly skipped. This is the
    /// case that took the whole corpus down until blobs got their memory: without a mapping there
    /// is nothing to read commands out of, and pretending otherwise would report success for a
    /// ring that can never deliver a command.
    #[test]
    fn a_ring_in_a_resource_with_no_host_mapping_is_refused() {
        use super::super::proto::types::vn_command_vkCreateRingMESA as Create;

        let info = ring_info();
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(None));
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let mut args = Create { ring: 1, pCreateInfo: Some(&info), ..Default::default() };
        h.vkCreateRingMESA(&mut args);
        assert!(h.reject.is_some());
        assert!(rings.is_empty(), "and nothing was registered");
    }

    /// A ring with no description at all. The guest sent the command; the decoder gave us no
    /// struct, so there is nothing to validate and nothing to create.
    #[test]
    fn a_ring_create_with_no_description_is_refused() {
        use super::super::proto::types::vn_command_vkCreateRingMESA as Create;

        let t = ring_table();
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(None));
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &t,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let mut args = Create { ring: 1, pCreateInfo: None, ..Default::default() };
        h.vkCreateRingMESA(&mut args);
        assert!(h.reject.is_some());
        assert!(rings.is_empty());
    }

    /// Nothing in this file's handlers reaches for `unsafe`, and this is what keeps it that way.
    ///
    /// Handlers adjudicate the guest's decisions; a raw pointer in one is the generator having
    /// handed it the wrong thing, so the fix belongs in the template and never in a block here.
    /// That was a rule written down and then broken five times, always the same way -- an
    /// out-parameter dereferenced because there was no other way to answer the guest. There is
    /// one now: the command structs' single-value members are shut away behind generated
    /// accessors, the way their arrays already were.
    ///
    /// Counting the source is a blunt instrument and deliberately so. The alternative is trusting
    /// the next person to remember, which is what produced the five.
    #[test]
    fn no_handler_here_reaches_for_unsafe() {
        let src = include_str!("context.rs");
        let (handlers, _tests) = src
            .split_once("#[cfg(test)]\nmod tests")
            .expect("this file ends in its own test module");
        let n = handlers.matches("unsafe").count();
        assert_eq!(n, 0, "venus/context.rs is not on the list of modules allowed unsafe");
    }

    /// The third door, and the only one with no count to reconcile: a NUL-terminated string
    /// carries its length in its own bytes, and the decoder writes the terminator rather than
    /// trusting the guest to have. `has_` and `None` are the same question here as everywhere --
    /// `vkEnumerateDeviceExtensionProperties` reads an absent `pLayerName` as "no layer", which
    /// is not the same request as an empty one.
    #[test]
    fn a_string_arrives_as_a_string_and_absent_is_not_empty() {
        use super::super::proto::types::vn_command_vkEnumerateDeviceExtensionProperties;

        let mut args = vn_command_vkEnumerateDeviceExtensionProperties::default();
        assert!(!args.has_pLayerName(), "no layer named");
        assert!(args.pLayerName().is_none(), "and absent is not the empty layer name");

        let name = c"VK_LAYER_KHRONOS_validation";
        args.plant_pLayerName(name);
        assert!(args.has_pLayerName());
        assert_eq!(args.pLayerName(), Some(name), "the string, terminator and all");
    }

    /// The mechanism the sixteen queries all rest on, tested once on the one that shows it best.
    ///
    /// A query is three lines in this file because the interesting work is a memory contract, not
    /// code: the decoder allocates the guest's struct from the arena at the layout a C compiler
    /// agrees with, so the driver fills the very bytes the reply encoder will read back. The
    /// chained `pNext` structs come along for free -- nobody copies them, nobody re-links them.
    /// That is the claim, and a chain is the only way to see it, because an unchained struct would
    /// pass just as well if the handler had quietly filled a copy.
    #[test]
    fn a_query_fills_the_guest_s_chain_where_it_lies() {
        use super::super::proto::types::{
            VkBool32, VkPhysicalDevice, VkPhysicalDeviceFeatures2,
            VkPhysicalDeviceVulkan11Features, VkStructureType,
            vn_command_vkGetPhysicalDeviceFeatures2,
        };

        const PD: VkPhysicalDevice = VkPhysicalDevice(0x9001);

        /// A driver that answers through the chain, which is what a real one does.
        unsafe extern "C" fn features(pd: VkPhysicalDevice, out: *mut VkPhysicalDeviceFeatures2) {
            assert_eq!(pd, PD, "the handle the guest named reaches the driver");
            // SAFETY: the handler passed an exclusive borrow of a live struct.
            let out = unsafe { &mut *out };
            out.features.geometryShader = VkBool32(1);
            assert_eq!(out.sType, VkStructureType::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
            // SAFETY: the guest chained one struct, and it is live for this call.
            let link = unsafe { &mut *(out.pNext as *mut VkPhysicalDeviceVulkan11Features) };
            assert_eq!(
                link.sType,
                VkStructureType::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                "the chain arrives with the guest's own sType, not a rebuilt one"
            );
            link.multiview = VkBool32(1);
        }

        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkGetPhysicalDeviceFeatures2(features);
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(fns);

        let mut link = VkPhysicalDeviceVulkan11Features {
            sType: VkStructureType::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            ..Default::default()
        };
        let mut asked = VkPhysicalDeviceFeatures2 {
            sType: VkStructureType::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            pNext: &raw mut link as *mut _,
            ..Default::default()
        };

        let mut args = vn_command_vkGetPhysicalDeviceFeatures2::default();
        args.physicalDevice = PD;
        args.plant_pFeatures(&mut asked);

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetPhysicalDeviceFeatures2(&mut args);
        assert!(h.reject.is_none(), "a query this driver can answer is not refused");

        assert_eq!(
            asked.features.geometryShader,
            VkBool32(1),
            "the struct the guest sent was filled"
        );
        assert_eq!(link.multiview, VkBool32(1), "and so was the one it chained behind it");

        driver.abandon_planted();
    }

    /// The guest is told what this build can serialize, not what the driver can do.
    ///
    /// A driver newer than the pinned vk.xml is the normal case, not an exotic one, and passing
    /// its version straight through is an invitation the renderer cannot honour: the guest enables
    /// that version, sends a struct from it, and the decoder poisons the ring for asking. The
    /// patch level is the driver's own and stays -- it names the implementation and carries no
    /// structs with it.
    #[test]
    fn a_driver_newer_than_the_protocol_is_capped_to_the_protocol() {
        use super::super::proto::info::VK_XML_VERSION;

        const MINOR: u32 = 12;
        let make = |major: u32, minor: u32, patch: u32| (major << 22) | (minor << MINOR) | patch;
        let (major, minor) = (VK_XML_VERSION >> 22, (VK_XML_VERSION >> MINOR) & 0x3ff);

        let newer = make(major, minor + 1, 77);
        assert_eq!(
            cap_api_version(newer),
            make(major, minor, 77),
            "a driver a minor version ahead is capped, and keeps its own patch level"
        );

        let older = make(major, minor - 1, 3);
        assert_eq!(cap_api_version(older), older, "a driver behind us is reported as it is");
        assert_eq!(cap_api_version(VK_XML_VERSION), VK_XML_VERSION, "our own version is untouched");
    }

    /// The cap is not a fact about the handler, it is a fact about the reply, so it is watched
    /// where the guest would read it: in the struct the driver filled.
    ///
    /// Both spellings of the query are driven, because both carry the cap and a guest picks
    /// whichever its instance version allows. They are the same three lines today, which is
    /// exactly why one of them can lose the cap in a refactor and stay green on its own.
    #[test]
    fn the_properties_query_caps_the_version_the_driver_reported() {
        use super::super::proto::info::VK_XML_VERSION;
        use super::super::proto::types::{
            VkPhysicalDevice, VkPhysicalDeviceProperties, VkPhysicalDeviceProperties2,
            vn_command_vkGetPhysicalDeviceProperties, vn_command_vkGetPhysicalDeviceProperties2,
        };

        /// A driver from the future, which is what every driver eventually is.
        unsafe extern "C" fn properties(
            _pd: VkPhysicalDevice,
            out: *mut VkPhysicalDeviceProperties,
        ) {
            // SAFETY: the handler passed an exclusive borrow of a live struct.
            let out = unsafe { &mut *out };
            out.apiVersion = ((VK_XML_VERSION >> 12) + 1) << 12 | 99;
            out.driverVersion = 0xabcd;
        }

        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkGetPhysicalDeviceProperties(properties);
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(fns);

        let mut asked = VkPhysicalDeviceProperties::default();
        let mut args = vn_command_vkGetPhysicalDeviceProperties::default();
        args.plant_pProperties(&mut asked);

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetPhysicalDeviceProperties(&mut args);
        assert!(h.reject.is_none());

        assert_eq!(
            asked.apiVersion,
            (VK_XML_VERSION & !0xfff) | 99,
            "the guest reads this build's ceiling, with the driver's own patch level"
        );
        assert_eq!(asked.driverVersion, 0xabcd, "and everything else the driver said is untouched");

        /// The same driver, reached through the `2` spelling: the version sits one struct deeper.
        unsafe extern "C" fn properties2(
            _pd: VkPhysicalDevice,
            out: *mut VkPhysicalDeviceProperties2,
        ) {
            // SAFETY: the handler passed an exclusive borrow of a live struct.
            let out = unsafe { &mut *out };
            out.properties.apiVersion = ((VK_XML_VERSION >> 12) + 1) << 12 | 99;
            out.properties.driverVersion = 0xabcd;
        }

        let mut fns2 = crate::vulkan::Instance::default();
        fns2.plant_vkGetPhysicalDeviceProperties2(properties2);
        driver.abandon_planted();
        driver.plant_instance(fns2);

        let mut asked2 = VkPhysicalDeviceProperties2::default();
        let mut args2 = vn_command_vkGetPhysicalDeviceProperties2::default();
        args2.plant_pProperties(&mut asked2);

        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetPhysicalDeviceProperties2(&mut args2);
        assert!(h.reject.is_none());

        assert_eq!(
            asked2.properties.apiVersion,
            (VK_XML_VERSION & !0xfff) | 99,
            "the `2` spelling caps it in the nested struct, where the guest reads it"
        );
        assert_eq!(
            asked2.properties.driverVersion, 0xabcd,
            "and it too leaves the rest of the driver's answer alone"
        );

        driver.abandon_planted();
    }

    /// The one query answered off the global table, before any instance exists. It reaches the
    /// real loader, so there is no driver to plant and the cap cannot be forced: measured
    /// 2026-09-01, the loader here reports 4211029 against this build's 4211045, which leaves the
    /// ceiling check true for a reason that has nothing to do with the handler. It is kept as the
    /// invariant it states, not as a witness; what this test actually holds down is the second
    /// half, that the handler asks the driver *before* checking the guest left it somewhere to
    /// answer -- an ordering a refactor reverses without noticing.
    #[test]
    fn the_instance_version_query_is_capped_and_needs_somewhere_to_answer() {
        use super::super::proto::info::VK_XML_VERSION;
        use super::super::proto::types::vn_command_vkEnumerateInstanceVersion as Cmd;

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut driver = Driver::new(Account::for_test(None));

        let mut args = Cmd::default();
        let mut out = 0u32;
        args.plant_pApiVersion(&mut out);

        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkEnumerateInstanceVersion(&mut args);
        assert!(h.reject.is_none(), "a real loader answering is never a reason to poison a ring");
        let ret = args.ret;
        if ret == VkResult::VK_SUCCESS {
            assert!(
                out <= VK_XML_VERSION,
                "whatever the loader reports, the guest never hears a version past this build's"
            );
        }

        // No `pApiVersion` at all: nothing to fill, and that is the guest's fault, not the host's.
        let mut empty = Cmd::default();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkEnumerateInstanceVersion(&mut empty);
        assert!(h.reject.is_some(), "a query with nowhere to answer is refused, not answered");
    }

    /// A host handle must never reach a guest, and this is the one query that hands them back
    /// somewhere no shadow can be put: inside a fixed array, inside an out-struct the driver
    /// filled. Every other command names its objects in a member the generator can shadow.
    ///
    /// The test plants a driver whose handles are deliberately nothing like the ids -- equal
    /// numbers would let a handler that never swaps pass.
    #[test]
    fn a_device_group_reaches_the_guest_as_ids_and_never_as_host_handles() {
        use super::super::proto::types::{
            VkInstance, VkPhysicalDeviceGroupProperties,
            vn_command_vkEnumeratePhysicalDeviceGroups as Cmd,
        };

        const INSTANCE: VkInstance = VkInstance(0x5000);
        const HOSTS: [u64; 2] = [0xfeed_0001, 0xfeed_0002];
        const IDS: [u64; 2] = [11, 12];

        unsafe extern "C" fn groups(
            _i: VkInstance,
            count: *mut u32,
            out: *mut VkPhysicalDeviceGroupProperties,
        ) -> VkResult {
            // SAFETY: the driver passed a live count, and an array of that length or null.
            let count = unsafe { &mut *count };
            if out.is_null() {
                *count = 1;
                return VkResult::VK_SUCCESS;
            }
            // SAFETY: `count` is the length the caller sized the array to.
            let out = unsafe { core::slice::from_raw_parts_mut(out, *count as usize) };
            out[0].physicalDeviceCount = 2;
            out[0].physicalDevices[0] = VkPhysicalDevice(HOSTS[0]);
            out[0].physicalDevices[1] = VkPhysicalDevice(HOSTS[1]);
            *count = 1;
            VkResult::VK_SUCCESS
        }

        let objects = Shared::new();
        for (id, host) in IDS.iter().zip(HOSTS) {
            objects
                .borrow_mut()
                .add(
                    ObjectId(*id),
                    VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE,
                    HostHandle(host),
                    None,
                )
                .unwrap();
        }

        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkEnumeratePhysicalDeviceGroups(groups);
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(fns);
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();

        let mut props = [VkPhysicalDeviceGroupProperties::default(); 1];
        let mut n = 1u32;
        let mut args = Cmd::default();
        args.instance = INSTANCE;
        args.plant_pPhysicalDeviceGroupCount(&mut n);
        args.plant_pPhysicalDeviceGroupProperties(&mut props);

        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkEnumeratePhysicalDeviceGroups(&mut args);
        assert!(h.reject.is_none());
        assert_eq!(args.ret, VkResult::VK_SUCCESS);
        assert_eq!(
            [props[0].physicalDevices[0].0, props[0].physicalDevices[1].0],
            IDS,
            "the guest reads back the ids it chose, not the handles the driver returned"
        );

        // A handle the guest has no name for: it never enumerated, so there is nothing honest to
        // hand it. Half a swapped group would be worse than none -- the guest cannot tell which
        // half is which.
        objects.borrow_mut().remove(ObjectId(IDS[1]));
        let mut props = [VkPhysicalDeviceGroupProperties::default(); 1];
        let mut n = 1u32;
        let mut args = Cmd::default();
        args.instance = INSTANCE;
        args.plant_pPhysicalDeviceGroupCount(&mut n);
        args.plant_pPhysicalDeviceGroupProperties(&mut props);
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkEnumeratePhysicalDeviceGroups(&mut args);
        assert_eq!(
            args.ret,
            VkResult::VK_ERROR_INITIALIZATION_FAILED,
            "a group this guest has no names for is refused, not half-translated"
        );

        driver.abandon_planted();
    }

    /// The guest is told what this build can serialize, and never what the driver merely has.
    ///
    /// Three of the extensions this renderer puts on its own devices are host-side machinery --
    /// Metal interop, the portability subset -- with no wire encoding at all. Advertising one
    /// would have the guest enable it, send one of its structs, and get its ring poisoned for
    /// doing exactly what it was told it could.
    #[test]
    fn the_guest_is_told_only_the_extensions_this_build_can_serialize() {
        use super::super::proto::info;
        use super::super::proto::types::{VkExtensionProperties, VkPhysicalDevice};

        const PD: VkPhysicalDevice = VkPhysicalDevice(7);

        // What a driver hands back: two the protocol knows, two it does not.
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_extensions(
            PD,
            &[
                "VK_KHR_external_memory_fd",
                "VK_EXT_metal_objects",
                "VK_EXT_external_memory_dma_buf",
                "VK_KHR_portability_subset",
            ],
        );

        let advertised = driver.advertised_extensions(PD);
        let names: Vec<String> = advertised
            .iter()
            .map(|e| {
                e.extensionName.iter().take_while(|c| **c != 0).map(|c| *c as u8 as char).collect()
            })
            .collect();
        assert_eq!(
            names,
            ["VK_EXT_external_memory_dma_buf", "VK_KHR_external_memory_fd"],
            "the host's own Metal and portability extensions are not the guest's business"
        );
        assert_eq!(
            advertised[0].specVersion,
            info::spec_version("VK_EXT_external_memory_dma_buf"),
            "and the version is the one whose structs this decoder knows"
        );
        assert_ne!(advertised[0].specVersion, 0);

        // A physical device nobody enumerated has nothing to say, rather than a panic.
        assert!(driver.advertised_extensions(VkPhysicalDevice(999)).is_empty());
        let _ = VkExtensionProperties::default();

        driver.abandon_planted();
    }

    /// A driver that will not say what a device supports is not a device that supports nothing.
    ///
    /// `learn_extensions` used to swallow the failure and record nothing, and nothing downstream
    /// could tell that from an answer. The guest would then be told the device had no extensions
    /// at all, and `vkCreateDevice` would quietly strip every extension it asked for -- a device
    /// that comes back VK_SUCCESS and cannot do what the guest built it to do.
    #[test]
    fn a_device_whose_extensions_could_not_be_read_is_not_a_device_with_none() {
        use super::super::proto::types::{VkExtensionProperties, VkPhysicalDevice};

        const PD: VkPhysicalDevice = VkPhysicalDevice(7);

        unsafe extern "C" fn refuse(
            _pd: VkPhysicalDevice,
            _layer: *const core::ffi::c_char,
            _count: *mut u32,
            _out: *mut VkExtensionProperties,
        ) -> VkResult {
            VkResult::VK_ERROR_OUT_OF_HOST_MEMORY
        }

        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkEnumerateDeviceExtensionProperties(refuse);
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(fns);

        assert_eq!(
            driver.learn_extensions(PD),
            Err(VkResult::VK_ERROR_OUT_OF_HOST_MEMORY),
            "the driver's refusal is the answer, not an empty list"
        );
        assert!(
            driver.advertised_extensions(PD).is_empty(),
            "and nothing was recorded that a later query could mistake for one"
        );

        // A driver with no instance behind it cannot be asked at all, which is the same failure.
        assert!(Driver::new(Account::for_test(None)).learn_extensions(PD).is_err());

        driver.abandon_planted();
    }

    /// A layer is host-side software a venus guest cannot see and this renderer does not load, so
    /// naming one is not a request that can be honoured or a mistake worth smoothing over.
    #[test]
    fn naming_a_layer_is_refused_rather_than_ignored() {
        use super::super::proto::types::{
            VkPhysicalDevice, vn_command_vkEnumerateDeviceExtensionProperties as Cmd,
        };

        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_extensions(VkPhysicalDevice(1), &["VK_KHR_external_memory_fd"]);
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();

        let mut n = 0u32;
        let mut args = Cmd::default();
        args.physicalDevice = VkPhysicalDevice(1);
        args.plant_pPropertyCount(&mut n);
        args.plant_pLayerName(c"VK_LAYER_KHRONOS_validation");

        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkEnumerateDeviceExtensionProperties(&mut args);
        assert!(h.reject.is_some(), "a layer this renderer has no way to load");
        assert_eq!(n, 0, "and nothing was answered");

        driver.abandon_planted();
    }

    /// Vulkan's two-call enumeration, both halves, and the ordering the second half forces.
    ///
    /// The count is written *after* the array is filled, and not by preference: it lives in the
    /// same struct the array was borrowed from, so the borrow checker will not hold both -- which
    /// is the right answer, because how many there are is not known until the driver has said.
    #[test]
    fn an_enumeration_answers_the_count_call_and_the_fill_call() {
        use super::super::proto::types::{
            VkPhysicalDevice, VkQueueFamilyProperties2,
            vn_command_vkGetPhysicalDeviceQueueFamilyProperties2 as Cmd,
        };

        const PD: VkPhysicalDevice = VkPhysicalDevice(0x33);
        const FAMILIES: u32 = 3;

        unsafe extern "C" fn families(
            _pd: VkPhysicalDevice,
            count: *mut u32,
            out: *mut VkQueueFamilyProperties2,
        ) {
            // SAFETY: the driver passed a live count, and an array of that length or null.
            let count = unsafe { &mut *count };
            if out.is_null() {
                *count = FAMILIES;
                return;
            }
            // SAFETY: `count` is the length the caller sized the array to.
            let out = unsafe { core::slice::from_raw_parts_mut(out, *count as usize) };
            for (i, f) in out.iter_mut().enumerate() {
                f.queueFamilyProperties.queueCount = i as u32 + 10;
            }
            *count = FAMILIES.min(*count);
        }

        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkGetPhysicalDeviceQueueFamilyProperties2(families);
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(fns);

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();

        // The count call: a count member, no array behind it.
        let mut n = 0u32;
        let mut args = Cmd::default();
        args.physicalDevice = PD;
        args.plant_pQueueFamilyPropertyCount(&mut n);
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetPhysicalDeviceQueueFamilyProperties2(&mut args);
        assert!(h.reject.is_none());
        assert_eq!(n, FAMILIES, "a null array is the guest asking how many there are");

        // The fill call, with room to spare on purpose. The count the guest reads back has to be
        // the driver's answer and not the size it asked with, so the two are kept different --
        // equal, and a handler that never writes the count back would pass this.
        let mut props = [VkQueueFamilyProperties2::default(); 4];
        let mut n = 4u32;
        let mut args = Cmd::default();
        args.physicalDevice = PD;
        args.plant_pQueueFamilyPropertyCount(&mut n);
        args.plant_pQueueFamilyProperties(&mut props);
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetPhysicalDeviceQueueFamilyProperties2(&mut args);
        assert!(h.reject.is_none());
        assert_eq!(n, FAMILIES, "the driver's count reaches the guest, not the size it asked with");
        assert_eq!(
            props.map(|p| p.queueFamilyProperties.queueCount),
            [10, 11, 12, 13],
            "and the guest's own array is where the driver wrote"
        );

        driver.abandon_planted();
    }

    /// Where a command has a field designed to say "no", that field is the answer and refusal is
    /// not. A guest asking for an extension this driver lacks has asked a fair question; poisoning
    /// its ring answers a different one.
    #[test]
    fn a_query_with_a_ret_reports_the_refusal_instead_of_poisoning() {
        use super::super::proto::types::{
            VkImage, VkImageDrmFormatModifierPropertiesEXT,
            vn_command_vkGetImageDrmFormatModifierPropertiesEXT,
        };

        const DEVICE: VkDevice = VkDevice(0x700);

        // A device whose table has no `VK_EXT_image_drm_format_modifier` in it.
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(DEVICE, crate::vulkan::Device::default());

        let mut props = VkImageDrmFormatModifierPropertiesEXT::default();
        let mut args = vn_command_vkGetImageDrmFormatModifierPropertiesEXT::default();
        args.device = DEVICE;
        args.image = VkImage(1);
        args.plant_pProperties(&mut props);

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetImageDrmFormatModifierPropertiesEXT(&mut args);

        assert!(h.reject.is_none(), "a fair question is not a poisoned ring");
        assert_eq!(
            args.ret,
            VkResult::VK_ERROR_EXTENSION_NOT_PRESENT,
            "and the answer goes back where the command has room for it"
        );

        driver.abandon_planted();
    }

    /// The two ways a query cannot be carried out, and neither may pass for success. A reply built
    /// from an unfilled struct is the guest reading its own zeroes back as the host's answer.
    #[test]
    fn a_query_that_cannot_be_answered_is_refused() {
        use super::super::proto::types::{
            VkPhysicalDeviceFeatures2, VkStructureType, vn_command_vkGetPhysicalDeviceFeatures2,
        };

        // No struct to answer into.
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(crate::vulkan::Instance::default());
        let mut args = vn_command_vkGetPhysicalDeviceFeatures2::default();
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetPhysicalDeviceFeatures2(&mut args);
        assert!(h.reject.is_some(), "a query with nowhere to put the answer");

        // A struct, but a driver with no such entry point. The guest can steer this one, so it is
        // a refusal and not the panic the advertised-command accessor would raise.
        let mut asked = VkPhysicalDeviceFeatures2 {
            sType: VkStructureType::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            ..Default::default()
        };
        let mut args = vn_command_vkGetPhysicalDeviceFeatures2::default();
        args.plant_pFeatures(&mut asked);
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        h.vkGetPhysicalDeviceFeatures2(&mut args);
        assert!(h.reject.is_some(), "a query this driver does not export");

        driver.abandon_planted();
    }

    #[test]
    fn a_blob_is_as_long_as_the_count_beside_it() {
        use super::super::proto::types::vn_command_vkCmdPushConstants;

        // A blob is the array wall's case with the element type left unsaid: vk.xml calls it
        // `void`, the wire counts it in bytes. So the door has to hand back exactly the bytes
        // the count claims -- planting more than the count says must not widen the slice.
        let bytes = [0xde_u8, 0xad, 0xbe, 0xef, 0x11, 0x22];
        let mut args = vn_command_vkCmdPushConstants::default();
        assert!(!args.has_pValues(), "a blob the guest never sent");
        args.size = 4;
        assert!(args.pValues().is_none(), "a count with no blob behind it is refused, not emptied");
        args.plant_pValues(&bytes);
        assert_eq!(args.pValues(), Some(&bytes[..]), "the plant sets the count and the pointer");
        args.size = 4;
        assert_eq!(args.pValues(), Some(&bytes[..4]), "the count measures it, not the pointer");
    }

    /// The pairing the whole shadow mechanism exists for: the guest names an object by an id it
    /// chose, and the host knows it by a handle the driver chose. Until a handler runs the two are
    /// the same number, which is exactly why a test that leaves them equal proves nothing -- this
    /// one plants a handle that is not the id, and then destroys by id.
    ///
    /// Without the shadows the destroy reads the target member, which the lookup has already
    /// replaced with the host handle, and removes nothing.
    #[test]
    fn an_object_is_registered_by_id_and_found_by_id_when_the_handle_differs() {
        use super::super::cs::{Lookup, Objects};
        use super::super::proto::types::{VkFence, VkStructureType, vn_command_vkCreateFence};

        const HOST: u64 = 0xfeed_face_0000_0001;
        const DEVICE: u64 = 9;
        const FENCE: u64 = 4;

        /// What a real handler will look like: it writes the shadow, never the wire member, and
        /// the pairing it registers is the one the generator hands it.
        struct Driver<'a> {
            objects: &'a Shared,
        }
        impl Commands for Driver<'_> {
            fn unsupported(&mut self, _cmd: VkCommandTypeEXT) {}

            fn vkCreateFence(&mut self, args: &mut vn_command_vkCreateFence<'_>) {
                let out = args.handle_pFence_mut().expect("the decoder owes a place to write");
                *out = VkFence(HOST);
            }

            fn object_created(
                &mut self,
                ty: VkObjectType,
                id: ObjectId,
                host: HostHandle,
                owner: Option<ObjectId>,
            ) {
                self.objects.borrow_mut().add(id, ty, host, owner).expect("a fresh id");
            }

            fn object_destroyed(&mut self, _ty: VkObjectType, id: ObjectId) {
                self.objects.borrow_mut().remove(id).expect("a live id");
            }
        }

        fn run(h: &mut Driver<'_>, objects: &Shared, wire: &[u8]) {
            let temp = Bump::new();
            let hard = AtomicBool::new(false);
            let mut dec = Decoder::new(wire, &temp, objects, &hard);
            let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
            let _flags = dec.decode_scalar::<VkFlags>();
            assert_eq!(vn_dispatch_command(&mut dec, None, cmd, h), Dispatched::Served);
            assert!(!dec.fatal(), "the command must decode");
            assert_eq!(dec.pos(), wire.len(), "the command must be fully consumed");
        }

        let objects = Shared::new();
        objects
            .borrow_mut()
            .add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(1), None)
            .unwrap();
        let mut h = Driver { objects: &objects };

        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkCreateFence_EXT, 0);
        w.extend_from_slice(&DEVICE.to_le_bytes());
        w.extend_from_slice(&1u64.to_le_bytes()); // pCreateInfo: present
        w.extend_from_slice(
            &(VkStructureType::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO.0).to_le_bytes(),
        );
        w.extend_from_slice(&0u64.to_le_bytes()); // pNext: absent
        w.extend_from_slice(&0u32.to_le_bytes()); // flags
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent
        w.extend_from_slice(&1u64.to_le_bytes()); // pFence: present
        w.extend_from_slice(&FENCE.to_le_bytes()); // the id the guest chose
        run(&mut h, &objects, &w);

        // Registered under the guest's id, holding the driver's handle. Neither half swapped.
        assert_eq!(
            objects.lookup(ObjectId(FENCE), VkObjectType::VK_OBJECT_TYPE_FENCE.0),
            Lookup::Found(HostHandle(HOST))
        );
        assert_eq!(
            objects.lookup(ObjectId(HOST), VkObjectType::VK_OBJECT_TYPE_FENCE.0),
            Lookup::Missing,
            "a host handle is not an id the guest may name"
        );

        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyFence_EXT, 0);
        w.extend_from_slice(&DEVICE.to_le_bytes());
        w.extend_from_slice(&FENCE.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent
        run(&mut h, &objects, &w);

        assert_eq!(
            objects.lookup(ObjectId(FENCE), VkObjectType::VK_OBJECT_TYPE_FENCE.0),
            Lookup::Missing,
            "the destroy names the guest id, so it must have found it"
        );
    }

    /// A create the driver refused must stay refused.
    ///
    /// The generated hook runs for every create, served or not, and its only evidence is the
    /// shadow -- which a refusal leaves zero, exactly as an unserved command does. If the hook
    /// treats that as "nobody decided" it registers the id as its own handle, and the ghost the
    /// handler just recorded is gone: the guest holds a device the driver never made, and every
    /// command behind it resolves to a handle pointing at nothing.
    #[test]
    fn a_create_the_driver_refused_leaves_a_ghost_and_not_an_object() {
        use super::super::cs::{Lookup, Objects};
        use super::super::proto::types::VkStructureType;

        const PHYSICAL_DEVICE: u64 = 3;
        const DEVICE: u64 = 11;

        let objects = Shared::new();
        objects
            .borrow_mut()
            .add(
                ObjectId(PHYSICAL_DEVICE),
                VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE,
                HostHandle(PHYSICAL_DEVICE),
                None,
            )
            .unwrap();

        // A driver with no instance refuses every device without reaching Vulkan, which is the
        // refusal this test wants: the interesting half is what happens after the `Err`.
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkCreateDevice_EXT, 0);
        w.extend_from_slice(&PHYSICAL_DEVICE.to_le_bytes());
        w.extend_from_slice(&1u64.to_le_bytes()); // pCreateInfo: present
        w.extend_from_slice(
            &(VkStructureType::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO.0).to_le_bytes(),
        );
        w.extend_from_slice(&0u64.to_le_bytes()); // pNext: absent
        w.extend_from_slice(&0u32.to_le_bytes()); // flags
        w.extend_from_slice(&0u32.to_le_bytes()); // queueCreateInfoCount
        w.extend_from_slice(&0u64.to_le_bytes()); // pQueueCreateInfos: empty
        w.extend_from_slice(&0u32.to_le_bytes()); // enabledLayerCount
        w.extend_from_slice(&0u64.to_le_bytes()); // ppEnabledLayerNames: empty
        w.extend_from_slice(&0u32.to_le_bytes()); // enabledExtensionCount
        w.extend_from_slice(&0u64.to_le_bytes()); // ppEnabledExtensionNames: empty
        w.extend_from_slice(&0u64.to_le_bytes()); // pEnabledFeatures: absent
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent
        w.extend_from_slice(&1u64.to_le_bytes()); // pDevice: present
        w.extend_from_slice(&DEVICE.to_le_bytes()); // the id the guest chose

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Dispatched::Served);
        assert!(!dec.fatal(), "a refusal is the driver's answer, not a protocol error");

        assert_eq!(
            objects.lookup(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE.0),
            Lookup::Ghost,
            "the id the driver refused must not become an object"
        );
    }

    /// A failed enumeration refuses every id the guest offered, not just the tail.
    ///
    /// The out-handles arrive in the request, so the guest has already chosen ids for physical
    /// devices the host may not have. When the enumeration itself fails there is no short answer
    /// to bound -- every id it named is refused, and each has to say so, or the hook registers the
    /// lot as their own handles and the guest holds a GPU per id it guessed at.
    #[test]
    fn a_failed_enumeration_refuses_every_id_the_guest_offered() {
        use super::super::cs::{Lookup, Objects};

        const INSTANCE: u64 = 2;
        const IDS: [u64; 3] = [21, 22, 23];

        let objects = Shared::new();
        objects
            .borrow_mut()
            .add(
                ObjectId(INSTANCE),
                VkObjectType::VK_OBJECT_TYPE_INSTANCE,
                HostHandle(INSTANCE),
                None,
            )
            .unwrap();

        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkEnumeratePhysicalDevices_EXT, 0);
        w.extend_from_slice(&INSTANCE.to_le_bytes());
        w.extend_from_slice(&1u64.to_le_bytes()); // pPhysicalDeviceCount: present
        w.extend_from_slice(&(IDS.len() as u32).to_le_bytes());
        w.extend_from_slice(&(IDS.len() as u64).to_le_bytes()); // pPhysicalDevices: the array size
        for id in IDS {
            w.extend_from_slice(&id.to_le_bytes());
        }

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Dispatched::Served);
        assert!(!dec.fatal(), "a refusal is the driver's answer, not a protocol error");

        for id in IDS {
            assert_eq!(
                objects.lookup(ObjectId(id), VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE.0),
                Lookup::Ghost,
                "id {id} was offered to a failed enumeration and must not become an object"
            );
        }
    }

    /// A driver with fewer devices than the guest guessed answers for the ones it has, and
    /// refuses the rest.
    ///
    /// The shape the generated accessor witnesses cannot reach: this array's count is
    /// `*pPhysicalDeviceCount`, behind an out-pointer, so nothing can plant it from a slice. It
    /// is also the only handler where the guest's array and the host's answer may legitimately be
    /// different lengths, and three separate things have to agree about which:
    ///
    /// - the count written back, which is what the guest reads to size its own array;
    /// - the ids that become objects, which is the head of what it offered;
    /// - the ids that become ghosts, which is the tail it offered and did not get.
    ///
    /// Ghosting the whole array instead would take a device the guest does hold; ghosting none
    /// would let the hook register a GPU per id it guessed at. Both leave every command that
    /// follows naming something -- the wrong thing.
    ///
    /// Extensions are learned here too, and only for the devices that came back. Learning one for
    /// a slot the driver did not fill would file a stranger's extension list under a handle that
    /// is about to be something else.
    #[test]
    fn a_short_enumeration_answers_for_what_it_got_and_refuses_the_rest() {
        use super::super::cs::{Lookup, Objects};
        use super::super::proto::types::{VkExtensionProperties, VkInstance, VkPhysicalDevice};
        use std::cell::RefCell;

        const INSTANCE: u64 = 2;
        /// Three ids offered, two devices behind them.
        const IDS: [u64; 3] = [21, 22, 23];
        const HOST: [u64; 2] = [0x9100, 0x9200];

        thread_local! {
            /// Every physical device `learn_extensions` was asked about, in order.
            static LEARNED: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "C" fn enumerate(
            _instance: VkInstance,
            n: *mut u32,
            out: *mut VkPhysicalDevice,
        ) -> VkResult {
            // SAFETY: the wrapper passes its slice's own length and pointer, which is the
            // pairing under test.
            let room = unsafe { *n } as usize;
            assert!(room >= HOST.len(), "the guest sized for three and the room says {room}");
            // SAFETY: `out` has room for `room` elements, which the count just said.
            let out = unsafe { core::slice::from_raw_parts_mut(out, room) };
            for (e, h) in out.iter_mut().zip(HOST) {
                *e = VkPhysicalDevice(h);
            }
            // Fewer than asked for: the driver has two, and says so.
            unsafe { *n = HOST.len() as u32 };
            VkResult::VK_SUCCESS
        }

        unsafe extern "C" fn extensions(
            pd: VkPhysicalDevice,
            _layer: *const core::ffi::c_char,
            n: *mut u32,
            props: *mut VkExtensionProperties,
        ) -> VkResult {
            // SAFETY: both are the caller's locals, and the array is null on the count query.
            unsafe {
                if props.is_null() {
                    LEARNED.with_borrow_mut(|l| l.push(pd.0));
                    *n = 0;
                } else {
                    *n = 0;
                }
            }
            VkResult::VK_SUCCESS
        }

        let objects = Shared::new();
        objects
            .borrow_mut()
            .add(
                ObjectId(INSTANCE),
                VkObjectType::VK_OBJECT_TYPE_INSTANCE,
                HostHandle(INSTANCE),
                None,
            )
            .unwrap();

        let mut inst = crate::vulkan::Instance::default();
        inst.plant_vkEnumeratePhysicalDevices(enumerate);
        inst.plant_vkEnumerateDeviceExtensionProperties(extensions);
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_instance(inst);

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkEnumeratePhysicalDevices_EXT, 0);
        w.extend_from_slice(&INSTANCE.to_le_bytes());
        w.extend_from_slice(&1u64.to_le_bytes()); // pPhysicalDeviceCount: present
        w.extend_from_slice(&(IDS.len() as u32).to_le_bytes());
        w.extend_from_slice(&(IDS.len() as u64).to_le_bytes()); // pPhysicalDevices: the array size
        for id in IDS {
            w.extend_from_slice(&id.to_le_bytes());
        }

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Dispatched::Served);
        assert!(!dec.fatal());

        // The two the driver had are real objects under the guest's own ids, each holding the
        // handle that came back in its slot -- not the slot beside it.
        for (id, host) in IDS.iter().zip(HOST) {
            assert_eq!(
                objects.lookup(ObjectId(*id), VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE.0),
                Lookup::Found(HostHandle(host)),
                "id {id} holds the handle from its own slot"
            );
        }
        // The third is a ghost: offered, not answered. A command naming it is absorbed rather
        // than poisoning the context, and it is not an object the guest may use.
        assert_eq!(
            objects.lookup(ObjectId(IDS[2]), VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE.0),
            Lookup::Ghost,
            "the id the driver had no device for is refused, not registered"
        );

        LEARNED.with_borrow(|l| {
            assert_eq!(
                l.as_slice(),
                &HOST,
                "extensions are learned for the devices that came back, and no others"
            );
        });

        // And the count the guest reads back, which the dispatch above cannot show: the storage
        // for it belongs to the decoder's arena and is gone with the batch. So the handler is
        // called once more against storage this test holds.
        //
        // It is the number the guest sizes its own array by, and reporting the length it asked
        // for instead would have it walk two devices' worth of handles plus whatever the third
        // slot was left holding -- while every other assertion above still passed.
        let mut n = IDS.len() as u32;
        let mut wire: [VkPhysicalDevice; 3] = core::array::from_fn(|i| VkPhysicalDevice(IDS[i]));
        let mut shadow = [VkPhysicalDevice(0); 3];
        let mut args = vn_command_vkEnumeratePhysicalDevices::default();
        args.instance = VkInstance(INSTANCE);
        args.plant_pPhysicalDeviceCount(&mut n);
        args.plant_pPhysicalDevices(&mut wire);
        args.plant_handle_pPhysicalDevices(&mut shadow);
        h.vkEnumeratePhysicalDevices(&mut args);
        assert_eq!(
            n,
            HOST.len() as u32,
            "the guest is told how many it got, not how many it asked for"
        );
        assert_eq!(
            shadow.map(|p| p.0),
            [HOST[0], HOST[1], 0],
            "and the slots past that answer are left as they were"
        );

        h.driver.abandon_planted();
    }

    /// Three commands a guest can send that name no work the host could do, each refused before
    /// anything crosses into the driver.
    ///
    /// These are trust-boundary refusals (CLAUDE.md): nothing here is a host invariant, so none of
    /// them asserts. What matters is *where* the refusal happens. Reaching the driver first and
    /// letting it decide means handing a Vulkan implementation a length it will read past, a byte
    /// count that is not a whole number of words, or a queue with no device -- and mesa's runtime
    /// carries assertions on all three, on a ring thread where an abort takes the whole worker.
    ///
    /// So each is asked twice: that the context is poisoned, and that the driver was never called.
    /// A test that only checked the first would pass against a handler that called the driver and
    /// then complained.
    #[test]
    fn the_commands_a_guest_can_botch_are_refused_before_the_driver_sees_them() {
        use super::super::proto::types::{
            VkAllocationCallbacks, VkCommandBuffer, VkCommandPool, VkDevice, VkFence,
            VkPipelineLayout, VkQueue, VkShaderModule, VkShaderModuleCreateInfo,
            VkShaderStageFlags, VkSubmitInfo, vn_command_vkCmdPushConstants,
            vn_command_vkCreateShaderModule, vn_command_vkQueueSubmit,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        const CB: (u64, u64) = (11, 110);

        thread_local! {
            /// Every entry point that was reached. Empty is the whole point.
            static REACHED: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "C" fn push(
            _cb: VkCommandBuffer,
            _layout: VkPipelineLayout,
            _stages: VkShaderStageFlags,
            _offset: u32,
            _size: u32,
            _values: *const core::ffi::c_void,
        ) {
            REACHED.with_borrow_mut(|r| r.push("vkCmdPushConstants"));
        }

        unsafe extern "C" fn create_shader(
            _device: VkDevice,
            _info: *const VkShaderModuleCreateInfo,
            _alloc: *const VkAllocationCallbacks,
            out: *mut VkShaderModule,
        ) -> VkResult {
            REACHED.with_borrow_mut(|r| r.push("vkCreateShaderModule"));
            // A real create that succeeds returns a handle, and `Driver::create_object` asserts
            // it -- a null one here would be testing that assert rather than the refusal.
            // SAFETY: the caller's local.
            unsafe { *out = VkShaderModule(0x5000) };
            VkResult::VK_SUCCESS
        }

        unsafe extern "C" fn submit(
            _queue: VkQueue,
            _n: u32,
            _submits: *const VkSubmitInfo,
            _fence: VkFence,
        ) -> VkResult {
            REACHED.with_borrow_mut(|r| r.push("vkQueueSubmit"));
            VkResult::VK_SUCCESS
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkCmdPushConstants(push);
        fns.plant_vkCreateShaderModule(create_shader);
        fns.plant_vkQueueSubmit(submit);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_pool(
            VkDevice(DEVICE),
            VkCommandPool(POOL),
            &[(VkCommandBuffer(CB.0), ObjectId(CB.1))],
        );

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        // Constants pushed without saying what they are. The decoder cannot make a slice of a
        // null pointer with a non-zero size, and pushing whatever the layout last held would hand
        // the next draw constants the guest never sent -- so the answer is not an empty push.
        let mut args = vn_command_vkCmdPushConstants::default();
        args.commandBuffer = VkCommandBuffer(CB.0);
        args.size = 16;
        h.vkCmdPushConstants(&mut args);
        assert!(h.reject.is_some(), "sixteen bytes of nothing is not a push");
        h.reject = None;

        // A shader whose code is not a whole number of words. `pCode` is `uint32_t*` and the size
        // is in bytes, so a size Vulkan cannot divide is a driver reading a partial word past the
        // end of what the guest sent.
        let info = VkShaderModuleCreateInfo { codeSize: 7, ..Default::default() };
        let mut args = vn_command_vkCreateShaderModule::default();
        args.device = VkDevice(DEVICE);
        args.pCreateInfo = Some(&info);
        h.vkCreateShaderModule(&mut args);
        assert!(h.reject.is_some(), "seven bytes is not a whole number of words");
        h.reject = None;

        // A submit to a queue this context never retrieved. There is no device behind it, so
        // there is no entry point to call -- and inventing a success would tell the guest work it
        // is waiting on has been queued.
        let mut args = vn_command_vkQueueSubmit::default();
        args.queue = VkQueue(0xdead);
        let submits: [VkSubmitInfo; 0] = [];
        args.plant_pSubmits(&submits);
        h.vkQueueSubmit(&mut args);
        assert!(h.reject.is_some(), "a queue with no device behind it cannot be submitted to");
        h.reject = None;

        REACHED.with_borrow(|r| {
            assert!(r.is_empty(), "the driver was reached by {r:?}, after the guest was refused");
        });

        // And the same three, sent properly, do reach it -- otherwise the assertion above would
        // hold just as well against a handler that refuses everything.
        let values = [1u8, 2, 3, 4];
        let mut args = vn_command_vkCmdPushConstants::default();
        args.commandBuffer = VkCommandBuffer(CB.0);
        args.plant_pValues(&values);
        h.vkCmdPushConstants(&mut args);
        assert!(h.reject.is_none());

        let info = VkShaderModuleCreateInfo { codeSize: 8, ..Default::default() };
        let mut args = vn_command_vkCreateShaderModule::default();
        args.device = VkDevice(DEVICE);
        args.pCreateInfo = Some(&info);
        h.vkCreateShaderModule(&mut args);
        assert!(h.reject.is_none());

        REACHED.with_borrow(|r| {
            assert_eq!(r.as_slice(), &["vkCmdPushConstants", "vkCreateShaderModule"]);
        });

        h.driver.abandon_planted();
    }

    /// The four commands a GTK client asks for that synoik and vkcube never did: saving the
    /// pipeline cache -- count then fill, in bytes -- merging caches, and the YCbCr conversion a
    /// video texture needs. Each reaches the driver in its own shape, and each answers the guest.
    ///
    /// A client with a pipeline cache asks for its data a few seconds after every new pipeline,
    /// to save it to disk; refused, it is poisoned a little after its first frame.
    #[test]
    fn the_commands_a_gtk_client_asks_for_are_served() {
        use super::super::proto::types::{
            VkAllocationCallbacks, VkDevice, VkPipelineCache, VkSamplerYcbcrConversion,
            VkSamplerYcbcrConversionCreateInfo, vn_command_vkCreateSamplerYcbcrConversion,
            vn_command_vkGetPipelineCacheData, vn_command_vkMergePipelineCaches,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const CACHE: u64 = 0x500;
        const BLOB: [u8; 5] = [0xca, 0xc4, 0xed, 0xda, 0x7a];

        #[derive(Default)]
        struct Saw {
            /// `(dst, the sources)`.
            merged: Vec<(u64, Vec<u64>)>,
            conversions: u32,
        }
        thread_local! { static SAW: RefCell<Saw> = RefCell::new(Saw::default()); }

        unsafe extern "C" fn cache_data(
            _d: VkDevice,
            _c: VkPipelineCache,
            size: *mut usize,
            data: *mut core::ffi::c_void,
        ) -> VkResult {
            // SAFETY: the wrapper passes its own count and the slice it was given, or null.
            unsafe {
                if data.is_null() {
                    *size = BLOB.len();
                    return VkResult::VK_SUCCESS;
                }
                let room = *size;
                let n = room.min(BLOB.len());
                core::ptr::copy_nonoverlapping(BLOB.as_ptr(), data.cast::<u8>(), n);
                *size = n;
                if n < BLOB.len() { VkResult::VK_INCOMPLETE } else { VkResult::VK_SUCCESS }
            }
        }
        unsafe extern "C" fn merge(
            _d: VkDevice,
            dst: VkPipelineCache,
            n: u32,
            srcs: *const VkPipelineCache,
        ) -> VkResult {
            // SAFETY: the wrapper passes the slice's own pointer and length.
            let srcs = unsafe { core::slice::from_raw_parts(srcs, n as usize) };
            SAW.with_borrow_mut(|w| w.merged.push((dst.0, srcs.iter().map(|c| c.0).collect())));
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn conversion(
            _d: VkDevice,
            _i: *const VkSamplerYcbcrConversionCreateInfo,
            _a: *const VkAllocationCallbacks,
            out: *mut VkSamplerYcbcrConversion,
        ) -> VkResult {
            SAW.with_borrow_mut(|w| w.conversions += 1);
            // SAFETY: the caller's local.
            unsafe { *out = VkSamplerYcbcrConversion(0x9c) };
            VkResult::VK_SUCCESS
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkGetPipelineCacheData(cache_data);
        fns.plant_vkMergePipelineCaches(merge);
        fns.plant_vkCreateSamplerYcbcrConversion(conversion);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        // Count: no array, and the size comes back.
        let mut size = 0usize;
        {
            let mut args = vn_command_vkGetPipelineCacheData::default();
            args.device = VkDevice(DEVICE);
            args.pipelineCache = VkPipelineCache(CACHE);
            args.plant_pDataSize(&mut size);
            h.vkGetPipelineCacheData(&mut args);
            assert!(h.reject.is_none(), "served now; a build that still refuses it fails here");
            assert!(h.reject.is_none());
            assert_eq!(args.ret, VkResult::VK_SUCCESS);
        }
        assert_eq!(size, BLOB.len(), "the count call says how many bytes there are");

        // Fill, with less room than that: what fits, the size it actually wrote, and INCOMPLETE
        // -- the guest's business, not an error.
        let mut short = [0u8; 3];
        let mut size = short.len();
        {
            let mut args = vn_command_vkGetPipelineCacheData::default();
            args.device = VkDevice(DEVICE);
            args.pipelineCache = VkPipelineCache(CACHE);
            args.plant_pDataSize(&mut size);
            args.plant_pData(&mut short);
            h.vkGetPipelineCacheData(&mut args);
            assert_eq!(args.ret, VkResult::VK_INCOMPLETE);
        }
        assert_eq!(size, 3);
        assert_eq!(short, BLOB[..3], "the bytes that fit, and no more");

        // Fill with room enough: all of it.
        let mut full = [0u8; 8];
        let mut size = full.len();
        {
            let mut args = vn_command_vkGetPipelineCacheData::default();
            args.device = VkDevice(DEVICE);
            args.pipelineCache = VkPipelineCache(CACHE);
            args.plant_pDataSize(&mut size);
            args.plant_pData(&mut full);
            h.vkGetPipelineCacheData(&mut args);
            assert_eq!(args.ret, VkResult::VK_SUCCESS);
        }
        assert_eq!(size, BLOB.len(), "the size written, not the room offered");
        assert_eq!(full[..5], BLOB);

        // Merge: every source, under the destination.
        let srcs = [VkPipelineCache(0x501), VkPipelineCache(0x502)];
        let mut args = vn_command_vkMergePipelineCaches::default();
        args.device = VkDevice(DEVICE);
        args.dstCache = VkPipelineCache(CACHE);
        args.plant_pSrcCaches(&srcs);
        h.vkMergePipelineCaches(&mut args);
        assert!(h.reject.is_none());
        assert_eq!(args.ret, VkResult::VK_SUCCESS);
        SAW.with_borrow(|w| assert_eq!(w.merged, [(CACHE, vec![0x501, 0x502])]));

        // A YCbCr conversion is a plain create: the driver's handle lands in the shadow.
        let info = VkSamplerYcbcrConversionCreateInfo::default();
        let mut wire = VkSamplerYcbcrConversion(77);
        let mut shadow = VkSamplerYcbcrConversion(0);
        let mut args = vn_command_vkCreateSamplerYcbcrConversion::default();
        args.device = VkDevice(DEVICE);
        args.pCreateInfo = Some(&info);
        args.plant_pYcbcrConversion(&mut wire);
        args.plant_handle_pYcbcrConversion(&mut shadow);
        h.vkCreateSamplerYcbcrConversion(&mut args);
        assert!(h.reject.is_none());
        assert_eq!(args.ret, VkResult::VK_SUCCESS);
        assert_eq!(shadow, VkSamplerYcbcrConversion(0x9c));
        assert_eq!(wire, VkSamplerYcbcrConversion(77), "the guest's id on the wire is left alone");
        SAW.with_borrow(|w| assert_eq!(w.conversions, 1));

        h.driver.abandon_planted();
    }

    /// Freeing descriptor sets one at a time is served: the run reaches the driver under its
    /// pool, the pool stops holding the sets, and the guest is answered `VK_SUCCESS`.
    ///
    /// GTK's Vulkan renderer frees sets individually rather than resetting the pool, so a build
    /// that refuses this poisons every GTK client a few frames in -- the client sees
    /// `VK_ERROR_DEVICE_LOST`, then aborts on the garbage size a failed pipeline-cache query hands
    /// it. The object-table half is the generated lifecycle hook's, as for every `vkFree*`; what is
    /// pinned here is the driver call and the pool's bookkeeping.
    #[test]
    fn freeing_descriptor_sets_releases_them_from_their_pool() {
        use super::super::proto::types::{
            VkDescriptorPool, VkDescriptorSet, VkDevice, vn_command_vkFreeDescriptorSets,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        const SETS: [(u64, u64); 2] = [(0x300, 30), (0x400, 40)];

        thread_local! {
            /// `(pool, the sets)` the driver was told to free.
            static SAW: RefCell<Vec<(u64, Vec<u64>)>> = const { RefCell::new(Vec::new()) };
        }
        unsafe extern "C" fn free(
            _d: VkDevice,
            pool: VkDescriptorPool,
            n: u32,
            sets: *const VkDescriptorSet,
        ) -> VkResult {
            // SAFETY: the wrapper passes the slice's own pointer and length.
            let sets = unsafe { core::slice::from_raw_parts(sets, n as usize) };
            SAW.with_borrow_mut(|w| w.push((pool.0, sets.iter().map(|s| s.0).collect())));
            VkResult::VK_SUCCESS
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkFreeDescriptorSets(free);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_pool(
            VkDevice(DEVICE),
            VkDescriptorPool(POOL),
            &[
                (VkDescriptorSet(SETS[0].0), ObjectId(SETS[0].1)),
                (VkDescriptorSet(SETS[1].0), ObjectId(SETS[1].1)),
            ],
        );

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        // Only the first: GTK frees a set at a time, and the second has to survive it.
        let one = [VkDescriptorSet(SETS[0].0)];
        let mut args = vn_command_vkFreeDescriptorSets::default();
        args.device = VkDevice(DEVICE);
        args.descriptorPool = VkDescriptorPool(POOL);
        args.plant_pDescriptorSets(&one);
        h.vkFreeDescriptorSets(&mut args);
        assert!(
            h.reject.is_none(),
            "the command is served now; a build that still refuses it fails here"
        );
        assert!(h.reject.is_none());
        assert_eq!(args.ret, VkResult::VK_SUCCESS);
        SAW.with_borrow(|w| assert_eq!(w, &[(POOL, vec![SETS[0].0])], "that set, under its pool"));
        assert_eq!(
            h.driver.pool_child_id(VkDescriptorPool(POOL), VkDescriptorSet(SETS[0].0)),
            None,
            "the pool no longer holds what was freed"
        );
        assert_eq!(
            h.driver.pool_child_id(VkDescriptorPool(POOL), VkDescriptorSet(SETS[1].0)),
            Some(ObjectId(SETS[1].1)),
            "and still holds what was not"
        );

        // Freeing nothing is not a failure, and does not reach the driver.
        let none: [VkDescriptorSet; 0] = [];
        let mut args = vn_command_vkFreeDescriptorSets::default();
        args.device = VkDevice(DEVICE);
        args.descriptorPool = VkDescriptorPool(POOL);
        args.plant_pDescriptorSets(&none);
        h.vkFreeDescriptorSets(&mut args);
        assert_eq!(args.ret, VkResult::VK_SUCCESS);
        SAW.with_borrow(|w| assert_eq!(w.len(), 1, "an empty run is answered without a call"));

        // A set under a pool that did not allocate it is refused before the driver sees the
        // pair: the driver would free it from the wrong pool, and the pool that holds it would
        // go on holding a handle the driver has reused.
        const OTHER: u64 = 8;
        h.driver.plant_pool(VkDevice(DEVICE), VkDescriptorPool(OTHER), &[]);
        let two = [VkDescriptorSet(SETS[1].0)];
        let mut args = vn_command_vkFreeDescriptorSets::default();
        args.device = VkDevice(DEVICE);
        args.descriptorPool = VkDescriptorPool(OTHER);
        args.plant_pDescriptorSets(&two);
        h.vkFreeDescriptorSets(&mut args);
        assert!(h.reject.take().is_some(), "a run that is not the pool's is refused");
        SAW.with_borrow(|w| assert_eq!(w.len(), 1, "and never reached the driver"));
        assert_eq!(
            h.driver.pool_child_id(VkDescriptorPool(POOL), VkDescriptorSet(SETS[1].0)),
            Some(ObjectId(SETS[1].1)),
            "its own pool still holds it"
        );

        // So is a set freed twice: the second time it is nobody's.
        let mut args = vn_command_vkFreeDescriptorSets::default();
        args.device = VkDevice(DEVICE);
        args.descriptorPool = VkDescriptorPool(POOL);
        args.plant_pDescriptorSets(&one);
        h.vkFreeDescriptorSets(&mut args);
        assert!(h.reject.take().is_some(), "a set already freed is not the pool's to free again");
        SAW.with_borrow(|w| assert_eq!(w.len(), 1));

        h.driver.abandon_planted();
    }

    /// The three commands whose two arrays are counted separately, and the one whose payload is
    /// counted in bytes: every array reaches the driver at its own length.
    ///
    /// `a_recording_handler_hands_the_driver_what_the_guest_sent` pins the shapes on a command
    /// buffer; these are the same question asked where the driver call is not a recording. Each
    /// pair is deliberately of unequal length, because every count crossing here is a `u32` and
    /// passing one array's length where the other's belongs compiles.
    ///
    /// `vkQueueSubmit`'s empty case is here rather than with the refusals: submitting no work is
    /// how a guest signals a fence, so it has to reach the driver, and a handler that turned an
    /// empty array away would leave that fence unsignalled and the guest waiting on it forever.
    #[test]
    fn the_commands_with_two_counts_deliver_both_arrays() {
        use super::super::proto::types::{
            VkCommandBuffer, VkCommandPool, VkCopyDescriptorSet, VkDescriptorSet, VkDevice,
            VkFence, VkPipelineBindPoint, VkPipelineLayout, VkQueue, VkSubmitInfo,
            VkWriteDescriptorSet, vn_command_vkCmdBindDescriptorSets, vn_command_vkQueueSubmit,
            vn_command_vkUpdateDescriptorSets,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        const CB: (u64, u64) = (11, 110);
        const QUEUE: u64 = 0x2200;

        #[derive(Default)]
        struct Saw {
            /// `(firstSet, the sets, the dynamic offsets)`.
            bound: Vec<(u32, Vec<u64>, Vec<u32>)>,
            /// `(writes, copies)` -- the two counts, which have no reason to be equal.
            updated: Vec<(u32, u32)>,
            /// How many submits, and the fence they carry.
            submitted: Vec<(u32, u64)>,
            /// The bytes a push carried, which are the command.
            pushed: Vec<(u32, Vec<u8>)>,
        }
        thread_local! {
            static SAW: RefCell<Saw> = RefCell::new(Saw::default());
        }

        #[allow(clippy::too_many_arguments)]
        unsafe extern "C" fn bind(
            _cb: VkCommandBuffer,
            _bind_point: VkPipelineBindPoint,
            _layout: VkPipelineLayout,
            first: u32,
            n_sets: u32,
            sets: *const VkDescriptorSet,
            n_offsets: u32,
            offsets: *const u32,
        ) {
            // SAFETY: the wrapper passes each slice's own pointer and length.
            let (s, o) = unsafe {
                (
                    core::slice::from_raw_parts(sets, n_sets as usize),
                    core::slice::from_raw_parts(offsets, n_offsets as usize),
                )
            };
            SAW.with_borrow_mut(|w| {
                w.bound.push((first, s.iter().map(|h| h.0).collect(), o.to_vec()))
            });
        }

        unsafe extern "C" fn update(
            _device: VkDevice,
            n_writes: u32,
            _writes: *const VkWriteDescriptorSet,
            n_copies: u32,
            _copies: *const VkCopyDescriptorSet,
        ) {
            SAW.with_borrow_mut(|w| w.updated.push((n_writes, n_copies)));
        }

        unsafe extern "C" fn submit(
            _queue: VkQueue,
            n: u32,
            _submits: *const VkSubmitInfo,
            fence: VkFence,
        ) -> VkResult {
            SAW.with_borrow_mut(|w| w.submitted.push((n, fence.0)));
            VkResult::VK_SUCCESS
        }

        unsafe extern "C" fn push(
            _cb: VkCommandBuffer,
            _layout: VkPipelineLayout,
            _stages: super::super::proto::types::VkShaderStageFlags,
            offset: u32,
            size: u32,
            values: *const core::ffi::c_void,
        ) {
            // SAFETY: the wrapper passes the slice's own pointer and its length in bytes.
            let b = unsafe { core::slice::from_raw_parts(values.cast::<u8>(), size as usize) };
            SAW.with_borrow_mut(|w| w.pushed.push((offset, b.to_vec())));
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkCmdBindDescriptorSets(bind);
        fns.plant_vkUpdateDescriptorSets(update);
        fns.plant_vkQueueSubmit(submit);
        fns.plant_vkCmdPushConstants(push);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_pool(
            VkDevice(DEVICE),
            VkCommandPool(POOL),
            &[(VkCommandBuffer(CB.0), ObjectId(CB.1))],
        );
        driver.plant_queue(VkDevice(DEVICE), VkQueue(QUEUE));

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let cb = VkCommandBuffer(CB.0);

        // Two sets, three dynamic offsets. A layout with no dynamic descriptors legitimately
        // binds none of the second, so the two counts are genuinely independent.
        let sets = [VkDescriptorSet(0x300), VkDescriptorSet(0x400)];
        let offsets = [16u32, 32, 48];
        let mut args = vn_command_vkCmdBindDescriptorSets::default();
        args.commandBuffer = cb;
        args.firstSet = 5;
        args.plant_pDescriptorSets(&sets);
        args.plant_pDynamicOffsets(&offsets);
        h.vkCmdBindDescriptorSets(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|w| {
            assert_eq!(w.bound, [(5, vec![0x300, 0x400], vec![16, 32, 48])]);
        });

        // Three writes, one copy.
        let writes = [VkWriteDescriptorSet::default(); 3];
        let copies = [VkCopyDescriptorSet::default(); 1];
        let mut args = vn_command_vkUpdateDescriptorSets::default();
        args.device = VkDevice(DEVICE);
        args.plant_pDescriptorWrites(&writes);
        args.plant_pDescriptorCopies(&copies);
        h.vkUpdateDescriptorSets(&mut args);
        SAW.with_borrow(|w| assert_eq!(w.updated, [(3, 1)], "each count with its own array"));

        // The bytes are the command, and the offset travels beside them without becoming them.
        let values = [0xdeu8, 0xad, 0xbe, 0xef, 0x01];
        let mut args = vn_command_vkCmdPushConstants::default();
        args.commandBuffer = cb;
        args.offset = 12;
        args.plant_pValues(&values);
        h.vkCmdPushConstants(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|w| {
            assert_eq!(w.pushed, [(12, values.to_vec())], "all five bytes, at the offset given");
        });

        // No work at all, which is how a guest signals a fence. It has to reach the driver.
        let submits: [VkSubmitInfo; 0] = [];
        let mut args = vn_command_vkQueueSubmit::default();
        args.queue = VkQueue(QUEUE);
        args.fence = VkFence(0x77);
        args.plant_pSubmits(&submits);
        h.vkQueueSubmit(&mut args);
        assert!(h.reject.is_none(), "an empty submit is a fence signal, not a botched command");
        assert_eq!(args.ret, VkResult::VK_SUCCESS);
        SAW.with_borrow(|w| {
            assert_eq!(w.submitted, [(0, 0x77)], "no work, and the fence that is waiting on it");
        });

        h.driver.abandon_planted();
    }

    /// A pipeline run the driver only partly completes hands the guest nothing, and leaves
    /// nothing behind on the host either.
    ///
    /// Vulkan is explicit that `vkCreateGraphicsPipelines` may fill some slots and fail: the
    /// handles it did produce are real, they are the caller's to destroy, and the ones it did not
    /// are `VK_NULL_HANDLE`. That makes the failure path the interesting one. The run is refused
    /// as a whole -- so the survivors have to be destroyed here, because the guest never learns
    /// their ids and nothing else will ever name them -- and the slots have to be cleared, or the
    /// reply hands back a handle that was destroyed on the way out.
    ///
    /// Neither is visible from any corpus: a replay strips replies, and a leaked pipeline is a
    /// host-side object no census reads.
    #[test]
    fn a_partly_failed_pipeline_run_destroys_what_it_made_and_keeps_none_of_it() {
        use super::super::proto::types::{
            VkAllocationCallbacks, VkDevice, VkGraphicsPipelineCreateInfo, VkPipeline,
            VkPipelineCache, vn_command_vkCreateGraphicsPipelines,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const IDS: [u64; 3] = [61, 62, 63];
        /// The driver compiles the first two and then gives up on the third.
        const MADE: [u64; 2] = [0x7100, 0x7200];

        thread_local! {
            /// Every pipeline handed to `vkDestroyPipeline`, in order.
            static DESTROYED: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
            /// The create-info count the driver was told.
            static ASKED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
        }

        #[allow(clippy::too_many_arguments)]
        unsafe extern "C" fn create(
            _device: VkDevice,
            _cache: VkPipelineCache,
            n: u32,
            _infos: *const VkGraphicsPipelineCreateInfo,
            _alloc: *const VkAllocationCallbacks,
            out: *mut VkPipeline,
        ) -> VkResult {
            ASKED.with_borrow_mut(|a| a.push(n));
            // SAFETY: the wrapper passes its slice's own pointer and length.
            let out = unsafe { core::slice::from_raw_parts_mut(out, n as usize) };
            for (e, h) in out.iter_mut().zip(MADE) {
                *e = VkPipeline(h);
            }
            VkResult::VK_ERROR_INVALID_SHADER_NV
        }

        unsafe extern "C" fn destroy(
            _device: VkDevice,
            pipeline: VkPipeline,
            _alloc: *const VkAllocationCallbacks,
        ) {
            DESTROYED.with_borrow_mut(|d| d.push(pipeline.0));
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkCreateGraphicsPipelines(create);
        fns.plant_vkDestroyPipeline(destroy);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let infos = [VkGraphicsPipelineCreateInfo::default(); 3];
        let mut wire: [VkPipeline; 3] = core::array::from_fn(|i| VkPipeline(IDS[i]));
        let mut shadow = [VkPipeline(0); 3];
        let mut args = vn_command_vkCreateGraphicsPipelines::default();
        args.device = VkDevice(DEVICE);
        args.plant_pCreateInfos(&infos);
        args.plant_pPipelines(&mut wire);
        args.plant_handle_pPipelines(&mut shadow);

        h.vkCreateGraphicsPipelines(&mut args);
        assert!(h.reject.is_none(), "a driver refusing to compile is an answer, not a bad command");
        assert_eq!(args.ret, VkResult::VK_ERROR_INVALID_SHADER_NV);

        ASKED.with_borrow(|a| {
            assert_eq!(a.as_slice(), &[3], "one handle slot per create-info, and it says so");
        });
        DESTROYED.with_borrow(|d| {
            assert_eq!(
                d.as_slice(),
                &MADE,
                "the two the driver did make are destroyed: the guest never learns their ids, \
                 so nothing else can ever name them"
            );
        });
        assert_eq!(
            shadow.map(|p| p.0),
            [0, 0, 0],
            "and no destroyed handle is left in the reply's shadow"
        );

        for id in IDS {
            assert!(
                objects.borrow().is_ghost(ObjectId(id)),
                "id {id} was asked for and the run failed, so it names a ghost"
            );
        }

        h.driver.abandon_planted();
    }

    /// A create the macro serves, refused because the driver is not there.
    ///
    /// The whole point of routing every simple object through `Driver::create_object` is that the
    /// device is re-checked against the driver's own table rather than trusted because the object
    /// table resolved it. A miss has to ghost, not quietly succeed: the guest is already sending
    /// commands that name the fence.
    #[test]
    fn a_simple_create_with_no_device_ghosts_rather_than_registering() {
        use super::super::cs::{Lookup, Objects};
        use super::super::proto::types::VkStructureType;

        const DEVICE: u64 = 9;
        const FENCE: u64 = 4;

        let objects = Shared::new();
        objects
            .borrow_mut()
            .add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(DEVICE), None)
            .unwrap();

        // The object table has the device; the driver does not. That is exactly the split the
        // re-check exists for -- a guest that destroyed a device and then created against it.
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkCreateFence_EXT, 0);
        w.extend_from_slice(&DEVICE.to_le_bytes());
        w.extend_from_slice(&1u64.to_le_bytes()); // pCreateInfo: present
        w.extend_from_slice(
            &(VkStructureType::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO.0).to_le_bytes(),
        );
        w.extend_from_slice(&0u64.to_le_bytes()); // pNext: absent
        w.extend_from_slice(&0u32.to_le_bytes()); // flags
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent
        w.extend_from_slice(&1u64.to_le_bytes()); // pFence: present
        w.extend_from_slice(&FENCE.to_le_bytes()); // the id the guest chose

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Dispatched::Served);
        assert!(!dec.fatal(), "a refusal is the driver's answer, not a protocol error");
        assert!(h.reject.is_none(), "a refused create is not a protocol violation");

        assert_eq!(
            objects.lookup(ObjectId(FENCE), VkObjectType::VK_OBJECT_TYPE_FENCE.0),
            Lookup::Ghost
        );
    }

    /// A shader whose code size is not a whole number of words poisons the context.
    ///
    /// Called directly rather than over the wire: the wire path only proves the decoder round
    /// trips, and what is under test is the guard in front of the driver call. The decoder
    /// allocates `codeSize / 4` words with truncating division, so a `codeSize` of 7 leaves the
    /// driver reading three bytes past a four-byte allocation -- on a number the guest chose.
    #[test]
    fn a_shader_whose_code_is_not_whole_words_is_refused() {
        use super::super::proto::types::{
            VkShaderModuleCreateInfo, vn_command_vkCreateShaderModule,
        };

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let odd = VkShaderModuleCreateInfo { codeSize: 7, ..Default::default() };
        let mut args = vn_command_vkCreateShaderModule::default();
        args.pCreateInfo = Some(&odd);
        h.vkCreateShaderModule(&mut args);
        assert!(h.reject.is_some(), "a code size of 7 must not reach the driver");

        // Four is a whole word, so the guard lets it through; there is no device, so the driver
        // refuses it -- which is a different answer from a protocol violation.
        let whole = VkShaderModuleCreateInfo { codeSize: 4, ..Default::default() };
        let mut args = vn_command_vkCreateShaderModule::default();
        args.pCreateInfo = Some(&whole);
        h.reject = None;
        h.vkCreateShaderModule(&mut args);
        assert!(h.reject.is_none(), "a whole number of words is not a protocol violation");
    }

    /// A command that claims an array and sends none is refused, not quietly done as nothing.
    ///
    /// Called directly rather than over the wire, and deliberately so. `vkBindBufferMemory2`'s
    /// array is a *required* one, so the decoder checks its size against the count and poisons
    /// before a handler could ever see the pair disagree -- there is no wire that reaches this.
    /// What is under test is the verdict [`Handlers::array`] returns, which the eighty-odd
    /// optional arrays *can* reach, and which every handler that carries one will rely on.
    ///
    /// The verdict has to be a refusal. Passing a count with no array behind it on to the driver
    /// walks it off the end of nothing; calling it empty reports success for a bind that never
    /// happened, and the guest then draws from a buffer it believes has memory.
    /// The generated accessor hands back the array the guest sent -- all of it, and no more.
    ///
    /// The count and the pointer are still two members, because `vn_command_*` has to keep C's
    /// layout; the accessor is the one place they become one thing. Nothing else in the harness
    /// can see it get that wrong: the wire round trip never calls an accessor, and the replay gate
    /// counts commands accounted for rather than what they did -- an accessor handing back one
    /// element too many replays with every command accepted and the census unchanged.
    #[test]
    fn a_counted_array_arrives_as_exactly_the_elements_behind_it() {
        use super::super::proto::types::{
            VkBindBufferMemoryInfo, VkDeviceSize, vn_command_vkBindBufferMemory2,
        };

        let infos: [VkBindBufferMemoryInfo; 3] = core::array::from_fn(|i| VkBindBufferMemoryInfo {
            memoryOffset: VkDeviceSize(100 + i as u64),
            ..Default::default()
        });
        let mut args = vn_command_vkBindBufferMemory2::default();
        args.plant_pBindInfos(&infos);
        let got = args.pBindInfos();
        assert_eq!(
            got.iter().map(|i| i.memoryOffset.0).collect::<Vec<_>>(),
            [100, 101, 102],
            "the slice must be the array, not a prefix of it and not a step past its end"
        );

        // A count of none is an empty slice, not an absent one: the guest asked for no binds,
        // which is a legal thing to ask for and a different answer from a broken pair.
        let mut args = vn_command_vkBindBufferMemory2::default();
        args.plant_pBindInfos(&infos[..0]);
        assert_eq!(args.pBindInfos().len(), 0);
    }

    /// A count with no array behind it never reaches a handler.
    ///
    /// The pair is reconciled where the truth is: `decode_array_size` checks the wire's own size
    /// against the count the command already gave, and the generated dispatch drops a command
    /// whose decode poisoned. That is what lets a validated array's accessor hand back a slice
    /// rather than an `Option` -- if this refusal ever stopped happening, the accessor would be
    /// describing a state it can no longer rule out, and would panic instead of poisoning.
    ///
    /// `vkCmdSetViewport` stands in for the validated class, as `vkFreeCommandBuffers` does above
    /// for the `noautovalidity` one whose absent-array size is deliberately left unchecked.
    #[test]
    fn an_array_the_guest_counted_but_did_not_send_never_reaches_a_handler() {
        use super::super::proto::types::vn_command_vkCmdSetViewport;

        const DEVICE: u64 = 9;
        const BUFFER: u64 = 11;
        /// Six `f32`, each a four-byte scalar on the wire.
        const VIEWPORT: usize = 24;

        #[derive(Default)]
        struct Recorder {
            saw: Option<usize>,
        }
        impl Commands for Recorder {
            fn unsupported(&mut self, _cmd: VkCommandTypeEXT) {}

            fn vkCmdSetViewport(&mut self, args: &mut vn_command_vkCmdSetViewport<'_>) {
                self.saw = Some(args.pViewports().len());
            }

            fn object_created(
                &mut self,
                _ty: VkObjectType,
                _id: ObjectId,
                _host: HostHandle,
                _owner: Option<ObjectId>,
            ) {
            }
            fn object_destroyed(&mut self, _ty: VkObjectType, _id: ObjectId) {}
        }

        /// The wire for a command setting `sent` viewports that claims to be setting `counted`.
        fn wire(counted: u32, sent: usize) -> Vec<u8> {
            let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkCmdSetViewport_EXT, 0);
            w.extend_from_slice(&BUFFER.to_le_bytes());
            w.extend_from_slice(&0u32.to_le_bytes()); // firstViewport
            w.extend_from_slice(&counted.to_le_bytes());
            w.extend_from_slice(&(sent as u64).to_le_bytes());
            w.resize(w.len() + sent * VIEWPORT, 0);
            w
        }

        fn run(w: &[u8], objects: &Shared) -> (Option<usize>, bool) {
            let temp = Bump::new();
            let hard = AtomicBool::new(false);
            let mut dec = Decoder::new(w, &temp, objects, &hard);
            let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
            let _flags = dec.decode_scalar::<VkFlags>();
            let mut h = Recorder::default();
            let verdict = vn_dispatch_command(&mut dec, None, cmd, &mut h);
            assert_eq!(
                verdict == Dispatched::Undecodable,
                dec.fatal(),
                "the verdict and the poison flag are one fact"
            );
            (h.saw, verdict == Dispatched::Undecodable)
        }

        let objects = Shared::new();
        {
            let mut t = objects.borrow_mut();
            t.add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(1), None)
                .unwrap();
            t.add(
                ObjectId(BUFFER),
                VkObjectType::VK_OBJECT_TYPE_COMMAND_BUFFER,
                HostHandle(2),
                Some(ObjectId(DEVICE)),
            )
            .unwrap();
        }

        // Three counted, none sent: the one shape the accessor is no longer able to describe.
        assert_eq!(run(&wire(3, 0), &objects), (None, true), "a split pair must not be dispatched");

        // Nothing counted and nothing sent is legal, and must still reach the handler as an empty
        // slice -- absence and emptiness are not the same answer.
        assert_eq!(run(&wire(0, 0), &objects), (Some(0), false), "setting nothing is legal");

        // And a pair that agrees passes through with its length intact.
        assert_eq!(run(&wire(2, 2), &objects), (Some(2), false), "an agreeing pair is dispatched");
    }

    /// A refused pipeline run ghosts every id in it, not just the first.
    ///
    /// One command makes a run of pipelines and the guest is answered all-or-nothing, so a
    /// refusal owes a decision about every id it named. An id left undecided is registered as its
    /// own handle by the unserved-command fiction, and the next `vkCmdBindPipeline` hands the
    /// driver a number the guest invented.
    #[test]
    fn a_refused_pipeline_run_ghosts_the_whole_run() {
        use super::super::proto::types::{
            VkGraphicsPipelineCreateInfo, VkPipeline, vn_command_vkCreateGraphicsPipelines,
        };

        const IDS: [u64; 3] = [41, 42, 43];

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        // There is no device, so the driver refuses the whole run -- which is the only way to
        // reach the refusal path without a driver that fails on demand.
        let infos = [VkGraphicsPipelineCreateInfo::default(); IDS.len()];
        let mut ids = IDS.map(VkPipeline);
        let mut shadow = [VkPipeline(0); IDS.len()];
        let mut args = vn_command_vkCreateGraphicsPipelines::default();
        args.plant_pCreateInfos(&infos);
        args.plant_pPipelines(&mut ids);
        args.plant_handle_pPipelines(&mut shadow);
        h.vkCreateGraphicsPipelines(&mut args);

        assert_ne!(args.ret, VkResult::VK_SUCCESS, "no device means no pipelines");
        for id in IDS {
            assert!(
                objects.borrow().is_ghost(ObjectId(id)),
                "every id in a refused run is a ghost, not just the first"
            );
        }
    }

    /// The query commands are served through the record their create left, and refused without
    /// it.
    ///
    /// The driver's bounds check is [`super::driver::tests`]'s to pin; this is the handler's
    /// wiring around it: the pool the guest creates is the pool every query command is measured
    /// against, each hands the driver the guest's own arguments in the guest's own order, the
    /// driver's answer is what goes back, a query past the pool or a read past the room is a
    /// refusal rather than a driver call, and a destroyed pool's record goes with it.
    #[test]
    fn query_results_are_read_through_the_pool_the_guest_created() {
        use super::super::proto::types::{
            VkAllocationCallbacks, VkBuffer, VkCommandBuffer, VkCommandPool, VkDeviceSize,
            VkPipelineStageFlagBits, VkQueryControlFlags, VkQueryPool, VkQueryPoolCreateInfo,
            VkQueryResultFlags, VkQueryType,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const POOL: u64 = 40;
        const HOST_POOL: VkQueryPool = VkQueryPool(0x50);
        const CB: VkCommandBuffer = VkCommandBuffer(0x30);

        thread_local! {
            static READS: RefCell<u32> = const { RefCell::new(0) };
            static SAW: RefCell<Vec<(&'static str, u64, u64, u64)>> = const { RefCell::new(Vec::new()) };
        }
        fn saw(what: &'static str, a: u64, b: u64, c: u64) {
            SAW.with_borrow_mut(|s| s.push((what, a, b, c)));
        }
        unsafe extern "C" fn reset(_d: VkDevice, _p: VkQueryPool, first: u32, count: u32) {
            saw("reset", first.into(), count.into(), 0);
        }
        unsafe extern "C" fn begin(
            _cb: VkCommandBuffer,
            _p: VkQueryPool,
            query: u32,
            flags: VkQueryControlFlags,
        ) {
            saw("begin", query.into(), flags.0.into(), 0);
        }
        unsafe extern "C" fn end(_cb: VkCommandBuffer, _p: VkQueryPool, query: u32) {
            saw("end", query.into(), 0, 0);
        }
        unsafe extern "C" fn cmd_reset(
            _cb: VkCommandBuffer,
            _p: VkQueryPool,
            first: u32,
            count: u32,
        ) {
            saw("cmd_reset", first.into(), count.into(), 0);
        }
        unsafe extern "C" fn timestamp(
            _cb: VkCommandBuffer,
            stage: VkPipelineStageFlagBits,
            _p: VkQueryPool,
            query: u32,
        ) {
            saw("timestamp", stage.0 as u64, query.into(), 0);
        }
        unsafe extern "C" fn copy(
            _cb: VkCommandBuffer,
            _p: VkQueryPool,
            first: u32,
            count: u32,
            buffer: VkBuffer,
            offset: VkDeviceSize,
            stride: VkDeviceSize,
            flags: VkQueryResultFlags,
        ) {
            saw("copy", first.into(), count.into(), buffer.0);
            saw("copy'", offset.0, stride.0, flags.0.into());
        }
        unsafe extern "C" fn create(
            _d: VkDevice,
            _i: *const VkQueryPoolCreateInfo,
            _a: *const VkAllocationCallbacks,
            out: *mut VkQueryPool,
        ) -> VkResult {
            // SAFETY: the caller passes a local of its own.
            unsafe { *out = HOST_POOL };
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn destroy(
            _d: VkDevice,
            _p: VkQueryPool,
            _a: *const VkAllocationCallbacks,
        ) {
        }
        unsafe extern "C" fn results(
            _d: VkDevice,
            _p: VkQueryPool,
            _first: u32,
            count: u32,
            size: usize,
            data: *mut core::ffi::c_void,
            _stride: VkDeviceSize,
            _flags: VkQueryResultFlags,
        ) -> VkResult {
            READS.with_borrow_mut(|n| *n += 1);
            // SAFETY: `size` bytes at `data` are the caller's buffer, as just measured.
            let out = unsafe { core::slice::from_raw_parts_mut(data.cast::<u8>(), size) };
            for (i, b) in out.iter_mut().enumerate() {
                *b = i as u8;
            }
            assert_eq!(count as usize * 4, size);
            VkResult::VK_SUCCESS
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkCreateQueryPool(create);
        fns.plant_vkDestroyQueryPool(destroy);
        fns.plant_vkGetQueryPoolResults(results);
        fns.plant_vkResetQueryPool(reset);
        fns.plant_vkCmdBeginQuery(begin);
        fns.plant_vkCmdEndQuery(end);
        fns.plant_vkCmdResetQueryPool(cmd_reset);
        fns.plant_vkCmdWriteTimestamp(timestamp);
        fns.plant_vkCmdCopyQueryPoolResults(copy);

        let objects = Shared::new();
        objects
            .borrow_mut()
            .add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(DEVICE), None)
            .unwrap();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_pool(VkDevice(DEVICE), VkCommandPool(0x20), &[(CB, ObjectId(9))]);
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let device = VkDevice(DEVICE);

        /// A read of `count` 32-bit results from the start of the pool, four bytes apart.
        fn read<'a>(device: VkDevice, count: u32) -> vn_command_vkGetQueryPoolResults<'a> {
            let mut args = vn_command_vkGetQueryPoolResults::default();
            args.device = device;
            args.queryPool = HOST_POOL;
            args.queryCount = count;
            args.stride = VkDeviceSize(4);
            args
        }

        // Two timestamps.
        let info = VkQueryPoolCreateInfo {
            queryType: VkQueryType::VK_QUERY_TYPE_TIMESTAMP,
            queryCount: 2,
            ..Default::default()
        };
        let mut id = VkQueryPool(POOL);
        let mut shadow = VkQueryPool(0);
        let mut args = vn_command_vkCreateQueryPool::default();
        args.device = device;
        args.pCreateInfo = Some(&info);
        args.plant_pQueryPool(&mut id);
        args.plant_handle_pQueryPool(&mut shadow);
        h.vkCreateQueryPool(&mut args);
        assert_eq!(args.ret, VkResult::VK_SUCCESS);
        assert_eq!(shadow, HOST_POOL, "the driver's handle, in the shadow the reply reads");
        assert!(h.reject.is_none());

        // Both, 32-bit, four apart, into eight bytes: exactly enough.
        let mut room = [0xffu8; 8];
        let mut args = read(device, 2);
        args.plant_pData(&mut room);
        h.vkGetQueryPoolResults(&mut args);
        assert_eq!(args.ret, VkResult::VK_SUCCESS, "the driver's answer");
        assert!(h.reject.is_none());
        assert_eq!(room, [0, 1, 2, 3, 4, 5, 6, 7], "and the driver's bytes, in the guest's room");

        // The same read into seven bytes: refused, and the driver never sees it.
        let mut short = [0xffu8; 7];
        let mut args = read(device, 2);
        args.plant_pData(&mut short);
        h.vkGetQueryPoolResults(&mut args);
        assert!(h.reject.take().is_some(), "a read past the room is a refusal");
        assert_eq!(short, [0xff; 7], "and nothing was written");
        READS.with_borrow(|n| assert_eq!(*n, 1, "only the read that fit reached the driver"));

        // The other seven, each with distinct values in every interchangeable-looking slot, so
        // a handler that swapped two of them would be seen. All within the pool's two queries.
        let mut args = vn_command_vkResetQueryPool {
            device,
            queryPool: HOST_POOL,
            firstQuery: 1,
            queryCount: 1,
            ..Default::default()
        };
        h.vkResetQueryPool(&mut args);
        let mut args = vn_command_vkCmdBeginQuery {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            query: 1,
            flags: VkQueryControlFlags(0x1),
            ..Default::default()
        };
        h.vkCmdBeginQuery(&mut args);
        let mut args = vn_command_vkCmdEndQuery {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            query: 1,
            ..Default::default()
        };
        h.vkCmdEndQuery(&mut args);
        let mut args = vn_command_vkCmdResetQueryPool {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            firstQuery: 0,
            queryCount: 2,
            ..Default::default()
        };
        h.vkCmdResetQueryPool(&mut args);
        let mut args = vn_command_vkCmdWriteTimestamp {
            commandBuffer: CB,
            pipelineStage: VkPipelineStageFlagBits(0x400),
            queryPool: HOST_POOL,
            query: 1,
            ..Default::default()
        };
        h.vkCmdWriteTimestamp(&mut args);
        let mut args = vn_command_vkCmdCopyQueryPoolResults {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            firstQuery: 1,
            queryCount: 1,
            dstBuffer: VkBuffer(0x60),
            dstOffset: VkDeviceSize(0x70),
            stride: VkDeviceSize(0x80),
            flags: VkQueryResultFlags(0x1),
            ..Default::default()
        };
        h.vkCmdCopyQueryPoolResults(&mut args);
        assert!(h.reject.is_none(), "every one of them was within the pool");
        SAW.with_borrow(|s| {
            assert_eq!(
                s.as_slice(),
                [
                    ("reset", 1, 1, 0),
                    ("begin", 1, 0x1, 0),
                    ("end", 1, 0, 0),
                    ("cmd_reset", 0, 2, 0),
                    ("timestamp", 0x400, 1, 0),
                    ("copy", 1, 1, 0x60),
                    ("copy'", 0x70, 0x80, 0x1),
                ],
                "the guest's own arguments, in the guest's own order"
            );
        });
        SAW.with_borrow_mut(Vec::clear);

        // And each of them, one query past the pool: refused, and the driver never sees it.
        let mut args = vn_command_vkResetQueryPool {
            device,
            queryPool: HOST_POOL,
            firstQuery: 0,
            queryCount: 0x7fff_ffff,
            ..Default::default()
        };
        h.vkResetQueryPool(&mut args);
        assert!(h.reject.take().is_some(), "a host-side reset past the pool is a heap scribble");
        let mut args = vn_command_vkCmdBeginQuery {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            query: 2,
            ..Default::default()
        };
        h.vkCmdBeginQuery(&mut args);
        assert!(h.reject.take().is_some());
        let mut args = vn_command_vkCmdEndQuery {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            query: u32::MAX,
            ..Default::default()
        };
        h.vkCmdEndQuery(&mut args);
        assert!(h.reject.take().is_some());
        let mut args = vn_command_vkCmdResetQueryPool {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            firstQuery: 1,
            queryCount: 2,
            ..Default::default()
        };
        h.vkCmdResetQueryPool(&mut args);
        assert!(h.reject.take().is_some());
        let mut args = vn_command_vkCmdWriteTimestamp {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            query: 2,
            ..Default::default()
        };
        h.vkCmdWriteTimestamp(&mut args);
        assert!(h.reject.take().is_some());
        let mut args = vn_command_vkCmdCopyQueryPoolResults {
            commandBuffer: CB,
            queryPool: HOST_POOL,
            firstQuery: 2,
            queryCount: 1,
            ..Default::default()
        };
        h.vkCmdCopyQueryPoolResults(&mut args);
        assert!(h.reject.take().is_some());
        SAW.with_borrow(|s| assert!(s.is_empty(), "no refusal reached the driver"));

        // Destroyed, the pool's record goes with it, and a read of it is a refusal too.
        let mut args =
            vn_command_vkDestroyQueryPool { device, queryPool: HOST_POOL, ..Default::default() };
        h.vkDestroyQueryPool(&mut args);
        let mut args = read(device, 1);
        args.plant_pData(&mut room[..4]);
        h.vkGetQueryPoolResults(&mut args);
        assert!(h.reject.take().is_some(), "a pool with no record is not read");
        READS.with_borrow(|n| assert_eq!(*n, 1));

        h.driver.abandon_planted();
    }

    /// A refused pool allocation ghosts every id in the run, not just the first.
    ///
    /// Vulkan fills the whole array or none of it, and the generated lifecycle hook walks all of
    /// it either way -- so a refusal that decided about only one id leaves the guest holding
    /// command buffers the driver never made.
    #[test]
    fn a_refused_pool_allocation_ghosts_the_whole_run() {
        use super::super::cs::{Lookup, Objects};
        use super::super::proto::types::{VkCommandBuffer, VkCommandBufferAllocateInfo};

        const IDS: [u64; 3] = [31, 32, 33];

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let info = VkCommandBufferAllocateInfo {
            commandBufferCount: IDS.len() as u32,
            ..Default::default()
        };
        // What the decoder hands a handler: the guest's chosen ids in the wire member, and a
        // parallel run of zeroed shadows for the host handles that are never going to arrive.
        let mut asked = IDS.map(VkCommandBuffer);
        let mut shadow = [VkCommandBuffer(0); IDS.len()];
        // The count is `pAllocateInfo`'s, so the planters set only the pointers -- which is the
        // shape the decoder leaves too.
        let mut args = vn_command_vkAllocateCommandBuffers::default();
        args.pAllocateInfo = Some(&info);
        args.plant_pCommandBuffers(&mut asked);
        args.plant_handle_pCommandBuffers(&mut shadow);
        h.vkAllocateCommandBuffers(&mut args);

        assert_ne!(args.ret, VkResult::VK_SUCCESS, "there is no device to allocate from");
        for id in IDS {
            assert_eq!(
                objects.lookup(ObjectId(id), VkObjectType::VK_OBJECT_TYPE_COMMAND_BUFFER.0),
                Lookup::Ghost,
                "id {id} was in a refused run and must not become an object"
            );
        }
    }

    /// A guest whose count disagrees with the array behind it never reaches a handler.
    ///
    /// This is the premise every generated array accessor's `SAFETY` rests on. The accessor
    /// lengths its slice by the *count member*, while the arena allocation was sized by the
    /// *array size on the wire*. They are only ever the same number because the decoder refuses
    /// the command when they differ -- so if that refusal ever stopped happening, a guest sending
    /// `count = 7` behind three elements would hand a handler a seven-element slice over a
    /// three-element allocation, and nothing else in the harness would notice.
    ///
    /// `vkFreeCommandBuffers` stands in for the whole class: its array is the plain counted shape
    /// the other hundred-odd share.
    #[test]
    fn a_count_that_disagrees_with_the_array_behind_it_never_reaches_a_handler() {
        const DEVICE: u64 = 9;
        const POOL: u64 = 10;
        const BUFFERS: [u64; 3] = [21, 22, 23];

        #[derive(Default)]
        struct Recorder {
            saw: Option<usize>,
        }
        impl Commands for Recorder {
            fn unsupported(&mut self, _cmd: VkCommandTypeEXT) {}

            fn vkFreeCommandBuffers(&mut self, args: &mut vn_command_vkFreeCommandBuffers<'_>) {
                self.saw = Some(args.pCommandBuffers().map_or(usize::MAX, <[_]>::len));
            }

            fn object_created(
                &mut self,
                _ty: VkObjectType,
                _id: ObjectId,
                _host: HostHandle,
                _owner: Option<ObjectId>,
            ) {
            }
            fn object_destroyed(&mut self, _ty: VkObjectType, _id: ObjectId) {}
        }

        /// The wire for a free of `sent` buffers that claims to be freeing `counted` of them.
        fn wire(counted: u32, sent: &[u64]) -> Vec<u8> {
            let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkFreeCommandBuffers_EXT, 0);
            w.extend_from_slice(&DEVICE.to_le_bytes());
            w.extend_from_slice(&POOL.to_le_bytes());
            w.extend_from_slice(&counted.to_le_bytes());
            w.extend_from_slice(&(sent.len() as u64).to_le_bytes());
            for id in sent {
                w.extend_from_slice(&id.to_le_bytes());
            }
            w
        }

        fn run(w: &[u8], objects: &Shared) -> (Option<usize>, bool) {
            let temp = Bump::new();
            let hard = AtomicBool::new(false);
            let mut dec = Decoder::new(w, &temp, objects, &hard);
            let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
            let _flags = dec.decode_scalar::<VkFlags>();
            let mut h = Recorder::default();
            let verdict = vn_dispatch_command(&mut dec, None, cmd, &mut h);
            assert_eq!(
                verdict == Dispatched::Undecodable,
                dec.fatal(),
                "the verdict and the poison flag are one fact"
            );
            (h.saw, verdict == Dispatched::Undecodable)
        }

        let objects = Shared::new();
        {
            let mut t = objects.borrow_mut();
            t.add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(1), None)
                .unwrap();
            t.add(
                ObjectId(POOL),
                VkObjectType::VK_OBJECT_TYPE_COMMAND_POOL,
                HostHandle(2),
                Some(ObjectId(DEVICE)),
            )
            .unwrap();
            for (i, id) in BUFFERS.iter().enumerate() {
                t.add(
                    ObjectId(*id),
                    VkObjectType::VK_OBJECT_TYPE_COMMAND_BUFFER,
                    HostHandle(100 + i as u64),
                    Some(ObjectId(DEVICE)),
                )
                .unwrap();
            }
        }

        // The control: the guest agrees with itself, and the handler gets exactly what it sent.
        let (saw, fatal) = run(&wire(BUFFERS.len() as u32, &BUFFERS), &objects);
        assert!(!fatal, "a guest that agrees with itself is not a protocol violation");
        assert_eq!(saw, Some(BUFFERS.len()), "the handler gets the array the guest sent");

        // Overclaiming: seven counted, three sent. The ring stops and no handler runs.
        let (saw, fatal) = run(&wire(7, &BUFFERS), &objects);
        assert!(fatal, "a count with a shorter array behind it must poison the ring");
        assert_eq!(saw, None, "and must not reach a handler at all");

        // Underclaiming is the same violation from the other side: a handler given a count of one
        // over a three-element allocation is not unsound, but the guest still disagreed with
        // itself, and letting it through would mean the two numbers are not tied after all.
        let (saw, fatal) = run(&wire(1, &BUFFERS), &objects);
        assert!(fatal, "a count with a longer array behind it must poison the ring too");
        assert_eq!(saw, None);
    }

    /// A destroyed pool takes its objects out of the object table, not just out of the driver's.
    ///
    /// Vulkan frees a command pool's buffers with the pool, and the guest sends no command per
    /// buffer -- so nothing else in the stream says those ids stopped naming anything. Left in
    /// the table they go on resolving to host handles the driver has freed and may already have
    /// handed back out for something else, and the next `vkCmd*` naming one would carry that
    /// handle to Vulkan. This is why the recording commands need no liveness check of their own:
    /// the lookup is the gate, which is also what the C does (`vkr_command_pool_release`).
    #[test]
    fn a_destroyed_pool_takes_its_command_buffers_out_of_the_object_table() {
        use super::super::cs::{Lookup, Objects};
        use super::super::proto::types::{
            VkCommandBuffer, VkCommandPool, VkDevice, vn_command_vkDestroyCommandPool,
        };

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        /// The guest ids of two command buffers, and the host handles they were allocated as.
        const BUFFERS: [(u64, u64); 2] = [(11, 110), (12, 120)];
        const COMMAND_BUFFER: VkObjectType = VkObjectType::VK_OBJECT_TYPE_COMMAND_BUFFER;

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        {
            let mut t = objects.borrow_mut();
            for (host, id) in BUFFERS {
                t.add(ObjectId(id), COMMAND_BUFFER, HostHandle(host), None).unwrap();
            }
        }
        driver.plant_pool(
            VkDevice(DEVICE),
            VkCommandPool(POOL),
            &BUFFERS.map(|(host, id)| (VkCommandBuffer(host), ObjectId(id))),
        );
        assert_eq!(
            objects.lookup(ObjectId(BUFFERS[0].1), COMMAND_BUFFER.0),
            Lookup::Found(HostHandle(11))
        );

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        // The pool arrives as the host handle the lookup resolved, which is what the driver is
        // keyed by. There is no device registered, so the driver call is skipped -- the object
        // table is what is under test.
        let mut args = vn_command_vkDestroyCommandPool {
            device: VkDevice(DEVICE),
            commandPool: VkCommandPool(POOL),
            ..Default::default()
        };
        h.vkDestroyCommandPool(&mut args);

        for (_, id) in BUFFERS {
            assert_eq!(
                objects.lookup(ObjectId(id), COMMAND_BUFFER.0),
                Lookup::Missing,
                "id {id} was freed with its pool and must stop naming anything"
            );
        }
    }

    /// A device's objects are destroyed on the way out, before the device, in Vulkan's order.
    ///
    /// Forgetting a guest id and freeing a host handle are two different debts, and the table
    /// fix only paid the first. Vulkan does not free a device's objects when the device goes --
    /// it is undefined to destroy a device that still owns any -- so every fence, image and pool
    /// the guest left behind has to be destroyed here, while the device is still alive to destroy
    /// them on. The guest owes nothing: it may send `vkDestroyDevice` with everything still live,
    /// and a VM that stops mid-frame sends no destroy at all.
    ///
    /// So this plants real entry points and asks what the driver called, in what order. It is the
    /// only way to see it: no corpus contains a guest that leaves objects behind, and the object
    /// table is already empty by the time anyone could look.
    #[test]
    fn a_destroyed_device_destroys_what_it_owned_first_and_waits_for_it_to_be_idle() {
        use std::cell::RefCell;

        use super::super::proto::types::{
            VkAllocationCallbacks, VkCommandPool, VkDevice, VkFence, VkImage,
        };

        const DEVICE: u64 = 3;
        /// Guest id, host handle -- kept apart so a wrapper passing one for the other shows up.
        const FENCE: (u64, u64) = (11, 0xf0);
        const IMAGE: (u64, u64) = (12, 0xf1);
        const POOL: (u64, u64) = (13, 0xf2);

        #[derive(Default)]
        struct Saw {
            /// Every call, in order, so "waited then destroyed then dropped the device" is
            /// checkable rather than assumed.
            calls: Vec<(&'static str, u64)>,
        }
        thread_local! { static SAW: RefCell<Saw> = RefCell::new(Saw::default()); }
        fn saw(what: &'static str, h: u64) {
            SAW.with_borrow_mut(|s| s.calls.push((what, h)));
        }

        unsafe extern "C" fn wait_idle(_d: VkDevice) -> VkResult {
            saw("wait", 0);
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn fence(_d: VkDevice, h: VkFence, _a: *const VkAllocationCallbacks) {
            saw("fence", h.0);
        }
        unsafe extern "C" fn image(_d: VkDevice, h: VkImage, _a: *const VkAllocationCallbacks) {
            saw("image", h.0);
        }
        unsafe extern "C" fn pool(
            _d: VkDevice,
            h: VkCommandPool,
            _a: *const VkAllocationCallbacks,
        ) {
            saw("pool", h.0);
        }
        unsafe extern "C" fn device(h: VkDevice, _a: *const VkAllocationCallbacks) {
            saw("device", h.0);
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkDeviceWaitIdle(wait_idle);
        fns.plant_vkDestroyFence(fence);
        fns.plant_vkDestroyImage(image);
        fns.plant_vkDestroyCommandPool(pool);
        fns.plant_vkDestroyDevice(device);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        {
            let mut t = objects.borrow_mut();
            t.add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(DEVICE), None)
                .unwrap();
            let under = Some(ObjectId(DEVICE));
            t.add(
                ObjectId(FENCE.0),
                VkObjectType::VK_OBJECT_TYPE_FENCE,
                HostHandle(FENCE.1),
                under,
            )
            .unwrap();
            t.add(
                ObjectId(IMAGE.0),
                VkObjectType::VK_OBJECT_TYPE_IMAGE,
                HostHandle(IMAGE.1),
                under,
            )
            .unwrap();
            t.add(
                ObjectId(POOL.0),
                VkObjectType::VK_OBJECT_TYPE_COMMAND_POOL,
                HostHandle(POOL.1),
                under,
            )
            .unwrap();
        }

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyDevice_EXT, 0);
        w.extend_from_slice(&DEVICE.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Dispatched::Served);

        SAW.with_borrow(|s| {
            assert_eq!(s.calls.first(), Some(&("wait", 0)), "nothing may be destroyed while busy");
            assert_eq!(s.calls.last(), Some(&("device", DEVICE)), "the device goes last of all");
            let mut middle: Vec<_> = s.calls[1..s.calls.len() - 1].to_vec();
            middle.sort_unstable();
            assert_eq!(
                middle,
                [("fence", FENCE.1), ("image", IMAGE.1), ("pool", POOL.1)],
                "each object destroyed once, by its own entry point, with its host handle"
            );
        });
    }

    /// A context that nobody tore down still gives its host handles back.
    ///
    /// Three of the four ways a context ends never called a teardown: a duplicate context id
    /// replacing a live one, the renderer being dropped with contexts still in it, and a panic on
    /// the way up. Only the guest's own context destroy did -- so the common ending, a VM stopped
    /// mid-workload, leaked an instance and every device under it unless the VMM happened to send
    /// the destroy first. The teardown is the drop now, and this is what says so: nothing here
    /// calls it, and the destroys still happen.
    #[test]
    fn a_context_nobody_tore_down_still_destroys_what_it_stood_up() {
        use std::cell::RefCell;

        use super::super::proto::types::{VkAllocationCallbacks, VkFence};

        const DEVICE: u64 = 3;
        const FENCE: (u64, u64) = (11, 0xf0);

        thread_local! { static SAW: RefCell<Vec<(&'static str, u64)>> = const { RefCell::new(Vec::new()) }; }
        fn saw(what: &'static str, h: u64) {
            SAW.with_borrow_mut(|s| s.push((what, h)));
        }

        unsafe extern "C" fn wait_idle(_d: VkDevice) -> VkResult {
            saw("wait", 0);
            VkResult::VK_SUCCESS
        }
        unsafe extern "C" fn fence(_d: VkDevice, h: VkFence, _a: *const VkAllocationCallbacks) {
            saw("fence", h.0);
        }
        unsafe extern "C" fn device(h: VkDevice, _a: *const VkAllocationCallbacks) {
            saw("device", h.0);
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkDeviceWaitIdle(wait_idle);
        fns.plant_vkDestroyFence(fence);
        fns.plant_vkDestroyDevice(device);

        let mut ctx =
            Context::new(ContextId::new(7).expect("7 is not zero"), &Budget::with_cap(None, false));
        ctx.driver_mut().plant_device(VkDevice(DEVICE), fns);
        {
            let mut t = ctx.objects().borrow_mut();
            t.add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, HostHandle(DEVICE), None)
                .unwrap();
            t.add(
                ObjectId(FENCE.0),
                VkObjectType::VK_OBJECT_TYPE_FENCE,
                HostHandle(FENCE.1),
                Some(ObjectId(DEVICE)),
            )
            .unwrap();
        }

        drop(ctx);

        SAW.with_borrow(|s| {
            assert_eq!(
                s.as_slice(),
                [("wait", 0), ("fence", FENCE.1), ("device", DEVICE)],
                "the drop is the teardown, in the order a teardown owes Vulkan"
            );
        });
    }

    /// And an instance takes the whole tree, which is the same rule one level up.
    ///
    /// `vkDestroyInstance` tears the driver down without a command naming a single device,
    /// physical device or fence underneath -- so before parentage was recorded, every one of
    /// their ids went on resolving to a freed handle. Nothing here is special-cased for the
    /// instance: it is the root, and the cascade that empties a device empties it.
    #[test]
    fn destroying_the_instance_takes_the_whole_tree_with_it() {
        use super::super::cs::{Lookup, Objects};

        const INSTANCE: u64 = 1;
        const PHYSICAL_DEVICE: u64 = 2;
        const DEVICE: u64 = 3;
        const FENCE: u64 = 4;
        const FENCE_TY: VkObjectType = VkObjectType::VK_OBJECT_TYPE_FENCE;

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        {
            let mut t = objects.borrow_mut();
            t.add(
                ObjectId(INSTANCE),
                VkObjectType::VK_OBJECT_TYPE_INSTANCE,
                HostHandle(INSTANCE),
                None,
            )
            .unwrap();
            t.add(
                ObjectId(PHYSICAL_DEVICE),
                VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE,
                HostHandle(PHYSICAL_DEVICE),
                Some(ObjectId(INSTANCE)),
            )
            .unwrap();
            t.add(
                ObjectId(DEVICE),
                VkObjectType::VK_OBJECT_TYPE_DEVICE,
                HostHandle(DEVICE),
                Some(ObjectId(PHYSICAL_DEVICE)),
            )
            .unwrap();
            t.add(ObjectId(FENCE), FENCE_TY, HostHandle(0xfeed), Some(ObjectId(DEVICE))).unwrap();
        }

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyInstance_EXT, 0);
        w.extend_from_slice(&INSTANCE.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Dispatched::Served);
        assert!(!dec.fatal(), "the destroy must decode");

        assert_eq!(objects.borrow().len(), 0, "the instance was the root of everything");
        assert_eq!(objects.lookup(ObjectId(FENCE), FENCE_TY.0), Lookup::Missing);
        assert_eq!(
            objects.lookup(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE.0),
            Lookup::Missing
        );
    }

    /// A device takes *all* its objects with it, not just the ones that sat in a pool.
    ///
    /// Vulkan destroys a device's fences, semaphores, buffers and images along with it, naming
    /// none of them -- exactly as it does for pool contents. An entry left behind goes on
    /// resolving a guest id to a handle the driver has freed, and the guest does not even have to
    /// name the dead device to use it: it names a *live* one. That is what this test sends. The
    /// fence belongs to device A; A is destroyed; the guest then sends a command against device
    /// B carrying A's fence id. B resolves, the id resolves, and the driver is handed a freed
    /// handle with a live device to use it on -- a use-after-free a guest reaches on purpose.
    ///
    /// Neither corpus can witness this: a recording is a guest that behaved. The lookup is where
    /// it has to be caught, so the lookup is what is asserted.
    #[test]
    fn a_destroyed_device_takes_every_object_it_owned_and_not_only_its_pools() {
        use super::super::cs::{Lookup, Objects};

        /// The device the fence belongs to. Device B, the live one the guest would name next,
        /// needs no constant: the table is asked directly what its id would have resolved to,
        /// which is the same question any command against B would have asked.
        const DEVICE_A: u64 = 3;
        /// A fence created on A, under a guest id, holding a host handle that is not that id.
        const FENCE_ID: u64 = 11;
        const FENCE_HOST: u64 = 0xfeed_face_0000_0011;
        const FENCE: VkObjectType = VkObjectType::VK_OBJECT_TYPE_FENCE;

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        {
            let mut t = objects.borrow_mut();
            t.add(
                ObjectId(DEVICE_A),
                VkObjectType::VK_OBJECT_TYPE_DEVICE,
                HostHandle(DEVICE_A),
                None,
            )
            .unwrap();
            t.add(ObjectId(FENCE_ID), FENCE, HostHandle(FENCE_HOST), Some(ObjectId(DEVICE_A)))
                .unwrap();
        }
        assert_eq!(
            objects.lookup(ObjectId(FENCE_ID), FENCE.0),
            Lookup::Found(HostHandle(FENCE_HOST))
        );

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        // Driven over the wire rather than by calling the handler, because the handler is only
        // half of a destroy: the generated lifecycle hook is what tells the table an object died,
        // and it runs after the handler and nowhere else. No device is planted in the driver, so
        // no entry point is called -- what is under test is the table.
        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyDevice_EXT, 0);
        w.extend_from_slice(&DEVICE_A.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Dispatched::Served);
        assert!(!dec.fatal(), "the destroy must decode");

        assert_eq!(
            objects.lookup(ObjectId(FENCE_ID), FENCE.0),
            Lookup::Missing,
            "the fence died with device A, so its id must stop naming anything -- \
             left behind, a command against the still-live device B would hand the driver \
             a freed handle"
        );
    }

    /// What a recording handler hands the driver is exactly what the guest sent -- no more, no
    /// less, and in order.
    ///
    /// The boundary nothing else in the harness can see. The wire round trip never calls an
    /// accessor; the replay gate counts commands accounted for, not what they did, and measurably
    /// so -- an accessor handing back one element too many replays with every command accepted
    /// and the census unchanged. So the driver is stood up out of planted entry points and asked
    /// what it was called with.
    ///
    /// Four shapes, which is what the fifteen recording commands are made of: one counted array,
    /// two arrays under one count, three arrays with three counts, and a command with a result to
    /// carry back.
    #[test]
    fn a_recording_handler_hands_the_driver_what_the_guest_sent() {
        use super::super::proto::types::{
            VkBuffer, VkBufferMemoryBarrier, VkCommandBuffer, VkCommandBufferBeginInfo,
            VkCommandPool, VkDependencyFlags, VkDevice, VkDeviceSize, VkImageMemoryBarrier,
            VkMemoryBarrier, VkPipelineStageFlags, VkViewport, vn_command_vkBeginCommandBuffer,
            vn_command_vkCmdBindVertexBuffers, vn_command_vkCmdPipelineBarrier,
            vn_command_vkCmdSetViewport,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        /// The host handle of the one command buffer, and the guest id it was allocated under.
        const CB: (u64, u64) = (11, 110);

        #[derive(Default)]
        struct Saw {
            viewports: Vec<(u32, Vec<f32>)>,
            vertex_buffers: Vec<(u32, Vec<u64>, Vec<u64>)>,
            barriers: Vec<(u32, u32, u32)>,
            began: u32,
        }
        thread_local! {
            static SAW: RefCell<Saw> = RefCell::new(Saw::default());
        }

        unsafe extern "C" fn set_viewport(
            _cb: VkCommandBuffer,
            first: u32,
            count: u32,
            p: *const VkViewport,
        ) {
            // SAFETY: the wrapper under test passes a slice's own pointer and length.
            let vps = unsafe { core::slice::from_raw_parts(p, count as usize) };
            SAW.with_borrow_mut(|s| s.viewports.push((first, vps.iter().map(|v| v.x).collect())));
        }

        unsafe extern "C" fn bind_vertex_buffers(
            _cb: VkCommandBuffer,
            first: u32,
            count: u32,
            buffers: *const VkBuffer,
            offsets: *const VkDeviceSize,
        ) {
            // SAFETY: as above -- one count, and the wrapper asserts both slices share it.
            let (b, o) = unsafe {
                (
                    core::slice::from_raw_parts(buffers, count as usize),
                    core::slice::from_raw_parts(offsets, count as usize),
                )
            };
            SAW.with_borrow_mut(|s| {
                s.vertex_buffers.push((
                    first,
                    b.iter().map(|h| h.0).collect(),
                    o.iter().map(|v| v.0).collect(),
                ))
            });
        }

        #[allow(clippy::too_many_arguments)]
        unsafe extern "C" fn pipeline_barrier(
            _cb: VkCommandBuffer,
            _src: VkPipelineStageFlags,
            _dst: VkPipelineStageFlags,
            _dependency: VkDependencyFlags,
            memory: u32,
            _pm: *const VkMemoryBarrier,
            buffers: u32,
            _pb: *const VkBufferMemoryBarrier,
            images: u32,
            _pi: *const VkImageMemoryBarrier,
        ) {
            SAW.with_borrow_mut(|s| s.barriers.push((memory, buffers, images)));
        }

        unsafe extern "C" fn begin(
            _cb: VkCommandBuffer,
            _info: *const VkCommandBufferBeginInfo,
        ) -> VkResult {
            SAW.with_borrow_mut(|s| s.began += 1);
            VkResult::VK_ERROR_OUT_OF_DEVICE_MEMORY
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkCmdSetViewport(set_viewport);
        fns.plant_vkCmdBindVertexBuffers(bind_vertex_buffers);
        fns.plant_vkCmdPipelineBarrier(pipeline_barrier);
        fns.plant_vkBeginCommandBuffer(begin);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_pool(
            VkDevice(DEVICE),
            VkCommandPool(POOL),
            &[(VkCommandBuffer(CB.0), ObjectId(CB.1))],
        );

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let cb = VkCommandBuffer(CB.0);

        // One counted array: three viewports, and a first-index that is not zero so that a
        // wrapper passing the count where the index goes cannot pass unnoticed.
        let vps: [VkViewport; 3] =
            core::array::from_fn(|i| VkViewport { x: 10.0 + i as f32, ..Default::default() });
        let mut args = vn_command_vkCmdSetViewport::default();
        args.commandBuffer = cb;
        args.firstViewport = 2;
        args.plant_pViewports(&vps);
        h.vkCmdSetViewport(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|s| {
            assert_eq!(s.viewports, [(2, vec![10.0, 11.0, 12.0])], "all three, in order, at 2");
        });

        // Two arrays under one count: they have to arrive the same length and stay paired.
        let buffers = [VkBuffer(0x100), VkBuffer(0x200)];
        let offsets = [VkDeviceSize(64), VkDeviceSize(128)];
        let mut args = vn_command_vkCmdBindVertexBuffers::default();
        args.commandBuffer = cb;
        args.firstBinding = 1;
        args.plant_pBuffers(&buffers);
        args.plant_pOffsets(&offsets);
        h.vkCmdBindVertexBuffers(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|s| {
            assert_eq!(s.vertex_buffers, [(1, vec![0x100, 0x200], vec![64, 128])]);
        });

        // Three arrays with three counts of their own. Deliberately unequal lengths: every count
        // is a `u32` and only the pointer types differ, so passing one array's length where
        // another's belongs compiles, and nothing outside this can see it.
        let memory = [VkMemoryBarrier::default(); 1];
        let buffers = [VkBufferMemoryBarrier::default(); 2];
        let images = [VkImageMemoryBarrier::default(); 3];
        let mut args = vn_command_vkCmdPipelineBarrier::default();
        args.commandBuffer = cb;
        args.plant_pMemoryBarriers(&memory);
        args.plant_pBufferMemoryBarriers(&buffers);
        args.plant_pImageMemoryBarriers(&images);
        h.vkCmdPipelineBarrier(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|s| assert_eq!(s.barriers, [(1, 2, 3)], "each count with its own array"));

        // A result the guest is owed: the driver's answer has to reach the reply, not be
        // replaced by a success the renderer invented.
        let begin = VkCommandBufferBeginInfo::default();
        let mut args = vn_command_vkBeginCommandBuffer { commandBuffer: cb, ..Default::default() };
        args.pBeginInfo = Some(&begin);
        h.vkBeginCommandBuffer(&mut args);
        assert!(h.reject.is_none());
        assert_eq!(args.ret, VkResult::VK_ERROR_OUT_OF_DEVICE_MEMORY);
        SAW.with_borrow(|s| assert_eq!(s.began, 1));

        // And a command buffer the driver has no device for stops the ring rather than being
        // recorded into nothing.
        let mut args = vn_command_vkCmdSetViewport::default();
        args.commandBuffer = VkCommandBuffer(0xdead);
        args.plant_pViewports(&vps);
        h.vkCmdSetViewport(&mut args);
        assert!(h.reject.is_some(), "there is no device to record into");

        // Nothing here came from Vulkan, so there is nothing to destroy. See `abandon_planted`.
        h.driver.abandon_planted();
    }

    /// A pool allocation hands the driver one out-array of exactly the length the guest asked
    /// for, and files both names of everything that comes back.
    ///
    /// The shape the generated accessor witnesses cannot reach: the count is
    /// `pAllocateInfo->commandBufferCount`, inside a struct rather than beside the pointer, so a
    /// planter cannot establish it from a slice and the round trip has nothing to plant.
    ///
    /// Three things travel together here and are separately wrong-able. The wire array carries
    /// the *guest's* ids; the shadow beside it is where the driver writes *host* handles; and the
    /// pool has to end up holding each pair. Hand the driver the shadow and the ids in different
    /// orders, or one element short, and every id still resolves -- to the wrong object, for the
    /// rest of the context's life.
    #[test]
    fn a_pool_allocation_hands_over_the_run_the_guest_asked_for() {
        use super::super::proto::types::{
            VkCommandBuffer, VkCommandBufferAllocateInfo, VkCommandBufferLevel, VkCommandPool,
            VkDevice, vn_command_vkAllocateCommandBuffers,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        /// The host handles the driver hands back, and the guest ids the wire named them by.
        const HOST: [u64; 3] = [0x1100, 0x1200, 0x1300];
        const IDS: [u64; 3] = [510, 520, 530];

        thread_local! {
            /// The room the driver was given, and what it was asked to fill it from.
            static SAW: RefCell<Vec<(u32, usize)>> = const { RefCell::new(Vec::new()) };
            /// What the next allocation answers with.
            static ANSWER: RefCell<VkResult> = const {
                RefCell::new(VkResult::VK_SUCCESS)
            };
        }

        unsafe extern "C" fn allocate(
            _device: VkDevice,
            info: *const VkCommandBufferAllocateInfo,
            out: *mut VkCommandBuffer,
        ) -> VkResult {
            // SAFETY: the wrapper passes an arena allocation and an array it sized from the count
            // inside it -- which is the pairing under test.
            let info = unsafe { &*info };
            let n = info.commandBufferCount as usize;
            SAW.with_borrow_mut(|s| s.push((info.commandBufferCount, n)));
            let r = ANSWER.with_borrow(|a| *a);
            if r == VkResult::VK_SUCCESS {
                // SAFETY: Vulkan fills the whole array on success, and the count it was given is
                // the length of the array it was given.
                let out = unsafe { core::slice::from_raw_parts_mut(out, n) };
                for (e, h) in out.iter_mut().zip(HOST) {
                    *e = VkCommandBuffer(h);
                }
            }
            r
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkAllocateCommandBuffers(allocate);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_pool(VkDevice(DEVICE), VkCommandPool(POOL), &[]);

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        // What the decoder would have built: the guest's ids on the wire, a shadow of the same
        // length beside them, and the count inside the create-info that sized both.
        let info = VkCommandBufferAllocateInfo {
            commandPool: VkCommandPool(POOL),
            level: VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: IDS.len() as u32,
            ..Default::default()
        };
        let mut wire: [VkCommandBuffer; 3] = core::array::from_fn(|i| VkCommandBuffer(IDS[i]));
        let mut shadow = [VkCommandBuffer(0); 3];
        let mut args = vn_command_vkAllocateCommandBuffers::default();
        args.device = VkDevice(DEVICE);
        args.pAllocateInfo = Some(&info);
        args.plant_pCommandBuffers(&mut wire);
        args.plant_handle_pCommandBuffers(&mut shadow);

        h.vkAllocateCommandBuffers(&mut args);
        assert!(h.reject.is_none());
        assert_eq!(args.ret, VkResult::VK_SUCCESS);
        SAW.with_borrow(|s| {
            assert_eq!(s, &[(3, 3)], "the count it was told and the room it was given are one");
        });
        assert_eq!(shadow.map(|c| c.0), HOST, "the driver's handles land in the shadow");
        assert_eq!(wire.map(|c| c.0), IDS, "and the guest's ids on the wire are left alone");

        // Both names of every object, paired the way the guest sent them. A run recorded in the
        // wrong order resolves every id to a live object -- someone else's.
        for (host, id) in HOST.iter().zip(IDS) {
            assert_eq!(
                h.driver.pool_child_id(VkCommandPool(POOL), VkCommandBuffer(*host)),
                Some(ObjectId(id)),
                "the pool holds {host:#x} under the id the guest named it by"
            );
        }

        // A refusal ghosts every id in the run. Left plain missing, each of the commands the
        // guest already has in flight against them would poison the context instead of being
        // absorbed -- and the guest asked for a whole run, so it is every id or none.
        ANSWER.with_borrow_mut(|a| *a = VkResult::VK_ERROR_OUT_OF_POOL_MEMORY);
        // A fresh set of ids, so a ghost found below is this run's and not the last one's.
        let mut shadow = [VkCommandBuffer(0); 3];
        let mut args = vn_command_vkAllocateCommandBuffers::default();
        args.device = VkDevice(DEVICE);
        args.pAllocateInfo = Some(&info);
        args.plant_pCommandBuffers(&mut wire);
        args.plant_handle_pCommandBuffers(&mut shadow);
        h.vkAllocateCommandBuffers(&mut args);
        assert_eq!(args.ret, VkResult::VK_ERROR_OUT_OF_POOL_MEMORY);
        for id in IDS {
            assert!(
                objects.borrow().is_ghost(ObjectId(id)),
                "id {id} was asked for and refused, so it names a ghost and not nothing"
            );
        }

        h.driver.abandon_planted();
    }

    /// The commands that put bytes into memory, and the two shapes the four of them add.
    ///
    /// Left unserved, these read as an accounting gap -- so many commands the build refused --
    /// and the census reads as memory nothing wrote. That is the same evidence a renderer that
    /// records them and gets them wrong produces, which is why what they were called with is
    /// asserted here rather than inferred from a score.
    ///
    /// Two shapes beyond the recording four: two arrays under two independent counts, cleared as
    /// a product rather than a pair; and a blob whose length is its own `size` member.
    /// The two directions of an image/buffer copy take the same five values in a different order,
    /// and the layout belongs to whichever side is the image. A handler that mirrors its sibling
    /// too faithfully attaches the layout to the buffer -- which still compiles, because a layout
    /// is a layout -- so this asserts each value arrives where it belongs and as itself.
    ///
    /// Absent from three of the four corpora and present once in the fourth, which is what a
    /// live desktop asked for and this build refused.
    #[test]
    fn copying_an_image_to_a_buffer_hands_the_layout_to_the_image() {
        use super::super::proto::types::{
            VkBuffer, VkBufferImageCopy, VkCommandBuffer, VkCommandPool, VkDevice, VkImage,
            VkImageLayout, vn_command_vkCmdCopyImageToBuffer,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        const CB: (u64, u64) = (11, 110);

        thread_local! {
            static SAW: RefCell<Vec<(u64, i32, u64, u32)>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "C" fn copy(
            _cb: VkCommandBuffer,
            src: VkImage,
            layout: VkImageLayout,
            dst: VkBuffer,
            count: u32,
            p: *const VkBufferImageCopy,
        ) {
            // SAFETY: the wrapper passes a slice's own pointer and its own length.
            let regions = unsafe { core::slice::from_raw_parts(p, count as usize) };
            SAW.with_borrow_mut(|s| {
                s.push((src.0, layout.0, dst.0, regions.len() as u32));
            });
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkCmdCopyImageToBuffer(copy);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_pool(
            VkDevice(DEVICE),
            VkCommandPool(POOL),
            &[(VkCommandBuffer(CB.0), ObjectId(CB.1))],
        );

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        let regions = [VkBufferImageCopy::default(); 3];
        let mut args = vn_command_vkCmdCopyImageToBuffer::default();
        args.commandBuffer = VkCommandBuffer(CB.0);
        args.srcImage = VkImage(0x44);
        args.srcImageLayout = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        args.dstBuffer = VkBuffer(0x55);
        args.plant_pRegions(&regions);
        h.vkCmdCopyImageToBuffer(&mut args);

        assert!(h.reject.is_none(), "a served command does not reject");
        assert!(
            h.reject.is_none(),
            "the command is served now; a build that still refuses it fails here"
        );
        SAW.with_borrow(|s| {
            assert_eq!(
                s.as_slice(),
                [(0x44, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.0, 0x55, 3)],
                "the image, its layout, the buffer and all three regions, none standing in \
                 for another"
            );
        });

        // Nothing here came from Vulkan, so there is nothing to destroy.
        h.driver.abandon_planted();
    }

    #[test]
    fn the_commands_that_write_pixels_hand_the_driver_what_the_guest_sent() {
        use super::super::proto::types::{
            VkClearAttachment, VkClearColorValue, VkClearRect, VkCommandBuffer, VkCommandPool,
            VkDevice, VkFilter, VkImage, VkImageBlit, VkImageCopy, VkImageLayout,
            VkImageSubresourceRange, VkPipelineLayout, VkShaderStageFlags,
            vn_command_vkCmdBlitImage, vn_command_vkCmdClearAttachments,
            vn_command_vkCmdClearColorImage, vn_command_vkCmdCopyImage,
            vn_command_vkCmdPushConstants,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        const CB: (u64, u64) = (11, 110);

        #[derive(Default)]
        struct Saw {
            blits: Vec<(u64, u64, u32, i32)>,
            copies: Vec<(u64, u32, u64, u32, u32)>,
            cleared_image: Vec<(u64, u32, u32)>,
            cleared_attachments: Vec<(u32, u32)>,
            pushed: Vec<(u64, u32, Vec<u8>)>,
        }
        thread_local! {
            static SAW: RefCell<Saw> = RefCell::new(Saw::default());
        }

        #[allow(clippy::too_many_arguments)]
        unsafe extern "C" fn blit(
            _cb: VkCommandBuffer,
            src: VkImage,
            _src_layout: VkImageLayout,
            dst: VkImage,
            _dst_layout: VkImageLayout,
            count: u32,
            _p: *const VkImageBlit,
            filter: VkFilter,
        ) {
            SAW.with_borrow_mut(|s| s.blits.push((src.0, dst.0, count, filter.0)));
        }

        unsafe extern "C" fn copy(
            _cb: VkCommandBuffer,
            src: VkImage,
            src_layout: VkImageLayout,
            dst: VkImage,
            dst_layout: VkImageLayout,
            count: u32,
            _p: *const VkImageCopy,
        ) {
            SAW.with_borrow_mut(|s| {
                s.copies.push((src.0, src_layout.0 as u32, dst.0, dst_layout.0 as u32, count))
            });
        }

        unsafe extern "C" fn clear_color(
            _cb: VkCommandBuffer,
            image: VkImage,
            _layout: VkImageLayout,
            color: *const VkClearColorValue,
            count: u32,
            p: *const VkImageSubresourceRange,
        ) {
            // SAFETY: the wrapper passes a reference for the colour and a slice's own pointer
            // and length for the ranges.
            let (c, ranges) = unsafe { (&*color, core::slice::from_raw_parts(p, count as usize)) };
            let first = ranges.first().map(|r| r.baseMipLevel).unwrap_or(u32::MAX);
            SAW.with_borrow_mut(|s| s.cleared_image.push((image.0, unsafe { c.uint32[0] }, first)));
        }

        unsafe extern "C" fn clear_attachments(
            _cb: VkCommandBuffer,
            attachments: u32,
            _pa: *const VkClearAttachment,
            rects: u32,
            _pr: *const VkClearRect,
        ) {
            SAW.with_borrow_mut(|s| s.cleared_attachments.push((attachments, rects)));
        }

        unsafe extern "C" fn push(
            _cb: VkCommandBuffer,
            layout: VkPipelineLayout,
            _stages: VkShaderStageFlags,
            offset: u32,
            size: u32,
            values: *const core::ffi::c_void,
        ) {
            // SAFETY: the wrapper passes the slice's own pointer and its length in bytes.
            let bytes = unsafe { core::slice::from_raw_parts(values.cast::<u8>(), size as usize) };
            SAW.with_borrow_mut(|s| s.pushed.push((layout.0, offset, bytes.to_vec())));
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkCmdBlitImage(blit);
        fns.plant_vkCmdCopyImage(copy);
        fns.plant_vkCmdClearColorImage(clear_color);
        fns.plant_vkCmdClearAttachments(clear_attachments);
        fns.plant_vkCmdPushConstants(push);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_pool(
            VkDevice(DEVICE),
            VkCommandPool(POOL),
            &[(VkCommandBuffer(CB.0), ObjectId(CB.1))],
        );

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };
        let cb = VkCommandBuffer(CB.0);

        // A counted array with a scalar behind it: the filter follows the pointer, so a wrapper
        // that passes the count and pointer in the wrong order still compiles and lands here.
        let regions = [VkImageBlit::default(); 2];
        let mut args = vn_command_vkCmdBlitImage::default();
        args.commandBuffer = cb;
        args.srcImage = VkImage(0x11);
        args.dstImage = VkImage(0x22);
        args.filter = VkFilter::VK_FILTER_LINEAR;
        args.plant_pRegions(&regions);
        h.vkCmdBlitImage(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|s| {
            assert_eq!(
                s.blits,
                [(0x11, 0x22, 2, VkFilter::VK_FILTER_LINEAR.0)],
                "source, destination, both regions and the filter, none of them each other"
            );
        });

        // The same five values as the blit above with the filter removed, which is the whole
        // difference between the two: a copy's regions name one extent, so there is nothing to
        // filter. That makes them the pair a handler is most likely to mirror into each other,
        // and the two layouts are what catches it -- a blit's stub ignores them, so only asking
        // for them separately here shows they arrived on the sides the guest put them on.
        let regions = [VkImageCopy::default(); 3];
        let mut args = vn_command_vkCmdCopyImage::default();
        args.commandBuffer = cb;
        args.srcImage = VkImage(0x66);
        args.srcImageLayout = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        args.dstImage = VkImage(0x77);
        args.dstImageLayout = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        args.plant_pRegions(&regions);
        h.vkCmdCopyImage(&mut args);
        assert!(h.reject.is_none());
        assert!(
            h.reject.is_none(),
            "the command is served now; a build that still refuses it fails here"
        );
        SAW.with_borrow(|s| {
            assert_eq!(
                s.copies,
                [(
                    0x66,
                    VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.0 as u32,
                    0x77,
                    VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.0 as u32,
                    3
                )],
                "each image with its own layout, and all three regions"
            );
        });

        // A by-ref value beside a counted array. The colour is the whole point of the command,
        // so it has to arrive as the guest set it and not as a default.
        let color = VkClearColorValue { uint32: [0xabcd_ef01, 0, 0, 0] };
        let ranges = [VkImageSubresourceRange { baseMipLevel: 4, ..Default::default() }];
        let mut args = vn_command_vkCmdClearColorImage::default();
        args.commandBuffer = cb;
        args.image = VkImage(0x33);
        args.pColor = Some(&color);
        args.plant_pRanges(&ranges);
        h.vkCmdClearColorImage(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|s| {
            assert_eq!(s.cleared_image, [(0x33, 0xabcd_ef01, 4)], "the guest's colour and range");
        });

        // Two arrays under two counts. Unequal on purpose: both counts are `u32` and the
        // pointers differ only in type, so passing one where the other belongs compiles.
        let attachments = [VkClearAttachment::default(); 2];
        let rects = [VkClearRect::default(); 3];
        let mut args = vn_command_vkCmdClearAttachments::default();
        args.commandBuffer = cb;
        args.plant_pAttachments(&attachments);
        args.plant_pRects(&rects);
        h.vkCmdClearAttachments(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|s| {
            assert_eq!(s.cleared_attachments, [(2, 3)], "each count with its own array");
        });

        // A blob measured by its own `size`, with a non-zero offset so a wrapper that passes the
        // length where the offset goes cannot pass unnoticed.
        let bytes = [0xde_u8, 0xad, 0xbe, 0xef, 0x11, 0x22];
        let mut args = vn_command_vkCmdPushConstants::default();
        args.commandBuffer = cb;
        args.layout = VkPipelineLayout(0x44);
        args.offset = 8;
        args.plant_pValues(&bytes);
        h.vkCmdPushConstants(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|s| {
            assert_eq!(s.pushed, [(0x44, 8, bytes.to_vec())], "the offset and every byte");
        });

        // A size with no bytes behind it. Pushing nothing would leave the layout holding whatever
        // the last push left, and the next draw would read constants the guest never sent.
        let mut args = vn_command_vkCmdPushConstants::default();
        args.commandBuffer = cb;
        args.size = 4;
        h.vkCmdPushConstants(&mut args);
        assert!(h.reject.is_some(), "a count with no blob behind it stops the ring");
        SAW.with_borrow(|s| assert_eq!(s.pushed.len(), 1, "and pushes nothing"));

        // Nothing here came from Vulkan, so there is nothing to destroy.
        h.driver.abandon_planted();
    }

    /// What a sync handler hands the driver, and what it does when there is nothing behind the
    /// handle the guest named.
    ///
    /// `vkQueueSubmit` is the one command in the corpus that carries a whole tree of the guest's
    /// work -- the submit infos, and inside each the semaphores and command buffers the decoder
    /// already resolved. The replay gate sees it as one command accounted for either way, so the
    /// count and the fence arriving intact is measured here or nowhere.
    #[test]
    fn a_sync_handler_hands_the_driver_what_the_guest_sent() {
        use super::super::proto::types::{
            VkBool32, VkDevice, VkFence, VkImportSemaphoreFdInfoKHR,
            VkImportSemaphoreResourceInfoMESA, VkQueue, VkSemaphore, VkSemaphoreGetFdInfoKHR,
            VkSubmitInfo, vn_command_vkImportSemaphoreResourceMESA, vn_command_vkQueueSubmit,
            vn_command_vkResetFences, vn_command_vkWaitForFences,
            vn_command_vkWaitSemaphoreResourceMESA,
        };
        use std::cell::RefCell;

        const DEVICE: u64 = 3;
        const QUEUE: u64 = 21;

        #[derive(Default)]
        struct Saw {
            submits: Vec<(u64, u32, u64)>,
            reset: Vec<u32>,
            waited: Vec<(u32, u32, u64)>,
            imported: u32,
            /// The read end of each pipe the export handed the write end of. Asking whether a
            /// descriptor number is still open would be asking the wrong question: the tests run
            /// in one process, and another thread's open takes the number the moment it is free.
            /// A read end at EOF says the write end was closed, whoever holds that number now.
            exported: Vec<core::ffi::c_int>,
        }
        thread_local! {
            static SAW: RefCell<Saw> = RefCell::new(Saw::default());
        }

        unsafe extern "C" fn submit(
            queue: VkQueue,
            count: u32,
            _p: *const VkSubmitInfo,
            fence: VkFence,
        ) -> VkResult {
            SAW.with_borrow_mut(|s| s.submits.push((queue.0, count, fence.0)));
            VkResult::VK_SUCCESS
        }

        unsafe extern "C" fn reset(_d: VkDevice, count: u32, _p: *const VkFence) -> VkResult {
            SAW.with_borrow_mut(|s| s.reset.push(count));
            VkResult::VK_SUCCESS
        }

        unsafe extern "C" fn wait(
            _d: VkDevice,
            count: u32,
            _p: *const VkFence,
            all: VkBool32,
            timeout: u64,
        ) -> VkResult {
            SAW.with_borrow_mut(|s| s.waited.push((count, all.0, timeout)));
            VkResult::VK_TIMEOUT
        }

        unsafe extern "C" fn import(
            _d: VkDevice,
            _info: *const VkImportSemaphoreFdInfoKHR,
        ) -> VkResult {
            SAW.with_borrow_mut(|s| s.imported += 1);
            VkResult::VK_SUCCESS
        }

        unsafe extern "C" fn export(
            _d: VkDevice,
            _info: *const VkSemaphoreGetFdInfoKHR,
            out: *mut core::ffi::c_int,
        ) -> VkResult {
            // A pipe, so the close the handler owes the descriptor is a thing the test can
            // see: the read end reaches EOF exactly when the write end is closed.
            let mut ends = [0 as core::ffi::c_int; 2];
            // SAFETY: `pipe` writes two descriptors into the array it is given.
            assert_eq!(unsafe { libc::pipe(ends.as_mut_ptr()) }, 0, "the test needs a pipe");
            // SAFETY: a flag on a descriptor this test owns; a read must not block if the
            // write end is still open, which is the failure being tested for.
            unsafe { libc::fcntl(ends[0], libc::F_SETFL, libc::O_NONBLOCK) };
            // SAFETY: the caller is `export_semaphore_sync_fd`, which passes a live `c_int`.
            unsafe { *out = ends[1] };
            SAW.with_borrow_mut(|s| s.exported.push(ends[0]));
            VkResult::VK_SUCCESS
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkQueueSubmit(submit);
        fns.plant_vkGetSemaphoreFdKHR(export);
        fns.plant_vkResetFences(reset);
        fns.plant_vkWaitForFences(wait);
        fns.plant_vkImportSemaphoreFdKHR(import);

        let objects = Shared::new();
        let mut driver = Driver::new(Account::for_test(None));
        driver.plant_device(VkDevice(DEVICE), fns);
        driver.plant_queue(VkDevice(DEVICE), VkQueue(QUEUE));

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut rings = BTreeMap::new();
        let mut ctx_reply = None;
        let mut monitor = None;
        let mut jrnl = Journal::new();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            ctx: ContextId::new(1).expect("1 is not zero"),
            reject: None,
            resources: &NO_RESOURCES,
            rings: &mut rings,
            monitor: &mut monitor,
            wait: None,
            execute: None,
            replaying: false,
            current_ring: None,
            reply: &mut ctx_reply,
            note: None,
            journal: &mut jrnl,
        };

        // A submit is two submit infos and a fence. The fence has to arrive as itself: it is the
        // handle every later wait in the frame is keyed by, and passing the wrong one -- or none
        // -- would leave the guest waiting on something nothing signals.
        let submits = [VkSubmitInfo::default(); 2];
        let mut args = vn_command_vkQueueSubmit::default();
        args.queue = VkQueue(QUEUE);
        args.fence = VkFence(99);
        args.plant_pSubmits(&submits);
        h.vkQueueSubmit(&mut args);
        assert!(h.reject.is_none());
        assert_eq!(args.ret, VkResult::VK_SUCCESS, "the driver's answer is the guest's");
        SAW.with_borrow(|s| assert_eq!(s.submits, [(QUEUE, 2, 99)]));

        // Three fences to reset, then a wait on two of them. `waitAll` and the timeout are the
        // guest's own: a wait that returned early would be a lie it cannot tell from the truth.
        let fences = [VkFence(1), VkFence(2), VkFence(3)];
        let mut args = vn_command_vkResetFences::default();
        args.device = VkDevice(DEVICE);
        args.plant_pFences(&fences);
        h.vkResetFences(&mut args);
        SAW.with_borrow(|s| assert_eq!(s.reset, [3]));

        let mut args = vn_command_vkWaitForFences::default();
        args.device = VkDevice(DEVICE);
        args.plant_pFences(&fences[..2]);
        args.waitAll = VkBool32(1);
        args.timeout = u64::MAX;
        h.vkWaitForFences(&mut args);
        assert_eq!(args.ret, VkResult::VK_TIMEOUT, "a timeout is an answer, not a failure");
        SAW.with_borrow(|s| assert_eq!(s.waited, [(2, 1, u64::MAX)]));

        // The import that stands in for a signal the host never saw.
        let info =
            VkImportSemaphoreResourceInfoMESA { semaphore: VkSemaphore(5), ..Default::default() };
        let mut args = vn_command_vkImportSemaphoreResourceMESA {
            device: VkDevice(DEVICE),
            pImportSemaphoreResourceInfo: Some(&info),
            ..Default::default()
        };
        h.vkImportSemaphoreResourceMESA(&mut args);
        assert!(h.reject.is_none());
        SAW.with_borrow(|s| assert_eq!(s.imported, 1));

        // The export half. Its whole effect is outside Vulkan, so what there is to check is that
        // the descriptor it produced was closed -- an fd leaked once per frame is a renderer that
        // runs out of them.
        let mut args = vn_command_vkWaitSemaphoreResourceMESA {
            device: VkDevice(DEVICE),
            semaphore: VkSemaphore(5),
            ..Default::default()
        };
        h.vkWaitSemaphoreResourceMESA(&mut args);
        assert!(h.reject.is_none());
        let read_end = SAW.with_borrow(|s| {
            assert_eq!(s.exported.len(), 1);
            s.exported[0]
        });
        let mut byte = 0u8;
        // SAFETY: a one-byte read from a descriptor this test owns, into a live byte.
        let n = unsafe { libc::read(read_end, (&raw mut byte).cast(), 1) };
        assert_eq!(n, 0, "the exported descriptor must not outlive the command that made it");
        // SAFETY: the read end, which nothing else holds.
        unsafe { libc::close(read_end) };

        // A resource id the C asserts on. The number is the guest's, so it is a rejection here --
        // an assert would hand a guest the power to abort the process.
        let info = VkImportSemaphoreResourceInfoMESA {
            semaphore: VkSemaphore(5),
            resourceId: 1,
            ..Default::default()
        };
        let mut args = vn_command_vkImportSemaphoreResourceMESA {
            device: VkDevice(DEVICE),
            pImportSemaphoreResourceInfo: Some(&info),
            ..Default::default()
        };
        h.vkImportSemaphoreResourceMESA(&mut args);
        assert!(h.reject.is_some(), "a resource-backed import is not something this serves");
        SAW.with_borrow(|s| assert_eq!(s.imported, 1, "and it must not have reached the driver"));

        // A driver with no `vkImportSemaphoreFdKHR` at all. The proc table's own answer to a
        // missing entry point is `.expect()`, and this build aborts on panic -- so for the two
        // commands a guest reaches an extension through, the predicate is what stands between a
        // driver we cannot serve on and a guest that can kill the process by asking.
        const BARE: u64 = 8;
        h.driver.plant_device(VkDevice(BARE), crate::vulkan::Device::default());
        let info =
            VkImportSemaphoreResourceInfoMESA { semaphore: VkSemaphore(5), ..Default::default() };
        let mut args = vn_command_vkImportSemaphoreResourceMESA {
            device: VkDevice(BARE),
            pImportSemaphoreResourceInfo: Some(&info),
            ..Default::default()
        };
        h.reject = None;
        h.vkImportSemaphoreResourceMESA(&mut args);
        assert!(h.reject.is_some(), "a driver without the extension is a rejection, not an abort");
        SAW.with_borrow(|s| assert_eq!(s.imported, 1));

        // A queue the context never retrieved stops the ring rather than reaching Vulkan with a
        // handle nothing vouches for.
        h.reject = None;
        let mut args = vn_command_vkQueueSubmit::default();
        args.queue = VkQueue(4242);
        args.plant_pSubmits(&submits);
        h.vkQueueSubmit(&mut args);
        assert!(h.reject.is_some(), "a queue with no device behind it must poison the ring");
        SAW.with_borrow(|s| assert_eq!(s.submits.len(), 1, "and must not reach the driver"));

        // Nothing here came from Vulkan, so there is nothing to destroy. See `abandon_planted`.
        h.driver.abandon_planted();
    }

    /// The recorder's classifiers, which decide what a journal keeps. Tested apart from the tee
    /// because a wrong answer here is silent: the command is still served, and only a restore
    /// months later finds the entry missing or stale. The tee's own wiring -- drain before every
    /// branch, commit only after the command has fully succeeded -- is scored by the harness's
    /// `--rebuild` gate, which is the fixed point this cannot check on its own.
    mod recorder {
        use super::super::{GENERATE_REPLY, VkCommandTypeEXT};
        use super::super::{Recording, mutates, recording_class, strip_reply_flag};

        #[test]
        fn every_vk_cmd_records_and_the_brackets_say_which_start_over() {
            let adds = |c| matches!(recording_class(c), Some(Recording::Adds));
            let resets = |c| matches!(recording_class(c), Some(Recording::Resets));

            assert!(adds(VkCommandTypeEXT::VK_COMMAND_TYPE_vkCmdBindPipeline_EXT));
            assert!(adds(VkCommandTypeEXT::VK_COMMAND_TYPE_vkCmdDraw_EXT));
            assert!(adds(VkCommandTypeEXT::VK_COMMAND_TYPE_vkEndCommandBuffer_EXT));

            assert!(resets(VkCommandTypeEXT::VK_COMMAND_TYPE_vkBeginCommandBuffer_EXT));
            assert!(resets(VkCommandTypeEXT::VK_COMMAND_TYPE_vkResetCommandBuffer_EXT));

            // Not a recording. It destroys the buffers, which takes their keys with them, and
            // every recording naming one stops being true without anything being pruned. Keeping
            // it would replay a free of buffers the restore has just rebuilt.
            assert!(
                recording_class(VkCommandTypeEXT::VK_COMMAND_TYPE_vkFreeCommandBuffers_EXT)
                    .is_none()
            );
            // Nor is a create, however much it looks like a pool operation.
            assert!(
                recording_class(VkCommandTypeEXT::VK_COMMAND_TYPE_vkCreateCommandPool_EXT)
                    .is_none()
            );
        }

        /// A command class is one or the other, never both: a recording hangs off its buffer and a
        /// mutation off every object it wrote, and an entry cannot be true under two rules.
        #[test]
        fn nothing_is_both_a_recording_and_a_mutation() {
            for raw in 0..2000i32 {
                let cmd = VkCommandTypeEXT(raw);
                assert!(
                    !(recording_class(cmd).is_some() && mutates(cmd)),
                    "{cmd:?} is classified twice"
                );
            }
        }

        #[test]
        fn the_reply_flag_is_cleared_and_nothing_else_moves() {
            let mut wire = vec![0u8; 16];
            wire[0..4].copy_from_slice(&7u32.to_le_bytes());
            wire[4..8].copy_from_slice(&(GENERATE_REPLY | 0x40).to_le_bytes());
            wire[8..16].copy_from_slice(&[0xab; 8]);

            let out = strip_reply_flag(&wire);
            assert_eq!(u32::from_le_bytes(out[0..4].try_into().unwrap()), 7, "the type is intact");
            assert_eq!(
                u32::from_le_bytes(out[4..8].try_into().unwrap()),
                0x40,
                "the reply bit goes and every other flag stays"
            );
            assert_eq!(&out[8..16], &[0xab; 8], "the arguments are untouched");
        }

        /// A command with no flags word cannot be a recorded one -- the loop breaks on a short
        /// header before dispatching -- but the strip must not panic if one ever reaches it.
        #[test]
        fn a_wire_too_short_to_have_flags_is_left_alone() {
            assert_eq!(strip_reply_flag(&[1, 2, 3]), vec![1, 2, 3]);
            assert_eq!(strip_reply_flag(&[]), Vec::<u8>::new());
        }
    }
}
