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
use std::cell::Cell;

use crate::ids::{CtxId, RingIdx};

use super::cs::Decoder;
use super::cs::Handle;
use super::cs::ObjectId;
use super::driver::Driver;
use super::objects::Shared;
use super::proto::serialize::{Commands, vn_command_name, vn_dispatch_command};
use super::proto::types::{
    VkCommandTypeEXT, VkFlags, VkObjectType, VkResult, vn_command_vkAllocateCommandBuffers,
    vn_command_vkAllocateDescriptorSets, vn_command_vkAllocateMemory,
    vn_command_vkBeginCommandBuffer, vn_command_vkBindBufferMemory2, vn_command_vkBindImageMemory2,
    vn_command_vkCmdBeginRenderPass, vn_command_vkCmdBindDescriptorSets,
    vn_command_vkCmdBindPipeline, vn_command_vkCmdBindVertexBuffers, vn_command_vkCmdCopyBuffer,
    vn_command_vkCmdCopyBufferToImage, vn_command_vkCmdDraw, vn_command_vkCmdEndRenderPass,
    vn_command_vkCmdFillBuffer, vn_command_vkCmdPipelineBarrier, vn_command_vkCmdSetScissor,
    vn_command_vkCmdSetViewport, vn_command_vkCreateBuffer, vn_command_vkCreateCommandPool,
    vn_command_vkCreateDescriptorPool, vn_command_vkCreateDescriptorSetLayout,
    vn_command_vkCreateDevice, vn_command_vkCreateFence, vn_command_vkCreateFramebuffer,
    vn_command_vkCreateGraphicsPipelines, vn_command_vkCreateImage, vn_command_vkCreateImageView,
    vn_command_vkCreateInstance, vn_command_vkCreatePipelineCache,
    vn_command_vkCreatePipelineLayout, vn_command_vkCreateRenderPass, vn_command_vkCreateSampler,
    vn_command_vkCreateSemaphore, vn_command_vkCreateShaderModule, vn_command_vkDestroyBuffer,
    vn_command_vkDestroyCommandPool, vn_command_vkDestroyDescriptorPool,
    vn_command_vkDestroyDescriptorSetLayout, vn_command_vkDestroyDevice, vn_command_vkDestroyFence,
    vn_command_vkDestroyFramebuffer, vn_command_vkDestroyImage, vn_command_vkDestroyImageView,
    vn_command_vkDestroyInstance, vn_command_vkDestroyPipeline, vn_command_vkDestroyPipelineCache,
    vn_command_vkDestroyPipelineLayout, vn_command_vkDestroyRenderPass,
    vn_command_vkDestroySampler, vn_command_vkDestroySemaphore, vn_command_vkDestroyShaderModule,
    vn_command_vkEndCommandBuffer, vn_command_vkEnumeratePhysicalDevices,
    vn_command_vkFreeCommandBuffers, vn_command_vkFreeMemory, vn_command_vkGetDeviceQueue2,
    vn_command_vkResetCommandBuffer, vn_command_vkUpdateDescriptorSets,
};
use crate::vulkan::Global;

/// `VK_COMMAND_GENERATE_REPLY_BIT_EXT`: the guest wants an answer to this command.
const GENERATE_REPLY: u32 = 0x1;

pub struct Context {
    pub id: CtxId,
    /// The hard poison. It outlives any one command and any one submission: once the stream cannot
    /// be trusted, nothing later in it can be either.
    fatal: Cell<bool>,
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
}

impl Context {
    pub fn new(id: CtxId) -> Context {
        Context {
            id,
            fatal: Cell::new(false),
            objects: Shared::new(),
            driver: Driver::new(),
            replay: false,
            dispatched: 0,
            unhandled: 0,
        }
    }

    pub fn fatal(&self) -> bool {
        self.fatal.get()
    }

    pub fn objects(&self) -> &Shared {
        &self.objects
    }

    /// Enter replay mode: the journal is about to be fed in, so nothing may answer it.
    pub fn replay_begin(&mut self) {
        self.replay = true;
    }

    /// Leave replay mode. In the C this is also where deferred rings start; there are no rings to
    /// start until a ring loop exists.
    pub fn replay_end(&mut self) {
        self.replay = false;
    }

    /// Drain one submission, dispatching every command in it.
    ///
    /// Returns false when the context was poisoned -- by this batch or by an earlier one. The C
    /// bails early on an already-fatal context for the same reason: a stream we stopped trusting
    /// does not become trustworthy because the guest sent more of it.
    pub fn submit(&mut self, buf: &[u8], todo: &mut Unimplemented, global: &Global) -> bool {
        if self.fatal.get() {
            return false;
        }

        // One arena for the batch. Every temporary a command decodes into lives until the batch
        // ends, which is the same bargain the C makes with its temp pool -- and the decoder's own
        // cap, not this arena, is what stops a guest from asking for all of memory.
        // Read out what the poison path needs before the handlers borrow the rest of the
        // context: they hold the driver mutably for as long as the loop runs.
        let id = self.id;
        let replay = self.replay;
        let fatal = &self.fatal;
        let (mut dispatched, mut unhandled) = (0u64, 0u64);

        let temp = Bump::new();
        let mut dec = Decoder::new(buf, &temp, &self.objects, fatal);
        let mut h = Handlers {
            objects: &self.objects,
            todo,
            driver: &mut self.driver,
            global,
            reject: None,
        };

        while dec.has_command() {
            dec.clear_soft_fatal();

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

            // A command that wants an answer has nowhere to be answered into: the reply buffer is
            // the ring's, and there is no ring loop yet. Poisoning is the honest response --
            // dispatching and dropping the reply would leave the guest waiting on a reply that was
            // never written, which is a hang rather than an error. Replay never takes this branch:
            // the journal's entries have had their reply flag stripped already.
            if flags.0 & GENERATE_REPLY != 0 && !replay {
                unhandled += 1;
                poison(fatal, id, &dec, cmd, "wants a reply, and there is no ring to answer into");
                break;
            }

            if vn_dispatch_command(&mut dec, None, cmd, &mut h).is_none() {
                // A command type this protocol does not define. We cannot even skip it: its length
                // is only knowable by decoding it.
                poison(fatal, id, &dec, cmd, "is not a command type this protocol defines");
                break;
            }
            dispatched += 1;
            // A handler that found the command itself unusable -- an id the guest cannot have, a
            // length that would send the driver off the end of what was decoded. The handler has
            // no decoder to say so with; this is where its verdict lands.
            if let Some(why) = h.reject.take() {
                poison(fatal, id, &dec, cmd, why);
            }

            if fatal.get() {
                // The decoder poisoned itself inside the command: a malformed argument, or a
                // shape the generator has no decoder for. Either way the command is what a
                // reader needs, because without it a gap reaches a user as a hung guest.
                poison(fatal, id, &dec, cmd, "did not decode");
                break;
            }
        }
        self.dispatched += dispatched;
        self.unhandled += unhandled;
        // Every exit from the loop is one place, so a branch that poisons and breaks cannot report
        // success on the way out.
        !self.fatal.get()
    }

    /// A ring-scoped submission. The ring the command belongs to is recorded but not yet acted on:
    /// the replay feed hands commands straight to the dispatcher, which is what makes a VM-free
    /// replay possible, and a real ring loop is what will need the index.
    pub fn submit_ring(
        &mut self,
        _ring: RingIdx,
        buf: &[u8],
        todo: &mut Unimplemented,
        global: &Global,
    ) -> bool {
        self.submit(buf, todo, global)
    }

    /// The driver state, for the teardown that has to destroy what it holds.
    pub fn driver_mut(&mut self) -> &mut Driver {
        &mut self.driver
    }

    /// The driver state, for the census that has to read what it holds.
    pub fn driver(&self) -> &Driver {
        &self.driver
    }
}

/// Poison a context, naming the command that did it -- once.
///
/// A ring the guest can no longer use looks the same from inside the guest whatever caused it, so
/// the command type is the only thing that tells a bug report from a hostile stream apart.
///
/// Free-standing rather than a method because the dispatch loop has already lent the rest of the
/// context to the handlers by the time it needs this.
fn poison(fatal: &Cell<bool>, id: CtxId, dec: &Decoder<'_>, cmd: VkCommandTypeEXT, why: &str) {
    if !fatal.get() {
        let name = vn_command_name(cmd)
            .map(str::to_string)
            .unwrap_or_else(|| format!("command type {}", cmd.0));
        eprintln!("[virglrs] ctx {id}: {name} {why}, {} bytes in", dec.pos());
    }
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
}

impl Handlers<'_> {
    /// The guest id a single out-handle carries, or None when the guest asked for no object.
    fn out_id<T: Handle>(&self, out: *const T) -> Option<ObjectId> {
        if out.is_null() {
            return None;
        }
        // SAFETY: non-null, and the decoder allocated one element in the arena.
        Some(ObjectId(unsafe { *out }.raw()))
    }

    /// Write what the driver produced into the shadow the generated hook will read, or ghost the
    /// id when it produced nothing.
    ///
    /// A guest pipelines: it sends a create and the commands using it without waiting for an
    /// answer, so those are already in flight when the create fails. A ghost turns each of them
    /// into one lost command instead of a poisoned ring -- see `objects::Table::ghosts`. Leaving
    /// the shadow zero would instead register the id as its own handle, which is the unserved
    /// command's fiction and a lie for a served one.
    fn plant<T: Handle>(
        &mut self,
        what: &str,
        out: *const T,
        shadow: *mut T,
        host: Result<u64, VkResult>,
    ) {
        let Some(id) = self.out_id(out) else {
            return;
        };
        match host {
            Ok(h) if h != 0 => {
                if !shadow.is_null() {
                    // SAFETY: non-null, and the decoder allocated one element in the arena.
                    unsafe { *shadow = T::from_raw(h) };
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

    /// The pool an allocation names, as a host handle -- zero when the info is missing, which no
    /// live pool can be, so the driver's re-check refuses it.
    fn pool_of<I>(&self, info: Option<&I>, pool: impl FnOnce(&I) -> u64) -> u64 {
        info.map_or(0, pool)
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

    /// The other honest reading of a split pair, for the arrays where it is the right one.
    ///
    /// Exactly the arrays vk.xml marks `noautovalidity` -- `vkFreeCommandBuffers`,
    /// `vkFreeDescriptorSets` -- where the decoder deliberately does not check the size, so the
    /// pair can genuinely arrive apart. Freeing "three, list not supplied" identifies nothing to
    /// free, which is not the same as claiming work was done: there is no work to claim. Poisoning
    /// a ring over it would cost the guest everything to punish a request that asked for nothing.
    fn array_or_empty<'w, T>(&mut self, a: Option<&'w [T]>) -> &'w [T] {
        a.unwrap_or_default()
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

    fn ghost_ids<T: Handle>(&mut self, ids: &[T]) {
        for id in ids {
            self.objects.borrow_mut().add_ghost(ObjectId(id.raw()));
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
            let host =
                self.driver.create_object(args.device, |d| d.$cmd(), args.$info, args.pAllocator);
            args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
            self.plant(stringify!($cmd), args.$out, args.$shadow, host);
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
            let host =
                self.driver.create_pool(args.device, |d| d.$cmd(), args.$info, args.pAllocator);
            args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
            self.plant(stringify!($cmd), args.$out, args.$shadow, host);
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

impl Commands for Handlers<'_> {
    fn unsupported(&mut self, cmd: VkCommandTypeEXT) {
        *self.todo.seen.entry(cmd.0).or_default() += 1;
    }

    fn object_created(&mut self, ty: VkObjectType, id: ObjectId, host: u64) {
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
        let handle = if host == 0 { id.0 } else { host };
        if self.objects.borrow_mut().add(id, ty.0, handle).is_err() {
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
        let host = self.driver.create_instance(self.global, args.pCreateInfo, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateInstance", args.pInstance, args.handle_pInstance, host.map(|h| h.0));
    }

    fn vkDestroyInstance(&mut self, _args: &mut vn_command_vkDestroyInstance<'_>) {
        // Every device under it dies first: Vulkan's teardown order is not advisory, and a guest
        // that skipped its own destroys does not get to leak them onto the host.
        self.driver.teardown();
    }

    fn vkEnumeratePhysicalDevices(&mut self, args: &mut vn_command_vkEnumeratePhysicalDevices<'_>) {
        if args.pPhysicalDeviceCount.is_null() {
            return;
        }
        // A null array is the guest asking how many there are. Answering it needs no ids, so
        // there is nothing to register and nothing to plant.
        if !args.has_pPhysicalDevices() {
            if let Ok(n) = self.driver.physical_device_count(args.instance) {
                // SAFETY: non-null, and the decoder allocated it in the arena.
                unsafe { *args.pPhysicalDeviceCount = n };
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
        let (instance, count_out) = (args.instance, args.pPhysicalDeviceCount);
        let Some(out) = self.array(args.handle_pPhysicalDevices_mut()) else { return };
        let Ok(got) = self.driver.physical_devices(instance, out) else {
            args.ret = VkResult::VK_ERROR_INITIALIZATION_FAILED;
            self.ghost_ids(ids);
            return;
        };
        // SAFETY: as above.
        unsafe { *count_out = got };
        // What each one supports is asked once, here, because device creation is filtered against
        // it and there is no later point where the guest is guaranteed to have named them all.
        for pd in out.iter().take(got as usize) {
            self.driver.learn_extensions(*pd);
        }
        self.ghost_ids(&ids[got as usize..]);
    }

    fn vkCreateDevice(&mut self, args: &mut vn_command_vkCreateDevice<'_>) {
        let host =
            self.driver.create_device(args.physicalDevice, args.pCreateInfo, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateDevice", args.pDevice, args.handle_pDevice, host.map(|h| h.0));
    }

    fn vkDestroyDevice(&mut self, args: &mut vn_command_vkDestroyDevice<'_>) {
        let orphans = self.driver.destroy_device(args.device);
        self.forget(orphans);
    }

    // ------------------------------------------------------------------- device memory
    //
    // What the memory census reads back, and the first thing the guest does with a device.

    fn vkAllocateMemory(&mut self, args: &mut vn_command_vkAllocateMemory<'_>) {
        // The id has to be read before the allocation, because it is the key the driver files it
        // under -- and it is the guest's, chosen in the request, not anything the host picks.
        let Some(id) = self.out_id(args.pMemory) else {
            return;
        };
        let host =
            self.driver.allocate_memory(args.device, id.0, args.pAllocateInfo, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkAllocateMemory", args.pMemory, args.handle_pMemory, host.map(|m| m.0));
    }

    fn vkFreeMemory(&mut self, args: &mut vn_command_vkFreeMemory<'_>) {
        // A null handle is a legal no-op in Vulkan, and the guest sends it: the id is then zero
        // and the driver's table has nothing under it, so this needs no guard of its own.
        self.driver.free_memory(args.id_memory.0);
    }

    // --------------------------------------------------------------- the simple objects
    //
    // One `vkCreateX`/`vkDestroyX` pair each, with no host state beyond the object table. The
    // whole list is here rather than behind a loop in the generator so that adding one is a
    // visible line in a diff, and so that an object needing more than the pair cannot be added by
    // accident -- `vkCreateShaderModule` below is what that looks like.

    simple_create!(vkCreateFence, vn_command_vkCreateFence, pCreateInfo, pFence, handle_pFence);
    simple_destroy!(vkDestroyFence, vn_command_vkDestroyFence, fence);

    simple_create!(
        vkCreateSemaphore,
        vn_command_vkCreateSemaphore,
        pCreateInfo,
        pSemaphore,
        handle_pSemaphore
    );
    simple_destroy!(vkDestroySemaphore, vn_command_vkDestroySemaphore, semaphore);

    pool_create!(
        vkCreateCommandPool,
        vn_command_vkCreateCommandPool,
        pCreateInfo,
        pCommandPool,
        handle_pCommandPool
    );
    pool_destroy!(vkDestroyCommandPool, vn_command_vkDestroyCommandPool, commandPool);

    simple_create!(vkCreateBuffer, vn_command_vkCreateBuffer, pCreateInfo, pBuffer, handle_pBuffer);
    simple_destroy!(vkDestroyBuffer, vn_command_vkDestroyBuffer, buffer);

    simple_create!(vkCreateImage, vn_command_vkCreateImage, pCreateInfo, pImage, handle_pImage);
    simple_destroy!(vkDestroyImage, vn_command_vkDestroyImage, image);

    simple_create!(
        vkCreateImageView,
        vn_command_vkCreateImageView,
        pCreateInfo,
        pView,
        handle_pView
    );
    simple_destroy!(vkDestroyImageView, vn_command_vkDestroyImageView, imageView);

    simple_create!(
        vkCreateSampler,
        vn_command_vkCreateSampler,
        pCreateInfo,
        pSampler,
        handle_pSampler
    );
    simple_destroy!(vkDestroySampler, vn_command_vkDestroySampler, sampler);

    simple_create!(
        vkCreateRenderPass,
        vn_command_vkCreateRenderPass,
        pCreateInfo,
        pRenderPass,
        handle_pRenderPass
    );
    simple_destroy!(vkDestroyRenderPass, vn_command_vkDestroyRenderPass, renderPass);

    simple_create!(
        vkCreateFramebuffer,
        vn_command_vkCreateFramebuffer,
        pCreateInfo,
        pFramebuffer,
        handle_pFramebuffer
    );
    simple_destroy!(vkDestroyFramebuffer, vn_command_vkDestroyFramebuffer, framebuffer);

    simple_create!(
        vkCreateDescriptorSetLayout,
        vn_command_vkCreateDescriptorSetLayout,
        pCreateInfo,
        pSetLayout,
        handle_pSetLayout
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
        handle_pDescriptorPool
    );
    pool_destroy!(vkDestroyDescriptorPool, vn_command_vkDestroyDescriptorPool, descriptorPool);

    simple_create!(
        vkCreatePipelineLayout,
        vn_command_vkCreatePipelineLayout,
        pCreateInfo,
        pPipelineLayout,
        handle_pPipelineLayout
    );
    simple_destroy!(vkDestroyPipelineLayout, vn_command_vkDestroyPipelineLayout, pipelineLayout);

    simple_create!(
        vkCreatePipelineCache,
        vn_command_vkCreatePipelineCache,
        pCreateInfo,
        pPipelineCache,
        handle_pPipelineCache
    );
    simple_destroy!(vkDestroyPipelineCache, vn_command_vkDestroyPipelineCache, pipelineCache);

    /// The one simple object with a check in front of it.
    ///
    /// `codeSize` is a byte count, uniquely among Vulkan's typed arrays, and the wire carries
    /// `codeSize / 4` words -- so a `codeSize` that is not a multiple of four decodes into an
    /// allocation shorter than the number the driver is then handed, and the driver reads off the
    /// end of it. The guest chooses that number, which makes rejecting it the boundary's job.
    fn vkCreateShaderModule(&mut self, args: &mut vn_command_vkCreateShaderModule<'_>) {
        if args.pCreateInfo.is_none_or(|i| i.codeSize % 4 != 0) {
            self.reject = Some("gave a shader a code size that is not a whole number of words");
            return;
        }
        let host = self.driver.create_object(
            args.device,
            |d| d.vkCreateShaderModule(),
            args.pCreateInfo,
            args.pAllocator,
        );
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateShaderModule", args.pShaderModule, args.handle_pShaderModule, host);
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
        let Some(ids) = self.array(args.pCommandBuffers()) else { return };
        // Read before the shadow is borrowed: see `vkEnumeratePhysicalDevices`.
        let (device, info) = (args.device, args.pAllocateInfo);
        let pool = self.pool_of(info, |i| i.commandPool.raw());
        // The pool records both names of every object it holds, so that destroying it can take
        // the guest's out of the object table. Built before the shadow is borrowed.
        let named: Vec<ObjectId> = ids.iter().map(|h| ObjectId(h.raw())).collect();
        let Some(out) = self.array(args.handle_pCommandBuffers_mut()) else { return };
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
        self.driver.free_objects(
            args.device,
            |d| d.vkFreeCommandBuffers(),
            args.commandPool,
            buffers,
        );
    }

    fn vkAllocateDescriptorSets(&mut self, args: &mut vn_command_vkAllocateDescriptorSets<'_>) {
        let Some(ids) = self.array(args.pDescriptorSets()) else { return };
        // Read before the shadow is borrowed: see `vkEnumeratePhysicalDevices`.
        let (device, info) = (args.device, args.pAllocateInfo);
        let pool = self.pool_of(info, |i| i.descriptorPool.raw());
        // The pool records both names of every object it holds, so that destroying it can take
        // the guest's out of the object table. Built before the shadow is borrowed.
        let named: Vec<ObjectId> = ids.iter().map(|h| ObjectId(h.raw())).collect();
        let Some(out) = self.array(args.handle_pDescriptorSets_mut()) else { return };
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
        let host = self.driver.device_queue(args.device, args.pQueueInfo);
        self.plant(
            "vkGetDeviceQueue2",
            args.pQueue,
            args.handle_pQueue,
            host.map(|q| q.0).ok_or(VkResult::VK_ERROR_INITIALIZATION_FAILED),
        );
    }

    // ------------------------------------------------------------------------ pipelines
    //
    // Not a `simple_create`: one command makes a run of them, and it is the only create that can
    // come back part real. What that costs is in [`Driver::create_pipelines`]; what is left here
    // is the all-or-nothing the guest sees.

    fn vkCreateGraphicsPipelines(&mut self, args: &mut vn_command_vkCreateGraphicsPipelines<'_>) {
        let Some(infos) = self.array(args.pCreateInfos()) else { return };
        let Some(ids) = self.array(args.pPipelines()) else {
            return;
        };
        // Read before the shadow is borrowed: see `vkEnumeratePhysicalDevices`.
        let (device, cache, alloc) = (args.device, args.pipelineCache, args.pAllocator);
        let Some(out) = self.array(args.handle_pPipelines_mut()) else {
            return;
        };
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
        let Some(infos) = self.array(args.pBindInfos()) else { return };
        args.ret = self.driver.bind_memory(args.device, |d| d.vkBindBufferMemory2(), infos);
    }

    fn vkBindImageMemory2(&mut self, args: &mut vn_command_vkBindImageMemory2<'_>) {
        let Some(infos) = self.array(args.pBindInfos()) else { return };
        args.ret = self.driver.bind_memory(args.device, |d| d.vkBindImageMemory2(), infos);
    }

    fn vkUpdateDescriptorSets(&mut self, args: &mut vn_command_vkUpdateDescriptorSets<'_>) {
        let Some(writes) = self.array(args.pDescriptorWrites()) else {
            return;
        };
        let Some(copies) = self.array(args.pDescriptorCopies()) else {
            return;
        };
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
        let Some(ret) = self.driver.begin_command_buffer(args.commandBuffer, args.pBeginInfo)
        else {
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
        let memory = self.array_or_empty(args.pMemoryBarriers());
        let buffers = self.array_or_empty(args.pBufferMemoryBarriers());
        let images = self.array_or_empty(args.pImageMemoryBarriers());
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
        let done = self.driver.cmd_begin_render_pass(
            args.commandBuffer,
            args.pRenderPassBegin,
            args.contents,
        );
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
        let Some(sets) = self.array(args.pDescriptorSets()) else { return };
        let offsets = self.array_or_empty(args.pDynamicOffsets());
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
        let Some(viewports) = self.array(args.pViewports()) else { return };
        let done = self.driver.cmd_set_viewport(args.commandBuffer, args.firstViewport, viewports);
        self.recorded(done);
    }

    fn vkCmdSetScissor(&mut self, args: &mut vn_command_vkCmdSetScissor<'_>) {
        let Some(scissors) = self.array(args.pScissors()) else { return };
        let done = self.driver.cmd_set_scissor(args.commandBuffer, args.firstScissor, scissors);
        self.recorded(done);
    }

    fn vkCmdBindVertexBuffers(&mut self, args: &mut vn_command_vkCmdBindVertexBuffers<'_>) {
        // One count, two arrays. Both accessors read that same count, so the two slices are the
        // same length by construction -- the driver asserts it rather than trusting the pair.
        let Some(buffers) = self.array(args.pBuffers()) else { return };
        let Some(offsets) = self.array(args.pOffsets()) else { return };
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
        let Some(regions) = self.array(args.pRegions()) else { return };
        let done = self.driver.cmd_copy_buffer(
            args.commandBuffer,
            args.srcBuffer,
            args.dstBuffer,
            regions,
        );
        self.recorded(done);
    }

    fn vkCmdCopyBufferToImage(&mut self, args: &mut vn_command_vkCmdCopyBufferToImage<'_>) {
        let Some(regions) = self.array(args.pRegions()) else { return };
        let done = self.driver.cmd_copy_buffer_to_image(
            args.commandBuffer,
            args.srcBuffer,
            args.dstImage,
            args.dstImageLayout,
            regions,
        );
        self.recorded(done);
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
    use super::*;

    fn header(cmd: VkCommandTypeEXT, flags: u32) -> Vec<u8> {
        let mut w = (cmd.0 as u32).to_le_bytes().to_vec();
        w.extend_from_slice(&flags.to_le_bytes());
        w
    }

    /// A shape the generator has no decoder for poisons the ring, and the log line that says so
    /// has to be able to name the command. Without the name a gap reaches a user as a hung guest
    /// with nothing to report; `vn_command_name` returning `None` here would be that silently.
    #[test]
    fn an_undecodable_command_poisons_the_context_by_name() {
        let cmd = VkCommandTypeEXT::VK_COMMAND_TYPE_vkGetPipelineCacheData_EXT;
        assert_eq!(vn_command_name(cmd), Some("vkGetPipelineCacheData"));

        let g = crate::vulkan::global();
        let mut ctx = Context::new(CtxId::new(1).unwrap());
        ctx.replay_begin();
        let mut todo = Unimplemented::default();
        assert!(!ctx.submit(&header(cmd, 0), &mut todo, &g), "a stubbed decoder must poison");
        assert!(ctx.fatal());

        // The poison outlives the batch: a stream we stopped trusting stays untrusted.
        assert!(!ctx.submit(&header(cmd, 0), &mut todo, &g));
    }

    /// A command that wants an answer has nowhere to be answered into, so it poisons -- but only
    /// outside replay, where the journal's replies have already been stripped.
    #[test]
    fn a_reply_request_poisons_only_outside_replay() {
        let cmd = VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyInstance_EXT;
        let mut todo = Unimplemented::default();
        let g = crate::vulkan::global();

        let mut ctx = Context::new(CtxId::new(1).unwrap());
        let w = header(cmd, GENERATE_REPLY);
        let mut full = w.clone();
        full.extend_from_slice(&1u64.to_le_bytes()); // instance id
        full.extend_from_slice(&0u64.to_le_bytes()); // no allocator
        assert!(!ctx.submit(&full, &mut todo, &g));
        assert_eq!(ctx.unhandled, 1);

        // In replay the flag is stripped, so the command reaches the dispatcher instead of the
        // poison. It still names an instance nothing created, which poisons for its own reason --
        // what separates the two paths is whether the command was dispatched at all.
        let mut ctx = Context::new(CtxId::new(1).unwrap());
        ctx.replay_begin();
        assert!(!ctx.submit(&full, &mut todo, &g));
        assert_eq!(ctx.dispatched, 1);
        assert_eq!(ctx.unhandled, 0);
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
                assert!(!args.handle_pFence.is_null(), "the decoder owes a place to write");
                // The guest id in `pFence` has to survive: the reply sends it back.
                // SAFETY: the decoder allocated one element there.
                unsafe { *args.handle_pFence = VkFence(HOST) };
            }

            fn object_created(&mut self, ty: VkObjectType, id: ObjectId, host: u64) {
                self.objects.borrow_mut().add(id, ty.0, host).expect("a fresh id");
            }

            fn object_destroyed(&mut self, _ty: VkObjectType, id: ObjectId) {
                self.objects.borrow_mut().remove(id).expect("a live id");
            }
        }

        fn run(h: &mut Driver<'_>, objects: &Shared, wire: &[u8]) {
            let temp = Bump::new();
            let hard = Cell::new(false);
            let mut dec = Decoder::new(wire, &temp, objects, &hard);
            let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
            let _flags = dec.decode_scalar::<VkFlags>();
            assert_eq!(vn_dispatch_command(&mut dec, None, cmd, h), Some(()));
            assert!(!dec.fatal(), "the command must decode");
            assert_eq!(dec.pos(), wire.len(), "the command must be fully consumed");
        }

        let objects = Shared::new();
        objects
            .borrow_mut()
            .add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE.0, 1)
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
            Lookup::Found(HOST)
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
                VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE.0,
                PHYSICAL_DEVICE,
            )
            .unwrap();

        // A driver with no instance refuses every device without reaching Vulkan, which is the
        // refusal this test wants: the interesting half is what happens after the `Err`.
        let mut driver = Driver::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let hard = Cell::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Some(()));
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
            .add(ObjectId(INSTANCE), VkObjectType::VK_OBJECT_TYPE_INSTANCE.0, INSTANCE)
            .unwrap();

        let mut driver = Driver::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let hard = Cell::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Some(()));
        assert!(!dec.fatal(), "a refusal is the driver's answer, not a protocol error");

        for id in IDS {
            assert_eq!(
                objects.lookup(ObjectId(id), VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE.0),
                Lookup::Ghost,
                "id {id} was offered to a failed enumeration and must not become an object"
            );
        }
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
            .add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE.0, DEVICE)
            .unwrap();

        // The object table has the device; the driver does not. That is exactly the split the
        // re-check exists for -- a guest that destroyed a device and then created against it.
        let mut driver = Driver::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let hard = Cell::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Some(()));
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
        let mut driver = Driver::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
        };

        let odd = VkShaderModuleCreateInfo { codeSize: 7, ..Default::default() };
        let mut args =
            vn_command_vkCreateShaderModule { pCreateInfo: Some(&odd), ..Default::default() };
        h.vkCreateShaderModule(&mut args);
        assert!(h.reject.is_some(), "a code size of 7 must not reach the driver");

        // Four is a whole word, so the guard lets it through; there is no device, so the driver
        // refuses it -- which is a different answer from a protocol violation.
        let whole = VkShaderModuleCreateInfo { codeSize: 4, ..Default::default() };
        let mut args =
            vn_command_vkCreateShaderModule { pCreateInfo: Some(&whole), ..Default::default() };
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
        let got = args.pBindInfos().expect("three counted, three sent");
        assert_eq!(
            got.iter().map(|i| i.memoryOffset.0).collect::<Vec<_>>(),
            [100, 101, 102],
            "the slice must be the array, not a prefix of it and not a step past its end"
        );

        // A count of none is an empty slice, not an absent one: the guest asked for no binds,
        // which is a legal thing to ask for and a different answer from a broken pair.
        let mut args = vn_command_vkBindBufferMemory2::default();
        args.plant_pBindInfos(&infos[..0]);
        assert_eq!(args.pBindInfos().map(<[_]>::len), Some(0));
    }

    #[test]
    fn an_array_the_guest_counted_but_did_not_send_is_refused() {
        use super::super::proto::types::{VkBindBufferMemoryInfo, vn_command_vkBindBufferMemory2};

        let objects = Shared::new();
        let mut driver = Driver::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
        };

        // Counted without planting: the pointer stays null, which is the one shape the
        // planter cannot build and the only one this handler is here to refuse.
        let mut args = vn_command_vkBindBufferMemory2::default();
        args.bindInfoCount = 3;
        h.vkBindBufferMemory2(&mut args);
        assert!(h.reject.is_some(), "three binds with no array behind them must not pass");

        // No array and nothing counted is the ordinary optional array, and binding nothing is a
        // legal no-op -- a different answer from a protocol violation.
        h.reject = None;
        let mut args = vn_command_vkBindBufferMemory2::default();
        h.vkBindBufferMemory2(&mut args);
        assert!(h.reject.is_none(), "binding nothing is not a protocol violation");

        // An array that is there passes the guard; there is no device, so the driver refuses it,
        // which is again not a protocol violation.
        h.reject = None;
        let info = VkBindBufferMemoryInfo::default();
        let mut args = vn_command_vkBindBufferMemory2::default();
        args.plant_pBindInfos(core::slice::from_ref(&info));
        h.vkBindBufferMemory2(&mut args);
        assert!(h.reject.is_none(), "an array the guest actually sent is not a violation");
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
        let mut driver = Driver::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut driver = Driver::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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

            fn object_created(&mut self, _ty: VkObjectType, _id: ObjectId, _host: u64) {}
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
            let hard = Cell::new(false);
            let mut dec = Decoder::new(w, &temp, objects, &hard);
            let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
            let _flags = dec.decode_scalar::<VkFlags>();
            let mut h = Recorder::default();
            assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Some(()));
            (h.saw, dec.fatal())
        }

        let objects = Shared::new();
        {
            let mut t = objects.borrow_mut();
            t.add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE.0, 1).unwrap();
            t.add(ObjectId(POOL), VkObjectType::VK_OBJECT_TYPE_COMMAND_POOL.0, 2).unwrap();
            for (i, id) in BUFFERS.iter().enumerate() {
                t.add(ObjectId(*id), VkObjectType::VK_OBJECT_TYPE_COMMAND_BUFFER.0, 100 + i as u64)
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
            VkCommandPool, VkDevice, vn_command_vkDestroyCommandPool,
        };

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        /// The guest ids of two command buffers, and the host handles they were allocated as.
        const BUFFERS: [(u64, u64); 2] = [(11, 110), (12, 120)];
        const COMMAND_BUFFER: i32 = VkObjectType::VK_OBJECT_TYPE_COMMAND_BUFFER.0;

        let objects = Shared::new();
        let mut driver = Driver::new();
        {
            let mut t = objects.borrow_mut();
            for (host, id) in BUFFERS {
                t.add(ObjectId(id), COMMAND_BUFFER, host).unwrap();
            }
        }
        driver.plant_pool(DEVICE, POOL, &BUFFERS.map(|(host, id)| (host, ObjectId(id))));
        assert_eq!(objects.lookup(ObjectId(BUFFERS[0].1), COMMAND_BUFFER), Lookup::Found(11));

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
                objects.lookup(ObjectId(id), COMMAND_BUFFER),
                Lookup::Missing,
                "id {id} was freed with its pool and must stop naming anything"
            );
        }
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
    /// Three shapes, which is what the fifteen recording commands are made of: one counted array,
    /// two arrays under one count, and a command with a result to carry back.
    #[test]
    fn a_recording_handler_hands_the_driver_what_the_guest_sent() {
        use super::super::proto::types::{
            VkBuffer, VkCommandBuffer, VkCommandBufferBeginInfo, VkDeviceSize, VkViewport,
            vn_command_vkBeginCommandBuffer, vn_command_vkCmdBindVertexBuffers,
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
        fns.plant_vkBeginCommandBuffer(begin);

        let objects = Shared::new();
        let mut driver = Driver::new();
        driver.plant_device(DEVICE, fns);
        driver.plant_pool(DEVICE, POOL, &[(CB.0, ObjectId(CB.1))]);

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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

        // A result the guest is owed: the driver's answer has to reach the reply, not be
        // replaced by a success the renderer invented.
        let mut args = vn_command_vkBeginCommandBuffer { commandBuffer: cb, ..Default::default() };
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
    }
}
