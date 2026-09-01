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
use super::driver::{Driver, MemoryError, NoSyncFd};
use super::objects::Shared;
use super::proto::serialize::{Commands, vn_command_name, vn_dispatch_command};
use super::proto::types::{
    VkCommandTypeEXT, VkDevice, VkDeviceMemory, VkFlags, VkObjectType, VkPhysicalDevice, VkResult,
    vn_command_vkAllocateCommandBuffers, vn_command_vkAllocateDescriptorSets,
    vn_command_vkAllocateMemory, vn_command_vkBeginCommandBuffer, vn_command_vkBindBufferMemory2,
    vn_command_vkBindImageMemory2, vn_command_vkCmdBeginRenderPass,
    vn_command_vkCmdBindDescriptorSets, vn_command_vkCmdBindPipeline,
    vn_command_vkCmdBindVertexBuffers, vn_command_vkCmdCopyBuffer,
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
    vn_command_vkEndCommandBuffer, vn_command_vkEnumerateDeviceExtensionProperties,
    vn_command_vkEnumerateInstanceVersion, vn_command_vkEnumeratePhysicalDeviceGroups,
    vn_command_vkEnumeratePhysicalDevices, vn_command_vkFreeCommandBuffers,
    vn_command_vkFreeMemory, vn_command_vkGetBufferMemoryRequirements2,
    vn_command_vkGetDeviceQueue2, vn_command_vkGetImageDrmFormatModifierPropertiesEXT,
    vn_command_vkGetImageMemoryRequirements2, vn_command_vkGetImageSubresourceLayout,
    vn_command_vkGetPhysicalDeviceExternalFenceProperties,
    vn_command_vkGetPhysicalDeviceExternalSemaphoreProperties,
    vn_command_vkGetPhysicalDeviceFeatures2, vn_command_vkGetPhysicalDeviceFormatProperties2,
    vn_command_vkGetPhysicalDeviceImageFormatProperties2,
    vn_command_vkGetPhysicalDeviceMemoryProperties2, vn_command_vkGetPhysicalDeviceProperties,
    vn_command_vkGetPhysicalDeviceProperties2,
    vn_command_vkGetPhysicalDeviceQueueFamilyProperties2, vn_command_vkImportSemaphoreResourceMESA,
    vn_command_vkQueueSubmit, vn_command_vkResetCommandBuffer, vn_command_vkResetFences,
    vn_command_vkUpdateDescriptorSets, vn_command_vkWaitForFences,
    vn_command_vkWaitSemaphoreResourceMESA,
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
            .map(|o| VkDeviceMemory(o.handle))
            .ok_or(MemoryError::NoSuchAllocation)?;
        let device = objects.device_of(id).ok_or(MemoryError::NoSuchAllocation)?;
        self.driver.memory_read(VkDevice(device), handle, id, buf)
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
        let doomed = self.objects.borrow_mut().take_all();
        self.driver.teardown(&doomed);
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
    fn out_id<T: Handle>(&self, out: Option<&T>) -> Option<ObjectId> {
        Some(ObjectId(out?.raw()))
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
        out: Option<&T>,
        shadow: Option<&mut T>,
        host: Result<u64, VkResult>,
    ) {
        let Some(id) = self.out_id(out) else {
            return;
        };
        match host {
            Ok(h) if h != 0 => {
                if let Some(shadow) = shadow {
                    *shadow = T::from_raw(h);
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
            let host =
                self.driver.create_pool(args.device, |d| d.$cmd(), args.$info, args.pAllocator);
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
    fn unsupported(&mut self, cmd: VkCommandTypeEXT) {
        *self.todo.seen.entry(cmd.0).or_default() += 1;
    }

    fn object_created(
        &mut self,
        ty: VkObjectType,
        id: ObjectId,
        host: u64,
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
        let handle = if host == 0 { id.0 } else { host };
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
        let host = self.driver.create_instance(self.global, args.pCreateInfo, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant(
            "vkCreateInstance",
            args.pInstance(),
            args.handle_pInstance_mut(),
            host.map(|h| h.0),
        );
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
        for pd in out.iter().take(got as usize) {
            self.driver.learn_extensions(*pd);
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
        let host =
            self.driver.create_device(args.physicalDevice, args.pCreateInfo, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateDevice", args.pDevice(), args.handle_pDevice_mut(), host.map(|h| h.0));
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
        let host =
            self.driver.allocate_memory(args.device, id, args.pAllocateInfo, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant(
            "vkAllocateMemory",
            args.pMemory(),
            args.handle_pMemory_mut(),
            host.map(|m| m.0),
        );
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

    simple_create!(vkCreateImage, vn_command_vkCreateImage, pCreateInfo, pImage, handle_pImage_mut);
    simple_destroy!(vkDestroyImage, vn_command_vkDestroyImage, image);

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
            args.pQueue(),
            args.handle_pQueue_mut(),
            host.map(|q| q.0).ok_or(VkResult::VK_ERROR_INITIALIZATION_FAILED),
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
        if !args.has_pPhysicalDeviceGroupCount() {
            self.reject = Some("enumerated without asking for a count");
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
                match table.id_of_handle(VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE, pd.0) {
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
        if !args.has_pPropertyCount() {
            self.reject = Some("enumerated without asking for a count");
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
        if !args.has_pQueueFamilyPropertyCount() {
            self.reject = Some("enumerated without asking for a count");
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

    // The queries that carry a `ret`. Where a command has a field designed to say "no", that is
    // the honest channel and refusal is not: a driver without `VK_EXT_image_drm_format_modifier`
    // is an answer the guest asked for and can act on, not a reason to take its ring down. The
    // void queries above have no such field, which is why refusal is all they have.

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
        let (pd, info) = (args.physicalDevice, args.pImageFormatInfo);
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
        let (pd, info) = (args.physicalDevice, args.pExternalFenceInfo);
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
        let (pd, info) = (args.physicalDevice, args.pExternalSemaphoreInfo);
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
        let (device, info) = (args.device, args.pInfo);
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
        let (device, info) = (args.device, args.pInfo);
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
        let (device, image, sub) = (args.device, args.image, args.pSubresource);
        let Some(out) = self.fills(args.pLayout_mut()) else { return };
        let r = self
            .driver
            .dev_query_arg_info(device, image, sub, out, |d| d.try_vkGetImageSubresourceLayout());
        self.asked(r);
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

    // --------------------------------------------------------------------------- sync
    //
    // Where a frame is handed to the GPU and waited for. Everything the recording section built
    // is inert until a submit names it, and everything after a submit is the guest asking whether
    // the work is done yet.

    fn vkQueueSubmit(&mut self, args: &mut vn_command_vkQueueSubmit<'_>) {
        // Submitting nothing is legal -- it is how a guest signals a fence with no work -- so the
        // empty slice goes through rather than being turned away.
        let submits = self.array_or_empty(args.pSubmits());
        let Some(ret) = self.driver.queue_submit(args.queue, submits, args.fence) else {
            self.reject = Some("submitted to a queue with no device behind it");
            return;
        };
        args.ret = ret;
    }

    fn vkResetFences(&mut self, args: &mut vn_command_vkResetFences<'_>) {
        let Some(fences) = self.array(args.pFences()) else { return };
        args.ret = self.driver.reset_fences(args.device, fences);
    }

    /// Blocks the caller for as long as the guest asked, up to forever. That is the guest's own
    /// thread being spent on the guest's own wait; answering early would be answering wrongly.
    fn vkWaitForFences(&mut self, args: &mut vn_command_vkWaitForFences<'_>) {
        let Some(fences) = self.array(args.pFences()) else { return };
        args.ret = self.driver.wait_for_fences(args.device, fences, args.waitAll, args.timeout);
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
        let mut driver = Driver::new();
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
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut driver = Driver::new();
        driver.plant_instance(fns);

        let mut asked = VkPhysicalDeviceProperties::default();
        let mut args = vn_command_vkGetPhysicalDeviceProperties::default();
        args.plant_pProperties(&mut asked);

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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

        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut driver = Driver::new();

        let mut args = Cmd::default();
        let mut out = 0u32;
        args.plant_pApiVersion(&mut out);

        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
                .add(ObjectId(*id), VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE, host, None)
                .unwrap();
        }

        let mut fns = crate::vulkan::Instance::default();
        fns.plant_vkEnumeratePhysicalDeviceGroups(groups);
        let mut driver = Driver::new();
        driver.plant_instance(fns);
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();

        let mut props = [VkPhysicalDeviceGroupProperties::default(); 1];
        let mut n = 1u32;
        let mut args = Cmd::default();
        args.instance = INSTANCE;
        args.plant_pPhysicalDeviceGroupCount(&mut n);
        args.plant_pPhysicalDeviceGroupProperties(&mut props);

        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut driver = Driver::new();
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

    /// A layer is host-side software a venus guest cannot see and this renderer does not load, so
    /// naming one is not a request that can be honoured or a mistake worth smoothing over.
    #[test]
    fn naming_a_layer_is_refused_rather_than_ignored() {
        use super::super::proto::types::{
            VkPhysicalDevice, vn_command_vkEnumerateDeviceExtensionProperties as Cmd,
        };

        let mut driver = Driver::new();
        driver.plant_extensions(VkPhysicalDevice(1), &["VK_KHR_external_memory_fd"]);
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();

        let mut n = 0u32;
        let mut args = Cmd::default();
        args.physicalDevice = VkPhysicalDevice(1);
        args.plant_pPropertyCount(&mut n);
        args.plant_pLayerName(c"VK_LAYER_KHRONOS_validation");

        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut driver = Driver::new();
        driver.plant_instance(fns);

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();

        // The count call: a count member, no array behind it.
        let mut n = 0u32;
        let mut args = Cmd::default();
        args.physicalDevice = PD;
        args.plant_pQueueFamilyPropertyCount(&mut n);
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut driver = Driver::new();
        driver.plant_device(DEVICE.0, crate::vulkan::Device::default());

        let mut props = VkImageDrmFormatModifierPropertiesEXT::default();
        let mut args = vn_command_vkGetImageDrmFormatModifierPropertiesEXT::default();
        args.device = DEVICE;
        args.image = VkImage(1);
        args.plant_pProperties(&mut props);

        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut driver = Driver::new();
        driver.plant_instance(crate::vulkan::Instance::default());
        let mut args = vn_command_vkGetPhysicalDeviceFeatures2::default();
        let objects = Shared::new();
        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
                host: u64,
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
            .add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, 1, None)
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
                VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE,
                PHYSICAL_DEVICE,
                None,
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
            .add(ObjectId(INSTANCE), VkObjectType::VK_OBJECT_TYPE_INSTANCE, INSTANCE, None)
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
            .add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, DEVICE, None)
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

            fn object_created(
                &mut self,
                _ty: VkObjectType,
                _id: ObjectId,
                _host: u64,
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
            t.add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, 1, None).unwrap();
            t.add(
                ObjectId(POOL),
                VkObjectType::VK_OBJECT_TYPE_COMMAND_POOL,
                2,
                Some(ObjectId(DEVICE)),
            )
            .unwrap();
            for (i, id) in BUFFERS.iter().enumerate() {
                t.add(
                    ObjectId(*id),
                    VkObjectType::VK_OBJECT_TYPE_COMMAND_BUFFER,
                    100 + i as u64,
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
            VkCommandPool, VkDevice, vn_command_vkDestroyCommandPool,
        };

        const DEVICE: u64 = 3;
        const POOL: u64 = 7;
        /// The guest ids of two command buffers, and the host handles they were allocated as.
        const BUFFERS: [(u64, u64); 2] = [(11, 110), (12, 120)];
        const COMMAND_BUFFER: VkObjectType = VkObjectType::VK_OBJECT_TYPE_COMMAND_BUFFER;

        let objects = Shared::new();
        let mut driver = Driver::new();
        {
            let mut t = objects.borrow_mut();
            for (host, id) in BUFFERS {
                t.add(ObjectId(id), COMMAND_BUFFER, host, None).unwrap();
            }
        }
        driver.plant_pool(DEVICE, POOL, &BUFFERS.map(|(host, id)| (host, ObjectId(id))));
        assert_eq!(objects.lookup(ObjectId(BUFFERS[0].1), COMMAND_BUFFER.0), Lookup::Found(11));

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
        let mut driver = Driver::new();
        driver.plant_device(DEVICE, fns);
        {
            let mut t = objects.borrow_mut();
            t.add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, DEVICE, None).unwrap();
            let under = Some(ObjectId(DEVICE));
            t.add(ObjectId(FENCE.0), VkObjectType::VK_OBJECT_TYPE_FENCE, FENCE.1, under).unwrap();
            t.add(ObjectId(IMAGE.0), VkObjectType::VK_OBJECT_TYPE_IMAGE, IMAGE.1, under).unwrap();
            t.add(ObjectId(POOL.0), VkObjectType::VK_OBJECT_TYPE_COMMAND_POOL, POOL.1, under)
                .unwrap();
        }

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
        };
        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyDevice_EXT, 0);
        w.extend_from_slice(&DEVICE.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent

        let temp = Bump::new();
        let hard = Cell::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Some(()));

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

        let mut ctx = Context::new(CtxId::new(7).expect("7 is not zero"));
        ctx.driver_mut().plant_device(DEVICE, fns);
        {
            let mut t = ctx.objects().borrow_mut();
            t.add(ObjectId(DEVICE), VkObjectType::VK_OBJECT_TYPE_DEVICE, DEVICE, None).unwrap();
            t.add(
                ObjectId(FENCE.0),
                VkObjectType::VK_OBJECT_TYPE_FENCE,
                FENCE.1,
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
        let mut driver = Driver::new();
        {
            let mut t = objects.borrow_mut();
            t.add(ObjectId(INSTANCE), VkObjectType::VK_OBJECT_TYPE_INSTANCE, INSTANCE, None)
                .unwrap();
            t.add(
                ObjectId(PHYSICAL_DEVICE),
                VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE,
                PHYSICAL_DEVICE,
                Some(ObjectId(INSTANCE)),
            )
            .unwrap();
            t.add(
                ObjectId(DEVICE),
                VkObjectType::VK_OBJECT_TYPE_DEVICE,
                DEVICE,
                Some(ObjectId(PHYSICAL_DEVICE)),
            )
            .unwrap();
            t.add(ObjectId(FENCE), FENCE_TY, 0xfeed, Some(ObjectId(DEVICE))).unwrap();
        }

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
        };
        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyInstance_EXT, 0);
        w.extend_from_slice(&INSTANCE.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent

        let temp = Bump::new();
        let hard = Cell::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Some(()));
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
        let mut driver = Driver::new();
        {
            let mut t = objects.borrow_mut();
            t.add(ObjectId(DEVICE_A), VkObjectType::VK_OBJECT_TYPE_DEVICE, DEVICE_A, None).unwrap();
            t.add(ObjectId(FENCE_ID), FENCE, FENCE_HOST, Some(ObjectId(DEVICE_A))).unwrap();
        }
        assert_eq!(objects.lookup(ObjectId(FENCE_ID), FENCE.0), Lookup::Found(FENCE_HOST));

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
        };
        // Driven over the wire rather than by calling the handler, because the handler is only
        // half of a destroy: the generated lifecycle hook is what tells the table an object died,
        // and it runs after the handler and nowhere else. No device is planted in the driver, so
        // no entry point is called -- what is under test is the table.
        let mut w = header(VkCommandTypeEXT::VK_COMMAND_TYPE_vkDestroyDevice_EXT, 0);
        w.extend_from_slice(&DEVICE_A.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes()); // pAllocator: absent

        let temp = Bump::new();
        let hard = Cell::new(false);
        let mut dec = Decoder::new(&w, &temp, &objects, &hard);
        let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
        let _flags = dec.decode_scalar::<VkFlags>();
        assert_eq!(vn_dispatch_command(&mut dec, None, cmd, &mut h), Some(()));
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
            VkDependencyFlags, VkDeviceSize, VkImageMemoryBarrier, VkMemoryBarrier,
            VkPipelineStageFlags, VkViewport, vn_command_vkBeginCommandBuffer,
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

        // Nothing here came from Vulkan, so there is nothing to destroy. See `abandon_planted`.
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
            /// The descriptor the export handed over, so the test can ask whether it was closed.
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
            // A real descriptor, so the close the handler owes it is a thing the test can see.
            let fd = unsafe { libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY) };
            assert!(fd >= 0, "the test needs a descriptor to watch");
            // SAFETY: the caller is `export_semaphore_sync_fd`, which passes a live `c_int`.
            unsafe { *out = fd };
            SAW.with_borrow_mut(|s| s.exported.push(fd));
            VkResult::VK_SUCCESS
        }

        let mut fns = crate::vulkan::Device::default();
        fns.plant_vkQueueSubmit(submit);
        fns.plant_vkGetSemaphoreFdKHR(export);
        fns.plant_vkResetFences(reset);
        fns.plant_vkWaitForFences(wait);
        fns.plant_vkImportSemaphoreFdKHR(import);

        let objects = Shared::new();
        let mut driver = Driver::new();
        driver.plant_device(DEVICE, fns);
        driver.plant_queue(DEVICE, QUEUE);

        let mut todo = Unimplemented::default();
        let global = crate::vulkan::global();
        let mut h = Handlers {
            objects: &objects,
            todo: &mut todo,
            driver: &mut driver,
            global: &global,
            reject: None,
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
        let fd = SAW.with_borrow(|s| {
            assert_eq!(s.exported.len(), 1);
            s.exported[0]
        });
        // SAFETY: a plain query on a descriptor number; it is closed, which is what is asserted.
        assert_eq!(
            unsafe { libc::fcntl(fd, libc::F_GETFD) },
            -1,
            "the exported descriptor must not outlive the command that made it"
        );

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
        h.driver.plant_device(BARE, crate::vulkan::Device::default());
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
}
