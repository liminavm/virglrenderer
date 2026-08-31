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
    VkCommandTypeEXT, VkFlags, VkObjectType, VkResult, vn_command_vkCreateDevice,
    vn_command_vkCreateInstance, vn_command_vkDestroyDevice, vn_command_vkDestroyInstance,
    vn_command_vkEnumeratePhysicalDevices, vn_command_vkGetDeviceQueue2,
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
            bad_id: false,
        };

        while dec.has_command() {
            dec.clear_soft_fatal();

            let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
            let flags = dec.decode_scalar::<VkFlags>();
            if dec.hard_fatal() {
                // The header itself was short: there is no command here to lose.
                eprintln!(
                    "[virglrs] ctx {}: submission ends mid-header, {} bytes into {}",
                    id.0,
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
            // A guest that named id zero, or reused one that is still live, is naming objects
            // it cannot have. The handler had no decoder to say so; this is where it lands.
            if h.bad_id {
                poison(fatal, id, &dec, cmd, "named an object it cannot have");
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
        eprintln!("[virglrs] ctx {}: {name} {why}, {} bytes in", id.0, dec.pos());
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
    /// A guest that reused a live id or named id zero. It cannot be reported from here -- the
    /// handler has no decoder -- so the loop reads it back and poisons.
    bad_id: bool,
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

    /// Refuse a run of ids in an out-array the guest sent.
    ///
    /// The generated hook walks the whole array whatever the handler did with it, so every id it
    /// will reach needs a decision recorded against it. The ones a short answer did not fill, and
    /// all of them when the enumeration failed outright, are refusals: left alone they would be
    /// registered as their own handles and the guest would hold objects that do not exist.
    fn ghost_range<T: Handle>(&mut self, out: *const T, range: core::ops::Range<usize>) {
        for i in range {
            // SAFETY: `i` is inside the array, which the decoder allocated with `asked` elements.
            if let Some(id) = self.out_id(unsafe { out.add(i) }) {
                self.objects.borrow_mut().add_ghost(id);
            }
        }
    }
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
            self.bad_id = true;
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

    fn vkCreateInstance(&mut self, args: &mut vn_command_vkCreateInstance) {
        let host = self.driver.create_instance(self.global, args.pCreateInfo, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateInstance", args.pInstance, args.handle_pInstance, host.map(|h| h.0));
    }

    fn vkDestroyInstance(&mut self, _args: &mut vn_command_vkDestroyInstance) {
        // Every device under it dies first: Vulkan's teardown order is not advisory, and a guest
        // that skipped its own destroys does not get to leak them onto the host.
        self.driver.teardown();
    }

    fn vkEnumeratePhysicalDevices(&mut self, args: &mut vn_command_vkEnumeratePhysicalDevices) {
        if args.pPhysicalDeviceCount.is_null() {
            return;
        }
        // A null array is the guest asking how many there are. Answering it needs no ids, so
        // there is nothing to register and nothing to plant.
        if args.pPhysicalDevices.is_null() {
            if let Ok(n) = self.driver.physical_device_count(args.instance) {
                // SAFETY: non-null, and the decoder allocated it in the arena.
                unsafe { *args.pPhysicalDeviceCount = n };
            }
            return;
        }

        // SAFETY: both are non-null and the decoder allocated the arrays with this many elements.
        let asked = unsafe { *args.pPhysicalDeviceCount } as usize;
        let out = unsafe { core::slice::from_raw_parts_mut(args.handle_pPhysicalDevices, asked) };
        let Ok(got) = self.driver.physical_devices(args.instance, out) else {
            args.ret = VkResult::VK_ERROR_INITIALIZATION_FAILED;
            self.ghost_range(args.pPhysicalDevices, 0..asked);
            return;
        };
        // SAFETY: as above.
        unsafe { *args.pPhysicalDeviceCount = got };
        // What each one supports is asked once, here, because device creation is filtered against
        // it and there is no later point where the guest is guaranteed to have named them all.
        for pd in out.iter().take(got as usize) {
            self.driver.learn_extensions(*pd);
        }
        self.ghost_range(args.pPhysicalDevices, got as usize..asked);
    }

    fn vkCreateDevice(&mut self, args: &mut vn_command_vkCreateDevice) {
        let host =
            self.driver.create_device(args.physicalDevice, args.pCreateInfo, args.pAllocator);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        self.plant("vkCreateDevice", args.pDevice, args.handle_pDevice, host.map(|h| h.0));
    }

    fn vkDestroyDevice(&mut self, args: &mut vn_command_vkDestroyDevice) {
        self.driver.destroy_device(args.device);
    }

    fn vkGetDeviceQueue2(&mut self, args: &mut vn_command_vkGetDeviceQueue2) {
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
        let mut ctx = Context::new(CtxId(1));
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

        let mut ctx = Context::new(CtxId(1));
        let w = header(cmd, GENERATE_REPLY);
        let mut full = w.clone();
        full.extend_from_slice(&1u64.to_le_bytes()); // instance id
        full.extend_from_slice(&0u64.to_le_bytes()); // no allocator
        assert!(!ctx.submit(&full, &mut todo, &g));
        assert_eq!(ctx.unhandled, 1);

        // In replay the flag is stripped, so the command reaches the dispatcher instead of the
        // poison. It still names an instance nothing created, which poisons for its own reason --
        // what separates the two paths is whether the command was dispatched at all.
        let mut ctx = Context::new(CtxId(1));
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

            fn vkCreateFence(&mut self, args: &mut vn_command_vkCreateFence) {
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
            bad_id: false,
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
            bad_id: false,
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
}
