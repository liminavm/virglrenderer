// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Replay a recorded venus corpus into a virglrenderer build, with no VM, no guest and no
//! hypervisor -- harness layer 2 for venus (`harness/README.md`).
//!
//! The corpus is a prologue plus a stream (`src/venus/vkr_record.h`). This drives it through the
//! public C ABI only, so the same binary runs against the C renderer and against virglrs and the
//! two outputs are comparable. That constraint is the whole point of the harness and nothing here
//! may reach around it.
//!
//!   vkr-replay <corpus.vkrc> --renderer <libvirglrenderer.dylib> [--flags N] [--verbose]
//!
//! Exit status is 0 when every record was accepted, 1 when any was rejected or the corpus itself
//! is malformed. A truncated corpus is not a failure: truncation is a prefix, and a prefix replays.

mod abi;
mod corpus;

use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;

use corpus::{Ctl, Record};

const BLOB_MEM_GUEST: u32 = 0x1;
const BLOB_MEM_HOST3D_GUEST: u32 = 0x3;

/// `VIRTGPU_DRM_CAPSET_VENUS`; a context's capset is the low byte of its context_init.
const CAPSET_VENUS: u32 = 4;
const CAPSET_MASK: u32 = 0xff;

/// Ring flow-control commands: a conversation with a guest, not work for the renderer.
///
/// `replay_ring_cmd` hands each command straight to the ring's dispatcher, bypassing the ring
/// buffer entirely -- that is what makes a VM-free replay possible at all. The consequence is that
/// the buffer's head, tail and seqno never move, so every command whose meaning is "the buffer
/// advanced" or "wait until it advances" is either meaningless or a deadlock. `vkWaitRingSeqnoMESA`
/// is the deadlock: it waits for a seqno the guest would have written and blocks forever.
///
/// The renderer's own snapshot journal reaches the same conclusion by a different route -- it
/// classifies these TRANSIENT and never records them. This recorder deliberately records
/// everything, so the judgement lives here instead.
const RING_FLOW_CONTROL: &[u32] = &[
    190, // vkNotifyRingMESA        -- "the buffer advanced"
    251, // vkSubmitVirtqueueSeqnoMESA
    252, // vkWaitVirtqueueSeqnoMESA
    253, // vkWaitRingSeqnoMESA     -- blocks on a seqno no one will write
];

struct Args {
    corpus: String,
    renderer: String,
    flags: i32,
    verbose: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut corpus = None;
    let mut renderer = None;
    let mut flags = abi::DEFAULT_FLAGS;
    let mut verbose = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--renderer" => renderer = it.next(),
            "--flags" => {
                let v = it.next().ok_or("--flags wants a value")?;
                flags = i32::from_str_radix(v.trim_start_matches("0x"), 16)
                    .or_else(|_| v.parse())
                    .map_err(|_| format!("--flags {v}: not a number"))?;
            }
            "--verbose" => verbose = true,
            _ if corpus.is_none() => corpus = Some(a),
            _ => return Err(format!("unexpected argument {a}")),
        }
    }

    Ok(Args {
        corpus: corpus.ok_or("no corpus given")?,
        renderer: renderer
            .or_else(|| std::env::var("VIRGL_RENDERER_LIB").ok())
            .ok_or("no --renderer and no VIRGL_RENDERER_LIB")?,
        flags,
        verbose,
    })
}

/// What the replay did, per context. Pass one's oracle: a renderer that accepts every command and
/// builds nothing looks identical to a working one in a pass/fail count alone, so the memory
/// census is reported beside the counts rather than instead of them.
#[derive(Default)]
struct Tally {
    prologue_ok: u64,
    prologue_fail: u64,
    cmd_ok: u64,
    cmd_fail: u64,
    ctl_ok: u64,
    ctl_fail: u64,
    /// Recorded so a replay says which resources it could not reconstruct, instead of failing
    /// obscurely on the first command that names one.
    unreplayable_imports: Vec<u32>,
    /// Ring flow-control commands skipped; see RING_FLOW_CONTROL.
    skipped_flow_control: u64,
    /// Device-memory blob creates that had to be parked and retried; see `deferred`.
    parked: u64,
    /// The largest number of stream records a parked event had to wait through.
    park_depth: u64,
    /// Classic (VIRGL2) contexts in the corpus. A venus corpus captured on a real desktop carries
    /// them -- the synoik image runs X clients through classic virgl alongside its Vulkan
    /// compositor -- and this replayer initializes venus only.
    skipped_classic: Vec<u32>,
}

impl Tally {
    fn failed(&self) -> u64 {
        self.prologue_fail + self.cmd_fail + self.ctl_fail
    }
}

/// Backing store synthesized for a guest-storage blob. The guest RAM it pointed into is gone, so
/// only the size was recorded; the bytes are ours. Kept alive for the resource's lifetime because
/// the renderer holds the iovec.
struct Backing {
    _bytes: Vec<u8>,
}

struct Replay<'a> {
    r: &'a abi::Renderer,
    verbose: bool,
    tally: Tally,
    /// Contexts on which replay_begin has been called and replay_end has not.
    open: BTreeSet<u32>,
    backings: BTreeMap<u32, Backing>,
    /// Classic contexts, and every resource event naming one, are skipped rather than failed.
    classic: BTreeSet<u32>,
    /// DIAGNOSTIC, not the shipped semantics. A create_blob that exports a VkDeviceMemory
    /// (blob_id != 0) can be recorded ahead of the vkAllocateMemory that made it: the recorder
    /// orders events by when each thread reached its lock, and that is not the order they
    /// executed in. Parking such a create and retrying it as the stream advances tells us whether
    /// ordering is the LAST thing wrong with a corpus -- and how far off the order actually is.
    /// The real fix is a recorded dependency fence, not a retry loop.
    deferred: Vec<(Ctl, u64)>,
}

impl<'a> Replay<'a> {
    fn note(&mut self, ok: bool, what: &str, rc: i32, kind: &str) {
        if !ok {
            eprintln!("FAIL {kind}: {what} -> {rc}");
        } else if self.verbose {
            println!("ok   {kind}: {what}");
        }
    }

    fn begin(&mut self, ctx_id: u32) {
        if self.open.contains(&ctx_id) {
            return;
        }
        // virgl_renderer_context_create returns as soon as the render-server socketpair is up; the
        // vkr context itself is created on a worker thread (render_client_worker_thread ->
        // render_context_main), so it may not exist yet. A VMM never notices -- guest traffic
        // always intervenes -- but a replayer's next call is immediate, and replay_begin then
        // fails its context lookup with -EINVAL. Wait for the worker rather than race it.
        let mut rc = self.r.replay_begin(ctx_id);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while rc != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(2));
            rc = self.r.replay_begin(ctx_id);
        }
        self.note(rc == 0, &format!("replay_begin ctx {ctx_id}"), rc, "ctl");
        if rc == 0 {
            self.open.insert(ctx_id);
            self.tally.ctl_ok += 1;
        } else {
            self.tally.ctl_fail += 1;
        }
    }

    fn end(&mut self, ctx_id: u32) {
        if !self.open.remove(&ctx_id) {
            return;
        }
        let rc = self.r.replay_end(ctx_id);
        self.note(rc == 0, &format!("replay_end ctx {ctx_id}"), rc, "ctl");
        if rc == 0 {
            self.tally.ctl_ok += 1;
        } else {
            self.tally.ctl_fail += 1;
        }
    }

    /// Resource events naming a context this replayer never created say nothing about the
    /// renderer under test, so they are skipped rather than counted either way.
    fn is_classic(&self, event: &Ctl) -> bool {
        match event {
            Ctl::CtxCreate { ctx_id, .. }
            | Ctl::CtxDestroy { ctx_id }
            | Ctl::AttachResource { ctx_id, .. }
            | Ctl::DetachResource { ctx_id, .. }
            | Ctl::CreateBlob { ctx_id, .. } => self.classic.contains(ctx_id),
            _ => false,
        }
    }

    fn ctl(&mut self, event: &Ctl) {
        if self.is_classic(event) {
            return;
        }
        let (rc, what) = match event {
            Ctl::CtxCreate { ctx_id, context_init, name } => {
                if context_init & CAPSET_MASK != CAPSET_VENUS {
                    // Not a failure: a real desktop capture carries classic contexts too, and a
                    // venus-only renderer legitimately refuses them. Remember the id so the
                    // resource events that name it are skipped rather than counted as failures.
                    self.classic.insert(*ctx_id);
                    self.tally.skipped_classic.push(*ctx_id);
                    eprintln!(
                        "SKIP ctl: context_create {ctx_id} {name:?} is capset {} (classic), \
                         not venus",
                        context_init & CAPSET_MASK
                    );
                    return;
                }
                let rc = self.r.context_create(*ctx_id, *context_init, name);
                // A context created inside the stream must be replaying before the commands that
                // follow it reach it.
                if rc == 0 {
                    self.begin(*ctx_id);
                }
                (rc, format!("context_create {ctx_id} flags={context_init:#x} {name:?}"))
            }
            Ctl::CtxDestroy { ctx_id } => {
                // End the replay first: replay_end starts the deferred ring threads, and tearing
                // the context down with them unstarted loses whatever they had parked.
                self.end(*ctx_id);
                self.r.context_destroy(*ctx_id);
                (0, format!("context_destroy {ctx_id}"))
            }
            Ctl::CreateBlob { res_handle, ctx_id, blob_mem, blob_flags, blob_id, size, num_iovs } => {
                // Guest-storage blobs were backed by guest RAM. Only the count and total size were
                // recorded -- the addresses named a machine that no longer exists -- so supply one
                // contiguous backing of the right size instead. It is the shape the renderer
                // validates (total iov size >= blob size), not the layout.
                let (iovecs, iov_count) = if matches!(*blob_mem, BLOB_MEM_GUEST | BLOB_MEM_HOST3D_GUEST)
                    || *num_iovs > 0
                {
                    let mut bytes = vec![0u8; *size as usize];
                    let iov = libc::iovec {
                        iov_base: bytes.as_mut_ptr().cast(),
                        iov_len: bytes.len(),
                    };
                    self.backings.insert(*res_handle, Backing { _bytes: bytes });
                    (Box::leak(Box::new(iov)) as *const libc::iovec, 1u32)
                } else {
                    (std::ptr::null(), 0u32)
                };

                let args = abi::CreateBlobArgs {
                    res_handle: *res_handle,
                    ctx_id: *ctx_id,
                    blob_mem: *blob_mem,
                    blob_flags: *blob_flags,
                    blob_id: *blob_id,
                    size: *size,
                    iovecs,
                    num_iovs: iov_count,
                };
                let rc = self.r.create_blob(&args);
                (
                    rc,
                    format!(
                        "create_blob res={res_handle} ctx={ctx_id} mem={blob_mem} \
                         flags={blob_flags:#x} blob_id={blob_id} size={size}"
                    ),
                )
            }
            Ctl::ImportBlob { res_handle, fd_type, size } => {
                // The fd came from outside the renderer; nothing here can reconstruct it. Say so
                // once, by handle, rather than letting the first command that names it fail with
                // an unrelated message.
                self.tally.unreplayable_imports.push(*res_handle);
                eprintln!(
                    "SKIP ctl: import_blob res={res_handle} fd_type={fd_type} size={size} \
                     -- the fd came from outside the renderer and cannot be reconstructed"
                );
                return;
            }
            Ctl::AttachResource { ctx_id, res_handle } => {
                self.r.attach_resource(*ctx_id, *res_handle);
                (0, format!("attach_resource ctx={ctx_id} res={res_handle}"))
            }
            Ctl::DetachResource { ctx_id, res_handle } => {
                self.r.detach_resource(*ctx_id, *res_handle);
                (0, format!("detach_resource ctx={ctx_id} res={res_handle}"))
            }
            Ctl::ResourceUnref { res_handle } => {
                self.r.resource_unref(*res_handle);
                self.backings.remove(res_handle);
                (0, format!("resource_unref res={res_handle}"))
            }
        };

        if rc != 0 {
            if let Ctl::CreateBlob { blob_id, .. } = event {
                if *blob_id != 0 {
                    self.deferred.push((event.clone(), 0));
                    return;
                }
            }
        }
        self.note(rc == 0, &what, rc, "ctl");
        if rc == 0 {
            self.tally.ctl_ok += 1;
        } else {
            self.tally.ctl_fail += 1;
        }
    }

    /// Retry every parked event, in the order they were parked. One that still fails stays parked
    /// with its age bumped, so the report can say how far the corpus's order is from the truth.
    fn drain_deferred(&mut self) {
        if self.deferred.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.deferred);
        for (event, age) in pending {
            let retry_ok = match &event {
                Ctl::CreateBlob {
                    res_handle,
                    ctx_id,
                    blob_mem,
                    blob_flags,
                    blob_id,
                    size,
                    ..
                } => {
                    let args = abi::CreateBlobArgs {
                        res_handle: *res_handle,
                        ctx_id: *ctx_id,
                        blob_mem: *blob_mem,
                        blob_flags: *blob_flags,
                        blob_id: *blob_id,
                        size: *size,
                        iovecs: std::ptr::null(),
                        num_iovs: 0,
                    };
                    self.r.create_blob(&args) == 0
                }
                _ => true,
            };
            if retry_ok {
                self.tally.ctl_ok += 1;
                self.tally.parked += 1;
                self.tally.park_depth = self.tally.park_depth.max(age);
                if self.verbose {
                    println!("ok   ctl: parked event landed after {age} records");
                }
            } else {
                self.deferred.push((event, age + 1));
            }
        }
    }
}

fn run(args: &Args) -> Result<Tally, String> {
    let blob = std::fs::read(&args.corpus)
        .map_err(|e| format!("{}: {e}", args.corpus))?;
    let c = corpus::parse(&blob).map_err(|e| format!("{}: {e}", args.corpus))?;

    if c.flags & corpus::FLAG_TRUNC_FULL != 0 {
        eprintln!("note: corpus hit the recorder cap; it is a valid prefix, not the whole run");
    }
    if c.flags & corpus::FLAG_TRUNC_FATAL != 0 {
        eprintln!("note: recording stopped on a fatal decode; the prefix is what ran cleanly");
    }

    let r = abi::Renderer::open(&args.renderer)?;
    r.init(args.flags)?;

    let mut rp = Replay {
        r: &r,
        verbose: args.verbose,
        tally: Tally::default(),
        open: BTreeSet::new(),
        backings: BTreeMap::new(),
        classic: BTreeSet::new(),
        deferred: Vec::new(),
    };

    // 1-2. Contexts already alive when the recorder armed: they have a prologue and no CtxCreate
    // event, so create them here and feed their journal. In generation order, because two
    // prologues may share a ctx_id and the later generation is a different context that reused
    // the number. Contexts created after arming get an empty prologue and are handled by their
    // CtxCreate event in the stream.
    let created_in_stream: BTreeSet<u32> = c
        .records
        .iter()
        .filter_map(|rec| match rec {
            Record::Ctl { event: Ctl::CtxCreate { ctx_id, .. }, .. } => Some(*ctx_id),
            _ => None,
        })
        .collect();

    let mut prologues: Vec<&corpus::Prologue> = c.prologues.iter().collect();
    prologues.sort_by_key(|p| p.ctx.generation);

    for p in &prologues {
        if p.entries.is_empty() && created_in_stream.contains(&p.ctx.id) {
            continue;
        }
        // Capset venus, not a guess: the prologue IS a vkr_journal export, so a context that has
        // one is a venus context by construction. Its original ctx_flags are unknown -- it
        // predates the recorder -- but the capset is the only part that matters here.
        let rc = rp.r.context_create(p.ctx.id, CAPSET_VENUS, "vkr-replay");
        if rc != 0 {
            eprintln!("FAIL ctl: context_create {} for prologue -> {rc}", p.ctx);
        }
        rp.begin(p.ctx.id);

        for (i, e) in p.entries.iter().enumerate() {
            let mut wire = e.wire.clone();
            let rc = if e.ring_key != 0 {
                rp.r.replay_ring_cmd(p.ctx.id, e.ring_key, &mut wire)
            } else {
                rp.r.replay_submit(p.ctx.id, &mut wire)
            };
            if rc == 0 {
                rp.tally.prologue_ok += 1;
            } else {
                rp.tally.prologue_fail += 1;
                eprintln!(
                    "FAIL prologue: {} entry {i} seq {} cmd_type {} klass {} -> {rc}",
                    p.ctx, e.seq, e.cmd_type, e.klass
                );
            }
        }
    }

    // 3. The stream, in recorded order. Control events are applied AT THEIR RECORDED POSITION and
    // never hoisted: the interleaving is the dependency order both ways round -- a blob exports
    // memory an earlier command allocated, and a later command reads a blob created before it.
    for rec in &c.records {
        match rec {
            Record::Ctl { event, .. } => rp.ctl(event),
            Record::Cmd { ctx, ring_id, cmd_type, wire, .. } => {
                if rp.classic.contains(&ctx.id) {
                    continue;
                }
                if RING_FLOW_CONTROL.contains(cmd_type) {
                    rp.tally.skipped_flow_control += 1;
                    continue;
                }
                if !rp.open.contains(&ctx.id) {
                    rp.tally.cmd_fail += 1;
                    eprintln!("FAIL cmd: {ctx} is not replaying; cmd_type {cmd_type} dropped");
                    continue;
                }
                let mut w = wire.clone();
                let rc = if *ring_id != 0 {
                    rp.r.replay_ring_cmd(ctx.id, *ring_id, &mut w)
                } else {
                    rp.r.replay_submit(ctx.id, &mut w)
                };
                if rc == 0 {
                    rp.tally.cmd_ok += 1;
                    if rp.verbose {
                        println!("ok   cmd: {ctx} ring={ring_id:#x} type={cmd_type}");
                    }
                } else {
                    rp.tally.cmd_fail += 1;
                    eprintln!("FAIL cmd: {ctx} ring={ring_id:#x} type={cmd_type} -> {rc}");
                }
                rp.drain_deferred();
            }
        }
    }

    for (event, age) in std::mem::take(&mut rp.deferred) {
        rp.tally.ctl_fail += 1;
        eprintln!("FAIL ctl: still parked after {age} records: {event:?}");
    }

    // 4. replay_end last, for everything still open: it starts the deferred ring threads, and a
    // ring thread running earlier would race the replayed commands into an order that never ran.
    for ctx_id in rp.open.clone() {
        rp.end(ctx_id);
    }

    // The oracle. Counts say the commands were accepted; the census says something was built.
    let mut ctxs: BTreeSet<u32> = c.prologues.iter().map(|p| p.ctx.id).collect();
    ctxs.extend(c.records.iter().map(|rec| rec.seq_ctx().id).filter(|id| *id != 0));
    for ctx_id in ctxs {
        if rp.classic.contains(&ctx_id) {
            continue;
        }
        match rp.r.memory_census(ctx_id) {
            Ok(pairs) => {
                let total: u64 = pairs.iter().map(|(_, sz)| sz).sum();
                println!(
                    "CENSUS ctx={ctx_id} allocations={} bytes={total}",
                    pairs.len()
                );
            }
            Err(rc) => println!("CENSUS ctx={ctx_id} unavailable ({rc})"),
        }
    }
    rp.r.dump_state();

    let tally = std::mem::take(&mut rp.tally);
    r.cleanup();
    Ok(tally)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("vkr-replay: {e}");
            eprintln!("usage: vkr-replay <corpus.vkrc> --renderer <lib> [--flags N] [--verbose]");
            return ExitCode::FAILURE;
        }
    };

    match run(&args) {
        Ok(t) => {
            println!(
                "REPLAY prologue={}/{} cmds={}/{} ctl={}/{}",
                t.prologue_ok,
                t.prologue_ok + t.prologue_fail,
                t.cmd_ok,
                t.cmd_ok + t.cmd_fail,
                t.ctl_ok,
                t.ctl_ok + t.ctl_fail
            );
            if t.parked != 0 {
                println!(
                    "REPLAY parked-and-retried blob creates: {} (deepest wait {} records)",
                    t.parked, t.park_depth
                );
            }
            if t.skipped_flow_control != 0 {
                println!(
                    "REPLAY skipped ring flow-control commands: {}",
                    t.skipped_flow_control
                );
            }
            if !t.skipped_classic.is_empty() {
                println!("REPLAY skipped classic contexts: {:?}", t.skipped_classic);
            }
            if !t.unreplayable_imports.is_empty() {
                println!(
                    "REPLAY unreplayable imported blobs: {:?}",
                    t.unreplayable_imports
                );
            }
            if t.failed() == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("vkr-replay: {e}");
            ExitCode::FAILURE
        }
    }
}
