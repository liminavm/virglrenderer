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
use std::os::raw::c_int;
use std::process::ExitCode;
use std::time::Duration;

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
    /// Write the score here instead of only printing it.
    score: Option<String>,
    /// Compare the score against this file and fail on any difference.
    expect: Option<String>,
    /// Exercise only what a skeleton owes: init, context create, resource create. No replay feed,
    /// no commands, no scoring. This is P1's gate -- a renderer that gets through it has a working
    /// ABI, resource table and context table, which is all a skeleton claims.
    smoke: bool,
    /// Export each venus context's snapshot journal, rebuild a fresh context from it, and require
    /// the rebuilt context's own journal to be the same one. See `rebuild_gate`.
    rebuild: bool,
    /// Run the gate after this many commands have been replayed, instead of only at teardown.
    /// A real suspend happens mid-workload; a capture of a workload that exits ends with the
    /// guest's own teardown, where there is nothing left to rebuild.
    rebuild_at: Option<u64>,
}

fn parse_args() -> Result<Args, String> {
    let mut corpus = None;
    let mut renderer = None;
    let mut flags = abi::DEFAULT_FLAGS;
    let mut verbose = false;
    let mut score = None;
    let mut expect = None;
    let mut smoke = false;
    let mut rebuild = false;
    let mut rebuild_at = None;

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
            "--score" => score = Some(it.next().ok_or("--score wants a path")?),
            "--expect" => expect = Some(it.next().ok_or("--expect wants a path")?),
            "--smoke" => smoke = true,
            "--rebuild" => rebuild = true,
            "--rebuild-at" => {
                let v = it.next().ok_or("--rebuild-at wants a command count")?;
                rebuild_at =
                    Some(v.parse().map_err(|_| format!("--rebuild-at {v}: not a number"))?);
                rebuild = true;
            }
            _ if corpus.is_none() => corpus = Some(a),
            _ => return Err(format!("unexpected argument {a}")),
        }
    }

    if smoke && (score.is_some() || expect.is_some()) {
        // A smoke score is a strict subset of a real one. Pinning it would replace a golden with
        // a weaker one that still passes, which is the failure a fixture exists to prevent.
        return Err("--smoke scores nothing; drop --score/--expect".into());
    }
    if smoke && rebuild {
        return Err("--smoke replays nothing, so there is no journal to rebuild from".into());
    }
    Ok(Args {
        corpus: corpus.ok_or("no corpus given")?,
        renderer: renderer
            .or_else(|| std::env::var("VIRGL_RENDERER_LIB").ok())
            .ok_or("no --renderer and no VIRGL_RENDERER_LIB")?,
        flags,
        verbose,
        score,
        expect,
        smoke,
        rebuild,
        rebuild_at,
    })
}

/// What the replay did, per context. Pass one's oracle: a renderer that accepts every command and
/// builds nothing looks identical to a working one in a pass/fail count alone, so the memory
/// census is reported beside the counts rather than instead of them.
#[derive(Default)]
struct Tally {
    /// Smoke mode only: blob creates that export an object a command would have made.
    smoke_skipped_exports: u64,
    prologue_ok: u64,
    prologue_fail: u64,
    /// Contexts whose journal rebuilt into the same journal, and those that did not.
    rebuild_ok: u64,
    rebuild_fail: u64,
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
    /// What makes the run fail. `rebuild_fail` is here and deliberately not in the score text: the
    /// gate is a claim about the renderer, not about the corpus, so a tree that passes it must
    /// produce the same score file as one that was never asked to run it -- otherwise turning the
    /// gate on would force every pinned fixture to be re-recorded.
    fn failed(&self) -> u64 {
        self.prologue_fail + self.cmd_fail + self.ctl_fail + self.rebuild_fail
    }
}

/// Backing store synthesized for a guest-storage blob. The guest RAM it pointed into is gone, so
/// only the size was recorded; the bytes are ours. Kept alive for the resource's lifetime because
/// the renderer holds the iovec.
struct Backing {
    _bytes: Vec<u8>,
}

/// A `create_blob` that exported host-side memory: everything needed to make it again.
#[derive(Clone, Copy)]
struct ExportedBlob {
    blob_mem: u32,
    blob_flags: u32,
    blob_id: u64,
    size: u64,
    /// The context's journal watermark when this blob was created. This is the fence: the guest
    /// may free the allocation later in the same journal, and a rebuilt context that replays the
    /// whole journal before creating its blobs is asked to export something already gone.
    at: u64,
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
    /// Score lines, in the order the contexts produced them. A context is scored when it is
    /// destroyed and once more at the end if it is still alive -- a workload that tears down
    /// cleanly leaves nothing to census, so scoring only at the end would score nothing at all.
    score: Vec<String>,
    /// Contexts already scored at their destroy, so the final sweep does not score them twice.
    scored: BTreeSet<u32>,
    /// Whether to run the snapshot fixed-point gate, and which contexts it has already run on.
    rebuild: bool,
    rebuilt: BTreeSet<u32>,
    /// The resources each context currently has attached.
    ///
    /// A rebuilt context needs the same ones: a journal's `vkAllocateMemory` may import from a
    /// resource, and a context that cannot reach it fails that create -- which then reads as the
    /// journal having lost the allocation, when what it lost was the attachment.
    attached: BTreeMap<u32, BTreeSet<u32>>,
    /// The exporting blobs each context currently has alive, by resource handle.
    ///
    /// These are the VMM's half of the world. A journal entry retained *because* a blob still
    /// holds the allocation it exported has no counterpart in a rebuilt context until the same
    /// blobs are created against it, so the gate replays these too -- which is the fence, taken
    /// once, in the one shape this harness can reach.
    exports: BTreeMap<u32, BTreeMap<u32, ExportedBlob>>,
    /// Gate after this many replayed commands, and how many have gone by.
    rebuild_at: Option<u64>,
    replayed: u64,
    /// DIAGNOSTIC, not the shipped semantics. A create_blob that exports a VkDeviceMemory
    /// (blob_id != 0) can be recorded ahead of the vkAllocateMemory that made it: the recorder
    /// orders events by when each thread reached its lock, and that is not the order they
    /// executed in. Parking such a create and retrying it as the stream advances tells us whether
    /// ordering is the LAST thing wrong with a corpus -- and how far off the order actually is.
    /// The real fix is a recorded dependency fence, not a retry loop.
    deferred: Vec<(Ctl, u64)>,
    smoke: bool,
    /// Blobs that came back IOSurface-backed, by the context that created them. IOSurface is the
    /// whole present path on this host -- a venus scanout blob has no CPU transfer_read, so its
    /// pixels exist nowhere but the surface -- and backing the wrong set of blobs is silent in
    /// every other line of the score.
    ///
    /// Handles, never ids. An id is host-private, recycled the instant its surface dies, and free
    /// to change across a snapshot restore, so pinning one would pin a number no implementation
    /// owes us.
    iosurf: BTreeMap<u32, BTreeSet<u32>>,
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
        if self.smoke || self.open.contains(&ctx_id) {
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
        if self.smoke || !self.open.remove(&ctx_id) {
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
        // A blob with a non-zero blob_id EXPORTS an object a command created -- a VkDeviceMemory,
        // usually. Smoke mode runs no commands, so there is nothing to export and the create
        // rightly fails. Skipping it is not lowering the bar: the C renderer fails these in smoke
        // mode too, identically, which is how we know the bar was in the wrong place.
        if self.smoke {
            if let Ctl::CreateBlob { blob_id, .. } = event {
                if *blob_id != 0 {
                    self.tally.smoke_skipped_exports += 1;
                    return;
                }
            }
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
                // The guest reuses context ids. This is a DIFFERENT context wearing the number of
                // one already scored at its destroy, so it owes a score of its own.
                self.scored.remove(ctx_id);
                // Its blobs die with the context it replaced, so its IOSurface tally starts at
                // zero too -- carrying the dead generation's count forward would report a port
                // that backs nothing as backing everything the previous one did.
                self.iosurf.remove(ctx_id);
                // A reused id is a different context, so it owes the gate its own answer. Without
                // this the second life of an id is silently taken as already gated -- and this
                // corpus reuses ctx 8 three times.
                self.rebuilt.remove(ctx_id);
                self.attached.remove(ctx_id);
                self.exports.remove(ctx_id);
                (rc, format!("context_create {ctx_id} flags={context_init:#x} {name:?}"))
            }
            Ctl::CtxDestroy { ctx_id } => {
                // End the replay first: replay_end starts the deferred ring threads, and tearing
                // the context down with them unstarted loses whatever they had parked.
                self.end(*ctx_id);
                // Score before destroying: this is the last moment the context's device memory
                // exists, and for a workload that exits cleanly it is the ONLY moment.
                if !self.smoke {
                    let blobs = self.exports.get(ctx_id).cloned().unwrap_or_default();
                    let lines = score_context(&self.r, *ctx_id, &blobs);
                    self.score.extend(lines);
                    self.score.push(iosurf_line(&self.iosurf, *ctx_id));
                    // And rebuild it here for the same reason it is scored here. A capture of a
                    // workload that exits carries the guest's own teardown, so by end of stream
                    // this context is gone and its journal with it -- a gate that waited would be
                    // asking a context that no longer exists and calling the silence a pass.
                    if self.rebuild {
                        let res = self.attached.get(ctx_id).cloned().unwrap_or_default();
                        let exp = self.exports.get(ctx_id).cloned().unwrap_or_default();
                        rebuild_gate(&self.r, *ctx_id, &res, &exp, &mut self.tally);
                        self.rebuilt.insert(*ctx_id);
                    }
                }
                self.scored.insert(*ctx_id);
                self.r.context_destroy(*ctx_id);
                (0, format!("context_destroy {ctx_id}"))
            }
            Ctl::CreateBlob {
                res_handle,
                ctx_id,
                blob_mem,
                blob_flags,
                blob_id,
                size,
                num_iovs,
            } => {
                // Guest-storage blobs were backed by guest RAM. Only the count and total size were
                // recorded -- the addresses named a machine that no longer exists -- so supply one
                // contiguous backing of the right size instead. It is the shape the renderer
                // validates (total iov size >= blob size), not the layout.
                let (iovecs, iov_count) =
                    if matches!(*blob_mem, BLOB_MEM_GUEST | BLOB_MEM_HOST3D_GUEST) || *num_iovs > 0
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
                if rc == 0 && self.r.iosurface_id(*res_handle).is_some() {
                    self.iosurf.entry(*ctx_id).or_default().insert(*res_handle);
                }
                if rc == 0 && *blob_id != 0 {
                    self.exports.entry(*ctx_id).or_default().insert(
                        *res_handle,
                        ExportedBlob {
                            blob_mem: *blob_mem,
                            blob_flags: *blob_flags,
                            blob_id: *blob_id,
                            size: *size,
                            at: self.r.journal_seq(*ctx_id),
                        },
                    );
                }
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
                self.attached.entry(*ctx_id).or_default().insert(*res_handle);
                (0, format!("attach_resource ctx={ctx_id} res={res_handle}"))
            }
            Ctl::DetachResource { ctx_id, res_handle } => {
                self.r.detach_resource(*ctx_id, *res_handle);
                self.attached.entry(*ctx_id).or_default().remove(res_handle);
                (0, format!("detach_resource ctx={ctx_id} res={res_handle}"))
            }
            Ctl::ResourceUnref { res_handle } => {
                self.r.resource_unref(*res_handle);
                self.backings.remove(res_handle);
                for blobs in self.exports.values_mut() {
                    blobs.remove(res_handle);
                }
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
                    res_handle, ctx_id, blob_mem, blob_flags, blob_id, size, ..
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
                    let ok = self.r.create_blob(&args) == 0;
                    if ok {
                        self.exports.entry(*ctx_id).or_default().insert(
                            *res_handle,
                            ExportedBlob {
                                blob_mem: *blob_mem,
                                blob_flags: *blob_flags,
                                blob_id: *blob_id,
                                size: *size,
                                at: self.r.journal_seq(*ctx_id),
                            },
                        );
                    }
                    ok
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

fn run(args: &Args) -> Result<(Tally, Vec<String>), String> {
    let blob = std::fs::read(&args.corpus).map_err(|e| format!("{}: {e}", args.corpus))?;
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
        score: Vec::new(),
        scored: BTreeSet::new(),
        deferred: Vec::new(),
        smoke: args.smoke,
        iosurf: BTreeMap::new(),
        rebuild: args.rebuild,
        rebuilt: BTreeSet::new(),
        attached: BTreeMap::new(),
        exports: BTreeMap::new(),
        rebuild_at: args.rebuild_at,
        replayed: 0,
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
        if rp.smoke {
            continue;
        }

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
                if rp.smoke {
                    continue;
                }
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

                // Mid-stream, which is the only shape a real suspend has: the guest is still
                // running and its world is at its richest. Gating only at teardown asks a context
                // that has already destroyed everything it built, and calls the empty answer a
                // pass.
                rp.replayed += 1;
                if rp.rebuild_at == Some(rp.replayed) {
                    for ctx_id in rp.open.clone() {
                        if rp.classic.contains(&ctx_id) || !rp.rebuilt.insert(ctx_id) {
                            continue;
                        }
                        let res = rp.attached.get(&ctx_id).cloned().unwrap_or_default();
                        let exp = rp.exports.get(&ctx_id).cloned().unwrap_or_default();
                        rebuild_gate(&rp.r, ctx_id, &res, &exp, &mut rp.tally);
                    }
                }
            }
        }
    }

    for (event, age) in std::mem::take(&mut rp.deferred) {
        rp.tally.ctl_fail += 1;
        eprintln!("FAIL ctl: still parked after {age} records: {event:?}");
    }

    // 4. replay_end last, for everything still open: it starts the deferred ring threads, and a
    // ring thread running earlier would race the replayed commands into an order that never ran.
    // Read before `end` empties it: these are the contexts the stream never destroyed, and they
    // are exactly the ones the gate below still owes an answer for.
    let still_open: BTreeSet<u32> = rp.open.clone();
    for ctx_id in rp.open.clone() {
        rp.end(ctx_id);
    }

    // The snapshot gate, before the score is taken: it creates and destroys contexts of its own,
    // and running it after scoring would score a world it had already added to.
    // Contexts the stream never destroyed -- the shape a real suspend has, where the guest is
    // still running. Those the stream did destroy were gated at that point, where they still
    // existed.
    if args.rebuild && !rp.smoke {
        for ctx_id in still_open {
            if rp.classic.contains(&ctx_id) || !rp.rebuilt.insert(ctx_id) {
                continue;
            }
            let res = rp.attached.get(&ctx_id).cloned().unwrap_or_default();
            let exp = rp.exports.get(&ctx_id).cloned().unwrap_or_default();
            rebuild_gate(&rp.r, ctx_id, &res, &exp, &mut rp.tally);
        }
    }

    // The oracle. Counts say the commands were accepted; the census says something was built, and
    // the content hashes say it was built the same way.
    let mut ctxs: BTreeSet<u32> = c.prologues.iter().map(|p| p.ctx.id).collect();
    ctxs.extend(c.records.iter().map(|rec| rec.seq_ctx().id).filter(|id| *id != 0));
    for ctx_id in ctxs {
        if rp.classic.contains(&ctx_id) || rp.scored.contains(&ctx_id) {
            continue;
        }
        if rp.smoke {
            continue;
        }
        let blobs = rp.exports.get(&ctx_id).cloned().unwrap_or_default();
        rp.score.extend(score_context(&rp.r, ctx_id, &blobs));
        rp.score.push(iosurf_line(&rp.iosurf, ctx_id));
    }
    let score = std::mem::take(&mut rp.score);
    rp.r.dump_state();

    let tally = std::mem::take(&mut rp.tally);
    r.cleanup();
    Ok((tally, score))
}

/// Where a rebuilt context is stood up, added to the id of the context it was built from.
///
/// Far enough above any id a corpus uses that the two cannot collide, and deliberately derived
/// from the original rather than allocated: a failure names `1008` and the reader knows it was
/// ctx 8 that could not be rebuilt.
const REBUILT_BASE: u32 = 1000;

/// Where a rebuilt context's re-exported blobs are stood up, added to the handle each had. Above
/// every handle a corpus uses, for the same reason and read the same way.
const REBUILT_RES_BASE: u32 = 0x0100_0000;

/// Require that replaying a context's journal yields a context whose journal is that same journal.
///
/// This is a fixed point, and so it is the floor rather than the ceiling. It cannot see what the
/// recorder never learned to keep -- a command dropped on the way in is equally absent from both
/// exports, and the two agree about a world that is missing it. What it does catch is everything
/// the recorder keeps and the replay cannot use: an entry whose objects are rebuilt in the wrong
/// order, one that names something the closure did not drag in, and any retention rule that is not
/// stable under being applied twice.
///
/// It does cross the fence -- the interleave where the VMM creates a blob against a half-replayed
/// context. Each blob the original exported is remade at the journal watermark it was first made
/// at, which is what a resuming VMM does and what a single `replay_upto(MAX)` cannot: an
/// allocation the guest frees later in the same journal is exportable only before that free.
/// What this cannot reach is the guest side of a resume -- there is no VM here, so nothing ever
/// reads back what the rebuilt blobs point at.
///
/// `seq` is deliberately not compared. A rebuilt context numbers its own journal from one, so the
/// sequence numbers differ by construction; what has to match is which commands were retained, in
/// which order, with which bytes, on which ring.
fn rebuild_gate(
    r: &abi::Renderer,
    ctx_id: u32,
    resources: &BTreeSet<u32>,
    exports: &BTreeMap<u32, ExportedBlob>,
    tally: &mut Tally,
) {
    let before = match r.journal_export(ctx_id) {
        Ok(b) => b,
        // Nothing retained is a real answer, not a failure: a context whose every command was
        // transient has nothing to rebuild and nothing to disagree about.
        Err(_) => {
            // Two very different answers, and an absent journal cannot tell them apart: a
            // recorder that saw nothing is a broken tee, and one that saw commands and kept none
            // of them is a context the guest tore down before the capture ended.
            println!(
                "rebuild ctx={ctx_id} nothing retained ({} commands recorded)",
                r.journal_seq(ctx_id)
            );
            return;
        }
    };

    let fresh = ctx_id + REBUILT_BASE;
    let mut fail = |why: String| {
        tally.rebuild_fail += 1;
        eprintln!("FAIL rebuild: ctx {ctx_id}: {why}");
    };

    let rc = r.context_create(fresh, CAPSET_VENUS, "vkr-rebuild");
    if rc != 0 {
        return fail(format!("context_create {fresh} -> {rc}"));
    }
    // The same resources the original can reach. A journal's `vkAllocateMemory` may import from
    // one, and without this the create fails and reads as a journal that lost the allocation --
    // which is how this gate first reported a renderer bug that was its own.
    for res in resources {
        r.attach_resource(fresh, *res);
    }
    let rc = r.replay_begin(fresh);
    if rc != 0 {
        r.context_destroy(fresh);
        return fail(format!("replay_begin {fresh} -> {rc}"));
    }
    let rc = r.journal_restore(fresh, &before);
    if rc != 0 {
        r.replay_end(fresh);
        r.context_destroy(fresh);
        return fail(format!("journal_restore {fresh} ({} bytes) -> {rc}", before.len()));
    }
    // The fence, walked station by station. An entry is retained partly BECAUSE a blob still
    // holds the allocation it exported, so a rebuilt context with no blobs keeps less than the
    // original and the two journals differ by this gate's own gap rather than by anything the
    // renderer did. And each blob has to be made where it was made: the guest may free the
    // allocation later in the same journal, so a context that replays to the end first is asked
    // to export memory that is already gone -- which is the interleave the watermark exists for.
    let mut ordered: Vec<(&u32, &ExportedBlob)> = exports.iter().collect();
    ordered.sort_by_key(|(res, b)| (b.at, **res));

    let mut remade = Vec::with_capacity(ordered.len());
    let feed = |upto: u64| {
        let rc = r.journal_replay_upto(fresh, upto);
        if rc != 0 {
            Err(format!("journal_replay_upto {fresh} to {upto} -> {rc}"))
        } else {
            Ok(())
        }
    };

    let mut trouble = None;
    for (res_handle, b) in ordered {
        if let Err(e) = feed(b.at) {
            trouble = Some(e);
            break;
        }
        let handle = res_handle + REBUILT_RES_BASE;
        let args = abi::CreateBlobArgs {
            res_handle: handle,
            ctx_id: fresh,
            blob_mem: b.blob_mem,
            blob_flags: b.blob_flags,
            blob_id: b.blob_id,
            size: b.size,
            iovecs: std::ptr::null(),
            num_iovs: 0,
        };
        let rc = r.create_blob(&args);
        if rc != 0 {
            trouble = Some(format!(
                "re-exporting blob_id {} ({} bytes, was res {res_handle}) into {fresh} at seq {} \
                 -> {rc} -- the rebuilt world does not have the allocation the original exported",
                b.blob_id, b.size, b.at
            ));
            break;
        }
        remade.push(handle);
    }
    if trouble.is_none() {
        if let Err(e) = feed(u64::MAX) {
            trouble = Some(e);
        }
    }
    if let Some(why) = trouble {
        for h in &remade {
            r.resource_unref(*h);
        }
        r.replay_end(fresh);
        r.context_destroy(fresh);
        return fail(why);
    }

    // Before `replay_end`, because that is where the VMM writes them: a restore puts the contents
    // in while the rings are still idle, and starting a ring over an allocation whose bytes have
    // not landed is the interleave the whole two-call shape exists to avoid. Held, not reported,
    // until the journals have been compared -- a held mismatch is the cause and a write refused
    // for an allocation the rebuild did not make is its symptom.
    let content = content_gate(r, ctx_id, fresh);
    let sync = sync_gate(r, ctx_id, fresh);

    let rc = r.replay_end(fresh);
    if rc != 0 {
        for h in &remade {
            r.resource_unref(*h);
        }
        r.context_destroy(fresh);
        return fail(format!("replay_end {fresh} -> {rc}"));
    }

    let after = r.journal_export(fresh).unwrap_or_default();
    // Said before the journals are compared, because it is the cause and they are the symptom: an
    // entry retained by a held allocation on one side and not the other makes the two differ, and
    // "the rebuilt journal has fewer entries" does not tell the reader which allocation went
    // missing or that a blob is why.
    let (held, held_after) = (r.journal_held(ctx_id), r.journal_held(fresh));
    for h in &remade {
        r.resource_unref(*h);
    }
    r.context_destroy(fresh);
    if held != held_after {
        return fail(format!(
            "{held_after:?} allocation(s) held by blobs in the rebuilt context, {held:?} in the \
             original -- re-exporting {} blob(s) did not reproduce the same held set",
            remade.len()
        ));
    }

    let key = |id| corpus::CtxKey { id, generation: 0 };
    let (a, b) = match (
        corpus::parse_journal(&before, key(ctx_id)),
        corpus::parse_journal(&after, key(fresh)),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return fail(format!("journal does not parse: {e}")),
    };

    if a.len() != b.len() {
        return fail(format!(
            "rebuilt journal has {} entries, the original {} -- a replayed command retained \
             something the guest's did not, or the reverse",
            b.len(),
            a.len()
        ));
    }
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if x.cmd_type != y.cmd_type || x.ring_key != y.ring_key || x.wire != y.wire {
            return fail(format!(
                "entry {i} differs: original cmd_type={} ring={:#x} {} bytes, \
                 rebuilt cmd_type={} ring={:#x} {} bytes",
                x.cmd_type,
                x.ring_key,
                x.wire.len(),
                y.cmd_type,
                y.ring_key,
                y.wire.len()
            ));
        }
    }
    match content {
        Ok(n) => println!("rebuild ctx={ctx_id} contents restored into {n} allocation(s)"),
        Err(why) => return fail(why),
    }
    match sync {
        Ok((n, moved)) => {
            println!("rebuild ctx={ctx_id} sync {n} object(s), {moved} fast-forwarded");
            // The one number that says whether this scored anything. A rebuilt world that already
            // agrees with the capture is compared against itself, and the fixed point passes
            // whatever either half does -- so a corpus that reports zero here says so on every
            // run rather than reading as a pass. It cannot be pinned: gate results stay out of
            // the score deliberately (see `Tally::failed`), and this is the substitute.
            if moved == 0 {
                eprintln!(
                    "sync: ctx {ctx_id} scored nothing -- the rebuilt world already matched the \
                     capture. Only a world with work in flight exercises this; try --rebuild-at."
                );
            }
        }
        Err(why) => return fail(why),
    }
    tally.rebuild_ok += 1;
    println!("rebuild ctx={ctx_id} {} entries, {} bytes, identical", a.len(), before.len());
}

/// FNV-1a, 64-bit. Inline because a content hash needs to be reproducible and comparable by hand,
/// not fast or cryptographic -- and a score file is worth no new dependency.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// What one sampled read is of: an allocation the context still holds, or a blob the VMM holds
/// over one.
///
/// The two are sampled together and scored by one stability rule, because they are two views of
/// the same bytes and an answer that moves moves in both. `Mem` sorts first so the score reads
/// allocation-then-blob for a context.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum Read {
    Mem(u64),
    Blob(u32),
}

/// Cap what a single read can cost us: a 64 MB scanout blob is real, and reading it whole on
/// every pass is the difference between a score and a stall. The prefix is still content, and a
/// divergence that misses the first megabyte is not one we can miss for long.
const READ_CAP: usize = 1 << 20;

/// One pass over what a context's memory holds: the census, read allocation by allocation, and
/// every exporting blob read the way the guest reads it. Keyed by what was read so passes can be
/// compared entry by entry rather than as a block of text -- which is what lets one moving
/// allocation be named instead of spoiling the whole context's score.
///
/// The blobs are here and not beside the census on purpose. `memory_census` deliberately skips an
/// exported allocation -- its bytes are the blob's, and reporting them under both would read one
/// buffer twice -- so without this the whole exported half of a context is unscored. It is also
/// the only read that survives the guest freeing the allocation: the resource holds a share of the
/// storage, so `vkFreeMemory` retires the record and leaves these bytes standing. Every venus

/// Put the original context's sync state into the rebuilt one, and require a capture of the
/// result to be the capture that went in.
///
/// The third half of a snapshot, after the world and its bytes. A rebuilt fence is freshly
/// created and unsignalled and a rebuilt timeline sits at its create value, so a resume that
/// stopped at the journal comes back to a guest waiting on fences that will never signal --
/// the submits that would have signalled them died with the renderer.
///
/// What makes it non-vacuous is the count reported here: `moved` is how many entries the blank
/// world disagreed with the capture about, and a corpus whose blank world already matches scores
/// nothing for sync however green it looks. That number is in the score, so a pin is what says
/// which corpora actually exercise this -- run at end of stream it is usually zero, and
/// `--rebuild-at` mid-stream is where a live world has fences in flight.
///
/// Before `replay_end` for the reason the contents are: the VMM restores sync while the rings are
/// still idle, because a started ring can consume a guest wait rooted in the pre-suspend epoch
/// before the fast-forward has run.
fn sync_gate(r: &abi::Renderer, ctx_id: u32, fresh: u32) -> Result<(usize, usize), String> {
    let want = match r.sync_export(ctx_id) {
        Ok(b) => b,
        Err(rc) => return Err(format!("sync_export {ctx_id} -> {rc}")),
    };
    let blank = match r.sync_export(fresh) {
        Ok(b) => b,
        Err(rc) => return Err(format!("sync_export {fresh} -> {rc}")),
    };
    let entries = sync_entries(&want)?;
    let moved = sync_entries(&blank).map(|b| entries.iter().filter(|e| !b.contains(e)).count())?;

    let rc = r.sync_restore(fresh, &want);
    if rc != 0 {
        return Err(format!("sync_restore {fresh} -> {rc}"));
    }
    let got = match r.sync_export(fresh) {
        Ok(b) => b,
        Err(rc) => return Err(format!("sync_export {fresh} after restore -> {rc}")),
    };
    if got != want {
        return Err(format!(
            "sync ctx={ctx_id}: the restored state is not the captured one ({} vs {} bytes)",
            got.len(),
            want.len()
        ));
    }
    Ok((entries.len(), moved))
}

/// A sync blob's entries, as opaque fixed-size records.
///
/// The harness parses the blob only far enough to count and compare entries -- what each field
/// means is the renderer's business, and a second reading of it here would be a second
/// implementation to keep in step.
fn sync_entries(blob: &[u8]) -> Result<Vec<[u8; 24]>, String> {
    if blob.len() < 8 || blob[0..4] != 0x4e59_5a4cu32.to_le_bytes() {
        return Err(format!("sync: {} bytes that are not a sync blob", blob.len()));
    }
    let count = u32::from_le_bytes(blob[4..8].try_into().expect("four bytes")) as usize;
    if blob.len() - 8 != count * 24 {
        return Err(format!("sync: {count} entries in {} bytes", blob.len()));
    }
    Ok((0..count)
        .map(|i| blob[8 + i * 24..8 + (i + 1) * 24].try_into().expect("24 bytes"))
        .collect())
}

/// Put the original context's allocation contents into the rebuilt one, and require them to read
/// back as themselves.
///
/// The other half of what a snapshot is. `rebuild_gate` proves a journal rebuilds the *world* --
/// the objects, and the commands that would make them again -- and says nothing about the bytes
/// in it: a rebuilt allocation is freshly allocated and blank, so a resume that stopped there
/// would come back to a desktop of empty windows. This is `memory_read` and `memory_write` closing
/// on each other over a live pair of contexts, which is the only place the pair can be scored --
/// at end of stream there is no rebuilt context to write into, and comparing a dead world to a
/// dead world compares nothing.
///
/// A prefix, capped the same way the census is: what the VMM keeps is what it can put back, and
/// the cap is the harness's, not the mechanism's.
///
/// Run where the VMM runs it -- after the journal has been fed and before `replay_end` starts the
/// rings -- so what this scores is the order a resume actually uses.
fn content_gate(r: &abi::Renderer, ctx_id: u32, fresh: u32) -> Result<usize, String> {
    let pairs = r
        .memory_census(ctx_id)
        .map_err(|rc| format!("memory_census {ctx_id} for the content gate -> {rc}"))?;
    let mut n = 0;
    for (mem_id, size) in pairs {
        let want = size.min(READ_CAP as u64) as usize;
        let mut src = vec![0u8; want];
        let rc = r.memory_read(ctx_id, mem_id, &mut src);
        if rc != 0 {
            return Err(format!("memory_read {ctx_id} mem {mem_id} ({want} bytes) -> {rc}"));
        }
        let rc = r.memory_write(fresh, mem_id, &src);
        if rc != 0 {
            return Err(format!(
                "memory_write {fresh} mem {mem_id} ({want} bytes) -> {rc} -- the rebuilt context                  has no allocation of that size under the id the original was read from"
            ));
        }
        let mut back = vec![0u8; want];
        let rc = r.memory_read(fresh, mem_id, &mut back);
        if rc != 0 {
            return Err(format!("memory_read {fresh} mem {mem_id} ({want} bytes) -> {rc}"));
        }
        if back != src {
            return Err(format!(
                "mem {mem_id} reads back as {:016x} in the rebuilt context, {:016x} in the                  original -- the write landed somewhere the read does not look",
                fnv1a(&back),
                fnv1a(&src)
            ));
        }
        n += 1;
    }
    Ok(n)
}

/// corpus we hold reaches that state.
fn census_pass(
    r: &abi::Renderer,
    ctx_id: u32,
    blobs: &BTreeMap<u32, ExportedBlob>,
) -> Result<BTreeMap<Read, (u64, Result<u64, c_int>)>, c_int> {
    let pairs = r.memory_census(ctx_id)?;
    let mut out = BTreeMap::new();
    for (mem_id, size) in pairs {
        let mut buf = vec![0u8; size.min(READ_CAP as u64) as usize];
        let rc = r.memory_read(ctx_id, mem_id, &mut buf);
        out.insert(Read::Mem(mem_id), (size, if rc != 0 { Err(rc) } else { Ok(fnv1a(&buf)) }));
    }
    for (&res_handle, b) in blobs {
        // The mapped extent is the renderer's answer, not the blob's recorded size: a resource
        // that maps short of what it was created for is the divergence this is here to catch.
        let entry = match r.blob_read(res_handle, READ_CAP) {
            Ok((size, buf)) => (size, Ok(fnv1a(&buf))),
            // The size it was created for, so a refusal still says which blob was refused.
            Err(rc) => (b.size, Err(rc)),
        };
        out.insert(Read::Blob(res_handle), entry);
    }
    Ok(out)
}

/// How many of a context's blobs came back IOSurface-backed.
///
/// A count, not the pixels. Reading a surface needs its geometry -- `read_iosurface` takes a byte
/// stride and a row count -- and the venus corpus does not carry it: the dimensions live in
/// SET_SCANOUT_BLOB, a virtio-gpu control command, and what this stream records is ring traffic.
/// Guessing them would read a surface out of bounds. Recording the scanout geometry alongside the
/// ring stream is what unlocks hashing venus frames here, and it is the one Layer 2 oracle for
/// venus pixels that exists at all.
fn iosurf_line(map: &BTreeMap<u32, BTreeSet<u32>>, ctx_id: u32) -> String {
    let n = map.get(&ctx_id).map_or(0, |s| s.len());
    format!("iosurface ctx={ctx_id} backed={n}")
}

/// Score a context's device memory, hashing only what holds still.
///
/// `replay_end` starts the deferred ring threads, and no ABI reports when they have drained, so a
/// census taken afterwards races work that is still running. The old rule -- resample the whole
/// context until two passes agreed -- fails in both directions. It can agree early, when two
/// samples catch the same half-drawn frame, which happens readily on a host with a VM running,
/// which is to say while anyone is capturing. And a single moving allocation makes the whole
/// context look unsettled when every other allocation in it is long finished.
///
/// So stability is decided per allocation. An allocation whose hash is identical across every
/// sample is scored by that hash. One that moves is scored as `unstable`, with how many distinct
/// values it took, and is NOT given a hash -- because a hash of a moving target pins nothing and
/// reads as a renderer divergence on the next run.
///
/// This is measured, not assumed: on `synoik-glclient` two 4,128,768-byte allocations are the
/// compositor's framebuffers, and their contents never converge -- 500 ms, 3 s and 8 s of settling
/// give three different answers, and the two renderer legs disagree by timing alone. Everything
/// else in that corpus is rock stable. Waiting longer is not the fix and never was.
fn score_context(
    r: &abi::Renderer,
    ctx_id: u32,
    blobs: &BTreeMap<u32, ExportedBlob>,
) -> Vec<String> {
    /// How many samples decide stability, and how far apart. The lead exists because the first
    /// sample after `replay_end` is the least representative one.
    const SAMPLES: u32 = 4;
    const SAMPLE_WAIT: Duration = Duration::from_millis(200);
    const SAMPLE_LEAD: Duration = Duration::from_millis(500);

    let lead = std::env::var("VKR_SETTLE_LEAD_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map_or(SAMPLE_LEAD, Duration::from_millis);
    std::thread::sleep(lead);

    let mut samples = Vec::with_capacity(SAMPLES as usize);
    for i in 0..SAMPLES {
        if i > 0 {
            std::thread::sleep(SAMPLE_WAIT);
        }
        match census_pass(r, ctx_id, blobs) {
            Ok(p) => samples.push(p),
            // A refused census is a fact about the run, not a reason to stop scoring the others.
            Err(rc) => return vec![format!("census ctx={ctx_id} UNAVAILABLE rc={rc}")],
        }
    }

    let last = samples.last().expect("SAMPLES > 0");
    let mut lines = Vec::with_capacity(last.len() + 2);
    let allocations = last.keys().filter(|k| matches!(k, Read::Mem(_))).count();
    lines.push(format!("census ctx={ctx_id} allocations={allocations}"));

    // Membership that moves is a different fact from content that moves, and hiding it inside a
    // per-allocation verdict would lose it: an allocation freed or made between samples is not an
    // unstable hash, it is a census that has not stopped changing shape.
    if samples.iter().any(|p| p.keys().ne(last.keys())) {
        lines.push(format!("census ctx={ctx_id} MEMBERSHIP UNSETTLED over {SAMPLES} samples"));
    }

    for (&what, &(size, ref v)) in last {
        // What each line is about, spelled once: the rest of the line is the same question asked
        // of an allocation and of the blob published over one.
        let subject = match what {
            Read::Mem(mem_id) => format!("mem ctx={ctx_id} id={mem_id}"),
            Read::Blob(res_handle) => format!("blob ctx={ctx_id} res={res_handle}"),
        };
        let seen: BTreeSet<_> =
            samples.iter().filter_map(|p| p.get(&what)).map(|(_, h)| h).collect();
        if seen.len() > 1 {
            lines.push(format!("{subject} size={size} unstable values={}", seen.len()));
            continue;
        }
        match v {
            Err(rc) => lines.push(format!("{subject} size={size} UNREADABLE rc={rc}")),
            Ok(h) => {
                let want = size.min(READ_CAP as u64);
                lines.push(format!("{subject} size={size} read={want} hash={h:016x}"))
            }
        }
    }
    lines.sort();
    lines
}

/// The score: every fact a second implementation replaying the same corpus must reproduce, one
/// per line, in a fixed order so `diff` is the whole comparison tool.
///
/// It scores RENDERER STATE, not pixels. A VM-free replay has no scanout and presents no frames,
/// so what it can compare is what the commands built -- the accept/reject counts and the contents
/// of the device memory left behind. That is the right target here: a port gets object lifetimes,
/// descriptor writes and memory bindings wrong long before it gets a colour space wrong, and those
/// are exactly what device memory shows.
fn score_text(t: &Tally, census: &[String]) -> String {
    let mut out = String::new();
    out.push_str(&format!("prologue {} / {}\n", t.prologue_ok, t.prologue_ok + t.prologue_fail));
    out.push_str(&format!("cmds {} / {}\n", t.cmd_ok, t.cmd_ok + t.cmd_fail));
    out.push_str(&format!("ctl {} / {}\n", t.ctl_ok, t.ctl_ok + t.ctl_fail));
    out.push_str(&format!("skipped flow-control {}\n", t.skipped_flow_control));
    out.push_str(&format!("skipped classic {:?}\n", t.skipped_classic));
    out.push_str(&format!("unreplayable imports {:?}\n", t.unreplayable_imports));
    // Zero on a correct recorder: a parked create means the corpus needed reordering the execution
    // clock should already have done, so it belongs in the score rather than only in the log.
    out.push_str(&format!("parked {}\n", t.parked));
    for line in census {
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("vkr-replay: {e}");
            eprintln!(
                "usage: vkr-replay <corpus.vkrc> --renderer <lib> [--flags N] [--verbose]\n\
                 \x20               [--score <file>] [--expect <file>] [--smoke] [--rebuild]"
            );
            return ExitCode::FAILURE;
        }
    };

    let (t, census) = match run(&args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("vkr-replay: {e}");
            return ExitCode::FAILURE;
        }
    };

    if args.smoke {
        // A smoke run exercises a strict subset, so its output is not a score and must never be
        // pinned as one: --score/--expect are refused above rather than writing a golden that
        // silently means less than the one it replaces.
        let ok = t.failed() == 0;
        println!(
            "smoke: ctl {} / {} ({} exporting blobs skipped) -- init, contexts and resources {}",
            t.ctl_ok,
            t.ctl_ok + t.ctl_fail,
            t.smoke_skipped_exports,
            if ok { "OK" } else { "FAILED" }
        );
        return if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }

    let score = score_text(&t, &census);
    print!("{score}");

    let mut ok = t.failed() == 0;

    if let Some(path) = &args.score {
        if let Err(e) = std::fs::write(path, &score) {
            eprintln!("vkr-replay: writing {path}: {e}");
            ok = false;
        } else {
            eprintln!("score written to {path}");
        }
    }

    if let Some(path) = &args.expect {
        match std::fs::read_to_string(path) {
            Ok(want) if want == score => eprintln!("score matches {path}"),
            Ok(want) => {
                // Say WHICH lines moved. A golden that only reports "differs" makes the reader
                // re-run by hand to find out what changed, every time.
                eprintln!("SCORE DIFFERS from {path}:");
                let (a, b): (Vec<&str>, Vec<&str>) =
                    (want.lines().collect(), score.lines().collect());
                for i in 0..a.len().max(b.len()) {
                    let (x, y) = (a.get(i).copied(), b.get(i).copied());
                    if x != y {
                        eprintln!("  - {}", x.unwrap_or("<missing>"));
                        eprintln!("  + {}", y.unwrap_or("<missing>"));
                    }
                }
                ok = false;
            }
            Err(e) => {
                eprintln!("vkr-replay: reading {path}: {e}");
                ok = false;
            }
        }
    }

    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
