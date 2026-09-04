# Rewriting virglrenderer in Rust

The host renderer is the last large C dependency limina owns, and it is where our
crashes live. This plan takes it to Rust in one multi-session pass, with no
requirement that intermediate states boot, and no obligation to older users.

The destination — **virglrs** — is a Rust `cdylib` that exports the same C ABI as
`libvirglrenderer.1.dylib` and installs the same `virglrenderer.pc` layout, so it is
swapped in by pointing `VIRGL_PREFIX` at it. libkrun, rutabaga, `build-app.sh` and
`check-virgl-link.sh` are untouched by the rewrite. Deleting the FFI in favour of a
native crate is a follow-up, after the Rust renderer is the only implementation.

The work is contained in this repository — plan, harness and the Rust tree — and
reconciled with limina (vendoring, manifest pin, `build-virglrenderer.sh`) once it
has something to vendor. This repository is a **hard fork**: nothing here needs to
keep the C implementation viable past the switch, and the C tree's only remaining job
is to be the reference the harness records goldens from.

## Point A, measured

Only what our build actually compiles counts. `-Dvenus=true -Dvideo=true
-Dplatforms=egl -Drender-server-mode=thread -Drender-server-worker=thread
-Dvulkan-dload=false` on macOS/aarch64 produces 88 objects:

| Subsystem | Hand-written C | What it is |
|---|---|---|
| `src/vrend/` | 32.4k | GL renderer: state machine, TGSI→GLSL, decode loop, blitter, formats, EGL winsys, IOSurface scanout |
| `src/venus/` | 18.1k + 0.8k ObjC | vkr: Vulkan passthrough, rings, object tables, memory, budget, journal, Metal/IOSurface helpers |
| `src/gallium/` + `src/mesa/util` | ~11k | TGSI parser/scanner (7.1k), `u_format`, `ralloc`, hash tables, `cso_cache` |
| `src/vrend/virgl_video_*` (ours) | 4.1k | VideoToolbox backend, AV1 OBU synthesizer, H.264 parameter-set serializer, dav1d bridge |
| `src/` core | 3.2k | The ABI, the resource table, contexts, fences |
| `server/` + `src/proxy/` | 5.0k | Render-server split — in thread mode, a socketpair between threads of one process |
| **Total** | **~74k** | |
| `subprojects/venus-protocol` | ~78k **generated** | 40 headers emitted from 3.1k Python + 2.5k mako templates over `vk.xml` |

Our fork's delta against upstream is 158 files, +19.7k lines (the −78.8k is
un-vendoring venus-protocol), across 172 commits.

## The contract Limina depends on

The C dylib exports 69 `virgl_*` symbols; libkrun references 62 of them (`nm -gU`
on the dylib intersected with the symbols libkrun names). The harness pins all 69 —
what this tree exports is ours to define, while who calls what moves without warning —
but these are the ones with known callers:

- The classic surface bound by `rutabaga_gfx/src/generated/virgl_renderer_bindings.rs`
  — init, cleanup, contexts, fences, resources (create/blob/import/export/map/unmap/
  iov), transfers, `submit_cmd`, caps, poll — including the IOSurface family
  (`resource_get_iosurface_id`, `resource_sync_iosurface`, `resource_read_iosurface`,
  `republish_iosurface`, `resource_get_map_ptr`) that carries the zero-copy present
  and the Mach-port publish.
- The `virgl_renderer_limina_*` family, hand-declared in
  `rutabaga_gfx/src/virgl_renderer.rs` and called from
  `devices/src/virtio/gpu/journal.rs`: `journal_export/seq/unpin`,
  `replay_begin/submit/ring_cmd/end`, `memory_census/read/write`,
  `sync_export/restore`, `classic_content_export/restore`, `dump_state`.

**The ABI is layouts, not only symbols.** libkrun's bindgen pins the layout of
`virgl_renderer_callbacks` (its version negotiation and `write_context_fence`),
`virgl_renderer_resource_create_args`, and the blob/import arg structs. A layout
mismatch compiles clean and corrupts at runtime, and no existing test would catch
it — so P0 pins the symbol list *and* struct layouts as harness fixtures the Rust
build is diffed against.

Behavioural contract, not just symbols: `RENDER_SERVER | THREAD_SYNC |
ASYNC_FENCE_CB` must retire venus fences **asynchronously** through
`write_context_fence`, or the guest hangs in `vkQueueWaitIdle`. `virgl_renderer_init`
must accept the flag word libkrun passes (`VENUS | USE_EGL | USE_GLES |
USE_SURFACELESS | THREAD_SYNC | ASYNC_FENCE_CB | RENDER_SERVER | USE_VIDEO`) and
advertise exactly the capsets those flags imply.

## What we delete rather than port

Not compiled today, or not needed once the C fork is gone:

- `src/drm/` native contexts, GLX winsys, GBM/minigbm, `mman_win32`, `vtest/`,
  `perf-testing/`, the Linux `virgl_video.c` libva backend.
- **The process-mode render server.** In thread mode, `proxy/` + `server/` (5k
  lines) marshal commands over a socketpair between threads of the same process for
  no reason but upstream's fork model. Rust accepts `RENDER_SERVER` at the ABI and
  calls vkr directly. This is the largest pure deletion in the plan.
- **epoxy.** Bind GLES entry points directly, `dlopen`ing the zink-on-KK
  `libGLESv2` **by absolute path** — the hardened runtime strips `DYLD_*`, which is
  the same trap that forced `vulkan-dload=false`. The Vulkan library is linked, not
  dlopened, for the same reason.
- Upstream's tracing backends (percetto/perfetto/sysprof); keep `stderr`.

## IOSurface is the present path, and it has rules

There is no dma-buf on macOS; an IOSurface is the currency, and both renderers are
built on it. vrend renders *into* the display surface through an `EGL_IOSURFACE_LIMINA`
EGLImage — the framebuffer's storage *is* the surface, no readback, no blit — and venus
presents via `SET_SCANOUT_BLOB`, importing the guest image as an `MTLTexture` over the
same surface. Ids cross to the supervisor as Mach ports, not as global ids. The
invariants a port owes, none of which the C encodes as a type:

- **An id is worth nothing after its surface dies.** `get_iosurface_id` returns 0 for
  "not backed", and must return 0 the instant the backing is freed; ids are recycled
  immediately. A cached id names a stranger's surface, and releasing it frees *theirs*
  irrecoverably — a non-global surface can only be re-minted by its creator. Nothing
  may persist an id, across a call or across a snapshot restore.
- **`sync_iosurface` is classic-only.** It is a blit-and-wait the VMM issues on
  `RESOURCE_FLUSH` for ctx 0. A venus blob renders into its surface directly and must
  never be synced.
- **`read_iosurface` writes top-down BGRA and its stride is in BYTES.** Passing a pixel
  width yields a quarter-width image tiled four across and squashed four down.
- **`republish_iosurface` is keyed by id, not by resource** — the resource may be gone
  while the surface lives — and the registry is process-global and mutex-guarded, so it
  must be safe off the renderer thread.
- **`get_map_ptr` is called eagerly at blob create** and its pointer feeds `hv_vm_map`,
  so it must stay valid for the resource's whole life. `resource_unmap` is called
  unconditionally at unref and must return a harmless `-EINVAL` when nothing was mapped.
- **Scanout stride must be GPU-row-aligned.** `IOSurfaceCreate` accepts a tight
  `width*4`, and CoreAnimation then composites blank.

## Ranked by difficulty

1. **The venus wire decoder (78k generated lines).** Decision: **fork
   venus-protocol's generator to emit Rust.** Only `vkxml.py` (the vk.xml model) is
   language-neutral. `vn_protocol.py`'s `Gen` class emits C *statements* — the
   `VariableInfo` machinery and `_sizeof/_encode/_decode_variable` are as much of the
   backend as `templates/` is — so the fork is emitter plus template, not templates
   alone: 2.6k lines of Python emitting 193k lines of Rust. The fork lives in this tree
   (`virglrs/venus-gen/`) and imports the subproject's model over `sys.path`: the
   subproject is wrap-managed and any edit inside it is eaten by the next re-clone.
   The generated decode is *safe* Rust — bounds-checked slices over guest bytes,
   poison instead of panic — with the raw Vulkan structs it fills handed to the
   bindings module, which is where the unsafe already lives. This is the
   highest-leverage item in the rewrite and the crash-prone code the user wants gone.
   The differential test is a round trip, not a C dump: the recorded ring bytes *are*
   C-encoder output from the guest's mesa driver, so Rust-decode → Rust-re-encode →
   byte-compare against the original wire diffs the two implementations over every
   recorded command for free. It does not cover reply encoding, and **nothing else
   does either**: both replay entry points call `vkr_replay_strip_reply`, so a replay
   never encodes a reply, and the state score measures what commands did rather than
   what was said back. That is what `venus-reply-oracle` covers: the same command
   struct handed to both the generated Rust reply encoder and venus-protocol's own
   generated C renderer encoder, and the bytes compared. One struct, two encoders —
   `vn_command_*` is `#[repr(C)]`, so the C reads the memory Rust filled rather than a
   second construction of it. It reaches the 326 per-command reply wrappers, which the
   round trip never touches, and its ground truth is what every venus guest in
   existence decodes.
2. **`vrend_shader.c` (8.6k).** TGSI→GLSL with variant keys. No crate exists; a
   direct port. Mechanical but unforgiving — the shader key logic is where subtle
   divergence hides, and it is exercised by every draw.
3. **`vrend_renderer.c` (15.7k).** The GL state machine. Large, but flat: mostly
   independent command handlers over shared state. Good subagent-parallel material
   once the state types exist.
4. **vkr object lifetime and rings (`vkr_context.c`, `vkr_ring.c`,
   `vkr_device_memory.c`, `vkr_image.c`).** The semantics are subtle (cross-context
   shares, ghost containment, poisoned replay, budget accounting) but this is
   exactly where Rust's ownership model pays; our own ghost/poison containment
   commits are C workarounds for problems Rust makes structural. One constraint the
   C encodes only by accident: a handler publishes its reply into guest-visible
   memory *partway through* ring submit, so the guest can act on a reply before the
   submit call returns. That visibility point is observable and must be reproduced —
   see the ordering rule in `src/venus/vkr_record.h`. It is also the one place a
   process-global ordering counter is not a design failure.
   This is also where the guest stops being able to kill the VM. Mesa's Vulkan runtime
   and KosmicKrisp carry ~820 `assert()`s, many on values a guest controls, and an
   assert on the vkr ring thread aborts the whole worker — limina compiles them out
   with `-Db_ndebug=true` because it has no better lever. The lever belongs here:
   reject at the boundary and poison the context. Four cases have already killed a VM
   and are the first validation rules and harness cases: degenerate
   `vkCmdClearAttachments` rects, `vkCreateBuffer` with `size == 0`, a render-pass
   format mismatch, and an attachment-less pass with `defaultRasterSampleCount == 0`.
5. **The snapshot family** — journal export, sync export/restore, classic content
   export/restore. Subtlest behaviour, smallest code, and it has no meaning until
   the thing it journals works. Its *siblings* do not wait: the replay feed
   (`replay_begin/submit/ring_cmd/end`) and `memory_census`/`memory_read` are what
   the venus harness drives, so they are P2 infrastructure, not P5 work.
6. **The VideoToolbox backend + AV1/H.264 bitstream synthesis (4.1k).** Ours
   already, well understood, and it maps cleanly onto `objc2` +
   `objc2-video-toolbox` + `objc2-io-surface`. `dav1d` → `rav1d` (the Rust port).

Crates that carry weight: `objc2` family (IOSurface, Metal, VideoToolbox, Mach
ports), `rav1d`. `u_format`'s table is already generated from Mesa's XML — port that
generator to emit Rust alongside the venus one.

**Vulkan is reached through a generated proc table, not `ash`.** The decoder fills our
own `#[repr(C)]` structs from the pinned vk.xml; ash's are different Rust types for
the same layout, so using it means either generating a field-by-field conversion for
~200 struct types or transmuting on a layout equality nothing checks — the exact class
of bug this rewrite exists to delete. The C already does the right thing: decoded
struct pointer straight to the ICD through a resolved proc pointer. We generate that
table from the same vk.xml, which also covers the private MESA commands ash has never
seen. Layout parity gets pinned the way the capset is, with `cc` and `offsetof`
against `subprojects/venus-protocol-1.0/include/vulkan/vulkan.h` — the header matching
the pinned vk.xml. Skew is bounded: Vulkan extends through `pNext` and never adds
fields to an existing struct.

**The Khronos loader is linked, not dlopened, and it picks the driver.** KosmicKrisp
exports only `vk_icdGetInstanceProcAddr` and `vk_icdNegotiateLoaderICDInterfaceVersion`
— it is an ICD, and talking to it directly would mean implementing a loader. Driver
selection is therefore the loader's `VK_DRIVER_FILES`/`VK_ICD_FILENAMES`, which is what
limina already sets; virglrs hardcodes no driver. The loader is *linked* rather than
dlopened for the reason limina's C build passes `-Dvulkan-dload=false`: the worker is
codesigned with a hardened runtime, which strips `DYLD_*`, so a bare-name runtime
dlopen resolves nothing and venus silently enumerates zero GPUs. Going through the
loader also buys the validation layers and vulkan-tools' mock ICD, which keeps the
GPU-free replay gate alive once real handlers exist.

## Why vrend is pass-1 scope

There is no video-only subset of vrend. mesa's VA post-processing draws go through
the **general** draw path — shader-key construction, TGSI→GLSL variant compilation,
constant-buffer binding in both the inline and resource-backed forms, sampler-view
target lowering, FBOs (`d1fdc034`). Video therefore requires the whole GL renderer,
and a partial one would need an arbitrated boundary plus software-2D kept wired as a
stopgap for stock guests. vrend lands in pass 1, in full.

## The harness

Built first, validated against the C implementation, baseline recorded, then held
constant across the rewrite. The design rule that makes it survive A→B: **it drives
only the public ABI.** Three layers.

### Layer 1 — VM-level replay (mostly exists; extend the corpus)

limina's `crates/limina-test/tests/venus_replay.rs` already replays an apitrace capture of
the real seated gnome-shell and a gfxreconstruct capture of native Vulkan, comparing
pixels against an llvmpipe/lavapipe reference — pixel-exact today. Alongside it sit
`venus*.rs`, `virgl*.rs`, `vkr_*.rs`, `vrend_session_restore.rs`,
`l2_video_vaapi.rs`, `l2_stock_vulkan_window.rs`, `scanout_churn_retention.rs`.
These are implementation-agnostic already. Work here is corpus, not framework, and
each corpus is owed by the phase it gates — not collected up front:

- **P2** — a seated compositor driving Vulkan clients (mutter or synoik with real
  clients), which is the only workload that produces scanout blobs in quantity. The
  present venus corpora carry nine surfaces between them, in two formats and two
  extents. That was enough to find that a scanout is recognised by shape rather than by
  a flag, and not enough to pin the shapes a real desktop produces -- multi-plane, tiled,
  or an extent whose driver pitch IOSurface will not take.
- **P3** — classic-vrend capture on a **stock** guest (apitrace over virgl, not
  zink→venus); today's Layer 1 corpus only exercises the venus path. GL clients and
  hardware video belong to this phase and not before it: the replayer skips classic
  contexts wholesale, and `USE_VIDEO` is vrend's flag — venus carries no video
  commands at all, so capturing either today grows a skip tally and nothing else.
- **P4** — a video clip per codec × path (H.264, HEVC, VP9 hardware; AV1 software),
  each with a VPP conversion leg, since that is what `d1fdc034` shows is fragile.
- **P5** — a suspend/resume cycle taken **mid-workload**, not from idle.

### Layer 2 — host-side, VM-free replay (the real new work)

A Rust crate that loads
`libvirglrenderer.dylib` — either implementation, selected by prefix — and replays
pinned corpora through the public ABI, with no VM, no guest, no HVF. This is what
makes the rewrite testable at subagent speed instead of boot speed.

- **Classic corpus**: the `vrend_trace` format we already record — submits, decoded
  commands with full payloads, resource create/blob/unref events in a never-evicted
  side store, transfer bytes. `harness/replay/vrend-replay.c`
  is the working seed; port it to Rust and make it a library.
- **Venus corpus**: the `limina_journal_export` → `replay_begin/submit/ring_cmd/end`
  path, which already solves handle remapping.
- **The recorder's vocabulary bounds the corpus, and it is missing the mapping
  calls.** `resource_map`, `get_map_ptr`, `get_map_info` and `unmap` are how a VMM
  actually collects a blob, and no `Ctl` kind names any of them — so a replay cannot
  call them however rich the guest workload is, and a stub behind them scores clean.
  The hooks belong beside `vkr_record_create_blob` in `virglrenderer.c`. Golden the
  return code and `map_info`, never the address: it is ASLR-fresh every run. New
  record kinds need a format version bump or a replayer that skips what it does not
  know, so the pinned corpora stay valid.
- **Oracle**: plain-text scores, diffed against fixtures pinned from the C build.
  Classic scores a content hash and an ink count per offscreen via `transfer_read_iov`;
  venus scores renderer *state* — accept counts plus the contents of the device memory
  the commands left behind, read through `limina_memory_census`/`memory_read`. A
  zero-allocation census at every context destroy is the venus leak oracle.
- **The IOSurface leg.** Classic scores its scanout surfaces at end of stream —
  `sync_iosurface` then `read_iosurface`, hashed like any other readback. That is a
  different path from `transfer_read_iov`: the scanout is an `EGL_IOSURFACE_LIMINA`
  EGLImage, so the surface *is* the framebuffer's storage, and a port can get one
  right while getting the other wrong. Venus scores how many blobs came back backed.
  `republish_iosurface` stays Layer 1 — it answers over a Mach port to a supervisor
  that a VM-free replay does not have.

### Layer 3 — fuzz corpora and the perf ledger

`tests/test_virgl_*` are **not** carried over. They are not ABI-level: every one of
them includes internal headers and links the static library, so none can run against
a Rust dylib, and they do not build on this platform anyway. Rewriting them
public-ABI-only would duplicate Layer 2 at the same speed. They stay as C-side
regression tests with no role in the rewrite.

What is carried: `tests/fuzzer/` corpora move to `cargo-fuzz` once the Rust decode
paths exist — corpora are data and survive the language change. And the perf bench
stays a **trend ledger, never a gate** — a rewrite regresses performance invisibly,
and gating on it stops work for the wrong reason.

## Phases

Each phase ends with the harness green against the phase's scope; the C build stays
buildable throughout as the A-side reference.

- **P0 — Harness.** Done. Both replayers run against the C dylib and score into
  pinned fixtures; the ABI's exported symbols and struct layouts are pinned too;
  the C tree is tagged `virgl-pre-rewrite-2026-08-31`. See `harness/README.md`.
  The corpora the replayers read are not in git and have no permanent home yet.
- **P1 — Skeleton.** The virglrs tree scaffolded in this repository, producing a
  dylib and a prefix layout interchangeable with the C build's. All 69 symbols
  exported and stubbed; the ABI types, resource table, context table, fence tracking,
  and async fence retirement implemented for real. Vendored into limina the way this
  tree already is — pinned in `third_party/manifest.toml`, with
  `build-virglrenderer.sh` grown a `VIRGL_IMPL=rust` leg producing the same prefix
  layout — so the switch is a manifest edit, not a build-system change. Gate: `abi/abi-fixture.sh` green
  with `VIRGL_PREFIX` pointed at the Rust build, and both replayers loading that dylib
  and getting through init, context create and resource create without error. All of
  it VM-free — a phase whose point is going fast does not gate on a boot.
- **P2 — venus.** Fork venus-protocol's generator to emit Rust; gate the decoder on a
  byte-identical wire round trip over both corpora. Then vkr: instance/device/queue/memory/
  image/buffer/descriptor/command-buffer, rings, budget, the Metal + IOSurface
  helpers. An output blob — `vkGetPipelineCacheData`, `vkGetQueryPoolResults` and the six
  others venus-protocol marks `need_blob_encode` — is room the guest offers as a count with no
  bytes behind it; the decoder allocates that room in the arena, bounded like every other
  array, and the reply copies what the handler wrote. The C writes into the reply buffer in
  place instead, which saves one copy of a pipeline cache and couples the decoder to an
  encoder that does not exist yet when it runs; the copy is the price of keeping them apart.
  Two pieces of the budget land after the ledger itself: `VK_EXT_memory_budget`, which is the
  only backpressure that reaches a guest at all and needs
  `vkGetPhysicalDeviceMemoryProperties2` intercepted rather than forwarded; and the HostShm
  blob carrier in `renderer.rs`, which is a second host allocator the C charges and this tree
  does not yet.
  Midpoint gate, before any VM: both corpora replay to completion, their scores match
  the fixtures pinned from the C build, **and every handler whose contents matter
  carries its own witness**. Replay strips replies and needs no display, so score
  parity is reachable with handlers alone, and it localises a failure where a boot only
  reports that something is broken. What it cannot do is check what a handler passed
  on: it measures that a command was *accounted for*, and a build whose every array
  accessor hands its handler one element fewer than the guest sent replays both corpora
  clean, census unchanged (`harness/README.md`, measured). Score parity alone is
  therefore not the midpoint. Ends at a seated venus
  GNOME desktop, booted with the existing venus-only
  `virgl_override` limina already has for forcing venus-only flags — no new
  machinery, and no classic stubs that have to lie about capsets. Carries the replay
  feed and `memory_census`/`memory_read` with it, because those are what the venus
  harness drives. This is where the crash pain is; it goes first among the renderers.
  Recording scanout geometry beside the ring stream lands here too — it is what turns
  the venus IOSurface score from a count into a frame hash, and a zero-copy blob has no
  other CPU-readable copy of its pixels.
  The ring transport is served: all ten of the C's transport commands
  (`src/venus/vkr_transport.c`). None of the ten is reachable by
  replay (`harness/README.md`), which is why the seated boot is a gate and not a
  formality: a build missing all of them scores every corpus clean and puts nothing on
  the screen. What the compositor does varies — it has both exited at startup and stayed
  running with every systemd field healthy — and neither presents a frame, so the state
  of the process is not the measurement. The frame is (`harness/vm/frame.py`).
  **synoik seats on virglrs**: a wallpaper, a top bar and a clock, 59,053 distinct
  colours against the C leg's 58,808 on the same guest, same dominant colour on the same
  0.77% of the frame.

  Three things stood between a served transport and that frame, and none of them was in
  a corpus.
  The Metal path emulates `VK_KHR_external_memory_fd` and
  `VK_EXT_external_memory_dma_buf`, and a driver that has neither must still be told it
  has both — mesa gates its renderer handle type on the second, so advertising only what
  the driver holds leaves the guest's device four extensions short and its compositor
  unable to add the primary node.
  `vkGetMemoryResourcePropertiesMESA` and `vkAllocateMemory` resolved a resource two
  different ways, so the query refused a scanout buffer the allocation right behind it
  would have imported; there is one resolution now (`ShmResources::bytes`,
  `Driver::span`), which is the shape the C's own comment says it learned the hard way.
  And `virgl_renderer_resource_read_iosurface` was a stub: a venus scanout blob has no
  CPU transfer path, so the surface's shared storage is the only place the frame exists
  and a headless boot without that read captures the boot console.

  One thing outside this tree stood there too. synoik took a gbm device on the primary
  node as mandatory, and gbm needs a gallium driver — which a Vulkan-only host renderer
  does not provide. It is the *cursor*-plane allocator and nothing else; scanout goes
  through the renderer's own Vulkan device and a PRIME import. The C leg passed only
  because it also serves classic virgl. Made optional guest-side.

  **A transport wait suspends the batch; it never blocks a handler.** The C sleeps inside
  the handler — a ring thread in its own dispatch, a ring-seqno wait on the virtio-gpu
  control queue thread — and neither ports, because here a handler runs with the context
  locked and, on the ABI path, the one global renderer mutex behind that. Sleeping would
  hold both against the thread whose progress is being waited for, and `RingThread::stop`
  — called from `vkDestroyRingMESA`, itself inside that lock — would join a thread that
  never reaches its stop check. So `Submitted::Waiting` says how much of the batch ran and
  what to wait for; the caller waits with nothing held and returns with the remainder. The
  wait command is deliberately not consumed, so the resume re-decodes it, which is what
  makes a reply-carrying wait truthful with no special case: the answer is encoded on the
  pass that proceeds.

  **A recorded command stream is copied out one at a time, and a wait inside one is
  refused.** `vkExecuteCommandStreamsMESA` is how every recorded `vkCmd*` arrives: mesa
  fills a resource and names it rather than sending the commands inline. `streamCount` is
  bounded only by what fits in a batch and each descriptor may name a whole resource, so
  the descriptors are recorded and the bytes copied per stream, after the bounds check —
  peak cost is the largest single stream. The C points its one decoder into guest memory
  and saves and restores its state around the nested run; each level here builds its own
  decoder, arena and reply scratch, so the outer decode is untouched by construction.

  A transport wait inside an executed stream is refused, which the C allows. A suspension
  unwinds to `ffi.rs` carrying a position in the *outer* stream, and that position cannot
  name a byte of the copy the wait came from. Mesa records `vkCmd*` work into these
  streams and never transport waits — a hypothesis, held by a poison that names the
  command if a boot ever proves it wrong.

  Three facts about the transport, established against the C and the guest mesa rather
  than inferred, because each one changes a design:

  *The dispatch origin has three classes, not two.* Context-only (`vkCreateRingMESA`,
  `vkDestroyRingMESA`, `vkNotifyRingMESA`, `vkWriteRingExtraMESA`,
  `vkSubmitVirtqueueSeqnoMESA`, `vkWaitRingSeqnoMESA` — they reach the context's ring
  table or its pending wait); ring-only (`vkWaitVirtqueueSeqnoMESA` — it blocks the ring
  it was found from); and either (the reply-stream pair, `vkExecuteCommandStreamsMESA`,
  and every ordinary Vulkan command, each acting on whichever dispatch it arrived on).
  A two-state typestate would therefore be a lie. The shape that stays true is a
  `Dispatch` trait both origins implement, with the context-only class bounded by a
  capability the ring type lacks and the ring-only class by one the context lacks — and
  the generator emitting the bound per command from a three-valued attribute. Commands
  nested inside an execute inherit the outer origin.

  *The guest aborts if the host stops stamping a liveness bit.* The guest's wait loop
  clears `VK_RING_STATUS_ALIVE_BIT` on the instance ring, and at the first warn
  iteration — around three and a half seconds of accumulated sleep, reached by any
  fence, semaphore or seqno wait — re-reads it and calls `abort()` if the host has not
  set it again. It kills the guest; it never hangs it. A cold shader cache or a first
  frame crosses that threshold routinely. The bit answers "is the renderer still
  scheduled", not "is this ring advancing", which is what the fatal bit is for — so
  `venus/monitor.rs` stamps from a thread that shares no lock with dispatch, and a
  registry of `Weak<RingStatus>` is what makes a destroyed ring un-register itself.
  Stamping happens at a third of the requested period above 300ms: the period is an
  upper bound, and the C tree records the exact-period stamp arriving 489ms late on an
  idle host even with the thread's QoS pinned. A `maxReportingPeriodMicroseconds` of
  zero is a guest error and is refused, never defaulted.

  *A guest can deadlock the whole device, and the C does not stop it.* The C's guard only
  fires in the ring thread's idle branch, so it misses this: a guest sends
  `vkWaitVirtqueueSeqnoMESA` on a ring with no submit behind it, blocking that ring, then
  `vkWaitRingSeqnoMESA` for a head the blocked ring can no longer advance. The only
  producer of the virtqueue seqno is the stream now waiting. Nothing times out, and the
  control queue is one queue for the whole device, so every context's submits, every
  scanout flush and every fence stop with it. A buggy guest reaches it as easily as a
  hostile one. The fix here is structural, not a timeout: a ring publishes what virtqueue
  seqno it is blocked on before it sleeps and wakes the waiter, and the waiter checks —
  before its first sleep and on every wake — whether it is itself the only thing that
  could release the ring. It poisons by name. "Blocked" and "blocked on something it
  cannot get" are read as one question under one lock: asked separately, a waiter can land
  in the gap where a ring has been released but not yet woken, and poison a context that
  was about to proceed. The C's tail-too-short guard is ported alongside it, asked from
  the waiting side, where the head advancing to meet the tail is itself the wake that
  triggers the re-check.

  They are also why an unserved command poisons the context whether or not it carries a
  reply. A command whose only product is its answer costs the guest one command when it
  is dropped; a command whose product is a *side effect* costs it a lie, and every seqno
  wait is exactly that — the guest is not waiting on a reply, it is waiting on the host
  to have blocked. The symptom of the dropped wait is a generic
  `VK_ERROR_OUT_OF_HOST_MEMORY` several commands later, out of a driver that refused
  nothing, which is as far from the cause as a report can land. The reply flag says who
  is blocked, never whether the command mattered. Upstream reaches the same rule from
  the other end: its generated wrapper for a handler-less command sets fatal before it
  decodes, and reads no flag. `vkCmdCopyImageToBuffer` is the ordinary kind of gap,
  merely absent from the corpora we had; `synoik-vkcube` is the corpus that carries it.

  **What venus still owes, carried into P3 rather than blocking it.** Each is a shape
  `CLAUDE.md` names, present in code that works, listed so vrend does not copy it and so
  the next venus pass starts here. Two vrend-facing items head the list because vrend
  consumes them on the first seated GL desktop.
  - *The cross-import, both directions.* A Vulkan client under a GL compositor is
    venus→vrend: `Storage::Texture` adopts the IOSurface as an EGLImage, zero-copy;
    `Storage::Linear` uploads into a placeholder texture, re-read per batch, zeroed on
    failure, and never `EINVAL` on a non-dmabuf fd type (that poisons the compositor's
    context). A GL compositor's buffer reaching a venus client is vrend→venus, the
    fd-less IOSurface attach of a classic resource. `Storage` is the currency for both;
    there is deliberately no venus-side presentation of `Linear` pages, whose consumer
    is vrend.
  - *Facts keyed by host handle are a second container.* `Driver::images` and
    `Driver::query_pools` each hold facts the object table already vouches for, keyed by
    a handle the driver may recycle, dropped at two destroy sites with a per-kind match
    in `empty_device` that a third recorded kind must extend. Safe today by discipline:
    every recorded object enters through its create and leaves through the table. The
    structural shape is one `records` map keyed by `HostHandle`, removed by
    `destroy_object` and by every doomed handle at teardown with no type match.
  - *A handler's refusal is still a second channel.* `vn_dispatch_command` returns one
    `Dispatched` verdict, and `Handlers::reject` beside it is what the handler could not
    say through it. The shape is `Dispatched::Refused(why)`, with the generator asking
    the handler for its verdict after the call.
  - *`NoSurface::NotExported` is representable in pages that can never carry it*, and
    `NoSurface::Layout` folds six distinct failures. `scanout_surface` wants
    `Option<Result<Surface, NoSurface>>` -- `None` for "not a scanout question" -- and
    a reason per question.
  - *`Storage::first_refusal` is a latch shaped like a predicate*, ordered after
    `surface()` by convention; `read_scanout_rows` is a second asker that never says why.
    One call, returning the refusal with its unsaid reason.
  - *`Decoder::verdict()` is called for its side effect* in the two oracle generators.
  - *`#[allow(clippy::too_many_arguments)]`* on the query read-backs stands where
    `first, count, stride, flags` is one value decoded once and measured once.
  - *Kept as designs, not built:* the bind-time surface link as a query over the image
    table (only zink-as-a-venus-client reaches it, and that configuration is dropped);
    the linear-vs-optimal tiling rule, adopted from the C for parity and unmeasured; the
    `-22` root cause, a diagnostic gap.
- **P3 — vrend.** TGSI parser, `u_format` generator, the GL state machine,
  TGSI→GLSL, blitter, EGL/GLES winsys, IOSurface scanout. Ends at accelerated GL for
  stock guests.
  The order is by what unblocks the next observable thing: decode, resources and
  formats, and enough of the state machine to CLEAR, BLIT, copy and transfer, scored
  against `vrend-nodraw.score` -- the classic corpus with every `DRAW_VBO` dropped,
  10,876 commands and 316 readback hashes that need no shader; then the IOSurface
  scanout and the first pixel gate, **kmscube on the stock image at
  `multi-user.target`** against the C's frame of the same; then TGSI and the shader
  translation against `vrend.score`; then the cross-import both ways (the seated GNOME
  with `vkcube --wsi wayland` is the frame with both halves of virglrs in it); video
  last.
  **What the host GL offers, measured 2026-09-03** with the replayer's environment
  (zink on KosmicKrisp through Mesa's EGL, surfaceless): OpenGL ES 3.1 (not 3.2), and
  desktop OpenGL 3.3 in both core and compatibility profiles (4.3 refused) carrying
  `ARB_compute_shader`, `ARB_shader_storage_buffer_object`,
  `ARB_shader_image_load_store`, `ARB_tessellation_shader`, and `ARB_gl_spirv` with one
  shader binary format. GLES has no SPIR-V path. limina's C build binds GLES
  (`VIRGL_RENDERER_USE_GLES` is in its init flags), so a SPIR-V shader target means a
  desktop-GL host context, which is the C's other winsys leg.
  **The host context is GLES 3.1 and the shader target is GLSL ES**, the C's gles leg,
  because that is the leg the goldens were recorded on: a differential against a
  different host profile would be comparing two renderers *and* two drivers. A
  desktop-GL context is its own gated change afterwards, one differential at a time.
  **Where P3 stands.** Decode, resources, transfers, the context layer and the IOSurface
  scanout are in: sub-contexts with a GL context each, object tables, every state
  command recorded, the immediate GL the C emits on a bind, framebuffer state, clears,
  copy-image and framebuffer blits, resource copies through them, queries and
  streamout. A scanout or shared 2D BGRA/RGBA resource is minted as an IOSurface at the
  pitch a linear Metal texture takes, adopted as the texture's storage through an
  `EGL_IOSURFACE_LIMINA` image (`egl::Image` owns the surface, so the id the VMM is
  handed is good exactly as long as the texture), and carries the C's rules for a
  resource that cannot be viewed: no texture view of an IOSurface-backed BGR* texture,
  the red/blue and sRGB conversions moved into the sampler swizzle, the clear colour,
  and the framebuffer bits the shader key reads. Shaders and draws are in: the TGSI
  parser, the translation to the GLSL the C emits byte for byte, the key each bound
  state makes and the variant chain it selects, program link with the C's attribute,
  fragment-data and transform-feedback bindings, `LINK_SHADER`, and `vrend_draw_vbo`
  with its bind of constants, UBOs, samplers, SSBOs, atomics, images, the sysval block
  and the vertex bindings. Both `vrend.score` and `vrend-nodraw.score` are a zero-line
  diff against the C, and the shader log is block-for-block identical to the C's for all
  34 translations. Two of the C's EGL-image rules are deliberately not
  carried: refusing `glCopyImageSubData` between two `B8G8R8X8` textures when one is an
  EGL image is a Mesa dmabuf quirk (a `GL_RGB8` import) the C's own macOS path says does
  not apply to an IOSurface, and the `LIMINA_VREND_*IOSURFACE*` environment switches are
  debugging aids with no reader here. Not served yet, each counted and named in the log
  when a stream asks: tessellation without a control shader (the C's injected TCS),
  advanced blend equations, a layered image bound as a subset of its levels or layers,
  the C's bridge of UBO 0 into the constant array, the shader blitter (a blit whose
  formats swizzle differently, or that swaps red and blue for an IOSurface-backed end),
  implicit-multisample surfaces, the resource-copy fallback through guest memory, blob
  resources, video. Next is the first pixel gate, kmscube against the C's frame.
- **P4 — video.** Decode command path, VideoToolbox backend via `objc2`, AV1 OBU
  synthesis, H.264 parameter sets, `rav1d`. Ends at hardware decode per codec plus
  the VPP legs.
- **P5 — snapshot.** Journal export, `memory_write`, sync export/restore, classic
  content export/restore. Ends at suspend/resume parity.
- **P6 — cutover.** Rust becomes the default prefix; limina's manifest and
  `build-virglrenderer.sh` are reconciled to it and the C tree is tagged and
  archived. Then the follow-up: delete rutabaga's FFI shim and depend on the crate
  directly.

## Consequences to accept

- **Upstream gets the fixes made up to the switch, and nothing after.** The delta
  this fork carries against upstream virglrenderer is still worth contributing, and
  that contribution is bounded at the point virglrs takes over. Past it, this tree
  diverges permanently and no change here is written with upstream in mind.
- **Every fix becomes ours to carry forever.** Nothing flows the other way either:
  upstream's future vrend and vkr work stops being something we can merge.
- The C tree stays buildable for the whole rewrite. It is the reference the harness
  records goldens from, and the fallback if a phase stalls.
- **Fork discipline still applies.** This tree is pinned by revision in limina's
  `third_party/manifest.toml`, so tag before any branch rewrite — every pinned rev has
  to stay reachable, and a rebase that orphans one breaks a build nobody is watching.
