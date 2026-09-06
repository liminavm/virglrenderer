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
   `objc2-video-toolbox` + `objc2-io-surface`. The C's dav1d fallback is not ported:
   see P4.

Crates that carry weight: `objc2` family (IOSurface, Metal, VideoToolbox, Mach
ports). `u_format`'s table is already generated from Mesa's XML — port that
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
  33 translations. The classic capsets are served: `virgl_caps_v1` and `v2`, probed once
  at init from the same features, limits and format table the renderer runs on, and
  pinned against the C's `vrend.caps` field for field. Two of the C's EGL-image rules are deliberately not
  carried: refusing `glCopyImageSubData` between two `B8G8R8X8` textures when one is an
  EGL image is a Mesa dmabuf quirk (a `GL_RGB8` import) the C's own macOS path says does
  not apply to an IOSurface, and the `LIMINA_VREND_*IOSURFACE*` environment switches are
  debugging aids with no reader here. Not served yet, each counted and named in the log
  when a stream asks: tessellation without a control shader (the C's injected TCS),
  advanced blend equations, a layered image bound as a subset of its levels or layers,
  the C's bridge of UBO 0 into the constant array,
  implicit-multisample surfaces, the resource-copy fallback through guest memory,
  video. The shader blitter is in, both paths: a blit whose ends disagree
  about their swizzle, that swaps red and blue for an IOSurface-backed end, or that has
  to convert a colourspace by hand, runs as a textured quad in the blitter's own shared
  GL context, and `blit.score` pins five such blits against the C. A blit whose two ends
  both carry depth runs the blitter's other shader, writing `gl_FragDepth` and hanging its
  destination off the depth attachment; `sampled.score` pins one against the C, scored by
  sampling the blitted depth into a colour target because the sweep reads neither depth nor
  anything but a plain 2D colour offscreen. Two branches of the blitter have no fixture and
  are faithful by reading only: its multisample resolves, and the depth shader's 1D and
  multisample spellings. The first pixel gate has passed: the stock guest's GNOME session,
  booted headless on virglrs and on the C and read from the presented frame, is
  bit-identical between the two but for the clock's digits. (kmscube was the planned
  workload; the stock guest autologs into GNOME, which holds DRM master, so the desktop
  itself is the workload.) A caution the gate taught: the Fedora logo the background
  extension draws is present or absent from one boot of the *same* renderer to the next,
  a guest-side race in the extension, so a frame diff has to be read against a second boot
  of the reference before a difference is charged to the port. That session, wallpaper
  included, is now the classic corpus: `vrend.bin` and the `vrend.score` pinned from it.

  **The cross-import is served, both halves of virglrs in one frame.** A `vkcube` window
  composited into the enhanced guest's GNOME overview, the client's pixels rendered through
  venus and sampled by a classic-virgl compositor through the IOSurface it exported, with no
  copy between them. Read from two samples with the cube at different angles, which is what
  separates a live desktop from the last frame a dead one left behind.

  **Blob resources were what P3 owed, and they were a blocker rather than a gap.**
  `PipeResourceSetType` is where a blob becomes a typed image: it carries the format, bind,
  extent, modifier and per-plane stride and offset that the blob's own creation does not. The
  decoder reads all of it; only the dispatch refuses, and the refusal poisons the context that
  asked. On the enhanced image that context is gnome-shell's, at session start and with no
  Vulkan client involved -- one refusal, then every submit for the rest of the boot fails and
  the desktop never presents again. The stock guest does not reach the command, which is why
  its pixel gate passes; both images render through vrend, so serving this has to leave the
  one that works untouched.

  It was also the whole of the cross-import gap: one consumer to build, not a pipeline. A
  handle now names a resource or storage nothing has typed yet, `SET_TYPE` is the upgrade that
  converts the second into the first, and it consumes what it converts, so nothing can type a
  handle and leave the old entry standing. The surface is adopted, never minted -- a surface
  minted at the upgrade would be a second copy of the frame the client is presenting, and the
  guest would composite the one nobody draws into.

  **The blank-texture case is reachable and unwanted, and filling it waits for a workload that
  needs it.** A share that is not a surface -- an export that is not a dedicated, linear image in
  a format IOSurface has -- gets a zeroed texture. No session produces one: a real GNOME desktop
  with a Vulkan client logs none, because the two external-memory extensions this tree injects
  are what steer mesa's WSI onto the dedicated-image path and away from prime-blit, whose
  staging buffer is exactly the linear-with-no-image shape that would land here. Forced (by
  making every share report itself surfaceless) it behaves as designed: `vkcube`'s window is
  black, the desktop stays live, nothing poisons and nothing aborts, and the reason is said once
  per storage rather than per frame. Black rather than garbage is the zeroing doing its job.

  So uploading the exporter's bytes into that texture is not the next thing to build. The C
  names a workload that would need it -- a software-decoded video frame reaching the GPU through
  a guest-memory blob's iovecs, which is every GStreamer `glupload` whose buffers qualify -- and
  that arrives with video, not before it. Building the upload now would be a copy path with no
  caller and no way to score it.

  Three outcomes there, each answering to whoever caused it, because conflating them is how a
  host bug ends up wearing a guest's clothes. Arguments describing no image are the guest's
  error and refuse its own context. A blob whose share is not a surface, or a host that adopts
  none, is nobody's error: a blank texture, zeroed because `glTexStorage` leaves contents
  undefined and undefined is another context's memory read as pixels. A driver that said it
  imports IOSurfaces and then refuses one aborts, because degrading there hides our own defect
  behind a window that merely renders something else.

  **Ported as the C has it, and owed a redesign.** What lands here is a workaround, adopted
  deliberately so the desktop runs; three things under it are wrong, and all three are this
  tree's own stated rules broken by the shape virgl hands us.

  *An import that copies is not an import.* The classic path models a shared buffer as a dmabuf
  the GL driver aliases. There is no dmabuf here, so a share that is not an IOSurface degrades
  to a copy refreshed before every batch that samples it -- the guest's pages and the GL texture
  are two containers holding one fact, reconciled on a schedule. That is the pair rule broken,
  and the per-batch re-read is exactly the layer that quietly repairs a mismatch. The IOSurface
  leg is a real alias; the other leg only looks like one.

  *A resource exists before anything knows what it is.* `PipeResourceSetType` retro-types a
  handle that was already created and attached, so there is a window in which a handle names an
  untyped thing. The window is protocol-inherent -- `SET_TYPE` arrives in the command stream and
  attach precedes it by ordering, which no amount of deleting `ffi.rs` changes -- so what is owed
  is a better representation of it, not its removal. See *existence precedes purpose* below.

  *A placeholder reports success for work it did not do.* When the adopt refuses, the fallback
  is a texture whose contents are wrong, and the guest is told the command succeeded. The C's
  own comment says as much. Zeroing keeps it from leaking another context's memory; it does not
  make the answer true.

  **Existence precedes purpose, and it does so in both protocols.** Neither wire says what a
  buffer is *for* at the moment it is created, and the two renderers are forced to opposite
  answers. venus must commit early: the guest is handed a host pointer at `vkAllocateMemory`, so
  where the bytes live is fixed then and can never move, and a scanout is recognised by shape --
  export plus dedicated image -- because nothing declares one. vrend must defer: a blob is
  attached by the VMM before any command stream can type it. `mint_surface` is the contrast that
  proves the point, committing at create because `BIND_SCANOUT` and `BIND_SHARED` are
  declarations rather than inferences. The linear-vs-optimal tiling rule belongs to the same
  family, committing to a CPU-addressable layout before knowing whether the image is ever
  presented.

  The mechanisms cannot be unified -- one protocol forces early, the other forces late. The
  representation can be, and the tree already converges on it: `Pages.why` latches a resolved
  negative with its reason and says it once. The rule that generalises, and the one a redesign
  should aim at: **when purpose arrives after existence, "not yet known" is a typed state with
  one owner, carrying what is known and why it is unresolved -- never a side table, never a
  silent guess, never success reported for wrong contents.**
- **P4 — video.** Hardware decode of H.264, HEVC, VP9 and AV1 through VideoToolbox.
  Ends with a video playing in the guest on virglrs, sampled twice and moving.

  **Ported from this tree, not rebased onto upstream.** The video code here is two things
  with different provenance. `virgl_video.c` is upstream's, and it is the VA-API layer.
  Everything that makes video work on this platform -- the VideoToolbox backend, the AV1 OBU
  synthesiser, the H.264 and HEVC parameter-set builders, the dav1d fallback -- was written
  here, and upstream has no VideoToolbox backend and no reason to grow one. A rebase would
  therefore import churn in the one file we are about to stop using and none of the ~4,600
  lines we are actually porting. If a bug is ever fixed upstream in `virgl_video.c`, the tool
  is a cherry-pick.

  **The descriptor is read, never rewritten.** The C translates every `ref[i]` in a picture
  descriptor from a guest buffer handle into a host buffer id before passing it on, and the
  VideoToolbox backend then reads none of them -- it keeps its own reference-picture buffer and
  parses the real bitstream. So the Rust leg decodes the handful of fields the *container* needs
  before a bitstream can be handed over, out of a fixed prefix, and carries no translation step
  and no host-side buffer ids at all.

  **Decode only.** `virgl_video_encode_bitstream` is a stub returning -1 and `fill_caps`
  advertises no encode entrypoint, so the guest cannot reach it. `EncodeBitstream` is refused
  and counted, like any other command this build does not serve, and the encode callbacks are
  not ported. Porting a dead seam would be porting the C's layering rather than its semantics.

  **What scores it.** Both legs call the same VideoToolbox on the same host, so a golden is
  exact -- not because these codecs are normatively bit-exact, though they are, but because
  even a quirky VT produces the same bytes twice. The decoded planes land in guest resources
  and leave by the ordinary readback path, so a replay corpus scores video through the sweep
  like anything else -- measured while pinning the first corpus: 240 of 243 decode targets carry
  pixels, a distinct hash per frame, three replays byte-identical. It cost one flag. The replay
  harness never passed `USE_VIDEO`, and that flag is what registers VP9 and AV1 as supplemental
  VideoToolbox decoders, without which every session create fails and the targets score empty
  while everything else looks clean.

  **There is one decoder, and it is VideoToolbox.** The C carries a dav1d fallback entered
  mid-stream for AV1 frames the hardware returns wrongly (super-resolution) and for a host with
  no AV1 silicon at all. Neither is ported. A superres frame is refused, because a refused frame
  is a lost frame while a delivered one is a wrong picture nothing reports; and a host without
  the silicon advertises no AV1, which leaves the stream on the guest's own dav1d -- better
  tested than ours, and the same place the C's stock tier leaves it. That also removes the one
  thing that could break the two legs' equivalence for a reason that is not a bug: a switch to
  a different decoder at a different unit.

  **VP9 is the only codec a stock guest can drive, and so it goes first.** Stock Fedora's
  mesa is built `-Dvideo-codecs=all_free` and its VA frontend enforces that
  driver-independently, so H.264 and HEVC cannot be reached from an unmodified guest at all.
  VP9 is also the codec that needs the least of us: the guest's slice buffers already hold a
  complete frame, so the backend concatenates, wraps and delivers, with no parameter sets to
  synthesize. That makes it the one slice that exercises the command path, the buffer
  lifetime and the delivery path without any bitstream work underneath -- which is exactly
  what should be proven first. H.264 and HEVC corpora need a guest mesa rebuilt with the
  codecs on; AV1 is free but wants M3-or-later silicon to decode in hardware.

  **Order, and why.** *A VP9 corpus first*: only the C leg can record it, the end gate needs
  it regardless, and it carries real `virgl_picture_desc` inputs -- synthetic descriptors
  exercise paths no player takes. *Then the whole VP9 leg*: caps, codec and buffer lifetime,
  command path, VT session, delivery. That is a scoring, end-to-end video renderer, and every
  later codec is bitstream work behind an interface it has already proven -- **both are done,
  and every later codec now needs only its bitstream work and its capset line.** *Then the
  builders*: `virgl_h264_build_parameter_sets`, `annexb_to_avcc`,
  `virgl_av1_build_temporal_unit` and their HEVC counterparts are pure bytes-in/bytes-out,
  roughly 2,600 lines with an exact oracle that needs no GL, no VT and no VM -- ordinary
  `cargo test` work against fixtures dumped from the C by a small generator linking the
  builder objects. *Then* H.264, HEVC, and AV1 last, being the only codec that must
  synthesize a header the guest's parser already destroyed.

  The `LIMINA_AV1_CAPTURE` dumps are not those fixtures: the C calls them an ABI-shaped dump
  for a spike built from the same tree, never a persisted format. Descriptors come from the
  wire corpus, which is one.

  **A decoded frame is delivered twice, and only the second half is negotiated.** VideoToolbox
  hands back a CVPixelBuffer, never a dmabuf. The unconditional half puts the pixels in the host
  GL texture, and that is all a guest that *samples* the target needs -- which is why VP9
  hardware decode works on a stock guest, byte-identical to the software decoder, with the
  writeback never firing once. The negotiated half writes each plane into the *guest's pages*
  too, announced by `VIRGL_CAP_V2_VIDEO_GUEST_PLANES`, and it exists for a guest that **exports**
  the target as a dmabuf: the fd has to name storage that actually holds the frame, and a
  one-page stub does not. Firefox's zero-copy import is the caller. A writeback that falls short
  is therefore a steady state and stays silent -- storage being big enough *is* the test for
  "the guest wants the frame here" -- so a log full of skips is not a fault to chase. A second
  bit,
  `VIRGL_CAP_V2_VIDEO_PLANAR_TARGET`, says the host will take a decode target as **one**
  resource in a planar format with its planes chained behind it, rather than one resource per
  plane -- backed by a planar IOSurface where it can be, and an RGBA conversion everywhere
  else. The two are separate because the writeback shipped first and a host can do it without
  accepting the composite shape.

  Both bits are protocol, so virglrs advertises them only when it serves them, and the
  samplable-format set has to agree with them exactly. Advertising a planar format the host
  cannot actually back is not a negotiation the guest can recover from: the kernel hands out
  the handle before we are asked, so a refused create is followed by an attach and sampler
  views on a resource that does not exist, and the context is poisoned for the rest of its
  life without the guest ever learning why. Only NV12 and NV21 can back a composite target, so
  only those two are advertised -- virglrs currently advertises IYUV and YV12 as well, which is
  a live deviation recorded in `harness/README.md` and closed by this phase.

  **`create_buffer` resolves its planes once.** It takes up to three guest resource handles,
  one per plane -- or one handle for the whole composite target, which is why the plane index
  has to travel with the delivery rather than being inferred from "one plane, one resource".
  The handles become shares at create, not handles re-looked-up at delivery, because the guest
  can free a plane mid-decode and a delivery path consulting a table the guest can empty is the
  lifetime bug this tree keeps refusing to write. Same shape as the resource work already
  landed.

  **One new unsafe module.** VideoToolbox, CoreMedia and CoreVideo are C APIs, so no `objc2`
  and no second Objective-C file -- `metal.rs` stays the only one. The new module goes on
  `CLAUDE.md`'s unsafe list, which is exhaustive on purpose.

  **Three gates, all of them already built.** The caps deviation the replay gate prints every
  run -- `num_video_caps`, `video_caps`, `capability_bits_v2`, and the planar half of
  `sampler` -- closes as video lands, and is a free regression line from the first commit. The
  corpus score is exact and pinned (`fixtures/vrend-vp9stock.score`). The end gate is a video
  playing in the guest, read under the liveness discipline: a playing video is its own moving
  element.

  **What a decode golden has to survive before it is pinned.** VP9 is normatively exact, so the
  hardware decode must agree byte for byte with `avdec_vp9` -- a verdict rather than a smell
  test, and one that catches "the pipeline ran and drew grey", which a buffer count does not.
  And it must agree with *itself*: this tree has already been bitten by a fault that fired two to
  four runs in twenty, which reads as "it never works" from a short streak and as "it works" from
  a lucky one. A golden taken from a single run of an intermittently faulting leg is worse than
  no golden, because the port is then graded against a bad frame. N runs, one hash.
- **P5 — snapshot.** Journal export, `memory_write`, sync export/restore, classic
  content export/restore. Ends at suspend/resume parity.
- **P6 — cutover.** Rust becomes the default prefix; limina's manifest and
  `build-virglrenderer.sh` are reconciled to it and the C tree is tagged and
  archived. Then the follow-up: delete rutabaga's FFI shim and depend on the crate
  directly.

## Where virglrs deliberately differs from the C

Parity with the C is the default and the harness measures it. These are the places we chose not
to have it, each because reproducing the C would mean reproducing a defect.

- **The count of bound vertex buffers belongs to the bind, not to the state-set.** The C keeps
  `num_vbos` beside `old_num_vbos`, written at `SET_VERTEX_BUFFERS`; two sets between draws leave
  the slots the first set bound still attached. `hw_num_vbos` is written by the bind that bound
  them.
- **The fourteen `ASTC_*_SRGB` formats are registered.** The C's `ASTC_FORMAT` macro never adds
  the sRGB rows, so its capset advertises fewer formats than its host has
  (`virglrs/vrend-gen/gl_formats.py`, and `harness/README.md` on `vrend.caps`).
- **The blitter reads the two sRGB features under their own names.** The C fills
  `has_srgb_write_control` from `feat_texture_srgb_decode` and `has_texture_srgb_decode` from
  `feat_srgb_write_control` — a transposition, and the first of the pair gates whether the blit
  enables `GL_FRAMEBUFFER_SRGB`. Unobservable on this host, where the pinned `blit.score` matches
  the C on an sRGB blit either way, and a host with one extension and not the other would diverge.
- **The sample-count ceiling is applied before the probe, not repaired after it.**
  `VREND_MAX_SAMPLES` exists in both trees for the same reason — a host whose multisample path is
  unsafe degrades instead of dying — but the C clamps `max_samples` after probing and then
  `memset`s all eight `sample_locations` words, which blanks the positions of the counts still
  under the ceiling too: a ceiling of 4 on a host with 8 leaves 2 and 4 selectable and describes
  neither. virglrs hands the ceiling to the probe, which already skips every count above what it
  is given, so the counts advertised and the positions published come from one pass and there is
  nothing to repair. Unobservable at limina's setting of 1, where nothing survives either way.
- **A blit leaves nothing on its source that a later draw can read.** `vrend_set_tex_param` writes
  the base and max level, the filters, the wrap modes and the format swizzle onto the *source
  texture object*, and the sampler-view bind skips its work when the same view handle is set into
  the same slot again — so on the reading, a draw after a blit samples through the blitter's
  settings. Measured, it does not: `sampled.score` draws twice through one view with a blit
  between, and virglrs returns the same pixels both times while the C's second read comes back
  byte-identical to the blit's destination, carrying the identity swizzle and the forced alpha
  `vrend_set_tex_param` wrote. Two identical draws with no state change between them have only one
  correct answer, so the C is wrong here and this is the one fixture in the tree pinned from
  virglrs rather than from the C — reproducing the bug to keep a golden green is not a trade worth
  making, and a permanently red line is a gate nobody reads. What shields virglrs is not
  established; the fixture pins the invariant, so a driver or a cache change that lets the write
  through moves the score.

## Open, and owed a decision

Each of these is a question about the renderers rather than about the harness, and each is
waiting on a call rather than on work.

- **virglrs serves no push-descriptor command.** The C dispatches `vkCmdPushDescriptorSet` and
  `vkCmdPushDescriptorSet2`; virglrs implements neither, so a guest using them lands on the
  generated default and is counted rather than served. Nothing we capture uses them — zero
  occurrences across all six venus corpora — so this is a gap with no workload behind it yet, and
  the decision is whether to serve them before one appears or wait for one to.

- **`MultisampleArrayUnsupported` is latent.** `vrend/resource.rs` refuses a multisampled array
  texture. Nothing on this host asks for one, so no corpus scores it and no boot has hit it. It is
  a known refusal waiting for either a workload that needs it or a decision that it never will be.

- **A blob typed with a planar format gets no picture.** The C converts one: `SET_TYPE` on an
  NV12, NV21, I420 or YV12 blob runs a CPU YUV-to-RGBA pass over the guest's planes into an RGBA
  texture at luma resolution. virglrs refuses it by name in `fill_texture` and blanks the
  texture, which is wrong-but-not-a-leak and says so on stderr.

  Nothing we hold reaches it. Every fixture but `vrend-vkclient` leaves all its blobs untyped,
  and vkclient's six typed blobs are `R16G16B16X16_FLOAT`. The C's other planar consumer does not
  reach it either: composite decode targets arrive through `resource_create`, and the C sets
  `guest_pixels` at one site only, in `SET_TYPE`.

  **No video route on either GNOME tier types a blob at all**, which is stronger than the fixture
  census and was measured rather than reasoned. Measured 2026-09-05, both guests seated, the
  discriminator `PIPE_RESOURCE_SET_TYPE` in the dumped command histogram:

  - stock, one 267 s window over 819,698 records: software H.264 into `waylandsink`, into
    `glupload ! glimagesink`, and Showtime with the VA decoders deranked; then `vah264dec`, and
    Showtime's default. 3,506 `TRANSFER3D`, 165 `DECODE_BITSTREAM`, zero `SET_TYPE`.
  - enhanced, one 123 s window over 772,843 records: `vah264dec ! waylandsink` — the VA decoder
    exporting a dmabuf for the compositor to import, which is the shape a planar `SET_TYPE` would
    come from — then Showtime hardware and Showtime deranked. 833 `DECODE_BITSTREAM`, 3,183
    video buffers, zero `SET_TYPE`.

  Decoded frames reach the renderer as transfers into ordinary resources, or as video buffers.
  Never as a typed blob. The instrument was checked before the negative was believed: the decoder
  names the command and counts six of it in `vrend-vkclient.bin`.

  Two gaps, so the negative is not read wider than it is. The C's comment above this code says the
  guest-pages fill is how "every GStreamer glupload whose buffers qualify" reaches the GPU;
  qualifying means dmabuf-backed, and the one arm pairing a dmabuf source with `glupload`
  (`vah264dec ! glupload ! glimagesink`) died guest-side on a bus error before it drew. That arm
  is unobserved, not disproved. And synoik was not booted — it has no GNOME and no video stack, so
  it is the least likely of the three, but it was not tried. Until a workload is found the
  conversion is unreachable and unscoreable, and refusing it by name is the resting state; the
  decision is whether to keep hunting for one.

## Owed, and waiting on work

These are not decisions. Each is settled in shape and unwritten in code, and each is here so that
it survives the session it was found in.

- **A classic resource does not import into a venus context.** The Vulkan compositor answers
  `create_immed failed and produced an invalid wl_buffer` and kills a classic-virgl GL client
  after one benchmark; the C serves the same client on the same guest. It is the mirror of the
  venus-to-venus import, which works. Attributed in time as well as between the legs — the build
  before host-visible allocations became minted pages fails identically, so it is neither that
  change nor the force-LINEAR rule. `harness/vm/client-gl-synoik.sh` is the reproducer, and
  `synoik-glclient.vkrc` was recorded from the C, which is why replaying it green said nothing.

- **`Exporter.ctx` names a context id, not a generation of one.** A `Shared` blob left by an
  earlier life of a reused context id is counted against the new one, inflating `journal_held`.
  The keys elsewhere are generational; this one is not, and the fix is to make it so rather than
  to purge at the reuse site.

- **The libkrun opaque-journal branch is parked and ready.** `limina-p5-opaque-journal` merges into
  `third_party/libkrun`'s `limina` branch with a `third_party/manifest.toml` bump. Nothing blocks
  it now.

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
