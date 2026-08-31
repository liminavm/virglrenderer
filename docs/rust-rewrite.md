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
on the dylib intersected with the symbols libkrun names). Those 62 are the contract:

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

## Ranked by difficulty

1. **The venus wire decoder (78k generated lines).** Decision: **fork
   venus-protocol's mako templates to emit Rust.** The model layer
   (`vkxml.py`, `vn_protocol.py`, 3.1k lines) is language-neutral; only
   `templates/renderer_*.h` and `templates/types*.h` (~1.5k lines of the 2.5k) emit
   the renderer-side C we need. Porting ~1.5k lines of template buys 78k lines of
   *safe* generated decode — bounds-checked slices instead of pointer arithmetic
   over guest memory. This is the highest-leverage item in the whole rewrite and it
   is the crash-prone code the user wants gone. It is also the item most likely to
   be underestimated: budget it as its own phase with its own differential test
   (decode the same recorded ring bytes through C and Rust, compare the decoded
   struct dumps).
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
   commits are C workarounds for problems Rust makes structural.
5. **The journal/replay/snapshot family.** Subtlest behaviour, smallest code. Do it
   **last** — it is ABI contract, libkrun calls it, and it has no meaning until the
   thing it journals works.
6. **The VideoToolbox backend + AV1/H.264 bitstream synthesis (4.1k).** Ours
   already, well understood, and it maps cleanly onto `objc2` +
   `objc2-video-toolbox` + `objc2-io-surface`. `dav1d` → `rav1d` (the Rust port).

Crates that carry weight: `ash` (Vulkan), `objc2` family (IOSurface, Metal,
VideoToolbox, Mach ports), `rav1d`. `u_format`'s table is already generated from
Mesa's XML — port that generator to emit Rust alongside the venus one.

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
These are implementation-agnostic already. Work here is corpus, not framework:

- classic-vrend capture on a **stock** guest (apitrace over virgl, not zink→venus) —
  today's corpus only exercises the venus path;
- a video clip per codec × path (H.264, HEVC, VP9 hardware; AV1 software), each with
  a VPP conversion leg, since that is what `d1fdc034` shows is fragile;
- a suspend/resume cycle taken **mid-workload**, not from idle.

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
- **Oracle**: `resource_read_iosurface` and `transfer_read_iov` pixels, plus return
  codes and fence-retirement *order*. Record goldens from the C build now; pin them
  as fixtures in the repo.
- **Named gap, the riskiest harness item**: there is no host-side capture of full
  venus *frame* traffic — only the re-creation journal. Extending `vkr_journal` to a
  full ring-stream recording mode is the cheapest route, and it must be done in the
  **C** implementation first so the goldens are recorded from A. Flag it early; if it
  turns out expensive, Layer 1 covers venus frames at boot speed and Layer 2 covers
  venus object lifetime only.

### Layer 3 — carried over

Upstream's `tests/test_virgl_*` are ABI-level and should link against either
implementation; keep them. Carry `tests/fuzzer/` corpora into `cargo-fuzz` once the
decode paths exist. Keep the perf bench as a **trend ledger, not a gate** — a
rewrite regresses performance invisibly, and gating on it this early stops work for
the wrong reason.

## Phases

Each phase ends with the harness green against the phase's scope; the C build stays
buildable throughout as the A-side reference.

- **P0 — Harness.** Done. Both replayers run against the C dylib and score into
  pinned fixtures; the ABI's exported symbols and struct layouts are pinned too;
  the C tree is tagged `virgl-pre-rewrite-2026-08-31`. See `harness/README.md`.
  The corpora the replayers read are not in git and have no permanent home yet.
- **P1 — Skeleton.** The virglrs tree scaffolded in this repository, producing a
  dylib and a prefix layout interchangeable with the C build's. All 69 symbols
  exported and stubbed; the ABI
  types, resource table, context table, fence tracking, and async fence retirement
  implemented for real. Boots to software-2D.
- **P2 — venus.** Fork venus-protocol's templates to emit Rust; differential-test the
  decoder against C on recorded ring bytes. Then vkr: instance/device/queue/memory/
  image/buffer/descriptor/command-buffer, rings, budget, the Metal + IOSurface
  helpers. Ends at a seated venus GNOME desktop, booted with the existing venus-only
  `virgl_override` limina already has for forcing venus-only flags — no new
  machinery, and no classic stubs that have to lie about capsets. This is where the crash pain is; it
  goes first among the renderers.
- **P3 — vrend.** TGSI parser, `u_format` generator, the GL state machine,
  TGSI→GLSL, blitter, EGL/GLES winsys, IOSurface scanout. Ends at accelerated GL for
  stock guests.
- **P4 — video.** Decode command path, VideoToolbox backend via `objc2`, AV1 OBU
  synthesis, H.264 parameter sets, `rav1d`. Ends at hardware decode per codec plus
  the VPP legs.
- **P5 — snapshot/replay.** Journal, replay, memory census/read/write, sync
  export/restore, classic content export/restore. Ends at suspend/resume parity.
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
