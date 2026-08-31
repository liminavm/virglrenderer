# The renderer test harness

The harness exists so the Rust rewrite (`docs/rust-rewrite.md`) has a fixed oracle: the same
corpora, run against the C renderer and against virglrs, producing comparable output. Its one
design rule is what makes it survive the transition — **it drives only the public C ABI.**
Nothing here may reach into renderer internals.

Three layers, in descending order of fidelity and ascending order of speed.

## Layer 1 — VM-level replay (`capture/`)

Capture a real workload inside a booted guest, replay it on both the accelerated backend and a
software reference, compare pixels. The highest-fidelity layer and the slowest.

- `capture-replay.sh` — glmark2 scene via apitrace over zink→venus.
- `capture-replay-shell.sh` — the real seated gnome-shell session; the closest thing to a
  desktop-correctness test we have.
- `capture-replay-vk.sh` — native Vulkan via gfxreconstruct, replayed against lavapipe.

These scripts drive a guest over SSH and were written against limina's boot scripts and fixture
paths; the paths need reconciling when the harness is wired up here. The knowledge in them —
which environment a capture needs, which replays silently fall back to llvmpipe, which oracle
actually distinguishes the backends — is the part worth carrying, and it is in the comments.

The reference leg is the oracle, not FPS: replays of the same trace on two backends can report
identical frame rates while rendering different pixels.

## Layer 2 — host-side replay (`replay/`)

Feed a recorded command stream straight into `libvirglrenderer.dylib` with no VM, no guest and
no hypervisor. This is the layer the rewrite is actually tested by, because it runs in seconds.

- `vrend-replay.c` — replays a classic-context stream: resource creates from the recorded
  arguments, transfer contents memcpy'd into synthesized backings, command batches submitted in
  captured order. Reads back scored resources and prints a content hash per readback. Build with
  `build.sh`, pointing `VIRGL_PREFIX` at whichever implementation is under test.
- `vrend-trace-decode.py` — decodes the same dump format for human inspection.
- `rgba2png.py` — turns raw readbacks into viewable PNGs.
- `rs/` — `vkr-replay`, the venus replayer. Creates each context, feeds the prologue journals and
  then the whole stream in execution order through the limina replay ABI, and scores the result.
  Run it with `vkr-replay.sh <corpus>`; it builds the crate and points the Vulkan loader at the
  ICD under test. `--score <file>` writes the score, `--expect <file>` diffs against a pinned one
  and exits non-zero, so `diff` is the whole comparison tool.
- `fixtures/` — pinned scores, recorded from the C build. `synoik.score` is the content fixture:
  its capture was taken mid-workload, so 22 device allocations are still live and half of them
  carry GPU-written bytes. `venus.score` is the lifecycle fixture: vkmark runs to completion, and
  every context censuses zero at its destroy — a port that leaks a VkDeviceMemory fails there.
- `vkr-record-decode.py` — decodes a venus full-stream capture (`--check` validates a capture
  structurally before it is pinned as a fixture, and reports how many records were recorded out of
  execution order — see the ordering rule in `src/venus/vkr_record.h`).

Corpora come from vrend's in-memory tracer (`LIMINA_VREND_TRACE=<MB>`) for classic contexts, and
from the venus recorder (`LIMINA_VKR_RECORD=<MB>`, `src/venus/vkr_record.[ch]`) for venus. Both
dump on demand through a FIFO rather than on a timer, so asking for a capture costs the render
path nothing until it happens.

### What the venus score is, and is not

It scores RENDERER STATE, not pixels. A VM-free replay has no scanout and presents no frames, so
what it compares is the accept counts and the contents of the device memory the commands left
behind. That is the right target for the rewrite: a port gets object lifetimes, descriptor writes
and memory bindings wrong long before it gets a colour space wrong. The census deliberately
excludes map_ptr-exported blobs — those are the VMM's mapped-blob capture and are already
host-mapped — so what gets hashed is GPU-produced state plus allocations nothing has written.

Each context is scored when it is destroyed, and once more at the end if it is still alive. A
workload that exits cleanly frees everything, so scoring only at the end would score nothing.

Each score is taken twice, 50 ms apart, and re-taken until two passes agree. The replay skips
every ring flow-control command, so nothing in the stream waits on the GPU: a hash read the
instant `replay_end` returns can race queue work still executing and look nondeterministic when
the renderer is perfectly deterministic.

### What P0 still has to build

- **Fixture-named scoring** for the classic replayer. The default readback target is picked by a heuristic inherited from
  the debugging spike this grew out of. A corpus wants its scored resources named by the fixture.
- **The `force_ctx_0` readback limitation** documented at the top of `vrend-replay.c`, which is a
  precondition for scoring more than one resource per run reliably.
- **ABI fixtures.** The symbol list (`nm -gU` on the dylib ∩ what libkrun names — 62 today) and
  the layouts of `virgl_renderer_callbacks`, `virgl_renderer_resource_create_args` and the
  blob/import arg structs, pinned as files the Rust build is diffed against. A layout mismatch
  compiles clean and corrupts at runtime, and nothing else here would catch it.
- **Pinned goldens** recorded from the C build before any Rust lands.

## The capture rig (`vm/`)

Layers 1 and 2 both need corpora, and a corpus comes from a real guest. `vm/` builds a
self-contained rig — limina's app bundle with this tree's renderer swapped in, plus APFS clones of
an enhanced and a stock disk — so captures never mutate limina's working set. See `vm/README.md`.

## Layer 3 — carried over

`tests/test_virgl_*` in this tree are ABI-level and should link against either implementation.
`tests/fuzzer/` corpora move to `cargo-fuzz` once the Rust decode paths exist. Performance is a
trend ledger, never a gate — a rewrite regresses performance invisibly, and gating on it stops
work for the wrong reason.
