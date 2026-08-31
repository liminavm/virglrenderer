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
  then the whole stream in execution order through the limina replay ABI, and reports a per-kind
  tally. Run it with `vkr-replay.sh <corpus>`; it builds the crate and points the Vulkan loader at
  the ICD under test.
- `vkr-record-decode.py` — decodes a venus full-stream capture (`--check` validates a capture
  structurally before it is pinned as a fixture, and reports how many records were recorded out of
  execution order — see the ordering rule in `src/venus/vkr_record.h`).

Corpora come from vrend's in-memory tracer (`LIMINA_VREND_TRACE=<MB>`) for classic contexts, and
from the venus recorder (`LIMINA_VKR_RECORD=<MB>`, `src/venus/vkr_record.[ch]`) for venus. Both
dump on demand through a FIFO rather than on a timer, so asking for a capture costs the render
path nothing until it happens.

### What P0 still has to build

- **Scoring for the venus replayer.** `replay/rs` replays a corpus end to end — a 580k-record
  vkmark capture feeds 506657 commands and 606 control events with no failures — but what it
  prints (an accepted/rejected tally, a per-context allocation count and byte total) is not
  something two implementations can be diffed on, and none of it is pinned. It can fail a crash,
  not a regression.
- **Fixture-named scoring.** The default readback target is picked by a heuristic inherited from
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
