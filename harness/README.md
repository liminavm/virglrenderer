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
  captured order. Scores every colour offscreen at its unref — a content hash and an ink count
  per readback, in stream order — and every IOSurface-backed resource at end of stream, with the
  same `--score`/`--expect` contract as the venus side.
  Run it with `vrend-replay.sh <corpus>`, which supplies the zink-on-KosmicKrisp environment the
  renderer needs; `build.sh` alone builds it, pointing `VIRGL_PREFIX` at the implementation under
  test.
- `vrend-trace-decode.py` — decodes the same dump format for human inspection.
- `rgba2png.py` — turns raw readbacks into viewable PNGs.
- `rs/` — `vkr-replay`, the venus replayer. Creates each context, feeds the prologue journals and
  then the whole stream in execution order through the limina replay ABI, and scores the result.
  Run it with `vkr-replay.sh <corpus>`; it builds the crate and points the Vulkan loader at the
  ICD under test. `--score <file>` writes the score, `--expect <file>` diffs against a pinned one
  and exits non-zero, so `diff` is the whole comparison tool.
- `rs/` also builds `venus-roundtrip`, the venus decoder's differential test. It decodes every
  recorded command with the Rust decoder, encodes it straight back, and compares against the bytes
  the guest sent. There is no C dump to diff against because there does not need to be one: the
  recorded wire IS mesa's encoder output, so a byte-identical re-encode is a diff between the two
  implementations over every command in the corpus. It needs no GPU and no ICD — run it as
  `cargo run --release --bin venus-roundtrip -- <corpus>`, which is a second and a half over the
  61 MB capture. Failure is per command: it names each command type, its outcome, and the first
  divergent offset, so one unimplemented shape cannot hide the thousand commands behind it. Both
  corpora reproduce exactly. It does not cover reply encoding — a recording holds requests only,
  never the bytes a renderer wrote back — and it tolerates the guest's padding bytes, which mesa
  leaves uninitialised and this renderer zeroes on purpose; the encoder reports exactly which
  ranges those are, so the tolerance is a set of offsets rather than a loose comparison.
- `rs/` builds `venus-reply-oracle` for the half `venus-roundtrip` cannot reach. The corpus asks
  for replies and holds none of them, and those are two separate facts. The requests keep their
  `VN_CS_COMMAND_FLAG_GENERATE_REPLY` bit: 137 of venus's commands in 24 kinds and 64 of synoik's
  in 25 kinds ask to be answered. What no recording holds is the *answer* — a recording is the
  guest's side of the wire — and replay never produces one either, because both replay entry points
  call `vkr_replay_strip_reply` to clear that bit before dispatch. So the 326 per-command reply
  wrappers have no witness in the corpus and none from replay, and they are generated: one template
  mistake is 326 identical bugs, each reaching a guest as plausible garbage rather than as an
  error. The oracle encodes every recorded command's reply twice, once with the generated Rust and
  once with venus-protocol's own generated C renderer encoder, and compares the bytes. That C is
  ground truth because it is what every venus guest in existence decodes. Both sides encode the *same*
  `vn_command_*`: it is `#[repr(C)]`, so the C reads the memory the Rust decoder filled rather than
  a second construction of it, and nothing has to agree about filling. It needs the C toolchain, so
  it is behind a feature — `cargo run --release --features reply-oracle --bin venus-reply-oracle --
  <corpus>`. Every corpus matches on every command. A recorded command carries the guest's
  *request*, so its outputs arrive zeroed — and zero is the value that hides a content mistake,
  since two encoders reading different members of the same zeroed struct write the same bytes. The
  outputs are therefore planted with distinct values before encoding. Measured: a corruption that
  only perturbs non-zero payload bytes is caught on 449 of synoik's 1397 replies with the fill and
  331 without it. Still out of reach are chained outputs — the fill leaves `pNext` null, because
  planting one link takes the reachable type set from 42 structs to 235.
- The Rust renderer has a second, GPU-free gate on the same corpora: `vkr-replay.sh` with
  `VIRGL_PREFIX` pointed at `virglrs/prefix`. It drives the real ABI — context create, the replay
  feed, the decode loop, the object table — and every command is accounted for rather than merely
  parsed, which is what separates it from `venus-roundtrip`. Every corpus reaches `cmds N/N` with
  no `FAIL` line. What it catches is the layer between the bytes and Vulkan, and it caught two things
  already — a `VK_NULL_HANDLE` treated as a missing object, and every command after the first that
  named one.

  **It measures that a command was accounted for, not that it was carried out correctly.** Measured
  by sabotage against a build that serves the whole frame: every array accessor cut to hand its
  handler one element fewer than the guest sent replays both corpora at `cmds 506657/506657` and
  `1348/1348`, census unchanged — and KosmicKrisp's own workload counters (barriers, render pass
  starts, clears) come back byte-identical too, so the one host-side signal in the output cannot
  see it either. `venus-roundtrip` does not help: it never calls a handler or an accessor. What
  covers that gap is two layers of witness, and `harness/sabotage/sweep.py` is what says so.

  **The accessors are witnessed by generation, because the bug is in the template.** One template
  mistake is a hundred identical bugs, so `venus-gen` emits a test per command that plants its
  arrays, encodes with the real argument encoder, decodes with the real decoder, and asks the
  accessor two things: its length, and which member it read. Those are the accessor's only two
  degrees of freedom — and the second is not redundant, because where several arrays share a
  count that is all that separates them, and where they share an element type too (`pOffsets`,
  `pSizes`, `pStrides`) the type system does not. 52 commands are covered; the six arrays whose
  count lives in another struct, behind an out-pointer, or in arithmetic are named in the
  generated module, because a planter cannot establish both halves of those.

  **What a handler does between the accessor and the driver is witnessed by hand.** The seam is
  generated: `Device::plant_<cmd>` puts a plain function of the entry point's own shape into a
  proc table, `Driver::plant_device`/`plant_pool` stand that table up as a device a handler can
  record into, and `Driver::pool_child_id` reads the pairing back. See
  `a_recording_handler_hands_the_driver_what_the_guest_sent` for the four argument shapes and
  `a_pool_allocation_hands_over_the_run_the_guest_asked_for` for a run allocated from a pool,
  where the guest's ids, the driver's handles and the pool's record of both are three things that
  can go out of step with each other and nothing afterwards can tell.

  **A passing suite says nothing about what it would catch, so `harness/sabotage/sweep.py` asks
  directly.** Each entry is a one-line edit that makes the renderer wrong in a way a guest would
  see, applied to a clean tree, tested, and reverted; `RED` names the test that noticed and
  `SURVIVED` names a hole. Every edit asserts it matched, because a sweep reporting `RED` for an
  edit it never made is worse than no sweep. Add an entry with each witness rather than after.

- The layout oracle is the third leg of the same feature, and it runs as a plain unit test:
  `cargo test --features reply-oracle` in `virglrs/`. `venus-roundtrip` proves the wire and
  `venus-reply-oracle` proves the replies, but both compare *bytes*, and the reply oracle only
  reaches them by handing a Rust pointer to a C encoder. What no byte comparison can see is the
  memory contract underneath that — nor the one the driver relies on when it is given a `Vk*` we
  filled. So every generated type is asked for its offsets twice, `offsetof` against `offset_of!`,
  and the answers are diffed. It caught a real one on its first run: the model reorders a struct's
  members to put a length before its array, which is what the wire wants and not what the layout
  is, and `VkHostAddressRangeEXT` reached the driver with its address and size swapped.
- `fixtures/` — pinned scores, recorded from the C build. `vrend.score` scores 310 offscreens
  from the classic corpus; `vrend-nodraw.score` is the same run with every `DRAW_VBO` dropped, and
  the diff between the two is the positive control: 19 offscreens lose their ink, and all three
  1280x800 scanout IOSurfaces go from distinct fully-inked hashes to one shared all-zero hash. An
  empty diff would mean the oracle measures nothing. `synoik.score` is the venus content fixture:
  its capture was taken mid-workload, so 22 device allocations are still live and half of them
  carry GPU-written bytes. `venus.score` and `synoik-lifecycle.score` are the lifecycle fixtures:
  vkmark runs to completion and the synoik session is stopped before its dump, so every context
  censuses zero at its destroy — a port that leaks a VkDeviceMemory fails there. The two synoik
  fixtures are one workload measured twice on purpose, and neither one can be the other: the
  census scores memory that is still live, so the corpus that proves teardown has nothing left to
  hash (`vm/README.md`).
- `vkr-record-decode.py` — decodes a venus full-stream capture (`--check` validates a capture
  structurally before it is pinned as a fixture, and reports how many records were recorded out of
  execution order — see the ordering rule in `src/venus/vkr_record.h`).

Corpora come from vrend's in-memory tracer (`LIMINA_VREND_TRACE=<MB>`) for classic contexts, and
from the venus recorder (`LIMINA_VKR_RECORD=<MB>`, `src/venus/vkr_record.[ch]`) for venus. Both
dump on demand through a FIFO rather than on a timer, so asking for a capture costs the render
path nothing until it happens.

The corpora themselves are NOT in git — they run from kilobytes to tens of megabytes, and a
permanent home for them is still to be decided. Recapture them with `vm/capture.sh` (see
`vm/README.md`); the pinned scores here only regress against the corpus they were recorded from.

### The IOSurface leg

IOSurface is the macOS dma-buf and the whole present path. vrend renders *into* the display
surface — it is wrapped as an `EGL_IOSURFACE_LIMINA` EGLImage, so the framebuffer's storage IS the
surface — and venus imports the guest image as an `MTLTexture` over one. So `transfer_read_iov`
and `read_iosurface` are not two views of one thing; they are different paths, and a port can get
either right while getting the other wrong.

Classic scores the surface itself: `sync_iosurface` (a classic-only blit-and-wait — a venus blob
renders into its surface directly and must never be synced), then `read_iosurface`, hashed. It is
scored at END of stream because a capture never unrefs its scanout: the opposite end of the run
from the colour offscreens, each read at the last moment it is both complete and still alive.

Venus scores only a count of backed blobs. `read_iosurface` takes a byte stride and a row count,
and the venus corpus carries no geometry — dimensions live in `SET_SCANOUT_BLOB`, a virtio-gpu
control command, while the stream records ring traffic. Recording scanout geometry beside it is
what would unlock hashing venus frames, and it is the only Layer 2 oracle for venus pixels there
can be: a zero-copy scanout blob has no `transfer_read` at all.

**No score contains an IOSurface id.** An id is host-private, recycled the instant its surface
dies, and free to change across a snapshot restore. What a port owes is that a resource is backed
and what its surface contains.

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

### Scoring the port against the C

A fixture is recorded from the C, so the gate ladder compares Rust to Rust and a fixture mismatch
means a regression. Reading a Rust score against the *C's* score is a different question, and on
all three corpora the answer is currently agreement: every census id, size and hash, and every
`iosurface backed` count, matches byte-for-byte.

Equal totals are not agreement. These two censuses once read 22 against 22 while sharing only
twenty entries, because a missing export and a missing import cancelled. Compare the ids, never
the count.

Claims about what a corpus contains are settled against the corpus. That an allocation is an
import was established by printing every allocation's `pNext` verdict during a replay — two
imports in 31 allocations, exactly the two ids in dispute — not by reasoning from the C's source
about which ones ought to be. The same probe is what found four `R8G8B8A8_SRGB` scanouts in a
corpus everything else in it said was BGRA.

Before concluding anything from a hash, compute the all-zero FNV-1a for that read length. Empty
and wrong are different findings and the score does not distinguish them: a refused command leaves
memory no one wrote, and reads as a hash like any other. A scanout export that mints a surface
nothing writes through scores exactly this way, and reports an empty surface as content — which is
worse than reporting nothing. The lengths the corpora actually read:

| bytes | all-zero FNV-1a | | bytes | all-zero FNV-1a |
|---|---|---|---|---|
| 65536 | `eb05052ea5b62325` | | 262144 | `9c735bed0a722325` |
| 131072 | `c74b47c8c74a2325` | | 393216 | `156ad9514d9a2325` |
| 196608 | `1a0564b2e8de2325` | | 1048576 | `a96777069d622325` |

Two habits worth keeping. A count that reads zero is worth less than one that reads busy —
`render_pass_starts=0` said neither replay rendered while the allocation-pool counters in the same
log showed real command buffers on both, and the zero was the counter that did not cover the
device. And whole-allocation divergence and partial divergence can share one cause: a refused
command that clears an image leaves it empty, while the same command refused inside a render pass
that other commands did write leaves the allocation merely wrong. Do not read those as two
findings before checking whether one refusal explains both.

### `--smoke`, the skeleton gate

Both replayers take `--smoke`: apply the resource and context events, skip commands, transfers and
the venus replay feed, and exit on whether every create landed. It is what a renderer that has an
ABI, a resource table and a context table — and nothing else yet — can be held to, and it makes
P1's exit condition a command rather than a judgement call.

It refuses `--score`/`--expect`. A smoke score is a strict subset of a real one, so pinning it
would replace a golden with a weaker one that still passes.

Blob creates with a non-zero `blob_id` are skipped and counted: they export an object a command
would have made, and smoke runs no commands. The C renderer fails them in smoke mode too, which is
how we know the bar was in the wrong place rather than the port.

## Layer 0 — the ABI itself (`abi/`)

`abi-fixture.sh` pins two files and checks a build against them: `symbols.txt`, every symbol the
dylib exports, and `layout.txt`, the size, alignment and field offsets of every struct that
crosses the ABI. `VIRGL_PREFIX` selects the build under test, so pointing it at a Rust build is
how the port gets checked — the same variable `vrend-replay.sh`, `vkr-replay.sh` and `build.sh`
resolve, so one setting drives every layer at once; `--pin` re-records, and is only for a change to the ABI that is meant.

**No gate here has a manual install step, and adding one to a ladder buys nothing.** A script that
scores `virglrs/prefix` builds and installs it first — `abi-fixture.sh` and `vkr-replay.sh` both
do — so what gets measured is always the tree as it stands. A prefix nobody here owns gets a
staleness warning instead, and the caller decides.

Both failures are invisible to every other layer here. A missing symbol shows up at `dlopen` and
nowhere earlier; a wrong field offset never shows up at all — it compiles clean on both sides and
corrupts at runtime. The layout comes from the compiler (`abi-dump.c`, built against the header),
not from a transcription anyone has to keep in step.

The symbol fixture pins *everything exported*, not the subset libkrun happens to call. What the
dylib exports is defined by this tree; who calls what is defined outside it and moves without
warning, and a port that exports the whole list satisfies every consumer of it.

## The capture rig (`vm/`)

Layers 1 and 2 both need corpora, and a corpus comes from a real guest. `vm/` builds a
self-contained rig — limina's app bundle with this tree's renderer swapped in, plus APFS clones of
an enhanced and a stock disk — so captures never mutate limina's working set. See `vm/README.md`.

## Layer 3 — fuzz corpora and the perf ledger

`tests/test_virgl_*` are not part of the harness. They look ABI-level and are not: every one of
them includes internal headers and links the static library, so none can run against a Rust
dylib. They also need `check` and do not build on this platform. They stay as C-side regression
tests with no role in the rewrite; Layer 2 covers the same ground through the ABI at the same
speed.

`tests/fuzzer/` corpora move to `cargo-fuzz` once the Rust decode paths exist — corpora are data
and survive the language change. Performance is a trend ledger, never a gate — a rewrite regresses
performance invisibly, and gating on it stops work for the wrong reason.
