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
  corpora reproduce exactly. It does not cover reply encoding — no recording contains one — and
  it tolerates the guest's padding bytes, which mesa leaves uninitialised and this renderer zeroes
  on purpose; the encoder reports exactly which ranges those are, so the tolerance is a set of
  offsets rather than a loose comparison.
- `rs/` builds `venus-reply-oracle` for the half `venus-roundtrip` cannot reach. No recording holds
  a reply — both replay entry points call `vkr_replay_strip_reply` — so the 326 per-command reply
  wrappers have no witness in the corpus, and they are generated: one template mistake is 326
  identical bugs, each reaching a guest as plausible garbage rather than as an error. The oracle
  encodes every recorded command's reply twice, once with the generated Rust and once with
  venus-protocol's own generated C renderer encoder, and compares the bytes. That C is ground truth
  because it is what every venus guest in existence decodes. Both sides encode the *same*
  `vn_command_*`: it is `#[repr(C)]`, so the C reads the memory the Rust decoder filled rather than
  a second construction of it, and nothing has to agree about filling. It needs the C toolchain, so
  it is behind a feature — `cargo run --release --features reply-oracle --bin venus-reply-oracle --
  <corpus>`. Both corpora match on every command. A recorded command carries the guest's
  *request*, so its outputs arrive zeroed — and zero is the value that hides a content mistake,
  since two encoders reading different members of the same zeroed struct write the same bytes. The
  outputs are therefore planted with distinct values before encoding. Measured: a corruption that
  only perturbs non-zero payload bytes is caught on 449 of synoik's 1397 replies with the fill and
  331 without it. Still out of reach are chained outputs — the fill leaves `pNext` null, because
  planting one link takes the reachable type set from 42 structs to 235.
- The Rust renderer has a second, GPU-free gate on the same corpora: `vkr-replay.sh` with
  `VIRGL_PREFIX` pointed at `virglrs/prefix`. It drives the real ABI — context create, the replay
  feed, the decode loop, the object table — and every command is accounted for rather than merely
  parsed, which is what separates it from `venus-roundtrip`. Both corpora reach `cmds N/N` with no
  `FAIL` line. It cannot be compared against a pinned score: no command is served yet, so there is
  no device memory to census and nothing to hash. What it catches is the layer between the bytes
  and Vulkan, and it caught two things already — a `VK_NULL_HANDLE` treated as a missing object,
  and every command after the first that named one.
- `fixtures/` — pinned scores, recorded from the C build. `vrend.score` scores 310 offscreens
  from the classic corpus; `vrend-nodraw.score` is the same run with every `DRAW_VBO` dropped, and
  the diff between the two is the positive control: 19 offscreens lose their ink, and all three
  1280x800 scanout IOSurfaces go from distinct fully-inked hashes to one shared all-zero hash. An
  empty diff would mean the oracle measures nothing. `synoik.score` is the venus content fixture:
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
