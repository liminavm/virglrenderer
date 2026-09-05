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
  Run it with `vrend-replay.sh <corpus> --renderer rs|c`, which supplies the zink-on-KosmicKrisp
  environment the renderer needs; `build.sh` alone builds it, pointing `VIRGL_PREFIX` at the
  implementation under test. Two switches bisect a score line that differs: `--until <seq>` stops the stream there
  and scores what the surfaces hold at that point, and `--readback <res>` on a resource still
  alive at the end reads its texture as well, so a scanout's surface and its texture can be
  compared — the pair disagreeing is how a stale surface read was told from a wrong render.
  `REPLAY_DUMP_DIR` (narrowed by `REPLAY_DUMP_W`) writes every scored readback and surface as raw
  BGRA, for `rgba2png.py` and a pixel diff.
- `vrend-trace-decode.py` — decodes the same dump format for human inspection.
- `corpus.py` — the synthetic-corpus writer: the trace container and the virgl commands, shared
  by the `make-*-corpus.py` scripts.
- `make-blit-corpus.py` — writes a synthetic classic corpus that forces the shader blitter, for
  the gate no recorded session provides (see `fixtures/blit.score` below).
- `make-sampled-corpus.py` — the same, for the blits whose destination the sweep cannot read
  back; it scores them through a draw (see `fixtures/sampled.score` below).
- `make-surface-corpus.py` — writes a synthetic classic corpus that destroys a surface the
  framebuffer is still drawing through (see `fixtures/surface.score` below).
- `rgba2png.py` — turns raw readbacks into viewable PNGs.
- `rs/` — `vkr-replay`, the venus replayer. Creates each context, feeds the prologue journals and
  then the whole stream in execution order through the limina replay ABI, and scores the result.
  Run it with `vkr-replay.sh <corpus> --renderer rs|c`; it builds the crate and points the Vulkan
  loader at the ICD under test. `--score <file>` writes the score, `--expect <file>` diffs against a pinned one
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
- `rs/` builds `vrend-roundtrip`, the same differential for the classic decoder. A vrend trace
  dump holds every command the C dispatched, grouped by the submit that carried it, so the Rust
  decoder frames each recorded batch, decodes each command, re-encodes it and compares dwords:
  `cargo run --release --bin vrend-roundtrip -- ../../vm/captures/vrend.bin`, well under a second.
  Two things make it a gate rather than a smoke test. The C recorded only what it *accepted*, so
  a refusal here is a command a real guest sent and the C served -- a decoder stricter than the
  wire shows up as one. And the decoder is exact where the C is loose (a trailing partial element,
  dwords past the ones a command reads), because a decoder that ignores dwords cannot reproduce
  them and the comparison would be blind exactly there; `END_TRANSFERS` keeps the slack mesa pads
  the transfer prologue with for the same reason. All 13,726 commands of `vrend.bin` reproduce.
  Like `venus-roundtrip` it never calls a handler: it proves the wire is read where it lives,
  not what is done with it.
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
- The Rust renderer has a second, GPU-free gate on the same corpora: `vkr-replay.sh --renderer rs`. It drives the real ABI — context create, the replay
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

  **The host memory budget is invisible to every other gate, and that is the point.** With no
  cap configured nothing is ever refused, so a replay scores identically whether the ledger is
  right, wrong, or absent -- byte-identical corpora are the proof the feature changed nothing a
  guest can see, and no evidence at all that it works. Its witnesses are the whole of its
  coverage: `an_allocation_over_the_budget_never_reaches_the_driver` (a refusal that arrives
  after `vkAllocateMemory` has already run costs exactly the memory the cap exists to save, and
  only the call counter can see the difference), `a_scanout_is_charged_for_the_pages_the_surface_took`,
  and `an_import_is_not_charged_because_its_bytes_are_the_exporters`. One sabotage that ought to
  be there cannot be written: a charge is credited by the record that holds it going away, so
  "a free forgets to credit" has no line to break. `sweep.py` says so where the entry would be.

- The layout oracle is the third leg of the same feature, and it runs as a plain unit test:
  `cargo test --features reply-oracle` in `virglrs/`. `venus-roundtrip` proves the wire and
  `venus-reply-oracle` proves the replies, but both compare *bytes*, and the reply oracle only
  reaches them by handing a Rust pointer to a C encoder. What no byte comparison can see is the
  memory contract underneath that — nor the one the driver relies on when it is given a `Vk*` we
  filled. So every generated type is asked for its offsets twice, `offsetof` against `offset_of!`,
  and the answers are diffed. It caught a real one on its first run: the model reorders a struct's
  members to put a length before its array, which is what the wire wants and not what the layout
  is, and `VkHostAddressRangeEXT` reached the driver with its address and size swapped.
- The video oracle is the same idea for the parameter sets the VideoToolbox backend synthesizes:
  `cargo test --features video-oracle` in `virglrs/`, which builds the C tree's serializers into
  the test binary and drives both sides from one script. These have no other reference — there is
  no conformance vector for an SPS we invented, only the bytes that have played — so agreeing with
  the C is the whole standard, and it is checked byte for byte rather than asserted.
- **A refusal is scored, not dropped.** The sweep used to write a line for a readback that
  succeeded and nothing at all for one that failed, so a port that started refusing what the
  reference serves scored byte-identical -- the one class of divergence the oracle could not see,
  and the one a hardening change makes real. Every refused readback now carries its resource, its
  extent and the errno, 2132 lines across eight fixtures. One line more comes from a probe: a
  transfer naming a handle nothing created, because no recorded guest ever asks for something
  refused with an errno rather than with the `-1` that means "cannot serve this readback", and the
  transfer entry points answer with a POSITIVE errno unlike most of the ABI. That probe caught the
  port answering -22 where the reference answers 22.

  The sweep does not ask for a planar resource as one RGBA texture. It is scored through its
  IOSurface plane by plane, the request has no answer, and the C refuses it by poisoning the
  context -- which dropped 893 later submits and replayed half the composite corpus into a dead
  context. A sweep must not manufacture the guest-hostile request it exists to observe the absence
  of.
- `fixtures/` — pinned scores, recorded from the C build. `vrend.score` scores 310 offscreens and
  5 IOSurfaces from the classic corpus: the stock guest's GNOME session with its wallpaper, which
  arrives as one 64 MiB transfer and is why the recorder's default capacity is 512 MB. That one
  transfer is most of what the corpus measures — the mip chain the guest builds from it with
  eleven blits, and the overview composited over it — so a recorder that drops it replays to a
  flat-colour desktop that measures none of this. `vrend-nodraw.score` is the same run with every
  `DRAW_VBO` dropped, and the diff between the two is the positive control: 19 offscreens lose
  their ink, and all three 1280x800 scanout IOSurfaces go from fully inked to one shared all-zero
  hash. An empty diff would mean the oracle measures nothing. `synoik.score` is the venus content
  fixture: its capture was taken mid-workload, so 22 device allocations are still live and half
  of them carry GPU-written bytes. `venus.score` and `synoik-lifecycle.score` are the lifecycle
  fixtures: vkmark runs to completion and the synoik session is stopped before its dump, so every context
  censuses zero at its destroy — a port that leaks a VkDeviceMemory fails there. The two synoik
  fixtures are one workload measured twice on purpose, and neither one can be the other: the
  census scores memory that is still live, so the corpus that proves teardown has nothing left to
  hash (`vm/README.md`).
  `vrend-composite-h264.score` and `vrend-composite-vp9.score` are hardware decode into the
  **composite planar** target -- one NV12 resource with its two planes chained behind it, against
  the per-plane shape `vrend-vp9stock` holds. Both come from one capture on the enhanced guest,
  with H.264 and VP9 played one after the other in Showtime.

  **Showtime splits each playback across two virgl contexts, and neither half is the workload.**
  The player's GL context draws the video (`--ctx 8` for H.264, `--ctx 10` for VP9); its decoder
  runs in a context of its own (`--ctx 9` and `--ctx 11`), and that is where every
  `DECODE_BITSTREAM` and every bitstream upload lands. A player context replayed alone has its
  bitstream nowhere and every plane reads back zero; a decoder context alone fills planes that
  nothing samples. So both fixtures name both: `--ctx 8,9` and `--ctx 10,11`. `--ctx` takes a list
  for this reason, and the replayer keeps each context's commands in their own submit -- a batch
  handed to the wrong context is rejected wholesale as "Illegal resource", which reads exactly
  like the resource bug this replay exists to find. The way to find the pair in a new capture is
  to count `XFERDATA` records per context: the decoder's is the one holding them. Naming contexts
  is not optional -- the busiest is the shell's, and scoring it measures the desktop and none of
  the decode.

  **The plane lines are the decoded picture.** A composite target is an IOSurface-backed planar
  surface, which is neither a GL texture nor a BGRA IOSurface, so the sweep resolves it with
  `IOSurfaceLookup` and hashes it plane by plane -- and only the tight rows: the pitch is the
  allocator's choice and the surface id is the run's, so neither is scored. H.264 pins six DPB
  slots as 2560x1440 luma and 1280x720 two-byte chroma, VP9 thirty-five at 352x240 and 176x120,
  every plane inked and every hash distinct. Around the picture the fixtures gate 5018
  resources created with none refused, 334 IOSurface-backed, and the decode commands served to
  the end with no submit errors.

  **The conversion runs but its pixels are not hashed.** `convert_planes` fills a composite
  target's base RGBA texture from its two planes, and with both contexts in one pass it now
  executes -- six passes on the H.264 leg -- so a GL error or a crash in it fails the gate. Its
  output is still not scored, and the reason has moved: a composite target is read through its
  IOSurface planes, and its base texture has no readback route at all, because the resource's
  format says planar and every transfer read of a planar format is refused. Scoring it means the
  renderer answering "hand me this resource as it is sampled", which is an ABI question, not a
  harness one.

  `vrend-overview.score` is the GNOME shell with someone **typing in the overview's search
  entry**, and it exists because that one act allocates a resource no other corpus contains: a
  `PIPE_BUFFER` carrying `SAMPLER_VIEW | RENDER_TARGET`, a texture buffer. virglrs refused that
  shape, which killed the session on the first glyph -- and no pinned corpus held one, because
  every capture had been a boot and a workload and nobody had ever typed. The tier had nothing
  to do with it: it is the same GNOME on the same virgl driver on all three images.

  Recorded headless, because typing needs no window: `/tmp/type.py` in the guest drives
  `/dev/uinput` directly, so mutter sees a real keyboard and a capture can search for four words
  without a human in front of it. That is worth keeping for any corpus needing input.

  ```sh
  ./capture.sh video --renderer c --out overview --mb 2048   # then, in the guest:
  #   sudo systemctl isolate graphical.target
  #   sudo python3 /tmp/type.py firefox settings terminal files
  ./dump.sh vrend-overview
  ```

  It also carries 58 composite planar decode targets, which GNOME probes at 64x64 during boot,
  so it gates the composite target path against a build that cannot make one.

  `vrend-shm.score` is the workload with **no GPU client in it at all**: a GTK4 terminal run with
  `GSK_RENDERER=cairo`, `GDK_DEBUG=gl-disable` and `LIBGL_ALWAYS_SOFTWARE=1`, so its surface
  reaches the compositor as a `wl_shm` buffer. The client issues no GL, and the corpus is
  therefore entirely the shell uploading and sampling somebody else's pixels. That inverts the
  usual balance and is the point of keeping it: `COPY_TRANSFER3D` dominates at 7760 against the
  browser corpus's 1532, and no draw target belongs to the client. Every accelerated corpus here
  measures the path a client's own rendering takes; this one measures the path taken when a
  client has none, which is what a guest without working acceleration falls back to.

  ```sh
  ./capture.sh vrend --renderer c --out shm --mb 1024   # then, in the guest, with the session
  #   env: GSK_RENDERER=cairo GDK_DEBUG=gl-disable LIBGL_ALWAYS_SOFTWARE=1 ptyxis -x ...
  ./dump.sh vrend-shm      # replayed at --ctx 2
  ```

  `vrend.caps` is the classic capsets, `virgl_caps_v1` and `virgl_caps_v2`, as the C fills
  them on this host: `./vrend-replay.sh ../vm/captures/vrend.bin --renderer c --caps FILE` writes
  them a field a line, and `--renderer rs` diffs the Rust side against the fixture. The guest's virgl driver
  configures itself from nothing else -- a format missing from `sampler` is a format the guest
  never creates, a wrong `glsl_level` is a whole feature set switched off -- and none of it is a
  pixel, so a score cannot see it. Three lines are expected to differ, all by design, and every
  one of them is this build declining to promise something it does not yet serve. **The capset
  is a promise, not a summary.** A guest reads it before it allocates anything, and the kernel
  has already handed it the handle by the time the host is asked -- so a host that advertises
  what it will later refuse does not degrade that guest, it poisons it for the life of its
  context. Advertising less than the C is therefore the safe direction and advertising more is
  never one; each line below closes when its path lands, not before.

  `num_video_caps`/`video_caps` carry only the profiles this build both has silicon for and has
  a decode path for. On a host with the silicon that is now the C's own six, byte for byte; a
  machine missing a codec's hardware reports fewer on both legs and still matches.

  `capability_bits_v2` is short `VIDEO_GUEST_PLANES` (1<<19), which the C sets on there being a
  decoder at all. That is right for the C, which writes the decoded frame back into the guest's
  own pages; the bit tells the guest that backing a decode target's planes with its memory is
  worthwhile, and a guest that took the offer here would export an honest-looking dmabuf fd
  naming a black frame. It turns on with the writeback, not with the decoder.

  `sampler` differs for two separate reasons. It carries the fourteen `ASTC_*_SRGB` formats the
  C's `ASTC_FORMAT` macro never registers (`virglrs/vrend-gen/gl_formats.py`), which is
  deliberate and permanent. And it advertises `NV12` where the C advertises `NV12` and `NV21`: a
  multi-plane format is samplable only as a composite decode target, the bitmask *is* the guest's
  permission to create one, and the surface behind one here is two 4:2:0 planes -- which NV12 is
  and NV21 is not, its chroma pair being exchanged within the plane, where no CoreVideo output
  produces it and no plane ordering repairs it. The C's own rule is the same one
  (`vrend_planar_target_backable`); it answers yes for one format more.
  A fourth line is a regression.
  `vrend-vp9stock.score` is VP9 hardware decode, 963 pictures through VideoToolbox, scored the
  ordinary way: the decoded planes land in guest resources and the sweep reads them back, so 240
  of the 243 decode-target resources carry pixels with a distinct hash per frame. It was recorded
  from the **stock** guest, which is the per-plane decode target -- one resource per plane, the
  shape a guest that never negotiates the composite planar target takes, and the shape the stock
  tier keeps. The enhanced tier's delivered mesa takes the other shape (one composite resource,
  planes chained behind it, one two-plane IOSurface), and that is a second contract needing its
  own corpus, not a newer version of this one.

  The corpus is `vrend-vp9stock.bin`, captured with `capture.sh vrend` while decoding
  `spikes/vt-vp9-decode/vp90-2-09-aq2.webm` from the limina tree -- a real conformance clip, not
  `videotestsrc`, which has no hidden frames and no reference management to get wrong. The decode
  is checked against `avdec_vp9` before anything is pinned: VP9 is normatively exact, so the two
  must agree byte for byte, and eight consecutive runs must agree with each other. A golden taken
  from one run of an intermittently faulting leg grades the port against a bad frame.

  `vrend-h264.score` and `vrend-hevc.score` are the parameter-set codecs, 300 pictures each
  through VideoToolbox, scored the same way -- and, like the VP9 one, from the **stock** guest and
  its per-plane decode target. What they add over VP9 is the whole synthesis path: the guest sends
  a parsed picture descriptor and slice NALs, and the SPS/PPS (and VPS) the decoder is configured
  from are written host-side out of the descriptor. The H.264 corpus also exercises the
  **session-preserving** path 290 times in 300 frames: `num_ref_idx_lX_active_minus1` reaches us
  as the effective per-slice count, so the PPS bytes change mid-GOP, and a renderer that rebuilds
  the session there loses the reference pictures and decodes visibly wrong pixels from that frame
  on.

  A stock Fedora cannot reach either codec: mesa is built `-Dvideo-codecs=all_free` and the VA
  frontend refuses both in `vl_codec.c` before the driver is consulted, so no host advertisement
  gets through. The rig guest therefore carries RPM Fusion's `mesa-va-drivers-freeworld`, which is
  the route a real Fedora user takes for the same reason and which libva already probes ahead of
  `/usr/lib64/dri/` -- limina's `scripts/provision/install-freeworld-va.sh` does this to its own
  images, and the rig clone gets the same two `dnf install` lines by hand. `vainfo` naming a
  driver under `dri-freeworld/` is the check that it took.

  The clips are 300 frames of real screen content at 1280x720, encoded with four reference frames
  and a B-pyramid so the decoder has a DPB to get wrong, and the decode is checked against
  `avdec_h264`/`avdec_h265` in the guest before anything is pinned: both codecs are normatively
  exact, so hardware and software must agree byte for byte, and eight consecutive C runs must
  agree with each other.

  ```sh
  ./capture.sh vrend --out h264 --mb 1024        # then, in the guest:
  #   gst-launch-1.0 -q filesrc location=rig-h264.mp4 ! qtdemux ! h264parse ! vah264dec   #     ! videoconvert ! video/x-raw,format=I420 ! filesink location=/dev/stdout | md5sum
  ./dump.sh vrend-h264
  ```

  `vrend-av1.score` is 400 pictures across six clips, and it is **not scored on this host**: AV1
  decode needs M3-or-later silicon, so a machine without it advertises no AV1 and the corpus
  measures nothing on either leg. A change to what the sweep records therefore reaches it only
  when someone re-records it on that machine: a bulk re-record here cannot include it, and it lags
  the others until then. It is recorded and scored on the AV1 machine, against a rig
  copied there rather than rebuilt: the bundles are self-contained after `make-rig.sh`, the two
  renderer prefixes and the KosmicKrisp/epoxy prefixes are a few tens of MB, and only the guest
  disk is large. What the copy does need is the prefixes' own dependencies present at the paths
  they were linked against -- `install_name_tool -id` onto the new path so the replayer links
  what is there, Homebrew's `vulkan-loader`, `dav1d` and `spirv-tools`, and the *same* `libLLVM`
  the mesa prefix was built against, dropped beside it so `DYLD_LIBRARY_PATH` finds it ahead of
  the other machine's.

  The clips are limina's own AV1 spike set (`spikes/av1-obu-serializer/clips`), which is what the
  serializer was developed against: baseline, global motion, tiles, low delay, aom pyramid, pan.
  Two of the eight are deliberately left out. `superres` because **this build refuses
  super-resolution frames** -- the hardware returns them wrongly and there is no software decoder
  here -- so it would diverge by design. `filmgrain` because its hardware decode **is not
  reproducible run to run**: three consecutive decodes of the same file give three different
  md5s, which is a golden that grades the weather.

  **The reference decoder for AV1 is libaom, not dav1d.** `dav1ddec` tags its output
  `chroma-site=jpeg, colorimetry=bt709` where the VA path tags nothing, so `videoconvert` does
  different work on the two and every clip's md5 differs for a reason that is not the decode --
  which reads exactly like a broken decoder. `av1dec` (libaom) and `vaav1dec` agree byte for byte
  on all six.

  **These three are scored with `--ctx 8`, and that is not optional.** They were captured with a
  whole guest booted, so the decode is one context of four; the replayer with no `--ctx` picks the
  busiest, which is the console's, and scores 2317 empty resources and not one picture -- a clean
  exit, a full score file, and no measurement of the thing the corpus exists for. The context to
  name is the one holding `DECODE_BITSTREAM`. Six clips played one after another still land in
  one context -- GStreamer's VA elements share the render node's -- so six codecs' worth of
  create, decode and destroy score as one run.

  **A resource is zeroed when it is created, so a score line is the renderer's answer and not
  its allocator's.** A texture's contents are undefined until something writes them, and this
  driver does not zero them, so an unwritten resource reads back whatever the last tenant of that
  GPU memory left -- deterministic per renderer and different between them, which grades the
  allocation history rather than the port. This corpus is where it surfaced: the guest allocates
  224 decode-target planes it never decodes into, and the two renderers disagreed on 22 of them
  while agreeing on all 963 decoded pictures. The replayer now writes zeros over each new
  resource through exactly the call shape the score reads it with, so the untouched case is
  defined and identical. Measured before adopting it: not one line moves on `vrend`, `blit`,
  `sampled` or `surface`, so it disturbs nothing any renderer actually draws. `--no-zero-new`
  turns it off, which is how to look at what was there instead -- never record a golden that way.

  `vrend-shaders.txt` is the classic corpus's shaders as the C saw them: for each of the 33
  shaders created, `tgsi_dump` of the tokens the C parsed from the guest's text and the GLSL
  `vrend_convert_shader` emitted. It is the shader translator's differential -- a score compares
  pixels, which cannot say *which* line of a 200-line shader went wrong; the GLSL can. It is
  recorded with `vm/prefix-debug` (a `-Db_ndebug=false` build of the C; the release prefix
  compiles the dump out) as
  `VREND_DEBUG=shader ./vrend-replay.sh ../vm/captures/vrend.bin --renderer c`,
  normalised by `vrend-shader-log.py`. The Rust tests in `virglrs/src/vrend/tgsi/fixture.rs`
  read it and hold the parser to the TGSI half: every dump parses and prints back byte for
  byte; those in `virglrs/src/vrend/shader/glsl/mod.rs` hold the translator to the GLSL half.
  The fixture cannot say which *key* a block was translated under, so the live differential is
  the same replay against the Rust prefix with `VIRGLRS_DEBUG=shader`, normalised by the same
  script: the two logs must diff empty, block for block and in order, which holds the key
  construction to the C as well as the translation. Under `--nodraw` that is the 16 blocks of
  shader creation and `LINK_SHADER`; with draws it is all 33, the seventeen more being the
  variants selected at draw time.
  `blit.score` is the shader blitter's gate, and it is synthetic on purpose: no recorded session
  reaches the blitter at all. A desktop's blits are format-matched mip-chain reductions, which
  take `glBlitFramebuffer`, so a blitter could be ported, get every other fixture in this tree
  green, and have been measured by none of them. `make-blit-corpus.py` writes
  `vm/captures/blit.bin` — five blits, each a destination only the blitter can fill, four by a
  swizzle disagreement between an X-channel format and its A-channel twin and one by the red/blue
  swap an IOSurface-backed BGRA scanout forces. Its own comments carry which variants exist and
  why the depth-writing ones do not. The corpus is generated rather than recorded, so it is
  reproducible from the script and the score is pinned from the C the same way as the rest.
  `sampled.score` covers what `blit.score` cannot. The sweep reads back only a plain 2D colour
  offscreen, so a blit into a layer of an array — or out of one slice of a 3D texture — has no
  line in any score and would replay green having measured nothing. `make-sampled-corpus.py`
  writes `vm/captures/sampled.bin`, which SAMPLES each such destination in a draw whose own
  destination the sweep does read. The oracle is deliberately a different mechanism from the one
  under test: a blit that writes the wrong layer, read back by a copy that reads the wrong layer,
  hashes exactly like a correct pair, whereas a draw naming its layer in a texture coordinate
  shares nothing with the blitter's attachment path. Every layer the corpus does not blit into
  carries its own fill, so a stray blit is visible from the untouched end too.
  **`sampled.score` is the one fixture pinned from virglrs and not from the C.** Its last three
  lines draw twice through one sampler view with a blit between, and the two implementations
  disagree: virglrs returns the same pixels both times, the C's second read comes back
  byte-identical to the blit's destination — the blitter's own texture parameters, reaching a draw
  that never asked for them. Two identical draws with nothing between them have one correct
  answer, so the C is the wrong golden here; reproducing its bug to keep a fixture green is not a
  trade worth making, and a permanently red line is a gate nobody reads. The deviation is in
  `docs/rust-rewrite.md`; every other line of this fixture, and every other fixture in the tree,
  is still pinned from the C. **A bulk re-record overwrites that line with the C's answer**, and
  it is one line in a fixture nobody rereads, so restore it deliberately afterwards -- the value
  to restore is the one `res=52` carries, because the whole point is that the two reads agree.
  The last two lines are the depth-writing blit, which takes the blitter's other fragment
  shader — `gl_FragDepth` instead of a colour, and the depth attachment instead of colour
  attachment 0. Two disagreeing depth formats force it off `glBlitFramebuffer`, and the Z mask
  keeps it off the copy path. The blitted depth is sampled into a colour target, and so is the
  blit's SOURCE through the same shader and the same view shape, so a wrong number in the
  destination cannot be blamed on the depth sampling path.
  `surface.score` gates a surface's lifetime against its framebuffer. A guest may destroy a
  surface it has bound, and the C's framebuffer holds a reference: the attachment goes on taking
  pixels until the next `SET_FRAMEBUFFER_STATE`. No recorded session does this — a desktop unbinds
  before it destroys — so the ordering that matters is never sampled, and a renderer that deletes
  the surface's texture view at `DESTROY_OBJECT` scores green everywhere else. `surface.bin`
  destroys a bound surface and then CLEARS to a flat colour, over a resource pre-filled with a
  pattern, so a clear that lands nowhere is a different hash from one that lands and neither is
  uninitialised storage. Its second case destroys the surface, creates another under the same
  handle, and rebinds: a slot that compared handles rather than descriptions would answer "already
  bound" and skip re-attaching a different texture. Its third frees the resource under a bound
  surface and clears again into a second colour buffer, which is what reads back once the first is
  gone. The attachment survives on its own -- deleting a texture releases its name, not the object
  an attachment still holds -- so what this case actually scores is everything the renderer does
  *around* the freed resource: a readback must put the framebuffer binding back the way it found
  it, and a clear must already know its colour fixup rather than ask a resource that is no longer
  there. Still unmeasured: *re-binding* a surface whose resource was freed, which the C serves from
  its refcount and this tree refuses.
- `vkr-record-decode.py` — decodes a venus full-stream capture (`--check` validates a capture
  structurally before it is pinned as a fixture, and reports how many records were recorded out of
  execution order — see the ordering rule in `src/venus/vkr_record.h`).

**Replay cannot reach the ring transport, so a seated boot is not a redundant gate.** The transport
commands drive a ring rather than travel on one, and every one of them is out of replay's reach --
by two distinct mechanisms, neither of which more replayer work would close.

`vkExecuteCommandStreamsMESA` is never recorded. It dispatches the commands it carries through the
same tee the recorder sits on, so writing the outer command as well would apply its contents twice
on replay (`src/venus/vkr_journal.c`). A corpus holds the flattened contents; the command itself
appears in none, so there is nothing to drive.

The `RING_FLOW_CONTROL` set -- `vkNotifyRingMESA`, `vkSubmitVirtqueueSeqnoMESA`,
`vkWaitVirtqueueSeqnoMESA`, `vkWaitRingSeqnoMESA` -- *is* recorded, and the replayer skips it on
purpose. Replay hands each command straight to the dispatcher with no ring buffer behind it, so
head, tail and seqno never move: a command meaning "the buffer advanced" is meaningless and a wait
on a seqno nothing will write blocks forever.

So a build can score every corpus perfectly and still refuse the first thing a live guest asks.
Replay stays the fine-grained oracle; booting is the only thing that scores the transport.

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

**Which renderer to score is never defaulted.** Both replay wrappers refuse to run until they are
told, by `--renderer rs|c` or by `VIRGL_PREFIX`, and each echoes the prefix it used. A default is
worse than an argument here: picking one silently produces a full, plausible score for whichever
implementation the caller forgot to name, and a score file for the wrong renderer reads exactly
like a regression in the right one.

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
