# The capture rig

A self-contained way to boot a real Linux guest against **this tree's** renderer, so a real
workload becomes a replay corpus for `harness/replay/`. Nothing here is tracked by git except the
scripts and this file — the bundle, the disks, the build and the captures are all local artifacts.

```sh
./build-renderer.sh          # this tree -> harness/vm/prefix
./make-rig.sh                # clone limina's bundle + two disks, swap our renderer in
./capture.sh venus  # boot the enhanced guest with the venus recorder armed
./dump.sh venus              # ask for the dump, then check it
```

## What the rig is

`make-rig.sh` clones limina's signed `Limina.app`, replaces
`Contents/Frameworks/libvirglrenderer.1.dylib` with ours, rewrites its dependencies from absolute
paths back to `@rpath` so the bundle stays self-contained, and re-signs inside-out preserving each
binary's own entitlements (the worker needs `com.apple.security.hypervisor`).

Working against a copy rather than limina's own bundle is the point: capturing a corpus means
booting a guest against a renderer we are actively changing, and doing that in limina's tree would
make every capture a mutation of their working set.

The guest disks are **APFS clones** (`cp -c`). They cost no space until the guest writes, and the
source images are never touched — an ordinary copy of two ~15 GB images would not fit on this
host.

## What the three images are

The tiers are **not** graphics capability tiers, and reading them as such is how a bug gets
explained away as "the stock guest would never do that". Every foundational graphics capability is
in all three.

- **stock** — a stock Fedora Workstation. Renders through virgl/vrend, has venus available, and
  hardware acceleration works.
- **enhanced** — stock, plus a 16k-page-size kernel and fixes to guest-mesa issues we found. It
  adds no foundational graphics capability over stock.
- **synoik** — enhanced, with a Vulkan compositor in place of GNOME.

So a difference in what a workload does between stock and enhanced is a guest-mesa behaviour
difference or a workload difference, never "that tier can't". When a capture on one tier shows a
command the other's corpus lacks, the question to ask is what the two sessions *did* differently.

## The three guests, and what each one needs

All three images boot to **multi-user**, not to a seated desktop, so a capture that just boots
records nothing. What starts the workload differs per image, and so does which renderer it drives:

| `capture.sh` | guest | workload | drives |
|---|---|---|---|
| `synoik` | enhanced + synoik | starts on its own at boot | venus, end to end |
| `venus` | enhanced GNOME | `sudo systemctl isolate graphical.target`, then a Vulkan client (`vkcube`, `vulkaninfo`) over SSH | venus from clients only |
| `vrend` | stock | `sudo systemctl isolate graphical.target` | classic vrend, plus the video path |

The surprise worth keeping: on the **enhanced** image gnome-shell renders through **classic virgl**
(`GALLIUM_DRIVER=virgl` in its environ), not zink→venus. Its venus traffic comes from Vulkan
clients, which makes it the mixed case rather than the venus one. Synoik is the desktop workload
that is venus throughout.

## Pixels are the oracle

Everything else in this rig is a proxy: a systemd target, a `vulkaninfo` line, a renderer log,
a corpus score. Each answers a real question and none of them answers "is there a desktop on the
screen" -- a compositor that exits at startup leaves `graphical-session.target` active, a clean
renderer log and a corpus that replays. The frame is the only thing that settles it.

`frame.py [PNG...]` reports on a captured frame: dimensions, distinct colours, dominant colour and
its share, and a coarse luminance sketch, then the path. It prints no verdict on purpose -- a
"looks seated" line would be a new proxy, and a more dangerous one for sounding like it looked.
Use the numbers to know where to look and then open the file.

Two properties of the capture decide how to use it. It exists only for a **headless** boot, since
limina refuses `--display-capture` together with a window -- so a windowed boot's oracle is the
human in front of it, and a headless boot's is the file. And the file holds the **last presented
frame**, rewritten as frames arrive: read it while the workload runs and it is the current screen,
read it after shutdown and it is the teardown console. The `*-frame.png` files a capture leaves
behind are therefore poweroff screens and evidence of nothing.

Which makes one frame insufficient on its own: a compositor whose context died leaves the last
frame it presented sitting in the surface, and that frame looks exactly like a seated desktop.
**Two samples, and something in them that has to move.** The clock is the obvious candidate and
a poor one -- it changes once a minute, so two samples inside the same minute prove nothing.
Anything animating is better: a rotating `vkcube` in the frame settles both liveness and its own
import at once, since a cube at a different angle in the second sample is a client still drawing
and a compositor still sampling it.

`--renderer c|rust` picks which build to boot, and picks the bundle with it: `Limina.app` holds
the C, `Limina-rust.app` holds virglrs. Only the C records — the recorder is a C-tree feature, so a
virglrs boot yields no corpus and the two legs divide accordingly: the C leg captures, the Rust leg
is what the capture is then replayed against. A boot also reaches what replay cannot -- the ring
transport commands, for the reasons `../README.md` gives. Booting both on the same guest with the same client is
how a refusal is attributed: `vkcube` running its full 25 seconds on one and segfaulting on the
other is a gap in this tree, not in vkcube.

A Vulkan client over SSH needs only `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/virtio_icd.aarch64.json`
plus `XDG_RUNTIME_DIR` and `WAYLAND_DISPLAY` for a windowed one. A **GL** client needs the session's
zink environment too, which an SSH shell does not inherit — without it the stack silently falls back
to llvmpipe and the capture records nothing while looking healthy.

**For a video workload, drive Showtime or `waylandsink` — not `glimagesink`.** A `gst-launch-1.0`
pipeline ending in `glimagesink` decodes correctly and runs to EOS with rc=0, and never puts a
window on the screen. It does this on **both** renderer legs, so it is the sink and not the port;
the C leg's captured frame shows the same empty desktop after the same playback. It is worth
finding out why, because a decoder whose only visible failure is an absent window is a workload
that scores green while measuring nothing on screen — but until it is, a corpus recorded through
it carries the decode and not the presentation. Showtime drives the whole path, composite planar
decode target included, with the picture up.

Two parser notes that cost nothing to know: `matroskademux ! vavp9dec` fails to negotiate without
a `vp9parse` between them, and `ffplay` is not a route here at all — no `libopenh264`, and its
vaapi-from-vulkan derivation fails.

**A capture can tap keys.** `tap-keys.py` is `type-into-overview.py`'s other half: it taps named
keys through `/dev/uinput` rather than typing words (`sudo python3 /tmp/tap-keys.py esc`). A guest
boots into the overview, where every window is a still thumbnail and the compositor stops
presenting -- so a client that draws continuously records its first frames and nothing after, and
the captured frame stops being rewritten while everything else still looks healthy. Escaping to
the focused window is what makes the rest of the capture move.

## A WebGL client asking for MSAA takes the VM down

`webgl.html` is the browser workload, run as `firefox --kiosk file:///tmp/webgl.html`. Requesting
the WebGL context with `antialias:true` reliably kills the VM, and the sequence is worth knowing
because only its first step names the cause:

    MESA: error: ZINK: vkQueueSubmit failed (VK_ERROR_DEVICE_LOST)
    [LIMINA-ALLOC-POOL] class 0 grew to 65 allocators -- in-flight depth is outrunning completion
    ... to 7291 ...
    VM stopped -- worker terminated by signal 6

The device is lost first; after that nothing completes, so the allocator pool grows without bound
until the abort about a minute later. The runaway pool is the loud part and is only the symptom.

It is **not** ours, and the corpus must not be read as if it were: the same page kills the VM on
the C leg and on virglrs alike, and a desktop booted without the page runs indefinitely. It is
below virglrenderer, in the host's zink-on-KosmicKrisp path. Measured 2026-09-05 on alface: four
runs with MSAA, all dead inside ~50 s of the page loading; the identical page with
`antialias:false` runs on with `device-lost=0`.

So the recorded corpus asks for `antialias:false`. That is a deliberate retreat from the crashing
case, not a belief that MSAA works -- and the crash keeps a reproducer here rather than a fixture,
because a corpus of it would only ever record the seconds before the host gave up.

**A capture can type.** `type-into-overview.py` creates a keyboard on `/dev/uinput` and taps
keys through it, so mutter sees a real device and a headless capture can drive the overview's
search entry -- no window, no human. Copy it into the guest and run it as root
(`sudo python3 /tmp/type.py firefox settings`). It is how `vrend-overview.bin` was recorded, and
that corpus exists because typing allocates a resource nothing else in the tree did.

## Two synoik corpora, and why one cannot do both jobs

The synoik guest yields two corpora, and they measure different things because a capture cannot
carry both properties at once:

| | `synoik.vkrc` | `synoik-lifecycle.vkrc` |
|---|---|---|
| dumped | mid-workload, compositor still running | after the session is stopped |
| teardown command kinds | 9 | 19 — `vkDestroyDevice`, `vkDestroyInstance`, `vkDestroyRingMESA`, the pipeline, render-pass, pool and layout destroys |
| `vkFreeMemory` | 4 | 41 |
| device allocations left to census | 22 | 0 |

The census hashes device memory that is still live, so it can only score what the workload has
not freed. A workload that exits cleanly frees everything — which is exactly what makes the
lifecycle corpus prove teardown, and exactly why it has no content hashes to offer. Keep both:
`synoik.score` is the content oracle, `synoik-lifecycle.score` is the teardown one, and a renderer
that leaks a `VkDeviceMemory` fails the second while passing the first.

### Capturing the lifecycle one

synoik runs as the user unit `org.gnome.Shell@user.service`, which sets `RefuseManualStop=yes`, as
do `gnome-session@gnome.target` and `gnome-session.target`. The one stoppable handle is
`graphical-session.target`; synoik's SIGTERM handler quits its loop outright rather than waiting
for clients, so stopping the target runs its drops and puts real `vkDestroy*` traffic on the wire
instead of a bare fd close at exit.

```sh
./capture.sh synoik --out synoik-lifecycle       # boot; synoik starts itself
# let it render for a couple of minutes, then, in the guest:
#   ssh -p <port from the boot log> claude@127.0.0.1 \
#     'XDG_RUNTIME_DIR=/run/user/1000 systemctl --user stop graphical-session.target'
./dump.sh synoik-lifecycle
```

`--out` is what keeps the two apart. Without it every synoik capture writes `synoik.vkrc` and
replaces the corpus a pinned score was recorded from.

## Client corpora, and why the C cannot score them

Two more corpora come from the synoik guest with a Vulkan client in front of the compositor:

| | `synoik-vkcube.vkrc` | `synoik-ptyxis.vkrc` |
|---|---|---|
| client | `vkcube --wsi wayland` | Ptyxis (GTK4, `GSK_RENDERER=vulkan`), rendering a long directory listing |
| adds to the other corpora | the cross-context import: a compositor reaching a client's buffer; `vkCmdCopyImageToBuffer` | `vkGetPipelineCacheData`, both calls -- an out-blob, the guest saving its pipeline cache; `vkFreeDescriptorSets` |
| dumped | mid-workload, client still running | mid-workload, after the guest wrote `~/.cache/gtk-4.0/vulkan-pipeline-cache` |

Neither has a fixture under `../replay/fixtures`, because the C build cannot replay them: the
client's export and the compositor's import are recorded out of execution order, the C's import
of the not-yet-created resource tombstones the image, and KosmicKrisp asserts on the next
descriptor that samples it. So they gate virglrs against its own previous build -- the score at
HEAD against the score with the change -- rather than against the C. What a score sees of the
commands they add is that each decodes and is served to the end of the stream: a build that
refuses or poisons on one shows as `cmds` short of the total. A reply's *shape* it cannot see,
because a recording holds requests only -- an out-blob decoded as absent replays to a
byte-identical score, and that half is the unit tests' to pin (`../sabotage/sweep.py` lists
which).

To capture the Ptyxis one:

```sh
./capture.sh synoik --renderer c --out synoik-ptyxis
# in the guest, once /run/user/1000/wayland-1 exists:
#   XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-1 GSK_RENDERER=vulkan \
#     ptyxis -- sh -c 'ls -lR /usr/share/icons | head -n 4000; sleep 300'
# wait for a new file under ~/.cache/gtk-4.0/vulkan-pipeline-cache, then
./dump.sh synoik-ptyxis
```

GTK's gpu renderer creates no query pools, so no corpus carries the query-pool commands; they
are pinned by unit tests alone.

## Capturing

Both recorders are armed by capacity and write only when asked, through a FIFO — so arming one
costs the render path a predictable branch and nothing is written until `dump.sh` asks. A capture
defaults to **headless**, which still drives the whole renderer (the display sink reads the
scanout IOSurface back instead of presenting it) without taking over the screen; `--window` shows
the guest.

Venus captures stop at the cap and say so in the header. That is deliberate: a truncated corpus is
a valid **prefix** and still replays from its own beginning, where a FIFO ring would keep the most
recent commands and be replayable from nowhere. A small cap costs coverage, never validity.

`dump.sh` runs the decoder's structural check on what came back. Run it before pinning anything as
a fixture — a lost head record, an orphan context or a miscounted header all read as a working
capture right up until something tries to replay it.

## What the rig still borrows from limina

Two host prefixes, because reproducing them here would mean vendoring two more forks:

- an epoxy built **with** EGL (Homebrew's is CGL-only), from `third_party/epoxy-egl-prefix`;
- the zink-on-KosmicKrisp Mesa on its case-sensitive volume, `/Volumes/mesa-cs/zink-kk-prefix`.

It also seeds itself from limina's built `Limina.app` and its disk images. `LIMINA_ROOT` overrides
where it looks. Settling these is part of the P6 reconcile in `docs/rust-rewrite.md`.
