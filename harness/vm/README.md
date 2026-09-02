# The capture rig

A self-contained way to boot a real Linux guest against **this tree's** renderer, so a real
workload becomes a replay corpus for `harness/replay/`. Nothing here is tracked by git except the
scripts and this file — the bundle, the disks, the build and the captures are all local artifacts.

```sh
./build-renderer.sh          # this tree -> harness/vm/prefix
./make-rig.sh                # clone limina's bundle + two disks, swap our renderer in
./capture.sh venus --mb 256  # boot the enhanced guest with the venus recorder armed
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
host. The two tiers are both here on purpose: the enhanced image boots the venus desktop, the
stock image exercises classic vrend and the VA-API video path.

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

A Vulkan client over SSH needs only `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/virtio_icd.aarch64.json`
plus `XDG_RUNTIME_DIR` and `WAYLAND_DISPLAY` for a windowed one. A **GL** client needs the session's
zink environment too, which an SSH shell does not inherit — without it the stack silently falls back
to llvmpipe and the capture records nothing while looking healthy.

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
./capture.sh synoik --mb 256 --out synoik-lifecycle    # boot; synoik starts itself
# let it render for a couple of minutes, then, in the guest:
#   ssh -p <port from the boot log> claude@127.0.0.1 \
#     'XDG_RUNTIME_DIR=/run/user/1000 systemctl --user stop graphical-session.target'
./dump.sh synoik-lifecycle
```

`--out` is what keeps the two apart. Without it every synoik capture writes `synoik.vkrc` and
replaces the corpus a pinned score was recorded from.

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
