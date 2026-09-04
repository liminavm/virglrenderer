#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Boot a rig guest with a recorder armed, so a real workload becomes a replay corpus.
#
#   capture.sh synoik [options]   synoik (a Vulkan compositor), venus recorder — the desktop
#                                 workload that is venus end to end
#   capture.sh venus  [options]   enhanced GNOME guest, venus recorder. Its shell renders through
#                                 classic virgl, so venus traffic here comes from Vulkan clients
#   capture.sh vrend  [options]   stock guest, classic command tracer (vrend_trace)
#   capture.sh video  [options]   enhanced guest, classic command tracer. The video codecs the
#                                 stock guest cannot reach: its mesa is built
#                                 -Dvideo-codecs=all_free and the VA frontend enforces that
#                                 driver-independently, so a decode corpus has to come from here
#
#   --mb N        recorder capacity, MB (default 512). The venus recorder STOPS at the cap and
#                 says so — a truncated corpus is a valid prefix, so a small cap costs coverage,
#                 never validity. The default holds a desktop's 64 MiB wallpaper upload, which a
#                 smaller one drops: the corpus still replays, as a flat-colour desktop that
#                 measures none of it.
#   --out NAME    write captures/NAME.vkrc instead of captures/<mode>.vkrc, and dump it with
#                 `dump.sh NAME`. A corpus is pinned by the score recorded from it, so a second
#                 capture under the same name silently replaces what a fixture was measured
#                 against — give a new capture its own name rather than the one already spoken
#                 for. `synoik` and `synoik-lifecycle` are two such corpora of one workload.
#                 A vrend capture is named vrend-NAME and written as a .bin.
#   --window      show the guest in a window (default: headless, so a capture does not take over
#                 the screen). Headless still attaches a virtio-gpu and drives the whole renderer
#                 -- presented frames go to a PNG instead of a window. Omitting BOTH would attach
#                 no GPU at all and record nothing, which is what --display-capture prevents.
#   --seconds N   dump automatically N seconds after boot, then leave the VM running
#   --renderer R  which rig bundle to boot: c (default) or rust
#   --            everything after is passed to the limina binary
#
# Dump at any time with dump.sh. The recorders write only when asked: arming one costs the render
# path a predictable branch, and nothing is written until a dump is requested.
set -euo pipefail
cd "$(dirname "$0")"
RIG="$(pwd)"

MODE="${1:-}"; shift || true
MB=512; WINDOW=0; SECONDS_TO_DUMP=""; NAME=""; RENDERER=c
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mb) MB="$2"; shift 2 ;;
    --out) NAME="$2"; shift 2 ;;
    --window) WINDOW=1; shift ;;
    --seconds) SECONDS_TO_DUMP="$2"; shift 2 ;;
    --renderer) RENDERER="$2"; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

case "$RENDERER" in
  c)    BUNDLE="$RIG/Limina.app" ;;
  rust) BUNDLE="$RIG/Limina-rust.app" ;;
  *) echo "unknown renderer: $RENDERER (c|rust)" >&2; exit 2 ;;
esac
APP="$BUNDLE/Contents/MacOS/limina"
[ -x "$APP" ] || { echo "no rig — run make-rig.sh --renderer $RENDERER" >&2; exit 1; }

NAME="${NAME:-$MODE}"

case "$MODE" in
  synoik|venus)
    [ "$MODE" = synoik ] \
      && DISK="$RIG/disks/Fedora-Workstation-44.enhanced.synoik.raw" \
      || DISK="$RIG/disks/Fedora-Workstation-44.enhanced.test.raw"
    export LIMINA_VKR_RECORD="$MB"
    export LIMINA_VKR_RECORD_OUT="$RIG/captures/$NAME.vkrc"
    export LIMINA_VKR_RECORD_FIFO="$RIG/captures/$NAME.fifo"
    OUT="$LIMINA_VKR_RECORD_OUT" ;;
  vrend|video)
    [ "$MODE" = video ] \
      && DISK="$RIG/disks/Fedora-Workstation-44.enhanced.test.raw" \
      || DISK="$RIG/disks/Fedora-Workstation-44.stock.test.raw"
    # A classic corpus is named vrend*, which is how dump.sh knows to write a .bin.
    case "$NAME" in vrend*) ;; *) NAME="vrend-$NAME" ;; esac
    export LIMINA_VREND_TRACE="$MB"
    export LIMINA_VREND_TRACE_OUT="$RIG/captures/$NAME.bin"
    export LIMINA_VREND_TRACE_FIFO="$RIG/captures/$NAME.fifo"
    OUT="$LIMINA_VREND_TRACE_OUT" ;;
  *) echo "usage: capture.sh {synoik|venus|vrend|video} [--mb N] [--out NAME] [--window] [--seconds N]" >&2
     exit 2 ;;
esac

[ -f "$DISK" ] || { echo "missing disk: $DISK — run make-rig.sh" >&2; exit 1; }
mkdir -p "$RIG/captures"

ARGS=(--disk "$DISK" --net)
if [ "$WINDOW" = 1 ]; then
  ARGS+=(--window)
else
  # --display-capture is what makes a headless boot still a GRAPHICS boot: it attaches the
  # virtio-gpu and writes presented frames to a PNG rather than a window. A boot with neither
  # --window nor --display-capture gets no virtio-gpu at all -- the guest comes up with an empty
  # /sys/class/drm, no venus context is ever created, and the recorder never even arms.
  ARGS+=(--display-capture "$RIG/captures/$MODE-frame.png"
         --display-size "${LIMINA_DISPLAY_SIZE:-1280x800}"
         --firmware "$BUNDLE/Contents/Resources/KRUN_EFI.gop.fd")
fi

echo "==> $MODE capture, ${MB} MB -> $OUT"
echo "    dump with: $RIG/dump.sh $NAME"

if [ -n "$SECONDS_TO_DUMP" ]; then
  ( sleep "$SECONDS_TO_DUMP"; "$RIG/dump.sh" "$NAME" ) &
fi

exec "$APP" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
