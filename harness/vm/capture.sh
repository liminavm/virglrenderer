#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Boot a rig guest with a recorder armed, so a real workload becomes a replay corpus.
#
#   capture.sh venus  [options]   enhanced guest, venus full-stream recorder (vkr_record)
#   capture.sh vrend  [options]   stock guest, classic command tracer (vrend_trace)
#
#   --mb N        recorder capacity, MB (default 256). The venus recorder STOPS at the cap and
#                 says so — a truncated corpus is a valid prefix, so a small cap costs coverage,
#                 never validity.
#   --window      show the guest in a window (default: headless, so a capture does not take over
#                 the screen)
#   --seconds N   dump automatically N seconds after boot, then leave the VM running
#   --            everything after is passed to the limina binary
#
# Dump at any time with dump.sh. The recorders write only when asked: arming one costs the render
# path a predictable branch, and nothing is written until a dump is requested.
set -euo pipefail
cd "$(dirname "$0")"
RIG="$(pwd)"

MODE="${1:-}"; shift || true
MB=256; WINDOW=0; SECONDS_TO_DUMP=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mb) MB="$2"; shift 2 ;;
    --window) WINDOW=1; shift ;;
    --seconds) SECONDS_TO_DUMP="$2"; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

APP="$RIG/Limina.app/Contents/MacOS/limina"
[ -x "$APP" ] || { echo "no rig — run make-rig.sh" >&2; exit 1; }

case "$MODE" in
  venus)
    DISK="$RIG/disks/Fedora-Workstation-44.enhanced.test.raw"
    export LIMINA_VKR_RECORD="$MB"
    export LIMINA_VKR_RECORD_OUT="$RIG/captures/venus.vkrc"
    export LIMINA_VKR_RECORD_FIFO="$RIG/captures/venus.fifo"
    OUT="$LIMINA_VKR_RECORD_OUT" ;;
  vrend)
    DISK="$RIG/disks/Fedora-Workstation-44.stock.test.raw"
    export LIMINA_VREND_TRACE="$MB"
    export LIMINA_VREND_TRACE_OUT="$RIG/captures/vrend.bin"
    export LIMINA_VREND_TRACE_FIFO="$RIG/captures/vrend.fifo"
    OUT="$LIMINA_VREND_TRACE_OUT" ;;
  *) echo "usage: capture.sh {venus|vrend} [--mb N] [--window] [--seconds N]" >&2; exit 2 ;;
esac

[ -f "$DISK" ] || { echo "missing disk: $DISK — run make-rig.sh" >&2; exit 1; }
mkdir -p "$RIG/captures"

ARGS=(--disk "$DISK" --net)
if [ "$WINDOW" = 1 ]; then
  ARGS+=(--window)
else
  # A headless boot still drives the whole renderer: the display sink reads the scanout
  # IOSurface back rather than presenting it, so venus and vrend see the same traffic.
  ARGS+=(--firmware "$RIG/Limina.app/Contents/Resources/KRUN_EFI.gop.fd")
fi

echo "==> $MODE capture, ${MB} MB -> $OUT"
echo "    dump with: $RIG/dump.sh $MODE"

if [ -n "$SECONDS_TO_DUMP" ]; then
  ( sleep "$SECONDS_TO_DUMP"; "$RIG/dump.sh" "$MODE" ) &
fi

exec "$APP" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
