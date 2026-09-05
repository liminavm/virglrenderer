#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 the limina authors
#
# Replay a classic (vrend) corpus against a renderer build, with no VM.
#
#   vrend-replay.sh <corpus.bin> --renderer rs|c [replayer options]
#
# Which renderer to run is mandatory: pass --renderer rs (the Rust tree) or --renderer c (the
# reference C), or set VIRGL_PREFIX to a prefix yourself. There is deliberately no default. A
# wrapper that picks one silently scores whichever renderer the caller forgot to name, and the
# result is a full, plausible score file for the wrong implementation -- which reads exactly like
# a regression in the one you meant.
#
# The environment is the point of this wrapper. vrend runs on zink over KosmicKrisp through an
# epoxy built WITH EGL, and none of that is on the default loader path: a golden recorded under an
# environment that lives in someone's shell history is not reproducible. Set MESA_PREFIX or
# EPOXY_PREFIX to move either one.
#
# --score <file> writes the score; --expect <file> compares against a pinned one and exits
# non-zero. The score is the readback sequence -- content hashes and ink counts, in stream order
# -- plus the run counters, so diff is the whole comparison tool.
set -euo pipefail

usage() {
  echo "usage: vrend-replay.sh <corpus.bin> --renderer rs|c [options]" >&2
  echo "       (or set VIRGL_PREFIX to a prefix; there is no default)" >&2
  exit 2
}
[ $# -ge 1 ] || usage
case "$1" in /*) CORPUS="$1" ;; *) CORPUS="$PWD/$1" ;; esac
shift

# Resolve --score/--expect against the caller's directory: everything below runs from the script's.
ARGS=()
RENDERER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --renderer)
      case "${2:-}" in
        rs|c) RENDERER="$2" ;;
        *) echo "--renderer wants rs or c" >&2; exit 2 ;;
      esac
      shift 2 ;;
    --score|--expect)
      case "${2:-}" in
        /*) ARGS+=("$1" "$2") ;;
        "") echo "$1 wants a path" >&2; exit 2 ;;
        *)  ARGS+=("$1" "$PWD/$2") ;;
      esac
      shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

cd "$(dirname "$0")"
HERE="$(pwd)"
ROOT="$(cd ../.. && pwd)"

MESA_PREFIX="${MESA_PREFIX:-/Volumes/mesa-cs/zink-kk-prefix}"
EPOXY_PREFIX="${EPOXY_PREFIX:-/Users/kov/Projects/limina/third_party/epoxy-egl-prefix}"
VULKAN_LIB="${VULKAN_LIB:-/opt/homebrew/opt/vulkan-loader/lib}"
ICD="$MESA_PREFIX/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json"

[ -f "$ICD" ] || { echo "no KosmicKrisp ICD at $ICD (set MESA_PREFIX)" >&2; exit 1; }
# libEGL comes from Mesa; epoxy is the GL dispatch the renderer links, and Homebrew's is CGL-only.
[ -f "$MESA_PREFIX/lib/libEGL.dylib" ] || {
  echo "no libEGL at $MESA_PREFIX (set MESA_PREFIX)" >&2; exit 1; }
[ -f "$EPOXY_PREFIX/lib/libepoxy.0.dylib" ] || {
  echo "no epoxy-with-EGL at $EPOXY_PREFIX (set EPOXY_PREFIX)" >&2; exit 1; }

case "$RENDERER" in
  rs) VIRGL_PREFIX="$ROOT/virglrs/prefix" ;;
  c)  VIRGL_PREFIX="$ROOT/harness/vm/prefix" ;;
  "") [ -n "${VIRGL_PREFIX:-}" ] || usage ;;
esac
# Build what is about to be scored. This script builds the replayer, never the renderer, so a
# prefix laid down by an earlier install.sh scores whatever was in the tree then -- silently, and
# with a full plausible score. vkr-replay.sh already takes this precaution for the same reason.
if [ "$VIRGL_PREFIX" = "$ROOT/virglrs/prefix" ]; then
  "$ROOT/virglrs/install.sh" >/dev/null
fi
echo "replay: $VIRGL_PREFIX" >&2

VIRGL_PREFIX="$VIRGL_PREFIX" "$HERE/build.sh" >/dev/null

# zink needs the Vulkan loader on the dyld path: it dlopens @rpath/libvulkan.1.dylib, and the
# replayer is not the app bundle whose rpath would resolve it.
exec env \
  DYLD_LIBRARY_PATH="$MESA_PREFIX/lib:$EPOXY_PREFIX/lib:$VULKAN_LIB" \
  VK_ICD_FILENAMES="$ICD" \
  MESA_LOADER_DRIVER_OVERRIDE=zink \
  GALLIUM_DRIVER=zink \
  "$HERE/vrend-replay" "$CORPUS" "$@"
