#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 the limina authors
#
# Replay a classic (vrend) corpus against a renderer build, with no VM.
#
#   vrend-replay.sh <corpus.bin> [replayer options]
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

[ $# -ge 1 ] || { echo "usage: vrend-replay.sh <corpus.bin> [options]" >&2; exit 2; }
case "$1" in /*) CORPUS="$1" ;; *) CORPUS="$PWD/$1" ;; esac
shift

# Resolve --score/--expect against the caller's directory: everything below runs from the script's.
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
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

VIRGL_PREFIX="${VIRGL_PREFIX:-$ROOT/harness/vm/prefix}" "$HERE/build.sh" >/dev/null

# zink needs the Vulkan loader on the dyld path: it dlopens @rpath/libvulkan.1.dylib, and the
# replayer is not the app bundle whose rpath would resolve it.
exec env \
  DYLD_LIBRARY_PATH="$MESA_PREFIX/lib:$EPOXY_PREFIX/lib:$VULKAN_LIB" \
  VK_ICD_FILENAMES="$ICD" \
  MESA_LOADER_DRIVER_OVERRIDE=zink \
  GALLIUM_DRIVER=zink \
  "$HERE/vrend-replay" "$CORPUS" "$@"
