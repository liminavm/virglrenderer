#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Replay a venus corpus against a renderer build, with no VM.
#
#   vkr-replay.sh <corpus.vkrc> [--renderer <libvirglrenderer.dylib>] [replayer options]
#
# --score <file> writes the score; --expect <file> compares against a pinned one and exits
# non-zero on any difference. The score is renderer state, not pixels: a VM-free replay has no
# scanout, so what it compares is the accept counts and the device memory the commands left.
#
# The renderer defaults to this tree's rig build (harness/vm/prefix). Point it at another build to
# compare implementations against the same corpus -- which is the entire reason this layer exists.
#
# venus needs a host Vulkan driver and nothing else: no EGL, no GL, no winsys. On macOS that is
# KosmicKrisp, taken from the same Mesa prefix the renderer was built against.
set -euo pipefail

# Resolve the corpus against the CALLER's directory before moving to our own.
CORPUS="${1:-}"
[ -n "$CORPUS" ] || { echo "usage: vkr-replay.sh <corpus.vkrc> [options]" >&2; exit 1; }
case "$CORPUS" in /*) ;; *) CORPUS="$(pwd)/$CORPUS" ;; esac
shift

# Resolve --score/--expect against the caller's directory too: everything below runs from the
# script's own, and a relative golden path would otherwise land somewhere the caller cannot see.
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
ICD="$MESA_PREFIX/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json"
[ -f "$ICD" ] || { echo "no KosmicKrisp ICD at $ICD (set MESA_PREFIX)" >&2; exit 1; }

BIN="$HERE/rs/target/release/vkr-replay"
cargo build --release --manifest-path "$HERE/rs/Cargo.toml" >/dev/null 2>&1

# VIRGL_PREFIX selects the implementation under test, the same variable harness/abi and the vrend
# replayer's build.sh use. It defaults to this tree's rig build; pointing it at the Rust prefix is
# how the port gets driven. An explicit --renderer still wins.
VIRGL_PREFIX="${VIRGL_PREFIX:-$ROOT/harness/vm/prefix}"
case " $* " in
  *" --renderer "*) RENDERER=() ;;
  *)
    LIB=""
    for cand in "$VIRGL_PREFIX/lib/libvirglrenderer.1.dylib" \
                "$VIRGL_PREFIX/lib/libvirglrenderer.dylib"; do
      [ -f "$cand" ] && LIB="$cand" && break
    done
    [ -n "$LIB" ] || { echo "no libvirglrenderer under $VIRGL_PREFIX" >&2; exit 1; }
    # This script builds the replayer, never the renderer -- so a prefix laid down by an earlier
    # install.sh scores whatever was in the tree then, silently. That has already put a claim in
    # harness/README.md that was measured against a build three commits old. For the Rust prefix
    # the fix is to build it; for any other, say so and let the caller decide.
    if [ "$VIRGL_PREFIX" = "$ROOT/virglrs/prefix" ]; then
      "$ROOT/virglrs/install.sh" >/dev/null
    elif find "$ROOT/src" -name '*.c' -newer "$LIB" 2>/dev/null | read -r _; then
      echo "warning: $LIB is older than the C sources it was built from -- stale build" >&2
    fi
    RENDERER=(--renderer "$LIB")
    ;;
esac

exec env VK_ICD_FILENAMES="$ICD" "$BIN" "$CORPUS" "${RENDERER[@]+"${RENDERER[@]}"}" "$@"
