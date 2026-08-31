#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Replay a venus corpus against a renderer build, with no VM.
#
#   vkr-replay.sh <corpus.vkrc> [--renderer <libvirglrenderer.dylib>] [replayer options]
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

cd "$(dirname "$0")"
HERE="$(pwd)"
ROOT="$(cd ../.. && pwd)"

MESA_PREFIX="${MESA_PREFIX:-/Volumes/mesa-cs/zink-kk-prefix}"
ICD="$MESA_PREFIX/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json"
[ -f "$ICD" ] || { echo "no KosmicKrisp ICD at $ICD (set MESA_PREFIX)" >&2; exit 1; }

BIN="$HERE/rs/target/release/vkr-replay"
cargo build --release --manifest-path "$HERE/rs/Cargo.toml" >/dev/null 2>&1

case " $* " in
  *" --renderer "*) RENDERER=() ;;
  *) RENDERER=(--renderer "$ROOT/harness/vm/prefix/lib/libvirglrenderer.1.dylib") ;;
esac

exec env VK_ICD_FILENAMES="$ICD" "$BIN" "$CORPUS" "${RENDERER[@]}" "$@"
