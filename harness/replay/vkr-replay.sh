#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Replay a venus corpus against a renderer build, with no VM.
#
#   vkr-replay.sh <corpus.vkrc> --renderer rs|c|<libvirglrenderer.dylib> [replayer options]
#
# --score <file> writes the score; --expect <file> compares against a pinned one and exits
# non-zero on any difference. The score is renderer state, not pixels: a VM-free replay has no
# scanout, so what it compares is the accept counts and the device memory the commands left.
#
# Which renderer to run is mandatory: --renderer rs (the Rust tree) or c (the reference C), or a
# dylib path of your own, or VIRGL_PREFIX set to a prefix. There is deliberately no default. A
# wrapper that picks one silently scores whichever renderer the caller forgot to name, and the
# result is a full, plausible score file for the wrong implementation -- which reads exactly like
# a regression in the one you meant. Running both against one corpus is the entire reason this
# layer exists, so naming which is not a burden.
#
# venus needs a host Vulkan driver and nothing else: no EGL, no GL, no winsys. On macOS that is
# KosmicKrisp, taken from the same Mesa prefix the renderer was built against.
set -euo pipefail

# Resolve the corpus against the CALLER's directory before moving to our own.
CORPUS="${1:-}"
usage() {
  echo "usage: vkr-replay.sh <corpus.vkrc> --renderer rs|c|<dylib> [options]" >&2
  echo "       (or set VIRGL_PREFIX to a prefix; there is no default)" >&2
  exit 2
}
[ -n "$CORPUS" ] || usage
case "$CORPUS" in /*) ;; *) CORPUS="$(pwd)/$CORPUS" ;; esac
shift

# Resolve --score/--expect against the caller's directory too: everything below runs from the
# script's own, and a relative golden path would otherwise land somewhere the caller cannot see.
ARGS=()
CHOICE=""
# The snapshot-journal gate runs by default on the Rust tree, the same as the classic replayer's.
# A gate that has to be remembered is a gate that stops being run. --no-rebuild opts out; the C
# tree is not asked, because journal_held is a virglrs extension it does not export.
REBUILD=1
ASKED=0
AT=0        # whether the caller named --rebuild-at, which already implies --rebuild
while [ $# -gt 0 ]; do
  case "$1" in
    --renderer)
      CHOICE="${2:-}"
      [ -n "$CHOICE" ] || { echo "--renderer wants rs, c, or a dylib path" >&2; exit 2; }
      shift 2 ;;
    --score|--expect)
      case "${2:-}" in
        /*) ARGS+=("$1" "$2") ;;
        "") echo "$1 wants a path" >&2; exit 2 ;;
        *)  ARGS+=("$1" "$PWD/$2") ;;
      esac
      shift 2 ;;
    --rebuild)    REBUILD=1; ASKED=1; shift ;;
    --no-rebuild) REBUILD=0; shift ;;
    --rebuild-at)
      [ -n "${2:-}" ] || { echo "--rebuild-at wants a command count" >&2; exit 2; }
      AT=1; ASKED=1; ARGS+=("$1" "$2"); shift 2 ;;
    --smoke)      REBUILD=0; ARGS+=("$1"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
# The C reference has no journal_held and the replayer resolves the rest of the journal ABI
# eagerly, so asking it to rebuild reports about the ABI rather than about the corpus. Refuse
# rather than quietly skip: a gate that silently does not run is worse than one not asked for.
if [ "$CHOICE" = c ]; then
  if [ "$ASKED" = 1 ]; then
    echo "--rebuild needs --renderer rs (the C tree has no journal_held)" >&2; exit 2
  fi
  REBUILD=0
fi
if [ "$REBUILD" = 1 ] && [ "$AT" = 0 ]; then
  ARGS+=(--rebuild)
fi
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
# replayer's build.sh use. --renderer names it more directly and wins where both are given.
case "$CHOICE" in
  rs) VIRGL_PREFIX="$ROOT/virglrs/prefix" ;;
  c)  VIRGL_PREFIX="$ROOT/harness/vm/prefix" ;;
  "") [ -n "${VIRGL_PREFIX:-}" ] || usage ;;
  *)  VIRGL_PREFIX="" ;;   # a dylib path, used as given
esac

if [ -n "${VIRGL_PREFIX:-}" ]; then
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
else
  LIB="$CHOICE"
  [ -f "$LIB" ] || { echo "no renderer at $LIB" >&2; exit 1; }
fi
echo "replay: $LIB" >&2
RENDERER=(--renderer "$LIB")

exec env VK_ICD_FILENAMES="$ICD" "$BIN" "$CORPUS" "${RENDERER[@]+"${RENDERER[@]}"}" "$@"
