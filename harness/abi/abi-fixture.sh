#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 the limina authors
#
# Pin, and check, the C ABI: the symbols the dylib exports and the layout of every struct that
# crosses it. Both are things a Rust port declares by hand on its own side, and both fail in a way
# nothing else in the harness sees -- a missing symbol only shows at dlopen, a wrong offset never
# shows at all, it just corrupts.
#
#   abi-fixture.sh            check the current build against the pinned fixtures
#   abi-fixture.sh --pin      re-record them (do this only when the ABI is meant to change)
#
# VIRGL_PREFIX selects the build under test; it defaults to this tree's rig prefix, so pointing it
# at a Rust build is how the port gets checked.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
ROOT="$(cd ../.. && pwd)"

PIN=0
[ "${1:-}" = "--pin" ] && PIN=1

PREFIX="${VIRGL_PREFIX:-$ROOT/harness/vm/prefix}"

# Own the freshness of the prefix we are about to score, the way vkr-replay.sh does. This gate
# reads a built dylib and nothing else, so without this it happily pins or checks a dylib from
# whatever the tree looked like when someone last installed -- and reports "symbols matches" about
# a build that no longer exists. The replay ladder already learned this once; there is no reason
# for the ABI gate to learn it again.
if [ "$PREFIX" = "$ROOT/virglrs/prefix" ]; then
  "$ROOT/virglrs/install.sh" >/dev/null
fi

LIB=""
for cand in "$PREFIX/lib/libvirglrenderer.1.dylib" "$PREFIX/lib/libvirglrenderer.dylib" \
            "$PREFIX/src/libvirglrenderer.dylib"; do
  [ -f "$cand" ] && LIB="$cand" && break
done
[ -n "$LIB" ] || { echo "no libvirglrenderer under $PREFIX" >&2; exit 1; }

# Any other prefix is built by someone else, so all we can do is say when it looks stale.
if [ "$PREFIX" != "$ROOT/virglrs/prefix" ] \
   && find "$ROOT/src" -name '*.c' -newer "$LIB" 2>/dev/null | read -r _; then
  echo "warning: $LIB is older than the C sources it was built from -- stale build" >&2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Every exported symbol, not an intersection with any one consumer: what the dylib exports is
# defined by this tree, while who calls what is defined outside it and moves without warning.
# A port that exports the whole list satisfies every consumer of it.
nm -gU "$LIB" | awk '$2 == "T" || $2 == "S" { print $3 }' | LC_ALL=C sort -u > "$TMP/symbols.txt"

cc -O0 -o "$TMP/abi-dump" "$HERE/abi-dump.c" -I"$ROOT/src" -I"$ROOT/harness/vm/build/src"
"$TMP/abi-dump" > "$TMP/layout.txt"

fail=0
for f in symbols layout; do
  if [ "$PIN" = 1 ]; then
    cp "$TMP/$f.txt" "$HERE/$f.txt"
    echo "pinned $f.txt ($(wc -l < "$TMP/$f.txt" | tr -d ' ') lines)"
  elif diff -u "$HERE/$f.txt" "$TMP/$f.txt"; then
    echo "$f matches"
  else
    echo "ABI FIXTURE MISMATCH: $f" >&2
    fail=1
  fi
done
exit $fail
