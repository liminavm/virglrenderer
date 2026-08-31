#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 Gustavo Noronha Silva
#
# Build vrend-replay against a chosen virglrenderer prefix.
#
# The prefix is the whole point: this tool exists to run the SAME corpus against two
# implementations, so which library it links is a parameter, not a constant. Linking a stock
# system virglrenderer would produce a replayer that runs and reports nothing meaningful —
# `otool -L` afterwards is the check, exactly as for the limina worker.
#
#   VIRGL_PREFIX=/path/to/prefix ./build.sh
#
# Defaults to this tree's own build/ if it has been configured, else the limina C prefix.
set -eu
cd "$(dirname "$0")"
SRC="$(cd ../.. && pwd)"

PREFIX="${VIRGL_PREFIX:-}"
if [ -z "$PREFIX" ]; then
  if [ -f "$SRC/build/src/libvirglrenderer.dylib" ]; then
    PREFIX="$SRC/build"
  else
    PREFIX="$HOME/Projects/limina/third_party/virgl-prefix"
  fi
fi

for cand in "$PREFIX/lib/libvirglrenderer.dylib" "$PREFIX/src/libvirglrenderer.dylib"; do
  [ -f "$cand" ] && LIBDIR="$(dirname "$cand")" && break
done
[ -n "${LIBDIR:-}" ] || { echo "no libvirglrenderer.dylib under $PREFIX — build it first" >&2; exit 1; }

cc -O2 -Wall -Wextra -o vrend-replay vrend-replay.c \
   -I"$SRC/src" -I"$SRC/build/src" -I"$PREFIX/include/virgl" \
   -L"$LIBDIR" -lvirglrenderer -Wl,-rpath,"$LIBDIR"

echo "built: $PWD/vrend-replay"
otool -L vrend-replay | grep virgl
