#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Build a self-contained VM rig under harness/vm: limina's app bundle with THIS tree's
# virglrenderer swapped in, plus APFS clones of a couple of guest disks.
#
# WHY A COPY AND NOT limina's OWN BUNDLE. Capturing a corpus means booting a real guest against a
# renderer we are actively changing. Doing that in limina's tree would make every capture a
# mutation of their working set — and the whole point of this repository right now is that the
# rewrite is contained. The rig is ours to break.
#
# The disks are APFS clones (cp -c): they cost no space until the guest writes, and the originals
# are never touched. Copying 15 GB per image the ordinary way would not fit on this host.
#
# Two renderers are under test in this tree, so the rig is two bundles. `--renderer` picks one,
# and it picks the build and the bundle together: a bundle named for one tree holding the other
# tree's dylib is a rig that lies about what it booted, and nothing downstream could tell.
#
# Usage: make-rig.sh [--disks-only | --bundle-only] [--renderer c|rust]
set -euo pipefail
cd "$(dirname "$0")"
RIG="$(pwd)"
cd ../..
ROOT="$(pwd)"

LIMINA="${LIMINA_ROOT:-$HOME/Projects/limina}"
SRC_APP="$LIMINA/target/Limina.app"
DISKS="$RIG/disks"

# Three guests, because the corpus needs three different renderers exercised:
#   synoik    a Vulkan compositor — the desktop workload that is venus end to end
#   enhanced  GNOME, whose shell runs on classic virgl (GALLIUM_DRIVER=virgl); venus here comes
#             from Vulkan clients, so it is the mixed vrend+venus case
#   stock     an unmodified guest: classic vrend and the VA-API video path
SRC_DISKS=(
  "$LIMINA/Fedora-Workstation-44.enhanced.synoik.raw"
  "$LIMINA/Fedora-Workstation-44.enhanced.test.raw"
  "$LIMINA/Fedora-Workstation-44.stock.test.raw"
)

want_bundle=1 want_disks=1 renderer=c
while [ $# -gt 0 ]; do
  case "$1" in
    --disks-only) want_bundle=0; shift ;;
    --bundle-only) want_disks=0; shift ;;
    --renderer) renderer="$2"; shift 2 ;;
    *) echo "usage: make-rig.sh [--disks-only|--bundle-only] [--renderer c|rust]" >&2; exit 2 ;;
  esac
done

case "$renderer" in
  c)    PREFIX="$ROOT/harness/vm/prefix"; APP="$RIG/Limina.app" ;;
  rust) PREFIX="$ROOT/virglrs/prefix";    APP="$RIG/Limina-rust.app" ;;
  *) echo "unknown renderer: $renderer (c|rust)" >&2; exit 2 ;;
esac

if [ "$want_disks" = 1 ]; then
  mkdir -p "$DISKS"
  for src in "${SRC_DISKS[@]}"; do
    [ -f "$src" ] || { echo "missing source disk: $src" >&2; exit 1; }
    dst="$DISKS/$(basename "$src")"
    if [ -f "$dst" ]; then
      echo "==> keeping existing $(basename "$dst") (delete it to re-clone)"
      continue
    fi
    # -c is the whole point: an APFS clone, not a copy. Without it this is 15 GB per image.
    echo "==> cloning $(basename "$src")"
    cp -c "$src" "$dst"
  done
fi

[ "$want_bundle" = 1 ] || exit 0

DYLIB="$PREFIX/lib/libvirglrenderer.1.dylib"
[ -f "$DYLIB" ] || {
  case "$renderer" in
    c)    echo "build it first: harness/vm/build-renderer.sh" >&2 ;;
    rust) echo "build it first: virglrs/install.sh" >&2 ;;
  esac
  exit 1
}
[ -d "$SRC_APP" ] || { echo "missing source bundle: $SRC_APP (cargo xtask app in limina)" >&2; exit 1; }

echo "==> cloning $SRC_APP"
rm -rf "$APP"
cp -Rc "$SRC_APP" "$APP"

FW="$APP/Contents/Frameworks"
MACOS="$APP/Contents/MacOS"

echo "==> swapping in $(basename "$DYLIB")"
cp "$DYLIB" "$FW/libvirglrenderer.1.dylib"

# Our build links its dependencies by absolute path; the bundle reaches them through @rpath. Left
# as built, the bundled worker would load epoxy/vulkan/dav1d from OUTSIDE the bundle — which
# happens to work on this host and would silently stop being the thing under test the moment one
# of those prefixes moved.
install_name_tool -id "@rpath/libvirglrenderer.1.dylib" "$FW/libvirglrenderer.1.dylib"
otool -L "$FW/libvirglrenderer.1.dylib" | awk 'NR>1 {print $1}' | while read -r dep; do
  case "$dep" in
    /Users/*|/Volumes/*|/opt/homebrew/*)
      install_name_tool -change "$dep" "@rpath/$(basename "$dep")" "$FW/libvirglrenderer.1.dylib"
      echo "    $dep -> @rpath/$(basename "$dep")" ;;
  esac
done

# Anything we rewrote to @rpath must actually BE in Frameworks, or the worker fails to launch with
# a dyld error that reads like a signing problem. Check now, while the cause is obvious.
missing=0
otool -L "$FW/libvirglrenderer.1.dylib" | awk 'NR>1 && $1 ~ /^@rpath\// {print $1}' | while read -r dep; do
  [ -f "$FW/${dep#@rpath/}" ] || { echo "    MISSING in bundle: $dep" >&2; missing=1; }
  [ "$missing" = 0 ] || exit 1
done

# Re-sign inside-out, exactly as build-app.sh does: the bundle seal covers Frameworks, so
# replacing a file there invalidates the app whether or not the worker validates libraries.
SIGN_ID="${LIMINA_SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null \
          | sed -n 's/^ *1) [0-9A-F]* "\(.*\)"$/\1/p' | head -1)}"
[ -n "$SIGN_ID" ] || SIGN_ID="-"
echo "==> re-signing as '$SIGN_ID'"

ENT="$(mktemp -t limina-ent).plist"
sign_like() {  # preserve each binary's own entitlements rather than inventing them
  local bin="$1"
  if codesign -d --entitlements "$ENT" --xml "$bin" >/dev/null 2>&1 && [ -s "$ENT" ]; then
    codesign -s "$SIGN_ID" --options runtime --entitlements "$ENT" --force "$bin"
  else
    codesign -s "$SIGN_ID" --options runtime --force "$bin"
  fi
}

codesign -s "$SIGN_ID" --force "$FW/libvirglrenderer.1.dylib"
sign_like "$MACOS/limina-vmm"
sign_like "$MACOS/limina"
[ -f "$MACOS/gvproxy" ] && sign_like "$MACOS/gvproxy"
sign_like "$APP"
rm -f "$ENT"

codesign --verify --deep --strict "$APP" && echo "==> bundle verifies"
echo "==> rig ready: $APP"
otool -L "$MACOS/limina-vmm" | grep -i virgl || true
