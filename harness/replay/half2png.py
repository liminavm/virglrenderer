#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 Gustavo Noronha Silva
#
# Turn vrend-replay's REPLAY_DUMP_DIR dumps of a HALF-FLOAT resource into viewable PNGs.
# rgba2png.py reads the 8-bit BGRA the offscreens are; the blob fixture's windows are
# R16G16B16X16_FLOAT (virgl format 236), eight bytes a texel, and reading those as bytes shows
# noise rather than a picture -- which is indistinguishable from the corruption one would be
# looking for. The values are linear, so they are gamma-encoded on the way out: skipping that
# step washes the frame out and invites a verdict about a renderer from a property of the dump.
import struct, sys, zlib

def png(path, w, h, rgb):
    raw = b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    open(path, "wb").write(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b""))

def encode(v):
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    s = v * 12.92 if v <= 0.0031308 else 1.055 * (v ** (1 / 2.4)) - 0.055
    return int(s * 255 + 0.5)

# A 256-entry table over the half floats that actually occur is not possible -- they are not
# 8-bit -- but the encode is pure, so memoising it keeps a 250k-texel frame to a fraction of a
# second without giving up exactness.
cache = {}
for src in sys.argv[1:]:
    import re
    m = re.search(r"res(\d+)_(\d+)x(\d+)\.rgba$", src)
    if not m:
        print("skipping (no WxH in name):", src, file=sys.stderr)
        continue
    w, h = int(m.group(2)), int(m.group(3))
    data = open(src, "rb").read()
    need = w * h * 8
    if len(data) < need:
        print(f"skipping {src}: {len(data)} bytes, need {need}", file=sys.stderr)
        continue
    halves = struct.unpack_from("<%de" % (w * h * 4), data, 0)
    out = bytearray(w * h * 3)
    lo = hi = halves[0]
    for i in range(w * h):
        for c in range(3):
            v = halves[i * 4 + c]
            if v < lo: lo = v
            if v > hi: hi = v
            b = cache.get(v)
            if b is None:
                b = cache[v] = encode(v)
            out[i * 3 + c] = b
    dst = src[:-5] + ".png"
    png(dst, w, h, bytes(out))
    print(f"{dst}  {w}x{h}  min={lo:.4f} max={hi:.4f}")
