#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Look at what the guest actually put on the screen.
#
# WHY THIS EXISTS. Every other signal about a seated desktop is a proxy, and proxies lie in the
# one direction that costs the most: they report success. `graphical-session.target` reads
# `active` while the compositor is exiting 101 behind it, because the target is reached by its
# dependencies and knows nothing about whether a frame was drawn. `vulkaninfo` reports a working
# venus device from an SSH shell that never touches the display. A renderer log shows contexts
# created and torn down cleanly, which is exactly what a compositor that died at startup also
# produces. Each of those is a real signal about a real thing; none of them is a pixel, and a
# claim that a desktop is up is a claim about pixels.
#
# So this prints facts about a frame and refuses to render a verdict. There is no PASS here and
# there should never be one: a script that concludes "looks seated" is a new proxy, indistinguishable
# from the ones that already lied, and it would be trusted faster because it sounds like it looked.
# The numbers narrow down where to look. The last line is the path, because opening it is the
# oracle and everything above is a summary of something you can see for yourself in one second.
#
# Usage: frame.py [PNG...]     (default: every *-frame.png under captures/)

import pathlib
import struct
import sys
import zlib
from collections import Counter

# Only what limina writes: 8-bit RGBA, non-interlaced. Anything else is a change worth noticing
# rather than a case worth silently handling, so it is refused by name.
COLORTYPE_RGBA = 6


def decode(path):
    """(width, height, rgba bytes) from a PNG, using nothing but the standard library."""
    d = path.read_bytes()
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path}: not a PNG")
    w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", d[16:29])
    if (depth, ctype, interlace) != (8, COLORTYPE_RGBA, 0):
        raise SystemExit(
            f"{path}: expected 8-bit RGBA non-interlaced, got depth={depth} "
            f"colortype={ctype} interlace={interlace}"
        )

    idat, off = bytearray(), 8
    while off < len(d):
        (length,) = struct.unpack(">I", d[off : off + 4])
        kind = d[off + 4 : off + 8]
        if kind == b"IDAT":
            idat += d[off + 8 : off + 8 + length]
        off += 12 + length
    raw = zlib.decompress(bytes(idat))

    # Undo the per-scanline filters. bpp is 4 for RGBA8, and the filters are defined against the
    # reconstructed bytes, so this walks forward and never re-reads a filtered byte.
    stride, bpp = w * 4, 4
    out = bytearray(h * stride)
    pos = 0
    for y in range(h):
        f = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride
        up = out[(y - 1) * stride : y * stride] if y else bytes(stride)
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = up[x]
            if f == 1:
                line[x] = (line[x] + a) & 0xFF
            elif f == 2:
                line[x] = (line[x] + b) & 0xFF
            elif f == 3:
                line[x] = (line[x] + ((a + b) >> 1)) & 0xFF
            elif f == 4:
                c = up[x - bpp] if x >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pr) & 0xFF
            elif f != 0:
                raise SystemExit(f"{path}: unknown scanline filter {f} on row {y}")
        out[y * stride : (y + 1) * stride] = line
    return w, h, bytes(out)


def thumbnail(w, h, rgba, cols=64):
    """A coarse luminance sketch, so the shape of the frame is visible without leaving the terminal.

    This is a rendering of the pixels rather than a judgement about them -- it shows a console
    with a cursor as a console with a cursor. It is still smaller than the truth; the file is
    the truth.
    """
    ramp = " .:-=+*#%@"
    rows = max(1, round(cols * h / w / 2.2))
    lines = []
    for ry in range(rows):
        line = ""
        for rx in range(cols):
            x = min(w - 1, rx * w // cols)
            y = min(h - 1, ry * h // rows)
            i = (y * w + x) * 4
            lum = (rgba[i] * 299 + rgba[i + 1] * 587 + rgba[i + 2] * 114) // 1000
            line += ramp[min(len(ramp) - 1, lum * len(ramp) // 256)]
        lines.append(line)
    return lines


def report(path):
    w, h, rgba = decode(path)
    px = [rgba[i : i + 4] for i in range(0, len(rgba), 4)]
    counts = Counter(px)
    total = len(px)
    (dom, dom_n) = counts.most_common(1)[0]

    print(f"\n=== {path}")
    print(f"    {w}x{h}, {total} pixels")
    print(f"    distinct colours : {len(counts)}")
    print(
        f"    dominant colour  : rgba{tuple(dom)} on {100 * dom_n / total:.2f}% of the frame"
    )
    print(f"    everything else  : {100 * (total - dom_n) / total:.2f}%")
    for line in thumbnail(w, h, rgba):
        print(f"    |{line}|")
    # Last, and deliberately: the numbers above are a summary, and the file is the thing.
    print(f"    look at it: {path}")


def main():
    args = [pathlib.Path(a) for a in sys.argv[1:]]
    if not args:
        here = pathlib.Path(__file__).resolve().parent
        args = sorted((here / "captures").glob("*-frame.png"))
    if not args:
        raise SystemExit(
            "no frames. A windowed boot writes none -- limina refuses --display-capture "
            "with a window, so the human at the screen is that boot's only oracle. Boot "
            "headless (capture.sh without --window) to get a frame on disk."
        )
    for p in args:
        report(p)


if __name__ == "__main__":
    main()
