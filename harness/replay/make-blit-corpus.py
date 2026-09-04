#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus whose every BLIT needs the shader blitter.
#
#   make-blit-corpus.py [OUT]      default ../vm/captures/blit.bin
#
# The recorded corpora do not reach the blitter. A GNOME session's blits are all format-matched
# mip-chain reductions, which take glBlitFramebuffer; replaying the classic corpus with the
# blitter's variants counted prints nothing. So a port of the blitter could be scored green by
# every pinned fixture in the tree while doing nothing at all, and the gate has to be built
# before the code it gates.
#
# Each blit here is a destination that only the blitter can fill. The reason is always
# `needs_swizzle` or `needs_redblue_swizzle` -- the two predicates in `vrend_renderer_blit_int`
# that a synthetic corpus can force without a draw:
#
#   swizzle    An X-channel format is an RGBA texture with a texture swizzle of (X,Y,Z,1) on it,
#              so a blit between the A and the X spelling of one layout has ends whose swizzles
#              disagree. A framebuffer blit reads the storage and would carry the wrong alpha.
#   redblue    An IOSurface-backed BGRA resource cannot be viewed, so a blit that names it in the
#              opposite channel order has to swap red and blue in the shader.
#
# Every destination here is a plain 2D colour offscreen, which is what the replay sweep reads
# back. The blits whose destinations it cannot read -- a layer of an array, a depth texture --
# are in `make-sampled-corpus.py`, which scores them through a draw.
#
# Sources are filled by RESOURCE_INLINE_WRITE rather than TRANSFER3D: the bytes travel in the
# command stream, so the corpus needs no iov and no XFERDATA records to be replayable.
#
# Every destination is exactly the size of the region its blit writes. The score reads a whole
# resource back, so a destination larger than its blit would carry uninitialised storage into the
# hash -- which is not a pinnable number: two runs of the C alone disagree on it.
import sys, zlib

from corpus import (Corpus, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW, BIND_SCANOUT,
                    FILTER_NEAREST, FILTER_LINEAR,
                    B8G8R8A8_UNORM, B8G8R8X8_UNORM, R8G8B8A8_UNORM, R8G8B8X8_UNORM,
                    B8G8R8A8_SRGB, B8G8R8X8_SRGB)

SIDE = 64   # every resource is SIDE x SIDE; the sweep scores 2D colour targets wider than 8


def source_pixels(handle):
    """A pattern that differs per source and has no channel symmetry, so a blit that swaps red
    and blue, or that carries the wrong alpha, cannot hash equal to one that does not."""
    px = bytearray()
    for y in range(SIDE):
        for x in range(SIDE):
            px += bytes(((x * 4 + handle) & 0xFF, (y * 4 + handle * 3) & 0xFF,
                         ((x ^ y) * 3 + handle * 7) & 0xFF, (0x40 + x + y) & 0xFF))
    return bytes(px)


def fill(c, handle):
    c.inline_write(handle, source_pixels(handle), SIDE, SIDE, SIDE * 4)


def blit(c, src, src_fmt, dst, dst_fmt, dst_side=SIDE, filt=FILTER_NEAREST):
    c.blit(src, src_fmt, (0, 0, 0, SIDE, SIDE, 1),
           dst, dst_fmt, (0, 0, 0, dst_side, dst_side, 1), filt=filt)


def build():
    c = Corpus()
    # Every variant is a source filled from the command stream and a destination only the shader
    # blitter can fill. No resource is shared between variants, so a hash that moves names
    # exactly one variant.
    #
    # The direction matters, and only one of the two bites. Gallium calls the X spelling
    # compatible with the A spelling -- the ignored channel may be dropped -- so an A -> X blit
    # is a plain `glCopyImageSubData` and never reaches `vrend_renderer_blit_int` at all. X -> A
    # is not compatible, because the destination's alpha has to come from somewhere, and that is
    # the blit that lands on the blitter.
    #
    # (name, src handle, src format, dst handle, dst handle's format, dst side, filter)
    variants = [
        ("bgra X->A", 10, B8G8R8X8_UNORM, 11, B8G8R8A8_UNORM, SIDE, FILTER_NEAREST),
        ("rgba X->A", 12, R8G8B8X8_UNORM, 13, R8G8B8A8_UNORM, SIDE, FILTER_NEAREST),
        ("srgb X->A", 14, B8G8R8X8_SRGB, 15, B8G8R8A8_SRGB, SIDE, FILTER_NEAREST),
        # Shrunk and filtered: the blitter's texcoords and its sampler, which a 1:1 nearest blit
        # would let a wrong implementation get right by accident.
        ("bgra X->A scaled", 16, B8G8R8X8_UNORM, 17, B8G8R8A8_UNORM, SIDE // 2, FILTER_LINEAR),
    ]
    # The redblue destination is a SCANOUT, which is what mints an IOSurface -- and an
    # IOSurface-backed BGRA resource is exactly the one that cannot be viewed. Naming it in the
    # opposite channel order is what `vrend_blit_needs_redblue_swizzle` reports.
    REDBLUE_SRC, REDBLUE_DST = 20, 21
    tex = BIND_RENDER_TARGET | BIND_SAMPLER_VIEW

    c.submit()
    for _, src, src_fmt, dst, dst_fmt, dst_side, _ in variants:
        c.create(src, src_fmt, tex, SIDE)
        c.create(dst, dst_fmt, tex, dst_side)
    c.create(REDBLUE_SRC, R8G8B8A8_UNORM, tex, SIDE)
    c.create(REDBLUE_DST, B8G8R8A8_UNORM, tex | BIND_SCANOUT, SIDE // 2)

    for _, src, src_fmt, dst, dst_fmt, dst_side, filt in variants:
        fill(c, src)
        blit(c, src, src_fmt, dst, dst_fmt, dst_side=dst_side, filt=filt)
    fill(c, REDBLUE_SRC)
    blit(c, REDBLUE_SRC, R8G8B8A8_UNORM, REDBLUE_DST, R8G8B8A8_UNORM,
         dst_side=SIDE // 2, filt=FILTER_LINEAR)
    c.submit()

    # Unref every scored offscreen: the sweep reads each back AT its unref. The scanout is left
    # alive on purpose -- it is scored through its IOSurface at the end of the stream, which is
    # the leg a capture of a real session exercises and a readback does not.
    for _, src, _, dst, _, _, _ in variants:
        c.unref(src)
        c.unref(dst)
    c.unref(REDBLUE_SRC)
    c.submit()
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "../vm/captures/blit.bin"
    blob = build().dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x" % (out, len(blob), zlib.crc32(blob)))


if __name__ == "__main__":
    main()
