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
# What is deliberately NOT here: the depth-writing variants (`blit_build_frag_depth`). The score
# reads back colour offscreens, so a depth destination would sit in the corpus looking covered
# while measuring nothing. They need a corpus that samples the blitted depth into a colour
# target, which needs shaders, which is a different corpus.
#
# Sources are filled by RESOURCE_INLINE_WRITE rather than TRANSFER3D: the bytes travel in the
# command stream, so the corpus needs no iov and no XFERDATA records to be replayable.
#
# Every destination is exactly the size of the region its blit writes. The score reads a whole
# resource back, so a destination larger than its blit would carry uninitialised storage into the
# hash -- which is not a pinnable number: two runs of the C alone disagree on it.
import struct, sys, zlib

MAGIC = 0x4C4D5654
HDR = struct.Struct("<IBBHQQII")   # total_len, type, cmd, ctx, seq, mono_ns, payload_len, aux_count
RES = struct.Struct("<Q12I")       # seq, kind, handle, target, format, bind, w, h, d, array, levels, samples, flags

T_SUBMIT, T_CMD = 1, 2
RES_CREATE, RES_UNREF = 0, 2
CCMD_INLINE_WRITE, CCMD_BLIT = 9, 16

CTX = 1
TARGET_2D = 2

# virgl_hw.h format numbers. The A/X pairs are what force `needs_swizzle`.
B8G8R8A8_UNORM, B8G8R8X8_UNORM = 1, 2
R8G8B8A8_UNORM, R8G8B8X8_UNORM = 67, 134
B8G8R8A8_SRGB, B8G8R8X8_SRGB = 100, 101

BIND_RENDER_TARGET = 1 << 1
BIND_SAMPLER_VIEW = 1 << 3
BIND_SCANOUT = 1 << 18

PIPE_MASK_RGBA = 0xF
FILTER_NEAREST, FILTER_LINEAR = 0, 1

SIDE = 64   # every resource is SIDE x SIDE; the sweep scores 2D colour targets wider than 8


def cmd0(cmd, obj, length):
    return cmd | (obj << 8) | (length << 16)


def source_pixels(handle):
    """A pattern that differs per source and has no channel symmetry, so a blit that swaps red
    and blue, or that carries the wrong alpha, cannot hash equal to one that does not."""
    px = bytearray()
    for y in range(SIDE):
        for x in range(SIDE):
            px += bytes(((x * 4 + handle) & 0xFF, (y * 4 + handle * 3) & 0xFF,
                         ((x ^ y) * 3 + handle * 7) & 0xFF, (0x40 + x + y) & 0xFF))
    return bytes(px)


class Corpus:
    def __init__(self):
        self.res = []
        self.recs = []
        self.seq = 1

    def create(self, handle, fmt, bind, side=None):
        side = SIDE if side is None else side
        self.res.append(RES.pack(self.seq, RES_CREATE, handle, TARGET_2D, fmt, bind,
                                 side, side, 1, 1, 0, 0, 0))

    def unref(self, handle):
        self.res.append(RES.pack(self.seq, RES_UNREF, handle, TARGET_2D, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    def record(self, typ, cmd, aux, payload):
        total = HDR.size + len(aux) * 4 + len(payload)
        self.recs.append(HDR.pack(total, typ, cmd, CTX, self.seq, self.seq * 1000,
                                  len(payload), len(aux))
                         + struct.pack("<%dI" % len(aux), *aux) + payload)
        self.seq += 1

    def submit(self):
        self.record(T_SUBMIT, 0, (0,), b"")

    def command(self, ccmd, dwords):
        self.record(T_CMD, ccmd, (), struct.pack("<%dI" % len(dwords), *dwords))

    def fill(self, handle):
        px = source_pixels(handle)
        dw = list(struct.unpack("<%dI" % (len(px) // 4), px))
        body = [handle, 0, 0, SIDE * 4, 0, 0, 0, 0, SIDE, SIDE, 1] + dw
        self.command(CCMD_INLINE_WRITE, [cmd0(CCMD_INLINE_WRITE, 0, len(body))] + body)

    def blit(self, src, src_fmt, dst, dst_fmt, dst_side=SIDE, filt=FILTER_NEAREST):
        s0 = PIPE_MASK_RGBA | (filt << 8)
        body = [s0, 0, 0,
                dst, 0, dst_fmt, 0, 0, 0, dst_side, dst_side, 1,
                src, 0, src_fmt, 0, 0, 0, SIDE, SIDE, 1]
        self.command(CCMD_BLIT, [cmd0(CCMD_BLIT, 0, len(body))] + body)

    def dump(self):
        body = b"".join(self.res) + b"".join(self.recs)
        # head[13] is `res_full`: the resource ring OVERFLOWED and the log no longer reaches
        # the start. A synthesised log is complete by construction, so it is 0.
        head = [MAGIC, 2, 512, len(body), len(self.recs), 0, 0, 0, 0, 0, 0, 0, len(self.res), 0, 0, 0]
        return struct.pack("<16I", *head) + body


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
        c.create(src, src_fmt, tex)
        c.create(dst, dst_fmt, tex, side=dst_side)
    c.create(REDBLUE_SRC, R8G8B8A8_UNORM, tex)
    c.create(REDBLUE_DST, B8G8R8A8_UNORM, tex | BIND_SCANOUT, side=SIDE // 2)

    for _, src, src_fmt, dst, dst_fmt, dst_side, filt in variants:
        c.fill(src)
        c.blit(src, src_fmt, dst, dst_fmt, dst_side=dst_side, filt=filt)
    c.fill(REDBLUE_SRC)
    c.blit(REDBLUE_SRC, R8G8B8A8_UNORM, REDBLUE_DST, R8G8B8A8_UNORM,
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
