#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus for the blits whose destination the replay sweep cannot read.
#
#   make-sampled-corpus.py [OUT]      default ../vm/captures/sampled.bin
#
# The sweep scores a resource only when it is `target == 2 && format != 20 && width > 8` -- a
# plain 2D colour offscreen. A blit into a layer of an array, or out of one slice of a 3D
# texture, has no line in any score: the corpus would replay green while measuring nothing. So
# every destination here is read by SAMPLING it in a draw whose own destination is a plain 2D
# offscreen, which the sweep does read. The draw is the oracle; `make-blit-corpus.py` holds the
# blits that need none.
#
# Why not read the array directly instead. The oracle has to be a DIFFERENT mechanism from the
# one under test, or two wrongs cancel: a blit that writes the wrong layer, read back by a copy
# that reads the wrong layer, hashes exactly like a correct pair. A draw that names its layer in
# a texture coordinate does not share the blitter's attachment path, and every layer this corpus
# does not blit into carries its own distinct fill, so a blit that lands in the wrong one is
# visible from both ends.
import sys, zlib

from corpus import (Corpus, cmd0, f32, BIND_DEPTH_STENCIL, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW,
                    BIND_VERTEX_BUFFER, PIPE_MASK_Z,
                    B8G8R8A8_UNORM, B8G8R8X8_UNORM, R32G32B32A32_FLOAT,
                    Z32_FLOAT, Z24X8_UNORM,
                    OBJ_BLEND, OBJ_DSA, OBJ_RASTERIZER, OBJ_VERTEX_ELEMENTS,
                    STAGE_VERTEX, STAGE_FRAGMENT,
                    SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W,
                    TARGET_2D, TARGET_3D, TARGET_2D_ARRAY)

SIDE = 64
SLICES = 8      # depth of the 3D source
LAYERS = 4      # array size of the layered destination

R8_UNORM = 64

# A quad covering the whole target, as a triangle strip. Positions are clip space; the texture
# coordinate's third component is the layer (2D array) or the normalised slice (3D), filled in
# per draw.
QUAD = [(-1.0, -1.0, 0.0, 0.0), (1.0, -1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)]

VS = """VERT
DCL IN[0]
DCL IN[1]
DCL OUT[0], POSITION
DCL OUT[1], GENERIC[0]
0: MOV OUT[0], IN[0]
1: MOV OUT[1], IN[1]
2: END"""


def fs(target):
    return """FRAG
DCL IN[0], GENERIC[0], PERSPECTIVE
DCL OUT[0], COLOR
DCL SAMP[0]
0: TEX OUT[0], IN[0], SAMP[0], %s
1: END""" % target


def pattern(tag, w, h):
    """Pixels that differ per tag and have no channel symmetry, so a wrong layer, a wrong slice
    or a swapped channel pair cannot hash equal to the right one."""
    px = bytearray()
    for y in range(h):
        for x in range(w):
            px += bytes(((x * 3 + tag * 37) & 0xFF, (y * 5 + tag * 11) & 0xFF,
                         ((x ^ y) * 7 + tag * 53) & 0xFF, (0x30 + x + y + tag) & 0xFF))
    return bytes(px)


def depth_pattern(w, h):
    """A depth ramp with no symmetry in x or y, written as Z24X8_UNORM: the depth is the high 24
    bits and the low byte is padding."""
    px = bytearray()
    for y in range(h):
        for x in range(w):
            d = ((x * 977 + y * 4093) & 0xFFFFFF)
            px += bytes((0, d & 0xFF, (d >> 8) & 0xFF, (d >> 16) & 0xFF))
    return bytes(px)


class Rig:
    """The objects every draw in this corpus shares, and the one draw it makes."""

    # Handle numbering is by kind so a refused object names itself in a log.
    VS_H = 100
    FS_ARRAY_H, FS_3D_H, FS_2D_H = 101, 102, 103
    RAST_H, BLEND_H, DSA_H, VE_H, SAMP_H = 110, 111, 112, 113, 114
    NEXT_TRANSIENT = 200   # surfaces, sampler views and vertex buffers, one set per draw

    def __init__(self, c):
        self.c = c
        self.next = Rig.NEXT_TRANSIENT
        c.shader(Rig.VS_H, STAGE_VERTEX, VS)
        c.shader(Rig.FS_ARRAY_H, STAGE_FRAGMENT, fs("2D_ARRAY"))
        c.shader(Rig.FS_3D_H, STAGE_FRAGMENT, fs("3D"))
        c.shader(Rig.FS_2D_H, STAGE_FRAGMENT, fs("2D"))
        c.rasterizer(Rig.RAST_H)
        c.blend(Rig.BLEND_H)
        c.dsa(Rig.DSA_H)
        # Two attributes out of one buffer: position at 0, texture coordinate at 16.
        c.vertex_elements(Rig.VE_H, [(0, 0, R32G32B32A32_FLOAT), (16, 0, R32G32B32A32_FLOAT)])
        c.sampler_state(Rig.SAMP_H)
        c.bind_object(OBJ_RASTERIZER, Rig.RAST_H)
        c.bind_object(OBJ_BLEND, Rig.BLEND_H)
        c.bind_object(OBJ_DSA, Rig.DSA_H)
        c.bind_object(OBJ_VERTEX_ELEMENTS, Rig.VE_H)
        c.bind_shader(Rig.VS_H, STAGE_VERTEX)

    def handle(self):
        self.next += 1
        return self.next

    def vertex_buffer(self, layer_coord):
        """A vertex buffer holding the quad, with the sampled layer in every vertex."""
        import struct
        h = self.handle()
        self.c.create(h, R8_UNORM, BIND_VERTEX_BUFFER, 128, 1, target=0)
        data = b"".join(struct.pack("<8f", x, y, 0.0, 1.0, u, v, layer_coord, 0.0)
                        for x, y, u, v in QUAD)
        self.c.inline_write(h, data, len(data), 1, 0)
        return h

    FS_FOR = {TARGET_2D_ARRAY: FS_ARRAY_H, TARGET_3D: FS_3D_H, TARGET_2D: FS_2D_H}

    def view(self, src, src_fmt, src_target, first_layer=0, last_layer=0,
             first_level=0, last_level=0, swizzle=(0, 1, 2, 3)):
        h = self.handle()
        self.c.sampler_view(h, src, src_fmt, src_target, first_layer=first_layer,
                            last_layer=last_layer, first_level=first_level,
                            last_level=last_level, swizzle=swizzle)
        return h

    def sample(self, view, src_target, layer_coord, dst, dst_fmt, side):
        """Draw what `view` samples into `dst`, which the sweep will read back."""
        c = self.c
        surf = self.handle()
        vbo = self.vertex_buffer(layer_coord)
        c.surface(surf, dst, dst_fmt)
        c.bind_shader(Rig.FS_FOR[src_target], STAGE_FRAGMENT)
        c.set_framebuffer([surf])
        c.set_viewport(side, side)
        c.set_vertex_buffers([(32, 0, vbo)])
        c.set_sampler_views(STAGE_FRAGMENT, [view])
        c.bind_sampler_states(STAGE_FRAGMENT, [Rig.SAMP_H])
        c.draw(4)


def build():
    c = Corpus()
    tex = BIND_RENDER_TARGET | BIND_SAMPLER_VIEW

    # --- the layered destination ---
    # A blit into layer 2 of a four-layer array. `vrend_renderer_blit_gl` attaches the layer the
    # DESTINATION names, and the destination is the only end that knows it: a source-keyed
    # attach lands every pixel in layer 0. X -> A forces the blitter (the two ends' table
    # swizzles disagree, so no framebuffer blit can serve it).
    ARR_SRC, ARR_DST = 30, 31
    ARR_READ_2, ARR_READ_0 = 32, 33
    BLIT_LAYER = 2

    # --- the 3D source ---
    # A blit out of one slice of an eight-slice 3D texture. The blitter normalises the slice by
    # the source TEXTURE's depth, not by the box's: dividing by the box makes every sub-range
    # blit sample the wrong slice. Two slices, because a wrong divisor can land on the right
    # slice for one z by coincidence and never for two.
    VOL_SRC = 40
    VOL_READ_5, VOL_READ_1 = 41, 42

    # --- what a blit leaves on its source's texture object ---
    # The blitter samples its source through the source's OWN texture object, and writes that
    # object's swizzle, base and max level, filters and wrap modes to suit itself. The sampler
    # view bind is cached on the view handle, so re-binding the identical view after a blit
    # re-applies none of them. On the reading, the next draw samples through the blitter's
    # settings. Two draws through one view, identical in every respect, with a blit between:
    # what the second reads settles it.
    #
    # The two implementations disagree, and the C is the one that is wrong. virglrs reads the
    # same pixels twice, which is the only answer two identical draws with nothing between them
    # can have. The C's second read comes back byte-identical to the blit's DESTINATION --
    # identity swizzle, alpha forced to one, exactly what `vrend_set_tex_param` wrote -- so the
    # blitter's settings reached a draw that never asked for them.
    #
    # So this is the one fixture in the tree whose golden is not the C: pinning the C here would
    # mean reproducing a bug in code already written correctly, and a permanently red line is a
    # gate nobody reads. The deviation is in `docs/rust-rewrite.md`.
    #
    # The observable has to be the SWIZZLE, which is a finding and not a preference. A view that
    # restricts levels is backed by a GL texture view carrying its own parameters, so the
    # blitter's writes land on the parent and no draw through that view could see them whatever
    # happened. Filters and wrap are no better: those come from the bound sampler object, which
    # overrides the texture's. Only the swizzle is a parameter both ends write on the same
    # object, and only a full-range view has no view object in between.
    #
    # What shields virglrs is not established, and these lines do not depend on it. They pin the
    # invariant, so if a driver or a change to the view cache ever lets the write through, the
    # score moves and says so.
    SAMP_SRC = 50
    SAMP_BLIT_DST = 51
    SAMP_READ_BEFORE, SAMP_READ_AFTER = 52, 53
    SAMP_VIEW_SWIZZLE = (SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W)   # red and blue swapped

    # --- the depth-writing blit ---
    # A blit whose destination is depth takes the blitter's OTHER fragment shader, which writes
    # gl_FragDepth instead of a colour, and attaches its destination to the depth attachment
    # rather than to colour. Two different depth formats force it there: a framebuffer blit
    # cannot serve a depth pair whose formats disagree (`vrend_renderer_prepare_blit`), and the
    # Z mask keeps it off the copy path, which takes only a full RGBA mask.
    #
    # The sweep skips depth and reads only colour, so the destination is scored the way the array
    # is: sampled in a draw. The SOURCE is sampled too, through the same shader and the same view
    # shape, so a wrong number in the destination cannot be blamed on the depth sampling path.
    DEPTH_SRC, DEPTH_DST = 60, 61
    DEPTH_READ_SRC, DEPTH_READ_DST = 62, 63

    c.submit()
    c.create(ARR_SRC, B8G8R8X8_UNORM, tex, SIDE)
    c.create(ARR_DST, B8G8R8A8_UNORM, tex, SIDE, array=LAYERS, target=TARGET_2D_ARRAY)
    c.create(ARR_READ_2, B8G8R8A8_UNORM, tex, SIDE)
    c.create(ARR_READ_0, B8G8R8A8_UNORM, tex, SIDE)
    c.create(VOL_SRC, B8G8R8X8_UNORM, tex, SIDE, depth=SLICES, target=TARGET_3D)
    c.create(VOL_READ_5, B8G8R8A8_UNORM, tex, SIDE)
    c.create(VOL_READ_1, B8G8R8A8_UNORM, tex, SIDE)
    c.create(SAMP_SRC, B8G8R8X8_UNORM, tex, SIDE)
    c.create(SAMP_BLIT_DST, B8G8R8A8_UNORM, tex, SIDE)
    c.create(SAMP_READ_BEFORE, B8G8R8A8_UNORM, tex, SIDE)
    c.create(SAMP_READ_AFTER, B8G8R8A8_UNORM, tex, SIDE)
    c.create(DEPTH_SRC, Z24X8_UNORM, BIND_DEPTH_STENCIL | BIND_SAMPLER_VIEW, SIDE)
    c.create(DEPTH_DST, Z32_FLOAT, BIND_DEPTH_STENCIL | BIND_SAMPLER_VIEW, SIDE)
    c.create(DEPTH_READ_SRC, B8G8R8A8_UNORM, tex, SIDE)
    c.create(DEPTH_READ_DST, B8G8R8A8_UNORM, tex, SIDE)

    rig = Rig(c)

    c.inline_write(ARR_SRC, pattern(1, SIDE, SIDE), SIDE, SIDE, SIDE * 4)
    # Every layer carries its own fill, so the layer the blit does NOT write still says which
    # layer it is -- that is what makes a wrong-layer blit visible from the untouched end too.
    for layer in range(LAYERS):
        c.inline_write(ARR_DST, pattern(10 + layer, SIDE, SIDE), SIDE, SIDE, SIDE * 4, z=layer)
    for slice_ in range(SLICES):
        c.inline_write(VOL_SRC, pattern(20 + slice_, SIDE, SIDE), SIDE, SIDE, SIDE * 4, z=slice_)
    c.inline_write(SAMP_SRC, pattern(30, SIDE, SIDE), SIDE, SIDE, SIDE * 4)
    c.inline_write(DEPTH_SRC, depth_pattern(SIDE, SIDE), SIDE, SIDE, SIDE * 4)

    c.blit(ARR_SRC, B8G8R8X8_UNORM, (0, 0, 0, SIDE, SIDE, 1),
           ARR_DST, B8G8R8A8_UNORM, (0, 0, BLIT_LAYER, SIDE, SIDE, 1))
    # The 3D blits go to plain 2D offscreens, which the sweep reads directly -- what needs the
    # draw is the array, not the volume. They are here because the two fixes are one commit's
    # worth of the same arithmetic, and a corpus that measures one of them is half a gate.
    c.blit(VOL_SRC, B8G8R8X8_UNORM, (0, 0, 5, SIDE, SIDE, 1),
           VOL_READ_5, B8G8R8A8_UNORM, (0, 0, 0, SIDE, SIDE, 1))
    c.blit(VOL_SRC, B8G8R8X8_UNORM, (0, 0, 1, SIDE, SIDE, 1),
           VOL_READ_1, B8G8R8A8_UNORM, (0, 0, 0, SIDE, SIDE, 1))

    arr_view = rig.view(ARR_DST, B8G8R8A8_UNORM, TARGET_2D_ARRAY, last_layer=LAYERS - 1)
    rig.sample(arr_view, TARGET_2D_ARRAY, BLIT_LAYER, ARR_READ_2, B8G8R8A8_UNORM, SIDE)
    rig.sample(arr_view, TARGET_2D_ARRAY, 0, ARR_READ_0, B8G8R8A8_UNORM, SIDE)

    # One view, reading the texture with red and blue swapped. Draw through it, blit out of the
    # same texture -- which writes the format's own swizzle over the view's -- then re-bind the
    # SAME view and draw again. The two reads agreeing would mean the view's swizzle survived
    # the blit.
    samp_view = rig.view(SAMP_SRC, B8G8R8X8_UNORM, TARGET_2D, swizzle=SAMP_VIEW_SWIZZLE)
    rig.sample(samp_view, TARGET_2D, 0, SAMP_READ_BEFORE, B8G8R8A8_UNORM, SIDE)
    c.blit(SAMP_SRC, B8G8R8X8_UNORM, (0, 0, 0, SIDE, SIDE, 1),
           SAMP_BLIT_DST, B8G8R8A8_UNORM, (0, 0, 0, SIDE, SIDE, 1))
    rig.sample(samp_view, TARGET_2D, 0, SAMP_READ_AFTER, B8G8R8A8_UNORM, SIDE)

    c.blit(DEPTH_SRC, Z24X8_UNORM, (0, 0, 0, SIDE, SIDE, 1),
           DEPTH_DST, Z32_FLOAT, (0, 0, 0, SIDE, SIDE, 1), mask=PIPE_MASK_Z)
    rig.sample(rig.view(DEPTH_SRC, Z24X8_UNORM, TARGET_2D), TARGET_2D, 0,
               DEPTH_READ_SRC, B8G8R8A8_UNORM, SIDE)
    rig.sample(rig.view(DEPTH_DST, Z32_FLOAT, TARGET_2D), TARGET_2D, 0,
               DEPTH_READ_DST, B8G8R8A8_UNORM, SIDE)
    c.submit()

    # The sweep reads each scored offscreen AT its unref, so everything it scores is unref'd.
    for h in (ARR_SRC, ARR_READ_2, ARR_READ_0, VOL_READ_5, VOL_READ_1,
              SAMP_BLIT_DST, SAMP_READ_BEFORE, SAMP_READ_AFTER,
              DEPTH_READ_SRC, DEPTH_READ_DST):
        c.unref(h)
    c.submit()
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "../vm/captures/sampled.bin"
    blob = build().dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x" % (out, len(blob), zlib.crc32(blob)))


if __name__ == "__main__":
    main()
