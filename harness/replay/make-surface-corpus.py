#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus that destroys a surface the framebuffer is still using.
#
#   make-surface-corpus.py [OUT]      default ../vm/captures/surface.bin
#
# A surface whose format reinterprets its resource is backed by a GL texture view, and that view
# has to outlive the surface object the guest destroys -- the framebuffer goes on naming it until
# the next SET_FRAMEBUFFER_STATE. No recorded corpus does this: a GNOME session destroys its
# surfaces only after unbinding them, so the ordering that matters is never sampled and a renderer
# that deletes the view at DESTROY_OBJECT scores green everywhere.
#
# Both cases here are scored by clearing to a flat colour after the destroy and reading the
# resource back. The resource is pre-filled with a pattern, so a clear that lands nowhere is a
# different hash from a clear that lands -- and neither is the uninitialised storage that would
# make the fixture unpinnable.
#
#   destroy-then-clear   The plain lifetime: the surface goes away, the framebuffer slot does not.
#   destroy-then-rebind  A second surface created under the same handle. The bind path compares
#                        the slot against the handle it is asked for, so a stale slot value can
#                        skip a re-attach that has to happen -- the new surface's view is a
#                        different texture from the dead one's.
#   unref-while-bound    The resource under a bound surface, freed. The C's surface holds a
#                        reference to it, so the framebuffer goes on taking pixels; the second
#                        colour buffer is what says whether it did, since the first is gone.
import sys, zlib

from corpus import (Corpus, OBJ_SURFACE, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW,
                    PIPE_CLEAR_COLOR0, B8G8R8A8_UNORM, B8G8R8X8_UNORM)

SIDE = 64   # the sweep scores 2D colour targets wider than 8

DESTROY_RES, DESTROY_SURF = 10, 200
REBIND_RES, REBIND_SURF = 11, 201
UNREF_RES, UNREF_SURF = 12, 202
SURVIVE_RES, SURVIVE_SURF = 13, 203

BOTH_COLOURS = PIPE_CLEAR_COLOR0 | (PIPE_CLEAR_COLOR0 << 1)


def fill_pixels(handle):
    """A pattern with no flat channel, so a clear that fails to land cannot hash equal to one
    that does."""
    px = bytearray()
    for y in range(SIDE):
        for x in range(SIDE):
            px += bytes(((x * 4 + handle) & 0xFF, (y * 4 + handle * 3) & 0xFF,
                         ((x ^ y) * 3 + handle * 7) & 0xFF, (0x40 + x + y) & 0xFF))
    return bytes(px)


def build():
    c = Corpus()
    tex = BIND_RENDER_TARGET | BIND_SAMPLER_VIEW

    c.submit()
    c.create(DESTROY_RES, B8G8R8A8_UNORM, tex, SIDE)
    c.create(REBIND_RES, B8G8R8A8_UNORM, tex, SIDE)
    c.create(UNREF_RES, B8G8R8A8_UNORM, tex, SIDE)
    c.create(SURVIVE_RES, B8G8R8A8_UNORM, tex, SIDE)

    # The surfaces name the X spelling of their resource's A format. That mismatch is what mints
    # a texture view rather than attaching the resource's own texture, so the thing the destroy
    # would delete is a distinct GL object and its loss is visible in the pixels.
    c.inline_write(DESTROY_RES, fill_pixels(DESTROY_RES), SIDE, SIDE, SIDE * 4)
    c.surface(DESTROY_SURF, DESTROY_RES, B8G8R8X8_UNORM)
    c.set_framebuffer([DESTROY_SURF])
    c.set_viewport(SIDE, SIDE)
    c.destroy_object(OBJ_SURFACE, DESTROY_SURF)
    c.clear((0.25, 0.5, 0.75, 1.0))

    c.inline_write(REBIND_RES, fill_pixels(REBIND_RES), SIDE, SIDE, SIDE * 4)
    c.surface(REBIND_SURF, REBIND_RES, B8G8R8X8_UNORM)
    c.set_framebuffer([REBIND_SURF])
    c.destroy_object(OBJ_SURFACE, REBIND_SURF)
    # Same handle, same description, a different object. Binding it must re-attach.
    c.surface(REBIND_SURF, REBIND_RES, B8G8R8X8_UNORM)
    c.set_framebuffer([REBIND_SURF])
    c.clear((0.75, 0.5, 0.25, 1.0))

    # Two resources under one framebuffer, so that freeing the first leaves a second whose
    # pixels can still be read. A resource event applies after the record whose sequence number
    # it carries, so each unref rides the SUBMIT that follows it.
    c.inline_write(UNREF_RES, fill_pixels(UNREF_RES), SIDE, SIDE, SIDE * 4)
    c.inline_write(SURVIVE_RES, fill_pixels(SURVIVE_RES), SIDE, SIDE, SIDE * 4)
    c.surface(UNREF_SURF, UNREF_RES, B8G8R8X8_UNORM)
    c.surface(SURVIVE_SURF, SURVIVE_RES, B8G8R8X8_UNORM)
    c.set_framebuffer([UNREF_SURF, SURVIVE_SURF])
    c.clear((0.125, 0.25, 0.375, 1.0), BOTH_COLOURS)
    c.unref(UNREF_RES)   # read back here: the first clear
    c.submit()
    # The framebuffer still names a resource the guest has freed. The second clear is what says
    # whether it kept taking pixels: the survivor carries this colour if it did, and the first
    # colour if the attachment died with its neighbour.
    c.clear((0.875, 0.75, 0.625, 1.0), BOTH_COLOURS)

    # Drop the framebuffer before the unrefs, so the readbacks are not racing an attachment.
    c.set_framebuffer([])
    c.submit()

    c.unref(DESTROY_RES)
    c.unref(REBIND_RES)
    c.unref(SURVIVE_RES)
    c.submit()
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "../vm/captures/surface.bin"
    blob = build().dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x" % (out, len(blob), zlib.crc32(blob)))


if __name__ == "__main__":
    main()
