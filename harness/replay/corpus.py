# SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
# Copyright © 2026 Gustavo Noronha Silva
#
# Writing a synthetic classic corpus: the trace container, and the virgl commands a synthetic
# corpus needs. Shared by every `make-*-corpus.py` in this directory.
#
# A synthetic corpus exists because a recorded one cannot reach the code it has to score. What is
# here is the container plus the commands that need no guest shader; a corpus that needs a draw
# builds its rig on top of `Corpus.command`.
import struct

MAGIC = 0x4C4D5654
HDR = struct.Struct("<IBBHQQII")   # total_len, type, cmd, ctx, seq, mono_ns, payload_len, aux_count
RES = struct.Struct("<Q12I")       # seq, kind, handle, target, format, bind, w, h, d, array, levels, samples, flags

T_SUBMIT, T_CMD = 1, 2
RES_CREATE, RES_UNREF = 0, 2

# virgl_context_cmd
CCMD_CREATE_OBJECT = 1
CCMD_BIND_OBJECT = 2
CCMD_SET_VIEWPORT_STATE = 4
CCMD_SET_FRAMEBUFFER_STATE = 5
CCMD_SET_VERTEX_BUFFERS = 6
CCMD_DRAW_VBO = 8
CCMD_INLINE_WRITE = 9
CCMD_SET_SAMPLER_VIEWS = 10
CCMD_BLIT = 16
CCMD_BIND_SAMPLER_STATES = 18
CCMD_BIND_SHADER = 31

# virgl_object_type
OBJ_BLEND, OBJ_RASTERIZER, OBJ_DSA, OBJ_SHADER = 1, 2, 3, 4
OBJ_VERTEX_ELEMENTS, OBJ_SAMPLER_VIEW, OBJ_SAMPLER_STATE, OBJ_SURFACE = 5, 6, 7, 8

# pipe_texture_target
TARGET_BUFFER, TARGET_2D, TARGET_3D, TARGET_CUBE, TARGET_2D_ARRAY = 0, 2, 3, 4, 7

# pipe_shader_type
STAGE_VERTEX, STAGE_FRAGMENT = 0, 1

# virgl_hw.h format numbers
B8G8R8A8_UNORM, B8G8R8X8_UNORM = 1, 2
R8G8B8A8_UNORM, R8G8B8X8_UNORM = 67, 134
B8G8R8A8_SRGB, B8G8R8X8_SRGB = 100, 101
R32G32B32A32_FLOAT = 31

BIND_DEPTH_STENCIL = 1 << 0
BIND_RENDER_TARGET = 1 << 1
BIND_SAMPLER_VIEW = 1 << 3
BIND_VERTEX_BUFFER = 1 << 4
BIND_SCANOUT = 1 << 18

SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W = 0, 1, 2, 3

PIPE_MASK_RGBA = 0xF
FILTER_NEAREST, FILTER_LINEAR = 0, 1
PRIM_TRIANGLE_STRIP = 5

CTX = 1


def cmd0(cmd, obj, length):
    """VIRGL_CMD0: the command header. `length` counts the dwords AFTER it."""
    return cmd | (obj << 8) | (length << 16)


def f32(x):
    """A float as the dword the wire carries it in."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


class Corpus:
    """A trace under construction: resource events and command records, in stream order.

    The replay driver applies a resource event after the record carrying its sequence number, so
    a create issued while the stream is at record N exists from N+1 onwards.
    """

    def __init__(self):
        self.res = []
        self.recs = []
        self.seq = 1

    # ---- resources ----

    def create(self, handle, fmt, bind, w, h=None, depth=1, array=1, target=TARGET_2D,
               levels=0, samples=0):
        h = w if h is None else h
        self.res.append(RES.pack(self.seq, RES_CREATE, handle, target, fmt, bind,
                                 w, h, depth, array, levels, samples, 0))

    def unref(self, handle):
        self.res.append(RES.pack(self.seq, RES_UNREF, handle, TARGET_2D, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    # ---- records ----

    def record(self, typ, cmd, aux, payload):
        total = HDR.size + len(aux) * 4 + len(payload)
        self.recs.append(HDR.pack(total, typ, cmd, CTX, self.seq, self.seq * 1000,
                                  len(payload), len(aux))
                         + struct.pack("<%dI" % len(aux), *aux) + payload)
        self.seq += 1

    def submit(self):
        """Start a batch. Every CMD record up to the next SUBMIT is one submission."""
        self.record(T_SUBMIT, 0, (0,), b"")

    def command(self, ccmd, dwords):
        self.record(T_CMD, ccmd, (), struct.pack("<%dI" % len(dwords), *dwords))

    def emit(self, ccmd, obj, body):
        """A command whose header length is its body's."""
        self.command(ccmd, [cmd0(ccmd, obj, len(body))] + list(body))

    # ---- the commands a synthetic corpus needs ----

    def inline_write(self, handle, pixels, w, h, stride, x=0, y=0, z=0, depth=1, level=0):
        """RESOURCE_INLINE_WRITE: the bytes travel in the command stream, so a corpus that fills
        this way needs no iov and no XFERDATA records to be replayable."""
        assert len(pixels) % 4 == 0
        dw = list(struct.unpack("<%dI" % (len(pixels) // 4), pixels))
        self.emit(CCMD_INLINE_WRITE, 0,
                  [handle, level, 0, stride, 0, x, y, z, w, h, depth] + dw)

    def blit(self, src, src_fmt, src_box, dst, dst_fmt, dst_box,
             src_level=0, dst_level=0, filt=FILTER_NEAREST, mask=PIPE_MASK_RGBA):
        """A box is (x, y, z, w, h, d)."""
        s0 = mask | (filt << 8)
        self.emit(CCMD_BLIT, 0,
                  [s0, 0, 0, dst, dst_level, dst_fmt] + list(dst_box)
                  + [src, src_level, src_fmt] + list(src_box))

    # ---- objects ----

    def shader(self, handle, stage, text):
        """CREATE_OBJECT(SHADER). The text is TGSI, NUL-terminated and dword-padded; num_tokens
        must be non-zero or the translator refuses it."""
        blob = text.encode() + b"\0"
        blob += b"\0" * (-len(blob) % 4)
        dw = list(struct.unpack("<%dI" % (len(blob) // 4), blob))
        self.emit(CCMD_CREATE_OBJECT, OBJ_SHADER,
                  [handle, stage, len(blob), text.count("\n") + 2, 0] + dw)

    def rasterizer(self, handle):
        """A rasterizer that does nothing but let the quad through: no cull, filled, and the
        half-pixel centre and front-CCW winding a full-viewport quad is written for."""
        s0 = (1 << 15) | (1 << 29)   # front_ccw, half_pixel_center
        self.emit(CCMD_CREATE_OBJECT, OBJ_RASTERIZER,
                  [handle, s0, f32(1.0), 0, 0, f32(1.0), 0, 0, 0])

    def blend(self, handle):
        """Blending off, all four channels written."""
        rt0 = 0xF << 27
        self.emit(CCMD_CREATE_OBJECT, OBJ_BLEND, [handle, 0, 0, rt0] + [0] * 7)

    def dsa(self, handle):
        """Depth test, stencil test and alpha test all off."""
        self.emit(CCMD_CREATE_OBJECT, OBJ_DSA, [handle, 0, 0, 0, 0])

    def vertex_elements(self, handle, elements):
        """Each element is (src_offset, buffer_index, format)."""
        body = [handle]
        for off, idx, fmt in elements:
            body += [off, 0, idx, fmt]
        self.emit(CCMD_CREATE_OBJECT, OBJ_VERTEX_ELEMENTS, body)

    def sampler_state(self, handle, wrap=1, filt=FILTER_NEAREST):
        """wrap 1 is CLAMP_TO_EDGE. Mip filter NONE, no compare, no anisotropy."""
        s0 = wrap | (wrap << 3) | (wrap << 6) | (filt << 9) | (filt << 13)
        self.emit(CCMD_CREATE_OBJECT, OBJ_SAMPLER_STATE,
                  [handle, s0, 0, 0, 0, 0, 0, 0, 0])

    def sampler_view(self, handle, res, fmt, target, first_layer=0, last_layer=0,
                     first_level=0, last_level=0, swizzle=(0, 1, 2, 3)):
        """`swizzle` is four of PIPE_SWIZZLE_X/Y/Z/W/0/1 (0..5), defaulting to identity."""
        swizzle = sum(s << (3 * i) for i, s in enumerate(swizzle))
        self.emit(CCMD_CREATE_OBJECT, OBJ_SAMPLER_VIEW,
                  [handle, res, fmt | (target << 24),
                   first_layer | (last_layer << 16),
                   first_level | (last_level << 8), swizzle])

    def surface(self, handle, res, fmt, level=0, first_layer=0, last_layer=0):
        self.emit(CCMD_CREATE_OBJECT, OBJ_SURFACE,
                  [handle, res, fmt, level, first_layer | (last_layer << 16)])

    # ---- state and the draw ----

    def bind_object(self, kind, handle):
        self.emit(CCMD_BIND_OBJECT, kind, [handle])

    def bind_shader(self, handle, stage):
        self.emit(CCMD_BIND_SHADER, 0, [handle, stage])

    def set_framebuffer(self, cbufs, zsurf=0):
        self.emit(CCMD_SET_FRAMEBUFFER_STATE, 0, [len(cbufs), zsurf] + list(cbufs))

    def set_vertex_buffers(self, buffers):
        """Each buffer is (stride, offset, resource)."""
        body = []
        for stride, offset, res in buffers:
            body += [stride, offset, res]
        self.emit(CCMD_SET_VERTEX_BUFFERS, 0, body)

    def set_sampler_views(self, stage, views, start_slot=0):
        self.emit(CCMD_SET_SAMPLER_VIEWS, 0, [stage, start_slot] + list(views))

    def bind_sampler_states(self, stage, states, start_slot=0):
        self.emit(CCMD_BIND_SAMPLER_STATES, 0, [stage, start_slot] + list(states))

    def set_viewport(self, w, h):
        """A viewport covering the whole target, y up -- the scale/translate pair gallium sends
        for a framebuffer whose origin is bottom-left."""
        self.emit(CCMD_SET_VIEWPORT_STATE, 0,
                  [0, f32(w / 2.0), f32(h / 2.0), f32(0.5),
                   f32(w / 2.0), f32(h / 2.0), f32(0.5)])

    def draw(self, count, mode=PRIM_TRIANGLE_STRIP):
        self.emit(CCMD_DRAW_VBO, 0, [0, count, mode, 0, 1, 0, 0, 0, 0, 0, count - 1, 0])

    # ---- output ----

    def dump(self):
        body = b"".join(self.res) + b"".join(self.recs)
        # head[13] is `res_full`: the resource ring OVERFLOWED and the log no longer reaches the
        # start. A synthesised log is complete by construction, so it is 0.
        head = [MAGIC, 2, 512, len(body), len(self.recs), 0, 0, 0, 0, 0, 0, 0,
                len(self.res), 0, 0, 0]
        return struct.pack("<16I", *head) + body
