#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
"""Generate the classic renderer's format tables.

    gen.py --outdir DIR --virgl-hw PATH --gallium DIR

Emits `formats.rs` into DIR: gallium's description of every `virgl_formats` value (block size,
channels, swizzle, colour space -- what `util_format_*` computes in the C), indexed by the wire's
format number, and the GL triples vrend maps each format to, grouped the way `vrend_formats.c`
groups them so the host-side probe can add a group under the same condition the C does.

The descriptions are the tree's `u_format.yaml`, parsed by the `u_format_parse.py` beside it --
the same two files the C's `util_format_description` is generated from, read where they are so
there is one copy. The wire numbering comes from `virgl_hw.h`, the ABI shared with the guest,
read from the tree for the same reason. The GL triples are `gl_formats.py`, converted once from
`vrend_formats.c`.
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gl_formats  # noqa: E402

ufp = None  # u_format_parse, imported from --gallium in main()

# The order `vrend_build_format_list_common` and `_gles` add the groups in, with the condition
# each is added under. Order matters: the first table to name a format wins.
GROUPS = [
    ('base_rgba_formats', 'Always'),
    ('base_depth_formats', 'Always'),
    ('float_base_formats', 'Always'),
    ('la_formats_compat', 'Always'),
    ('la_formats_fallback', 'Always'),
    ('float_3comp_formats', 'Always'),
    ('integer_base_formats', 'Always'),
    ('integer_3comp_formats', 'Always'),
    ('rg_base_formats', 'Always'),
    ('integer_rg_formats', 'Always'),
    ('float_rg_formats', 'Always'),
    ('snorm_formats', 'Always'),
    ('snorm_la_formats', 'Always'),
    ('dxtn_formats', 'S3tc'),
    ('dxtn_srgb_formats', 'S3tc'),
    ('rgtc_formats', 'Rgtc'),
    ('bptc_formats', 'Bptc'),
    ('srgb_formats', 'Always'),
    ('bit10_formats', 'Always'),
    ('packed_float_formats', 'Always'),
    ('exponent_float_formats', 'Always'),
    ('yuv_planar_formats', 'SamplerOnly'),
    ('gles_bgra_formats', 'Gles'),
    ('gles_z32_format', 'Gles'),
    ('gles_bit10_formats', 'Gles'),
    ('astc_formats', 'Astc'),
    ('etc2_formats', 'Etc2'),
    ('gl_base_rgba_formats', 'DesktopGl'),
    ('gl_z32_format', 'DesktopGl'),
    ('gl_bgra_formats', 'DesktopGl'),
    ('gl_bit10_formats', 'DesktopGl'),
]

# `vrend_add_compressed_formats` groups: inserted with SAMPLER_VIEW and no probe.
COMPRESSED_GROUPS = {'dxtn_formats', 'dxtn_srgb_formats', 'rgtc_formats', 'bptc_formats',
                     'astc_formats', 'etc2_formats'}

SWIZZLES = {
    'NO_SWIZZLE': None,
    'RRR1_SWIZZLE': ['X', 'X', 'X', 'One'],
    'RRRG_SWIZZLE': ['X', 'X', 'X', 'Y'],
    'RGB1_SWIZZLE': ['X', 'Y', 'Z', 'One'],
    'OOOR_SWIZZLE': ['Zero', 'Zero', 'Zero', 'X'],
    'BGR1_SWIZZLE': ['Z', 'Y', 'X', 'One'],
    'BGRA_SWIZZLE': ['Z', 'Y', 'X', 'W'],
}


def virgl_enum(path):
    """name -> value for `enum virgl_formats`, explicit values honoured."""
    src = Path(path).read_text()
    body = re.search(r'enum virgl_formats\s*\{(.*?)\};', src, re.S).group(1)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
    v = -1
    out = {}
    for line in body.splitlines():
        m = re.match(r'\s*VIRGL_FORMAT_(\w+)\s*(?:=\s*(\w+))?\s*,?\s*$', line)
        if not m:
            continue
        v = int(m.group(2), 0) if m.group(2) else v + 1
        out[m.group(1)] = v
    return out


def chan(c):
    if c.size == 0:
        return 'Channel::VOID'
    ty = {ufp.VOID: 'Void', ufp.UNSIGNED: 'Unsigned', ufp.SIGNED: 'Signed',
          ufp.FIXED: 'Fixed', ufp.FLOAT: 'Float'}[c.type]
    return 'Channel { ty: ChannelType::%s, normalized: %s, pure: %s, bits: %d }' % (
        ty, 'true' if c.norm else 'false', 'true' if c.pure else 'false', c.size)


def swz(s):
    """A description's swizzle, `None` where gallium says the channel is not there at all."""
    if s == ufp.SWIZZLE_NONE:
        return 'None'
    return 'Some(Swizzle::%s)' % {ufp.SWIZZLE_X: 'X', ufp.SWIZZLE_Y: 'Y', ufp.SWIZZLE_Z: 'Z',
                                  ufp.SWIZZLE_W: 'W', ufp.SWIZZLE_0: 'Zero', ufp.SWIZZLE_1: 'One'}[s]


def desc(f, index):
    layout = {'plain': 'Plain', 'subsampled': 'Subsampled', 'planar2': 'Planar2',
              'planar3': 'Planar3', 's3tc': 'S3tc', 'rgtc': 'Rgtc', 'etc': 'Etc',
              'bptc': 'Bptc', 'astc': 'Astc', 'atc': 'Atc', 'fxt1': 'Fxt1',
              'other': 'Other'}[f.layout]
    cs = {ufp.RGB: 'Rgb', ufp.SRGB: 'Srgb', ufp.YUV: 'Yuv', ufp.ZS: 'Zs'}[f.colorspace]
    equiv = 'None'
    if f.srgb_equivalent and f.srgb_equivalent.name in index:
        equiv = 'Some(Equivalent::Srgb(%d))' % index[f.srgb_equivalent.name]
    elif f.linear_equivalent and f.linear_equivalent.name in index:
        equiv = 'Some(Equivalent::Linear(%d))' % index[f.linear_equivalent.name]
    return ('Description { name: "%s", layout: Layout::%s, block: Block { width: %d, height: %d, '
            'depth: %d, bits: %d }, nr_channels: %d, channels: [%s], swizzle: [%s], '
            'colorspace: Colorspace::%s, is_array: %s, is_bitmask: %s, is_mixed: %s, '
            'is_unorm: %s, is_snorm: %s, equivalent: %s }' % (
                f.name[len('PIPE_FORMAT_'):], layout, f.block_width, f.block_height,
                f.block_depth, f.block_size(), f.nr_channels(),
                ', '.join(chan(c) for c in f.le_channels),
                ', '.join(swz(s) for s in f.le_swizzles),
                cs, *('true' if b else 'false' for b in (f.is_array(), f.is_bitmask(), f.is_mixed(),
                                                       f.is_unorm(), f.is_snorm())),
                equiv))


def view_class(vc):
    """`view_class_bptc_unorm` -> `BptcUnorm`, `view_class_32` -> `Bits32`, ASTC keeps its size."""
    tail = vc[len('view_class_'):]
    if tail.isdigit():
        return 'Bits' + tail
    if tail.startswith('astc_'):
        return 'Astc' + tail[len('astc_'):-len('_rgba')].replace('x', 'X')
    return ''.join(p.capitalize() for p in tail.split('_'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--virgl-hw', required=True)
    ap.add_argument('--gallium', required=True, help='directory holding u_format.yaml and its parser')
    args = ap.parse_args()
    sys.path.insert(0, args.gallium)
    global ufp
    import u_format_parse as ufp
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    numbering = virgl_enum(args.virgl_hw)
    count = numbering['MAX']
    formats = ufp.parse(str(Path(args.gallium) / 'u_format.yaml'))
    by_name = {f.name[len('PIPE_FORMAT_'):]: f for f in formats}
    aliases = {f.alias[len('PIPE_FORMAT_'):]: f.name[len('PIPE_FORMAT_'):] for f in formats if f.alias}
    index = {'PIPE_FORMAT_' + n: v for n, v in numbering.items()}

    lines = ['// GENERATED by virglrs/vrend-gen/gen.py -- do not edit.', '',
             'pub const FORMAT_COUNT: usize = %d;' % count, '',
             '/// Every `virgl_formats` value, described. A number the wire assigns no name to is',
             '/// `None`.',
             'pub static DESCRIPTIONS: [Option<Description>; FORMAT_COUNT] = [']
    described = 0
    for v in range(count):
        name = next((n for n, val in numbering.items() if val == v and n != 'MAX'), None)
        f = by_name.get(name) if name else None
        if f is None:
            assert name is None, '%s is on the wire but not in u_format.yaml' % name
            lines.append('    None, // %d' % v)
        else:
            lines.append('    Some(%s),' % desc(f, index))
            described += 1
    lines += ['];', '']

    lines += ['/// The GL triples, in the groups and order `vrend_formats.c` adds them.',
              'pub static GL_GROUPS: &[GlGroup] = &[']
    for group, when in GROUPS:
        rows = getattr(gl_formats, group)
        lines.append('    GlGroup { name: "%s", when: When::%s, compressed: %s, formats: &[' % (
            group, when, 'true' if group in COMPRESSED_GROUPS else 'false'))
        for fmt, internal, glformat, gltype, sw, vc in rows:
            fmt = aliases.get(fmt, fmt)
            if fmt not in numbering:
                # The C names these through gallium's enum, whose aliases reach numbers the wire
                # header never assigned; a guest cannot send them, so a row is nothing here.
                print('%s: %s has no wire number, dropped' % (group, fmt), file=sys.stderr)
                continue
            swizzle = SWIZZLES[sw]
            swizzle = 'None' if swizzle is None else 'Some([%s])' % ', '.join(
                'Swizzle::%s' % s for s in swizzle)
            lines.append('        GlFormat { format: Format::table(%d), internalformat: %s, glformat: %s, '
                         'gltype: %s, swizzle: %s, view_class: ViewClass::%s },' % (
                             numbering[fmt], internal, glformat, gltype, swizzle,
                             view_class(vc)))
        lines.append('    ] },')
    lines += ['];', '']
    (out / 'formats.rs').write_text('\n'.join(lines))
    print('%d of %d formats described' % (described, count), file=sys.stderr)


if __name__ == '__main__':
    main()
