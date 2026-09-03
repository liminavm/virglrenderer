# vrend-gen

Generates the classic renderer's format tables (`OUT_DIR/vrend/formats.rs`) at build time, the
way `gl-gen` and `venus-gen` generate theirs. The output is never checked in.

Its inputs are read where they already are, so each exists once:

- `src/virgl_hw.h` -- the wire numbering of `enum virgl_formats`, the ABI shared with the guest.
- `src/gallium/auxiliary/util/u_format.yaml`, parsed by the `u_format_parse.py` beside it
  (Mesa, MIT) -- what each format is: block geometry, channels, swizzle, colour space. The C's
  `util_format_description` table is generated from the same file.

`gl_formats.py` is this directory's own: the GL triple `vrend_formats.c` gives each format,
converted once from the C's tables, grouped as the C groups them and in its order. The condition
each group is added under, and whether it is probed or inserted blind, is `gen.py`'s `GROUPS`.
Needs python3 with PyYAML.
