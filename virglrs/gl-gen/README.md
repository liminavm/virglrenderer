# gl-gen

Generates the GLES and EGL bindings (`OUT_DIR/gl/{types,gles,egl}.rs`) from the Khronos
registries, at build time, the way `venus-gen` generates the venus decoder from vk.xml. The
output is never checked in.

`registry/gl.xml` and `registry/egl.xml` are the Khronos OpenGL and EGL API registries
(Apache-2.0), vendored unmodified from libepoxy at d1f952c4 (2026-04-03). Update them by copying
newer ones in; nothing here depends on their version beyond the features and extensions
`gen.py` names, and it fails the build if one of those is missing.

Which entry points and constants exist is decided in `gen.py`'s lists, once. An extension the
driver lacks costs nothing: its entry points load as `None`, and `Gles::has_all_of` /
`Egl::has_all_of` answer for it by name.
