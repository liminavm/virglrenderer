# venus-gen

The Rust backend for venus-protocol's generator.

`vkxml.py` in `subprojects/venus-protocol-1.0` is a language-neutral model of `vk.xml`.
`vn_protocol.py`'s `Gen` is a *C* backend on top of it — it selects which types and commands venus
serializes, and it emits C statements — and this directory is the Rust one. Only the selection is
reused; none of the emission is.

It lives here rather than in the subproject because the subproject is wrap-managed: the next
`meson subprojects update` re-clones it and eats any local change. `gen.py` puts the checkout on
`sys.path` and imports the model from it, so the pin in `subprojects/venus-protocol.wrap` remains
the single source of truth for which Vulkan the wire speaks.

    gen.py --outdir <dir> [--protocol <checkout>]

`build.rs` runs it into `OUT_DIR` on every build. The output is never checked in — generated code in
the tree is code someone edits by hand, and the next regeneration eats the edit.

Needs `python3` with `mako`, which the C tree already required of anyone building venus.
