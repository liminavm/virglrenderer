# virglrenderer → virglrs

This tree is being rewritten in Rust. The plan is `docs/rust-rewrite.md`; the test floor is
`harness/README.md`. The C implementation still here is the reference the harness records
goldens from, and nothing else — it is a hard fork with no obligation to upstream past the
switch point, and no obligation to older users at any point.

## Why Rust, and what follows from it

We are not porting for fashion. The language is the point: it is meant to remove whole classes
of the bugs this renderer has cost us. Every rule below is that reason, applied.

**Keep `unsafe` minimal and wrapped.** Unsafe lives in named modules and nowhere else. They are:
the Vulkan and GLES bindings (`vulkan.rs`, `venus/driver.rs`); the IOSurface and Metal bindings
(`metal.rs`), which are the only Objective-C in the tree; the guest-memory
mapping (`guest_mem.rs`); the C shim (`ffi.rs`, `abi.rs`); and the venus wire decoder (`venus/cs.rs`), which owns
the arena every decoded pointer points into. Every unsafe block carries a `SAFETY:` comment naming
the invariant and who upholds it. The rest of the renderer is safe Rust, and an unsafe block
outside these modules is a design failure, not a shortcut.

`metal.rs` exists because the scanout path has no Vulkan route on this platform: KosmicKrisp
imports an IOSurface but does not export one, so a surface has to be minted host-side. It owns the
surface's lifetime and hands the rest of the tree a safe handle — never a raw `IOSurfaceRef`, and
never an id, which is worth nothing the moment its surface dies.

The list is exhaustive on purpose: a module that starts needing unsafe is a module whose types are
wrong. Handlers in `venus/context.rs` in particular must stay safe — when one needs a raw pointer,
the fix is for the generator to hand it a reference or a slice, not for the handler to dereference.

**Make bad state unrepresentable; fail the build, not the run.** Use the type system in our
favour. The concrete form this takes here: virgl is a soup of bare `uint32_t` — resource
handles, context ids, fence ids, object handles, ring ids, blob ids, all mutually
interchangeable and none of them checked. Each gets its own newtype. Where an operation is only
valid in one state, that state is a distinct type, not a boolean someone must remember to test.

**When the language can't enforce it, crash loudly.** `assert!` and `.expect()` are runtime
safety nets that surface a problem at the moment it is introduced, not three frames later.
Never silently ignore, never paper over, never `let _ =` a real error.

**Except at the trust boundary, where a guest is never allowed to crash us.** An assert fires on
a violated *host* invariant — something we got wrong. Malformed, hostile or nonsensical input
from the guest is not that: a guest that sends a bad ring, an impossible descriptor or an
out-of-range handle must be rejected and its context poisoned, never allowed to abort the
worker. One guest must not be able to take down the process. Validate and reject at the
boundary; assert behind it.

**Panics abort; they never unwind into C.** Unwinding across `extern "C"` is undefined
behaviour, so the crate builds with `panic = "abort"`. A loud abort is the behaviour we want
anyway, and it means the FFI shim needs no unwind machinery.

**No global mutable state.** virglrenderer's C is built on file-scope statics and an implicit
current context (`force_ctx_0`), which is where the data races we have chased live. State hangs
off an explicitly owned root and is reached through it. The C ABI's implicit global becomes a
single owned root at the shim — one place, not a habit.

**C and C++ disappear; FFI never dictates design.** The C-ABI dylib is a compatibility shim with
an end date. limina reaches us today through rutabaga in its libkrun fork, which links the C
dylib; **the plan is to change rutabaga to consume this crate's Rust API directly**, at which
point the shim serves no first-party caller. So the consumer to design for is a Rust one, and
**that API is the product; the C ABI is a translation of it.** So wherever the two are
in tension, the ABI takes the hit — in performance, in efficiency, in ergonomics. Never the other
way round.

Concretely: C's idiosyncrasies stop at `ffi.rs` and never leak inward. No errno in a Rust
signature, no bare-integer id where a newtype belongs, no `bool` standing in for a `Result`, no
`repr(C)` argument struct in a Rust API, no length travelling beside the array it measures, and no
concept — like the implicit global context — that exists only because a C header says so. Review
every signature against one question: **would this still make sense if `ffi.rs` were deleted?** If the answer needs a C header to explain it, the
translation belongs in the shim.

**Two values that must agree are one value.** A count and its pointer, an offset and its base, a
capacity and its buffer. Passed onward as a pair, every layer in between has to be trusted to keep
them in step — and the layer that quietly repairs a mismatch is worse than the one that crashes,
because it reports success for work it did not do. Reconcile the pair once, at the boundary that
knows the truth (the decoder for a wire array, `ffi.rs` for a C one), and pass a slice or a newtype
from there on. Where the wire genuinely lets the two disagree, that same boundary is where the
guest's version is rejected — never averaged, never clamped. The pair may reappear only as the
arguments of the foreign call that needs it, rebuilt from the single value.

The rule is about state as much as about wire pairs, and that is where it has been learned the
hard way. Two containers holding one fact, two maps each holding a copy of a handle, two writers
deciding one thing — each has cost us a bug. The same reconciliation applies: one owner, and
everything else holds a key to it rather than a copy of it.

**A lifetime mismatch is a design bug, and wants a structural fix.** When something outlives what
it describes — an id still naming a freed handle, a record surviving the thing it recorded, a
cached answer outliving what it was true of — the fix is never another purge to remember at
another destroy site. Adding one leaves the next destroy path to be found by whoever hits it in
production. Restructure so the stale thing cannot be reached: make one place the owner, have
everything else name it indirectly, and let a reference to something gone fail on its own. The
test of a proposed fix is whether a future call site can still get it wrong. If it can, it is a
patch, not a fix.

**Generated code is generated.** The venus decoder is emitted from templates. Fixing a bug by
editing generated output puts it somewhere no one will find it and the next regeneration eats
it. Fix the template.

## Working here

- Every behaviour fix carries a harness case that would have caught it (`harness/README.md`).
- **A documented C limitation is a hypothesis until the harness reproduces it.** The C tree
  carries workarounds whose comments assert things that are no longer true; porting one
  faithfully carries the folklore forward and hides that the real bug was fixed elsewhere.
  Reproduce it against a pinned score first, then port it or delete it.
- Commit as work finishes. Never `git add -A` — this tree has untracked local files that must
  not be committed. Never push without asking.
