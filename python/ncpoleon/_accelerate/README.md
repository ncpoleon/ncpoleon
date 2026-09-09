# `_accelerate` type stubs

`_accelerate` is the compiled Rust extension (`_accelerate.*.so`), which type
checkers cannot introspect, so every `from ncpoleon._accelerate... import ...`
in the runtime code is unresolvable without help. This directory is that help:
a stub-only mirror of the extension's module tree that simply re-exports the
hand-written declarations living next to the Python packages
(`ncpoleon/polynomials/__init__.pyi`, `ncpoleon/relaxations/__init__.pyi`, ...),
so the types stay declared in exactly one place.

It contains no `__init__.py` on purpose. At import time Python's `FileFinder`
prefers a matching extension module over a directory with no `__init__`, so
`_accelerate.*.so` still wins and this directory is never imported; it is only
ever read by a type checker.

Keep the tree in sync with the `sys.modules` registrations in `src/lib.rs`,
`src/polynomials/mod.rs` and the per-polynomial-kind `mod.rs` files.
