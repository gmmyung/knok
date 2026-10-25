# Developer Notes

This document is for maintainers and contributors working on knok internals.
See `CONTRIBUTING.md` for the public contribution workflow and `AGENTS.md` for
automation-specific rules.

## Workspace Layout

- `knok-core`: graph IR, op definitions, and type checking.
- `knok-compile`: MLIR lowering and IREE compilation.
- `knok-build`: build-script tracing frontend and generated wrapper codegen.
- `knok-build-macros`: build-time graph registration macros.
- `knok`: public tensors, runtime engine API, feature gates, examples, tests,
  and generated graph imports.

## Compile-Time Flow

`#[knok_build::graph]` marks graph functions that live in `build.rs` or modules
included by `build.rs`. The macro records signature/backend metadata and emits
registration glue; `compile_graphs!` executes the function with traced tensor
inputs on the build host, typechecks the resulting graph, builds MLIR through
melior operation builders, validates it, compiles it in-process with Eerie,
and generates target wrapper modules. Eerie discovers the stock compiler wheel
through Python; `KNOK_IREE_COMPILER` can select a custom shared library. Compiler
initialization is shared across graphs, with per-compilation options and MLIR
contexts. MLIR text crosses the Melior/IREE boundary; raw handles do not.

Multi-output graph ops such as `split` are represented as tuple projection
expressions with a per-call `tuple_id`. Lowering caches projections by
`(inline scope, tuple_id)`: projections from the same source op share one
lowered operation, while two separate source calls remain separate even if their
expression trees are structurally identical.

Single-output traced ops use the same identity rule with per-call `node_id`s.
Cloning a traced tensor handle preserves the `node_id` and reuses the lowered
SSA value; calling the same Rust op expression a second time creates a new
`node_id` and remains a separate graph operation.

## Autograd Layout

Static autograd lives in `knok-core` because it rewrites typed graph IR before
lowering. Keep semantic validation and gradient graph construction there rather
than in `knok-build` or `knok-compile`.

- `crates/knok-core/src/autograd/mod.rs`: public value-and-gradient entry
  points, graph flattening, same-registry graph-call inlining, builder state,
  gradient accumulation, and shared expression/type helpers.
- `crates/knok-core/src/autograd/reverse.rs`: reverse-mode gradient rules for
  differentiable ops. Add new op-specific adjoints here unless they require
  changes to flattening, graph-call inlining, or builder state.
- `crates/knok-core/src/tests/autograd.rs`: graph-transform tests for static
  autograd, including unsupported cases and graph-call scope behavior.
- `crates/knok-core/src/tests/typecheck.rs`: AST, parser, type-checking, and op
  shape rule tests.

Runtime autograd coverage is intentionally separate from transform tests.
`crates/knok/tests/runtime-e2e/build_graphs/` contains build-script graph
fixtures grouped by op family, while
`crates/knok/tests/runtime-e2e/src/tests/autograd/` contains runtime assertions
for generated value-and-gradient wrappers.

## Runtime Flow

The hosted path uses `Engine` to own the IREE instance and device and cache
loaded modules by static VMFB address/length. Prepared `Model` handles borrow
the engine and retain resolved functions, avoiding cache access and lookup in
loops. Inputs borrow host tensors; output counts, dtypes, and full shapes are
validated before host copies. Generated `run` helpers resolve a function per
call, while `call` also creates an engine.

No-default-features builds expose tensors and checked graph/artifact metadata.
Execution traits and helpers are gated by the dependency's `runtime` feature;
generated code uses `knok::__runtime!` so this does not depend on the consuming
crate's feature names. There are no placeholder engines.

## Release Flow

Published crates are versioned in lockstep. Prepare one release commit that
bumps all crate versions, updates internal workspace dependency versions, and
moves `CHANGELOG.md` entries from `Unreleased` into a dated release section.

Check the release metadata and full local validation before tagging:

```sh
scripts/verify-release.sh v0.1.1
scripts/release-check.sh
scripts/publish.sh --dry-run
```

Publishing order is scripted as:

1. `knok-core`
2. `knok-compile`
3. `knok-build-macros`
4. `knok-build`
5. `knok`

Push a `vMAJOR.MINOR.PATCH` tag on the release commit to publish from GitHub
Actions:

```sh
git tag v0.1.1
git push origin v0.1.1
```

The release workflow verifies the tag, crate versions, workspace dependency
versions, and changelog section before running `scripts/publish.sh --execute`.
Manual workflow dispatch remains available for dry-runs and controlled retries.

For crates.io authentication, create an API token on crates.io and store it as
the GitHub Actions secret `CARGO_REGISTRY_TOKEN`. The token needs publish access
for all published crates.

## Platform Notes

The Nix shell pins LLVM/MLIR and provisions IREE 3.13.0 from the Python
wheel unless `EERIE_PYTHON` or `KNOK_IREE_COMPILER` selects a custom installation.
The Eerie runtime is built from source and needs no compiler library search
paths. docs.rs builds the default `knok` feature set, including the hosted
runtime API, so public docs should describe the normal inference path first.

On macOS the shell selects an executable linker that hides host LLVM/MLIR
exports, including Cargo build scripts. Shared-library and proc-macro exports
are preserved. The same adapter handles rustc's LTO options in the optional
`cross-lang-lto` shell, which uses Rust's LLVM toolchain for native runtime code.

See also:

- `docs/compiler.md` for compiler acquisition and cache configuration
- `docs/dtypes.md` for dtype policy
- `docs/lowering.md` for MLIR builder and parsed-attribute policy
- `docs/release-readiness.md` for the pre-release checklist
