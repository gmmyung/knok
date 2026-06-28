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
inputs on the build host, typechecks the resulting graph, emits textual MLIR,
validates it with `melior`, invokes `iree-compile`, and generates target
wrapper modules. `KNOK_IREE_COMPILE` can point at a specific compiler binary;
otherwise `knok-compile` expects `iree-compile` on `PATH`.

Multi-output graph ops such as `split` are represented as tuple projection
expressions with a per-call `tuple_id`. Lowering caches projections by
`(inline scope, tuple_id)`: projections from the same source op share one
lowered operation, while two separate source calls remain separate even if their
expression trees are structurally identical.

Single-output traced ops use the same identity rule with per-call `node_id`s.
Cloning a traced tensor handle preserves the `node_id` and reuses the lowered
SSA value; calling the same Rust op expression a second time creates a new
`node_id` and remains a separate graph operation.

## Runtime Flow

The hosted path uses `Engine` for repeated inference. Generated
`graphs::<name>::run` wrappers reuse the engine's IREE instance, driver, device,
loaded module, and compiled artifact. Convenience wrappers are useful for
examples and small one-off calls but include runtime setup cost.

No-default-features builds expose compile-time artifacts but do not provide the
hosted runtime convenience path.

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

The Nix shell pins LLVM/MLIR and provisions the IREE compiler from the Python
wheel when `LIB_IREE_COMPILER` is not already set. docs.rs is configured to
build `knok` with `no-default-features` and `features = ["std"]`, so public docs
must not require the hosted runtime feature.
