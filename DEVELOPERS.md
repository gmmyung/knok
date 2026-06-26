# Developer Notes

This document is for maintainers and contributors working on knok internals.
See `CONTRIBUTING.md` for the public contribution workflow and `AGENTS.md` for
automation-specific rules.

## Workspace Layout

- `knok-core`: graph AST, parser, op definitions, and type checking.
- `knok-compile`: MLIR lowering, IREE compilation, artifact generation, and
  proc-macro codegen helpers.
- `knok-macros`: procedural macro entrypoints.
- `knok`: public tensors, runtime engine API, feature gates, examples, tests,
  and benchmarks.

## Compile-Time Flow

`#[knok::graph]` parses a restricted Rust function body into the core graph IR.
The compiler typechecks the graph, emits MLIR with `melior`, invokes the IREE
compiler, embeds the resulting VMFB artifact, and generates typed wrapper
functions.

`knok::mlir_model!` follows the same artifact path but starts from a local MLIR
file and an optional typed signature.

## Runtime Flow

The hosted path uses `Engine` for repeated inference. Generated `*_run` and
`invoke_run` wrappers reuse the engine's IREE instance, driver, device, loaded
module, and compiled artifact. Convenience wrappers are useful for examples and
small one-off calls but include runtime setup cost.

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
3. `knok-macros`
4. `knok`

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
for all four crates.

## Platform Notes

The Nix shell pins LLVM/MLIR and provisions the IREE compiler from the Python
wheel when `LIB_IREE_COMPILER` is not already set. docs.rs is configured to
build `knok` with `no-default-features` and `features = ["std"]`, so public docs
must not require the hosted runtime feature.
