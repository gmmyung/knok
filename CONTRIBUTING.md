# Contributing to knok

`knok` is an experimental Rust linalg graph frontend. Build scripts execute
traced Rust graph functions on the compile host, lower the recorded graph
through MLIR, compile it to IREE VM bytecode, and expose typed Rust tensor
wrappers in the target crate.

## Development Setup

Use the Nix development shell. It provides Rust, LLVM/MLIR, `melior`
dependencies, and the IREE compiler library used by build scripts and tests.

```sh
nix develop
```

The main validation command is:

```sh
scripts/release-check.sh
```

Coverage reports use the same Nix-provisioned toolchain:

```sh
scripts/coverage.sh
```

The script writes an LCOV report to `target/coverage/lcov.info` and prints a
line coverage summary.

For faster iteration, run focused checks first:

```sh
cargo fmt --all -- --check
cargo test -p knok-core -p knok-compile -p knok-build -p knok-build-macros -p knok-macros
cargo test -p knok
cargo test -p knok-build-tracing-runtime
cargo check -p knok --no-default-features
cargo check -p knok-no-std-smoke
scripts/coverage.sh
```

## Pull Requests

Keep PRs focused on one behavior or API change. Public API changes should update
README examples, tests, and `CHANGELOG.md`.

Before opening a PR:

- Run the narrowest relevant tests while developing.
- Run `scripts/release-check.sh` before requesting review when practical.
- Include the commands you ran in the PR description.
- Keep new dependencies justified and feature-gated unless they are needed by
  the core default path.

## Versioning and Branches

All published crates use the same version: `knok-core`, `knok-compile`,
`knok-build-macros`, `knok-build`, `knok-macros`, and `knok` are released in
lockstep.

Before `1.0.0`, public breaking changes and notable additive features should use
a minor version bump, such as `0.1.0` to `0.2.0`. Fixes, documentation updates,
and internal-only changes should use a patch bump, such as `0.1.0` to `0.1.1`.
After `1.0.0`, use standard semantic versioning.

`main` should stay releasable and is protected on GitHub. Use short topic
branches such as `codex/add-op` or `name/add-op` for PRs. Required checks are
`fmt`, `core`, `no-std`, `docs`, `host-runtime`, and `coverage`. Release
branches are not needed unless an already-published line needs an urgent patch.

Release tags use `vMAJOR.MINOR.PATCH`. Pushing a valid tag publishes to
crates.io after CI verifies crate versions, workspace dependency versions, and
the matching `CHANGELOG.md` section.

## Feature Modes

`knok` must keep these modes healthy:

- default features: hosted runtime, `mlir_model!`, and `std`
- `default-features = false`: `no_std + alloc`
- `features = ["half"]`: `f16` and `bf16` tensor element support
- build-time graph tracing through `knok-build` as a build-dependency

When changing runtime code, also consider the no-default-features build. When
changing tracing, macro, or lowering behavior, add tests in the appropriate
crate and update README examples if users see a new API.

## Benchmarks

Benchmark notes are recorded in `BENCHMARKS.md`. Treat those numbers as trend
checks, not publish-grade performance claims.
