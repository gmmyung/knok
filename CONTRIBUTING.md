# Contributing to knok

`knok` is an experimental Rust linalg graph frontend. It parses restricted
Rust function bodies at compile time, lowers them through MLIR, compiles them to
IREE VM bytecode, and exposes typed Rust tensor wrappers.

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

For faster iteration, run focused checks first:

```sh
cargo fmt --all -- --check
cargo test -p knok-core -p knok-compile -p knok-macros
cargo test -p knok
cargo check -p knok --no-default-features
cargo check -p knok-no-std-smoke
```

## Pull Requests

Keep PRs focused on one behavior or API change. Public API changes should update
README examples, tests, and `CHANGELOG.md`.

Before opening a PR:

- Run the narrowest relevant tests while developing.
- Run `scripts/release-check.sh` before requesting review when practical.
- Include the commands you ran in the PR description.
- Update trybuild `.stderr` snapshots only when diagnostics intentionally
  change.
- Keep new dependencies justified and feature-gated unless they are needed by
  the core default path.

## Versioning and Branches

All published crates use the same version: `knok-core`, `knok-compile`,
`knok-macros`, and `knok` are released in lockstep.

Before `1.0.0`, public breaking changes and notable additive features should use
a minor version bump, such as `0.1.0` to `0.2.0`. Fixes, documentation updates,
and internal-only changes should use a patch bump, such as `0.1.0` to `0.1.1`.
After `1.0.0`, use standard semantic versioning.

`main` should stay releasable. Use short topic branches such as
`codex/add-op` or `name/add-op` for PRs. Release branches are not needed unless
an already-published line needs an urgent patch.

Release tags use `vMAJOR.MINOR.PATCH`. Pushing a valid tag publishes to
crates.io after CI verifies crate versions, workspace dependency versions, and
the matching `CHANGELOG.md` section.

## Feature Modes

`knok` must keep these modes healthy:

- default features: hosted runtime, proc macros, and `std`
- `default-features = false`: `no_std + alloc`
- `features = ["half"]`: `f16` and `bf16` tensor element support
- `features = ["macros"]`: compile-time graph expansion without hosted runtime

When changing runtime code, also consider the no-default-features build. When
changing macro or lowering behavior, add tests in the appropriate crate and
update README examples if users see a new API.

## Benchmarks

Criterion benchmarks live under `crates/knok/benches`. Current local snapshots
are recorded in `BENCHMARKS.md`. Treat those numbers as trend checks, not
publish-grade performance claims.
