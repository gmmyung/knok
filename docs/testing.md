# Testing

`knok` uses layered tests so failures point to the right subsystem.

## Test Layers

- `knok-core`: AST, tensor type parsing, shape rules, and type checking.
- `knok-compile`: MLIR lowering, cache behavior, backend flags, and MLIR
  verification.
- `knok-build`: proc macro diagnostics and generated wrapper code shape.
- `knok`: public tensor/runtime/backend API behavior.
- `knok-build-tracing-runtime`: small build-script tracing smoke test.
- `knok-runtime-e2e`: actual build.rs tracing, IREE compilation, generated
  wrapper import, runtime execution, and output assertions.
- `knok-no-std-smoke`: no-std/check-only generated wrapper coverage with stub
  artifacts.

## Commands

```sh
nix develop --command scripts/release-check.sh
nix develop --command cargo test -p knok-runtime-e2e
nix develop --command scripts/coverage.sh
nix develop --command scripts/benchmark.sh
```

`coverage.sh` writes:

- `target/coverage/lcov.info`
- `target/coverage/html/index.html`
- `target/coverage/badge/coverage.svg`

Runtime E2E tests should assert actual tensor outputs, not only generated MLIR
text. MLIR string assertions are reserved for internal lowering invariants such
as cache reuse or a specific lowering form.

Before a release, also run the dry-run publishing checklist in
[release-readiness.md](release-readiness.md).

`scripts/benchmark.sh` is not part of `release-check.sh`. It runs the separate
release-mode benchmark crate under `benchmarks/runtime` and should be run when
changing runtime dispatch, backend selection, tensor conversion, or generated
wrapper code. It writes `target/benchmark-summary.csv` and
`target/benchmark-summary.json` inside the benchmark crate.
