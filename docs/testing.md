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
nix develop .#coverage --command scripts/coverage.sh
nix develop --command scripts/benchmark.sh
```

Coverage is validated on Linux CI. Its separate shell keeps profiling tools out
of ordinary compiler/runtime development.

`coverage.sh` writes:

- `target/coverage/lcov.info`
- `target/coverage/html/index.html`
- `target/coverage/badge/coverage.svg`

Runtime E2E tests should assert actual tensor outputs, not only generated MLIR
text. MLIR string assertions are reserved for internal lowering invariants such
as cache reuse or a specific lowering form.

## Adding Tests

- Add parser, type-checker, op shape rule, and AST behavior tests under
  `crates/knok-core/src/tests/typecheck.rs`.
- Add static autograd graph-transform tests under
  `crates/knok-core/src/tests/autograd.rs`. These tests should cover generated
  output signatures, unsupported differentiated cases, graph-call inlining, and
  graph-size regressions before runtime execution is involved.
- Add runtime E2E graph fixtures under
  `crates/knok/tests/runtime-e2e/build_graphs/`, grouped by op family. Register
  new forward fixtures in `compile_graphs!` and value-and-gradient fixtures in
  `grad_graphs_with_options!` from `crates/knok/tests/runtime-e2e/build.rs`.
- Add generated-wrapper runtime assertions under
  `crates/knok/tests/runtime-e2e/src/tests/`. Autograd runtime assertions are
  split by op family under `src/tests/autograd/`.

Before a release, also run the dry-run publishing checklist in
[release-readiness.md](release-readiness.md).

`scripts/benchmark.sh` is not part of `release-check.sh`. It runs the separate
Criterion benchmark crate under `benchmarks/runtime` and should be run when
changing runtime dispatch, backend selection, tensor conversion, or generated
wrapper code.
