# Release Readiness

Use this checklist before cutting the next minor release.

## API and Docs

- Confirm README examples compile against the current public API.
- Confirm crate-level docs explain build-time tracing and hosted runtime use.
- Confirm feature modes are documented: default runtime, no-default-features,
  `half`, and `embedded-runtime`.
- Confirm known limitations are documented rather than hidden.

## Compiler and Runtime

- Confirm `docs/compiler.md` matches the current `iree-compile` lookup logic.
- Confirm `docs/backends.md` matches supported compile backends and runtime
  drivers.
- Confirm runtime E2E tests cover creation, elementwise, linalg, reductions,
  indexing, layout, multi-output graphs, and a reusable engine path.

## Validation

Run:

```sh
nix develop --command scripts/release-check.sh
nix develop --command scripts/publish.sh --dry-run
```

For a tag release, also run:

```sh
nix develop --command scripts/verify-release.sh vMAJOR.MINOR.PATCH
```

## Publishing

All published crates are versioned in lockstep:

1. `knok-core`
2. `knok-compile`
3. `knok-build-macros`
4. `knok-build`
5. `knok`

GitHub Actions publishes from a `vMAJOR.MINOR.PATCH` tag after release metadata
validation passes and `CARGO_REGISTRY_TOKEN` is available.
