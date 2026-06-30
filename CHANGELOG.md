# Changelog

All notable user-facing changes to this project should be recorded here.

This project follows a lightweight Keep a Changelog style. Dates use
`YYYY-MM-DD`.

Before publishing, move the relevant `Unreleased` entries into a versioned
section such as `## 0.1.1 - 2026-06-26`. Release tags use the matching
`v0.1.1` form.

## Unreleased

### Added

- Added an experimental `knok-build` build-time tracing frontend that records
  host-executed graph functions from `build.rs`, emits generated target wrappers
  into `OUT_DIR`, and imports them with `knok::generated_graphs!`.
- Added Rust-native build-time tracing wrappers for static shape/indexing ops,
  creation ops, reductions, linalg, conv2d options, predicates, and
  floating-point math.
- Added build-time tracing compile-fail coverage for graph macro diagnostics
  and build script type-checking failures.
- Added Nix-provisioned `cargo-llvm-cov` coverage tooling and a GitHub Actions
  coverage workflow with a README badge.
- Added coverage HTML and badge artifacts, a configurable line coverage gate,
  and additional public API coverage for artifact metadata and runtime errors.
- Added more rustdoc for the `knok-build` build-script API.
- Added release-readiness, compiler setup, dtype policy, and lowering
  architecture documentation.
- Added `compile_mlir_models!` and `compile_mlir_models_with_options!` for
  build-time compilation of external `.mlir` files into generated wrappers.
- Added feature-gated Vulkan/SPIR-V and CUDA backend/driver variants, and made
  Metal/SPIR-V a macOS-only backend.
- Added runtime workflow documentation and a standalone release-mode runtime
  benchmark harness.
- Added Criterion runtime benchmarks for matmul, batched matmul, large
  elementwise graphs, reductions, softmax, layout operations, MLP, and conv2d.

### Changed

- Replaced the parser-first graph authoring surface with build-time host
  tracing through `knok-build`.
- Changed traced op reuse to use explicit node and tuple projection identities
  instead of thread-local tracing state.
- Renamed the default hosted execution feature from `host-runtime` to `runtime`
  and let docs.rs build the default runtime-enabled API surface.
- Replaced additional tensor lowering dense-array attributes with structured
  melior attribute builders.
- Replaced generated raw runtime invocation glue with typed `Graph<I, O>`
  handles. Generated `run` / `call` wrappers remain the supported execution
  surface; callers should not call raw runtime internals directly.
- Simplified the low-level `Graph` API to `artifact`, `run`, and `run_once`;
  generated modules still expose `call` for one-shot execution.
- Changed the runtime benchmark baseline from hand-written scalar loops to
  `ndarray`.
- Changed `scripts/benchmark.sh` to run the Criterion benchmark target directly.

### Fixed

- Exported `squeeze` from `knok_build::prelude`.
- Namespaced external MLIR VMFB artifact filenames so they cannot overwrite
  traced graph artifacts with the same logical name.

### Removed

- Removed the old `#[knok::graph]` parser macro, parser-specific tests, UI
  snapshots, graph examples, and benchmark entrypoint.

## 0.2.0 - 2026-06-28

### Added

- Added typed backend and driver selection for graph and MLIR model macros.
- Added `Tensor5` and `Tensor6` containers and rank-6 graph support.
- Added rank-6 NumPy-style shape, broadcasting, selection, and reduction test
  coverage for existing graph ops.
- Added static layout operations for shape-inferred `transpose` axes,
  `permute_dims`, `swapaxes`, `moveaxis`, `split`, `tile`, `repeat`, `pad`,
  `flip`, and `roll`.
- Added static linalg contraction graph ops: `dot`, `vecdot`, `inner`, `outer`,
  `trace`, and `diagonal`.
- Added full-tensor and axis-aware `prod`, `max` / `amax`, `min` / `amin`,
  `argmin`, `var`, `std`, and `ptp` graph ops.
- Added tensor-index `gather` and `take_along_axis` graph operations with
  static dtype, rank, and shape validation.
- Added static graph tensor creation helpers: `zeros_like`, `ones_like`,
  `full_like`, static literal `arange`, static literal `linspace`, and
  `eye`/`identity`.
- Added NumPy-style elementwise math graph ops: `square`, `reciprocal`,
  `floor`, `ceil`, `round`, `rint`, `sin`, `cos`, `tan`, `log2`, `log10`,
  `log1p`, `exp2`, and `expm1`.
- Added contributor, developer, agent, and changelog documentation.
- Added tag-triggered release automation with release metadata validation.
- Added README badges for CI, docs.rs, and crates.io.

### Changed

- Changed `GraphArtifact` metadata from shape-only fields to typed
  `input_descs` and `output_descs`, and exported `DType` and `TensorDesc`.
- Replaced string backend and driver macro attributes with typed `Backend` and
  `Driver` paths.
- Changed `transpose` to match NumPy's default rank-N axis reversal instead of
  only accepting rank-2 tensors.
- Changed `knok::Error` to implement `core::error::Error`, including no-std
  builds, instead of gating the trait impl on `std`.

### Fixed

- Nothing yet.

### Removed

- Nothing yet.
