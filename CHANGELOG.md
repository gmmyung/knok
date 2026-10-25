# Changelog

All notable user-facing changes to this project should be recorded here.

This project follows a lightweight Keep a Changelog style. Dates use
`YYYY-MM-DD`.

Before publishing, move the relevant `Unreleased` entries into a versioned
section such as `## 0.1.1 - 2026-06-26`. Release tags use the matching
`v0.1.1` form.

## Unreleased

### Added

- Added generated `run_into` wrappers and `Graph::run_into` for reusing host
  output tensor storage, including mixed-dtype tuples.
- Added opt-in `cross-lang-lto` for the native runtime, with a matching Linux
  and macOS Nix shell.
- Added experimental build-time value-and-gradient graph generation through
  `knok-build` `grad_graphs!` macros, including scalar-loss autograd coverage
  for arithmetic, math/reduction/layout operations, linalg contractions,
  pooling and convolution, selection ops, shape/indexing ops, same-registry
  graph calls, documented edge-case gradient policies, no-grad index/predicate
  values, and explicit `detach` stop-gradient.
- Added deterministic random tensor tracing helpers: compile-time
  `uniform_static` / `randint_static` constants and runtime key-threaded
  `rand` / `uniform` / `randint` graph ops.
- Added an optional `serde` feature for fixed-rank tensor
  `Serialize`/`Deserialize` support using a `{ shape, data }` representation.

### Changed

- Updated Melior to 0.28.2 with LLVM/MLIR 23, Syn to 3, and the benchmark
  harness to Criterion 0.8.2 and ndarray 0.17.2. Refreshed both dependency
  lockfiles and the Nix toolchain, and removed unused compiler dependencies.
- Moved coverage tooling to `nix develop .#coverage`, using Rust-matched LLVM
  readers and keeping profiling dependencies out of the default shell.
- Compile graphs in-process through Eerie's automatically discovered compiler
  library, sharing initialization across graphs and eliminating compiler
  subprocesses and temporary MLIR files. `KNOK_IREE_COMPILER` selects a custom
  library; the old `KNOK_IREE_COMPILE` executable override is removed.
- Configure macOS build-host exports in the Nix shell so Melior and the stock
  IREE wheel can coexist, including in Cargo build scripts. Remove native
  library paths and rpaths from the shell's Rust flags.
- Lower axis softmax with the same stable max-shifted reductions as global
  softmax, avoiding an IREE 3.13 release-mode distribution error in its gradient.
- Preserve compiler diagnostic chains and graph names in build-script errors.
- Updated the runtime to Eerie 0.7.0 and aligned the development compiler with
  IREE 3.13.0. Existing VMFBs from IREE 3.11 or earlier must be recompiled;
  build-time graph caches are invalidated automatically by the compiler version.
- Rejected reused `Expr::Node` ids with different expression payloads during
  `knok_core::type_check` instead of reusing cached node output types.

## 0.3.3 - 2026-07-02

### Fixed

- Validated static creation call arguments for hoisted graph let bindings so
  check-only builds reject non-literal `arange` and `linspace` calls before
  lowering.

## 0.3.2 - 2026-07-02

### Fixed

- Flattened build-time traced node DAGs into graph let bindings to avoid
  exponential expression cloning and memory blowups in larger demos.
- Memoized repeated traced node type checking while preserving tuple projection
  validation for manually constructed graphs.

## 0.3.1 - 2026-07-02

### Fixed

- Added crate-local README files and explicit crate readme metadata for the
  published workspace crates.

## 0.3.0 - 2026-07-01

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
- Added static NHWC `max_pool2d` and `avg_pool2d` graph operations with
  build-time shape validation and runtime E2E coverage.
- Added a Criterion max-pooling benchmark with an `ndarray` baseline.

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
