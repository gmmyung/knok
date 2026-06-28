# Changelog

All notable user-facing changes to this project should be recorded here.

This project follows a lightweight Keep a Changelog style. Dates use
`YYYY-MM-DD`.

Before publishing, move the relevant `Unreleased` entries into a versioned
section such as `## 0.1.1 - 2026-06-26`. Release tags use the matching
`v0.1.1` form.

## Unreleased

### Added

- Added rustdoc coverage for public APIs across all published crates and made
  CI/release checks fail on missing public item documentation.

### Changed

- Nothing yet.

### Fixed

- Nothing yet.

### Removed

- Nothing yet.

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
