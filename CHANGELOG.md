# Changelog

All notable user-facing changes to this project should be recorded here.

This project follows a lightweight Keep a Changelog style. Dates use
`YYYY-MM-DD`.

Before publishing, move the relevant `Unreleased` entries into a versioned
section such as `## 0.1.1 - 2026-06-26`. Release tags use the matching
`v0.1.1` form.

## Unreleased

### Added

- Added typed backend and driver selection for graph and MLIR model macros.
- Added `Tensor5` and `Tensor6` containers and rank-6 graph support.
- Added rank-6 NumPy-style shape, broadcasting, selection, and reduction test
  coverage for existing graph ops.
- Added contributor, developer, agent, and changelog documentation.
- Added tag-triggered release automation with release metadata validation.

### Changed

- Changed `GraphArtifact` metadata from shape-only fields to typed
  `input_descs` and `output_descs`, and exported `DType` and `TensorDesc`.
- Replaced string backend and driver macro attributes with typed `Backend` and
  `Driver` paths.
- Changed `transpose` to match NumPy's default rank-N axis reversal instead of
  only accepting rank-2 tensors.

### Fixed

- Nothing yet.

### Removed

- Nothing yet.
