# Lowering Notes

`knok-compile` lowers typed graph IR to MLIR with melior operation builders.
Graph bodies should not be assembled by concatenating textual MLIR operations.

## Builder Boundary

The lowering layer directly builds:

- modules, functions, blocks, regions, and operations
- SSA operands and result types
- common tensor/linalg operation regions
- dense i32/i64 array attributes where melior exposes typed constructors

Some MLIR fragments are still parsed from short strings because melior does not
currently expose convenient typed builders for every attribute form used here.
Those cases should stay centralized in lowering helpers.

Allowed parsed snippets:

- scalar and dense constant attributes
- affine maps for `linalg.generic`
- linalg iterator type enum attributes
- comparison predicate enum attributes
- reassociation attributes
- MLIR type strings for statically shaped tensor types

When melior exposes a suitable typed constructor, prefer that over a parsed
string and add a focused regression test.

## Regression Cases

Lowering tests should keep covering:

- rank-0 tensors and scalar slices
- empty dense-array attributes
- axis and shape attributes on tensor ops
- linalg regions with explicit yields
- tuple projection reuse for multi-output graph ops
- generated MLIR verification before invoking `iree-compile`
