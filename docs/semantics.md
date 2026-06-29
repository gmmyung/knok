# Tensor Semantics

`knok` uses statically shaped tensor types. Graph signatures name both dtype and
shape in Rust types, and the build script rejects shape or dtype mismatches
before producing VMFB artifacts.

## Rank and Layout

- Public tensor containers are `Tensor0` through `Tensor6`.
- Build-time aliases `T0` through `T6` mirror the target-side tensor types.
- Tensor storage is row-major.
- Rank-0 tensors are scalar tensors with one stored element.

## Dtypes

Supported element types are:

- `bool`
- `f32`, `f64`
- `i32`, `i64`
- `f16`, `bf16` when the `half` feature is enabled

Implicit dtype promotion is intentionally not part of the current API. Graphs
should use matching dtypes, and future casting operations should be explicit.
See [dtypes.md](dtypes.md) for the support matrix and operation categories.

## Broadcasting

Elementwise tensor operations use NumPy-style trailing-dimension broadcasting
when the type checker can prove the resulting shape. Scalar tensors can be used
with shaped tensors. Incompatible shapes are build-time errors.

## Axes

Axis arguments are `usize` values known at build time. Negative axes are not
accepted. Axis-bearing operations validate rank, bounds, and target shape during
build-time type checking.

## Multi-Output Graphs

Graph functions can return tuples. Generated wrappers preserve tuple order and
dtype. Raw single-output helpers reject multi-output artifacts; generated typed
wrappers should be preferred.

## Operation Surface

- Elementwise arithmetic: `+`, `-`, `*`, `/`, unary `-`, `abs`, `minimum`,
  `maximum`, `clip`, `pow`, `square`, `reciprocal`, `relu`.
- Comparisons and predicates: `greater`, `greater_equal`, `less`,
  `less_equal`, `equal`, `not_equal`, logical ops, `isnan`, `r#where`.
- Reductions: `sum`, `prod`, `mean`, extrema, arg extrema, `var`, `std`, `ptp`,
  `all`, `any`, and axis variants.
- Shape and indexing: `reshape`, `broadcast`, `squeeze`, `unsqueeze`, `slice`,
  `take`, `gather`, `take_along_axis`, `split`, `concat`, `stack`, `tile`,
  `repeat`, `pad`, `flip`, `roll`.
- Axis movement: `transpose`, `transpose_axes`, `permute`, `permute_dims`,
  `swapaxes`, `moveaxis`.
- Linalg and convolution: `matmul`, `dot`, `vecdot`, `inner`, `outer`, `trace`,
  `diagonal`, `conv2d`, `conv2d_options`.
- Creation and math: `zeros_like`, `ones_like`, `full_like`, `arange`,
  `linspace`, `eye`, `identity`, floating-point math functions.
