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

## Gradient Graphs

`knok-build` can generate an experimental value-and-gradient graph from a
single-output graph whose output is a rank-0 floating-point loss. The generated
graph is named `{graph_name}_value_and_grad` and returns the original loss
followed by gradients for every floating-point input, in input order.

Gradient generation is a build-time static graph transform, not a runtime tape.
The MVP differentiates elementwise arithmetic and math (`abs`, `exp`, `exp2`,
`expm1`, `log`, `log1p`, `log2`, `log10`, `sin`, `cos`, `tan`, `tanh`, `relu`,
`sigmoid`, `square`, `sqrt`, `reciprocal`, `pow`), selection ops (`where`,
`minimum`, `maximum`, `clip`), reductions (`sum`, `mean`, `prod`, `min`, `max`,
`ptp`, `var`, `std`, `softmax`), linalg ops (`matmul`, `dot`, `vecdot`,
`inner`, `outer`, `trace`, `diagonal`, `conv2d`), pooling ops (`max_pool2d`,
`avg_pool2d`), and shape/indexing ops (`reshape`, `broadcast`, `squeeze`,
`unsqueeze`, `transpose`, `permute`, `permute_dims`, `swapaxes`, `moveaxis`,
`flip`, `roll`, `pad`, `slice`, `concat`, `stack`, `split`, `take`, `gather`,
`take_along_axis`, `tile`, `repeat`). Graph calls registered in the same
`grad_graphs!` invocation are inlined before differentiation.
Non-floating outputs such as `argmax`, `argmin`, boolean predicates, and
boolean reductions may feed index or condition operands, but they do not
receive gradients. Unsupported differentiated floating-point operations or
unsupported static cases are build-time errors.
Random tensors are treated as non-differentiable sources. They may feed
differentiable operations, in which case gradients flow to differentiable
operands that consume the sampled tensor but not to the random source or key.

Use `detach(x)` to preserve `x` in the forward graph while stopping gradient
propagation through that value. Unsupported single-output operations are allowed
behind `detach` as long as the detached value is not differentiated through.

Gradient rules are intentionally explicit in edge cases. Rounding, index, and
predicate ops (`ceil`, `floor`, `rint`, `round`, comparisons, logical ops,
`isnan`, `argmax`, `argmin`, `all`, `any`) do not propagate gradients. Reduction
extrema (`min`, `max`, `ptp`) and `max_pool2d` route the full upstream gradient
to every selected value, so ties can duplicate gradient rather than split it.
Binary `minimum`/`maximum` break exact ties toward the left operand. `clip`
propagates through the value at inclusive bounds and to
the bound operands only outside the clipped range. `prod` uses the algebraic
`output / input` local gradient, so zeros in the reduced input remain an MVP
edge case that can produce non-finite gradients.

Some reverse rules expand static loops into graph IR. `conv2d`, pooling,
`gather`, `take_along_axis`, `tile`, and `repeat` gradients are generated from
existing tensor primitives instead of dedicated backward kernels, so larger
static shapes can increase generated graph size and compile time. `repeat`
gradients that require an intermediate rank above the Tensor6 limit are rejected
at build time.

## Random Tensors

Random tensor helpers are deterministic and seed-based.

- `uniform_static(seed, low, high)` and `randint_static(seed, low, high)` run
  during build-time lowering and embed the generated tensor as an MLIR dense
  constant.
- `rand(key)`, `uniform(key, low, high)`, and `randint(key, low, high)` run in
  the compiled graph and return `(tensor, next_key)`. Passing the same key
  produces the same tensor; pass `next_key` forward to advance the stream.
- Floating helpers sample `[low, high)`. `rand(key)` is shorthand for
  `uniform(key, 0.0, 1.0)`.
- Integer helpers sample `[low, high)` for `i32` and `i64` outputs.
- Static helper bounds are validated at build time. Runtime helpers currently
  take literal bounds from the traced graph; callers should provide `low < high`.

## Pooling

`max_pool2d` and `avg_pool2d` use NHWC rank-4 input layout. Options are static:
kernel, stride, dilation, and top/bottom/left/right padding are known at build
time. `avg_pool2d` uses zero padding and divides by the full window size, so
padded elements are included in the average denominator.

## Operation Surface

- Elementwise arithmetic: `+`, `-`, `*`, `/`, unary `-`, `abs`, `minimum`,
  `maximum`, `clip`, `pow`, `square`, `reciprocal`, `relu`, `detach`.
- Comparisons and predicates: `greater`, `greater_equal`, `less`,
  `less_equal`, `equal`, `not_equal`, logical ops, `isnan`, `r#where`.
- Reductions: `sum`, `prod`, `mean`, extrema, arg extrema, `var`, `std`, `ptp`,
  `all`, `any`, and axis variants.
- Shape and indexing: `reshape`, `broadcast`, `squeeze`, `unsqueeze`, `slice`,
  `take`, `gather`, `take_along_axis`, `split`, `concat`, `stack`, `tile`,
  `repeat`, `pad`, `flip`, `roll`.
- Axis movement: `transpose`, `transpose_axes`, `permute`, `permute_dims`,
  `swapaxes`, `moveaxis`.
- Linalg, convolution, and pooling: `matmul`, `dot`, `vecdot`, `inner`,
  `outer`, `trace`, `diagonal`, `conv2d`, `conv2d_options`, `max_pool2d`,
  `max_pool2d_options`, `avg_pool2d`, `avg_pool2d_options`.
- Creation and math: `zeros_like`, `ones_like`, `full_like`, `arange`,
  `linspace`, `eye`, `identity`, `rand`, `uniform`, `randint`,
  `uniform_static`, `randint_static`, floating-point math functions.
