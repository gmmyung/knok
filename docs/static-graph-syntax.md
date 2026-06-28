# Static Graph Syntax

Graph bodies intentionally accept a small, static subset of Rust. Function-like
ops are parsed by the macro, so they do not need ordinary Rust functions in
scope.

Static creation ops are graph operations, not dynamic host array constructors.
Their shape and dtype are known at macro expansion time:

```rust
fn make_bias(x: Tensor2<f32, 2, 3>) -> Tensor2<f32, 2, 3> {
    full_like(x, 0.25)
}

fn make_positions() -> Tensor1<i32, 4> {
    arange::<Tensor1<i32, 4>>(0, 8, 2)
}

fn make_grid() -> Tensor1<f32, 5> {
    linspace::<Tensor1<f32, 5>>(0.0, 1.0)
}

fn make_identity() -> Tensor2<f32, 3, 3> {
    eye::<Tensor2<f32, 3, 3>>()
}
```

`zeros_like(x)` and `ones_like(x)` return a tensor with `x`'s static type.
`full_like(x, value)` requires `value` to be a rank-0 scalar with the same
element type as `x`. `arange::<Target>(stop)`,
`arange::<Target>(start, stop)`, and `arange::<Target>(start, stop, step)`
require numeric literal parameters and a rank-1 numeric `Target` whose length
matches the generated sequence. `linspace::<Target>(start, stop)` requires
numeric literal endpoints and a rank-1 numeric `Target`; integer targets are
accepted only when the endpoints divide evenly across the target length.
`eye::<Target>()` and `identity::<Target>()` require a rank-2 square `Target`
and support bool, integer, and floating-point element types.

Shape-changing ops are type-directed:

```rust
fn reshape_example(x: Tensor1<f32, 6>) -> Tensor2<f32, 2, 3> {
    reshape::<Tensor2<f32, 2, 3>>(x)
}

fn unsqueeze_example(x: Tensor2<f32, 2, 3>) -> Tensor4<f32, 1, 2, 1, 3> {
    unsqueeze::<Tensor4<f32, 1, 2, 1, 3>>(x)
}
```

Axis/index ops use const generics:

```rust
fn row_sums(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    sum::<1>(x)
}

fn middle_columns(x: Tensor2<f32, 2, 4>) -> Tensor2<f32, 2, 2> {
    slice::<Tensor2<f32, 2, 2>, 0, 1>(x)
}

fn last_column(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    take::<1, 2>(x)
}

fn move_batch_axis(x: Tensor3<f32, 2, 3, 4>) -> Tensor3<f32, 3, 4, 2> {
    moveaxis::<0, 2>(x)
}

fn static_pad(x: Tensor2<f32, 2, 2>) -> Tensor2<f32, 4, 5> {
    pad::<Tensor2<f32, 4, 5>, 1, 2>(x)
}

fn rows_by_index(x: Tensor2<f32, 2, 3>, indices: Tensor1<i64, 2>) -> Tensor2<f32, 2, 3> {
    gather::<Tensor2<f32, 2, 3>, 0>(x, indices)
}

fn columns_by_row(x: Tensor2<f32, 2, 3>, indices: Tensor2<i64, 2, 2>) -> Tensor2<f32, 2, 2> {
    take_along_axis::<1>(x, indices)
}
```

`slice::<Target, START...>(x)` keeps rank and uses the target shape as static
slice sizes. `take::<AXIS, INDEX>(x)` removes one axis and returns `Tensor0<_>`
when that is the remaining scalar. `gather::<Target, AXIS>(x, indices)` is
NumPy-style `take` with an `i32` or `i64` index tensor; the output shape must be
`input[..AXIS] + indices.shape + input[AXIS + 1..]`. `take_along_axis::<AXIS>(x,
indices)` uses an `i32` or `i64` index tensor with the same rank as `x`; all
dimensions except `AXIS` must match, and the result shape is `indices.shape`.
Negative runtime indices wrap from the end of the selected axis. Out-of-bounds
runtime indices fail the invocation. `concat::<AXIS>(a, b)` joins two tensors
along one existing axis. `stack::<AXIS>(a, b)` inserts a new axis of size 2.
`permute_dims::<AXES...>(x)`, `transpose::<AXES...>(x)`, `swapaxes::<A, B>(x)`,
and `moveaxis::<SRC, DST>(x)` use const axes and infer output shapes.
`split::<AXIS, SECTION...>(x)` returns one statically shaped tensor per section
and the section sizes must sum to the selected axis. `tile::<MULTIPLE...>(x)`,
`repeat::<AXIS, COUNT>(x)`, `pad::<Target, LOW...>(x)`, `flip::<AXES...>(x)`,
and `roll::<AXIS, SHIFT>(x)` are also static; `flip(x)` flips all axes.

Predicate tensors use `TensorN<bool, ...>` and lower to MLIR `i1`, not numeric
masks. Comparisons return bool tensors and can feed logical ops, bool
reductions, and selection:

```rust
fn select_positive(x: Tensor1<f32, 4>, fallback: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    r#where(greater(x, 0.0), x, fallback)
}

fn all_positive(x: Tensor1<f32, 4>) -> Tensor0<bool> {
    all(greater(x, 0.0))
}
```

The selection op is the graph op `where(condition, x, y)`, but Rust source must
spell it as the raw identifier `r#where(...)` because `where` is a keyword.
