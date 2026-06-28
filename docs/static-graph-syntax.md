# Build-Time Tracing Syntax

Graph functions are normal Rust functions compiled into the build script. They
operate on traced tensor aliases from `knok_build::prelude`; both short
`T2<...>` aliases and descriptive `Tensor2<...>` aliases are available:

```rust
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
    relu(matmul(x.clone(), x) + 1.0)
}
```

`compile_graphs!` invokes the function with traced inputs during `cargo build`;
tensor operations append nodes to a graph IR, which is then type-checked and
compiled to IREE VMFB. The attribute macro records the signature and backend
metadata; the function body remains ordinary Rust.

## Supported Patterns

- Host-side helper functions.
- Host-side loops with statically known iteration counts.
- Numeric scalar literals on the right-hand side of tensor arithmetic and graph
  op calls.
- Tuple outputs for multi-output generated wrappers.

```rust
fn block(x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
    relu(matmul(x.clone(), x) + 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn stacked(mut x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
    for _ in 0..2 {
        x = block(x);
    }
    x
}
```

Tensor-value-dependent Rust control flow is not supported. Use graph operations
for data-dependent behavior.

## Operation Surface

The tracing surface is a Rust-native graph API:

- Elementwise arithmetic: `+`, `-`, `*`, `/`, unary `-`, `abs`, `minimum`,
  `maximum`, `clip`, `pow`, `square`, `reciprocal`, and `relu`.
- Comparisons and predicates: `greater`, `greater_equal`, `less`,
  `less_equal`, `equal`, `not_equal`, `logical_and`, `logical_or`,
  `logical_not`, `logical_xor`, `all`, `any`, `all_axis`, `any_axis`, `isnan`,
  and `r#where`.
- Shape and indexing: `reshape`, `broadcast`, `squeeze`, `unsqueeze`, `slice`,
  `take`, `gather`, `take_along_axis`, `split`, `concat`, `stack`, `tile`,
  `repeat`, `pad`, `flip`, and `roll`.
- Axis movement: `transpose`, `permute`, `permute_dims`, `swapaxes`, and
  `moveaxis`.
- Creation: `zeros_like`, `ones_like`, `full_like`, `arange`, `linspace`,
  `eye`, and `identity`.
- Reductions and statistics: `sum`, `sum_axis`, `prod`, `prod_axis`, `mean`,
  `mean_axis`, `max` / `amax`, `max_axis` / `amax_axis`, `min` / `amin`,
  `min_axis` / `amin_axis`, `argmax`, `argmax_axis`, `argmin`, `argmin_axis`,
  `var`, `var_axis`, `std`, `std_axis`, `ptp`, `ptp_axis`, `all`, `all_axis`,
  `any`, `any_axis`, `softmax`, and `softmax_axis`.
- Linalg and convolution: `matmul`, `dot`, `vecdot`, `inner`, `outer`,
  `trace`, `trace_axes`, `diagonal`, `diagonal_axes`, `conv2d`, and
  `conv2d_options`.
- Floating-point math: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`,
  `log1p`, `sqrt`, `floor`, `ceil`, `round`, `rint`, `sin`, `cos`, `tan`,
  `tanh`, and `sigmoid`.

Axis and shape metadata are ordinary value arguments:

```rust
#[knok_build::graph(backend = Backend::LlvmCpu)]
fn ops(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    sum_axis(softmax_axis(x, 1), 1)
}
```

Shape-changing operations follow the same rule:

```rust
let parts: (Tensor2<f32, 2, 2>, Tensor2<f32, 2, 3>) = split(x, 1, [2, 3]);
let y: Tensor3<f32, 3, 4, 2> = transpose_axes(z, [1, 2, 0]);
```

Creation and convolution helpers use explicit function names instead of
overloaded graph-only syntax:

```rust
let r: Tensor1<i32, 4> = arange_step(0, 8, 2);
let y: Tensor4<f32, 1, 2, 2, 1> =
    conv2d_options(x, k, Conv2dOptions::new().padding(1, 1, 1, 1).stride(2, 2));
```

## Target Import

Target crates import generated wrappers with:

```rust
knok::generated_graphs!(pub mod graphs);
```

When using `BuildOptions::output_file(...)`, pass the same filename as the
second macro argument:

```rust
knok::generated_graphs!(pub mod graphs, "custom_knok_graphs.rs");
```

Generated modules expose `artifact()`, `run(&Engine, ...)`, and `call(...)`.
