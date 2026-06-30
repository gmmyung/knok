# Build-Time Graph Syntax

Graph functions are normal Rust functions compiled into the build script. They
operate on traced tensor aliases from `knok_build::prelude`; both short
`T2<...>` aliases and descriptive `Tensor2<...>` aliases are available.

```rust
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
    relu(matmul(x.clone(), x) + 1.0)
}

fn main() {
    knok_build::compile_graphs!(forward);
}
```

`compile_graphs!` invokes each graph function with traced inputs during
`cargo build`. Tensor operations append nodes to graph IR, the IR is
type-checked, lowered to MLIR, compiled to IREE VMFB, and emitted as generated
Rust wrappers in `OUT_DIR`.

The `#[knok_build::graph]` attribute records the Rust signature and backend
metadata. The function body remains ordinary Rust executed on traced tensor
values.

## Supported Rust Patterns

- Host-side helper functions.
- Host-side loops with statically known iteration counts.
- Numeric scalar literals as tensor operands.
- Tuple outputs for multi-output generated wrappers.
- Local `let` bindings and cloned traced tensors.

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

Tensor-value-dependent Rust control flow is not supported because tensor values
are unknown on the build host. Use graph operations such as `r#where` for
data-dependent behavior.

## Shape Metadata

Axis, shape, and convolution options are normal Rust value arguments:

```rust
#[knok_build::graph(backend = Backend::LlvmCpu)]
fn ops(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    sum_axis(softmax_axis(x, 1), 1)
}

let parts: (Tensor2<f32, 2, 2>, Tensor2<f32, 2, 3>) = split(x, 1, [2, 3]);
let y: Tensor3<f32, 3, 4, 2> = transpose_axes(z, [1, 2, 0]);
```

Creation and convolution helpers use explicit function names:

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

Generated modules expose:

- `GRAPH`, a typed `knok::Graph<I, O>` handle.
- `artifact()` for static metadata and embedded VMFB bytes.
- `run(&Engine, ...)` for repeated hosted inference.
- `call(...)` for one-off hosted inference with a convenience engine.

## External MLIR Models

Custom `.mlir` files can be compiled in `build.rs` and exposed with the same
generated wrapper shape. This path is separate from traced graph IR: the MLIR is
compiled as an external artifact, while Rust types provide the wrapper
signature.

```rust
use knok_build::prelude::*;

fn main() {
    knok_build::compile_mlir_models!(
        imported_add {
            path: "models/add.mlir",
            function: "imported.add",
            backend: Backend::LlvmCpu,
            inputs: [x: T1<f32, 4>, y: T1<f32, 4>],
            outputs: [T1<f32, 4>],
        },
    );
}
```

Then import and call it from target code:

```rust
knok::generated_graphs!(pub mod graphs);

let z = graphs::imported_add::call(x, y)?;
```

When traced graphs and external MLIR models are generated from the same
`build.rs`, write one set to a custom output file:

```rust
knok_build::compile_mlir_models_with_options!(
    BuildOptions::default().output_file("knok_mlir_models.rs");
    imported_add {
        path: "models/add.mlir",
        function: "imported.add",
        backend: Backend::LlvmCpu,
        inputs: [x: T1<f32, 4>, y: T1<f32, 4>],
        outputs: [T1<f32, 4>],
    },
);
```

```rust
knok::generated_graphs!(pub mod graphs);
knok::generated_graphs!(pub mod mlir_models, "knok_mlir_models.rs");
```
