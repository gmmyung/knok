# knok

`knok` is an experimental Rust linalg graph frontend that compiles restricted
Rust function blocks into IREE VM bytecode at compile time.

## Current API

Graph definitions are regular Rust functions decorated with `#[knok::graph]`.
The function body is parsed as a restricted graph expression, compiled to MLIR
with `melior`, lowered to an IREE VMFB with `iree-compile`, and replaced by a
runtime wrapper.

```rust
use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn forward(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x + y)
}

fn main() -> knok::Result<()> {
    let x = Tensor1::from_array([-1.0, 2.0, -3.0, 4.0]);
    let y = Tensor1::from_array([0.5, 1.0, 10.0, -10.0]);
    let output = forward(x, y)?;
    println!("{:?}", output.into_vec());
    Ok(())
}
```

Local MLIR files can also be compiled into embedded artifacts:

```rust
knok::mlir_model! {
    name: imported_add4,
    path: "tests/fixtures/add4.mlir",
    backend: "llvm-cpu",
    function: "imported.add4",
}

let output = imported_add4::invoke_f32(&[(&[4], &x), (&[4], &y)])?;
```

Graphs can call earlier graph functions. Calls are inlined into the caller at
macro expansion time, so the outer graph still compiles to one VMFB:

```rust
#[knok::graph(backend = "llvm-cpu")]
fn layer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn model(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    layer(x + y)
}
```

Recursive graph calls are rejected.

`Tensor1<T, D0>` and `Tensor2<T, D0, D1>` are used because stable Rust does not
support `Tensor<T, [D0, D1]>` as a const-generic type parameter.

## Feature Modes

- Default features enable the hosted runtime convenience path.
- `default-features = false` builds `knok` as `no_std + alloc`.
- Proc macros, `melior`, and `iree-compile` still run on the compile host.
- In no-default-features mode, generated graph functions compile and expose
  `<name>_artifact()`, but the convenience invocation function returns
  `Error::HostedRuntimeDisabled`.

## MVP limits

- `f32` tensors only.
- Static rank-1 and rank-2 shapes only.
- Explicit `backend = "llvm-cpu"` or `backend = "metal-spirv"`.
- Supported graph operations: `+`, `-`, `*`, `/`, unary `-`, `relu`, `matmul`,
  and rank-2 `transpose`.
- Function bodies may contain `let` bindings and one final expression. Arbitrary
  Rust control flow and function calls are rejected.
- Graph calls must refer to earlier `#[knok::graph]` functions in the same
  macro expansion process.

## Development

Use the Nix shell so `melior` can find LLVM/MLIR 22 and `knok` can find the IREE
compiler:

```sh
nix develop
cargo test
```
