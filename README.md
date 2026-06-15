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

For repeated inference, create a reusable runtime engine and call the generated
`*_run` wrapper. This reuses the IREE instance, driver, device, loaded module,
and resolved function instead of rebuilding runtime state on every call:

```rust
use knok::{Engine, RuntimeConfig};

let engine = Engine::new(RuntimeConfig::auto())?;
let output = forward_run(&engine, x, y)?;
```

Graphs can also embed multiple IREE backend variants. The runtime selects the
variant whose driver matches the engine:

```rust
#[knok::graph(backends = [
    backend("llvm-cpu", driver = "local-task"),
    backend("metal-spirv", driver = "metal"),
])]
fn forward(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x + y)
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

Typed MLIR imports also expose `invoke_run(&engine, ...)` and
`invoke_f32_run(&engine, ...)` for reusable execution.

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
  `<name>_artifact()` with backend variant metadata. Runtime execution functions
  return `Error::HostedRuntimeDisabled`.

## MVP limits

- `f32` tensors only.
- Static rank-1 and rank-2 shapes only.
- Explicit `backend = "llvm-cpu"` or `backend = "metal-spirv"`, or
  `backends = [backend("...", driver = "...")]`.
- Supported graph operations: `+`, `-`, `*`, `/`, unary `-`, `relu`, `matmul`,
  rank-2 `transpose`, rank-1/rank-2 `reshape`, scalar-like `broadcast`, and
  full-tensor `sum`.
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
