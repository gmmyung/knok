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
use knok::{Backend, Engine, RuntimeConfig};

let engine = Engine::new(RuntimeConfig::auto())?;
let output = forward_run(&engine, x, y)?;

let cpu_engine = Engine::new(RuntimeConfig::backend(Backend::LlvmCpu))?;
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

`Tensor1<T, D0>` through `Tensor4<T, D0, ...>` are used because stable Rust does
not support `Tensor<T, [D0, D1]>` as a const-generic type parameter.
They expose `from_array`, `from_vec`, `TryFrom<Vec<_>>`, `zeros`, `ones`,
`filled`, `as_slice`, `as_mut_slice`, `into_vec`, and checked indexing helpers.

## Examples

The examples are assertion-backed and can be run directly:

```sh
cargo run -p knok --example mlp
cargo run -p knok --example classifier
cargo run -p knok --example imported_mlir
cargo run -p knok --example multi_backend
```

They cover the recommended hosted workflow:

- define or import a static graph at compile time
- construct one reusable `Engine`
- call generated `*_run` or `invoke_run` wrappers for repeated inference
- select CPU or Metal by choosing an engine driver when the artifact contains
  matching backend variants

## Feature Modes

- Default features enable the proc macros and hosted runtime convenience path.
- `default-features = false` builds `knok` as `no_std + alloc`.
- Use `default-features = false, features = ["macros"]` when a no-std target
  still needs compile-time graph expansion.
- Proc macros, `melior`, and `iree-compile` still run on the compile host.
- In no-default-features mode, generated graph functions compile and expose
  `<name>_artifact()` with backend variant metadata. Runtime execution functions
  return `Error::HostedRuntimeDisabled`.

## MVP limits

- `f32` tensors only.
- Static rank-1 through rank-4 shapes only.
- Explicit `backend = "llvm-cpu"` or `backend = "metal-spirv"`, or
  `backends = [backend("...", driver = "...")]`.
  Backend names and driver compatibility are validated at macro expansion time.
- Supported graph operations: `+`, `-`, `*`, `/`, unary `-`, `relu`, `matmul`,
  batched rank-3 `matmul`, NHWC/HWCF `conv2d`, rank-2 `transpose`, reshape
  across ranks 1-4, scalar-like `broadcast`, full-tensor `sum`, full-tensor
  `mean`, full-tensor `softmax`, rank-1 `argmax`, `exp`, `log`, `sqrt`, `tanh`,
  and `sigmoid`.
- Function bodies may contain `let` bindings and one final expression. Arbitrary
  Rust control flow and function calls are rejected.
- Graph calls must refer to earlier `#[knok::graph]` functions in the same
  macro expansion process.
- `softmax` normalizes over the whole tensor using max-subtracted exponentials.
  `argmax` currently returns the rank-1 index as `Tensor1<f32, 1>` because
  public tensors are still `f32` only.

## Development

Use the Nix shell so `melior` can find LLVM/MLIR 22 and `knok` can find the IREE
compiler:

```sh
nix develop
cargo fmt --all -- --check
cargo test --workspace
cargo doc -p knok --no-default-features --features std --no-deps
```

CI intentionally tests against the published `eerie` crate. For local
co-development with an adjacent checkout, add a temporary local Cargo patch
outside committed files:

```toml
[patch.crates-io]
eerie = { path = "../eerie" }
```

Release checks and publishing order are scripted:

```sh
scripts/release-check.sh
scripts/publish.sh --dry-run
scripts/publish.sh --execute
```

## Benchmarks

Runtime and tensor benchmarks use Criterion:

```sh
cargo bench -p knok --bench runtime
```

For a quick smoke run while developing:

```sh
cargo bench -p knok --bench runtime -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.2
```

The runtime benchmark includes both reusable `Engine` calls and the convenience
wrapper path that constructs runtime state per invocation. Larger benchmark
shapes currently include:

- `matmul_64x64`: `Tensor2<f32, 64, 64> @ Tensor2<f32, 64, 64>`
- `batched_matmul_32x32x32`: `Tensor3<f32, 32, 32, 32>`
- `conv2d_nhwc_16x32x32x3_hwcf_3x3x3x16`
- `mlp_128x128x64`: `128x128 @ 128x64 + 128x64`, then `relu`
- `softmax_64x1000`: `Tensor2<f32, 64, 1000>`

Comparison groups include `ndarray` equivalents for matmul, batched matmul,
MLP, softmax, and a direct NHWC/HWCF convolution loop over `ndarray::Array4`;
`nalgebra` is included for dense `64x64` matrix multiplication.
