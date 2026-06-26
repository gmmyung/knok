# knok

`knok` is an experimental Rust linalg graph frontend that compiles restricted
Rust function blocks into IREE VM bytecode at compile time.

## Current API

Graph definitions are regular Rust functions decorated with `#[knok::graph]`.
The function body is parsed as a restricted graph expression, compiled to MLIR
with `melior`, lowered to an IREE VMFB with `iree-compile`, and replaced by a
runtime wrapper.

The recommended hosted API is:

- use `#[knok::graph]` for Rust-authored static tensor graphs
- use `knok::mlir_model!` for checked local MLIR imports
- call `foo(...)` for simple one-off execution
- call `foo_run(&engine, ...)` or `model::invoke_run(&engine, ...)` when running
  inference repeatedly
- use `foo_artifact()` or `model::artifact()` only when integrating a custom
  runtime path

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
and VMFB program instead of rebuilding runtime state on every call:

```rust
use knok::{Backend, Engine, RuntimeConfig};

let engine = Engine::new(RuntimeConfig::auto())?;
let output = forward_run(&engine, x, y)?;

let cpu_engine = Engine::new(RuntimeConfig::backend(Backend::LlvmCpu))?;
```

Graphs can return multiple tensors by returning a Rust tuple:

```rust
#[knok::graph(backend = "llvm-cpu")]
fn add_sub(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> (Tensor1<f32, 4>, Tensor1<f32, 4>) {
    (x + y, x - y)
}

let (sum, diff) = add_sub_run(&engine, x, y)?;

#[knok::graph(backend = "llvm-cpu")]
fn combine(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    let (sum, diff) = add_sub(x, y);
    sum * diff
}
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
    inputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
    output: Tensor1<f32, 4>,
}

let output = imported_add4::invoke(x, y)?;
```

Typed MLIR imports also expose `invoke_run(&engine, ...)` for reusable
execution. Multi-output MLIR imports use `outputs: [...]` and return a tuple.
MLIR imports without a declared signature still expose `artifact()`; use
`Engine::invoke` with `RuntimeInput` values when you need raw runtime execution.

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

## Static graph syntax

Graph bodies intentionally accept a small, static subset of Rust. Function-like
ops are parsed by the macro, so they do not need ordinary Rust functions in
scope.

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
```

`slice::<Target, START...>(x)` keeps rank and uses the target shape as static
slice sizes. `take::<AXIS, INDEX>(x)` removes one axis, returning `Tensor1<_, 1>`
when that would otherwise be scalar. `concat::<AXIS>(a, b)` joins two tensors
along one existing axis. `stack::<AXIS>(a, b)` inserts a new axis of size 2.

Predicate tensors use `TensorN<bool, ...>` and lower to MLIR `i1`, not numeric
masks. Comparisons return bool tensors and can feed logical ops, bool
reductions, and selection:

```rust
fn select_positive(x: Tensor1<f32, 4>, fallback: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    r#where(greater(x, 0.0), x, fallback)
}

fn all_positive(x: Tensor1<f32, 4>) -> Tensor1<bool, 1> {
    all(greater(x, 0.0))
}
```

The selection op is the graph op `where(condition, x, y)`, but Rust source must
spell it as the raw identifier `r#where(...)` because `where` is a keyword.

## Examples

The examples are assertion-backed and can be run directly:

```sh
cargo run -p knok --example mlp
cargo run -p knok --example classifier
cargo run -p knok --example imported_mlir
cargo run -p knok --example multi_backend
cargo run -p knok --example dtypes
cargo test -p knok --features half half_graphs_run
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
- `features = ["half"]` enables `half::f16` and `half::bf16` tensor element
  types and re-exports them as `knok::half::{f16, bf16}`.
- Use `default-features = false, features = ["macros"]` when a no-std target
  still needs compile-time graph expansion.
- Proc macros, `melior`, and `iree-compile` still run on the compile host.
- In no-default-features mode, generated graph functions compile and expose
  `<name>_artifact()` with backend variant metadata. Runtime execution functions
  return `Error::HostedRuntimeDisabled`.

## MVP limits

- Graph tensor element types: `bool`, `f32`, `f64`, `i32`, and `i64`; gated
  `half::f16` and `half::bf16` support is available with `features = ["half"]`.
- Graph operations can mix bool predicates with numeric tensors where the op
  requires it, such as `r#where(condition, x, y)`. Numeric operands still must
  have matching element types; mixed dtype promotion is not implemented.
- Quantized integer types, complex numbers, and string/object-like values are
  not implemented yet.
- Static rank-1 through rank-4 shapes only.
- Explicit `backend = "llvm-cpu"` or `backend = "metal-spirv"`, or
  `backends = [backend("...", driver = "...")]`.
  Backend names and driver compatibility are validated at macro expansion time.
- Supported graph operations: `+`, `-`, `*`, `/`, unary `-`, trailing
  broadcasting, comparisons (`greater`, `greater_equal`, `less`, `less_equal`,
  `equal`, `not_equal`), `r#where`, `logical_and`, `logical_or`, `logical_not`,
  `logical_xor`, `all`, `any`, `isnan`, `abs`, `minimum`, `maximum`, `clip`,
  `pow`, `relu`, `matmul`, batched rank-3 `matmul`, NHWC/HWCF `conv2d`, rank-2
  `transpose`, reshape across ranks 1-4, `broadcast`, `squeeze`, `unsqueeze`,
  static `slice`, static `take`, binary `concat`, binary `stack`, full-tensor
  and axis-aware `sum`, full-tensor and axis-aware `mean`, full-tensor and
  axis-aware `softmax`, rank-1 `argmax`, `exp`, `log`, `sqrt`, `tanh`, and
  `sigmoid`.
- Axis-aware reductions use const generic syntax, for example `sum::<1>(x)`,
  `mean::<0>(x)`, and `softmax::<1>(logits)`.
- Floating-point classifier/math ops (`relu`, `mean`, `softmax`, `argmax`,
  `pow`, `exp`, `log`, `sqrt`, `tanh`, and `sigmoid`) require a floating-point
  element type. Backend support for `f16`/`bf16` math can vary. Integer tensors
  support arithmetic, `abs`, `minimum`, `maximum`, `clip`, reshape/broadcast,
  sum, matmul, and conv lowering where IREE accepts the resulting MLIR.
- `isfinite` and `isinf` are not exposed yet; the current lowering only adds
  `isnan`, which maps cleanly to `arith.cmpf uno`.
- The compiler, MLIR lowering, and hosted runtime wrappers support real bool
  tensors. Bool tensors lower to MLIR `i1` and use `eerie` bool buffer support
  at the runtime boundary.
- Function bodies may contain `let` bindings and one final expression. Arbitrary
  Rust control flow and function calls are rejected.
- Graph calls must refer to earlier `#[knok::graph]` functions in the same
  macro expansion process.
- `softmax(x)` normalizes over the whole tensor using max-subtracted
  exponentials; `softmax::<AXIS>(x)` normalizes over one axis using
  `linalg.softmax`.
  `argmax` currently returns the rank-1 index in the same floating-point element
  type as its input.

## Dtype support

Numeric literals are typed by their Rust suffix, defaulting to `f32` for float
literals and `i32` for integer literals. Bool literals are accepted where a
scalar bool predicate is valid.

| dtype | Tensor container | Graph parse/typecheck | Hosted runtime | Notes |
| --- | --- | --- | --- | --- |
| `f32` | yes | yes | yes | Primary path; all floating-point graph ops target this first. |
| `f64` | yes | yes | yes | Uses `--iree-input-demote-f64-to-f32=false`; backend support may vary. |
| `f16` | `half` feature | `half` feature | `half` feature | Uses `half::f16`; backend math support may vary. |
| `bf16` | `half` feature | `half` feature | `half` feature | Uses `half::bf16`; currently best treated as typed storage/roundtrip unless the selected backend accepts the op. |
| `i32` | yes | yes | yes | Arithmetic, reductions, shape/index ops, matmul/conv where IREE accepts the MLIR. |
| `i64` | yes | yes | yes | Same policy as `i32`. |
| `bool` | yes | yes | yes | Lowers to MLIR `i1`; used for comparisons, logical ops, `r#where`, `all`, `any`, and `isnan`. |
| quantized ints | no | no | no | Deferred; requires explicit scale/zero-point semantics, not just smaller integer storage. |

There is no implicit promotion, mixed numeric dtype execution, complex dtype,
string/object dtype, or NumPy object-array equivalent.

## Development

Use the Nix shell so `melior` can find LLVM/MLIR 22 and `knok` can find the IREE
compiler:

```sh
nix develop
cargo fmt --all -- --check
cargo test --workspace
cargo doc -p knok --no-default-features --features std --no-deps
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

Current local benchmark snapshots are recorded in `BENCHMARKS.md`.

The runtime benchmark includes both reusable `Engine` calls and the convenience
wrapper path that constructs runtime state per invocation. Larger benchmark
shapes currently include:

- `matmul_128x128`: `Tensor2<f32, 128, 128> @ Tensor2<f32, 128, 128>`
- `batched_matmul_16x128x128`: `Tensor3<f32, 16, 128, 128>` where each batch
  computes `128x128 @ 128x128`
- `conv2d_nhwc_16x32x32x3_hwcf_3x3x3x16`
- `mlp_128x128x64`: `128x128 @ 128x64 + 128x64`, then `relu`
- `softmax_64x1000`: `Tensor2<f32, 64, 1000>`

Comparison groups include `ndarray` equivalents for matmul, batched matmul,
MLP, softmax, and a direct NHWC/HWCF convolution loop over `ndarray::Array4`;
`nalgebra` is included for dense `128x128` matrix multiplication.
