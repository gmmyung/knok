# knok

[![CI](https://github.com/gmmyung/knok/actions/workflows/rust.yml/badge.svg)](https://github.com/gmmyung/knok/actions/workflows/rust.yml)
[![docs.rs](https://docs.rs/knok/badge.svg)](https://docs.rs/knok)
[![crates.io](https://img.shields.io/crates/v/knok.svg)](https://crates.io/crates/knok)

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

#[knok::graph(backend = Backend::LlvmCpu)]
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
#[knok::graph(backend = Backend::LlvmCpu)]
fn add_sub(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> (Tensor1<f32, 4>, Tensor1<f32, 4>) {
    (x + y, x - y)
}

let (sum, diff) = add_sub_run(&engine, x, y)?;

#[knok::graph(backend = Backend::LlvmCpu)]
fn combine(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    let (sum, diff) = add_sub(x, y);
    sum * diff
}
```

Graphs can also embed multiple IREE backend variants. The runtime selects the
variant whose driver matches the engine:

```rust
#[knok::graph(backends = [
    backend(Backend::LlvmCpu, driver = Driver::LocalTask),
    backend(Backend::MetalSpirv, driver = Driver::Metal),
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
    backend: Backend::LlvmCpu,
    function: "imported.add4",
    inputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
    output: Tensor1<f32, 4>,
}

let output = imported_add4::invoke(x, y)?;
```

Typed MLIR imports also expose `invoke_run(&engine, ...)` for reusable
execution. Multi-output MLIR imports use `outputs: [...]` and return a tuple.
MLIR imports without a declared signature still expose `artifact()`; use
`Engine::invoke` with `knok::runtime::raw::Input` values when you need raw
runtime execution. Artifacts generated from typed graphs or typed MLIR imports
record input and output `TensorDesc` metadata, so raw `Engine::invoke` rejects
input count, dtype, and shape mismatches before entering IREE.

Graphs can call earlier graph functions. Calls are inlined into the caller at
macro expansion time, so the outer graph still compiles to one VMFB:

```rust
#[knok::graph(backend = Backend::LlvmCpu)]
fn layer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn model(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    layer(x + y)
}
```

Recursive graph calls are rejected.

`Tensor0<T>` through `Tensor6<T, D0, ...>` are used because stable Rust does
not support `Tensor<T, [D0, D1]>` as a const-generic type parameter.
They expose `from_array`, `from_vec`, `TryFrom<Vec<_>>`, `zeros`, `ones`,
`filled`, `as_slice`, `as_mut_slice`, `into_vec`, and checked indexing helpers.
`Tensor0<T>` represents a rank-0 scalar tensor and stores exactly one element.
Those associated functions are host-side tensor constructors for preparing
inputs and checking outputs; they are separate from graph ops parsed inside
`#[knok::graph]` bodies.

## Static graph syntax

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
- Static rank-0 through rank-6 shapes only.
- Host tensors are contiguous row-major value containers. Graph operations are
  value operations, not NumPy-style host views.
- Explicit `backend = Backend::LlvmCpu` or `backend = Backend::MetalSpirv`, or
  `backends = [backend(Backend::..., driver = Driver::...)]`.
  Backend names and driver compatibility are validated at macro expansion time.
- Supported graph operations: `+`, `-`, `*`, `/`, unary `-`, trailing
  broadcasting, comparisons (`greater`, `greater_equal`, `less`, `less_equal`,
  `equal`, `not_equal`), `r#where`, `logical_and`, `logical_or`, `logical_not`,
  `logical_xor`, `all`, `any`, `isnan`, `abs`, `minimum`, `maximum`, `clip`,
  `pow`, `square`, `reciprocal`, `relu`, NumPy-style `matmul` for ranks 1-6,
  `dot`, `vecdot`, `inner`, `outer`, `trace`, `diagonal`, NHWC/HWCF `conv2d`
  with static `Pad<TOP, BOTTOM, LEFT, RIGHT>`, `Stride<H, W>`, and
  `Dilation<H, W>`, and `Groups<N>` options, NumPy-style rank-N `transpose`
  with optional const axes, explicit `permute::<Target, AXES...>`,
  shape-inferred `permute_dims::<AXES...>`, `swapaxes`, `moveaxis`, reshape
  across ranks 0-6, `broadcast`, `squeeze`, `unsqueeze`, static `slice`,
  static `take`, tensor-index `gather`, `take_along_axis`, static `split`,
  binary `concat`, binary `stack`, `tile`, `repeat`, `pad`, `flip`, `roll`,
  `zeros_like`, `ones_like`, `full_like`, static literal `arange`, static
  literal `linspace`, static square `eye`/`identity`, full-tensor and
  axis-aware `sum`, `prod`, `mean`, `max` / `amax`, `min` / `amin`, `argmax`,
  `argmin`, `var`, `std`, and `ptp`, full-tensor and axis-aware `softmax`,
  `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `floor`,
  `ceil`, `round`, `rint`, `sin`, `cos`, `tan`, `tanh`, and `sigmoid`.
- Axis-aware reductions use const generic syntax, for example `sum::<1>(x)`,
  `prod::<1>(x)`, `mean::<0>(x)`, `amax::<1>(x)`, `var::<1>(x)`,
  `softmax::<1>(logits)`, and `argmax::<1>(logits)`.
- `dot(x, y)` accepts rank-1 numeric vectors. `vecdot(x, y)` contracts the last
  axis of same-shaped numeric tensors, and `vecdot::<AXIS>(x, y)` contracts a
  specific axis. `inner(x, y)` contracts the last axis of both numeric operands
  and treats scalar operands as elementwise multiplication. `outer(x, y)`
  flattens both operands and returns a rank-2 outer product.
- `trace(x)` and `diagonal(x)` use the last two axes by default; pass
  `trace::<AXIS0, AXIS1>(x)` or `diagonal::<AXIS0, AXIS1>(x)` to select explicit
  axes. `trace` requires equal diagonal dimensions and numeric tensors.
  `diagonal` supports all tensor element types and also requires equal diagonal
  dimensions.
- `conv2d(x, k)` defaults to valid convolution. Options use type-style generic
  markers, for example `conv2d::<Pad<1, 1, 1, 1>, Stride<2, 2>>(x, k)`.
  `Groups<N>` follows PyTorch-style grouped convolution shape rules: input
  channels and output channels must both be divisible by `N`, and kernel input
  channels must equal `input_channels / N`.
- Floating-point classifier/math/statistics ops (`relu`, `mean`, `var`, `std`,
  `softmax`, `pow`, `exp`,
  `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `floor`, `ceil`,
  `round`, `rint`, `sin`, `cos`, `tan`, `tanh`, and `sigmoid`) require a
  floating-point element type. Backend support for `f16`/`bf16` math can vary.
  Integer tensors support arithmetic, `abs`, `square`, `reciprocal`,
  `minimum`, `maximum`, `clip`, reshape/broadcast, static creators, `sum`,
  `prod`, `max` / `amax`, `min` / `amin`, `ptp`, `argmax`, `argmin`, matmul,
  linalg contractions, and conv lowering where IREE accepts the resulting MLIR.
  Bool tensors support `zeros_like`, `ones_like`, `full_like`, and
  `eye`/`identity`; `arange` and `linspace` require numeric targets.
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
- Full-tensor reductions and rank-1 axis reductions return `Tensor0<_>`.
  Full-tensor reductions over `Tensor0<_>` return `Tensor0<_>` where the
  operation is defined.
- `argmax(x)` and `argmin(x)` accept ordered tensor elements and return the
  row-major flattened index as `Tensor0<i64>`. Axis-aware forms return
  per-slice indices along the reduced axis. Ties select the lowest index.
- `var` and `std` compute population statistics with the same output dtype as
  the input. Integer and bool statistics are not implicitly promoted.
- Empty `sum`, `prod`, `all`, and `any` reductions use their identity values.
  Empty `mean`, `max`, `min`, `softmax`, `argmax`, `argmin`, `var`, `std`, and
  `ptp` reductions are rejected because there is no well-defined selected value
  or denominator.

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
| `bool` | yes | yes | yes | Lowers to MLIR `i1`; used for comparisons, logical ops, `r#where`, `all`, `any`, `max`, `min`, `argmax`, `argmin`, `isnan`, and value tensors in indexing ops. |
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

Additional project documents:

| Document | Purpose |
| --- | --- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, PR expectations, validation commands, branch naming, and versioning policy. |
| [DEVELOPERS.md](DEVELOPERS.md) | Crate layout, compile/runtime flow, release flow, and platform notes. |
| [AGENTS.md](AGENTS.md) | Rules for Codex and other automated agents working in this repository. |
| [CHANGELOG.md](CHANGELOG.md) | User-facing changes and release notes. |

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
- `vecdot_128x1024_axis1`: row-wise vector dot over `Tensor2<f32, 128, 1024>`
- `conv2d_nhwc_16x32x32x3_hwcf_3x3x3x16`
- `mlp_128x128x64`: `128x128 @ 128x64 + 128x64`, then `relu`
- `softmax_64x1000`: `Tensor2<f32, 64, 1000>`

Comparison groups include `ndarray` equivalents for matmul, batched matmul,
vecdot, MLP, softmax, and a direct NHWC/HWCF convolution loop over
`ndarray::Array4`; `nalgebra` is included for dense `128x128` matrix
multiplication.
