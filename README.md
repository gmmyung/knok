# knok

[![CI](https://github.com/gmmyung/knok/actions/workflows/rust.yml/badge.svg)](https://github.com/gmmyung/knok/actions/workflows/rust.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgmmyung%2Fknok%2Fbadges%2Fcoverage.json)](https://github.com/gmmyung/knok/actions/workflows/coverage.yml)
[![docs.rs](https://docs.rs/knok/badge.svg)](https://docs.rs/knok)
[![crates.io](https://img.shields.io/crates/v/knok.svg)](https://crates.io/crates/knok)

`knok` lets you write static-shape tensor graphs in Rust, compile them during
`cargo build`, and call the compiled graph as a typed Rust function.

Graphs are authored in `build.rs`, lowered through MLIR, compiled with IREE,
and embedded into your crate as generated wrappers. Runtime code imports those
wrappers with `knok::generated_graphs!` and calls typed Rust functions.

## Install

Add `knok` to your target crate and `knok-build` to build dependencies:

```toml
[dependencies]
knok = "0.3"

[build-dependencies]
knok-build = "0.3"
```

Build-time compilation uses the IREE compiler from your Python environment:

```sh
python -m pip install 'iree-base-compiler==3.13.0'
```

The Nix development shell provides the compiler, LLVM/MLIR, and macOS build-host
configuration:

```sh
nix develop
```

See [docs/compiler.md](docs/compiler.md) for compiler setup, cache knobs, and
troubleshooting.

## Quickstart

Define and compile a graph in `build.rs`:

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

Import the generated module and call it from normal Rust code. `call` is the
one-shot convenience path:

```rust
use knok::prelude::*;

knok::generated_graphs!(pub mod graphs);

fn run_once(x: Tensor2<f32, 2, 2>) -> knok::Result<Tensor2<f32, 2, 2>> {
    graphs::forward::call(x)
}
```

Each generated module also exposes a typed `GRAPH` handle for code that wants
the lower-level `knok::Graph<I, O>` value.

For repeated inference, create an engine from the generated artifact once and
use `run`:

```rust
use knok::Engine;
use knok::prelude::*;

knok::generated_graphs!(pub mod graphs);

fn run_many(xs: impl IntoIterator<Item = Tensor2<f32, 2, 2>>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::forward::artifact())?;

    for x in xs {
        let y = graphs::forward::run(&engine, x)?;
        // use y
    }

    Ok(())
}
```

`Engine::for_artifact` uses the runtime driver recorded by `knok-build`, so the
engine matches the backend used to compile the graph.

Use `graphs::forward::run_into(&engine, x, &mut output)` to reuse an existing
output tensor's host storage. Multiple outputs use a mutable tuple of tensors.
Device buffers and invocation metadata may still allocate.

Existing `.mlir` files can also be compiled from `build.rs` with
`knok_build::compile_mlir_models!` and imported through the same
`knok::generated_graphs!` flow.

## What It Supports

- Static-shape tensors with ranks 0 through 6.
- Explicit dtype handling. `f32`, integer, and bool tensors are available by
  default; `f16` and `bf16` are behind the `half` feature.
- Single-output and multi-output graphs.
- Elementwise arithmetic, comparisons, logical predicates, reductions, shape
  transforms, slicing/padding, matmul, convolution, pooling, and related tensor
  operations.
- Deterministic random tensor helpers, including compile-time constants and
  runtime key-threaded random streams.
- Build-time graph construction with `knok-build`.
- Experimental build-time value-and-gradient graph generation for scalar
  floating-point losses, including explicit `detach` stop-gradient and no-grad
  index/predicate values for conditions and indexing.
- Hosted execution through IREE with the `knok` runtime API.
- External MLIR model import for users who already have MLIR.

The operation semantics are documented in [docs/semantics.md](docs/semantics.md)
and dtype support is tracked in [docs/dtypes.md](docs/dtypes.md).

Random helpers are deterministic. Static helpers embed build-time constants,
while runtime helpers take and return an explicit `T0<i64>` key:

```rust
#[knok_build::graph(backend = Backend::LlvmCpu)]
fn sample(key: T0<i64>) -> (T2<f32, 4, 4>, T0<i64>) {
    uniform(key, -1.0, 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn weights() -> T2<f32, 4, 4> {
    uniform_static(42, -0.05, 0.05)
}
```

## Backends

`knok` separates the compile-time IREE target backend from the runtime driver.
Most users should start with `Backend::LlvmCpu`.

| Backend | Runtime driver | Availability |
| --- | --- | --- |
| `Backend::LlvmCpu` | `local-task` | Always |
| `Backend::MetalSpirv` | `metal` | macOS |
| `Backend::VulkanSpirv` | `vulkan` | `vulkan` feature |
| `Backend::Cuda` | `cuda` | `cuda` feature |

Enable feature-gated backends on both sides when you compile graphs and run them:

```toml
[dependencies]
knok = { version = "0.3", features = ["vulkan"] }

[build-dependencies]
knok-build = { version = "0.3", features = ["vulkan"] }
```

Use `cuda` instead of `vulkan` for CUDA. Metal is exposed only on macOS targets.
See [docs/backends.md](docs/backends.md) for platform notes.

## Feature Flags

| Feature | Effect |
| --- | --- |
| `runtime` | Enables hosted execution through IREE. Included by default. |
| `std` | Enables standard-library support. Included by default. |
| `half` | Enables `half::f16` and `half::bf16` tensor element types. |
| `serde` | Implements Serde `Serialize` and `Deserialize` for fixed-rank tensors. |
| `vulkan` | Exposes the Vulkan SPIR-V backend and runtime driver. |
| `cuda` | Exposes the CUDA backend and runtime driver. |
| `embedded-runtime` | Enables the IREE runtime dependency without enabling `std`. |
| `cross-lang-lto` | Enables native runtime ThinLTO; requires the [matching toolchain](docs/compiler.md#cross-language-lto). |

With `default-features = false`, generated wrapper types can still typecheck in
`no_std + alloc` targets, but hosted runtime execution is disabled.

Enable `serde` when tensors need to cross a serialization boundary:

```toml
[dependencies]
knok = { version = "0.3", features = ["serde"] }
```

Tensor serialization uses a format-neutral `{ shape, data }` representation.
The `data` field stores flat row-major elements, and deserialization validates
the recorded shape against the tensor's static shape.

## Learn More

- [Static graph syntax](docs/static-graph-syntax.md): author graphs in
  `build.rs`, including gradient wrappers and external MLIR imports.
- [Tensor semantics](docs/semantics.md): rank, broadcasting, axis handling, and
  operation behavior.
- [Dtypes](docs/dtypes.md): supported element types and explicit casting policy.
- [Backends](docs/backends.md): compiler backends, runtime drivers, and platform
  notes.
- [Runtime workflow](docs/runtime.md): generated wrappers, `Graph`, and
  reusable engines.
- [Compiler setup](docs/compiler.md): compiler installation and cache
  behavior.
- [Benchmarks](docs/benchmarks.md): local runtime benchmark harness and shapes.

## AI-Assisted Development

Parts of this project have been developed with assistance from AI coding tools.
AI-generated suggestions are reviewed, edited, and tested before they are
accepted. The maintainer remains responsible for the project's design,
implementation, security, correctness, and licensing.

## Development

Run the full local validation suite:

```sh
scripts/release-check.sh
```

Generate coverage artifacts:

```sh
scripts/coverage.sh
```

Run local Criterion runtime benchmarks:

```sh
scripts/benchmark.sh
```

Contributor-oriented notes live in [CONTRIBUTING.md](CONTRIBUTING.md) and
[DEVELOPERS.md](DEVELOPERS.md). User-facing changes are tracked in
[CHANGELOG.md](CHANGELOG.md).
