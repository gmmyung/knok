# knok

[![CI](https://github.com/gmmyung/knok/actions/workflows/rust.yml/badge.svg)](https://github.com/gmmyung/knok/actions/workflows/rust.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgmmyung%2Fknok%2Fbadges%2Fcoverage.json)](https://github.com/gmmyung/knok/actions/workflows/coverage.yml)
[![docs.rs](https://docs.rs/knok/badge.svg)](https://docs.rs/knok)
[![crates.io](https://img.shields.io/crates/v/knok.svg)](https://crates.io/crates/knok)

`knok` lets you write static-shape tensor graphs in Rust, compile them during
`cargo build`, and call the compiled graph as a typed Rust function.

Graphs are authored in `build.rs`, lowered through MLIR, compiled with IREE, and
embedded into your crate as generated wrappers. The target code only sees typed
tensor inputs and outputs.

## Install

Add `knok` to your target crate and `knok-build` to build dependencies:

```toml
[dependencies]
knok = "0.2"

[build-dependencies]
knok-build = "0.2"
```

Build-time compilation needs `iree-compile` on `PATH`. You can also point to it
explicitly:

```sh
export KNOK_IREE_COMPILE=/path/to/iree-compile
```

The Nix development shell in this repository provides the pinned compiler:

```sh
nix develop
```

See [docs/compiler.md](docs/compiler.md) for compiler setup, cache knobs, and
troubleshooting.

## Quickstart

Define a graph in `build.rs`:

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

Import the generated module and call it from normal Rust code:

```rust
use knok::prelude::*;

knok::generated_graphs!(pub mod graphs);

fn run_once(x: Tensor2<f32, 2, 2>) -> knok::Result<Tensor2<f32, 2, 2>> {
    graphs::forward::call(x)
}
```

For repeated inference, create an engine once and reuse it:

```rust
use knok::{Backend, Engine, RuntimeConfig};
use knok::prelude::*;

knok::generated_graphs!(pub mod graphs);

fn run_many(xs: impl IntoIterator<Item = Tensor2<f32, 2, 2>>) -> knok::Result<()> {
    let engine = Engine::new(RuntimeConfig::backend(Backend::LlvmCpu))?;

    for x in xs {
        let y = graphs::forward::run(&engine, x)?;
        // use y
    }

    Ok(())
}
```

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
- Build-time graph construction with `knok-build`.
- Hosted execution through IREE with the `knok` runtime API.
- External MLIR model import for users who already have MLIR.

The operation semantics are documented in [docs/semantics.md](docs/semantics.md)
and dtype support is tracked in [docs/dtypes.md](docs/dtypes.md).

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
knok = { version = "0.2", features = ["vulkan"] }

[build-dependencies]
knok-build = { version = "0.2", features = ["vulkan"] }
```

Use `cuda` instead of `vulkan` for CUDA. Metal is exposed only on macOS targets.
See [docs/backends.md](docs/backends.md) for platform notes.

## Feature Flags

| Feature | Effect |
| --- | --- |
| `runtime` | Enables hosted execution through IREE. Included by default. |
| `std` | Enables standard-library support. Included by default. |
| `half` | Enables `half::f16` and `half::bf16` tensor element types. |
| `vulkan` | Exposes the Vulkan SPIR-V backend and runtime driver. |
| `cuda` | Exposes the CUDA backend and runtime driver. |
| `embedded-runtime` | Enables the IREE runtime dependency without enabling `std`. |

With `default-features = false`, generated wrapper types can still typecheck in
`no_std + alloc` targets, but hosted runtime execution is disabled.

## Learn More

- [Static graph syntax](docs/static-graph-syntax.md): author graphs in
  `build.rs`, including external MLIR imports.
- [Tensor semantics](docs/semantics.md): rank, broadcasting, axis handling, and
  operation behavior.
- [Dtypes](docs/dtypes.md): supported element types and explicit casting policy.
- [Backends](docs/backends.md): compiler backends, runtime drivers, and platform
  notes.
- [Compiler setup](docs/compiler.md): `iree-compile` installation and cache
  behavior.

## Development

Run the full local validation suite:

```sh
scripts/release-check.sh
```

Generate coverage artifacts:

```sh
scripts/coverage.sh
```

Contributor-oriented notes live in [CONTRIBUTING.md](CONTRIBUTING.md) and
[DEVELOPERS.md](DEVELOPERS.md). User-facing changes are tracked in
[CHANGELOG.md](CHANGELOG.md).
