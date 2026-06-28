# knok

[![CI](https://github.com/gmmyung/knok/actions/workflows/rust.yml/badge.svg)](https://github.com/gmmyung/knok/actions/workflows/rust.yml)
[![Coverage](https://github.com/gmmyung/knok/actions/workflows/coverage.yml/badge.svg)](https://github.com/gmmyung/knok/actions/workflows/coverage.yml)
[![docs.rs](https://docs.rs/knok/badge.svg)](https://docs.rs/knok)
[![crates.io](https://img.shields.io/crates/v/knok.svg)](https://crates.io/crates/knok)

`knok` is an experimental static-shape tensor graph frontend for Rust. Graphs
are traced by executing host Rust from `build.rs`, compiled to IREE VM bytecode
during `cargo build`, and embedded into the target crate as generated wrappers.

## Current API

Graph definitions live in `build.rs` or a module included by `build.rs`.
`#[knok_build::graph]` records only signature and backend metadata; the function
body is ordinary Rust executed on traced tensor values during the build.

```rust
// build.rs
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
    relu(matmul(x.clone(), x) + 1.0)
}

fn main() {
    knok_build::compile_graphs!(forward);
}
```

Target code imports the generated wrappers with `generated_graphs!`:

```rust
use knok::prelude::*;

knok::generated_graphs!(pub mod graphs);

fn run(x: Tensor2<f32, 2, 2>) -> knok::Result<Tensor2<f32, 2, 2>> {
    graphs::forward::call(x)
}
```

If the build script writes a custom wrapper filename with
`BuildOptions::output_file(...)`, pass the same filename to the import macro:

```rust
knok::generated_graphs!(pub mod graphs, "custom_knok_graphs.rs");
```

Each generated graph module exposes:

- `artifact()` for custom runtime integration
- `run(&Engine, ...)` for repeated hosted inference
- `call(...)` for one-off hosted inference

For repeated inference, construct one reusable runtime engine:

```rust
use knok::{Backend, Engine, RuntimeConfig};

let engine = Engine::new(RuntimeConfig::backend(Backend::LlvmCpu))?;
let output = graphs::forward::run(&engine, x)?;
```

## Build-Time Tracing

Tracing records tensor operations instead of computing tensor values. Host-side
helpers, loops, constants, and generic Rust abstraction are allowed when they
operate on traced tensors. Tensor-value-dependent Rust control flow is not
supported because tensor values are unknown on the build host.

Build scripts can use `T0<T>` through `T6<T, D0, ...>` aliases or the
descriptive `Tensor0<T>` through `Tensor6<T, D0, ...>` aliases. The tracing
surface covers arithmetic, comparisons, logical ops, creation helpers,
shape/indexing ops, axis reductions, linalg contractions, conv2d options, and
floating-point math.

Axis, shape, and convolution options are normal Rust value arguments:

```rust
let reduced: Tensor1<f32, 2> = sum_axis(x.clone(), 1);
let stepped: Tensor1<i32, 4> = arange_step(0, 8, 2);
let convolved: Tensor4<f32, 1, 2, 2, 1> =
    conv2d_options(x, k, Conv2dOptions::new().padding(1, 1, 1, 1).stride(2, 2));
```

Build scripts can use stub artifacts for no-std/check-only fixtures:

```rust
fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
```

Stub artifacts compile generated wrappers but are not executable.

## IREE Compiler

Build-time graph compilation uses `melior` in the build script process to build
and validate MLIR, then invokes the `iree-compile` command line tool to produce
VMFB artifacts. Put `iree-compile` on `PATH`, or point `KNOK_IREE_COMPILE` at
the compiler binary:

```sh
export KNOK_IREE_COMPILE=/path/to/iree-compile
```

The Nix development shell installs the pinned IREE compiler wheel and adds
`iree-compile` to `PATH` automatically.

## Feature Modes

- Default features enable `std` and hosted runtime execution.
- `default-features = false` builds `knok` as `no_std + alloc`.
- `features = ["half"]` enables `half::f16` and `half::bf16` tensor element
  types and re-exports them as `knok::half::{f16, bf16}`.
- Build-time graph tracing lives in the separate `knok-build` build-dependency
  crate and runs on the compile host.
- Hosted runtime execution is unavailable in no-default-features mode.

## Validation

Use the release check from a shell with the IREE compiler/runtime available:

```sh
scripts/release-check.sh
```

The check covers formatting, core/lowering/build tracing tests, no-std wrapper
checks, docs, and an end-to-end build.rs traced runtime fixture.

Coverage reports are available through the Nix shell:

```sh
scripts/coverage.sh
```

The script writes `target/coverage/lcov.info` and prints a summary.

## Documentation Map

| File | Purpose |
| --- | --- |
| [docs/static-graph-syntax.md](docs/static-graph-syntax.md) | Build-time tracing graph syntax and examples. |
| [DEVELOPERS.md](DEVELOPERS.md) | Crate layout, compile/runtime flow, and release notes. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor workflow and validation commands. |
| [CHANGELOG.md](CHANGELOG.md) | User-facing changes. |
