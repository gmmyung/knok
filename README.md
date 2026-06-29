# knok

[![CI](https://github.com/gmmyung/knok/actions/workflows/rust.yml/badge.svg)](https://github.com/gmmyung/knok/actions/workflows/rust.yml)
[![Coverage](https://raw.githubusercontent.com/gmmyung/knok/badges/coverage.svg)](https://github.com/gmmyung/knok/actions/workflows/coverage.yml)
[![docs.rs](https://docs.rs/knok/badge.svg)](https://docs.rs/knok)
[![crates.io](https://img.shields.io/crates/v/knok.svg)](https://crates.io/crates/knok)

`knok` is an experimental static-shape tensor graph frontend for Rust. Graphs
are traced by executing host Rust from `build.rs`, compiled to IREE VM bytecode
during `cargo build`, and embedded into the target crate as generated wrappers.

## Quickstart

Define graph functions in `build.rs`:

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

Import and run the generated wrapper from target code:

```rust
use knok::prelude::*;

knok::generated_graphs!(pub mod graphs);

fn run(x: Tensor2<f32, 2, 2>) -> knok::Result<Tensor2<f32, 2, 2>> {
    graphs::forward::call(x)
}
```

For repeated hosted inference, reuse an engine:

```rust
use knok::{Backend, Engine, RuntimeConfig};

let engine = Engine::new(RuntimeConfig::backend(Backend::LlvmCpu))?;
let output = graphs::forward::run(&engine, x)?;
```

Build-time graph compilation uses `melior` for MLIR construction/validation and
the `iree-compile` command line tool for VMFB generation. Put `iree-compile` on
`PATH`, or set `KNOK_IREE_COMPILE=/path/to/iree-compile`. The Nix development
shell provides the pinned compiler automatically. See
[docs/compiler.md](docs/compiler.md) for setup details.

## Feature Modes

- Default features enable `std` and hosted runtime execution.
- `default-features = false` builds `knok` as `no_std + alloc`; generated
  wrappers can typecheck, but hosted runtime execution is unavailable.
- `features = ["half"]` enables `half::f16` and `half::bf16` tensor element
  types and re-exports them as `knok::half::{f16, bf16}`.
- Build-time graph tracing lives in `knok-build`, a build-dependency crate that
  runs on the compile host.

## Validation

```sh
scripts/release-check.sh
scripts/coverage.sh
```

`release-check.sh` covers formatting, core/lowering/build tracing tests, no-std
wrapper checks, docs, and runtime E2E fixtures. `coverage.sh` writes LCOV, HTML,
and badge outputs under `target/coverage`.

## Documentation Map

| File | Purpose |
| --- | --- |
| [docs/static-graph-syntax.md](docs/static-graph-syntax.md) | Build-time graph authoring syntax. |
| [docs/semantics.md](docs/semantics.md) | Tensor ranks, dtype, broadcasting, axis, and op semantics. |
| [docs/dtypes.md](docs/dtypes.md) | Supported dtype matrix and explicit casting policy. |
| [docs/backends.md](docs/backends.md) | Compiler backends, runtime drivers, and platform notes. |
| [docs/compiler.md](docs/compiler.md) | `iree-compile` setup, cache knobs, and troubleshooting. |
| [docs/lowering.md](docs/lowering.md) | MLIR lowering architecture and parsed-attribute policy. |
| [docs/testing.md](docs/testing.md) | Test layers and validation commands. |
| [docs/release-readiness.md](docs/release-readiness.md) | Release checklist and dry-run commands. |
| [DEVELOPERS.md](DEVELOPERS.md) | Crate layout, compile/runtime flow, and release notes. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor workflow and validation commands. |
| [CHANGELOG.md](CHANGELOG.md) | User-facing changes. |
