# Benchmarks

`knok` keeps runtime benchmarks outside the default workspace checks so normal
CI and release validation stay focused on correctness.

## Command

Run the local benchmark harness from the repository root:

```sh
nix develop --command scripts/benchmark.sh
```

The script builds and runs `benchmarks/runtime` in release mode. It uses the
pinned `iree-compile` from the Nix shell.

Tune iteration counts with environment variables:

```sh
KNOK_BENCH_WARMUP=5 KNOK_BENCH_ITERS=50 nix develop --command scripts/benchmark.sh
```

## Cases

The harness currently reports:

| Case | Shape | What it measures |
| --- | --- | --- |
| `knok tiny_relu` | `2x2 -> 2x2` | Lower bound for hosted graph call overhead. |
| `knok matmul` | `128x128 @ 128x128 -> 128x128` | Single matrix multiply through IREE. |
| `ndarray matmul` | `128x128 @ 128x128 -> 128x128` | `ndarray::Array2::dot` baseline. |
| `knok batched matmul` | `16x128x128 @ 16x128x128 -> 16x128x128` | Batched matrix multiply through IREE. |
| `ndarray batched matmul` | `16x128x128 @ 16x128x128 -> 16x128x128` | Batch loop over `ndarray::Array2::dot`. |

The `ndarray` baselines reuse prebuilt arrays in the timed loop. The batched
case loops over the batch axis and calls `Array2::dot` per batch.

## Engine Reuse

Benchmarks use reusable engines:

```rust
let engine = knok::Engine::for_artifact(graphs::matmul_128::artifact())?;
let y = graphs::matmul_128::run(&engine, x, w)?;
```

This excludes one-shot engine construction from the timed loop and matches the
recommended path for repeated inference.

The generated `call(...)` convenience wrapper is useful for occasional
execution, but it constructs an engine from the artifact each time. Do not use
`call(...)` for steady-state throughput measurements.

## Interpreting Results

Runtime measurements include:

- host tensor to IREE buffer-view creation
- VM function invocation
- backend dispatch
- output buffer reads into Rust tensors

They do not include build-time graph tracing or VMFB compilation. Those happen
when the benchmark crate builds.
