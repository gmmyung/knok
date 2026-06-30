# Benchmarks

`knok` keeps runtime benchmarks outside the default workspace checks so normal
CI and release validation stay focused on correctness. The benchmark harness is
a lightweight standalone crate under `benchmarks/runtime`.

## Command

Run the local benchmark harness from the repository root:

```sh
nix develop --command scripts/benchmark.sh
```

The script builds and runs `benchmarks/runtime` in release mode. It uses the
pinned `iree-compile` from the Nix shell.

Tune sample counts with environment variables:

```sh
KNOK_BENCH_WARMUP=5 \
KNOK_BENCH_SAMPLES=50 \
nix develop --command scripts/benchmark.sh
```

The script prints mean and median timings and also writes:

- `benchmarks/runtime/target/benchmark-summary.csv`
- `benchmarks/runtime/target/benchmark-summary.json`

## Cases

The harness currently reports:

| Case | Shape | What it measures |
| --- | --- | --- |
| `knok tiny_relu` | `2x2 -> 2x2` | Lower bound for hosted graph call overhead. |
| `knok matmul` | `128x128 @ 128x128 -> 128x128` | Single matrix multiply through IREE. |
| `ndarray matmul` | `128x128 @ 128x128 -> 128x128` | `ndarray::Array2::dot` baseline. |
| `knok batched matmul` | `16x128x128 @ 16x128x128 -> 16x128x128` | Batched matrix multiply through IREE. |
| `ndarray batched matmul` | `16x128x128 @ 16x128x128 -> 16x128x128` | Batch loop over `ndarray::Array2::dot`. |
| `knok elementwise` | `1024x1024 -> 1024x1024` | Fused `exp + tanh * 0.5` followed by ReLU. |
| `ndarray elementwise` | `1024x1024 -> 1024x1024` | Matching `ndarray::mapv` baseline. |
| `knok sum_axis1` | `512x512 -> 512` | Axis reduction through IREE. |
| `ndarray sum_axis1` | `512x512 -> 512` | `ndarray::sum_axis` baseline. |
| `knok softmax_axis1` | `512x1024 -> 512x1024` | Row-wise softmax through IREE. |
| `ndarray softmax_axis1` | `512x1024 -> 512x1024` | Row-wise ndarray implementation. |
| `knok transpose` | `512x256 -> 256x512` | Layout transform through IREE. |
| `ndarray transpose` | `512x256 -> 256x512` | `ndarray` transpose materialized with `to_owned`. |
| `knok broadcast` | `512 -> 512x512` | Broadcast materialization through IREE. |
| `ndarray broadcast` | `512 -> 512x512` | `ndarray` broadcast materialized with `to_owned`. |
| `knok MLP` | `64x128 -> 64x64` | `matmul + bias + relu + matmul`. |
| `ndarray MLP` | `64x128 -> 64x64` | Matching ndarray matmul pipeline. |
| `knok conv2d` | `8x32x32x3, 3x3x3x16 -> 8x30x30x16` | NHWC/HWCF convolution through IREE. |

The `ndarray` baselines reuse prebuilt arrays in the timed loop. `knok` cases
reuse an `Engine` and clone input tensors per iteration because `run` consumes
typed tensor inputs.

Pooling is not benchmarked yet because there is no public pooling graph op.

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

`knok` runtime measurements include:

- host tensor to IREE buffer-view creation
- VM function invocation
- backend dispatch
- output buffer reads into Rust tensors

They do not include build-time graph tracing or VMFB compilation. Those happen
when the benchmark crate builds.
