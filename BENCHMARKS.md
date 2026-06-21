# Benchmarks

These are quick Criterion runs for tracking broad performance shape. They use a
short sample window, so treat them as a local snapshot instead of publish-grade
numbers.

Environment:

- Date: 2026-06-21
- Machine: Apple M5 Pro, 15 CPU cores
- OS: macOS Darwin 25.5.0 arm64
- Command:
  `cargo bench -p knok --bench runtime -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.2`
- Cargo profile: `bench` / optimized release build

## Runtime

| Benchmark | Median-ish Criterion Estimate |
| --- | ---: |
| `tensor1_from_array_shape_4` | 10.8 ns |
| `tensor4_zeros_shape_1x8x8x3` | 20.7 ns |
| `add4_knok_reusable_engine` | 11.6 us |
| `add4_knok_convenience_engine_per_call` | 2.02 ms |
| `matmul_128x128_reusable_engine` | 81.5 us |
| `batched_matmul_16x128x128_reusable_engine` | 299 us |
| `conv2d_nhwc_16x32x32x3_hwcf_3x3x3x16` | 127 us |
| `mlp_128x128x64_reusable_engine` | 79.7 us |
| `softmax_64x1000_reusable_engine` | 223 us |

## Comparisons

| Operation | knok | ndarray | nalgebra |
| --- | ---: | ---: | ---: |
| `matmul_128x128` | 81.5 us | 42.9 us | 43.5 us |
| `batched_matmul_16x128x128` | 299 us | 718 us | n/a |
| `conv2d_nhwc_16x32x32x3_hwcf_3x3x3x16` | 127 us | 1.73 ms | n/a |
| `mlp_128x128x64` | 79.7 us | 24.5 us | n/a |
| `softmax_64x1000` | 223 us | 86.8 us | n/a |

Observations:

- Reusing an `Engine` is mandatory for meaningful inference timings on small
  graphs. Constructing runtime state per call is about 2 ms in this run.
- `matmul_128x128`, MLP, and softmax are still slower than ndarray/nalgebra on
  this CPU path.
- Batched matmul and convolution are faster than the current ndarray comparison
  implementations, though those comparison kernels are not a substitute for a
  tuned convolution library.
