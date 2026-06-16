use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use knok::prelude::*;
use knok::{Engine, RuntimeConfig};
use nalgebra::DMatrix;
use ndarray::{s, Array2, Array3, Array4};
use std::hint::black_box;

#[knok::graph(backend = "llvm-cpu")]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[knok::graph(backend = "llvm-cpu")]
fn matmul_64(x: Tensor2<f32, 64, 64>, y: Tensor2<f32, 64, 64>) -> Tensor2<f32, 64, 64> {
    matmul(x, y)
}

#[knok::graph(backend = "llvm-cpu")]
fn batched_matmul_32x32(
    x: Tensor3<f32, 32, 32, 32>,
    y: Tensor3<f32, 32, 32, 32>,
) -> Tensor3<f32, 32, 32, 32> {
    matmul(x, y)
}

#[knok::graph(backend = "llvm-cpu")]
fn conv2d_16x32x32x3_16(
    x: Tensor4<f32, 16, 32, 32, 3>,
    k: Tensor4<f32, 3, 3, 3, 16>,
) -> Tensor4<f32, 16, 30, 30, 16> {
    conv2d(x, k)
}

#[knok::graph(backend = "llvm-cpu")]
fn mlp_128_batch_128_64(
    x: Tensor2<f32, 128, 128>,
    w: Tensor2<f32, 128, 64>,
    b: Tensor2<f32, 128, 64>,
) -> Tensor2<f32, 128, 64> {
    relu(matmul(x, w) + b)
}

#[knok::graph(backend = "llvm-cpu")]
fn softmax_64x1000(logits: Tensor2<f32, 64, 1000>) -> Tensor2<f32, 64, 1000> {
    softmax(logits)
}

fn patterned_vec(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| ((index % 31) as f32 - 15.0) / 16.0)
        .collect()
}

fn small_benches(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("small");
    let engine = Engine::new(RuntimeConfig::auto()).expect("failed to create runtime engine");

    group.bench_function("tensor1_from_array_shape_4", |bencher| {
        bencher.iter(|| Tensor1::from_array(black_box([1.0, 2.0, 3.0, 4.0])))
    });

    group.bench_function("tensor4_zeros_shape_1x8x8x3", |bencher| {
        bencher.iter(|| Tensor4::<f32, 1, 8, 8, 3>::zeros())
    });

    group.bench_function("add4_knok_reusable_engine", |bencher| {
        bencher.iter(|| {
            let x = Tensor1::from_array(black_box([1.0, 2.0, 3.0, 4.0]));
            let y = Tensor1::from_array(black_box([10.0, 20.0, 30.0, 40.0]));
            black_box(add4_run(&engine, x, y).unwrap())
        })
    });

    group.bench_function("add4_knok_convenience_engine_per_call", |bencher| {
        bencher.iter(|| {
            let x = Tensor1::from_array(black_box([1.0, 2.0, 3.0, 4.0]));
            let y = Tensor1::from_array(black_box([10.0, 20.0, 30.0, 40.0]));
            black_box(add4(x, y).unwrap())
        })
    });

    group.finish();
}

fn large_knok_benches(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("large_knok");
    group.sample_size(10);
    let engine = Engine::new(RuntimeConfig::auto()).expect("failed to create runtime engine");

    let matmul_lhs = patterned_vec(64 * 64);
    let matmul_rhs = patterned_vec(64 * 64);
    group.bench_function("matmul_64x64_reusable_engine", |bencher| {
        bencher.iter(|| {
            let lhs = Tensor2::<f32, 64, 64>::from_vec(black_box(matmul_lhs.clone())).unwrap();
            let rhs = Tensor2::<f32, 64, 64>::from_vec(black_box(matmul_rhs.clone())).unwrap();
            black_box(matmul_64_run(&engine, lhs, rhs).unwrap())
        })
    });

    let batch_lhs = patterned_vec(32 * 32 * 32);
    let batch_rhs = patterned_vec(32 * 32 * 32);
    group.bench_function("batched_matmul_32x32x32_reusable_engine", |bencher| {
        bencher.iter(|| {
            let lhs = Tensor3::<f32, 32, 32, 32>::from_vec(black_box(batch_lhs.clone())).unwrap();
            let rhs = Tensor3::<f32, 32, 32, 32>::from_vec(black_box(batch_rhs.clone())).unwrap();
            black_box(batched_matmul_32x32_run(&engine, lhs, rhs).unwrap())
        })
    });

    let conv_input = patterned_vec(16 * 32 * 32 * 3);
    let conv_kernel = patterned_vec(3 * 3 * 3 * 16);
    group.bench_function("conv2d_nhwc_16x32x32x3_hwcf_3x3x3x16", |bencher| {
        bencher.iter(|| {
            let input =
                Tensor4::<f32, 16, 32, 32, 3>::from_vec(black_box(conv_input.clone())).unwrap();
            let kernel =
                Tensor4::<f32, 3, 3, 3, 16>::from_vec(black_box(conv_kernel.clone())).unwrap();
            black_box(conv2d_16x32x32x3_16_run(&engine, input, kernel).unwrap())
        })
    });

    let mlp_x = patterned_vec(128 * 128);
    let mlp_w = patterned_vec(128 * 64);
    let mlp_b = patterned_vec(128 * 64);
    group.bench_function("mlp_128x128x64_reusable_engine", |bencher| {
        bencher.iter(|| {
            let x = Tensor2::<f32, 128, 128>::from_vec(black_box(mlp_x.clone())).unwrap();
            let w = Tensor2::<f32, 128, 64>::from_vec(black_box(mlp_w.clone())).unwrap();
            let b = Tensor2::<f32, 128, 64>::from_vec(black_box(mlp_b.clone())).unwrap();
            black_box(mlp_128_batch_128_64_run(&engine, x, w, b).unwrap())
        })
    });

    let logits = patterned_vec(64 * 1000);
    group.bench_function("softmax_64x1000_reusable_engine", |bencher| {
        bencher.iter(|| {
            let logits = Tensor2::<f32, 64, 1000>::from_vec(black_box(logits.clone())).unwrap();
            black_box(softmax_64x1000_run(&engine, logits).unwrap())
        })
    });

    group.finish();
}

fn ndarray_comparison_benches(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("compare_ndarray");
    group.sample_size(10);

    let matmul_lhs = Array2::from_shape_vec((64, 64), patterned_vec(64 * 64)).unwrap();
    let matmul_rhs = Array2::from_shape_vec((64, 64), patterned_vec(64 * 64)).unwrap();
    group.bench_function("matmul_64x64", |bencher| {
        bencher.iter(|| black_box(matmul_lhs.dot(black_box(&matmul_rhs))))
    });

    let batch_lhs = Array3::from_shape_vec((32, 32, 32), patterned_vec(32 * 32 * 32)).unwrap();
    let batch_rhs = Array3::from_shape_vec((32, 32, 32), patterned_vec(32 * 32 * 32)).unwrap();
    group.bench_function("batched_matmul_32x32x32", |bencher| {
        bencher.iter(|| {
            let mut output = Array3::<f32>::zeros((32, 32, 32));
            for batch in 0..32 {
                let lhs = batch_lhs.index_axis(ndarray::Axis(0), batch);
                let rhs = batch_rhs.index_axis(ndarray::Axis(0), batch);
                output.slice_mut(s![batch, .., ..]).assign(&lhs.dot(&rhs));
            }
            black_box(output)
        })
    });

    let conv_input =
        Array4::from_shape_vec((16, 32, 32, 3), patterned_vec(16 * 32 * 32 * 3)).unwrap();
    let conv_kernel = Array4::from_shape_vec((3, 3, 3, 16), patterned_vec(3 * 3 * 3 * 16)).unwrap();
    group.bench_function("conv2d_nhwc_16x32x32x3_hwcf_3x3x3x16", |bencher| {
        bencher.iter(|| black_box(ndarray_conv2d_valid(&conv_input, &conv_kernel)))
    });

    let mlp_x = Array2::from_shape_vec((128, 128), patterned_vec(128 * 128)).unwrap();
    let mlp_w = Array2::from_shape_vec((128, 64), patterned_vec(128 * 64)).unwrap();
    let mlp_b = Array2::from_shape_vec((128, 64), patterned_vec(128 * 64)).unwrap();
    group.bench_function("mlp_128x128x64", |bencher| {
        bencher.iter(|| {
            let mut output = mlp_x.dot(black_box(&mlp_w)) + black_box(&mlp_b);
            output.mapv_inplace(|value| value.max(0.0));
            black_box(output)
        })
    });

    let logits = Array2::from_shape_vec((64, 1000), patterned_vec(64 * 1000)).unwrap();
    group.bench_function("softmax_64x1000", |bencher| {
        bencher.iter(|| black_box(ndarray_softmax(&logits)))
    });

    group.finish();
}

fn nalgebra_comparison_benches(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("compare_nalgebra");
    group.sample_size(10);

    let matmul_lhs = DMatrix::from_row_slice(64, 64, &patterned_vec(64 * 64));
    let matmul_rhs = DMatrix::from_row_slice(64, 64, &patterned_vec(64 * 64));
    group.bench_function(BenchmarkId::new("matmul", "64x64"), |bencher| {
        bencher.iter(|| black_box(black_box(&matmul_lhs) * black_box(&matmul_rhs)))
    });

    group.finish();
}

fn ndarray_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let exp = logits.mapv(|value| (value - max).exp());
    let sum = exp.sum();
    exp / sum
}

fn ndarray_conv2d_valid(input: &Array4<f32>, kernel: &Array4<f32>) -> Array4<f32> {
    let mut output = Array4::<f32>::zeros((16, 30, 30, 16));
    for n in 0..16 {
        for oh in 0..30 {
            for ow in 0..30 {
                for oc in 0..16 {
                    let mut acc = 0.0;
                    for kh in 0..3 {
                        for kw in 0..3 {
                            for ic in 0..3 {
                                acc += input[[n, oh + kh, ow + kw, ic]] * kernel[[kh, kw, ic, oc]];
                            }
                        }
                    }
                    output[[n, oh, ow, oc]] = acc;
                }
            }
        }
    }
    output
}

criterion_group!(
    benches,
    small_benches,
    large_knok_benches,
    ndarray_comparison_benches,
    nalgebra_comparison_benches,
);
criterion_main!(benches);
