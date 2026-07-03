#![allow(dead_code)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use knok::{prelude::*, Engine};
use ndarray::{Array1, Array2, Array3, Array4, Axis};

knok::generated_graphs!(mod graphs);

fn runtime_benches(c: &mut Criterion) {
    bench_overhead(c);
    bench_matmul(c);
    bench_batched_matmul(c);
    bench_elementwise(c);
    bench_reductions(c);
    bench_softmax(c);
    bench_layout(c);
    bench_mlp(c);
    bench_conv2d(c);
    bench_pooling(c);
}

fn bench_overhead(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::tiny_relu::artifact()).unwrap();
    let input = Tensor2::<f32, 2, 2>::from_array([[-1.0, 2.0], [3.0, -4.0]]);

    c.bench_function("knok/tiny_relu_2x2", |b| {
        b.iter(|| {
            let output = graphs::tiny_relu::run(&engine, black_box(input.clone())).unwrap();
            black_box(checksum(output.as_slice()))
        })
    });
}

fn bench_matmul(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::matmul_128::artifact()).unwrap();
    let x_data = filled_data(128 * 128, 0.001, 1.0);
    let y_data = filled_data(128 * 128, 0.002, -0.5);
    let x = Tensor2::<f32, 128, 128>::from_vec(x_data.clone()).unwrap();
    let y = Tensor2::<f32, 128, 128>::from_vec(y_data.clone()).unwrap();
    let ndarray_x = Array2::from_shape_vec((128, 128), x_data).unwrap();
    let ndarray_y = Array2::from_shape_vec((128, 128), y_data).unwrap();

    c.bench_function("knok/matmul_128x128", |b| {
        b.iter(|| {
            let output =
                graphs::matmul_128::run(&engine, black_box(x.clone()), black_box(y.clone()))
                    .unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/matmul_128x128", |b| {
        b.iter(|| {
            let output = black_box(&ndarray_x).dot(black_box(&ndarray_y));
            black_box(checksum_iter(output.iter()))
        })
    });
}

fn bench_batched_matmul(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::batched_matmul_16x128::artifact()).unwrap();
    let len = 16 * 128 * 128;
    let x_data = filled_data(len, 0.0005, 0.25);
    let y_data = filled_data(len, 0.0007, -0.75);
    let x = Tensor3::<f32, 16, 128, 128>::from_vec(x_data.clone()).unwrap();
    let y = Tensor3::<f32, 16, 128, 128>::from_vec(y_data.clone()).unwrap();
    let ndarray_x = Array3::from_shape_vec((16, 128, 128), x_data).unwrap();
    let ndarray_y = Array3::from_shape_vec((16, 128, 128), y_data).unwrap();

    c.bench_function("knok/batched_matmul_16x128x128", |b| {
        b.iter(|| {
            let output = graphs::batched_matmul_16x128::run(
                &engine,
                black_box(x.clone()),
                black_box(y.clone()),
            )
            .unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/batched_matmul_16x128x128", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for batch in 0..16 {
                let output = black_box(&ndarray_x)
                    .index_axis(Axis(0), batch)
                    .dot(&black_box(&ndarray_y).index_axis(Axis(0), batch));
                sum += checksum_iter(output.iter());
            }
            black_box(sum)
        })
    });
}

fn bench_elementwise(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::elementwise_1024::artifact()).unwrap();
    let data = filled_data(1024 * 1024, 0.0001, -0.25);
    let input = Tensor2::<f32, 1024, 1024>::from_vec(data.clone()).unwrap();
    let ndarray_input = Array2::from_shape_vec((1024, 1024), data).unwrap();

    c.bench_function("knok/elementwise_exp_tanh_relu_1024x1024", |b| {
        b.iter(|| {
            let output = graphs::elementwise_1024::run(&engine, black_box(input.clone())).unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/elementwise_exp_tanh_relu_1024x1024", |b| {
        b.iter(|| {
            let output = black_box(&ndarray_input).mapv(|value| {
                let value = value.exp() + value.tanh() * 0.5;
                value.max(0.0)
            });
            black_box(checksum_iter(output.iter()))
        })
    });
}

fn bench_reductions(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::sum_axis1_512::artifact()).unwrap();
    let data = filled_data(512 * 512, 0.001, 1.0);
    let input = Tensor2::<f32, 512, 512>::from_vec(data.clone()).unwrap();
    let ndarray_input = Array2::from_shape_vec((512, 512), data).unwrap();

    c.bench_function("knok/sum_axis1_512x512", |b| {
        b.iter(|| {
            let output = graphs::sum_axis1_512::run(&engine, black_box(input.clone())).unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/sum_axis1_512x512", |b| {
        b.iter(|| {
            let output = black_box(&ndarray_input).sum_axis(Axis(1));
            black_box(checksum_iter(output.iter()))
        })
    });
}

fn bench_softmax(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::softmax_axis1_512x1024::artifact()).unwrap();
    let data = filled_data(512 * 1024, 0.0001, -0.5);
    let input = Tensor2::<f32, 512, 1024>::from_vec(data.clone()).unwrap();
    let ndarray_input = Array2::from_shape_vec((512, 1024), data).unwrap();

    c.bench_function("knok/softmax_axis1_512x1024", |b| {
        b.iter(|| {
            let output =
                graphs::softmax_axis1_512x1024::run(&engine, black_box(input.clone())).unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/softmax_axis1_512x1024", |b| {
        b.iter(|| black_box(ndarray_softmax_axis1_checksum(&ndarray_input)))
    });
}

fn bench_layout(c: &mut Criterion) {
    let transpose_engine = Engine::for_artifact(graphs::transpose_512x256::artifact()).unwrap();
    let broadcast_engine = Engine::for_artifact(graphs::broadcast_row_512::artifact()).unwrap();
    let matrix_data = filled_data(512 * 256, 0.001, 0.0);
    let row_data = filled_data(512, 0.01, 1.0);
    let matrix = Tensor2::<f32, 512, 256>::from_vec(matrix_data.clone()).unwrap();
    let row = Tensor1::<f32, 512>::from_vec(row_data.clone()).unwrap();
    let ndarray_matrix = Array2::from_shape_vec((512, 256), matrix_data).unwrap();
    let ndarray_row = Array1::from_vec(row_data);

    c.bench_function("knok/transpose_512x256", |b| {
        b.iter(|| {
            let output =
                graphs::transpose_512x256::run(&transpose_engine, black_box(matrix.clone()))
                    .unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/transpose_512x256", |b| {
        b.iter(|| {
            let output = black_box(&ndarray_matrix).t().to_owned();
            black_box(checksum_iter(output.iter()))
        })
    });

    c.bench_function("knok/broadcast_row_512_to_512x512", |b| {
        b.iter(|| {
            let output =
                graphs::broadcast_row_512::run(&broadcast_engine, black_box(row.clone())).unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/broadcast_row_512_to_512x512", |b| {
        b.iter(|| {
            let output = black_box(&ndarray_row)
                .broadcast((512, 512))
                .unwrap()
                .to_owned();
            black_box(checksum_iter(output.iter()))
        })
    });
}

fn bench_mlp(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::mlp_64x128x256x64::artifact()).unwrap();
    let x_data = filled_data(64 * 128, 0.001, 0.5);
    let w1_data = filled_data(128 * 256, 0.0005, 0.01);
    let b1_data = filled_data(256, 0.001, 0.1);
    let w2_data = filled_data(256 * 64, 0.0007, 0.02);
    let x = Tensor2::<f32, 64, 128>::from_vec(x_data.clone()).unwrap();
    let w1 = Tensor2::<f32, 128, 256>::from_vec(w1_data.clone()).unwrap();
    let b1 = Tensor1::<f32, 256>::from_vec(b1_data.clone()).unwrap();
    let w2 = Tensor2::<f32, 256, 64>::from_vec(w2_data.clone()).unwrap();
    let ndarray_x = Array2::from_shape_vec((64, 128), x_data).unwrap();
    let ndarray_w1 = Array2::from_shape_vec((128, 256), w1_data).unwrap();
    let ndarray_b1 = Array1::from_vec(b1_data);
    let ndarray_w2 = Array2::from_shape_vec((256, 64), w2_data).unwrap();

    c.bench_function("knok/mlp_64x128x256x64", |b| {
        b.iter(|| {
            let output = graphs::mlp_64x128x256x64::run(
                &engine,
                black_box(x.clone()),
                black_box(w1.clone()),
                black_box(b1.clone()),
                black_box(w2.clone()),
            )
            .unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/mlp_64x128x256x64", |b| {
        b.iter(|| {
            let mut hidden = black_box(&ndarray_x).dot(black_box(&ndarray_w1));
            hidden += black_box(&ndarray_b1);
            hidden.mapv_inplace(|value| value.max(0.0));
            let output = hidden.dot(black_box(&ndarray_w2));
            black_box(checksum_iter(output.iter()))
        })
    });
}

fn bench_conv2d(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::conv2d_nhwc_8x32x32x3::artifact()).unwrap();
    let x =
        Tensor4::<f32, 8, 32, 32, 3>::from_vec(filled_data(8 * 32 * 32 * 3, 0.001, 0.0)).unwrap();
    let k = Tensor4::<f32, 3, 3, 3, 16>::from_vec(filled_data(3 * 3 * 3 * 16, 0.01, -0.2)).unwrap();

    c.bench_function("knok/conv2d_nhwc_8x32x32x3_hwcf_3x3x3x16", |b| {
        b.iter(|| {
            let output = graphs::conv2d_nhwc_8x32x32x3::run(
                &engine,
                black_box(x.clone()),
                black_box(k.clone()),
            )
            .unwrap();
            black_box(checksum(output.as_slice()))
        })
    });
}

fn bench_pooling(c: &mut Criterion) {
    let engine = Engine::for_artifact(graphs::max_pool2d_nhwc_16x64x64x16::artifact()).unwrap();
    let len = 16 * 64 * 64 * 16;
    let data = filled_data(len, 0.001, -0.5);
    let input = Tensor4::<f32, 16, 64, 64, 16>::from_vec(data.clone()).unwrap();
    let ndarray_input = Array4::from_shape_vec((16, 64, 64, 16), data).unwrap();

    c.bench_function("knok/max_pool2d_nhwc_16x64x64x16", |b| {
        b.iter(|| {
            let output =
                graphs::max_pool2d_nhwc_16x64x64x16::run(&engine, black_box(input.clone()))
                    .unwrap();
            black_box(checksum(output.as_slice()))
        })
    });

    c.bench_function("ndarray/max_pool2d_nhwc_16x64x64x16", |b| {
        b.iter(|| black_box(ndarray_max_pool2d_nhwc_checksum(&ndarray_input)))
    });
}

fn filled_data(len: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|index| (index % 257) as f32 * scale + offset)
        .collect()
}

fn checksum(values: &[f32]) -> f32 {
    values.iter().copied().sum::<f32>()
}

fn checksum_iter<'a>(values: impl IntoIterator<Item = &'a f32>) -> f32 {
    values.into_iter().copied().sum::<f32>()
}

fn ndarray_max_pool2d_nhwc_checksum(input: &Array4<f32>) -> f32 {
    let (batch, height, width, channels) = input.dim();
    let mut sum = 0.0;
    for n in 0..batch {
        for h in 0..(height / 2) {
            for w in 0..(width / 2) {
                for c in 0..channels {
                    let h0 = h * 2;
                    let w0 = w * 2;
                    let value = input[[n, h0, w0, c]]
                        .max(input[[n, h0 + 1, w0, c]])
                        .max(input[[n, h0, w0 + 1, c]])
                        .max(input[[n, h0 + 1, w0 + 1, c]]);
                    sum += value;
                }
            }
        }
    }
    sum
}

fn ndarray_softmax_axis1_checksum(input: &Array2<f32>) -> f32 {
    let mut output = input.clone();
    for mut row in output.axis_iter_mut(Axis(0)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|value| (value - max).exp());
        let sum = row.iter().copied().sum::<f32>();
        row.mapv_inplace(|value| value / sum);
    }
    checksum_iter(output.iter())
}

criterion_group!(benches, runtime_benches);
criterion_main!(benches);
