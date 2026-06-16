use criterion::{criterion_group, criterion_main, Criterion};
use knok::prelude::*;
use knok::{Engine, RuntimeConfig};
use std::hint::black_box;

#[knok::graph(backend = "llvm-cpu")]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[knok::graph(backend = "llvm-cpu")]
fn mlp_block(
    x: Tensor2<f32, 1, 3>,
    w: Tensor2<f32, 3, 2>,
    b: Tensor2<f32, 1, 2>,
) -> Tensor2<f32, 1, 2> {
    relu(matmul(x, w) + b)
}

#[knok::graph(backend = "llvm-cpu")]
fn classifier_head(logits: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    softmax(logits)
}

fn tensor_benches(criterion: &mut Criterion) {
    criterion.bench_function("tensor1_from_array", |bencher| {
        bencher.iter(|| Tensor1::from_array(black_box([1.0, 2.0, 3.0, 4.0])))
    });

    criterion.bench_function("tensor4_zeros", |bencher| {
        bencher.iter(|| Tensor4::<f32, 1, 8, 8, 3>::zeros())
    });

    criterion.bench_function("tensor3_checked_index_mut", |bencher| {
        bencher.iter(|| {
            let mut tensor = Tensor3::<f32, 4, 4, 4>::ones();
            *tensor
                .get_mut(black_box(2), black_box(1), black_box(3))
                .unwrap() = black_box(7.0);
            black_box(tensor)
        })
    });
}

fn runtime_benches(criterion: &mut Criterion) {
    let engine = Engine::new(RuntimeConfig::auto()).expect("failed to create runtime engine");

    criterion.bench_function("add4_reusable_engine", |bencher| {
        bencher.iter(|| {
            let x = Tensor1::from_array(black_box([1.0, 2.0, 3.0, 4.0]));
            let y = Tensor1::from_array(black_box([10.0, 20.0, 30.0, 40.0]));
            black_box(add4_run(&engine, x, y).unwrap())
        })
    });

    criterion.bench_function("add4_convenience_engine_per_call", |bencher| {
        bencher.iter(|| {
            let x = Tensor1::from_array(black_box([1.0, 2.0, 3.0, 4.0]));
            let y = Tensor1::from_array(black_box([10.0, 20.0, 30.0, 40.0]));
            black_box(add4(x, y).unwrap())
        })
    });

    criterion.bench_function("mlp_block_reusable_engine", |bencher| {
        bencher.iter(|| {
            let x = Tensor2::from_array(black_box([[1.0, 2.0, 3.0]]));
            let w = Tensor2::from_array(black_box([[1.0, -1.0], [0.5, 2.0], [-1.0, 0.25]]));
            let b = Tensor2::from_array(black_box([[0.25, -0.5]]));
            black_box(mlp_block_run(&engine, x, w, b).unwrap())
        })
    });

    criterion.bench_function("classifier_softmax_reusable_engine", |bencher| {
        bencher.iter(|| {
            let logits = Tensor1::from_array(black_box([1000.0, 1001.0, 1002.0, 1003.0]));
            black_box(classifier_head_run(&engine, logits).unwrap())
        })
    });
}

criterion_group!(benches, tensor_benches, runtime_benches);
criterion_main!(benches);
