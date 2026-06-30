use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use knok::{prelude::*, Engine};
use ndarray::{Array2, Array3, Axis};

knok::generated_graphs!(pub mod graphs);

fn main() -> knok::Result<()> {
    let iterations = env_usize("KNOK_BENCH_ITERS", 20);
    let warmup = env_usize("KNOK_BENCH_WARMUP", 3);

    println!("backend=llvm-cpu driver=local-task");
    println!("warmup={warmup} iterations={iterations}");

    bench_tiny_relu(warmup, iterations)?;
    bench_matmul_128(warmup, iterations)?;
    bench_batched_matmul_16x128(warmup, iterations)?;

    Ok(())
}

fn bench_tiny_relu(warmup: usize, iterations: usize) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::tiny_relu::artifact())?;
    let input = Tensor2::<f32, 2, 2>::from_array([[-1.0, 2.0], [3.0, -4.0]]);

    for _ in 0..warmup {
        black_box(graphs::tiny_relu::run(&engine, input.clone())?);
    }

    let (duration, checksum) = measure(iterations, || {
        let output = graphs::tiny_relu::run(&engine, input.clone())?;
        Ok(checksum(output.as_slice()))
    })?;

    report("knok tiny_relu 2x2", iterations, duration, checksum);
    Ok(())
}

fn bench_matmul_128(warmup: usize, iterations: usize) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::matmul_128::artifact())?;
    let x_data = filled_data(128 * 128, 0.001, 1.0);
    let y_data = filled_data(128 * 128, 0.002, -0.5);
    let x = Tensor2::<f32, 128, 128>::from_vec(x_data.clone())?;
    let y = Tensor2::<f32, 128, 128>::from_vec(y_data.clone())?;
    let ndarray_x =
        Array2::from_shape_vec((128, 128), x_data.clone()).expect("valid lhs matrix shape");
    let ndarray_y =
        Array2::from_shape_vec((128, 128), y_data.clone()).expect("valid rhs matrix shape");

    for _ in 0..warmup {
        black_box(graphs::matmul_128::run(&engine, x.clone(), y.clone())?);
    }

    let (knok_duration, knok_checksum) = measure(iterations, || {
        let output = graphs::matmul_128::run(&engine, x.clone(), y.clone())?;
        Ok(checksum(output.as_slice()))
    })?;
    report(
        "knok matmul 128x128 @ 128x128",
        iterations,
        knok_duration,
        knok_checksum,
    );

    for _ in 0..warmup {
        black_box(ndarray_matmul_checksum(&ndarray_x, &ndarray_y));
    }

    let (ndarray_duration, ndarray_checksum) = measure(iterations, || {
        Ok(ndarray_matmul_checksum(&ndarray_x, &ndarray_y))
    })?;
    report(
        "ndarray matmul 128x128 @ 128x128",
        iterations,
        ndarray_duration,
        ndarray_checksum,
    );

    Ok(())
}

fn bench_batched_matmul_16x128(warmup: usize, iterations: usize) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::batched_matmul_16x128::artifact())?;
    let len = 16 * 128 * 128;
    let x_data = filled_data(len, 0.0005, 0.25);
    let y_data = filled_data(len, 0.0007, -0.75);
    let x = Tensor3::<f32, 16, 128, 128>::from_vec(x_data.clone())?;
    let y = Tensor3::<f32, 16, 128, 128>::from_vec(y_data.clone())?;
    let ndarray_x =
        Array3::from_shape_vec((16, 128, 128), x_data.clone()).expect("valid lhs batch shape");
    let ndarray_y =
        Array3::from_shape_vec((16, 128, 128), y_data.clone()).expect("valid rhs batch shape");

    for _ in 0..warmup {
        black_box(graphs::batched_matmul_16x128::run(
            &engine,
            x.clone(),
            y.clone(),
        )?);
    }

    let (knok_duration, knok_checksum) = measure(iterations, || {
        let output = graphs::batched_matmul_16x128::run(&engine, x.clone(), y.clone())?;
        Ok(checksum(output.as_slice()))
    })?;
    report(
        "knok batched matmul 16x128x128 @ 16x128x128",
        iterations,
        knok_duration,
        knok_checksum,
    );

    for _ in 0..warmup {
        black_box(ndarray_batched_matmul_checksum(&ndarray_x, &ndarray_y));
    }

    let (ndarray_duration, ndarray_checksum) = measure(iterations, || {
        Ok(ndarray_batched_matmul_checksum(&ndarray_x, &ndarray_y))
    })?;
    report(
        "ndarray batched matmul 16x128x128 @ 16x128x128",
        iterations,
        ndarray_duration,
        ndarray_checksum,
    );

    Ok(())
}

fn measure(
    mut iterations: usize,
    mut run: impl FnMut() -> knok::Result<f32>,
) -> knok::Result<(Duration, f32)> {
    iterations = iterations.max(1);
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        checksum = black_box(run()?);
    }
    Ok((start.elapsed(), checksum))
}

fn report(name: &str, iterations: usize, duration: Duration, checksum: f32) {
    let per_iter = duration.as_secs_f64() * 1_000.0 / iterations.max(1) as f64;
    println!("{name}: {per_iter:.3} ms/iter checksum={checksum:.6}");
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn filled_data(len: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|index| (index % 257) as f32 * scale + offset)
        .collect()
}

fn checksum(values: &[f32]) -> f32 {
    values.iter().copied().sum::<f32>()
}

fn ndarray_matmul_checksum(lhs: &Array2<f32>, rhs: &Array2<f32>) -> f32 {
    let output = lhs.dot(rhs);
    checksum(output.as_slice().expect("ndarray dot output is contiguous"))
}

fn ndarray_batched_matmul_checksum(lhs: &Array3<f32>, rhs: &Array3<f32>) -> f32 {
    let mut sum = 0.0;
    for b in 0..lhs.len_of(Axis(0)) {
        let output = lhs.index_axis(Axis(0), b).dot(&rhs.index_axis(Axis(0), b));
        sum += checksum(output.as_slice().expect("ndarray dot output is contiguous"));
    }
    sum
}
