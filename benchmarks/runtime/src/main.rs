use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use knok::{prelude::*, Engine};

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

    let (rust_duration, rust_checksum) = measure(iterations, || {
        let output = rust_matmul(&x_data, &y_data, 128, 128, 128);
        Ok(checksum(&output))
    })?;
    report(
        "rust matmul 128x128 @ 128x128",
        iterations,
        rust_duration,
        rust_checksum,
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

    let (rust_duration, rust_checksum) = measure(iterations, || {
        let output = rust_batched_matmul(&x_data, &y_data, 16, 128, 128, 128);
        Ok(checksum(&output))
    })?;
    report(
        "rust batched matmul 16x128x128 @ 16x128x128",
        iterations,
        rust_duration,
        rust_checksum,
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

fn rust_matmul(x: &[f32], y: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += x[row * k + inner] * y[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    out
}

fn rust_batched_matmul(
    x: &[f32],
    y: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; batch * m * n];
    let lhs_stride = m * k;
    let rhs_stride = k * n;
    let out_stride = m * n;
    for b in 0..batch {
        let lhs = &x[b * lhs_stride..][..lhs_stride];
        let rhs = &y[b * rhs_stride..][..rhs_stride];
        let dst = &mut out[b * out_stride..][..out_stride];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0;
                for inner in 0..k {
                    sum += lhs[row * k + inner] * rhs[inner * n + col];
                }
                dst[row * n + col] = sum;
            }
        }
    }
    out
}
