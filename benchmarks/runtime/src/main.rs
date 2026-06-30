use std::{
    env, fs,
    hint::black_box,
    path::Path,
    time::{Duration, Instant},
};

use knok::{prelude::*, Engine};
use ndarray::{Array1, Array2, Array3, Axis};

knok::generated_graphs!(pub mod graphs);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BenchConfig::from_env();
    let mut results = Vec::new();

    println!("backend=llvm-cpu driver=local-task");
    println!(
        "warmup={} samples={} output_dir={}",
        config.warmup,
        config.samples,
        config.output_dir.display()
    );

    bench_overhead(&config, &mut results)?;
    bench_matmul(&config, &mut results)?;
    bench_batched_matmul(&config, &mut results)?;
    bench_elementwise(&config, &mut results)?;
    bench_reductions(&config, &mut results)?;
    bench_softmax(&config, &mut results)?;
    bench_layout(&config, &mut results)?;
    bench_mlp(&config, &mut results)?;
    bench_conv2d(&config, &mut results)?;

    write_summary_files(&config.output_dir, &results)?;
    Ok(())
}

struct BenchConfig {
    warmup: usize,
    samples: usize,
    output_dir: std::path::PathBuf,
}

impl BenchConfig {
    fn from_env() -> Self {
        Self {
            warmup: env_usize("KNOK_BENCH_WARMUP", 3),
            samples: env_usize("KNOK_BENCH_SAMPLES", 20).max(1),
            output_dir: env::var_os("KNOK_BENCH_OUTPUT_DIR")
                .map(Into::into)
                .unwrap_or_else(|| "target".into()),
        }
    }
}

#[derive(Debug)]
struct BenchResult {
    name: &'static str,
    mean: Duration,
    median: Duration,
    min: Duration,
    max: Duration,
    checksum: f32,
}

fn bench_overhead(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::tiny_relu::artifact())?;
    let input = Tensor2::<f32, 2, 2>::from_array([[-1.0, 2.0], [3.0, -4.0]]);

    measure_case(config, results, "knok_tiny_relu_2x2", || {
        let output = graphs::tiny_relu::run(&engine, input.clone())?;
        Ok(checksum(output.as_slice()))
    })
}

fn bench_matmul(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::matmul_128::artifact())?;
    let x_data = filled_data(128 * 128, 0.001, 1.0);
    let y_data = filled_data(128 * 128, 0.002, -0.5);
    let x = Tensor2::<f32, 128, 128>::from_vec(x_data.clone())?;
    let y = Tensor2::<f32, 128, 128>::from_vec(y_data.clone())?;
    let ndarray_x = Array2::from_shape_vec((128, 128), x_data).expect("valid lhs shape");
    let ndarray_y = Array2::from_shape_vec((128, 128), y_data).expect("valid rhs shape");

    measure_case(config, results, "knok_matmul_128x128", || {
        let output = graphs::matmul_128::run(&engine, x.clone(), y.clone())?;
        Ok(checksum(output.as_slice()))
    })?;

    measure_case(config, results, "ndarray_matmul_128x128", || {
        let output = black_box(&ndarray_x).dot(black_box(&ndarray_y));
        Ok(checksum_iter(output.iter()))
    })
}

fn bench_batched_matmul(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::batched_matmul_16x128::artifact())?;
    let len = 16 * 128 * 128;
    let x_data = filled_data(len, 0.0005, 0.25);
    let y_data = filled_data(len, 0.0007, -0.75);
    let x = Tensor3::<f32, 16, 128, 128>::from_vec(x_data.clone())?;
    let y = Tensor3::<f32, 16, 128, 128>::from_vec(y_data.clone())?;
    let ndarray_x = Array3::from_shape_vec((16, 128, 128), x_data).expect("valid lhs shape");
    let ndarray_y = Array3::from_shape_vec((16, 128, 128), y_data).expect("valid rhs shape");

    measure_case(config, results, "knok_batched_matmul_16x128x128", || {
        let output = graphs::batched_matmul_16x128::run(&engine, x.clone(), y.clone())?;
        Ok(checksum(output.as_slice()))
    })?;

    measure_case(config, results, "ndarray_batched_matmul_16x128x128", || {
        let mut sum = 0.0;
        for batch in 0..16 {
            let output = black_box(&ndarray_x)
                .index_axis(Axis(0), batch)
                .dot(&black_box(&ndarray_y).index_axis(Axis(0), batch));
            sum += checksum_iter(output.iter());
        }
        Ok(sum)
    })
}

fn bench_elementwise(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::elementwise_1024::artifact())?;
    let data = filled_data(1024 * 1024, 0.0001, -0.25);
    let input = Tensor2::<f32, 1024, 1024>::from_vec(data.clone())?;
    let ndarray_input = Array2::from_shape_vec((1024, 1024), data).expect("valid input shape");

    measure_case(
        config,
        results,
        "knok_elementwise_exp_tanh_relu_1024x1024",
        || {
            let output = graphs::elementwise_1024::run(&engine, input.clone())?;
            Ok(checksum(output.as_slice()))
        },
    )?;

    measure_case(
        config,
        results,
        "ndarray_elementwise_exp_tanh_relu_1024x1024",
        || {
            let output = black_box(&ndarray_input).mapv(|value| {
                let value = value.exp() + value.tanh() * 0.5;
                value.max(0.0)
            });
            Ok(checksum_iter(output.iter()))
        },
    )
}

fn bench_reductions(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::sum_axis1_512::artifact())?;
    let data = filled_data(512 * 512, 0.001, 1.0);
    let input = Tensor2::<f32, 512, 512>::from_vec(data.clone())?;
    let ndarray_input = Array2::from_shape_vec((512, 512), data).expect("valid input shape");

    measure_case(config, results, "knok_sum_axis1_512x512", || {
        let output = graphs::sum_axis1_512::run(&engine, input.clone())?;
        Ok(checksum(output.as_slice()))
    })?;

    measure_case(config, results, "ndarray_sum_axis1_512x512", || {
        let output = black_box(&ndarray_input).sum_axis(Axis(1));
        Ok(checksum_iter(output.iter()))
    })
}

fn bench_softmax(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::softmax_axis1_512x1024::artifact())?;
    let data = filled_data(512 * 1024, 0.0001, -0.5);
    let input = Tensor2::<f32, 512, 1024>::from_vec(data.clone())?;
    let ndarray_input = Array2::from_shape_vec((512, 1024), data).expect("valid input shape");

    measure_case(config, results, "knok_softmax_axis1_512x1024", || {
        let output = graphs::softmax_axis1_512x1024::run(&engine, input.clone())?;
        Ok(checksum(output.as_slice()))
    })?;

    measure_case(config, results, "ndarray_softmax_axis1_512x1024", || {
        Ok(ndarray_softmax_axis1_checksum(&ndarray_input))
    })
}

fn bench_layout(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let transpose_engine = Engine::for_artifact(graphs::transpose_512x256::artifact())?;
    let broadcast_engine = Engine::for_artifact(graphs::broadcast_row_512::artifact())?;
    let matrix_data = filled_data(512 * 256, 0.001, 0.0);
    let row_data = filled_data(512, 0.01, 1.0);
    let matrix = Tensor2::<f32, 512, 256>::from_vec(matrix_data.clone())?;
    let row = Tensor1::<f32, 512>::from_vec(row_data.clone())?;
    let ndarray_matrix = Array2::from_shape_vec((512, 256), matrix_data).expect("valid shape");
    let ndarray_row = Array1::from_vec(row_data);

    measure_case(config, results, "knok_transpose_512x256", || {
        let output = graphs::transpose_512x256::run(&transpose_engine, matrix.clone())?;
        Ok(checksum(output.as_slice()))
    })?;

    measure_case(config, results, "ndarray_transpose_512x256", || {
        let output = black_box(&ndarray_matrix).t().to_owned();
        Ok(checksum_iter(output.iter()))
    })?;

    measure_case(config, results, "knok_broadcast_row_512_to_512x512", || {
        let output = graphs::broadcast_row_512::run(&broadcast_engine, row.clone())?;
        Ok(checksum(output.as_slice()))
    })?;

    measure_case(
        config,
        results,
        "ndarray_broadcast_row_512_to_512x512",
        || {
            let output = black_box(&ndarray_row)
                .broadcast((512, 512))
                .expect("valid broadcast")
                .to_owned();
            Ok(checksum_iter(output.iter()))
        },
    )
}

fn bench_mlp(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::mlp_64x128x256x64::artifact())?;
    let x_data = filled_data(64 * 128, 0.001, 0.5);
    let w1_data = filled_data(128 * 256, 0.0005, 0.01);
    let b1_data = filled_data(256, 0.001, 0.1);
    let w2_data = filled_data(256 * 64, 0.0007, 0.02);
    let x = Tensor2::<f32, 64, 128>::from_vec(x_data.clone())?;
    let w1 = Tensor2::<f32, 128, 256>::from_vec(w1_data.clone())?;
    let b1 = Tensor1::<f32, 256>::from_vec(b1_data.clone())?;
    let w2 = Tensor2::<f32, 256, 64>::from_vec(w2_data.clone())?;
    let ndarray_x = Array2::from_shape_vec((64, 128), x_data).expect("valid input shape");
    let ndarray_w1 = Array2::from_shape_vec((128, 256), w1_data).expect("valid w1 shape");
    let ndarray_b1 = Array1::from_vec(b1_data);
    let ndarray_w2 = Array2::from_shape_vec((256, 64), w2_data).expect("valid w2 shape");

    measure_case(config, results, "knok_mlp_64x128x256x64", || {
        let output =
            graphs::mlp_64x128x256x64::run(&engine, x.clone(), w1.clone(), b1.clone(), w2.clone())?;
        Ok(checksum(output.as_slice()))
    })?;

    measure_case(config, results, "ndarray_mlp_64x128x256x64", || {
        let mut hidden = black_box(&ndarray_x).dot(black_box(&ndarray_w1));
        hidden += black_box(&ndarray_b1);
        hidden.mapv_inplace(|value| value.max(0.0));
        let output = hidden.dot(black_box(&ndarray_w2));
        Ok(checksum_iter(output.iter()))
    })
}

fn bench_conv2d(config: &BenchConfig, results: &mut Vec<BenchResult>) -> knok::Result<()> {
    let engine = Engine::for_artifact(graphs::conv2d_nhwc_8x32x32x3::artifact())?;
    let x = Tensor4::<f32, 8, 32, 32, 3>::from_vec(filled_data(8 * 32 * 32 * 3, 0.001, 0.0))?;
    let k = Tensor4::<f32, 3, 3, 3, 16>::from_vec(filled_data(3 * 3 * 3 * 16, 0.01, -0.2))?;

    measure_case(
        config,
        results,
        "knok_conv2d_nhwc_8x32x32x3_hwcf_3x3x3x16",
        || {
            let output = graphs::conv2d_nhwc_8x32x32x3::run(&engine, x.clone(), k.clone())?;
            Ok(checksum(output.as_slice()))
        },
    )
}

fn measure_case(
    config: &BenchConfig,
    results: &mut Vec<BenchResult>,
    name: &'static str,
    mut run: impl FnMut() -> knok::Result<f32>,
) -> knok::Result<()> {
    for _ in 0..config.warmup {
        black_box(run()?);
    }

    let mut samples = Vec::with_capacity(config.samples);
    let mut final_checksum = 0.0;
    for _ in 0..config.samples {
        let start = Instant::now();
        final_checksum = black_box(run()?);
        samples.push(start.elapsed());
    }

    samples.sort_unstable();
    let total_nanos = samples.iter().map(Duration::as_nanos).sum::<u128>();
    let result = BenchResult {
        name,
        mean: Duration::from_nanos((total_nanos / samples.len() as u128) as u64),
        median: samples[samples.len() / 2],
        min: samples[0],
        max: samples[samples.len() - 1],
        checksum: final_checksum,
    };

    println!(
        "{:<52} median={:>9.3} us mean={:>9.3} us checksum={:.6}",
        result.name,
        duration_us(result.median),
        duration_us(result.mean),
        result.checksum
    );

    results.push(result);
    Ok(())
}

fn write_summary_files(
    output_dir: &Path,
    results: &[BenchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;
    let csv_path = output_dir.join("benchmark-summary.csv");
    let json_path = output_dir.join("benchmark-summary.json");

    let mut csv =
        String::from("benchmark,mean_ns,median_ns,min_ns,max_ns,mean_us,median_us,checksum\n");
    for result in results {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6}\n",
            result.name,
            result.mean.as_nanos(),
            result.median.as_nanos(),
            result.min.as_nanos(),
            result.max.as_nanos(),
            duration_us(result.mean),
            duration_us(result.median),
            result.checksum
        ));
    }
    fs::write(&csv_path, csv)?;

    let mut json = String::from("[\n");
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            json.push_str(",\n");
        }
        json.push_str(&format!(
            "  {{\"benchmark\":\"{}\",\"mean_ns\":{},\"median_ns\":{},\"min_ns\":{},\"max_ns\":{},\"mean_us\":{:.6},\"median_us\":{:.6},\"checksum\":{:.6}}}",
            result.name,
            result.mean.as_nanos(),
            result.median.as_nanos(),
            result.min.as_nanos(),
            result.max.as_nanos(),
            duration_us(result.mean),
            duration_us(result.median),
            result.checksum
        ));
    }
    json.push_str("\n]\n");
    fs::write(&json_path, json)?;

    println!("wrote {}", csv_path.display());
    println!("wrote {}", json_path.display());
    Ok(())
}

fn duration_us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
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

fn checksum_iter<'a>(values: impl IntoIterator<Item = &'a f32>) -> f32 {
    values.into_iter().copied().sum::<f32>()
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
