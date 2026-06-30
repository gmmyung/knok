use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn tiny_relu(x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
    relu(x + 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn matmul_128(x: T2<f32, 128, 128>, y: T2<f32, 128, 128>) -> T2<f32, 128, 128> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn batched_matmul_16x128(
    x: T3<f32, 16, 128, 128>,
    y: T3<f32, 16, 128, 128>,
) -> T3<f32, 16, 128, 128> {
    matmul(x, y)
}

fn main() {
    knok_build::compile_graphs!(tiny_relu, matmul_128, batched_matmul_16x128);
}
