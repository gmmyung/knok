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

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn elementwise_1024(x: T2<f32, 1024, 1024>) -> T2<f32, 1024, 1024> {
    relu(exp(x.clone()) + tanh(x) * 0.5)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn sum_axis1_512(x: T2<f32, 512, 512>) -> T1<f32, 512> {
    sum_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn softmax_axis1_512x1024(x: T2<f32, 512, 1024>) -> T2<f32, 512, 1024> {
    softmax_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn transpose_512x256(x: T2<f32, 512, 256>) -> T2<f32, 256, 512> {
    transpose(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn broadcast_row_512(x: T1<f32, 512>) -> T2<f32, 512, 512> {
    broadcast(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mlp_64x128x256x64(
    x: T2<f32, 64, 128>,
    w1: T2<f32, 128, 256>,
    b1: T1<f32, 256>,
    w2: T2<f32, 256, 64>,
) -> T2<f32, 64, 64> {
    matmul(relu(matmul(x, w1) + b1), w2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn conv2d_nhwc_8x32x32x3(
    x: T4<f32, 8, 32, 32, 3>,
    k: T4<f32, 3, 3, 3, 16>,
) -> T4<f32, 8, 30, 30, 16> {
    conv2d(x, k)
}

fn main() {
    knok_build::compile_graphs!(
        tiny_relu,
        matmul_128,
        batched_matmul_16x128,
        elementwise_1024,
        sum_axis1_512,
        softmax_axis1_512x1024,
        transpose_512x256,
        broadcast_row_512,
        mlp_64x128x256x64,
        conv2d_nhwc_8x32x32x3
    );
}
