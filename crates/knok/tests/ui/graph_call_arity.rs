use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn layer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn outer(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    layer(x, y)
}

fn main() {}

