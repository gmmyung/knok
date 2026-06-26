use knok::prelude::*;

#[knok::graph(backends = [
    backend(Backend::LlvmCpu, driver = Backend::LlvmCpu),
])]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

fn main() {}
