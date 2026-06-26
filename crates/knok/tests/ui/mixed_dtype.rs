use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn mixed(x: Tensor1<f32, 4>, y: Tensor1<f64, 4>) -> Tensor1<f32, 4> {
    x + y
}

fn main() {}
