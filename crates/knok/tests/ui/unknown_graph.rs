use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn outer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    missing_graph(x)
}

fn main() {}

