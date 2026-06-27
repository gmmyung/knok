use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn full_like_shape(x: Tensor1<f32, 4>, fill: Tensor1<f32, 1>) -> Tensor1<f32, 4> {
    full_like(x, fill)
}

fn main() {}
