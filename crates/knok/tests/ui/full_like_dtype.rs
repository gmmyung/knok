use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn full_like_dtype(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    full_like(x, 1i32)
}

fn main() {}
