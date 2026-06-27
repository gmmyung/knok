use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_diagonal_rect(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    diagonal(x)
}

fn main() {}
