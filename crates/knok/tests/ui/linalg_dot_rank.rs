use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_dot_rank(x: Tensor2<f32, 2, 2>, y: Tensor2<f32, 2, 2>) -> Tensor0<f32> {
    dot(x, y)
}

fn main() {}
