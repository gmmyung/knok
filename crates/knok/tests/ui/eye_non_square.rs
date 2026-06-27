use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn eye_non_square() -> Tensor2<f32, 2, 3> {
    eye::<Tensor2<f32, 2, 3>>()
}

fn main() {}
