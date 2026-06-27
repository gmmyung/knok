use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn linspace_rank() -> Tensor2<f32, 2, 2> {
    linspace::<Tensor2<f32, 2, 2>>(0.0, 1.0)
}

fn main() {}
