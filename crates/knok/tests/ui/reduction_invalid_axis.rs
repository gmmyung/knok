use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_reduction_axis(x: Tensor2<f32, 2, 3>) -> Tensor0<f32> {
    max::<2>(x)
}

fn main() {}
