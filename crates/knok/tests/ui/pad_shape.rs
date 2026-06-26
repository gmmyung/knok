use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_pad(x: Tensor2<f32, 2, 2>) -> Tensor2<f32, 2, 3> {
    pad::<Tensor2<f32, 2, 3>, 1, 0>(x)
}

fn main() {}
