use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_outer_shape(
    x: Tensor4<f32, 1, 1, 1, 1>,
    y: Tensor3<f32, 1, 1, 1>,
) -> Tensor6<f32, 1, 1, 1, 1, 1, 1> {
    outer(x, y)
}

fn main() {}
