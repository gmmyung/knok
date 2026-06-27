use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_take_along_axis_shape(
    x: Tensor2<f32, 2, 3>,
    indices: Tensor2<i64, 3, 2>,
) -> Tensor2<f32, 3, 2> {
    take_along_axis::<1>(x, indices)
}

fn main() {}
