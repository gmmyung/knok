use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_take_along_axis_rank(
    x: Tensor2<f32, 2, 3>,
    indices: Tensor1<i64, 2>,
) -> Tensor1<f32, 2> {
    take_along_axis::<1>(x, indices)
}

fn main() {}
