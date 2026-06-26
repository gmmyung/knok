use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_gather_shape(
    x: Tensor2<f32, 2, 3>,
    indices: Tensor1<i64, 2>,
) -> Tensor2<f32, 3, 2> {
    gather::<Tensor2<f32, 3, 2>, 1>(x, indices)
}

fn main() {}
