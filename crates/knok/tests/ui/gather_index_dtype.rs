use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_gather_index_dtype(
    x: Tensor2<f32, 2, 3>,
    indices: Tensor1<f32, 2>,
) -> Tensor2<f32, 2, 3> {
    gather::<Tensor2<f32, 2, 3>, 0>(x, indices)
}

fn main() {}
