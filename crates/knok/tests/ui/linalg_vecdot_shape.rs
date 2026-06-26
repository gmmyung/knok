use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_vecdot_shape(x: Tensor2<f32, 2, 3>, y: Tensor2<f32, 3, 3>) -> Tensor1<f32, 2> {
    vecdot(x, y)
}

fn main() {}
