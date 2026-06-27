use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_repeat(x: Tensor2<f32, 2, 2>) -> Tensor2<f32, 2, 2> {
    repeat::<2, 2>(x)
}

fn main() {}
