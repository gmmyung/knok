use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_split(x: Tensor2<f32, 2, 4>) -> (Tensor2<f32, 2, 1>, Tensor2<f32, 2, 2>) {
    let (a, b) = split::<1, 1, 2>(x);
    (a, b)
}

fn main() {}
