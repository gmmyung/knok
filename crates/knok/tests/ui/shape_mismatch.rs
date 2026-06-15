use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn add_bad(x: Tensor1<f32, 4>, y: Tensor1<f32, 5>) -> Tensor1<f32, 4> {
    x + y
}

fn main() {}

