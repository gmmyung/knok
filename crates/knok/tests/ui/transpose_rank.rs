use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn bad_transpose(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    transpose(x)
}

fn main() {}

