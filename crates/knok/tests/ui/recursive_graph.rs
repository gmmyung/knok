use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn recurse(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    recurse(x)
}

fn main() {}

