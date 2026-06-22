use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn bool_add(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    x + y
}

fn main() {}
