use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn integer_relu(x: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    relu(x)
}

fn main() {}
