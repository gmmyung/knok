use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn integer_relu(x: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    relu(x)
}

fn main() {}
