use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn integer_floor(x: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    floor(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn bool_square(x: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    square(x)
}

fn main() {}
