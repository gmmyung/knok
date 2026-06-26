use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bool_add(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    x + y
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn bool_neg(x: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    -x
}

fn main() {}
