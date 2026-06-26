use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bool_greater(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    greater(x, y)
}

fn main() {}
