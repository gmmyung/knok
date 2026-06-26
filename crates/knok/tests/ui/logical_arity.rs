use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_logical_arity(x: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    logical_and(x)
}

fn main() {}
