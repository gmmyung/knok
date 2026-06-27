use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_reduction_dtype(x: Tensor1<i32, 4>) -> Tensor0<i32> {
    var(x)
}

fn main() {}
