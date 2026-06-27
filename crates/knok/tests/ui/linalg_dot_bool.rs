use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_dot_bool(x: Tensor1<bool, 2>, y: Tensor1<bool, 2>) -> Tensor0<bool> {
    dot(x, y)
}

fn main() {}
