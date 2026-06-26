use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_reduction_arity(x: Tensor1<f32, 4>) -> Tensor0<i64> {
    argmin(x, x)
}

fn main() {}
