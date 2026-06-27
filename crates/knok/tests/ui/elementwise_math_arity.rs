use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_elementwise_math_arity(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    sin(x, x)
}

fn main() {}
