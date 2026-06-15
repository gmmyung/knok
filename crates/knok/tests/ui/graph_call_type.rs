use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn layer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn outer(x: Tensor1<f32, 5>) -> Tensor1<f32, 4> {
    layer(x)
}

fn main() {}

