use knok::prelude::*;

#[knok::graph(backend = "vulkan-spirv")]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

fn main() {}
