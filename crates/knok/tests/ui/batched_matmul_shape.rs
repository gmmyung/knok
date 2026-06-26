use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn bad_batch_mm(
    x: Tensor3<f32, 2, 2, 3>,
    y: Tensor3<f32, 3, 3, 2>,
) -> Tensor3<f32, 2, 2, 2> {
    matmul(x, y)
}

fn main() {}
