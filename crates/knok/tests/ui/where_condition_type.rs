use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn bad_where_condition(
    c: Tensor1<f32, 4>,
    x: Tensor1<f32, 4>,
    y: Tensor1<f32, 4>,
) -> Tensor1<f32, 4> {
    r#where(c, x, y)
}

fn main() {}
