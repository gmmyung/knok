use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn arange_shape() -> Tensor1<i32, 3> {
    arange::<Tensor1<i32, 3>>(0, 8, 2)
}

fn main() {}
