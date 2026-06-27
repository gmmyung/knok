use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn arange_dynamic(stop: Tensor0<i32>) -> Tensor1<i32, 4> {
    arange::<Tensor1<i32, 4>>(0, stop, 1)
}

fn main() {}
