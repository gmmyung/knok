use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn arange_bool() -> Tensor1<bool, 4> {
    arange::<Tensor1<bool, 4>>(0, 4)
}

fn main() {}
