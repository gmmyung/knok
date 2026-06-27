use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn linspace_integer_step() -> Tensor1<i32, 4> {
    linspace::<Tensor1<i32, 4>>(0, 10)
}

fn main() {}
