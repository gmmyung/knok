use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn unsupported(x: Tensor1<u8, 4>) -> Tensor1<u8, 4> {
    x
}

fn main() {}
