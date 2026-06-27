use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_permute(x: Tensor3<f32, 2, 3, 4>) -> Tensor3<f32, 2, 3, 4> {
    permute_dims::<0, 0, 2>(x)
}

fn main() {}
