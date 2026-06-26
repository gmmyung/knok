use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_trace_axes(x: Tensor2<f32, 2, 2>) -> Tensor0<f32> {
    trace::<1, 1>(x)
}

fn main() {}
