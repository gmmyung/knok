use knok::prelude::*;

#[knok::graph(backend = Backend::LlvmCpu)]
fn bad_trace_rect(x: Tensor2<f32, 2, 3>) -> Tensor0<f32> {
    trace(x)
}

fn main() {}
