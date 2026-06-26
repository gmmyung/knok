#[knok::graph(backends = [
    backend(Backend::LlvmCpu, driver = Driver::LocalTask),
    backend(Backend::LlvmCpu, driver = Driver::LocalTask),
])]
fn add4(
    x: knok::tensor::Tensor1<f32, 4>,
    y: knok::tensor::Tensor1<f32, 4>,
) -> knok::tensor::Tensor1<f32, 4> {
    x + y
}

fn main() {}
