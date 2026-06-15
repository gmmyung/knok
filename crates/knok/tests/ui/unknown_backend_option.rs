#[knok::graph(backends = [
    backend("llvm-cpu", device = "default"),
])]
fn add4(
    x: knok::tensor::Tensor1<f32, 4>,
    y: knok::tensor::Tensor1<f32, 4>,
) -> knok::tensor::Tensor1<f32, 4> {
    x + y
}

fn main() {}
