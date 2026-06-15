knok::mlir_model! {
    name: imported_add4,
    path: "../../../../crates/knok/tests/fixtures/add4.mlir",
    backend: "llvm-cpu",
    function: "imported.add4",
    inputs: [knok::tensor::Tensor1<f32, 5>, knok::tensor::Tensor1<f32, 4>],
    output: knok::tensor::Tensor1<f32, 4>,
}

fn main() {}
