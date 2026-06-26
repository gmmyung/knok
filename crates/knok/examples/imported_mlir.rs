use knok::prelude::*;
use knok::{Engine, RuntimeConfig};

knok::mlir_model! {
    name: imported_add4,
    path: "tests/fixtures/add4.mlir",
    backend: Backend::LlvmCpu,
    function: "imported.add4",
    inputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
    output: Tensor1<f32, 4>,
}

fn main() -> knok::Result<()> {
    let engine = Engine::new(RuntimeConfig::auto())?;
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let output = imported_add4::invoke_run(&engine, x, y)?;

    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    Ok(())
}
