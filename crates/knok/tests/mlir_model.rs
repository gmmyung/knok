use knok::prelude::*;
use knok::{Engine, RuntimeConfig};

knok::mlir_model! {
    name: imported_add4,
    path: "tests/fixtures/add4.mlir",
    backend: "llvm-cpu",
    function: "imported.add4",
    inputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
    output: Tensor1<f32, 4>,
}

knok::mlir_model! {
    name: imported_add4_bundle,
    path: "tests/fixtures/add4.mlir",
    backends: [backend("llvm-cpu", driver = "local-task")],
    function: "imported.add4",
    inputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
    output: Tensor1<f32, 4>,
}

#[test]
fn imported_mlir_model_untyped_wrapper_runs() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [10.0, 20.0, 30.0, 40.0];
    let output = imported_add4::invoke_f32(&[(&[4], &x), (&[4], &y)]).unwrap();
    assert_eq!(output, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn imported_mlir_model_typed_wrapper_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let output = imported_add4::invoke(x, y).unwrap();

    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn imported_mlir_model_engine_wrapper_runs() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let output = imported_add4::invoke_run(&engine, x, y).unwrap();

    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn imported_mlir_model_backend_bundle_runs() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let output = imported_add4_bundle::invoke_run(&engine, x, y).unwrap();

    assert_eq!(imported_add4_bundle::artifact().variants.len(), 1);
    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}
