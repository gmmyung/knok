use knok::prelude::*;
use knok::runtime::raw;
use knok::{Engine, Error, RuntimeConfig};

knok::mlir_model! {
    name: imported_add4,
    path: "tests/fixtures/add4.mlir",
    backend: Backend::LlvmCpu,
    function: "imported.add4",
    inputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
    output: Tensor1<f32, 4>,
}

knok::mlir_model! {
    name: imported_add4_bundle,
    path: "tests/fixtures/add4.mlir",
    backends: [backend(Backend::LlvmCpu, driver = Driver::LocalTask)],
    function: "imported.add4",
    inputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
    output: Tensor1<f32, 4>,
}

knok::mlir_model! {
    name: imported_add4_i32,
    path: "tests/fixtures/add4_i32.mlir",
    backend: Backend::LlvmCpu,
    function: "imported.add4",
    inputs: [Tensor1<i32, 4>, Tensor1<i32, 4>],
    output: Tensor1<i32, 4>,
}

knok::mlir_model! {
    name: imported_add_sub4,
    path: "tests/fixtures/add_sub4.mlir",
    backend: Backend::LlvmCpu,
    function: "imported.add_sub4",
    inputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
    outputs: [Tensor1<f32, 4>, Tensor1<f32, 4>],
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

#[test]
fn imported_mlir_model_i32_runs() {
    let x = Tensor1::from_array([1i32, 2, 3, 4]);
    let y = Tensor1::from_array([10i32, 20, 30, 40]);

    let output = imported_add4_i32::invoke(x, y).unwrap();

    assert_eq!(output.into_vec(), vec![11i32, 22, 33, 44]);
}

#[test]
fn imported_mlir_model_multi_output_runs() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let (sum, diff) = imported_add_sub4::invoke_run(&engine, x, y).unwrap();

    assert_eq!(sum.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(diff.into_vec(), vec![-9.0, -18.0, -27.0, -36.0]);
}

#[test]
fn raw_runtime_multi_output_runs() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [10.0, 20.0, 30.0, 40.0];

    let outputs = engine
        .invoke(
            imported_add_sub4::artifact(),
            &[raw::Input::F32(&[4], &x), raw::Input::F32(&[4], &y)],
        )
        .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs.read::<f32>(0).unwrap(),
        vec![11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(
        outputs.read::<f32>(1).unwrap(),
        vec![-9.0, -18.0, -27.0, -36.0]
    );
}

#[test]
fn single_output_helper_rejects_multi_output_function() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [10.0, 20.0, 30.0, 40.0];

    let error = engine
        .invoke_one::<f32>(
            imported_add_sub4::artifact(),
            &[raw::Input::F32(&[4], &x), raw::Input::F32(&[4], &y)],
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::OutputCountMismatch {
            expected: 1,
            actual: 2
        }
    ));
}
