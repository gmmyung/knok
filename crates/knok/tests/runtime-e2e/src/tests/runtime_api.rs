use knok::{tensor::*, Engine};

use crate::{graphs, mlir_models};

#[test]
fn generated_build_traced_affine_relu_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let output = graphs::affine_relu::call(x).unwrap();
    assert_eq!(output.as_slice(), &[8.0, 11.0, 16.0, 23.0]);
}

#[test]
fn reusable_engine_runs_multiple_generated_graphs() {
    let engine = Engine::for_artifact(graphs::affine_relu::artifact()).unwrap();

    let selected = graphs::elementwise_select::run(
        &engine,
        Tensor1::from_array([-1.0, 2.0, 7.0, 4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    assert_eq!(selected.as_slice(), &[0.0, 6.0, 6.0, 6.0]);

    let product = graphs::matmul_2x3_3x2::run(
        &engine,
        Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        Tensor2::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
    )
    .unwrap();
    assert_eq!(product.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn multi_output_graph_preserves_output_order_and_dtype() {
    let x = Tensor2::from_array([[1.0, 5.0, 3.0], [2.0, -4.0, 6.0]]);

    let (sum, argmax) = graphs::multi_output_stats::call(x).unwrap();

    assert_eq!(sum.as_slice(), &[9.0, 4.0]);
    assert_eq!(argmax.as_slice(), &[1, 2]);
}

#[test]
fn imported_mlir_models_run_through_generated_wrappers() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let sum = mlir_models::imported_add4::call(x.clone(), y.clone()).unwrap();
    assert_eq!(sum.as_slice(), &[11.0, 22.0, 33.0, 44.0]);

    let (sum, diff) = mlir_models::imported_add_sub4::call(x, y).unwrap();
    assert_eq!(sum.as_slice(), &[11.0, 22.0, 33.0, 44.0]);
    assert_eq!(diff.as_slice(), &[-9.0, -18.0, -27.0, -36.0]);
}

#[test]
fn run_into_reuses_single_output_storage() {
    let engine = Engine::for_artifact(graphs::affine_relu::artifact()).unwrap();
    let mut output = Tensor2::from_array([[0.0; 2]; 2]);
    let storage = output.as_slice().as_ptr();

    for _ in 0..3 {
        graphs::affine_relu::run_into(
            &engine,
            Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]),
            &mut output,
        )
        .unwrap();
        assert_eq!(output.as_slice(), &[8.0, 11.0, 16.0, 23.0]);
        assert_eq!(output.as_slice().as_ptr(), storage);
    }
}

#[test]
fn run_into_reuses_mixed_dtype_output_storage() {
    let engine = Engine::for_artifact(graphs::multi_output_stats::artifact()).unwrap();
    let mut outputs = (
        Tensor1::from_array([0.0; 2]),
        Tensor1::from_array([0_i64; 2]),
    );
    let storage = (outputs.0.as_slice().as_ptr(), outputs.1.as_slice().as_ptr());

    for _ in 0..3 {
        graphs::multi_output_stats::run_into(
            &engine,
            Tensor2::from_array([[1.0, 5.0, 3.0], [2.0, -4.0, 6.0]]),
            &mut outputs,
        )
        .unwrap();
        assert_eq!(outputs.0.as_slice(), &[9.0, 4.0]);
        assert_eq!(outputs.1.as_slice(), &[1, 2]);
        assert_eq!(outputs.0.as_slice().as_ptr(), storage.0);
        assert_eq!(outputs.1.as_slice().as_ptr(), storage.1);
    }
}

#[test]
fn run_into_supports_bool_outputs_and_no_inputs() {
    let engine = Engine::for_artifact(graphs::isnan_f32::artifact()).unwrap();
    let mut predicates = Tensor1::from_array([false; 4]);
    graphs::isnan_f32::run_into(
        &engine,
        Tensor1::from_array([f32::NAN, 0.0, 1.0, f32::NAN]),
        &mut predicates,
    )
    .unwrap();
    assert_eq!(predicates.as_slice(), &[true, false, false, true]);

    let mut output = Tensor1::from_array([0_i32; 4]);
    graphs::arange_step_i32::run_into(&engine, &mut output).unwrap();
    assert_eq!(output.as_slice(), &[0, 2, 4, 6]);
}

#[test]
fn run_into_checks_output_count_before_writing() {
    type Output = Tensor2<f32, 2, 2>;
    let artifact = graphs::affine_relu::artifact();
    let graph = knok::Graph::<Output, (Output, Output)>::new(artifact);
    let engine = Engine::for_artifact(artifact).unwrap();
    let mut outputs = (
        Output::from_array([[42.0; 2]; 2]),
        Output::from_array([[42.0; 2]; 2]),
    );
    let error = graph
        .run_into(&engine, Output::from_array([[1.0; 2]; 2]), &mut outputs)
        .unwrap_err();
    assert!(matches!(
        error,
        knok::Error::OutputCountMismatch {
            expected: 2,
            actual: 1
        }
    ));
    assert_eq!(outputs.0.as_slice(), &[42.0; 4]);
    assert_eq!(outputs.1.as_slice(), &[42.0; 4]);
}
