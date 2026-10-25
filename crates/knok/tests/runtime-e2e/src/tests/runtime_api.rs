use knok::{tensor::*, Engine};

use crate::{graphs, mlir_models};

#[test]
fn generated_build_traced_affine_relu_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let output = graphs::affine_relu::call(&x).unwrap();
    assert_eq!(output.as_slice(), &[8.0, 11.0, 16.0, 23.0]);
}

#[test]
fn reusable_engine_runs_multiple_generated_graphs() {
    let engine = Engine::for_artifact(graphs::affine_relu::artifact()).unwrap();

    let selected = graphs::elementwise_select::run(
        &engine,
        &Tensor1::from_array([-1.0, 2.0, 7.0, 4.0]),
        &Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    assert_eq!(selected.as_slice(), &[0.0, 6.0, 6.0, 6.0]);

    let product = graphs::matmul_2x3_3x2::run(
        &engine,
        &Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &Tensor2::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
    )
    .unwrap();
    assert_eq!(product.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn multi_output_graph_preserves_output_order_and_dtype() {
    let x = Tensor2::from_array([[1.0, 5.0, 3.0], [2.0, -4.0, 6.0]]);

    let (sum, argmax) = graphs::multi_output_stats::call(&x).unwrap();

    assert_eq!(sum.as_slice(), &[9.0, 4.0]);
    assert_eq!(argmax.as_slice(), &[1, 2]);
}

#[test]
fn imported_mlir_models_run_through_generated_wrappers() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let sum = mlir_models::imported_add4::call(&x, &y).unwrap();
    assert_eq!(sum.as_slice(), &[11.0, 22.0, 33.0, 44.0]);

    let (sum, diff) = mlir_models::imported_add_sub4::call(&x, &y).unwrap();
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
            &Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]),
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
            &Tensor2::from_array([[1.0, 5.0, 3.0], [2.0, -4.0, 6.0]]),
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
        &Tensor1::from_array([f32::NAN, 0.0, 1.0, f32::NAN]),
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
    let engine = Engine::for_artifact(mlir_models::wrong_count::artifact()).unwrap();
    let input = Tensor1::from_array([1.0; 4]);
    let mut output = Tensor1::from_array([42.0; 4]);
    let error =
        mlir_models::wrong_count::run_into(&engine, &input, &input, &mut output).unwrap_err();
    assert!(matches!(
        error,
        knok::Error::OutputCountMismatch {
            expected: 1,
            actual: 2
        }
    ));
    assert_eq!(output.as_slice(), &[42.0; 4]);
}

#[test]
fn prepared_models_reuse_borrowed_inputs_across_compatible_cpu_drivers() {
    let input = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let input_storage = input.as_slice().as_ptr();
    for driver in [knok::Driver::LocalTask, knok::Driver::LocalSync] {
        let engine = Engine::new(knok::RuntimeConfig::driver(driver).device_ordinal(0)).unwrap();
        let model = graphs::affine_relu::load(&engine).unwrap();
        let mut output = Tensor2::from_array([[0.0; 2]; 2]);
        let output_storage = output.as_slice().as_ptr();
        for _ in 0..3 {
            model.run_into(&input, &mut output).unwrap();
            assert_eq!(output.as_slice(), &[8.0, 11.0, 16.0, 23.0]);
            assert_eq!(output.as_slice().as_ptr(), output_storage);
            assert_eq!(model.run(&input).unwrap().as_slice(), output.as_slice());
        }
        assert_eq!(
            model.artifact().function_name(),
            graphs::affine_relu::artifact().function_name()
        );
    }
    assert_eq!(input.as_slice().as_ptr(), input_storage);
    assert_eq!(input.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn prepared_models_handle_tuples_and_zero_inputs() {
    let engine = Engine::for_artifact(mlir_models::imported_add4::artifact()).unwrap();
    let model = mlir_models::imported_add_sub4::load(&engine).unwrap();
    let x = Tensor1::from_array([1.0; 4]);
    let y = Tensor1::from_array([2.0; 4]);
    let (sum, difference) = model.run((&x, &y)).unwrap();
    assert_eq!(sum.as_slice(), &[3.0; 4]);
    assert_eq!(difference.as_slice(), &[-1.0; 4]);
    let model = graphs::arange_step_i32::load(&engine).unwrap();
    assert_eq!(model.run(()).unwrap().as_slice(), &[0, 2, 4, 6]);
}

#[test]
fn imported_output_shape_is_checked_even_when_element_counts_match() {
    let engine = Engine::for_artifact(mlir_models::wrong_shape::artifact()).unwrap();
    let input = Tensor1::from_array([1.0; 4]);
    let model = mlir_models::wrong_shape::load(&engine).unwrap();
    assert!(
        matches!(model.run((&input, &input)), Err(knok::Error::OutputShapeMismatch { index: 0, expected: &[2, 2], ref actual }) if actual == &[4])
    );
    let mut output = Tensor2::from_array([[42.0; 2]; 2]);
    assert!(model.run_into((&input, &input), &mut output).is_err());
    assert_eq!(output.as_slice(), &[42.0; 4]);
}

#[test]
fn later_output_mismatches_leave_all_host_outputs_untouched() {
    let engine = Engine::for_artifact(mlir_models::wrong_second_shape::artifact()).unwrap();
    let input = Tensor1::from_array([1.0; 4]);
    let mut outputs = (
        Tensor1::from_array([42.0; 4]),
        Tensor2::from_array([[42.0; 2]; 2]),
    );
    let error = mlir_models::wrong_second_shape::run_into(&engine, &input, &input, &mut outputs)
        .unwrap_err();
    assert!(matches!(
        error,
        knok::Error::OutputShapeMismatch { index: 1, .. }
    ));
    assert_eq!(outputs.0.as_slice(), &[42.0; 4]);
    assert_eq!(outputs.1.as_slice(), &[42.0; 4]);
    let mut outputs = (
        Tensor1::from_array([42.0; 4]),
        Tensor1::from_array([42_i32; 4]),
    );
    assert!(
        mlir_models::wrong_second_dtype::run_into(&engine, &input, &input, &mut outputs).is_err()
    );
    assert_eq!(outputs.0.as_slice(), &[42.0; 4]);
    assert_eq!(outputs.1.as_slice(), &[42; 4]);
}

#[test]
fn explicit_compiler_tuning_is_applied_and_recorded() {
    use crate::{tuned, tuned_grad};
    let artifact = tuned::affine_relu::artifact();
    let flags = artifact.first_variant().unwrap().compile_flags();
    assert!(flags.contains(&"--iree-llvmcpu-target-cpu=generic"));
    assert!(flags.contains(&"--iree-llvmcpu-target-cpu-features="));
    assert!(flags.contains(&"--iree-opt-level=O1"));
    let input = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(
        tuned::affine_relu::call(&input).unwrap().as_slice(),
        &[8.0, 11.0, 16.0, 23.0]
    );
    let engine = Engine::for_artifact(artifact).unwrap();
    let model = tuned_grad::sum_square_loss_value_and_grad::load(&engine).unwrap();
    assert!(model
        .artifact()
        .first_variant()
        .unwrap()
        .compile_flags()
        .contains(&"--iree-llvmcpu-target-cpu=generic"));
    let input = Tensor1::from_array([1.0, 2.0, -3.0]);
    let (loss, gradient) = model.run(&input).unwrap();
    assert_eq!(loss.as_slice(), &[14.0]);
    assert_eq!(gradient.as_slice(), &[2.0, 4.0, -6.0]);
    assert!(mlir_models::imported_add4::artifact()
        .first_variant()
        .unwrap()
        .compile_flags()
        .contains(&"--iree-llvmcpu-target-cpu=generic"));
}
