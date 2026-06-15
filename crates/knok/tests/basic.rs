use knok::prelude::*;
use knok::{Engine, Error, GraphArtifact, GraphArtifactVariant, RuntimeConfig};

#[knok::graph(backend = "llvm-cpu")]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[knok::graph(backends = [backend("llvm-cpu", driver = "local-task")])]
fn add4_bundle(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[knok::graph(backend = "llvm-cpu")]
fn arithmetic4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    ((x - y) * 2.0) / 4.0
}

#[knok::graph(backend = "llvm-cpu")]
fn neg4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    -x
}

#[knok::graph(backend = "llvm-cpu")]
fn let_chain4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    let sum = x + y;
    let shifted = sum - 1.0;
    relu(shifted)
}

#[knok::graph(backend = "llvm-cpu")]
fn relu4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn mm2x2(x: Tensor2<f32, 2, 2>, y: Tensor2<f32, 2, 2>) -> Tensor2<f32, 2, 2> {
    matmul(x, y)
}

#[knok::graph(backend = "llvm-cpu")]
fn transpose2x3(x: Tensor2<f32, 2, 3>) -> Tensor2<f32, 3, 2> {
    transpose(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn reshape2x2(x: Tensor1<f32, 4>) -> Tensor2<f32, 2, 2> {
    reshape::<Tensor2<f32, 2, 2>>(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn flatten2x2(x: Tensor2<f32, 2, 2>) -> Tensor1<f32, 4> {
    reshape::<Tensor1<f32, 4>>(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn broadcast1to4(x: Tensor1<f32, 1>) -> Tensor1<f32, 4> {
    broadcast::<Tensor1<f32, 4>>(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn sum4(x: Tensor1<f32, 4>) -> Tensor1<f32, 1> {
    sum(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn sum2x2(x: Tensor2<f32, 2, 2>) -> Tensor1<f32, 1> {
    sum(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn reshape1_to_3d(x: Tensor1<f32, 8>) -> Tensor3<f32, 2, 2, 2> {
    reshape::<Tensor3<f32, 2, 2, 2>>(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn flatten4d(x: Tensor4<f32, 1, 2, 2, 1>) -> Tensor1<f32, 4> {
    reshape::<Tensor1<f32, 4>>(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn exp4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    exp(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn log4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    log(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn sqrt4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    sqrt(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn tanh4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    tanh(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn sigmoid4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    sigmoid(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn softmax4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    softmax(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn mean4(x: Tensor1<f32, 4>) -> Tensor1<f32, 1> {
    mean(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn argmax4(x: Tensor1<f32, 4>) -> Tensor1<f32, 1> {
    argmax(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn batch_mm(x: Tensor3<f32, 1, 2, 3>, y: Tensor3<f32, 1, 3, 2>) -> Tensor3<f32, 1, 2, 2> {
    matmul(x, y)
}

#[knok::graph(backend = "llvm-cpu")]
fn conv2d_1x1(
    x: Tensor4<f32, 1, 2, 2, 1>,
    k: Tensor4<f32, 1, 1, 1, 1>,
) -> Tensor4<f32, 1, 2, 2, 1> {
    conv2d(x, k)
}

#[knok::graph(backend = "llvm-cpu")]
fn layer4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x)
}

#[knok::graph(backend = "llvm-cpu")]
fn composed4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    layer4(x + y)
}

#[knok::graph(backend = "llvm-cpu")]
fn composed_twice4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    let first = composed4(x, y);
    layer4(first - 2.0)
}

#[test]
fn add_graph_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);
    let output = add4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn artifact_records_backend_variant() {
    let artifact = add4_artifact();
    assert_eq!(artifact.function_name, "knok.add4");
    assert_eq!(artifact.variants.len(), 1);
    let variant = artifact.first_variant().unwrap();
    assert_eq!(variant.backend, "llvm-cpu");
    assert_eq!(variant.driver, "local-task");
    assert!(variant
        .compile_flags
        .contains(&"--iree-hal-target-backends=llvm-cpu"));
}

#[test]
fn explicit_backend_bundle_syntax_runs() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let output = add4_bundle_run(
        &engine,
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();

    assert_eq!(add4_bundle_artifact().variants.len(), 1);
    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn reusable_engine_runs_graph_repeatedly() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();

    let first = add4_run(
        &engine,
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    let second = add4_run(
        &engine,
        Tensor1::from_array([5.0, 6.0, 7.0, 8.0]),
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();

    assert_eq!(engine.driver_name(), "local-task");
    assert_eq!(first.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(second.into_vec(), vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn engine_reports_missing_artifact_variant_for_driver() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let artifact = add4_artifact();
    let variant = artifact.first_variant().unwrap();
    let variants = Box::leak(Box::new([GraphArtifactVariant {
        backend: "metal-spirv",
        driver: "metal",
        ..variant
    }]));
    let artifact = GraphArtifact {
        variants,
        ..artifact
    };
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [10.0, 20.0, 30.0, 40.0];
    let error = engine
        .invoke_f32(artifact, &[(&[4], &x), (&[4], &y)])
        .unwrap_err();

    assert!(matches!(
        error,
        Error::MissingArtifactVariant {
            function_name: "knok.add4",
            ..
        }
    ));
}

#[test]
fn arithmetic_graph_runs() {
    let x = Tensor1::from_array([9.0, 10.0, 11.0, 12.0]);
    let y = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let output = arithmetic4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn unary_negation_graph_runs() {
    let x = Tensor1::from_array([1.0, -2.0, 3.5, -4.5]);
    let output = neg4(x).unwrap();
    assert_eq!(output.into_vec(), vec![-1.0, 2.0, -3.5, 4.5]);
}

#[test]
fn let_chain_and_scalar_broadcast_run() {
    let x = Tensor1::from_array([1.0, 2.0, -3.0, 4.0]);
    let y = Tensor1::from_array([1.0, -10.0, 10.0, -2.0]);
    let output = let_chain4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![1.0, 0.0, 6.0, 1.0]);
}

#[test]
fn relu_graph_runs() {
    let x = Tensor1::from_array([-1.0, 2.0, -3.0, 4.0]);
    let output = relu4(x).unwrap();
    assert_eq!(output.into_vec(), vec![0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn matmul_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let y = Tensor2::from_array([[5.0, 6.0], [7.0, 8.0]]);
    let output = mm2x2(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn transpose_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let output = transpose2x3(x).unwrap();
    assert_eq!(output.into_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn reshape_graph_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let output = reshape2x2(x).unwrap();
    assert_eq!(output.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn flatten_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let output = flatten2x2(x).unwrap();
    assert_eq!(output.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn broadcast_graph_runs() {
    let x = Tensor1::from_array([7.0]);
    let output = broadcast1to4(x).unwrap();
    assert_eq!(output.into_vec(), vec![7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn sum_graph_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let output = sum4(x).unwrap();
    assert_eq!(output.into_vec(), vec![10.0]);
}

#[test]
fn rank2_sum_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let output = sum2x2(x).unwrap();
    assert_eq!(output.into_vec(), vec![10.0]);
}

#[test]
fn rank3_and_rank4_tensors_run() {
    let rank3 = reshape1_to_3d(Tensor1::from_array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ]))
    .unwrap();
    assert_eq!(
        rank3.into_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );

    let rank4 = Tensor4::from_array([[[[1.0], [2.0]], [[3.0], [4.0]]]]);
    let flat = flatten4d(rank4).unwrap();
    assert_eq!(flat.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn math_op_graphs_run() {
    let exp_output = exp4(Tensor1::from_array([0.0, 1.0, 2.0, 3.0])).unwrap();
    assert_close(
        &exp_output.into_vec(),
        &[1.0, core::f32::consts::E, 7.389056, 20.085537],
    );

    let log_output = log4(Tensor1::from_array([1.0, core::f32::consts::E, 4.0, 8.0])).unwrap();
    assert_close(&log_output.into_vec(), &[0.0, 1.0, 1.3862944, 2.0794415]);

    let sqrt_output = sqrt4(Tensor1::from_array([1.0, 4.0, 9.0, 16.0])).unwrap();
    assert_close(&sqrt_output.into_vec(), &[1.0, 2.0, 3.0, 4.0]);

    let tanh_output = tanh4(Tensor1::from_array([0.0, 1.0, -1.0, 2.0])).unwrap();
    assert_close(
        &tanh_output.into_vec(),
        &[0.0, 0.7615942, -0.7615942, 0.9640276],
    );

    let sigmoid_output = sigmoid4(Tensor1::from_array([0.0, 2.0, -2.0, 4.0])).unwrap();
    assert_close(
        &sigmoid_output.into_vec(),
        &[0.5, 0.880797, 0.11920292, 0.98201376],
    );
}

#[test]
fn reduction_and_classifier_op_graphs_run() {
    let softmax_output = softmax4(Tensor1::from_array([1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_close(
        &softmax_output.into_vec(),
        &[0.032058604, 0.08714432, 0.23688284, 0.6439143],
    );

    let mean_output = mean4(Tensor1::from_array([1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_close(&mean_output.into_vec(), &[2.5]);

    let argmax_output = argmax4(Tensor1::from_array([1.0, 10.0, 3.0, 4.0])).unwrap();
    assert_eq!(argmax_output.into_vec(), vec![1.0]);
}

#[test]
fn batched_matmul_and_conv2d_graphs_run() {
    let lhs = Tensor3::from_array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
    let rhs = Tensor3::from_array([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]);
    let mm = batch_mm(lhs, rhs).unwrap();
    assert_eq!(mm.into_vec(), vec![58.0, 64.0, 139.0, 154.0]);

    let image = Tensor4::from_array([[[[1.0], [2.0]], [[3.0], [4.0]]]]);
    let kernel = Tensor4::from_array([[[[2.0]]]]);
    let conv = conv2d_1x1(image, kernel).unwrap();
    assert_eq!(conv.into_vec(), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn graph_calls_are_inlined() {
    let x = Tensor1::from_array([-1.0, 2.0, -3.0, 4.0]);
    let y = Tensor1::from_array([0.5, 1.0, 10.0, -10.0]);
    let output = composed4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![0.0, 3.0, 7.0, 0.0]);
}

#[test]
fn nested_graph_calls_are_inlined() {
    let x = Tensor1::from_array([-1.0, 2.0, -3.0, 4.0]);
    let y = Tensor1::from_array([0.5, 1.0, 10.0, -10.0]);
    let output = composed_twice4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![0.0, 1.0, 5.0, 0.0]);
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-4,
            "expected {expected}, got {actual}"
        );
    }
}
