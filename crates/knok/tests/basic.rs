use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
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
