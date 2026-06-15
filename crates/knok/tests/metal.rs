#![cfg(target_os = "macos")]

use knok::prelude::*;

#[knok::graph(backend = "metal-spirv")]
fn add4_metal(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[test]
fn metal_add_graph_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);
    let output = add4_metal(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}
