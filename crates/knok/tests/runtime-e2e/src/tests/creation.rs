use knok::tensor::*;

use crate::{common::*, graphs};

#[test]
fn no_input_creation_graph_runs() {
    let range = graphs::arange_step_i32::call().unwrap();
    let linspace = graphs::linspace_f32::call().unwrap();
    let eye = graphs::identity_f32::call().unwrap();

    assert_exact(range.as_slice(), &[0, 2, 4, 6]);
    assert_close(linspace.as_slice(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    assert_close(
        eye.as_slice(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
}

#[test]
fn additional_creation_ops_run() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0]);
    let to = graphs::arange_to_i32::call().unwrap();
    let range = graphs::arange_i32::call().unwrap();
    let zeros = graphs::zeros_like_f32::call(x.clone()).unwrap();
    let ones = graphs::ones_like_f32::call(x.clone()).unwrap();
    let full = graphs::full_like_f32::call(x).unwrap();

    assert_exact(to.as_slice(), &[0, 1, 2, 3]);
    assert_exact(range.as_slice(), &[2, 3, 4, 5]);
    assert_close(zeros.as_slice(), &[0.0, 0.0, 0.0]);
    assert_close(ones.as_slice(), &[1.0, 1.0, 1.0]);
    assert_close(full.as_slice(), &[3.5, 3.5, 3.5]);
}
