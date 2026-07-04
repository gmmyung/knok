use knok::tensor::*;

use crate::{common::*, graphs};

#[test]
fn no_input_creation_graph_runs() {
    let range = graphs::arange_step_i32::call().unwrap();
    let linspace = graphs::linspace_f32::call().unwrap();
    let identity = graphs::identity_f32::call().unwrap();
    let eye = graphs::eye_f32::call().unwrap();

    assert_exact(range.as_slice(), &[0, 2, 4, 6]);
    assert_close(linspace.as_slice(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    assert_close(
        identity.as_slice(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
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

#[test]
fn random_creation_ops_run() {
    let static_uniform = graphs::static_uniform_f32::call().unwrap();
    let static_uniform_edge = graphs::static_uniform_f32_edge::call().unwrap();
    let static_ints = graphs::static_randint_i32::call().unwrap();

    assert!(
        static_uniform
            .as_slice()
            .iter()
            .all(|value| (-1.0..1.0).contains(value)),
        "static uniform values were outside [-1, 1): {:?}",
        static_uniform.as_slice()
    );
    assert!(
        static_uniform_edge.as_slice()[0] < 1.0,
        "static uniform rounded up to high: {:?}",
        static_uniform_edge.as_slice()
    );
    assert!(
        static_ints
            .as_slice()
            .iter()
            .all(|value| (2..5).contains(value)),
        "static randint values were outside [2, 5): {:?}",
        static_ints.as_slice()
    );

    let key = Tensor0::from_scalar(123_i64);
    let (uniform, next_key) = graphs::runtime_uniform_f32::call(key.clone()).unwrap();
    let (same_uniform, same_next_key) = graphs::runtime_uniform_f32::call(key).unwrap();

    assert_close(uniform.as_slice(), same_uniform.as_slice());
    assert_exact(next_key.as_slice(), &[127_i64]);
    assert_exact(same_next_key.as_slice(), &[127_i64]);
    assert!(
        uniform
            .as_slice()
            .iter()
            .all(|value| (0.0..1.0).contains(value)),
        "runtime uniform values were outside [0, 1): {:?}",
        uniform.as_slice()
    );

    let (advanced_uniform, _) = graphs::runtime_uniform_f32::call(next_key).unwrap();
    assert_ne!(uniform.as_slice(), advanced_uniform.as_slice());

    let (subnormal_uniform, next_key) =
        graphs::runtime_uniform_f32_subnormal::call(Tensor0::from_scalar(0_i64)).unwrap();
    assert_exact(next_key.as_slice(), &[8_i64]);
    assert_exact(subnormal_uniform.as_slice(), &[0.0_f32; 8]);

    let (ints, next_key) = graphs::runtime_randint_i64::call(Tensor0::from_scalar(10_i64)).unwrap();
    assert_exact(next_key.as_slice(), &[16_i64]);
    assert!(
        ints.as_slice().iter().all(|value| (-3..4).contains(value)),
        "runtime randint values were outside [-3, 4): {:?}",
        ints.as_slice()
    );

    let (wide_ints, next_key) =
        graphs::runtime_randint_i32_wide::call(Tensor0::from_scalar(10_i64)).unwrap();
    assert_exact(next_key.as_slice(), &[18_i64]);
    assert!(
        wide_ints.as_slice().iter().all(|value| *value != i32::MAX),
        "wide runtime randint produced excluded high bound: {:?}",
        wide_ints.as_slice()
    );
    assert!(
        wide_ints.as_slice().iter().any(|value| *value != i32::MIN),
        "wide runtime randint collapsed to the low bound: {:?}",
        wide_ints.as_slice()
    );
}
