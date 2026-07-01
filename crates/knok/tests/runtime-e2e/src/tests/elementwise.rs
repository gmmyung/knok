use knok::tensor::*;

use crate::{common::*, graphs};

#[test]
fn unary_rounding_and_arithmetic_ops_run() {
    let x = Tensor1::from_array([-1.25, -0.5, 1.5, 2.25]);

    let abs = graphs::abs_f32::call(x.clone()).unwrap();
    let ceil = graphs::ceil_f32::call(x.clone()).unwrap();
    let floor = graphs::floor_f32::call(x.clone()).unwrap();
    let rint = graphs::rint_f32::call(x.clone()).unwrap();
    let round = graphs::round_f32::call(x.clone()).unwrap();
    let relu = graphs::relu_f32::call(x.clone()).unwrap();
    assert_close(abs.as_slice(), &[1.25, 0.5, 1.5, 2.25]);
    assert_close(ceil.as_slice(), &[-1.0, -0.0, 2.0, 3.0]);
    assert_close(floor.as_slice(), &[-2.0, -1.0, 1.0, 2.0]);
    assert_close(rint.as_slice(), &[-1.0, -0.0, 2.0, 2.0]);
    assert_close(round.as_slice(), &[-1.0, -0.0, 2.0, 2.0]);
    assert_close(relu.as_slice(), &[0.0, 0.0, 1.5, 2.25]);

    let square = graphs::square_f32::call(x.clone()).unwrap();
    let reciprocal = graphs::reciprocal_f32::call(x).unwrap();
    assert_close(square.as_slice(), &[1.5625, 0.25, 2.25, 5.0625]);
    assert_close(reciprocal.as_slice(), &[-0.8, -2.0, 2.0 / 3.0, 4.0 / 9.0]);
}

#[test]
fn transcendental_and_activation_ops_run() {
    let x = Tensor1::from_array([0.25, 0.5, 1.0, 2.0]);

    let exp = graphs::exp_f32::call(x.clone()).unwrap();
    let exp2 = graphs::exp2_f32::call(x.clone()).unwrap();
    let expm1 = graphs::expm1_f32::call(x.clone()).unwrap();
    let log = graphs::log_f32::call(x.clone()).unwrap();
    let log1p = graphs::log1p_f32::call(x.clone()).unwrap();
    let log2 = graphs::log2_f32::call(x.clone()).unwrap();
    let values = x.as_slice().to_vec();
    assert_close(
        exp.as_slice(),
        &values.iter().map(|v| v.exp()).collect::<Vec<_>>(),
    );
    assert_close(
        exp2.as_slice(),
        &values.iter().map(|v| v.exp2()).collect::<Vec<_>>(),
    );
    assert_close(
        expm1.as_slice(),
        &values.iter().map(|v| v.exp_m1()).collect::<Vec<_>>(),
    );
    assert_close(
        log.as_slice(),
        &values.iter().map(|v| v.ln()).collect::<Vec<_>>(),
    );
    assert_close(
        log1p.as_slice(),
        &values.iter().map(|v| v.ln_1p()).collect::<Vec<_>>(),
    );
    assert_close(
        log2.as_slice(),
        &values.iter().map(|v| v.log2()).collect::<Vec<_>>(),
    );

    let log10 = graphs::log10_f32::call(x.clone()).unwrap();
    let sqrt = graphs::sqrt_f32::call(x.clone()).unwrap();
    let isnan = graphs::isnan_f32::call(x.clone()).unwrap();
    assert_close(
        log10.as_slice(),
        &values.iter().map(|v| v.log10()).collect::<Vec<_>>(),
    );
    assert_close(
        sqrt.as_slice(),
        &values.iter().map(|v| v.sqrt()).collect::<Vec<_>>(),
    );
    assert_exact(isnan.as_slice(), &[false, false, false, false]);

    let sin = graphs::sin_f32::call(x.clone()).unwrap();
    let cos = graphs::cos_f32::call(x.clone()).unwrap();
    let tan = graphs::tan_f32::call(x.clone()).unwrap();
    let tanh = graphs::tanh_f32::call(x.clone()).unwrap();
    let sigmoid = graphs::sigmoid_f32::call(x).unwrap();
    assert_close(
        sin.as_slice(),
        &values.iter().map(|v| v.sin()).collect::<Vec<_>>(),
    );
    assert_close(
        cos.as_slice(),
        &values.iter().map(|v| v.cos()).collect::<Vec<_>>(),
    );
    assert_close(
        tan.as_slice(),
        &values.iter().map(|v| v.tan()).collect::<Vec<_>>(),
    );
    assert_close(
        tanh.as_slice(),
        &values.iter().map(|v| v.tanh()).collect::<Vec<_>>(),
    );
    assert_close(
        sigmoid.as_slice(),
        &values
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn binary_predicate_and_logical_ops_run() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([2.0, 2.0, 1.0, 4.0]);

    let minimum = graphs::minimum_f32::call(x.clone(), y.clone()).unwrap();
    let maximum = graphs::maximum_f32::call(x.clone(), y.clone()).unwrap();
    let pow = graphs::pow_f32::call(x.clone(), y.clone()).unwrap();
    let clip = graphs::clip_f32::call(x.clone()).unwrap();
    let full = graphs::full_like_4_f32::call(x.clone()).unwrap();
    assert_close(minimum.as_slice(), &[1.0, 2.0, 1.0, 4.0]);
    assert_close(maximum.as_slice(), &[2.0, 2.0, 3.0, 4.0]);
    assert_close(pow.as_slice(), &[1.0, 4.0, 3.0, 256.0]);
    assert_close(clip.as_slice(), &[1.5, 2.0, 3.0, 3.5]);
    assert_close(full.as_slice(), &[7.0, 7.0, 7.0, 7.0]);

    let gt = graphs::greater_f32::call(x.clone(), y.clone()).unwrap();
    let ge = graphs::greater_equal_f32::call(x.clone(), y.clone()).unwrap();
    let lt = graphs::less_f32::call(x.clone(), y.clone()).unwrap();
    let le = graphs::less_equal_f32::call(x.clone(), y.clone()).unwrap();
    let eq = graphs::equal_f32::call(x.clone(), y.clone()).unwrap();
    let ne = graphs::not_equal_f32::call(x, y).unwrap();
    assert_exact(gt.as_slice(), &[false, false, true, false]);
    assert_exact(ge.as_slice(), &[false, true, true, true]);
    assert_exact(lt.as_slice(), &[true, false, false, false]);
    assert_exact(le.as_slice(), &[true, true, false, true]);
    assert_exact(eq.as_slice(), &[false, true, false, true]);
    assert_exact(ne.as_slice(), &[true, false, true, false]);

    let a = Tensor1::from_array([true, true, false, false]);
    let b = Tensor1::from_array([true, false, true, false]);
    let and = graphs::logical_and_bool::call(a.clone(), b.clone()).unwrap();
    let or = graphs::logical_or_bool::call(a.clone(), b.clone()).unwrap();
    let xor = graphs::logical_xor_bool::call(a.clone(), b).unwrap();
    let not = graphs::logical_not_bool::call(a).unwrap();
    assert_exact(and.as_slice(), &[true, false, false, false]);
    assert_exact(or.as_slice(), &[true, true, true, false]);
    assert_exact(xor.as_slice(), &[false, true, true, false]);
    assert_exact(not.as_slice(), &[false, false, true, true]);
}
