use knok::tensor::*;

use crate::{common::*, graphs};

#[test]
fn reduction_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let sum = graphs::sum_all_f32::call(&x).unwrap();
    let prod = graphs::prod_all_f32::call(&x).unwrap();
    let mean = graphs::mean_all_f32::call(&x).unwrap();
    let min = graphs::min_all_f32::call(&x).unwrap();
    let max = graphs::max_all_f32::call(&x).unwrap();
    let var = graphs::var_all_f32::call(&x).unwrap();
    assert_close(sum.as_slice(), &[21.0]);
    assert_close(prod.as_slice(), &[720.0]);
    assert_close(mean.as_slice(), &[3.5]);
    assert_close(min.as_slice(), &[1.0]);
    assert_close(max.as_slice(), &[6.0]);
    assert_close(var.as_slice(), &[35.0 / 12.0]);

    let std = graphs::std_all_f32::call(&x).unwrap();
    let argmin = graphs::argmin_all_f32::call(&x).unwrap();
    let argmax = graphs::argmax_all_f32::call(&x).unwrap();
    let sum_axis = graphs::sum_axis1_f32::call(&x).unwrap();
    let mean_axis = graphs::mean_axis0_f32::call(&x).unwrap();
    let argmax_axis = graphs::argmax_axis1_f32::call(&x).unwrap();
    let argmin_axis = graphs::argmin_axis0_f32::call(&x).unwrap();
    assert_close(std.as_slice(), &[(35.0_f32 / 12.0).sqrt()]);
    assert_exact(argmin.as_slice(), &[0]);
    assert_exact(argmax.as_slice(), &[5]);
    assert_close(sum_axis.as_slice(), &[6.0, 15.0]);
    assert_close(mean_axis.as_slice(), &[2.5, 3.5, 4.5]);
    assert_exact(argmax_axis.as_slice(), &[2, 2]);
    assert_exact(argmin_axis.as_slice(), &[0, 0, 0]);

    let amin = graphs::amin_all_f32::call(&x).unwrap();
    let amax = graphs::amax_all_f32::call(&x).unwrap();
    let amin_axis = graphs::amin_axis1_f32::call(&x).unwrap();
    let amax_axis = graphs::amax_axis0_f32::call(&x).unwrap();
    let min_axis = graphs::min_axis0_f32::call(&x).unwrap();
    let max_axis = graphs::max_axis1_f32::call(&x).unwrap();
    let prod_axis = graphs::prod_axis0_f32::call(&x).unwrap();
    let ptp = graphs::ptp_all_f32::call(&x).unwrap();
    let ptp_axis = graphs::ptp_axis1_f32::call(&x).unwrap();
    let var_axis = graphs::var_axis0_f32::call(&x).unwrap();
    let std_axis = graphs::std_axis1_f32::call(&x).unwrap();
    assert_close(amin.as_slice(), &[1.0]);
    assert_close(amax.as_slice(), &[6.0]);
    assert_close(amin_axis.as_slice(), &[1.0, 4.0]);
    assert_close(amax_axis.as_slice(), &[4.0, 5.0, 6.0]);
    assert_close(min_axis.as_slice(), &[1.0, 2.0, 3.0]);
    assert_close(max_axis.as_slice(), &[3.0, 6.0]);
    assert_close(prod_axis.as_slice(), &[4.0, 10.0, 18.0]);
    assert_close(ptp.as_slice(), &[5.0]);
    assert_close(ptp_axis.as_slice(), &[2.0, 2.0]);
    assert_close(var_axis.as_slice(), &[2.25, 2.25, 2.25]);
    assert_close(
        std_axis.as_slice(),
        &[(2.0_f32 / 3.0).sqrt(), (2.0_f32 / 3.0).sqrt()],
    );
}

#[test]
fn bool_reduction_ops_run() {
    let b = Tensor2::from_array([[true, true, false], [true, true, true]]);

    let all = graphs::all_bool::call(&b).unwrap();
    let any = graphs::any_bool::call(&b).unwrap();
    let all_axis = graphs::all_axis1_bool::call(&b).unwrap();
    let any_axis = graphs::any_axis0_bool::call(&b).unwrap();

    assert_exact(all.as_slice(), &[false]);
    assert_exact(any.as_slice(), &[true]);
    assert_exact(all_axis.as_slice(), &[false, true]);
    assert_exact(any_axis.as_slice(), &[true, true, true]);
}

#[test]
fn softmax_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]);

    let all = graphs::softmax_all_f32::call(&x).unwrap();
    let axis = graphs::softmax_axis1_f32::call(&x).unwrap();

    let flat_den = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp() + 3.0 * 1.0_f32.exp();
    assert_close(
        all.as_slice(),
        &[
            1.0_f32.exp() / flat_den,
            2.0_f32.exp() / flat_den,
            3.0_f32.exp() / flat_den,
            1.0_f32.exp() / flat_den,
            1.0_f32.exp() / flat_den,
            1.0_f32.exp() / flat_den,
        ],
    );
    let row0_den = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp();
    assert_close(
        axis.as_slice(),
        &[
            1.0_f32.exp() / row0_den,
            2.0_f32.exp() / row0_den,
            3.0_f32.exp() / row0_den,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
    );
    let shifted = Tensor2::from_array([[1001.0, 1002.0, 1003.0], [1001.0, 1001.0, 1001.0]]);
    let shifted_all = graphs::softmax_all_f32::call(&shifted).unwrap();
    let shifted_axis = graphs::softmax_axis1_f32::call(&shifted).unwrap();
    assert_close(shifted_all.as_slice(), all.as_slice());
    assert_close(shifted_axis.as_slice(), axis.as_slice());
}
