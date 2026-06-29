use knok::tensor::*;

use crate::{common::*, graphs};

#[test]
fn generated_shape_ops_return_expected_layouts() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let flat = graphs::reshape_2x3_to_6::call(x.clone()).unwrap();
    let transposed = graphs::transpose_2x3::call(x).unwrap();

    assert_close(flat.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_close(transposed.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn shape_and_index_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let squeeze_input = Tensor3::from_array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
    let broadcast = graphs::broadcast_first_row_2x3::call(x.clone()).unwrap();
    let reshape = graphs::reshape_2x3_to_6::call(x.clone()).unwrap();
    let unsqueeze = graphs::unsqueeze_2x3::call(x.clone()).unwrap();
    let squeeze = graphs::squeeze_1x2x3::call(squeeze_input).unwrap();
    let slice = graphs::slice_2x3::call(x.clone()).unwrap();
    let pad = graphs::pad_2x3::call(x.clone()).unwrap();
    assert_close(broadcast.as_slice(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_close(reshape.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_close(unsqueeze.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_close(squeeze.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_close(slice.as_slice(), &[2.0, 3.0, 5.0, 6.0]);
    assert_close(
        pad.as_slice(),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0],
    );

    let idx = Tensor2::from_array([[0_i64, 2], [1, 0]]);
    let take_idx = Tensor2::from_array([[2_i64, 1], [0, 2]]);
    let gather = graphs::gather_axis1_f32::call(x.clone(), idx).unwrap();
    let take = graphs::take_axis1_index1_f32::call(x.clone()).unwrap();
    let take_along_axis = graphs::take_along_axis1_f32::call(x, take_idx).unwrap();
    assert_close(gather.as_slice(), &[1.0, 3.0, 2.0, 1.0, 4.0, 6.0, 5.0, 4.0]);
    assert_close(take.as_slice(), &[2.0, 5.0]);
    assert_close(take_along_axis.as_slice(), &[3.0, 2.0, 4.0, 6.0]);
}
