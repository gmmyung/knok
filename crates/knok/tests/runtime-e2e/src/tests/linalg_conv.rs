use knok::tensor::*;

use crate::{common::*, graphs};

#[test]
fn batched_matmul_broadcasts_rhs() {
    let x = Tensor3::from_array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[2.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ]);
    let y = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

    let output = graphs::batched_matmul::call(x, y).unwrap();

    assert_close(
        output.as_slice(),
        &[22.0, 28.0, 49.0, 64.0, 7.0, 10.0, 9.0, 12.0],
    );
}

#[test]
fn matmul_rank_variants_run() {
    let v = Tensor1::from_array([1.0, 2.0, 3.0]);
    let n = Tensor2::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

    let vecvec = graphs::vecvec_matmul_f32::call(v.clone(), v.clone()).unwrap();
    let vecmat = graphs::vecmat_matmul_f32::call(v, n.clone()).unwrap();
    assert_close(vecvec.as_slice(), &[14.0]);
    assert_close(vecmat.as_slice(), &[58.0, 64.0]);

    let x3 = Tensor3::from_array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[2.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ]);
    let y3 = Tensor3::from_array([
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ]);
    let same_batch = graphs::same_batch_matmul::call(x3, y3).unwrap();
    assert_close(
        same_batch.as_slice(),
        &[58.0, 64.0, 139.0, 154.0, 7.0, 10.0, 9.0, 12.0],
    );

    let x4 = Tensor4::from_array([
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        [[[2.0, 0.0, 1.0], [1.0, 1.0, 1.0]]],
    ]);
    let y4 = Tensor3::from_array([
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ]);
    let broadcast = graphs::broadcast_4d_matmul::call(x4, y4).unwrap();
    assert_close(
        broadcast.as_slice(),
        &[
            58.0, 64.0, 139.0, 154.0, 22.0, 28.0, 49.0, 64.0, 25.0, 28.0, 27.0, 30.0, 7.0, 10.0,
            9.0, 12.0,
        ],
    );
}

#[test]
fn linalg_and_conv_ops_run() {
    let v = Tensor1::from_array([1.0, 2.0, 3.0]);
    let m = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let n = Tensor2::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let dot = graphs::dot_f32::call(v.clone()).unwrap();
    let inner = graphs::inner_f32::call(m.clone(), v.clone()).unwrap();
    let vecdot_axis = graphs::vecdot_axis1_f32::call(m.clone()).unwrap();
    let outer = graphs::outer_f32::call(v.clone()).unwrap();
    let matvec = graphs::matvec_f32::call(m.clone(), v).unwrap();
    let matmul = graphs::matmul_2x3_3x2::call(m.clone(), n.clone()).unwrap();
    assert_close(dot.as_slice(), &[14.0]);
    assert_close(inner.as_slice(), &[14.0, 32.0]);
    assert_close(vecdot_axis.as_slice(), &[14.0, 77.0]);
    assert_close(
        outer.as_slice(),
        &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
    );
    assert_close(matvec.as_slice(), &[14.0, 32.0]);
    assert_close(matmul.as_slice(), &[58.0, 64.0, 139.0, 154.0]);

    let square = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let batched_square = Tensor3::from_array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
    ]);
    let batched_mat = Tensor3::from_array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[2.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ]);
    let trace = graphs::trace_f32::call(square.clone()).unwrap();
    let trace_axes = graphs::trace_axes_f32::call(batched_square.clone()).unwrap();
    let diagonal = graphs::diagonal_f32::call(square).unwrap();
    let diagonal_axes = graphs::diagonal_axes_f32::call(batched_square).unwrap();
    let batched_matmul = graphs::batched_matmul::call(batched_mat, n).unwrap();
    assert_close(trace.as_slice(), &[15.0]);
    assert_close(trace_axes.as_slice(), &[15.0, 15.0]);
    assert_close(diagonal.as_slice(), &[1.0, 5.0, 9.0]);
    assert_close(diagonal_axes.as_slice(), &[1.0, 5.0, 9.0, 9.0, 5.0, 1.0]);
    assert_close(
        batched_matmul.as_slice(),
        &[58.0, 64.0, 139.0, 154.0, 25.0, 28.0, 27.0, 30.0],
    );

    let x = Tensor4::from_array([[
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
        [[7.0], [8.0], [9.0]],
    ]]);
    let k = Tensor4::from_array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]);
    let xg = Tensor4::from_array([[
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        [[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]],
    ]]);
    let kg = Tensor4::from_array([[[[1.0, 1.0]], [[0.0, 0.0]]], [[[0.0, 0.0]], [[1.0, 1.0]]]]);
    let conv = graphs::conv2d_f32::call(x, k).unwrap();
    let grouped = graphs::grouped_conv2d_f32::call(xg, kg).unwrap();
    assert_close(conv.as_slice(), &[6.0, 8.0, 12.0, 14.0]);
    assert_close(
        grouped.as_slice(),
        &[6.0, 60.0, 8.0, 80.0, 12.0, 120.0, 14.0, 140.0],
    );

    let pool_input = Tensor4::from_array([[
        [[-8.0], [-7.0], [-6.0], [-5.0]],
        [[-4.0], [-3.0], [-2.0], [-1.0]],
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
    ]]);
    let max_pool = graphs::max_pool2d_f32::call(pool_input).unwrap();
    assert_close(max_pool.as_slice(), &[-3.0, -1.0, 6.0, 8.0]);

    let avg_input = Tensor4::from_array([[
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
        [[7.0], [8.0], [9.0]],
    ]]);
    let avg_pool = graphs::avg_pool2d_padded_f32::call(avg_input).unwrap();
    assert_close(avg_pool.as_slice(), &[0.25, 1.25, 2.75, 7.0]);
}
