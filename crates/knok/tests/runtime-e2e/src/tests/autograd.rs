use knok::tensor::*;

use crate::{autograd_graphs, common::*};

#[test]
fn value_and_grad_sum_square_runs() {
    let x = Tensor1::from_array([1.0, 2.0, -3.0]);

    let (loss, grad_x) = autograd_graphs::sum_square_loss_value_and_grad::call(x).unwrap();

    assert_close(loss.as_slice(), &[14.0]);
    assert_close(grad_x.as_slice(), &[2.0, 4.0, -6.0]);
}

#[test]
fn value_and_grad_unbroadcasts_bias_gradient() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let bias = Tensor1::from_array([10.0, 20.0, 30.0]);

    let (loss, grad_x, grad_bias) =
        autograd_graphs::bias_sum_loss_value_and_grad::call(x, bias).unwrap();

    assert_close(loss.as_slice(), &[141.0]);
    assert_close(grad_x.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_close(grad_bias.as_slice(), &[2.0, 2.0, 2.0]);
}

#[test]
fn value_and_grad_rank2_matmul_runs() {
    let lhs = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let rhs = Tensor2::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::matmul_sum_loss_value_and_grad::call(lhs, rhs).unwrap();

    assert_close(loss.as_slice(), &[415.0]);
    assert_close(grad_lhs.as_slice(), &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
    assert_close(grad_rhs.as_slice(), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn value_and_grad_matmul_rank_variants_run() {
    let lhs = Tensor1::from_array([1.0, 2.0, 3.0]);
    let rhs = Tensor1::from_array([4.0, 5.0, 6.0]);
    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::vecvec_matmul_loss_value_and_grad::call(lhs, rhs).unwrap();
    assert_close(loss.as_slice(), &[32.0]);
    assert_close(grad_lhs.as_slice(), &[4.0, 5.0, 6.0]);
    assert_close(grad_rhs.as_slice(), &[1.0, 2.0, 3.0]);

    let lhs = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let rhs = Tensor1::from_array([1.0, 2.0, 3.0]);
    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::matvec_sum_loss_value_and_grad::call(lhs, rhs).unwrap();
    assert_close(loss.as_slice(), &[46.0]);
    assert_close(grad_lhs.as_slice(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_close(grad_rhs.as_slice(), &[5.0, 7.0, 9.0]);

    let lhs = Tensor1::from_array([1.0, 2.0, 3.0]);
    let rhs = Tensor2::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::vecmat_sum_loss_value_and_grad::call(lhs, rhs).unwrap();
    assert_close(loss.as_slice(), &[122.0]);
    assert_close(grad_lhs.as_slice(), &[15.0, 19.0, 23.0]);
    assert_close(grad_rhs.as_slice(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn value_and_grad_batched_matmul_broadcasts_gradients() {
    let lhs = Tensor4::from_array([
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        [[[2.0, 0.0, 1.0], [1.0, 1.0, 1.0]]],
    ]);
    let rhs = Tensor3::from_array([
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ]);

    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::broadcast_matmul_sum_loss_value_and_grad::call(lhs, rhs).unwrap();

    assert_close(loss.as_slice(), &[726.0]);
    assert_close(
        grad_lhs.as_slice(),
        &[
            18.0, 26.0, 34.0, 18.0, 26.0, 34.0, 18.0, 26.0, 34.0, 18.0, 26.0, 34.0,
        ],
    );
    assert_close(
        grad_rhs.as_slice(),
        &[
            8.0, 8.0, 8.0, 8.0, 11.0, 11.0, 8.0, 8.0, 8.0, 8.0, 11.0, 11.0,
        ],
    );
}

#[test]
fn value_and_grad_shape_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let (loss, grad_x) = autograd_graphs::shape_loss_value_and_grad::call(x).unwrap();

    assert_close(loss.as_slice(), &[21.0]);
    assert_close(grad_x.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn value_and_grad_relu_mean_and_detach_run() {
    let x = Tensor2::from_array([[1.0, -2.0], [3.0, -4.0]]);
    let w = Tensor2::from_array([[1.0, 0.0], [0.0, 1.0]]);
    let b = Tensor1::from_array([0.0, 1.0]);

    let (loss, grad_x, grad_w, grad_b) =
        autograd_graphs::relu_mean_loss_value_and_grad::call(x, w, b).unwrap();

    assert_close(loss.as_slice(), &[1.0]);
    assert_close(grad_x.as_slice(), &[0.25, 0.0, 0.25, 0.0]);
    assert_close(grad_w.as_slice(), &[1.0, 0.0, -1.5, 0.0]);
    assert_close(grad_b.as_slice(), &[0.5, 0.0]);

    let x = Tensor1::from_array([1.0, 2.0, -3.0]);
    let (loss, grad_x) = autograd_graphs::detached_square_loss_value_and_grad::call(x).unwrap();
    assert_close(loss.as_slice(), &[14.0]);
    assert_close(grad_x.as_slice(), &[0.0, 0.0, 0.0]);
}
