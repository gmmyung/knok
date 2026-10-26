use super::*;

#[test]
fn value_and_grad_rank2_matmul_runs() {
    let lhs = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let rhs = Tensor2::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::matmul_sum_loss_value_and_grad::call(&lhs, &rhs).unwrap();

    assert_close(loss.as_slice(), &[415.0]);
    assert_close(grad_lhs.as_slice(), &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
    assert_close(grad_rhs.as_slice(), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn value_and_grad_matmul_rank_variants_run() {
    let lhs = Tensor1::from_array([1.0, 2.0, 3.0]);
    let rhs = Tensor1::from_array([4.0, 5.0, 6.0]);
    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::vecvec_matmul_loss_value_and_grad::call(&lhs, &rhs).unwrap();
    assert_close(loss.as_slice(), &[32.0]);
    assert_close(grad_lhs.as_slice(), &[4.0, 5.0, 6.0]);
    assert_close(grad_rhs.as_slice(), &[1.0, 2.0, 3.0]);

    let lhs = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let rhs = Tensor1::from_array([1.0, 2.0, 3.0]);
    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::matvec_sum_loss_value_and_grad::call(&lhs, &rhs).unwrap();
    assert_close(loss.as_slice(), &[46.0]);
    assert_close(grad_lhs.as_slice(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_close(grad_rhs.as_slice(), &[5.0, 7.0, 9.0]);

    let lhs = Tensor1::from_array([1.0, 2.0, 3.0]);
    let rhs = Tensor2::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let (loss, grad_lhs, grad_rhs) =
        autograd_graphs::vecmat_sum_loss_value_and_grad::call(&lhs, &rhs).unwrap();
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
        autograd_graphs::broadcast_matmul_sum_loss_value_and_grad::call(&lhs, &rhs).unwrap();

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
fn value_and_grad_linalg_ext_ops_run() {
    let v = Tensor1::from_array([1.0, 2.0, 3.0]);
    let w = Tensor1::from_array([4.0, 5.0, 6.0]);
    let m = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let n = Tensor2::from_array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);

    let (loss, grad_v, grad_w, grad_m, grad_n) =
        autograd_graphs::linalg_ext_loss_value_and_grad::call(&v, &w, &m, &n).unwrap();

    assert_close(loss.as_slice(), &[746.0]);
    assert_close(grad_v.as_slice(), &[19.0, 20.0, 21.0]);
    assert_close(grad_w.as_slice(), &[7.0, 8.0, 9.0]);
    assert_close(grad_m.as_slice(), &[24.0, 27.0, 30.0, 27.0, 30.0, 33.0]);
    assert_close(grad_n.as_slice(), &[6.0, 9.0, 12.0, 9.0, 12.0, 15.0]);
}

#[test]
fn value_and_grad_trace_diagonal_runs() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    let (loss, grad_x) = autograd_graphs::trace_diagonal_loss_value_and_grad::call(&x).unwrap();

    assert_close(loss.as_slice(), &[30.0]);
    assert_close(
        grad_x.as_slice(),
        &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
    );
}

#[test]
fn value_and_grad_pooling_runs() {
    let max_input = Tensor4::from_array([[
        [[-8.0], [-7.0], [-6.0], [-5.0]],
        [[-4.0], [-3.0], [-2.0], [-1.0]],
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
    ]]);
    let (loss, grad_x) = autograd_graphs::max_pool_loss_value_and_grad::call(&max_input).unwrap();
    assert_close(loss.as_slice(), &[10.0]);
    assert_close(
        grad_x.as_slice(),
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ],
    );

    let avg_input = Tensor4::from_array([[
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
        [[7.0], [8.0], [9.0]],
    ]]);
    let (loss, grad_x) = autograd_graphs::avg_pool_loss_value_and_grad::call(&avg_input).unwrap();
    assert_close(loss.as_slice(), &[11.25]);
    assert_close(grad_x.as_slice(), &[0.25; 9]);
}

#[test]
fn value_and_grad_conv2d_runs() {
    let x = Tensor4::from_array([[
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
        [[7.0], [8.0], [9.0]],
    ]]);
    let k = Tensor4::from_array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]);

    let (loss, grad_x, grad_k) =
        autograd_graphs::conv2d_sum_loss_value_and_grad::call(&x, &k).unwrap();

    assert_close(loss.as_slice(), &[40.0]);
    assert_close(
        grad_x.as_slice(),
        &[1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0],
    );
    assert_close(grad_k.as_slice(), &[12.0, 16.0, 24.0, 28.0]);

    let x = Tensor4::from_array([[
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        [[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]],
    ]]);
    let k = Tensor4::from_array([[[[1.0, 1.0]], [[0.0, 0.0]]], [[[0.0, 0.0]], [[1.0, 1.0]]]]);
    let (loss, grad_x, grad_k) =
        autograd_graphs::grouped_conv2d_sum_loss_value_and_grad::call(&x, &k).unwrap();
    assert_close(loss.as_slice(), &[440.0]);
    assert_close(
        grad_x.as_slice(),
        &[
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            1.0,
        ],
    );
    assert_close(
        grad_k.as_slice(),
        &[12.0, 120.0, 16.0, 160.0, 24.0, 240.0, 28.0, 280.0],
    );

    let x = Tensor4::from_array([[
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
        [[6.0], [7.0], [8.0], [9.0], [10.0]],
        [[11.0], [12.0], [13.0], [14.0], [15.0]],
        [[16.0], [17.0], [18.0], [19.0], [20.0]],
        [[21.0], [22.0], [23.0], [24.0], [25.0]],
    ]]);
    let k = Tensor4::from_array([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]);
    let (loss, grad_x, grad_k) =
        autograd_graphs::padded_stride_dilated_conv2d_sum_loss_value_and_grad::call(&x, &k)
            .unwrap();
    assert_close(loss.as_slice(), &[520.0]);
    assert_close(
        grad_x.as_slice(),
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
            0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    assert_close(grad_k.as_slice(), &[52.0, 52.0, 52.0, 52.0]);
}
