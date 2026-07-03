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

#[test]
fn value_and_grad_extended_elementwise_ops_run() {
    let x_values = [1.0_f32, 1.5, 2.0];
    let y_values = [2.0_f32, 0.5, 3.0];
    let x = Tensor1::from_array(x_values);
    let y = Tensor1::from_array(y_values);

    let (loss, grad_x, grad_y) =
        autograd_graphs::extended_elementwise_loss_value_and_grad::call(x, y).unwrap();

    let expected_loss = x_values
        .iter()
        .zip(y_values)
        .map(|(&x, y)| {
            x.abs()
                + x.recip()
                + x.sin()
                + x.cos()
                + x.tan()
                + x.ln_1p()
                + x.log2()
                + x.log10()
                + x.exp2()
                + x.exp_m1()
                + x.powf(y)
        })
        .sum::<f32>();
    let expected_grad_x = x_values
        .iter()
        .zip(y_values)
        .map(|(&x, y)| {
            1.0 - x.recip().powi(2) + x.cos() - x.sin()
                + (1.0 + x.tan().powi(2))
                + 1.0 / (1.0 + x)
                + 1.0 / (x * core::f32::consts::LN_2)
                + 1.0 / (x * core::f32::consts::LN_10)
                + core::f32::consts::LN_2 * x.exp2()
                + x.exp()
                + y * x.powf(y - 1.0)
        })
        .collect::<Vec<_>>();
    let expected_grad_y = x_values
        .iter()
        .zip(y_values)
        .map(|(&x, y)| x.powf(y) * x.ln())
        .collect::<Vec<_>>();

    assert_close(loss.as_slice(), &[expected_loss]);
    assert_close(grad_x.as_slice(), &expected_grad_x);
    assert_close(grad_y.as_slice(), &expected_grad_y);
}

#[test]
fn value_and_grad_selection_ops_run() {
    let x = Tensor1::from_array([-2.0, 0.5, 3.0]);
    let y = Tensor1::from_array([-1.0, 2.0, 1.0]);

    let (loss, grad_x, grad_y) =
        autograd_graphs::selection_loss_value_and_grad::call(x, y).unwrap();

    assert_close(loss.as_slice(), &[4.5]);
    assert_close(grad_x.as_slice(), &[1.0, 1.0, 1.0]);
    assert_close(grad_y.as_slice(), &[0.0, 1.0, 0.0]);
}

#[test]
fn value_and_grad_reduction_extrema_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]]);

    let (loss, grad_x) = autograd_graphs::reduction_extrema_loss_value_and_grad::call(x).unwrap();

    assert_close(loss.as_slice(), &[128.0]);
    assert_close(grad_x.as_slice(), &[9.0, 4.0, 3.0, 36.0, 21.0, 16.0]);
}

#[test]
fn value_and_grad_layout_and_index_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let y = Tensor2::from_array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]);

    let (loss, grad_x, grad_y) =
        autograd_graphs::layout_index_loss_value_and_grad::call(x, y).unwrap();

    assert_close(loss.as_slice(), &[336.0]);
    assert_close(grad_x.as_slice(), &[6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);
    assert_close(grad_y.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn value_and_grad_full_like_runs() {
    let fill = Tensor0::from_scalar(2.5);
    let x = Tensor1::from_array([1.0, 2.0, 3.0]);

    let (loss, grad_fill, grad_x) =
        autograd_graphs::full_like_loss_value_and_grad::call(fill, x).unwrap();

    assert_close(loss.as_slice(), &[7.5]);
    assert_close(grad_fill.as_slice(), &[3.0]);
    assert_close(grad_x.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn value_and_grad_softmax_runs() {
    let values = [[1.0_f32, 2.0, 3.0], [0.5, 1.5, -0.5]];
    let x = Tensor2::from_array(values);

    let (loss, grad_x) = autograd_graphs::softmax_square_loss_value_and_grad::call(x).unwrap();

    let mut expected_loss = 0.0;
    let mut expected_grad = Vec::new();
    for row in values {
        let max = row.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let exp = row.map(|value| (value - max).exp());
        let denom = exp.iter().sum::<f32>();
        let softmax = exp.map(|value| value / denom);
        let upstream = softmax.map(|value| 2.0 * value);
        let dot = upstream
            .iter()
            .zip(softmax)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f32>();
        expected_loss += softmax.iter().map(|value| value * value).sum::<f32>();
        expected_grad.extend(
            softmax
                .iter()
                .zip(upstream)
                .map(|(value, upstream)| value * (upstream - dot)),
        );
    }

    assert_close(loss.as_slice(), &[expected_loss]);
    assert_close(grad_x.as_slice(), &expected_grad);
}

#[test]
fn value_and_grad_variance_std_ptp_runs() {
    let values = [[1.0_f32, 2.0, 4.0], [3.0, 5.0, 7.0]];
    let x = Tensor2::from_array(values);

    let (loss, grad_x) = autograd_graphs::variance_std_ptp_loss_value_and_grad::call(x).unwrap();

    let mut expected_loss = 0.0;
    let mut expected_grad = Vec::new();
    for row in values {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let centered = row.map(|value| value - mean);
        let variance = centered.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
        let std = variance.sqrt();
        let min = row.into_iter().fold(f32::INFINITY, f32::min);
        let max = row.into_iter().fold(f32::NEG_INFINITY, f32::max);
        expected_loss += variance + std + (max - min);
        expected_grad.extend(row.iter().zip(centered).map(|(&value, centered)| {
            let var_grad = 2.0 * centered / row.len() as f32;
            let std_grad = centered / (row.len() as f32 * std);
            let ptp_grad = if value == max {
                1.0
            } else if value == min {
                -1.0
            } else {
                0.0
            };
            var_grad + std_grad + ptp_grad
        }));
    }

    assert_close(loss.as_slice(), &[expected_loss]);
    assert_close(grad_x.as_slice(), &expected_grad);
}

#[test]
fn value_and_grad_linalg_ext_ops_run() {
    let v = Tensor1::from_array([1.0, 2.0, 3.0]);
    let w = Tensor1::from_array([4.0, 5.0, 6.0]);
    let m = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let n = Tensor2::from_array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);

    let (loss, grad_v, grad_w, grad_m, grad_n) =
        autograd_graphs::linalg_ext_loss_value_and_grad::call(v, w, m, n).unwrap();

    assert_close(loss.as_slice(), &[746.0]);
    assert_close(grad_v.as_slice(), &[19.0, 20.0, 21.0]);
    assert_close(grad_w.as_slice(), &[7.0, 8.0, 9.0]);
    assert_close(grad_m.as_slice(), &[24.0, 27.0, 30.0, 27.0, 30.0, 33.0]);
    assert_close(grad_n.as_slice(), &[6.0, 9.0, 12.0, 9.0, 12.0, 15.0]);
}

#[test]
fn value_and_grad_trace_diagonal_runs() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    let (loss, grad_x) = autograd_graphs::trace_diagonal_loss_value_and_grad::call(x).unwrap();

    assert_close(loss.as_slice(), &[30.0]);
    assert_close(
        grad_x.as_slice(),
        &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
    );
}

#[test]
fn value_and_grad_tile_repeat_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);

    let (loss, grad_x) = autograd_graphs::tile_repeat_loss_value_and_grad::call(x).unwrap();

    assert_close(loss.as_slice(), &[80.0]);
    assert_close(grad_x.as_slice(), &[8.0, 8.0, 8.0, 8.0]);
}

#[test]
fn value_and_grad_tile_and_repeat_run_independently() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let (loss, grad_x) = autograd_graphs::tile_only_loss_value_and_grad::call(x.clone()).unwrap();
    assert_close(loss.as_slice(), &[60.0]);
    assert_close(grad_x.as_slice(), &[6.0, 6.0, 6.0, 6.0]);

    let (loss, grad_x) = autograd_graphs::repeat_only_loss_value_and_grad::call(x).unwrap();
    assert_close(loss.as_slice(), &[20.0]);
    assert_close(grad_x.as_slice(), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn value_and_grad_split_runs() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let (loss, grad_x) = autograd_graphs::split_loss_value_and_grad::call(x).unwrap();

    assert_close(loss.as_slice(), &[33.0]);
    assert_close(grad_x.as_slice(), &[2.0, 1.0, 1.0, 8.0, 1.0, 1.0]);
}

#[test]
fn value_and_grad_indexed_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let gather_idx = Tensor2::from_array([[0_i64, -1], [1, 0]]);
    let take_idx = Tensor2::from_array([[2_i64, 1], [0, -1]]);

    let (loss, grad_x) =
        autograd_graphs::indexed_loss_value_and_grad::call(x, gather_idx, take_idx).unwrap();

    assert_close(loss.as_slice(), &[41.0]);
    assert_close(grad_x.as_slice(), &[2.0, 2.0, 2.0, 3.0, 1.0, 2.0]);

    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let gather_idx = Tensor1::from_array([0_i64, 0, -1]);
    let take_idx = Tensor2::from_array([[1_i64, 0, -1], [0, -2, 1]]);
    let (loss, grad_x) =
        autograd_graphs::indexed_axis0_loss_value_and_grad::call(x, gather_idx, take_idx).unwrap();

    assert_close(loss.as_slice(), &[48.0]);
    assert_close(grad_x.as_slice(), &[3.0, 4.0, 2.0, 2.0, 1.0, 3.0]);
}

#[test]
fn value_and_grad_argmax_index_path_is_no_grad() {
    let x = Tensor2::from_array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]);

    let (loss, grad_x) = autograd_graphs::argmax_indexed_loss_value_and_grad::call(x).unwrap();

    assert_close(loss.as_slice(), &[11.0]);
    assert_close(grad_x.as_slice(), &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn value_and_grad_pooling_runs() {
    let max_input = Tensor4::from_array([[
        [[-8.0], [-7.0], [-6.0], [-5.0]],
        [[-4.0], [-3.0], [-2.0], [-1.0]],
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
    ]]);
    let (loss, grad_x) = autograd_graphs::max_pool_loss_value_and_grad::call(max_input).unwrap();
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
    let (loss, grad_x) = autograd_graphs::avg_pool_loss_value_and_grad::call(avg_input).unwrap();
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
        autograd_graphs::conv2d_sum_loss_value_and_grad::call(x, k).unwrap();

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
        autograd_graphs::grouped_conv2d_sum_loss_value_and_grad::call(x, k).unwrap();
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
        autograd_graphs::padded_stride_dilated_conv2d_sum_loss_value_and_grad::call(x, k).unwrap();
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
