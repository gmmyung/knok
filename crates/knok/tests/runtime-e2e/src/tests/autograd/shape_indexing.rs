use super::*;

#[test]
fn value_and_grad_shape_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let (loss, grad_x) = autograd_graphs::shape_loss_value_and_grad::call(x).unwrap();

    assert_close(loss.as_slice(), &[21.0]);
    assert_close(grad_x.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
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
