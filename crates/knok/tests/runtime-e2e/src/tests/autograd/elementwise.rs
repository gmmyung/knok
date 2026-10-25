use super::*;

#[test]
fn value_and_grad_extended_elementwise_ops_run() {
    let x_values = [1.0_f32, 1.5, 2.0];
    let y_values = [2.0_f32, 0.5, 3.0];
    let x = Tensor1::from_array(x_values);
    let y = Tensor1::from_array(y_values);

    let (loss, grad_x, grad_y) =
        autograd_graphs::extended_elementwise_loss_value_and_grad::call(&x, &y).unwrap();

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
        autograd_graphs::selection_loss_value_and_grad::call(&x, &y).unwrap();

    assert_close(loss.as_slice(), &[4.5]);
    assert_close(grad_x.as_slice(), &[1.0, 1.0, 1.0]);
    assert_close(grad_y.as_slice(), &[0.0, 1.0, 0.0]);
}

#[test]
fn value_and_grad_full_like_runs() {
    let fill = Tensor0::from_scalar(2.5);
    let x = Tensor1::from_array([1.0, 2.0, 3.0]);

    let (loss, grad_fill, grad_x) =
        autograd_graphs::full_like_loss_value_and_grad::call(&fill, &x).unwrap();

    assert_close(loss.as_slice(), &[7.5]);
    assert_close(grad_fill.as_slice(), &[3.0]);
    assert_close(grad_x.as_slice(), &[0.0, 0.0, 0.0]);
}
