use super::*;

#[test]
fn value_and_grad_reduction_extrema_ops_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]]);

    let (loss, grad_x) = autograd_graphs::reduction_extrema_loss_value_and_grad::call(&x).unwrap();

    assert_close(loss.as_slice(), &[128.0]);
    assert_close(grad_x.as_slice(), &[9.0, 4.0, 3.0, 36.0, 21.0, 16.0]);
}

#[test]
fn value_and_grad_softmax_runs() {
    let values = [[1.0_f32, 2.0, 3.0], [0.5, 1.5, -0.5]];
    let x = Tensor2::from_array(values);

    let (loss, grad_x) = autograd_graphs::softmax_square_loss_value_and_grad::call(&x).unwrap();

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

    let (loss, grad_x) = autograd_graphs::variance_std_ptp_loss_value_and_grad::call(&x).unwrap();

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
