use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn sum_square_loss(x: T1<f32, 3>) -> T0<f32> {
    sum(square(x))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn bias_sum_loss(x: T2<f32, 2, 3>, bias: T1<f32, 3>) -> T0<f32> {
    sum(x + bias)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn shape_loss(x: T2<f32, 2, 3>) -> T0<f32> {
    let flat: T1<f32, 6> = reshape(x);
    let matrix: T2<f32, 3, 2> = reshape(flat);
    let transposed: T2<f32, 2, 3> = transpose(matrix);
    sum(transposed)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn relu_mean_loss(x: T2<f32, 2, 2>, w: T2<f32, 2, 2>, b: T1<f32, 2>) -> T0<f32> {
    mean(relu(matmul(x, w) + b))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn detached_square_loss(x: T1<f32, 3>) -> T0<f32> {
    sum(square(detach(x)))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn random_weighted_sum_loss(x: T1<f32, 4>, key: T0<i64>) -> T0<f32> {
    let (weights, _next_key): (T1<f32, 4>, T0<i64>) = rand(key);
    sum(x * weights)
}
