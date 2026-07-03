use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn affine_relu(x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
    relu(matmul(x.clone(), x) + 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn elementwise_select(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    r#where(greater(x.clone(), 0.0), clip(maximum(x, y), 0.0, 6.0), 0.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn multi_output_stats(x: T2<f32, 2, 3>) -> (T1<f32, 2>, T1<i64, 2>) {
    (sum_axis(x.clone(), 1), argmax_axis(x, 1))
}
