use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn reduction_extrema_loss(x: T2<f32, 2, 3>) -> T0<f32> {
    let prod: T1<f32, 2> = prod_axis(x.clone(), 1);
    let max: T1<f32, 2> = max_axis(x.clone(), 1);
    let min: T1<f32, 2> = min_axis(x, 1);
    sum(prod + max + min)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn softmax_square_loss(x: T2<f32, 2, 3>) -> T0<f32> {
    sum(square(softmax_axis(x, 1)))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn variance_std_ptp_loss(x: T2<f32, 2, 3>) -> T0<f32> {
    let var: T1<f32, 2> = var_axis(x.clone(), 1);
    let std: T1<f32, 2> = std_axis(x.clone(), 1);
    let ptp: T1<f32, 2> = ptp_axis(x, 1);
    sum(var + std + ptp)
}
