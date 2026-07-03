use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn extended_elementwise_loss(x: T1<f32, 3>, y: T1<f32, 3>) -> T0<f32> {
    sum(abs(x.clone())
        + reciprocal(x.clone())
        + sin(x.clone())
        + cos(x.clone())
        + tan(x.clone())
        + log1p(x.clone())
        + log2(x.clone())
        + log10(x.clone())
        + exp2(x.clone())
        + expm1(x.clone())
        + pow(x, y))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn selection_loss(x: T1<f32, 3>, y: T1<f32, 3>) -> T0<f32> {
    let positive = greater(x.clone(), 0.0);
    let selected: T1<f32, 3> = r#where(
        positive,
        maximum(x.clone(), y.clone()),
        minimum(x.clone(), y),
    );
    sum(selected + clip(x, -1.0, 2.0))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn full_like_loss(fill: T0<f32>, x: T1<f32, 3>) -> T0<f32> {
    sum(full_like(x, fill))
}
