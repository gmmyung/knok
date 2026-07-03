use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn sum_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    sum(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn prod_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    prod(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn mean_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    mean(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn min_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    min(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn max_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    max(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn var_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    var(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn std_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    std(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn argmin_all_f32(x: T2<f32, 2, 3>) -> T0<i64> {
    argmin(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn argmax_all_f32(x: T2<f32, 2, 3>) -> T0<i64> {
    argmax(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn sum_axis1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    sum_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn mean_axis0_f32(x: T2<f32, 2, 3>) -> T1<f32, 3> {
    mean_axis(x, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn argmax_axis1_f32(x: T2<f32, 2, 3>) -> T1<i64, 2> {
    argmax_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn argmin_axis0_f32(x: T2<f32, 2, 3>) -> T1<i64, 3> {
    argmin_axis(x, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn amin_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    amin(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn amax_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    amax(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn amin_axis1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    amin_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn amax_axis0_f32(x: T2<f32, 2, 3>) -> T1<f32, 3> {
    amax_axis(x, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn min_axis0_f32(x: T2<f32, 2, 3>) -> T1<f32, 3> {
    min_axis(x, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn max_axis1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    max_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn prod_axis0_f32(x: T2<f32, 2, 3>) -> T1<f32, 3> {
    prod_axis(x, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn ptp_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    ptp(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn ptp_axis1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    ptp_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn var_axis0_f32(x: T2<f32, 2, 3>) -> T1<f32, 3> {
    var_axis(x, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn std_axis1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    std_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn all_bool(b: T2<bool, 2, 3>) -> T0<bool> {
    all(b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn any_bool(b: T2<bool, 2, 3>) -> T0<bool> {
    any(b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn all_axis1_bool(b: T2<bool, 2, 3>) -> T1<bool, 2> {
    all_axis(b, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn any_axis0_bool(b: T2<bool, 2, 3>) -> T1<bool, 3> {
    any_axis(b, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn softmax_all_f32(x: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    softmax(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn softmax_axis1_f32(x: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    softmax_axis(x, 1)
}
