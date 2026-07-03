use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn abs_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    abs(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn ceil_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    ceil(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn floor_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    floor(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn rint_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    rint(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn round_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    round(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn relu_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    relu(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn square_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    square(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn reciprocal_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    reciprocal(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn exp_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    exp(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn exp2_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    exp2(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn expm1_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    expm1(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn log_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    log(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn log1p_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    log1p(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn log2_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    log2(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn log10_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    log10(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn sqrt_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    sqrt(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn isnan_f32(x: T1<f32, 4>) -> T1<bool, 4> {
    isnan(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn sin_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    sin(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn cos_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    cos(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn tan_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    tan(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn tanh_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    tanh(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn sigmoid_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    sigmoid(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn minimum_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    minimum(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn maximum_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    maximum(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn pow_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    pow(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn clip_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    clip(x, 1.5, 3.5)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn full_like_4_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    full_like(x, 7.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn greater_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    greater(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn greater_equal_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    greater_equal(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn less_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    less(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn less_equal_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    less_equal(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn equal_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    equal(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn not_equal_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    not_equal(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn logical_and_bool(a: T1<bool, 4>, b: T1<bool, 4>) -> T1<bool, 4> {
    logical_and(a, b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn logical_or_bool(a: T1<bool, 4>, b: T1<bool, 4>) -> T1<bool, 4> {
    logical_or(a, b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn logical_xor_bool(a: T1<bool, 4>, b: T1<bool, 4>) -> T1<bool, 4> {
    logical_xor(a, b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn logical_not_bool(a: T1<bool, 4>) -> T1<bool, 4> {
    logical_not(a)
}
