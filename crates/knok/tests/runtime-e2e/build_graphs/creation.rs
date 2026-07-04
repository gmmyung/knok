use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn arange_step_i32() -> T1<i32, 4> {
    arange_step(0, 8, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn linspace_f32() -> T1<f32, 5> {
    linspace(0.0, 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn identity_f32() -> T2<f32, 3, 3> {
    identity()
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn eye_f32() -> T2<f32, 3, 3> {
    eye()
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn arange_to_i32() -> T1<i32, 4> {
    arange_to(4)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn arange_i32() -> T1<i32, 4> {
    arange(2, 6)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn zeros_like_f32(x: T1<f32, 3>) -> T1<f32, 3> {
    zeros_like(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn ones_like_f32(x: T1<f32, 3>) -> T1<f32, 3> {
    ones_like(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn full_like_f32(x: T1<f32, 3>) -> T1<f32, 3> {
    full_like(x, 3.5)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn static_uniform_f32() -> T1<f32, 4> {
    uniform_static(123, -1.0, 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn static_uniform_f32_edge() -> T1<f32, 1> {
    uniform_static(63_433_462, 0.0, 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn static_randint_i32() -> T1<i32, 8> {
    randint_static(7, 2, 5)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn runtime_uniform_f32(key: T0<i64>) -> (T1<f32, 4>, T0<i64>) {
    rand(key)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn runtime_randint_i64(key: T0<i64>) -> (T1<i64, 6>, T0<i64>) {
    randint(key, -3, 4)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn runtime_randint_i32_wide(key: T0<i64>) -> (T1<i32, 8>, T0<i64>) {
    randint(key, i32::MIN, i32::MAX)
}
