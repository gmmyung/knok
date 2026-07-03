use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn matmul_2x3_3x2(x: T2<f32, 2, 3>, y: T2<f32, 3, 2>) -> T2<f32, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn vecvec_matmul_f32(x: T1<f32, 3>, y: T1<f32, 3>) -> T0<f32> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn vecmat_matmul_f32(x: T1<f32, 3>, y: T2<f32, 3, 2>) -> T1<f32, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn batched_matmul(x: T3<f32, 2, 2, 3>, y: T2<f32, 3, 2>) -> T3<f32, 2, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn same_batch_matmul(x: T3<f32, 2, 2, 3>, y: T3<f32, 2, 3, 2>) -> T3<f32, 2, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn broadcast_4d_matmul(
    x: T4<f32, 2, 1, 2, 3>,
    y: T3<f32, 2, 3, 2>,
) -> T4<f32, 2, 2, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn dot_f32(v: T1<f32, 3>) -> T0<f32> {
    dot(v.clone(), v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn inner_f32(m: T2<f32, 2, 3>, v: T1<f32, 3>) -> T1<f32, 2> {
    inner(m, v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn vecdot_axis1_f32(m: T2<f32, 2, 3>) -> T1<f32, 2> {
    vecdot_axis(m.clone(), m, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn vecdot_f32(v: T1<f32, 3>) -> T0<f32> {
    vecdot(v.clone(), v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn outer_f32(v: T1<f32, 3>) -> T2<f32, 3, 3> {
    outer(v.clone(), v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn matvec_f32(m: T2<f32, 2, 3>, v: T1<f32, 3>) -> T1<f32, 2> {
    matmul(m, v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn trace_f32(square: T2<f32, 3, 3>) -> T0<f32> {
    trace(square)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn trace_axes_f32(batched_square: T3<f32, 2, 3, 3>) -> T1<f32, 2> {
    trace_axes(batched_square, 1, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn diagonal_f32(square: T2<f32, 3, 3>) -> T1<f32, 3> {
    diagonal(square)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn diagonal_axes_f32(batched_square: T3<f32, 2, 3, 3>) -> T2<f32, 2, 3> {
    diagonal_axes(batched_square, 1, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn conv2d_f32(x: T4<f32, 1, 3, 3, 1>, k: T4<f32, 2, 2, 1, 1>) -> T4<f32, 1, 2, 2, 1> {
    conv2d(x, k)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn grouped_conv2d_f32(
    x: T4<f32, 1, 3, 3, 2>,
    k: T4<f32, 2, 2, 1, 2>,
) -> T4<f32, 1, 2, 2, 2> {
    conv2d_options(x, k, Conv2dOptions::new().groups(2))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn padded_stride_dilated_conv2d_f32(
    x: T4<f32, 1, 5, 5, 1>,
    k: T4<f32, 2, 2, 1, 1>,
) -> T4<f32, 1, 3, 3, 1> {
    conv2d_options(
        x,
        k,
        Conv2dOptions::new()
            .padding(1, 1, 1, 1)
            .stride(2, 2)
            .dilation(2, 2),
    )
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn max_pool2d_f32(x: T4<f32, 1, 4, 4, 1>) -> T4<f32, 1, 2, 2, 1> {
    max_pool2d(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn avg_pool2d_padded_f32(x: T4<f32, 1, 3, 3, 1>) -> T4<f32, 1, 2, 2, 1> {
    avg_pool2d_options(x, Pool2dOptions::new(2, 2).padding(1, 0, 1, 0).stride(2, 2))
}
