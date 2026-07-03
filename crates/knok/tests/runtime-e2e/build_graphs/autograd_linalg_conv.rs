use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn matmul_sum_loss(lhs: T2<f32, 2, 3>, rhs: T2<f32, 3, 2>) -> T0<f32> {
    sum(matmul(lhs, rhs))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn vecvec_matmul_loss(lhs: T1<f32, 3>, rhs: T1<f32, 3>) -> T0<f32> {
    matmul(lhs, rhs)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn matvec_sum_loss(lhs: T2<f32, 2, 3>, rhs: T1<f32, 3>) -> T0<f32> {
    sum(matmul(lhs, rhs))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn vecmat_sum_loss(lhs: T1<f32, 3>, rhs: T2<f32, 3, 2>) -> T0<f32> {
    sum(matmul(lhs, rhs))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn broadcast_matmul_sum_loss(
    lhs: T4<f32, 2, 1, 2, 3>,
    rhs: T3<f32, 2, 3, 2>,
) -> T0<f32> {
    sum(matmul(lhs, rhs))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn linalg_ext_loss(
    v: T1<f32, 3>,
    w: T1<f32, 3>,
    m: T2<f32, 2, 3>,
    n: T2<f32, 2, 3>,
) -> T0<f32> {
    let dot_loss: T0<f32> = dot(v.clone(), w.clone());
    let vecdot_values: T1<f32, 2> = vecdot_axis(m.clone(), n.clone(), 1);
    let inner_values: T2<f32, 2, 2> = inner(m, n);
    let outer_values: T2<f32, 3, 3> = outer(v, w);
    let vecdot_loss: T0<f32> = sum(vecdot_values);
    let inner_loss: T0<f32> = sum(inner_values);
    let outer_loss: T0<f32> = sum(outer_values);
    dot_loss + vecdot_loss + inner_loss + outer_loss
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn trace_diagonal_loss(x: T2<f32, 3, 3>) -> T0<f32> {
    let trace_loss: T0<f32> = trace(x.clone());
    let diagonal_values: T1<f32, 3> = diagonal(x);
    let diagonal_loss: T0<f32> = sum(diagonal_values);
    trace_loss + diagonal_loss
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn max_pool_loss(x: T4<f32, 1, 4, 4, 1>) -> T0<f32> {
    let pooled: T4<f32, 1, 2, 2, 1> = max_pool2d(x);
    sum(pooled)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn avg_pool_loss(x: T4<f32, 1, 3, 3, 1>) -> T0<f32> {
    let pooled: T4<f32, 1, 2, 2, 1> =
        avg_pool2d_options(x, Pool2dOptions::new(2, 2).padding(1, 0, 1, 0).stride(2, 2));
    sum(pooled)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn conv2d_sum_loss(x: T4<f32, 1, 3, 3, 1>, k: T4<f32, 2, 2, 1, 1>) -> T0<f32> {
    let y: T4<f32, 1, 2, 2, 1> = conv2d(x, k);
    sum(y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn grouped_conv2d_sum_loss(x: T4<f32, 1, 3, 3, 2>, k: T4<f32, 2, 2, 1, 2>) -> T0<f32> {
    let y: T4<f32, 1, 2, 2, 2> = conv2d_options(x, k, Conv2dOptions::new().groups(2));
    sum(y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn padded_stride_dilated_conv2d_sum_loss(
    x: T4<f32, 1, 5, 5, 1>,
    k: T4<f32, 2, 2, 1, 1>,
) -> T0<f32> {
    let y: T4<f32, 1, 3, 3, 1> = conv2d_options(
        x,
        k,
        Conv2dOptions::new()
            .padding(1, 1, 1, 1)
            .stride(2, 2)
            .dilation(2, 2),
    );
    sum(y)
}
