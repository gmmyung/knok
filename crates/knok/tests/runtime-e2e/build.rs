use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn affine_relu(x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
    relu(matmul(x.clone(), x) + 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn elementwise_select(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    r#where(greater(x.clone(), 0.0), clip(maximum(x, y), 0.0, 6.0), 0.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn multi_output_stats(x: T2<f32, 2, 3>) -> (T1<f32, 2>, T1<i64, 2>) {
    (sum_axis(x.clone(), 1), argmax_axis(x, 1))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn matmul_2x3_3x2(x: T2<f32, 2, 3>, y: T2<f32, 3, 2>) -> T2<f32, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn vecvec_matmul_f32(x: T1<f32, 3>, y: T1<f32, 3>) -> T0<f32> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn vecmat_matmul_f32(x: T1<f32, 3>, y: T2<f32, 3, 2>) -> T1<f32, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn batched_matmul(x: T3<f32, 2, 2, 3>, y: T2<f32, 3, 2>) -> T3<f32, 2, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn same_batch_matmul(x: T3<f32, 2, 2, 3>, y: T3<f32, 2, 3, 2>) -> T3<f32, 2, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn broadcast_4d_matmul(x: T4<f32, 2, 1, 2, 3>, y: T3<f32, 2, 3, 2>) -> T4<f32, 2, 2, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn arange_step_i32() -> T1<i32, 4> {
    arange_step(0, 8, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn linspace_f32() -> T1<f32, 5> {
    linspace(0.0, 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn identity_f32() -> T2<f32, 3, 3> {
    identity()
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn arange_to_i32() -> T1<i32, 4> {
    arange_to(4)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn arange_i32() -> T1<i32, 4> {
    arange(2, 6)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn zeros_like_f32(x: T1<f32, 3>) -> T1<f32, 3> {
    zeros_like(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn ones_like_f32(x: T1<f32, 3>) -> T1<f32, 3> {
    ones_like(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn full_like_f32(x: T1<f32, 3>) -> T1<f32, 3> {
    full_like(x, 3.5)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn abs_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    abs(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn ceil_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    ceil(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn floor_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    floor(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn rint_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    rint(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn round_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    round(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn relu_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    relu(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn square_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    square(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn reciprocal_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    reciprocal(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn exp_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    exp(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn exp2_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    exp2(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn expm1_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    expm1(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn log_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    log(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn log1p_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    log1p(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn log2_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    log2(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn log10_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    log10(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn sqrt_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    sqrt(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn isnan_f32(x: T1<f32, 4>) -> T1<bool, 4> {
    isnan(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn sin_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    sin(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn cos_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    cos(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn tan_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    tan(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn sigmoid_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    sigmoid(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn minimum_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    minimum(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn maximum_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    maximum(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn pow_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    pow(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn clip_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    clip(x, 1.5, 3.5)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn full_like_4_f32(x: T1<f32, 4>) -> T1<f32, 4> {
    full_like(x, 7.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn greater_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    greater(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn greater_equal_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    greater_equal(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn less_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    less(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn less_equal_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    less_equal(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn equal_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    equal(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn not_equal_f32(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<bool, 4> {
    not_equal(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn logical_and_bool(a: T1<bool, 4>, b: T1<bool, 4>) -> T1<bool, 4> {
    logical_and(a, b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn logical_or_bool(a: T1<bool, 4>, b: T1<bool, 4>) -> T1<bool, 4> {
    logical_or(a, b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn logical_xor_bool(a: T1<bool, 4>, b: T1<bool, 4>) -> T1<bool, 4> {
    logical_xor(a, b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn logical_not_bool(a: T1<bool, 4>) -> T1<bool, 4> {
    logical_not(a)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn sum_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    sum(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn prod_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    prod(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mean_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    mean(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn min_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    min(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn max_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    max(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn var_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    var(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn std_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    std(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn argmin_all_f32(x: T2<f32, 2, 3>) -> T0<i64> {
    argmin(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn argmax_all_f32(x: T2<f32, 2, 3>) -> T0<i64> {
    argmax(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn sum_axis1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    sum_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mean_axis0_f32(x: T2<f32, 2, 3>) -> T1<f32, 3> {
    mean_axis(x, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn argmax_axis1_f32(x: T2<f32, 2, 3>) -> T1<i64, 2> {
    argmax_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn amin_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    amin(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn amax_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    amax(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn amin_axis1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    amin_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn amax_axis0_f32(x: T2<f32, 2, 3>) -> T1<f32, 3> {
    amax_axis(x, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn ptp_all_f32(x: T2<f32, 2, 3>) -> T0<f32> {
    ptp(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn ptp_axis1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    ptp_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn all_bool(b: T2<bool, 2, 3>) -> T0<bool> {
    all(b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn any_bool(b: T2<bool, 2, 3>) -> T0<bool> {
    any(b)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn all_axis1_bool(b: T2<bool, 2, 3>) -> T1<bool, 2> {
    all_axis(b, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn any_axis0_bool(b: T2<bool, 2, 3>) -> T1<bool, 3> {
    any_axis(b, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn softmax_all_f32(x: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    softmax(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn softmax_axis1_f32(x: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    softmax_axis(x, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn reshape_2x3_to_6(x: T2<f32, 2, 3>) -> T1<f32, 6> {
    reshape(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn transpose_2x3(x: T2<f32, 2, 3>) -> T2<f32, 3, 2> {
    transpose(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn broadcast_first_row_2x3(x: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    broadcast::<T2<f32, 2, 3>>(take::<T1<f32, 3>>(x, 0, 0))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn unsqueeze_2x3(x: T2<f32, 2, 3>) -> T3<f32, 1, 2, 3> {
    unsqueeze(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn squeeze_1x2x3(x: T3<f32, 1, 2, 3>) -> T2<f32, 2, 3> {
    squeeze(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn slice_2x3(x: T2<f32, 2, 3>) -> T2<f32, 2, 2> {
    slice(x, [0, 1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn slice_scalar(x: T0<f32>) -> T0<f32> {
    slice(x, [])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn pad_2x3(x: T2<f32, 2, 3>) -> T2<f32, 3, 4> {
    pad(x, [1, 1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn gather_axis1_f32(x: T2<f32, 2, 3>, idx: T2<i64, 2, 2>) -> T3<f32, 2, 2, 2> {
    gather(x, idx, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn take_axis1_index1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    take(x, 1, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn take_along_axis1_f32(x: T2<f32, 2, 3>, idx: T2<i64, 2, 2>) -> T2<f32, 2, 2> {
    take_along_axis(x, idx, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn concat_axis0_f32(a: T2<f32, 2, 3>, b: T2<f32, 2, 3>) -> T2<f32, 4, 3> {
    concat(a, b, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn stack_axis1_f32(a: T2<f32, 2, 3>, b: T2<f32, 2, 3>) -> T3<f32, 2, 2, 3> {
    stack(a, b, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn split_left_reshape_f32(a: T2<f32, 2, 3>) -> T1<f32, 2> {
    let (left, _right): (T2<f32, 2, 1>, T2<f32, 2, 2>) = split(a, 1, [1, 2]);
    reshape(left)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn tile_f32(a: T2<f32, 2, 3>) -> T2<f32, 4, 3> {
    tile::<T2<f32, 4, 3>, 2>(a, [2, 1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn repeat_axis1_f32(a: T2<f32, 2, 3>) -> T2<f32, 2, 6> {
    repeat(a, 1, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn flip_f32(a: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    flip(a)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn flip_axis1_f32(a: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    flip_axes(a, [1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn roll_axis1_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 2, 3, 2> {
    roll(x, 1, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn transpose_3d_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 2, 3, 2> {
    transpose(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn transpose_axes_3d_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 3, 2, 2> {
    transpose_axes(x, [1, 0, 2])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn permute_dims_3d_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 3, 2, 2> {
    permute_dims(x, [1, 0, 2])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn swapaxes_3d_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 2, 2, 3> {
    swapaxes(x, 1, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn moveaxis_after_permute_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 2, 2, 3> {
    moveaxis(permute::<T3<f32, 3, 2, 2>, 3>(x, [1, 0, 2]), 0, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn dot_f32(v: T1<f32, 3>) -> T0<f32> {
    dot(v.clone(), v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn inner_f32(m: T2<f32, 2, 3>, v: T1<f32, 3>) -> T1<f32, 2> {
    inner(m, v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn vecdot_axis1_f32(m: T2<f32, 2, 3>) -> T1<f32, 2> {
    vecdot_axis(m.clone(), m, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn outer_f32(v: T1<f32, 3>) -> T2<f32, 3, 3> {
    outer(v.clone(), v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn matvec_f32(m: T2<f32, 2, 3>, v: T1<f32, 3>) -> T1<f32, 2> {
    matmul(m, v)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn trace_f32(square: T2<f32, 3, 3>) -> T0<f32> {
    trace(square)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn trace_axes_f32(batched_square: T3<f32, 2, 3, 3>) -> T1<f32, 2> {
    trace_axes(batched_square, 1, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn diagonal_f32(square: T2<f32, 3, 3>) -> T1<f32, 3> {
    diagonal(square)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn diagonal_axes_f32(batched_square: T3<f32, 2, 3, 3>) -> T2<f32, 2, 3> {
    diagonal_axes(batched_square, 1, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn conv2d_f32(x: T4<f32, 1, 3, 3, 1>, k: T4<f32, 2, 2, 1, 1>) -> T4<f32, 1, 2, 2, 1> {
    conv2d(x, k)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn grouped_conv2d_f32(x: T4<f32, 1, 3, 3, 2>, k: T4<f32, 2, 2, 1, 2>) -> T4<f32, 1, 2, 2, 2> {
    conv2d_options(x, k, Conv2dOptions::new().groups(2))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn max_pool2d_f32(x: T4<f32, 1, 4, 4, 1>) -> T4<f32, 1, 2, 2, 1> {
    max_pool2d(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn avg_pool2d_padded_f32(x: T4<f32, 1, 3, 3, 1>) -> T4<f32, 1, 2, 2, 1> {
    avg_pool2d_options(x, Pool2dOptions::new(2, 2).padding(1, 0, 1, 0).stride(2, 2))
}

fn main() {
    knok_build::compile_graphs!(
        affine_relu,
        elementwise_select,
        multi_output_stats,
        matmul_2x3_3x2,
        vecvec_matmul_f32,
        vecmat_matmul_f32,
        batched_matmul,
        same_batch_matmul,
        broadcast_4d_matmul,
        arange_step_i32,
        linspace_f32,
        identity_f32,
        arange_to_i32,
        arange_i32,
        zeros_like_f32,
        ones_like_f32,
        full_like_f32,
        abs_f32,
        ceil_f32,
        floor_f32,
        rint_f32,
        round_f32,
        relu_f32,
        square_f32,
        reciprocal_f32,
        exp_f32,
        exp2_f32,
        expm1_f32,
        log_f32,
        log1p_f32,
        log2_f32,
        log10_f32,
        sqrt_f32,
        isnan_f32,
        sin_f32,
        cos_f32,
        tan_f32,
        sigmoid_f32,
        minimum_f32,
        maximum_f32,
        pow_f32,
        clip_f32,
        full_like_4_f32,
        greater_f32,
        greater_equal_f32,
        less_f32,
        less_equal_f32,
        equal_f32,
        not_equal_f32,
        logical_and_bool,
        logical_or_bool,
        logical_xor_bool,
        logical_not_bool,
        sum_all_f32,
        prod_all_f32,
        mean_all_f32,
        min_all_f32,
        max_all_f32,
        var_all_f32,
        std_all_f32,
        argmin_all_f32,
        argmax_all_f32,
        sum_axis1_f32,
        mean_axis0_f32,
        argmax_axis1_f32,
        amin_all_f32,
        amax_all_f32,
        amin_axis1_f32,
        amax_axis0_f32,
        ptp_all_f32,
        ptp_axis1_f32,
        all_bool,
        any_bool,
        all_axis1_bool,
        any_axis0_bool,
        softmax_all_f32,
        softmax_axis1_f32,
        reshape_2x3_to_6,
        transpose_2x3,
        broadcast_first_row_2x3,
        unsqueeze_2x3,
        squeeze_1x2x3,
        slice_2x3,
        slice_scalar,
        pad_2x3,
        gather_axis1_f32,
        take_axis1_index1_f32,
        take_along_axis1_f32,
        concat_axis0_f32,
        stack_axis1_f32,
        split_left_reshape_f32,
        tile_f32,
        repeat_axis1_f32,
        flip_f32,
        flip_axis1_f32,
        roll_axis1_f32,
        transpose_3d_f32,
        transpose_axes_3d_f32,
        permute_dims_3d_f32,
        swapaxes_3d_f32,
        moveaxis_after_permute_f32,
        dot_f32,
        inner_f32,
        vecdot_axis1_f32,
        outer_f32,
        matvec_f32,
        trace_f32,
        trace_axes_f32,
        diagonal_f32,
        diagonal_axes_f32,
        conv2d_f32,
        grouped_conv2d_f32,
        max_pool2d_f32,
        avg_pool2d_padded_f32
    );

    knok_build::compile_mlir_models_with_options!(
        BuildOptions::default().output_file("knok_mlir_models.rs");
        imported_add4 {
            path: "../fixtures/add4.mlir",
            function: "imported.add4",
            backend: Backend::LlvmCpu,
            inputs: [x: T1<f32, 4>, y: T1<f32, 4>],
            outputs: [T1<f32, 4>],
        },
        imported_add_sub4 {
            path: "../fixtures/add_sub4.mlir",
            function: "imported.add_sub4",
            backend: Backend::LlvmCpu,
            inputs: [x: T1<f32, 4>, y: T1<f32, 4>],
            outputs: [T1<f32, 4>, T1<f32, 4>],
        },
    );
}
