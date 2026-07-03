use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn reshape_2x3_to_6(x: T2<f32, 2, 3>) -> T1<f32, 6> {
    reshape(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn transpose_2x3(x: T2<f32, 2, 3>) -> T2<f32, 3, 2> {
    transpose(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn broadcast_first_row_2x3(x: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    broadcast::<T2<f32, 2, 3>>(take::<T1<f32, 3>>(x, 0, 0))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn unsqueeze_2x3(x: T2<f32, 2, 3>) -> T3<f32, 1, 2, 3> {
    unsqueeze(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn squeeze_1x2x3(x: T3<f32, 1, 2, 3>) -> T2<f32, 2, 3> {
    squeeze(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn slice_2x3(x: T2<f32, 2, 3>) -> T2<f32, 2, 2> {
    slice(x, [0, 1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn slice_scalar(x: T0<f32>) -> T0<f32> {
    slice(x, [])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn pad_2x3(x: T2<f32, 2, 3>) -> T2<f32, 3, 4> {
    pad(x, [1, 1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn gather_axis1_f32(x: T2<f32, 2, 3>, idx: T2<i64, 2, 2>) -> T3<f32, 2, 2, 2> {
    gather(x, idx, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn take_axis1_index1_f32(x: T2<f32, 2, 3>) -> T1<f32, 2> {
    take(x, 1, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn take_along_axis1_f32(x: T2<f32, 2, 3>, idx: T2<i64, 2, 2>) -> T2<f32, 2, 2> {
    take_along_axis(x, idx, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn concat_axis0_f32(a: T2<f32, 2, 3>, b: T2<f32, 2, 3>) -> T2<f32, 4, 3> {
    concat(a, b, 0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn stack_axis1_f32(a: T2<f32, 2, 3>, b: T2<f32, 2, 3>) -> T3<f32, 2, 2, 3> {
    stack(a, b, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn split_left_reshape_f32(a: T2<f32, 2, 3>) -> T1<f32, 2> {
    let (left, _right): (T2<f32, 2, 1>, T2<f32, 2, 2>) = split(a, 1, [1, 2]);
    reshape(left)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn tile_f32(a: T2<f32, 2, 3>) -> T2<f32, 4, 3> {
    tile::<T2<f32, 4, 3>, 2>(a, [2, 1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn repeat_axis1_f32(a: T2<f32, 2, 3>) -> T2<f32, 2, 6> {
    repeat(a, 1, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn flip_f32(a: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    flip(a)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn flip_axis1_f32(a: T2<f32, 2, 3>) -> T2<f32, 2, 3> {
    flip_axes(a, [1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn roll_axis1_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 2, 3, 2> {
    roll(x, 1, 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn transpose_3d_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 2, 3, 2> {
    transpose(x)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn transpose_axes_3d_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 3, 2, 2> {
    transpose_axes(x, [1, 0, 2])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn permute_dims_3d_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 3, 2, 2> {
    permute_dims(x, [1, 0, 2])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn swapaxes_3d_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 2, 2, 3> {
    swapaxes(x, 1, 2)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn moveaxis_after_permute_f32(x: T3<f32, 2, 3, 2>) -> T3<f32, 2, 2, 3> {
    moveaxis(permute::<T3<f32, 3, 2, 2>, 3>(x, [1, 0, 2]), 0, 2)
}
