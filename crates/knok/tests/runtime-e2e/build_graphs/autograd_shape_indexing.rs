use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn layout_index_loss(x: T2<f32, 2, 3>, y: T2<f32, 2, 3>) -> T0<f32> {
    let swapped: T2<f32, 3, 2> = swapaxes(x.clone(), 0, 1);
    let moved: T2<f32, 3, 2> = moveaxis(x.clone(), 0, 1);
    let flipped = flip(x.clone());
    let rolled = roll(x.clone(), 1, 1);
    let padded: T2<f32, 4, 5> = pad(x.clone(), [1, 1]);
    let sliced: T2<f32, 2, 3> = slice(padded, [1, 1]);
    let joined: T2<f32, 4, 3> = concat(x, y, 0);
    let left: T2<f32, 2, 3> = slice(joined.clone(), [0, 0]);
    let right: T2<f32, 2, 3> = slice(joined, [2, 0]);
    let stacked: T3<f32, 2, 2, 3> = stack(left, right, 0);
    let first: T2<f32, 2, 3> = take(stacked.clone(), 0, 0);
    let second: T2<f32, 2, 3> = take(stacked, 0, 1);
    let swapped_sum: T0<f32> = sum(swapped);
    let moved_sum: T0<f32> = sum(moved);
    let flipped_sum: T0<f32> = sum(flipped);
    let rolled_sum: T0<f32> = sum(rolled);
    let sliced_sum: T0<f32> = sum(sliced);
    let first_sum: T0<f32> = sum(first);
    let second_sum: T0<f32> = sum(second);
    swapped_sum + moved_sum + flipped_sum + rolled_sum + sliced_sum + first_sum + second_sum
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn tile_repeat_loss(x: T2<f32, 2, 2>) -> T0<f32> {
    let tiled: T2<f32, 4, 6> = tile(x.clone(), [2, 3]);
    let repeated: T2<f32, 2, 4> = repeat(x, 1, 2);
    let tiled_loss: T0<f32> = sum(tiled);
    let repeated_loss: T0<f32> = sum(repeated);
    tiled_loss + repeated_loss
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn tile_only_loss(x: T2<f32, 2, 2>) -> T0<f32> {
    let tiled: T2<f32, 4, 6> = tile(x, [2, 3]);
    sum(tiled)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn repeat_only_loss(x: T2<f32, 2, 2>) -> T0<f32> {
    let repeated: T2<f32, 2, 4> = repeat(x, 1, 2);
    sum(repeated)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn split_loss(x: T2<f32, 2, 3>) -> T0<f32> {
    let (left, right): (T2<f32, 2, 1>, T2<f32, 2, 2>) = split(x, 1, [1, 2]);
    let left_loss: T0<f32> = sum(square(left));
    let right_loss: T0<f32> = sum(right);
    left_loss + right_loss
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn indexed_loss(
    x: T2<f32, 2, 3>,
    gather_idx: T2<i64, 2, 2>,
    take_idx: T2<i64, 2, 2>,
) -> T0<f32> {
    let gathered: T3<f32, 2, 2, 2> = gather(x.clone(), gather_idx, 1);
    let taken: T2<f32, 2, 2> = take_along_axis(x, take_idx, 1);
    let gather_loss: T0<f32> = sum(gathered);
    let take_loss: T0<f32> = sum(taken);
    gather_loss + take_loss
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn indexed_axis0_loss(
    x: T2<f32, 2, 3>,
    gather_idx: T1<i64, 3>,
    take_idx: T2<i64, 2, 3>,
) -> T0<f32> {
    let gathered: T2<f32, 3, 3> = gather(x.clone(), gather_idx, 0);
    let taken: T2<f32, 2, 3> = take_along_axis(x, take_idx, 0);
    let gather_loss: T0<f32> = sum(gathered);
    let take_loss: T0<f32> = sum(taken);
    gather_loss + take_loss
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
pub(crate) fn argmax_indexed_loss(x: T2<f32, 2, 3>) -> T0<f32> {
    let indices: T1<i64, 2> = argmax_axis(x.clone(), 1);
    let indices: T2<i64, 2, 1> = reshape(indices);
    let taken: T2<f32, 2, 1> = take_along_axis(x, indices, 1);
    sum(taken)
}
