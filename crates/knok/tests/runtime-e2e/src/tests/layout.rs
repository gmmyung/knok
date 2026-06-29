use knok::tensor::*;

use crate::{common::*, graphs};

#[test]
fn layout_ops_run() {
    let a = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b = Tensor2::from_array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);
    let x = Tensor3::from_array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    ]);

    let concat = graphs::concat_axis0_f32::call(a.clone(), b.clone()).unwrap();
    let stack = graphs::stack_axis1_f32::call(a.clone(), b).unwrap();
    let split_reshape = graphs::split_left_reshape_f32::call(a.clone()).unwrap();
    let tile = graphs::tile_f32::call(a.clone()).unwrap();
    let repeat = graphs::repeat_axis1_f32::call(a.clone()).unwrap();
    let flip = graphs::flip_f32::call(a.clone()).unwrap();
    assert_close(
        concat.as_slice(),
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    assert_close(
        stack.as_slice(),
        &[
            1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0,
        ],
    );
    assert_close(split_reshape.as_slice(), &[1.0, 4.0]);
    assert_close(
        tile.as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    assert_close(
        repeat.as_slice(),
        &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
    );
    assert_close(flip.as_slice(), &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

    let flip_axes = graphs::flip_axis1_f32::call(a).unwrap();
    let roll = graphs::roll_axis1_f32::call(x.clone()).unwrap();
    let transpose = graphs::transpose_3d_f32::call(x.clone()).unwrap();
    let transpose_axes = graphs::transpose_axes_3d_f32::call(x.clone()).unwrap();
    let permute_dims = graphs::permute_dims_3d_f32::call(x.clone()).unwrap();
    let swapaxes = graphs::swapaxes_3d_f32::call(x.clone()).unwrap();
    assert_close(flip_axes.as_slice(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    assert_close(
        roll.as_slice(),
        &[
            5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0,
        ],
    );
    assert_close(
        transpose.as_slice(),
        &[
            1.0, 7.0, 3.0, 9.0, 5.0, 11.0, 2.0, 8.0, 4.0, 10.0, 6.0, 12.0,
        ],
    );
    assert_close(
        transpose_axes.as_slice(),
        &[
            1.0, 2.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 5.0, 6.0, 11.0, 12.0,
        ],
    );
    assert_close(permute_dims.as_slice(), transpose_axes.as_slice());
    assert_close(
        swapaxes.as_slice(),
        &[
            1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0, 8.0, 10.0, 12.0,
        ],
    );

    let moved = graphs::moveaxis_after_permute_f32::call(x).unwrap();
    assert_close(
        moved.as_slice(),
        &[
            1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0, 8.0, 10.0, 12.0,
        ],
    );
}
