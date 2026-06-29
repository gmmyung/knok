use knok_core::{CallOp, Expr};

use super::{
    context::{tuple_id, TraceVars},
    expr::{call_output, unary},
    tensor::TraceTensor,
};

pub fn reshape<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(
        CallOp::Reshape(Output::tensor_type()),
        vec![value.into_expr()],
    )
}

pub fn broadcast<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(
        CallOp::Broadcast(Output::tensor_type()),
        vec![value.into_expr()],
    )
}

pub fn squeeze<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(
        CallOp::Squeeze(Output::tensor_type()),
        vec![value.into_expr()],
    )
}

pub fn unsqueeze<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(
        CallOp::Unsqueeze(Output::tensor_type()),
        vec![value.into_expr()],
    )
}

pub fn slice<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    starts: [usize; N],
) -> Output {
    trace_slice(value, &starts)
}

fn trace_slice<Output: TraceTensor>(value: impl TraceTensor, starts: &[usize]) -> Output {
    call_output(
        CallOp::Slice {
            target: Output::tensor_type(),
            starts: starts.to_vec(),
        },
        vec![value.into_expr()],
    )
}

fn trace_pad<Output: TraceTensor>(value: impl TraceTensor, lows: &[usize]) -> Output {
    call_output(
        CallOp::Pad {
            target: Output::tensor_type(),
            lows: lows.to_vec(),
        },
        vec![value.into_expr()],
    )
}

pub fn pad<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    lows: [usize; N],
) -> Output {
    trace_pad(value, &lows)
}

pub fn gather<Output: TraceTensor>(
    value: impl TraceTensor,
    indices: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_gather(value, indices, axis)
}

fn trace_gather<Output: TraceTensor>(
    value: impl TraceTensor,
    indices: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(
        CallOp::Gather {
            target: Output::tensor_type(),
            axis,
        },
        vec![value.into_expr(), indices.into_expr()],
    )
}

pub fn take<Output: TraceTensor>(value: impl TraceTensor, axis: usize, index: usize) -> Output {
    trace_take(value, axis, index)
}

fn trace_take<Output: TraceTensor>(value: impl TraceTensor, axis: usize, index: usize) -> Output {
    call_output(CallOp::Take { axis, index }, vec![value.into_expr()])
}

pub fn take_along_axis<Output: TraceTensor>(
    value: impl TraceTensor,
    indices: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_take_along_axis(value, indices, axis)
}

fn trace_take_along_axis<Output: TraceTensor>(
    value: impl TraceTensor,
    indices: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(
        CallOp::TakeAlongAxis { axis },
        vec![value.into_expr(), indices.into_expr()],
    )
}

pub fn concat<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_concat(lhs, rhs, axis)
}

fn trace_concat<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(CallOp::Concat(axis), vec![lhs.into_expr(), rhs.into_expr()])
}

pub fn stack<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_stack(lhs, rhs, axis)
}

fn trace_stack<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(CallOp::Stack(axis), vec![lhs.into_expr(), rhs.into_expr()])
}

pub fn split<Output: TraceVars, const N: usize>(
    value: impl TraceTensor,
    axis: usize,
    sections: [usize; N],
) -> Output {
    trace_split(value, axis, &sections)
}

fn trace_split<Output: TraceVars>(
    value: impl TraceTensor,
    axis: usize,
    sections: &[usize],
) -> Output {
    if sections.len() != Output::var_count() {
        panic!(
            "split sections produce {} outputs, but the requested Rust output type has {} values",
            sections.len(),
            Output::var_count()
        );
    }
    let value = Expr::Call {
        op: CallOp::Split {
            axis,
            sections: sections.to_vec(),
        },
        args: vec![value.into_expr()],
    };
    Output::from_tuple_expr(tuple_id(), value)
}

pub fn tile<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    multiples: [usize; N],
) -> Output {
    trace_tile(value, &multiples)
}

fn trace_tile<Output: TraceTensor>(value: impl TraceTensor, multiples: &[usize]) -> Output {
    call_output(CallOp::Tile(multiples.to_vec()), vec![value.into_expr()])
}

pub fn repeat<Output: TraceTensor>(value: impl TraceTensor, axis: usize, count: usize) -> Output {
    trace_repeat(value, axis, count)
}

fn trace_repeat<Output: TraceTensor>(value: impl TraceTensor, axis: usize, count: usize) -> Output {
    call_output(CallOp::Repeat { axis, count }, vec![value.into_expr()])
}

pub fn flip<T: TraceTensor>(value: T) -> T {
    trace_flip(value)
}

fn trace_flip<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Flip(Vec::new()), value)
}

pub fn flip_axes<T: TraceTensor, const N: usize>(value: T, axes: [usize; N]) -> T {
    trace_flip_axes(value, &axes)
}

fn trace_flip_axes<T: TraceTensor>(value: T, axes: &[usize]) -> T {
    unary(CallOp::Flip(axes.to_vec()), value)
}

pub fn roll<T: TraceTensor>(value: T, axis: usize, shift: usize) -> T {
    trace_roll(value, axis, shift)
}

fn trace_roll<T: TraceTensor>(value: T, axis: usize, shift: usize) -> T {
    unary(CallOp::Roll { axis, shift }, value)
}

pub fn transpose<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_transpose(value)
}

fn trace_transpose<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(CallOp::Transpose(Vec::new()), vec![value.into_expr()])
}

pub fn transpose_axes<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    axes: [usize; N],
) -> Output {
    trace_transpose_axes(value, &axes)
}

fn trace_transpose_axes<Output: TraceTensor>(value: impl TraceTensor, axes: &[usize]) -> Output {
    call_output(CallOp::Transpose(axes.to_vec()), vec![value.into_expr()])
}

pub fn permute<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    axes: [usize; N],
) -> Output {
    trace_permute(value, &axes)
}

fn trace_permute<Output: TraceTensor>(value: impl TraceTensor, axes: &[usize]) -> Output {
    call_output(
        CallOp::Permute {
            target: Output::tensor_type(),
            axes: axes.to_vec(),
        },
        vec![value.into_expr()],
    )
}

pub fn permute_dims<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    axes: [usize; N],
) -> Output {
    trace_permute_dims(value, &axes)
}

fn trace_permute_dims<Output: TraceTensor>(value: impl TraceTensor, axes: &[usize]) -> Output {
    call_output(CallOp::PermuteDims(axes.to_vec()), vec![value.into_expr()])
}

pub fn swapaxes<Output: TraceTensor>(
    value: impl TraceTensor,
    axis0: usize,
    axis1: usize,
) -> Output {
    trace_swapaxes(value, axis0, axis1)
}

fn trace_swapaxes<Output: TraceTensor>(
    value: impl TraceTensor,
    axis0: usize,
    axis1: usize,
) -> Output {
    call_output(CallOp::SwapAxes { axis0, axis1 }, vec![value.into_expr()])
}

pub fn moveaxis<Output: TraceTensor>(
    value: impl TraceTensor,
    source: usize,
    destination: usize,
) -> Output {
    trace_moveaxis(value, source, destination)
}

fn trace_moveaxis<Output: TraceTensor>(
    value: impl TraceTensor,
    source: usize,
    destination: usize,
) -> Output {
    call_output(
        CallOp::MoveAxis {
            source,
            destination,
        },
        vec![value.into_expr()],
    )
}
