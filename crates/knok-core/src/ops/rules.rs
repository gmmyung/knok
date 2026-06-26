use proc_macro2::Span;

use super::{conv2d_result_type, matmul_result_type, validate_permute};
use crate::{AxisSpec, CallOp, ElementType, TensorType};

pub(crate) fn infer_call_result(op: &CallOp, args: &[TensorType]) -> syn::Result<TensorType> {
    let results = infer_call_results(op, args)?;
    if results.len() != 1 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{op:?} produces {} values and cannot be used as a tensor expression",
                results.len()
            ),
        ));
    }
    Ok(results
        .into_iter()
        .next()
        .expect("single-result op produced no type"))
}

pub(crate) fn infer_call_results(op: &CallOp, args: &[TensorType]) -> syn::Result<Vec<TensorType>> {
    let ty = match op {
        CallOp::Split { axis, sections } => {
            expect_arity(op, args, 1)?;
            return split_result_types(&args[0], *axis, sections);
        }
        CallOp::Abs => {
            expect_arity(op, args, 1)?;
            let ty = args[0].clone();
            expect_numeric_element(ty.elem, "abs")?;
            Ok(ty)
        }
        CallOp::All(axis) | CallOp::Any(axis) => {
            expect_arity(op, args, 1)?;
            let input = args[0].clone();
            expect_bool_element(input.elem, "bool reductions")?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "bool reductions expect a tensor input",
                ));
            }
            reduction_output_type(&input, *axis)
        }
        CallOp::Argmax(axis) => {
            expect_arity(op, args, 1)?;
            let input = args[0].clone();
            expect_numeric_element(input.elem, "argmax")?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "argmax expects a tensor input",
                ));
            }
            expect_non_empty_reduction(&input, *axis, "argmax")?;
            let mut output = reduction_output_type(&input, *axis)?;
            output.elem = ElementType::I64;
            Ok(output)
        }
        CallOp::Exp
        | CallOp::IsNan
        | CallOp::Log
        | CallOp::Relu
        | CallOp::Sigmoid
        | CallOp::Sqrt
        | CallOp::Tanh => {
            expect_arity(op, args, 1)?;
            let ty = args[0].clone();
            expect_float(op, ty.elem)?;
            if matches!(op, CallOp::IsNan) {
                Ok(TensorType {
                    elem: ElementType::Bool,
                    shape: ty.shape,
                })
            } else {
                Ok(ty)
            }
        }
        CallOp::Greater | CallOp::GreaterEqual | CallOp::Less | CallOp::LessEqual => {
            expect_arity(op, args, 2)?;
            comparison_result_type(&args[0], &args[1])
        }
        CallOp::Equal | CallOp::NotEqual => {
            expect_arity(op, args, 2)?;
            equality_result_type(&args[0], &args[1])
        }
        CallOp::LogicalAnd | CallOp::LogicalOr | CallOp::LogicalXor => {
            expect_arity(op, args, 2)?;
            logical_result_type(&args[0], &args[1])
        }
        CallOp::LogicalNot => {
            expect_arity(op, args, 1)?;
            let ty = args[0].clone();
            expect_bool_element(ty.elem, "logical_not")?;
            Ok(ty)
        }
        CallOp::Minimum | CallOp::Maximum => {
            expect_arity(op, args, 2)?;
            expect_numeric_element(args[0].elem, "min/max ops")?;
            expect_numeric_element(args[1].elem, "min/max ops")?;
            binary_result_type(&args[0], &args[1])
        }
        CallOp::Clip => {
            expect_arity(op, args, 3)?;
            expect_numeric_element(args[0].elem, "clip")?;
            expect_numeric_element(args[1].elem, "clip")?;
            expect_numeric_element(args[2].elem, "clip")?;
            let value = binary_result_type(&args[0], &args[1])?;
            binary_result_type(&value, &args[2])
        }
        CallOp::Pow => {
            expect_arity(op, args, 2)?;
            expect_float(op, args[0].elem)?;
            expect_float(op, args[1].elem)?;
            binary_result_type(&args[0], &args[1])
        }
        CallOp::Concat(axis) => {
            expect_arity(op, args, 2)?;
            concat_result_type(&args[0], &args[1], *axis)
        }
        CallOp::Flip(axes) => {
            expect_arity(op, args, 1)?;
            validate_axis_list(&args[0], axes, "flip")?;
            Ok(args[0].clone())
        }
        CallOp::MoveAxis {
            source,
            destination,
        } => {
            expect_arity(op, args, 1)?;
            moveaxis_result_type(&args[0], *source, *destination)
        }
        CallOp::Pad { target, lows } => {
            expect_arity(op, args, 1)?;
            validate_pad(&args[0], target, lows)?;
            Ok(target.clone())
        }
        CallOp::Softmax(axis) => {
            expect_arity(op, args, 1)?;
            let ty = args[0].clone();
            expect_float(op, ty.elem)?;
            if ty.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "softmax expects a tensor input",
                ));
            }
            expect_non_empty_reduction(&ty, *axis, "softmax")?;
            Ok(ty)
        }
        CallOp::Transpose(axes) => {
            expect_arity(op, args, 1)?;
            transpose_result_type(&args[0], axes)
        }
        CallOp::Permute { target, axes } => {
            expect_arity(op, args, 1)?;
            validate_permute(&args[0], target, axes)?;
            Ok(target.clone())
        }
        CallOp::PermuteDims(axes) => {
            expect_arity(op, args, 1)?;
            permute_dims_result_type(&args[0], axes, "permute_dims")
        }
        CallOp::Repeat { axis, count } => {
            expect_arity(op, args, 1)?;
            repeat_result_type(&args[0], *axis, *count)
        }
        CallOp::Reshape(target) => {
            expect_arity(op, args, 1)?;
            let input = &args[0];
            if input.elem != target.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "reshape input and output element types must match",
                ));
            }
            if element_count(input) != element_count(target) {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "reshape element counts must match, got {} and {}",
                        element_count(input),
                        element_count(target)
                    ),
                ));
            }
            Ok(target.clone())
        }
        CallOp::Roll { axis, .. } => {
            expect_arity(op, args, 1)?;
            expect_axis(&args[0], *axis)?;
            Ok(args[0].clone())
        }
        CallOp::Broadcast(target) => {
            expect_arity(op, args, 1)?;
            let input = &args[0];
            if input.elem != target.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "broadcast input and output element types must match",
                ));
            }
            if let Err(message) = broadcast_shape(input, target) {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!("broadcast input and output shapes are incompatible: {message}"),
                ));
            }
            Ok(target.clone())
        }
        CallOp::Slice { target, starts } => {
            expect_arity(op, args, 1)?;
            validate_slice(&args[0], target, starts)?;
            Ok(target.clone())
        }
        CallOp::Squeeze(target) => {
            expect_arity(op, args, 1)?;
            validate_squeeze(&args[0], target)?;
            Ok(target.clone())
        }
        CallOp::Stack(axis) => {
            expect_arity(op, args, 2)?;
            stack_result_type(&args[0], &args[1], *axis)
        }
        CallOp::SwapAxes { axis0, axis1 } => {
            expect_arity(op, args, 1)?;
            swapaxes_result_type(&args[0], *axis0, *axis1)
        }
        CallOp::Sum(axis) => {
            expect_arity(op, args, 1)?;
            let input = args[0].clone();
            expect_numeric_element(input.elem, "sum")?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "sum expects a tensor input",
                ));
            }
            reduction_output_type(&input, *axis)
        }
        CallOp::Mean(axis) => {
            expect_arity(op, args, 1)?;
            let input = args[0].clone();
            expect_float(op, input.elem)?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "mean expects a tensor input",
                ));
            }
            expect_non_empty_reduction(&input, *axis, "mean")?;
            reduction_output_type(&input, *axis)
        }
        CallOp::Take { axis, index } => {
            expect_arity(op, args, 1)?;
            take_result_type(&args[0], *axis, *index)
        }
        CallOp::Matmul => {
            expect_arity(op, args, 2)?;
            matmul_result_type(&args[0], &args[1])
        }
        CallOp::Conv2d(options) => {
            expect_arity(op, args, 2)?;
            conv2d_result_type(&args[0], &args[1], options)
        }
        CallOp::Tile(multiples) => {
            expect_arity(op, args, 1)?;
            tile_result_type(&args[0], multiples)
        }
        CallOp::Unsqueeze(target) => {
            expect_arity(op, args, 1)?;
            validate_unsqueeze(&args[0], target)?;
            Ok(target.clone())
        }
        CallOp::Where => {
            expect_arity(op, args, 3)?;
            where_result_type(&args[0], &args[1], &args[2])
        }
        CallOp::Graph(_) => Err(syn::Error::new(
            Span::call_site(),
            "graph calls are resolved by the graph type checker",
        )),
    }?;
    Ok(vec![ty])
}

fn binary_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_numeric_element(lhs.elem, "arithmetic operators")?;
    expect_numeric_element(rhs.elem, "arithmetic operators")?;
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "elementwise operands must have the same element type, got {lhs:?} and {rhs:?}"
            ),
        ));
    }
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: lhs.elem,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("elementwise operands are not broadcast-compatible: {message}"),
            )
        })
}

fn comparison_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_numeric_element(lhs.elem, "comparison ops")?;
    expect_numeric_element(rhs.elem, "comparison ops")?;
    predicate_result_type(lhs, rhs, "comparison")
}

fn equality_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    predicate_result_type(lhs, rhs, "equality")
}

fn predicate_result_type(
    lhs: &TensorType,
    rhs: &TensorType,
    op_name: &str,
) -> syn::Result<TensorType> {
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} operands must have the same element type, got {lhs:?} and {rhs:?}"),
        ));
    }
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: ElementType::Bool,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("{op_name} operands are not broadcast-compatible: {message}"),
            )
        })
}

fn logical_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_bool_element(lhs.elem, "logical ops")?;
    expect_bool_element(rhs.elem, "logical ops")?;
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: ElementType::Bool,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("logical operands are not broadcast-compatible: {message}"),
            )
        })
}

fn where_result_type(
    condition: &TensorType,
    lhs: &TensorType,
    rhs: &TensorType,
) -> syn::Result<TensorType> {
    expect_bool_element(condition.elem, "where condition")?;
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "where value operands must have the same element type, got {lhs:?} and {rhs:?}"
            ),
        ));
    }
    let value_shape = broadcast_shape(lhs, rhs).map_err(|message| {
        syn::Error::new(
            Span::call_site(),
            format!("where value operands are not broadcast-compatible: {message}"),
        )
    })?;
    let result = TensorType {
        elem: lhs.elem,
        shape: value_shape,
    };
    let shape = broadcast_shape(condition, &result).map_err(|message| {
        syn::Error::new(
            Span::call_site(),
            format!("where condition is not broadcast-compatible with values: {message}"),
        )
    })?;
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn element_count(ty: &TensorType) -> usize {
    ty.shape.iter().product()
}

pub(crate) fn broadcast_shape_slices(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let lhs_dim = dim_from_trailing(lhs, rank, offset);
        let rhs_dim = dim_from_trailing(rhs, rank, offset);
        let dim = match (lhs_dim, rhs_dim) {
            (Some(lhs_dim), Some(rhs_dim)) if lhs_dim == rhs_dim => lhs_dim,
            (Some(1), Some(rhs_dim)) => rhs_dim,
            (Some(lhs_dim), Some(1)) => lhs_dim,
            (None, Some(dim)) | (Some(dim), None) => dim,
            (None, None) => unreachable!("rank is derived from at least one shape"),
            (Some(lhs_dim), Some(rhs_dim)) => {
                return Err(format!(
                    "dimension {offset} differs: {lhs_dim} vs {rhs_dim}"
                ));
            }
        };
        shape.push(dim);
    }
    Ok(shape)
}

fn broadcast_shape(lhs: &TensorType, rhs: &TensorType) -> Result<Vec<usize>, String> {
    broadcast_shape_slices(&lhs.shape, &rhs.shape)
}

fn dim_from_trailing(shape: &[usize], rank: usize, offset: usize) -> Option<usize> {
    let pad = rank - shape.len();
    (offset >= pad).then(|| shape[offset - pad])
}

fn validate_slice(input: &TensorType, target: &TensorType, starts: &[usize]) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "slice input and output element types must match",
        ));
    }
    if input.rank() != target.rank() || starts.len() != input.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "slice expects one start const per dimension and equal input/output rank, got input rank {}, output rank {}, starts {:?}",
                input.rank(),
                target.rank(),
                starts
            ),
        ));
    }
    for (axis, ((start, size), input_dim)) in starts
        .iter()
        .zip(&target.shape)
        .zip(&input.shape)
        .enumerate()
    {
        if *start + *size > *input_dim {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "slice dimension {axis} is out of bounds: start {start} + size {size} exceeds input dimension {input_dim}",
                ),
            ));
        }
    }
    Ok(())
}

fn take_result_type(input: &TensorType, axis: usize, index: usize) -> syn::Result<TensorType> {
    expect_axis(input, axis)?;
    if index >= input.shape[axis] {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "take index {index} is out of bounds for axis {axis} with dimension {}",
                input.shape[axis]
            ),
        ));
    }
    let mut shape = input.shape.clone();
    shape.remove(axis);
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn validate_squeeze(input: &TensorType, target: &TensorType) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "squeeze input and output element types must match",
        ));
    }
    let squeezed = input
        .shape
        .iter()
        .copied()
        .filter(|dim| *dim != 1)
        .collect::<Vec<_>>();
    if squeezed == target.shape {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "squeeze target shape {:?} does not match input shape {:?} after removing singleton dimensions",
                target.shape, input.shape
            ),
        ))
    }
}

fn validate_unsqueeze(input: &TensorType, target: &TensorType) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "unsqueeze input and output element types must match",
        ));
    }
    let target_without_singletons = target
        .shape
        .iter()
        .copied()
        .filter(|dim| *dim != 1)
        .collect::<Vec<_>>();
    let input_without_rank1_scalar = if input.shape == [1] {
        Vec::new()
    } else {
        input.shape.clone()
    };
    if target_without_singletons == input.shape
        || target_without_singletons == input_without_rank1_scalar
    {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "unsqueeze target shape {:?} must insert only singleton dimensions into input shape {:?}",
                target.shape, input.shape
            ),
        ))
    }
}

fn concat_result_type(lhs: &TensorType, rhs: &TensorType, axis: usize) -> syn::Result<TensorType> {
    if lhs.elem != rhs.elem || lhs.rank() != rhs.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            "concat expects tensors with the same element type and rank",
        ));
    }
    expect_axis(lhs, axis)?;
    let mut shape = lhs.shape.clone();
    for (dim, shape_dim) in shape.iter_mut().enumerate() {
        if dim == axis {
            *shape_dim = lhs.shape[dim] + rhs.shape[dim];
        } else if lhs.shape[dim] != rhs.shape[dim] {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "concat dimension {dim} must match outside axis {axis}, got {} and {}",
                    lhs.shape[dim], rhs.shape[dim]
                ),
            ));
        }
    }
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn stack_result_type(lhs: &TensorType, rhs: &TensorType, axis: usize) -> syn::Result<TensorType> {
    if lhs != rhs {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("stack expects matching tensor types, got {lhs:?} and {rhs:?}"),
        ));
    }
    if axis > lhs.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "stack axis {axis} is out of bounds for rank-{} tensor",
                lhs.rank()
            ),
        ));
    }
    let mut shape = lhs.shape.clone();
    shape.insert(axis, 2);
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn transpose_result_type(input: &TensorType, axes: &[usize]) -> syn::Result<TensorType> {
    if axes.is_empty() {
        let mut ty = input.clone();
        ty.shape.reverse();
        return Ok(ty);
    }
    permute_dims_result_type(input, axes, "transpose")
}

fn permute_dims_result_type(
    input: &TensorType,
    axes: &[usize],
    op_name: &str,
) -> syn::Result<TensorType> {
    validate_axis_permutation(input, axes, op_name)?;
    Ok(TensorType {
        elem: input.elem,
        shape: axes.iter().map(|axis| input.shape[*axis]).collect(),
    })
}

fn swapaxes_result_type(input: &TensorType, axis0: usize, axis1: usize) -> syn::Result<TensorType> {
    expect_axis(input, axis0)?;
    expect_axis(input, axis1)?;
    let mut shape = input.shape.clone();
    shape.swap(axis0, axis1);
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn moveaxis_result_type(
    input: &TensorType,
    source: usize,
    destination: usize,
) -> syn::Result<TensorType> {
    let axes = moveaxis_permutation(input.rank(), source, destination)?;
    Ok(TensorType {
        elem: input.elem,
        shape: axes.iter().map(|axis| input.shape[*axis]).collect(),
    })
}

fn split_result_types(
    input: &TensorType,
    axis: usize,
    sections: &[usize],
) -> syn::Result<Vec<TensorType>> {
    expect_axis(input, axis)?;
    let total: usize = sections.iter().sum();
    if total != input.shape[axis] {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "split sections {:?} sum to {total}, but axis {axis} has dimension {}",
                sections, input.shape[axis]
            ),
        ));
    }
    Ok(sections
        .iter()
        .map(|section| {
            let mut shape = input.shape.clone();
            shape[axis] = *section;
            TensorType {
                elem: input.elem,
                shape,
            }
        })
        .collect())
}

fn tile_result_type(input: &TensorType, multiples: &[usize]) -> syn::Result<TensorType> {
    if multiples.len() != input.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "tile expects one multiple per input dimension, got rank {} and multiples {:?}",
                input.rank(),
                multiples
            ),
        ));
    }
    Ok(TensorType {
        elem: input.elem,
        shape: input
            .shape
            .iter()
            .zip(multiples)
            .map(|(dim, multiple)| dim * multiple)
            .collect(),
    })
}

fn repeat_result_type(input: &TensorType, axis: usize, count: usize) -> syn::Result<TensorType> {
    expect_axis(input, axis).map_err(|_| {
        syn::Error::new(
            Span::call_site(),
            format!(
                "repeat axis {axis} is out of bounds for rank-{} tensor {:?}; repeat requires an existing axis",
                input.rank(),
                input.shape
            ),
        )
    })?;
    let mut shape = input.shape.clone();
    shape[axis] *= count;
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn validate_pad(input: &TensorType, target: &TensorType, lows: &[usize]) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "pad input and output element types must match",
        ));
    }
    if input.rank() != target.rank() || lows.len() != input.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "pad expects one low-padding const per dimension and equal input/output rank, got input rank {}, output rank {}, lows {:?}",
                input.rank(),
                target.rank(),
                lows
            ),
        ));
    }
    for (axis, ((input_dim, target_dim), low)) in
        input.shape.iter().zip(&target.shape).zip(lows).enumerate()
    {
        if *input_dim + *low > *target_dim {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "pad dimension {axis} is out of bounds: input dimension {input_dim} + low padding {low} exceeds output dimension {target_dim}",
                ),
            ));
        }
    }
    Ok(())
}

fn validate_axis_list(input: &TensorType, axes: &[usize], op_name: &str) -> syn::Result<()> {
    let axes = if axes.is_empty() {
        (0..input.rank()).collect::<Vec<_>>()
    } else {
        axes.to_vec()
    };
    let mut seen = vec![false; input.rank()];
    for axis in axes {
        if axis >= input.rank() {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "{op_name} axis {axis} is out of bounds for rank-{} tensor {:?}",
                    input.rank(),
                    input.shape
                ),
            ));
        }
        if seen[axis] {
            return Err(syn::Error::new(
                Span::call_site(),
                format!("{op_name} axes must not contain duplicates"),
            ));
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_axis_permutation(input: &TensorType, axes: &[usize], op_name: &str) -> syn::Result<()> {
    if axes.len() != input.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{op_name} expects one axis per input dimension, got rank {} and axes {:?}",
                input.rank(),
                axes
            ),
        ));
    }
    let mut seen = vec![false; input.rank()];
    for &axis in axes {
        if axis >= input.rank() || seen[axis] {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "{op_name} axes must be a permutation of 0..{}, got {:?}",
                    input.rank(),
                    axes
                ),
            ));
        }
        seen[axis] = true;
    }
    Ok(())
}

pub(crate) fn moveaxis_permutation(
    rank: usize,
    source: usize,
    destination: usize,
) -> syn::Result<Vec<usize>> {
    if source >= rank || destination >= rank {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "moveaxis source {source} and destination {destination} must be in bounds for rank-{rank} tensor"
            ),
        ));
    }
    let mut axes = (0..rank).collect::<Vec<_>>();
    let axis = axes.remove(source);
    axes.insert(destination, axis);
    Ok(axes)
}

fn reduction_output_type(input: &TensorType, axis: AxisSpec) -> syn::Result<TensorType> {
    let shape = match axis {
        AxisSpec::One(axis) => {
            expect_axis(input, axis)?;
            let mut shape = input.shape.clone();
            shape.remove(axis);
            shape
        }
        AxisSpec::All => Vec::new(),
    };
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn expect_non_empty_reduction(
    input: &TensorType,
    axis: AxisSpec,
    op_name: &str,
) -> syn::Result<()> {
    match axis {
        AxisSpec::One(axis) => {
            expect_axis(input, axis)?;
            if input.shape[axis] == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "{op_name} cannot reduce empty axis {axis} for tensor shape {:?}",
                        input.shape
                    ),
                ));
            }
        }
        AxisSpec::All if element_count(input) == 0 => {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "{op_name} cannot reduce empty tensor shape {:?}",
                    input.shape
                ),
            ));
        }
        AxisSpec::All => {}
    }
    Ok(())
}

fn expect_axis(input: &TensorType, axis: usize) -> syn::Result<()> {
    if axis < input.rank() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "axis {axis} is out of bounds for rank-{} tensor {:?}",
                input.rank(),
                input.shape
            ),
        ))
    }
}

fn expect_arity(op: &CallOp, args: &[TensorType], expected: usize) -> syn::Result<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op:?} expects {expected} arguments, got {}", args.len()),
        ))
    }
}

fn expect_float(op: &CallOp, elem: ElementType) -> syn::Result<()> {
    if elem.is_float() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op:?} supports floating-point tensors only"),
        ))
    }
}

pub(crate) fn expect_numeric_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
    if elem.is_numeric() {
        Ok(())
    } else {
        let verb = if op_name.ends_with('s') {
            "support"
        } else {
            "supports"
        };
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} {verb} numeric tensors only"),
        ))
    }
}

fn expect_bool_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
    if elem.is_bool() {
        Ok(())
    } else {
        let verb = if op_name.ends_with('s') {
            "support"
        } else {
            "supports"
        };
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} {verb} bool tensors only"),
        ))
    }
}
