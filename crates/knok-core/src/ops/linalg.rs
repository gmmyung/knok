use proc_macro2::Span;

use super::expect_numeric_element;
use crate::TensorType;

const MAX_TENSOR_RANK: usize = 6;

pub(crate) fn dot_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_same_numeric_element(lhs, rhs, "dot")?;
    if lhs.rank() != 1 || rhs.rank() != 1 {
        return Err(syn::Error::new(
            Span::call_site(),
            "dot expects rank-1 vector operands",
        ));
    }
    expect_matching_dim(lhs.shape[0], rhs.shape[0], "dot vector dimensions")?;
    Ok(TensorType {
        elem: lhs.elem,
        shape: Vec::new(),
    })
}

pub(crate) fn vecdot_result_type(
    lhs: &TensorType,
    rhs: &TensorType,
    axis: Option<usize>,
) -> syn::Result<TensorType> {
    expect_same_numeric_element(lhs, rhs, "vecdot")?;
    if lhs.rank() == 0 || rhs.rank() == 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            "vecdot expects tensor operands with rank at least 1",
        ));
    }
    if lhs.shape != rhs.shape {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("vecdot expects operands with identical shapes, got {lhs:?} and {rhs:?}"),
        ));
    }
    let axis = axis.unwrap_or(lhs.rank() - 1);
    expect_axis(lhs, axis, "vecdot")?;
    let mut shape = lhs.shape.clone();
    shape.remove(axis);
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

pub(crate) fn inner_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_same_numeric_element(lhs, rhs, "inner")?;
    if lhs.rank() == 0 || rhs.rank() == 0 {
        let shape = if lhs.rank() == 0 {
            rhs.shape.clone()
        } else {
            lhs.shape.clone()
        };
        return Ok(TensorType {
            elem: lhs.elem,
            shape,
        });
    }
    expect_matching_dim(
        lhs.shape[lhs.rank() - 1],
        rhs.shape[rhs.rank() - 1],
        "inner contracted dimensions",
    )?;
    let mut shape = lhs.shape[..lhs.rank() - 1].to_vec();
    shape.extend_from_slice(&rhs.shape[..rhs.rank() - 1]);
    if shape.len() > MAX_TENSOR_RANK {
        return Err(result_rank_error("inner", shape.len()));
    }
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

pub(crate) fn outer_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_same_numeric_element(lhs, rhs, "outer")?;
    Ok(TensorType {
        elem: lhs.elem,
        shape: vec![element_count(lhs), element_count(rhs)],
    })
}

pub(crate) fn trace_result_type(
    input: &TensorType,
    axes: Option<[usize; 2]>,
) -> syn::Result<TensorType> {
    expect_numeric_element(input.elem, "trace")?;
    let [axis0, axis1] = default_axis_pair(input, axes, "trace")?;
    expect_matching_dim(
        input.shape[axis0],
        input.shape[axis1],
        "trace diagonal dimensions",
    )?;
    Ok(TensorType {
        elem: input.elem,
        shape: remove_axes_shape(input, axis0, axis1),
    })
}

pub(crate) fn diagonal_result_type(
    input: &TensorType,
    axes: Option<[usize; 2]>,
) -> syn::Result<TensorType> {
    let [axis0, axis1] = default_axis_pair(input, axes, "diagonal")?;
    expect_matching_dim(
        input.shape[axis0],
        input.shape[axis1],
        "diagonal dimensions",
    )?;
    let mut shape = remove_axes_shape(input, axis0, axis1);
    shape.push(input.shape[axis0]);
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn expect_same_numeric_element(
    lhs: &TensorType,
    rhs: &TensorType,
    op_name: &str,
) -> syn::Result<()> {
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} expects operands with the same element type"),
        ));
    }
    expect_numeric_element(lhs.elem, op_name)
}

fn default_axis_pair(
    input: &TensorType,
    axes: Option<[usize; 2]>,
    op_name: &str,
) -> syn::Result<[usize; 2]> {
    if input.rank() < 2 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} expects a tensor input with rank at least 2"),
        ));
    }
    let axes = axes.unwrap_or([input.rank() - 2, input.rank() - 1]);
    expect_axis(input, axes[0], op_name)?;
    expect_axis(input, axes[1], op_name)?;
    if axes[0] == axes[1] {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} axes must be distinct"),
        ));
    }
    Ok(axes)
}

fn expect_axis(input: &TensorType, axis: usize, op_name: &str) -> syn::Result<()> {
    if axis < input.rank() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{op_name} axis {axis} is out of bounds for rank-{} tensor {:?}",
                input.rank(),
                input.shape
            ),
        ))
    }
}

fn expect_matching_dim(lhs: usize, rhs: usize, name: &str) -> syn::Result<()> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{name} must match, got {lhs} and {rhs}"),
        ))
    }
}

fn remove_axes_shape(input: &TensorType, axis0: usize, axis1: usize) -> Vec<usize> {
    input
        .shape
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| (axis != axis0 && axis != axis1).then_some(*dim))
        .collect()
}

fn element_count(input: &TensorType) -> usize {
    input.shape.iter().copied().product()
}

fn result_rank_error(op_name: &str, rank: usize) -> syn::Error {
    syn::Error::new(
        Span::call_site(),
        format!("{op_name} result rank {rank} exceeds the current Tensor6 limit"),
    )
}
