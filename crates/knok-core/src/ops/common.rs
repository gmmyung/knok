use proc_macro2::Span;

use crate::{AxisSpec, CallOp, ElementType, TensorType};

pub(crate) fn element_count(ty: &TensorType) -> usize {
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

pub(crate) fn broadcast_shape(lhs: &TensorType, rhs: &TensorType) -> Result<Vec<usize>, String> {
    broadcast_shape_slices(&lhs.shape, &rhs.shape)
}

fn dim_from_trailing(shape: &[usize], rank: usize, offset: usize) -> Option<usize> {
    let pad = rank - shape.len();
    (offset >= pad).then(|| shape[offset - pad])
}

pub(crate) fn reduction_output_type(input: &TensorType, axis: AxisSpec) -> syn::Result<TensorType> {
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

pub(crate) fn expect_non_empty_reduction(
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

pub(crate) fn expect_axis(input: &TensorType, axis: usize) -> syn::Result<()> {
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

pub(crate) fn expect_arity(op: &CallOp, args: &[TensorType], expected: usize) -> syn::Result<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{} expects {expected} arguments, got {}",
                op.name(),
                args.len()
            ),
        ))
    }
}

pub(crate) fn expect_float(op: &CallOp, elem: ElementType) -> syn::Result<()> {
    if elem.is_float() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{} supports floating-point tensors only", op.name()),
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

pub(crate) fn expect_ordered_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
    if elem.is_numeric() || elem.is_bool() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} supports ordered tensor elements only"),
        ))
    }
}

pub(crate) fn expect_index_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
    if matches!(elem, ElementType::I32 | ElementType::I64) {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} must be i32 or i64"),
        ))
    }
}

pub(crate) fn expect_bool_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
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
