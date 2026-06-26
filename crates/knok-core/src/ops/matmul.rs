use proc_macro2::Span;

use crate::typecheck::{broadcast_shape_slices, expect_numeric_element};
use crate::TensorType;

pub(crate) fn matmul_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "matmul expects operands with the same element type",
        ));
    }
    expect_numeric_element(lhs.elem, "matmul")?;
    if lhs.rank() == 0 || rhs.rank() == 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            "matmul expects operands with rank at least 1",
        ));
    }
    if lhs.rank() > 4 || rhs.rank() > 4 {
        return Err(syn::Error::new(
            Span::call_site(),
            "matmul currently supports ranks 1 through 4",
        ));
    }

    let lhs_is_vector = lhs.rank() == 1;
    let rhs_is_vector = rhs.rank() == 1;
    let lhs_k = if lhs_is_vector {
        lhs.shape[0]
    } else {
        lhs.shape[lhs.rank() - 1]
    };
    let rhs_k = if rhs_is_vector {
        rhs.shape[0]
    } else {
        rhs.shape[rhs.rank() - 2]
    };
    if lhs_k != rhs_k {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("matmul inner dimensions must match, got {lhs_k} and {rhs_k}"),
        ));
    }

    let lhs_batch = if lhs.rank() > 2 {
        &lhs.shape[..lhs.rank() - 2]
    } else {
        &[]
    };
    let rhs_batch = if rhs.rank() > 2 {
        &rhs.shape[..rhs.rank() - 2]
    } else {
        &[]
    };
    let mut shape = broadcast_shape_slices(lhs_batch, rhs_batch).map_err(|message| {
        syn::Error::new(
            Span::call_site(),
            format!("matmul batch dimensions are not broadcastable: {message}"),
        )
    })?;

    if !lhs_is_vector {
        shape.push(lhs.shape[lhs.rank() - 2]);
    }
    if !rhs_is_vector {
        shape.push(rhs.shape[rhs.rank() - 1]);
    }

    if shape.len() > 4 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "matmul result rank {} exceeds the current Tensor4 limit",
                shape.len()
            ),
        ));
    }

    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}
