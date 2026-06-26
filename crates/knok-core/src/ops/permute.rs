use proc_macro2::Span;

use crate::TensorType;

pub(crate) fn validate_permute(
    input: &TensorType,
    target: &TensorType,
    axes: &[usize],
) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "permute input and output element types must match",
        ));
    }
    if axes.len() != input.rank() || target.rank() != input.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "permute expects one axis per input dimension and equal input/output rank, got input rank {}, output rank {}, axes {:?}",
                input.rank(),
                target.rank(),
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
                    "permute axes must be a permutation of 0..{}, got {:?}",
                    input.rank(),
                    axes
                ),
            ));
        }
        seen[axis] = true;
    }
    let expected = axes
        .iter()
        .map(|axis| input.shape[*axis])
        .collect::<Vec<_>>();
    if expected != target.shape {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "permute target shape {:?} does not match input shape {:?} with axes {:?}; expected {:?}",
                target.shape, input.shape, axes, expected
            ),
        ));
    }
    Ok(())
}
