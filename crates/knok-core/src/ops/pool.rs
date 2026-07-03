use proc_macro2::Span;

use super::expect_numeric_element;
use crate::{Pool2dOptions, TensorType};

pub(crate) fn max_pool2d_result_type(
    input: &TensorType,
    options: &Pool2dOptions,
) -> syn::Result<TensorType> {
    pool2d_result_type(input, options, "max_pool2d")
}

pub(crate) fn avg_pool2d_result_type(
    input: &TensorType,
    options: &Pool2dOptions,
) -> syn::Result<TensorType> {
    if !input.elem.is_float() {
        return Err(syn::Error::new(
            Span::call_site(),
            "avg_pool2d expects a floating-point element type",
        ));
    }
    pool2d_result_type(input, options, "avg_pool2d")
}

fn pool2d_result_type(
    input: &TensorType,
    options: &Pool2dOptions,
    op_name: &str,
) -> syn::Result<TensorType> {
    if input.rank() != 4 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} expects an NHWC rank-4 input tensor"),
        ));
    }
    expect_numeric_element(input.elem, op_name)?;
    if options.kernel[0] == 0 || options.kernel[1] == 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} kernel dimensions must be non-zero"),
        ));
    }
    if options.stride[0] == 0
        || options.stride[1] == 0
        || options.dilation[0] == 0
        || options.dilation[1] == 0
    {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} stride and dilation options must be non-zero"),
        ));
    }
    if input.shape[0] == 0 || input.shape[3] == 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} batch and channel dimensions must be non-zero"),
        ));
    }

    let effective_h = (options.kernel[0] - 1) * options.dilation[0] + 1;
    let effective_w = (options.kernel[1] - 1) * options.dilation[1] + 1;
    let padded_h = input.shape[1] + options.padding.top + options.padding.bottom;
    let padded_w = input.shape[2] + options.padding.left + options.padding.right;
    if padded_h < effective_h || padded_w < effective_w {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{op_name} effective kernel spatial dimensions must fit inside the padded input"
            ),
        ));
    }

    Ok(TensorType {
        elem: input.elem,
        shape: vec![
            input.shape[0],
            (padded_h - effective_h) / options.stride[0] + 1,
            (padded_w - effective_w) / options.stride[1] + 1,
            input.shape[3],
        ],
    })
}
