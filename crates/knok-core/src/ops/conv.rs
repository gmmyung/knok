use proc_macro2::Span;

use crate::typecheck::expect_numeric_element;
use crate::{Conv2dOptions, TensorType};

pub(crate) fn conv2d_result_type(
    input: &TensorType,
    kernel: &TensorType,
    options: &Conv2dOptions,
) -> syn::Result<TensorType> {
    if input.rank() != 4 || kernel.rank() != 4 || input.elem != kernel.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "conv2d expects NHWC input and HWCF kernel rank-4 tensors with the same element type",
        ));
    }
    expect_numeric_element(input.elem, "conv2d")?;
    if options.groups == 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            "conv2d groups must be non-zero",
        ));
    }
    if options.stride[0] == 0
        || options.stride[1] == 0
        || options.dilation[0] == 0
        || options.dilation[1] == 0
    {
        return Err(syn::Error::new(
            Span::call_site(),
            "conv2d stride and dilation options must be non-zero",
        ));
    }
    if kernel.shape[0] == 0 || kernel.shape[1] == 0 || kernel.shape[2] == 0 || kernel.shape[3] == 0
    {
        return Err(syn::Error::new(
            Span::call_site(),
            "conv2d kernel dimensions must be non-zero",
        ));
    }
    let input_channels = input.shape[3];
    let output_channels = kernel.shape[3];
    if input_channels == 0 || output_channels == 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            "conv2d input and output channels must be non-zero",
        ));
    }
    if input_channels % options.groups != 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "conv2d input channels must be divisible by groups, got {input_channels} channels and {} groups",
                options.groups
            ),
        ));
    }
    if output_channels % options.groups != 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "conv2d output channels must be divisible by groups, got {output_channels} channels and {} groups",
                options.groups
            ),
        ));
    }
    let expected_kernel_channels = input_channels / options.groups;
    if kernel.shape[2] != expected_kernel_channels {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "conv2d kernel input channels must equal input channels / groups, got {} and expected {}",
                kernel.shape[2], expected_kernel_channels
            ),
        ));
    }

    let effective_h = (kernel.shape[0] - 1) * options.dilation[0] + 1;
    let effective_w = (kernel.shape[1] - 1) * options.dilation[1] + 1;
    let padded_h = input.shape[1] + options.padding.top + options.padding.bottom;
    let padded_w = input.shape[2] + options.padding.left + options.padding.right;
    if padded_h < effective_h || padded_w < effective_w {
        return Err(syn::Error::new(
            Span::call_site(),
            "conv2d effective kernel spatial dimensions must fit inside the padded input",
        ));
    }

    Ok(TensorType {
        elem: input.elem,
        shape: vec![
            input.shape[0],
            (padded_h - effective_h) / options.stride[0] + 1,
            (padded_w - effective_w) / options.stride[1] + 1,
            output_channels,
        ],
    })
}
