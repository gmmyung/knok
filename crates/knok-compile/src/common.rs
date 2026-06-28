use knok_core::{ElementType, TensorType};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub(crate) fn rust_element_type(elem: ElementType) -> TokenStream {
    match elem {
        ElementType::Bool => quote!(bool),
        ElementType::F32 => quote!(f32),
        ElementType::F64 => quote!(f64),
        ElementType::F16 => quote!(::knok::half::f16),
        ElementType::BF16 => quote!(::knok::half::bf16),
        ElementType::I32 => quote!(i32),
        ElementType::I64 => quote!(i64),
    }
}

pub(crate) fn dtype_expr(elem: ElementType) -> TokenStream {
    match elem {
        ElementType::Bool => quote!(::knok::DType::Bool),
        ElementType::F32 => quote!(::knok::DType::F32),
        ElementType::F64 => quote!(::knok::DType::F64),
        ElementType::F16 => quote!(::knok::DType::F16),
        ElementType::BF16 => quote!(::knok::DType::BF16),
        ElementType::I32 => quote!(::knok::DType::I32),
        ElementType::I64 => quote!(::knok::DType::I64),
    }
}

pub(crate) fn tensor_desc_expr(ty: &TensorType) -> TokenStream {
    let elem = dtype_expr(ty.elem);
    let dims = ty.shape.iter().copied();
    quote!(::knok::TensorDesc::new(#elem, &[#(#dims),*]))
}

pub(crate) fn mlir_result_types(outputs: &[TensorType]) -> String {
    let types = outputs
        .iter()
        .map(TensorType::mlir_type)
        .collect::<Vec<_>>()
        .join(", ");
    if outputs.len() == 1 {
        types
    } else {
        format!("({types})")
    }
}

pub(crate) fn runtime_input_variant(elem: ElementType) -> proc_macro2::Ident {
    match elem {
        ElementType::Bool => format_ident!("Bool"),
        ElementType::F32 => format_ident!("F32"),
        ElementType::F64 => format_ident!("F64"),
        ElementType::F16 => format_ident!("F16"),
        ElementType::BF16 => format_ident!("BF16"),
        ElementType::I32 => format_ident!("I32"),
        ElementType::I64 => format_ident!("I64"),
    }
}
