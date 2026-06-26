use knok_core::{parse_tensor_type, ElementType, TensorType};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{spanned::Spanned, FnArg, ReturnType, Type};

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

pub(crate) fn parse_return_output_types(output: &ReturnType) -> syn::Result<Vec<Type>> {
    match output {
        ReturnType::Type(_, ty) => parse_output_types(ty),
        ReturnType::Default => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "graph functions must return a Tensor type or tuple of Tensor types",
        )),
    }
}

fn parse_output_types(ty: &Type) -> syn::Result<Vec<Type>> {
    match ty {
        Type::Tuple(tuple) => {
            if tuple.elems.is_empty() {
                return Err(syn::Error::new(
                    tuple.span(),
                    "output tuple must contain at least one Tensor type",
                ));
            }
            for elem in &tuple.elems {
                parse_tensor_type(elem)?;
            }
            Ok(tuple.elems.iter().cloned().collect())
        }
        _ => {
            parse_tensor_type(ty)?;
            Ok(vec![ty.clone()])
        }
    }
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

pub(crate) fn input_name(input: &FnArg) -> syn::Result<proc_macro2::Ident> {
    let FnArg::Typed(pat_ty) = input else {
        return Err(syn::Error::new_spanned(
            input,
            "graph methods with self receivers are not supported",
        ));
    };
    let syn::Pat::Ident(ident) = pat_ty.pat.as_ref() else {
        return Err(syn::Error::new_spanned(
            &pat_ty.pat,
            "graph argument patterns must be simple identifiers",
        ));
    };
    Ok(ident.ident.clone())
}
