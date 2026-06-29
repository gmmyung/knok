use proc_macro2::Span;
use syn::{spanned::Spanned, GenericArgument, Lit, Type, TypePath};

use crate::{ElementType, TensorType};

pub fn parse_tensor_type(ty: &Type) -> syn::Result<TensorType> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return Err(syn::Error::new(
            ty.span(),
            "expected Tensor0/T0 through Tensor6/T6 type",
        ));
    };
    let segment = path.segments.last().ok_or_else(|| {
        syn::Error::new(path.span(), "expected Tensor0/T0 through Tensor6/T6 type")
    })?;
    let rank = match segment.ident.to_string().as_str() {
        "Tensor0" | "T0" => 0,
        "Tensor1" | "T1" => 1,
        "Tensor2" | "T2" => 2,
        "Tensor3" | "T3" => 3,
        "Tensor4" | "T4" => 4,
        "Tensor5" | "T5" => 5,
        "Tensor6" | "T6" => 6,
        _ => {
            return Err(syn::Error::new(
                segment.ident.span(),
                "expected Tensor0<T>/T0<T>, Tensor1<T, D0>/T1<T, D0>, ..., or Tensor6<T, D0, D1, D2, D3, D4, D5>/T6<T, D0, D1, D2, D3, D4, D5>",
            ));
        }
    };
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return Err(syn::Error::new(
            segment.arguments.span(),
            "tensor type requires generic arguments",
        ));
    };
    let mut args = args.args.iter();
    let elem = parse_element_type(args.next())?;
    let mut shape = Vec::new();
    for _ in 0..rank {
        shape.push(parse_const_usize(args.next())?);
    }
    if args.next().is_some() {
        return Err(syn::Error::new(
            segment.arguments.span(),
            "too many tensor generic arguments",
        ));
    }
    Ok(TensorType { elem, shape })
}

fn parse_element_type(arg: Option<&GenericArgument>) -> syn::Result<ElementType> {
    let Some(GenericArgument::Type(Type::Path(path))) = arg else {
        return Err(syn::Error::new(
            Span::call_site(),
            supported_element_type_message(),
        ));
    };
    let Some(segment) = path.path.segments.last() else {
        return Err(syn::Error::new(
            path.span(),
            supported_element_type_message(),
        ));
    };
    for (name, elem) in [
        ("bool", ElementType::Bool),
        ("f32", ElementType::F32),
        ("f64", ElementType::F64),
        #[cfg(feature = "half")]
        ("f16", ElementType::F16),
        #[cfg(feature = "half")]
        ("bf16", ElementType::BF16),
        ("i32", ElementType::I32),
        ("i64", ElementType::I64),
    ] {
        if segment.ident == name {
            return Ok(elem);
        }
    }
    Err(syn::Error::new(
        path.span(),
        supported_element_type_message(),
    ))
}

fn supported_element_type_message() -> &'static str {
    "unsupported tensor element type; supported types are bool, f32, f64, i32, i64, and half::f16/half::bf16 when the `half` feature is enabled"
}

fn parse_const_usize(arg: Option<&GenericArgument>) -> syn::Result<usize> {
    let Some(GenericArgument::Const(syn::Expr::Lit(expr_lit))) = arg else {
        return Err(syn::Error::new(
            Span::call_site(),
            "tensor dimensions must be integer const arguments",
        ));
    };
    let Lit::Int(lit) = &expr_lit.lit else {
        return Err(syn::Error::new(
            expr_lit.span(),
            "tensor dimensions must be integer const arguments",
        ));
    };
    lit.base10_parse::<usize>()
}
