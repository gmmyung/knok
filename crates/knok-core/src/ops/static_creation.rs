use proc_macro2::Span;

use crate::{static_arange_literals, static_eye_literals, static_linspace_literals, CallOp, Expr};

pub(crate) fn validate_static_creation_call(op: &CallOp, args: &[Expr]) -> syn::Result<()> {
    match op {
        CallOp::Arange(target) => static_arange_literals(target, args)
            .map(|_| ())
            .map_err(|message| syn::Error::new(Span::call_site(), message)),
        CallOp::Linspace(target) => static_linspace_literals(target, args)
            .map(|_| ())
            .map_err(|message| syn::Error::new(Span::call_site(), message)),
        CallOp::Eye(target) => {
            if !args.is_empty() {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!("Eye expects 0 arguments, got {}", args.len()),
                ));
            }
            static_eye_literals(target)
                .map(|_| ())
                .map_err(|message| syn::Error::new(Span::call_site(), message))
        }
        _ => Ok(()),
    }
}
