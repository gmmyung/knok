//! Procedural macros for build-script graph tracing.
//!
//! These macros only generate Rust registration glue. Actual graph tracing and
//! IREE compilation happen when the user's `build.rs` runs.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::Parser, parse_macro_input, punctuated::Punctuated, Expr, FnArg, ItemFn, MetaNameValue,
    Pat, Path, ReturnType, Token,
};

/// Marks a build-script function as a traceable `knok` graph.
#[proc_macro_attribute]
pub fn graph(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_traced_graph(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Compiles registered graph functions with default build options.
#[proc_macro]
pub fn compile_graphs(input: TokenStream) -> TokenStream {
    let paths = parse_macro_input!(input with Punctuated::<Path, Token![,]>::parse_terminated);
    expand_compile_graphs(paths, quote!(::knok_build::BuildOptions::default())).into()
}

/// Compiles registered graph functions with an explicit [`knok_build::BuildOptions`] expression.
#[proc_macro]
pub fn compile_graphs_with_options(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as CompileGraphsWithOptions);
    let options = parsed.options;
    expand_compile_graphs(parsed.graphs, quote!(#options)).into()
}

struct CompileGraphsWithOptions {
    options: Expr,
    graphs: Punctuated<Path, Token![,]>,
}

impl syn::parse::Parse for CompileGraphsWithOptions {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let options = input.parse::<Expr>()?;
        input.parse::<Token![;]>()?;
        let graphs = Punctuated::<Path, Token![,]>::parse_terminated(input)?;
        Ok(Self { options, graphs })
    }
}

fn expand_traced_graph(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let backend = parse_backend(attr)?;
    let item_fn: ItemFn = syn::parse2(item)?;
    let name = item_fn.sig.ident.clone();
    let graph_name = name.to_string();
    let register_name = format_ident!("__knok_register_{name}");
    let inputs = item_fn
        .sig
        .inputs
        .iter()
        .map(|input| {
            let FnArg::Typed(pat_ty) = input else {
                return Err(syn::Error::new_spanned(
                    input,
                    "graph methods with self receivers are not supported",
                ));
            };
            let Pat::Ident(ident) = pat_ty.pat.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &pat_ty.pat,
                    "graph argument patterns must be simple identifiers",
                ));
            };
            let name = ident.ident.clone();
            let name_string = name.to_string();
            let ty = pat_ty.ty.clone();
            Ok(quote!(let #name = context.input::<#ty>(#name_string);))
        })
        .collect::<syn::Result<Vec<_>>>()?;
    if matches!(item_fn.sig.output, ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.ident,
            "graph functions must return a traced tensor type or tuple of traced tensor types",
        ));
    }
    let arg_names = item_fn
        .sig
        .inputs
        .iter()
        .map(|input| {
            let FnArg::Typed(pat_ty) = input else {
                unreachable!("validated above");
            };
            let Pat::Ident(ident) = pat_ty.pat.as_ref() else {
                unreachable!("validated above");
            };
            ident.ident.clone()
        })
        .collect::<Vec<_>>();

    Ok(quote! {
        #item_fn

        #[allow(non_snake_case)]
        pub(crate) fn #register_name(
            registry: &mut ::knok_build::GraphRegistry,
        ) -> ::knok_build::Result<()> {
            registry.trace(#graph_name, #backend, |context| {
                #(#inputs)*
                #name(#(#arg_names),*)
            })
        }
    })
}

fn parse_backend(attr: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let args = Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse2(attr)?;
    for arg in args {
        if arg.path.is_ident("backend") {
            let value = arg.value;
            return Ok(quote!(#value));
        }
    }
    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "missing required backend = Backend::... argument",
    ))
}

fn expand_compile_graphs(
    paths: Punctuated<Path, Token![,]>,
    options: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let calls = paths.iter().map(|path| {
        let mut register_path = path.clone();
        let last = register_path
            .segments
            .last_mut()
            .expect("path parser produced an empty path");
        let name = &last.ident;
        last.ident = format_ident!("__knok_register_{name}");
        quote!(#register_path(&mut registry)?;)
    });
    quote! {
        {
            let result: ::knok_build::Result<()> = (|| {
                let mut registry = ::knok_build::GraphRegistry::new();
                #(#calls)*
                ::knok_build::emit_registered_graphs_with_options(registry, #options)
            })();
            if let Err(error) = result {
                panic!("knok build failed: {error}");
            }
        }
    }
}
