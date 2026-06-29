//! Procedural macros for build-script graph tracing.
//!
//! These macros only generate Rust registration glue. Actual graph tracing and
//! IREE compilation happen when the user's `build.rs` runs.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    braced, bracketed, parse::Parser, parse_macro_input, punctuated::Punctuated, Expr, FnArg,
    Ident, ItemFn, LitStr, MetaNameValue, Pat, Path, ReturnType, Token, Type,
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

/// Compiles external MLIR files with default build options.
#[proc_macro]
pub fn compile_mlir_models(input: TokenStream) -> TokenStream {
    let models =
        parse_macro_input!(input with Punctuated::<MlirModelSpec, Token![,]>::parse_terminated);
    expand_compile_mlir_models(models, quote!(::knok_build::BuildOptions::default())).into()
}

/// Compiles external MLIR files with an explicit [`knok_build::BuildOptions`] expression.
#[proc_macro]
pub fn compile_mlir_models_with_options(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as CompileMlirModelsWithOptions);
    let options = parsed.options;
    expand_compile_mlir_models(parsed.models, quote!(#options)).into()
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

struct CompileMlirModelsWithOptions {
    options: Expr,
    models: Punctuated<MlirModelSpec, Token![,]>,
}

impl syn::parse::Parse for CompileMlirModelsWithOptions {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let options = input.parse::<Expr>()?;
        input.parse::<Token![;]>()?;
        let models = Punctuated::<MlirModelSpec, Token![,]>::parse_terminated(input)?;
        Ok(Self { options, models })
    }
}

struct MlirModelSpec {
    name: Ident,
    path: LitStr,
    function: LitStr,
    backend: Expr,
    inputs: Vec<MlirModelInput>,
    outputs: Vec<Type>,
}

struct MlirModelInput {
    name: Ident,
    ty: Type,
}

impl syn::parse::Parse for MlirModelSpec {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let name = input.parse::<Ident>()?;
        let content;
        braced!(content in input);

        let mut path = None;
        let mut function = None;
        let mut backend = None;
        let mut inputs = None;
        let mut outputs = None;

        while !content.is_empty() {
            let key = content.parse::<Ident>()?;
            content.parse::<Token![:]>()?;
            match key.to_string().as_str() {
                "path" => path = Some(content.parse::<LitStr>()?),
                "function" => function = Some(content.parse::<LitStr>()?),
                "backend" => backend = Some(content.parse::<Expr>()?),
                "inputs" => {
                    let inner;
                    bracketed!(inner in content);
                    let parsed = Punctuated::<MlirModelInput, Token![,]>::parse_terminated(&inner)?;
                    inputs = Some(parsed.into_iter().collect());
                }
                "outputs" => {
                    let inner;
                    bracketed!(inner in content);
                    let parsed = Punctuated::<Type, Token![,]>::parse_terminated(&inner)?;
                    outputs = Some(parsed.into_iter().collect());
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected one of path, function, backend, inputs, or outputs",
                    ));
                }
            }
            let _ = content.parse::<Token![,]>();
        }

        Ok(Self {
            name,
            path: path.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing path field")
            })?,
            function: function.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing function field")
            })?,
            backend: backend.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing backend field")
            })?,
            inputs: inputs.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing inputs field")
            })?,
            outputs: outputs.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing outputs field")
            })?,
        })
    }
}

impl syn::parse::Parse for MlirModelInput {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let name = input.parse::<Ident>()?;
        input.parse::<Token![:]>()?;
        let ty = input.parse::<Type>()?;
        Ok(Self { name, ty })
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

fn expand_compile_mlir_models(
    models: Punctuated<MlirModelSpec, Token![,]>,
    options: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let models = models.into_iter().map(mlir_model_tokens);
    quote! {
        {
            let result: ::knok_build::Result<()> = (|| {
                let models = vec![#(#models),*];
                ::knok_build::emit_mlir_models_with_options(models, #options)
            })();
            if let Err(error) = result {
                panic!("knok build failed: {error}");
            }
        }
    }
}

fn mlir_model_tokens(model: MlirModelSpec) -> proc_macro2::TokenStream {
    let name = model.name.to_string();
    let path = model.path;
    let function = model.function;
    let backend = model.backend;
    let inputs = model.inputs.into_iter().map(|input| {
        let name = input.name.to_string();
        let ty = tensor_type_tokens(&input.ty);
        quote! {
            ::knok_build::__private::Input {
                name: #name.into(),
                ty: #ty,
            }
        }
    });
    let outputs = model.outputs.iter().map(tensor_type_tokens);
    quote! {
        ::knok_build::MlirModel::new(
            #name,
            #path,
            #function,
            #backend,
            vec![#(#inputs),*],
            vec![#(#outputs),*],
        )
    }
}

fn tensor_type_tokens(ty: &Type) -> proc_macro2::TokenStream {
    match knok_core::parse_tensor_type(ty) {
        Ok(ty) => {
            let elem = element_type_tokens(ty.elem);
            let shape = ty.shape.iter();
            quote! {
                ::knok_build::__private::TensorType {
                    elem: #elem,
                    shape: vec![#(#shape),*],
                }
            }
        }
        Err(error) => error.to_compile_error(),
    }
}

fn element_type_tokens(elem: knok_core::ElementType) -> proc_macro2::TokenStream {
    match elem {
        knok_core::ElementType::Bool => quote!(::knok_build::__private::ElementType::Bool),
        knok_core::ElementType::F32 => quote!(::knok_build::__private::ElementType::F32),
        knok_core::ElementType::F64 => quote!(::knok_build::__private::ElementType::F64),
        knok_core::ElementType::F16 => quote!(::knok_build::__private::ElementType::F16),
        knok_core::ElementType::BF16 => quote!(::knok_build::__private::ElementType::BF16),
        knok_core::ElementType::I32 => quote!(::knok_build::__private::ElementType::I32),
        knok_core::ElementType::I64 => quote!(::knok_build::__private::ElementType::I64),
    }
}
