use std::{env, fs, path::Path};

use knok_core::parse_tensor_type;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    parse2,
    punctuated::Punctuated,
    Ident, LitStr, Token, Type,
};

use crate::{
    backend::{parse_backend_array, parse_backend_expr, reject_duplicate_drivers, BackendSpec},
    common::{runtime_input_variant, rust_element_type, tensor_desc_expr},
    compile::compile_mlir_variants,
    mlir_signature::validate_mlir_model_signature,
};

/// Expands a `knok::mlir_model!` declaration.
///
/// The generated module embeds one or more compiled VMFB variants and exposes
/// typed invocation helpers when the macro input declares input and output
/// tensor types.
pub fn expand_mlir_model(input: TokenStream) -> TokenStream {
    match expand_mlir_model_result(input) {
        Ok(tokens) => tokens,
        Err(error) => error.to_compile_error(),
    }
}

fn expand_mlir_model_result(input: TokenStream) -> syn::Result<TokenStream> {
    let model: MlirModel = parse2(input)?;
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_err(|error| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("CARGO_MANIFEST_DIR is not set: {error}"),
        )
    })?;
    let path = Path::new(&manifest_dir).join(model.path.value());
    let mlir = fs::read_to_string(&path).map_err(|error| {
        syn::Error::new(
            model.path.span(),
            format!("failed to read MLIR file `{}`: {error}", path.display()),
        )
    })?;
    if let (Some(inputs), Some(outputs)) = (&model.inputs, &model.outputs) {
        let expected_inputs = inputs
            .iter()
            .map(parse_tensor_type)
            .collect::<syn::Result<Vec<_>>>()?;
        let expected_outputs = outputs
            .iter()
            .map(parse_tensor_type)
            .collect::<syn::Result<Vec<_>>>()?;
        validate_mlir_model_signature(
            &mlir,
            &model.function.value(),
            &expected_inputs,
            &expected_outputs,
        )?;
    }
    let compiled = compile_mlir_variants(&model.backend_specs, &mlir).map_err(|error| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("failed to compile MLIR file `{}`: {error}", path.display()),
        )
    })?;
    let module_name = model.name;
    let function_name = model.function.value();
    let input_descs = model
        .inputs
        .as_ref()
        .map(|types| {
            types
                .iter()
                .map(|ty| parse_tensor_type(ty).map(|tensor_ty| tensor_desc_expr(&tensor_ty)))
                .collect::<syn::Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let output_descs = model
        .outputs
        .as_ref()
        .map(|types| {
            types
                .iter()
                .map(|ty| parse_tensor_type(ty).map(|tensor_ty| tensor_desc_expr(&tensor_ty)))
                .collect::<syn::Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let input_types = model.inputs.unwrap_or_default();
    let output_types = model.outputs;
    let typed_scope_import = output_types.as_ref().map(|_| {
        quote!(
            use super::*;
        )
    });
    let typed_invoke = if let Some(output_types) = output_types {
        let input_names = (0..input_types.len())
            .map(|index| format_ident!("input{index}"))
            .collect::<Vec<_>>();
        let runtime_inputs = input_types
            .iter()
            .zip(input_names.iter())
            .map(|(ty, input_name)| {
                parse_tensor_type(ty).map(|tensor_ty| {
                    let variant = runtime_input_variant(tensor_ty.elem);
                    quote!(::knok::runtime::raw::Input::#variant(<#ty>::SHAPE, #input_name.as_slice()))
                })
            })
            .collect::<syn::Result<Vec<_>>>()?;
        let output_tensor_types = output_types
            .iter()
            .map(parse_tensor_type)
            .collect::<syn::Result<Vec<_>>>()?;
        let output_ty = if output_types.len() == 1 {
            let output_ty = &output_types[0];
            quote!(#output_ty)
        } else {
            quote!((#(#output_types),*))
        };
        let invoke_body = if output_types.len() == 1 {
            let output_ty = &output_types[0];
            let output_elem_ty = rust_element_type(output_tensor_types[0].elem);
            quote! {
                let output = ::knok::__private::invoke_one_with_engine::<#output_elem_ty>(
                    engine,
                    artifact(),
                    &[#(#runtime_inputs),*],
                )?;
                <#output_ty>::from_vec(output)
            }
        } else {
            let output_reads = output_types
                .iter()
                .zip(output_tensor_types.iter())
                .enumerate()
                .map(|(index, (output_ty, output_tensor_ty))| {
                    let output_elem_ty = rust_element_type(output_tensor_ty.elem);
                    quote!(<#output_ty>::from_vec(outputs.read::<#output_elem_ty>(#index)?)?)
                });
            quote! {
                let outputs = engine.invoke(artifact(), &[#(#runtime_inputs),*])?;
                Ok((#(#output_reads),*))
            }
        };
        Some(quote! {
            pub fn invoke_run(
                engine: &::knok::Engine,
                #(#input_names: #input_types),*
            ) -> ::knok::Result<#output_ty> {
                #invoke_body
            }

            pub fn invoke(#(#input_names: #input_types),*) -> ::knok::Result<#output_ty> {
                let artifact = artifact();
                let engine = ::knok::Engine::for_artifact(artifact)?;
                invoke_run(&engine, #(#input_names),*)
            }
        })
    } else {
        None
    };
    let variant_statics = compiled.iter().enumerate().map(|(index, variant)| {
        let vmfb_name = format_ident!("VMFB_{index}");
        let flags_name = format_ident!("COMPILE_FLAGS_{index}");
        let vmfb_bytes = variant.vmfb.iter().copied();
        let flags = &variant.compile_flags;
        quote! {
            static #vmfb_name: &[u8] = &[#(#vmfb_bytes),*];
            static #flags_name: &[&str] = &[#(#flags),*];
        }
    });
    let variants = compiled.iter().enumerate().map(|(index, variant)| {
        let vmfb_name = format_ident!("VMFB_{index}");
        let flags_name = format_ident!("COMPILE_FLAGS_{index}");
        let backend = &variant.backend;
        let driver = &variant.driver;
        quote! {
            ::knok::GraphArtifactVariant {
                vmfb: #vmfb_name,
                backend: #backend,
                driver: #driver,
                compile_flags: #flags_name,
            }
        }
    });

    Ok(quote! {
        pub mod #module_name {
            #typed_scope_import

            pub fn artifact() -> ::knok::GraphArtifact {
                #(#variant_statics)*
                static VARIANTS: &[::knok::GraphArtifactVariant] = &[#(#variants),*];
                static INPUT_DESCS: &[::knok::TensorDesc] = &[#(#input_descs),*];
                static OUTPUT_DESCS: &[::knok::TensorDesc] = &[#(#output_descs),*];
                ::knok::GraphArtifact {
                    function_name: #function_name,
                    input_descs: INPUT_DESCS,
                    output_descs: OUTPUT_DESCS,
                    variants: VARIANTS,
                }
            }

            #typed_invoke
        }
    })
}

struct MlirModel {
    name: Ident,
    path: LitStr,
    backend_specs: Vec<BackendSpec>,
    function: LitStr,
    inputs: Option<Vec<Type>>,
    outputs: Option<Vec<Type>>,
}

impl Parse for MlirModel {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut name = None;
        let mut path = None;
        let mut backend_specs = None;
        let mut function = None;
        let mut inputs = None;
        let mut outputs = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            match key.to_string().as_str() {
                "name" => name = Some(input.parse()?),
                "path" => path = Some(input.parse()?),
                "backend" => {
                    if backend_specs.is_some() {
                        return Err(syn::Error::new(
                            key.span(),
                            "backend and backends are mutually exclusive",
                        ));
                    }
                    let value: syn::Expr = input.parse()?;
                    backend_specs = Some(vec![parse_backend_expr(&value)?]);
                }
                "backends" => {
                    if backend_specs.is_some() {
                        return Err(syn::Error::new(
                            key.span(),
                            "backend and backends are mutually exclusive",
                        ));
                    }
                    let value: syn::Expr = input.parse()?;
                    backend_specs = Some(parse_backend_array(&value)?);
                }
                "function" => function = Some(input.parse()?),
                "inputs" => {
                    let content;
                    bracketed!(content in input);
                    inputs = Some(
                        Punctuated::<Type, Token![,]>::parse_terminated(&content)?
                            .into_iter()
                            .collect(),
                    );
                }
                "output" => {
                    if outputs.is_some() {
                        return Err(syn::Error::new(
                            key.span(),
                            "output and outputs are mutually exclusive",
                        ));
                    }
                    outputs = Some(vec![input.parse()?]);
                }
                "outputs" => {
                    if outputs.is_some() {
                        return Err(syn::Error::new(
                            key.span(),
                            "output and outputs are mutually exclusive",
                        ));
                    }
                    let content;
                    bracketed!(content in input);
                    let parsed = Punctuated::<Type, Token![,]>::parse_terminated(&content)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    if parsed.is_empty() {
                        return Err(syn::Error::new(
                            key.span(),
                            "outputs must contain at least one Tensor type",
                        ));
                    }
                    outputs = Some(parsed);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown mlir_model key `{other}`"),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        if inputs.is_some() != outputs.is_some() {
            return Err(input.error("inputs and output(s) must be provided together"));
        }
        Ok(Self {
            name: name.ok_or_else(|| input.error("missing name: <ident>"))?,
            path: path.ok_or_else(|| input.error("missing path: \"...\""))?,
            backend_specs: {
                let specs =
                    backend_specs.ok_or_else(|| input.error("missing backend: Backend::..."))?;
                reject_duplicate_drivers(&specs)?;
                specs
            },
            function: function.ok_or_else(|| input.error("missing function: \"...\""))?,
            inputs,
            outputs,
        })
    }
}
