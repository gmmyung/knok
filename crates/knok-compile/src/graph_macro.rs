use knok_core::parse_graph_with_signatures;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse2, ItemFn, ReturnType};

use crate::{
    backend::parse_backend_specs,
    common::{input_name, parse_return_output_types, runtime_input_variant, rust_element_type},
    compile::compile_graph_variants_with_registry,
    registry::{register_graph, registered_graphs, registered_signatures},
};

pub fn expand_graph(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_graph_result(attr, item) {
        Ok(tokens) => tokens,
        Err(error) => error.to_compile_error(),
    }
}

fn expand_graph_result(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let backend_specs = parse_backend_specs(attr.clone())?;
    let item_fn: ItemFn = parse2(item)?;
    let visibility = item_fn.vis.clone();
    let signature = item_fn.sig.clone();
    let output_ty = match &signature.output {
        ReturnType::Type(_, ty) => ty.clone(),
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                &signature.ident,
                "graph functions must return a Tensor type or tuple of Tensor types",
            ));
        }
    };
    let graph = parse_graph_with_signatures(attr, item_fn, &registered_signatures())?;
    let output_tys = parse_return_output_types(&signature.output)?;
    let graphs = registered_graphs();
    let compiled =
        compile_graph_variants_with_registry(&graph, &graphs, &backend_specs).map_err(|error| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("failed to compile knok graph `{}`: {error}", graph.name),
            )
        })?;
    register_graph(graph.clone());

    let name = &signature.ident;
    let inputs = signature.inputs.iter().collect::<Vec<_>>();
    let arg_names = signature
        .inputs
        .iter()
        .map(input_name)
        .collect::<syn::Result<Vec<_>>>()?;
    let function_name = format!("knok.{}", graph.name);
    let artifact_name = format_ident!("{}_artifact", name);
    let run_name = format_ident!("{}_run", name);
    let output_shapes = graph.outputs.iter().map(|output| {
        let dims = output.shape.iter().copied();
        quote!(&[#(#dims),*])
    });
    let runtime_inputs = graph
        .inputs
        .iter()
        .zip(arg_names.iter())
        .map(|(input, arg_name)| {
            let shape = {
                let dims = input.ty.shape.iter().copied();
                quote!(&[#(#dims),*])
            };
            let variant = runtime_input_variant(input.ty.elem);
            quote!(::knok::runtime::RuntimeInput::#variant(#shape, #arg_name.as_slice()))
        });
    let artifact_input_shapes = graph.inputs.iter().map(|input| {
        let dims = input.ty.shape.iter().copied();
        quote!(&[#(#dims),*])
    });
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

    let run_body = if output_tys.len() == 1 {
        let output_elem_ty = rust_element_type(graph.outputs[0].elem);
        quote! {
            let output = ::knok::__private::invoke_one_with_engine::<#output_elem_ty>(
                engine,
                artifact,
                &[#(#runtime_inputs),*],
            )?;
            <#output_ty>::from_vec(output)
        }
    } else {
        let output_reads = graph.outputs.iter().zip(output_tys.iter()).enumerate().map(
            |(index, (output, output_ty))| {
                let output_elem_ty = rust_element_type(output.elem);
                quote!(<#output_ty>::from_vec(outputs.read::<#output_elem_ty>(#index)?)?)
            },
        );
        quote! {
            let outputs = engine.invoke(artifact, &[#(#runtime_inputs),*])?;
            Ok((#(#output_reads),*))
        }
    };

    Ok(quote! {
        #visibility fn #artifact_name() -> ::knok::GraphArtifact {
            #(#variant_statics)*
            static VARIANTS: &[::knok::GraphArtifactVariant] = &[#(#variants),*];
            static INPUT_SHAPES: &[&[usize]] = &[#(#artifact_input_shapes),*];
            static OUTPUT_SHAPES: &[&[usize]] = &[#(#output_shapes),*];
            ::knok::GraphArtifact {
                function_name: #function_name,
                input_shapes: INPUT_SHAPES,
                output_shapes: OUTPUT_SHAPES,
                variants: VARIANTS,
            }
        }

        #visibility fn #run_name(engine: &::knok::Engine, #(#inputs),*) -> ::knok::Result<#output_ty> {
            let artifact = #artifact_name();
            #run_body
        }

        #visibility fn #name(#(#inputs),*) -> ::knok::Result<#output_ty> {
            let artifact = #artifact_name();
            let engine = ::knok::Engine::for_artifact(artifact)?;
            #run_name(&engine, #(#arg_names),*)
        }
    })
}
