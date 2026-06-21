use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
};

use knok_core::{
    parse_graph_with_signatures, parse_tensor_type, BinaryOp, CallOp, ElementType, Expr,
    GraphSignature, TensorType, TypedGraph, UnaryOp,
};
use melior::{dialect::DialectRegistry, ir::operation::OperationLike, ir::Module, Context};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    bracketed,
    parse::{Parse, ParseStream, Parser},
    parse2,
    punctuated::Punctuated,
    spanned::Spanned,
    FnArg, Ident, ItemFn, Lit, LitStr, MetaNameValue, ReturnType, Token, Type,
};

pub fn expand_graph(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_graph_result(attr, item) {
        Ok(tokens) => tokens,
        Err(error) => error.to_compile_error(),
    }
}

static GRAPH_REGISTRY: OnceLock<Mutex<BTreeMap<String, TypedGraph>>> = OnceLock::new();

fn graph_registry() -> &'static Mutex<BTreeMap<String, TypedGraph>> {
    GRAPH_REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn registered_graphs() -> BTreeMap<String, TypedGraph> {
    graph_registry()
        .lock()
        .expect("knok graph registry lock poisoned")
        .clone()
}

fn registered_signatures() -> Vec<(String, GraphSignature)> {
    registered_graphs()
        .into_iter()
        .map(|(name, graph)| {
            (
                name,
                GraphSignature {
                    inputs: graph.inputs.into_iter().map(|input| input.ty).collect(),
                    output: graph.output,
                },
            )
        })
        .collect()
}

fn register_graph(graph: TypedGraph) {
    graph_registry()
        .lock()
        .expect("knok graph registry lock poisoned")
        .insert(graph.name.clone(), graph);
}

#[derive(Clone, Debug)]
struct BackendSpec {
    backend: String,
    driver: String,
    extra_flags: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IreeBackend {
    LlvmCpu,
    MetalSpirv,
}

impl IreeBackend {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "llvm-cpu" => Some(Self::LlvmCpu),
            "metal-spirv" => Some(Self::MetalSpirv),
            _ => None,
        }
    }

    fn default_driver(self) -> &'static str {
        match self {
            Self::LlvmCpu => "local-task",
            Self::MetalSpirv => "metal",
        }
    }

    fn target_backend(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
        }
    }

    fn supports_driver(self, driver: &str) -> bool {
        self.default_driver() == driver
    }
}

impl BackendSpec {
    fn new(
        backend: String,
        driver: Option<String>,
        extra_flags: Vec<String>,
        span: proc_macro2::Span,
    ) -> syn::Result<Self> {
        let capability = IreeBackend::parse(&backend).ok_or_else(|| {
            syn::Error::new(
                span,
                format!(
                    "unsupported IREE backend `{backend}`; expected `llvm-cpu` or `metal-spirv`"
                ),
            )
        })?;
        let driver = driver.unwrap_or_else(|| capability.default_driver().to_string());
        if !capability.supports_driver(&driver) {
            return Err(syn::Error::new(
                span,
                format!(
                    "backend `{}` expects runtime driver `{}`, got `{driver}`",
                    capability.target_backend(),
                    capability.default_driver(),
                ),
            ));
        }
        Ok(Self {
            backend,
            driver,
            extra_flags,
        })
    }
}

fn parse_backend_specs(attr: TokenStream) -> syn::Result<Vec<BackendSpec>> {
    let args = Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse2(attr)?;
    let mut backend = None;
    let mut backends = None;
    for arg in args {
        if arg.path.is_ident("backend") {
            if backend.is_some() || backends.is_some() {
                return Err(syn::Error::new(
                    arg.span(),
                    "backend and backends are mutually exclusive",
                ));
            }
            let syn::Expr::Lit(expr_lit) = &arg.value else {
                return Err(syn::Error::new(
                    arg.value.span(),
                    "backend must be a string literal",
                ));
            };
            let Lit::Str(lit) = &expr_lit.lit else {
                return Err(syn::Error::new(
                    expr_lit.span(),
                    "backend must be a string literal",
                ));
            };
            backend = Some(vec![BackendSpec::new(
                lit.value(),
                None,
                Vec::new(),
                lit.span(),
            )?]);
        } else if arg.path.is_ident("backends") {
            if backend.is_some() || backends.is_some() {
                return Err(syn::Error::new(
                    arg.span(),
                    "backend and backends are mutually exclusive",
                ));
            }
            backends = Some(parse_backend_array(&arg.value)?);
        } else {
            return Err(syn::Error::new(
                arg.path.span(),
                "unknown graph attribute argument",
            ));
        }
    }
    let specs = backend.or(backends).ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "missing required backend = \"...\" argument",
        )
    })?;
    reject_duplicate_drivers(&specs)?;
    Ok(specs)
}

fn parse_backend_array(value: &syn::Expr) -> syn::Result<Vec<BackendSpec>> {
    let syn::Expr::Array(array) = value else {
        return Err(syn::Error::new(
            value.span(),
            "backends must be an array of backend(...) declarations",
        ));
    };
    if array.elems.is_empty() {
        return Err(syn::Error::new(
            array.span(),
            "backends must contain at least one backend(...) declaration",
        ));
    }
    array.elems.iter().map(parse_backend_call).collect()
}

fn parse_backend_call(expr: &syn::Expr) -> syn::Result<BackendSpec> {
    let syn::Expr::Call(call) = expr else {
        return Err(syn::Error::new(expr.span(), "expected backend(...)"));
    };
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return Err(syn::Error::new(call.func.span(), "expected backend(...)"));
    };
    if !path.path.is_ident("backend") {
        return Err(syn::Error::new(call.func.span(), "expected backend(...)"));
    }
    let Some(first) = call.args.first() else {
        return Err(syn::Error::new(call.span(), "backend name is required"));
    };
    let syn::Expr::Lit(expr_lit) = first else {
        return Err(syn::Error::new(
            first.span(),
            "backend name must be a string literal",
        ));
    };
    let Lit::Str(backend_lit) = &expr_lit.lit else {
        return Err(syn::Error::new(
            expr_lit.span(),
            "backend name must be a string literal",
        ));
    };

    let mut driver = None;
    let mut extra_flags = Vec::new();
    for arg in call.args.iter().skip(1) {
        let syn::Expr::Assign(assign) = arg else {
            return Err(syn::Error::new(
                arg.span(),
                "backend options must be assignments such as driver = \"...\"",
            ));
        };
        let syn::Expr::Path(key_path) = assign.left.as_ref() else {
            return Err(syn::Error::new(assign.left.span(), "expected option name"));
        };
        let key = key_path.path.require_ident()?.to_string();
        match key.as_str() {
            "driver" => {
                if driver.is_some() {
                    return Err(syn::Error::new(assign.span(), "duplicate driver option"));
                }
                let syn::Expr::Lit(expr_lit) = assign.right.as_ref() else {
                    return Err(syn::Error::new(
                        assign.right.span(),
                        "driver must be a string literal",
                    ));
                };
                let Lit::Str(lit) = &expr_lit.lit else {
                    return Err(syn::Error::new(
                        expr_lit.span(),
                        "driver must be a string literal",
                    ));
                };
                driver = Some(lit.value());
            }
            "flags" => {
                let syn::Expr::Array(array) = assign.right.as_ref() else {
                    return Err(syn::Error::new(
                        assign.right.span(),
                        "flags must be an array of string literals",
                    ));
                };
                for flag in &array.elems {
                    let syn::Expr::Lit(expr_lit) = flag else {
                        return Err(syn::Error::new(
                            flag.span(),
                            "flags must be string literals",
                        ));
                    };
                    let Lit::Str(lit) = &expr_lit.lit else {
                        return Err(syn::Error::new(
                            expr_lit.span(),
                            "flags must be string literals",
                        ));
                    };
                    extra_flags.push(lit.value());
                }
            }
            _ => {
                return Err(syn::Error::new(
                    key_path.path.span(),
                    format!("unknown backend option `{key}`"),
                ));
            }
        }
    }
    BackendSpec::new(backend_lit.value(), driver, extra_flags, backend_lit.span())
}

fn reject_duplicate_drivers(specs: &[BackendSpec]) -> syn::Result<()> {
    let mut drivers = BTreeMap::<&str, &str>::new();
    for spec in specs {
        if let Some(existing_backend) = drivers.insert(&spec.driver, &spec.backend) {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "duplicate runtime driver `{}` for backends `{}` and `{}`",
                    spec.driver, existing_backend, spec.backend
                ),
            ));
        }
    }
    Ok(())
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
                "graph functions must return a Tensor type",
            ));
        }
    };
    let graph = parse_graph_with_signatures(attr, item_fn, &registered_signatures())?;
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
    let input_shapes = graph.inputs.iter().map(|input| {
        let dims = input.ty.shape.iter().copied();
        quote!(&[#(#dims),*])
    });
    let function_name = format!("knok.{}", graph.name);
    let artifact_name = format_ident!("{}_artifact", name);
    let run_name = format_ident!("{}_run", name);
    let output_dims = graph.output.shape.iter().copied();
    let output_elem_ty = rust_element_type(graph.output.elem);
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

    Ok(quote! {
        #visibility fn #artifact_name() -> ::knok::GraphArtifact {
            #(#variant_statics)*
            static VARIANTS: &[::knok::GraphArtifactVariant] = &[#(#variants),*];
            static INPUT_SHAPES: &[&[usize]] = &[#(#artifact_input_shapes),*];
            ::knok::GraphArtifact {
                function_name: #function_name,
                input_shapes: INPUT_SHAPES,
                output_shape: &[#(#output_dims),*],
                variants: VARIANTS,
            }
        }

        #visibility fn #run_name(engine: &::knok::Engine, #(#inputs),*) -> ::knok::Result<#output_ty> {
            let artifact = #artifact_name();
            let output = ::knok::__private::invoke_with_engine::<#output_elem_ty>(
                engine,
                artifact,
                &[#((#input_shapes, #arg_names.as_slice())),*],
            )?;
            <#output_ty>::from_vec(output)
        }

        #visibility fn #name(#(#inputs),*) -> ::knok::Result<#output_ty> {
            let artifact = #artifact_name();
            let engine = ::knok::Engine::for_artifact(artifact)?;
            #run_name(&engine, #(#arg_names),*)
        }
    })
}

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
    if let (Some(inputs), Some(output)) = (&model.inputs, &model.output) {
        let expected_inputs = inputs
            .iter()
            .map(parse_tensor_type)
            .collect::<syn::Result<Vec<_>>>()?;
        let expected_output = parse_tensor_type(output)?;
        validate_mlir_model_signature(
            &mlir,
            &model.function.value(),
            &expected_inputs,
            &expected_output,
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
    let input_types = model.inputs.unwrap_or_default();
    let output_shape = model
        .output
        .as_ref()
        .map(|ty| quote!(<#ty>::SHAPE))
        .unwrap_or_else(|| quote!(&[]));
    let output_elem_ty = model
        .output
        .as_ref()
        .map(|ty| parse_tensor_type(ty).map(|ty| rust_element_type(ty.elem)))
        .transpose()?;
    let typed_scope_import = model.output.as_ref().map(|_| {
        quote!(
            use super::*;
        )
    });
    let typed_invoke = if let Some(output_ty) = model.output {
        let output_elem_ty = output_elem_ty.expect("output element type was parsed");
        let input_names = (0..input_types.len())
            .map(|index| format_ident!("input{index}"))
            .collect::<Vec<_>>();
        Some(quote! {
            pub fn invoke_run(
                engine: &::knok::Engine,
                #(#input_names: #input_types),*
            ) -> ::knok::Result<#output_ty> {
                let output = ::knok::__private::invoke_with_engine::<#output_elem_ty>(engine, artifact(), &[
                    #((<#input_types>::SHAPE, #input_names.as_slice())),*
                ])?;
                <#output_ty>::from_vec(output)
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
                static INPUT_SHAPES: &[&[usize]] = &[#(<#input_types>::SHAPE),*];
                ::knok::GraphArtifact {
                    function_name: #function_name,
                    input_shapes: INPUT_SHAPES,
                    output_shape: #output_shape,
                    variants: VARIANTS,
                }
            }

            #typed_invoke
        }
    })
}

fn validate_mlir_model_signature(
    mlir: &str,
    function_name: &str,
    expected_inputs: &[TensorType],
    expected_output: &TensorType,
) -> syn::Result<()> {
    let symbol_name = function_name.rsplit('.').next().unwrap_or(function_name);
    let signature = find_mlir_function_signature(mlir, symbol_name).ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("failed to find MLIR function symbol `@{symbol_name}`"),
        )
    })?;
    if signature.inputs != expected_inputs {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "mlir_model inputs do not match MLIR function `{function_name}`: declared {:?}, MLIR has {:?}",
                expected_inputs, signature.inputs
            ),
        ));
    }
    if &signature.output != expected_output {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "mlir_model output does not match MLIR function `{function_name}`: declared {:?}, MLIR has {:?}",
                expected_output, signature.output
            ),
        ));
    }
    Ok(())
}

fn rust_element_type(elem: ElementType) -> TokenStream {
    match elem {
        ElementType::F32 => quote!(f32),
        ElementType::F64 => quote!(f64),
        ElementType::I32 => quote!(i32),
        ElementType::I64 => quote!(i64),
    }
}

struct MlirSignature {
    inputs: Vec<TensorType>,
    output: TensorType,
}

fn find_mlir_function_signature(mlir: &str, symbol_name: &str) -> Option<MlirSignature> {
    let needle = format!("func.func @{symbol_name}");
    let start = mlir.find(&needle)? + needle.len();
    let rest = &mlir[start..];
    let args_start = rest.find('(')? + 1;
    let args_end = args_start + rest[args_start..].find(')')?;
    let args = &rest[args_start..args_end];
    let after_args = &rest[args_end + 1..];
    let arrow = after_args.find("->")? + 2;
    let output = after_args[arrow..]
        .trim_start()
        .split_whitespace()
        .next()?
        .trim_end_matches('{');

    let inputs = if args.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(args, ',')
            .into_iter()
            .map(|arg| {
                let ty = arg.rsplit_once(':')?.1.trim();
                parse_mlir_tensor_type(ty)
            })
            .collect::<Option<Vec<_>>>()?
    };
    Some(MlirSignature {
        inputs,
        output: parse_mlir_tensor_type(output)?,
    })
}

fn split_top_level(input: &str, separator: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in input.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if ch == separator && depth == 0 => {
                parts.push(input[start..index].trim());
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(input[start..].trim());
    parts
}

fn parse_mlir_tensor_type(ty: &str) -> Option<TensorType> {
    let body = ty.strip_prefix("tensor<")?.strip_suffix('>')?;
    if let Some(elem) = parse_mlir_element_type(body) {
        return Some(TensorType {
            elem,
            shape: Vec::new(),
        });
    }
    let (dims, elem) = body.rsplit_once('x')?;
    let elem = parse_mlir_element_type(elem)?;
    let shape = dims
        .split('x')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()
        .ok()?;
    Some(TensorType { elem, shape })
}

fn parse_mlir_element_type(elem: &str) -> Option<ElementType> {
    match elem {
        "f32" => Some(ElementType::F32),
        "f64" => Some(ElementType::F64),
        "i32" => Some(ElementType::I32),
        "i64" => Some(ElementType::I64),
        _ => None,
    }
}

fn element_count(ty: &TensorType) -> usize {
    ty.shape.iter().product()
}

fn format_shape_list(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_usize_list(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn reassociation_for_rank(rank: usize) -> String {
    let dims = (0..rank)
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[[{dims}]]")
}

fn collapse_reassociation_for_removed_axis(rank: usize, axis: usize) -> String {
    if rank <= 1 {
        return reassociation_for_rank(rank);
    }
    let mut groups = Vec::new();
    let mut index = 0;
    while index < rank {
        if index == axis {
            if groups.is_empty() {
                groups.push(vec![index, index + 1]);
                index += 2;
            } else {
                groups.last_mut().expect("group exists").push(index);
                index += 1;
            }
        } else {
            groups.push(vec![index]);
            index += 1;
        }
    }
    format_reassociation_groups(groups)
}

fn expand_reassociation_for_inserted_axis(input_rank: usize, axis: usize) -> String {
    let mut groups = Vec::new();
    for input_axis in 0..input_rank {
        if input_axis == axis {
            groups.push(vec![input_axis, input_axis + 1]);
        } else if input_axis < axis {
            groups.push(vec![input_axis]);
        } else {
            groups.push(vec![input_axis + 1]);
        }
    }
    if axis == input_rank {
        if let Some(last) = groups.last_mut() {
            last.push(axis);
        } else {
            groups.push(vec![axis]);
        }
    }
    format_reassociation_groups(groups)
}

fn format_reassociation_groups(groups: Vec<Vec<usize>>) -> String {
    let groups = groups
        .into_iter()
        .map(|group| {
            format!(
                "[{}]",
                group
                    .into_iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{groups}]")
}

fn format_dim_list(rank: usize) -> String {
    (0..rank)
        .map(|index| format!("d{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn broadcast_result_type(lhs: &TensorType, rhs: &TensorType) -> anyhow::Result<TensorType> {
    if lhs.elem != rhs.elem {
        anyhow::bail!("binary operands have different element types");
    }
    let shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn ensure_broadcastable(input: &TensorType, output: &TensorType) -> anyhow::Result<()> {
    if input.elem != output.elem {
        anyhow::bail!("broadcast input and output element types differ");
    }
    let shape = broadcast_shape(&input.shape, &output.shape)?;
    if shape != output.shape {
        anyhow::bail!(
            "broadcast result shape {:?} does not match requested output {:?}",
            shape,
            output.shape
        );
    }
    Ok(())
}

fn axis_broadcast_dimensions(
    input_rank: usize,
    output_rank: usize,
    axis: usize,
) -> anyhow::Result<Vec<usize>> {
    if input_rank + 1 != output_rank {
        anyhow::bail!("axis broadcast expects exactly one reduced dimension");
    }
    if axis >= output_rank {
        anyhow::bail!("axis {axis} is out of bounds for rank {output_rank}");
    }
    Ok(vec![axis])
}

fn ensure_axis_broadcastable(
    input: &TensorType,
    output: &TensorType,
    axis: usize,
) -> anyhow::Result<()> {
    if input.elem != output.elem {
        anyhow::bail!("broadcast input and output element types differ");
    }
    if input.rank() + 1 != output.rank() {
        anyhow::bail!("axis broadcast expects exactly one reduced dimension");
    }
    for output_index in 0..output.rank() {
        if output_index == axis {
            continue;
        }
        let input_index = if output_index < axis {
            output_index
        } else {
            output_index - 1
        };
        if input.shape[input_index] != output.shape[output_index] {
            anyhow::bail!(
                "axis broadcast dimension mismatch at output dimension {}: input {} vs output {}",
                output_index,
                input.shape[input_index],
                output.shape[output_index]
            );
        }
    }
    Ok(())
}

fn collapse_reassociation_for_squeezed_broadcast(
    input_shape: &[usize],
    aligned_output_shape: &[usize],
) -> String {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut pending = Vec::new();
    for (index, (input_dim, output_dim)) in input_shape.iter().zip(aligned_output_shape).enumerate()
    {
        pending.push(index);
        if !(*input_dim == 1 && *output_dim != 1) {
            groups.push(core::mem::take(&mut pending));
        }
    }
    if !pending.is_empty() {
        if let Some(last) = groups.last_mut() {
            last.extend(pending);
        } else {
            groups.push(pending);
        }
    }
    let groups = groups
        .into_iter()
        .map(|group| {
            format!(
                "[{}]",
                group
                    .into_iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{groups}]")
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> anyhow::Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let lhs_dim = dim_from_trailing(lhs, rank, offset);
        let rhs_dim = dim_from_trailing(rhs, rank, offset);
        let dim = match (lhs_dim, rhs_dim) {
            (Some(lhs_dim), Some(rhs_dim)) if lhs_dim == rhs_dim => lhs_dim,
            (Some(1), Some(rhs_dim)) => rhs_dim,
            (Some(lhs_dim), Some(1)) => lhs_dim,
            (None, Some(dim)) | (Some(dim), None) => dim,
            (None, None) => unreachable!("rank is derived from at least one shape"),
            (Some(lhs_dim), Some(rhs_dim)) => {
                anyhow::bail!("broadcast dimension {offset} differs: {lhs_dim} vs {rhs_dim}");
            }
        };
        shape.push(dim);
    }
    Ok(shape)
}

fn dim_from_trailing(shape: &[usize], rank: usize, offset: usize) -> Option<usize> {
    let padding = rank - shape.len();
    (offset >= padding).then(|| shape[offset - padding])
}

fn reduction_output_shape(
    input_shape: &[usize],
    axis: Option<usize>,
    keep_dims: bool,
) -> Vec<usize> {
    match axis {
        Some(axis) if keep_dims => input_shape
            .iter()
            .enumerate()
            .map(|(index, dim)| if index == axis { 1 } else { *dim })
            .collect(),
        Some(axis) => {
            let mut shape = input_shape.to_vec();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        }
        None => vec![1],
    }
}

fn reduction_output_map(input_rank: usize, axis: Option<usize>, keep_dims: bool) -> String {
    match axis {
        Some(axis) if keep_dims => {
            let dims = (0..input_rank)
                .map(|index| {
                    if index == axis {
                        "0".to_string()
                    } else {
                        format!("d{index}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("({dims})")
        }
        Some(_) if input_rank == 1 => "(0)".to_string(),
        Some(axis) => {
            let dims = (0..input_rank)
                .filter(|index| *index != axis)
                .map(|index| format!("d{index}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({dims})")
        }
        None => "(0)".to_string(),
    }
}

fn min_float_literal(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 => "-3.40282347E+38",
        ElementType::F64 => "-1.7976931348623157E+308",
        ElementType::I32 | ElementType::I64 => "0",
    }
}

struct MlirModel {
    name: Ident,
    path: LitStr,
    backend_specs: Vec<BackendSpec>,
    function: LitStr,
    inputs: Option<Vec<Type>>,
    output: Option<Type>,
}

impl Parse for MlirModel {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut name = None;
        let mut path = None;
        let mut backend_specs = None;
        let mut function = None;
        let mut inputs = None;
        let mut output = None;
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
                    let lit: LitStr = input.parse()?;
                    backend_specs = Some(vec![BackendSpec::new(
                        lit.value(),
                        None,
                        Vec::new(),
                        lit.span(),
                    )?]);
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
                "output" => output = Some(input.parse()?),
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
        if inputs.is_some() != output.is_some() {
            return Err(input.error("inputs and output must be provided together"));
        }
        Ok(Self {
            name: name.ok_or_else(|| input.error("missing name: <ident>"))?,
            path: path.ok_or_else(|| input.error("missing path: \"...\""))?,
            backend_specs: {
                let specs = backend_specs.ok_or_else(|| input.error("missing backend: \"...\""))?;
                reject_duplicate_drivers(&specs)?;
                specs
            },
            function: function.ok_or_else(|| input.error("missing function: \"...\""))?,
            inputs,
            output,
        })
    }
}

fn input_name(input: &FnArg) -> syn::Result<proc_macro2::Ident> {
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

pub struct CompiledGraph {
    pub mlir: String,
    pub vmfb: Vec<u8>,
}

struct CompiledVariant {
    backend: String,
    driver: String,
    compile_flags: Vec<String>,
    vmfb: Vec<u8>,
}

pub fn compile_graph(graph: &TypedGraph) -> anyhow::Result<CompiledGraph> {
    compile_graph_with_registry(graph, &BTreeMap::new())
}

pub fn compile_graph_with_registry(
    graph: &TypedGraph,
    graphs: &BTreeMap<String, TypedGraph>,
) -> anyhow::Result<CompiledGraph> {
    let mlir = lower_to_mlir_with_registry(graph, graphs)?;
    verify_with_melior(&mlir)?;
    let vmfb = compile_mlir_source(&graph.backend, &mlir)?;
    Ok(CompiledGraph { mlir, vmfb })
}

fn compile_graph_variants_with_registry(
    graph: &TypedGraph,
    graphs: &BTreeMap<String, TypedGraph>,
    specs: &[BackendSpec],
) -> anyhow::Result<Vec<CompiledVariant>> {
    let mlir = lower_to_mlir_with_registry(graph, graphs)?;
    verify_with_melior(&mlir)?;
    specs
        .iter()
        .map(|spec| {
            let compile_flags = backend_flags(&spec.backend, &spec.extra_flags);
            let vmfb = compile_with_iree(&spec.backend, &spec.extra_flags, &mlir)?;
            Ok(CompiledVariant {
                backend: spec.backend.clone(),
                driver: spec.driver.clone(),
                compile_flags,
                vmfb,
            })
        })
        .collect()
}

fn compile_mlir_variants(
    specs: &[BackendSpec],
    mlir: &str,
) -> anyhow::Result<Vec<CompiledVariant>> {
    verify_with_melior(mlir)?;
    specs
        .iter()
        .map(|spec| {
            let compile_flags = backend_flags(&spec.backend, &spec.extra_flags);
            let vmfb = compile_with_iree(&spec.backend, &spec.extra_flags, mlir)?;
            Ok(CompiledVariant {
                backend: spec.backend.clone(),
                driver: spec.driver.clone(),
                compile_flags,
                vmfb,
            })
        })
        .collect()
}

pub fn compile_mlir_source(backend: &str, mlir: &str) -> anyhow::Result<Vec<u8>> {
    compile_with_iree(backend, &[], mlir)
}

pub fn lower_to_mlir(graph: &TypedGraph) -> anyhow::Result<String> {
    lower_to_mlir_with_registry(graph, &BTreeMap::new())
}

pub fn lower_to_mlir_with_registry(
    graph: &TypedGraph,
    graphs: &BTreeMap<String, TypedGraph>,
) -> anyhow::Result<String> {
    let mut lowerer = Lowerer::new(graph, graphs);
    lowerer.lower()
}

fn verify_with_melior(mlir: &str) -> anyhow::Result<()> {
    let registry = DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::parse(&context, mlir)
        .ok_or_else(|| anyhow::anyhow!("melior failed to parse generated MLIR"))?;
    if !module.as_operation().verify() {
        anyhow::bail!("melior rejected generated MLIR");
    }
    Ok(())
}

fn compile_with_iree(backend: &str, extra_flags: &[String], mlir: &str) -> anyhow::Result<Vec<u8>> {
    if IreeBackend::parse(backend).is_none() {
        anyhow::bail!("unsupported IREE backend `{backend}`; expected `llvm-cpu` or `metal-spirv`");
    }
    let cache_dir = cache_dir()?;
    fs::create_dir_all(&cache_dir)?;
    let iree_compile = iree_compile_command();
    let flags = backend_flags(backend, extra_flags);
    let key = cache_key(backend, mlir, &iree_compile, &flags);
    let vmfb_path = cache_dir.join(format!("{key}.vmfb"));
    let mlir_path = cache_dir.join(format!("{key}.mlir"));
    if vmfb_path.exists() {
        return Ok(fs::read(vmfb_path)?);
    }

    fs::write(&mlir_path, mlir)?;
    let mut command = Command::new(&iree_compile);
    command
        .arg(&mlir_path)
        .args(&flags)
        .arg("-o")
        .arg(&vmfb_path);
    let output = command.output()?;
    if !output.status.success() {
        anyhow::bail!(
            "iree-compile failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(fs::read(vmfb_path)?)
}

fn iree_compile_command() -> String {
    env::var("KNOK_IREE_COMPILE").unwrap_or_else(|_| "iree-compile".to_string())
}

fn backend_flags(backend: &str, extra_flags: &[String]) -> Vec<String> {
    let capability = IreeBackend::parse(backend)
        .unwrap_or_else(|| panic!("unsupported IREE backend `{backend}`"));
    let mut flags = vec![
        format!("--iree-hal-target-backends={backend}"),
        "--iree-input-demote-f64-to-f32=false".to_string(),
    ];
    if capability == IreeBackend::MetalSpirv {
        flags.push("--iree-metal-compile-to-metallib=false".to_string());
    }
    flags.extend(extra_flags.iter().cloned());
    flags
}

fn cache_dir() -> anyhow::Result<PathBuf> {
    if let Ok(path) = env::var("KNOK_CACHE_DIR") {
        return Ok(PathBuf::from(path));
    }
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    Ok(Path::new(&manifest_dir).join("target/knok-cache"))
}

fn cache_key(backend: &str, mlir: &str, iree_compile: &str, flags: &[String]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"knok-cache-v2");
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.update(backend.as_bytes());
    hasher.update(iree_compile.as_bytes());
    hasher.update(iree_compile_version(iree_compile).as_bytes());
    for flag in flags {
        hasher.update(flag.as_bytes());
    }
    for var in [
        "CARGO_CFG_TARGET_ARCH",
        "CARGO_CFG_TARGET_ENV",
        "CARGO_CFG_TARGET_OS",
        "CARGO_CFG_TARGET_VENDOR",
    ] {
        if let Ok(value) = env::var(var) {
            hasher.update(var.as_bytes());
            hasher.update(value.as_bytes());
        }
    }
    hasher.update(mlir.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn iree_compile_version(iree_compile: &str) -> String {
    match Command::new(iree_compile).arg("--version").output() {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout).into(),
        Ok(output) => format!(
            "unavailable:{}:{}:{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(error) => format!("unavailable:{error}"),
    }
}

struct Lowerer<'a> {
    graph: &'a TypedGraph,
    graphs: &'a BTreeMap<String, TypedGraph>,
    call_stack: Vec<String>,
    next_value: usize,
    lines: Vec<String>,
    values: BTreeMap<String, Value>,
}

#[derive(Clone)]
struct Value {
    name: String,
    ty: TensorType,
}

impl<'a> Lowerer<'a> {
    fn new(graph: &'a TypedGraph, graphs: &'a BTreeMap<String, TypedGraph>) -> Self {
        Self {
            graph,
            graphs,
            call_stack: vec![graph.name.clone()],
            next_value: 0,
            lines: Vec::new(),
            values: BTreeMap::new(),
        }
    }

    fn lower(&mut self) -> anyhow::Result<String> {
        let arg_list = self
            .graph
            .inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                self.values.insert(
                    input.name.clone(),
                    Value {
                        name: format!("%arg{index}"),
                        ty: input.ty.clone(),
                    },
                );
                format!("%arg{index}: {}", input.ty.mlir_type())
            })
            .collect::<Vec<_>>()
            .join(", ");
        for binding in &self.graph.lets {
            let value = self.lower_expr(&binding.value.kind)?;
            self.values.insert(binding.name.clone(), value);
        }
        let body = self.lower_expr(&self.graph.body.kind)?;
        self.lines.push(format!(
            "    return {} : {}",
            body.name,
            body.ty.mlir_type()
        ));

        let mut mlir = String::new();
        mlir.push_str("module @knok {\n");
        mlir.push_str(&format!(
            "  func.func @{}({}) -> {} {{\n",
            self.graph.name,
            arg_list,
            self.graph.output.mlir_type()
        ));
        for line in &self.lines {
            mlir.push_str(line);
            mlir.push('\n');
        }
        mlir.push_str("  }\n");
        mlir.push_str("}\n");
        Ok(mlir)
    }

    fn lower_expr(&mut self, expr: &Expr) -> anyhow::Result<Value> {
        match expr {
            Expr::Var(name) => self
                .values
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("unknown value `{name}` during lowering")),
            Expr::Const { value, elem } => self.constant(value, *elem),
            Expr::Unary { op, value } => match op {
                UnaryOp::Neg => {
                    let value = self.lower_expr(value)?;
                    let zero = self.zero_like(&value.ty)?;
                    self.binary_value(BinaryOp::Sub, zero, value)
                }
            },
            Expr::Binary { op, lhs, rhs } => {
                let lhs = self.lower_expr(lhs)?;
                let rhs = self.lower_expr(rhs)?;
                self.binary_value(*op, lhs, rhs)
            }
            Expr::Call { op, args } => match op {
                CallOp::Abs => {
                    let input = self.lower_expr(&args[0])?;
                    let op_name = if input.ty.elem.is_float() {
                        "math.absf"
                    } else {
                        "math.absi"
                    };
                    self.emit_unary(op_name, input)
                }
                CallOp::Argmax => {
                    let input = self.lower_expr(&args[0])?;
                    self.argmax(input)
                }
                CallOp::Clip => {
                    let value = self.lower_expr(&args[0])?;
                    let min = self.lower_expr(&args[1])?;
                    let max = self.lower_expr(&args[2])?;
                    let clipped_high = self.minimum(value, max)?;
                    self.maximum(clipped_high, min)
                }
                CallOp::Concat(axis) => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.concat(lhs, rhs, *axis)
                }
                CallOp::Conv2d => {
                    let input = self.lower_expr(&args[0])?;
                    let kernel = self.lower_expr(&args[1])?;
                    self.conv2d(input, kernel)
                }
                CallOp::Exp => {
                    let value = self.lower_expr(&args[0])?;
                    self.emit_unary("math.exp", value)
                }
                CallOp::Log => {
                    let value = self.lower_expr(&args[0])?;
                    self.emit_unary("math.log", value)
                }
                CallOp::Relu => {
                    let value = self.lower_expr(&args[0])?;
                    let zero = self.zero_like(&value.ty)?;
                    self.emit_binary("arith.maximumf", zero, value)
                }
                CallOp::Mean(axis) => {
                    let input = self.lower_expr(&args[0])?;
                    self.mean(input, *axis)
                }
                CallOp::Matmul => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.matmul(lhs, rhs)
                }
                CallOp::Minimum => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.minimum(lhs, rhs)
                }
                CallOp::Maximum => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.maximum(lhs, rhs)
                }
                CallOp::Pow => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.emit_binary("math.powf", lhs, rhs)
                }
                CallOp::Sigmoid => {
                    let value = self.lower_expr(&args[0])?;
                    self.sigmoid(value)
                }
                CallOp::Softmax(axis) => {
                    let value = self.lower_expr(&args[0])?;
                    self.softmax(value, *axis)
                }
                CallOp::Sqrt => {
                    let value = self.lower_expr(&args[0])?;
                    self.emit_unary("math.sqrt", value)
                }
                CallOp::Tanh => {
                    let value = self.lower_expr(&args[0])?;
                    self.emit_unary("math.tanh", value)
                }
                CallOp::Transpose => {
                    let input = self.lower_expr(&args[0])?;
                    self.transpose(input)
                }
                CallOp::Reshape(ty) => {
                    let input = self.lower_expr(&args[0])?;
                    self.reshape(input, ty)
                }
                CallOp::Broadcast(ty) => {
                    let input = self.lower_expr(&args[0])?;
                    self.broadcast(input, ty)
                }
                CallOp::Slice { target, starts } => {
                    let input = self.lower_expr(&args[0])?;
                    self.slice(input, target, starts)
                }
                CallOp::Squeeze(ty) => {
                    let input = self.lower_expr(&args[0])?;
                    self.reshape(input, ty)
                }
                CallOp::Stack(axis) => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.stack(lhs, rhs, *axis)
                }
                CallOp::Sum(axis) => {
                    let input = self.lower_expr(&args[0])?;
                    self.sum(input, *axis)
                }
                CallOp::Take { axis, index } => {
                    let input = self.lower_expr(&args[0])?;
                    self.take(input, *axis, *index)
                }
                CallOp::Unsqueeze(ty) => {
                    let input = self.lower_expr(&args[0])?;
                    self.reshape(input, ty)
                }
                CallOp::Graph(name) => {
                    let args = args
                        .iter()
                        .map(|arg| self.lower_expr(arg))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    self.inline_graph(name, args)
                }
            },
        }
    }

    fn inline_graph(&mut self, name: &str, args: Vec<Value>) -> anyhow::Result<Value> {
        if self.call_stack.iter().any(|candidate| candidate == name) {
            anyhow::bail!("recursive graph call `{name}` is not supported");
        }
        let graph = self
            .graphs
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("unknown graph `{name}` during lowering"))?;
        if graph.inputs.len() != args.len() {
            anyhow::bail!(
                "graph `{name}` expects {} arguments, got {}",
                graph.inputs.len(),
                args.len()
            );
        }

        self.call_stack.push(name.to_string());
        let mut overwritten = Vec::new();
        for (input, value) in graph.inputs.iter().zip(args) {
            overwritten.push((
                input.name.clone(),
                self.values.insert(input.name.clone(), value),
            ));
        }

        let result = (|| {
            for binding in &graph.lets {
                let value = self.lower_expr(&binding.value.kind)?;
                overwritten.push((
                    binding.name.clone(),
                    self.values.insert(binding.name.clone(), value),
                ));
            }
            self.lower_expr(&graph.body.kind)
        })();

        for (name, old_value) in overwritten.into_iter().rev() {
            if let Some(old_value) = old_value {
                self.values.insert(name, old_value);
            } else {
                self.values.remove(&name);
            }
        }
        self.call_stack.pop();
        result
    }

    fn constant(&mut self, value: &str, elem: ElementType) -> anyhow::Result<Value> {
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = arith.constant {value} : {}",
            elem.mlir_type()
        ));
        Ok(Value {
            name,
            ty: TensorType {
                elem,
                shape: vec![],
            },
        })
    }

    fn zero_like(&mut self, ty: &TensorType) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            return self.constant(ty.elem.zero_literal(), ty.elem);
        }
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = arith.constant dense<{}> : {}",
            ty.elem.zero_literal(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn binary_value(&mut self, op: BinaryOp, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = match (elem.is_float(), op) {
            (true, BinaryOp::Add) => "arith.addf",
            (true, BinaryOp::Sub) => "arith.subf",
            (true, BinaryOp::Mul) => "arith.mulf",
            (true, BinaryOp::Div) => "arith.divf",
            (false, BinaryOp::Add) => "arith.addi",
            (false, BinaryOp::Sub) => "arith.subi",
            (false, BinaryOp::Mul) => "arith.muli",
            (false, BinaryOp::Div) => "arith.divsi",
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    fn minimum(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = if elem.is_float() {
            "arith.minimumf"
        } else {
            "arith.minsi"
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    fn maximum(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = if elem.is_float() {
            "arith.maximumf"
        } else {
            "arith.maxsi"
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    fn emit_unary(&mut self, op_name: &str, value: Value) -> anyhow::Result<Value> {
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = {op_name} {} : {}",
            value.name,
            value.ty.mlir_type()
        ));
        Ok(Value { name, ty: value.ty })
    }

    fn emit_binary(&mut self, op_name: &str, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let ty = broadcast_result_type(&lhs.ty, &rhs.ty)?;
        let lhs = if lhs.ty == ty {
            lhs
        } else {
            self.broadcast(lhs, &ty)?
        };
        let rhs = if rhs.ty == ty {
            rhs
        } else {
            self.broadcast(rhs, &ty)?
        };
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = {op_name} {}, {} : {}",
            lhs.name,
            rhs.name,
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn splat(&mut self, scalar: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.fill ins({} : {}) outs({empty} : {}) -> {}",
            scalar.name,
            scalar.ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn matmul(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        if lhs.ty.rank() == 3 {
            return self.batch_matmul(lhs, rhs);
        }
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape: vec![lhs.ty.shape[0], rhs.ty.shape[1]],
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({zero} : {}) outs({empty} : {}) -> {}",
            ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.matmul ins({}, {} : {}, {}) outs({init} : {}) -> {}",
            lhs.name,
            rhs.name,
            lhs.ty.mlir_type(),
            rhs.ty.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn batch_matmul(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape: vec![lhs.ty.shape[0], lhs.ty.shape[1], rhs.ty.shape[2]],
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({zero} : {}) outs({empty} : {}) -> {}",
            ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.batch_matmul ins({}, {} : {}, {}) outs({init} : {}) -> {}",
            lhs.name,
            rhs.name,
            lhs.ty.mlir_type(),
            rhs.ty.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn conv2d(&mut self, input: Value, kernel: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![
                input.ty.shape[0],
                input.ty.shape[1] - kernel.ty.shape[0] + 1,
                input.ty.shape[2] - kernel.ty.shape[1] + 1,
                kernel.ty.shape[3],
            ],
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({zero} : {}) outs({empty} : {}) -> {}",
            ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.conv_2d_nhwc_hwcf ins({}, {} : {}, {}) outs({init} : {}) -> {}",
            input.name,
            kernel.name,
            input.ty.mlir_type(),
            kernel.ty.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn transpose(&mut self, input: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![input.ty.shape[1], input.ty.shape[0]],
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.transpose ins({} : {}) outs({empty} : {}) permutation = [1, 0]",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn reshape(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        if input.ty == *ty {
            return Ok(input);
        }
        let flat = if input.ty.rank() == 1 {
            input
        } else {
            let flat_ty = TensorType {
                elem: input.ty.elem,
                shape: vec![element_count(&input.ty)],
            };
            self.collapse_to_rank1(input, &flat_ty)?
        };
        if ty.rank() == 1 {
            Ok(flat)
        } else {
            self.expand_rank1(flat, ty)
        }
    }

    fn expand_rank1(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let name = self.fresh();
        let output_shape = format_shape_list(&ty.shape);
        let reassociation = reassociation_for_rank(ty.rank());
        self.lines.push(format!(
            "    {name} = tensor.expand_shape {} {reassociation} output_shape {output_shape} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn collapse_to_rank1(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let name = self.fresh();
        let reassociation = reassociation_for_rank(input.ty.rank());
        self.lines.push(format!(
            "    {name} = tensor.collapse_shape {} {reassociation} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn slice(&mut self, input: Value, ty: &TensorType, starts: &[usize]) -> anyhow::Result<Value> {
        let offsets = format_usize_list(starts);
        let sizes = format_usize_list(&ty.shape);
        let strides = format_usize_list(&vec![1; ty.rank()]);
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.extract_slice {}{offsets} {sizes} {strides} : {} to {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn take(&mut self, input: Value, axis: usize, index: usize) -> anyhow::Result<Value> {
        let mut starts = vec![0; input.ty.rank()];
        starts[axis] = index;
        let mut slice_shape = input.ty.shape.clone();
        slice_shape[axis] = 1;
        let slice_ty = TensorType {
            elem: input.ty.elem,
            shape: slice_shape,
        };
        let sliced = self.slice(input, &slice_ty, &starts)?;
        let mut output_shape = sliced.ty.shape.clone();
        output_shape.remove(axis);
        if output_shape.is_empty() {
            output_shape.push(1);
        }
        let output_ty = TensorType {
            elem: sliced.ty.elem,
            shape: output_shape,
        };
        if sliced.ty == output_ty {
            return Ok(sliced);
        }
        let name = self.fresh();
        let reassociation = collapse_reassociation_for_removed_axis(sliced.ty.rank(), axis);
        self.lines.push(format!(
            "    {name} = tensor.collapse_shape {} {reassociation} : {} into {}",
            sliced.name,
            sliced.ty.mlir_type(),
            output_ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: output_ty,
        })
    }

    fn concat(&mut self, lhs: Value, rhs: Value, axis: usize) -> anyhow::Result<Value> {
        let mut shape = lhs.ty.shape.clone();
        shape[axis] += rhs.ty.shape[axis];
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape,
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let lhs_offsets = vec![0; ty.rank()];
        let first = self.insert_slice(lhs, empty, &ty, &lhs_offsets)?;
        let mut rhs_offsets = vec![0; ty.rank()];
        rhs_offsets[axis] = ty.shape[axis] - rhs.ty.shape[axis];
        self.insert_slice(rhs, first.name, &ty, &rhs_offsets)
    }

    fn stack(&mut self, lhs: Value, rhs: Value, axis: usize) -> anyhow::Result<Value> {
        let mut unit_shape = lhs.ty.shape.clone();
        unit_shape.insert(axis, 1);
        let unit_ty = TensorType {
            elem: lhs.ty.elem,
            shape: unit_shape,
        };
        let lhs = self.expand_insert_axis(lhs, &unit_ty, axis)?;
        let rhs = self.expand_insert_axis(rhs, &unit_ty, axis)?;
        self.concat(lhs, rhs, axis)
    }

    fn insert_slice(
        &mut self,
        source: Value,
        dest: String,
        dest_ty: &TensorType,
        offsets: &[usize],
    ) -> anyhow::Result<Value> {
        let offsets = format_usize_list(offsets);
        let sizes = format_usize_list(&source.ty.shape);
        let strides = format_usize_list(&vec![1; source.ty.rank()]);
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.insert_slice {} into {dest}{offsets} {sizes} {strides} : {} into {}",
            source.name,
            source.ty.mlir_type(),
            dest_ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: dest_ty.clone(),
        })
    }

    fn expand_insert_axis(
        &mut self,
        input: Value,
        ty: &TensorType,
        axis: usize,
    ) -> anyhow::Result<Value> {
        let name = self.fresh();
        let output_shape = format_shape_list(&ty.shape);
        let reassociation = expand_reassociation_for_inserted_axis(input.ty.rank(), axis);
        self.lines.push(format!(
            "    {name} = tensor.expand_shape {} {reassociation} output_shape {output_shape} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn broadcast(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        if input.ty == *ty {
            return Ok(input);
        }
        if input.ty.rank() == 0 {
            return self.splat(input, ty);
        }
        if element_count(&input.ty) == 1 {
            let scalar = self.extract_first_scalar(input)?;
            return self.splat(scalar, ty);
        }
        ensure_broadcastable(&input.ty, ty)?;
        let (input, dimensions) = self.squeeze_singleton_broadcast_dims(input, ty)?;
        self.emit_linalg_broadcast(input, ty, &dimensions)
    }

    fn broadcast_along_axis(
        &mut self,
        input: Value,
        ty: &TensorType,
        axis: usize,
    ) -> anyhow::Result<Value> {
        if input.ty == *ty {
            return Ok(input);
        }
        if input.ty.rank() == 0 || element_count(&input.ty) == 1 {
            let scalar = if input.ty.rank() == 0 {
                input
            } else {
                self.extract_first_scalar(input)?
            };
            return self.splat(scalar, ty);
        }
        ensure_axis_broadcastable(&input.ty, ty, axis)?;
        let dimensions = axis_broadcast_dimensions(input.ty.rank(), ty.rank(), axis)?;
        self.emit_linalg_broadcast(input, ty, &dimensions)
    }

    fn emit_linalg_broadcast(
        &mut self,
        input: Value,
        ty: &TensorType,
        dimensions: &[usize],
    ) -> anyhow::Result<Value> {
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let dimensions = format_usize_list(dimensions);
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.broadcast ins({} : {}) outs({empty} : {}) dimensions = {dimensions}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn extract_first_scalar(&mut self, input: Value) -> anyhow::Result<Value> {
        let name = self.fresh();
        let zero = self.fresh();
        self.lines
            .push(format!("    {zero} = arith.constant 0 : index"));
        let indices = vec![zero; input.ty.rank()].join(", ");
        self.lines.push(format!(
            "    {name} = tensor.extract {}[{}] : {}",
            input.name,
            indices,
            input.ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: TensorType {
                elem: input.ty.elem,
                shape: Vec::new(),
            },
        })
    }

    fn squeeze_singleton_broadcast_dims(
        &mut self,
        input: Value,
        ty: &TensorType,
    ) -> anyhow::Result<(Value, Vec<usize>)> {
        let padding = ty.rank() - input.ty.rank();
        let singleton_dimensions = input
            .ty
            .shape
            .iter()
            .enumerate()
            .filter_map(|(index, input_dim)| {
                let output_dim = ty.shape[padding + index];
                (*input_dim == 1 && output_dim != 1).then_some(index)
            })
            .collect::<Vec<_>>();

        let mut dimensions = (0..padding).collect::<Vec<_>>();
        dimensions.extend(singleton_dimensions.iter().map(|index| padding + *index));

        if singleton_dimensions.is_empty() {
            return Ok((input, dimensions));
        }

        let squeezed_shape = input
            .ty
            .shape
            .iter()
            .enumerate()
            .filter_map(|(index, input_dim)| {
                let output_dim = ty.shape[padding + index];
                (!(*input_dim == 1 && output_dim != 1)).then_some(*input_dim)
            })
            .collect::<Vec<_>>();
        if squeezed_shape.is_empty() {
            return Ok((self.extract_first_scalar(input)?, (0..ty.rank()).collect()));
        }

        let squeezed_ty = TensorType {
            elem: input.ty.elem,
            shape: squeezed_shape,
        };
        let aligned_output_shape = &ty.shape[padding..];
        let reassociation =
            collapse_reassociation_for_squeezed_broadcast(&input.ty.shape, aligned_output_shape);
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.collapse_shape {} {reassociation} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            squeezed_ty.mlir_type()
        ));
        Ok((
            Value {
                name,
                ty: squeezed_ty,
            },
            dimensions,
        ))
    }

    fn mean(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        let count = axis.map_or_else(|| element_count(&input.ty), |axis| input.ty.shape[axis]);
        let elem = input.ty.elem;
        let sum = self.sum(input, axis)?;
        let scale = self.constant(&format!("{count}.0"), elem)?;
        self.emit_binary("arith.divf", sum, scale)
    }

    fn softmax(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        if let Some(axis) = axis {
            return self.axis_softmax(input, axis);
        }
        let max = self.max(input.clone(), axis, false)?;
        let max = if let Some(axis) = axis {
            self.broadcast_along_axis(max, &input.ty, axis)?
        } else {
            self.broadcast(max, &input.ty)?
        };
        let shifted = self.emit_binary("arith.subf", input, max)?;
        let exp = self.emit_unary("math.exp", shifted)?;
        let denominator = self.reduce(
            exp.clone(),
            exp.ty.elem.zero_literal(),
            "arith.addf",
            axis,
            false,
        )?;
        let denominator = if let Some(axis) = axis {
            self.broadcast_along_axis(denominator, &exp.ty, axis)?
        } else {
            self.broadcast(denominator, &exp.ty)?
        };
        self.emit_binary("arith.divf", exp, denominator)
    }

    fn axis_softmax(&mut self, input: Value, axis: usize) -> anyhow::Result<Value> {
        let empty = self.fresh();
        self.lines.push(format!(
            "    {empty} = tensor.empty() : {}",
            input.ty.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.softmax dimension({axis}) ins({} : {}) outs({empty} : {}) -> {}",
            input.name,
            input.ty.mlir_type(),
            input.ty.mlir_type(),
            input.ty.mlir_type()
        ));
        Ok(Value { name, ty: input.ty })
    }

    fn sigmoid(&mut self, input: Value) -> anyhow::Result<Value> {
        let one = self.constant(input.ty.elem.one_literal(), input.ty.elem)?;
        let zero = self.zero_like(&input.ty)?;
        let neg = self.emit_binary("arith.subf", zero, input)?;
        let exp = self.emit_unary("math.exp", neg)?;
        let denominator = self.emit_binary("arith.addf", one.clone(), exp)?;
        self.emit_binary("arith.divf", one, denominator)
    }

    fn argmax(&mut self, input: Value) -> anyhow::Result<Value> {
        if input.ty.rank() != 1 {
            anyhow::bail!("argmax lowering currently supports rank-1 tensors only");
        }
        let len = input.ty.shape[0];
        if len == 0 {
            anyhow::bail!("argmax lowering expects a non-empty tensor");
        }
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![1],
        };
        let zero = self.fresh();
        self.lines
            .push(format!("    {zero} = arith.constant 0 : index"));
        let one = self.fresh();
        self.lines
            .push(format!("    {one} = arith.constant 1 : index"));
        let upper = self.fresh();
        self.lines
            .push(format!("    {upper} = arith.constant {len} : index"));
        let first = self.fresh();
        self.lines.push(format!(
            "    {first} = tensor.extract {}[{zero}] : {}",
            input.name,
            input.ty.mlir_type()
        ));
        let best_index = self.fresh();
        let best_value = self.fresh();
        let next_value = self.fresh();
        let better = self.fresh();
        let selected_index = self.fresh();
        let selected_value = self.fresh();
        self.lines.push(format!(
            "    {best_index}, {best_value} = scf.for %i = {one} to {upper} step {one} iter_args(%best_i = {zero}, %best_v = {first}) -> (index, {}) {{",
            input.ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {next_value} = tensor.extract {}[%i] : {}",
            input.name,
            input.ty.mlir_type()
        ));
        self.lines.push(format!(
            "      {better} = arith.cmpf ogt, {next_value}, %best_v : {}",
            input.ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {selected_index} = arith.select {better}, %i, %best_i : index"
        ));
        self.lines.push(format!(
            "      {selected_value} = arith.select {better}, {next_value}, %best_v : {}",
            input.ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      scf.yield {selected_index}, {selected_value} : index, {}",
            input.ty.elem.mlir_type()
        ));
        self.lines.push("    }".to_string());
        let index_i64 = self.fresh();
        self.lines.push(format!(
            "    {index_i64} = arith.index_cast {best_index} : index to i64"
        ));
        let index_value = self.fresh();
        let conversion_op = match input.ty.elem {
            ElementType::F32 | ElementType::F64 => "arith.uitofp",
            ElementType::I32 | ElementType::I64 => "arith.index_cast",
        };
        self.lines.push(format!(
            "    {index_value} = {conversion_op} {index_i64} : i64 to {}",
            input.ty.elem.mlir_type()
        ));
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.insert {index_value} into {empty}[{zero}] : {}",
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn sum(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        let reducer_op = if input.ty.elem.is_float() {
            "arith.addf"
        } else {
            "arith.addi"
        };
        let initial_value = input.ty.elem.zero_literal();
        self.reduce(input, initial_value, reducer_op, axis, false)
    }

    fn max(&mut self, input: Value, axis: Option<usize>, keep_dims: bool) -> anyhow::Result<Value> {
        self.reduce(
            input.clone(),
            min_float_literal(input.ty.elem),
            "arith.maximumf",
            axis,
            keep_dims,
        )
    }

    fn reduce(
        &mut self,
        input: Value,
        initial_value: &str,
        reducer_op: &str,
        axis: Option<usize>,
        keep_dims: bool,
    ) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: reduction_output_shape(&input.ty.shape, axis, keep_dims),
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {initial_value} : {}",
            ty.elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({zero} : {}) outs({empty} : {}) -> {}",
            ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));

        let rank = input.ty.rank();
        let dims = format_dim_list(rank);
        let input_map = format!("({dims})");
        let iterator_types = (0..rank)
            .map(|index| {
                if axis.is_none() || axis == Some(index) {
                    "\"reduction\""
                } else {
                    "\"parallel\""
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let output_map = reduction_output_map(rank, axis, keep_dims);
        let reduced = self.fresh();
        let name = self.fresh();
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {input_map}>, affine_map<({dims}) -> {output_map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterator_types}]"));
        self.lines.push(format!(
            "    }} ins({} : {}) outs({init} : {}) {{",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%value: {}, %acc: {}):",
            ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {reduced} = {reducer_op} %acc, %value : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield {reduced} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value { name, ty })
    }

    fn fresh(&mut self) -> String {
        let name = format!("%{}", self.next_value);
        self.next_value += 1;
        name
    }
}
