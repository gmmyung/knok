use knok_core::{ElementType, Input, TensorType, TypedGraph};

use crate::{Backend, MlirModel, Result};

const MAX_GRAPH_TUPLE_ARITY: usize = 12;

pub(crate) fn graph_module(
    graph: &TypedGraph,
    backend: Backend,
    vmfb_name: &str,
    compile_flags: &[String],
) -> Result<String> {
    let function_name = format!("knok.{}", graph.name);
    generated_module(
        &graph.name,
        &function_name,
        &graph.inputs,
        &graph.outputs,
        backend,
        vmfb_name,
        compile_flags,
    )
}

pub(crate) fn mlir_model_module(
    model: &MlirModel,
    vmfb_name: &str,
    compile_flags: &[String],
) -> Result<String> {
    generated_module(
        &model.name,
        &model.function_name,
        &model.inputs,
        &model.outputs,
        model.backend,
        vmfb_name,
        compile_flags,
    )
}

fn generated_module(
    name: &str,
    function_name: &str,
    inputs: &[Input],
    outputs: &[TensorType],
    backend: Backend,
    vmfb_name: &str,
    compile_flags: &[String],
) -> Result<String> {
    let module_name = sanitize_ident(name)?;
    if !is_vm_function_name(function_name) {
        anyhow::bail!("`{function_name}` is not a valid IREE VM function name");
    }
    ensure_supported_tuple_arity("input", inputs.len())?;
    ensure_supported_tuple_arity("output", outputs.len())?;
    let input_descs = inputs
        .iter()
        .map(|input| tensor_desc(&input.ty))
        .collect::<Vec<_>>()
        .join(", ");
    let output_descs = outputs
        .iter()
        .map(tensor_desc)
        .collect::<Vec<_>>()
        .join(", ");
    let compile_flags = compile_flags
        .iter()
        .map(|flag| format!("{flag:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let input_names = inputs
        .iter()
        .map(|input| sanitize_ident(&input.name))
        .collect::<Result<Vec<_>>>()?;
    let input_params = inputs
        .iter()
        .zip(input_names.iter())
        .map(|(input, name)| format!("{}: {}", name, rust_tensor_type(&input.ty)))
        .collect::<Vec<_>>()
        .join(", ");
    let input_type = rust_input_type(inputs);
    let input_expr = rust_input_expr(&input_names);
    let output_type = rust_output_type(outputs);
    let run_params = prefixed_params("engine: &::knok::Engine", &input_params);
    let call_params = input_params.clone();

    Ok(format!(
        r#"pub mod {module_name} {{
    static VMFB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/{vmfb_name}"));
    static COMPILE_FLAGS: &[&str] = &[{compile_flags}];
    static VARIANTS: &[::knok::GraphArtifactVariant] = &[::knok::GraphArtifactVariant {{
        vmfb: VMFB,
        backend: "{backend}",
        driver: "{driver}",
        compile_flags: COMPILE_FLAGS,
    }}];
    static INPUT_DESCS: &[::knok::TensorDesc] = &[{input_descs}];
    static OUTPUT_DESCS: &[::knok::TensorDesc] = &[{output_descs}];

    pub const fn artifact() -> ::knok::GraphArtifact {{
        ::knok::GraphArtifact {{
            function_name: "{function_name}",
            input_descs: INPUT_DESCS,
            output_descs: OUTPUT_DESCS,
            variants: VARIANTS,
        }}
    }}

    pub static GRAPH: ::knok::Graph<{input_type}, {output_type}> = ::knok::Graph::new(artifact());

    pub fn run({run_params}) -> ::knok::Result<{output_type}> {{
        GRAPH.run(engine, {input_expr})
    }}

    pub fn call({call_params}) -> ::knok::Result<{output_type}> {{
        GRAPH.run_once({input_expr})
    }}
}}
"#,
        backend = backend.name(),
        driver = backend.default_driver(),
    ))
}

fn ensure_supported_tuple_arity(kind: &str, len: usize) -> Result<()> {
    if len > MAX_GRAPH_TUPLE_ARITY {
        anyhow::bail!(
            "generated graph {kind} arity {len} exceeds the supported maximum of {MAX_GRAPH_TUPLE_ARITY}"
        );
    }
    Ok(())
}

fn prefixed_params(prefix: &str, params: &str) -> String {
    if params.is_empty() {
        prefix.into()
    } else {
        format!("{prefix}, {params}")
    }
}

fn sanitize_ident(name: &str) -> Result<String> {
    if is_ident(name) && !is_keyword(name) {
        Ok(name.into())
    } else {
        anyhow::bail!("`{name}` is not a valid generated Rust identifier")
    }
}

fn is_ident(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn is_keyword(name: &str) -> bool {
    matches!(
        name,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
    )
}

fn is_vm_function_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|ch| ch == '.' || ch == '_' || ch.is_ascii_alphanumeric())
}

fn tensor_desc(ty: &TensorType) -> String {
    format!(
        "::knok::TensorDesc::new({}, &{})",
        dtype_expr(ty.elem),
        shape_array(ty)
    )
}

fn shape_array(ty: &TensorType) -> String {
    let dims = ty
        .shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{dims}]")
}

fn rust_output_type(outputs: &[TensorType]) -> String {
    if outputs.len() == 1 {
        rust_tensor_type(&outputs[0])
    } else {
        format!(
            "({})",
            outputs
                .iter()
                .map(rust_tensor_type)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn rust_input_type(inputs: &[Input]) -> String {
    match inputs {
        [] => "()".into(),
        [input] => rust_tensor_type(&input.ty),
        _ => format!(
            "({})",
            inputs
                .iter()
                .map(|input| rust_tensor_type(&input.ty))
                .map(|ty| format!("{ty},"))
                .collect::<Vec<_>>()
                .join(" ")
        ),
    }
}

fn rust_input_expr(input_names: &[String]) -> String {
    match input_names {
        [] => "()".into(),
        [name] => name.clone(),
        _ => format!(
            "({})",
            input_names
                .iter()
                .map(|name| format!("{name},"))
                .collect::<Vec<_>>()
                .join(" ")
        ),
    }
}

fn rust_tensor_type(ty: &TensorType) -> String {
    let elem = rust_element_type(ty.elem);
    match ty.shape.as_slice() {
        [] => format!("::knok::tensor::Tensor0<{elem}>"),
        [d0] => format!("::knok::tensor::Tensor1<{elem}, {d0}>"),
        [d0, d1] => format!("::knok::tensor::Tensor2<{elem}, {d0}, {d1}>"),
        [d0, d1, d2] => format!("::knok::tensor::Tensor3<{elem}, {d0}, {d1}, {d2}>"),
        [d0, d1, d2, d3] => {
            format!("::knok::tensor::Tensor4<{elem}, {d0}, {d1}, {d2}, {d3}>")
        }
        [d0, d1, d2, d3, d4] => {
            format!("::knok::tensor::Tensor5<{elem}, {d0}, {d1}, {d2}, {d3}, {d4}>")
        }
        [d0, d1, d2, d3, d4, d5] => {
            format!("::knok::tensor::Tensor6<{elem}, {d0}, {d1}, {d2}, {d3}, {d4}, {d5}>")
        }
        _ => panic!(
            "rank {} cannot be represented by knok tensor containers",
            ty.shape.len()
        ),
    }
}

fn rust_element_type(elem: ElementType) -> &'static str {
    match elem {
        ElementType::Bool => "bool",
        ElementType::F32 => "f32",
        ElementType::F64 => "f64",
        ElementType::F16 => "::knok::half::f16",
        ElementType::BF16 => "::knok::half::bf16",
        ElementType::I32 => "i32",
        ElementType::I64 => "i64",
    }
}

fn dtype_expr(elem: ElementType) -> &'static str {
    match elem {
        ElementType::Bool => "::knok::DType::Bool",
        ElementType::F32 => "::knok::DType::F32",
        ElementType::F64 => "::knok::DType::F64",
        ElementType::F16 => "::knok::DType::F16",
        ElementType::BF16 => "::knok::DType::BF16",
        ElementType::I32 => "::knok::DType::I32",
        ElementType::I64 => "::knok::DType::I64",
    }
}

#[cfg(test)]
mod tests {
    use knok_core::{Input, TensorType, TypedGraph};

    use super::*;

    fn ty(elem: ElementType, shape: &[usize]) -> TensorType {
        TensorType {
            elem,
            shape: shape.to_vec(),
        }
    }

    fn input(name: &str, elem: ElementType, shape: &[usize]) -> Input {
        Input {
            name: name.into(),
            ty: ty(elem, shape),
        }
    }

    fn typed_graph(name: &str, inputs: Vec<Input>, outputs: Vec<TensorType>) -> TypedGraph {
        TypedGraph {
            name: name.into(),
            backend: "llvm-cpu".into(),
            inputs,
            outputs,
            lets: Vec::new(),
            body: Vec::new(),
        }
    }

    #[test]
    fn generates_single_output_wrapper_metadata_and_runtime_inputs() {
        let graph = typed_graph(
            "forward",
            vec![input("x", ElementType::F32, &[2, 3])],
            vec![ty(ElementType::F32, &[2, 3])],
        );

        let module = graph_module(
            &graph,
            Backend::LlvmCpu,
            "forward.vmfb",
            &["--some-flag".into()],
        )
        .unwrap();

        assert!(module.contains("pub mod forward"));
        assert!(module.contains("include_bytes!(concat!(env!(\"OUT_DIR\"), \"/forward.vmfb\"))"));
        assert!(module.contains("function_name: \"knok.forward\""));
        assert!(module.contains("backend: \"llvm-cpu\""));
        assert!(module.contains("driver: \"local-task\""));
        assert!(module.contains("static COMPILE_FLAGS: &[&str] = &[\"--some-flag\"]"));
        assert!(module.contains("::knok::TensorDesc::new(::knok::DType::F32, &[2, 3])"));
        assert!(module.contains("x: ::knok::tensor::Tensor2<f32, 2, 3>"));
        assert!(module.contains(
            "pub static GRAPH: ::knok::Graph<::knok::tensor::Tensor2<f32, 2, 3>, ::knok::tensor::Tensor2<f32, 2, 3>>"
        ));
        assert!(module.contains("GRAPH.run(engine, x)"));
        assert!(module.contains("GRAPH.run_once(x)"));
        assert!(!module.contains("__private"));
    }

    #[test]
    fn generates_multi_output_reads_by_index_and_dtype() {
        let graph = typed_graph(
            "stats",
            vec![input("values", ElementType::F32, &[2, 3])],
            vec![ty(ElementType::F32, &[2]), ty(ElementType::I64, &[2])],
        );

        let module = graph_module(&graph, Backend::LlvmCpu, "stats.vmfb", &[]).unwrap();

        assert!(module.contains(
            "pub fn run(engine: &::knok::Engine, values: ::knok::tensor::Tensor2<f32, 2, 3>) -> ::knok::Result<(::knok::tensor::Tensor1<f32, 2>, ::knok::tensor::Tensor1<i64, 2>)>"
        ));
        assert!(module.contains(
            "pub static GRAPH: ::knok::Graph<::knok::tensor::Tensor2<f32, 2, 3>, (::knok::tensor::Tensor1<f32, 2>, ::knok::tensor::Tensor1<i64, 2>)>"
        ));
        assert!(module.contains("GRAPH.run(engine, values)"));
        assert!(module.contains("GRAPH.run_once(values)"));
        assert!(!module.contains("__private"));
    }

    #[test]
    fn generates_external_mlir_model_wrapper_with_imported_function_name() {
        let model = MlirModel::new(
            "imported_add",
            "models/add.mlir",
            "imported.add",
            Backend::LlvmCpu,
            vec![
                input("x", ElementType::F32, &[4]),
                input("y", ElementType::F32, &[4]),
            ],
            vec![ty(ElementType::F32, &[4])],
        );

        let module = mlir_model_module(&model, "mlir-model-imported_add.vmfb", &[]).unwrap();

        assert!(module.contains("pub mod imported_add"));
        assert!(module.contains(
            "include_bytes!(concat!(env!(\"OUT_DIR\"), \"/mlir-model-imported_add.vmfb\"))"
        ));
        assert!(module.contains("function_name: \"imported.add\""));
        assert!(module.contains("x: ::knok::tensor::Tensor1<f32, 4>"));
        assert!(module.contains("y: ::knok::tensor::Tensor1<f32, 4>"));
        assert!(module.contains(
            "pub static GRAPH: ::knok::Graph<(::knok::tensor::Tensor1<f32, 4>, ::knok::tensor::Tensor1<f32, 4>,), ::knok::tensor::Tensor1<f32, 4>>"
        ));
        assert!(module.contains("GRAPH.run(engine, (x, y,))"));
        assert!(module.contains("GRAPH.run_once((x, y,))"));
        assert!(!module.contains("__private"));
    }

    #[test]
    fn rejects_invalid_external_vm_function_names() {
        let model = MlirModel::new(
            "imported_add",
            "models/add.mlir",
            "imported.add\"",
            Backend::LlvmCpu,
            Vec::new(),
            vec![ty(ElementType::F32, &[4])],
        );

        assert!(mlir_model_module(&model, "imported_add.vmfb", &[])
            .unwrap_err()
            .to_string()
            .contains("not a valid IREE VM function name"));
    }

    #[test]
    fn rejects_tuple_arities_that_runtime_traits_do_not_support() {
        let inputs = (0..13)
            .map(|index| input(&format!("x{index}"), ElementType::F32, &[1]))
            .collect::<Vec<_>>();
        let too_many_inputs =
            typed_graph("too_many_inputs", inputs, vec![ty(ElementType::F32, &[1])]);

        assert!(
            graph_module(&too_many_inputs, Backend::LlvmCpu, "bad.vmfb", &[])
                .unwrap_err()
                .to_string()
                .contains("input arity 13 exceeds the supported maximum of 12")
        );

        let outputs = (0..13)
            .map(|_| ty(ElementType::F32, &[1]))
            .collect::<Vec<_>>();
        let too_many_outputs = typed_graph(
            "too_many_outputs",
            vec![input("x", ElementType::F32, &[1])],
            outputs,
        );

        assert!(
            graph_module(&too_many_outputs, Backend::LlvmCpu, "bad.vmfb", &[])
                .unwrap_err()
                .to_string()
                .contains("output arity 13 exceeds the supported maximum of 12")
        );
    }

    #[test]
    fn generates_rank_zero_and_rank_six_tensor_types() {
        assert_eq!(
            rust_tensor_type(&ty(ElementType::Bool, &[])),
            "::knok::tensor::Tensor0<bool>"
        );
        assert_eq!(
            rust_tensor_type(&ty(ElementType::I32, &[1, 2, 3, 4, 5, 6])),
            "::knok::tensor::Tensor6<i32, 1, 2, 3, 4, 5, 6>"
        );
        assert_eq!(shape_array(&ty(ElementType::F64, &[])), "[]");
        assert_eq!(
            tensor_desc(&ty(ElementType::BF16, &[4])),
            "::knok::TensorDesc::new(::knok::DType::BF16, &[4])"
        );
    }

    #[test]
    fn rejects_generated_identifiers_that_are_invalid_or_keywords() {
        let bad_module = typed_graph("type", Vec::new(), vec![ty(ElementType::F32, &[])]);
        assert!(graph_module(&bad_module, Backend::LlvmCpu, "bad.vmfb", &[])
            .unwrap_err()
            .to_string()
            .contains("not a valid generated Rust identifier"));

        let bad_input = typed_graph(
            "ok",
            vec![input("not-valid", ElementType::F32, &[1])],
            vec![ty(ElementType::F32, &[1])],
        );
        assert!(graph_module(&bad_input, Backend::LlvmCpu, "bad.vmfb", &[])
            .unwrap_err()
            .to_string()
            .contains("not a valid generated Rust identifier"));
    }

    #[test]
    #[should_panic(expected = "rank 7 cannot be represented")]
    fn panics_when_codegen_is_asked_for_unsupported_tensor_rank() {
        let _ = rust_tensor_type(&ty(ElementType::F32, &[1, 1, 1, 1, 1, 1, 1]));
    }
}
