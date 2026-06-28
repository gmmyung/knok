use knok_core::{ElementType, TensorType, TypedGraph};

use crate::{Backend, Result};

pub(crate) fn graph_module(
    graph: &TypedGraph,
    backend: Backend,
    vmfb_name: &str,
    compile_flags: &[String],
) -> Result<String> {
    let module_name = sanitize_ident(&graph.name)?;
    let input_descs = graph
        .inputs
        .iter()
        .map(|input| tensor_desc(&input.ty))
        .collect::<Vec<_>>()
        .join(", ");
    let output_descs = graph
        .outputs
        .iter()
        .map(tensor_desc)
        .collect::<Vec<_>>()
        .join(", ");
    let compile_flags = compile_flags
        .iter()
        .map(|flag| format!("{flag:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let function_name = format!("knok.{}", graph.name);
    let input_names = graph
        .inputs
        .iter()
        .map(|input| sanitize_ident(&input.name))
        .collect::<Result<Vec<_>>>()?;
    let input_params = graph
        .inputs
        .iter()
        .zip(input_names.iter())
        .map(|(input, name)| format!("{}: {}", name, rust_tensor_type(&input.ty)))
        .collect::<Vec<_>>()
        .join(", ");
    let runtime_inputs = graph
        .inputs
        .iter()
        .zip(input_names.iter())
        .map(|(input, name)| {
            let shape = shape_array(&input.ty);
            format!(
                "::knok::runtime::raw::Input::{}(&{}, {}.as_slice())",
                runtime_input_variant(input.ty.elem),
                shape,
                name
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let output_type = rust_output_type(&graph.outputs);
    let call_args = input_names.join(", ");
    let run_body = run_body(&graph.outputs, &runtime_inputs);

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

    pub fn artifact() -> ::knok::GraphArtifact {{
        ::knok::GraphArtifact {{
            function_name: "{function_name}",
            input_descs: INPUT_DESCS,
            output_descs: OUTPUT_DESCS,
            variants: VARIANTS,
        }}
    }}

    pub fn run(engine: &::knok::Engine, {input_params}) -> ::knok::Result<{output_type}> {{
        let artifact = artifact();
        {run_body}
    }}

    pub fn call({input_params}) -> ::knok::Result<{output_type}> {{
        let engine = ::knok::Engine::for_artifact(artifact())?;
        run(&engine, {call_args})
    }}
}}
"#,
        backend = backend.name(),
        driver = backend.default_driver(),
    ))
}

fn run_body(outputs: &[TensorType], runtime_inputs: &str) -> String {
    if outputs.len() == 1 {
        let output = &outputs[0];
        format!(
            "let output = ::knok::__private::invoke_one_with_engine::<{}>(engine, artifact, &[{}])?;\n        <{}>::from_vec(output)",
            rust_element_type(output.elem),
            runtime_inputs,
            rust_tensor_type(output),
        )
    } else {
        let reads = outputs
            .iter()
            .enumerate()
            .map(|(index, output)| {
                format!(
                    "<{}>::from_vec(outputs.read::<{}>({index})?)?",
                    rust_tensor_type(output),
                    rust_element_type(output.elem)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "let outputs = engine.invoke(artifact, &[{}])?;\n        Ok(({reads}))",
            runtime_inputs
        )
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

fn runtime_input_variant(elem: ElementType) -> &'static str {
    match elem {
        ElementType::Bool => "Bool",
        ElementType::F32 => "F32",
        ElementType::F64 => "F64",
        ElementType::F16 => "F16",
        ElementType::BF16 => "BF16",
        ElementType::I32 => "I32",
        ElementType::I64 => "I64",
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
