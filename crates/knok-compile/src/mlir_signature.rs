use knok_core::{ElementType, TensorType};

pub(crate) fn validate_mlir_model_signature(
    mlir: &str,
    function_name: &str,
    expected_inputs: &[TensorType],
    expected_outputs: &[TensorType],
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
    if signature.outputs != expected_outputs {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "mlir_model outputs do not match MLIR function `{function_name}`: declared {:?}, MLIR has {:?}",
                expected_outputs, signature.outputs
            ),
        ));
    }
    Ok(())
}

struct MlirSignature {
    inputs: Vec<TensorType>,
    outputs: Vec<TensorType>,
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
    let after_arrow = after_args[arrow..].trim();
    let output = mlir_function_result_text(after_arrow)?;

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
        outputs: parse_mlir_result_types(output)?,
    })
}

fn mlir_function_result_text(after_arrow: &str) -> Option<&str> {
    let mut angle_depth = 0usize;
    let mut paren_depth = 0usize;
    for (index, ch) in after_arrow.char_indices() {
        match ch {
            '<' => angle_depth += 1,
            '>' => angle_depth = angle_depth.saturating_sub(1),
            '(' if angle_depth == 0 => paren_depth += 1,
            ')' if angle_depth == 0 => paren_depth = paren_depth.saturating_sub(1),
            '{' if angle_depth == 0 && paren_depth == 0 => {
                return Some(after_arrow[..index].trim());
            }
            _ if angle_depth == 0
                && paren_depth == 0
                && starts_mlir_keyword(after_arrow, index, "attributes") =>
            {
                return Some(after_arrow[..index].trim());
            }
            _ => {}
        }
    }
    None
}

fn starts_mlir_keyword(input: &str, index: usize, keyword: &str) -> bool {
    let tail = &input[index..];
    if !tail.starts_with(keyword) {
        return false;
    }
    let before_boundary = input[..index]
        .chars()
        .next_back()
        .map_or(true, |ch| ch.is_whitespace());
    let after_boundary = tail[keyword.len()..]
        .chars()
        .next()
        .map_or(true, |ch| ch.is_whitespace() || ch == '{');
    before_boundary && after_boundary
}

fn parse_mlir_result_types(output: &str) -> Option<Vec<TensorType>> {
    let output = output.trim();
    if output.starts_with('(') {
        let inner = output.strip_prefix('(')?.strip_suffix(')')?;
        if inner.trim().is_empty() {
            return None;
        }
        split_top_level(inner, ',')
            .into_iter()
            .map(parse_mlir_tensor_type)
            .collect()
    } else {
        Some(vec![parse_mlir_tensor_type(output)?])
    }
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
        "i1" => Some(ElementType::Bool),
        "f32" => Some(ElementType::F32),
        "f64" => Some(ElementType::F64),
        "f16" => Some(ElementType::F16),
        "bf16" => Some(ElementType::BF16),
        "i32" => Some(ElementType::I32),
        "i64" => Some(ElementType::I64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize]) -> TensorType {
        TensorType {
            elem: ElementType::F32,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn parses_mlir_signature_with_function_attributes() {
        let mlir = r#"
module @imported {
  func.func @foo(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes { iree.abi.stub } {
    return %arg0 : tensor<4xf32>
  }
}
"#;

        let signature = find_mlir_function_signature(mlir, "foo").unwrap();

        assert_eq!(signature.inputs, vec![tensor(&[4])]);
        assert_eq!(signature.outputs, vec![tensor(&[4])]);
    }

    #[test]
    fn parses_multi_result_mlir_signature_with_function_attributes() {
        let mlir = r#"
module @imported {
  func.func @foo(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes { iree.abi.stub } {
    return %arg0, %arg1 : tensor<4xf32>, tensor<4xf32>
  }
}
"#;

        let signature = find_mlir_function_signature(mlir, "foo").unwrap();

        assert_eq!(signature.inputs, vec![tensor(&[4]), tensor(&[4])]);
        assert_eq!(signature.outputs, vec![tensor(&[4]), tensor(&[4])]);
    }
}
