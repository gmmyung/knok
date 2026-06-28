use knok_core::TensorType;

pub(crate) fn mlir_result_types(outputs: &[TensorType]) -> String {
    let types = outputs
        .iter()
        .map(TensorType::mlir_type)
        .collect::<Vec<_>>()
        .join(", ");
    if outputs.len() == 1 {
        types
    } else {
        format!("({types})")
    }
}
