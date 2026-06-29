use std::{cell::RefCell, rc::Rc};

use melior::{dialect::DialectRegistry, ir::operation::OperationLike, ir::Module, Context};

fn context_with_all_dialects() -> (Context, Rc<RefCell<Vec<String>>>) {
    let registry = DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let diagnostics = Rc::new(RefCell::new(Vec::new()));
    let captured = diagnostics.clone();
    context.attach_diagnostic_handler(move |diagnostic| {
        captured.borrow_mut().push(diagnostic.to_string());
        true
    });

    (context, diagnostics)
}

pub(crate) fn canonicalize_and_verify(mlir: &str) -> anyhow::Result<String> {
    let (context, diagnostics) = context_with_all_dialects();
    let module = Module::parse(&context, mlir).ok_or_else(|| {
        let diagnostics = diagnostics.borrow();
        if diagnostics.is_empty() {
            anyhow::anyhow!("melior failed to parse generated MLIR")
        } else {
            anyhow::anyhow!(
                "melior failed to parse generated MLIR:\n{}",
                diagnostics.join("\n")
            )
        }
    })?;
    if !module.as_operation().verify() {
        let diagnostics = diagnostics.borrow();
        if diagnostics.is_empty() {
            anyhow::bail!("melior rejected generated MLIR");
        }
        anyhow::bail!(
            "melior rejected generated MLIR:\n{}",
            diagnostics.join("\n")
        );
    }
    Ok(module.as_operation().to_string())
}

#[cfg(test)]
mod tests {
    use super::canonicalize_and_verify;

    #[test]
    fn canonicalizes_valid_module() {
        let mlir = canonicalize_and_verify(
            r#"
module @knok {
  func.func @forward() -> tensor<1xf32> {
    %0 = tensor.empty() : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
"#,
        )
        .unwrap();

        assert!(mlir.contains("module @knok"));
        assert!(mlir.contains("func.func @forward"));
    }

    #[test]
    fn rejects_invalid_module() {
        let error = canonicalize_and_verify("module @knok {").unwrap_err();

        assert!(error.to_string().contains("generated MLIR"));
    }
}
