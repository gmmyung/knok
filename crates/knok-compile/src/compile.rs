use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use knok_core::TypedGraph;
use melior::{dialect::DialectRegistry, ir::operation::OperationLike, ir::Module, Context};

use crate::{
    backend::{backend_flags, BackendSpec, IreeBackend},
    lowering::lower_to_mlir_with_registry,
};

static CACHE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct CompiledGraph {
    pub mlir: String,
    pub vmfb: Vec<u8>,
}

pub(crate) struct CompiledVariant {
    pub(crate) backend: String,
    pub(crate) driver: String,
    pub(crate) compile_flags: Vec<String>,
    pub(crate) vmfb: Vec<u8>,
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

pub(crate) fn compile_mlir_variants(
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
    if IreeBackend::from_target_backend(backend).is_none() {
        anyhow::bail!("unsupported IREE backend `{backend}`; expected `llvm-cpu` or `metal-spirv`");
    }
    let cache_dir = cache_dir()?;
    fs::create_dir_all(&cache_dir)?;
    let iree_compile = iree_compile_command();
    let flags = backend_flags(backend, extra_flags);
    let key = cache_key(backend, mlir, &iree_compile, &flags);
    let vmfb_path = cache_dir.join(format!("{key}.vmfb"));
    if let Some(vmfb) = read_cached_vmfb(&vmfb_path)? {
        return Ok(vmfb);
    }

    let suffix = unique_cache_temp_suffix();
    let mlir_path = cache_dir.join(format!("{key}.{suffix}.mlir"));
    let tmp_vmfb_path = cache_dir.join(format!("{key}.{suffix}.vmfb.tmp"));
    fs::write(&mlir_path, mlir)?;
    let mut command = Command::new(&iree_compile);
    command
        .arg(&mlir_path)
        .args(&flags)
        .arg("-o")
        .arg(&tmp_vmfb_path);
    let output = command.output()?;
    let _ = fs::remove_file(&mlir_path);
    if !output.status.success() {
        let _ = fs::remove_file(&tmp_vmfb_path);
        anyhow::bail!(
            "iree-compile failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let vmfb = fs::read(&tmp_vmfb_path)?;
    if vmfb.len() < 16 {
        let _ = fs::remove_file(&tmp_vmfb_path);
        anyhow::bail!(
            "iree-compile produced an invalid VMFB cache artifact with {} bytes",
            vmfb.len()
        );
    }
    match fs::rename(&tmp_vmfb_path, &vmfb_path) {
        Ok(()) => Ok(vmfb),
        Err(_) => {
            let _ = fs::remove_file(&tmp_vmfb_path);
            read_cached_vmfb(&vmfb_path)?
                .ok_or_else(|| anyhow::anyhow!("failed to publish VMFB cache artifact"))
        }
    }
}

fn read_cached_vmfb(path: &Path) -> anyhow::Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(vmfb) if vmfb.len() >= 16 => Ok(Some(vmfb)),
        Ok(_) => {
            let _ = fs::remove_file(path);
            Ok(None)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn unique_cache_temp_suffix() -> String {
    let process_id = std::process::id();
    let counter = CACHE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{process_id}-{counter}")
}

fn iree_compile_command() -> String {
    env::var("KNOK_IREE_COMPILE").unwrap_or_else(|_| "iree-compile".to_string())
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

#[cfg(test)]
mod tests {
    use knok_core::{type_check, CallOp, ElementType, Expr, Graph, Input, TensorType};

    use crate::lowering::lower_to_mlir;

    use super::verify_with_melior;

    #[test]
    fn expm1_lowering_uses_direct_math_op() {
        let ty = TensorType {
            elem: ElementType::F32,
            shape: vec![4],
        };
        let graph = type_check(
            Graph {
                name: "expm1_graph".into(),
                backend: "llvm-cpu".into(),
                inputs: vec![Input {
                    name: "x".into(),
                    ty: ty.clone(),
                }],
                outputs: vec![ty],
                lets: Vec::new(),
                body: vec![Expr::Call {
                    op: CallOp::ExpM1,
                    args: vec![Expr::Var("x".into())],
                }],
            },
            &[],
        )
        .unwrap();

        let mlir = lower_to_mlir(&graph).unwrap();
        assert!(mlir.contains("math.expm1"), "{mlir}");
        assert!(!mlir.contains(" = math.exp "), "{mlir}");
        assert!(!mlir.contains("arith.subf"), "{mlir}");
        verify_with_melior(&mlir).unwrap();
    }

    #[test]
    fn tuple_projections_share_one_split_lowering() {
        let input = TensorType {
            elem: ElementType::F32,
            shape: vec![4],
        };
        let output = TensorType {
            elem: ElementType::F32,
            shape: vec![2],
        };
        let split = Expr::Call {
            op: CallOp::Split {
                axis: 0,
                sections: vec![2, 2],
            },
            args: vec![Expr::Var("x".into())],
        };
        let graph = type_check(
            Graph {
                name: "split_once".into(),
                backend: "llvm-cpu".into(),
                inputs: vec![Input {
                    name: "x".into(),
                    ty: input,
                }],
                outputs: vec![output.clone(), output],
                lets: Vec::new(),
                body: vec![
                    Expr::TupleGet {
                        tuple_id: 1,
                        value: Box::new(split.clone()),
                        index: 0,
                    },
                    Expr::TupleGet {
                        tuple_id: 1,
                        value: Box::new(split),
                        index: 1,
                    },
                ],
            },
            &[],
        )
        .unwrap();

        let mlir = lower_to_mlir(&graph).unwrap();
        assert_eq!(mlir.matches("tensor.extract_slice").count(), 2, "{mlir}");
        verify_with_melior(&mlir).unwrap();
    }

    #[test]
    fn cloned_single_output_node_lowers_once() {
        let lhs = TensorType {
            elem: ElementType::F32,
            shape: vec![2, 3],
        };
        let rhs = TensorType {
            elem: ElementType::F32,
            shape: vec![3, 2],
        };
        let output = TensorType {
            elem: ElementType::F32,
            shape: vec![2, 2],
        };
        let matmul = Expr::Node {
            node_id: 1,
            value: Box::new(Expr::Call {
                op: CallOp::Matmul,
                args: vec![Expr::Var("x".into()), Expr::Var("w".into())],
            }),
        };
        let graph = type_check(
            Graph {
                name: "reuse_matmul".into(),
                backend: "llvm-cpu".into(),
                inputs: vec![
                    Input {
                        name: "x".into(),
                        ty: lhs,
                    },
                    Input {
                        name: "w".into(),
                        ty: rhs,
                    },
                ],
                outputs: vec![output],
                lets: Vec::new(),
                body: vec![Expr::Binary {
                    op: knok_core::BinaryOp::Add,
                    lhs: Box::new(matmul.clone()),
                    rhs: Box::new(matmul),
                }],
            },
            &[],
        )
        .unwrap();

        let mlir = lower_to_mlir(&graph).unwrap();
        assert_eq!(mlir.matches("linalg.matmul").count(), 1, "{mlir}");
        verify_with_melior(&mlir).unwrap();
    }

    #[test]
    fn distinct_single_output_nodes_lower_separately() {
        let lhs = TensorType {
            elem: ElementType::F32,
            shape: vec![2, 3],
        };
        let rhs = TensorType {
            elem: ElementType::F32,
            shape: vec![3, 2],
        };
        let output = TensorType {
            elem: ElementType::F32,
            shape: vec![2, 2],
        };
        let matmul = |node_id| Expr::Node {
            node_id,
            value: Box::new(Expr::Call {
                op: CallOp::Matmul,
                args: vec![Expr::Var("x".into()), Expr::Var("w".into())],
            }),
        };
        let graph = type_check(
            Graph {
                name: "repeat_matmul".into(),
                backend: "llvm-cpu".into(),
                inputs: vec![
                    Input {
                        name: "x".into(),
                        ty: lhs,
                    },
                    Input {
                        name: "w".into(),
                        ty: rhs,
                    },
                ],
                outputs: vec![output],
                lets: Vec::new(),
                body: vec![Expr::Binary {
                    op: knok_core::BinaryOp::Add,
                    lhs: Box::new(matmul(1)),
                    rhs: Box::new(matmul(2)),
                }],
            },
            &[],
        )
        .unwrap();

        let mlir = lower_to_mlir(&graph).unwrap();
        assert_eq!(mlir.matches("linalg.matmul").count(), 2, "{mlir}");
        verify_with_melior(&mlir).unwrap();
    }

    #[test]
    fn distinct_tuple_ids_do_not_share_split_lowering() {
        let input = TensorType {
            elem: ElementType::F32,
            shape: vec![4],
        };
        let output = TensorType {
            elem: ElementType::F32,
            shape: vec![2],
        };
        let split = Expr::Call {
            op: CallOp::Split {
                axis: 0,
                sections: vec![2, 2],
            },
            args: vec![Expr::Var("x".into())],
        };
        let graph = type_check(
            Graph {
                name: "split_twice".into(),
                backend: "llvm-cpu".into(),
                inputs: vec![Input {
                    name: "x".into(),
                    ty: input,
                }],
                outputs: vec![output.clone(), output.clone(), output.clone(), output],
                lets: Vec::new(),
                body: vec![
                    Expr::TupleGet {
                        tuple_id: 1,
                        value: Box::new(split.clone()),
                        index: 0,
                    },
                    Expr::TupleGet {
                        tuple_id: 1,
                        value: Box::new(split.clone()),
                        index: 1,
                    },
                    Expr::TupleGet {
                        tuple_id: 2,
                        value: Box::new(split.clone()),
                        index: 0,
                    },
                    Expr::TupleGet {
                        tuple_id: 2,
                        value: Box::new(split),
                        index: 1,
                    },
                ],
            },
            &[],
        )
        .unwrap();

        let mlir = lower_to_mlir(&graph).unwrap();
        assert_eq!(mlir.matches("tensor.extract_slice").count(), 4, "{mlir}");
        verify_with_melior(&mlir).unwrap();
    }
}
