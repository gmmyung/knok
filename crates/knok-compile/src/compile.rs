use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use fs2::FileExt;
use knok_core::TypedGraph;
use melior::{dialect::DialectRegistry, ir::operation::OperationLike, ir::Module, Context};

use crate::{
    backend::{backend_flags, IreeBackend},
    lowering::lower_to_mlir_with_registry,
};

static CACHE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct CompiledGraph {
    /// MLIR emitted from the typed graph before IREE compilation.
    pub mlir: String,
    /// IREE VM bytecode module bytes produced from the MLIR.
    pub vmfb: Vec<u8>,
}

/// Lowers and compiles a typed graph without a graph-call registry.
pub fn compile_graph(graph: &TypedGraph) -> anyhow::Result<CompiledGraph> {
    compile_graph_with_registry(graph, &BTreeMap::new())
}

/// Lowers and compiles a typed graph with additional callable graph definitions.
pub fn compile_graph_with_registry(
    graph: &TypedGraph,
    graphs: &BTreeMap<String, TypedGraph>,
) -> anyhow::Result<CompiledGraph> {
    let mlir = lower_to_mlir_with_registry(graph, graphs)?;
    verify_with_melior(&mlir)?;
    let vmfb = compile_mlir_source(&graph.backend, &mlir)?;
    Ok(CompiledGraph { mlir, vmfb })
}

/// Compiles an MLIR module string to IREE VM bytecode for the given backend.
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
    let _compiler_lock = lock_compiler_cache(&cache_dir)?;
    let compiler = iree_compile_command();
    let flags = backend_flags(backend, extra_flags);
    let version = compiler_version(&compiler)?;
    let key = cache_key(backend, mlir, &compiler, &version, &flags);
    let vmfb_path = cache_dir.join(format!("{key}.vmfb"));
    if let Some(vmfb) = read_cached_vmfb(&vmfb_path)? {
        return Ok(vmfb);
    }

    let suffix = unique_cache_temp_suffix();
    let tmp_mlir_path = cache_dir.join(format!("{key}.{suffix}.mlir.tmp"));
    let tmp_vmfb_path = cache_dir.join(format!("{key}.{suffix}.vmfb.tmp"));
    fs::write(&tmp_mlir_path, mlir)?;
    if let Err(error) = compile_with_iree_compile(&compiler, &flags, &tmp_mlir_path, &tmp_vmfb_path)
    {
        let _ = fs::remove_file(&tmp_mlir_path);
        let _ = fs::remove_file(&tmp_vmfb_path);
        return Err(error);
    }
    let _ = fs::remove_file(&tmp_mlir_path);
    let vmfb = fs::read(&tmp_vmfb_path)?;
    if vmfb.len() < 16 {
        let _ = fs::remove_file(&tmp_mlir_path);
        let _ = fs::remove_file(&tmp_vmfb_path);
        anyhow::bail!(
            "IREE compiler produced an invalid VMFB cache artifact with {} bytes",
            vmfb.len()
        );
    }
    match fs::rename(&tmp_vmfb_path, &vmfb_path) {
        Ok(()) => Ok(vmfb),
        Err(_) => {
            let _ = fs::remove_file(&tmp_mlir_path);
            let _ = fs::remove_file(&tmp_vmfb_path);
            read_cached_vmfb(&vmfb_path)?
                .ok_or_else(|| anyhow::anyhow!("failed to publish VMFB cache artifact"))
        }
    }
}

fn lock_compiler_cache(cache_dir: &Path) -> anyhow::Result<fs::File> {
    let lock_path = cache_dir.join("iree-compiler.lock");
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    file.lock_exclusive()?;
    Ok(file)
}

fn iree_compile_command() -> String {
    env::var("KNOK_IREE_COMPILE").unwrap_or_else(|_| "iree-compile".to_string())
}

fn compiler_version(compiler: &str) -> anyhow::Result<String> {
    let output = Command::new(compiler).arg("--version").output().map_err(|error| {
        anyhow::anyhow!(
            "failed to run IREE compiler `{compiler}`: {error}; set KNOK_IREE_COMPILE or install iree-compile"
        )
    })?;
    if !output.status.success() {
        anyhow::bail!(
            "IREE compiler `{compiler} --version` failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn compile_with_iree_compile(
    compiler: &str,
    flags: &[String],
    input_path: &Path,
    output_path: &Path,
) -> anyhow::Result<()> {
    let output = Command::new(compiler)
        .args(flags)
        .arg("-o")
        .arg(output_path)
        .arg(input_path)
        .output()
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to run IREE compiler `{compiler}`: {error}; set KNOK_IREE_COMPILE or install iree-compile"
            )
        })?;
    if !output.status.success() {
        let _ = fs::remove_file(output_path);
        anyhow::bail!(
            "IREE compiler `{compiler}` failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
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

fn cache_dir() -> anyhow::Result<PathBuf> {
    if let Ok(path) = env::var("KNOK_CACHE_DIR") {
        return Ok(PathBuf::from(path));
    }
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    Ok(Path::new(&manifest_dir).join("target/knok-cache"))
}

fn cache_key(
    backend: &str,
    mlir: &str,
    compiler: &str,
    compiler_version: &str,
    flags: &[String],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"knok-cache-v3");
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.update(backend.as_bytes());
    hasher.update(compiler.as_bytes());
    hasher.update(compiler_version.as_bytes());
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

#[cfg(test)]
mod tests {
    use knok_core::{type_check, CallOp, ElementType, Expr, Graph, Input, TensorType};

    use crate::lowering::lower_to_mlir;

    use super::{
        cache_key, compile_mlir_source, read_cached_vmfb, unique_cache_temp_suffix,
        verify_with_melior,
    };

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

    #[test]
    fn cache_key_changes_when_compile_inputs_change() {
        let flags = vec!["--flag-a".to_string()];
        let base = cache_key("llvm-cpu", "module {}", "iree-compile", "1", &flags);
        let different_backend = cache_key("metal-spirv", "module {}", "iree-compile", "1", &flags);
        let different_mlir = cache_key("llvm-cpu", "module @x {}", "iree-compile", "1", &flags);
        let different_compiler = cache_key("llvm-cpu", "module {}", "custom-compile", "1", &flags);
        let different_version = cache_key("llvm-cpu", "module {}", "iree-compile", "2", &flags);
        let different_flags = cache_key(
            "llvm-cpu",
            "module {}",
            "iree-compile",
            "1",
            &["--flag-b".to_string()],
        );

        assert_ne!(base, different_backend);
        assert_ne!(base, different_mlir);
        assert_ne!(base, different_compiler);
        assert_ne!(base, different_version);
        assert_ne!(base, different_flags);
    }

    #[test]
    fn unique_cache_temp_suffixes_are_distinct() {
        let first = unique_cache_temp_suffix();
        let second = unique_cache_temp_suffix();

        assert_ne!(first, second);
        assert!(first.contains(&std::process::id().to_string()));
    }

    #[test]
    fn invalid_cached_vmfb_is_removed_and_treated_as_miss() {
        let dir = std::env::temp_dir().join(format!(
            "knok-cache-test-{}-{}",
            std::process::id(),
            unique_cache_temp_suffix()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.vmfb");
        std::fs::write(&path, [0_u8; 4]).unwrap();

        let cached = read_cached_vmfb(&path).unwrap();

        assert!(cached.is_none());
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn valid_cached_vmfb_is_returned() {
        let dir = std::env::temp_dir().join(format!(
            "knok-cache-test-{}-{}",
            std::process::id(),
            unique_cache_temp_suffix()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("good.vmfb");
        let bytes = vec![1_u8; 16];
        std::fs::write(&path, &bytes).unwrap();

        let cached = read_cached_vmfb(&path).unwrap();

        assert_eq!(cached, Some(bytes));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unsupported_backend_errors_before_spawning_compiler() {
        let error = compile_mlir_source("unsupported", "module {}").unwrap_err();

        assert!(error
            .to_string()
            .contains("unsupported IREE backend `unsupported`"));
    }
}
