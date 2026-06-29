use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use fs2::FileExt;
use knok_core::TypedGraph;

use crate::{
    backend::{backend_flags, IreeBackend},
    lowering::lower_to_mlir_with_registry,
    mlir::canonicalize_and_verify,
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
    let backend = parse_backend(&graph.backend)?;
    let vmfb = compile_verified_mlir_source(backend, &mlir)?;
    Ok(CompiledGraph { mlir, vmfb })
}

/// Compiles an MLIR module string to IREE VM bytecode for the given backend.
pub fn compile_mlir_source(backend: &str, mlir: &str) -> anyhow::Result<Vec<u8>> {
    let backend = parse_backend(backend)?;
    let mlir = canonicalize_and_verify(mlir)?;
    compile_verified_mlir_source(backend, &mlir)
}

fn parse_backend(backend: &str) -> anyhow::Result<IreeBackend> {
    IreeBackend::from_target_backend(backend).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported IREE backend `{backend}`; expected `llvm-cpu` or `metal-spirv`"
        )
    })
}

fn compile_verified_mlir_source(backend: IreeBackend, mlir: &str) -> anyhow::Result<Vec<u8>> {
    compile_with_iree(backend, &[], mlir)
}

fn compile_with_iree(
    backend: IreeBackend,
    extra_flags: &[String],
    mlir: &str,
) -> anyhow::Result<Vec<u8>> {
    let cache_dir = cache_dir()?;
    fs::create_dir_all(&cache_dir)?;
    let _compiler_lock = lock_compiler_cache(&cache_dir)?;
    let compiler = iree_compile_command();
    let backend_name = backend.target_name();
    let flags = backend_flags(backend_name, extra_flags);
    let version = compiler_version(&compiler)?;
    let key = cache_key(backend_name, mlir, &compiler, &version, &flags);
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
    use knok_core::{
        type_check, AxisSpec, CallOp, Conv2dOptions, ElementType, Expr, Graph, Input, Padding2d,
        TensorType,
    };

    use crate::{lowering::lower_to_mlir, mlir::canonicalize_and_verify};

    use super::{cache_key, compile_mlir_source, read_cached_vmfb, unique_cache_temp_suffix};

    fn tensor(elem: ElementType, shape: &[usize]) -> TensorType {
        TensorType {
            elem,
            shape: shape.to_vec(),
        }
    }

    fn input(name: &str, elem: ElementType, shape: &[usize]) -> Input {
        Input {
            name: name.into(),
            ty: tensor(elem, shape),
        }
    }

    fn var(name: &str) -> Expr {
        Expr::Var(name.into())
    }

    fn constant(value: &str, elem: ElementType) -> Expr {
        Expr::Const {
            value: value.into(),
            elem,
        }
    }

    fn call(op: CallOp, args: Vec<Expr>) -> Expr {
        Expr::Call { op, args }
    }

    fn typed_graph(
        name: &str,
        inputs: Vec<Input>,
        outputs: Vec<TensorType>,
        body: Vec<Expr>,
    ) -> knok_core::TypedGraph {
        type_check(
            Graph {
                name: name.into(),
                backend: "llvm-cpu".into(),
                inputs,
                outputs,
                lets: Vec::new(),
                body,
            },
            &[],
        )
        .unwrap()
    }

    fn lower_verified(graph: &knok_core::TypedGraph) -> String {
        let mlir = lower_to_mlir(graph).unwrap();
        canonicalize_and_verify(&mlir).unwrap();
        mlir
    }

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
        canonicalize_and_verify(&mlir).unwrap();
    }

    #[test]
    fn lowers_elementwise_creation_and_predicate_ops() {
        let f32x4 = tensor(ElementType::F32, &[4]);
        let boolx4 = tensor(ElementType::Bool, &[4]);
        let i32x4 = tensor(ElementType::I32, &[4]);
        let f32x2x2 = tensor(ElementType::F32, &[2, 2]);
        let unary_ops = [
            CallOp::Abs,
            CallOp::Ceil,
            CallOp::Exp,
            CallOp::Exp2,
            CallOp::ExpM1,
            CallOp::Floor,
            CallOp::Log,
            CallOp::Log1P,
            CallOp::Log2,
            CallOp::Log10,
            CallOp::Relu,
            CallOp::Rint,
            CallOp::Round,
            CallOp::Sigmoid,
            CallOp::Sin,
            CallOp::Cos,
            CallOp::Sqrt,
            CallOp::Tan,
            CallOp::Tanh,
            CallOp::Square,
            CallOp::Reciprocal,
            CallOp::ZerosLike,
            CallOp::OnesLike,
        ];
        let mut outputs = Vec::new();
        let mut body = Vec::new();
        for op in unary_ops {
            outputs.push(f32x4.clone());
            body.push(call(op, vec![var("x")]));
        }
        for op in [
            CallOp::Minimum,
            CallOp::Maximum,
            CallOp::Pow,
            CallOp::LogicalAnd,
            CallOp::LogicalOr,
            CallOp::LogicalXor,
        ] {
            outputs.push(
                if matches!(
                    op,
                    CallOp::LogicalAnd | CallOp::LogicalOr | CallOp::LogicalXor
                ) {
                    boolx4.clone()
                } else {
                    f32x4.clone()
                },
            );
            let args = if matches!(
                op,
                CallOp::LogicalAnd | CallOp::LogicalOr | CallOp::LogicalXor
            ) {
                vec![var("mask"), var("mask")]
            } else {
                vec![var("x"), var("y")]
            };
            body.push(call(op, args));
        }
        for op in [
            CallOp::Greater,
            CallOp::GreaterEqual,
            CallOp::Less,
            CallOp::LessEqual,
            CallOp::Equal,
            CallOp::NotEqual,
        ] {
            outputs.push(boolx4.clone());
            body.push(call(op, vec![var("x"), var("y")]));
        }
        outputs.extend([
            boolx4.clone(),
            boolx4.clone(),
            f32x4.clone(),
            f32x4.clone(),
            i32x4.clone(),
            f32x4.clone(),
            f32x2x2.clone(),
        ]);
        body.extend([
            call(CallOp::IsNan, vec![var("x")]),
            call(CallOp::LogicalNot, vec![var("mask")]),
            call(
                CallOp::Clip,
                vec![
                    var("x"),
                    constant("0.0", ElementType::F32),
                    constant("6.0", ElementType::F32),
                ],
            ),
            call(CallOp::Where, vec![var("mask"), var("x"), var("y")]),
            call(
                CallOp::Arange(i32x4),
                vec![
                    constant("0", ElementType::I32),
                    constant("8", ElementType::I32),
                    constant("2", ElementType::I32),
                ],
            ),
            call(
                CallOp::Linspace(f32x4),
                vec![
                    constant("0.0", ElementType::F32),
                    constant("1.0", ElementType::F32),
                ],
            ),
            call(CallOp::Eye(f32x2x2), Vec::new()),
        ]);
        let graph = typed_graph(
            "elementwise_creation",
            vec![
                input("x", ElementType::F32, &[4]),
                input("y", ElementType::F32, &[4]),
                input("mask", ElementType::Bool, &[4]),
            ],
            outputs,
            body,
        );

        let mlir = lower_verified(&graph);
        assert!(mlir.contains("math.expm1"), "{mlir}");
        assert!(mlir.contains("arith.select"), "{mlir}");
    }

    #[test]
    fn lowers_reduction_ops() {
        let f32_scalar = tensor(ElementType::F32, &[]);
        let bool_scalar = tensor(ElementType::Bool, &[]);
        let i64_scalar = tensor(ElementType::I64, &[]);
        let f32_rows = tensor(ElementType::F32, &[2]);
        let bool_rows = tensor(ElementType::Bool, &[2]);
        let i64_rows = tensor(ElementType::I64, &[2]);
        let mut outputs = Vec::new();
        let mut body = Vec::new();

        for op in [
            CallOp::Sum(AxisSpec::All),
            CallOp::Prod(AxisSpec::All),
            CallOp::Mean(AxisSpec::All),
            CallOp::Max(AxisSpec::All),
            CallOp::Min(AxisSpec::All),
            CallOp::Var(AxisSpec::All),
            CallOp::Std(AxisSpec::All),
            CallOp::Ptp(AxisSpec::All),
        ] {
            outputs.push(f32_scalar.clone());
            body.push(call(op, vec![var("x")]));
        }
        for op in [CallOp::Argmax(AxisSpec::All), CallOp::Argmin(AxisSpec::All)] {
            outputs.push(i64_scalar.clone());
            body.push(call(op, vec![var("x")]));
        }
        for op in [CallOp::All(AxisSpec::All), CallOp::Any(AxisSpec::All)] {
            outputs.push(bool_scalar.clone());
            body.push(call(op, vec![var("flags")]));
        }
        for op in [
            CallOp::Sum(AxisSpec::One(1)),
            CallOp::Prod(AxisSpec::One(1)),
            CallOp::Mean(AxisSpec::One(1)),
            CallOp::Max(AxisSpec::One(1)),
            CallOp::Min(AxisSpec::One(1)),
            CallOp::Var(AxisSpec::One(1)),
            CallOp::Std(AxisSpec::One(1)),
            CallOp::Ptp(AxisSpec::One(1)),
        ] {
            outputs.push(f32_rows.clone());
            body.push(call(op, vec![var("x")]));
        }
        for op in [
            CallOp::Argmax(AxisSpec::One(1)),
            CallOp::Argmin(AxisSpec::One(1)),
        ] {
            outputs.push(i64_rows.clone());
            body.push(call(op, vec![var("x")]));
        }
        for op in [CallOp::All(AxisSpec::One(1)), CallOp::Any(AxisSpec::One(1))] {
            outputs.push(bool_rows.clone());
            body.push(call(op, vec![var("flags")]));
        }
        outputs.push(tensor(ElementType::F32, &[2, 3]));
        body.push(call(CallOp::Softmax(AxisSpec::One(1)), vec![var("x")]));

        let graph = typed_graph(
            "reductions",
            vec![
                input("x", ElementType::F32, &[2, 3]),
                input("flags", ElementType::Bool, &[2, 3]),
            ],
            outputs,
            body,
        );

        let mlir = lower_verified(&graph);
        assert!(mlir.contains("linalg.generic"), "{mlir}");
    }

    #[test]
    fn lowers_linalg_conv_and_shape_ops() {
        let conv_options = Conv2dOptions {
            padding: Padding2d {
                top: 1,
                bottom: 1,
                left: 1,
                right: 1,
            },
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        };
        let graph = typed_graph(
            "linalg_conv_shape",
            vec![
                input("matrix", ElementType::F32, &[2, 3]),
                input("rhs", ElementType::F32, &[3, 2]),
                input("vector", ElementType::F32, &[3]),
                input("square", ElementType::F32, &[3, 3]),
                input("x", ElementType::F32, &[2, 3]),
                input("y", ElementType::F32, &[2, 3]),
                input("idx", ElementType::I64, &[2, 2]),
                input("cube", ElementType::F32, &[2, 3, 4]),
                input("image", ElementType::F32, &[1, 4, 4, 2]),
                input("kernel", ElementType::F32, &[3, 3, 2, 1]),
            ],
            vec![
                tensor(ElementType::F32, &[2, 2]),
                tensor(ElementType::F32, &[]),
                tensor(ElementType::F32, &[]),
                tensor(ElementType::F32, &[3, 3]),
                tensor(ElementType::F32, &[2]),
                tensor(ElementType::F32, &[3]),
                tensor(ElementType::F32, &[1, 4, 4, 1]),
                tensor(ElementType::F32, &[6]),
                tensor(ElementType::F32, &[3, 2]),
                tensor(ElementType::F32, &[2, 2, 2]),
                tensor(ElementType::F32, &[2, 2]),
                tensor(ElementType::F32, &[4, 3]),
                tensor(ElementType::F32, &[2, 2, 3]),
                tensor(ElementType::F32, &[4, 6]),
                tensor(ElementType::F32, &[2, 6]),
                tensor(ElementType::F32, &[4, 2, 3]),
            ],
            vec![
                call(CallOp::Matmul, vec![var("matrix"), var("rhs")]),
                call(CallOp::Dot, vec![var("vector"), var("vector")]),
                call(CallOp::Trace(None), vec![var("square")]),
                call(CallOp::Outer, vec![var("vector"), var("vector")]),
                call(CallOp::Vecdot(Some(1)), vec![var("matrix"), var("matrix")]),
                call(CallOp::Diagonal(None), vec![var("square")]),
                call(
                    CallOp::Conv2d(conv_options),
                    vec![var("image"), var("kernel")],
                ),
                call(
                    CallOp::Reshape(tensor(ElementType::F32, &[6])),
                    vec![var("x")],
                ),
                call(CallOp::Transpose(Vec::new()), vec![var("x")]),
                call(
                    CallOp::Gather {
                        target: tensor(ElementType::F32, &[2, 2, 2]),
                        axis: 1,
                    },
                    vec![var("x"), var("idx")],
                ),
                call(
                    CallOp::Slice {
                        target: tensor(ElementType::F32, &[2, 2]),
                        starts: vec![0, 1],
                    },
                    vec![var("x")],
                ),
                call(CallOp::Concat(0), vec![var("x"), var("y")]),
                call(CallOp::Stack(0), vec![var("x"), var("y")]),
                call(CallOp::Tile(vec![2, 2]), vec![var("x")]),
                call(CallOp::Repeat { axis: 1, count: 2 }, vec![var("x")]),
                call(CallOp::PermuteDims(vec![2, 0, 1]), vec![var("cube")]),
            ],
        );

        let mlir = lower_verified(&graph);
        assert!(mlir.contains("linalg.matmul"), "{mlir}");
        assert!(mlir.contains("tensor.extract_slice"), "{mlir}");
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
        canonicalize_and_verify(&mlir).unwrap();
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
        canonicalize_and_verify(&mlir).unwrap();
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
        canonicalize_and_verify(&mlir).unwrap();
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
        canonicalize_and_verify(&mlir).unwrap();
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

    #[test]
    fn compile_source_rejects_invalid_mlir_before_spawning_compiler() {
        let error = compile_mlir_source("llvm-cpu", "module @knok {").unwrap_err();

        assert!(error.to_string().contains("generated MLIR"));
    }
}
