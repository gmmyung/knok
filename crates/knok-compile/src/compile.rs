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

pub(crate) fn compile_graph_variants_with_registry(
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
    use knok_core::parse_graph;
    use quote::quote;
    use syn::parse_quote;

    use crate::lowering::lower_to_mlir;

    use super::verify_with_melior;

    #[test]
    fn expm1_lowering_uses_direct_math_op() {
        let graph = parse_graph(
            quote!(backend = Backend::LlvmCpu),
            parse_quote! {
                fn expm1_graph(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                    expm1(x)
                }
            },
        )
        .unwrap();

        let mlir = lower_to_mlir(&graph).unwrap();
        assert!(mlir.contains("math.expm1"), "{mlir}");
        assert!(!mlir.contains(" = math.exp "), "{mlir}");
        assert!(!mlir.contains("arith.subf"), "{mlir}");
        verify_with_melior(&mlir).unwrap();
    }
}
