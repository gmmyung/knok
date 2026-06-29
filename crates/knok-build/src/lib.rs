//! Build-script frontend for `knok` static graph tracing.
//!
//! This crate is used from `build.rs`, not from the final target crate. A build
//! script defines graph functions with [`graph`], executes them with traced
//! tensor values through [`compile_graphs`], and writes generated wrappers plus
//! VMFB artifacts into `OUT_DIR`.
//!
//! ```ignore
//! use knok_build::prelude::*;
//!
//! #[knok_build::graph(backend = Backend::LlvmCpu)]
//! fn forward(x: T2<f32, 2, 2>) -> T2<f32, 2, 2> {
//!     relu(matmul(x.clone(), x) + 1.0)
//! }
//!
//! fn main() {
//!     knok_build::compile_graphs!(forward);
//! }
//! ```
//!
//! Target crates import the generated wrappers with
//! `knok::generated_graphs!(pub mod graphs);`.

mod codegen;
mod trace;

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

use knok_compile::{compile_graph_with_registry, compile_mlir_source};
use knok_core::{Input, TensorType, TypedGraph};

pub use knok_build_macros::{
    compile_graphs, compile_graphs_with_options, compile_mlir_models,
    compile_mlir_models_with_options, graph,
};
pub use trace::*;

/// Result type used by build-script tracing and code generation.
pub type Result<T> = anyhow::Result<T>;

/// IREE target backend selected by a traced graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// CPU backend compiled through IREE's LLVM CPU target.
    LlvmCpu,
    /// Apple Metal backend compiled through IREE's Metal/SPIR-V path.
    MetalSpirv,
}

impl Backend {
    /// Returns the IREE target backend flag value.
    pub const fn name(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
        }
    }

    /// Returns the default IREE runtime driver used by generated wrappers.
    pub const fn default_driver(self) -> &'static str {
        match self {
            Self::LlvmCpu => "local-task",
            Self::MetalSpirv => "metal",
        }
    }
}

/// Options controlling generated wrapper output from `compile_graphs!`.
#[derive(Clone, Debug)]
pub struct BuildOptions {
    output_file: String,
    stub_artifacts: bool,
}

impl BuildOptions {
    /// Creates placeholder VMFB files for check-only builds.
    ///
    /// Stub artifacts let no-std or cross-target fixture crates typecheck
    /// generated wrappers without requiring a runnable IREE compiler artifact.
    /// The generated artifacts are intentionally not executable.
    pub fn stub_artifacts_for_check() -> Self {
        Self {
            stub_artifacts: true,
            ..Self::default()
        }
    }

    /// Writes generated Rust wrappers to `name` inside `OUT_DIR`.
    ///
    /// Target crates must pass the same filename to
    /// `knok::generated_graphs!(pub mod graphs, "...")`.
    pub fn output_file(mut self, name: impl Into<String>) -> Self {
        self.output_file = name.into();
        self
    }
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            output_file: "knok_graphs.rs".into(),
            stub_artifacts: false,
        }
    }
}

/// Registry populated by generated `#[graph]` registration glue.
#[derive(Default)]
pub struct GraphRegistry {
    graphs: Vec<RegisteredGraph>,
}

/// One traced graph registered for build-time compilation.
pub struct RegisteredGraph {
    graph: TypedGraph,
    backend: Backend,
}

/// One external MLIR module compiled and wrapped by `compile_mlir_models!`.
#[derive(Clone, Debug)]
pub struct MlirModel {
    /// Generated Rust module name.
    pub name: String,
    /// Path to the `.mlir` file, relative to the build-script crate unless absolute.
    pub path: PathBuf,
    /// IREE VM function name, such as `imported.forward`.
    pub function_name: String,
    /// IREE compile backend.
    pub backend: Backend,
    /// Input tensor names and types used by the generated wrapper.
    pub inputs: Vec<Input>,
    /// Output tensor types used by the generated wrapper.
    pub outputs: Vec<TensorType>,
}

impl MlirModel {
    /// Creates an external MLIR model descriptor.
    pub fn new(
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        function_name: impl Into<String>,
        backend: Backend,
        inputs: Vec<Input>,
        outputs: Vec<TensorType>,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            function_name: function_name.into(),
            backend,
            inputs,
            outputs,
        }
    }
}

impl GraphRegistry {
    /// Creates an empty graph registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Traces one graph body and stores the resulting typed graph.
    ///
    /// Most users call this indirectly through [`compile_graphs`] or
    /// [`compile_graphs_with_options`].
    pub fn trace<F, O>(&mut self, name: &str, backend: Backend, build: F) -> Result<()>
    where
        F: FnOnce(&mut TraceContext) -> O,
        O: TraceOutput,
    {
        let mut context = TraceContext::default();
        let output = build(&mut context);
        let graph = context.finish(name, backend.name(), output)?;
        self.graphs.push(RegisteredGraph { graph, backend });
        Ok(())
    }
}

/// Emits wrappers and VMFB artifacts for all graphs in `registry`.
pub fn emit_registered_graphs(registry: GraphRegistry) -> Result<()> {
    emit_registered_graphs_with_options(registry, BuildOptions::default())
}

/// Emits wrappers and VMFB artifacts with explicit build options.
pub fn emit_registered_graphs_with_options(
    registry: GraphRegistry,
    options: BuildOptions,
) -> Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    fs::create_dir_all(&out_dir)?;
    if env::var_os("KNOK_CACHE_DIR").is_none() {
        env::set_var("KNOK_CACHE_DIR", out_dir.join("knok-cache"));
    }
    println!("cargo:rerun-if-changed=build.rs");

    let mut generated = String::new();
    let graphs = registry.graphs;
    let graph_registry = graphs
        .iter()
        .map(|registered| (registered.graph.name.clone(), registered.graph.clone()))
        .collect::<BTreeMap<_, _>>();

    for registered in &graphs {
        let vmfb_name = graph_vmfb_name(&registered.graph.name);
        let vmfb_path = out_dir.join(&vmfb_name);
        let compile_flags;
        if options.stub_artifacts {
            fs::write(&vmfb_path, [0u8; 16])?;
            compile_flags = Vec::new();
        } else {
            let compiled = compile_registered_graph(&registered.graph, &graph_registry)?;
            fs::write(&vmfb_path, compiled.vmfb)?;
            compile_flags = Vec::new();
        }
        generated.push_str(&codegen::graph_module(
            &registered.graph,
            registered.backend,
            &vmfb_name,
            &compile_flags,
        )?);
        generated.push('\n');
    }

    fs::write(out_dir.join(options.output_file), generated)?;
    Ok(())
}

/// Emits wrappers and VMFB artifacts for external MLIR models.
pub fn emit_mlir_models(models: Vec<MlirModel>) -> Result<()> {
    emit_mlir_models_with_options(models, BuildOptions::default())
}

/// Emits external MLIR model wrappers and VMFB artifacts with explicit options.
pub fn emit_mlir_models_with_options(models: Vec<MlirModel>, options: BuildOptions) -> Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    fs::create_dir_all(&out_dir)?;
    if env::var_os("KNOK_CACHE_DIR").is_none() {
        env::set_var("KNOK_CACHE_DIR", out_dir.join("knok-cache"));
    }
    println!("cargo:rerun-if-changed=build.rs");

    let mut generated = String::new();
    for model in &models {
        let source_path = resolve_model_path(&model.path)?;
        println!("cargo:rerun-if-changed={}", source_path.display());
        let vmfb_name = mlir_model_vmfb_name(&model.name);
        let vmfb_path = out_dir.join(&vmfb_name);
        let compile_flags;
        if options.stub_artifacts {
            fs::write(&vmfb_path, [0u8; 16])?;
            compile_flags = Vec::new();
        } else {
            let source = fs::read_to_string(&source_path)?;
            let vmfb = compile_mlir_source(model.backend.name(), &source)?;
            fs::write(&vmfb_path, vmfb)?;
            compile_flags = Vec::new();
        }
        generated.push_str(&codegen::mlir_model_module(
            model,
            &vmfb_name,
            &compile_flags,
        )?);
        generated.push('\n');
    }

    fs::write(out_dir.join(options.output_file), generated)?;
    Ok(())
}

fn resolve_model_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join(path))
}

fn graph_vmfb_name(name: &str) -> String {
    format!("{name}.vmfb")
}

fn mlir_model_vmfb_name(name: &str) -> String {
    format!("mlir-model-{name}.vmfb")
}

fn compile_registered_graph(
    graph: &TypedGraph,
    graphs: &BTreeMap<String, TypedGraph>,
) -> Result<knok_compile::CompiledGraph> {
    compile_graph_with_registry(graph, graphs).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_mlir_artifacts_do_not_collide_with_traced_graph_artifacts() {
        assert_eq!(graph_vmfb_name("forward"), "forward.vmfb");
        assert_eq!(mlir_model_vmfb_name("forward"), "mlir-model-forward.vmfb");
        assert_ne!(graph_vmfb_name("forward"), mlir_model_vmfb_name("forward"));
    }
}

/// Common build-script imports for graph definitions.
pub mod prelude {
    pub use crate::{
        abs, all, all_axis, amax, amax_axis, amin, amin_axis, any, any_axis, arange, arange_step,
        arange_to, argmax, argmax_axis, argmin, argmin_axis, broadcast, ceil, clip, compile_graphs,
        compile_graphs_with_options, compile_mlir_models, compile_mlir_models_with_options, concat,
        conv2d, conv2d_options, cos, diagonal, diagonal_axes, dot, equal, exp, exp2, expm1, eye,
        flip, flip_axes, floor, full_like, gather, graph, greater, greater_equal, identity, inner,
        isnan, less, less_equal, linspace, log, log10, log1p, log2, logical_and, logical_not,
        logical_or, logical_xor, matmul, max, max_axis, maximum, mean, mean_axis, min, min_axis,
        minimum, moveaxis, not_equal, ones_like, outer, pad, permute, permute_dims, pow, prod,
        prod_axis, ptp, ptp_axis, r#where, reciprocal, relu, repeat, reshape, rint, roll, round,
        sigmoid, sin, slice, softmax, softmax_axis, split, sqrt, square, squeeze, stack, std,
        std_axis, sum, sum_axis, swapaxes, take, take_along_axis, tan, tanh, tile, trace,
        trace_axes, transpose, transpose_axes, unsqueeze, var, var_axis, vecdot, vecdot_axis,
        zeros_like, Backend, BuildOptions, Conv2dOptions, MlirModel, Tensor0, Tensor1, Tensor2,
        Tensor3, Tensor4, Tensor5, Tensor6, T0, T1, T2, T3, T4, T5, T6,
    };
    pub use half::{bf16, f16};
}

#[doc(hidden)]
pub mod __private {
    pub use knok_core::{ElementType, Input, TensorType};
}
