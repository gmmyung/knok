mod codegen;
mod trace;

use std::{collections::BTreeMap, env, fs, path::PathBuf};

use knok_compile::compile_graph_with_registry;
use knok_core::TypedGraph;

pub use knok_build_macros::{compile_graphs, compile_graphs_with_options, graph};
pub use trace::*;

pub type Result<T> = anyhow::Result<T>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    LlvmCpu,
    MetalSpirv,
}

impl Backend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
        }
    }

    pub const fn default_driver(self) -> &'static str {
        match self {
            Self::LlvmCpu => "local-task",
            Self::MetalSpirv => "metal",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuildOptions {
    output_file: String,
    stub_artifacts: bool,
}

impl BuildOptions {
    pub fn stub_artifacts_for_check() -> Self {
        Self {
            stub_artifacts: true,
            ..Self::default()
        }
    }

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

#[derive(Default)]
pub struct GraphRegistry {
    graphs: Vec<RegisteredGraph>,
}

pub struct RegisteredGraph {
    graph: TypedGraph,
    backend: Backend,
}

impl GraphRegistry {
    pub fn new() -> Self {
        Self::default()
    }

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

pub fn emit_registered_graphs(registry: GraphRegistry) -> Result<()> {
    emit_registered_graphs_with_options(registry, BuildOptions::default())
}

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
        let vmfb_name = format!("{}.vmfb", registered.graph.name);
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

fn compile_registered_graph(
    graph: &TypedGraph,
    graphs: &BTreeMap<String, TypedGraph>,
) -> Result<knok_compile::CompiledGraph> {
    compile_graph_with_registry(graph, graphs).map_err(Into::into)
}

pub mod prelude {
    pub use crate::{
        abs, all, all_axis, amax, amax_axis, amin, amin_axis, any, any_axis, arange, arange_step,
        arange_to, argmax, argmax_axis, argmin, argmin_axis, broadcast, ceil, clip, compile_graphs,
        compile_graphs_with_options, concat, conv2d, conv2d_options, cos, diagonal, diagonal_axes,
        dot, equal, exp, exp2, expm1, eye, flip, flip_axes, floor, full_like, gather, graph,
        greater, greater_equal, identity, inner, isnan, less, less_equal, linspace, log, log10,
        log1p, log2, logical_and, logical_not, logical_or, logical_xor, matmul, max, max_axis,
        maximum, mean, mean_axis, min, min_axis, minimum, moveaxis, not_equal, ones_like, outer,
        pad, permute, permute_dims, pow, prod, prod_axis, ptp, ptp_axis, r#where, reciprocal, relu,
        repeat, reshape, rint, roll, round, sigmoid, sin, slice, softmax, softmax_axis, split,
        sqrt, square, squeeze, stack, std, std_axis, sum, sum_axis, swapaxes, take,
        take_along_axis, tan, tanh, tile, trace, trace_axes, transpose, transpose_axes, unsqueeze,
        var, var_axis, vecdot, vecdot_axis, zeros_like, Backend, BuildOptions, Conv2dOptions,
        Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, T0, T1, T2, T3, T4, T5, T6,
    };
    pub use half::{bf16, f16};
}
