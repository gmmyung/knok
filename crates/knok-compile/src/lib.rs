//! MLIR lowering and IREE compilation for `knok` graphs.
//!
//! `knok-build` uses this crate from build scripts to turn traced graph IR into
//! VMFB bytes. MLIR modules are built and verified in-process with `melior`;
//! final VMFB compilation uses Eerie's runtime-loaded IREE compiler in the same
//! process. Compiler initialization is shared across graph compilations.

mod backend;
mod compile;
mod lowering;
mod mlir;

pub use compile::{
    compile_graph, compile_graph_with_options, compile_graph_with_registry, compile_mlir_source,
    compile_mlir_source_with_options, CompileOptions, CompiledGraph,
};
pub use lowering::{lower_to_mlir, lower_to_mlir_with_registry};
