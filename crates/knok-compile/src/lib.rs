//! MLIR lowering and IREE compilation for `knok` graphs.
//!
//! `knok-build` uses this crate from build scripts to turn traced graph IR into
//! VMFB bytes. MLIR validation happens in-process with `melior`; IREE
//! compilation is delegated to the `iree-compile` command line tool.

mod backend;
mod common;
mod compile;
mod lowering;

pub use compile::{compile_graph, compile_graph_with_registry, compile_mlir_source, CompiledGraph};
pub use lowering::{lower_to_mlir, lower_to_mlir_with_registry};
