#![warn(missing_docs)]
//! Macro expansion, MLIR lowering, and IREE compilation support for `knok`.
//!
//! This crate backs `knok-macros`. It is public so tooling can reuse the same
//! parser/lowerer/compiler path, but most application code should use
//! `#[knok::graph]` and `knok::mlir_model!` from the top-level `knok` crate.

mod backend;
mod common;
mod compile;
mod graph_macro;
mod lowering;
mod mlir_model;
mod mlir_signature;
mod registry;

pub use compile::{compile_graph, compile_graph_with_registry, compile_mlir_source, CompiledGraph};
pub use graph_macro::expand_graph;
pub use lowering::{lower_to_mlir, lower_to_mlir_with_registry};
pub use mlir_model::expand_mlir_model;
