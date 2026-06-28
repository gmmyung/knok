#![warn(missing_docs)]
//! Procedural macros for the `knok` graph frontend.
//!
//! Application code normally imports these through the top-level `knok` crate.

use proc_macro::TokenStream;

/// Compiles a restricted Rust tensor function into an embedded IREE graph.
///
/// The decorated function body is parsed as static graph syntax, lowered to
/// MLIR, compiled to VMFB bytecode, and replaced with typed runtime wrappers.
#[proc_macro_attribute]
pub fn graph(attr: TokenStream, item: TokenStream) -> TokenStream {
    knok_compile::expand_graph(attr.into(), item.into()).into()
}

/// Imports and compiles a local MLIR file into an embedded graph artifact.
///
/// When a signature is declared, the macro also generates typed invocation
/// helpers for one-shot and reusable-engine execution.
#[proc_macro]
pub fn mlir_model(input: TokenStream) -> TokenStream {
    knok_compile::expand_mlir_model(input.into()).into()
}
