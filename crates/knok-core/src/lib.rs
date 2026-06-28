#![warn(missing_docs)]
//! Core parser, graph IR, and type checker for `knok`.
//!
//! This crate is used by `knok-compile` and the procedural macros. Most users
//! should depend on the top-level `knok` crate instead. The public API here is
//! useful for tests, tooling, and alternate macro frontends that want to parse
//! restricted Rust graph functions into the typed `knok` IR.

mod ast;
mod ops;
mod parse;
mod typecheck;

#[cfg(test)]
mod tests;

pub use ast::{
    static_arange_literals, static_eye_literals, static_linspace_literals, AxisSpec, BinaryOp,
    CallOp, Conv2dOptions, ElementType, Expr, Graph, GraphSignature, Input, Let, Padding2d,
    StaticScalar, TensorType, TypedExpr, TypedGraph, TypedLet, TypedValue, UnaryOp,
};
pub use parse::{parse_graph, parse_graph_with_signatures, parse_tensor_type};
pub use typecheck::type_check;
