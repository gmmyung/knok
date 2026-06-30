//! Core graph IR, tensor type metadata, and type checking for `knok`.
//!
//! This crate is primarily consumed by `knok-build` and `knok-compile`. End
//! users normally interact with the typed tensors in `knok` and the tracing
//! helpers in `knok-build`.

mod ast;
mod ops;
mod type_parse;
mod typecheck;

#[cfg(test)]
mod tests;

pub use ast::{
    static_arange_literals, static_eye_literals, static_linspace_literals, AxisSpec, BinaryOp,
    CallOp, Conv2dOptions, ElementType, Expr, Graph, GraphSignature, Input, Let, Padding2d,
    Pool2dOptions, StaticScalar, TensorType, TypedExpr, TypedGraph, TypedLet, TypedValue, UnaryOp,
};
pub use type_parse::parse_tensor_type;
pub use typecheck::type_check;
