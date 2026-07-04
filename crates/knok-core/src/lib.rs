//! Core graph IR, tensor type metadata, and type checking for `knok`.
//!
//! This crate is primarily consumed by `knok-build` and `knok-compile`. End
//! users normally interact with the typed tensors in `knok` and the tracing
//! helpers in `knok-build`.

mod ast;
mod autograd;
mod ops;
mod type_parse;
mod typecheck;

#[cfg(test)]
mod tests;

pub use ast::{
    AxisSpec, BinaryOp, CallOp, Conv2dOptions, ElementType, Expr, Graph, GraphSignature, Input,
    Let, Padding2d, Pool2dOptions, StaticScalar, TensorType, TypedExpr, TypedGraph, TypedLet,
    TypedValue, UnaryOp,
};
pub use autograd::{value_and_grad_graph, value_and_grad_graph_with_registry};
pub use ops::{
    static_arange_literals, static_eye_literals, static_linspace_literals,
    static_random_int_literals, static_random_uniform_literals,
};
pub use type_parse::parse_tensor_type;
pub use typecheck::type_check;
