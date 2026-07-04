use std::{collections::BTreeMap, sync::Arc};

use crate::{
    parse_tensor_type, static_random_uniform_literals, type_check, value_and_grad_graph,
    value_and_grad_graph_with_registry, AxisSpec, BinaryOp, CallOp, Conv2dOptions, ElementType,
    Expr, Graph, GraphSignature, Input, Let, Pool2dOptions, TensorType, TypedGraph,
};

fn tensor(elem: ElementType, shape: &[usize]) -> TensorType {
    TensorType {
        elem,
        shape: shape.to_vec(),
    }
}

fn input(name: &str, elem: ElementType, shape: &[usize]) -> Input {
    Input {
        name: name.into(),
        ty: tensor(elem, shape),
    }
}

fn var(name: &str) -> Expr {
    Expr::Var(name.into())
}

fn graph(name: &str, inputs: Vec<Input>, outputs: Vec<TensorType>, body: Expr) -> Graph {
    Graph {
        name: name.into(),
        backend: "llvm-cpu".into(),
        inputs,
        outputs,
        lets: Vec::new(),
        body: vec![body],
    }
}

fn graph_signature(graph: &TypedGraph) -> GraphSignature {
    GraphSignature {
        inputs: graph.inputs.iter().map(|input| input.ty.clone()).collect(),
        outputs: graph.outputs.clone(),
    }
}

fn expr_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Var(_) | Expr::Const { .. } => 1,
        Expr::Unary { value, .. } => 1 + expr_node_count(value),
        Expr::Binary { lhs, rhs, .. } => 1 + expr_node_count(lhs) + expr_node_count(rhs),
        Expr::Node { value, .. } => 1 + expr_node_count(value),
        Expr::TupleGet { value, .. } => 1 + expr_node_count(value),
        Expr::Call { args, .. } => 1 + args.iter().map(expr_node_count).sum::<usize>(),
    }
}

fn graph_expr_node_count(graph: &TypedGraph) -> usize {
    graph
        .lets
        .iter()
        .map(|binding| expr_node_count(&binding.value.kind))
        .sum::<usize>()
        + graph
            .body
            .iter()
            .map(|expr| expr_node_count(&expr.kind))
            .sum::<usize>()
}

mod autograd;
mod typecheck;
