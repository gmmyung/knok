use proc_macro2::Span;

use crate::ops::{broadcast_shape_slices, expect_numeric_element, infer_call_result};
use crate::{
    CallOp, Expr, Graph, GraphSignature, TensorType, TypedExpr, TypedGraph, TypedLet, TypedValue,
};

pub fn type_check(
    graph: Graph,
    graph_signatures: &[(String, GraphSignature)],
) -> syn::Result<TypedGraph> {
    let mut env = graph
        .inputs
        .iter()
        .map(|input| (input.name.clone(), input.ty.clone()))
        .collect::<Vec<_>>();
    let mut lets = Vec::new();
    for binding in graph.lets {
        let value = type_let_value(&binding.value, &env, graph_signatures, &graph.name)?;
        if binding.names.len() != value.tys.len() {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "let pattern expects {} values, initializer produces {}",
                    binding.names.len(),
                    value.tys.len()
                ),
            ));
        }
        for (name, ty) in binding.names.iter().zip(&value.tys) {
            env.push((name.clone(), ty.clone()));
        }
        lets.push(TypedLet {
            names: binding.names,
            value,
        });
    }
    let body = graph
        .body
        .iter()
        .map(|expr| type_expr(expr, &env, graph_signatures, &graph.name))
        .collect::<syn::Result<Vec<_>>>()?;
    let body_tys = body.iter().map(|expr| expr.ty.clone()).collect::<Vec<_>>();
    if body_tys != graph.outputs {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "return type mismatch: inferred {:?}, declared {:?}",
                body_tys, graph.outputs
            ),
        ));
    }
    Ok(TypedGraph {
        name: graph.name,
        backend: graph.backend,
        inputs: graph.inputs,
        outputs: graph.outputs,
        lets,
        body,
    })
}

fn type_let_value(
    expr: &Expr,
    env: &[(String, TensorType)],
    graph_signatures: &[(String, GraphSignature)],
    current_graph: &str,
) -> syn::Result<TypedValue> {
    let tys = match expr {
        Expr::Call {
            op: CallOp::Graph(name),
            args,
        } => graph_call_output_types(name, args, env, graph_signatures, current_graph)?,
        _ => vec![type_expr(expr, env, graph_signatures, current_graph)?.ty],
    };
    Ok(TypedValue {
        kind: expr.clone(),
        tys,
    })
}

fn type_expr(
    expr: &Expr,
    env: &[(String, TensorType)],
    graph_signatures: &[(String, GraphSignature)],
    current_graph: &str,
) -> syn::Result<TypedExpr> {
    let ty = match expr {
        Expr::Var(name) => env
            .iter()
            .rev()
            .find_map(|(candidate, ty)| (candidate == name).then(|| ty.clone()))
            .ok_or_else(|| syn::Error::new(Span::call_site(), format!("unknown value `{name}`")))?,
        Expr::Const { elem, .. } => TensorType {
            elem: *elem,
            shape: vec![],
        },
        Expr::Unary { value, .. } => {
            let ty = type_expr(value, env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(ty.elem, "arithmetic operators")?;
            ty
        }
        Expr::Binary { lhs, rhs, .. } => {
            let lhs = type_expr(lhs, env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(rhs, env, graph_signatures, current_graph)?.ty;
            binary_result_type(&lhs, &rhs)?
        }
        Expr::Call { op, args } => {
            call_result_type(op, args, env, graph_signatures, current_graph)?
        }
    };
    Ok(TypedExpr {
        kind: expr.clone(),
        ty,
    })
}

fn binary_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_numeric_element(lhs.elem, "arithmetic operators")?;
    expect_numeric_element(rhs.elem, "arithmetic operators")?;
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "elementwise operands must have the same element type, got {lhs:?} and {rhs:?}"
            ),
        ));
    }
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: lhs.elem,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("elementwise operands are not broadcast-compatible: {message}"),
            )
        })
}

fn call_result_type(
    op: &CallOp,
    args: &[Expr],
    env: &[(String, TensorType)],
    graph_signatures: &[(String, GraphSignature)],
    current_graph: &str,
) -> syn::Result<TensorType> {
    if let CallOp::Graph(name) = op {
        let outputs = graph_call_output_types(name, args, env, graph_signatures, current_graph)?;
        if outputs.len() != 1 {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "graph `{name}` returns {} values and cannot be used as a tensor expression yet",
                    outputs.len()
                ),
            ));
        }
        return Ok(outputs[0].clone());
    }

    let arg_tys = args
        .iter()
        .map(|arg| type_expr(arg, env, graph_signatures, current_graph).map(|typed| typed.ty))
        .collect::<syn::Result<Vec<_>>>()?;
    infer_call_result(op, &arg_tys)
}

fn graph_call_output_types(
    name: &str,
    args: &[Expr],
    env: &[(String, TensorType)],
    graph_signatures: &[(String, GraphSignature)],
    current_graph: &str,
) -> syn::Result<Vec<TensorType>> {
    if name == current_graph {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("recursive graph call `{name}` is not supported"),
        ));
    }
    let signature = graph_signatures
        .iter()
        .rev()
        .find_map(|(candidate, signature)| (candidate == name).then_some(signature))
        .ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                format!(
                    "unknown graph call `{name}`; graph calls must refer to earlier #[knok::graph] functions"
                ),
            )
        })?;
    if args.len() != signature.inputs.len() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "graph `{name}` expects {} arguments, got {}",
                signature.inputs.len(),
                args.len()
            ),
        ));
    }
    for (index, (arg, expected)) in args.iter().zip(&signature.inputs).enumerate() {
        let actual = type_expr(arg, env, graph_signatures, current_graph)?.ty;
        if &actual != expected {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "graph `{name}` argument {index} type mismatch: expected {expected:?}, got {actual:?}"
                ),
            ));
        }
    }
    Ok(signature.outputs.clone())
}

fn broadcast_shape(lhs: &TensorType, rhs: &TensorType) -> Result<Vec<usize>, String> {
    broadcast_shape_slices(&lhs.shape, &rhs.shape)
}
