use std::{collections::BTreeMap, sync::Arc};

use proc_macro2::Span;

use crate::ops::{
    broadcast_shape_slices, expect_numeric_element, infer_call_result, infer_call_results,
    validate_static_creation_call,
};
use crate::{
    CallOp, Expr, Graph, GraphSignature, Input, TensorType, TypedExpr, TypedGraph, TypedLet,
    TypedValue,
};

pub fn type_check(
    graph: Graph,
    graph_signatures: &[(String, GraphSignature)],
) -> syn::Result<TypedGraph> {
    let mut checker = TypeChecker::new(graph_signatures, &graph.name);
    let mut env = TypeEnv::from_inputs(&graph.inputs);
    let mut lets = Vec::new();
    for binding in graph.lets {
        let value = checker.type_let_value(&binding.value, &env)?;
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
        env.push_bindings(&binding.names, &value.tys);
        lets.push(TypedLet {
            names: binding.names,
            value,
        });
    }
    let body = graph
        .body
        .iter()
        .map(|expr| checker.type_tensor_expr(expr, &env))
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

struct TypeChecker<'a> {
    graph_signatures: &'a [(String, GraphSignature)],
    current_graph: &'a str,
    node_types: NodeTypeCache,
}

impl<'a> TypeChecker<'a> {
    fn new(graph_signatures: &'a [(String, GraphSignature)], current_graph: &'a str) -> Self {
        Self {
            graph_signatures,
            current_graph,
            node_types: NodeTypeCache::default(),
        }
    }

    fn type_let_value(&mut self, expr: &Expr, env: &TypeEnv) -> syn::Result<TypedValue> {
        let tys = self.type_value_outputs(expr, env)?;
        Ok(TypedValue {
            kind: expr.clone(),
            tys,
        })
    }

    fn type_value_outputs(&mut self, expr: &Expr, env: &TypeEnv) -> syn::Result<Vec<TensorType>> {
        Ok(match expr {
            Expr::Call {
                op: CallOp::Graph(name),
                args,
            } => self.type_graph_call_outputs(name, args, env)?,
            Expr::Call { op, args } => self.type_call_outputs(op, args, env)?,
            Expr::Node { node_id, value } => self.type_node_outputs(*node_id, value, env)?,
            _ => vec![self.type_tensor_expr(expr, env)?.ty],
        })
    }

    fn type_call_outputs(
        &mut self,
        op: &CallOp,
        args: &[Expr],
        env: &TypeEnv,
    ) -> syn::Result<Vec<TensorType>> {
        validate_static_creation_call(op, args)?;
        let arg_tys = self.type_arg_tys(args, env)?;
        infer_call_results(op, &arg_tys)
    }

    fn type_tensor_expr(&mut self, expr: &Expr, env: &TypeEnv) -> syn::Result<TypedExpr> {
        let ty = match expr {
            Expr::Var(name) => env.lookup(name).ok_or_else(|| {
                syn::Error::new(Span::call_site(), format!("unknown value `{name}`"))
            })?,
            Expr::Const { elem, .. } => TensorType {
                elem: *elem,
                shape: vec![],
            },
            Expr::Unary { value, .. } => {
                let ty = self.type_tensor_expr(value, env)?.ty;
                expect_numeric_element(ty.elem, "arithmetic operators")?;
                ty
            }
            Expr::Binary { lhs, rhs, .. } => {
                let lhs = self.type_tensor_expr(lhs, env)?.ty;
                let rhs = self.type_tensor_expr(rhs, env)?.ty;
                binary_result_type(&lhs, &rhs)?
            }
            Expr::Node { node_id, value } => {
                let outputs = self.type_node_outputs(*node_id, value, env)?;
                single_output_type(&outputs, "node")?
            }
            Expr::TupleGet { value, index, .. } => {
                let outputs = self.type_value_outputs(value, env)?;
                outputs.get(*index).cloned().ok_or_else(|| {
                    syn::Error::new(
                        Span::call_site(),
                        format!(
                            "tuple projection index {index} out of bounds for {} values",
                            outputs.len()
                        ),
                    )
                })?
            }
            Expr::Call { op, args } => self.type_call_tensor_result(op, args, env)?,
        };
        Ok(TypedExpr {
            kind: expr.clone(),
            ty,
        })
    }

    fn type_call_tensor_result(
        &mut self,
        op: &CallOp,
        args: &[Expr],
        env: &TypeEnv,
    ) -> syn::Result<TensorType> {
        if let CallOp::Graph(name) = op {
            let outputs = self.type_graph_call_outputs(name, args, env)?;
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

        validate_static_creation_call(op, args)?;

        let arg_tys = self.type_arg_tys(args, env)?;
        infer_call_result(op, &arg_tys)
    }

    fn type_graph_call_outputs(
        &mut self,
        name: &str,
        args: &[Expr],
        env: &TypeEnv,
    ) -> syn::Result<Vec<TensorType>> {
        if name == self.current_graph {
            return Err(syn::Error::new(
                Span::call_site(),
                format!("recursive graph call `{name}` is not supported"),
            ));
        }
        let signature = self
            .graph_signatures
            .iter()
            .rev()
            .find_map(|(candidate, signature)| (candidate == name).then_some(signature))
            .ok_or_else(|| {
                syn::Error::new(
                    Span::call_site(),
                    format!(
                        "unknown graph call `{name}`; graph calls must refer to registered graph signatures"
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
            let actual = self.type_tensor_expr(arg, env)?.ty;
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

    fn type_arg_tys(&mut self, args: &[Expr], env: &TypeEnv) -> syn::Result<Vec<TensorType>> {
        args.iter()
            .map(|arg| self.type_tensor_expr(arg, env).map(|typed| typed.ty))
            .collect()
    }

    fn type_node_outputs(
        &mut self,
        node_id: u64,
        value: &Arc<Expr>,
        env: &TypeEnv,
    ) -> syn::Result<Vec<TensorType>> {
        if let Some(tys) = self.node_types.cached_outputs(node_id, value)? {
            return Ok(tys);
        }

        let tys = self.type_value_outputs(value, env)?;
        self.node_types
            .insert(node_id, value.clone(), tys.clone())?;
        Ok(tys)
    }
}

#[derive(Default)]
struct TypeEnv {
    values: Vec<(String, TensorType)>,
}

impl TypeEnv {
    fn from_inputs(inputs: &[Input]) -> Self {
        Self {
            values: inputs
                .iter()
                .map(|input| (input.name.clone(), input.ty.clone()))
                .collect(),
        }
    }

    fn lookup(&self, name: &str) -> Option<TensorType> {
        self.values
            .iter()
            .rev()
            .find_map(|(candidate, ty)| (candidate == name).then(|| ty.clone()))
    }

    fn push_bindings(&mut self, names: &[String], tys: &[TensorType]) {
        self.values.extend(
            names
                .iter()
                .zip(tys)
                .map(|(name, ty)| (name.clone(), ty.clone())),
        );
    }
}

#[derive(Default)]
struct NodeTypeCache {
    outputs: BTreeMap<u64, CachedNodeTypes>,
}

struct CachedNodeTypes {
    value: Arc<Expr>,
    tys: Vec<TensorType>,
}

impl NodeTypeCache {
    fn cached_outputs(
        &self,
        node_id: u64,
        value: &Arc<Expr>,
    ) -> syn::Result<Option<Vec<TensorType>>> {
        let Some(cached) = self.outputs.get(&node_id) else {
            return Ok(None);
        };
        ensure_same_node_payload(node_id, &cached.value, value)?;
        Ok(Some(cached.tys.clone()))
    }

    fn insert(&mut self, node_id: u64, value: Arc<Expr>, tys: Vec<TensorType>) -> syn::Result<()> {
        if let Some(cached) = self.outputs.get(&node_id) {
            ensure_same_node_payload(node_id, &cached.value, &value)?;
            return Ok(());
        }
        self.outputs.insert(node_id, CachedNodeTypes { value, tys });
        Ok(())
    }
}

fn ensure_same_node_payload(
    node_id: u64,
    cached: &Arc<Expr>,
    current: &Arc<Expr>,
) -> syn::Result<()> {
    if cached.as_ref() != current.as_ref() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("node id {node_id} is used for multiple expression payloads"),
        ));
    }
    Ok(())
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

fn single_output_type(tys: &[TensorType], context: &str) -> syn::Result<TensorType> {
    if tys.len() != 1 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{context} produces {} values and cannot be used as a tensor expression",
                tys.len()
            ),
        ));
    }
    Ok(tys[0].clone())
}

fn broadcast_shape(lhs: &TensorType, rhs: &TensorType) -> Result<Vec<usize>, String> {
    broadcast_shape_slices(&lhs.shape, &rhs.shape)
}
