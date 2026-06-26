use proc_macro2::Span;

use crate::{
    CallOp, ElementType, Expr, Graph, GraphSignature, TensorType, TypedExpr, TypedGraph, TypedLet,
    TypedValue,
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

fn comparison_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_numeric_element(lhs.elem, "comparison ops")?;
    expect_numeric_element(rhs.elem, "comparison ops")?;
    predicate_result_type(lhs, rhs, "comparison")
}

fn equality_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    predicate_result_type(lhs, rhs, "equality")
}

fn predicate_result_type(
    lhs: &TensorType,
    rhs: &TensorType,
    op_name: &str,
) -> syn::Result<TensorType> {
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} operands must have the same element type, got {lhs:?} and {rhs:?}"),
        ));
    }
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: ElementType::Bool,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("{op_name} operands are not broadcast-compatible: {message}"),
            )
        })
}

fn logical_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_bool_element(lhs.elem, "logical ops")?;
    expect_bool_element(rhs.elem, "logical ops")?;
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: ElementType::Bool,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("logical operands are not broadcast-compatible: {message}"),
            )
        })
}

fn where_result_type(
    condition: &TensorType,
    lhs: &TensorType,
    rhs: &TensorType,
) -> syn::Result<TensorType> {
    expect_bool_element(condition.elem, "where condition")?;
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "where value operands must have the same element type, got {lhs:?} and {rhs:?}"
            ),
        ));
    }
    let value_shape = broadcast_shape(lhs, rhs).map_err(|message| {
        syn::Error::new(
            Span::call_site(),
            format!("where value operands are not broadcast-compatible: {message}"),
        )
    })?;
    let result = TensorType {
        elem: lhs.elem,
        shape: value_shape,
    };
    let shape = broadcast_shape(condition, &result).map_err(|message| {
        syn::Error::new(
            Span::call_site(),
            format!("where condition is not broadcast-compatible with values: {message}"),
        )
    })?;
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn call_result_type(
    op: &CallOp,
    args: &[Expr],
    env: &[(String, TensorType)],
    graph_signatures: &[(String, GraphSignature)],
    current_graph: &str,
) -> syn::Result<TensorType> {
    match op {
        CallOp::Abs => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(ty.elem, "abs")?;
            Ok(ty)
        }
        CallOp::All(axis) | CallOp::Any(axis) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_bool_element(input.elem, "bool reductions")?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "bool reductions expect a tensor input",
                ));
            }
            reduction_output_type(&input, *axis)
        }
        CallOp::Argmax(axis) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(input.elem, "argmax")?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "argmax expects a tensor input",
                ));
            }
            let mut output = reduction_output_type(&input, *axis)?;
            output.elem = ElementType::I64;
            Ok(output)
        }
        CallOp::Exp
        | CallOp::IsNan
        | CallOp::Log
        | CallOp::Relu
        | CallOp::Sigmoid
        | CallOp::Sqrt
        | CallOp::Tanh => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, ty.elem)?;
            if matches!(op, CallOp::IsNan) {
                Ok(TensorType {
                    elem: ElementType::Bool,
                    shape: ty.shape,
                })
            } else {
                Ok(ty)
            }
        }
        CallOp::Greater | CallOp::GreaterEqual | CallOp::Less | CallOp::LessEqual => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            comparison_result_type(&lhs, &rhs)
        }
        CallOp::Equal | CallOp::NotEqual => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            equality_result_type(&lhs, &rhs)
        }
        CallOp::LogicalAnd | CallOp::LogicalOr | CallOp::LogicalXor => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            logical_result_type(&lhs, &rhs)
        }
        CallOp::LogicalNot => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_bool_element(ty.elem, "logical_not")?;
            Ok(ty)
        }
        CallOp::Minimum | CallOp::Maximum => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(lhs.elem, "min/max ops")?;
            expect_numeric_element(rhs.elem, "min/max ops")?;
            binary_result_type(&lhs, &rhs)
        }
        CallOp::Clip => {
            expect_arity(op, args, 3)?;
            let value = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let min = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            let max = type_expr(&args[2], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(value.elem, "clip")?;
            expect_numeric_element(min.elem, "clip")?;
            expect_numeric_element(max.elem, "clip")?;
            let value = binary_result_type(&value, &min)?;
            binary_result_type(&value, &max)
        }
        CallOp::Pow => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            expect_float(op, lhs.elem)?;
            expect_float(op, rhs.elem)?;
            binary_result_type(&lhs, &rhs)
        }
        CallOp::Concat(axis) => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            concat_result_type(&lhs, &rhs, *axis)
        }
        CallOp::Softmax(axis) => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, ty.elem)?;
            if let Some(axis) = axis {
                expect_axis(&ty, *axis)?;
            }
            Ok(ty)
        }
        CallOp::Transpose => {
            expect_arity(op, args, 1)?;
            let mut ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            if ty.rank() != 2 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "transpose currently supports rank-2 tensors only",
                ));
            }
            ty.shape.swap(0, 1);
            Ok(ty)
        }
        CallOp::Reshape(target) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            if input.elem != target.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "reshape input and output element types must match",
                ));
            }
            if element_count(&input) != element_count(target) {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "reshape element counts must match, got {} and {}",
                        element_count(&input),
                        element_count(target)
                    ),
                ));
            }
            Ok(target.clone())
        }
        CallOp::Broadcast(target) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            if input.elem != target.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "broadcast input and output element types must match",
                ));
            }
            if let Err(message) = broadcast_shape(&input, target) {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!("broadcast input and output shapes are incompatible: {message}"),
                ));
            }
            Ok(target.clone())
        }
        CallOp::Slice { target, starts } => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            validate_slice(&input, target, starts)?;
            Ok(target.clone())
        }
        CallOp::Squeeze(target) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            validate_squeeze(&input, target)?;
            Ok(target.clone())
        }
        CallOp::Stack(axis) => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            stack_result_type(&lhs, &rhs, *axis)
        }
        CallOp::Sum(axis) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(input.elem, "sum")?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "sum expects a tensor input",
                ));
            }
            Ok(reduction_output_type(&input, *axis)?)
        }
        CallOp::Mean(axis) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, input.elem)?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "mean expects a tensor input",
                ));
            }
            Ok(reduction_output_type(&input, *axis)?)
        }
        CallOp::Take { axis, index } => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            take_result_type(&input, *axis, *index)
        }
        CallOp::Matmul => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            if lhs.elem != rhs.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "matmul expects operands with the same element type",
                ));
            }
            expect_numeric_element(lhs.elem, "matmul")?;
            match (lhs.rank(), rhs.rank()) {
                (2, 2) => {
                    if lhs.shape[1] != rhs.shape[0] {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            format!(
                                "matmul inner dimensions must match, got {} and {}",
                                lhs.shape[1], rhs.shape[0]
                            ),
                        ));
                    }
                    Ok(TensorType {
                        elem: lhs.elem,
                        shape: vec![lhs.shape[0], rhs.shape[1]],
                    })
                }
                (3, 3) => {
                    if lhs.shape[0] != rhs.shape[0] {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            format!(
                                "batched matmul batch dimensions must match, got {} and {}",
                                lhs.shape[0], rhs.shape[0]
                            ),
                        ));
                    }
                    if lhs.shape[2] != rhs.shape[1] {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            format!(
                                "batched matmul inner dimensions must match, got {} and {}",
                                lhs.shape[2], rhs.shape[1]
                            ),
                        ));
                    }
                    Ok(TensorType {
                        elem: lhs.elem,
                        shape: vec![lhs.shape[0], lhs.shape[1], rhs.shape[2]],
                    })
                }
                _ => Err(syn::Error::new(
                    Span::call_site(),
                    "matmul expects rank-2 operands or rank-3 batched operands",
                )),
            }
        }
        CallOp::Conv2d => {
            expect_arity(op, args, 2)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let kernel = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            if input.rank() != 4 || kernel.rank() != 4 || input.elem != kernel.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "conv2d expects NHWC input and HWCF kernel rank-4 tensors with the same element type",
                ));
            }
            expect_numeric_element(input.elem, "conv2d")?;
            if input.shape[3] != kernel.shape[2] {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "conv2d input channels must match kernel channels, got {} and {}",
                        input.shape[3], kernel.shape[2]
                    ),
                ));
            }
            if input.shape[1] < kernel.shape[0] || input.shape[2] < kernel.shape[1] {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "conv2d kernel spatial dimensions must fit inside the input",
                ));
            }
            Ok(TensorType {
                elem: input.elem,
                shape: vec![
                    input.shape[0],
                    input.shape[1] - kernel.shape[0] + 1,
                    input.shape[2] - kernel.shape[1] + 1,
                    kernel.shape[3],
                ],
            })
        }
        CallOp::Unsqueeze(target) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            validate_unsqueeze(&input, target)?;
            Ok(target.clone())
        }
        CallOp::Where => {
            expect_arity(op, args, 3)?;
            let condition = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let lhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[2], env, graph_signatures, current_graph)?.ty;
            where_result_type(&condition, &lhs, &rhs)
        }
        CallOp::Graph(name) => {
            let outputs =
                graph_call_output_types(name, args, env, graph_signatures, current_graph)?;
            if outputs.len() != 1 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "graph `{name}` returns {} values and cannot be used as a tensor expression yet",
                        outputs.len()
                    ),
                ));
            }
            Ok(outputs[0].clone())
        }
    }
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

fn element_count(ty: &TensorType) -> usize {
    ty.shape.iter().product()
}

fn broadcast_shape(lhs: &TensorType, rhs: &TensorType) -> Result<Vec<usize>, String> {
    let rank = lhs.rank().max(rhs.rank());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let lhs_dim = dim_from_trailing(&lhs.shape, rank, offset);
        let rhs_dim = dim_from_trailing(&rhs.shape, rank, offset);
        let dim = match (lhs_dim, rhs_dim) {
            (Some(lhs_dim), Some(rhs_dim)) if lhs_dim == rhs_dim => lhs_dim,
            (Some(1), Some(rhs_dim)) => rhs_dim,
            (Some(lhs_dim), Some(1)) => lhs_dim,
            (None, Some(dim)) | (Some(dim), None) => dim,
            (None, None) => unreachable!("rank is derived from at least one shape"),
            (Some(lhs_dim), Some(rhs_dim)) => {
                return Err(format!(
                    "dimension {} differs: {} vs {}",
                    offset, lhs_dim, rhs_dim
                ));
            }
        };
        shape.push(dim);
    }
    Ok(shape)
}

fn dim_from_trailing(shape: &[usize], rank: usize, offset: usize) -> Option<usize> {
    let pad = rank - shape.len();
    (offset >= pad).then(|| shape[offset - pad])
}

fn validate_slice(input: &TensorType, target: &TensorType, starts: &[usize]) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "slice input and output element types must match",
        ));
    }
    if input.rank() != target.rank() || starts.len() != input.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "slice expects one start const per dimension and equal input/output rank, got input rank {}, output rank {}, starts {:?}",
                input.rank(),
                target.rank(),
                starts
            ),
        ));
    }
    for (axis, ((start, size), input_dim)) in starts
        .iter()
        .zip(&target.shape)
        .zip(&input.shape)
        .enumerate()
    {
        if *start + *size > *input_dim {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "slice dimension {axis} is out of bounds: start {start} + size {size} exceeds input dimension {input_dim}",
                ),
            ));
        }
    }
    Ok(())
}

fn take_result_type(input: &TensorType, axis: usize, index: usize) -> syn::Result<TensorType> {
    expect_axis(input, axis)?;
    if index >= input.shape[axis] {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "take index {index} is out of bounds for axis {axis} with dimension {}",
                input.shape[axis]
            ),
        ));
    }
    let mut shape = input.shape.clone();
    shape.remove(axis);
    if shape.is_empty() {
        shape.push(1);
    }
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn validate_squeeze(input: &TensorType, target: &TensorType) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "squeeze input and output element types must match",
        ));
    }
    let squeezed = input
        .shape
        .iter()
        .copied()
        .filter(|dim| *dim != 1)
        .collect::<Vec<_>>();
    let squeezed = if squeezed.is_empty() {
        vec![1]
    } else {
        squeezed
    };
    if squeezed == target.shape {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "squeeze target shape {:?} does not match input shape {:?} after removing singleton dimensions",
                target.shape, input.shape
            ),
        ))
    }
}

fn validate_unsqueeze(input: &TensorType, target: &TensorType) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "unsqueeze input and output element types must match",
        ));
    }
    let target_without_singletons = target
        .shape
        .iter()
        .copied()
        .filter(|dim| *dim != 1)
        .collect::<Vec<_>>();
    let input_without_rank1_scalar = if input.shape == [1] {
        Vec::new()
    } else {
        input.shape.clone()
    };
    if target_without_singletons == input.shape
        || target_without_singletons == input_without_rank1_scalar
    {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "unsqueeze target shape {:?} must insert only singleton dimensions into input shape {:?}",
                target.shape, input.shape
            ),
        ))
    }
}

fn concat_result_type(lhs: &TensorType, rhs: &TensorType, axis: usize) -> syn::Result<TensorType> {
    if lhs.elem != rhs.elem || lhs.rank() != rhs.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            "concat expects tensors with the same element type and rank",
        ));
    }
    expect_axis(lhs, axis)?;
    let mut shape = lhs.shape.clone();
    for (dim, shape_dim) in shape.iter_mut().enumerate() {
        if dim == axis {
            *shape_dim = lhs.shape[dim] + rhs.shape[dim];
        } else if lhs.shape[dim] != rhs.shape[dim] {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "concat dimension {dim} must match outside axis {axis}, got {} and {}",
                    lhs.shape[dim], rhs.shape[dim]
                ),
            ));
        }
    }
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn stack_result_type(lhs: &TensorType, rhs: &TensorType, axis: usize) -> syn::Result<TensorType> {
    if lhs != rhs {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("stack expects matching tensor types, got {lhs:?} and {rhs:?}"),
        ));
    }
    if axis > lhs.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "stack axis {axis} is out of bounds for rank-{} tensor",
                lhs.rank()
            ),
        ));
    }
    let mut shape = lhs.shape.clone();
    shape.insert(axis, 2);
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn reduction_output_type(input: &TensorType, axis: Option<usize>) -> syn::Result<TensorType> {
    let shape = if let Some(axis) = axis {
        expect_axis(input, axis)?;
        let mut shape = input.shape.clone();
        shape.remove(axis);
        if shape.is_empty() {
            vec![1]
        } else {
            shape
        }
    } else {
        vec![1]
    };
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn expect_axis(input: &TensorType, axis: usize) -> syn::Result<()> {
    if axis < input.rank() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "axis {axis} is out of bounds for rank-{} tensor {:?}",
                input.rank(),
                input.shape
            ),
        ))
    }
}

fn expect_arity(op: &CallOp, args: &[Expr], expected: usize) -> syn::Result<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op:?} expects {expected} arguments, got {}", args.len()),
        ))
    }
}

fn expect_float(op: &CallOp, elem: ElementType) -> syn::Result<()> {
    if elem.is_float() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op:?} supports floating-point tensors only"),
        ))
    }
}

fn expect_numeric_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
    if elem.is_numeric() {
        Ok(())
    } else {
        let verb = if op_name.ends_with('s') {
            "support"
        } else {
            "supports"
        };
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} {verb} numeric tensors only"),
        ))
    }
}

fn expect_bool_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
    if elem.is_bool() {
        Ok(())
    } else {
        let verb = if op_name.ends_with('s') {
            "support"
        } else {
            "supports"
        };
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} {verb} bool tensors only"),
        ))
    }
}
