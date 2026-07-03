use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::Span;

use crate::ops::{broadcast_shape_slices, infer_call_results};
use crate::{
    type_check, AxisSpec, BinaryOp, CallOp, ElementType, Expr, Graph, Input, TensorType,
    TypedGraph, TypedLet, TypedValue, UnaryOp,
};

const MAX_VALUE_AND_GRAD_OUTPUTS: usize = 12;

#[derive(Clone)]
struct Node {
    name: String,
    expr: Expr,
    ty: TensorType,
}

/// Builds a static value-and-gradient graph for a typed scalar-loss graph.
///
/// The generated graph preserves the original inputs and returns the original
/// rank-0 floating-point loss followed by gradients for each floating-point
/// input in input order.
pub fn value_and_grad_graph(graph: &TypedGraph) -> syn::Result<TypedGraph> {
    ValueAndGradBuilder::new(graph)?.build()
}

struct ValueAndGradBuilder<'a> {
    graph: &'a TypedGraph,
    used_names: BTreeSet<String>,
    env: BTreeMap<String, TensorType>,
    lets: Vec<crate::Let>,
    forward_nodes: Vec<Node>,
    grads: BTreeMap<String, Expr>,
    next_name: usize,
    value_name: String,
    differentiable_inputs: Vec<Input>,
}

impl<'a> ValueAndGradBuilder<'a> {
    fn new(graph: &'a TypedGraph) -> syn::Result<Self> {
        if graph.outputs.len() != 1 {
            return Err(error("autograd MVP expects a single-output graph"));
        }
        let loss_ty = &graph.outputs[0];
        if loss_ty.rank() != 0 || !loss_ty.elem.is_float() {
            return Err(error(
                "autograd MVP expects the graph output to be a rank-0 floating-point loss",
            ));
        }
        let differentiable_inputs = graph
            .inputs
            .iter()
            .filter(|input| input.ty.elem.is_float())
            .cloned()
            .collect::<Vec<_>>();
        if differentiable_inputs.is_empty() {
            return Err(error(
                "autograd MVP expects at least one floating-point input",
            ));
        }
        if differentiable_inputs.len() + 1 > MAX_VALUE_AND_GRAD_OUTPUTS {
            return Err(error(format!(
                "autograd MVP supports at most {} floating-point inputs, got {}",
                MAX_VALUE_AND_GRAD_OUTPUTS - 1,
                differentiable_inputs.len()
            )));
        }

        let mut used_names = BTreeSet::new();
        for input in &graph.inputs {
            used_names.insert(input.name.clone());
        }
        for binding in &graph.lets {
            for name in &binding.names {
                used_names.insert(name.clone());
            }
        }
        let mut env = BTreeMap::new();
        for input in &graph.inputs {
            env.insert(input.name.clone(), input.ty.clone());
        }
        let value_name = unique_name("__knok_autograd_value", &mut used_names);
        Ok(Self {
            graph,
            used_names,
            env,
            lets: Vec::new(),
            forward_nodes: Vec::new(),
            grads: BTreeMap::new(),
            next_name: 0,
            value_name,
            differentiable_inputs,
        })
    }

    fn build(mut self) -> syn::Result<TypedGraph> {
        for binding in &self.graph.lets {
            self.flatten_typed_let(binding)?;
        }
        let body = self
            .graph
            .body
            .first()
            .expect("single output was validated");
        let (value_expr, value_ty) = self.flatten_expr(&body.kind)?;
        if value_ty != body.ty {
            return Err(error("internal autograd type mismatch for graph body"));
        }
        let value_name = self.value_name.clone();
        self.bind_forward_node(value_name.clone(), value_expr, value_ty.clone())?;
        self.grads.insert(
            value_name.clone(),
            Expr::Const {
                value: "1.0".into(),
                elem: value_ty.elem,
            },
        );

        for node in self.forward_nodes.clone().into_iter().rev() {
            let Some(grad) = self.grads.remove(&node.name) else {
                continue;
            };
            self.reverse_node(&node, grad)?;
        }

        let mut outputs = vec![self.graph.outputs[0].clone()];
        let mut body = vec![Expr::Var(value_name)];
        for input in self.differentiable_inputs.clone() {
            let grad = self
                .grads
                .remove(&input.name)
                .unwrap_or_else(|| zeros_like(Expr::Var(input.name.clone())));
            let grad = self.bind_auto(grad, input.ty.clone())?;
            outputs.push(input.ty);
            body.push(grad);
        }

        let graph = Graph {
            name: format!("{}_value_and_grad", self.graph.name),
            backend: self.graph.backend.clone(),
            inputs: self.graph.inputs.clone(),
            outputs,
            lets: self.lets,
            body,
        };
        type_check(graph, &[])
    }

    fn flatten_typed_let(&mut self, binding: &TypedLet) -> syn::Result<()> {
        if binding.names.len() != 1 || binding.value.tys.len() != 1 {
            return Err(error(
                "autograd MVP does not support multi-output let bindings",
            ));
        }
        let (expr, ty) = self.flatten_value(&binding.value)?;
        if ty != binding.value.tys[0] {
            return Err(error("internal autograd type mismatch for let binding"));
        }
        self.bind_forward_node(binding.names[0].clone(), expr, ty)
    }

    fn flatten_value(&mut self, value: &TypedValue) -> syn::Result<(Expr, TensorType)> {
        self.flatten_expr(&value.kind)
    }

    fn flatten_expr(&mut self, expr: &Expr) -> syn::Result<(Expr, TensorType)> {
        match expr {
            Expr::Var(name) => {
                let ty =
                    self.env.get(name).cloned().ok_or_else(|| {
                        error(format!("unknown value `{name}` in autograd graph"))
                    })?;
                Ok((Expr::Var(name.clone()), ty))
            }
            Expr::Const { value, elem } => Ok((
                Expr::Const {
                    value: value.clone(),
                    elem: *elem,
                },
                TensorType {
                    elem: *elem,
                    shape: Vec::new(),
                },
            )),
            Expr::Unary { op, value } => {
                let (value, ty) = self.flatten_expr(value)?;
                let expr = Expr::Unary {
                    op: *op,
                    value: Box::new(value),
                };
                let expr = self.bind_forward_auto(expr, ty.clone())?;
                Ok((expr, ty))
            }
            Expr::Binary { op, lhs, rhs } => {
                let (lhs, lhs_ty) = self.flatten_expr(lhs)?;
                let (rhs, rhs_ty) = self.flatten_expr(rhs)?;
                let ty = elementwise_type(&lhs_ty, &rhs_ty)?;
                let expr = Expr::Binary {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
                let expr = self.bind_forward_auto(expr, ty.clone())?;
                Ok((expr, ty))
            }
            Expr::Node { value, .. } => self.flatten_expr(value),
            Expr::TupleGet { .. } => Err(error(
                "autograd MVP does not support tuple projections or multi-output ops",
            )),
            Expr::Call { op, args } => {
                if matches!(op, CallOp::Graph(_) | CallOp::Split { .. }) {
                    return Err(error(format!(
                        "autograd MVP does not support `{}`",
                        op.name()
                    )));
                }
                let mut flat_args = Vec::with_capacity(args.len());
                let mut arg_tys = Vec::with_capacity(args.len());
                for arg in args {
                    let (arg, ty) = self.flatten_expr(arg)?;
                    flat_args.push(arg);
                    arg_tys.push(ty);
                }
                let results = infer_call_results(op, &arg_tys)?;
                if results.len() != 1 {
                    return Err(error(format!(
                        "autograd MVP does not support multi-output `{}`",
                        op.name()
                    )));
                }
                let ty = results[0].clone();
                let expr = Expr::Call {
                    op: op.clone(),
                    args: flat_args,
                };
                let expr = self.bind_forward_auto(expr, ty.clone())?;
                Ok((expr, ty))
            }
        }
    }

    fn reverse_node(&mut self, node: &Node, grad: Expr) -> syn::Result<()> {
        match &node.expr {
            Expr::Var(name) => self.add_grad(name, grad),
            Expr::Const { .. } => Ok(()),
            Expr::Unary {
                op: UnaryOp::Neg,
                value,
            } => self.add_grad_to_expr(value, neg(grad)),
            Expr::Binary { op, lhs, rhs } => self.reverse_binary(*op, lhs, rhs, &node.ty, grad),
            Expr::Call { op, args } => self.reverse_call(node, op, args, grad),
            Expr::Node { .. } | Expr::TupleGet { .. } => Err(error(
                "internal autograd error: forward graph was not flattened",
            )),
        }
    }

    fn reverse_binary(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(lhs)?;
        let rhs_ty = self.expr_ty(rhs)?;
        match op {
            BinaryOp::Add => {
                let lhs_grad = self.unbroadcast(grad.clone(), output_ty, &lhs_ty)?;
                let rhs_grad = self.unbroadcast(grad, output_ty, &rhs_ty)?;
                self.add_grad_to_expr(lhs, lhs_grad)?;
                self.add_grad_to_expr(rhs, rhs_grad)
            }
            BinaryOp::Sub => {
                let lhs_grad = self.unbroadcast(grad.clone(), output_ty, &lhs_ty)?;
                let rhs_grad = self.unbroadcast(neg(grad), output_ty, &rhs_ty)?;
                self.add_grad_to_expr(lhs, lhs_grad)?;
                self.add_grad_to_expr(rhs, rhs_grad)
            }
            BinaryOp::Mul => {
                let lhs_grad = binary(BinaryOp::Mul, grad.clone(), rhs.clone());
                let rhs_grad = binary(BinaryOp::Mul, grad, lhs.clone());
                let lhs_grad = self.unbroadcast(lhs_grad, output_ty, &lhs_ty)?;
                let rhs_grad = self.unbroadcast(rhs_grad, output_ty, &rhs_ty)?;
                self.add_grad_to_expr(lhs, lhs_grad)?;
                self.add_grad_to_expr(rhs, rhs_grad)
            }
            BinaryOp::Div => {
                let lhs_grad = binary(BinaryOp::Div, grad.clone(), rhs.clone());
                let rhs_sq = call(CallOp::Square, vec![rhs.clone()]);
                let rhs_grad = neg(binary(
                    BinaryOp::Div,
                    binary(BinaryOp::Mul, grad, lhs.clone()),
                    rhs_sq,
                ));
                let lhs_grad = self.unbroadcast(lhs_grad, output_ty, &lhs_ty)?;
                let rhs_grad = self.unbroadcast(rhs_grad, output_ty, &rhs_ty)?;
                self.add_grad_to_expr(lhs, lhs_grad)?;
                self.add_grad_to_expr(rhs, rhs_grad)
            }
        }
    }

    fn reverse_call(
        &mut self,
        node: &Node,
        op: &CallOp,
        args: &[Expr],
        grad: Expr,
    ) -> syn::Result<()> {
        match op {
            CallOp::Detach => Ok(()),
            CallOp::Abs => {
                let zero = zeros_like(args[0].clone());
                let positive = call(CallOp::Greater, vec![args[0].clone(), zero.clone()]);
                let negative = call(CallOp::Less, vec![args[0].clone(), zero.clone()]);
                let one = ones_like(args[0].clone());
                let local = call(
                    CallOp::Where,
                    vec![
                        positive,
                        one.clone(),
                        call(CallOp::Where, vec![negative, neg(one), zero]),
                    ],
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Exp => self.add_grad_to_expr(
                &args[0],
                binary(BinaryOp::Mul, grad, Expr::Var(node.name.clone())),
            ),
            CallOp::Exp2 => {
                let y = Expr::Var(node.name.clone());
                let local = binary(BinaryOp::Mul, scalar("0.6931471805599453", node.ty.elem), y);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::ExpM1 => {
                let y = Expr::Var(node.name.clone());
                let local = binary(BinaryOp::Add, y, scalar("1.0", node.ty.elem));
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Log => {
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, args[0].clone()))
            }
            CallOp::Log1P => {
                let denom = binary(BinaryOp::Add, args[0].clone(), scalar("1.0", node.ty.elem));
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, denom))
            }
            CallOp::Log2 => {
                let denom = binary(
                    BinaryOp::Mul,
                    args[0].clone(),
                    scalar("0.6931471805599453", node.ty.elem),
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, denom))
            }
            CallOp::Log10 => {
                let denom = binary(
                    BinaryOp::Mul,
                    args[0].clone(),
                    scalar("2.302585092994046", node.ty.elem),
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, denom))
            }
            CallOp::Sin => {
                let local = call(CallOp::Cos, vec![args[0].clone()]);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Cos => {
                let local = neg(call(CallOp::Sin, vec![args[0].clone()]));
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Tan => {
                let y = Expr::Var(node.name.clone());
                let local = binary(
                    BinaryOp::Add,
                    ones_like(y.clone()),
                    call(CallOp::Square, vec![y]),
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Tanh => {
                let y = Expr::Var(node.name.clone());
                let one_minus_y2 = binary(
                    BinaryOp::Sub,
                    ones_like(y.clone()),
                    call(CallOp::Square, vec![y]),
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, one_minus_y2))
            }
            CallOp::Relu => {
                let zero = zeros_like(args[0].clone());
                let condition = call(CallOp::Greater, vec![args[0].clone(), zero]);
                let zero_grad = zeros_like(grad.clone());
                self.add_grad_to_expr(
                    &args[0],
                    call(CallOp::Where, vec![condition, grad, zero_grad]),
                )
            }
            CallOp::Sigmoid => {
                let y = Expr::Var(node.name.clone());
                let one_minus_y = binary(BinaryOp::Sub, ones_like(y.clone()), y.clone());
                let local = binary(BinaryOp::Mul, y, one_minus_y);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Square => {
                let two = scalar("2.0", node.ty.elem);
                let local = binary(BinaryOp::Mul, two, args[0].clone());
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Reciprocal => {
                let y = Expr::Var(node.name.clone());
                let local = neg(call(CallOp::Square, vec![y]));
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Sqrt => {
                let y = Expr::Var(node.name.clone());
                let two_y = binary(BinaryOp::Mul, scalar("2.0", node.ty.elem), y);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, two_y))
            }
            CallOp::Pow => self.reverse_pow(args, Expr::Var(node.name.clone()), &node.ty, grad),
            CallOp::Minimum => self.reverse_minimum_maximum(args, &node.ty, grad, false),
            CallOp::Maximum => self.reverse_minimum_maximum(args, &node.ty, grad, true),
            CallOp::Clip => self.reverse_clip(args, &node.ty, grad),
            CallOp::Where => self.reverse_where(args, &node.ty, grad),
            CallOp::Sum(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let input_grad = self.expand_reduction_grad(grad, &node.ty, &input_ty, *axis)?;
                self.add_grad_to_expr(&args[0], input_grad)
            }
            CallOp::Prod(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let output = Expr::Var(node.name.clone());
                let output = self.expand_reduction_grad(output, &node.ty, &input_ty, *axis)?;
                let grad = self.expand_reduction_grad(grad, &node.ty, &input_ty, *axis)?;
                let local = binary(BinaryOp::Div, output, args[0].clone());
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Mean(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let count = reduction_count(&input_ty, *axis);
                let scaled = binary(
                    BinaryOp::Div,
                    grad,
                    scalar(&format!("{count}.0"), node.ty.elem),
                );
                let input_grad = self.expand_reduction_grad(scaled, &node.ty, &input_ty, *axis)?;
                self.add_grad_to_expr(&args[0], input_grad)
            }
            CallOp::Max(axis) | CallOp::Min(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let output = Expr::Var(node.name.clone());
                let selected = self.expand_reduction_grad(output, &node.ty, &input_ty, *axis)?;
                let grad = self.expand_reduction_grad(grad, &node.ty, &input_ty, *axis)?;
                let mask = call(CallOp::Equal, vec![args[0].clone(), selected]);
                self.add_grad_to_expr(&args[0], mask_grad(mask, grad))
            }
            CallOp::Matmul => self.reverse_matmul(args, &node.ty, grad),
            CallOp::Reshape(_) => {
                let input_ty = self.expr_ty(&args[0])?;
                self.add_grad_to_expr(&args[0], call(CallOp::Reshape(input_ty), vec![grad]))
            }
            CallOp::Squeeze(_) | CallOp::Unsqueeze(_) => {
                let input_ty = self.expr_ty(&args[0])?;
                self.add_grad_to_expr(&args[0], call(CallOp::Reshape(input_ty), vec![grad]))
            }
            CallOp::Broadcast(_) => {
                let input_ty = self.expr_ty(&args[0])?;
                let input_grad = self.unbroadcast(grad, &node.ty, &input_ty)?;
                self.add_grad_to_expr(&args[0], input_grad)
            }
            CallOp::Pad { lows, .. } => {
                let input_ty = self.expr_ty(&args[0])?;
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Slice {
                            target: input_ty,
                            starts: lows.clone(),
                        },
                        vec![grad],
                    ),
                )
            }
            CallOp::Slice { starts, .. } => {
                let input_ty = self.expr_ty(&args[0])?;
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Pad {
                            target: input_ty,
                            lows: starts.clone(),
                        },
                        vec![grad],
                    ),
                )
            }
            CallOp::Concat(axis) => self.reverse_concat(args, *axis, grad),
            CallOp::Stack(axis) => {
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Take {
                            axis: *axis,
                            index: 0,
                        },
                        vec![grad.clone()],
                    ),
                )?;
                self.add_grad_to_expr(
                    &args[1],
                    call(
                        CallOp::Take {
                            axis: *axis,
                            index: 1,
                        },
                        vec![grad],
                    ),
                )
            }
            CallOp::Take { axis, index } => {
                let input_ty = self.expr_ty(&args[0])?;
                let mut expanded_ty = input_ty.clone();
                expanded_ty.shape[*axis] = 1;
                let mut lows = vec![0; input_ty.rank()];
                lows[*axis] = *index;
                let expanded = call(CallOp::Reshape(expanded_ty), vec![grad]);
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Pad {
                            target: input_ty,
                            lows,
                        },
                        vec![expanded],
                    ),
                )
            }
            CallOp::Transpose(axes) => {
                let input_ty = self.expr_ty(&args[0])?;
                let inverse = inverse_permutation(default_axes(&input_ty, axes));
                self.add_grad_to_expr(&args[0], call(CallOp::Transpose(inverse), vec![grad]))
            }
            CallOp::SwapAxes { axis0, axis1 } => self.add_grad_to_expr(
                &args[0],
                call(
                    CallOp::SwapAxes {
                        axis0: *axis0,
                        axis1: *axis1,
                    },
                    vec![grad],
                ),
            ),
            CallOp::MoveAxis {
                source,
                destination,
            } => self.add_grad_to_expr(
                &args[0],
                call(
                    CallOp::MoveAxis {
                        source: *destination,
                        destination: *source,
                    },
                    vec![grad],
                ),
            ),
            CallOp::PermuteDims(axes) => {
                let inverse = inverse_permutation(axes.clone());
                self.add_grad_to_expr(&args[0], call(CallOp::PermuteDims(inverse), vec![grad]))
            }
            CallOp::Permute { axes, .. } => {
                let inverse = inverse_permutation(axes.clone());
                self.add_grad_to_expr(&args[0], call(CallOp::PermuteDims(inverse), vec![grad]))
            }
            CallOp::Flip(axes) => {
                self.add_grad_to_expr(&args[0], call(CallOp::Flip(axes.clone()), vec![grad]))
            }
            CallOp::Roll { axis, shift } => {
                let input_ty = self.expr_ty(&args[0])?;
                let dim = input_ty.shape[*axis];
                let reverse_shift = if dim == 0 {
                    0
                } else {
                    (dim - (shift % dim)) % dim
                };
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Roll {
                            axis: *axis,
                            shift: reverse_shift,
                        },
                        vec![grad],
                    ),
                )
            }
            CallOp::FullLike => {
                let fill_ty = self.expr_ty(&args[1])?;
                let fill_grad = self.unbroadcast(grad, &node.ty, &fill_ty)?;
                self.add_grad_to_expr(&args[1], fill_grad)
            }
            CallOp::Ceil
            | CallOp::Floor
            | CallOp::Rint
            | CallOp::Round
            | CallOp::IsNan
            | CallOp::Greater
            | CallOp::GreaterEqual
            | CallOp::Less
            | CallOp::LessEqual
            | CallOp::Equal
            | CallOp::NotEqual
            | CallOp::LogicalAnd
            | CallOp::LogicalOr
            | CallOp::LogicalXor
            | CallOp::LogicalNot
            | CallOp::Arange(_)
            | CallOp::Linspace(_)
            | CallOp::Eye(_)
            | CallOp::OnesLike
            | CallOp::ZerosLike => Ok(()),
            _ => Err(error(format!(
                "autograd MVP does not support differentiating `{}`",
                op.name()
            ))),
        }
    }

    fn reverse_matmul(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let spec = MatmulGradSpec::new(&lhs_ty, &rhs_ty, output_ty)?;
        let lhs = reshape_if_needed(args[0].clone(), &lhs_ty, &spec.lhs_matrix_ty);
        let rhs = reshape_if_needed(args[1].clone(), &rhs_ty, &spec.rhs_matrix_ty);
        let grad = reshape_if_needed(grad, output_ty, &spec.grad_matrix_ty);

        let rhs_t_ty = swap_last_two_type(&spec.rhs_matrix_ty)?;
        let rhs_t = swap_last_two(rhs, &spec.rhs_matrix_ty)?;
        let lhs_grad = call(CallOp::Matmul, vec![grad.clone(), rhs_t]);
        let lhs_grad_ty = matmul_type(&spec.grad_matrix_ty, &rhs_t_ty)?;
        let (lhs_grad, lhs_grad_ty) = spec.demote_lhs_grad(lhs_grad, lhs_grad_ty)?;
        let lhs_grad = self.unbroadcast(lhs_grad, &lhs_grad_ty, &lhs_ty)?;
        self.add_grad_to_expr(&args[0], lhs_grad)?;

        let lhs_t_ty = swap_last_two_type(&spec.lhs_matrix_ty)?;
        let lhs_t = swap_last_two(lhs, &spec.lhs_matrix_ty)?;
        let rhs_grad = call(CallOp::Matmul, vec![lhs_t, grad]);
        let rhs_grad_ty = matmul_type(&lhs_t_ty, &spec.grad_matrix_ty)?;
        let (rhs_grad, rhs_grad_ty) = spec.demote_rhs_grad(rhs_grad, rhs_grad_ty)?;
        let rhs_grad = self.unbroadcast(rhs_grad, &rhs_grad_ty, &rhs_ty)?;
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_pow(
        &mut self,
        args: &[Expr],
        output: Expr,
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let lhs_local = binary(
            BinaryOp::Mul,
            args[1].clone(),
            call(
                CallOp::Pow,
                vec![
                    args[0].clone(),
                    binary(
                        BinaryOp::Sub,
                        args[1].clone(),
                        scalar("1.0", output_ty.elem),
                    ),
                ],
            ),
        );
        let lhs_grad = binary(BinaryOp::Mul, grad.clone(), lhs_local);
        let lhs_grad = self.unbroadcast(lhs_grad, output_ty, &lhs_ty)?;
        self.add_grad_to_expr(&args[0], lhs_grad)?;

        let rhs_local = binary(
            BinaryOp::Mul,
            output,
            call(CallOp::Log, vec![args[0].clone()]),
        );
        let rhs_grad = binary(BinaryOp::Mul, grad, rhs_local);
        let rhs_grad = self.unbroadcast(rhs_grad, output_ty, &rhs_ty)?;
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_minimum_maximum(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
        maximum: bool,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let (lhs_op, rhs_op) = if maximum {
            (CallOp::GreaterEqual, CallOp::Greater)
        } else {
            (CallOp::LessEqual, CallOp::Less)
        };
        let lhs_mask = call(lhs_op, vec![args[0].clone(), args[1].clone()]);
        let rhs_mask = call(rhs_op, vec![args[1].clone(), args[0].clone()]);
        let lhs_grad = self.unbroadcast(mask_grad(lhs_mask, grad.clone()), output_ty, &lhs_ty)?;
        let rhs_grad = self.unbroadcast(mask_grad(rhs_mask, grad), output_ty, &rhs_ty)?;
        self.add_grad_to_expr(&args[0], lhs_grad)?;
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_clip(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let value_ty = self.expr_ty(&args[0])?;
        let min_ty = self.expr_ty(&args[1])?;
        let max_ty = self.expr_ty(&args[2])?;
        let above_min = call(CallOp::GreaterEqual, vec![args[0].clone(), args[1].clone()]);
        let below_max = call(CallOp::LessEqual, vec![args[0].clone(), args[2].clone()]);
        let value_mask = call(CallOp::LogicalAnd, vec![above_min, below_max]);
        let min_mask = call(CallOp::Less, vec![args[0].clone(), args[1].clone()]);
        let max_mask = call(CallOp::Greater, vec![args[0].clone(), args[2].clone()]);

        let value_grad =
            self.unbroadcast(mask_grad(value_mask, grad.clone()), output_ty, &value_ty)?;
        let min_grad = self.unbroadcast(mask_grad(min_mask, grad.clone()), output_ty, &min_ty)?;
        let max_grad = self.unbroadcast(mask_grad(max_mask, grad), output_ty, &max_ty)?;
        self.add_grad_to_expr(&args[0], value_grad)?;
        self.add_grad_to_expr(&args[1], min_grad)?;
        self.add_grad_to_expr(&args[2], max_grad)
    }

    fn reverse_where(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let true_ty = self.expr_ty(&args[1])?;
        let false_ty = self.expr_ty(&args[2])?;
        let true_grad = self.unbroadcast(
            mask_grad(args[0].clone(), grad.clone()),
            output_ty,
            &true_ty,
        )?;
        let false_grad = self.unbroadcast(
            call(
                CallOp::Where,
                vec![args[0].clone(), zeros_like(grad.clone()), grad],
            ),
            output_ty,
            &false_ty,
        )?;
        self.add_grad_to_expr(&args[1], true_grad)?;
        self.add_grad_to_expr(&args[2], false_grad)
    }

    fn reverse_concat(&mut self, args: &[Expr], axis: usize, grad: Expr) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let lhs_starts = vec![0; lhs_ty.rank()];
        let mut rhs_starts = vec![0; rhs_ty.rank()];
        rhs_starts[axis] = lhs_ty.shape[axis];
        let lhs_grad = call(
            CallOp::Slice {
                target: lhs_ty,
                starts: lhs_starts,
            },
            vec![grad.clone()],
        );
        let rhs_grad = call(
            CallOp::Slice {
                target: rhs_ty,
                starts: rhs_starts,
            },
            vec![grad],
        );
        self.add_grad_to_expr(&args[0], lhs_grad)?;
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn expand_reduction_grad(
        &mut self,
        grad: Expr,
        grad_ty: &TensorType,
        input_ty: &TensorType,
        axis: AxisSpec,
    ) -> syn::Result<Expr> {
        let (expanded, expanded_ty) = match axis {
            AxisSpec::All => (grad, grad_ty.clone()),
            AxisSpec::One(axis) => {
                let mut shape = grad_ty.shape.clone();
                shape.insert(axis, 1);
                let unsqueezed_ty = TensorType {
                    elem: grad_ty.elem,
                    shape,
                };
                (
                    call(CallOp::Reshape(unsqueezed_ty.clone()), vec![grad]),
                    unsqueezed_ty,
                )
            }
        };
        Ok(if expanded_ty == *input_ty {
            expanded
        } else {
            call(CallOp::Broadcast(input_ty.clone()), vec![expanded])
        })
    }

    fn unbroadcast(
        &mut self,
        mut expr: Expr,
        source_ty: &TensorType,
        target_ty: &TensorType,
    ) -> syn::Result<Expr> {
        let mut current = source_ty.clone();
        while current.rank() > target_ty.rank() {
            expr = call(CallOp::Sum(AxisSpec::One(0)), vec![expr]);
            current.shape.remove(0);
        }
        if current.rank() != target_ty.rank() {
            return Err(error("cannot unbroadcast gradient to higher-rank target"));
        }
        for axis in 0..target_ty.rank() {
            if target_ty.shape[axis] == 1 && current.shape[axis] != 1 {
                expr = call(CallOp::Sum(AxisSpec::One(axis)), vec![expr]);
                current.shape.remove(axis);
                let mut shape = current.shape.clone();
                shape.insert(axis, 1);
                current.shape = shape;
                expr = call(CallOp::Reshape(current.clone()), vec![expr]);
            }
        }
        if current != *target_ty {
            return Err(error(format!(
                "cannot unbroadcast gradient from {:?} to {:?}",
                source_ty, target_ty
            )));
        }
        Ok(expr)
    }

    fn add_grad_to_expr(&mut self, expr: &Expr, grad: Expr) -> syn::Result<()> {
        match expr {
            Expr::Var(name) => self.add_grad(name, grad),
            Expr::Const { .. } => Ok(()),
            _ => Err(error(
                "internal autograd error: gradient target was not flattened",
            )),
        }
    }

    fn add_grad(&mut self, name: &str, grad: Expr) -> syn::Result<()> {
        if !self.env.contains_key(name) {
            return Err(error(format!("unknown gradient target `{name}`")));
        }
        let grad = if let Some(existing) = self.grads.remove(name) {
            binary(BinaryOp::Add, existing, grad)
        } else {
            grad
        };
        self.grads.insert(name.into(), grad);
        Ok(())
    }

    fn expr_ty(&self, expr: &Expr) -> syn::Result<TensorType> {
        match expr {
            Expr::Var(name) => self
                .env
                .get(name)
                .cloned()
                .ok_or_else(|| error(format!("unknown value `{name}` in autograd graph"))),
            Expr::Const { elem, .. } => Ok(TensorType {
                elem: *elem,
                shape: Vec::new(),
            }),
            _ => Err(error(
                "internal autograd error: expected a flattened value reference",
            )),
        }
    }

    fn bind_forward_node(&mut self, name: String, expr: Expr, ty: TensorType) -> syn::Result<()> {
        self.bind_named(name.clone(), expr.clone(), ty.clone())?;
        self.forward_nodes.push(Node { name, expr, ty });
        Ok(())
    }

    fn bind_forward_auto(&mut self, expr: Expr, ty: TensorType) -> syn::Result<Expr> {
        let name = self.next_auto_name();
        self.bind_forward_node(name.clone(), expr, ty)?;
        Ok(Expr::Var(name))
    }

    fn bind_auto(&mut self, expr: Expr, ty: TensorType) -> syn::Result<Expr> {
        let name = self.next_auto_name();
        self.bind_named(name.clone(), expr, ty)?;
        Ok(Expr::Var(name))
    }

    fn bind_named(&mut self, name: String, expr: Expr, ty: TensorType) -> syn::Result<()> {
        self.lets.push(crate::Let {
            names: vec![name.clone()],
            value: expr,
        });
        self.env.insert(name, ty);
        Ok(())
    }

    fn next_auto_name(&mut self) -> String {
        loop {
            let candidate = format!("__knok_autograd_{}", self.next_name);
            self.next_name += 1;
            if self.used_names.insert(candidate.clone()) {
                return candidate;
            }
        }
    }
}

fn unique_name(base: &str, used: &mut BTreeSet<String>) -> String {
    let mut candidate = base.to_string();
    let mut suffix = 1;
    while used.contains(&candidate) {
        candidate = format!("{base}_{suffix}");
        suffix += 1;
    }
    used.insert(candidate.clone());
    candidate
}

fn elementwise_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    if lhs.elem != rhs.elem {
        return Err(error(
            "elementwise gradient operands must have matching dtypes",
        ));
    }
    let shape = broadcast_shape_slices(&lhs.shape, &rhs.shape)
        .map_err(|message| error(format!("elementwise gradient broadcast failed: {message}")))?;
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

struct MatmulGradSpec {
    lhs_is_vector: bool,
    rhs_is_vector: bool,
    lhs_matrix_ty: TensorType,
    rhs_matrix_ty: TensorType,
    grad_matrix_ty: TensorType,
}

impl MatmulGradSpec {
    fn new(lhs_ty: &TensorType, rhs_ty: &TensorType, output_ty: &TensorType) -> syn::Result<Self> {
        // Match NumPy matmul gradients by temporarily promoting vector operands
        // to matrices, then demoting and unbroadcasting the resulting gradients.
        let lhs_is_vector = lhs_ty.rank() == 1;
        let rhs_is_vector = rhs_ty.rank() == 1;
        let lhs_matrix_ty = promoted_lhs_matmul_type(lhs_ty);
        let rhs_matrix_ty = promoted_rhs_matmul_type(rhs_ty);
        let grad_matrix_ty = matmul_type(&lhs_matrix_ty, &rhs_matrix_ty)?;
        let expected_output =
            demoted_matmul_output_type(&grad_matrix_ty, lhs_is_vector, rhs_is_vector)?;
        if expected_output != *output_ty {
            return Err(error(format!(
                "internal autograd matmul type mismatch: expected {:?}, got {:?}",
                expected_output, output_ty
            )));
        }
        Ok(Self {
            lhs_is_vector,
            rhs_is_vector,
            lhs_matrix_ty,
            rhs_matrix_ty,
            grad_matrix_ty,
        })
    }

    fn demote_lhs_grad(&self, expr: Expr, ty: TensorType) -> syn::Result<(Expr, TensorType)> {
        if self.lhs_is_vector {
            let axis = ty.rank().saturating_sub(2);
            remove_axis(expr, ty, axis)
        } else {
            Ok((expr, ty))
        }
    }

    fn demote_rhs_grad(&self, expr: Expr, ty: TensorType) -> syn::Result<(Expr, TensorType)> {
        if self.rhs_is_vector {
            let axis = ty.rank().saturating_sub(1);
            remove_axis(expr, ty, axis)
        } else {
            Ok((expr, ty))
        }
    }
}

fn promoted_lhs_matmul_type(ty: &TensorType) -> TensorType {
    if ty.rank() == 1 {
        TensorType {
            elem: ty.elem,
            shape: vec![1, ty.shape[0]],
        }
    } else {
        ty.clone()
    }
}

fn promoted_rhs_matmul_type(ty: &TensorType) -> TensorType {
    if ty.rank() == 1 {
        TensorType {
            elem: ty.elem,
            shape: vec![ty.shape[0], 1],
        }
    } else {
        ty.clone()
    }
}

fn demoted_matmul_output_type(
    promoted: &TensorType,
    lhs_is_vector: bool,
    rhs_is_vector: bool,
) -> syn::Result<TensorType> {
    let mut shape = promoted.shape.clone();
    if rhs_is_vector {
        remove_shape_axis(&mut shape, promoted.rank().saturating_sub(1))?;
    }
    if lhs_is_vector {
        remove_shape_axis(&mut shape, promoted.rank().saturating_sub(2))?;
    }
    Ok(TensorType {
        elem: promoted.elem,
        shape,
    })
}

fn matmul_type(lhs_ty: &TensorType, rhs_ty: &TensorType) -> syn::Result<TensorType> {
    let results = infer_call_results(&CallOp::Matmul, &[lhs_ty.clone(), rhs_ty.clone()])?;
    let [ty]: [TensorType; 1] = results
        .try_into()
        .map_err(|_| error("internal autograd error: matmul gradient produced multiple outputs"))?;
    Ok(ty)
}

fn reshape_if_needed(expr: Expr, source_ty: &TensorType, target_ty: &TensorType) -> Expr {
    if source_ty == target_ty {
        expr
    } else {
        call(CallOp::Reshape(target_ty.clone()), vec![expr])
    }
}

fn remove_axis(expr: Expr, mut ty: TensorType, axis: usize) -> syn::Result<(Expr, TensorType)> {
    remove_shape_axis(&mut ty.shape, axis)?;
    let expr = call(CallOp::Reshape(ty.clone()), vec![expr]);
    Ok((expr, ty))
}

fn remove_shape_axis(shape: &mut Vec<usize>, axis: usize) -> syn::Result<()> {
    if axis >= shape.len() {
        return Err(error(format!(
            "internal autograd error: cannot remove axis {axis} from shape {:?}",
            shape
        )));
    }
    let dim = shape[axis];
    if dim != 1 {
        return Err(error(format!(
            "internal autograd error: expected singleton matmul axis {axis}, got {dim}"
        )));
    }
    shape.remove(axis);
    Ok(())
}

fn swap_last_two(expr: Expr, ty: &TensorType) -> syn::Result<Expr> {
    if ty.rank() < 2 {
        return Err(error(
            "internal autograd error: matmul operand was not promoted to matrix rank",
        ));
    }
    Ok(call(
        CallOp::SwapAxes {
            axis0: ty.rank() - 2,
            axis1: ty.rank() - 1,
        },
        vec![expr],
    ))
}

fn swap_last_two_type(ty: &TensorType) -> syn::Result<TensorType> {
    if ty.rank() < 2 {
        return Err(error(
            "internal autograd error: matmul operand was not promoted to matrix rank",
        ));
    }
    let mut swapped = ty.clone();
    let rank = swapped.rank();
    swapped.shape.swap(rank - 2, rank - 1);
    Ok(swapped)
}

fn default_axes(input_ty: &TensorType, axes: &[usize]) -> Vec<usize> {
    if axes.is_empty() {
        (0..input_ty.rank()).rev().collect()
    } else {
        axes.to_vec()
    }
}

fn inverse_permutation(axes: Vec<usize>) -> Vec<usize> {
    let mut inverse = vec![0; axes.len()];
    for (output_axis, input_axis) in axes.into_iter().enumerate() {
        inverse[input_axis] = output_axis;
    }
    inverse
}

fn reduction_count(input_ty: &TensorType, axis: AxisSpec) -> usize {
    match axis {
        AxisSpec::All => input_ty.shape.iter().product(),
        AxisSpec::One(axis) => input_ty.shape[axis],
    }
}

fn call(op: CallOp, args: Vec<Expr>) -> Expr {
    Expr::Call { op, args }
}

fn binary(op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn neg(value: Expr) -> Expr {
    Expr::Unary {
        op: UnaryOp::Neg,
        value: Box::new(value),
    }
}

fn scalar(value: &str, elem: ElementType) -> Expr {
    Expr::Const {
        value: value.into(),
        elem,
    }
}

fn zeros_like(value: Expr) -> Expr {
    call(CallOp::ZerosLike, vec![value])
}

fn ones_like(value: Expr) -> Expr {
    call(CallOp::OnesLike, vec![value])
}

fn mask_grad(condition: Expr, grad: Expr) -> Expr {
    call(
        CallOp::Where,
        vec![condition, grad.clone(), zeros_like(grad)],
    )
}

fn error(message: impl Into<String>) -> syn::Error {
    syn::Error::new(Span::call_site(), message.into())
}
