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
            CallOp::Exp => self.add_grad_to_expr(
                &args[0],
                binary(BinaryOp::Mul, grad, Expr::Var(node.name.clone())),
            ),
            CallOp::Log => {
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, args[0].clone()))
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
            CallOp::Sqrt => {
                let y = Expr::Var(node.name.clone());
                let two_y = binary(BinaryOp::Mul, scalar("2.0", node.ty.elem), y);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, two_y))
            }
            CallOp::Sum(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let input_grad = self.expand_reduction_grad(grad, &node.ty, &input_ty, *axis)?;
                self.add_grad_to_expr(&args[0], input_grad)
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
            CallOp::Matmul => self.reverse_matmul(args, grad),
            CallOp::Reshape(_) => {
                let input_ty = self.expr_ty(&args[0])?;
                self.add_grad_to_expr(&args[0], call(CallOp::Reshape(input_ty), vec![grad]))
            }
            CallOp::Broadcast(_) => {
                let input_ty = self.expr_ty(&args[0])?;
                let input_grad = self.unbroadcast(grad, &node.ty, &input_ty)?;
                self.add_grad_to_expr(&args[0], input_grad)
            }
            CallOp::Transpose(axes) => {
                let input_ty = self.expr_ty(&args[0])?;
                let inverse = inverse_permutation(default_axes(&input_ty, axes));
                self.add_grad_to_expr(&args[0], call(CallOp::Transpose(inverse), vec![grad]))
            }
            CallOp::PermuteDims(axes) => {
                let inverse = inverse_permutation(axes.clone());
                self.add_grad_to_expr(&args[0], call(CallOp::PermuteDims(inverse), vec![grad]))
            }
            CallOp::Permute { axes, .. } => {
                let input_ty = self.expr_ty(&args[0])?;
                let inverse = inverse_permutation(axes.clone());
                self.add_grad_to_expr(&args[0], call(CallOp::PermuteDims(inverse), vec![grad]))?;
                let _ = input_ty;
                Ok(())
            }
            CallOp::Arange(_) | CallOp::Linspace(_) | CallOp::Eye(_) => Ok(()),
            _ => Err(error(format!(
                "autograd MVP does not support differentiating `{}`",
                op.name()
            ))),
        }
    }

    fn reverse_matmul(&mut self, args: &[Expr], grad: Expr) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        if lhs_ty.rank() != 2 || rhs_ty.rank() != 2 {
            return Err(error(
                "autograd MVP only supports rank-2 matrix-matrix matmul gradients",
            ));
        }
        let rhs_t = call(CallOp::Transpose(Vec::new()), vec![args[1].clone()]);
        let lhs_t = call(CallOp::Transpose(Vec::new()), vec![args[0].clone()]);
        self.add_grad_to_expr(&args[0], call(CallOp::Matmul, vec![grad.clone(), rhs_t]))?;
        self.add_grad_to_expr(&args[1], call(CallOp::Matmul, vec![lhs_t, grad]))
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
                    call(CallOp::Unsqueeze(unsqueezed_ty.clone()), vec![grad]),
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
                expr = call(CallOp::Unsqueeze(current.clone()), vec![expr]);
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

fn error(message: impl Into<String>) -> syn::Error {
    syn::Error::new(Span::call_site(), message.into())
}
