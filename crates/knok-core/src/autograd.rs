use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::Span;

use crate::ops::{broadcast_shape_slices, infer_call_results};
use crate::{
    type_check, AxisSpec, BinaryOp, CallOp, Conv2dOptions, ElementType, Expr, Graph, Input,
    Pool2dOptions, TensorType, TypedGraph, TypedLet, TypedValue, UnaryOp,
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
    value_and_grad_graph_with_registry(graph, &BTreeMap::new())
}

/// Builds a static value-and-gradient graph, resolving nested graph calls from
/// `graph_registry` by inlining their typed bodies before differentiation.
pub fn value_and_grad_graph_with_registry(
    graph: &TypedGraph,
    graph_registry: &BTreeMap<String, TypedGraph>,
) -> syn::Result<TypedGraph> {
    ValueAndGradBuilder::new(graph, graph_registry)?.build()
}

struct ValueAndGradBuilder<'a> {
    graph: &'a TypedGraph,
    graph_registry: &'a BTreeMap<String, TypedGraph>,
    inline_stack: Vec<String>,
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
    fn new(
        graph: &'a TypedGraph,
        graph_registry: &'a BTreeMap<String, TypedGraph>,
    ) -> syn::Result<Self> {
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
            graph_registry,
            inline_stack: Vec::new(),
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
            let grad = self.gradient_output_for_input(&input)?;
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
        if binding.names.len() != binding.value.tys.len() {
            return Err(error(
                "internal autograd error: let binding arity did not match value arity",
            ));
        }
        let outputs = self.flatten_value_outputs(&binding.value)?;
        if outputs.len() != binding.names.len() {
            return Err(error(
                "internal autograd error: flattened let binding arity mismatch",
            ));
        }
        for ((name, expected_ty), (expr, ty)) in
            binding.names.iter().zip(&binding.value.tys).zip(outputs)
        {
            if ty != *expected_ty {
                return Err(error("internal autograd type mismatch for let binding"));
            }
            self.bind_forward_node(name.clone(), expr, ty)?;
        }
        Ok(())
    }

    fn flatten_value_outputs(
        &mut self,
        value: &TypedValue,
    ) -> syn::Result<Vec<(Expr, TensorType)>> {
        let outputs = self.flatten_expr_outputs(&value.kind)?;
        if outputs.len() != value.tys.len() {
            return Err(error(
                "internal autograd error: flattened value arity mismatch",
            ));
        }
        for ((_, ty), expected_ty) in outputs.iter().zip(&value.tys) {
            if ty != expected_ty {
                return Err(error("internal autograd type mismatch for value"));
            }
        }
        Ok(outputs)
    }

    fn flatten_expr(&mut self, expr: &Expr) -> syn::Result<(Expr, TensorType)> {
        self.flatten_expr_scoped(expr, &BTreeMap::new())
    }

    fn flatten_expr_outputs(&mut self, expr: &Expr) -> syn::Result<Vec<(Expr, TensorType)>> {
        self.flatten_expr_outputs_scoped(expr, &BTreeMap::new())
    }

    fn flatten_expr_scoped(
        &mut self,
        expr: &Expr,
        scope: &BTreeMap<String, (Expr, TensorType)>,
    ) -> syn::Result<(Expr, TensorType)> {
        match expr {
            Expr::Var(name) => {
                if let Some((expr, ty)) = scope.get(name) {
                    return Ok((expr.clone(), ty.clone()));
                }
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
                let (value, ty) = self.flatten_expr_scoped(value, scope)?;
                let expr = Expr::Unary {
                    op: *op,
                    value: Box::new(value),
                };
                let expr = self.bind_forward_auto(expr, ty.clone())?;
                Ok((expr, ty))
            }
            Expr::Binary { op, lhs, rhs } => {
                let (lhs, lhs_ty) = self.flatten_expr_scoped(lhs, scope)?;
                let (rhs, rhs_ty) = self.flatten_expr_scoped(rhs, scope)?;
                let ty = elementwise_type(&lhs_ty, &rhs_ty)?;
                let expr = Expr::Binary {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
                let expr = self.bind_forward_auto(expr, ty.clone())?;
                Ok((expr, ty))
            }
            Expr::Node { value, .. } => self.flatten_expr_scoped(value, scope),
            Expr::TupleGet { value, index, .. } => {
                let outputs = self.flatten_expr_outputs_scoped(value, scope)?;
                outputs.get(*index).cloned().ok_or_else(|| {
                    error(format!(
                        "tuple projection index {index} out of bounds for {} values",
                        outputs.len()
                    ))
                })
            }
            Expr::Call { op, args } => {
                if matches!(op, CallOp::Graph(_) | CallOp::Split { .. }) {
                    let outputs = self.flatten_expr_outputs_scoped(expr, scope)?;
                    let [output]: [(Expr, TensorType); 1] = outputs.try_into().map_err(|outputs: Vec<_>| {
                        error(format!(
                            "autograd MVP cannot use multi-output `{}` as a tensor expression ({} outputs)",
                            op.name(),
                            outputs.len()
                        ))
                    })?;
                    return Ok(output);
                }
                let mut flat_args = Vec::with_capacity(args.len());
                let mut arg_tys = Vec::with_capacity(args.len());
                for arg in args {
                    let (arg, ty) = self.flatten_expr_scoped(arg, scope)?;
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

    fn flatten_expr_outputs_scoped(
        &mut self,
        expr: &Expr,
        scope: &BTreeMap<String, (Expr, TensorType)>,
    ) -> syn::Result<Vec<(Expr, TensorType)>> {
        match expr {
            Expr::Node { value, .. } => self.flatten_expr_outputs_scoped(value, scope),
            Expr::Call {
                op: CallOp::Split { axis, sections },
                args,
            } => self.flatten_split_outputs(args, *axis, sections, scope),
            Expr::Call {
                op: CallOp::Graph(name),
                args,
            } => self.flatten_graph_outputs(name, args, scope),
            _ => self
                .flatten_expr_scoped(expr, scope)
                .map(|output| vec![output]),
        }
    }

    fn flatten_split_outputs(
        &mut self,
        args: &[Expr],
        axis: usize,
        sections: &[usize],
        scope: &BTreeMap<String, (Expr, TensorType)>,
    ) -> syn::Result<Vec<(Expr, TensorType)>> {
        if args.len() != 1 {
            return Err(error("split expects one argument"));
        }
        let (input, input_ty) = self.flatten_expr_scoped(&args[0], scope)?;
        let output_tys = infer_call_results(
            &CallOp::Split {
                axis,
                sections: sections.to_vec(),
            },
            core::slice::from_ref(&input_ty),
        )?;
        let mut offset = 0;
        output_tys
            .into_iter()
            .map(|ty| {
                let mut starts = vec![0; input_ty.rank()];
                starts[axis] = offset;
                offset += ty.shape[axis];
                let expr = call(
                    CallOp::Slice {
                        target: ty.clone(),
                        starts,
                    },
                    vec![input.clone()],
                );
                let expr = self.bind_forward_auto(expr, ty.clone())?;
                Ok((expr, ty))
            })
            .collect()
    }

    fn flatten_graph_outputs(
        &mut self,
        name: &str,
        args: &[Expr],
        scope: &BTreeMap<String, (Expr, TensorType)>,
    ) -> syn::Result<Vec<(Expr, TensorType)>> {
        if self.inline_stack.iter().any(|current| current == name) {
            return Err(error(format!(
                "recursive graph call `{name}` is not supported by autograd"
            )));
        }
        let graph = self.graph_registry.get(name).ok_or_else(|| {
            error(format!(
                "autograd MVP cannot resolve graph call `{name}` without a graph registry entry"
            ))
        })?;
        self.inline_stack.push(name.to_string());
        let result = (|| {
            if graph.inputs.len() != args.len() {
                return Err(error(format!(
                    "graph call `{name}` expected {} arguments, got {}",
                    graph.inputs.len(),
                    args.len()
                )));
            }

            let mut local_scope = BTreeMap::new();
            for (input, arg) in graph.inputs.iter().zip(args) {
                let (expr, ty) = self.flatten_expr_scoped(arg, scope)?;
                if ty != input.ty {
                    return Err(error(format!(
                        "graph call `{name}` argument `{}` expected {:?}, got {:?}",
                        input.name, input.ty, ty
                    )));
                }
                local_scope.insert(input.name.clone(), (expr, ty));
            }

            for binding in &graph.lets {
                let outputs =
                    self.flatten_expr_outputs_scoped(&binding.value.kind, &local_scope)?;
                if outputs.len() != binding.names.len() || outputs.len() != binding.value.tys.len()
                {
                    return Err(error(format!(
                        "graph call `{name}` has an internal let arity mismatch"
                    )));
                }
                for ((binding_name, expected_ty), (expr, ty)) in
                    binding.names.iter().zip(&binding.value.tys).zip(outputs)
                {
                    if ty != *expected_ty {
                        return Err(error(format!(
                            "graph call `{name}` has an internal let type mismatch"
                        )));
                    }
                    local_scope.insert(binding_name.clone(), (expr, ty));
                }
            }

            let outputs = graph
                .body
                .iter()
                .map(|expr| self.flatten_expr_scoped(&expr.kind, &local_scope))
                .collect::<syn::Result<Vec<_>>>()?;
            if outputs.len() != graph.outputs.len() {
                return Err(error(format!("graph call `{name}` output arity mismatch")));
            }
            for ((_, ty), expected_ty) in outputs.iter().zip(&graph.outputs) {
                if ty != expected_ty {
                    return Err(error(format!("graph call `{name}` output type mismatch")));
                }
            }
            Ok(outputs)
        })();
        self.inline_stack.pop();
        result
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
        if call_stops_gradient(op, &node.ty) {
            return Ok(());
        }
        match op {
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
            CallOp::Var(axis) => self.reverse_var(args, &node.ty, *axis, grad),
            CallOp::Std(axis) => {
                self.reverse_std(args, Expr::Var(node.name.clone()), &node.ty, *axis, grad)
            }
            CallOp::Ptp(axis) => self.reverse_ptp(args, &node.ty, *axis, grad),
            CallOp::Softmax(axis) => {
                self.reverse_softmax(args, Expr::Var(node.name.clone()), *axis, grad)
            }
            CallOp::Matmul => self.reverse_matmul(args, &node.ty, grad),
            CallOp::Dot => self.reverse_dot(args, grad),
            CallOp::Vecdot(axis) => self.reverse_vecdot(args, &node.ty, *axis, grad),
            CallOp::Inner => self.reverse_inner(args, &node.ty, grad),
            CallOp::Outer => self.reverse_outer(args, &node.ty, grad),
            CallOp::Trace(axes) => self.reverse_trace(args, &node.ty, *axes, grad),
            CallOp::Diagonal(axes) => self.reverse_diagonal(args, *axes, grad),
            CallOp::Reshape(_) | CallOp::Squeeze(_) | CallOp::Unsqueeze(_) => {
                self.add_reshape_grad_to_arg(&args[0], grad)
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
            CallOp::Tile(multiples) => self.reverse_tile(args, &node.ty, multiples, grad),
            CallOp::Repeat { axis, count } => self.reverse_repeat(args, *axis, *count, grad),
            CallOp::Gather { axis, .. } => self.reverse_gather(args, &node.ty, *axis, grad),
            CallOp::TakeAlongAxis { axis } => {
                self.reverse_take_along_axis(args, &node.ty, *axis, grad)
            }
            CallOp::AvgPool2d(options) => self.reverse_avg_pool2d(args, options, grad),
            CallOp::MaxPool2d(options) => {
                self.reverse_max_pool2d(args, Expr::Var(node.name.clone()), &node.ty, options, grad)
            }
            CallOp::Conv2d(options) => self.reverse_conv2d(args, &node.ty, options, grad),
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
            _ => Err(error(format!(
                "autograd MVP does not support differentiating floating-point `{}`",
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

    fn reverse_softmax(
        &mut self,
        args: &[Expr],
        output: Expr,
        axis: AxisSpec,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let reduced_ty = single_call_type(&CallOp::Sum(axis), core::slice::from_ref(&input_ty))?;
        let weighted = binary(BinaryOp::Mul, grad.clone(), output.clone());
        let summed = call(CallOp::Sum(axis), vec![weighted]);
        let summed = self.expand_reduction_grad(summed, &reduced_ty, &input_ty, axis)?;
        let input_grad = binary(BinaryOp::Mul, output, binary(BinaryOp::Sub, grad, summed));
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_var(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: AxisSpec,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if input_ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(());
        }
        let mean = call(CallOp::Mean(axis), vec![args[0].clone()]);
        let mean = self.expand_reduction_grad(mean, output_ty, &input_ty, axis)?;
        let centered = binary(BinaryOp::Sub, args[0].clone(), mean);
        let grad = self.expand_reduction_grad(grad, output_ty, &input_ty, axis)?;
        let local = binary(
            BinaryOp::Div,
            binary(BinaryOp::Mul, scalar("2.0", input_ty.elem), centered),
            scalar(
                &format!("{}.0", reduction_count(&input_ty, axis)),
                input_ty.elem,
            ),
        );
        let input_grad = binary(BinaryOp::Mul, grad, local);
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_std(
        &mut self,
        args: &[Expr],
        output: Expr,
        output_ty: &TensorType,
        axis: AxisSpec,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if input_ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(());
        }
        let mean = call(CallOp::Mean(axis), vec![args[0].clone()]);
        let mean = self.expand_reduction_grad(mean, output_ty, &input_ty, axis)?;
        let centered = binary(BinaryOp::Sub, args[0].clone(), mean);
        let grad = self.expand_reduction_grad(grad, output_ty, &input_ty, axis)?;
        let std = self.expand_reduction_grad(output, output_ty, &input_ty, axis)?;
        let count = scalar(
            &format!("{}.0", reduction_count(&input_ty, axis)),
            input_ty.elem,
        );
        let denom = binary(BinaryOp::Mul, count, std);
        let input_grad = binary(BinaryOp::Mul, grad, binary(BinaryOp::Div, centered, denom));
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_ptp(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: AxisSpec,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if input_ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(());
        }
        let max = call(CallOp::Max(axis), vec![args[0].clone()]);
        let min = call(CallOp::Min(axis), vec![args[0].clone()]);
        let max = self.expand_reduction_grad(max, output_ty, &input_ty, axis)?;
        let min = self.expand_reduction_grad(min, output_ty, &input_ty, axis)?;
        let grad = self.expand_reduction_grad(grad, output_ty, &input_ty, axis)?;
        let max_grad = mask_grad(
            call(CallOp::Equal, vec![args[0].clone(), max]),
            grad.clone(),
        );
        let min_grad = mask_grad(call(CallOp::Equal, vec![args[0].clone(), min]), neg(grad));
        self.add_grad_to_expr(&args[0], binary(BinaryOp::Add, max_grad, min_grad))
    }

    fn reverse_dot(&mut self, args: &[Expr], grad: Expr) -> syn::Result<()> {
        self.add_grad_to_expr(
            &args[0],
            binary(BinaryOp::Mul, grad.clone(), args[1].clone()),
        )?;
        self.add_grad_to_expr(&args[1], binary(BinaryOp::Mul, grad, args[0].clone()))
    }

    fn reverse_vecdot(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: Option<usize>,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let axis = AxisSpec::One(axis.unwrap_or(lhs_ty.rank() - 1));
        let grad = self.expand_reduction_grad(grad, output_ty, &lhs_ty, axis)?;
        self.add_grad_to_expr(
            &args[0],
            binary(BinaryOp::Mul, grad.clone(), args[1].clone()),
        )?;
        self.add_grad_to_expr(&args[1], binary(BinaryOp::Mul, grad, args[0].clone()))
    }

    fn reverse_inner(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        if lhs_ty.rank() == 0 || rhs_ty.rank() == 0 {
            let lhs_grad = self.unbroadcast(
                binary(BinaryOp::Mul, grad.clone(), args[1].clone()),
                output_ty,
                &lhs_ty,
            )?;
            let rhs_grad = self.unbroadcast(
                binary(BinaryOp::Mul, grad, args[0].clone()),
                output_ty,
                &rhs_ty,
            )?;
            self.add_grad_to_expr(&args[0], lhs_grad)?;
            return self.add_grad_to_expr(&args[1], rhs_grad);
        }

        let k = lhs_ty.shape[lhs_ty.rank() - 1];
        let lhs_prefix = element_count_slice(&lhs_ty.shape[..lhs_ty.rank() - 1]);
        let rhs_prefix = element_count_slice(&rhs_ty.shape[..rhs_ty.rank() - 1]);
        let matrix_elem = lhs_ty.elem;
        let lhs_matrix_ty = tensor(matrix_elem, vec![lhs_prefix, k]);
        let rhs_matrix_ty = tensor(matrix_elem, vec![rhs_prefix, k]);
        let grad_matrix_ty = tensor(matrix_elem, vec![lhs_prefix, rhs_prefix]);
        let lhs = call(
            CallOp::Reshape(lhs_matrix_ty.clone()),
            vec![args[0].clone()],
        );
        let rhs = call(
            CallOp::Reshape(rhs_matrix_ty.clone()),
            vec![args[1].clone()],
        );
        let grad = reshape_if_needed(grad, output_ty, &grad_matrix_ty);

        let lhs_grad = call(CallOp::Matmul, vec![grad.clone(), rhs]);
        let lhs_grad = call(CallOp::Reshape(lhs_ty.clone()), vec![lhs_grad]);
        self.add_grad_to_expr(&args[0], lhs_grad)?;

        let grad_t = call(CallOp::Transpose(Vec::new()), vec![grad]);
        let rhs_grad = call(CallOp::Matmul, vec![grad_t, lhs]);
        let rhs_grad = call(CallOp::Reshape(rhs_ty), vec![rhs_grad]);
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_outer(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let lhs_len = element_count_slice(&lhs_ty.shape);
        let rhs_len = element_count_slice(&rhs_ty.shape);
        let elem = lhs_ty.elem;
        let lhs_row_ty = tensor(elem, vec![1, lhs_len]);
        let rhs_col_ty = tensor(elem, vec![rhs_len, 1]);
        let grad_ty = tensor(elem, vec![lhs_len, rhs_len]);
        let lhs_row = call(CallOp::Reshape(lhs_row_ty), vec![args[0].clone()]);
        let rhs_col = call(CallOp::Reshape(rhs_col_ty), vec![args[1].clone()]);
        let grad = reshape_if_needed(grad, output_ty, &grad_ty);

        let lhs_grad = call(CallOp::Matmul, vec![grad.clone(), rhs_col]);
        let lhs_grad = call(CallOp::Reshape(lhs_ty), vec![lhs_grad]);
        self.add_grad_to_expr(&args[0], lhs_grad)?;

        let rhs_grad = call(CallOp::Matmul, vec![lhs_row, grad]);
        let rhs_grad = call(CallOp::Reshape(rhs_ty), vec![rhs_grad]);
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_trace(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axes: Option<[usize; 2]>,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let axes = default_axis_pair(&input_ty, axes)?;
        let compact_ty = diagonal_compact_type(&input_ty, axes);
        let mut grad_shape = output_ty.shape.clone();
        grad_shape.push(1);
        grad_shape.push(1);
        let grad = call(
            CallOp::Reshape(tensor(input_ty.elem, grad_shape)),
            vec![grad],
        );
        let grad = call(CallOp::Broadcast(compact_ty.clone()), vec![grad]);
        let eye = diagonal_eye(&input_ty, axes, &compact_ty);
        let compact = binary(BinaryOp::Mul, grad, eye);
        self.add_grad_to_expr(&args[0], permute_compact_diagonal(compact, &input_ty, axes))
    }

    fn reverse_diagonal(
        &mut self,
        args: &[Expr],
        axes: Option<[usize; 2]>,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let axes = default_axis_pair(&input_ty, axes)?;
        let compact_ty = diagonal_compact_type(&input_ty, axes);
        let mut grad_shape = remove_axes_shape(&input_ty.shape, axes);
        grad_shape.push(input_ty.shape[axes[0]]);
        grad_shape.push(1);
        let grad = call(
            CallOp::Reshape(tensor(input_ty.elem, grad_shape)),
            vec![grad],
        );
        let grad = call(CallOp::Broadcast(compact_ty.clone()), vec![grad]);
        let eye = diagonal_eye(&input_ty, axes, &compact_ty);
        let compact = binary(BinaryOp::Mul, grad, eye);
        self.add_grad_to_expr(&args[0], permute_compact_diagonal(compact, &input_ty, axes))
    }

    fn reverse_tile(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        multiples: &[usize],
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if multiples.contains(&0) {
            return self.add_grad_to_expr(&args[0], zeros_like(args[0].clone()));
        }
        let mut current = grad;
        let mut current_ty = output_ty.clone();
        for (axis, multiple) in multiples.iter().copied().enumerate().rev() {
            if multiple <= 1 {
                continue;
            }
            let mut slice_ty = current_ty.clone();
            slice_ty.shape[axis] = input_ty.shape[axis];
            let mut acc = None;
            for index in 0..multiple {
                let mut starts = vec![0; current_ty.rank()];
                starts[axis] = index * input_ty.shape[axis];
                let slice = call(
                    CallOp::Slice {
                        target: slice_ty.clone(),
                        starts,
                    },
                    vec![current.clone()],
                );
                acc = Some(match acc {
                    Some(existing) => binary(BinaryOp::Add, existing, slice),
                    None => slice,
                });
            }
            current = acc.ok_or_else(|| error("internal autograd error: empty tile gradient"))?;
            current_ty = slice_ty;
        }
        self.add_grad_to_expr(&args[0], current)
    }

    fn reverse_repeat(
        &mut self,
        args: &[Expr],
        axis: usize,
        count: usize,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if count == 0 || input_ty.shape[axis] == 0 {
            return self.add_grad_to_expr(&args[0], zeros_like(args[0].clone()));
        }
        if count == 1 {
            return self.add_grad_to_expr(&args[0], grad);
        }
        let mut expanded_shape = input_ty.shape.clone();
        expanded_shape.insert(axis + 1, count);
        if expanded_shape.len() > 6 {
            return Err(error(
                "autograd MVP does not support repeat gradients that require rank above 6",
            ));
        }
        let expanded_ty = tensor(input_ty.elem, expanded_shape);
        let expanded = call(CallOp::Reshape(expanded_ty), vec![grad]);
        let result = call(CallOp::Sum(AxisSpec::One(axis + 1)), vec![expanded]);
        self.add_grad_to_expr(&args[0], result)
    }

    fn reverse_gather(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: usize,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let indices_ty = self.expr_ty(&args[1])?;
        let reduce_axes = (axis..axis + indices_ty.rank()).collect::<Vec<_>>();
        let input_grad = indexed_axis_grad(
            &args[0],
            &args[1],
            &indices_ty,
            output_ty,
            &input_ty,
            axis,
            &reduce_axes,
            grad,
        )?;
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_take_along_axis(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: usize,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let indices_ty = self.expr_ty(&args[1])?;
        let input_grad = indexed_axis_grad(
            &args[0],
            &args[1],
            &indices_ty,
            output_ty,
            &input_ty,
            axis,
            &[axis],
            grad,
        )?;
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_avg_pool2d(
        &mut self,
        args: &[Expr],
        options: &Pool2dOptions,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let output_ty = single_call_type(
            &CallOp::AvgPool2d(options.clone()),
            core::slice::from_ref(&input_ty),
        )?;
        let unit_ty = pool_unit_type(&input_ty);
        let scale = scalar(
            &format!("{}.0", options.kernel[0] * options.kernel[1]),
            input_ty.elem,
        );
        let mut acc = None;
        for oh in 0..output_ty.shape[1] {
            for ow in 0..output_ty.shape[2] {
                let grad_unit = pool_output_unit(grad.clone(), &output_ty, oh, ow);
                let grad_unit = binary(BinaryOp::Div, grad_unit, scale.clone());
                for kh in 0..options.kernel[0] {
                    for kw in 0..options.kernel[1] {
                        if let Some((ih, iw)) = pool_input_position(options, oh, ow, kh, kw) {
                            if ih >= input_ty.shape[1] || iw >= input_ty.shape[2] {
                                continue;
                            }
                            let padded = pad_pool_unit_to_input(
                                grad_unit.clone(),
                                &unit_ty,
                                &input_ty,
                                ih,
                                iw,
                            );
                            push_sum(&mut acc, padded);
                        }
                    }
                }
            }
        }
        self.add_grad_to_expr(&args[0], acc.unwrap_or_else(|| zeros_like(args[0].clone())))
    }

    fn reverse_max_pool2d(
        &mut self,
        args: &[Expr],
        output: Expr,
        output_ty: &TensorType,
        options: &Pool2dOptions,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let unit_ty = pool_unit_type(&input_ty);
        let mut acc = None;
        for oh in 0..output_ty.shape[1] {
            for ow in 0..output_ty.shape[2] {
                let grad_unit = pool_output_unit(grad.clone(), output_ty, oh, ow);
                let output_unit = pool_output_unit(output.clone(), output_ty, oh, ow);
                for kh in 0..options.kernel[0] {
                    for kw in 0..options.kernel[1] {
                        if let Some((ih, iw)) = pool_input_position(options, oh, ow, kh, kw) {
                            if ih >= input_ty.shape[1] || iw >= input_ty.shape[2] {
                                continue;
                            }
                            let input_unit = call(
                                CallOp::Slice {
                                    target: unit_ty.clone(),
                                    starts: vec![0, ih, iw, 0],
                                },
                                vec![args[0].clone()],
                            );
                            let masked = mask_grad(
                                call(CallOp::Equal, vec![input_unit, output_unit.clone()]),
                                grad_unit.clone(),
                            );
                            let padded =
                                pad_pool_unit_to_input(masked, &unit_ty, &input_ty, ih, iw);
                            push_sum(&mut acc, padded);
                        }
                    }
                }
            }
        }
        self.add_grad_to_expr(&args[0], acc.unwrap_or_else(|| zeros_like(args[0].clone())))
    }

    fn reverse_conv2d(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        options: &Conv2dOptions,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let kernel_ty = self.expr_ty(&args[1])?;
        let groups = options.groups;
        let input_channels_per_group = input_ty.shape[3] / groups;
        let output_channels_per_group = kernel_ty.shape[3] / groups;

        let mut input_acc = None;
        let mut kernel_acc = None;
        for group in 0..groups {
            let output_channel_start = group * output_channels_per_group;
            let input_channel_start = group * input_channels_per_group;
            for kh in 0..kernel_ty.shape[0] {
                for kw in 0..kernel_ty.shape[1] {
                    for oh in 0..output_ty.shape[1] {
                        for ow in 0..output_ty.shape[2] {
                            let Some((ih, iw)) = conv_input_position(options, oh, ow, kh, kw)
                            else {
                                continue;
                            };
                            if ih >= input_ty.shape[1] || iw >= input_ty.shape[2] {
                                continue;
                            }
                            let grad_block = conv_output_channel_block(
                                grad.clone(),
                                output_ty,
                                oh,
                                ow,
                                output_channel_start,
                                output_channels_per_group,
                            );
                            for local_ic in 0..input_channels_per_group {
                                let input_channel = input_channel_start + local_ic;
                                let input_unit = conv_input_unit(
                                    args[0].clone(),
                                    &input_ty,
                                    ih,
                                    iw,
                                    input_channel,
                                );
                                let kernel_block = conv_kernel_output_block(
                                    args[1].clone(),
                                    &kernel_ty,
                                    kh,
                                    kw,
                                    local_ic,
                                    output_channel_start,
                                    output_channels_per_group,
                                );

                                let input_piece =
                                    binary(BinaryOp::Mul, grad_block.clone(), kernel_block);
                                let input_piece =
                                    call(CallOp::Sum(AxisSpec::One(3)), vec![input_piece]);
                                let input_piece = call(
                                    CallOp::Reshape(conv_input_unit_type(&input_ty)),
                                    vec![input_piece],
                                );
                                let input_piece = call(
                                    CallOp::Pad {
                                        target: input_ty.clone(),
                                        lows: vec![0, ih, iw, input_channel],
                                    },
                                    vec![input_piece],
                                );
                                push_sum(&mut input_acc, input_piece);

                                let kernel_piece =
                                    binary(BinaryOp::Mul, input_unit, grad_block.clone());
                                let kernel_piece =
                                    call(CallOp::Sum(AxisSpec::One(2)), vec![kernel_piece]);
                                let kernel_piece =
                                    call(CallOp::Sum(AxisSpec::One(1)), vec![kernel_piece]);
                                let kernel_piece =
                                    call(CallOp::Sum(AxisSpec::One(0)), vec![kernel_piece]);
                                let kernel_unit_ty = tensor(
                                    kernel_ty.elem,
                                    vec![1, 1, 1, output_channels_per_group],
                                );
                                let kernel_piece = call(
                                    CallOp::Reshape(kernel_unit_ty.clone()),
                                    vec![kernel_piece],
                                );
                                let kernel_piece = call(
                                    CallOp::Pad {
                                        target: kernel_ty.clone(),
                                        lows: vec![kh, kw, local_ic, output_channel_start],
                                    },
                                    vec![kernel_piece],
                                );
                                push_sum(&mut kernel_acc, kernel_piece);
                            }
                        }
                    }
                }
            }
        }

        self.add_grad_to_expr(
            &args[0],
            input_acc.unwrap_or_else(|| zeros_like(args[0].clone())),
        )?;
        self.add_grad_to_expr(
            &args[1],
            kernel_acc.unwrap_or_else(|| zeros_like(args[1].clone())),
        )
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

    fn add_reshape_grad_to_arg(&mut self, arg: &Expr, grad: Expr) -> syn::Result<()> {
        let input_ty = self.expr_ty(arg)?;
        self.add_grad_to_expr(arg, call(CallOp::Reshape(input_ty), vec![grad]))
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
        let Some(ty) = self.env.get(name) else {
            return Err(error(format!("unknown gradient target `{name}`")));
        };
        if !ty.elem.is_float() {
            return Ok(());
        }
        let grad = if let Some(existing) = self.grads.remove(name) {
            binary(BinaryOp::Add, existing, grad)
        } else {
            grad
        };
        self.grads.insert(name.into(), grad);
        Ok(())
    }

    fn gradient_output_for_input(&mut self, input: &Input) -> syn::Result<Expr> {
        let grad = self
            .grads
            .remove(&input.name)
            .unwrap_or_else(|| zeros_like(Expr::Var(input.name.clone())));
        self.bind_auto(grad, input.ty.clone())
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

fn call_stops_gradient(op: &CallOp, output_ty: &TensorType) -> bool {
    !output_ty.elem.is_float()
        || matches!(
            op,
            CallOp::Detach
                | CallOp::Ceil
                | CallOp::Floor
                | CallOp::Rint
                | CallOp::Round
                | CallOp::All(_)
                | CallOp::Any(_)
                | CallOp::Argmax(_)
                | CallOp::Argmin(_)
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
                | CallOp::ZerosLike
        )
}

fn single_call_type(op: &CallOp, args: &[TensorType]) -> syn::Result<TensorType> {
    let results = infer_call_results(op, args)?;
    let [ty]: [TensorType; 1] = results.try_into().map_err(|_| {
        error(format!(
            "internal autograd error: `{}` is not single-output",
            op.name()
        ))
    })?;
    Ok(ty)
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
    single_call_type(&CallOp::Matmul, &[lhs_ty.clone(), rhs_ty.clone()])
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

fn tensor(elem: ElementType, shape: Vec<usize>) -> TensorType {
    TensorType { elem, shape }
}

fn element_count_slice(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn indexed_axis_grad(
    input: &Expr,
    indices: &Expr,
    indices_ty: &TensorType,
    output_ty: &TensorType,
    input_ty: &TensorType,
    axis: usize,
    reduce_axes: &[usize],
    grad: Expr,
) -> syn::Result<Expr> {
    if input_ty.shape[axis] == 0 {
        return Ok(zeros_like(input.clone()));
    }
    let mut pieces = Vec::with_capacity(input_ty.shape[axis]);
    for index in 0..input_ty.shape[axis] {
        let positive = call(
            CallOp::Equal,
            vec![indices.clone(), scalar(&index.to_string(), indices_ty.elem)],
        );
        let negative = call(
            CallOp::Equal,
            vec![
                indices.clone(),
                scalar(
                    &format!("{}", index as isize - input_ty.shape[axis] as isize),
                    indices_ty.elem,
                ),
            ],
        );
        let condition = call(CallOp::LogicalOr, vec![positive, negative]);
        let condition = if indices_ty.shape == output_ty.shape {
            condition
        } else {
            reshape_index_condition(condition, indices_ty, output_ty, axis)?
        };
        let mut piece = mask_grad(condition, grad.clone());
        for reduce_axis in reduce_axes.iter().copied().rev() {
            piece = call(CallOp::Sum(AxisSpec::One(reduce_axis)), vec![piece]);
        }
        let mut unit_ty = input_ty.clone();
        unit_ty.shape[axis] = 1;
        piece = call(CallOp::Reshape(unit_ty.clone()), vec![piece]);
        pieces.push(piece);
    }
    concat_all(pieces, axis).ok_or_else(|| error("internal autograd error: empty index gradient"))
}

fn reshape_index_condition(
    condition: Expr,
    indices_ty: &TensorType,
    output_ty: &TensorType,
    axis: usize,
) -> syn::Result<Expr> {
    let mut mask_shape = vec![1; output_ty.rank()];
    for (offset, dim) in indices_ty.shape.iter().copied().enumerate() {
        mask_shape[axis + offset] = dim;
    }
    let mask_ty = tensor(ElementType::Bool, mask_shape);
    let condition = reshape_if_needed(
        condition,
        &tensor(ElementType::Bool, indices_ty.shape.clone()),
        &mask_ty,
    );
    let output_bool_ty = tensor(ElementType::Bool, output_ty.shape.clone());
    Ok(if mask_ty == output_bool_ty {
        condition
    } else {
        call(CallOp::Broadcast(output_bool_ty), vec![condition])
    })
}

fn pool_unit_type(input_ty: &TensorType) -> TensorType {
    tensor(
        input_ty.elem,
        vec![input_ty.shape[0], 1, 1, input_ty.shape[3]],
    )
}

fn pool_output_unit(output: Expr, output_ty: &TensorType, oh: usize, ow: usize) -> Expr {
    let unit_ty = tensor(
        output_ty.elem,
        vec![output_ty.shape[0], 1, 1, output_ty.shape[3]],
    );
    call(
        CallOp::Slice {
            target: unit_ty,
            starts: vec![0, oh, ow, 0],
        },
        vec![output],
    )
}

fn pool_input_position(
    options: &Pool2dOptions,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Option<(usize, usize)> {
    let padded_h = output_h * options.stride[0] + kernel_h * options.dilation[0];
    let padded_w = output_w * options.stride[1] + kernel_w * options.dilation[1];
    let input_h = padded_h.checked_sub(options.padding.top)?;
    let input_w = padded_w.checked_sub(options.padding.left)?;
    Some((input_h, input_w))
}

fn pad_pool_unit_to_input(
    unit: Expr,
    unit_ty: &TensorType,
    input_ty: &TensorType,
    input_h: usize,
    input_w: usize,
) -> Expr {
    debug_assert_eq!(
        unit_ty.shape,
        vec![input_ty.shape[0], 1, 1, input_ty.shape[3]]
    );
    call(
        CallOp::Pad {
            target: input_ty.clone(),
            lows: vec![0, input_h, input_w, 0],
        },
        vec![unit],
    )
}

fn conv_input_position(
    options: &Conv2dOptions,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Option<(usize, usize)> {
    let padded_h = output_h * options.stride[0] + kernel_h * options.dilation[0];
    let padded_w = output_w * options.stride[1] + kernel_w * options.dilation[1];
    let input_h = padded_h.checked_sub(options.padding.top)?;
    let input_w = padded_w.checked_sub(options.padding.left)?;
    Some((input_h, input_w))
}

fn conv_input_unit_type(input_ty: &TensorType) -> TensorType {
    tensor(input_ty.elem, vec![input_ty.shape[0], 1, 1, 1])
}

fn conv_input_unit(input: Expr, input_ty: &TensorType, h: usize, w: usize, channel: usize) -> Expr {
    call(
        CallOp::Slice {
            target: conv_input_unit_type(input_ty),
            starts: vec![0, h, w, channel],
        },
        vec![input],
    )
}

fn conv_output_channel_block(
    grad: Expr,
    output_ty: &TensorType,
    output_h: usize,
    output_w: usize,
    output_channel_start: usize,
    output_channel_count: usize,
) -> Expr {
    let ty = tensor(
        output_ty.elem,
        vec![output_ty.shape[0], 1, 1, output_channel_count],
    );
    call(
        CallOp::Slice {
            target: ty,
            starts: vec![0, output_h, output_w, output_channel_start],
        },
        vec![grad],
    )
}

fn conv_kernel_output_block(
    kernel: Expr,
    kernel_ty: &TensorType,
    kernel_h: usize,
    kernel_w: usize,
    input_channel: usize,
    output_channel_start: usize,
    output_channel_count: usize,
) -> Expr {
    let ty = tensor(kernel_ty.elem, vec![1, 1, 1, output_channel_count]);
    call(
        CallOp::Slice {
            target: ty,
            starts: vec![kernel_h, kernel_w, input_channel, output_channel_start],
        },
        vec![kernel],
    )
}

fn push_sum(acc: &mut Option<Expr>, expr: Expr) {
    *acc = Some(match acc.take() {
        Some(existing) => binary(BinaryOp::Add, existing, expr),
        None => expr,
    });
}

fn concat_all(pieces: Vec<Expr>, axis: usize) -> Option<Expr> {
    let mut iter = pieces.into_iter();
    let mut acc = iter.next()?;
    for piece in iter {
        acc = call(CallOp::Concat(axis), vec![acc, piece]);
    }
    Some(acc)
}

fn default_axis_pair(input_ty: &TensorType, axes: Option<[usize; 2]>) -> syn::Result<[usize; 2]> {
    if input_ty.rank() < 2 {
        return Err(error(
            "internal autograd error: diagonal op input rank was below 2",
        ));
    }
    let axes = axes.unwrap_or([input_ty.rank() - 2, input_ty.rank() - 1]);
    if axes[0] >= input_ty.rank() || axes[1] >= input_ty.rank() || axes[0] == axes[1] {
        return Err(error(format!(
            "internal autograd error: invalid diagonal axes {:?} for {:?}",
            axes, input_ty
        )));
    }
    Ok(axes)
}

fn remove_axes_shape(shape: &[usize], axes: [usize; 2]) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| (axis != axes[0] && axis != axes[1]).then_some(*dim))
        .collect()
}

fn diagonal_compact_type(input_ty: &TensorType, axes: [usize; 2]) -> TensorType {
    let mut shape = remove_axes_shape(&input_ty.shape, axes);
    shape.push(input_ty.shape[axes[0]]);
    shape.push(input_ty.shape[axes[1]]);
    tensor(input_ty.elem, shape)
}

fn diagonal_eye(input_ty: &TensorType, axes: [usize; 2], compact_ty: &TensorType) -> Expr {
    let eye_ty = tensor(
        input_ty.elem,
        vec![input_ty.shape[axes[0]], input_ty.shape[axes[1]]],
    );
    call(
        CallOp::Broadcast(compact_ty.clone()),
        vec![call(CallOp::Eye(eye_ty), Vec::new())],
    )
}

fn permute_compact_diagonal(compact: Expr, input_ty: &TensorType, axes: [usize; 2]) -> Expr {
    let non_axes = (0..input_ty.rank())
        .filter(|axis| *axis != axes[0] && *axis != axes[1])
        .collect::<Vec<_>>();
    let compact_axes = (0..input_ty.rank())
        .map(|axis| {
            if axis == axes[0] {
                non_axes.len()
            } else if axis == axes[1] {
                non_axes.len() + 1
            } else {
                non_axes
                    .iter()
                    .position(|non_axis| *non_axis == axis)
                    .expect("non-diagonal axis missing from compact axis map")
            }
        })
        .collect::<Vec<_>>();
    call(CallOp::PermuteDims(compact_axes), vec![compact])
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
