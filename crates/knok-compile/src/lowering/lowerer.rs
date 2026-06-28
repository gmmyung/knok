use std::collections::BTreeMap;

use knok_core::{
    static_arange_literals, static_eye_literals, static_linspace_literals, BinaryOp, CallOp, Expr,
    TensorType, TypedGraph, UnaryOp,
};

use crate::common::mlir_result_types;

pub fn lower_to_mlir(graph: &TypedGraph) -> anyhow::Result<String> {
    lower_to_mlir_with_registry(graph, &BTreeMap::new())
}

pub fn lower_to_mlir_with_registry(
    graph: &TypedGraph,
    graphs: &BTreeMap<String, TypedGraph>,
) -> anyhow::Result<String> {
    let mut lowerer = Lowerer::new(graph, graphs);
    lowerer.lower()
}

pub(super) struct Lowerer<'a> {
    pub(super) graph: &'a TypedGraph,
    pub(super) graphs: &'a BTreeMap<String, TypedGraph>,
    pub(super) call_stack: Vec<String>,
    pub(super) tuple_scope_stack: Vec<u64>,
    pub(super) next_tuple_scope: u64,
    pub(super) next_value: usize,
    pub(super) lines: Vec<String>,
    pub(super) values: BTreeMap<String, Value>,
    pub(super) node_values: BTreeMap<(u64, u64), Value>,
    pub(super) tuple_values: BTreeMap<(u64, u64), Vec<Value>>,
}

#[derive(Clone)]
pub(super) struct Value {
    pub(super) name: String,
    pub(super) ty: TensorType,
    pub(super) kind: ValueKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ValueKind {
    Scalar,
    Tensor,
}

impl Value {
    pub(super) fn scalar(name: String, elem: knok_core::ElementType) -> Self {
        Self {
            name,
            ty: TensorType {
                elem,
                shape: Vec::new(),
            },
            kind: ValueKind::Scalar,
        }
    }

    pub(super) fn tensor(name: String, ty: TensorType) -> Self {
        Self {
            name,
            ty,
            kind: ValueKind::Tensor,
        }
    }

    pub(super) fn mlir_type(&self) -> String {
        match self.kind {
            ValueKind::Scalar => self.ty.elem.mlir_type().to_string(),
            ValueKind::Tensor => self.ty.mlir_type(),
        }
    }
}

impl<'a> Lowerer<'a> {
    fn new(graph: &'a TypedGraph, graphs: &'a BTreeMap<String, TypedGraph>) -> Self {
        Self {
            graph,
            graphs,
            call_stack: vec![graph.name.clone()],
            tuple_scope_stack: vec![0],
            next_tuple_scope: 1,
            next_value: 0,
            lines: Vec::new(),
            values: BTreeMap::new(),
            node_values: BTreeMap::new(),
            tuple_values: BTreeMap::new(),
        }
    }

    fn lower(&mut self) -> anyhow::Result<String> {
        let arg_list = self
            .graph
            .inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                self.values.insert(
                    input.name.clone(),
                    Value::tensor(format!("%arg{index}"), input.ty.clone()),
                );
                format!("%arg{index}: {}", input.ty.mlir_type())
            })
            .collect::<Vec<_>>()
            .join(", ");
        for binding in &self.graph.lets {
            let values = self.lower_let_values(&binding.value.kind)?;
            self.bind_values(&binding.names, values, None)?;
        }
        let body = self.lower_body_outputs(&self.graph.body, &self.graph.outputs)?;
        let return_values = body
            .iter()
            .map(|value| value.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let return_types = body
            .iter()
            .map(|value| value.ty.mlir_type())
            .collect::<Vec<_>>()
            .join(", ");
        self.lines
            .push(format!("    return {} : {}", return_values, return_types));

        let mut mlir = String::new();
        mlir.push_str("module @knok {\n");
        mlir.push_str(&format!(
            "  func.func @{}({}) -> {} {{\n",
            self.graph.name,
            arg_list,
            mlir_result_types(&self.graph.outputs)
        ));
        for line in &self.lines {
            mlir.push_str(line);
            mlir.push('\n');
        }
        mlir.push_str("  }\n");
        mlir.push_str("}\n");
        Ok(mlir)
    }

    fn lower_expr(&mut self, expr: &Expr) -> anyhow::Result<Value> {
        match expr {
            Expr::Var(name) => self
                .values
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("unknown value `{name}` during lowering")),
            Expr::Const { value, elem } => self.constant(value, *elem),
            Expr::Unary { op, value } => match op {
                UnaryOp::Neg => {
                    let value = self.lower_expr(value)?;
                    let zero = self.zero_like(&value.ty)?;
                    self.binary_value(BinaryOp::Sub, zero, value)
                }
            },
            Expr::Binary { op, lhs, rhs } => {
                let lhs = self.lower_expr(lhs)?;
                let rhs = self.lower_expr(rhs)?;
                self.binary_value(*op, lhs, rhs)
            }
            Expr::Node { node_id, value } => self.lower_node(*node_id, value),
            Expr::TupleGet {
                tuple_id,
                value,
                index,
            } => self.lower_tuple_get(*tuple_id, value, *index),
            Expr::Call { op, args } => {
                let values = self.lower_call_values(op, args)?;
                if values.len() != 1 {
                    anyhow::bail!(
                        "{op:?} produces {} values and cannot be used as a tensor expression",
                        values.len()
                    );
                }
                Ok(values
                    .into_iter()
                    .next()
                    .expect("single-result call produced no value"))
            }
        }
    }

    fn lower_call(&mut self, op: &CallOp, args: &[Expr]) -> anyhow::Result<Value> {
        match op {
            CallOp::Abs => {
                let input = self.lower_expr(&args[0])?;
                let op_name = if input.ty.elem.is_float() {
                    "math.absf"
                } else {
                    "math.absi"
                };
                self.emit_unary(op_name, input)
            }
            CallOp::All(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.all(input, *axis)
            }
            CallOp::Argmax(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.argmax(input, *axis)
            }
            CallOp::Argmin(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.argmin(input, *axis)
            }
            CallOp::Any(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.any(input, *axis)
            }
            CallOp::Clip => {
                let value = self.lower_expr(&args[0])?;
                let min = self.lower_expr(&args[1])?;
                let max = self.lower_expr(&args[2])?;
                let clipped_high = self.minimum(value, max)?;
                self.maximum(clipped_high, min)
            }
            CallOp::Ceil => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.ceil", value)
            }
            CallOp::Concat(axis) => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.concat(lhs, rhs, *axis)
            }
            CallOp::Conv2d(options) => {
                let input = self.lower_expr(&args[0])?;
                let kernel = self.lower_expr(&args[1])?;
                self.conv2d(input, kernel, options)
            }
            CallOp::Diagonal(axes) => {
                let input = self.lower_expr(&args[0])?;
                self.diagonal(input, *axes)
            }
            CallOp::Dot => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.dot(lhs, rhs)
            }
            CallOp::Arange(ty) => {
                let values =
                    static_arange_literals(ty, args).map_err(|message| anyhow::anyhow!(message))?;
                self.dense_constant(ty, &values)
            }
            CallOp::Eye(ty) => {
                let values = static_eye_literals(ty).map_err(|message| anyhow::anyhow!(message))?;
                self.dense_constant(ty, &values)
            }
            CallOp::Cos => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.cos", value)
            }
            CallOp::Exp => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.exp", value)
            }
            CallOp::FullLike => {
                let input = self.lower_expr(&args[0])?;
                let fill = self.lower_expr(&args[1])?;
                self.splat(fill, &input.ty)
            }
            CallOp::Exp2 => {
                let value = self.lower_expr(&args[0])?;
                let ln2 = self.constant("0.6931471805599453", value.ty.elem)?;
                let scaled = self.binary_value(BinaryOp::Mul, value, ln2)?;
                self.emit_unary("math.exp", scaled)
            }
            CallOp::ExpM1 => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.expm1", value)
            }
            CallOp::Floor => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.floor", value)
            }
            CallOp::Greater => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.comparison("ogt", "sgt", lhs, rhs)
            }
            CallOp::GreaterEqual => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.comparison("oge", "sge", lhs, rhs)
            }
            CallOp::Gather { target, axis } => {
                let input = self.lower_expr(&args[0])?;
                let indices = self.lower_expr(&args[1])?;
                self.gather(input, indices, *axis, target)
            }
            CallOp::IsNan => {
                let value = self.lower_expr(&args[0])?;
                self.isnan(value)
            }
            CallOp::Less => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.comparison("olt", "slt", lhs, rhs)
            }
            CallOp::LessEqual => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.comparison("ole", "sle", lhs, rhs)
            }
            CallOp::Linspace(ty) => {
                let values = static_linspace_literals(ty, args)
                    .map_err(|message| anyhow::anyhow!(message))?;
                self.dense_constant(ty, &values)
            }
            CallOp::Log => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.log", value)
            }
            CallOp::Log1P => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.log1p", value)
            }
            CallOp::Log2 => {
                let value = self.lower_expr(&args[0])?;
                let elem = value.ty.elem;
                let log = self.emit_unary("math.log", value)?;
                let ln2 = self.constant("0.6931471805599453", elem)?;
                self.binary_value(BinaryOp::Div, log, ln2)
            }
            CallOp::Log10 => {
                let value = self.lower_expr(&args[0])?;
                let elem = value.ty.elem;
                let log = self.emit_unary("math.log", value)?;
                let ln10 = self.constant("2.302585092994046", elem)?;
                self.binary_value(BinaryOp::Div, log, ln10)
            }
            CallOp::LogicalAnd => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.logical_binary("arith.andi", lhs, rhs)
            }
            CallOp::LogicalNot => {
                let value = self.lower_expr(&args[0])?;
                self.logical_not(value)
            }
            CallOp::LogicalOr => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.logical_binary("arith.ori", lhs, rhs)
            }
            CallOp::LogicalXor => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.logical_binary("arith.xori", lhs, rhs)
            }
            CallOp::Inner => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.inner(lhs, rhs)
            }
            CallOp::Relu => {
                let value = self.lower_expr(&args[0])?;
                let zero = self.zero_like(&value.ty)?;
                self.emit_binary("arith.maximumf", zero, value)
            }
            CallOp::Mean(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.mean(input, *axis)
            }
            CallOp::Max(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.max(input, *axis)
            }
            CallOp::Matmul => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.matmul(lhs, rhs)
            }
            CallOp::Min(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.min(input, *axis)
            }
            CallOp::Minimum => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.minimum(lhs, rhs)
            }
            CallOp::Maximum => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.maximum(lhs, rhs)
            }
            CallOp::Equal => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.comparison("oeq", "eq", lhs, rhs)
            }
            CallOp::NotEqual => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.comparison("une", "ne", lhs, rhs)
            }
            CallOp::Outer => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.outer(lhs, rhs)
            }
            CallOp::Flip(axes) => {
                let input = self.lower_expr(&args[0])?;
                self.flip(input, axes)
            }
            CallOp::MoveAxis {
                source,
                destination,
            } => {
                let input = self.lower_expr(&args[0])?;
                self.moveaxis(input, *source, *destination)
            }
            CallOp::Pad { target, lows } => {
                let input = self.lower_expr(&args[0])?;
                self.pad(input, target, lows)
            }
            CallOp::Pow => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.emit_binary("math.powf", lhs, rhs)
            }
            CallOp::OnesLike => {
                let input = self.lower_expr(&args[0])?;
                self.one_like(&input.ty)
            }
            CallOp::Prod(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.prod(input, *axis)
            }
            CallOp::Ptp(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.ptp(input, *axis)
            }
            CallOp::Permute { target, axes } => {
                let input = self.lower_expr(&args[0])?;
                self.permute(input, target, axes)
            }
            CallOp::PermuteDims(axes) => {
                let input = self.lower_expr(&args[0])?;
                self.permute_dims(input, axes)
            }
            CallOp::Reciprocal => {
                let value = self.lower_expr(&args[0])?;
                let one = self.one_like(&value.ty)?;
                self.binary_value(BinaryOp::Div, one, value)
            }
            CallOp::Repeat { axis, count } => {
                let input = self.lower_expr(&args[0])?;
                self.repeat(input, *axis, *count)
            }
            CallOp::Sigmoid => {
                let value = self.lower_expr(&args[0])?;
                self.sigmoid(value)
            }
            CallOp::Rint | CallOp::Round => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.roundeven", value)
            }
            CallOp::Sin => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.sin", value)
            }
            CallOp::Softmax(axis) => {
                let value = self.lower_expr(&args[0])?;
                self.softmax(value, *axis)
            }
            CallOp::Sqrt => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.sqrt", value)
            }
            CallOp::Square => {
                let value = self.lower_expr(&args[0])?;
                self.binary_value(BinaryOp::Mul, value.clone(), value)
            }
            CallOp::Tan => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.tan", value)
            }
            CallOp::Tanh => {
                let value = self.lower_expr(&args[0])?;
                self.emit_unary("math.tanh", value)
            }
            CallOp::Transpose(axes) => {
                let input = self.lower_expr(&args[0])?;
                self.transpose(input, axes)
            }
            CallOp::Reshape(ty) => {
                let input = self.lower_expr(&args[0])?;
                self.reshape(input, ty)
            }
            CallOp::Roll { axis, shift } => {
                let input = self.lower_expr(&args[0])?;
                self.roll(input, *axis, *shift)
            }
            CallOp::Broadcast(ty) => {
                let input = self.lower_expr(&args[0])?;
                self.broadcast(input, ty)
            }
            CallOp::Slice { target, starts } => {
                let input = self.lower_expr(&args[0])?;
                self.slice(input, target, starts)
            }
            CallOp::Squeeze(ty) => {
                let input = self.lower_expr(&args[0])?;
                self.reshape(input, ty)
            }
            CallOp::Stack(axis) => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.stack(lhs, rhs, *axis)
            }
            CallOp::Std(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.std(input, *axis)
            }
            CallOp::SwapAxes { axis0, axis1 } => {
                let input = self.lower_expr(&args[0])?;
                self.swapaxes(input, *axis0, *axis1)
            }
            CallOp::Sum(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.sum(input, *axis)
            }
            CallOp::Split { .. } => {
                anyhow::bail!(
                    "split produces multiple values and must be lowered through tuple projections"
                )
            }
            CallOp::Take { axis, index } => {
                let input = self.lower_expr(&args[0])?;
                self.take(input, *axis, *index)
            }
            CallOp::Trace(axes) => {
                let input = self.lower_expr(&args[0])?;
                self.trace(input, *axes)
            }
            CallOp::TakeAlongAxis { axis } => {
                let input = self.lower_expr(&args[0])?;
                let indices = self.lower_expr(&args[1])?;
                self.take_along_axis(input, indices, *axis)
            }
            CallOp::Tile(multiples) => {
                let input = self.lower_expr(&args[0])?;
                self.tile(input, multiples)
            }
            CallOp::Unsqueeze(ty) => {
                let input = self.lower_expr(&args[0])?;
                self.reshape(input, ty)
            }
            CallOp::Vecdot(axis) => {
                let lhs = self.lower_expr(&args[0])?;
                let rhs = self.lower_expr(&args[1])?;
                self.vecdot(lhs, rhs, *axis)
            }
            CallOp::Var(axis) => {
                let input = self.lower_expr(&args[0])?;
                self.var(input, *axis)
            }
            CallOp::Where => {
                let condition = self.lower_expr(&args[0])?;
                let true_value = self.lower_expr(&args[1])?;
                let false_value = self.lower_expr(&args[2])?;
                self.where_select(condition, true_value, false_value)
            }
            CallOp::ZerosLike => {
                let input = self.lower_expr(&args[0])?;
                self.zero_like(&input.ty)
            }
            CallOp::Graph(name) => {
                let args = args
                    .iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                self.inline_graph(name, args)
            }
        }
    }

    fn lower_call_values(&mut self, op: &CallOp, args: &[Expr]) -> anyhow::Result<Vec<Value>> {
        match op {
            CallOp::Graph(name) => {
                let args = args
                    .iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                self.inline_graph_values(name, args)
            }
            CallOp::Split { axis, sections } => {
                let input = self.lower_expr(&args[0])?;
                self.split(input, *axis, sections)
            }
            _ => Ok(vec![self.lower_call(op, args)?]),
        }
    }

    fn inline_graph(&mut self, name: &str, args: Vec<Value>) -> anyhow::Result<Value> {
        let values = self.inline_graph_values(name, args)?;
        if values.len() != 1 {
            anyhow::bail!(
                "graph `{name}` returns {} values and cannot be inlined as a tensor expression yet",
                values.len()
            );
        }
        Ok(values
            .into_iter()
            .next()
            .expect("single-output graph call produced no values"))
    }

    fn lower_let_values(&mut self, expr: &Expr) -> anyhow::Result<Vec<Value>> {
        match expr {
            Expr::Node { value, .. } => self.lower_let_values(value),
            Expr::Call { op, args } => self.lower_call_values(op, args),
            _ => Ok(vec![self.lower_expr(expr)?]),
        }
    }

    fn current_tuple_scope(&self) -> u64 {
        *self
            .tuple_scope_stack
            .last()
            .expect("lowerer always has an active tuple scope")
    }

    fn lower_node(&mut self, node_id: u64, value: &Expr) -> anyhow::Result<Value> {
        let key = (self.current_tuple_scope(), node_id);
        if !self.node_values.contains_key(&key) {
            let value = self.lower_expr(value)?;
            self.node_values.insert(key, value);
        }
        self.node_values
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("node value was not cached"))
    }

    fn lower_tuple_get(
        &mut self,
        tuple_id: u64,
        value: &Expr,
        index: usize,
    ) -> anyhow::Result<Value> {
        let key = (self.current_tuple_scope(), tuple_id);
        if !self.tuple_values.contains_key(&key) {
            let values = self.lower_let_values(value)?;
            self.tuple_values.insert(key, values);
        }
        let values = self
            .tuple_values
            .get(&key)
            .expect("tuple values were inserted above");
        values.get(index).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "tuple projection index {index} out of bounds for {} values",
                values.len()
            )
        })
    }

    fn bind_values(
        &mut self,
        names: &[String],
        values: Vec<Value>,
        mut overwritten: Option<&mut Vec<(String, Option<Value>)>>,
    ) -> anyhow::Result<()> {
        if names.len() != values.len() {
            anyhow::bail!(
                "internal error: let binding expected {} values, lowering produced {}",
                names.len(),
                values.len()
            );
        }
        for (name, value) in names.iter().zip(values) {
            let old_value = self.values.insert(name.clone(), value);
            if let Some(overwritten) = &mut overwritten {
                overwritten.push((name.clone(), old_value));
            }
        }
        Ok(())
    }

    fn inline_graph_values(&mut self, name: &str, args: Vec<Value>) -> anyhow::Result<Vec<Value>> {
        if self.call_stack.iter().any(|candidate| candidate == name) {
            anyhow::bail!("recursive graph call `{name}` is not supported");
        }
        let graph = self
            .graphs
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown graph `{name}` during lowering"))?;
        if graph.inputs.len() != args.len() {
            anyhow::bail!(
                "graph `{name}` expects {} arguments, got {}",
                graph.inputs.len(),
                args.len()
            );
        }

        self.call_stack.push(name.to_string());
        let tuple_scope = self.next_tuple_scope;
        self.next_tuple_scope += 1;
        self.tuple_scope_stack.push(tuple_scope);
        let mut overwritten = Vec::new();
        for (input, value) in graph.inputs.iter().zip(args) {
            overwritten.push((
                input.name.clone(),
                self.values.insert(input.name.clone(), value),
            ));
        }

        let result = (|| {
            for binding in &graph.lets {
                let values = self.lower_let_values(&binding.value.kind)?;
                self.bind_values(&binding.names, values, Some(&mut overwritten))?;
            }
            self.lower_body_outputs(&graph.body, &graph.outputs)
        })();

        for (name, old_value) in overwritten.into_iter().rev() {
            if let Some(old_value) = old_value {
                self.values.insert(name, old_value);
            } else {
                self.values.remove(&name);
            }
        }
        self.node_values
            .retain(|(scope, _), _| *scope != tuple_scope);
        self.tuple_values
            .retain(|(scope, _), _| *scope != tuple_scope);
        self.tuple_scope_stack.pop();
        self.call_stack.pop();
        result
    }

    pub(super) fn fresh(&mut self) -> String {
        let name = format!("%{}", self.next_value);
        self.next_value += 1;
        name
    }

    fn lower_body_outputs(
        &mut self,
        body: &[knok_core::TypedExpr],
        outputs: &[TensorType],
    ) -> anyhow::Result<Vec<Value>> {
        if body.len() != outputs.len() {
            anyhow::bail!(
                "internal error: graph body has {} values but signature has {} outputs",
                body.len(),
                outputs.len()
            );
        }
        body.iter()
            .zip(outputs)
            .map(|(expr, output)| {
                let value = self.lower_expr(&expr.kind)?;
                self.output_value(value, output)
            })
            .collect()
    }

    fn output_value(&mut self, value: Value, output: &TensorType) -> anyhow::Result<Value> {
        if value.ty != *output {
            anyhow::bail!(
                "internal error: lowered output type {:?} does not match graph output {:?}",
                value.ty,
                output
            );
        }
        match value.kind {
            ValueKind::Tensor => Ok(value),
            ValueKind::Scalar if output.rank() == 0 => self.splat(value, output),
            ValueKind::Scalar => anyhow::bail!(
                "internal error: scalar SSA value cannot be returned as rank-{} tensor",
                output.rank()
            ),
        }
    }
}
