use std::collections::BTreeMap;

use knok_core::{BinaryOp, CallOp, Expr, TensorType, TypedGraph, UnaryOp};

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
    pub(super) next_value: usize,
    pub(super) lines: Vec<String>,
    pub(super) values: BTreeMap<String, Value>,
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
            next_value: 0,
            lines: Vec::new(),
            values: BTreeMap::new(),
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
            Expr::Call { op, args } => match op {
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
                CallOp::Exp => {
                    let value = self.lower_expr(&args[0])?;
                    self.emit_unary("math.exp", value)
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
                CallOp::Log => {
                    let value = self.lower_expr(&args[0])?;
                    self.emit_unary("math.log", value)
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
                CallOp::Matmul => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.matmul(lhs, rhs)
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
                CallOp::Pow => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.emit_binary("math.powf", lhs, rhs)
                }
                CallOp::Outer => {
                    let lhs = self.lower_expr(&args[0])?;
                    let rhs = self.lower_expr(&args[1])?;
                    self.outer(lhs, rhs)
                }
                CallOp::Permute { target, axes } => {
                    let input = self.lower_expr(&args[0])?;
                    self.permute(input, target, axes)
                }
                CallOp::Sigmoid => {
                    let value = self.lower_expr(&args[0])?;
                    self.sigmoid(value)
                }
                CallOp::Softmax(axis) => {
                    let value = self.lower_expr(&args[0])?;
                    self.softmax(value, *axis)
                }
                CallOp::Sqrt => {
                    let value = self.lower_expr(&args[0])?;
                    self.emit_unary("math.sqrt", value)
                }
                CallOp::Tanh => {
                    let value = self.lower_expr(&args[0])?;
                    self.emit_unary("math.tanh", value)
                }
                CallOp::Transpose => {
                    let input = self.lower_expr(&args[0])?;
                    self.transpose(input)
                }
                CallOp::Reshape(ty) => {
                    let input = self.lower_expr(&args[0])?;
                    self.reshape(input, ty)
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
                CallOp::Sum(axis) => {
                    let input = self.lower_expr(&args[0])?;
                    self.sum(input, *axis)
                }
                CallOp::Take { axis, index } => {
                    let input = self.lower_expr(&args[0])?;
                    self.take(input, *axis, *index)
                }
                CallOp::Trace(axes) => {
                    let input = self.lower_expr(&args[0])?;
                    self.trace(input, *axes)
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
                CallOp::Where => {
                    let condition = self.lower_expr(&args[0])?;
                    let true_value = self.lower_expr(&args[1])?;
                    let false_value = self.lower_expr(&args[2])?;
                    self.where_select(condition, true_value, false_value)
                }
                CallOp::Graph(name) => {
                    let args = args
                        .iter()
                        .map(|arg| self.lower_expr(arg))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    self.inline_graph(name, args)
                }
            },
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
            Expr::Call {
                op: CallOp::Graph(name),
                args,
            } => {
                let args = args
                    .iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                self.inline_graph_values(name, args)
            }
            _ => Ok(vec![self.lower_expr(expr)?]),
        }
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
