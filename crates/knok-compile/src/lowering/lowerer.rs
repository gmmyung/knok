use std::collections::BTreeMap;

use knok_core::{
    static_arange_literals, static_eye_literals, static_linspace_literals, BinaryOp, CallOp, Expr,
    TensorType, TypedGraph, UnaryOp,
};
use melior::{
    dialect::{func, DialectRegistry},
    ir::{
        attribute::{
            AttributeLike, DenseI32ArrayAttribute, DenseI64ArrayAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::{OperationBuilder, OperationMutLike},
        r#type::FunctionType,
        Attribute, Block, BlockLike, Identifier, Location, Module, Region, RegionLike, Type,
        Value as MlirValue,
    },
    Context,
};

pub(super) use super::value::{RawValue, Value, ValueKind};

use crate::mlir::canonicalize_and_verify;

pub fn lower_to_mlir(graph: &TypedGraph) -> anyhow::Result<String> {
    lower_to_mlir_with_registry(graph, &BTreeMap::new())
}

/// Lowers a typed graph to an MLIR module, resolving graph calls from `graphs`.
pub fn lower_to_mlir_with_registry(
    graph: &TypedGraph,
    graphs: &BTreeMap<String, TypedGraph>,
) -> anyhow::Result<String> {
    let registry = DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let lowerer = Lowerer::new(&context, graph, graphs)?;
    lowerer.lower()
}

pub(super) struct Lowerer<'a, 'c> {
    pub(super) context: &'c Context,
    pub(super) location: Location<'c>,
    pub(super) block: Block<'c>,
    pub(super) graph: &'a TypedGraph,
    pub(super) graphs: &'a BTreeMap<String, TypedGraph>,
    pub(super) call_stack: Vec<String>,
    pub(super) tuple_scope_stack: Vec<u64>,
    pub(super) next_tuple_scope: u64,
    pub(super) values: BTreeMap<String, Value>,
    pub(super) node_values: BTreeMap<(u64, u64), Value>,
    pub(super) tuple_values: BTreeMap<(u64, u64), Vec<Value>>,
}

impl<'a, 'c> Lowerer<'a, 'c> {
    fn new(
        context: &'c Context,
        graph: &'a TypedGraph,
        graphs: &'a BTreeMap<String, TypedGraph>,
    ) -> anyhow::Result<Self> {
        let location = Location::unknown(context);
        let arg_types = graph
            .inputs
            .iter()
            .map(|input| mlir_tensor_type(context, &input.ty))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let block = Block::new(
            &arg_types
                .iter()
                .copied()
                .map(|ty| (ty, location))
                .collect::<Vec<_>>(),
        );
        let mut lowerer = Self {
            context,
            location,
            block,
            graph,
            graphs,
            call_stack: vec![graph.name.clone()],
            tuple_scope_stack: vec![0],
            next_tuple_scope: 1,
            values: BTreeMap::new(),
            node_values: BTreeMap::new(),
            tuple_values: BTreeMap::new(),
        };
        for (index, input) in graph.inputs.iter().enumerate() {
            let arg = lowerer.block.argument(index)?;
            lowerer.values.insert(
                input.name.clone(),
                Value::tensor(RawValue::from_value(arg.into()), input.ty.clone()),
            );
        }
        Ok(lowerer)
    }

    fn lower(mut self) -> anyhow::Result<String> {
        for binding in &self.graph.lets {
            let values = self.lower_let_values(&binding.value.kind)?;
            self.bind_values(&binding.names, values, None)?;
        }
        let body = self.lower_body_outputs(&self.graph.body, &self.graph.outputs)?;
        let return_values = body
            .iter()
            .map(|value| value.raw.as_value())
            .collect::<Vec<_>>();
        self.block
            .append_operation(func::r#return(&return_values, self.location));

        let input_types = self
            .graph
            .inputs
            .iter()
            .map(|input| mlir_tensor_type(self.context, &input.ty))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let output_types = self
            .graph
            .outputs
            .iter()
            .map(|output| mlir_tensor_type(self.context, output))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let region = Region::new();
        region.append_block(self.block);
        let mut module = Module::new(self.location);
        module.as_operation_mut().set_attribute(
            "sym_name",
            StringAttribute::new(self.context, "knok").into(),
        );
        let function_type = FunctionType::new(self.context, &input_types, &output_types);
        module.body().append_operation(func::func(
            self.context,
            StringAttribute::new(self.context, &self.graph.name),
            TypeAttribute::new(function_type.into()),
            region,
            &[],
            self.location,
        ));
        canonicalize_and_verify(&module.as_operation().to_string())
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
            CallOp::MaxPool2d(options) => {
                let input = self.lower_expr(&args[0])?;
                self.max_pool2d(input, options)
            }
            CallOp::AvgPool2d(options) => {
                let input = self.lower_expr(&args[0])?;
                self.avg_pool2d(input, options)
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

    pub(super) fn append_op(
        &mut self,
        op_name: &str,
        operands: &[Value],
        result_ty: &TensorType,
        result_kind: ValueKind,
    ) -> anyhow::Result<Value> {
        let result_type = self.mlir_value_type(result_ty, result_kind)?;
        let operands = operands
            .iter()
            .map(|value| value.raw.as_value())
            .collect::<Vec<_>>();
        let op = OperationBuilder::new(op_name, self.location)
            .add_operands(&operands)
            .add_results(&[result_type])
            .build()?;
        let result = self.block.append_operation(op).result(0)?;
        Ok(match result_kind {
            ValueKind::Scalar => Value::scalar(RawValue::from_value(result.into()), result_ty.elem),
            ValueKind::Tensor => {
                Value::tensor(RawValue::from_value(result.into()), result_ty.clone())
            }
        })
    }

    pub(super) fn append_op_with_attrs(
        &mut self,
        op_name: &str,
        operands: &[Value],
        result_ty: &TensorType,
        result_kind: ValueKind,
        attrs: &[(String, String)],
    ) -> anyhow::Result<Value> {
        let result_type = self.mlir_value_type(result_ty, result_kind)?;
        let results =
            self.append_op_with_result_types(op_name, operands, &[result_type], attrs, Vec::new())?;
        Ok(match result_kind {
            ValueKind::Scalar => Value::scalar(results[0], result_ty.elem),
            ValueKind::Tensor => Value::tensor(results[0], result_ty.clone()),
        })
    }

    pub(super) fn append_op_with_result_types(
        &mut self,
        op_name: &str,
        operands: &[Value],
        result_types: &[Type<'c>],
        attrs: &[(String, String)],
        regions: Vec<Region<'c>>,
    ) -> anyhow::Result<Vec<RawValue>> {
        let operands = operands
            .iter()
            .map(|value| value.raw.as_value())
            .collect::<Vec<_>>();
        let attrs = self.parse_attrs(attrs)?;
        let op = OperationBuilder::new(op_name, self.location)
            .add_operands(&operands)
            .add_results(result_types)
            .add_attributes(&attrs)
            .add_regions_vec(regions)
            .build()?;
        let op = self.block.append_operation(op);
        (0..result_types.len())
            .map(|index| Ok(RawValue::from_value(op.result(index)?.into())))
            .collect()
    }

    pub(super) fn append_op_with_built_attrs(
        &mut self,
        op_name: &str,
        operands: &[Value],
        result_types: &[Type<'c>],
        attrs: &[(Identifier<'c>, Attribute<'c>)],
        regions: Vec<Region<'c>>,
    ) -> anyhow::Result<Vec<RawValue>> {
        let operands = operands
            .iter()
            .map(|value| value.raw.as_value())
            .collect::<Vec<_>>();
        let op = OperationBuilder::new(op_name, self.location)
            .add_operands(&operands)
            .add_results(result_types)
            .add_attributes(attrs)
            .add_regions_vec(regions)
            .build()?;
        let op = self.block.append_operation(op);
        (0..result_types.len())
            .map(|index| Ok(RawValue::from_value(op.result(index)?.into())))
            .collect()
    }

    pub(super) fn append_tensor_empty(&mut self, ty: &TensorType) -> anyhow::Result<Value> {
        self.append_op("tensor.empty", &[], ty, ValueKind::Tensor)
    }

    pub(super) fn append_linalg_fill(
        &mut self,
        scalar: Value,
        output: Value,
        ty: &TensorType,
    ) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            let result_type = mlir_tensor_type(self.context, ty)?;
            let results = self.append_op_with_result_types(
                "tensor.from_elements",
                &[scalar],
                &[result_type],
                &[],
                Vec::new(),
            )?;
            return Ok(Value::tensor(results[0], ty.clone()));
        }
        let output_map = if ty.rank() == 0 {
            "()".to_string()
        } else {
            format!(
                "({})",
                (0..ty.rank())
                    .map(|axis| format!("d{axis}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let iterators = vec!["parallel"; ty.rank()];
        let results = self.append_linalg_generic(
            &[scalar],
            &[output],
            &[ty.clone()],
            ty.rank(),
            &["()".to_string(), output_map],
            &iterators,
            |_, _, args| Ok(vec![RawValue::from_value(args[0])]),
        )?;
        Ok(Value::tensor(results[0], ty.clone()))
    }

    pub(super) fn append_named_linalg(
        &mut self,
        op_name: &str,
        inputs: &[Value],
        output: Value,
        ty: &TensorType,
        attrs: &[(String, String)],
    ) -> anyhow::Result<Value> {
        let attrs = self.parse_attrs(attrs)?;
        self.append_named_linalg_with_built_attrs(op_name, inputs, output, ty, &attrs)
    }

    pub(super) fn append_named_linalg_with_built_attrs(
        &mut self,
        op_name: &str,
        inputs: &[Value],
        output: Value,
        ty: &TensorType,
        attrs: &[(Identifier<'c>, Attribute<'c>)],
    ) -> anyhow::Result<Value> {
        let mut operands = inputs.to_vec();
        operands.push(output);
        let mut all_attrs = vec![dense_i32_attr(
            self.context,
            "operand_segment_sizes",
            &[inputs.len() as i32, 1],
        )];
        all_attrs.extend_from_slice(attrs);
        let result_type = mlir_tensor_type(self.context, ty)?;
        let regions = if op_name == "linalg.softmax" {
            Vec::new()
        } else {
            vec![self.named_linalg_region(op_name, inputs, ty)?]
        };
        let results = self.append_op_with_built_attrs(
            op_name,
            &operands,
            &[result_type],
            &all_attrs,
            regions,
        )?;
        Ok(Value::tensor(results[0], ty.clone()))
    }

    fn named_linalg_region(
        &self,
        op_name: &str,
        inputs: &[Value],
        ty: &TensorType,
    ) -> anyhow::Result<Region<'c>> {
        let block_arg_types = inputs
            .iter()
            .map(|value| value.ty.elem)
            .chain(std::iter::once(ty.elem))
            .map(|elem| mlir_element_type(self.context, elem).map(|ty| (ty, self.location)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let block = Block::new(&block_arg_types);
        let args = (0..block_arg_types.len())
            .map(|index| block.argument(index).map(Into::into))
            .collect::<Result<Vec<MlirValue<'c, '_>>, _>>()?;
        let yielded = if matches!(
            op_name,
            "linalg.matmul" | "linalg.batch_matmul" | "linalg.conv_2d_nhwc_hwcf"
        ) {
            let elem_ty = mlir_element_type(self.context, ty.elem)?;
            let mul_op = if ty.elem.is_float() {
                "arith.mulf"
            } else {
                "arith.muli"
            };
            let add_op = if ty.elem.is_float() {
                "arith.addf"
            } else {
                "arith.addi"
            };
            let product = append_block_op(
                self.context,
                &block,
                self.location,
                mul_op,
                &[args[0], args[1]],
                &[elem_ty],
                &[],
                Vec::new(),
            )?[0];
            append_block_op(
                self.context,
                &block,
                self.location,
                add_op,
                &[product.as_value(), args[2]],
                &[elem_ty],
                &[],
                Vec::new(),
            )?[0]
        } else {
            RawValue::from_value(args[0])
        };
        append_block_op(
            self.context,
            &block,
            self.location,
            "linalg.yield",
            &[yielded.as_value()],
            &[],
            &[],
            Vec::new(),
        )?;
        let region = Region::new();
        region.append_block(block);
        Ok(region)
    }

    pub(super) fn append_linalg_generic<F>(
        &mut self,
        inputs: &[Value],
        outputs: &[Value],
        result_tys: &[TensorType],
        loop_rank: usize,
        indexing_maps: &[String],
        iterator_types: &[&str],
        build_body: F,
    ) -> anyhow::Result<Vec<RawValue>>
    where
        F: FnOnce(&mut Self, &Block<'c>, &[MlirValue<'c, '_>]) -> anyhow::Result<Vec<RawValue>>,
    {
        let mut operands = inputs.to_vec();
        operands.extend_from_slice(outputs);
        let block_arg_types = inputs
            .iter()
            .chain(outputs)
            .map(|value| {
                mlir_element_type(self.context, value.ty.elem).map(|ty| (ty, self.location))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let block = Block::new(&block_arg_types);
        let args = (0..block_arg_types.len())
            .map(|index| block.argument(index).map(Into::into))
            .collect::<Result<Vec<MlirValue<'c, '_>>, _>>()?;
        let yields = build_body(self, &block, &args)?
            .into_iter()
            .map(RawValue::as_value)
            .collect::<Vec<_>>();
        append_block_op(
            self.context,
            &block,
            self.location,
            "linalg.yield",
            &yields,
            &[],
            &[],
            Vec::new(),
        )?;
        let region = Region::new();
        region.append_block(block);
        let result_types = result_tys
            .iter()
            .map(|ty| mlir_tensor_type(self.context, ty))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let dims = (0..loop_rank)
            .map(|index| format!("d{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let maps = indexing_maps
            .iter()
            .map(|map| format!("affine_map<({dims}) -> {map}>"))
            .collect::<Vec<_>>()
            .join(", ");
        let iterators = iterator_types
            .iter()
            .map(|kind| format!("#linalg.iterator_type<{kind}>"))
            .collect::<Vec<_>>()
            .join(", ");
        let attrs = vec![
            dense_i32_attr(
                self.context,
                "operand_segment_sizes",
                &[inputs.len() as i32, outputs.len() as i32],
            ),
            (
                Identifier::new(self.context, "indexing_maps"),
                parse_attr(self.context, &format!("[{maps}]"))?,
            ),
            (
                Identifier::new(self.context, "iterator_types"),
                parse_attr(self.context, &format!("[{iterators}]"))?,
            ),
        ];
        self.append_op_with_built_attrs(
            "linalg.generic",
            &operands,
            &result_types,
            &attrs,
            vec![region],
        )
    }

    pub(super) fn append_index_constant(&mut self, value: usize) -> anyhow::Result<RawValue> {
        let result_type = Type::index(self.context);
        let results = self.append_op_with_result_types(
            "arith.constant",
            &[],
            &[result_type],
            &[("value".to_string(), format!("{value} : index"))],
            Vec::new(),
        )?;
        Ok(results[0])
    }

    pub(super) fn append_tensor_extract(
        &mut self,
        input: Value,
        indices: &[RawValue],
    ) -> anyhow::Result<Value> {
        let mut operands = vec![input.raw.as_value()];
        operands.extend(indices.iter().copied().map(RawValue::as_value));
        let result_type = mlir_element_type(self.context, input.ty.elem)?;
        let op = OperationBuilder::new("tensor.extract", self.location)
            .add_operands(&operands)
            .add_results(&[result_type])
            .build()?;
        let op = self.block.append_operation(op);
        let result = RawValue::from_value(op.result(0)?.into());
        Ok(Value::scalar(result, input.ty.elem))
    }

    pub(super) fn append_tensor_pad(
        &mut self,
        input: Value,
        ty: &TensorType,
        lows: &[usize],
        highs: &[usize],
        pad_value: Value,
    ) -> anyhow::Result<Value> {
        let block_args = (0..ty.rank())
            .map(|_| Type::index(self.context))
            .map(|ty| (ty, self.location))
            .collect::<Vec<_>>();
        let block = Block::new(&block_args);
        append_block_op(
            self.context,
            &block,
            self.location,
            "tensor.yield",
            &[pad_value.raw.as_value()],
            &[],
            &[],
            Vec::new(),
        )?;
        let region = Region::new();
        region.append_block(block);
        let attrs = [
            dense_i32_attr(self.context, "operand_segment_sizes", &[1, 0, 0]),
            dense_i64_attr(self.context, "static_low", lows)?,
            dense_i64_attr(self.context, "static_high", highs)?,
        ];
        let result_type = mlir_tensor_type(self.context, ty)?;
        let results = self.append_op_with_built_attrs(
            "tensor.pad",
            &[input],
            &[result_type],
            &attrs,
            vec![region],
        )?;
        Ok(Value::tensor(results[0], ty.clone()))
    }

    pub(super) fn append_tensor_extract_slice(
        &mut self,
        input: Value,
        ty: &TensorType,
        offsets: &[usize],
        sizes: &[usize],
        strides: &[usize],
    ) -> anyhow::Result<Value> {
        let result_type = mlir_tensor_type(self.context, ty)?;
        let attrs = [
            dense_i32_attr(self.context, "operand_segment_sizes", &[1, 0, 0, 0]),
            dense_i64_attr(self.context, "static_offsets", offsets)?,
            dense_i64_attr(self.context, "static_sizes", sizes)?,
            dense_i64_attr(self.context, "static_strides", strides)?,
        ];
        let results = self.append_op_with_built_attrs(
            "tensor.extract_slice",
            &[input],
            &[result_type],
            &attrs,
            Vec::new(),
        )?;
        Ok(Value::tensor(results[0], ty.clone()))
    }

    pub(super) fn append_tensor_insert_slice(
        &mut self,
        source: Value,
        dest: Value,
        dest_ty: &TensorType,
        offsets: &[usize],
    ) -> anyhow::Result<Value> {
        let result_type = mlir_tensor_type(self.context, dest_ty)?;
        let strides = vec![1; source.ty.rank()];
        let attrs = [
            dense_i32_attr(self.context, "operand_segment_sizes", &[1, 1, 0, 0, 0]),
            dense_i64_attr(self.context, "static_offsets", offsets)?,
            dense_i64_attr(self.context, "static_sizes", &source.ty.shape)?,
            dense_i64_attr(self.context, "static_strides", &strides)?,
        ];
        let results = self.append_op_with_built_attrs(
            "tensor.insert_slice",
            &[source, dest],
            &[result_type],
            &attrs,
            Vec::new(),
        )?;
        Ok(Value::tensor(results[0], dest_ty.clone()))
    }

    pub(super) fn append_reassociation_op(
        &mut self,
        op_name: &str,
        input: Value,
        ty: &TensorType,
        reassociation: &str,
        output_shape: Option<&[usize]>,
    ) -> anyhow::Result<Value> {
        let mut attrs = vec![(
            Identifier::new(self.context, "reassociation"),
            parse_attr(self.context, reassociation)?,
        )];
        if let Some(output_shape) = output_shape {
            attrs.push(dense_i64_attr(
                self.context,
                "static_output_shape",
                output_shape,
            )?);
            attrs.push(dense_i32_attr(self.context, "operand_segment_sizes", &[0]));
        }
        let result_type = mlir_tensor_type(self.context, ty)?;
        let results =
            self.append_op_with_built_attrs(op_name, &[input], &[result_type], &attrs, Vec::new())?;
        Ok(Value::tensor(results[0], ty.clone()))
    }

    pub(super) fn mlir_value_type(
        &self,
        ty: &TensorType,
        kind: ValueKind,
    ) -> anyhow::Result<Type<'c>> {
        match kind {
            ValueKind::Scalar => mlir_element_type(self.context, ty.elem),
            ValueKind::Tensor => mlir_tensor_type(self.context, ty),
        }
    }

    pub(super) fn parse_attrs(
        &self,
        attrs: &[(String, String)],
    ) -> anyhow::Result<Vec<(Identifier<'c>, Attribute<'c>)>> {
        attrs
            .iter()
            .map(|(name, value)| {
                Ok((
                    Identifier::new(self.context, name),
                    Attribute::parse(self.context, value).ok_or_else(|| {
                        anyhow::anyhow!("failed to parse MLIR attribute `{value}`")
                    })?,
                ))
            })
            .collect()
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

pub(super) fn mlir_element_type<'c>(
    context: &'c Context,
    elem: knok_core::ElementType,
) -> anyhow::Result<Type<'c>> {
    Type::parse(context, elem.mlir_type())
        .ok_or_else(|| anyhow::anyhow!("failed to parse MLIR element type `{}`", elem.mlir_type()))
}

pub(super) fn mlir_tensor_type<'c>(
    context: &'c Context,
    ty: &TensorType,
) -> anyhow::Result<Type<'c>> {
    Type::parse(context, &ty.mlir_type())
        .ok_or_else(|| anyhow::anyhow!("failed to parse MLIR tensor type `{}`", ty.mlir_type()))
}

pub(super) fn append_block_op<'c>(
    context: &'c Context,
    block: &Block<'c>,
    location: Location<'c>,
    op_name: &str,
    operands: &[MlirValue<'c, '_>],
    result_types: &[Type<'c>],
    attrs: &[(String, String)],
    regions: Vec<Region<'c>>,
) -> anyhow::Result<Vec<RawValue>> {
    let attrs = attrs
        .iter()
        .map(|(name, value)| {
            Ok((
                Identifier::new(context, name),
                Attribute::parse(context, value)
                    .ok_or_else(|| anyhow::anyhow!("failed to parse MLIR attribute `{value}`"))?,
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let op = OperationBuilder::new(op_name, location)
        .add_operands(operands)
        .add_results(result_types)
        .add_attributes(&attrs)
        .add_regions_vec(regions)
        .build()?;
    let op = block.append_operation(op);
    (0..result_types.len())
        .map(|index| Ok(RawValue::from_value(op.result(index)?.into())))
        .collect()
}

fn parse_attr<'c>(context: &'c Context, value: &str) -> anyhow::Result<Attribute<'c>> {
    Attribute::parse(context, value)
        .ok_or_else(|| anyhow::anyhow!("failed to parse MLIR attribute `{value}`"))
}

fn dense_i32_attr<'c>(
    context: &'c Context,
    name: &str,
    values: &[i32],
) -> (Identifier<'c>, Attribute<'c>) {
    let attr = DenseI32ArrayAttribute::new(context, values);
    (Identifier::new(context, name), unsafe {
        Attribute::from_raw(attr.to_raw())
    })
}

pub(super) fn dense_i64_attr<'c>(
    context: &'c Context,
    name: &str,
    values: &[usize],
) -> anyhow::Result<(Identifier<'c>, Attribute<'c>)> {
    let values = values
        .iter()
        .copied()
        .map(i64::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let attr = DenseI64ArrayAttribute::new(context, &values);
    Ok((Identifier::new(context, name), unsafe {
        Attribute::from_raw(attr.to_raw())
    }))
}
