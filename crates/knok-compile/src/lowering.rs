use std::collections::BTreeMap;

use knok_core::{BinaryOp, CallOp, ElementType, Expr, TensorType, TypedGraph, UnaryOp};

use crate::common::mlir_result_types;

fn element_count(ty: &TensorType) -> usize {
    ty.shape.iter().product()
}

fn format_shape_list(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_usize_list(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn reassociation_for_rank(rank: usize) -> String {
    let dims = (0..rank)
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[[{dims}]]")
}

fn collapse_reassociation_for_removed_axis(rank: usize, axis: usize) -> String {
    if rank <= 1 {
        return reassociation_for_rank(rank);
    }
    let mut groups = Vec::new();
    let mut index = 0;
    while index < rank {
        if index == axis {
            if groups.is_empty() {
                groups.push(vec![index, index + 1]);
                index += 2;
            } else {
                groups.last_mut().expect("group exists").push(index);
                index += 1;
            }
        } else {
            groups.push(vec![index]);
            index += 1;
        }
    }
    format_reassociation_groups(groups)
}

fn expand_reassociation_for_inserted_axis(input_rank: usize, axis: usize) -> String {
    let mut groups = Vec::new();
    for input_axis in 0..input_rank {
        if input_axis == axis {
            groups.push(vec![input_axis, input_axis + 1]);
        } else if input_axis < axis {
            groups.push(vec![input_axis]);
        } else {
            groups.push(vec![input_axis + 1]);
        }
    }
    if axis == input_rank {
        if let Some(last) = groups.last_mut() {
            last.push(axis);
        } else {
            groups.push(vec![axis]);
        }
    }
    format_reassociation_groups(groups)
}

fn format_reassociation_groups(groups: Vec<Vec<usize>>) -> String {
    let groups = groups
        .into_iter()
        .map(|group| {
            format!(
                "[{}]",
                group
                    .into_iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{groups}]")
}

fn format_dim_list(rank: usize) -> String {
    (0..rank)
        .map(|index| format!("d{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn parallel_iterators(rank: usize) -> String {
    (0..rank)
        .map(|_| "\"parallel\"")
        .collect::<Vec<_>>()
        .join(", ")
}

fn broadcast_result_type(lhs: &TensorType, rhs: &TensorType) -> anyhow::Result<TensorType> {
    if lhs.elem != rhs.elem {
        anyhow::bail!("binary operands have different element types");
    }
    let shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn ensure_broadcastable(input: &TensorType, output: &TensorType) -> anyhow::Result<()> {
    if input.elem != output.elem {
        anyhow::bail!("broadcast input and output element types differ");
    }
    let shape = broadcast_shape(&input.shape, &output.shape)?;
    if shape != output.shape {
        anyhow::bail!(
            "broadcast result shape {:?} does not match requested output {:?}",
            shape,
            output.shape
        );
    }
    Ok(())
}

fn axis_broadcast_dimensions(
    input_rank: usize,
    output_rank: usize,
    axis: usize,
) -> anyhow::Result<Vec<usize>> {
    if input_rank + 1 != output_rank {
        anyhow::bail!("axis broadcast expects exactly one reduced dimension");
    }
    if axis >= output_rank {
        anyhow::bail!("axis {axis} is out of bounds for rank {output_rank}");
    }
    Ok(vec![axis])
}

fn ensure_axis_broadcastable(
    input: &TensorType,
    output: &TensorType,
    axis: usize,
) -> anyhow::Result<()> {
    if input.elem != output.elem {
        anyhow::bail!("broadcast input and output element types differ");
    }
    if input.rank() + 1 != output.rank() {
        anyhow::bail!("axis broadcast expects exactly one reduced dimension");
    }
    for output_index in 0..output.rank() {
        if output_index == axis {
            continue;
        }
        let input_index = if output_index < axis {
            output_index
        } else {
            output_index - 1
        };
        if input.shape[input_index] != output.shape[output_index] {
            anyhow::bail!(
                "axis broadcast dimension mismatch at output dimension {}: input {} vs output {}",
                output_index,
                input.shape[input_index],
                output.shape[output_index]
            );
        }
    }
    Ok(())
}

fn collapse_reassociation_for_squeezed_broadcast(
    input_shape: &[usize],
    aligned_output_shape: &[usize],
) -> String {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut pending = Vec::new();
    for (index, (input_dim, output_dim)) in input_shape.iter().zip(aligned_output_shape).enumerate()
    {
        pending.push(index);
        if !(*input_dim == 1 && *output_dim != 1) {
            groups.push(core::mem::take(&mut pending));
        }
    }
    if !pending.is_empty() {
        if let Some(last) = groups.last_mut() {
            last.extend(pending);
        } else {
            groups.push(pending);
        }
    }
    let groups = groups
        .into_iter()
        .map(|group| {
            format!(
                "[{}]",
                group
                    .into_iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{groups}]")
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> anyhow::Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let lhs_dim = dim_from_trailing(lhs, rank, offset);
        let rhs_dim = dim_from_trailing(rhs, rank, offset);
        let dim = match (lhs_dim, rhs_dim) {
            (Some(lhs_dim), Some(rhs_dim)) if lhs_dim == rhs_dim => lhs_dim,
            (Some(1), Some(rhs_dim)) => rhs_dim,
            (Some(lhs_dim), Some(1)) => lhs_dim,
            (None, Some(dim)) | (Some(dim), None) => dim,
            (None, None) => unreachable!("rank is derived from at least one shape"),
            (Some(lhs_dim), Some(rhs_dim)) => {
                anyhow::bail!("broadcast dimension {offset} differs: {lhs_dim} vs {rhs_dim}");
            }
        };
        shape.push(dim);
    }
    Ok(shape)
}

fn dim_from_trailing(shape: &[usize], rank: usize, offset: usize) -> Option<usize> {
    let padding = rank - shape.len();
    (offset >= padding).then(|| shape[offset - padding])
}

fn reduction_output_shape(
    input_shape: &[usize],
    axis: Option<usize>,
    keep_dims: bool,
) -> Vec<usize> {
    match axis {
        Some(axis) if keep_dims => input_shape
            .iter()
            .enumerate()
            .map(|(index, dim)| if index == axis { 1 } else { *dim })
            .collect(),
        Some(axis) => {
            let mut shape = input_shape.to_vec();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        }
        None => vec![1],
    }
}

fn reduction_output_map(input_rank: usize, axis: Option<usize>, keep_dims: bool) -> String {
    match axis {
        Some(axis) if keep_dims => {
            let dims = (0..input_rank)
                .map(|index| {
                    if index == axis {
                        "0".to_string()
                    } else {
                        format!("d{index}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("({dims})")
        }
        Some(_) if input_rank == 1 => "(0)".to_string(),
        Some(axis) => {
            let dims = (0..input_rank)
                .filter(|index| *index != axis)
                .map(|index| format!("d{index}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({dims})")
        }
        None => "(0)".to_string(),
    }
}

fn min_float_literal(elem: ElementType) -> &'static str {
    match elem {
        ElementType::Bool => "0",
        ElementType::F32 => "-3.40282347E+38",
        ElementType::F64 => "-1.7976931348623157E+308",
        ElementType::F16 => "-65504.0",
        ElementType::BF16 => "-3.38953139E+38",
        ElementType::I32 | ElementType::I64 => "0",
    }
}

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

struct Lowerer<'a> {
    graph: &'a TypedGraph,
    graphs: &'a BTreeMap<String, TypedGraph>,
    call_stack: Vec<String>,
    next_value: usize,
    lines: Vec<String>,
    values: BTreeMap<String, Value>,
}

#[derive(Clone)]
struct Value {
    name: String,
    ty: TensorType,
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
                    Value {
                        name: format!("%arg{index}"),
                        ty: input.ty.clone(),
                    },
                );
                format!("%arg{index}: {}", input.ty.mlir_type())
            })
            .collect::<Vec<_>>()
            .join(", ");
        for binding in &self.graph.lets {
            let values = self.lower_let_values(&binding.value.kind)?;
            self.bind_values(&binding.names, values, None)?;
        }
        let body = self
            .graph
            .body
            .iter()
            .map(|expr| self.lower_expr(&expr.kind))
            .collect::<anyhow::Result<Vec<_>>>()?;
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
                CallOp::Argmax => {
                    let input = self.lower_expr(&args[0])?;
                    self.argmax(input)
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
                CallOp::Conv2d => {
                    let input = self.lower_expr(&args[0])?;
                    let kernel = self.lower_expr(&args[1])?;
                    self.conv2d(input, kernel)
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
                CallOp::Unsqueeze(ty) => {
                    let input = self.lower_expr(&args[0])?;
                    self.reshape(input, ty)
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
            graph
                .body
                .iter()
                .map(|expr| self.lower_expr(&expr.kind))
                .collect()
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

    fn constant(&mut self, value: &str, elem: ElementType) -> anyhow::Result<Value> {
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = arith.constant {value} : {}",
            elem.mlir_type()
        ));
        Ok(Value {
            name,
            ty: TensorType {
                elem,
                shape: vec![],
            },
        })
    }

    fn zero_like(&mut self, ty: &TensorType) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            return self.constant(ty.elem.zero_literal(), ty.elem);
        }
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = arith.constant dense<{}> : {}",
            ty.elem.zero_literal(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn binary_value(&mut self, op: BinaryOp, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = match (elem.is_float(), op) {
            (true, BinaryOp::Add) => "arith.addf",
            (true, BinaryOp::Sub) => "arith.subf",
            (true, BinaryOp::Mul) => "arith.mulf",
            (true, BinaryOp::Div) => "arith.divf",
            (false, BinaryOp::Add) => "arith.addi",
            (false, BinaryOp::Sub) => "arith.subi",
            (false, BinaryOp::Mul) => "arith.muli",
            (false, BinaryOp::Div) => "arith.divsi",
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    fn minimum(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = if elem.is_float() {
            "arith.minimumf"
        } else {
            "arith.minsi"
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    fn maximum(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = if elem.is_float() {
            "arith.maximumf"
        } else {
            "arith.maxsi"
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    fn emit_unary(&mut self, op_name: &str, value: Value) -> anyhow::Result<Value> {
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = {op_name} {} : {}",
            value.name,
            value.ty.mlir_type()
        ));
        Ok(Value { name, ty: value.ty })
    }

    fn emit_binary(&mut self, op_name: &str, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let ty = broadcast_result_type(&lhs.ty, &rhs.ty)?;
        let lhs = if lhs.ty == ty {
            lhs
        } else {
            self.broadcast(lhs, &ty)?
        };
        let rhs = if rhs.ty == ty {
            rhs
        } else {
            self.broadcast(rhs, &ty)?
        };
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = {op_name} {}, {} : {}",
            lhs.name,
            rhs.name,
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn comparison(
        &mut self,
        float_predicate: &str,
        integer_predicate: &str,
        lhs: Value,
        rhs: Value,
    ) -> anyhow::Result<Value> {
        let shape = broadcast_shape(&lhs.ty.shape, &rhs.ty.shape)?;
        let ty = TensorType {
            elem: ElementType::Bool,
            shape,
        };
        let op_name = if lhs.ty.elem.is_float() {
            "arith.cmpf"
        } else {
            "arith.cmpi"
        };
        let predicate = if lhs.ty.elem.is_float() {
            float_predicate
        } else {
            integer_predicate
        };
        self.emit_elementwise_binary(&ty, lhs, rhs, |result, lhs, rhs, elem| {
            format!("      {result} = {op_name} {predicate}, {lhs}, {rhs} : {elem}")
        })
    }

    fn isnan(&mut self, value: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: ElementType::Bool,
            shape: value.ty.shape.clone(),
        };
        self.emit_elementwise_unary(&ty, value, |result, value, elem| {
            format!("      {result} = arith.cmpf uno, {value}, {value} : {elem}")
        })
    }

    fn logical_binary(&mut self, op_name: &str, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let ty = broadcast_result_type(&lhs.ty, &rhs.ty)?;
        self.emit_elementwise_binary(&ty, lhs, rhs, |result, lhs, rhs, elem| {
            format!("      {result} = {op_name} {lhs}, {rhs} : {elem}")
        })
    }

    fn logical_not(&mut self, value: Value) -> anyhow::Result<Value> {
        let ty = value.ty.clone();
        let true_value = self.constant("1", ElementType::Bool)?;
        let true_value = self.broadcast(true_value, &ty)?;
        self.logical_binary("arith.xori", value, true_value)
    }

    fn where_select(
        &mut self,
        condition: Value,
        true_value: Value,
        false_value: Value,
    ) -> anyhow::Result<Value> {
        let value_shape = broadcast_shape(&true_value.ty.shape, &false_value.ty.shape)?;
        let value_ty = TensorType {
            elem: true_value.ty.elem,
            shape: value_shape,
        };
        let shape = broadcast_shape(&condition.ty.shape, &value_ty.shape)?;
        let ty = TensorType {
            elem: true_value.ty.elem,
            shape,
        };
        let condition_ty = TensorType {
            elem: ElementType::Bool,
            shape: ty.shape.clone(),
        };
        let condition = if condition.ty == condition_ty {
            condition
        } else {
            self.broadcast(condition, &condition_ty)?
        };
        let true_ty = TensorType {
            elem: true_value.ty.elem,
            shape: ty.shape.clone(),
        };
        let true_value = if true_value.ty == true_ty {
            true_value
        } else {
            self.broadcast(true_value, &true_ty)?
        };
        let false_ty = TensorType {
            elem: false_value.ty.elem,
            shape: ty.shape.clone(),
        };
        let false_value = if false_value.ty == false_ty {
            false_value
        } else {
            self.broadcast(false_value, &false_ty)?
        };
        self.emit_elementwise_ternary(&ty, condition, true_value, false_value)
    }

    fn emit_elementwise_unary<F>(
        &mut self,
        ty: &TensorType,
        value: Value,
        emit_body: F,
    ) -> anyhow::Result<Value>
    where
        F: FnOnce(&str, &str, &str) -> String,
    {
        if ty.rank() == 0 {
            let name = self.fresh();
            self.lines
                .push(emit_body(&name, &value.name, value.ty.elem.mlir_type()));
            return Ok(Value {
                name,
                ty: ty.clone(),
            });
        }
        let value_ty = TensorType {
            elem: value.ty.elem,
            shape: ty.shape.clone(),
        };
        let value = if value.ty == value_ty {
            value
        } else {
            self.broadcast(value, &value_ty)?
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let dims = format_dim_list(ty.rank());
        let map = format!("({dims})");
        let iterators = parallel_iterators(ty.rank());
        let result = self.fresh();
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
        self.lines.push(format!(
            "    }} ins({} : {}) outs({empty} : {}) {{",
            value.name,
            value.ty.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%value: {}, %out: {}):",
            value.ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines
            .push(emit_body(&result, "%value", value.ty.elem.mlir_type()));
        self.lines.push(format!(
            "      linalg.yield {result} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn emit_elementwise_binary<F>(
        &mut self,
        ty: &TensorType,
        lhs: Value,
        rhs: Value,
        emit_body: F,
    ) -> anyhow::Result<Value>
    where
        F: FnOnce(&str, &str, &str, &str) -> String,
    {
        if ty.rank() == 0 {
            let name = self.fresh();
            self.lines.push(emit_body(
                &name,
                &lhs.name,
                &rhs.name,
                lhs.ty.elem.mlir_type(),
            ));
            return Ok(Value {
                name,
                ty: ty.clone(),
            });
        }
        let lhs_ty = TensorType {
            elem: lhs.ty.elem,
            shape: ty.shape.clone(),
        };
        let lhs = if lhs.ty == lhs_ty {
            lhs
        } else {
            self.broadcast(lhs, &lhs_ty)?
        };
        let rhs_ty = TensorType {
            elem: rhs.ty.elem,
            shape: ty.shape.clone(),
        };
        let rhs = if rhs.ty == rhs_ty {
            rhs
        } else {
            self.broadcast(rhs, &rhs_ty)?
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let dims = format_dim_list(ty.rank());
        let map = format!("({dims})");
        let iterators = parallel_iterators(ty.rank());
        let result = self.fresh();
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
        self.lines.push(format!(
            "    }} ins({}, {} : {}, {}) outs({empty} : {}) {{",
            lhs.name,
            rhs.name,
            lhs.ty.mlir_type(),
            rhs.ty.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%lhs: {}, %rhs: {}, %out: {}):",
            lhs.ty.elem.mlir_type(),
            rhs.ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines
            .push(emit_body(&result, "%lhs", "%rhs", lhs.ty.elem.mlir_type()));
        self.lines.push(format!(
            "      linalg.yield {result} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn emit_elementwise_ternary(
        &mut self,
        ty: &TensorType,
        condition: Value,
        true_value: Value,
        false_value: Value,
    ) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            let name = self.fresh();
            self.lines.push(format!(
                "    {name} = arith.select {}, {}, {} : {}",
                condition.name,
                true_value.name,
                false_value.name,
                ty.elem.mlir_type()
            ));
            return Ok(Value {
                name,
                ty: ty.clone(),
            });
        }
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let dims = format_dim_list(ty.rank());
        let map = format!("({dims})");
        let iterators = parallel_iterators(ty.rank());
        let selected = self.fresh();
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
        self.lines.push(format!(
            "    }} ins({}, {}, {} : {}, {}, {}) outs({empty} : {}) {{",
            condition.name,
            true_value.name,
            false_value.name,
            condition.ty.mlir_type(),
            true_value.ty.mlir_type(),
            false_value.ty.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%condition: i1, %true_value: {}, %false_value: {}, %out: {}):",
            ty.elem.mlir_type(),
            ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {selected} = arith.select %condition, %true_value, %false_value : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield {selected} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn splat(&mut self, scalar: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.fill ins({} : {}) outs({empty} : {}) -> {}",
            scalar.name,
            scalar.ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn matmul(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        if lhs.ty.rank() == 3 {
            return self.batch_matmul(lhs, rhs);
        }
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape: vec![lhs.ty.shape[0], rhs.ty.shape[1]],
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({zero} : {}) outs({empty} : {}) -> {}",
            ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.matmul ins({}, {} : {}, {}) outs({init} : {}) -> {}",
            lhs.name,
            rhs.name,
            lhs.ty.mlir_type(),
            rhs.ty.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn batch_matmul(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape: vec![lhs.ty.shape[0], lhs.ty.shape[1], rhs.ty.shape[2]],
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({zero} : {}) outs({empty} : {}) -> {}",
            ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.batch_matmul ins({}, {} : {}, {}) outs({init} : {}) -> {}",
            lhs.name,
            rhs.name,
            lhs.ty.mlir_type(),
            rhs.ty.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn conv2d(&mut self, input: Value, kernel: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![
                input.ty.shape[0],
                input.ty.shape[1] - kernel.ty.shape[0] + 1,
                input.ty.shape[2] - kernel.ty.shape[1] + 1,
                kernel.ty.shape[3],
            ],
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({zero} : {}) outs({empty} : {}) -> {}",
            ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.conv_2d_nhwc_hwcf ins({}, {} : {}, {}) outs({init} : {}) -> {}",
            input.name,
            kernel.name,
            input.ty.mlir_type(),
            kernel.ty.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn transpose(&mut self, input: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![input.ty.shape[1], input.ty.shape[0]],
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.transpose ins({} : {}) outs({empty} : {}) permutation = [1, 0]",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn reshape(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        if input.ty == *ty {
            return Ok(input);
        }
        let flat = if input.ty.rank() == 1 {
            input
        } else {
            let flat_ty = TensorType {
                elem: input.ty.elem,
                shape: vec![element_count(&input.ty)],
            };
            self.collapse_to_rank1(input, &flat_ty)?
        };
        if ty.rank() == 1 {
            Ok(flat)
        } else {
            self.expand_rank1(flat, ty)
        }
    }

    fn expand_rank1(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let name = self.fresh();
        let output_shape = format_shape_list(&ty.shape);
        let reassociation = reassociation_for_rank(ty.rank());
        self.lines.push(format!(
            "    {name} = tensor.expand_shape {} {reassociation} output_shape {output_shape} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn collapse_to_rank1(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let name = self.fresh();
        let reassociation = reassociation_for_rank(input.ty.rank());
        self.lines.push(format!(
            "    {name} = tensor.collapse_shape {} {reassociation} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn slice(&mut self, input: Value, ty: &TensorType, starts: &[usize]) -> anyhow::Result<Value> {
        let offsets = format_usize_list(starts);
        let sizes = format_usize_list(&ty.shape);
        let strides = format_usize_list(&vec![1; ty.rank()]);
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.extract_slice {}{offsets} {sizes} {strides} : {} to {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn take(&mut self, input: Value, axis: usize, index: usize) -> anyhow::Result<Value> {
        let mut starts = vec![0; input.ty.rank()];
        starts[axis] = index;
        let mut slice_shape = input.ty.shape.clone();
        slice_shape[axis] = 1;
        let slice_ty = TensorType {
            elem: input.ty.elem,
            shape: slice_shape,
        };
        let sliced = self.slice(input, &slice_ty, &starts)?;
        let mut output_shape = sliced.ty.shape.clone();
        output_shape.remove(axis);
        if output_shape.is_empty() {
            output_shape.push(1);
        }
        let output_ty = TensorType {
            elem: sliced.ty.elem,
            shape: output_shape,
        };
        if sliced.ty == output_ty {
            return Ok(sliced);
        }
        let name = self.fresh();
        let reassociation = collapse_reassociation_for_removed_axis(sliced.ty.rank(), axis);
        self.lines.push(format!(
            "    {name} = tensor.collapse_shape {} {reassociation} : {} into {}",
            sliced.name,
            sliced.ty.mlir_type(),
            output_ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: output_ty,
        })
    }

    fn concat(&mut self, lhs: Value, rhs: Value, axis: usize) -> anyhow::Result<Value> {
        let mut shape = lhs.ty.shape.clone();
        shape[axis] += rhs.ty.shape[axis];
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape,
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let lhs_offsets = vec![0; ty.rank()];
        let first = self.insert_slice(lhs, empty, &ty, &lhs_offsets)?;
        let mut rhs_offsets = vec![0; ty.rank()];
        rhs_offsets[axis] = ty.shape[axis] - rhs.ty.shape[axis];
        self.insert_slice(rhs, first.name, &ty, &rhs_offsets)
    }

    fn stack(&mut self, lhs: Value, rhs: Value, axis: usize) -> anyhow::Result<Value> {
        let mut unit_shape = lhs.ty.shape.clone();
        unit_shape.insert(axis, 1);
        let unit_ty = TensorType {
            elem: lhs.ty.elem,
            shape: unit_shape,
        };
        let lhs = self.expand_insert_axis(lhs, &unit_ty, axis)?;
        let rhs = self.expand_insert_axis(rhs, &unit_ty, axis)?;
        self.concat(lhs, rhs, axis)
    }

    fn insert_slice(
        &mut self,
        source: Value,
        dest: String,
        dest_ty: &TensorType,
        offsets: &[usize],
    ) -> anyhow::Result<Value> {
        let offsets = format_usize_list(offsets);
        let sizes = format_usize_list(&source.ty.shape);
        let strides = format_usize_list(&vec![1; source.ty.rank()]);
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.insert_slice {} into {dest}{offsets} {sizes} {strides} : {} into {}",
            source.name,
            source.ty.mlir_type(),
            dest_ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: dest_ty.clone(),
        })
    }

    fn expand_insert_axis(
        &mut self,
        input: Value,
        ty: &TensorType,
        axis: usize,
    ) -> anyhow::Result<Value> {
        let name = self.fresh();
        let output_shape = format_shape_list(&ty.shape);
        let reassociation = expand_reassociation_for_inserted_axis(input.ty.rank(), axis);
        self.lines.push(format!(
            "    {name} = tensor.expand_shape {} {reassociation} output_shape {output_shape} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn broadcast(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        if input.ty == *ty {
            return Ok(input);
        }
        if input.ty.rank() == 0 {
            return self.splat(input, ty);
        }
        if element_count(&input.ty) == 1 {
            let scalar = self.extract_first_scalar(input)?;
            return self.splat(scalar, ty);
        }
        ensure_broadcastable(&input.ty, ty)?;
        let (input, dimensions) = self.squeeze_singleton_broadcast_dims(input, ty)?;
        self.emit_linalg_broadcast(input, ty, &dimensions)
    }

    fn broadcast_along_axis(
        &mut self,
        input: Value,
        ty: &TensorType,
        axis: usize,
    ) -> anyhow::Result<Value> {
        if input.ty == *ty {
            return Ok(input);
        }
        if input.ty.rank() == 0 || element_count(&input.ty) == 1 {
            let scalar = if input.ty.rank() == 0 {
                input
            } else {
                self.extract_first_scalar(input)?
            };
            return self.splat(scalar, ty);
        }
        ensure_axis_broadcastable(&input.ty, ty, axis)?;
        let dimensions = axis_broadcast_dimensions(input.ty.rank(), ty.rank(), axis)?;
        self.emit_linalg_broadcast(input, ty, &dimensions)
    }

    fn emit_linalg_broadcast(
        &mut self,
        input: Value,
        ty: &TensorType,
        dimensions: &[usize],
    ) -> anyhow::Result<Value> {
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let dimensions = format_usize_list(dimensions);
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.broadcast ins({} : {}) outs({empty} : {}) dimensions = {dimensions}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
        })
    }

    fn extract_first_scalar(&mut self, input: Value) -> anyhow::Result<Value> {
        let name = self.fresh();
        let zero = self.fresh();
        self.lines
            .push(format!("    {zero} = arith.constant 0 : index"));
        let indices = vec![zero; input.ty.rank()].join(", ");
        self.lines.push(format!(
            "    {name} = tensor.extract {}[{}] : {}",
            input.name,
            indices,
            input.ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: TensorType {
                elem: input.ty.elem,
                shape: Vec::new(),
            },
        })
    }

    fn squeeze_singleton_broadcast_dims(
        &mut self,
        input: Value,
        ty: &TensorType,
    ) -> anyhow::Result<(Value, Vec<usize>)> {
        let padding = ty.rank() - input.ty.rank();
        let singleton_dimensions = input
            .ty
            .shape
            .iter()
            .enumerate()
            .filter_map(|(index, input_dim)| {
                let output_dim = ty.shape[padding + index];
                (*input_dim == 1 && output_dim != 1).then_some(index)
            })
            .collect::<Vec<_>>();

        let mut dimensions = (0..padding).collect::<Vec<_>>();
        dimensions.extend(singleton_dimensions.iter().map(|index| padding + *index));

        if singleton_dimensions.is_empty() {
            return Ok((input, dimensions));
        }

        let squeezed_shape = input
            .ty
            .shape
            .iter()
            .enumerate()
            .filter_map(|(index, input_dim)| {
                let output_dim = ty.shape[padding + index];
                (!(*input_dim == 1 && output_dim != 1)).then_some(*input_dim)
            })
            .collect::<Vec<_>>();
        if squeezed_shape.is_empty() {
            return Ok((self.extract_first_scalar(input)?, (0..ty.rank()).collect()));
        }

        let squeezed_ty = TensorType {
            elem: input.ty.elem,
            shape: squeezed_shape,
        };
        let aligned_output_shape = &ty.shape[padding..];
        let reassociation =
            collapse_reassociation_for_squeezed_broadcast(&input.ty.shape, aligned_output_shape);
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.collapse_shape {} {reassociation} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            squeezed_ty.mlir_type()
        ));
        Ok((
            Value {
                name,
                ty: squeezed_ty,
            },
            dimensions,
        ))
    }

    fn mean(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        let count = axis.map_or_else(|| element_count(&input.ty), |axis| input.ty.shape[axis]);
        let elem = input.ty.elem;
        let sum = self.sum(input, axis)?;
        let scale = self.constant(&format!("{count}.0"), elem)?;
        self.emit_binary("arith.divf", sum, scale)
    }

    fn softmax(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        if let Some(axis) = axis {
            return self.axis_softmax(input, axis);
        }
        let max = self.max(input.clone(), axis, false)?;
        let max = if let Some(axis) = axis {
            self.broadcast_along_axis(max, &input.ty, axis)?
        } else {
            self.broadcast(max, &input.ty)?
        };
        let shifted = self.emit_binary("arith.subf", input, max)?;
        let exp = self.emit_unary("math.exp", shifted)?;
        let denominator = self.reduce(
            exp.clone(),
            exp.ty.elem.zero_literal(),
            "arith.addf",
            axis,
            false,
        )?;
        let denominator = if let Some(axis) = axis {
            self.broadcast_along_axis(denominator, &exp.ty, axis)?
        } else {
            self.broadcast(denominator, &exp.ty)?
        };
        self.emit_binary("arith.divf", exp, denominator)
    }

    fn axis_softmax(&mut self, input: Value, axis: usize) -> anyhow::Result<Value> {
        let empty = self.fresh();
        self.lines.push(format!(
            "    {empty} = tensor.empty() : {}",
            input.ty.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.softmax dimension({axis}) ins({} : {}) outs({empty} : {}) -> {}",
            input.name,
            input.ty.mlir_type(),
            input.ty.mlir_type(),
            input.ty.mlir_type()
        ));
        Ok(Value { name, ty: input.ty })
    }

    fn sigmoid(&mut self, input: Value) -> anyhow::Result<Value> {
        let one = self.constant(input.ty.elem.one_literal(), input.ty.elem)?;
        let zero = self.zero_like(&input.ty)?;
        let neg = self.emit_binary("arith.subf", zero, input)?;
        let exp = self.emit_unary("math.exp", neg)?;
        let denominator = self.emit_binary("arith.addf", one.clone(), exp)?;
        self.emit_binary("arith.divf", one, denominator)
    }

    fn argmax(&mut self, input: Value) -> anyhow::Result<Value> {
        if input.ty.rank() != 1 {
            anyhow::bail!("argmax lowering currently supports rank-1 tensors only");
        }
        let len = input.ty.shape[0];
        if len == 0 {
            anyhow::bail!("argmax lowering expects a non-empty tensor");
        }
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![1],
        };
        let zero = self.fresh();
        self.lines
            .push(format!("    {zero} = arith.constant 0 : index"));
        let one = self.fresh();
        self.lines
            .push(format!("    {one} = arith.constant 1 : index"));
        let upper = self.fresh();
        self.lines
            .push(format!("    {upper} = arith.constant {len} : index"));
        let first = self.fresh();
        self.lines.push(format!(
            "    {first} = tensor.extract {}[{zero}] : {}",
            input.name,
            input.ty.mlir_type()
        ));
        let best_index = self.fresh();
        let best_value = self.fresh();
        let next_value = self.fresh();
        let better = self.fresh();
        let selected_index = self.fresh();
        let selected_value = self.fresh();
        self.lines.push(format!(
            "    {best_index}, {best_value} = scf.for %i = {one} to {upper} step {one} iter_args(%best_i = {zero}, %best_v = {first}) -> (index, {}) {{",
            input.ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {next_value} = tensor.extract {}[%i] : {}",
            input.name,
            input.ty.mlir_type()
        ));
        self.lines.push(format!(
            "      {better} = arith.cmpf ogt, {next_value}, %best_v : {}",
            input.ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {selected_index} = arith.select {better}, %i, %best_i : index"
        ));
        self.lines.push(format!(
            "      {selected_value} = arith.select {better}, {next_value}, %best_v : {}",
            input.ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      scf.yield {selected_index}, {selected_value} : index, {}",
            input.ty.elem.mlir_type()
        ));
        self.lines.push("    }".to_string());
        let index_i64 = self.fresh();
        self.lines.push(format!(
            "    {index_i64} = arith.index_cast {best_index} : index to i64"
        ));
        let index_value = self.fresh();
        let conversion_op = match input.ty.elem {
            ElementType::Bool => "arith.index_cast",
            ElementType::F32 | ElementType::F64 => "arith.uitofp",
            ElementType::F16 | ElementType::BF16 => "arith.uitofp",
            ElementType::I32 | ElementType::I64 => "arith.index_cast",
        };
        self.lines.push(format!(
            "    {index_value} = {conversion_op} {index_i64} : i64 to {}",
            input.ty.elem.mlir_type()
        ));
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.insert {index_value} into {empty}[{zero}] : {}",
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
    }

    fn sum(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        let reducer_op = if input.ty.elem.is_float() {
            "arith.addf"
        } else {
            "arith.addi"
        };
        let initial_value = input.ty.elem.zero_literal();
        self.reduce(input, initial_value, reducer_op, axis, false)
    }

    fn all(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        self.reduce(input, "1", "arith.andi", axis, false)
    }

    fn any(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        self.reduce(input, "0", "arith.ori", axis, false)
    }

    fn max(&mut self, input: Value, axis: Option<usize>, keep_dims: bool) -> anyhow::Result<Value> {
        self.reduce(
            input.clone(),
            min_float_literal(input.ty.elem),
            "arith.maximumf",
            axis,
            keep_dims,
        )
    }

    fn reduce(
        &mut self,
        input: Value,
        initial_value: &str,
        reducer_op: &str,
        axis: Option<usize>,
        keep_dims: bool,
    ) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: reduction_output_shape(&input.ty.shape, axis, keep_dims),
        };
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {initial_value} : {}",
            ty.elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({zero} : {}) outs({empty} : {}) -> {}",
            ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));

        let rank = input.ty.rank();
        let dims = format_dim_list(rank);
        let input_map = format!("({dims})");
        let iterator_types = (0..rank)
            .map(|index| {
                if axis.is_none() || axis == Some(index) {
                    "\"reduction\""
                } else {
                    "\"parallel\""
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let output_map = reduction_output_map(rank, axis, keep_dims);
        let reduced = self.fresh();
        let name = self.fresh();
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {input_map}>, affine_map<({dims}) -> {output_map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterator_types}]"));
        self.lines.push(format!(
            "    }} ins({} : {}) outs({init} : {}) {{",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%value: {}, %acc: {}):",
            ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {reduced} = {reducer_op} %acc, %value : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield {reduced} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value { name, ty })
    }

    fn fresh(&mut self) -> String {
        let name = format!("%{}", self.next_value);
        self.next_value += 1;
        name
    }
}
