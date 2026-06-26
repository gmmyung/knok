use knok_core::{ElementType, TensorType};

use super::lowerer::{Lowerer, Value};
use super::shape::{
    element_count, format_dim_list, min_float_literal, reduction_output_map, reduction_output_shape,
};

impl Lowerer<'_> {
    pub(super) fn mean(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "mean")?;
        let count = axis.map_or_else(|| element_count(&input.ty), |axis| input.ty.shape[axis]);
        let elem = input.ty.elem;
        let sum = self.sum(input, axis)?;
        let scale = self.constant(&format!("{count}.0"), elem)?;
        self.emit_binary("arith.divf", sum, scale)
    }

    pub(super) fn softmax(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "softmax")?;
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
        Ok(Value::tensor(name, input.ty))
    }

    pub(super) fn sigmoid(&mut self, input: Value) -> anyhow::Result<Value> {
        let one = self.constant(input.ty.elem.one_literal(), input.ty.elem)?;
        let zero = self.zero_like(&input.ty)?;
        let neg = self.emit_binary("arith.subf", zero, input)?;
        let exp = self.emit_unary("math.exp", neg)?;
        let denominator = self.emit_binary("arith.addf", one.clone(), exp)?;
        self.emit_binary("arith.divf", one, denominator)
    }

    pub(super) fn argmax(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "argmax")?;
        let index_ty = TensorType {
            elem: ElementType::I64,
            shape: reduction_output_shape(&input.ty.shape, axis, false),
        };
        let value_ty = TensorType {
            elem: input.ty.elem,
            shape: index_ty.shape.clone(),
        };
        let valid_ty = TensorType {
            elem: ElementType::Bool,
            shape: index_ty.shape.clone(),
        };

        let init_index = self.fill_tensor(&index_ty, "0", ElementType::I64);
        let init_value =
            self.fill_tensor(&value_ty, min_numeric_literal(input.ty.elem), input.ty.elem);
        let init_valid = self.fill_tensor(&valid_ty, "0", ElementType::Bool);

        let rank = input.ty.rank();
        let dims = format_dim_list(rank);
        let input_map = format!("({dims})");
        let output_map = reduction_output_map(rank, axis, false);
        let iterator_types = reduction_iterator_types(rank, axis);
        let result = self.fresh();
        self.lines
            .push(format!("    {result}:3 = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {input_map}>, affine_map<({dims}) -> {output_map}>, affine_map<({dims}) -> {output_map}>, affine_map<({dims}) -> {output_map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterator_types}]"));
        self.lines.push(format!(
            "    }} ins({} : {}) outs({}, {}, {} : {}, {}, {}) {{",
            input.name,
            input.ty.mlir_type(),
            init_index,
            init_value,
            init_valid,
            index_ty.mlir_type(),
            value_ty.mlir_type(),
            valid_ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%value: {}, %best_i: i64, %best_v: {}, %valid: i1):",
            input.ty.elem.mlir_type(),
            input.ty.elem.mlir_type()
        ));
        let candidate_index = self.argmax_candidate_index(&input.ty, axis);
        let better = self.fresh();
        let compare_op = if input.ty.elem.is_float() {
            "arith.cmpf"
        } else {
            "arith.cmpi"
        };
        let greater_predicate = if input.ty.elem.is_float() {
            "ogt"
        } else {
            "sgt"
        };
        let equal_predicate = if input.ty.elem.is_float() {
            "oeq"
        } else {
            "eq"
        };
        let greater = self.fresh();
        self.lines.push(format!(
            "      {greater} = {compare_op} {greater_predicate}, %value, %best_v : {}",
            input.ty.elem.mlir_type()
        ));
        let equal = self.fresh();
        self.lines.push(format!(
            "      {equal} = {compare_op} {equal_predicate}, %value, %best_v : {}",
            input.ty.elem.mlir_type()
        ));
        let lower_index = self.fresh();
        self.lines.push(format!(
            "      {lower_index} = arith.cmpi slt, {candidate_index}, %best_i : i64"
        ));
        let tied_before = self.fresh();
        self.lines.push(format!(
            "      {tied_before} = arith.andi {equal}, {lower_index} : i1"
        ));
        let candidate_better = self.fresh();
        self.lines.push(format!(
            "      {candidate_better} = arith.ori {greater}, {tied_before} : i1"
        ));
        let true_value = self.fresh();
        self.lines
            .push(format!("      {true_value} = arith.constant 1 : i1"));
        let not_valid = self.fresh();
        self.lines.push(format!(
            "      {not_valid} = arith.xori %valid, {true_value} : i1"
        ));
        self.lines.push(format!(
            "      {better} = arith.ori {not_valid}, {candidate_better} : i1"
        ));
        let selected_index = self.fresh();
        self.lines.push(format!(
            "      {selected_index} = arith.select {better}, {candidate_index}, %best_i : i64"
        ));
        let selected_value = self.fresh();
        self.lines.push(format!(
            "      {selected_value} = arith.select {better}, %value, %best_v : {}",
            input.ty.elem.mlir_type()
        ));
        let selected_valid = self.fresh();
        self.lines.push(format!(
            "      {selected_valid} = arith.ori %valid, {better} : i1"
        ));
        self.lines.push(format!(
            "      linalg.yield {selected_index}, {selected_value}, {selected_valid} : i64, {}, i1",
            input.ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "    }} -> ({}, {}, {})",
            index_ty.mlir_type(),
            value_ty.mlir_type(),
            valid_ty.mlir_type()
        ));
        Ok(Value {
            name: format!("{result}#0"),
            ty: index_ty,
            kind: super::lowerer::ValueKind::Tensor,
        })
    }

    fn argmax_candidate_index(&mut self, input: &TensorType, axis: Option<usize>) -> String {
        let index = if let Some(axis) = axis {
            let index = self.fresh();
            self.lines
                .push(format!("      {index} = linalg.index {axis} : index"));
            index
        } else {
            let first = self.fresh();
            self.lines
                .push(format!("      {first} = linalg.index 0 : index"));
            let mut flattened = first;
            for axis in 1..input.rank() {
                let axis_index = self.fresh();
                self.lines
                    .push(format!("      {axis_index} = linalg.index {axis} : index"));
                let dim_size = self.fresh();
                self.lines.push(format!(
                    "      {dim_size} = arith.constant {} : index",
                    input.shape[axis]
                ));
                let scaled = self.fresh();
                self.lines.push(format!(
                    "      {scaled} = arith.muli {flattened}, {dim_size} : index"
                ));
                let next = self.fresh();
                self.lines.push(format!(
                    "      {next} = arith.addi {scaled}, {axis_index} : index"
                ));
                flattened = next;
            }
            flattened
        };
        let index_i64 = self.fresh();
        self.lines.push(format!(
            "      {index_i64} = arith.index_cast {index} : index to i64"
        ));
        index_i64
    }

    pub(super) fn sum(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        let reducer_op = if input.ty.elem.is_float() {
            "arith.addf"
        } else {
            "arith.addi"
        };
        let initial_value = input.ty.elem.zero_literal();
        self.reduce(input, initial_value, reducer_op, axis, false)
    }

    pub(super) fn all(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        self.reduce(input, "1", "arith.andi", axis, false)
    }

    pub(super) fn any(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
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
        let init = self.fill_tensor(&ty, initial_value, ty.elem);

        let rank = input.ty.rank();
        let dims = format_dim_list(rank);
        let input_map = format!("({dims})");
        let iterator_types = reduction_iterator_types(rank, axis);
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
        Ok(Value::tensor(name, ty))
    }

    fn fill_tensor(&mut self, ty: &TensorType, value: &str, elem: ElementType) -> String {
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let scalar = self.fresh();
        self.lines.push(format!(
            "    {scalar} = arith.constant {value} : {}",
            elem.mlir_type()
        ));
        let init = self.fresh();
        self.lines.push(format!(
            "    {init} = linalg.fill ins({scalar} : {}) outs({empty} : {}) -> {}",
            elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        init
    }
}

fn reduction_iterator_types(rank: usize, axis: Option<usize>) -> String {
    (0..rank)
        .map(|index| {
            if axis.is_none() || axis == Some(index) {
                "\"reduction\""
            } else {
                "\"parallel\""
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn min_numeric_literal(elem: ElementType) -> &'static str {
    match elem {
        ElementType::Bool => "0",
        ElementType::F32 => "-3.40282347E+38",
        ElementType::F64 => "-1.7976931348623157E+308",
        ElementType::F16 => "-65504.0",
        ElementType::BF16 => "-3.38953139E+38",
        ElementType::I32 => "-2147483648",
        ElementType::I64 => "-9223372036854775808",
    }
}

fn ensure_non_empty_reduction(
    input: &TensorType,
    axis: Option<usize>,
    op_name: &str,
) -> anyhow::Result<()> {
    match axis {
        Some(axis) if axis >= input.rank() => {
            anyhow::bail!(
                "{op_name} axis {axis} is out of bounds for rank-{} tensor {:?}",
                input.rank(),
                input.shape
            );
        }
        Some(axis) if input.shape[axis] == 0 => {
            anyhow::bail!(
                "{op_name} cannot reduce empty axis {axis} for tensor shape {:?}",
                input.shape
            );
        }
        None if element_count(input) == 0 => {
            anyhow::bail!(
                "{op_name} cannot reduce empty tensor shape {:?}",
                input.shape
            );
        }
        _ => Ok(()),
    }
}
