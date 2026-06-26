use knok_core::{ElementType, TensorType};

use super::lowerer::{Lowerer, Value};
use super::shape::{
    element_count, format_dim_list, min_float_literal, reduction_output_map, reduction_output_shape,
};

impl Lowerer<'_> {
    pub(super) fn mean(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
        let count = axis.map_or_else(|| element_count(&input.ty), |axis| input.ty.shape[axis]);
        let elem = input.ty.elem;
        let sum = self.sum(input, axis)?;
        let scale = self.constant(&format!("{count}.0"), elem)?;
        self.emit_binary("arith.divf", sum, scale)
    }

    pub(super) fn softmax(&mut self, input: Value, axis: Option<usize>) -> anyhow::Result<Value> {
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

    pub(super) fn sigmoid(&mut self, input: Value) -> anyhow::Result<Value> {
        let one = self.constant(input.ty.elem.one_literal(), input.ty.elem)?;
        let zero = self.zero_like(&input.ty)?;
        let neg = self.emit_binary("arith.subf", zero, input)?;
        let exp = self.emit_unary("math.exp", neg)?;
        let denominator = self.emit_binary("arith.addf", one.clone(), exp)?;
        self.emit_binary("arith.divf", one, denominator)
    }

    pub(super) fn argmax(&mut self, input: Value) -> anyhow::Result<Value> {
        if input.ty.rank() != 1 {
            anyhow::bail!("argmax lowering currently supports rank-1 tensors only");
        }
        let len = input.ty.shape[0];
        if len == 0 {
            anyhow::bail!("argmax lowering expects a non-empty tensor");
        }
        let ty = TensorType {
            elem: ElementType::I64,
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
        let compare_op = if input.ty.elem.is_float() {
            "arith.cmpf"
        } else {
            "arith.cmpi"
        };
        let compare_predicate = if input.ty.elem.is_float() {
            "ogt"
        } else {
            "sgt"
        };
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
            "      {better} = {compare_op} {compare_predicate}, {next_value}, %best_v : {}",
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
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.insert {index_i64} into {empty}[{zero}] : {}",
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
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
}
