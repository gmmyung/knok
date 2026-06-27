use knok_core::{BinaryOp, TensorType};

use super::lowerer::{Lowerer, Value, ValueKind};
use super::shape::{element_count, format_dim_list, parallel_iterators};

impl Lowerer<'_> {
    pub(super) fn dot(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        self.inner(lhs, rhs)
    }

    pub(super) fn vecdot(
        &mut self,
        lhs: Value,
        rhs: Value,
        axis: Option<usize>,
    ) -> anyhow::Result<Value> {
        let axis = axis.unwrap_or(lhs.ty.rank() - 1);
        let mut ty = lhs.ty.clone();
        ty.shape.remove(axis);
        let output_rank = ty.rank();
        let reduction_dim = format!("d{output_rank}");
        let mut input_indices = Vec::with_capacity(lhs.ty.rank());
        let mut output_axis = 0;
        for input_axis in 0..lhs.ty.rank() {
            if input_axis == axis {
                input_indices.push(reduction_dim.clone());
            } else {
                input_indices.push(format!("d{output_axis}"));
                output_axis += 1;
            }
        }
        self.emit_two_input_reduction(
            lhs,
            rhs,
            ty,
            &input_indices,
            &input_indices,
            output_rank + 1,
        )
    }

    pub(super) fn inner(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        if lhs.ty.rank() == 0 || rhs.ty.rank() == 0 {
            return self.binary_value(BinaryOp::Mul, lhs, rhs);
        }
        let lhs_prefix_rank = lhs.ty.rank() - 1;
        let rhs_prefix_rank = rhs.ty.rank() - 1;
        let mut shape = lhs.ty.shape[..lhs_prefix_rank].to_vec();
        shape.extend_from_slice(&rhs.ty.shape[..rhs_prefix_rank]);
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape,
        };
        let output_rank = ty.rank();
        let reduction_dim = format!("d{output_rank}");
        let mut lhs_indices = (0..lhs_prefix_rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        lhs_indices.push(reduction_dim.clone());
        let mut rhs_indices = (0..rhs_prefix_rank)
            .map(|axis| format!("d{}", lhs_prefix_rank + axis))
            .collect::<Vec<_>>();
        rhs_indices.push(reduction_dim);
        self.emit_two_input_reduction(lhs, rhs, ty, &lhs_indices, &rhs_indices, output_rank + 1)
    }

    pub(super) fn outer(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let lhs_flat_ty = TensorType {
            elem: lhs.ty.elem,
            shape: vec![element_count(&lhs.ty)],
        };
        let rhs_flat_ty = TensorType {
            elem: rhs.ty.elem,
            shape: vec![element_count(&rhs.ty)],
        };
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape: vec![lhs_flat_ty.shape[0], rhs_flat_ty.shape[0]],
        };
        let lhs = self.reshape(lhs, &lhs_flat_ty)?;
        let rhs = self.reshape(rhs, &rhs_flat_ty)?;
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let product = self.fresh();
        let mul_op = if ty.elem.is_float() {
            "arith.mulf"
        } else {
            "arith.muli"
        };
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],"
        ));
        self.lines
            .push("      iterator_types = [\"parallel\", \"parallel\"]".to_string());
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
            ty.elem.mlir_type(),
            ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {product} = {mul_op} %lhs, %rhs : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield {product} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value::tensor(name, ty))
    }

    pub(super) fn trace(
        &mut self,
        input: Value,
        axes: Option<[usize; 2]>,
    ) -> anyhow::Result<Value> {
        let [axis0, axis1] = axes.unwrap_or([input.ty.rank() - 2, input.ty.rank() - 1]);
        let ty = TensorType {
            elem: input.ty.elem,
            shape: input
                .ty
                .shape
                .iter()
                .enumerate()
                .filter_map(|(axis, dim)| (axis != axis0 && axis != axis1).then_some(*dim))
                .collect(),
        };
        let output_rank = ty.rank();
        let reduction_dim = format!("d{output_rank}");
        let mut input_indices = Vec::with_capacity(input.ty.rank());
        let mut output_axis = 0;
        for input_axis in 0..input.ty.rank() {
            if input_axis == axis0 || input_axis == axis1 {
                input_indices.push(reduction_dim.clone());
            } else {
                input_indices.push(format!("d{output_axis}"));
                output_axis += 1;
            }
        }
        self.emit_one_input_reduction(input, ty, &input_indices, output_rank + 1)
    }

    pub(super) fn diagonal(
        &mut self,
        input: Value,
        axes: Option<[usize; 2]>,
    ) -> anyhow::Result<Value> {
        let [axis0, axis1] = axes.unwrap_or([input.ty.rank() - 2, input.ty.rank() - 1]);
        let mut shape = input
            .ty
            .shape
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| (axis != axis0 && axis != axis1).then_some(*dim))
            .collect::<Vec<_>>();
        shape.push(input.ty.shape[axis0]);
        let ty = TensorType {
            elem: input.ty.elem,
            shape,
        };
        let input = self.ensure_tensor_value(input)?;
        let diagonal_axis = ty.rank() - 1;
        let mut input_indices = Vec::with_capacity(input.ty.rank());
        let mut output_axis = 0;
        for input_axis in 0..input.ty.rank() {
            if input_axis == axis0 || input_axis == axis1 {
                input_indices.push(format!("d{diagonal_axis}"));
            } else {
                input_indices.push(format!("d{output_axis}"));
                output_axis += 1;
            }
        }
        let output_indices = (0..ty.rank())
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let dims = format_dim_list(ty.rank());
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {}>, affine_map<({dims}) -> {}>],",
            affine_tuple(&input_indices),
            affine_tuple(&output_indices)
        ));
        self.lines.push(format!(
            "      iterator_types = [{}]",
            parallel_iterators(ty.rank())
        ));
        self.lines.push(format!(
            "    }} ins({} : {}) outs({empty} : {}) {{",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%value: {}, %out: {}):",
            ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield %value : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value::tensor(name, ty))
    }

    fn emit_two_input_reduction(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: TensorType,
        lhs_indices: &[String],
        rhs_indices: &[String],
        loop_rank: usize,
    ) -> anyhow::Result<Value> {
        let lhs = self.ensure_tensor_value(lhs)?;
        let rhs = self.ensure_tensor_value(rhs)?;
        let init = self.zero_initialized_tensor(&ty)?;
        let output_rank = ty.rank();
        let dims = format_dim_list(loop_rank);
        let output_indices = (0..output_rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        let iterators = contraction_iterators(output_rank);
        let name = self.fresh();
        let product = self.fresh();
        let sum = self.fresh();
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
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {}>, affine_map<({dims}) -> {}>, affine_map<({dims}) -> {}>],",
            affine_tuple(lhs_indices),
            affine_tuple(rhs_indices),
            affine_tuple(&output_indices)
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
        self.lines.push(format!(
            "    }} ins({}, {} : {}, {}) outs({init} : {}) {{",
            lhs.name,
            rhs.name,
            lhs.ty.mlir_type(),
            rhs.ty.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%lhs: {}, %rhs: {}, %acc: {}):",
            ty.elem.mlir_type(),
            ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {product} = {mul_op} %lhs, %rhs : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {sum} = {add_op} %acc, {product} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield {sum} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value::tensor(name, ty))
    }

    fn emit_one_input_reduction(
        &mut self,
        input: Value,
        ty: TensorType,
        input_indices: &[String],
        loop_rank: usize,
    ) -> anyhow::Result<Value> {
        let input = self.ensure_tensor_value(input)?;
        let init = self.zero_initialized_tensor(&ty)?;
        let output_rank = ty.rank();
        let dims = format_dim_list(loop_rank);
        let output_indices = (0..output_rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        let iterators = contraction_iterators(output_rank);
        let name = self.fresh();
        let sum = self.fresh();
        let add_op = if ty.elem.is_float() {
            "arith.addf"
        } else {
            "arith.addi"
        };
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {}>, affine_map<({dims}) -> {}>],",
            affine_tuple(input_indices),
            affine_tuple(&output_indices)
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
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
            "      {sum} = {add_op} %acc, %value : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield {sum} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value::tensor(name, ty))
    }

    fn ensure_tensor_value(&mut self, value: Value) -> anyhow::Result<Value> {
        match value.kind {
            ValueKind::Tensor => Ok(value),
            ValueKind::Scalar => {
                let ty = value.ty.clone();
                self.splat(value, &ty)
            }
        }
    }
}

fn contraction_iterators(output_rank: usize) -> String {
    let mut values = vec!["\"parallel\""; output_rank];
    values.push("\"reduction\"");
    values.join(", ")
}

fn affine_tuple(indices: &[String]) -> String {
    if indices.is_empty() {
        "()".to_string()
    } else {
        format!("({})", indices.join(", "))
    }
}
