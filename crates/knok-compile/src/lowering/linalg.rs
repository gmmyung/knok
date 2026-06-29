use super::lowerer::{append_block_op, mlir_element_type, Lowerer, RawValue, Value, ValueKind};
use super::shape::element_count;
use knok_core::{BinaryOp, TensorType};

impl Lowerer<'_, '_> {
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
        let empty = self.append_tensor_empty(&ty)?;
        let mul_op = if ty.elem.is_float() {
            "arith.mulf"
        } else {
            "arith.muli"
        };
        let elem = ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[lhs, rhs],
            &[empty],
            &[ty.clone()],
            2,
            &[
                "(d0)".to_string(),
                "(d1)".to_string(),
                "(d0, d1)".to_string(),
            ],
            &["parallel", "parallel"],
            |_, block, args| {
                let elem_ty = mlir_element_type(context, elem)?;
                let product = append_block_op(
                    context,
                    block,
                    location,
                    mul_op,
                    &[args[0], args[1]],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?;
                Ok(vec![product[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty))
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
        let output = self.append_tensor_empty(&ty)?;
        let iterators = vec!["parallel"; ty.rank()];
        let raw = self.append_linalg_generic(
            &[input],
            &[output],
            &[ty.clone()],
            ty.rank(),
            &[affine_tuple(&input_indices), affine_tuple(&output_indices)],
            &iterators,
            |_, _, args| Ok(vec![RawValue::from_value(args[0])]),
        )?;
        Ok(Value::tensor(raw[0], ty))
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
        let output_indices = (0..output_rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        let mut iterators = vec!["parallel"; output_rank];
        iterators.push("reduction");
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
        let elem = ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[lhs, rhs],
            &[init],
            &[ty.clone()],
            loop_rank,
            &[
                affine_tuple(lhs_indices),
                affine_tuple(rhs_indices),
                affine_tuple(&output_indices),
            ],
            &iterators,
            |_, block, args| {
                let elem_ty = mlir_element_type(context, elem)?;
                let product = append_block_op(
                    context,
                    block,
                    location,
                    mul_op,
                    &[args[0], args[1]],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?;
                let sum = append_block_op(
                    context,
                    block,
                    location,
                    add_op,
                    &[args[2], product[0].as_value()],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?;
                Ok(vec![sum[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty))
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
        let output_indices = (0..output_rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        let mut iterators = vec!["parallel"; output_rank];
        iterators.push("reduction");
        let add_op = if ty.elem.is_float() {
            "arith.addf"
        } else {
            "arith.addi"
        };
        let elem = ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[input],
            &[init],
            &[ty.clone()],
            loop_rank,
            &[affine_tuple(input_indices), affine_tuple(&output_indices)],
            &iterators,
            |_, block, args| {
                let elem_ty = mlir_element_type(context, elem)?;
                let sum = append_block_op(
                    context,
                    block,
                    location,
                    add_op,
                    &[args[1], args[0]],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?;
                Ok(vec![sum[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty))
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

fn affine_tuple(indices: &[String]) -> String {
    if indices.is_empty() {
        "()".to_string()
    } else {
        format!("({})", indices.join(", "))
    }
}
