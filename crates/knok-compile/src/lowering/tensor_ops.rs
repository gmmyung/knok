use knok_core::TensorType;

use super::lowerer::{
    append_block_op, dense_i64_attr, mlir_element_type, Lowerer, RawValue, Value, ValueKind,
};
use super::shape::{
    axis_broadcast_dimensions, collapse_reassociation_for_removed_axis,
    collapse_reassociation_for_squeezed_broadcast, element_count, ensure_axis_broadcastable,
    ensure_broadcastable, expand_reassociation_for_inserted_axis, reassociation_for_rank,
};

impl Lowerer<'_, '_> {
    pub(super) fn transpose(&mut self, input: Value, axes: &[usize]) -> anyhow::Result<Value> {
        if input.ty.rank() <= 1 {
            return Ok(input);
        }
        if axes.is_empty() {
            let ty = TensorType {
                elem: input.ty.elem,
                shape: input.ty.shape.iter().rev().copied().collect(),
            };
            let axes = (0..input.ty.rank()).rev().collect::<Vec<_>>();
            self.permute(input, &ty, &axes)
        } else {
            self.permute_dims(input, axes)
        }
    }

    pub(super) fn permute_dims(&mut self, input: Value, axes: &[usize]) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: axes.iter().map(|axis| input.ty.shape[*axis]).collect(),
        };
        self.permute(input, &ty, axes)
    }

    pub(super) fn swapaxes(
        &mut self,
        input: Value,
        axis0: usize,
        axis1: usize,
    ) -> anyhow::Result<Value> {
        let mut axes = (0..input.ty.rank()).collect::<Vec<_>>();
        axes.swap(axis0, axis1);
        self.permute_dims(input, &axes)
    }

    pub(super) fn moveaxis(
        &mut self,
        input: Value,
        source: usize,
        destination: usize,
    ) -> anyhow::Result<Value> {
        let mut axes = (0..input.ty.rank()).collect::<Vec<_>>();
        let axis = axes.remove(source);
        axes.insert(destination, axis);
        self.permute_dims(input, &axes)
    }

    pub(super) fn permute(
        &mut self,
        input: Value,
        ty: &TensorType,
        axes: &[usize],
    ) -> anyhow::Result<Value> {
        if axes.iter().copied().eq(0..axes.len()) {
            return Ok(input);
        }
        let empty = self.append_tensor_empty(ty)?;
        let attrs = [dense_i64_attr(self.context, "permutation", axes)?];
        self.append_named_linalg_with_built_attrs("linalg.transpose", &[input], empty, ty, &attrs)
    }

    pub(super) fn reshape(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        if input.ty == *ty {
            return Ok(input);
        }
        if ty.rank() == 0 || input.ty.rank() == 0 {
            return self.splat(input, ty);
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
        let reassociation = reassociation_for_rank(ty.rank());
        self.append_reassociation_op(
            "tensor.expand_shape",
            input,
            ty,
            &reassociation,
            Some(&ty.shape),
        )
    }

    fn collapse_to_rank1(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let reassociation = reassociation_for_rank(input.ty.rank());
        self.append_reassociation_op("tensor.collapse_shape", input, ty, &reassociation, None)
    }

    pub(super) fn slice(
        &mut self,
        input: Value,
        ty: &TensorType,
        starts: &[usize],
    ) -> anyhow::Result<Value> {
        self.append_tensor_extract_slice(input, ty, starts, &ty.shape, &vec![1; ty.rank()])
    }

    pub(super) fn split(
        &mut self,
        input: Value,
        axis: usize,
        sections: &[usize],
    ) -> anyhow::Result<Vec<Value>> {
        let mut offset = 0;
        sections
            .iter()
            .map(|section| {
                let mut starts = vec![0; input.ty.rank()];
                starts[axis] = offset;
                let mut shape = input.ty.shape.clone();
                shape[axis] = *section;
                let ty = TensorType {
                    elem: input.ty.elem,
                    shape,
                };
                offset += *section;
                self.slice(input.clone(), &ty, &starts)
            })
            .collect()
    }

    pub(super) fn take(
        &mut self,
        input: Value,
        axis: usize,
        index: usize,
    ) -> anyhow::Result<Value> {
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
        let output_ty = TensorType {
            elem: sliced.ty.elem,
            shape: output_shape,
        };
        if sliced.ty == output_ty {
            return Ok(sliced);
        }
        if output_ty.rank() == 0 {
            return self.reshape(sliced, &output_ty);
        }
        let reassociation = collapse_reassociation_for_removed_axis(sliced.ty.rank(), axis);
        self.append_reassociation_op(
            "tensor.collapse_shape",
            sliced,
            &output_ty,
            &reassociation,
            None,
        )
    }

    pub(super) fn gather(
        &mut self,
        input: Value,
        indices: Value,
        axis: usize,
        ty: &TensorType,
    ) -> anyhow::Result<Value> {
        let index_rank = indices.ty.rank();
        let index_axes = (axis..axis + index_rank).collect::<Vec<_>>();
        let input_axes = (0..input.ty.rank())
            .map(|input_axis| {
                if input_axis < axis {
                    GatherInputAxis::Output(input_axis)
                } else if input_axis == axis {
                    GatherInputAxis::Index
                } else {
                    GatherInputAxis::Output(input_axis + index_rank - 1)
                }
            })
            .collect::<Vec<_>>();
        let indexed_axis_size = input.ty.shape[axis];
        self.gather_with_maps(
            input,
            indices,
            ty,
            &index_axes,
            &input_axes,
            indexed_axis_size,
        )
    }

    pub(super) fn take_along_axis(
        &mut self,
        input: Value,
        indices: Value,
        axis: usize,
    ) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: indices.ty.shape.clone(),
        };
        let index_axes = (0..indices.ty.rank()).collect::<Vec<_>>();
        let input_axes = (0..input.ty.rank())
            .map(|input_axis| {
                if input_axis == axis {
                    GatherInputAxis::Index
                } else {
                    GatherInputAxis::Output(input_axis)
                }
            })
            .collect::<Vec<_>>();
        let indexed_axis_size = input.ty.shape[axis];
        self.gather_with_maps(
            input,
            indices,
            &ty,
            &index_axes,
            &input_axes,
            indexed_axis_size,
        )
    }

    fn gather_with_maps(
        &mut self,
        input: Value,
        indices: Value,
        ty: &TensorType,
        index_axes: &[usize],
        input_axes: &[GatherInputAxis],
        indexed_axis_size: usize,
    ) -> anyhow::Result<Value> {
        let index_ty = indices.ty.clone();
        let indices = self.coerce_to_kind(indices, &index_ty, ValueKind::Tensor)?;
        if ty.rank() == 0 {
            let index_value = self.extract_index_value(&indices, &[])?;
            let index_value = self.normalize_gather_index(index_value, indexed_axis_size)?;
            let value = self.extract_gather_value(input, &index_value, input_axes)?;
            return self.splat(value, ty);
        }

        let output = self.append_tensor_empty(ty)?;
        let output_map = identity_map(ty.rank());
        let index_map = format_dim_subset(index_axes);
        let iterators = vec!["parallel"; ty.rank()];
        let input_elem = input.ty.elem;
        let input_ty = input.ty.clone();
        let input_raw = input.raw;
        let index_elem_ty = indices.ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[indices],
            &[output],
            &[ty.clone()],
            ty.rank(),
            &[index_map, output_map],
            &iterators,
            |_, block, args| {
                let output_indices = (0..ty.rank())
                    .map(|axis| {
                        append_block_op(
                            context,
                            block,
                            location,
                            "linalg.index",
                            &[],
                            &[melior::ir::Type::index(context)],
                            &[("dim".to_string(), format!("{axis} : i64"))],
                            Vec::new(),
                        )
                        .map(|values| values[0])
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let index_value =
                    index_cast_in_block(context, block, location, args[0], index_elem_ty)?;
                let index_value = normalize_index_in_block(
                    context,
                    block,
                    location,
                    index_value,
                    indexed_axis_size,
                )?;
                let input_indices = input_axes
                    .iter()
                    .map(|axis| match axis {
                        GatherInputAxis::Index => index_value,
                        GatherInputAxis::Output(axis) => output_indices[*axis],
                    })
                    .map(RawValue::as_value)
                    .collect::<Vec<_>>();
                let mut operands = vec![input_raw.as_value()];
                operands.extend(input_indices);
                let result = append_block_op(
                    context,
                    block,
                    location,
                    "tensor.extract",
                    &operands,
                    &[mlir_element_type(context, input_elem)?],
                    &[],
                    Vec::new(),
                )?;
                let _ = input_ty;
                Ok(vec![result[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty.clone()))
    }

    fn extract_index_value(
        &mut self,
        indices: &Value,
        index_axes: &[RawValue],
    ) -> anyhow::Result<RawValue> {
        let index_value = self.append_tensor_extract(indices.clone(), index_axes)?;
        let result_type = melior::ir::Type::index(self.context);
        let results = self.append_op_with_result_types(
            "arith.index_cast",
            &[index_value],
            &[result_type],
            &[],
            Vec::new(),
        )?;
        Ok(results[0])
    }

    fn extract_gather_value(
        &mut self,
        input: Value,
        index_value: &RawValue,
        input_axes: &[GatherInputAxis],
    ) -> anyhow::Result<Value> {
        let zero = self.append_index_constant(0)?;
        let indices = input_axes
            .iter()
            .map(|axis| match axis {
                GatherInputAxis::Index => *index_value,
                GatherInputAxis::Output(_) => zero,
            })
            .collect::<Vec<_>>();
        self.append_tensor_extract(input, &indices)
    }

    fn normalize_gather_index(
        &mut self,
        index_value: RawValue,
        axis_size: usize,
    ) -> anyhow::Result<RawValue> {
        normalize_index_on_main_block(self, index_value, axis_size)
    }

    pub(super) fn concat(&mut self, lhs: Value, rhs: Value, axis: usize) -> anyhow::Result<Value> {
        let mut shape = lhs.ty.shape.clone();
        shape[axis] += rhs.ty.shape[axis];
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape,
        };
        let empty = self.append_tensor_empty(&ty)?;
        let lhs_offsets = vec![0; ty.rank()];
        let first = self.insert_slice(lhs, empty, &ty, &lhs_offsets)?;
        let mut rhs_offsets = vec![0; ty.rank()];
        rhs_offsets[axis] = ty.shape[axis] - rhs.ty.shape[axis];
        self.insert_slice(rhs, first, &ty, &rhs_offsets)
    }

    pub(super) fn stack(&mut self, lhs: Value, rhs: Value, axis: usize) -> anyhow::Result<Value> {
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

    pub(super) fn tile(&mut self, input: Value, multiples: &[usize]) -> anyhow::Result<Value> {
        if input.ty.rank() == 0 {
            return Ok(input);
        }
        let final_ty = TensorType {
            elem: input.ty.elem,
            shape: input
                .ty
                .shape
                .iter()
                .zip(multiples)
                .map(|(dim, multiple)| dim * multiple)
                .collect(),
        };
        if multiples.contains(&0) {
            return self.empty_tensor(&final_ty);
        }
        let mut value = input;
        for (axis, multiple) in multiples.iter().copied().enumerate() {
            if multiple <= 1 {
                continue;
            }
            let pieces = (0..multiple).map(|_| value.clone()).collect::<Vec<_>>();
            value = self.concat_many(pieces, axis)?;
        }
        Ok(value)
    }

    pub(super) fn repeat(
        &mut self,
        input: Value,
        axis: usize,
        count: usize,
    ) -> anyhow::Result<Value> {
        let mut output_ty = input.ty.clone();
        output_ty.shape[axis] *= count;
        if count == 0 || input.ty.shape[axis] == 0 {
            return self.empty_tensor(&output_ty);
        }
        if count == 1 {
            return Ok(input);
        }

        let mut pieces = Vec::new();
        for index in 0..input.ty.shape[axis] {
            let mut starts = vec![0; input.ty.rank()];
            starts[axis] = index;
            let mut unit_shape = input.ty.shape.clone();
            unit_shape[axis] = 1;
            let unit_ty = TensorType {
                elem: input.ty.elem,
                shape: unit_shape,
            };
            let unit = self.slice(input.clone(), &unit_ty, &starts)?;
            pieces.push(self.concat_many((0..count).map(|_| unit.clone()).collect(), axis)?);
        }
        self.concat_many(pieces, axis)
    }

    pub(super) fn flip(&mut self, input: Value, axes: &[usize]) -> anyhow::Result<Value> {
        let axes = if axes.is_empty() {
            (0..input.ty.rank()).collect::<Vec<_>>()
        } else {
            axes.to_vec()
        };
        let mut value = input;
        for axis in axes {
            let dim = value.ty.shape[axis];
            if dim <= 1 {
                continue;
            }
            let mut pieces = Vec::with_capacity(dim);
            for index in (0..dim).rev() {
                let mut starts = vec![0; value.ty.rank()];
                starts[axis] = index;
                let mut unit_shape = value.ty.shape.clone();
                unit_shape[axis] = 1;
                let unit_ty = TensorType {
                    elem: value.ty.elem,
                    shape: unit_shape,
                };
                pieces.push(self.slice(value.clone(), &unit_ty, &starts)?);
            }
            value = self.concat_many(pieces, axis)?;
        }
        Ok(value)
    }

    pub(super) fn roll(
        &mut self,
        input: Value,
        axis: usize,
        shift: usize,
    ) -> anyhow::Result<Value> {
        let dim = input.ty.shape[axis];
        if dim <= 1 {
            return Ok(input);
        }
        let offset = shift % dim;
        if offset == 0 {
            return Ok(input);
        }
        let split = dim - offset;
        let mut high_starts = vec![0; input.ty.rank()];
        high_starts[axis] = split;
        let mut high_shape = input.ty.shape.clone();
        high_shape[axis] = offset;
        let high_ty = TensorType {
            elem: input.ty.elem,
            shape: high_shape,
        };
        let high = self.slice(input.clone(), &high_ty, &high_starts)?;

        let low_starts = vec![0; input.ty.rank()];
        let mut low_shape = input.ty.shape.clone();
        low_shape[axis] = split;
        let low_ty = TensorType {
            elem: input.ty.elem,
            shape: low_shape,
        };
        let low = self.slice(input, &low_ty, &low_starts)?;
        self.concat(high, low, axis)
    }

    pub(super) fn pad(
        &mut self,
        input: Value,
        ty: &TensorType,
        lows: &[usize],
    ) -> anyhow::Result<Value> {
        let highs = ty
            .shape
            .iter()
            .zip(&input.ty.shape)
            .zip(lows)
            .map(|((output, input), low)| output - input - low)
            .collect::<Vec<_>>();
        if lows.iter().all(|low| *low == 0) && highs.iter().all(|high| *high == 0) {
            return Ok(input);
        }
        let zero = self.constant(ty.elem.zero_literal(), ty.elem)?;
        self.append_tensor_pad(input, ty, lows, &highs, zero)
    }

    fn concat_many(&mut self, mut values: Vec<Value>, axis: usize) -> anyhow::Result<Value> {
        let Some(first) = values.first().cloned() else {
            anyhow::bail!("internal error: concat_many needs at least one tensor");
        };
        values
            .drain(1..)
            .try_fold(first, |acc, value| self.concat(acc, value, axis))
    }

    fn empty_tensor(&mut self, ty: &TensorType) -> anyhow::Result<Value> {
        self.append_tensor_empty(ty)
    }

    fn insert_slice(
        &mut self,
        source: Value,
        dest: Value,
        dest_ty: &TensorType,
        offsets: &[usize],
    ) -> anyhow::Result<Value> {
        self.append_tensor_insert_slice(source, dest, dest_ty, offsets)
    }

    fn expand_insert_axis(
        &mut self,
        input: Value,
        ty: &TensorType,
        axis: usize,
    ) -> anyhow::Result<Value> {
        let reassociation = expand_reassociation_for_inserted_axis(input.ty.rank(), axis);
        self.append_reassociation_op(
            "tensor.expand_shape",
            input,
            ty,
            &reassociation,
            Some(&ty.shape),
        )
    }

    pub(super) fn broadcast(&mut self, input: Value, ty: &TensorType) -> anyhow::Result<Value> {
        if input.ty == *ty && input.kind == ValueKind::Tensor {
            return Ok(input);
        }
        if input.kind == ValueKind::Scalar {
            return self.splat(input, ty);
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

    pub(super) fn broadcast_along_axis(
        &mut self,
        input: Value,
        ty: &TensorType,
        axis: usize,
    ) -> anyhow::Result<Value> {
        if input.ty == *ty && input.kind == ValueKind::Tensor {
            return Ok(input);
        }
        if input.kind == ValueKind::Scalar {
            return self.splat(input, ty);
        }
        if input.ty.rank() == 0 || element_count(&input.ty) == 1 {
            let scalar = self.coerce_to_scalar(input)?;
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
        let empty = self.append_tensor_empty(ty)?;
        let attrs = [dense_i64_attr(self.context, "dimensions", dimensions)?];
        self.append_named_linalg_with_built_attrs("linalg.broadcast", &[input], empty, ty, &attrs)
    }

    pub(super) fn extract_first_scalar(&mut self, input: Value) -> anyhow::Result<Value> {
        let zero = self.append_index_constant(0)?;
        let indices = vec![zero; input.ty.rank()];
        self.append_tensor_extract(input, &indices)
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
        let squeezed = self.append_reassociation_op(
            "tensor.collapse_shape",
            input,
            &squeezed_ty,
            &reassociation,
            None,
        )?;
        Ok((squeezed, dimensions))
    }
}

enum GatherInputAxis {
    Index,
    Output(usize),
}

fn format_dim_subset(axes: &[usize]) -> String {
    let dims = axes
        .iter()
        .map(|axis| format!("d{axis}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("({dims})")
}

fn identity_map(rank: usize) -> String {
    if rank == 0 {
        return "()".to_string();
    }
    format!(
        "({})",
        (0..rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn index_cast_in_block<'c>(
    context: &'c melior::Context,
    block: &melior::ir::Block<'c>,
    location: melior::ir::Location<'c>,
    value: melior::ir::Value<'c, '_>,
    _elem: knok_core::ElementType,
) -> anyhow::Result<RawValue> {
    let result = append_block_op(
        context,
        block,
        location,
        "arith.index_cast",
        &[value],
        &[melior::ir::Type::index(context)],
        &[],
        Vec::new(),
    )?;
    Ok(result[0])
}

fn normalize_index_on_main_block(
    lowerer: &mut Lowerer<'_, '_>,
    index_value: RawValue,
    axis_size: usize,
) -> anyhow::Result<RawValue> {
    normalize_index_in_block(
        lowerer.context,
        &lowerer.block,
        lowerer.location,
        index_value,
        axis_size,
    )
}

fn normalize_index_in_block<'c>(
    context: &'c melior::Context,
    block: &melior::ir::Block<'c>,
    location: melior::ir::Location<'c>,
    index_value: RawValue,
    axis_size: usize,
) -> anyhow::Result<RawValue> {
    let index_ty = melior::ir::Type::index(context);
    let i1_ty = mlir_element_type(context, knok_core::ElementType::Bool)?;
    let zero = append_block_op(
        context,
        block,
        location,
        "arith.constant",
        &[],
        &[index_ty],
        &[("value".to_string(), "0 : index".to_string())],
        Vec::new(),
    )?[0];
    let dim = append_block_op(
        context,
        block,
        location,
        "arith.constant",
        &[],
        &[index_ty],
        &[("value".to_string(), format!("{axis_size} : index"))],
        Vec::new(),
    )?[0];
    let is_negative = append_block_op(
        context,
        block,
        location,
        "arith.cmpi",
        &[index_value.as_value(), zero.as_value()],
        &[i1_ty],
        &[("predicate".to_string(), "2 : i64".to_string())],
        Vec::new(),
    )?[0];
    let wrapped = append_block_op(
        context,
        block,
        location,
        "arith.addi",
        &[index_value.as_value(), dim.as_value()],
        &[index_ty],
        &[],
        Vec::new(),
    )?[0];
    let normalized = append_block_op(
        context,
        block,
        location,
        "arith.select",
        &[
            is_negative.as_value(),
            wrapped.as_value(),
            index_value.as_value(),
        ],
        &[index_ty],
        &[],
        Vec::new(),
    )?[0];
    let lower_ok = append_block_op(
        context,
        block,
        location,
        "arith.cmpi",
        &[normalized.as_value(), zero.as_value()],
        &[i1_ty],
        &[("predicate".to_string(), "5 : i64".to_string())],
        Vec::new(),
    )?[0];
    let upper_ok = append_block_op(
        context,
        block,
        location,
        "arith.cmpi",
        &[normalized.as_value(), dim.as_value()],
        &[i1_ty],
        &[("predicate".to_string(), "2 : i64".to_string())],
        Vec::new(),
    )?[0];
    let in_bounds = append_block_op(
        context,
        block,
        location,
        "arith.andi",
        &[lower_ok.as_value(), upper_ok.as_value()],
        &[i1_ty],
        &[],
        Vec::new(),
    )?[0];
    append_block_op(
        context,
        block,
        location,
        "cf.assert",
        &[in_bounds.as_value()],
        &[],
        &[("msg".to_string(), "\"index out of bounds\"".to_string())],
        Vec::new(),
    )?;
    Ok(normalized)
}
