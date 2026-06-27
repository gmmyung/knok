use knok_core::TensorType;

use super::lowerer::{Lowerer, Value, ValueKind};
use super::shape::{
    axis_broadcast_dimensions, collapse_reassociation_for_removed_axis,
    collapse_reassociation_for_squeezed_broadcast, element_count, ensure_axis_broadcastable,
    ensure_broadcastable, expand_reassociation_for_inserted_axis, format_dim_list,
    format_shape_list, format_usize_list, parallel_iterators, reassociation_for_rank,
};

impl Lowerer<'_> {
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
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let permutation = format_usize_list(axes);
        self.lines.push(format!(
            "    {name} = linalg.transpose ins({} : {}) outs({empty} : {}) permutation = {permutation}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value::tensor(name, ty.clone()))
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
        let name = self.fresh();
        let output_shape = format_shape_list(&ty.shape);
        let reassociation = reassociation_for_rank(ty.rank());
        self.lines.push(format!(
            "    {name} = tensor.expand_shape {} {reassociation} output_shape {output_shape} : {} into {}",
            input.name,
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value::tensor(name, ty.clone()))
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
        Ok(Value::tensor(name, ty.clone()))
    }

    pub(super) fn slice(
        &mut self,
        input: Value,
        ty: &TensorType,
        starts: &[usize],
    ) -> anyhow::Result<Value> {
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
        Ok(Value::tensor(name, ty.clone()))
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
        let name = self.fresh();
        let reassociation = collapse_reassociation_for_removed_axis(sliced.ty.rank(), axis);
        self.lines.push(format!(
            "    {name} = tensor.collapse_shape {} {reassociation} : {} into {}",
            sliced.name,
            sliced.ty.mlir_type(),
            output_ty.mlir_type()
        ));
        Ok(Value::tensor(name, output_ty))
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
            let index_value = self.normalize_gather_index(&index_value, indexed_axis_size, "    ");
            let value = self.extract_gather_value(input, &index_value, input_axes)?;
            return self.splat(value, ty);
        }

        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let dims = format_dim_list(ty.rank());
        let output_map = format!("({dims})");
        let index_map = format_dim_subset(index_axes);
        let iterators = parallel_iterators(ty.rank());
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {index_map}>, affine_map<({dims}) -> {output_map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
        self.lines.push(format!(
            "    }} ins({} : {}) outs({empty} : {}) {{",
            indices.name,
            indices.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%index_value: {}, %out: {}):",
            indices.ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        let output_indices = (0..ty.rank())
            .map(|axis| {
                let value = self.fresh();
                self.lines
                    .push(format!("      {value} = linalg.index {axis} : index"));
                value
            })
            .collect::<Vec<_>>();
        let index_value = self.index_cast("%index_value", indices.ty.elem.mlir_type(), "      ");
        let index_value = self.normalize_gather_index(&index_value, indexed_axis_size, "      ");
        let input_indices = input_axes
            .iter()
            .map(|axis| match axis {
                GatherInputAxis::Index => index_value.clone(),
                GatherInputAxis::Output(axis) => output_indices[*axis].clone(),
            })
            .collect::<Vec<_>>();
        let result = self.fresh();
        self.lines.push(format!(
            "      {result} = tensor.extract {}[{}] : {}",
            input.name,
            input_indices.join(", "),
            input.ty.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield {result} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value::tensor(name, ty.clone()))
    }

    fn extract_index_value(
        &mut self,
        indices: &Value,
        index_axes: &[String],
    ) -> anyhow::Result<String> {
        let index_value = self.fresh();
        self.lines.push(format!(
            "    {index_value} = tensor.extract {}[{}] : {}",
            indices.name,
            index_axes.join(", "),
            indices.ty.mlir_type()
        ));
        Ok(self.index_cast(&index_value, indices.ty.elem.mlir_type(), "    "))
    }

    fn extract_gather_value(
        &mut self,
        input: Value,
        index_value: &str,
        input_axes: &[GatherInputAxis],
    ) -> anyhow::Result<Value> {
        let zero = self.fresh();
        self.lines
            .push(format!("    {zero} = arith.constant 0 : index"));
        let indices = input_axes
            .iter()
            .map(|axis| match axis {
                GatherInputAxis::Index => index_value.to_string(),
                GatherInputAxis::Output(_) => zero.clone(),
            })
            .collect::<Vec<_>>();
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.extract {}[{}] : {}",
            input.name,
            indices.join(", "),
            input.ty.mlir_type()
        ));
        Ok(Value::scalar(name, input.ty.elem))
    }

    fn index_cast(&mut self, value: &str, elem_type: &str, indent: &str) -> String {
        let index = self.fresh();
        self.lines.push(format!(
            "{indent}{index} = arith.index_cast {value} : {elem_type} to index"
        ));
        index
    }

    fn normalize_gather_index(
        &mut self,
        index_value: &str,
        axis_size: usize,
        indent: &str,
    ) -> String {
        let zero = self.fresh();
        self.lines
            .push(format!("{indent}{zero} = arith.constant 0 : index"));
        let dim = self.fresh();
        self.lines.push(format!(
            "{indent}{dim} = arith.constant {axis_size} : index"
        ));
        let is_negative = self.fresh();
        self.lines.push(format!(
            "{indent}{is_negative} = arith.cmpi slt, {index_value}, {zero} : index"
        ));
        let wrapped = self.fresh();
        self.lines.push(format!(
            "{indent}{wrapped} = arith.addi {index_value}, {dim} : index"
        ));
        let normalized = self.fresh();
        self.lines.push(format!(
            "{indent}{normalized} = arith.select {is_negative}, {wrapped}, {index_value} : index"
        ));
        let lower_ok = self.fresh();
        self.lines.push(format!(
            "{indent}{lower_ok} = arith.cmpi sge, {normalized}, {zero} : index"
        ));
        let upper_ok = self.fresh();
        self.lines.push(format!(
            "{indent}{upper_ok} = arith.cmpi slt, {normalized}, {dim} : index"
        ));
        let in_bounds = self.fresh();
        self.lines.push(format!(
            "{indent}{in_bounds} = arith.andi {lower_ok}, {upper_ok} : i1"
        ));
        self.lines.push(format!(
            "{indent}cf.assert {in_bounds}, \"index out of bounds\""
        ));
        normalized
    }

    pub(super) fn concat(&mut self, lhs: Value, rhs: Value, axis: usize) -> anyhow::Result<Value> {
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
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let name = self.fresh();
        let lows = format_usize_list(lows);
        let highs = format_usize_list(&highs);
        self.lines.push(format!(
            "    {name} = tensor.pad {} low{lows} high{highs} {{",
            input.name
        ));
        let block_args = (0..ty.rank())
            .map(|axis| format!("%d{axis}: index"))
            .collect::<Vec<_>>()
            .join(", ");
        self.lines.push(format!("    ^bb0({block_args}):"));
        self.lines.push(format!(
            "      tensor.yield {zero} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "    }} : {} to {}",
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value::tensor(name, ty.clone()))
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
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        Ok(Value::tensor(empty, ty.clone()))
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
        Ok(Value::tensor(name, dest_ty.clone()))
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
        Ok(Value::tensor(name, ty.clone()))
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
        Ok(Value::tensor(name, ty.clone()))
    }

    pub(super) fn extract_first_scalar(&mut self, input: Value) -> anyhow::Result<Value> {
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
        Ok(Value::scalar(name, input.ty.elem))
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
        Ok((Value::tensor(name, squeezed_ty), dimensions))
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
