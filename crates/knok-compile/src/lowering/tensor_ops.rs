use knok_core::TensorType;

use super::lowerer::{Lowerer, Value, ValueKind};
use super::shape::{
    axis_broadcast_dimensions, collapse_reassociation_for_removed_axis,
    collapse_reassociation_for_squeezed_broadcast, element_count, ensure_axis_broadcastable,
    ensure_broadcastable, expand_reassociation_for_inserted_axis, format_shape_list,
    format_usize_list, reassociation_for_rank,
};

impl Lowerer<'_> {
    pub(super) fn transpose(&mut self, input: Value) -> anyhow::Result<Value> {
        if input.ty.rank() <= 1 {
            return Ok(input);
        }
        let ty = TensorType {
            elem: input.ty.elem,
            shape: input.ty.shape.iter().rev().copied().collect(),
        };
        let axes = (0..input.ty.rank()).rev().collect::<Vec<_>>();
        self.permute(input, &ty, &axes)
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
