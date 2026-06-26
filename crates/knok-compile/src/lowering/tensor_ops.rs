use knok_core::{Conv2dOptions, TensorType};

use super::lowerer::{Lowerer, Value, ValueKind};
use super::shape::{
    axis_broadcast_dimensions, broadcast_shape, collapse_reassociation_for_removed_axis,
    collapse_reassociation_for_squeezed_broadcast, element_count, ensure_axis_broadcastable,
    ensure_broadcastable, expand_reassociation_for_inserted_axis, format_dim_list,
    format_shape_list, format_usize_list, reassociation_for_rank,
};

impl Lowerer<'_> {
    pub(super) fn matmul(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        if lhs.ty.rank() == 3 && rhs.ty.rank() == 3 && lhs.ty.shape[0] == rhs.ty.shape[0] {
            return self.batch_matmul(lhs, rhs);
        }
        if lhs.ty.rank() != 2 || rhs.ty.rank() != 2 {
            return self.generic_matmul(lhs, rhs);
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
        Ok(Value::tensor(name, ty))
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
        Ok(Value::tensor(name, ty))
    }

    fn generic_matmul(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let spec = MatmulSpec::new(&lhs.ty, &rhs.ty)?;
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape: spec.output_shape.clone(),
        };
        let init = self.zero_initialized_tensor(&ty)?;
        let output_rank = ty.rank();
        let reduction_dim = format!("d{output_rank}");
        let dims = format_dim_list(output_rank + 1);
        let output_map = affine_tuple(
            &(0..output_rank)
                .map(|index| format!("d{index}"))
                .collect::<Vec<_>>(),
        );
        let lhs_map = affine_tuple(&spec.lhs_indices(&reduction_dim));
        let rhs_map = affine_tuple(&spec.rhs_indices(&reduction_dim));
        let iterators = {
            let mut values = vec!["\"parallel\""; output_rank];
            values.push("\"reduction\"");
            values.join(", ")
        };
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
            "      indexing_maps = [affine_map<({dims}) -> {lhs_map}>, affine_map<({dims}) -> {rhs_map}>, affine_map<({dims}) -> {output_map}>],"
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

    pub(super) fn conv2d(
        &mut self,
        input: Value,
        kernel: Value,
        options: &Conv2dOptions,
    ) -> anyhow::Result<Value> {
        let input = self.pad_conv2d_input(input, options)?;
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![
                input.ty.shape[0],
                (input.ty.shape[1] - ((kernel.ty.shape[0] - 1) * options.dilation_h + 1))
                    / options.stride_h
                    + 1,
                (input.ty.shape[2] - ((kernel.ty.shape[1] - 1) * options.dilation_w + 1))
                    / options.stride_w
                    + 1,
                kernel.ty.shape[3],
            ],
        };
        let init = self.zero_initialized_tensor(&ty)?;
        let name = self.fresh();
        let attrs = if *options == Conv2dOptions::default() {
            String::new()
        } else {
            format!(
                " {{dilations = dense<[{}, {}]> : vector<2xi64>, strides = dense<[{}, {}]> : vector<2xi64>}}",
                options.dilation_h, options.dilation_w, options.stride_h, options.stride_w
            )
        };
        self.lines.push(format!(
            "    {name} = linalg.conv_2d_nhwc_hwcf{attrs} ins({}, {} : {}, {}) outs({init} : {}) -> {}",
            input.name,
            kernel.name,
            input.ty.mlir_type(),
            kernel.ty.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value::tensor(name, ty))
    }

    pub(super) fn transpose(&mut self, input: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![input.ty.shape[1], input.ty.shape[0]],
        };
        self.permute(input, &ty, &[1, 0])
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

    fn zero_initialized_tensor(&mut self, ty: &TensorType) -> anyhow::Result<String> {
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
        Ok(init)
    }

    fn pad_conv2d_input(&mut self, input: Value, options: &Conv2dOptions) -> anyhow::Result<Value> {
        if options.pad_h == 0 && options.pad_w == 0 {
            return Ok(input);
        }
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![
                input.ty.shape[0],
                input.ty.shape[1] + 2 * options.pad_h,
                input.ty.shape[2] + 2 * options.pad_w,
                input.ty.shape[3],
            ],
        };
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.pad {} low[0, {}, {}, 0] high[0, {}, {}, 0] {{",
            input.name, options.pad_h, options.pad_w, options.pad_h, options.pad_w
        ));
        self.lines
            .push("    ^bb0(%d0: index, %d1: index, %d2: index, %d3: index):".to_string());
        self.lines.push(format!(
            "      tensor.yield {zero} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "    }} : {} to {}",
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value::tensor(name, ty))
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
            let scalar = self.to_scalar(input)?;
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

struct MatmulSpec {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl MatmulSpec {
    fn new(lhs: &TensorType, rhs: &TensorType) -> anyhow::Result<Self> {
        if lhs.rank() == 0 || rhs.rank() == 0 {
            anyhow::bail!("matmul expects operands with rank at least 1");
        }
        let lhs_is_vector = lhs.rank() == 1;
        let rhs_is_vector = rhs.rank() == 1;
        let lhs_k = if lhs_is_vector {
            lhs.shape[0]
        } else {
            lhs.shape[lhs.rank() - 1]
        };
        let rhs_k = if rhs_is_vector {
            rhs.shape[0]
        } else {
            rhs.shape[rhs.rank() - 2]
        };
        if lhs_k != rhs_k {
            anyhow::bail!("matmul inner dimensions differ: {lhs_k} vs {rhs_k}");
        }
        let lhs_batch = if lhs.rank() > 2 {
            &lhs.shape[..lhs.rank() - 2]
        } else {
            &[]
        };
        let rhs_batch = if rhs.rank() > 2 {
            &rhs.shape[..rhs.rank() - 2]
        } else {
            &[]
        };
        let mut output_shape = broadcast_shape(lhs_batch, rhs_batch)?;
        if !lhs_is_vector {
            output_shape.push(lhs.shape[lhs.rank() - 2]);
        }
        if !rhs_is_vector {
            output_shape.push(rhs.shape[rhs.rank() - 1]);
        }
        Ok(Self {
            lhs_shape: lhs.shape.clone(),
            rhs_shape: rhs.shape.clone(),
            output_shape,
        })
    }

    fn lhs_indices(&self, reduction_dim: &str) -> Vec<String> {
        let lhs_rank = self.lhs_shape.len();
        if lhs_rank == 1 {
            return vec![reduction_dim.to_string()];
        }
        let output_batch_rank = self.output_batch_rank();
        let lhs_batch_rank = lhs_rank.saturating_sub(2);
        let mut indices = batch_indices(
            &self.lhs_shape[..lhs_batch_rank],
            &self.output_shape[..output_batch_rank],
        );
        indices.push(format!("d{output_batch_rank}"));
        indices.push(reduction_dim.to_string());
        indices
    }

    fn rhs_indices(&self, reduction_dim: &str) -> Vec<String> {
        let rhs_rank = self.rhs_shape.len();
        if rhs_rank == 1 {
            return vec![reduction_dim.to_string()];
        }
        let output_batch_rank = self.output_batch_rank();
        let rhs_batch_rank = rhs_rank.saturating_sub(2);
        let mut indices = batch_indices(
            &self.rhs_shape[..rhs_batch_rank],
            &self.output_shape[..output_batch_rank],
        );
        indices.push(reduction_dim.to_string());
        let n_axis = output_batch_rank + usize::from(self.lhs_shape.len() != 1);
        indices.push(format!("d{n_axis}"));
        indices
    }

    fn output_batch_rank(&self) -> usize {
        let matrix_axes =
            usize::from(self.lhs_shape.len() != 1) + usize::from(self.rhs_shape.len() != 1);
        self.output_shape.len() - matrix_axes
    }
}

fn batch_indices(input_batch: &[usize], output_batch: &[usize]) -> Vec<String> {
    let padding = output_batch.len() - input_batch.len();
    input_batch
        .iter()
        .enumerate()
        .map(|(index, dim)| {
            let output_axis = padding + index;
            if *dim == 1 && output_batch[output_axis] != 1 {
                "0".to_string()
            } else {
                format!("d{output_axis}")
            }
        })
        .collect()
}

fn affine_tuple(indices: &[String]) -> String {
    if indices.is_empty() {
        "()".to_string()
    } else {
        format!("({})", indices.join(", "))
    }
}
