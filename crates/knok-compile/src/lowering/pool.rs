use knok_core::{ElementType, Pool2dOptions, TensorType};
use melior::ir::{Block, Type, Value as MlirValue};

use super::lowerer::{append_block_op, mlir_element_type, Lowerer, RawValue, Value};

impl Lowerer<'_, '_> {
    pub(super) fn max_pool2d(
        &mut self,
        input: Value,
        options: &Pool2dOptions,
    ) -> anyhow::Result<Value> {
        let input = self.pad_pool2d_input(input, options, max_pool_padding_literal)?;
        let ty = pool2d_output_type(&input.ty, options);
        self.emit_pool2d_generic(input, &ty, options, Pool2dReduction::Max)
    }

    pub(super) fn avg_pool2d(
        &mut self,
        input: Value,
        options: &Pool2dOptions,
    ) -> anyhow::Result<Value> {
        let input = self.pad_pool2d_input(input, options, ElementType::zero_literal)?;
        let ty = pool2d_output_type(&input.ty, options);
        self.emit_pool2d_generic(input, &ty, options, Pool2dReduction::Average)
    }

    fn emit_pool2d_generic(
        &mut self,
        input: Value,
        ty: &TensorType,
        options: &Pool2dOptions,
        reduction: Pool2dReduction,
    ) -> anyhow::Result<Value> {
        let output = self.append_tensor_empty(ty)?;
        let input_raw = input.raw;
        let input_elem = input.ty.elem;
        let context = self.context;
        let location = self.location;
        let results = self.append_linalg_generic(
            &[],
            &[output],
            &[ty.clone()],
            ty.rank(),
            &["(d0, d1, d2, d3)".to_string()],
            &["parallel", "parallel", "parallel", "parallel"],
            |_, block, _| {
                let indices = (0..4)
                    .map(|axis| linalg_index(context, block, location, axis))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let indices = indices
                    .into_iter()
                    .map(RawValue::as_value)
                    .collect::<Vec<_>>();
                build_pool2d_body(
                    context, block, location, input_raw, input_elem, options, reduction, &indices,
                )
                .map(|value| vec![value])
            },
        )?;
        Ok(Value::tensor(results[0], ty.clone()))
    }

    fn pad_pool2d_input(
        &mut self,
        input: Value,
        options: &Pool2dOptions,
        pad_literal: fn(ElementType) -> &'static str,
    ) -> anyhow::Result<Value> {
        let padding = options.padding;
        if padding.top == 0 && padding.bottom == 0 && padding.left == 0 && padding.right == 0 {
            return Ok(input);
        }
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![
                input.ty.shape[0],
                input.ty.shape[1] + padding.top + padding.bottom,
                input.ty.shape[2] + padding.left + padding.right,
                input.ty.shape[3],
            ],
        };
        let pad = self.constant(pad_literal(ty.elem), ty.elem)?;
        self.append_tensor_pad(
            input,
            &ty,
            &[0, padding.top, padding.left, 0],
            &[0, padding.bottom, padding.right, 0],
            pad,
        )
    }
}

fn pool2d_output_type(input: &TensorType, options: &Pool2dOptions) -> TensorType {
    let effective_h = (options.kernel[0] - 1) * options.dilation[0] + 1;
    let effective_w = (options.kernel[1] - 1) * options.dilation[1] + 1;
    TensorType {
        elem: input.elem,
        shape: vec![
            input.shape[0],
            (input.shape[1] - effective_h) / options.stride[0] + 1,
            (input.shape[2] - effective_w) / options.stride[1] + 1,
            input.shape[3],
        ],
    }
}

fn max_pool_padding_literal(elem: ElementType) -> &'static str {
    match elem {
        ElementType::Bool => unreachable!("max_pool2d does not support bool tensors"),
        ElementType::F32 => "0xFF800000",
        ElementType::F64 => "0xFFF0000000000000",
        ElementType::F16 => "0xFC00",
        ElementType::BF16 => "0xFF80",
        ElementType::I32 => "-2147483648",
        ElementType::I64 => "-9223372036854775808",
    }
}

#[derive(Clone, Copy)]
enum Pool2dReduction {
    Average,
    Max,
}

fn build_pool2d_body<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: melior::ir::Location<'c>,
    input: RawValue,
    elem: ElementType,
    options: &Pool2dOptions,
    reduction: Pool2dReduction,
    output_indices: &[MlirValue<'c, '_>],
) -> anyhow::Result<RawValue> {
    let elem_ty = mlir_element_type(context, elem)?;
    let mut acc = scalar_constant_in_block(
        context,
        block,
        location,
        match reduction {
            Pool2dReduction::Average => elem.zero_literal(),
            Pool2dReduction::Max => max_pool_padding_literal(elem),
        },
        elem,
    )?;
    for kernel_h in 0..options.kernel[0] {
        for kernel_w in 0..options.kernel[1] {
            let h = scaled_index_in_block(
                context,
                block,
                location,
                output_indices[1],
                options.stride[0],
                kernel_h * options.dilation[0],
            )?;
            let w = scaled_index_in_block(
                context,
                block,
                location,
                output_indices[2],
                options.stride[1],
                kernel_w * options.dilation[1],
            )?;
            let value = tensor_extract_in_block(
                context,
                block,
                location,
                input,
                elem_ty,
                &[
                    output_indices[0],
                    h.as_value(),
                    w.as_value(),
                    output_indices[3],
                ],
            )?;
            let op = match (reduction, elem.is_float()) {
                (Pool2dReduction::Average, true) => "arith.addf",
                (Pool2dReduction::Average, false) => "arith.addi",
                (Pool2dReduction::Max, true) => "arith.maximumf",
                (Pool2dReduction::Max, false) => "arith.maxsi",
            };
            acc = append_block_op(
                context,
                block,
                location,
                op,
                &[acc.as_value(), value.as_value()],
                &[elem_ty],
                &[],
                Vec::new(),
            )?[0];
        }
    }
    if matches!(reduction, Pool2dReduction::Average) {
        let denominator = scalar_constant_in_block(
            context,
            block,
            location,
            &format!("{}.0", options.kernel[0] * options.kernel[1]),
            elem,
        )?;
        acc = append_block_op(
            context,
            block,
            location,
            "arith.divf",
            &[acc.as_value(), denominator.as_value()],
            &[elem_ty],
            &[],
            Vec::new(),
        )?[0];
    }
    Ok(acc)
}

fn scaled_index_in_block<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: melior::ir::Location<'c>,
    index: MlirValue<'c, '_>,
    scale: usize,
    offset: usize,
) -> anyhow::Result<RawValue> {
    let index_ty = Type::index(context);
    let scale = index_constant_in_block(context, block, location, scale)?;
    let scaled = append_block_op(
        context,
        block,
        location,
        "arith.muli",
        &[index, scale.as_value()],
        &[index_ty],
        &[],
        Vec::new(),
    )?[0];
    if offset == 0 {
        return Ok(scaled);
    }
    let offset = index_constant_in_block(context, block, location, offset)?;
    append_block_op(
        context,
        block,
        location,
        "arith.addi",
        &[scaled.as_value(), offset.as_value()],
        &[index_ty],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn index_constant_in_block<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: melior::ir::Location<'c>,
    value: usize,
) -> anyhow::Result<RawValue> {
    append_block_op(
        context,
        block,
        location,
        "arith.constant",
        &[],
        &[Type::index(context)],
        &[("value".to_string(), format!("{value} : index"))],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn scalar_constant_in_block<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: melior::ir::Location<'c>,
    value: &str,
    elem: ElementType,
) -> anyhow::Result<RawValue> {
    append_block_op(
        context,
        block,
        location,
        "arith.constant",
        &[],
        &[mlir_element_type(context, elem)?],
        &[(
            "value".to_string(),
            format!("{value} : {}", elem.mlir_type()),
        )],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn tensor_extract_in_block<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: melior::ir::Location<'c>,
    input: RawValue,
    elem_ty: Type<'c>,
    indices: &[MlirValue<'c, '_>],
) -> anyhow::Result<RawValue> {
    let mut operands = vec![input.as_value()];
    operands.extend_from_slice(indices);
    append_block_op(
        context,
        block,
        location,
        "tensor.extract",
        &operands,
        &[elem_ty],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn linalg_index<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: melior::ir::Location<'c>,
    axis: usize,
) -> anyhow::Result<RawValue> {
    append_block_op(
        context,
        block,
        location,
        "linalg.index",
        &[],
        &[Type::index(context)],
        &[("dim".to_string(), format!("{axis} : i64"))],
        Vec::new(),
    )
    .map(|values| values[0])
}
