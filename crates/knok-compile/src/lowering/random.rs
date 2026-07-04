use knok_core::{BinaryOp, ElementType, TensorType};
use melior::ir::{Block, Location, Type};

use super::lowerer::{append_block_op, mlir_element_type, Lowerer, RawValue, Value};

impl Lowerer<'_, '_> {
    pub(super) fn random_uniform(
        &mut self,
        target: &TensorType,
        key: Value,
        low: Value,
        high: Value,
    ) -> anyhow::Result<Vec<Value>> {
        let key = self.coerce_to_scalar(key)?;
        let low = self.coerce_to_scalar(low)?;
        let high = self.coerce_to_scalar(high)?;
        let tensor = self.random_uniform_tensor(target, key.clone(), low, high)?;
        let next_key = self.random_next_key(key, target)?;
        Ok(vec![tensor, next_key])
    }

    pub(super) fn random_int(
        &mut self,
        target: &TensorType,
        key: Value,
        low: Value,
        high: Value,
    ) -> anyhow::Result<Vec<Value>> {
        let key = self.coerce_to_scalar(key)?;
        let low = self.coerce_to_scalar(low)?;
        let high = self.coerce_to_scalar(high)?;
        let tensor = self.random_int_tensor(target, key.clone(), low, high)?;
        let next_key = self.random_next_key(key, target)?;
        Ok(vec![tensor, next_key])
    }

    fn random_uniform_tensor(
        &mut self,
        target: &TensorType,
        key: Value,
        low: Value,
        high: Value,
    ) -> anyhow::Result<Value> {
        let output = self.append_tensor_empty(target)?;
        let output_map = identity_map(target.rank());
        let iterators = vec!["parallel"; target.rank()];
        let elem = target.elem;
        let shape = target.shape.clone();
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[key, low, high],
            &[output],
            &[target.clone()],
            target.rank(),
            &[
                "()".to_string(),
                "()".to_string(),
                "()".to_string(),
                output_map,
            ],
            &iterators,
            |_, block, args| {
                let linear_index = flattened_linalg_index(context, block, location, &shape)?;
                let counter = append_block_op(
                    context,
                    block,
                    location,
                    "arith.addi",
                    &[args[0], linear_index.as_value()],
                    &[mlir_element_type(context, ElementType::I64)?],
                    &[],
                    Vec::new(),
                )?[0];
                let hash = splitmix64_in_block(context, block, location, counter)?;
                let unit = unit_float_from_hash(context, block, location, hash, elem)?;
                let span = append_block_op(
                    context,
                    block,
                    location,
                    "arith.subf",
                    &[args[2], args[1]],
                    &[mlir_element_type(context, elem)?],
                    &[],
                    Vec::new(),
                )?[0];
                let scaled = append_block_op(
                    context,
                    block,
                    location,
                    "arith.mulf",
                    &[unit.as_value(), span.as_value()],
                    &[mlir_element_type(context, elem)?],
                    &[],
                    Vec::new(),
                )?[0];
                let shifted = append_block_op(
                    context,
                    block,
                    location,
                    "arith.addf",
                    &[args[1], scaled.as_value()],
                    &[mlir_element_type(context, elem)?],
                    &[],
                    Vec::new(),
                )?[0];
                Ok(vec![shifted])
            },
        )?;
        Ok(Value::tensor(raw[0], target.clone()))
    }

    fn random_int_tensor(
        &mut self,
        target: &TensorType,
        key: Value,
        low: Value,
        high: Value,
    ) -> anyhow::Result<Value> {
        let output = self.append_tensor_empty(target)?;
        let output_map = identity_map(target.rank());
        let iterators = vec!["parallel"; target.rank()];
        let elem = target.elem;
        let shape = target.shape.clone();
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[key, low, high],
            &[output],
            &[target.clone()],
            target.rank(),
            &[
                "()".to_string(),
                "()".to_string(),
                "()".to_string(),
                output_map,
            ],
            &iterators,
            |_, block, args| {
                let linear_index = flattened_linalg_index(context, block, location, &shape)?;
                let counter = append_block_op(
                    context,
                    block,
                    location,
                    "arith.addi",
                    &[args[0], linear_index.as_value()],
                    &[mlir_element_type(context, ElementType::I64)?],
                    &[],
                    Vec::new(),
                )?[0];
                let hash = splitmix64_in_block(context, block, location, counter)?;
                let sample = integer_sample_from_hash(context, block, location, hash, elem)?;
                let elem_ty = mlir_element_type(context, elem)?;
                let range = append_block_op(
                    context,
                    block,
                    location,
                    "arith.subi",
                    &[args[2], args[1]],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?[0];
                let offset = append_block_op(
                    context,
                    block,
                    location,
                    "arith.remsi",
                    &[sample.as_value(), range.as_value()],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?[0];
                let shifted = append_block_op(
                    context,
                    block,
                    location,
                    "arith.addi",
                    &[args[1], offset.as_value()],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?[0];
                Ok(vec![shifted])
            },
        )?;
        Ok(Value::tensor(raw[0], target.clone()))
    }

    fn random_next_key(&mut self, key: Value, target: &TensorType) -> anyhow::Result<Value> {
        let count = target.shape.iter().product::<usize>();
        let count = i64::try_from(count)
            .map_err(|_| anyhow::anyhow!("random output element count exceeds i64"))?;
        let count = self.constant(&count.to_string(), ElementType::I64)?;
        self.binary_value(BinaryOp::Add, key, count)
    }
}

fn identity_map(rank: usize) -> String {
    if rank == 0 {
        "()".to_string()
    } else {
        format!(
            "({})",
            (0..rank)
                .map(|axis| format!("d{axis}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn flattened_linalg_index<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    shape: &[usize],
) -> anyhow::Result<RawValue> {
    let i64_ty = mlir_element_type(context, ElementType::I64)?;
    let mut flattened = int_constant_in_block(context, block, location, 0, ElementType::I64)?;
    for (axis, dim) in shape.iter().copied().enumerate() {
        let index = linalg_index(context, block, location, axis)?;
        let index = append_block_op(
            context,
            block,
            location,
            "arith.index_cast",
            &[index.as_value()],
            &[i64_ty],
            &[],
            Vec::new(),
        )?[0];
        if axis == 0 {
            flattened = index;
        } else {
            let dim = int_constant_in_block(
                context,
                block,
                location,
                i128::try_from(dim)?,
                ElementType::I64,
            )?;
            let scaled = append_block_op(
                context,
                block,
                location,
                "arith.muli",
                &[flattened.as_value(), dim.as_value()],
                &[i64_ty],
                &[],
                Vec::new(),
            )?[0];
            flattened = append_block_op(
                context,
                block,
                location,
                "arith.addi",
                &[scaled.as_value(), index.as_value()],
                &[i64_ty],
                &[],
                Vec::new(),
            )?[0];
        }
    }
    Ok(flattened)
}

fn splitmix64_in_block<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    value: RawValue,
) -> anyhow::Result<RawValue> {
    let value = add_i64(
        context,
        block,
        location,
        value,
        splitmix_const(0x9E37_79B9_7F4A_7C15),
    )?;
    let shifted = shift_right_i64(context, block, location, value, 30)?;
    let xored = xor_i64(context, block, location, value, shifted)?;
    let multiplied = mul_i64(
        context,
        block,
        location,
        xored,
        splitmix_const(0xBF58_476D_1CE4_E5B9),
    )?;
    let shifted = shift_right_i64(context, block, location, multiplied, 27)?;
    let xored = xor_i64(context, block, location, multiplied, shifted)?;
    let multiplied = mul_i64(
        context,
        block,
        location,
        xored,
        splitmix_const(0x94D0_49BB_1331_11EB),
    )?;
    let shifted = shift_right_i64(context, block, location, multiplied, 31)?;
    xor_i64(context, block, location, multiplied, shifted)
}

fn unit_float_from_hash<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    hash: RawValue,
    elem: ElementType,
) -> anyhow::Result<RawValue> {
    let (mask, scale) = match elem {
        ElementType::F64 => (9_007_199_254_740_991_i128, "9007199254740992.0"),
        ElementType::F32 | ElementType::F16 | ElementType::BF16 => (16_777_215_i128, "16777216.0"),
        _ => anyhow::bail!("random_uniform target element type must be floating-point"),
    };
    let masked = and_i64(context, block, location, hash, mask)?;
    let elem_ty = mlir_element_type(context, elem)?;
    let float = append_block_op(
        context,
        block,
        location,
        "arith.sitofp",
        &[masked.as_value()],
        &[elem_ty],
        &[],
        Vec::new(),
    )?[0];
    let scale = float_constant_in_block(context, block, location, scale, elem)?;
    append_block_op(
        context,
        block,
        location,
        "arith.divf",
        &[float.as_value(), scale.as_value()],
        &[elem_ty],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn integer_sample_from_hash<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    hash: RawValue,
    elem: ElementType,
) -> anyhow::Result<RawValue> {
    match elem {
        ElementType::I32 => {
            let i32_ty = mlir_element_type(context, ElementType::I32)?;
            let truncated = append_block_op(
                context,
                block,
                location,
                "arith.trunci",
                &[hash.as_value()],
                &[i32_ty],
                &[],
                Vec::new(),
            )?[0];
            let mask = int_constant_in_block(context, block, location, i32::MAX as i128, elem)?;
            append_block_op(
                context,
                block,
                location,
                "arith.andi",
                &[truncated.as_value(), mask.as_value()],
                &[i32_ty],
                &[],
                Vec::new(),
            )
            .map(|values| values[0])
        }
        ElementType::I64 => and_i64(context, block, location, hash, i64::MAX as i128),
        _ => anyhow::bail!("random_int target element type must be i32 or i64"),
    }
}

fn add_i64<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    lhs: RawValue,
    rhs: i64,
) -> anyhow::Result<RawValue> {
    let rhs = int_constant_in_block(context, block, location, rhs as i128, ElementType::I64)?;
    append_block_op(
        context,
        block,
        location,
        "arith.addi",
        &[lhs.as_value(), rhs.as_value()],
        &[mlir_element_type(context, ElementType::I64)?],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn mul_i64<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    lhs: RawValue,
    rhs: i64,
) -> anyhow::Result<RawValue> {
    let rhs = int_constant_in_block(context, block, location, rhs as i128, ElementType::I64)?;
    append_block_op(
        context,
        block,
        location,
        "arith.muli",
        &[lhs.as_value(), rhs.as_value()],
        &[mlir_element_type(context, ElementType::I64)?],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn and_i64<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    lhs: RawValue,
    rhs: i128,
) -> anyhow::Result<RawValue> {
    let rhs = int_constant_in_block(context, block, location, rhs, ElementType::I64)?;
    append_block_op(
        context,
        block,
        location,
        "arith.andi",
        &[lhs.as_value(), rhs.as_value()],
        &[mlir_element_type(context, ElementType::I64)?],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn xor_i64<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    lhs: RawValue,
    rhs: RawValue,
) -> anyhow::Result<RawValue> {
    append_block_op(
        context,
        block,
        location,
        "arith.xori",
        &[lhs.as_value(), rhs.as_value()],
        &[mlir_element_type(context, ElementType::I64)?],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn shift_right_i64<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    value: RawValue,
    amount: i128,
) -> anyhow::Result<RawValue> {
    let amount = int_constant_in_block(context, block, location, amount, ElementType::I64)?;
    append_block_op(
        context,
        block,
        location,
        "arith.shrui",
        &[value.as_value(), amount.as_value()],
        &[mlir_element_type(context, ElementType::I64)?],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn linalg_index<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
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

fn int_constant_in_block<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    value: i128,
    elem: ElementType,
) -> anyhow::Result<RawValue> {
    let ty = mlir_element_type(context, elem)?;
    append_block_op(
        context,
        block,
        location,
        "arith.constant",
        &[],
        &[ty],
        &[(
            "value".to_string(),
            format!("{value} : {}", elem.mlir_type()),
        )],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn float_constant_in_block<'c>(
    context: &'c melior::Context,
    block: &Block<'c>,
    location: Location<'c>,
    value: &str,
    elem: ElementType,
) -> anyhow::Result<RawValue> {
    let ty = mlir_element_type(context, elem)?;
    append_block_op(
        context,
        block,
        location,
        "arith.constant",
        &[],
        &[ty],
        &[(
            "value".to_string(),
            format!("{value} : {}", elem.mlir_type()),
        )],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn splitmix_const(value: u64) -> i64 {
    value as i64
}
