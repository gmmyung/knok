use knok_core::{AxisSpec, ElementType, TensorType};

use super::lowerer::{append_block_op, mlir_element_type, Lowerer, RawValue, Value};
use super::shape::{element_count, reduction_output_map, reduction_output_shape};

impl Lowerer<'_, '_> {
    pub(super) fn mean(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "mean")?;
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(input);
        }
        let count = axis
            .index()
            .map_or_else(|| element_count(&input.ty), |axis| input.ty.shape[axis]);
        let elem = input.ty.elem;
        let sum = self.sum(input, axis)?;
        let scale = self.constant(&format!("{count}.0"), elem)?;
        self.emit_binary("arith.divf", sum, scale)
    }

    pub(super) fn var(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "var")?;
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return self.zero_like(&input.ty);
        }
        let count = axis
            .index()
            .map_or_else(|| element_count(&input.ty), |axis| input.ty.shape[axis]);
        let elem = input.ty.elem;
        let mean = self.mean(input.clone(), axis)?;
        let mean = if let Some(axis) = axis.index() {
            self.broadcast_along_axis(mean, &input.ty, axis)?
        } else {
            self.broadcast(mean, &input.ty)?
        };
        let centered = self.emit_binary("arith.subf", input, mean)?;
        let squared = self.emit_binary("arith.mulf", centered.clone(), centered)?;
        let sum = self.sum(squared, axis)?;
        let scale = self.constant(&format!("{count}.0"), elem)?;
        self.emit_binary("arith.divf", sum, scale)
    }

    pub(super) fn std(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        let variance = self.var(input, axis)?;
        self.emit_unary("math.sqrt", variance)
    }

    pub(super) fn ptp(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "ptp")?;
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return self.zero_like(&input.ty);
        }
        let max = self.max(input.clone(), axis)?;
        let min = self.min(input, axis)?;
        let op = if max.ty.elem.is_float() {
            "arith.subf"
        } else {
            "arith.subi"
        };
        self.emit_binary(op, max, min)
    }

    pub(super) fn softmax(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "softmax")?;
        if let Some(axis) = axis.index() {
            return self.axis_softmax(input, axis);
        }
        let max = self.max_keep_dims(input.clone(), axis, false)?;
        let max = if let Some(axis) = axis.index() {
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
        let denominator = if let Some(axis) = axis.index() {
            self.broadcast_along_axis(denominator, &exp.ty, axis)?
        } else {
            self.broadcast(denominator, &exp.ty)?
        };
        self.emit_binary("arith.divf", exp, denominator)
    }

    fn axis_softmax(&mut self, input: Value, axis: usize) -> anyhow::Result<Value> {
        let ty = input.ty.clone();
        let empty = self.append_tensor_empty(&ty)?;
        self.append_named_linalg(
            "linalg.softmax",
            &[input],
            empty,
            &ty,
            &[("dimension".to_string(), format!("{axis} : i64"))],
        )
    }

    pub(super) fn sigmoid(&mut self, input: Value) -> anyhow::Result<Value> {
        let one = self.constant(input.ty.elem.one_literal(), input.ty.elem)?;
        let zero = self.zero_like(&input.ty)?;
        let neg = self.emit_binary("arith.subf", zero, input)?;
        let exp = self.emit_unary("math.exp", neg)?;
        let denominator = self.emit_binary("arith.addf", one.clone(), exp)?;
        self.emit_binary("arith.divf", one, denominator)
    }

    pub(super) fn argmax(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        self.arg_extreme(input, axis, Extreme::Max)
    }

    pub(super) fn argmin(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        self.arg_extreme(input, axis, Extreme::Min)
    }

    fn arg_extreme(
        &mut self,
        input: Value,
        axis: AxisSpec,
        extreme: Extreme,
    ) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, extreme.arg_op_name())?;
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            let index_ty = TensorType {
                elem: ElementType::I64,
                shape: Vec::new(),
            };
            let scalar = self.constant("0", ElementType::I64)?;
            return self.splat(scalar, &index_ty);
        }
        let index_ty = TensorType {
            elem: ElementType::I64,
            shape: reduction_output_shape(&input.ty.shape, axis, false),
        };
        let value_ty = TensorType {
            elem: input.ty.elem,
            shape: index_ty.shape.clone(),
        };
        let init_index = self.fill_tensor(&index_ty, "0", ElementType::I64)?;
        let init_value = self.fill_tensor(
            &value_ty,
            extreme.initial_value(input.ty.elem),
            input.ty.elem,
        )?;

        let rank = input.ty.rank();
        let input_map = identity_map(rank);
        let output_map = reduction_output_map(rank, axis, false);
        let iterator_types = reduction_iterator_types(rank, axis);
        let iterator_types = parse_iterator_types(&iterator_types);
        let compare_op = if input.ty.elem.is_float() {
            "arith.cmpf"
        } else {
            "arith.cmpi"
        };
        let ordering_predicate = extreme.ordering_predicate(input.ty.elem);
        let equal_predicate = if input.ty.elem.is_float() {
            "oeq"
        } else {
            "eq"
        };
        let input_ty = input.ty.clone();
        let elem = input.ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[input],
            &[init_index, init_value],
            &[index_ty.clone(), value_ty],
            rank,
            &[input_map, output_map.clone(), output_map],
            &iterator_types,
            |_, block, args| {
                let elem_ty = mlir_element_type(context, elem)?;
                let i1_ty = mlir_element_type(context, ElementType::Bool)?;
                let i64_ty = mlir_element_type(context, ElementType::I64)?;
                let candidate_index =
                    arg_candidate_index(context, block, location, &input_ty, axis)?;
                let ordered = append_block_op(
                    context,
                    block,
                    location,
                    compare_op,
                    &[args[0], args[2]],
                    &[i1_ty],
                    &[(
                        "predicate".to_string(),
                        cmp_predicate_attr(compare_op, ordering_predicate),
                    )],
                    Vec::new(),
                )?[0];
                let equal = append_block_op(
                    context,
                    block,
                    location,
                    compare_op,
                    &[args[0], args[2]],
                    &[i1_ty],
                    &[(
                        "predicate".to_string(),
                        cmp_predicate_attr(compare_op, equal_predicate),
                    )],
                    Vec::new(),
                )?[0];
                let lower_index = append_block_op(
                    context,
                    block,
                    location,
                    "arith.cmpi",
                    &[candidate_index.as_value(), args[1]],
                    &[i1_ty],
                    &[("predicate".to_string(), "2 : i64".to_string())],
                    Vec::new(),
                )?[0];
                let tied_before = append_block_op(
                    context,
                    block,
                    location,
                    "arith.andi",
                    &[equal.as_value(), lower_index.as_value()],
                    &[i1_ty],
                    &[],
                    Vec::new(),
                )?[0];
                let candidate_better = append_block_op(
                    context,
                    block,
                    location,
                    "arith.ori",
                    &[ordered.as_value(), tied_before.as_value()],
                    &[i1_ty],
                    &[],
                    Vec::new(),
                )?[0];
                let selected_index = append_block_op(
                    context,
                    block,
                    location,
                    "arith.select",
                    &[
                        candidate_better.as_value(),
                        candidate_index.as_value(),
                        args[1],
                    ],
                    &[i64_ty],
                    &[],
                    Vec::new(),
                )?[0];
                let selected_value = append_block_op(
                    context,
                    block,
                    location,
                    "arith.select",
                    &[candidate_better.as_value(), args[0], args[2]],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?[0];
                Ok(vec![selected_index, selected_value])
            },
        )?;
        Ok(Value::tensor(raw[0], index_ty))
    }

    pub(super) fn sum(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(input);
        }
        let reducer_op = if input.ty.elem.is_float() {
            "arith.addf"
        } else {
            "arith.addi"
        };
        let initial_value = input.ty.elem.zero_literal();
        self.reduce(input, initial_value, reducer_op, axis, false)
    }

    pub(super) fn prod(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(input);
        }
        let reducer_op = if input.ty.elem.is_float() {
            "arith.mulf"
        } else {
            "arith.muli"
        };
        let initial_value = input.ty.elem.one_literal();
        self.reduce(input, initial_value, reducer_op, axis, false)
    }

    pub(super) fn all(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(input);
        }
        self.reduce(input, "1", "arith.andi", axis, false)
    }

    pub(super) fn any(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(input);
        }
        self.reduce(input, "0", "arith.ori", axis, false)
    }

    pub(super) fn max(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        self.max_keep_dims(input, axis, false)
    }

    fn max_keep_dims(
        &mut self,
        input: Value,
        axis: AxisSpec,
        keep_dims: bool,
    ) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "max")?;
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(input);
        }
        let (initial_value, reducer_op) = match input.ty.elem {
            ElementType::Bool => ("0", "arith.ori"),
            elem if elem.is_float() => (negative_infinity_literal(elem), "arith.maximumf"),
            elem => (min_numeric_literal(elem), "arith.maxsi"),
        };
        self.reduce(input, initial_value, reducer_op, axis, keep_dims)
    }

    pub(super) fn min(&mut self, input: Value, axis: AxisSpec) -> anyhow::Result<Value> {
        ensure_non_empty_reduction(&input.ty, axis, "min")?;
        if input.ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(input);
        }
        let (initial_value, reducer_op) = match input.ty.elem {
            ElementType::Bool => ("1", "arith.andi"),
            elem if elem.is_float() => (positive_infinity_literal(elem), "arith.minimumf"),
            elem => (max_numeric_literal(elem), "arith.minsi"),
        };
        self.reduce(input, initial_value, reducer_op, axis, false)
    }

    fn reduce(
        &mut self,
        input: Value,
        initial_value: &str,
        reducer_op: &str,
        axis: AxisSpec,
        keep_dims: bool,
    ) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: reduction_output_shape(&input.ty.shape, axis, keep_dims),
        };
        let init = self.fill_tensor(&ty, initial_value, ty.elem)?;

        let rank = input.ty.rank();
        let input_map = identity_map(rank);
        let iterator_types = reduction_iterator_types(rank, axis);
        let output_map = reduction_output_map(rank, axis, keep_dims);
        let iterator_types = parse_iterator_types(&iterator_types);
        let elem = ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[input],
            &[init],
            &[ty.clone()],
            rank,
            &[input_map, output_map],
            &iterator_types,
            |_, block, args| {
                let elem_ty = mlir_element_type(context, elem)?;
                let reduced = append_block_op(
                    context,
                    block,
                    location,
                    reducer_op,
                    &[args[1], args[0]],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?;
                Ok(vec![reduced[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty))
    }

    fn fill_tensor(
        &mut self,
        ty: &TensorType,
        value: &str,
        elem: ElementType,
    ) -> anyhow::Result<Value> {
        let empty = self.append_tensor_empty(ty)?;
        let scalar = self.constant(value, elem)?;
        self.append_linalg_fill(scalar, empty, ty)
    }
}

fn reduction_iterator_types(rank: usize, axis: AxisSpec) -> String {
    (0..rank)
        .map(|index| {
            if matches!(axis, AxisSpec::All) || axis.index() == Some(index) {
                "\"reduction\""
            } else {
                "\"parallel\""
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn parse_iterator_types(text: &str) -> Vec<&'static str> {
    text.split(',')
        .map(str::trim)
        .map(|value| value.trim_matches('"'))
        .map(|value| match value {
            "parallel" => "parallel",
            "reduction" => "reduction",
            other => panic!("unknown linalg iterator type `{other}`"),
        })
        .collect()
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

fn cmp_predicate_attr(op_name: &str, predicate: &str) -> String {
    let value = match op_name {
        "arith.cmpf" => cmpf_predicate_value(predicate),
        "arith.cmpi" => cmpi_predicate_value(predicate),
        _ => unreachable!("comparison predicate requested for {op_name}"),
    };
    format!("{value} : i64")
}

fn cmpf_predicate_value(predicate: &str) -> i64 {
    match predicate {
        "false" => 0,
        "oeq" => 1,
        "ogt" => 2,
        "oge" => 3,
        "olt" => 4,
        "ole" => 5,
        "one" => 6,
        "ord" => 7,
        "ueq" => 8,
        "ugt" => 9,
        "uge" => 10,
        "ult" => 11,
        "ule" => 12,
        "une" => 13,
        "uno" => 14,
        "true" => 15,
        other => panic!("unknown arith.cmpf predicate `{other}`"),
    }
}

fn cmpi_predicate_value(predicate: &str) -> i64 {
    match predicate {
        "eq" => 0,
        "ne" => 1,
        "slt" => 2,
        "sle" => 3,
        "sgt" => 4,
        "sge" => 5,
        "ult" => 6,
        "ule" => 7,
        "ugt" => 8,
        "uge" => 9,
        other => panic!("unknown arith.cmpi predicate `{other}`"),
    }
}

fn arg_candidate_index<'c>(
    context: &'c melior::Context,
    block: &melior::ir::Block<'c>,
    location: melior::ir::Location<'c>,
    input: &TensorType,
    axis: AxisSpec,
) -> anyhow::Result<RawValue> {
    let index_ty = melior::ir::Type::index(context);
    let index = if let Some(axis) = axis.index() {
        linalg_index(context, block, location, axis)?
    } else {
        let mut flattened = linalg_index(context, block, location, 0)?;
        for axis in 1..input.rank() {
            let axis_index = linalg_index(context, block, location, axis)?;
            let dim_size = append_block_op(
                context,
                block,
                location,
                "arith.constant",
                &[],
                &[index_ty],
                &[(
                    "value".to_string(),
                    format!("{} : index", input.shape[axis]),
                )],
                Vec::new(),
            )?[0];
            let scaled = append_block_op(
                context,
                block,
                location,
                "arith.muli",
                &[flattened.as_value(), dim_size.as_value()],
                &[index_ty],
                &[],
                Vec::new(),
            )?[0];
            flattened = append_block_op(
                context,
                block,
                location,
                "arith.addi",
                &[scaled.as_value(), axis_index.as_value()],
                &[index_ty],
                &[],
                Vec::new(),
            )?[0];
        }
        flattened
    };
    append_block_op(
        context,
        block,
        location,
        "arith.index_cast",
        &[index.as_value()],
        &[mlir_element_type(context, ElementType::I64)?],
        &[],
        Vec::new(),
    )
    .map(|values| values[0])
}

fn linalg_index<'c>(
    context: &'c melior::Context,
    block: &melior::ir::Block<'c>,
    location: melior::ir::Location<'c>,
    axis: usize,
) -> anyhow::Result<RawValue> {
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

fn max_numeric_literal(elem: ElementType) -> &'static str {
    match elem {
        ElementType::Bool => "1",
        ElementType::F32 => "3.40282347E+38",
        ElementType::F64 => "1.7976931348623157E+308",
        ElementType::F16 => "65504.0",
        ElementType::BF16 => "3.38953139E+38",
        ElementType::I32 => "2147483647",
        ElementType::I64 => "9223372036854775807",
    }
}

fn negative_infinity_literal(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 => "0xFF800000",
        ElementType::F64 => "0xFFF0000000000000",
        ElementType::F16 => "0xFC00",
        ElementType::BF16 => "0xFF80",
        _ => unreachable!("negative infinity seed requested for non-float element"),
    }
}

fn positive_infinity_literal(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 => "0x7F800000",
        ElementType::F64 => "0x7FF0000000000000",
        ElementType::F16 => "0x7C00",
        ElementType::BF16 => "0x7F80",
        _ => unreachable!("positive infinity seed requested for non-float element"),
    }
}

#[derive(Clone, Copy)]
enum Extreme {
    Max,
    Min,
}

impl Extreme {
    fn arg_op_name(self) -> &'static str {
        match self {
            Self::Max => "argmax",
            Self::Min => "argmin",
        }
    }

    fn initial_value(self, elem: ElementType) -> &'static str {
        match self {
            Self::Max => min_numeric_literal(elem),
            Self::Min => max_numeric_literal(elem),
        }
    }

    fn ordering_predicate(self, elem: ElementType) -> &'static str {
        match (self, elem) {
            (Self::Max, elem) if elem.is_float() => "ogt",
            (Self::Min, elem) if elem.is_float() => "olt",
            (Self::Max, ElementType::Bool) => "ugt",
            (Self::Min, ElementType::Bool) => "ult",
            (Self::Max, _) => "sgt",
            (Self::Min, _) => "slt",
        }
    }
}

fn ensure_non_empty_reduction(
    input: &TensorType,
    axis: AxisSpec,
    op_name: &str,
) -> anyhow::Result<()> {
    match axis {
        AxisSpec::One(axis) if axis >= input.rank() => {
            anyhow::bail!(
                "{op_name} axis {axis} is out of bounds for rank-{} tensor {:?}",
                input.rank(),
                input.shape
            );
        }
        AxisSpec::One(axis) if input.shape[axis] == 0 => {
            anyhow::bail!(
                "{op_name} cannot reduce empty axis {axis} for tensor shape {:?}",
                input.shape
            );
        }
        AxisSpec::All if element_count(input) == 0 => {
            anyhow::bail!(
                "{op_name} cannot reduce empty tensor shape {:?}",
                input.shape
            );
        }
        _ => Ok(()),
    }
}
