use knok_core::{BinaryOp, ElementType, TensorType};

use super::lowerer::{append_block_op, mlir_element_type, Lowerer, Value, ValueKind};
use super::shape::{broadcast_result_type, broadcast_shape};

impl Lowerer<'_, '_> {
    pub(super) fn constant(&mut self, value: &str, elem: ElementType) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem,
            shape: Vec::new(),
        };
        self.append_op_with_attrs(
            "arith.constant",
            &[],
            &ty,
            ValueKind::Scalar,
            &[(
                "value".to_string(),
                format!("{value} : {}", elem.mlir_type()),
            )],
        )
    }

    pub(super) fn zero_like(&mut self, ty: &TensorType) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            return self.constant(ty.elem.zero_literal(), ty.elem);
        }
        self.append_op_with_attrs(
            "arith.constant",
            &[],
            ty,
            ValueKind::Tensor,
            &[(
                "value".to_string(),
                format!("dense<{}> : {}", ty.elem.zero_literal(), ty.mlir_type()),
            )],
        )
    }

    pub(super) fn one_like(&mut self, ty: &TensorType) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            return self.constant(ty.elem.one_literal(), ty.elem);
        }
        self.append_op_with_attrs(
            "arith.constant",
            &[],
            ty,
            ValueKind::Tensor,
            &[(
                "value".to_string(),
                format!("dense<{}> : {}", ty.elem.one_literal(), ty.mlir_type()),
            )],
        )
    }

    pub(super) fn dense_constant(
        &mut self,
        ty: &TensorType,
        values: &[String],
    ) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            let value = values
                .first()
                .ok_or_else(|| anyhow::anyhow!("rank-0 dense constant requires one value"))?;
            return self.constant(value, ty.elem);
        }
        let expected_len: usize = ty.shape.iter().product();
        if values.len() != expected_len {
            anyhow::bail!(
                "dense constant for {:?} expected {expected_len} values, got {}",
                ty,
                values.len()
            );
        }
        let literal = nested_dense_literal(values, &ty.shape);
        self.append_op_with_attrs(
            "arith.constant",
            &[],
            ty,
            ValueKind::Tensor,
            &[(
                "value".to_string(),
                format!("dense<{literal}> : {}", ty.mlir_type()),
            )],
        )
    }

    pub(super) fn zero_initialized_tensor(&mut self, ty: &TensorType) -> anyhow::Result<Value> {
        let empty = self.append_tensor_empty(ty)?;
        let zero = self.constant(ty.elem.zero_literal(), ty.elem)?;
        self.append_linalg_fill(zero, empty, ty)
    }

    pub(super) fn binary_value(
        &mut self,
        op: BinaryOp,
        lhs: Value,
        rhs: Value,
    ) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = match (elem.is_float(), op) {
            (true, BinaryOp::Add) => "arith.addf",
            (true, BinaryOp::Sub) => "arith.subf",
            (true, BinaryOp::Mul) => "arith.mulf",
            (true, BinaryOp::Div) => "arith.divf",
            (false, BinaryOp::Add) => "arith.addi",
            (false, BinaryOp::Sub) => "arith.subi",
            (false, BinaryOp::Mul) => "arith.muli",
            (false, BinaryOp::Div) => "arith.divsi",
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    pub(super) fn minimum(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = if elem.is_float() {
            "arith.minimumf"
        } else {
            "arith.minsi"
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    pub(super) fn maximum(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let elem = if lhs.ty.rank() == 0 {
            rhs.ty.elem
        } else {
            lhs.ty.elem
        };
        let op_name = if elem.is_float() {
            "arith.maximumf"
        } else {
            "arith.maxsi"
        };
        self.emit_binary(op_name, lhs, rhs)
    }

    pub(super) fn emit_unary(&mut self, op_name: &str, value: Value) -> anyhow::Result<Value> {
        let ty = value.ty.clone();
        let kind = value.kind;
        self.append_op(op_name, &[value], &ty, kind)
    }

    pub(super) fn emit_binary(
        &mut self,
        op_name: &str,
        lhs: Value,
        rhs: Value,
    ) -> anyhow::Result<Value> {
        let ty = broadcast_result_type(&lhs.ty, &rhs.ty)?;
        let result_kind =
            if ty.rank() == 0 && lhs.kind == ValueKind::Scalar && rhs.kind == ValueKind::Scalar {
                ValueKind::Scalar
            } else {
                ValueKind::Tensor
            };
        let lhs = self.coerce_to_kind(lhs, &ty, result_kind)?;
        let rhs = self.coerce_to_kind(rhs, &ty, result_kind)?;
        self.append_op(op_name, &[lhs, rhs], &ty, result_kind)
    }

    pub(super) fn comparison(
        &mut self,
        float_predicate: &str,
        integer_predicate: &str,
        lhs: Value,
        rhs: Value,
    ) -> anyhow::Result<Value> {
        let shape = broadcast_shape(&lhs.ty.shape, &rhs.ty.shape)?;
        let ty = TensorType {
            elem: ElementType::Bool,
            shape,
        };
        let op_name = if lhs.ty.elem.is_float() {
            "arith.cmpf"
        } else {
            "arith.cmpi"
        };
        let predicate = if lhs.ty.elem.is_float() {
            float_predicate
        } else {
            integer_predicate
        };
        self.emit_elementwise_binary_op(&ty, lhs, rhs, op_name, Some(predicate))
    }

    pub(super) fn isnan(&mut self, value: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: ElementType::Bool,
            shape: value.ty.shape.clone(),
        };
        self.emit_elementwise_unary_with_duplicate_operand(&ty, value, "arith.cmpf", Some("uno"))
    }

    pub(super) fn logical_binary(
        &mut self,
        op_name: &str,
        lhs: Value,
        rhs: Value,
    ) -> anyhow::Result<Value> {
        let ty = broadcast_result_type(&lhs.ty, &rhs.ty)?;
        self.emit_elementwise_binary_op(&ty, lhs, rhs, op_name, None)
    }

    pub(super) fn logical_not(&mut self, value: Value) -> anyhow::Result<Value> {
        let ty = value.ty.clone();
        let true_value = self.constant("1", ElementType::Bool)?;
        let true_value = self.broadcast(true_value, &ty)?;
        self.logical_binary("arith.xori", value, true_value)
    }

    pub(super) fn where_select(
        &mut self,
        condition: Value,
        true_value: Value,
        false_value: Value,
    ) -> anyhow::Result<Value> {
        let value_shape = broadcast_shape(&true_value.ty.shape, &false_value.ty.shape)?;
        let value_ty = TensorType {
            elem: true_value.ty.elem,
            shape: value_shape,
        };
        let shape = broadcast_shape(&condition.ty.shape, &value_ty.shape)?;
        let ty = TensorType {
            elem: true_value.ty.elem,
            shape,
        };
        let condition_ty = TensorType {
            elem: ElementType::Bool,
            shape: ty.shape.clone(),
        };
        let condition = self.coerce_to_kind(condition, &condition_ty, ValueKind::Tensor)?;
        let true_ty = TensorType {
            elem: true_value.ty.elem,
            shape: ty.shape.clone(),
        };
        let true_value = self.coerce_to_kind(true_value, &true_ty, ValueKind::Tensor)?;
        let false_ty = TensorType {
            elem: false_value.ty.elem,
            shape: ty.shape.clone(),
        };
        let false_value = self.coerce_to_kind(false_value, &false_ty, ValueKind::Tensor)?;
        self.emit_elementwise_ternary(&ty, condition, true_value, false_value)
    }

    fn emit_elementwise_unary_with_duplicate_operand(
        &mut self,
        ty: &TensorType,
        value: Value,
        op_name: &str,
        predicate: Option<&str>,
    ) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            let result_kind = value.kind;
            return self.append_region_like_scalar_op(
                op_name,
                &[value.clone(), value],
                ty,
                result_kind,
                predicate,
            );
        }
        let value_ty = TensorType {
            elem: value.ty.elem,
            shape: ty.shape.clone(),
        };
        let value = self.coerce_to_kind(value, &value_ty, ValueKind::Tensor)?;
        let output = self.append_tensor_empty(ty)?;
        let map = identity_map(ty.rank());
        let iterators = iterator_kinds(ty.rank(), "parallel");
        let result_elem = ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[value],
            &[output],
            &[ty.clone()],
            ty.rank(),
            &[map.clone(), map],
            &iterators,
            |_, block, args| {
                let attrs = predicate_attr(op_name, predicate);
                let result_type = mlir_element_type(context, result_elem)?;
                let results = append_block_op(
                    context,
                    block,
                    location,
                    op_name,
                    &[args[0], args[0]],
                    &[result_type],
                    &attrs,
                    Vec::new(),
                )?;
                Ok(vec![results[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty.clone()))
    }

    fn emit_elementwise_binary_op(
        &mut self,
        ty: &TensorType,
        lhs: Value,
        rhs: Value,
        op_name: &str,
        predicate: Option<&str>,
    ) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            let result_kind = if lhs.kind == ValueKind::Scalar && rhs.kind == ValueKind::Scalar {
                ValueKind::Scalar
            } else {
                ValueKind::Tensor
            };
            let lhs_ty = TensorType {
                elem: lhs.ty.elem,
                shape: Vec::new(),
            };
            let rhs_ty = TensorType {
                elem: rhs.ty.elem,
                shape: Vec::new(),
            };
            let lhs = self.coerce_to_kind(lhs, &lhs_ty, result_kind)?;
            let rhs = self.coerce_to_kind(rhs, &rhs_ty, result_kind)?;
            return self.append_region_like_scalar_op(
                op_name,
                &[lhs, rhs],
                ty,
                result_kind,
                predicate,
            );
        }
        let lhs_ty = TensorType {
            elem: lhs.ty.elem,
            shape: ty.shape.clone(),
        };
        let lhs = self.coerce_to_kind(lhs, &lhs_ty, ValueKind::Tensor)?;
        let rhs_ty = TensorType {
            elem: rhs.ty.elem,
            shape: ty.shape.clone(),
        };
        let rhs = self.coerce_to_kind(rhs, &rhs_ty, ValueKind::Tensor)?;
        let output = self.append_tensor_empty(ty)?;
        let map = identity_map(ty.rank());
        let iterators = iterator_kinds(ty.rank(), "parallel");
        let result_elem = ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[lhs, rhs],
            &[output],
            &[ty.clone()],
            ty.rank(),
            &[map.clone(), map.clone(), map],
            &iterators,
            |_, block, args| {
                let attrs = predicate_attr(op_name, predicate);
                let result_type = mlir_element_type(context, result_elem)?;
                let results = append_block_op(
                    context,
                    block,
                    location,
                    op_name,
                    &[args[0], args[1]],
                    &[result_type],
                    &attrs,
                    Vec::new(),
                )?;
                Ok(vec![results[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty.clone()))
    }

    fn emit_elementwise_ternary(
        &mut self,
        ty: &TensorType,
        condition: Value,
        true_value: Value,
        false_value: Value,
    ) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            let result_kind = if condition.kind == ValueKind::Scalar
                && true_value.kind == ValueKind::Scalar
                && false_value.kind == ValueKind::Scalar
            {
                ValueKind::Scalar
            } else {
                ValueKind::Tensor
            };
            let condition_ty = TensorType {
                elem: ElementType::Bool,
                shape: Vec::new(),
            };
            let value_ty = TensorType {
                elem: ty.elem,
                shape: Vec::new(),
            };
            let condition = self.coerce_to_kind(condition, &condition_ty, result_kind)?;
            let true_value = self.coerce_to_kind(true_value, &value_ty, result_kind)?;
            let false_value = self.coerce_to_kind(false_value, &value_ty, result_kind)?;
            return self.append_region_like_scalar_op(
                "arith.select",
                &[condition, true_value, false_value],
                ty,
                result_kind,
                None,
            );
        }
        let output = self.append_tensor_empty(ty)?;
        let map = identity_map(ty.rank());
        let iterators = iterator_kinds(ty.rank(), "parallel");
        let result_elem = ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[condition, true_value, false_value],
            &[output],
            &[ty.clone()],
            ty.rank(),
            &[map.clone(), map.clone(), map.clone(), map],
            &iterators,
            |_, block, args| {
                let result_type = mlir_element_type(context, result_elem)?;
                let results = append_block_op(
                    context,
                    block,
                    location,
                    "arith.select",
                    &[args[0], args[1], args[2]],
                    &[result_type],
                    &[],
                    Vec::new(),
                )?;
                Ok(vec![results[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty.clone()))
    }

    pub(super) fn splat(&mut self, scalar: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let scalar = self.coerce_to_scalar(scalar)?;
        let empty = self.append_tensor_empty(ty)?;
        self.append_linalg_fill(scalar, empty, ty)
    }

    pub(super) fn coerce_to_kind(
        &mut self,
        value: Value,
        ty: &TensorType,
        kind: ValueKind,
    ) -> anyhow::Result<Value> {
        match kind {
            ValueKind::Scalar => self.coerce_to_scalar(value),
            ValueKind::Tensor => self.broadcast(value, ty),
        }
    }

    pub(super) fn coerce_to_scalar(&mut self, value: Value) -> anyhow::Result<Value> {
        match value.kind {
            ValueKind::Scalar => Ok(value),
            ValueKind::Tensor
                if value.ty.rank() == 0 || value.ty.shape.iter().product::<usize>() == 1 =>
            {
                self.extract_first_scalar(value)
            }
            ValueKind::Tensor => {
                anyhow::bail!("expected scalar-compatible tensor, got {:?}", value.ty)
            }
        }
    }

    fn append_region_like_scalar_op(
        &mut self,
        op_name: &str,
        operands: &[Value],
        ty: &TensorType,
        result_kind: ValueKind,
        predicate: Option<&str>,
    ) -> anyhow::Result<Value> {
        self.append_op_with_attrs(
            op_name,
            operands,
            ty,
            result_kind,
            &predicate_attr(op_name, predicate),
        )
    }
}

fn nested_dense_literal(values: &[String], shape: &[usize]) -> String {
    if shape.is_empty() {
        return values
            .first()
            .expect("scalar dense literal has one value")
            .clone();
    }
    if shape[0] == 0 {
        return "[]".to_string();
    }
    let chunk_len = shape[1..].iter().product::<usize>();
    let items = (0..shape[0])
        .map(|index| {
            if chunk_len == 0 {
                nested_dense_literal(&[], &shape[1..])
            } else {
                let start = index * chunk_len;
                let end = start + chunk_len;
                nested_dense_literal(&values[start..end], &shape[1..])
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{items}]")
}

fn identity_map(rank: usize) -> String {
    if rank == 0 {
        return "()".to_string();
    }
    format!(
        "({})",
        (0..rank)
            .map(|index| format!("d{index}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn iterator_kinds(rank: usize, kind: &'static str) -> Vec<&'static str> {
    vec![kind; rank]
}

fn predicate_attr(op_name: &str, predicate: Option<&str>) -> Vec<(String, String)> {
    let Some(predicate) = predicate else {
        return Vec::new();
    };
    let value = match op_name {
        "arith.cmpf" => cmpf_predicate_value(predicate),
        "arith.cmpi" => cmpi_predicate_value(predicate),
        _ => return Vec::new(),
    };
    vec![("predicate".to_string(), format!("{value} : i64"))]
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
