use knok_core::{BinaryOp, ElementType, TensorType};

use super::lowerer::{Lowerer, Value, ValueKind};
use super::shape::{broadcast_result_type, broadcast_shape, format_dim_list, parallel_iterators};

impl Lowerer<'_> {
    pub(super) fn constant(&mut self, value: &str, elem: ElementType) -> anyhow::Result<Value> {
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = arith.constant {value} : {}",
            elem.mlir_type()
        ));
        Ok(Value::scalar(name, elem))
    }

    pub(super) fn zero_like(&mut self, ty: &TensorType) -> anyhow::Result<Value> {
        if ty.rank() == 0 {
            return self.constant(ty.elem.zero_literal(), ty.elem);
        }
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = arith.constant dense<{}> : {}",
            ty.elem.zero_literal(),
            ty.mlir_type()
        ));
        Ok(Value::tensor(name, ty.clone()))
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
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = {op_name} {} : {}",
            value.name,
            value.mlir_type()
        ));
        Ok(Value {
            name,
            ty: value.ty,
            kind: value.kind,
        })
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
        let result_type = match result_kind {
            ValueKind::Scalar => ty.elem.mlir_type().to_string(),
            ValueKind::Tensor => ty.mlir_type(),
        };
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = {op_name} {}, {} : {}",
            lhs.name, rhs.name, result_type
        ));
        Ok(Value {
            name,
            ty,
            kind: result_kind,
        })
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
        self.emit_elementwise_binary(&ty, lhs, rhs, |result, lhs, rhs, elem| {
            format!("      {result} = {op_name} {predicate}, {lhs}, {rhs} : {elem}")
        })
    }

    pub(super) fn isnan(&mut self, value: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: ElementType::Bool,
            shape: value.ty.shape.clone(),
        };
        self.emit_elementwise_unary(&ty, value, |result, value, elem| {
            format!("      {result} = arith.cmpf uno, {value}, {value} : {elem}")
        })
    }

    pub(super) fn logical_binary(
        &mut self,
        op_name: &str,
        lhs: Value,
        rhs: Value,
    ) -> anyhow::Result<Value> {
        let ty = broadcast_result_type(&lhs.ty, &rhs.ty)?;
        self.emit_elementwise_binary(&ty, lhs, rhs, |result, lhs, rhs, elem| {
            format!("      {result} = {op_name} {lhs}, {rhs} : {elem}")
        })
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

    fn emit_elementwise_unary<F>(
        &mut self,
        ty: &TensorType,
        value: Value,
        emit_body: F,
    ) -> anyhow::Result<Value>
    where
        F: FnOnce(&str, &str, &str) -> String,
    {
        if ty.rank() == 0 {
            let result_kind = value.kind;
            let value_type = value.mlir_type();
            let name = self.fresh();
            self.lines.push(emit_body(&name, &value.name, &value_type));
            return Ok(Value {
                name,
                ty: ty.clone(),
                kind: result_kind,
            });
        }
        let value_ty = TensorType {
            elem: value.ty.elem,
            shape: ty.shape.clone(),
        };
        let value = self.coerce_to_kind(value, &value_ty, ValueKind::Tensor)?;
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let dims = format_dim_list(ty.rank());
        let map = format!("({dims})");
        let iterators = parallel_iterators(ty.rank());
        let result = self.fresh();
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
        self.lines.push(format!(
            "    }} ins({} : {}) outs({empty} : {}) {{",
            value.name,
            value.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%value: {}, %out: {}):",
            value.ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines
            .push(emit_body(&result, "%value", value.ty.elem.mlir_type()));
        self.lines.push(format!(
            "      linalg.yield {result} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value {
            name,
            ty: ty.clone(),
            kind: ValueKind::Tensor,
        })
    }

    fn emit_elementwise_binary<F>(
        &mut self,
        ty: &TensorType,
        lhs: Value,
        rhs: Value,
        emit_body: F,
    ) -> anyhow::Result<Value>
    where
        F: FnOnce(&str, &str, &str, &str) -> String,
    {
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
            let operand_type = lhs.mlir_type();
            let name = self.fresh();
            self.lines
                .push(emit_body(&name, &lhs.name, &rhs.name, &operand_type));
            return Ok(Value {
                name,
                ty: ty.clone(),
                kind: result_kind,
            });
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
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let dims = format_dim_list(ty.rank());
        let map = format!("({dims})");
        let iterators = parallel_iterators(ty.rank());
        let result = self.fresh();
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
        self.lines.push(format!(
            "    }} ins({}, {} : {}, {}) outs({empty} : {}) {{",
            lhs.name,
            rhs.name,
            lhs.mlir_type(),
            rhs.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%lhs: {}, %rhs: {}, %out: {}):",
            lhs.ty.elem.mlir_type(),
            rhs.ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines
            .push(emit_body(&result, "%lhs", "%rhs", lhs.ty.elem.mlir_type()));
        self.lines.push(format!(
            "      linalg.yield {result} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value {
            name,
            ty: ty.clone(),
            kind: ValueKind::Tensor,
        })
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
            let value_type = true_value.mlir_type();
            let name = self.fresh();
            self.lines.push(format!(
                "    {name} = arith.select {}, {}, {} : {}",
                condition.name, true_value.name, false_value.name, value_type
            ));
            return Ok(Value {
                name,
                ty: ty.clone(),
                kind: result_kind,
            });
        }
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        let dims = format_dim_list(ty.rank());
        let map = format!("({dims})");
        let iterators = parallel_iterators(ty.rank());
        let selected = self.fresh();
        self.lines.push(format!("    {name} = linalg.generic {{"));
        self.lines.push(format!(
            "      indexing_maps = [affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>, affine_map<({dims}) -> {map}>],"
        ));
        self.lines
            .push(format!("      iterator_types = [{iterators}]"));
        self.lines.push(format!(
            "    }} ins({}, {}, {} : {}, {}, {}) outs({empty} : {}) {{",
            condition.name,
            true_value.name,
            false_value.name,
            condition.mlir_type(),
            true_value.mlir_type(),
            false_value.mlir_type(),
            ty.mlir_type()
        ));
        self.lines.push(format!(
            "    ^bb0(%condition: i1, %true_value: {}, %false_value: {}, %out: {}):",
            ty.elem.mlir_type(),
            ty.elem.mlir_type(),
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      {selected} = arith.select %condition, %true_value, %false_value : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "      linalg.yield {selected} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!("    }} -> {}", ty.mlir_type()));
        Ok(Value {
            name,
            ty: ty.clone(),
            kind: ValueKind::Tensor,
        })
    }

    pub(super) fn splat(&mut self, scalar: Value, ty: &TensorType) -> anyhow::Result<Value> {
        let scalar = self.to_scalar(scalar)?;
        let empty = self.fresh();
        self.lines
            .push(format!("    {empty} = tensor.empty() : {}", ty.mlir_type()));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = linalg.fill ins({} : {}) outs({empty} : {}) -> {}",
            scalar.name,
            scalar.ty.elem.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value {
            name,
            ty: ty.clone(),
            kind: ValueKind::Tensor,
        })
    }

    pub(super) fn coerce_to_kind(
        &mut self,
        value: Value,
        ty: &TensorType,
        kind: ValueKind,
    ) -> anyhow::Result<Value> {
        match kind {
            ValueKind::Scalar => self.to_scalar(value),
            ValueKind::Tensor => self.broadcast(value, ty),
        }
    }

    pub(super) fn to_scalar(&mut self, value: Value) -> anyhow::Result<Value> {
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
}
