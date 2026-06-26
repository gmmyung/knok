use knok_core::{BinaryOp, ElementType, TensorType};

use super::lowerer::{Lowerer, Value};
use super::shape::{broadcast_result_type, broadcast_shape, format_dim_list, parallel_iterators};

impl Lowerer<'_> {
    pub(super) fn constant(&mut self, value: &str, elem: ElementType) -> anyhow::Result<Value> {
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = arith.constant {value} : {}",
            elem.mlir_type()
        ));
        Ok(Value {
            name,
            ty: TensorType {
                elem,
                shape: vec![],
            },
        })
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
        Ok(Value {
            name,
            ty: ty.clone(),
        })
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
            value.ty.mlir_type()
        ));
        Ok(Value { name, ty: value.ty })
    }

    pub(super) fn emit_binary(
        &mut self,
        op_name: &str,
        lhs: Value,
        rhs: Value,
    ) -> anyhow::Result<Value> {
        let ty = broadcast_result_type(&lhs.ty, &rhs.ty)?;
        let lhs = if lhs.ty == ty {
            lhs
        } else {
            self.broadcast(lhs, &ty)?
        };
        let rhs = if rhs.ty == ty {
            rhs
        } else {
            self.broadcast(rhs, &ty)?
        };
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = {op_name} {}, {} : {}",
            lhs.name,
            rhs.name,
            ty.mlir_type()
        ));
        Ok(Value { name, ty })
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
        let condition = if condition.ty == condition_ty {
            condition
        } else {
            self.broadcast(condition, &condition_ty)?
        };
        let true_ty = TensorType {
            elem: true_value.ty.elem,
            shape: ty.shape.clone(),
        };
        let true_value = if true_value.ty == true_ty {
            true_value
        } else {
            self.broadcast(true_value, &true_ty)?
        };
        let false_ty = TensorType {
            elem: false_value.ty.elem,
            shape: ty.shape.clone(),
        };
        let false_value = if false_value.ty == false_ty {
            false_value
        } else {
            self.broadcast(false_value, &false_ty)?
        };
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
            let name = self.fresh();
            self.lines
                .push(emit_body(&name, &value.name, value.ty.elem.mlir_type()));
            return Ok(Value {
                name,
                ty: ty.clone(),
            });
        }
        let value_ty = TensorType {
            elem: value.ty.elem,
            shape: ty.shape.clone(),
        };
        let value = if value.ty == value_ty {
            value
        } else {
            self.broadcast(value, &value_ty)?
        };
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
            value.ty.mlir_type(),
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
            let name = self.fresh();
            self.lines.push(emit_body(
                &name,
                &lhs.name,
                &rhs.name,
                lhs.ty.elem.mlir_type(),
            ));
            return Ok(Value {
                name,
                ty: ty.clone(),
            });
        }
        let lhs_ty = TensorType {
            elem: lhs.ty.elem,
            shape: ty.shape.clone(),
        };
        let lhs = if lhs.ty == lhs_ty {
            lhs
        } else {
            self.broadcast(lhs, &lhs_ty)?
        };
        let rhs_ty = TensorType {
            elem: rhs.ty.elem,
            shape: ty.shape.clone(),
        };
        let rhs = if rhs.ty == rhs_ty {
            rhs
        } else {
            self.broadcast(rhs, &rhs_ty)?
        };
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
            lhs.ty.mlir_type(),
            rhs.ty.mlir_type(),
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
            let name = self.fresh();
            self.lines.push(format!(
                "    {name} = arith.select {}, {}, {} : {}",
                condition.name,
                true_value.name,
                false_value.name,
                ty.elem.mlir_type()
            ));
            return Ok(Value {
                name,
                ty: ty.clone(),
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
            condition.ty.mlir_type(),
            true_value.ty.mlir_type(),
            false_value.ty.mlir_type(),
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
        })
    }

    pub(super) fn splat(&mut self, scalar: Value, ty: &TensorType) -> anyhow::Result<Value> {
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
        })
    }
}
