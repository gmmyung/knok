use knok_core::TensorType;

use super::lowerer::{Lowerer, Value};
use super::shape::{broadcast_shape, format_dim_list};

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
        let init = self.zero_initialized_tensor(&ty)?;
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
        let init = self.zero_initialized_tensor(&ty)?;
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
