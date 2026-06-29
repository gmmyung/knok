use super::lowerer::{append_block_op, mlir_element_type, Lowerer, Value};
use super::shape::broadcast_shape;
use knok_core::TensorType;

impl Lowerer<'_, '_> {
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
        self.append_named_linalg("linalg.matmul", &[lhs, rhs], init, &ty, &[])
    }

    fn batch_matmul(&mut self, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: lhs.ty.elem,
            shape: vec![lhs.ty.shape[0], lhs.ty.shape[1], rhs.ty.shape[2]],
        };
        let init = self.zero_initialized_tensor(&ty)?;
        self.append_named_linalg("linalg.batch_matmul", &[lhs, rhs], init, &ty, &[])
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
        let output_map = affine_tuple(
            &(0..output_rank)
                .map(|index| format!("d{index}"))
                .collect::<Vec<_>>(),
        );
        let lhs_map = affine_tuple(&spec.lhs_indices(&reduction_dim));
        let rhs_map = affine_tuple(&spec.rhs_indices(&reduction_dim));
        let mut iterators = vec!["parallel"; output_rank];
        iterators.push("reduction");
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
        let elem = ty.elem;
        let context = self.context;
        let location = self.location;
        let raw = self.append_linalg_generic(
            &[lhs, rhs],
            &[init],
            &[ty.clone()],
            output_rank + 1,
            &[lhs_map, rhs_map, output_map],
            &iterators,
            |_, block, args| {
                let elem_ty = mlir_element_type(context, elem)?;
                let product = append_block_op(
                    context,
                    block,
                    location,
                    mul_op,
                    &[args[0], args[1]],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?;
                let sum = append_block_op(
                    context,
                    block,
                    location,
                    add_op,
                    &[args[2], product[0].as_value()],
                    &[elem_ty],
                    &[],
                    Vec::new(),
                )?;
                Ok(vec![sum[0]])
            },
        )?;
        Ok(Value::tensor(raw[0], ty))
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

#[cfg(test)]
mod tests {
    use knok_core::{ElementType, TensorType};

    use super::*;

    fn f32_tensor(shape: &[usize]) -> TensorType {
        TensorType {
            elem: ElementType::F32,
            shape: shape.to_vec(),
        }
    }

    fn matmul_spec_error(lhs: &[usize], rhs: &[usize]) -> String {
        match MatmulSpec::new(&f32_tensor(lhs), &f32_tensor(rhs)) {
            Ok(_) => panic!("expected matmul spec error for {lhs:?} @ {rhs:?}"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn vector_vector_matmul_is_scalar_reduction() {
        let spec = MatmulSpec::new(&f32_tensor(&[3]), &f32_tensor(&[3])).unwrap();

        assert!(spec.output_shape.is_empty());
        assert_eq!(spec.lhs_indices("d0"), vec!["d0"]);
        assert_eq!(spec.rhs_indices("d0"), vec!["d0"]);
        assert_eq!(spec.output_batch_rank(), 0);
    }

    #[test]
    fn matrix_vector_matmul_keeps_lhs_rows() {
        let spec = MatmulSpec::new(&f32_tensor(&[2, 3]), &f32_tensor(&[3])).unwrap();

        assert_eq!(spec.output_shape, vec![2]);
        assert_eq!(spec.lhs_indices("d1"), vec!["d0", "d1"]);
        assert_eq!(spec.rhs_indices("d1"), vec!["d1"]);
        assert_eq!(spec.output_batch_rank(), 0);
    }

    #[test]
    fn vector_matrix_matmul_keeps_rhs_columns() {
        let spec = MatmulSpec::new(&f32_tensor(&[3]), &f32_tensor(&[3, 2])).unwrap();

        assert_eq!(spec.output_shape, vec![2]);
        assert_eq!(spec.lhs_indices("d1"), vec!["d1"]);
        assert_eq!(spec.rhs_indices("d1"), vec!["d1", "d0"]);
        assert_eq!(spec.output_batch_rank(), 0);
    }

    #[test]
    fn batched_matmul_broadcasts_batch_dimensions() {
        let spec = MatmulSpec::new(&f32_tensor(&[2, 1, 4, 3]), &f32_tensor(&[1, 5, 3, 6])).unwrap();

        assert_eq!(spec.output_shape, vec![2, 5, 4, 6]);
        assert_eq!(spec.lhs_indices("d4"), vec!["d0", "0", "d2", "d4"]);
        assert_eq!(spec.rhs_indices("d4"), vec!["0", "d1", "d4", "d3"]);
        assert_eq!(spec.output_batch_rank(), 2);
    }

    #[test]
    fn affine_tuple_formats_empty_and_nonempty_maps() {
        assert_eq!(affine_tuple(&[]), "()");
        assert_eq!(affine_tuple(&["d0".into(), "d2".into()]), "(d0, d2)");
    }

    #[test]
    fn matmul_spec_rejects_invalid_inputs() {
        assert!(matmul_spec_error(&[], &[3]).contains("rank at least 1"));
        assert!(matmul_spec_error(&[2, 4], &[3, 2]).contains("inner dimensions differ"));
        assert!(matmul_spec_error(&[2, 3, 4], &[5, 4, 2]).contains("broadcast dimension"));
    }
}
