use knok_core::{type_check, Expr, Graph, Input, TensorType, TypedExpr, TypedGraph};

use crate::Result;

use super::{expr::next_tuple_id, tensor::TraceTensor};

#[derive(Default)]
pub struct TraceContext {
    inputs: Vec<Input>,
}

impl TraceContext {
    pub fn input<T: TraceTensor>(&mut self, name: &str) -> T {
        let ty = T::tensor_type();
        self.inputs.push(Input {
            name: name.into(),
            ty,
        });
        T::from_expr(Expr::Var(name.into()))
    }

    pub(crate) fn finish<O: TraceOutput>(
        self,
        name: &str,
        backend: &str,
        output: O,
    ) -> Result<TypedGraph> {
        let body = output.exprs();
        let outputs = O::types();
        let graph = Graph {
            name: name.into(),
            backend: backend.into(),
            inputs: self.inputs,
            outputs,
            lets: Vec::new(),
            body,
        };
        type_check(graph, &[]).map_err(Into::into)
    }
}

pub trait TraceOutput {
    fn exprs(self) -> Vec<Expr>;
    fn types() -> Vec<TensorType>;
}

pub trait TraceVars: TraceOutput {
    fn var_count() -> usize;
    fn from_tuple_expr(tuple_id: u64, value: Expr) -> Self;
}

impl<T: TraceTensor> TraceOutput for T {
    fn exprs(self) -> Vec<Expr> {
        vec![self.into_expr()]
    }

    fn types() -> Vec<TensorType> {
        vec![T::tensor_type()]
    }
}

impl<T: TraceTensor> TraceVars for T {
    fn var_count() -> usize {
        1
    }

    fn from_tuple_expr(tuple_id: u64, value: Expr) -> Self {
        T::from_expr(Expr::TupleGet {
            tuple_id,
            value: Box::new(value),
            index: 0,
        })
    }
}

macro_rules! impl_tuple_output {
    ($($name:ident: $index:tt),+) => {
        impl<$($name: TraceTensor),+> TraceOutput for ($($name,)+) {
            fn exprs(self) -> Vec<Expr> {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                vec![$($name.into_expr()),+]
            }

            fn types() -> Vec<TensorType> {
                vec![$($name::tensor_type()),+]
            }
        }

        impl<$($name: TraceTensor),+> TraceVars for ($($name,)+) {
            fn var_count() -> usize {
                [$($name::tensor_type()),+].len()
            }

            fn from_tuple_expr(tuple_id: u64, value: Expr) -> Self {
                (
                    $(
                        $name::from_expr(Expr::TupleGet {
                            tuple_id,
                            value: Box::new(value.clone()),
                            index: $index,
                        }),
                    )+
                )
            }
        }
    };
}

impl_tuple_output!(A: 0, B: 1);
impl_tuple_output!(A: 0, B: 1, C: 2);
impl_tuple_output!(A: 0, B: 1, C: 2, D: 3);
impl_tuple_output!(A: 0, B: 1, C: 2, D: 3, E: 4);
impl_tuple_output!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);

pub fn typed_expr<T: TraceTensor>(value: T) -> TypedExpr {
    TypedExpr {
        kind: value.expr().clone(),
        ty: T::tensor_type(),
    }
}

pub(crate) fn tuple_id() -> u64 {
    next_tuple_id()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{matmul, relu, split, T1, T2};
    use knok_core::ElementType;

    #[test]
    fn traces_helper_loop_to_expression_tree() {
        fn block(x: T2<f32, 4, 4>) -> T2<f32, 4, 4> {
            relu(matmul(x.clone(), x) + 1.0)
        }

        let mut context = TraceContext::default();
        let x = context.input::<T2<f32, 4, 4>>("x");
        let mut y = x;
        for _ in 0..2 {
            y = block(y);
        }
        let graph = context.finish("forward", "llvm-cpu", y).unwrap();

        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs[0].shape, vec![4, 4]);
        assert_eq!(graph.outputs[0].elem, ElementType::F32);
    }

    #[test]
    fn split_traces_as_tuple_projections_without_context_side_effects() {
        let mut context = TraceContext::default();
        let x = context.input::<T1<f32, 4>>("x");
        let output: (T1<f32, 2>, T1<f32, 2>) = split(x, 0, [2, 2]);
        let graph = context.finish("split", "llvm-cpu", output).unwrap();

        assert!(graph.lets.is_empty());
        assert_eq!(graph.outputs, vec![T1::<f32, 2>::tensor_type(); 2]);
    }
}
