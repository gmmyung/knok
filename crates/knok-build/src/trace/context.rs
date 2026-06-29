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
    use crate::trace::*;
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

    #[test]
    fn traces_elementwise_creation_and_predicate_ops() {
        let mut context = TraceContext::default();
        let x = context.input::<T1<f32, 4>>("x");
        let y = context.input::<T1<f32, 4>>("y");
        let mask = context.input::<T1<bool, 4>>("mask");

        let _: T1<f32, 4> = abs(x.clone());
        let _: T1<f32, 4> = ceil(x.clone());
        let _: T1<f32, 4> = exp(x.clone());
        let _: T1<f32, 4> = exp2(x.clone());
        let _: T1<f32, 4> = expm1(x.clone());
        let _: T1<f32, 4> = floor(x.clone());
        let _: T1<f32, 4> = log(x.clone());
        let _: T1<f32, 4> = log1p(x.clone());
        let _: T1<f32, 4> = log2(x.clone());
        let _: T1<f32, 4> = log10(x.clone());
        let _: T1<f32, 4> = rint(x.clone());
        let _: T1<f32, 4> = round(x.clone());
        let _: T1<f32, 4> = sigmoid(x.clone());
        let _: T1<f32, 4> = sin(x.clone());
        let _: T1<f32, 4> = cos(x.clone());
        let _: T1<f32, 4> = sqrt(x.clone());
        let _: T1<f32, 4> = tan(x.clone());
        let _: T1<f32, 4> = tanh(x.clone());
        let _: T1<f32, 4> = square(x.clone());
        let _: T1<f32, 4> = reciprocal(x.clone());
        let _: T1<bool, 4> = isnan(x.clone());
        let _: T1<f32, 4> = zeros_like(x.clone());
        let _: T1<f32, 4> = ones_like(x.clone());
        let _: T1<f32, 4> = minimum(x.clone(), y.clone());
        let _: T1<f32, 4> = pow(x.clone(), 2.0);
        let _: T1<bool, 4> = greater_equal(x.clone(), y.clone());
        let _: T1<bool, 4> = less(x.clone(), y.clone());
        let _: T1<bool, 4> = equal(x.clone(), y.clone());
        let _: T1<bool, 4> = not_equal(x.clone(), y.clone());
        let _: T1<bool, 4> = logical_or(mask.clone(), false);
        let _: T1<bool, 4> = logical_xor(mask.clone(), true);
        let _: T1<i32, 4> = arange_to(4);
        let _: T1<i32, 4> = arange(0, 4);
        let _: T2<f32, 2, 2> = identity();

        let predicate = logical_and(greater(x.clone(), 0.0), less_equal(y.clone(), 6.0));
        let selected: T1<f32, 4> = r#where(
            predicate.clone(),
            clip(maximum(x.clone(), y.clone()), 0.0, 6.0),
            full_like(x.clone(), 1.0),
        );
        let ints: T1<i32, 4> = arange_step(0, 8, 2);
        let line: T1<f32, 4> = linspace(0.0, 1.0);
        let eye_matrix: T2<f32, 2, 2> = eye();

        let graph = context
            .finish(
                "elementwise_creation",
                "llvm-cpu",
                (
                    relu(selected),
                    logical_not(predicate),
                    ints,
                    line,
                    eye_matrix,
                    ones_like(x),
                ),
            )
            .unwrap();

        assert_eq!(graph.outputs.len(), 6);
        assert_eq!(graph.outputs[0], T1::<f32, 4>::tensor_type());
        assert_eq!(graph.outputs[1], T1::<bool, 4>::tensor_type());
        assert_eq!(graph.outputs[2].elem, ElementType::I32);
    }

    #[test]
    fn traces_reduction_linalg_and_conv_ops() {
        let mut context = TraceContext::default();
        let matrix = context.input::<T2<f32, 2, 3>>("matrix");
        let rhs = context.input::<T2<f32, 3, 2>>("rhs");
        let vector = context.input::<T1<f32, 3>>("vector");
        let square = context.input::<T2<f32, 3, 3>>("square");
        let flags = context.input::<T2<bool, 2, 3>>("flags");
        let image = context.input::<T4<f32, 1, 4, 4, 2>>("image");
        let kernel = context.input::<T4<f32, 3, 3, 2, 1>>("kernel");

        let _: T0<f32> = sum(matrix.clone());
        let _: T0<f32> = prod(matrix.clone());
        let _: T0<f32> = mean(matrix.clone());
        let _: T0<f32> = max(matrix.clone());
        let _: T0<f32> = amax(matrix.clone());
        let _: T0<f32> = min(matrix.clone());
        let _: T0<f32> = amin(matrix.clone());
        let _: T0<i64> = argmax(matrix.clone());
        let _: T0<i64> = argmin(matrix.clone());
        let _: T0<f32> = var(matrix.clone());
        let _: T0<f32> = std(matrix.clone());
        let _: T0<f32> = ptp(matrix.clone());
        let _: T1<f32, 2> = prod_axis(matrix.clone(), 1);
        let _: T1<f32, 2> = mean_axis(matrix.clone(), 1);
        let _: T1<f32, 2> = max_axis(matrix.clone(), 1);
        let _: T1<f32, 2> = amax_axis(matrix.clone(), 1);
        let _: T1<f32, 2> = min_axis(matrix.clone(), 1);
        let _: T1<f32, 2> = amin_axis(matrix.clone(), 1);
        let _: T1<f32, 2> = var_axis(matrix.clone(), 1);
        let _: T1<f32, 2> = std_axis(matrix.clone(), 1);
        let _: T1<f32, 2> = ptp_axis(matrix.clone(), 1);
        let _: T0<bool> = all(flags.clone());
        let _: T0<bool> = any(flags.clone());
        let _: T1<bool, 2> = all_axis(flags.clone(), 1);
        let _: T1<bool, 2> = any_axis(flags.clone(), 1);
        let _: T1<i64, 2> = argmin_axis(matrix.clone(), 1);
        let _: T2<f32, 2, 3> = softmax(matrix.clone());
        let _: T0<f32> = dot(vector.clone(), vector.clone());
        let _: T0<f32> = inner(vector.clone(), vector.clone());
        let _: T1<f32, 2> = vecdot_axis(matrix.clone(), matrix.clone(), 1);
        let _: T0<f32> = trace(square.clone());
        let _: T0<f32> = trace_axes(square.clone(), 0, 1);
        let _: T1<f32, 3> = diagonal_axes(square.clone(), 0, 1);
        let _: T4<f32, 1, 4, 4, 1> = conv2d_options(
            image.clone(),
            kernel.clone(),
            Conv2dOptions::new()
                .padding(1, 1, 1, 1)
                .stride(1, 1)
                .dilation(1, 1)
                .groups(1),
        );

        let product: T2<f32, 2, 2> = matmul(matrix.clone(), rhs);
        let row_sum: T1<f32, 2> = sum_axis(matrix.clone(), 1);
        let row_argmax: T1<i64, 2> = argmax_axis(matrix.clone(), 1);
        let outer_product: T2<f32, 3, 3> = outer(vector.clone(), vector);
        let diagonal_values: T1<f32, 3> = diagonal(square);
        let conv: T4<f32, 1, 2, 2, 1> = conv2d(image, kernel);

        let graph = context
            .finish(
                "reduction_linalg_conv",
                "llvm-cpu",
                (
                    product,
                    row_sum,
                    row_argmax,
                    outer_product,
                    diagonal_values,
                    conv,
                ),
            )
            .unwrap();

        assert_eq!(graph.outputs.len(), 6);
        assert_eq!(graph.outputs[0], T2::<f32, 2, 2>::tensor_type());
        assert_eq!(graph.outputs[2], T1::<i64, 2>::tensor_type());
        assert_eq!(graph.outputs[5], T4::<f32, 1, 2, 2, 1>::tensor_type());
    }

    #[test]
    fn traces_shape_and_indexing_ops() {
        let mut context = TraceContext::default();
        let x = context.input::<T2<f32, 2, 3>>("x");
        let y = context.input::<T2<f32, 2, 3>>("y");
        let row = context.input::<T1<f32, 3>>("row");
        let indices = context.input::<T2<i64, 2, 2>>("indices");
        let cube = context.input::<T3<f32, 2, 3, 4>>("cube");

        let _: T2<f32, 2, 3> = broadcast(row.clone());
        let _: T1<f32, 3> = squeeze(unsqueeze::<T2<f32, 1, 3>>(row.clone()));
        let _: T2<f32, 4, 5> = pad(x.clone(), [1, 1]);
        let _: T1<f32, 3> = take(x.clone(), 0, 0);
        let _: T2<f32, 2, 2> = take_along_axis(x.clone(), indices.clone(), 1);
        let _: T2<f32, 2, 6> = repeat(x.clone(), 1, 2);
        let _: T2<f32, 2, 3> = flip(x.clone());
        let _: T2<f32, 2, 3> = flip_axes(x.clone(), [1]);
        let _: T2<f32, 2, 3> = roll(x.clone(), 1, 1);
        let _: T2<f32, 3, 2> = transpose_axes(x.clone(), [1, 0]);
        let _: T3<f32, 4, 2, 3> = permute(cube.clone(), [2, 0, 1]);
        let _: T3<f32, 4, 2, 3> = permute_dims(cube.clone(), [2, 0, 1]);
        let _: T3<f32, 2, 4, 3> = swapaxes(cube.clone(), 1, 2);
        let _: T3<f32, 3, 2, 4> = moveaxis(cube, 1, 0);

        let reshaped: T1<f32, 6> = reshape(x.clone());
        let transposed: T2<f32, 3, 2> = transpose(x.clone());
        let gathered: T3<f32, 2, 2, 2> = gather(x.clone(), indices, 1);
        let sliced: T2<f32, 2, 2> = slice(x.clone(), [0, 1]);
        let concatenated: T2<f32, 4, 3> = concat(x.clone(), y.clone(), 0);
        let stacked: T3<f32, 2, 2, 3> = stack(x, y, 0);
        let tiled: T2<f32, 4, 6> = tile(row, [4, 2]);

        let graph = context
            .finish(
                "shape_indexing",
                "llvm-cpu",
                (
                    reshaped,
                    transposed,
                    gathered,
                    sliced,
                    concatenated,
                    stacked,
                ),
            )
            .unwrap();

        assert_eq!(graph.outputs.len(), 6);
        assert_eq!(graph.outputs[0], T1::<f32, 6>::tensor_type());
        assert_eq!(graph.outputs[2], T3::<f32, 2, 2, 2>::tensor_type());
        assert_eq!(tiled.expr().clone(), tiled.into_expr());
    }
}
