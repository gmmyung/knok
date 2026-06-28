use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: T2<f32, 4, 4>) -> T2<f32, 4, 4> {
    relu(matmul(x.clone(), x) + 1.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn elementwise_and_predicate(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    r#where(greater(x.clone(), 0.0), clip(maximum(x, y), 0.0, 6.0), 0.0)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn math_variants(x: Tensor1<f32, 4>) -> (Tensor1<f32, 4>, Tensor1<f32, 4>, Tensor1<f32, 4>) {
    (exp2(x.clone()), log1p(square(x.clone())), tanh(cos(x)))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn axis_reductions(x: Tensor2<f32, 2, 3>) -> (Tensor1<f32, 2>, Tensor1<i64, 2>) {
    (sum_axis(x.clone(), 1), argmax_axis(x, 1))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn bool_reductions(x: Tensor2<bool, 2, 3>) -> Tensor1<bool, 2> {
    any_axis(logical_or(x.clone(), logical_not(x)), 1)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn static_shape_ops(x: Tensor2<f32, 2, 5>) -> (Tensor2<f32, 2, 2>, Tensor2<f32, 2, 3>) {
    split(x, 1, [2, 3])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn indexed_shape_ops(x: Tensor2<f32, 2, 3>, indices: Tensor2<i64, 2, 2>) -> Tensor3<f32, 2, 2, 2> {
    let gathered: Tensor3<f32, 2, 2, 2> = gather(x, indices, 1);
    permute(gathered, [0, 2, 1])
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn layout_ops(x: Tensor3<f32, 2, 3, 4>) -> (Tensor3<f32, 3, 4, 2>, Tensor3<f32, 3, 4, 2>) {
    (transpose_axes(x.clone(), [1, 2, 0]), moveaxis(x, 0, 2))
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn reshape_ops(
    x: Tensor2<f32, 2, 3>,
) -> (Tensor1<f32, 6>, Tensor3<f32, 1, 2, 3>, Tensor2<f32, 2, 3>) {
    let flat: Tensor1<f32, 6> = reshape(x.clone());
    let expanded: Tensor3<f32, 1, 2, 3> = unsqueeze(x);
    let restored: Tensor2<f32, 2, 3> = squeeze(expanded.clone());
    (flat, expanded, restored)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn creation_ops() -> (Tensor1<i32, 4>, Tensor1<f32, 5>, Tensor2<bool, 2, 2>) {
    (arange_step(0, 8, 2), linspace(0.0, 1.0), identity())
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn linalg_ops(x: Tensor2<f32, 2, 3>, y: Tensor2<f32, 3, 2>) -> Tensor2<f32, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn matmul_variants(
    x: Tensor4<f32, 2, 1, 2, 3>,
    y: Tensor3<f32, 3, 3, 2>,
) -> Tensor4<f32, 2, 3, 2, 2> {
    matmul(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn conv_ops(x: Tensor4<f32, 1, 3, 3, 1>, k: Tensor4<f32, 2, 2, 1, 1>) -> Tensor4<f32, 1, 2, 2, 1> {
    conv2d_options(x, k, Conv2dOptions::new().padding(1, 1, 1, 1).stride(2, 2))
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward,
        elementwise_and_predicate,
        math_variants,
        axis_reductions,
        bool_reductions,
        static_shape_ops,
        indexed_shape_ops,
        layout_ops,
        reshape_ops,
        creation_ops,
        linalg_ops,
        matmul_variants,
        conv_ops
    );
}
