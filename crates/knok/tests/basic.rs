use knok::prelude::*;
use knok::runtime::raw;
use knok::{DType, Engine, Error, GraphArtifact, GraphArtifactVariant, RuntimeConfig, TensorDesc};

#[knok::graph(backend = Backend::LlvmCpu)]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn add_sub4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> (Tensor1<f32, 4>, Tensor1<f32, 4>) {
    (x + y, x - y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn add_sub4_product(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    let (sum, diff) = add_sub4(x, y);
    sum * diff
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn add4_f64(x: Tensor1<f64, 4>, y: Tensor1<f64, 4>) -> Tensor1<f64, 4> {
    x + y
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn arithmetic4_i32(x: Tensor1<i32, 4>, y: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    ((x - y) * 2i32) / 4i32
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn add4_i64(x: Tensor1<i64, 4>, y: Tensor1<i64, 4>) -> Tensor1<i64, 4> {
    x + y
}

#[cfg(feature = "half")]
#[knok::graph(backend = Backend::LlvmCpu)]
fn add4_f16(x: Tensor1<f16, 4>, y: Tensor1<f16, 4>) -> Tensor1<f16, 4> {
    x + y
}

#[cfg(feature = "half")]
#[knok::graph(backend = Backend::LlvmCpu)]
fn identity4_bf16(x: Tensor1<bf16, 4>) -> Tensor1<bf16, 4> {
    x
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sum4_i32(x: Tensor1<i32, 4>) -> Tensor0<i32> {
    sum(x)
}

#[knok::graph(backends = [backend(Backend::LlvmCpu, driver = Driver::LocalTask)])]
fn add4_bundle(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn arithmetic4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    ((x - y) * 2.0) / 4.0
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn neg4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    -x
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn let_chain4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    let sum = x + y;
    let shifted = sum - 1.0;
    relu(shifted)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn relu4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn abs4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    abs(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn abs4_i32(x: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    abs(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn minimum4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    minimum(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn maximum4_i32(x: Tensor1<i32, 4>, y: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    maximum(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn clip4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    clip(x, 0.0, 2.0)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn pow4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    pow(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn mm2x2(x: Tensor2<f32, 2, 2>, y: Tensor2<f32, 2, 2>) -> Tensor2<f32, 2, 2> {
    matmul(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn transpose2x3(x: Tensor2<f32, 2, 3>) -> Tensor2<f32, 3, 2> {
    transpose(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn reshape2x2(x: Tensor1<f32, 4>) -> Tensor2<f32, 2, 2> {
    reshape::<Tensor2<f32, 2, 2>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn flatten2x2(x: Tensor2<f32, 2, 2>) -> Tensor1<f32, 4> {
    reshape::<Tensor1<f32, 4>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn broadcast1to4(x: Tensor1<f32, 1>) -> Tensor1<f32, 4> {
    broadcast::<Tensor1<f32, 4>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn zeros_like6d_i32(x: Tensor6<i32, 1, 1, 1, 1, 2, 3>) -> Tensor6<i32, 1, 1, 1, 1, 2, 3> {
    zeros_like(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn ones_like_bool2x2(x: Tensor2<bool, 2, 2>) -> Tensor2<bool, 2, 2> {
    ones_like(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn full_like4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    full_like(x, 3.5)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn arange_i32_0_8_2() -> Tensor1<i32, 4> {
    arange::<Tensor1<i32, 4>>(0, 8, 2)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn arange_f32_descending() -> Tensor1<f32, 4> {
    arange::<Tensor1<f32, 4>>(1.5, -0.5, -0.5)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn arange_f32_tiny() -> Tensor1<f32, 3> {
    arange::<Tensor1<f32, 3>>(1.0e-20, 4.0e-20, 1.0e-20)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn linspace_f32_0_1() -> Tensor1<f32, 5> {
    linspace::<Tensor1<f32, 5>>(0.0, 1.0)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn linspace_f64_tiny() -> Tensor1<f64, 3> {
    linspace::<Tensor1<f64, 3>>(1.0e-20f64, 3.0e-20f64)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn linspace_i64_2_8() -> Tensor1<i64, 4> {
    linspace::<Tensor1<i64, 4>>(2i64, 8i64)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn eye3_f32() -> Tensor2<f32, 3, 3> {
    eye::<Tensor2<f32, 3, 3>>()
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn identity2_bool() -> Tensor2<bool, 2, 2> {
    identity::<Tensor2<bool, 2, 2>>()
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sum4(x: Tensor1<f32, 4>) -> Tensor0<f32> {
    sum(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sum2x2(x: Tensor2<f32, 2, 2>) -> Tensor0<f32> {
    sum(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sum2x3_axis0(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 3> {
    sum::<0>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sum2x3_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    sum::<1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn mean2x3_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    mean::<1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn softmax2x3_axis1(x: Tensor2<f32, 2, 3>) -> Tensor2<f32, 2, 3> {
    softmax::<1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn add_bias2x3(x: Tensor2<f32, 2, 3>, bias: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
    x + bias
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn broadcast_bias2x3(bias: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
    broadcast::<Tensor2<f32, 2, 3>>(bias)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn add_column2x3(x: Tensor2<f32, 2, 3>, column: Tensor2<f32, 2, 1>) -> Tensor2<f32, 2, 3> {
    x + column
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn add_bias6d(
    x: Tensor6<f32, 1, 1, 1, 1, 1, 3>,
    bias: Tensor1<f32, 3>,
) -> Tensor6<f32, 1, 1, 1, 1, 1, 3> {
    x + bias
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn add_channel_bias4d(
    x: Tensor4<f32, 1, 2, 2, 3>,
    bias: Tensor1<f32, 3>,
) -> Tensor4<f32, 1, 2, 2, 3> {
    x + bias
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sum4d_axis3(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor3<f32, 1, 2, 2> {
    sum::<3>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn mean4d_axis2(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor3<f32, 1, 2, 3> {
    mean::<2>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn argmax4d_axis3(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor3<i64, 1, 2, 2> {
    argmax::<3>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sum6d_axis5(x: Tensor6<f32, 1, 1, 1, 1, 1, 3>) -> Tensor5<f32, 1, 1, 1, 1, 1> {
    sum::<5>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn slice2x4_to2x2(x: Tensor2<f32, 2, 4>) -> Tensor2<f32, 2, 2> {
    slice::<Tensor2<f32, 2, 2>, 0, 1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn slice4d_middle(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor4<f32, 1, 1, 2, 2> {
    slice::<Tensor4<f32, 1, 1, 2, 2>, 0, 1, 0, 1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn take2x3_axis0(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 3> {
    take::<0, 1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn take2x3_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    take::<1, 2>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn take4d_axis3(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor3<f32, 1, 2, 2> {
    take::<3, 1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn squeeze1x2x1x3(x: Tensor4<f32, 1, 2, 1, 3>) -> Tensor2<f32, 2, 3> {
    squeeze::<Tensor2<f32, 2, 3>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn unsqueeze2x3(x: Tensor2<f32, 2, 3>) -> Tensor4<f32, 1, 2, 1, 3> {
    unsqueeze::<Tensor4<f32, 1, 2, 1, 3>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn concat_rows2x2(x: Tensor2<f32, 1, 2>, y: Tensor2<f32, 2, 2>) -> Tensor2<f32, 3, 2> {
    concat::<0>(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn concat_cols2x2(x: Tensor2<f32, 2, 1>, y: Tensor2<f32, 2, 2>) -> Tensor2<f32, 2, 3> {
    concat::<1>(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn concat4d_channels(
    x: Tensor4<f32, 1, 1, 1, 1>,
    y: Tensor4<f32, 1, 1, 1, 2>,
) -> Tensor4<f32, 1, 1, 1, 3> {
    concat::<3>(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn stack_vectors(x: Tensor1<f32, 3>, y: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
    stack::<0>(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn reshape1_to_3d(x: Tensor1<f32, 8>) -> Tensor3<f32, 2, 2, 2> {
    reshape::<Tensor3<f32, 2, 2, 2>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn flatten4d(x: Tensor4<f32, 1, 2, 2, 1>) -> Tensor1<f32, 4> {
    reshape::<Tensor1<f32, 4>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn square4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    square(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn square4_i32(x: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    square(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn reciprocal4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    reciprocal(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn reciprocal4_i32(x: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    reciprocal(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn exp4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    exp(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn exp_variants4(x: Tensor1<f32, 4>) -> (Tensor1<f32, 4>, Tensor1<f32, 4>) {
    (exp2(x), expm1(x))
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn log4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    log(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn log_variants4(
    x: Tensor1<f32, 4>,
    y: Tensor1<f32, 4>,
    z: Tensor1<f32, 4>,
) -> (Tensor1<f32, 4>, Tensor1<f32, 4>, Tensor1<f32, 4>) {
    (log2(x), log10(y), log1p(z))
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sqrt4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    sqrt(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn rounding4(
    x: Tensor1<f32, 4>,
) -> (
    Tensor1<f32, 4>,
    Tensor1<f32, 4>,
    Tensor1<f32, 4>,
    Tensor1<f32, 4>,
) {
    (floor(x), ceil(x), round(x), rint(x))
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn trig4(x: Tensor1<f32, 4>) -> (Tensor1<f32, 4>, Tensor1<f32, 4>, Tensor1<f32, 4>) {
    (sin(x), cos(x), tan(x))
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn tanh4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    tanh(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sigmoid4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    sigmoid(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn softmax4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    softmax(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn mean4(x: Tensor1<f32, 4>) -> Tensor0<f32> {
    mean(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn argmax4(x: Tensor1<f32, 4>) -> Tensor0<i64> {
    argmax(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn argmax4_i32(x: Tensor1<i32, 4>) -> Tensor0<i64> {
    argmax(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn argmax2x3(x: Tensor2<f32, 2, 3>) -> Tensor0<i64> {
    argmax(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn argmax2x3_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<i64, 2> {
    argmax::<1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn batch_mm(x: Tensor3<f32, 1, 2, 3>, y: Tensor3<f32, 1, 3, 2>) -> Tensor3<f32, 1, 2, 2> {
    matmul(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn scalar_add(x: Tensor0<f32>, y: Tensor0<f32>) -> Tensor0<f32> {
    x + y
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn sum4_axis0(x: Tensor1<f32, 4>) -> Tensor0<f32> {
    sum::<0>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn dot4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor0<f32> {
    matmul(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn matvec2x3(x: Tensor2<f32, 2, 3>, y: Tensor1<f32, 3>) -> Tensor1<f32, 2> {
    matmul(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn vecmat3x2(x: Tensor1<f32, 3>, y: Tensor2<f32, 3, 2>) -> Tensor1<f32, 2> {
    matmul(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn broadcast_batch_mm(
    x: Tensor4<f32, 2, 1, 2, 3>,
    y: Tensor3<f32, 3, 3, 2>,
) -> Tensor4<f32, 2, 3, 2, 2> {
    matmul(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn batch_mm5d(
    x: Tensor5<f32, 1, 1, 2, 2, 3>,
    y: Tensor5<f32, 1, 1, 2, 3, 2>,
) -> Tensor5<f32, 1, 1, 2, 2, 2> {
    matmul(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn batch_mm6d(
    x: Tensor6<f32, 1, 1, 1, 1, 2, 3>,
    y: Tensor6<f32, 1, 1, 1, 1, 3, 2>,
) -> Tensor6<f32, 1, 1, 1, 1, 2, 2> {
    matmul(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn permute4d(x: Tensor4<f32, 1, 2, 3, 4>) -> Tensor4<f32, 1, 3, 2, 4> {
    permute::<Tensor4<f32, 1, 3, 2, 4>, 0, 2, 1, 3>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn transpose6d(x: Tensor6<f32, 1, 2, 1, 3, 2, 4>) -> Tensor6<f32, 4, 2, 3, 1, 2, 1> {
    transpose(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn slice6d(x: Tensor6<f32, 1, 1, 1, 1, 2, 4>) -> Tensor6<f32, 1, 1, 1, 1, 1, 2> {
    slice::<Tensor6<f32, 1, 1, 1, 1, 1, 2>, 0, 0, 0, 0, 1, 1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn take6d_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<f32, 1, 1, 1, 1, 2> {
    take::<5, 1>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn squeeze6d(x: Tensor6<f32, 1, 2, 1, 3, 1, 4>) -> Tensor3<f32, 2, 3, 4> {
    squeeze::<Tensor3<f32, 2, 3, 4>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn unsqueeze3d_to6d(x: Tensor3<f32, 2, 3, 4>) -> Tensor6<f32, 1, 2, 1, 3, 1, 4> {
    unsqueeze::<Tensor6<f32, 1, 2, 1, 3, 1, 4>>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn concat6d(
    x: Tensor6<f32, 1, 1, 1, 1, 2, 1>,
    y: Tensor6<f32, 1, 1, 1, 1, 2, 2>,
) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
    concat::<5>(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn stack5d_to6d(
    x: Tensor5<f32, 1, 1, 1, 2, 3>,
    y: Tensor5<f32, 1, 1, 1, 2, 3>,
) -> Tensor6<f32, 1, 1, 1, 2, 2, 3> {
    stack::<4>(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn softmax6d_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
    softmax::<5>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn mean6d_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<f32, 1, 1, 1, 1, 2> {
    mean::<5>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn argmax6d_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<i64, 1, 1, 1, 1, 2> {
    argmax::<5>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn any6d_axis5(x: Tensor6<bool, 1, 1, 1, 1, 2, 3>) -> Tensor5<bool, 1, 1, 1, 1, 2> {
    any::<5>(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn where6d(
    x: Tensor6<f32, 1, 1, 1, 1, 2, 3>,
    y: Tensor1<f32, 3>,
) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
    r#where(greater(x, y), x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn conv2d_1x1(
    x: Tensor4<f32, 1, 2, 2, 1>,
    k: Tensor4<f32, 1, 1, 1, 1>,
) -> Tensor4<f32, 1, 2, 2, 1> {
    conv2d(x, k)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn conv2d_pad_stride(
    x: Tensor4<f32, 1, 3, 3, 1>,
    k: Tensor4<f32, 2, 2, 1, 1>,
) -> Tensor4<f32, 1, 2, 2, 1> {
    conv2d::<Pad<1, 1, 1, 1>, Stride<2, 2>>(x, k)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn conv2d_dilated(
    x: Tensor4<f32, 1, 3, 3, 1>,
    k: Tensor4<f32, 2, 2, 1, 1>,
) -> Tensor4<f32, 1, 1, 1, 1> {
    conv2d::<Dilation<2, 2>>(x, k)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn conv2d_groups2(
    x: Tensor4<f32, 1, 1, 1, 4>,
    k: Tensor4<f32, 1, 1, 2, 4>,
) -> Tensor4<f32, 1, 1, 1, 4> {
    conv2d::<Groups<2>>(x, k)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn layer4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn composed4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    layer4(x + y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn composed_twice4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    let first = composed4(x, y);
    layer4(first - 2.0)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn select_positive4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    r#where(greater(x, 0.0), x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn select_positive_literals4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    r#where(greater(x, 0.0), 1.0, 0.0)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn select_with_bool_input4(
    condition: Tensor1<bool, 4>,
    x: Tensor1<f32, 4>,
    y: Tensor1<f32, 4>,
) -> Tensor1<f32, 4> {
    r#where(condition, x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn compare_greater4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
    greater(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn compare_greater_equal4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
    greater_equal(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn compare_less4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
    less(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn compare_less_equal4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
    less_equal(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn compare_equal4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
    equal(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn compare_not_equal4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
    not_equal(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn bool_equal4(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    equal(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn bool_not_equal4(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    not_equal(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn logical_from_comparisons4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
    logical_xor(greater(x, 0.0), less_equal(y, 2.0))
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn logical_and_input4(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    logical_and(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn logical_or_input4(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    logical_or(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn logical_xor_input4(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    logical_xor(x, y)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn logical_not_input4(x: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
    logical_not(x)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn all_positive4(x: Tensor1<f32, 4>) -> Tensor0<bool> {
    all(greater(x, 0.0))
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn any_nan4(x: Tensor1<f32, 4>) -> Tensor0<bool> {
    any(isnan(x))
}

#[test]
fn add_graph_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);
    let output = add4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn multi_output_graph_runs() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let (sum, diff) = add_sub4_run(&engine, x, y).unwrap();

    assert_eq!(sum.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(diff.into_vec(), vec![-9.0, -18.0, -27.0, -36.0]);
}

#[test]
fn multi_output_graph_call_can_be_destructured() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);

    let output = add_sub4_product_run(&engine, x, y).unwrap();

    assert_eq!(output.into_vec(), vec![-99.0, -396.0, -891.0, -1584.0]);
}

#[cfg(feature = "half")]
#[test]
fn half_graphs_run() {
    let f16_output = add4_f16(
        Tensor1::from_array([
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ]),
        Tensor1::from_array([
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
            f16::from_f32(40.0),
        ]),
    )
    .unwrap();
    assert_eq!(
        f16_output.into_vec(),
        vec![
            f16::from_f32(11.0),
            f16::from_f32(22.0),
            f16::from_f32(33.0),
            f16::from_f32(44.0)
        ]
    );

    let bf16_input = Tensor1::from_array([
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
    ]);
    let bf16_output = identity4_bf16(bf16_input.clone()).unwrap();
    assert_eq!(bf16_output, bf16_input);
}

#[test]
fn artifact_records_backend_variant() {
    let artifact = add4_artifact();
    assert_eq!(artifact.function_name, "knok.add4");
    assert_eq!(
        artifact.input_descs,
        &[
            TensorDesc::new(DType::F32, &[4]),
            TensorDesc::new(DType::F32, &[4])
        ]
    );
    assert_eq!(artifact.output_descs, &[TensorDesc::new(DType::F32, &[4])]);
    assert_eq!(artifact.variants.len(), 1);
    let variant = artifact.first_variant().unwrap();
    assert_eq!(variant.backend, "llvm-cpu");
    assert_eq!(variant.driver, "local-task");
    assert!(variant
        .compile_flags
        .contains(&"--iree-hal-target-backends=llvm-cpu"));
}

#[test]
fn explicit_backend_bundle_syntax_runs() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let output = add4_bundle_run(
        &engine,
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();

    assert_eq!(add4_bundle_artifact().variants.len(), 1);
    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn reusable_engine_runs_graph_repeatedly() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();

    let first = add4_run(
        &engine,
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    let second = add4_run(
        &engine,
        Tensor1::from_array([5.0, 6.0, 7.0, 8.0]),
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();

    assert_eq!(engine.driver_name(), "local-task");
    assert_eq!(first.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(second.into_vec(), vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn engine_reports_missing_artifact_variant_for_driver() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let artifact = add4_artifact();
    let variant = artifact.first_variant().unwrap();
    let variants = Box::leak(Box::new([GraphArtifactVariant {
        backend: "metal-spirv",
        driver: "metal",
        ..variant
    }]));
    let artifact = GraphArtifact {
        variants,
        ..artifact
    };
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [10.0, 20.0, 30.0, 40.0];
    let error = engine
        .invoke(
            artifact,
            &[raw::Input::F32(&[4], &x), raw::Input::F32(&[4], &y)],
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::MissingArtifactVariant {
            function_name: "knok.add4",
            ..
        }
    ));
}

#[test]
fn engine_rejects_raw_input_count_mismatch() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = [1.0, 2.0, 3.0, 4.0];
    let error = engine
        .invoke(add4_artifact(), &[raw::Input::F32(&[4], &x)])
        .unwrap_err();

    assert!(matches!(
        error,
        Error::InputCountMismatch {
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn engine_rejects_raw_input_dtype_mismatch() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let y = [10.0f64, 20.0, 30.0, 40.0];
    let error = engine
        .invoke(
            add4_artifact(),
            &[raw::Input::F32(&[4], &x), raw::Input::F64(&[4], &y)],
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::InputDTypeMismatch {
            index: 1,
            expected: DType::F32,
            actual: DType::F64
        }
    ));
}

#[test]
fn engine_rejects_raw_input_shape_mismatch() {
    let engine = Engine::new(RuntimeConfig::auto()).unwrap();
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let y = [10.0f32, 20.0, 30.0, 40.0];
    let error = engine
        .invoke(
            add4_artifact(),
            &[raw::Input::F32(&[2, 2], &x), raw::Input::F32(&[4], &y)],
        )
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Shape {
            expected: &[4],
            actual
        } if actual == vec![2, 2]
    ));
}

#[test]
fn non_f32_numeric_graphs_run() {
    let f64_output = add4_f64(
        Tensor1::from_array([1.0f64, 2.0, 3.0, 4.0]),
        Tensor1::from_array([10.0f64, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    assert_eq!(f64_output.into_vec(), vec![11.0f64, 22.0, 33.0, 44.0]);

    let i32_output = arithmetic4_i32(
        Tensor1::from_array([9i32, 10, 11, 12]),
        Tensor1::from_array([1i32, 2, 3, 4]),
    )
    .unwrap();
    assert_eq!(i32_output.into_vec(), vec![4i32, 4, 4, 4]);

    let i64_output = add4_i64(
        Tensor1::from_array([1i64, 2, 3, 4]),
        Tensor1::from_array([10i64, 20, 30, 40]),
    )
    .unwrap();
    assert_eq!(i64_output.into_vec(), vec![11i64, 22, 33, 44]);

    let sum_output = sum4_i32(Tensor1::from_array([1i32, 2, 3, 4])).unwrap();
    assert_eq!(sum_output.into_vec(), vec![10i32]);
}

#[test]
fn arithmetic_graph_runs() {
    let x = Tensor1::from_array([9.0, 10.0, 11.0, 12.0]);
    let y = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let output = arithmetic4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn unary_negation_graph_runs() {
    let x = Tensor1::from_array([1.0, -2.0, 3.5, -4.5]);
    let output = neg4(x).unwrap();
    assert_eq!(output.into_vec(), vec![-1.0, 2.0, -3.5, 4.5]);
}

#[test]
fn let_chain_and_scalar_broadcast_run() {
    let x = Tensor1::from_array([1.0, 2.0, -3.0, 4.0]);
    let y = Tensor1::from_array([1.0, -10.0, 10.0, -2.0]);
    let output = let_chain4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![1.0, 0.0, 6.0, 1.0]);
}

#[test]
fn relu_graph_runs() {
    let x = Tensor1::from_array([-1.0, 2.0, -3.0, 4.0]);
    let output = relu4(x).unwrap();
    assert_eq!(output.into_vec(), vec![0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn elementwise_op_graphs_run() {
    let abs = abs4(Tensor1::from_array([-1.0, 2.0, -3.5, 4.5])).unwrap();
    assert_eq!(abs.into_vec(), vec![1.0, 2.0, 3.5, 4.5]);

    let abs_i32 = abs4_i32(Tensor1::from_array([-1, 2, -3, 4])).unwrap();
    assert_eq!(abs_i32.into_vec(), vec![1, 2, 3, 4]);

    let min = minimum4(
        Tensor1::from_array([1.0, 5.0, -3.0, 10.0]),
        Tensor1::from_array([2.0, 4.0, -4.0, 20.0]),
    )
    .unwrap();
    assert_eq!(min.into_vec(), vec![1.0, 4.0, -4.0, 10.0]);

    let max_i32 = maximum4_i32(
        Tensor1::from_array([1, 5, -3, 10]),
        Tensor1::from_array([2, 4, -4, 20]),
    )
    .unwrap();
    assert_eq!(max_i32.into_vec(), vec![2, 5, -3, 20]);

    let clipped = clip4(Tensor1::from_array([-1.0, 0.5, 3.0, 2.0])).unwrap();
    assert_eq!(clipped.into_vec(), vec![0.0, 0.5, 2.0, 2.0]);

    let pow = pow4(
        Tensor1::from_array([2.0, 3.0, 4.0, 9.0]),
        Tensor1::from_array([3.0, 2.0, 0.5, 0.5]),
    )
    .unwrap();
    let output = pow.into_vec();
    let expected = [8.0, 9.0, 2.0, 3.0];
    for (actual, expected) in output.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-5);
    }
}

#[test]
fn matmul_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let y = Tensor2::from_array([[5.0, 6.0], [7.0, 8.0]]);
    let output = mm2x2(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn scalar_tensor_graph_runs() {
    let output = scalar_add(Tensor0::from_scalar(2.0), Tensor0::from_scalar(3.5)).unwrap();
    assert_eq!(output.into_vec(), vec![5.5]);

    let sum = sum4_axis0(Tensor1::from_array([1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_eq!(sum.into_vec(), vec![10.0]);
}

#[test]
fn expanded_matmul_graphs_run() {
    let dot = dot4(
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    assert_eq!(dot.into_vec(), vec![300.0]);

    let matvec = matvec2x3(
        Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        Tensor1::from_array([7.0, 8.0, 9.0]),
    )
    .unwrap();
    assert_eq!(matvec.into_vec(), vec![50.0, 122.0]);

    let vecmat = vecmat3x2(
        Tensor1::from_array([1.0, 2.0, 3.0]),
        Tensor2::from_array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]),
    )
    .unwrap();
    assert_eq!(vecmat.into_vec(), vec![40.0, 46.0]);

    let broadcast =
        broadcast_batch_mm(Tensor4::filled(1.0), Tensor3::<f32, 3, 3, 2>::filled(2.0)).unwrap();
    assert_eq!(broadcast.into_vec(), vec![6.0; 24]);

    let rank5_batch = batch_mm5d(
        Tensor5::<f32, 1, 1, 2, 2, 3>::filled(1.0),
        Tensor5::<f32, 1, 1, 2, 3, 2>::filled(2.0),
    )
    .unwrap();
    assert_eq!(rank5_batch.into_vec(), vec![6.0; 8]);

    let rank6_batch = batch_mm6d(
        Tensor6::<f32, 1, 1, 1, 1, 2, 3>::filled(1.0),
        Tensor6::<f32, 1, 1, 1, 1, 3, 2>::filled(2.0),
    )
    .unwrap();
    assert_eq!(rank6_batch.into_vec(), vec![6.0; 4]);
}

#[test]
fn transpose_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let output = transpose2x3(x).unwrap();
    assert_eq!(output.into_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn permute_graph_runs() {
    let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let output = permute4d(Tensor4::<f32, 1, 2, 3, 4>::from_vec(values.clone()).unwrap()).unwrap();
    let mut expected = Vec::new();
    for d0 in 0..1 {
        for d2 in 0..3 {
            for d1 in 0..2 {
                for d3 in 0..4 {
                    expected.push(values[((d0 * 2 + d1) * 3 + d2) * 4 + d3]);
                }
            }
        }
    }
    assert_eq!(output.into_vec(), expected);
}

#[test]
fn rank6_numpy_style_shape_ops_run() {
    let values = (0..48).map(|value| value as f32).collect::<Vec<_>>();
    let transposed =
        transpose6d(Tensor6::<f32, 1, 2, 1, 3, 2, 4>::from_vec(values.clone()).unwrap()).unwrap();
    assert_eq!(
        transposed.into_vec(),
        reverse_axes_flat(&values, &[1, 2, 1, 3, 2, 4])
    );

    let sliced = slice6d(
        Tensor6::<f32, 1, 1, 1, 1, 2, 4>::from_vec((1..=8).map(|v| v as f32).collect()).unwrap(),
    )
    .unwrap();
    assert_eq!(sliced.into_vec(), vec![6.0, 7.0]);

    let taken = take6d_axis5(
        Tensor6::<f32, 1, 1, 1, 1, 2, 3>::from_vec((1..=6).map(|v| v as f32).collect()).unwrap(),
    )
    .unwrap();
    assert_eq!(taken.into_vec(), vec![2.0, 5.0]);

    let rank3_values = (1..=24).map(|value| value as f32).collect::<Vec<_>>();
    let squeezed =
        squeeze6d(Tensor6::<f32, 1, 2, 1, 3, 1, 4>::from_vec(rank3_values.clone()).unwrap())
            .unwrap();
    assert_eq!(squeezed.into_vec(), rank3_values);

    let rank3_values = (1..=24).map(|value| value as f32).collect::<Vec<_>>();
    let unsqueezed =
        unsqueeze3d_to6d(Tensor3::<f32, 2, 3, 4>::from_vec(rank3_values.clone()).unwrap()).unwrap();
    assert_eq!(unsqueezed.into_vec(), rank3_values);

    let concatenated = concat6d(
        Tensor6::<f32, 1, 1, 1, 1, 2, 1>::from_vec(vec![1.0, 2.0]).unwrap(),
        Tensor6::<f32, 1, 1, 1, 1, 2, 2>::from_vec(vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        concatenated.into_vec(),
        vec![1.0, 10.0, 20.0, 2.0, 30.0, 40.0]
    );

    let stacked = stack5d_to6d(
        Tensor5::<f32, 1, 1, 1, 2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        Tensor5::<f32, 1, 1, 1, 2, 3>::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        stacked.into_vec(),
        vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0]
    );
}

#[test]
fn reshape_graph_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let output = reshape2x2(x).unwrap();
    assert_eq!(output.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn flatten_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let output = flatten2x2(x).unwrap();
    assert_eq!(output.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn broadcast_graph_runs() {
    let x = Tensor1::from_array([7.0]);
    let output = broadcast1to4(x).unwrap();
    assert_eq!(output.into_vec(), vec![7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn static_creator_graphs_run() {
    let zeros = zeros_like6d_i32(Tensor6::<i32, 1, 1, 1, 1, 2, 3>::filled(9)).unwrap();
    assert_eq!(zeros.into_vec(), vec![0; 6]);

    let bool_ones =
        ones_like_bool2x2(Tensor2::from_array([[false, true], [false, false]])).unwrap();
    assert_eq!(bool_ones.into_vec(), vec![true, true, true, true]);

    let full = full_like4(Tensor1::from_array([0.0, 1.0, 2.0, 3.0])).unwrap();
    assert_eq!(full.into_vec(), vec![3.5, 3.5, 3.5, 3.5]);

    let arange_i32 = arange_i32_0_8_2().unwrap();
    assert_eq!(arange_i32.into_vec(), vec![0, 2, 4, 6]);

    let arange_f32 = arange_f32_descending().unwrap();
    assert_eq!(arange_f32.into_vec(), vec![1.5, 1.0, 0.5, 0.0]);

    let arange_tiny = arange_f32_tiny().unwrap().into_vec();
    assert_close_with_tolerance(&arange_tiny, &[1.0e-20, 2.0e-20, 3.0e-20], 1.0e-25);

    let linspace_f32 = linspace_f32_0_1().unwrap().into_vec();
    assert_close(&linspace_f32, &[0.0, 0.25, 0.5, 0.75, 1.0]);

    let linspace_tiny = linspace_f64_tiny().unwrap().into_vec();
    assert_close_f64(&linspace_tiny, &[1.0e-20, 2.0e-20, 3.0e-20]);

    let linspace_i64 = linspace_i64_2_8().unwrap();
    assert_eq!(linspace_i64.into_vec(), vec![2, 4, 6, 8]);

    let eye = eye3_f32().unwrap();
    assert_eq!(
        eye.into_vec(),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );

    let identity = identity2_bool().unwrap();
    assert_eq!(identity.into_vec(), vec![true, false, false, true]);
}

#[test]
fn sum_graph_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let output = sum4(x).unwrap();
    assert_eq!(output.into_vec(), vec![10.0]);
}

#[test]
fn rank2_sum_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let output = sum2x2(x).unwrap();
    assert_eq!(output.into_vec(), vec![10.0]);
}

#[test]
fn axis_reduction_graphs_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);

    let axis0 = sum2x3_axis0(x.clone()).unwrap();
    assert_eq!(axis0.into_vec(), vec![11.0, 22.0, 33.0]);

    let axis1 = sum2x3_axis1(x.clone()).unwrap();
    assert_eq!(axis1.into_vec(), vec![6.0, 60.0]);

    let mean = mean2x3_axis1(x).unwrap();
    assert_eq!(mean.into_vec(), vec![2.0, 20.0]);

    let x4 = Tensor4::from_array([[
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ]]);
    let sum4d = sum4d_axis3(x4.clone()).unwrap();
    assert_eq!(sum4d.into_vec(), vec![6.0, 15.0, 24.0, 33.0]);

    let mean4d = mean4d_axis2(x4).unwrap();
    assert_eq!(mean4d.into_vec(), vec![2.5, 3.5, 4.5, 8.5, 9.5, 10.5]);

    let sum6d = sum6d_axis5(Tensor6::from_array([[[[[[1.0, 2.0, 3.0]]]]]])).unwrap();
    assert_eq!(sum6d.into_vec(), vec![6.0]);
}

#[test]
fn axis_softmax_graph_runs() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]);
    let output = softmax2x3_axis1(x).unwrap().into_vec();

    let expected = [
        0.09003057f32,
        0.24472848,
        0.66524094,
        0.33333334,
        0.33333334,
        0.33333334,
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-5);
    }
}

#[test]
fn rank6_numpy_style_reductions_run() {
    let input =
        Tensor6::<f32, 1, 1, 1, 1, 2, 3>::from_vec(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();

    let softmax = softmax6d_axis5(input.clone()).unwrap().into_vec();
    assert_close(
        &softmax,
        &[
            0.09003057, 0.24472848, 0.66524094, 0.33333334, 0.33333334, 0.33333334,
        ],
    );

    let mean = mean6d_axis5(input.clone()).unwrap();
    assert_close(&mean.into_vec(), &[2.0, 1.0]);

    let argmax = argmax6d_axis5(input).unwrap();
    assert_eq!(argmax.into_vec(), vec![2, 0]);

    let any = any6d_axis5(
        Tensor6::<bool, 1, 1, 1, 1, 2, 3>::from_vec(vec![false, false, true, false, false, false])
            .unwrap(),
    )
    .unwrap();
    assert_eq!(any.into_vec(), vec![true, false]);
}

#[test]
fn rank_broadcasting_graphs_run() {
    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
    let bias = Tensor1::from_array([100.0, 200.0, 300.0]);

    let added = add_bias2x3(x, bias.clone()).unwrap();
    assert_eq!(
        added.into_vec(),
        vec![101.0, 202.0, 303.0, 110.0, 220.0, 330.0]
    );

    let broadcast = broadcast_bias2x3(bias).unwrap();
    assert_eq!(
        broadcast.into_vec(),
        vec![100.0, 200.0, 300.0, 100.0, 200.0, 300.0]
    );

    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
    let column = Tensor2::from_array([[100.0], [200.0]]);
    let added = add_column2x3(x, column).unwrap();
    assert_eq!(
        added.into_vec(),
        vec![101.0, 102.0, 103.0, 210.0, 220.0, 230.0]
    );

    let rank6_added = add_bias6d(
        Tensor6::from_array([[[[[[1.0, 2.0, 3.0]]]]]]),
        Tensor1::from_array([10.0, 20.0, 30.0]),
    )
    .unwrap();
    assert_eq!(rank6_added.into_vec(), vec![11.0, 22.0, 33.0]);

    let x4 = Tensor4::from_array([[
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ]]);
    let bias = Tensor1::from_array([100.0, 200.0, 300.0]);
    let added = add_channel_bias4d(x4, bias).unwrap();
    assert_eq!(
        added.into_vec(),
        vec![101.0, 202.0, 303.0, 104.0, 205.0, 306.0, 107.0, 208.0, 309.0, 110.0, 211.0, 312.0]
    );
}

#[test]
fn rank6_numpy_style_broadcasting_and_selection_run() {
    let selected = where6d(
        Tensor6::<f32, 1, 1, 1, 1, 2, 3>::from_vec(vec![1.0, 0.0, 4.0, 2.0, 5.0, 1.0]).unwrap(),
        Tensor1::from_array([2.0, 3.0, 1.0]),
    )
    .unwrap();
    assert_eq!(selected.into_vec(), vec![2.0, 3.0, 4.0, 2.0, 5.0, 1.0]);
}

#[test]
fn shape_and_indexing_graphs_run() {
    let sliced = slice2x4_to2x2(Tensor2::from_array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ]))
    .unwrap();
    assert_eq!(sliced.into_vec(), vec![2.0, 3.0, 20.0, 30.0]);

    let x = Tensor2::from_array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
    assert_eq!(
        take2x3_axis0(x.clone()).unwrap().into_vec(),
        vec![10.0, 20.0, 30.0]
    );
    assert_eq!(take2x3_axis1(x).unwrap().into_vec(), vec![3.0, 30.0]);

    let x4 = Tensor4::from_array([[
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ]]);
    assert_eq!(
        take4d_axis3(x4.clone()).unwrap().into_vec(),
        vec![2.0, 5.0, 8.0, 11.0]
    );
    assert_eq!(
        slice4d_middle(x4).unwrap().into_vec(),
        vec![8.0, 9.0, 11.0, 12.0]
    );

    let x4 = Tensor4::from_array([[[[1.0, 2.0, 3.0]], [[10.0, 20.0, 30.0]]]]);
    let squeezed = squeeze1x2x1x3(x4).unwrap();
    assert_eq!(squeezed.into_vec(), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);

    let x2 = Tensor2::from_array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
    let unsqueezed = unsqueeze2x3(x2).unwrap();
    assert_eq!(unsqueezed.into_vec(), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);

    let rows = concat_rows2x2(
        Tensor2::from_array([[1.0, 2.0]]),
        Tensor2::from_array([[10.0, 20.0], [30.0, 40.0]]),
    )
    .unwrap();
    assert_eq!(rows.into_vec(), vec![1.0, 2.0, 10.0, 20.0, 30.0, 40.0]);

    let cols = concat_cols2x2(
        Tensor2::from_array([[1.0], [2.0]]),
        Tensor2::from_array([[10.0, 20.0], [30.0, 40.0]]),
    )
    .unwrap();
    assert_eq!(cols.into_vec(), vec![1.0, 10.0, 20.0, 2.0, 30.0, 40.0]);

    let channels = concat4d_channels(
        Tensor4::from_array([[[[1.0]]]]),
        Tensor4::from_array([[[[10.0, 20.0]]]]),
    )
    .unwrap();
    assert_eq!(channels.into_vec(), vec![1.0, 10.0, 20.0]);

    let stacked = stack_vectors(
        Tensor1::from_array([1.0, 2.0, 3.0]),
        Tensor1::from_array([10.0, 20.0, 30.0]),
    )
    .unwrap();
    assert_eq!(stacked.into_vec(), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
}

#[test]
fn rank3_and_rank4_tensors_run() {
    let rank3 = reshape1_to_3d(Tensor1::from_array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ]))
    .unwrap();
    assert_eq!(
        rank3.into_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );

    let rank4 = Tensor4::from_array([[[[1.0], [2.0]], [[3.0], [4.0]]]]);
    let flat = flatten4d(rank4).unwrap();
    assert_eq!(flat.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn math_op_graphs_run() {
    let square_output = square4(Tensor1::from_array([-2.0, -1.5, 0.5, 3.0])).unwrap();
    assert_close(&square_output.into_vec(), &[4.0, 2.25, 0.25, 9.0]);

    let square_i32_output = square4_i32(Tensor1::from_array([-2, -1, 3, 4])).unwrap();
    assert_eq!(square_i32_output.into_vec(), vec![4, 1, 9, 16]);

    let reciprocal_output = reciprocal4(Tensor1::from_array([1.0, 2.0, 4.0, -0.5])).unwrap();
    assert_close(&reciprocal_output.into_vec(), &[1.0, 0.5, 0.25, -2.0]);

    let reciprocal_i32_output = reciprocal4_i32(Tensor1::from_array([1, -1, 2, -2])).unwrap();
    assert_eq!(reciprocal_i32_output.into_vec(), vec![1, -1, 0, 0]);

    let exp_output = exp4(Tensor1::from_array([0.0, 1.0, 2.0, 3.0])).unwrap();
    assert_close(
        &exp_output.into_vec(),
        &[1.0, core::f32::consts::E, 7.389056, 20.085537],
    );

    let (exp2_output, expm1_output) =
        exp_variants4(Tensor1::from_array([0.0, 1.0e-8, -1.0e-8, 1.0])).unwrap();
    assert_close(&exp2_output.into_vec(), &[1.0, 1.0, 1.0, 2.0]);
    let expm1_values = expm1_output.into_vec();
    assert_close(&expm1_values, &[0.0, 1.0e-8, -1.0e-8, 1.7182817]);
    assert!(
        expm1_values[1] > 5.0e-9 && expm1_values[1] < 1.5e-8,
        "expected expm1(1e-8) to preserve a small positive signal, got {}",
        expm1_values[1]
    );
    assert!(
        expm1_values[2] < -5.0e-9 && expm1_values[2] > -1.5e-8,
        "expected expm1(-1e-8) to preserve a small negative signal, got {}",
        expm1_values[2]
    );

    let log_output = log4(Tensor1::from_array([1.0, core::f32::consts::E, 4.0, 8.0])).unwrap();
    assert_close(&log_output.into_vec(), &[0.0, 1.0, 1.3862944, 2.0794415]);

    let (log2_output, log10_output, log1p_output) = log_variants4(
        Tensor1::from_array([1.0, 2.0, 4.0, 8.0]),
        Tensor1::from_array([1.0, 10.0, 100.0, 1000.0]),
        Tensor1::from_array([0.0, 1.0, 3.0, 9.0]),
    )
    .unwrap();
    assert_close(&log2_output.into_vec(), &[0.0, 1.0, 2.0, 3.0]);
    assert_close(&log10_output.into_vec(), &[0.0, 1.0, 2.0, 3.0]);
    assert_close(
        &log1p_output.into_vec(),
        &[
            0.0,
            core::f32::consts::LN_2,
            core::f32::consts::LN_2 * 2.0,
            core::f32::consts::LN_10,
        ],
    );

    let sqrt_output = sqrt4(Tensor1::from_array([1.0, 4.0, 9.0, 16.0])).unwrap();
    assert_close(&sqrt_output.into_vec(), &[1.0, 2.0, 3.0, 4.0]);

    let (floor_output, ceil_output, round_output, rint_output) =
        rounding4(Tensor1::from_array([-1.7, -0.5, 1.5, 2.3])).unwrap();
    assert_close(&floor_output.into_vec(), &[-2.0, -1.0, 1.0, 2.0]);
    assert_close(&ceil_output.into_vec(), &[-1.0, 0.0, 2.0, 3.0]);
    assert_close(&round_output.into_vec(), &[-2.0, 0.0, 2.0, 2.0]);
    assert_close(&rint_output.into_vec(), &[-2.0, 0.0, 2.0, 2.0]);

    let (sin_output, cos_output, tan_output) = trig4(Tensor1::from_array([
        0.0,
        core::f32::consts::FRAC_PI_6,
        core::f32::consts::FRAC_PI_4,
        -core::f32::consts::FRAC_PI_4,
    ]))
    .unwrap();
    assert_close(&sin_output.into_vec(), &[0.0, 0.5, 0.70710677, -0.70710677]);
    assert_close(
        &cos_output.into_vec(),
        &[1.0, 0.8660254, 0.70710677, 0.70710677],
    );
    assert_close(&tan_output.into_vec(), &[0.0, 0.57735026, 1.0, -1.0]);

    let tanh_output = tanh4(Tensor1::from_array([0.0, 1.0, -1.0, 2.0])).unwrap();
    assert_close(
        &tanh_output.into_vec(),
        &[0.0, 0.7615942, -0.7615942, 0.9640276],
    );

    let sigmoid_output = sigmoid4(Tensor1::from_array([0.0, 2.0, -2.0, 4.0])).unwrap();
    assert_close(
        &sigmoid_output.into_vec(),
        &[0.5, 0.880797, 0.11920292, 0.98201376],
    );
}

#[test]
fn reduction_and_classifier_op_graphs_run() {
    let softmax_output = softmax4(Tensor1::from_array([1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_close(
        &softmax_output.into_vec(),
        &[0.032058604, 0.08714432, 0.23688284, 0.6439143],
    );

    let large_softmax_output =
        softmax4(Tensor1::from_array([1000.0, 1001.0, 1002.0, 1003.0])).unwrap();
    assert_close(
        &large_softmax_output.into_vec(),
        &[0.032058604, 0.08714432, 0.23688284, 0.6439143],
    );

    let mean_output = mean4(Tensor1::from_array([1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_close(&mean_output.into_vec(), &[2.5]);

    let argmax_output = argmax4(Tensor1::from_array([1.0, 10.0, 3.0, 4.0])).unwrap();
    assert_eq!(argmax_output.into_vec(), vec![1i64]);

    let integer_argmax_output = argmax4_i32(Tensor1::from_array([1i32, 10, 3, 4])).unwrap();
    assert_eq!(integer_argmax_output.into_vec(), vec![1i64]);

    let argmax_full_output =
        argmax2x3(Tensor2::from_array([[1.0, 9.0, 3.0], [7.0, 2.0, 8.0]])).unwrap();
    assert_eq!(argmax_full_output.into_vec(), vec![1i64]);

    let argmax_axis_output =
        argmax2x3_axis1(Tensor2::from_array([[1.0, 9.0, 3.0], [7.0, 2.0, 8.0]])).unwrap();
    assert_eq!(argmax_axis_output.into_vec(), vec![1i64, 2i64]);

    let argmax4d_output = argmax4d_axis3(Tensor4::from_array([[
        [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]],
        [[9.0, 8.0, 7.0], [0.0, 11.0, 10.0]],
    ]]))
    .unwrap();
    assert_eq!(argmax4d_output.into_vec(), vec![1i64, 2, 0, 1]);
}

#[test]
fn batched_matmul_and_conv2d_graphs_run() {
    let lhs = Tensor3::from_array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
    let rhs = Tensor3::from_array([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]);
    let mm = batch_mm(lhs, rhs).unwrap();
    assert_eq!(mm.into_vec(), vec![58.0, 64.0, 139.0, 154.0]);

    let image = Tensor4::from_array([[[[1.0], [2.0]], [[3.0], [4.0]]]]);
    let kernel = Tensor4::from_array([[[[2.0]]]]);
    let conv = conv2d_1x1(image, kernel).unwrap();
    assert_eq!(conv.into_vec(), vec![2.0, 4.0, 6.0, 8.0]);

    let image = Tensor4::from_array([[
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
        [[7.0], [8.0], [9.0]],
    ]]);
    let kernel = Tensor4::filled(1.0);
    let padded = conv2d_pad_stride(image.clone(), kernel.clone()).unwrap();
    assert_eq!(padded.into_vec(), vec![1.0, 5.0, 11.0, 28.0]);

    let dilated = conv2d_dilated(image, kernel).unwrap();
    assert_eq!(dilated.into_vec(), vec![20.0]);

    let image = Tensor4::from_array([[[[1.0, 2.0, 10.0, 20.0]]]]);
    let kernel = Tensor4::from_array([[[[1.0, 0.0, 3.0, 0.0], [0.0, 1.0, 0.0, 4.0]]]]);
    let grouped = conv2d_groups2(image, kernel).unwrap();
    assert_eq!(grouped.into_vec(), vec![1.0, 2.0, 30.0, 80.0]);
}

#[test]
fn graph_calls_are_inlined() {
    let x = Tensor1::from_array([-1.0, 2.0, -3.0, 4.0]);
    let y = Tensor1::from_array([0.5, 1.0, 10.0, -10.0]);
    let output = composed4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![0.0, 3.0, 7.0, 0.0]);
}

#[test]
fn nested_graph_calls_are_inlined() {
    let x = Tensor1::from_array([-1.0, 2.0, -3.0, 4.0]);
    let y = Tensor1::from_array([0.5, 1.0, 10.0, -10.0]);
    let output = composed_twice4(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![0.0, 1.0, 5.0, 0.0]);
}

#[test]
fn bool_predicate_and_where_graphs_run() {
    let selected = select_positive4(
        Tensor1::from_array([1.0, -2.0, 3.0, -4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    assert_eq!(selected.into_vec(), vec![1.0, 20.0, 3.0, 40.0]);

    let selected = select_positive_literals4(Tensor1::from_array([1.0, -2.0, 3.0, -4.0])).unwrap();
    assert_eq!(selected.into_vec(), vec![1.0, 0.0, 1.0, 0.0]);

    let selected = select_with_bool_input4(
        Tensor1::from_array([true, false, false, true]),
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    assert_eq!(selected.into_vec(), vec![1.0, 20.0, 30.0, 4.0]);

    let compared = compare_greater4(
        Tensor1::from_array([1.0, 3.0, 3.0, 5.0]),
        Tensor1::from_array([0.0, 4.0, 3.0, 2.0]),
    )
    .unwrap();
    assert_eq!(compared.into_vec(), vec![true, false, false, true]);

    let compared = compare_greater_equal4(
        Tensor1::from_array([1.0, 3.0, 3.0, 5.0]),
        Tensor1::from_array([0.0, 4.0, 3.0, 2.0]),
    )
    .unwrap();
    assert_eq!(compared.into_vec(), vec![true, false, true, true]);

    let compared = compare_less4(
        Tensor1::from_array([1.0, 3.0, 3.0, 5.0]),
        Tensor1::from_array([0.0, 4.0, 3.0, 2.0]),
    )
    .unwrap();
    assert_eq!(compared.into_vec(), vec![false, true, false, false]);

    let compared = compare_less_equal4(
        Tensor1::from_array([1.0, 3.0, 3.0, 5.0]),
        Tensor1::from_array([0.0, 4.0, 3.0, 2.0]),
    )
    .unwrap();
    assert_eq!(compared.into_vec(), vec![false, true, true, false]);

    let compared = compare_equal4(
        Tensor1::from_array([1.0, 3.0, 3.0, 5.0]),
        Tensor1::from_array([0.0, 4.0, 3.0, 5.0]),
    )
    .unwrap();
    assert_eq!(compared.into_vec(), vec![false, false, true, true]);

    let compared = compare_not_equal4(
        Tensor1::from_array([1.0, 3.0, 3.0, 5.0]),
        Tensor1::from_array([0.0, 4.0, 3.0, 5.0]),
    )
    .unwrap();
    assert_eq!(compared.into_vec(), vec![true, true, false, false]);

    let compared = bool_equal4(
        Tensor1::from_array([true, false, true, false]),
        Tensor1::from_array([true, true, false, false]),
    )
    .unwrap();
    assert_eq!(compared.into_vec(), vec![true, false, false, true]);

    let compared = bool_not_equal4(
        Tensor1::from_array([true, false, true, false]),
        Tensor1::from_array([true, true, false, false]),
    )
    .unwrap();
    assert_eq!(compared.into_vec(), vec![false, true, true, false]);

    let logical = logical_from_comparisons4(
        Tensor1::from_array([1.0, -2.0, 3.0, -4.0]),
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();
    assert_eq!(logical.into_vec(), vec![false, true, true, false]);

    let logical = logical_and_input4(
        Tensor1::from_array([true, true, false, false]),
        Tensor1::from_array([true, false, true, false]),
    )
    .unwrap();
    assert_eq!(logical.into_vec(), vec![true, false, false, false]);

    let logical = logical_or_input4(
        Tensor1::from_array([true, true, false, false]),
        Tensor1::from_array([true, false, true, false]),
    )
    .unwrap();
    assert_eq!(logical.into_vec(), vec![true, true, true, false]);

    let logical = logical_xor_input4(
        Tensor1::from_array([true, true, false, false]),
        Tensor1::from_array([true, false, true, false]),
    )
    .unwrap();
    assert_eq!(logical.into_vec(), vec![false, true, true, false]);

    let logical = logical_not_input4(Tensor1::from_array([true, false, true, false])).unwrap();
    assert_eq!(logical.into_vec(), vec![false, true, false, true]);

    assert_eq!(
        all_positive4(Tensor1::from_array([1.0, 2.0, 3.0, 4.0]))
            .unwrap()
            .into_vec(),
        vec![true]
    );
    assert_eq!(
        all_positive4(Tensor1::from_array([1.0, 2.0, -3.0, 4.0]))
            .unwrap()
            .into_vec(),
        vec![false]
    );
    assert_eq!(
        any_nan4(Tensor1::from_array([1.0, f32::NAN, 3.0, 4.0]))
            .unwrap()
            .into_vec(),
        vec![true]
    );
}

fn reverse_axes_flat(values: &[f32], shape: &[usize]) -> Vec<f32> {
    let output_shape = shape.iter().rev().copied().collect::<Vec<_>>();
    let output_len = output_shape.iter().product();
    (0..output_len)
        .map(|output_index| {
            let output_indices = unravel_index(output_index, &output_shape);
            let input_indices = output_indices.iter().rev().copied().collect::<Vec<_>>();
            values[ravel_index(&input_indices, shape)]
        })
        .collect()
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        indices[axis] = index % shape[axis];
        index /= shape[axis];
    }
    indices
}

fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
    indices
        .iter()
        .zip(shape)
        .fold(0, |index, (axis_index, dim)| index * dim + axis_index)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_close_with_tolerance(actual, expected, 1.0e-4);
}

fn assert_close_with_tolerance(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance,
            "expected {expected}, got {actual}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-30,
            "expected {expected}, got {actual}"
        );
    }
}
