use knok_core::{CallOp, Expr};

use super::{
    expr::{binary_bool, binary_same, call_output, node_expr, unary, unary_bool},
    tensor::{BoolTensor, TraceOperand, TraceTensor},
};

/// Records an elementwise absolute-value operation.
pub fn abs<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Abs, value)
}

/// Records an elementwise ceiling operation.
pub fn ceil<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Ceil, value)
}

/// Records a stop-gradient marker for generated gradient graphs.
pub fn detach<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Detach, value)
}

/// Records an elementwise natural exponential operation.
pub fn exp<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Exp, value)
}

/// Records an elementwise base-2 exponential operation.
pub fn exp2<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Exp2, value)
}

/// Records an elementwise exponential-minus-one operation.
pub fn expm1<T: TraceTensor>(value: T) -> T {
    unary(CallOp::ExpM1, value)
}

/// Records an elementwise floor operation.
pub fn floor<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Floor, value)
}

/// Records an elementwise natural logarithm operation.
pub fn log<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Log, value)
}

/// Records an elementwise natural logarithm of one plus the input.
pub fn log1p<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Log1P, value)
}

/// Records an elementwise base-2 logarithm operation.
pub fn log2<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Log2, value)
}

/// Records an elementwise base-10 logarithm operation.
pub fn log10<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Log10, value)
}

/// Records an elementwise rectified linear unit operation.
pub fn relu<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Relu, value)
}

/// Records an elementwise round-to-integer operation.
pub fn rint<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Rint, value)
}

/// Records an elementwise rounding operation.
pub fn round<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Round, value)
}

/// Records an elementwise sigmoid operation.
pub fn sigmoid<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Sigmoid, value)
}

/// Records an elementwise sine operation.
pub fn sin<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Sin, value)
}

/// Records an elementwise cosine operation.
pub fn cos<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Cos, value)
}

/// Records an elementwise square-root operation.
pub fn sqrt<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Sqrt, value)
}

/// Records an elementwise tangent operation.
pub fn tan<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Tan, value)
}

/// Records an elementwise hyperbolic tangent operation.
pub fn tanh<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Tanh, value)
}

/// Records an elementwise square operation.
pub fn square<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Square, value)
}

/// Records an elementwise reciprocal operation.
pub fn reciprocal<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Reciprocal, value)
}

/// Records an elementwise NaN predicate.
pub fn isnan<T: BoolTensor>(value: T) -> T::Bool {
    unary_bool(CallOp::IsNan, value)
}

/// Records a zero-filled tensor with the same shape as `value`.
pub fn zeros_like<T: TraceTensor>(value: T) -> T {
    unary(CallOp::ZerosLike, value)
}

/// Records a one-filled tensor with the same shape as `value`.
pub fn ones_like<T: TraceTensor>(value: T) -> T {
    unary(CallOp::OnesLike, value)
}

/// Records a tensor filled with `fill` and the same shape as `value`.
pub fn full_like<T, Fill>(value: T, fill: Fill) -> T
where
    T: TraceTensor,
    Fill: TraceOperand<T::Elem>,
{
    T::from_expr(node_expr(Expr::Call {
        op: CallOp::FullLike,
        args: vec![value.into_expr(), fill.into_operand_expr()],
    }))
}

/// Records an elementwise minimum operation.
pub fn minimum<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor,
    R: TraceOperand<L::Elem>,
{
    binary_same(CallOp::Minimum, lhs, rhs)
}

/// Records an elementwise maximum operation.
pub fn maximum<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor,
    R: TraceOperand<L::Elem>,
{
    binary_same(CallOp::Maximum, lhs, rhs)
}

/// Records an elementwise power operation.
pub fn pow<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor,
    R: TraceOperand<L::Elem>,
{
    binary_same(CallOp::Pow, lhs, rhs)
}

/// Records an elementwise clamp between `min` and `max`.
pub fn clip<T, Min, Max>(value: T, min: Min, max: Max) -> T
where
    T: TraceTensor,
    Min: TraceOperand<T::Elem>,
    Max: TraceOperand<T::Elem>,
{
    T::from_expr(node_expr(Expr::Call {
        op: CallOp::Clip,
        args: vec![
            value.into_expr(),
            min.into_operand_expr(),
            max.into_operand_expr(),
        ],
    }))
}

/// Records an elementwise greater-than comparison.
pub fn greater<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::Greater, lhs, rhs)
}

/// Records an elementwise greater-than-or-equal comparison.
pub fn greater_equal<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::GreaterEqual, lhs, rhs)
}

/// Records an elementwise less-than comparison.
pub fn less<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::Less, lhs, rhs)
}

/// Records an elementwise less-than-or-equal comparison.
pub fn less_equal<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::LessEqual, lhs, rhs)
}

/// Records an elementwise equality comparison.
pub fn equal<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::Equal, lhs, rhs)
}

/// Records an elementwise inequality comparison.
pub fn not_equal<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::NotEqual, lhs, rhs)
}

/// Records an elementwise boolean and operation.
pub fn logical_and<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor<Elem = bool>,
    R: TraceOperand<bool>,
{
    binary_same(CallOp::LogicalAnd, lhs, rhs)
}

/// Records an elementwise boolean or operation.
pub fn logical_or<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor<Elem = bool>,
    R: TraceOperand<bool>,
{
    binary_same(CallOp::LogicalOr, lhs, rhs)
}

/// Records an elementwise boolean xor operation.
pub fn logical_xor<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor<Elem = bool>,
    R: TraceOperand<bool>,
{
    binary_same(CallOp::LogicalXor, lhs, rhs)
}

/// Records an elementwise boolean not operation.
pub fn logical_not<T: TraceTensor<Elem = bool>>(value: T) -> T {
    unary(CallOp::LogicalNot, value)
}

/// Records an elementwise select operation.
pub fn r#where<Output, C, X, Y>(condition: C, x: X, y: Y) -> Output
where
    Output: TraceTensor,
    C: TraceTensor<Elem = bool>,
    X: TraceOperand<Output::Elem>,
    Y: TraceOperand<Output::Elem>,
{
    call_output(
        CallOp::Where,
        vec![
            condition.into_expr(),
            x.into_operand_expr(),
            y.into_operand_expr(),
        ],
    )
}
