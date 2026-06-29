use knok_core::{CallOp, Expr};

use super::{
    expr::{binary_bool, binary_same, call_output, node_expr, unary, unary_bool},
    tensor::{BoolTensor, TraceOperand, TraceTensor},
};

pub fn abs<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Abs, value)
}

pub fn ceil<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Ceil, value)
}

pub fn exp<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Exp, value)
}

pub fn exp2<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Exp2, value)
}

pub fn expm1<T: TraceTensor>(value: T) -> T {
    unary(CallOp::ExpM1, value)
}

pub fn floor<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Floor, value)
}

pub fn log<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Log, value)
}

pub fn log1p<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Log1P, value)
}

pub fn log2<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Log2, value)
}

pub fn log10<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Log10, value)
}

pub fn relu<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Relu, value)
}

pub fn rint<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Rint, value)
}

pub fn round<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Round, value)
}

pub fn sigmoid<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Sigmoid, value)
}

pub fn sin<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Sin, value)
}

pub fn cos<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Cos, value)
}

pub fn sqrt<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Sqrt, value)
}

pub fn tan<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Tan, value)
}

pub fn tanh<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Tanh, value)
}

pub fn square<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Square, value)
}

pub fn reciprocal<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Reciprocal, value)
}

pub fn isnan<T: BoolTensor>(value: T) -> T::Bool {
    unary_bool(CallOp::IsNan, value)
}

pub fn zeros_like<T: TraceTensor>(value: T) -> T {
    unary(CallOp::ZerosLike, value)
}

pub fn ones_like<T: TraceTensor>(value: T) -> T {
    unary(CallOp::OnesLike, value)
}

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

pub fn minimum<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor,
    R: TraceOperand<L::Elem>,
{
    binary_same(CallOp::Minimum, lhs, rhs)
}

pub fn maximum<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor,
    R: TraceOperand<L::Elem>,
{
    binary_same(CallOp::Maximum, lhs, rhs)
}

pub fn pow<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor,
    R: TraceOperand<L::Elem>,
{
    binary_same(CallOp::Pow, lhs, rhs)
}

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

pub fn greater<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::Greater, lhs, rhs)
}

pub fn greater_equal<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::GreaterEqual, lhs, rhs)
}

pub fn less<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::Less, lhs, rhs)
}

pub fn less_equal<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::LessEqual, lhs, rhs)
}

pub fn equal<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::Equal, lhs, rhs)
}

pub fn not_equal<L, R>(lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    binary_bool(CallOp::NotEqual, lhs, rhs)
}

pub fn logical_and<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor<Elem = bool>,
    R: TraceOperand<bool>,
{
    binary_same(CallOp::LogicalAnd, lhs, rhs)
}

pub fn logical_or<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor<Elem = bool>,
    R: TraceOperand<bool>,
{
    binary_same(CallOp::LogicalOr, lhs, rhs)
}

pub fn logical_xor<L, R>(lhs: L, rhs: R) -> L
where
    L: TraceTensor<Elem = bool>,
    R: TraceOperand<bool>,
{
    binary_same(CallOp::LogicalXor, lhs, rhs)
}

pub fn logical_not<T: TraceTensor<Elem = bool>>(value: T) -> T {
    unary(CallOp::LogicalNot, value)
}

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
