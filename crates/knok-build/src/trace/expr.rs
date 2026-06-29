use std::sync::atomic::{AtomicU64, Ordering};

use knok_core::{CallOp, Expr};

use super::tensor::{BoolTensor, TraceOperand, TraceTensor};

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_TUPLE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn call_output<Output: TraceTensor>(op: CallOp, args: Vec<Expr>) -> Output {
    Output::from_expr(node_expr(Expr::Call { op, args }))
}

pub(crate) fn node_expr(value: Expr) -> Expr {
    Expr::Node {
        node_id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
        value: Box::new(value),
    }
}

pub(crate) fn next_tuple_id() -> u64 {
    NEXT_TUPLE_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn unary<T: TraceTensor>(op: CallOp, value: T) -> T {
    T::from_expr(node_expr(Expr::Call {
        op,
        args: vec![value.into_expr()],
    }))
}

pub(crate) fn unary_bool<T: BoolTensor>(op: CallOp, value: T) -> T::Bool {
    <T::Bool as TraceTensor>::from_expr(node_expr(Expr::Call {
        op,
        args: vec![value.into_expr()],
    }))
}

pub(crate) fn binary_same<L, R>(op: CallOp, lhs: L, rhs: R) -> L
where
    L: TraceTensor,
    R: TraceOperand<L::Elem>,
{
    L::from_expr(node_expr(Expr::Call {
        op,
        args: vec![lhs.into_expr(), rhs.into_operand_expr()],
    }))
}

pub(crate) fn binary_bool<L, R>(op: CallOp, lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    <L::Bool as TraceTensor>::from_expr(node_expr(Expr::Call {
        op,
        args: vec![lhs.into_expr(), rhs.into_operand_expr()],
    }))
}
