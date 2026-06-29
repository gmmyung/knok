use knok_core::{CallOp, Expr};

use super::{
    expr::{call_output, node_expr},
    tensor::{TraceElement, TraceTensor, T0, T1, T2, T3, T4, T5, T6},
};

pub trait Matmul<Rhs>: TraceTensor {
    type Output: TraceTensor;

    fn matmul(self, rhs: Rhs) -> Self::Output;
}

impl<T: TraceElement, const K: usize> Matmul<T1<T, K>> for T1<T, K> {
    type Output = T0<T>;

    fn matmul(self, rhs: T1<T, K>) -> Self::Output {
        T0::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<T: TraceElement, const M: usize, const K: usize> Matmul<T1<T, K>> for T2<T, M, K> {
    type Output = T1<T, M>;

    fn matmul(self, rhs: T1<T, K>) -> Self::Output {
        T1::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<T: TraceElement, const K: usize, const N: usize> Matmul<T2<T, K, N>> for T1<T, K> {
    type Output = T1<T, N>;

    fn matmul(self, rhs: T2<T, K, N>) -> Self::Output {
        T1::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<T: TraceElement, const M: usize, const K: usize, const N: usize> Matmul<T2<T, K, N>>
    for T2<T, M, K>
{
    type Output = T2<T, M, N>;

    fn matmul(self, rhs: T2<T, K, N>) -> Self::Output {
        T2::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<T: TraceElement, const B0: usize, const M: usize, const K: usize, const N: usize>
    Matmul<T3<T, B0, K, N>> for T3<T, B0, M, K>
{
    type Output = T3<T, B0, M, N>;

    fn matmul(self, rhs: T3<T, B0, K, N>) -> Self::Output {
        T3::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<T: TraceElement, const B0: usize, const M: usize, const K: usize, const N: usize>
    Matmul<T2<T, K, N>> for T3<T, B0, M, K>
{
    type Output = T3<T, B0, M, N>;

    fn matmul(self, rhs: T2<T, K, N>) -> Self::Output {
        T3::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<T: TraceElement, const B0: usize, const M: usize, const K: usize, const N: usize>
    Matmul<T3<T, B0, K, N>> for T2<T, M, K>
{
    type Output = T3<T, B0, M, N>;

    fn matmul(self, rhs: T3<T, B0, K, N>) -> Self::Output {
        T3::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<
        T: TraceElement,
        const B0: usize,
        const B1: usize,
        const M: usize,
        const K: usize,
        const N: usize,
    > Matmul<T4<T, B0, B1, K, N>> for T4<T, B0, B1, M, K>
{
    type Output = T4<T, B0, B1, M, N>;

    fn matmul(self, rhs: T4<T, B0, B1, K, N>) -> Self::Output {
        T4::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<
        T: TraceElement,
        const B0: usize,
        const B1: usize,
        const M: usize,
        const K: usize,
        const N: usize,
    > Matmul<T3<T, B1, K, N>> for T4<T, B0, 1, M, K>
{
    type Output = T4<T, B0, B1, M, N>;

    fn matmul(self, rhs: T3<T, B1, K, N>) -> Self::Output {
        T4::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<
        T: TraceElement,
        const B0: usize,
        const B1: usize,
        const B2: usize,
        const M: usize,
        const K: usize,
        const N: usize,
    > Matmul<T5<T, B0, B1, B2, K, N>> for T5<T, B0, B1, B2, M, K>
{
    type Output = T5<T, B0, B1, B2, M, N>;

    fn matmul(self, rhs: T5<T, B0, B1, B2, K, N>) -> Self::Output {
        T5::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

impl<
        T: TraceElement,
        const B0: usize,
        const B1: usize,
        const B2: usize,
        const B3: usize,
        const M: usize,
        const K: usize,
        const N: usize,
    > Matmul<T6<T, B0, B1, B2, B3, K, N>> for T6<T, B0, B1, B2, B3, M, K>
{
    type Output = T6<T, B0, B1, B2, B3, M, N>;

    fn matmul(self, rhs: T6<T, B0, B1, B2, B3, K, N>) -> Self::Output {
        T6::from_expr(node_expr(Expr::Call {
            op: CallOp::Matmul,
            args: vec![self.into_expr(), rhs.into_expr()],
        }))
    }
}

pub fn matmul<L, R>(lhs: L, rhs: R) -> L::Output
where
    L: Matmul<R>,
{
    lhs.matmul(rhs)
}

pub fn dot<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    call_output(CallOp::Dot, vec![lhs.into_expr(), rhs.into_expr()])
}

pub fn inner<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    call_output(CallOp::Inner, vec![lhs.into_expr(), rhs.into_expr()])
}

pub fn outer<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    call_output(CallOp::Outer, vec![lhs.into_expr(), rhs.into_expr()])
}

pub fn vecdot<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    trace_vecdot(lhs, rhs)
}

fn trace_vecdot<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    call_output(CallOp::Vecdot(None), vec![lhs.into_expr(), rhs.into_expr()])
}

pub fn vecdot_axis<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_vecdot_axis(lhs, rhs, axis)
}

fn trace_vecdot_axis<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(
        CallOp::Vecdot(Some(axis)),
        vec![lhs.into_expr(), rhs.into_expr()],
    )
}

pub fn trace<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_trace(value)
}

fn trace_trace<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(CallOp::Trace(None), vec![value.into_expr()])
}

pub fn trace_axes<Output: TraceTensor>(
    value: impl TraceTensor,
    axis0: usize,
    axis1: usize,
) -> Output {
    trace_trace_axes(value, axis0, axis1)
}

fn trace_trace_axes<Output: TraceTensor>(
    value: impl TraceTensor,
    axis0: usize,
    axis1: usize,
) -> Output {
    call_output(CallOp::Trace(Some([axis0, axis1])), vec![value.into_expr()])
}

pub fn diagonal<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_diagonal(value)
}

fn trace_diagonal<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(CallOp::Diagonal(None), vec![value.into_expr()])
}

pub fn diagonal_axes<Output: TraceTensor>(
    value: impl TraceTensor,
    axis0: usize,
    axis1: usize,
) -> Output {
    trace_diagonal_axes(value, axis0, axis1)
}

fn trace_diagonal_axes<Output: TraceTensor>(
    value: impl TraceTensor,
    axis0: usize,
    axis1: usize,
) -> Output {
    call_output(
        CallOp::Diagonal(Some([axis0, axis1])),
        vec![value.into_expr()],
    )
}
