use knok_core::{AxisSpec, CallOp};

use super::{
    expr::{call_output, unary},
    tensor::TraceTensor,
};

pub fn sum<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_sum_all(value)
}

pub fn prod<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_prod_all(value)
}

pub fn mean<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_mean_all(value)
}

pub fn max<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_max_all(value)
}

pub fn amax<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_max_all(value)
}

pub fn min<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_min_all(value)
}

pub fn amin<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_min_all(value)
}

pub fn argmax<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_argmax_all(value)
}

pub fn argmin<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_argmin_all(value)
}

pub fn var<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_var_all(value)
}

pub fn std<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_std_all(value)
}

pub fn ptp<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_ptp_all(value)
}

pub fn softmax<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Softmax(AxisSpec::All), value)
}

pub fn all<Output: TraceTensor<Elem = bool>>(value: impl TraceTensor<Elem = bool>) -> Output {
    trace_all_all(value)
}

pub fn any<Output: TraceTensor<Elem = bool>>(value: impl TraceTensor<Elem = bool>) -> Output {
    trace_any_all(value)
}

pub fn all_axis<Output: TraceTensor<Elem = bool>>(
    value: impl TraceTensor<Elem = bool>,
    axis: usize,
) -> Output {
    trace_all_axis(value, axis)
}

pub fn any_axis<Output: TraceTensor<Elem = bool>>(
    value: impl TraceTensor<Elem = bool>,
    axis: usize,
) -> Output {
    trace_any_axis(value, axis)
}

pub fn argmax_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_argmax_axis(value, axis)
}

pub fn argmin_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_argmin_axis(value, axis)
}

pub fn max_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_max_axis(value, axis)
}

pub fn amax_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_max_axis(value, axis)
}

pub fn mean_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_mean_axis(value, axis)
}

pub fn min_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_min_axis(value, axis)
}

pub fn amin_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_min_axis(value, axis)
}

pub fn prod_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_prod_axis(value, axis)
}

pub fn ptp_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_ptp_axis(value, axis)
}

pub fn std_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_std_axis(value, axis)
}

pub fn sum_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_sum_axis(value, axis)
}

pub fn var_axis<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
    trace_var_axis(value, axis)
}

macro_rules! reduction_all {
    ($name:ident, $op:ident) => {
        fn $name<Output: TraceTensor>(value: impl TraceTensor) -> Output {
            call_output(CallOp::$op(AxisSpec::All), vec![value.into_expr()])
        }
    };
}

macro_rules! reduction_axis {
    ($name:ident, $op:ident) => {
        fn $name<Output: TraceTensor>(value: impl TraceTensor, axis: usize) -> Output {
            call_output(CallOp::$op(AxisSpec::One(axis)), vec![value.into_expr()])
        }
    };
}

reduction_all!(trace_all_all, All);
reduction_axis!(trace_all_axis, All);
reduction_all!(trace_any_all, Any);
reduction_axis!(trace_any_axis, Any);
reduction_all!(trace_argmax_all, Argmax);
reduction_axis!(trace_argmax_axis, Argmax);
reduction_all!(trace_argmin_all, Argmin);
reduction_axis!(trace_argmin_axis, Argmin);
reduction_all!(trace_max_all, Max);
reduction_axis!(trace_max_axis, Max);
reduction_all!(trace_mean_all, Mean);
reduction_axis!(trace_mean_axis, Mean);
reduction_all!(trace_min_all, Min);
reduction_axis!(trace_min_axis, Min);
reduction_all!(trace_prod_all, Prod);
reduction_axis!(trace_prod_axis, Prod);
reduction_all!(trace_ptp_all, Ptp);
reduction_axis!(trace_ptp_axis, Ptp);
reduction_all!(trace_std_all, Std);
reduction_axis!(trace_std_axis, Std);
reduction_all!(trace_sum_all, Sum);
reduction_axis!(trace_sum_axis, Sum);
reduction_all!(trace_var_all, Var);
reduction_axis!(trace_var_axis, Var);

fn trace_softmax_axis<T: TraceTensor>(value: T, axis: usize) -> T {
    unary(CallOp::Softmax(AxisSpec::One(axis)), value)
}

pub fn softmax_axis<T: TraceTensor>(value: T, axis: usize) -> T {
    trace_softmax_axis(value, axis)
}
