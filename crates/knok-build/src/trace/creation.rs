use knok_core::CallOp;

use super::{
    expr::call_output,
    tensor::{TraceOperand, TraceTensor},
};

pub fn arange_to<Output>(stop: impl TraceOperand<Output::Elem>) -> Output
where
    Output: TraceTensor,
{
    trace_arange1(stop)
}

fn trace_arange1<Output>(stop: impl TraceOperand<Output::Elem>) -> Output
where
    Output: TraceTensor,
{
    call_output(
        CallOp::Arange(Output::tensor_type()),
        vec![stop.into_operand_expr()],
    )
}

pub fn arange<Output>(
    start: impl TraceOperand<Output::Elem>,
    stop: impl TraceOperand<Output::Elem>,
) -> Output
where
    Output: TraceTensor,
{
    trace_arange2(start, stop)
}

fn trace_arange2<Output>(
    start: impl TraceOperand<Output::Elem>,
    stop: impl TraceOperand<Output::Elem>,
) -> Output
where
    Output: TraceTensor,
{
    call_output(
        CallOp::Arange(Output::tensor_type()),
        vec![start.into_operand_expr(), stop.into_operand_expr()],
    )
}

pub fn arange_step<Output>(
    start: impl TraceOperand<Output::Elem>,
    stop: impl TraceOperand<Output::Elem>,
    step: impl TraceOperand<Output::Elem>,
) -> Output
where
    Output: TraceTensor,
{
    trace_arange3(start, stop, step)
}

fn trace_arange3<Output>(
    start: impl TraceOperand<Output::Elem>,
    stop: impl TraceOperand<Output::Elem>,
    step: impl TraceOperand<Output::Elem>,
) -> Output
where
    Output: TraceTensor,
{
    call_output(
        CallOp::Arange(Output::tensor_type()),
        vec![
            start.into_operand_expr(),
            stop.into_operand_expr(),
            step.into_operand_expr(),
        ],
    )
}

pub fn linspace<Output>(
    start: impl TraceOperand<Output::Elem>,
    stop: impl TraceOperand<Output::Elem>,
) -> Output
where
    Output: TraceTensor,
{
    call_output(
        CallOp::Linspace(Output::tensor_type()),
        vec![start.into_operand_expr(), stop.into_operand_expr()],
    )
}

pub fn eye<Output: TraceTensor>() -> Output {
    call_output(CallOp::Eye(Output::tensor_type()), Vec::new())
}

pub fn identity<Output: TraceTensor>() -> Output {
    eye()
}
