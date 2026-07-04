use knok_core::{CallOp, Expr};

use super::{
    context::{tuple_id, TraceVars},
    tensor::{ScalarLiteral, TraceTensor, T0},
};

/// Floating-point scalar types supported by random uniform tracing helpers.
pub trait RandomFloatElement: ScalarLiteral {
    /// Additive identity used by [`rand`].
    fn zero() -> Self;
    /// Exclusive upper bound used by [`rand`].
    fn one() -> Self;
}

impl RandomFloatElement for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

impl RandomFloatElement for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

/// Integer scalar types supported by random integer tracing helpers.
pub trait RandomIntElement: ScalarLiteral {}

impl RandomIntElement for i32 {}
impl RandomIntElement for i64 {}

/// Records a compile-time deterministic uniform random tensor.
///
/// Values are sampled in `[low, high)` from `seed` while the graph is compiled,
/// then embedded as a constant in the generated artifact.
pub fn uniform_static<Output>(seed: u64, low: Output::Elem, high: Output::Elem) -> Output
where
    Output: TraceTensor,
    Output::Elem: RandomFloatElement,
{
    trace_static_random_uniform(seed, low, high)
}

fn trace_static_random_uniform<Output>(seed: u64, low: Output::Elem, high: Output::Elem) -> Output
where
    Output: TraceTensor,
    Output::Elem: RandomFloatElement,
{
    Output::from_expr(Expr::Call {
        op: CallOp::StaticRandomUniform {
            target: Output::tensor_type(),
            seed,
        },
        args: vec![low.const_expr(), high.const_expr()],
    })
}

/// Records a compile-time deterministic integer random tensor.
///
/// Values are sampled in `[low, high)` from `seed` while the graph is compiled,
/// then embedded as a constant in the generated artifact.
pub fn randint_static<Output>(seed: u64, low: Output::Elem, high: Output::Elem) -> Output
where
    Output: TraceTensor,
    Output::Elem: RandomIntElement,
{
    trace_static_random_int(seed, low, high)
}

fn trace_static_random_int<Output>(seed: u64, low: Output::Elem, high: Output::Elem) -> Output
where
    Output: TraceTensor,
    Output::Elem: RandomIntElement,
{
    Output::from_expr(Expr::Call {
        op: CallOp::StaticRandomInt {
            target: Output::tensor_type(),
            seed,
        },
        args: vec![low.const_expr(), high.const_expr()],
    })
}

/// Records a runtime deterministic uniform random tensor and the next key.
///
/// Values are sampled in `[low, high)` from `key`. Passing the same key produces
/// the same values; pass the returned key to advance the random stream.
pub fn uniform<Output>(key: T0<i64>, low: Output::Elem, high: Output::Elem) -> (Output, T0<i64>)
where
    Output: TraceTensor,
    Output::Elem: RandomFloatElement,
{
    trace_random_uniform(key, low, high)
}

fn trace_random_uniform<Output>(
    key: T0<i64>,
    low: Output::Elem,
    high: Output::Elem,
) -> (Output, T0<i64>)
where
    Output: TraceTensor,
    Output::Elem: RandomFloatElement,
{
    let value = Expr::Call {
        op: CallOp::RandomUniform(Output::tensor_type()),
        args: vec![key.into_expr(), low.const_expr(), high.const_expr()],
    };
    <(Output, T0<i64>) as TraceVars>::from_tuple_expr(tuple_id(), value)
}

/// Records a runtime deterministic uniform random tensor in `[0, 1)`.
pub fn rand<Output>(key: T0<i64>) -> (Output, T0<i64>)
where
    Output: TraceTensor,
    Output::Elem: RandomFloatElement,
{
    uniform(key, Output::Elem::zero(), Output::Elem::one())
}

/// Records a runtime deterministic integer random tensor and the next key.
///
/// Values are sampled in `[low, high)` from `key`. Passing the same key produces
/// the same values; pass the returned key to advance the random stream.
pub fn randint<Output>(key: T0<i64>, low: Output::Elem, high: Output::Elem) -> (Output, T0<i64>)
where
    Output: TraceTensor,
    Output::Elem: RandomIntElement,
{
    trace_random_int(key, low, high)
}

fn trace_random_int<Output>(
    key: T0<i64>,
    low: Output::Elem,
    high: Output::Elem,
) -> (Output, T0<i64>)
where
    Output: TraceTensor,
    Output::Elem: RandomIntElement,
{
    let value = Expr::Call {
        op: CallOp::RandomInt(Output::tensor_type()),
        args: vec![key.into_expr(), low.const_expr(), high.const_expr()],
    };
    <(Output, T0<i64>) as TraceVars>::from_tuple_expr(tuple_id(), value)
}
