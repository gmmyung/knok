use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::atomic::{AtomicU64, Ordering},
};

use knok_core::{
    type_check, AxisSpec, BinaryOp, CallOp, Conv2dOptions as CoreConv2dOptions, ElementType, Expr,
    Graph, Input, Padding2d, TensorType, TypedExpr, TypedGraph, UnaryOp,
};

use crate::Result;

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_TUPLE_ID: AtomicU64 = AtomicU64::new(1);

pub trait TraceElement: Copy + 'static {
    const ELEMENT: ElementType;
}

impl TraceElement for bool {
    const ELEMENT: ElementType = ElementType::Bool;
}

impl TraceElement for f32 {
    const ELEMENT: ElementType = ElementType::F32;
}

impl TraceElement for f64 {
    const ELEMENT: ElementType = ElementType::F64;
}

impl TraceElement for half::f16 {
    const ELEMENT: ElementType = ElementType::F16;
}

impl TraceElement for half::bf16 {
    const ELEMENT: ElementType = ElementType::BF16;
}

impl TraceElement for i32 {
    const ELEMENT: ElementType = ElementType::I32;
}

impl TraceElement for i64 {
    const ELEMENT: ElementType = ElementType::I64;
}

pub trait ScalarLiteral: TraceElement {
    fn const_expr(self) -> Expr;
}

impl ScalarLiteral for bool {
    fn const_expr(self) -> Expr {
        Expr::Const {
            value: if self { "1" } else { "0" }.into(),
            elem: Self::ELEMENT,
        }
    }
}

impl ScalarLiteral for f32 {
    fn const_expr(self) -> Expr {
        Expr::Const {
            value: format!("{self:?}"),
            elem: Self::ELEMENT,
        }
    }
}

impl ScalarLiteral for f64 {
    fn const_expr(self) -> Expr {
        Expr::Const {
            value: format!("{self:?}"),
            elem: Self::ELEMENT,
        }
    }
}

impl ScalarLiteral for i32 {
    fn const_expr(self) -> Expr {
        Expr::Const {
            value: self.to_string(),
            elem: Self::ELEMENT,
        }
    }
}

impl ScalarLiteral for i64 {
    fn const_expr(self) -> Expr {
        Expr::Const {
            value: self.to_string(),
            elem: Self::ELEMENT,
        }
    }
}

pub trait TraceTensor: Clone {
    type Elem: TraceElement;

    fn tensor_type() -> TensorType;
    fn from_expr(expr: Expr) -> Self;
    fn into_expr(self) -> Expr;
    fn expr(&self) -> &Expr;
}

pub trait BoolTensor: TraceTensor {
    type Bool: TraceTensor<Elem = bool>;
}

pub trait TraceOperand<E: TraceElement> {
    fn into_operand_expr(self) -> Expr;
}

impl<E: TraceElement, T: TraceTensor> TraceOperand<E> for T {
    fn into_operand_expr(self) -> Expr {
        self.into_expr()
    }
}

macro_rules! impl_scalar_operand {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TraceOperand<$ty> for $ty {
                fn into_operand_expr(self) -> Expr {
                    self.const_expr()
                }
            }
        )+
    };
}

impl_scalar_operand!(bool, f32, f64, i32, i64);

macro_rules! define_tensor {
    ($name:ident, []) => {
        #[derive(Clone, Debug)]
        pub struct $name<T: TraceElement> {
            expr: Expr,
            _marker: PhantomData<T>,
        }

        impl<T: TraceElement> TraceTensor for $name<T> {
            type Elem = T;

            fn tensor_type() -> TensorType {
                TensorType { elem: T::ELEMENT, shape: Vec::new() }
            }

            fn from_expr(expr: Expr) -> Self {
                Self { expr, _marker: PhantomData }
            }

            fn into_expr(self) -> Expr {
                self.expr
            }

            fn expr(&self) -> &Expr {
                &self.expr
            }
        }

        impl<T: TraceElement> BoolTensor for $name<T> {
            type Bool = $name<bool>;
        }
    };
    ($name:ident, [$($dim:ident),+]) => {
        #[derive(Clone, Debug)]
        pub struct $name<T: TraceElement, $(const $dim: usize),+> {
            expr: Expr,
            _marker: PhantomData<T>,
        }

        impl<T: TraceElement, $(const $dim: usize),+> TraceTensor for $name<T, $($dim),+> {
            type Elem = T;

            fn tensor_type() -> TensorType {
                TensorType { elem: T::ELEMENT, shape: vec![$($dim),+] }
            }

            fn from_expr(expr: Expr) -> Self {
                Self { expr, _marker: PhantomData }
            }

            fn into_expr(self) -> Expr {
                self.expr
            }

            fn expr(&self) -> &Expr {
                &self.expr
            }
        }

        impl<T: TraceElement, $(const $dim: usize),+> BoolTensor for $name<T, $($dim),+> {
            type Bool = $name<bool, $($dim),+>;
        }
    };
}

define_tensor!(T0, []);
define_tensor!(T1, [D0]);
define_tensor!(T2, [D0, D1]);
define_tensor!(T3, [D0, D1, D2]);
define_tensor!(T4, [D0, D1, D2, D3]);
define_tensor!(T5, [D0, D1, D2, D3, D4]);
define_tensor!(T6, [D0, D1, D2, D3, D4, D5]);

pub type Tensor0<T> = T0<T>;
pub type Tensor1<T, const D0: usize> = T1<T, D0>;
pub type Tensor2<T, const D0: usize, const D1: usize> = T2<T, D0, D1>;
pub type Tensor3<T, const D0: usize, const D1: usize, const D2: usize> = T3<T, D0, D1, D2>;
pub type Tensor4<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize> =
    T4<T, D0, D1, D2, D3>;
pub type Tensor5<
    T,
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
> = T5<T, D0, D1, D2, D3, D4>;
pub type Tensor6<
    T,
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
> = T6<T, D0, D1, D2, D3, D4, D5>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv2dOptions {
    padding: [usize; 4],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl Conv2dOptions {
    pub const fn new() -> Self {
        Self {
            padding: [0, 0, 0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }

    pub const fn padding(mut self, top: usize, bottom: usize, left: usize, right: usize) -> Self {
        self.padding = [top, bottom, left, right];
        self
    }

    pub const fn stride(mut self, height: usize, width: usize) -> Self {
        self.stride = [height, width];
        self
    }

    pub const fn dilation(mut self, height: usize, width: usize) -> Self {
        self.dilation = [height, width];
        self
    }

    pub const fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    fn into_core(self) -> CoreConv2dOptions {
        CoreConv2dOptions {
            padding: Padding2d {
                top: self.padding[0],
                bottom: self.padding[1],
                left: self.padding[2],
                right: self.padding[3],
            },
            stride: self.stride,
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_tensor_binary {
    ($trait:ident, $method:ident, $op:expr) => {
        impl_tensor_binary_for!($trait, $method, $op, T0, []);
        impl_tensor_binary_for!($trait, $method, $op, T1, [D0]);
        impl_tensor_binary_for!($trait, $method, $op, T2, [D0, D1]);
        impl_tensor_binary_for!($trait, $method, $op, T3, [D0, D1, D2]);
        impl_tensor_binary_for!($trait, $method, $op, T4, [D0, D1, D2, D3]);
        impl_tensor_binary_for!($trait, $method, $op, T5, [D0, D1, D2, D3, D4]);
        impl_tensor_binary_for!($trait, $method, $op, T6, [D0, D1, D2, D3, D4, D5]);
    };
}

macro_rules! impl_tensor_binary_for {
    ($trait:ident, $method:ident, $op:expr, $name:ident, []) => {
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [], T0, []);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [], T1, [R0]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [], T2, [R0, R1]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [], T3, [R0, R1, R2]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [], T4, [R0, R1, R2, R3]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [], T5, [R0, R1, R2, R3, R4]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [], T6, [R0, R1, R2, R3, R4, R5]);

        impl<T: ScalarLiteral> $trait<T> for $name<T> {
            type Output = Self;

            fn $method(self, rhs: T) -> Self::Output {
                Self::from_expr(node_expr(Expr::Binary {
                    op: $op,
                    lhs: Box::new(self.into_expr()),
                    rhs: Box::new(rhs.const_expr()),
                }))
            }
        }
    };
    ($trait:ident, $method:ident, $op:expr, $name:ident, [$($dim:ident),+]) => {
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [$($dim),+], T0, []);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [$($dim),+], T1, [R0]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [$($dim),+], T2, [R0, R1]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [$($dim),+], T3, [R0, R1, R2]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [$($dim),+], T4, [R0, R1, R2, R3]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [$($dim),+], T5, [R0, R1, R2, R3, R4]);
        impl_tensor_binary_rhs!($trait, $method, $op, $name, [$($dim),+], T6, [R0, R1, R2, R3, R4, R5]);

        impl<T: ScalarLiteral, $(const $dim: usize),+> $trait<T> for $name<T, $($dim),+> {
            type Output = Self;

            fn $method(self, rhs: T) -> Self::Output {
                Self::from_expr(node_expr(Expr::Binary {
                    op: $op,
                    lhs: Box::new(self.into_expr()),
                    rhs: Box::new(rhs.const_expr()),
                }))
            }
        }
    };
}

macro_rules! impl_tensor_binary_rhs {
    ($trait:ident, $method:ident, $op:expr, $lhs:ident, [], $rhs:ident, []) => {
        impl<L: TraceElement, R: TraceElement> $trait<$rhs<R>> for $lhs<L> {
            type Output = Self;

            fn $method(self, rhs: $rhs<R>) -> Self::Output {
                Self::from_expr(node_expr(Expr::Binary {
                    op: $op,
                    lhs: Box::new(self.into_expr()),
                    rhs: Box::new(rhs.into_expr()),
                }))
            }
        }
    };
    ($trait:ident, $method:ident, $op:expr, $lhs:ident, [], $rhs:ident, [$($rdim:ident),+]) => {
        impl<L: TraceElement, R: TraceElement, $(const $rdim: usize),+> $trait<$rhs<R, $($rdim),+>> for $lhs<L> {
            type Output = Self;

            fn $method(self, rhs: $rhs<R, $($rdim),+>) -> Self::Output {
                Self::from_expr(node_expr(Expr::Binary {
                    op: $op,
                    lhs: Box::new(self.into_expr()),
                    rhs: Box::new(rhs.into_expr()),
                }))
            }
        }
    };
    ($trait:ident, $method:ident, $op:expr, $lhs:ident, [$($ldim:ident),+], $rhs:ident, []) => {
        impl<L: TraceElement, R: TraceElement, $(const $ldim: usize),+> $trait<$rhs<R>> for $lhs<L, $($ldim),+> {
            type Output = Self;

            fn $method(self, rhs: $rhs<R>) -> Self::Output {
                Self::from_expr(node_expr(Expr::Binary {
                    op: $op,
                    lhs: Box::new(self.into_expr()),
                    rhs: Box::new(rhs.into_expr()),
                }))
            }
        }
    };
    ($trait:ident, $method:ident, $op:expr, $lhs:ident, [$($ldim:ident),+], $rhs:ident, [$($rdim:ident),+]) => {
        impl<L: TraceElement, R: TraceElement, $(const $ldim: usize),+, $(const $rdim: usize),+> $trait<$rhs<R, $($rdim),+>> for $lhs<L, $($ldim),+> {
            type Output = Self;

            fn $method(self, rhs: $rhs<R, $($rdim),+>) -> Self::Output {
                Self::from_expr(node_expr(Expr::Binary {
                    op: $op,
                    lhs: Box::new(self.into_expr()),
                    rhs: Box::new(rhs.into_expr()),
                }))
            }
        }
    };
}

impl_tensor_binary!(Add, add, BinaryOp::Add);
impl_tensor_binary!(Sub, sub, BinaryOp::Sub);
impl_tensor_binary!(Mul, mul, BinaryOp::Mul);
impl_tensor_binary!(Div, div, BinaryOp::Div);

macro_rules! impl_tensor_neg {
    ($name:ident, []) => {
        impl<T: TraceElement> Neg for $name<T> {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self::from_expr(node_expr(Expr::Unary {
                    op: UnaryOp::Neg,
                    value: Box::new(self.into_expr()),
                }))
            }
        }
    };
    ($name:ident, [$($dim:ident),+]) => {
        impl<T: TraceElement, $(const $dim: usize),+> Neg for $name<T, $($dim),+> {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self::from_expr(node_expr(Expr::Unary {
                    op: UnaryOp::Neg,
                    value: Box::new(self.into_expr()),
                }))
            }
        }
    };
}

impl_tensor_neg!(T0, []);
impl_tensor_neg!(T1, [D0]);
impl_tensor_neg!(T2, [D0, D1]);
impl_tensor_neg!(T3, [D0, D1, D2]);
impl_tensor_neg!(T4, [D0, D1, D2, D3]);
impl_tensor_neg!(T5, [D0, D1, D2, D3, D4]);
impl_tensor_neg!(T6, [D0, D1, D2, D3, D4, D5]);

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

fn call_output<Output: TraceTensor>(op: CallOp, args: Vec<Expr>) -> Output {
    Output::from_expr(node_expr(Expr::Call { op, args }))
}

fn node_expr(value: Expr) -> Expr {
    Expr::Node {
        node_id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
        value: Box::new(value),
    }
}

fn next_tuple_id() -> u64 {
    NEXT_TUPLE_ID.fetch_add(1, Ordering::Relaxed)
}

fn unary<T: TraceTensor>(op: CallOp, value: T) -> T {
    T::from_expr(node_expr(Expr::Call {
        op,
        args: vec![value.into_expr()],
    }))
}

fn unary_bool<T: BoolTensor>(op: CallOp, value: T) -> T::Bool {
    <T::Bool as TraceTensor>::from_expr(node_expr(Expr::Call {
        op,
        args: vec![value.into_expr()],
    }))
}

fn binary_same<L, R>(op: CallOp, lhs: L, rhs: R) -> L
where
    L: TraceTensor,
    R: TraceOperand<L::Elem>,
{
    L::from_expr(node_expr(Expr::Call {
        op,
        args: vec![lhs.into_expr(), rhs.into_operand_expr()],
    }))
}

fn binary_bool<L, R>(op: CallOp, lhs: L, rhs: R) -> L::Bool
where
    L: BoolTensor,
    R: TraceOperand<L::Elem>,
{
    <L::Bool as TraceTensor>::from_expr(node_expr(Expr::Call {
        op,
        args: vec![lhs.into_expr(), rhs.into_operand_expr()],
    }))
}

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

pub fn reshape<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    Output::from_expr(node_expr(Expr::Call {
        op: CallOp::Reshape(Output::tensor_type()),
        args: vec![value.into_expr()],
    }))
}

pub fn broadcast<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    Output::from_expr(node_expr(Expr::Call {
        op: CallOp::Broadcast(Output::tensor_type()),
        args: vec![value.into_expr()],
    }))
}

pub fn squeeze<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(
        CallOp::Squeeze(Output::tensor_type()),
        vec![value.into_expr()],
    )
}

pub fn unsqueeze<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(
        CallOp::Unsqueeze(Output::tensor_type()),
        vec![value.into_expr()],
    )
}

pub fn slice<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    starts: [usize; N],
) -> Output {
    trace_slice(value, &starts)
}

fn trace_slice<Output: TraceTensor>(value: impl TraceTensor, starts: &[usize]) -> Output {
    call_output(
        CallOp::Slice {
            target: Output::tensor_type(),
            starts: starts.to_vec(),
        },
        vec![value.into_expr()],
    )
}

fn trace_pad<Output: TraceTensor>(value: impl TraceTensor, lows: &[usize]) -> Output {
    call_output(
        CallOp::Pad {
            target: Output::tensor_type(),
            lows: lows.to_vec(),
        },
        vec![value.into_expr()],
    )
}

pub fn pad<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    lows: [usize; N],
) -> Output {
    trace_pad(value, &lows)
}

pub fn gather<Output: TraceTensor>(
    value: impl TraceTensor,
    indices: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_gather(value, indices, axis)
}

fn trace_gather<Output: TraceTensor>(
    value: impl TraceTensor,
    indices: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(
        CallOp::Gather {
            target: Output::tensor_type(),
            axis,
        },
        vec![value.into_expr(), indices.into_expr()],
    )
}

pub fn take<Output: TraceTensor>(value: impl TraceTensor, axis: usize, index: usize) -> Output {
    trace_take(value, axis, index)
}

fn trace_take<Output: TraceTensor>(value: impl TraceTensor, axis: usize, index: usize) -> Output {
    call_output(CallOp::Take { axis, index }, vec![value.into_expr()])
}

pub fn take_along_axis<Output: TraceTensor>(
    value: impl TraceTensor,
    indices: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_take_along_axis(value, indices, axis)
}

fn trace_take_along_axis<Output: TraceTensor>(
    value: impl TraceTensor,
    indices: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(
        CallOp::TakeAlongAxis { axis },
        vec![value.into_expr(), indices.into_expr()],
    )
}

pub fn concat<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_concat(lhs, rhs, axis)
}

fn trace_concat<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(CallOp::Concat(axis), vec![lhs.into_expr(), rhs.into_expr()])
}

pub fn stack<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    trace_stack(lhs, rhs, axis)
}

fn trace_stack<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    axis: usize,
) -> Output {
    call_output(CallOp::Stack(axis), vec![lhs.into_expr(), rhs.into_expr()])
}

pub fn split<Output: TraceVars, const N: usize>(
    value: impl TraceTensor,
    axis: usize,
    sections: [usize; N],
) -> Output {
    trace_split(value, axis, &sections)
}

fn trace_split<Output: TraceVars>(
    value: impl TraceTensor,
    axis: usize,
    sections: &[usize],
) -> Output {
    if sections.len() != Output::var_count() {
        panic!(
            "split sections produce {} outputs, but the requested Rust output type has {} values",
            sections.len(),
            Output::var_count()
        );
    }
    let value = Expr::Call {
        op: CallOp::Split {
            axis,
            sections: sections.to_vec(),
        },
        args: vec![value.into_expr()],
    };
    Output::from_tuple_expr(next_tuple_id(), value)
}

pub fn tile<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    multiples: [usize; N],
) -> Output {
    trace_tile(value, &multiples)
}

fn trace_tile<Output: TraceTensor>(value: impl TraceTensor, multiples: &[usize]) -> Output {
    call_output(CallOp::Tile(multiples.to_vec()), vec![value.into_expr()])
}

pub fn repeat<Output: TraceTensor>(value: impl TraceTensor, axis: usize, count: usize) -> Output {
    trace_repeat(value, axis, count)
}

fn trace_repeat<Output: TraceTensor>(value: impl TraceTensor, axis: usize, count: usize) -> Output {
    call_output(CallOp::Repeat { axis, count }, vec![value.into_expr()])
}

pub fn flip<T: TraceTensor>(value: T) -> T {
    trace_flip(value)
}

fn trace_flip<T: TraceTensor>(value: T) -> T {
    unary(CallOp::Flip(Vec::new()), value)
}

pub fn flip_axes<T: TraceTensor, const N: usize>(value: T, axes: [usize; N]) -> T {
    trace_flip_axes(value, &axes)
}

fn trace_flip_axes<T: TraceTensor>(value: T, axes: &[usize]) -> T {
    unary(CallOp::Flip(axes.to_vec()), value)
}

pub fn roll<T: TraceTensor>(value: T, axis: usize, shift: usize) -> T {
    trace_roll(value, axis, shift)
}

fn trace_roll<T: TraceTensor>(value: T, axis: usize, shift: usize) -> T {
    unary(CallOp::Roll { axis, shift }, value)
}

pub fn transpose<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    trace_transpose(value)
}

fn trace_transpose<Output: TraceTensor>(value: impl TraceTensor) -> Output {
    call_output(CallOp::Transpose(Vec::new()), vec![value.into_expr()])
}

pub fn transpose_axes<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    axes: [usize; N],
) -> Output {
    trace_transpose_axes(value, &axes)
}

fn trace_transpose_axes<Output: TraceTensor>(value: impl TraceTensor, axes: &[usize]) -> Output {
    call_output(CallOp::Transpose(axes.to_vec()), vec![value.into_expr()])
}

pub fn permute<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    axes: [usize; N],
) -> Output {
    trace_permute(value, &axes)
}

fn trace_permute<Output: TraceTensor>(value: impl TraceTensor, axes: &[usize]) -> Output {
    call_output(
        CallOp::Permute {
            target: Output::tensor_type(),
            axes: axes.to_vec(),
        },
        vec![value.into_expr()],
    )
}

pub fn permute_dims<Output: TraceTensor, const N: usize>(
    value: impl TraceTensor,
    axes: [usize; N],
) -> Output {
    trace_permute_dims(value, &axes)
}

fn trace_permute_dims<Output: TraceTensor>(value: impl TraceTensor, axes: &[usize]) -> Output {
    call_output(CallOp::PermuteDims(axes.to_vec()), vec![value.into_expr()])
}

pub fn swapaxes<Output: TraceTensor>(
    value: impl TraceTensor,
    axis0: usize,
    axis1: usize,
) -> Output {
    trace_swapaxes(value, axis0, axis1)
}

fn trace_swapaxes<Output: TraceTensor>(
    value: impl TraceTensor,
    axis0: usize,
    axis1: usize,
) -> Output {
    call_output(CallOp::SwapAxes { axis0, axis1 }, vec![value.into_expr()])
}

pub fn moveaxis<Output: TraceTensor>(
    value: impl TraceTensor,
    source: usize,
    destination: usize,
) -> Output {
    trace_moveaxis(value, source, destination)
}

fn trace_moveaxis<Output: TraceTensor>(
    value: impl TraceTensor,
    source: usize,
    destination: usize,
) -> Output {
    call_output(
        CallOp::MoveAxis {
            source,
            destination,
        },
        vec![value.into_expr()],
    )
}

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

pub fn conv2d<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    trace_conv2d(lhs, rhs)
}

fn trace_conv2d<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    conv2d_options(lhs, rhs, Conv2dOptions::default())
}

pub fn conv2d_options<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    options: Conv2dOptions,
) -> Output {
    call_output(
        CallOp::Conv2d(options.into_core()),
        vec![lhs.into_expr(), rhs.into_expr()],
    )
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
