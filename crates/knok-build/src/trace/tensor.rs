use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use knok_core::{BinaryOp, ElementType, Expr, TensorType, UnaryOp};

use super::expr::node_expr;

/// Element type that can appear in a build-time traced tensor.
pub trait TraceElement: Copy + 'static {
    /// Graph IR element type represented by this Rust scalar type.
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

/// Scalar value that can be embedded as a graph literal during tracing.
pub trait ScalarLiteral: TraceElement {
    /// Converts the scalar into a graph constant expression.
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
            value: mlir_float_literal(format!("{self:?}")),
            elem: Self::ELEMENT,
        }
    }
}

impl ScalarLiteral for f64 {
    fn const_expr(self) -> Expr {
        Expr::Const {
            value: mlir_float_literal(format!("{self:?}")),
            elem: Self::ELEMENT,
        }
    }
}

fn mlir_float_literal(mut text: String) -> String {
    if matches!(text.as_str(), "inf" | "-inf" | "NaN") {
        return text;
    }
    let exponent = text.find('e').or_else(|| text.find('E'));
    if let Some(index) = exponent {
        if !text[..index].contains('.') {
            text.insert_str(index, ".0");
        }
        return text;
    }
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
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

/// Build-time tensor handle used while tracing graph functions.
pub trait TraceTensor: Clone {
    /// Scalar element type of the tensor.
    type Elem: TraceElement;

    /// Returns the static tensor type represented by this handle type.
    fn tensor_type() -> TensorType;
    /// Wraps a graph expression in this tensor handle type.
    fn from_expr(expr: Expr) -> Self;
    /// Consumes the handle and returns the traced graph expression.
    fn into_expr(self) -> Expr;
    /// Borrows the traced graph expression.
    fn expr(&self) -> &Expr;
}

/// Traced tensor that has a shape-compatible boolean mask type.
pub trait BoolTensor: TraceTensor {
    /// Boolean tensor type with the same rank and shape.
    type Bool: TraceTensor<Elem = bool>;
}

/// Value accepted as an operand to a traced tensor operation.
pub trait TraceOperand<E: TraceElement> {
    /// Converts the operand into a graph expression.
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
        /// Rank-0 traced tensor handle used inside `build.rs` graph functions.
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
        /// Fixed-rank traced tensor handle used inside `build.rs` graph functions.
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

/// Alias for a rank-0 traced tensor handle.
pub type Tensor0<T> = T0<T>;
/// Alias for a rank-1 traced tensor handle.
pub type Tensor1<T, const D0: usize> = T1<T, D0>;
/// Alias for a rank-2 traced tensor handle.
pub type Tensor2<T, const D0: usize, const D1: usize> = T2<T, D0, D1>;
/// Alias for a rank-3 traced tensor handle.
pub type Tensor3<T, const D0: usize, const D1: usize, const D2: usize> = T3<T, D0, D1, D2>;
/// Alias for a rank-4 traced tensor handle.
pub type Tensor4<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize> =
    T4<T, D0, D1, D2, D3>;
/// Alias for a rank-5 traced tensor handle.
pub type Tensor5<
    T,
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
> = T5<T, D0, D1, D2, D3, D4>;
/// Alias for a rank-6 traced tensor handle.
pub type Tensor6<
    T,
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
> = T6<T, D0, D1, D2, D3, D4, D5>;

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
