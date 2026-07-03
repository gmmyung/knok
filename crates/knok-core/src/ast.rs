use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Untyped graph IR produced before semantic validation.
pub struct Graph {
    /// Graph function name used for generated modules and graph calls.
    pub name: String,
    /// IREE target backend name selected by the graph attribute.
    pub backend: String,
    /// Named graph inputs in declaration order.
    pub inputs: Vec<Input>,
    /// Declared output tensor types in return order.
    pub outputs: Vec<TensorType>,
    /// Let bindings emitted before the graph body.
    pub lets: Vec<Let>,
    /// Body expressions returned by the graph.
    pub body: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// One named graph input.
pub struct Input {
    /// Rust argument name exposed to graph expressions.
    pub name: String,
    /// Static tensor type of the input.
    pub ty: TensorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Untyped let binding in graph IR.
pub struct Let {
    /// Names bound by this let expression.
    pub names: Vec<String>,
    /// Expression assigned to the names.
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Untyped graph expression.
pub enum Expr {
    /// Reference to an input or let-bound value.
    Var(String),
    /// Scalar literal with an explicit element type.
    Const {
        /// Literal text used by lowering.
        value: String,
        /// Literal element type.
        elem: ElementType,
    },
    /// Unary arithmetic expression.
    Unary {
        /// Unary operator.
        op: UnaryOp,
        /// Operand expression.
        value: Box<Expr>,
    },
    /// Binary arithmetic expression.
    Binary {
        /// Binary operator.
        op: BinaryOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// Traced node that should be let-bound once even when cloned.
    Node {
        /// Stable trace node identifier.
        node_id: u64,
        /// Shared expression payload for this node.
        value: Arc<Expr>,
    },
    /// Projection from a traced multi-output expression.
    TupleGet {
        /// Stable tuple expression identifier.
        tuple_id: u64,
        /// Shared tuple-producing expression.
        value: Arc<Expr>,
        /// Zero-based tuple output index.
        index: usize,
    },
    /// Function-style graph operation call.
    Call {
        /// Operation kind.
        op: CallOp,
        /// Operation arguments.
        args: Vec<Expr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Unary arithmetic operator.
pub enum UnaryOp {
    /// Arithmetic negation.
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Binary arithmetic operator.
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Function-style graph operation.
pub enum CallOp {
    /// Absolute value.
    Abs,
    /// Boolean all reduction.
    All(AxisSpec),
    /// Index of the maximum value.
    Argmax(AxisSpec),
    /// Index of the minimum value.
    Argmin(AxisSpec),
    /// Boolean any reduction.
    Any(AxisSpec),
    /// Clamp values between lower and upper bounds.
    Clip,
    /// Concatenate two tensors along an axis.
    Concat(usize),
    /// 2D convolution with static options.
    Conv2d(Conv2dOptions),
    /// Ceiling.
    Ceil,
    /// Stop-gradient marker for static autograd.
    Detach,
    /// Extract a diagonal over the default or explicit axes.
    Diagonal(Option<[usize; 2]>),
    /// Dot product.
    Dot,
    /// Equality comparison.
    Equal,
    /// Identity matrix construction for a target type.
    Eye(TensorType),
    /// Natural exponential.
    Exp,
    /// Base-2 exponential.
    Exp2,
    /// Exponential minus one.
    ExpM1,
    /// Floor.
    Floor,
    /// Fill a tensor with the shape of another tensor.
    FullLike,
    /// Greater-than comparison.
    Greater,
    /// Greater-than-or-equal comparison.
    GreaterEqual,
    /// Gather values by indices along an axis.
    Gather {
        /// Result tensor type.
        target: TensorType,
        /// Gather axis.
        axis: usize,
    },
    /// NaN predicate.
    IsNan,
    /// Less-than comparison.
    Less,
    /// Less-than-or-equal comparison.
    LessEqual,
    /// Static arange construction for a target type.
    Arange(TensorType),
    /// Static linspace construction for a target type.
    Linspace(TensorType),
    /// Natural logarithm.
    Log,
    /// Natural logarithm of one plus the input.
    Log1P,
    /// Base-2 logarithm.
    Log2,
    /// Base-10 logarithm.
    Log10,
    /// Boolean and.
    LogicalAnd,
    /// Boolean not.
    LogicalNot,
    /// Boolean or.
    LogicalOr,
    /// Boolean xor.
    LogicalXor,
    /// Inner product.
    Inner,
    /// Matrix multiplication with NumPy-style rank handling.
    Matmul,
    /// 2D max pooling with static options.
    MaxPool2d(Pool2dOptions),
    /// Maximum reduction.
    Max(AxisSpec),
    /// 2D average pooling with static options.
    AvgPool2d(Pool2dOptions),
    /// Mean reduction.
    Mean(AxisSpec),
    /// Minimum reduction.
    Min(AxisSpec),
    /// Elementwise minimum.
    Minimum,
    /// Elementwise maximum.
    Maximum,
    /// Inequality comparison.
    NotEqual,
    /// Reverse tensor elements along all or selected axes.
    Flip(Vec<usize>),
    /// Move one axis to another position.
    MoveAxis {
        /// Source axis.
        source: usize,
        /// Destination axis.
        destination: usize,
    },
    /// Pad tensor lows to reach a target type.
    Pad {
        /// Result tensor type.
        target: TensorType,
        /// Low-side padding per axis.
        lows: Vec<usize>,
    },
    /// Outer product.
    Outer,
    /// Elementwise power.
    Pow,
    /// Product reduction.
    Prod(AxisSpec),
    /// Peak-to-peak reduction.
    Ptp(AxisSpec),
    /// Permute axes with an explicit target type.
    Permute {
        /// Result tensor type.
        target: TensorType,
        /// Axis permutation.
        axes: Vec<usize>,
    },
    /// Permute axes while inferring the target from the input.
    PermuteDims(Vec<usize>),
    /// Reciprocal.
    Reciprocal,
    /// Rectified linear unit.
    Relu,
    /// Repeat elements along an axis.
    Repeat {
        /// Repeated axis.
        axis: usize,
        /// Repeat count.
        count: usize,
    },
    /// Reshape to a target type.
    Reshape(TensorType),
    /// Roll elements along an axis.
    Roll {
        /// Rolled axis.
        axis: usize,
        /// Static shift amount.
        shift: usize,
    },
    /// Broadcast to a target type.
    Broadcast(TensorType),
    /// Round to nearest integer using IREE semantics.
    Rint,
    /// Round to nearest integer.
    Round,
    /// Sigmoid activation.
    Sigmoid,
    /// Slice from static starts to a target type.
    Slice {
        /// Result tensor type.
        target: TensorType,
        /// Start index per sliced axis.
        starts: Vec<usize>,
    },
    /// Softmax over all elements or one axis.
    Softmax(AxisSpec),
    /// Square root.
    Sqrt,
    /// Elementwise square.
    Square,
    /// Remove unit dimensions to a target type.
    Squeeze(TensorType),
    /// Stack two tensors along an axis.
    Stack(usize),
    /// Standard deviation reduction.
    Std(AxisSpec),
    /// Sum reduction.
    Sum(AxisSpec),
    /// Split a tensor into sections along an axis.
    Split {
        /// Split axis.
        axis: usize,
        /// Section sizes.
        sections: Vec<usize>,
    },
    /// Swap two axes.
    SwapAxes {
        /// First axis.
        axis0: usize,
        /// Second axis.
        axis1: usize,
    },
    /// Hyperbolic tangent.
    Tanh,
    /// Cosine.
    Cos,
    /// Sine.
    Sin,
    /// Tangent.
    Tan,
    /// Take one static index along an axis.
    Take {
        /// Source axis.
        axis: usize,
        /// Static index.
        index: usize,
    },
    /// Tile by static multiples.
    Tile(Vec<usize>),
    /// Take values according to per-element indices along an axis.
    TakeAlongAxis {
        /// Indexed axis.
        axis: usize,
    },
    /// Matrix trace over the default or explicit axes.
    Trace(Option<[usize; 2]>),
    /// Transpose with default or explicit axes.
    Transpose(Vec<usize>),
    /// Add unit dimensions to a target type.
    Unsqueeze(TensorType),
    /// Variance reduction.
    Var(AxisSpec),
    /// Vector dot product over the default or explicit axis.
    Vecdot(Option<usize>),
    /// Select between two values using a boolean condition.
    Where,
    /// Tensor of ones with the input shape.
    OnesLike,
    /// Tensor of zeros with the input shape.
    ZerosLike,
    /// Call another graph by name.
    Graph(String),
}

impl CallOp {
    /// Returns the stable textual operation name used in diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::All(_) => "all",
            Self::Argmax(_) => "argmax",
            Self::Argmin(_) => "argmin",
            Self::Any(_) => "any",
            Self::Clip => "clip",
            Self::Concat(_) => "concat",
            Self::Conv2d(_) => "conv2d",
            Self::Ceil => "ceil",
            Self::Detach => "detach",
            Self::Diagonal(_) => "diagonal",
            Self::Dot => "dot",
            Self::Equal => "equal",
            Self::Eye(_) => "eye",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::ExpM1 => "expm1",
            Self::Floor => "floor",
            Self::FullLike => "full_like",
            Self::Greater => "greater",
            Self::GreaterEqual => "greater_equal",
            Self::Gather { .. } => "gather",
            Self::IsNan => "isnan",
            Self::Less => "less",
            Self::LessEqual => "less_equal",
            Self::Arange(_) => "arange",
            Self::Linspace(_) => "linspace",
            Self::Log => "log",
            Self::Log1P => "log1p",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::LogicalAnd => "logical_and",
            Self::LogicalNot => "logical_not",
            Self::LogicalOr => "logical_or",
            Self::LogicalXor => "logical_xor",
            Self::Inner => "inner",
            Self::Matmul => "matmul",
            Self::MaxPool2d(_) => "max_pool2d",
            Self::Max(_) => "max",
            Self::AvgPool2d(_) => "avg_pool2d",
            Self::Mean(_) => "mean",
            Self::Min(_) => "min",
            Self::Minimum => "minimum",
            Self::Maximum => "maximum",
            Self::NotEqual => "not_equal",
            Self::Flip(_) => "flip",
            Self::MoveAxis { .. } => "moveaxis",
            Self::Pad { .. } => "pad",
            Self::Outer => "outer",
            Self::Pow => "pow",
            Self::Prod(_) => "prod",
            Self::Ptp(_) => "ptp",
            Self::Permute { .. } => "permute",
            Self::PermuteDims(_) => "permute_dims",
            Self::Reciprocal => "reciprocal",
            Self::Relu => "relu",
            Self::Repeat { .. } => "repeat",
            Self::Reshape(_) => "reshape",
            Self::Roll { .. } => "roll",
            Self::Broadcast(_) => "broadcast",
            Self::Rint => "rint",
            Self::Round => "round",
            Self::Sigmoid => "sigmoid",
            Self::Slice { .. } => "slice",
            Self::Softmax(_) => "softmax",
            Self::Sqrt => "sqrt",
            Self::Square => "square",
            Self::Squeeze(_) => "squeeze",
            Self::Stack(_) => "stack",
            Self::Std(_) => "std",
            Self::Sum(_) => "sum",
            Self::Split { .. } => "split",
            Self::SwapAxes { .. } => "swapaxes",
            Self::Tanh => "tanh",
            Self::Cos => "cos",
            Self::Sin => "sin",
            Self::Tan => "tan",
            Self::Take { .. } => "take",
            Self::Tile(_) => "tile",
            Self::TakeAlongAxis { .. } => "take_along_axis",
            Self::Trace(_) => "trace",
            Self::Transpose(_) => "transpose",
            Self::Unsqueeze(_) => "unsqueeze",
            Self::Var(_) => "var",
            Self::Vecdot(_) => "vecdot",
            Self::Where => "where",
            Self::OnesLike => "ones_like",
            Self::ZerosLike => "zeros_like",
            Self::Graph(_) => "graph",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Axis selection for reductions and axis-sensitive operations.
pub enum AxisSpec {
    /// Operate over all axes.
    All,
    /// Operate over one zero-based axis.
    One(usize),
}

impl AxisSpec {
    /// Converts an optional axis into a graph axis specification.
    pub fn from_optional(axis: Option<usize>) -> Self {
        match axis {
            Some(axis) => Self::One(axis),
            None => Self::All,
        }
    }

    /// Returns the selected axis, or `None` for all-axis operations.
    pub fn index(self) -> Option<usize> {
        match self {
            Self::All => None,
            Self::One(axis) => Some(axis),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Static 2D convolution options.
pub struct Conv2dOptions {
    /// Explicit top/bottom/left/right padding.
    pub padding: Padding2d,
    /// Height and width stride.
    pub stride: [usize; 2],
    /// Height and width dilation.
    pub dilation: [usize; 2],
    /// Number of convolution groups.
    pub groups: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Static 2D pooling options.
pub struct Pool2dOptions {
    /// Height and width kernel size.
    pub kernel: [usize; 2],
    /// Explicit top/bottom/left/right padding.
    pub padding: Padding2d,
    /// Height and width stride.
    pub stride: [usize; 2],
    /// Height and width dilation.
    pub dilation: [usize; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Explicit 2D padding amounts.
pub struct Padding2d {
    /// Top padding.
    pub top: usize,
    /// Bottom padding.
    pub bottom: usize,
    /// Left padding.
    pub left: usize,
    /// Right padding.
    pub right: usize,
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self {
            padding: Padding2d::default(),
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

impl Default for Pool2dOptions {
    fn default() -> Self {
        Self {
            kernel: [2, 2],
            padding: Padding2d::default(),
            stride: [2, 2],
            dilation: [1, 1],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Type-checked graph IR.
pub struct TypedGraph {
    /// Graph function name used for generated modules and graph calls.
    pub name: String,
    /// IREE target backend name selected by the graph attribute.
    pub backend: String,
    /// Named graph inputs in declaration order.
    pub inputs: Vec<Input>,
    /// Output tensor types in return order.
    pub outputs: Vec<TensorType>,
    /// Type-checked let bindings.
    pub lets: Vec<TypedLet>,
    /// Type-checked body expressions.
    pub body: Vec<TypedExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Type-checked let binding.
pub struct TypedLet {
    /// Names bound by this let expression.
    pub names: Vec<String>,
    /// Typed value assigned to the names.
    pub value: TypedValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Type-checked expression value that may produce multiple tensors.
pub struct TypedValue {
    /// Untyped expression kind.
    pub kind: Expr,
    /// Tensor types produced by the value.
    pub tys: Vec<TensorType>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Type-checked expression that produces one tensor.
pub struct TypedExpr {
    /// Untyped expression kind.
    pub kind: Expr,
    /// Tensor type produced by the expression.
    pub ty: TensorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Static tensor type.
pub struct TensorType {
    /// Tensor element type.
    pub elem: ElementType,
    /// Static shape dimensions.
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Input and output types for resolving graph calls.
pub struct GraphSignature {
    /// Input tensor types in call order.
    pub inputs: Vec<TensorType>,
    /// Output tensor types in return order.
    pub outputs: Vec<TensorType>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Supported graph element type.
pub enum ElementType {
    /// Boolean value lowered as `i1`.
    Bool,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
    /// IEEE 16-bit floating point.
    F16,
    /// Brain floating point.
    BF16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Parsed static scalar literal.
pub enum StaticScalar {
    /// Boolean literal.
    Bool(bool),
    /// Integer literal.
    Int(i128),
    /// Floating-point literal.
    Float(f64),
}

impl TensorType {
    /// Returns the tensor rank.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Formats the tensor type as an MLIR `tensor<...>` type.
    pub fn mlir_type(&self) -> String {
        if self.shape.is_empty() {
            return format!("tensor<{}>", self.elem.mlir_type());
        }
        let dims = self
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x");
        format!("tensor<{}x{}>", dims, self.elem.mlir_type())
    }
}

impl ElementType {
    /// Returns the MLIR scalar type name.
    pub fn mlir_type(self) -> &'static str {
        match self {
            Self::Bool => "i1",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    /// Returns true for floating-point element types.
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::F16 | Self::BF16)
    }

    /// Returns true for the boolean element type.
    pub fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    /// Returns true for non-boolean numeric element types.
    pub fn is_numeric(self) -> bool {
        !self.is_bool()
    }

    /// Returns a zero literal suitable for this element type.
    pub fn zero_literal(self) -> &'static str {
        match self {
            Self::Bool => "0",
            Self::F32 | Self::F64 | Self::F16 | Self::BF16 => "0.0",
            Self::I32 | Self::I64 => "0",
        }
    }

    /// Returns a one literal suitable for this element type.
    pub fn one_literal(self) -> &'static str {
        match self {
            Self::Bool => "1",
            Self::F32 | Self::F64 | Self::F16 | Self::BF16 => "1.0",
            Self::I32 | Self::I64 => "1",
        }
    }
}

impl Expr {
    /// Parses this expression as a static scalar when it is representable.
    pub fn static_scalar(&self) -> Option<StaticScalar> {
        match self {
            Self::Const { value, elem } => match elem {
                ElementType::Bool => match value.as_str() {
                    "0" => Some(StaticScalar::Bool(false)),
                    "1" => Some(StaticScalar::Bool(true)),
                    _ => None,
                },
                ElementType::I32 | ElementType::I64 => {
                    value.parse::<i128>().ok().map(StaticScalar::Int)
                }
                ElementType::F32 | ElementType::F64 | ElementType::F16 | ElementType::BF16 => {
                    value.parse::<f64>().ok().map(StaticScalar::Float)
                }
            },
            Self::Unary {
                op: UnaryOp::Neg,
                value,
            } => match value.static_scalar()? {
                StaticScalar::Int(value) => Some(StaticScalar::Int(-value)),
                StaticScalar::Float(value) => Some(StaticScalar::Float(-value)),
                StaticScalar::Bool(_) => None,
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_nodes_share_expression_payloads() {
        let expr = Expr::Node {
            node_id: 1,
            value: Arc::new(Expr::Var("x".into())),
        };
        let cloned = expr.clone();

        match (&expr, &cloned) {
            (Expr::Node { value, .. }, Expr::Node { value: cloned, .. }) => {
                assert!(Arc::ptr_eq(value, cloned));
            }
            _ => unreachable!("test constructs node expressions"),
        }
    }
}
