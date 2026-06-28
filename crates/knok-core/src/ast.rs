#[derive(Clone, Debug, PartialEq, Eq)]
/// Parsed graph function before type checking.
pub struct Graph {
    /// Rust function name used as the exported graph function name.
    pub name: String,
    /// Selected IREE target backend, such as `llvm-cpu` or `metal-spirv`.
    pub backend: String,
    /// Declared graph inputs in source order.
    pub inputs: Vec<Input>,
    /// Declared graph outputs in source order.
    pub outputs: Vec<TensorType>,
    /// Let bindings that precede the final body expression or expressions.
    pub lets: Vec<Let>,
    /// Final returned graph expressions.
    pub body: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// One parsed graph input.
pub struct Input {
    /// Source-level argument name.
    pub name: String,
    /// Static tensor type declared for the argument.
    pub ty: TensorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Parsed `let` binding inside a graph body.
pub struct Let {
    /// Bound names. Multi-output graph calls use more than one name.
    pub names: Vec<String>,
    /// Initializer expression for the binding.
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Untyped graph expression.
pub enum Expr {
    /// Reference to a graph input or local binding.
    Var(String),
    /// Literal scalar constant.
    Const {
        /// MLIR-compatible literal spelling.
        value: String,
        /// Static element type of the literal.
        elem: ElementType,
    },
    /// Unary operator expression.
    Unary {
        /// Operator to apply.
        op: UnaryOp,
        /// Operand expression.
        value: Box<Expr>,
    },
    /// Binary operator expression.
    Binary {
        /// Operator to apply.
        op: BinaryOp,
        /// Left-hand operand.
        lhs: Box<Expr>,
        /// Right-hand operand.
        rhs: Box<Expr>,
    },
    /// Function-like graph operation call.
    Call {
        /// Parsed operation.
        op: CallOp,
        /// Call arguments.
        args: Vec<Expr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Unary graph operator.
pub enum UnaryOp {
    /// Numeric negation.
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Binary graph operator.
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
/// Function-like operation accepted by the graph parser.
pub enum CallOp {
    /// Elementwise absolute value.
    Abs,
    /// Boolean all-reduction over all elements or one axis.
    All(AxisSpec),
    /// Index of the maximum element over all elements or one axis.
    Argmax(AxisSpec),
    /// Index of the minimum element over all elements or one axis.
    Argmin(AxisSpec),
    /// Boolean any-reduction over all elements or one axis.
    Any(AxisSpec),
    /// Elementwise clipping with lower and upper bounds.
    Clip,
    /// Concatenate two tensors along an existing axis.
    Concat(usize),
    /// NHWC/HWCF 2D convolution.
    Conv2d(Conv2dOptions),
    /// Elementwise ceiling.
    Ceil,
    /// Extract a diagonal over the default or explicit pair of axes.
    Diagonal(Option<[usize; 2]>),
    /// Vector dot product.
    Dot,
    /// Elementwise equality comparison.
    Equal,
    /// Static identity matrix creation.
    Eye(TensorType),
    /// Elementwise exponential.
    Exp,
    /// Elementwise base-2 exponential.
    Exp2,
    /// Elementwise `exp(x) - 1`.
    ExpM1,
    /// Elementwise floor.
    Floor,
    /// Fill a tensor with the shape and type of another tensor.
    FullLike,
    /// Elementwise greater-than comparison.
    Greater,
    /// Elementwise greater-than-or-equal comparison.
    GreaterEqual,
    /// Gather values along an axis using an index tensor.
    Gather {
        /// Statically declared output tensor type.
        target: TensorType,
        /// Axis to gather from.
        axis: usize,
    },
    /// Elementwise NaN predicate.
    IsNan,
    /// Elementwise less-than comparison.
    Less,
    /// Elementwise less-than-or-equal comparison.
    LessEqual,
    /// Static rank-1 range creation.
    Arange(TensorType),
    /// Static rank-1 evenly spaced value creation.
    Linspace(TensorType),
    /// Elementwise natural logarithm.
    Log,
    /// Elementwise `log(1 + x)`.
    Log1P,
    /// Elementwise base-2 logarithm.
    Log2,
    /// Elementwise base-10 logarithm.
    Log10,
    /// Elementwise boolean and.
    LogicalAnd,
    /// Elementwise boolean not.
    LogicalNot,
    /// Elementwise boolean or.
    LogicalOr,
    /// Elementwise boolean xor.
    LogicalXor,
    /// NumPy-style inner product.
    Inner,
    /// NumPy-style matrix multiplication for ranks 1 through 6.
    Matmul,
    /// Maximum reduction over all elements or one axis.
    Max(AxisSpec),
    /// Mean reduction over all elements or one axis.
    Mean(AxisSpec),
    /// Minimum reduction over all elements or one axis.
    Min(AxisSpec),
    /// Elementwise minimum.
    Minimum,
    /// Elementwise maximum.
    Maximum,
    /// Elementwise inequality comparison.
    NotEqual,
    /// Reverse all axes or selected axes.
    Flip(Vec<usize>),
    /// Move one axis to a new position.
    MoveAxis {
        /// Source axis.
        source: usize,
        /// Destination axis after removal of the source axis.
        destination: usize,
    },
    /// Static padding operation.
    Pad {
        /// Statically declared output tensor type.
        target: TensorType,
        /// Low padding amount per axis.
        lows: Vec<usize>,
    },
    /// Flattened outer product.
    Outer,
    /// Elementwise power.
    Pow,
    /// Product reduction over all elements or one axis.
    Prod(AxisSpec),
    /// Peak-to-peak range reduction over all elements or one axis.
    Ptp(AxisSpec),
    /// Explicit permutation with a declared output type.
    Permute {
        /// Statically declared output tensor type.
        target: TensorType,
        /// Axis order.
        axes: Vec<usize>,
    },
    /// Shape-inferred permutation.
    PermuteDims(Vec<usize>),
    /// Elementwise reciprocal.
    Reciprocal,
    /// Elementwise rectified linear unit.
    Relu,
    /// Repeat each element along one axis.
    Repeat {
        /// Axis to repeat.
        axis: usize,
        /// Repeat count.
        count: usize,
    },
    /// Type-directed reshape.
    Reshape(TensorType),
    /// Roll elements along one axis.
    Roll {
        /// Axis to roll.
        axis: usize,
        /// Positive shift amount.
        shift: usize,
    },
    /// Type-directed broadcast.
    Broadcast(TensorType),
    /// Elementwise round-to-integral using backend rint semantics.
    Rint,
    /// Elementwise round.
    Round,
    /// Elementwise sigmoid.
    Sigmoid,
    /// Static slice with a declared output type.
    Slice {
        /// Statically declared output tensor type.
        target: TensorType,
        /// Start index per sliced axis.
        starts: Vec<usize>,
    },
    /// Softmax over all elements or one axis.
    Softmax(AxisSpec),
    /// Elementwise square root.
    Sqrt,
    /// Elementwise square.
    Square,
    /// Type-directed squeeze.
    Squeeze(TensorType),
    /// Stack two tensors along a new axis.
    Stack(usize),
    /// Standard deviation reduction over all elements or one axis.
    Std(AxisSpec),
    /// Sum reduction over all elements or one axis.
    Sum(AxisSpec),
    /// Static split into multiple outputs.
    Split {
        /// Axis to split.
        axis: usize,
        /// Section sizes along the split axis.
        sections: Vec<usize>,
    },
    /// Swap two axes.
    SwapAxes {
        /// First axis.
        axis0: usize,
        /// Second axis.
        axis1: usize,
    },
    /// Elementwise hyperbolic tangent.
    Tanh,
    /// Elementwise cosine.
    Cos,
    /// Elementwise sine.
    Sin,
    /// Elementwise tangent.
    Tan,
    /// Take a static index from one axis.
    Take {
        /// Axis to index.
        axis: usize,
        /// Static index.
        index: usize,
    },
    /// Tile a tensor by static per-axis multiples.
    Tile(Vec<usize>),
    /// NumPy-style `take_along_axis`.
    TakeAlongAxis {
        /// Axis containing the indexed values.
        axis: usize,
    },
    /// Trace over the default or explicit pair of axes.
    Trace(Option<[usize; 2]>),
    /// Transpose with explicit axes, or reversed axes when empty.
    Transpose(Vec<usize>),
    /// Type-directed unsqueeze.
    Unsqueeze(TensorType),
    /// Variance reduction over all elements or one axis.
    Var(AxisSpec),
    /// Vector dot over the last axis or an explicit axis.
    Vecdot(Option<usize>),
    /// Elementwise select.
    Where,
    /// Create ones with the shape and type of another tensor.
    OnesLike,
    /// Create zeros with the shape and type of another tensor.
    ZerosLike,
    /// Call to another registered graph by name.
    Graph(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Reduction axis selection.
pub enum AxisSpec {
    /// Reduce over every element.
    All,
    /// Reduce over one static axis.
    One(usize),
}

impl AxisSpec {
    /// Converts an optional axis to a full-tensor or one-axis selection.
    pub fn from_optional(axis: Option<usize>) -> Self {
        match axis {
            Some(axis) => Self::One(axis),
            None => Self::All,
        }
    }

    /// Returns the selected axis, or `None` for a full-tensor selection.
    pub fn index(self) -> Option<usize> {
        match self {
            Self::All => None,
            Self::One(axis) => Some(axis),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Static options for the `conv2d` graph operation.
pub struct Conv2dOptions {
    /// Explicit zero padding.
    pub padding: Padding2d,
    /// Height and width stride.
    pub stride: [usize; 2],
    /// Height and width dilation.
    pub dilation: [usize; 2],
    /// Number of convolution groups.
    pub groups: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Top, bottom, left, and right 2D padding.
pub struct Padding2d {
    /// Padding before the height axis.
    pub top: usize,
    /// Padding after the height axis.
    pub bottom: usize,
    /// Padding before the width axis.
    pub left: usize,
    /// Padding after the width axis.
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

#[derive(Clone, Debug, PartialEq, Eq)]
/// Type-checked graph function.
pub struct TypedGraph {
    /// Rust function name used as the exported graph function name.
    pub name: String,
    /// Selected IREE target backend.
    pub backend: String,
    /// Declared graph inputs in source order.
    pub inputs: Vec<Input>,
    /// Declared graph outputs in source order.
    pub outputs: Vec<TensorType>,
    /// Type-checked let bindings.
    pub lets: Vec<TypedLet>,
    /// Type-checked final body expressions.
    pub body: Vec<TypedExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Type-checked `let` binding.
pub struct TypedLet {
    /// Bound names.
    pub names: Vec<String>,
    /// Typed initializer value.
    pub value: TypedValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Type-checked value, which may contain more than one tensor result.
pub struct TypedValue {
    /// Original expression.
    pub kind: Expr,
    /// Tensor types produced by the expression.
    pub tys: Vec<TensorType>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Type-checked single-result expression.
pub struct TypedExpr {
    /// Original expression.
    pub kind: Expr,
    /// Tensor type produced by the expression.
    pub ty: TensorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Static tensor element type and shape.
pub struct TensorType {
    /// Element type.
    pub elem: ElementType,
    /// Static dimensions in row-major order.
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Input and output tensor types for a graph callable by another graph.
pub struct GraphSignature {
    /// Input tensor types in call order.
    pub inputs: Vec<TensorType>,
    /// Output tensor types in return order.
    pub outputs: Vec<TensorType>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Tensor element type understood by the graph parser and lowerer.
pub enum ElementType {
    /// Boolean predicate element, lowered as MLIR `i1`.
    Bool,
    /// 32-bit IEEE floating-point element.
    F32,
    /// 64-bit IEEE floating-point element.
    F64,
    /// 16-bit IEEE floating-point element.
    F16,
    /// 16-bit brain floating-point element.
    BF16,
    /// Signed 32-bit integer element.
    I32,
    /// Signed 64-bit integer element.
    I64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Literal scalar value recoverable from a parsed expression.
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

    /// Formats the type as an MLIR ranked tensor type.
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
    /// Returns the MLIR scalar type spelling.
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

    /// Returns true for `bool`.
    pub fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    /// Returns true for every non-boolean numeric element type.
    pub fn is_numeric(self) -> bool {
        !self.is_bool()
    }

    /// Returns the MLIR literal spelling for zero of this element type.
    pub fn zero_literal(self) -> &'static str {
        match self {
            Self::Bool => "0",
            Self::F32 | Self::F64 | Self::F16 | Self::BF16 => "0.0",
            Self::I32 | Self::I64 => "0",
        }
    }

    /// Returns the MLIR literal spelling for one of this element type.
    pub fn one_literal(self) -> &'static str {
        match self {
            Self::Bool => "1",
            Self::F32 | Self::F64 | Self::F16 | Self::BF16 => "1.0",
            Self::I32 | Self::I64 => "1",
        }
    }
}

impl Expr {
    /// Returns this expression as a static scalar literal when possible.
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

/// Expands a static `arange::<Target>(...)` operation into MLIR literals.
pub fn static_arange_literals(target: &TensorType, args: &[Expr]) -> Result<Vec<String>, String> {
    validate_static_vector_target(target, "arange")?;
    let params = parse_numeric_params("arange", args, 1..=3)?;
    let zero = NumericParam::Int(0);
    let one = NumericParam::Int(1);
    let (start, stop, step) = match params.as_slice() {
        [stop] => (&zero, stop, &one),
        [start, stop] => (start, stop, &one),
        [start, stop, step] => (start, stop, step),
        _ => unreachable!("parse_numeric_params enforces arange arity"),
    };
    let len = target.shape[0];
    match target.elem {
        ElementType::I32 | ElementType::I64 => {
            let start = start
                .as_int()
                .ok_or_else(|| "arange integer targets require integer parameters".to_string())?;
            let stop = stop
                .as_int()
                .ok_or_else(|| "arange integer targets require integer parameters".to_string())?;
            let step = step
                .as_int()
                .ok_or_else(|| "arange integer targets require integer parameters".to_string())?;
            let expected = integer_arange_len(start, stop, step)?;
            if expected != len {
                return Err(format!(
                    "arange produces {expected} values but target shape {:?} has {len}",
                    target.shape
                ));
            }
            (0..len)
                .map(|index| {
                    let value = start + step * index as i128;
                    integer_literal_for_elem(value, target.elem)
                })
                .collect()
        }
        ElementType::F32 | ElementType::F64 | ElementType::F16 | ElementType::BF16 => {
            let start = start.as_float();
            let stop = stop.as_float();
            let step = step.as_float();
            let expected = float_arange_len(start, stop, step)?;
            if expected != len {
                return Err(format!(
                    "arange produces {expected} values but target shape {:?} has {len}",
                    target.shape
                ));
            }
            Ok((0..len)
                .map(|index| float_literal(start + step * index as f64))
                .collect())
        }
        ElementType::Bool => Err("arange target element type must be numeric".to_string()),
    }
}

/// Expands a static `linspace::<Target>(start, stop)` operation into MLIR literals.
pub fn static_linspace_literals(target: &TensorType, args: &[Expr]) -> Result<Vec<String>, String> {
    validate_static_vector_target(target, "linspace")?;
    let params = parse_numeric_params("linspace", args, 2..=2)?;
    let start = params[0];
    let stop = params[1];
    let len = target.shape[0];
    match target.elem {
        ElementType::I32 | ElementType::I64 => {
            let start = start
                .as_int()
                .ok_or_else(|| "linspace integer targets require integer parameters".to_string())?;
            let stop = stop
                .as_int()
                .ok_or_else(|| "linspace integer targets require integer parameters".to_string())?;
            let values = integer_linspace_values(start, stop, len)?;
            values
                .into_iter()
                .map(|value| integer_literal_for_elem(value, target.elem))
                .collect()
        }
        ElementType::F32 | ElementType::F64 | ElementType::F16 | ElementType::BF16 => {
            let start = start.as_float();
            let stop = stop.as_float();
            Ok(float_linspace_values(start, stop, len)
                .into_iter()
                .map(float_literal)
                .collect())
        }
        ElementType::Bool => Err("linspace target element type must be numeric".to_string()),
    }
}

/// Expands a static square `eye::<Target>()` operation into MLIR literals.
pub fn static_eye_literals(target: &TensorType) -> Result<Vec<String>, String> {
    if target.rank() != 2 {
        return Err(format!(
            "eye target must be rank-2, got rank-{} shape {:?}",
            target.rank(),
            target.shape
        ));
    }
    if target.shape[0] != target.shape[1] {
        return Err(format!(
            "eye target matrix must be square, got shape {:?}",
            target.shape
        ));
    }
    let rows = target.shape[0];
    let mut values = Vec::with_capacity(rows * rows);
    for row in 0..rows {
        for col in 0..rows {
            values.push(if row == col {
                target.elem.one_literal().to_string()
            } else {
                target.elem.zero_literal().to_string()
            });
        }
    }
    Ok(values)
}

fn validate_static_vector_target(target: &TensorType, op_name: &str) -> Result<(), String> {
    if !target.elem.is_numeric() {
        return Err(format!("{op_name} target element type must be numeric"));
    }
    if target.rank() != 1 {
        return Err(format!(
            "{op_name} target must be rank-1, got rank-{} shape {:?}",
            target.rank(),
            target.shape
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum NumericParam {
    Int(i128),
    Float(f64),
}

impl NumericParam {
    fn as_int(self) -> Option<i128> {
        match self {
            Self::Int(value) => Some(value),
            Self::Float(_) => None,
        }
    }

    fn as_float(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }
}

fn parse_numeric_params(
    op_name: &str,
    args: &[Expr],
    expected: core::ops::RangeInclusive<usize>,
) -> Result<Vec<NumericParam>, String> {
    if !expected.contains(&args.len()) {
        let expected = if expected.start() == expected.end() {
            expected.start().to_string()
        } else {
            format!("{} to {}", expected.start(), expected.end())
        };
        return Err(format!(
            "{op_name} expects {expected} literal arguments, got {}",
            args.len()
        ));
    }
    args.iter()
        .map(|arg| match arg.static_scalar() {
            Some(StaticScalar::Int(value)) => Ok(NumericParam::Int(value)),
            Some(StaticScalar::Float(value)) if value.is_finite() => Ok(NumericParam::Float(value)),
            Some(StaticScalar::Float(_)) => Err(format!(
                "{op_name} parameters must be finite numeric literals"
            )),
            Some(StaticScalar::Bool(_)) | None => {
                Err(format!("{op_name} parameters must be numeric literals"))
            }
        })
        .collect()
}

fn integer_arange_len(start: i128, stop: i128, step: i128) -> Result<usize, String> {
    if step == 0 {
        return Err("arange step must not be zero".to_string());
    }
    let distance = stop - start;
    if (step > 0 && distance <= 0) || (step < 0 && distance >= 0) {
        return Ok(0);
    }
    let distance = distance.unsigned_abs();
    let step = step.unsigned_abs();
    usize::try_from(distance.div_ceil(step)).map_err(|_| "arange length exceeds usize".to_string())
}

fn float_arange_len(start: f64, stop: f64, step: f64) -> Result<usize, String> {
    if step == 0.0 {
        return Err("arange step must not be zero".to_string());
    }
    let distance = stop - start;
    if (step > 0.0 && distance <= 0.0) || (step < 0.0 && distance >= 0.0) {
        return Ok(0);
    }
    let len = (distance / step).ceil();
    if !len.is_finite() || len < 0.0 || len > usize::MAX as f64 {
        return Err("arange length exceeds usize".to_string());
    }
    Ok(len as usize)
}

fn integer_linspace_values(start: i128, stop: i128, len: usize) -> Result<Vec<i128>, String> {
    match len {
        0 => Ok(Vec::new()),
        1 => Ok(vec![start]),
        _ => {
            let intervals = len as i128 - 1;
            let distance = stop - start;
            if distance % intervals != 0 {
                return Err(format!(
                    "linspace integer target requires evenly divisible endpoints for {len} values"
                ));
            }
            let step = distance / intervals;
            Ok((0..len).map(|index| start + step * index as i128).collect())
        }
    }
}

fn float_linspace_values(start: f64, stop: f64, len: usize) -> Vec<f64> {
    match len {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (len - 1) as f64;
            (0..len).map(|index| start + step * index as f64).collect()
        }
    }
}

fn integer_literal_for_elem(value: i128, elem: ElementType) -> Result<String, String> {
    match elem {
        ElementType::I32 if value < i32::MIN as i128 || value > i32::MAX as i128 => {
            Err(format!("integer literal {value} does not fit in i32"))
        }
        ElementType::I64 if value < i64::MIN as i128 || value > i64::MAX as i128 => {
            Err(format!("integer literal {value} does not fit in i64"))
        }
        ElementType::I32 | ElementType::I64 => Ok(value.to_string()),
        _ => Err("integer literal target must be i32 or i64".to_string()),
    }
}

fn float_literal(value: f64) -> String {
    if value == 0.0 {
        return "0.0".to_string();
    }
    let text = value.to_string();
    if text.contains('.') || text.contains('e') || text.contains('E') {
        text
    } else {
        format!("{text}.0")
    }
}
