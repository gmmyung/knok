#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    pub name: String,
    pub backend: String,
    pub inputs: Vec<Input>,
    pub outputs: Vec<TensorType>,
    pub lets: Vec<Let>,
    pub body: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub name: String,
    pub ty: TensorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Let {
    pub names: Vec<String>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Var(String),
    Const {
        value: String,
        elem: ElementType,
    },
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Call {
        op: CallOp,
        args: Vec<Expr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallOp {
    Abs,
    All(AxisSpec),
    Argmax(AxisSpec),
    Any(AxisSpec),
    Clip,
    Concat(usize),
    Conv2d(Conv2dOptions),
    Equal,
    Eye(TensorType),
    Exp,
    FullLike,
    Greater,
    GreaterEqual,
    IsNan,
    Less,
    LessEqual,
    Arange(TensorType),
    Linspace(TensorType),
    Log,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    LogicalXor,
    Matmul,
    Mean(AxisSpec),
    Minimum,
    Maximum,
    NotEqual,
    Pow,
    Permute {
        target: TensorType,
        axes: Vec<usize>,
    },
    Relu,
    Reshape(TensorType),
    Broadcast(TensorType),
    Sigmoid,
    Slice {
        target: TensorType,
        starts: Vec<usize>,
    },
    Softmax(AxisSpec),
    Sqrt,
    Squeeze(TensorType),
    Stack(usize),
    Sum(AxisSpec),
    Tanh,
    Take {
        axis: usize,
        index: usize,
    },
    Transpose,
    Unsqueeze(TensorType),
    Where,
    OnesLike,
    ZerosLike,
    Graph(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisSpec {
    All,
    One(usize),
}

impl AxisSpec {
    pub fn from_optional(axis: Option<usize>) -> Self {
        match axis {
            Some(axis) => Self::One(axis),
            None => Self::All,
        }
    }

    pub fn index(self) -> Option<usize> {
        match self {
            Self::All => None,
            Self::One(axis) => Some(axis),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conv2dOptions {
    pub padding: Padding2d,
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Padding2d {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
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
pub struct TypedGraph {
    pub name: String,
    pub backend: String,
    pub inputs: Vec<Input>,
    pub outputs: Vec<TensorType>,
    pub lets: Vec<TypedLet>,
    pub body: Vec<TypedExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedLet {
    pub names: Vec<String>,
    pub value: TypedValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedValue {
    pub kind: Expr,
    pub tys: Vec<TensorType>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedExpr {
    pub kind: Expr,
    pub ty: TensorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    pub elem: ElementType,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphSignature {
    pub inputs: Vec<TensorType>,
    pub outputs: Vec<TensorType>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    Bool,
    F32,
    F64,
    F16,
    BF16,
    I32,
    I64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticScalar {
    Bool(bool),
    Int(i128),
    Float(f64),
}

impl TensorType {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

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

    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::F16 | Self::BF16)
    }

    pub fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    pub fn is_numeric(self) -> bool {
        !self.is_bool()
    }

    pub fn zero_literal(self) -> &'static str {
        match self {
            Self::Bool => "0",
            Self::F32 | Self::F64 | Self::F16 | Self::BF16 => "0.0",
            Self::I32 | Self::I64 => "0",
        }
    }

    pub fn one_literal(self) -> &'static str {
        match self {
            Self::Bool => "1",
            Self::F32 | Self::F64 | Self::F16 | Self::BF16 => "1.0",
            Self::I32 | Self::I64 => "1",
        }
    }
}

impl Expr {
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
