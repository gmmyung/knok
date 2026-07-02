use std::sync::Arc;

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
    Node {
        node_id: u64,
        value: Arc<Expr>,
    },
    TupleGet {
        tuple_id: u64,
        value: Arc<Expr>,
        index: usize,
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
    Argmin(AxisSpec),
    Any(AxisSpec),
    Clip,
    Concat(usize),
    Conv2d(Conv2dOptions),
    Ceil,
    Diagonal(Option<[usize; 2]>),
    Dot,
    Equal,
    Eye(TensorType),
    Exp,
    Exp2,
    ExpM1,
    Floor,
    FullLike,
    Greater,
    GreaterEqual,
    Gather {
        target: TensorType,
        axis: usize,
    },
    IsNan,
    Less,
    LessEqual,
    Arange(TensorType),
    Linspace(TensorType),
    Log,
    Log1P,
    Log2,
    Log10,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    LogicalXor,
    Inner,
    Matmul,
    Max(AxisSpec),
    Mean(AxisSpec),
    Min(AxisSpec),
    Minimum,
    Maximum,
    NotEqual,
    Flip(Vec<usize>),
    MoveAxis {
        source: usize,
        destination: usize,
    },
    Pad {
        target: TensorType,
        lows: Vec<usize>,
    },
    Outer,
    Pow,
    Prod(AxisSpec),
    Ptp(AxisSpec),
    Permute {
        target: TensorType,
        axes: Vec<usize>,
    },
    PermuteDims(Vec<usize>),
    Reciprocal,
    Relu,
    Repeat {
        axis: usize,
        count: usize,
    },
    Reshape(TensorType),
    Roll {
        axis: usize,
        shift: usize,
    },
    Broadcast(TensorType),
    Rint,
    Round,
    Sigmoid,
    Slice {
        target: TensorType,
        starts: Vec<usize>,
    },
    Softmax(AxisSpec),
    Sqrt,
    Square,
    Squeeze(TensorType),
    Stack(usize),
    Std(AxisSpec),
    Sum(AxisSpec),
    Split {
        axis: usize,
        sections: Vec<usize>,
    },
    SwapAxes {
        axis0: usize,
        axis1: usize,
    },
    Tanh,
    Cos,
    Sin,
    Tan,
    Take {
        axis: usize,
        index: usize,
    },
    Tile(Vec<usize>),
    TakeAlongAxis {
        axis: usize,
    },
    Trace(Option<[usize; 2]>),
    Transpose(Vec<usize>),
    Unsqueeze(TensorType),
    Var(AxisSpec),
    Vecdot(Option<usize>),
    Where,
    OnesLike,
    ZerosLike,
    Graph(String),
}

impl CallOp {
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
            Self::Max(_) => "max",
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
