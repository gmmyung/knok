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
    All(Option<usize>),
    Argmax(Option<usize>),
    Any(Option<usize>),
    Clip,
    Concat(usize),
    Conv2d(Conv2dOptions),
    Equal,
    Exp,
    Greater,
    GreaterEqual,
    IsNan,
    Less,
    LessEqual,
    Log,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    LogicalXor,
    Matmul,
    Mean(Option<usize>),
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
    Softmax(Option<usize>),
    Sqrt,
    Squeeze(TensorType),
    Stack(usize),
    Sum(Option<usize>),
    Tanh,
    Take {
        axis: usize,
        index: usize,
    },
    Transpose,
    Unsqueeze(TensorType),
    Where,
    Graph(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conv2dOptions {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self {
            pad_h: 0,
            pad_w: 0,
            stride_h: 1,
            stride_w: 1,
            dilation_h: 1,
            dilation_w: 1,
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
