use proc_macro2::Span;
use syn::{
    parse::Parser, spanned::Spanned, Attribute, BinOp, Expr as SynExpr, FnArg, GenericArgument,
    ItemFn, Lit, MetaNameValue, Pat, PatIdent, ReturnType, Stmt, Type, TypePath, UnOp,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    pub name: String,
    pub backend: String,
    pub inputs: Vec<Input>,
    pub output: TensorType,
    pub lets: Vec<Let>,
    pub body: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub name: String,
    pub ty: TensorType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Let {
    pub name: String,
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
    Argmax,
    Any(Option<usize>),
    Clip,
    Concat(usize),
    Conv2d,
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
pub struct TypedGraph {
    pub name: String,
    pub backend: String,
    pub inputs: Vec<Input>,
    pub output: TensorType,
    pub lets: Vec<TypedLet>,
    pub body: TypedExpr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedLet {
    pub name: String,
    pub value: TypedExpr,
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
    pub output: TensorType,
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

pub fn parse_graph(attr: proc_macro2::TokenStream, item: ItemFn) -> syn::Result<TypedGraph> {
    parse_graph_with_signatures(attr, item, &[])
}

pub fn parse_graph_with_signatures(
    attr: proc_macro2::TokenStream,
    item: ItemFn,
    graph_signatures: &[(String, GraphSignature)],
) -> syn::Result<TypedGraph> {
    let backend = parse_backend(attr)?;
    let graph = parse_item_fn(item, backend)?;
    type_check(graph, graph_signatures)
}

fn parse_backend(attr: proc_macro2::TokenStream) -> syn::Result<String> {
    let args = syn::punctuated::Punctuated::<MetaNameValue, syn::Token![,]>::parse_terminated
        .parse2(attr)?;
    for arg in args {
        if arg.path.is_ident("backend") {
            if let SynExpr::Lit(expr_lit) = &arg.value {
                if let Lit::Str(lit) = &expr_lit.lit {
                    return Ok(lit.value());
                }
            }
            return Err(syn::Error::new(
                arg.span(),
                "backend must be a string literal",
            ));
        }
        if arg.path.is_ident("backends") {
            return parse_first_backend_from_array(&arg.value);
        }
    }
    Err(syn::Error::new(
        Span::call_site(),
        "missing required backend = \"...\" argument",
    ))
}

fn parse_first_backend_from_array(value: &SynExpr) -> syn::Result<String> {
    let SynExpr::Array(array) = value else {
        return Err(syn::Error::new(
            value.span(),
            "backends must be an array of backend(...) declarations",
        ));
    };
    let Some(SynExpr::Call(call)) = array.elems.first() else {
        return Err(syn::Error::new(
            array.span(),
            "backends must contain at least one backend(...) declaration",
        ));
    };
    let SynExpr::Path(path) = call.func.as_ref() else {
        return Err(syn::Error::new(call.func.span(), "expected backend(...)"));
    };
    if !path.path.is_ident("backend") {
        return Err(syn::Error::new(call.func.span(), "expected backend(...)"));
    }
    let Some(SynExpr::Lit(expr_lit)) = call.args.first() else {
        return Err(syn::Error::new(call.span(), "backend name is required"));
    };
    let Lit::Str(lit) = &expr_lit.lit else {
        return Err(syn::Error::new(
            expr_lit.span(),
            "backend name must be a string literal",
        ));
    };
    Ok(lit.value())
}

fn parse_item_fn(item: ItemFn, backend: String) -> syn::Result<Graph> {
    reject_unsupported_attrs(&item.attrs)?;
    let name = item.sig.ident.to_string();
    let mut inputs = Vec::new();
    for input in &item.sig.inputs {
        inputs.push(parse_input(input)?);
    }
    let output = match &item.sig.output {
        ReturnType::Type(_, ty) => parse_tensor_type(ty)?,
        ReturnType::Default => {
            return Err(syn::Error::new(
                item.sig.ident.span(),
                "graph functions must return a Tensor type",
            ));
        }
    };

    let (lets, body) = parse_block(&item.block.stmts)?;
    Ok(Graph {
        name,
        backend,
        inputs,
        output,
        lets,
        body,
    })
}

fn reject_unsupported_attrs(attrs: &[Attribute]) -> syn::Result<()> {
    if let Some(attr) = attrs.first() {
        return Err(syn::Error::new(
            attr.span(),
            "attributes inside #[knok::graph] functions are not supported yet",
        ));
    }
    Ok(())
}

fn parse_input(input: &FnArg) -> syn::Result<Input> {
    let FnArg::Typed(pat_ty) = input else {
        return Err(syn::Error::new(
            input.span(),
            "graph methods with self receivers are not supported",
        ));
    };
    let Pat::Ident(PatIdent { ident, .. }) = pat_ty.pat.as_ref() else {
        return Err(syn::Error::new(
            pat_ty.pat.span(),
            "graph argument patterns must be simple identifiers",
        ));
    };
    Ok(Input {
        name: ident.to_string(),
        ty: parse_tensor_type(&pat_ty.ty)?,
    })
}

pub fn parse_tensor_type(ty: &Type) -> syn::Result<TensorType> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return Err(syn::Error::new(
            ty.span(),
            "expected Tensor1, Tensor2, Tensor3, or Tensor4 type",
        ));
    };
    let segment = path.segments.last().ok_or_else(|| {
        syn::Error::new(
            path.span(),
            "expected Tensor1, Tensor2, Tensor3, or Tensor4 type",
        )
    })?;
    let rank = match segment.ident.to_string().as_str() {
        "Tensor1" => 1,
        "Tensor2" => 2,
        "Tensor3" => 3,
        "Tensor4" => 4,
        _ => {
            return Err(syn::Error::new(
                segment.ident.span(),
                "expected Tensor1<T, D0>, Tensor2<T, D0, D1>, Tensor3<T, D0, D1, D2>, or Tensor4<T, D0, D1, D2, D3>",
            ));
        }
    };
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return Err(syn::Error::new(
            segment.arguments.span(),
            "tensor type requires generic arguments",
        ));
    };
    let mut args = args.args.iter();
    let elem = parse_element_type(args.next())?;
    let mut shape = Vec::new();
    for _ in 0..rank {
        shape.push(parse_const_usize(args.next())?);
    }
    if args.next().is_some() {
        return Err(syn::Error::new(
            segment.arguments.span(),
            "too many tensor generic arguments",
        ));
    }
    Ok(TensorType { elem, shape })
}

fn parse_element_type(arg: Option<&GenericArgument>) -> syn::Result<ElementType> {
    let Some(GenericArgument::Type(Type::Path(path))) = arg else {
        return Err(syn::Error::new(
            Span::call_site(),
            supported_element_type_message(),
        ));
    };
    let Some(segment) = path.path.segments.last() else {
        return Err(syn::Error::new(
            path.span(),
            supported_element_type_message(),
        ));
    };
    for (name, elem) in [
        ("bool", ElementType::Bool),
        ("f32", ElementType::F32),
        ("f64", ElementType::F64),
        #[cfg(feature = "half")]
        ("f16", ElementType::F16),
        #[cfg(feature = "half")]
        ("bf16", ElementType::BF16),
        ("i32", ElementType::I32),
        ("i64", ElementType::I64),
    ] {
        if segment.ident == name {
            return Ok(elem);
        }
    }
    Err(syn::Error::new(
        path.span(),
        supported_element_type_message(),
    ))
}

fn supported_element_type_message() -> &'static str {
    "unsupported tensor element type; supported types are bool, f32, f64, i32, i64, and half::f16/half::bf16 when the `half` feature is enabled"
}

fn parse_const_usize(arg: Option<&GenericArgument>) -> syn::Result<usize> {
    let Some(GenericArgument::Const(SynExpr::Lit(expr_lit))) = arg else {
        return Err(syn::Error::new(
            Span::call_site(),
            "tensor dimensions must be integer const arguments",
        ));
    };
    let Lit::Int(lit) = &expr_lit.lit else {
        return Err(syn::Error::new(
            expr_lit.span(),
            "tensor dimensions must be integer const arguments",
        ));
    };
    lit.base10_parse::<usize>()
}

fn parse_block(stmts: &[Stmt]) -> syn::Result<(Vec<Let>, Expr)> {
    let mut lets = Vec::new();
    let mut body = None;
    for stmt in stmts {
        match stmt {
            Stmt::Local(local) => {
                let Pat::Ident(PatIdent { ident, .. }) = &local.pat else {
                    return Err(syn::Error::new(
                        local.pat.span(),
                        "let bindings must use simple identifiers",
                    ));
                };
                let Some(init) = &local.init else {
                    return Err(syn::Error::new(
                        local.span(),
                        "let bindings must have initializers",
                    ));
                };
                lets.push(Let {
                    name: ident.to_string(),
                    value: parse_expr(&init.expr)?,
                });
            }
            Stmt::Expr(expr, None) => {
                body = Some(parse_expr(expr)?);
            }
            Stmt::Expr(_, Some(_)) => {
                return Err(syn::Error::new(
                    stmt.span(),
                    "only let bindings and a final expression are supported",
                ));
            }
            Stmt::Item(_) | Stmt::Macro(_) => {
                return Err(syn::Error::new(
                    stmt.span(),
                    "items and macros inside graph functions are not supported",
                ));
            }
        }
    }
    let body = body.ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            "graph functions must end with a tensor expression",
        )
    })?;
    Ok((lets, body))
}

fn parse_expr(expr: &SynExpr) -> syn::Result<Expr> {
    match expr {
        SynExpr::Path(path) => Ok(Expr::Var(path.path.require_ident()?.to_string())),
        SynExpr::Lit(expr_lit) => match &expr_lit.lit {
            Lit::Float(lit) => Ok(Expr::Const {
                value: lit.base10_digits().to_string(),
                elem: parse_float_literal_element(lit)?,
            }),
            Lit::Int(lit) => Ok(Expr::Const {
                value: lit.base10_digits().to_string(),
                elem: parse_int_literal_element(lit)?,
            }),
            Lit::Bool(lit) => Ok(Expr::Const {
                value: if lit.value() { "1" } else { "0" }.to_string(),
                elem: ElementType::Bool,
            }),
            _ => Err(syn::Error::new(
                expr_lit.span(),
                "expected numeric or bool literal",
            )),
        },
        SynExpr::Paren(paren) => parse_expr(&paren.expr),
        SynExpr::Unary(unary) => {
            let UnOp::Neg(_) = unary.op else {
                return Err(syn::Error::new(
                    unary.op.span(),
                    "only unary - is supported",
                ));
            };
            Ok(Expr::Unary {
                op: UnaryOp::Neg,
                value: Box::new(parse_expr(&unary.expr)?),
            })
        }
        SynExpr::Binary(binary) => Ok(Expr::Binary {
            op: parse_binary_op(&binary.op)?,
            lhs: Box::new(parse_expr(&binary.left)?),
            rhs: Box::new(parse_expr(&binary.right)?),
        }),
        SynExpr::Call(call) => {
            let SynExpr::Path(path) = call.func.as_ref() else {
                return Err(syn::Error::new(call.func.span(), "expected graph op name"));
            };
            let (op_name, generics) = parse_call_path(&path.path)?;
            let op = match op_name.as_str() {
                "abs" => {
                    reject_any_generics(&generics, &path.path, "abs")?;
                    CallOp::Abs
                }
                "all" => {
                    reject_target_type(&generics, &path.path, "all")?;
                    CallOp::All(optional_axis(&generics, &path.path, "all")?)
                }
                "argmax" => {
                    reject_any_generics(&generics, &path.path, "argmax")?;
                    CallOp::Argmax
                }
                "any" => {
                    reject_target_type(&generics, &path.path, "any")?;
                    CallOp::Any(optional_axis(&generics, &path.path, "any")?)
                }
                "clip" => {
                    reject_any_generics(&generics, &path.path, "clip")?;
                    CallOp::Clip
                }
                "concat" => {
                    reject_target_type(&generics, &path.path, "concat")?;
                    CallOp::Concat(expect_one_const(&generics, &path.path, "concat")?)
                }
                "conv2d" => {
                    reject_any_generics(&generics, &path.path, "conv2d")?;
                    CallOp::Conv2d
                }
                "exp" => {
                    reject_any_generics(&generics, &path.path, "exp")?;
                    CallOp::Exp
                }
                "greater" => {
                    reject_any_generics(&generics, &path.path, "greater")?;
                    CallOp::Greater
                }
                "greater_equal" => {
                    reject_any_generics(&generics, &path.path, "greater_equal")?;
                    CallOp::GreaterEqual
                }
                "isnan" => {
                    reject_any_generics(&generics, &path.path, "isnan")?;
                    CallOp::IsNan
                }
                "less" => {
                    reject_any_generics(&generics, &path.path, "less")?;
                    CallOp::Less
                }
                "less_equal" => {
                    reject_any_generics(&generics, &path.path, "less_equal")?;
                    CallOp::LessEqual
                }
                "log" => {
                    reject_any_generics(&generics, &path.path, "log")?;
                    CallOp::Log
                }
                "logical_and" => {
                    reject_any_generics(&generics, &path.path, "logical_and")?;
                    CallOp::LogicalAnd
                }
                "logical_not" => {
                    reject_any_generics(&generics, &path.path, "logical_not")?;
                    CallOp::LogicalNot
                }
                "logical_or" => {
                    reject_any_generics(&generics, &path.path, "logical_or")?;
                    CallOp::LogicalOr
                }
                "logical_xor" => {
                    reject_any_generics(&generics, &path.path, "logical_xor")?;
                    CallOp::LogicalXor
                }
                "matmul" => {
                    reject_any_generics(&generics, &path.path, "matmul")?;
                    CallOp::Matmul
                }
                "mean" => {
                    reject_target_type(&generics, &path.path, "mean")?;
                    CallOp::Mean(optional_axis(&generics, &path.path, "mean")?)
                }
                "minimum" => {
                    reject_any_generics(&generics, &path.path, "minimum")?;
                    CallOp::Minimum
                }
                "maximum" => {
                    reject_any_generics(&generics, &path.path, "maximum")?;
                    CallOp::Maximum
                }
                "pow" => {
                    reject_any_generics(&generics, &path.path, "pow")?;
                    CallOp::Pow
                }
                "equal" => {
                    reject_any_generics(&generics, &path.path, "equal")?;
                    CallOp::Equal
                }
                "not_equal" => {
                    reject_any_generics(&generics, &path.path, "not_equal")?;
                    CallOp::NotEqual
                }
                "relu" => {
                    reject_any_generics(&generics, &path.path, "relu")?;
                    CallOp::Relu
                }
                "reshape" => {
                    reject_consts(&generics, &path.path, "reshape")?;
                    CallOp::Reshape(expect_target_type(
                        generics.target_ty.clone(),
                        &path.path,
                        "reshape",
                    )?)
                }
                "broadcast" => {
                    reject_consts(&generics, &path.path, "broadcast")?;
                    CallOp::Broadcast(expect_target_type(
                        generics.target_ty.clone(),
                        &path.path,
                        "broadcast",
                    )?)
                }
                "sigmoid" => {
                    reject_any_generics(&generics, &path.path, "sigmoid")?;
                    CallOp::Sigmoid
                }
                "softmax" => {
                    reject_target_type(&generics, &path.path, "softmax")?;
                    CallOp::Softmax(optional_axis(&generics, &path.path, "softmax")?)
                }
                "slice" => {
                    let target =
                        expect_target_type(generics.target_ty.clone(), &path.path, "slice")?;
                    CallOp::Slice {
                        target,
                        starts: generics.consts.clone(),
                    }
                }
                "sqrt" => {
                    reject_any_generics(&generics, &path.path, "sqrt")?;
                    CallOp::Sqrt
                }
                "squeeze" => {
                    reject_consts(&generics, &path.path, "squeeze")?;
                    CallOp::Squeeze(expect_target_type(
                        generics.target_ty.clone(),
                        &path.path,
                        "squeeze",
                    )?)
                }
                "stack" => {
                    reject_target_type(&generics, &path.path, "stack")?;
                    CallOp::Stack(expect_one_const(&generics, &path.path, "stack")?)
                }
                "sum" => {
                    reject_target_type(&generics, &path.path, "sum")?;
                    CallOp::Sum(optional_axis(&generics, &path.path, "sum")?)
                }
                "tanh" => {
                    reject_any_generics(&generics, &path.path, "tanh")?;
                    CallOp::Tanh
                }
                "take" => {
                    reject_target_type(&generics, &path.path, "take")?;
                    let values = expect_const_count(&generics, &path.path, "take", 2)?;
                    CallOp::Take {
                        axis: values[0],
                        index: values[1],
                    }
                }
                "transpose" => {
                    reject_any_generics(&generics, &path.path, "transpose")?;
                    CallOp::Transpose
                }
                "where" | "r#where" => {
                    reject_any_generics(&generics, &path.path, "where")?;
                    CallOp::Where
                }
                "unsqueeze" => {
                    reject_consts(&generics, &path.path, "unsqueeze")?;
                    CallOp::Unsqueeze(expect_target_type(
                        generics.target_ty.clone(),
                        &path.path,
                        "unsqueeze",
                    )?)
                }
                _ => {
                    reject_any_generics(&generics, &path.path, &op_name)?;
                    CallOp::Graph(op_name)
                }
            };
            let args = call
                .args
                .iter()
                .map(parse_expr)
                .collect::<syn::Result<_>>()?;
            Ok(Expr::Call { op, args })
        }
        _ => Err(syn::Error::new(
            expr.span(),
            "unsupported graph expression syntax",
        )),
    }
}

#[derive(Clone, Debug, Default)]
struct CallGenerics {
    target_ty: Option<TensorType>,
    consts: Vec<usize>,
}

fn parse_call_path(path: &syn::Path) -> syn::Result<(String, CallGenerics)> {
    let Some(segment) = path.segments.last() else {
        return Err(syn::Error::new(path.span(), "expected graph op name"));
    };
    if path.segments.len() != 1 {
        return Err(syn::Error::new(
            path.span(),
            "graph op names must be unqualified identifiers",
        ));
    }
    let mut generics = CallGenerics::default();
    match &segment.arguments {
        syn::PathArguments::None => {}
        syn::PathArguments::AngleBracketed(args) => {
            if args.args.is_empty() {
                return Err(syn::Error::new(
                    segment.arguments.span(),
                    "missing generic argument",
                ));
            }
            for arg in &args.args {
                match arg {
                    GenericArgument::Type(ty) => {
                        if generics.target_ty.is_some() {
                            return Err(syn::Error::new(
                                arg.span(),
                                "graph ops accept at most one target tensor type",
                            ));
                        }
                        generics.target_ty = Some(parse_tensor_type(ty)?);
                    }
                    GenericArgument::Const(expr) => {
                        generics.consts.push(parse_generic_const_usize(expr)?);
                    }
                    _ => {
                        return Err(syn::Error::new(
                            arg.span(),
                            "graph op generic argument must be a tensor type or integer const",
                        ));
                    }
                }
            }
        }
        syn::PathArguments::Parenthesized(_) => {
            return Err(syn::Error::new(
                segment.arguments.span(),
                "parenthesized graph op arguments are not supported",
            ));
        }
    }
    Ok((segment.ident.to_string(), generics))
}

fn parse_generic_const_usize(expr: &SynExpr) -> syn::Result<usize> {
    let SynExpr::Lit(expr_lit) = expr else {
        return Err(syn::Error::new(
            expr.span(),
            "generic const argument must be an integer",
        ));
    };
    let Lit::Int(lit) = &expr_lit.lit else {
        return Err(syn::Error::new(
            expr_lit.span(),
            "generic const argument must be an integer",
        ));
    };
    lit.base10_parse::<usize>()
}

fn expect_target_type(
    target_ty: Option<TensorType>,
    path: &syn::Path,
    op_name: &str,
) -> syn::Result<TensorType> {
    target_ty.ok_or_else(|| {
        syn::Error::new(
            path.span(),
            format!("{op_name} requires a target tensor type, for example {op_name}::<Tensor1<f32, 4>>(x)"),
        )
    })
}

fn reject_target_type(generics: &CallGenerics, path: &syn::Path, op_name: &str) -> syn::Result<()> {
    if generics.target_ty.is_some() {
        Err(syn::Error::new(
            path.span(),
            format!("{op_name} does not accept a target tensor type"),
        ))
    } else {
        Ok(())
    }
}

fn reject_consts(generics: &CallGenerics, path: &syn::Path, op_name: &str) -> syn::Result<()> {
    if !generics.consts.is_empty() {
        Err(syn::Error::new(
            path.span(),
            format!("{op_name} does not accept const generic arguments"),
        ))
    } else {
        Ok(())
    }
}

fn reject_any_generics(
    generics: &CallGenerics,
    path: &syn::Path,
    op_name: &str,
) -> syn::Result<()> {
    if generics.target_ty.is_some() || !generics.consts.is_empty() {
        Err(syn::Error::new(
            path.span(),
            format!("graph call `{op_name}` does not accept generic arguments"),
        ))
    } else {
        Ok(())
    }
}

fn optional_axis(
    generics: &CallGenerics,
    path: &syn::Path,
    op_name: &str,
) -> syn::Result<Option<usize>> {
    match generics.consts.as_slice() {
        [] => Ok(None),
        [axis] => Ok(Some(*axis)),
        _ => Err(syn::Error::new(
            path.span(),
            format!("{op_name} accepts at most one axis const generic"),
        )),
    }
}

fn expect_one_const(
    generics: &CallGenerics,
    path: &syn::Path,
    op_name: &str,
) -> syn::Result<usize> {
    let values = expect_const_count(generics, path, op_name, 1)?;
    Ok(values[0])
}

fn expect_const_count<'a>(
    generics: &'a CallGenerics,
    path: &syn::Path,
    op_name: &str,
    expected: usize,
) -> syn::Result<&'a [usize]> {
    if generics.consts.len() == expected {
        Ok(&generics.consts)
    } else {
        Err(syn::Error::new(
            path.span(),
            format!(
                "{op_name} expects {expected} const generic arguments, got {}",
                generics.consts.len()
            ),
        ))
    }
}

fn parse_float_literal_element(lit: &syn::LitFloat) -> syn::Result<ElementType> {
    match lit.suffix() {
        "" | "f32" => Ok(ElementType::F32),
        "f64" => Ok(ElementType::F64),
        suffix => Err(syn::Error::new(
            lit.span(),
            format!("unsupported float literal suffix `{suffix}`; expected f32 or f64"),
        )),
    }
}

fn parse_int_literal_element(lit: &syn::LitInt) -> syn::Result<ElementType> {
    match lit.suffix() {
        "" | "i32" => Ok(ElementType::I32),
        "i64" => Ok(ElementType::I64),
        suffix => Err(syn::Error::new(
            lit.span(),
            format!("unsupported integer literal suffix `{suffix}`; expected i32 or i64"),
        )),
    }
}

fn parse_binary_op(op: &BinOp) -> syn::Result<BinaryOp> {
    match op {
        BinOp::Add(_) => Ok(BinaryOp::Add),
        BinOp::Sub(_) => Ok(BinaryOp::Sub),
        BinOp::Mul(_) => Ok(BinaryOp::Mul),
        BinOp::Div(_) => Ok(BinaryOp::Div),
        _ => Err(syn::Error::new(op.span(), "unsupported binary operator")),
    }
}

pub fn type_check(
    graph: Graph,
    graph_signatures: &[(String, GraphSignature)],
) -> syn::Result<TypedGraph> {
    let mut env = graph
        .inputs
        .iter()
        .map(|input| (input.name.clone(), input.ty.clone()))
        .collect::<Vec<_>>();
    let mut lets = Vec::new();
    for binding in graph.lets {
        let value = type_expr(&binding.value, &env, graph_signatures, &graph.name)?;
        env.push((binding.name.clone(), value.ty.clone()));
        lets.push(TypedLet {
            name: binding.name,
            value,
        });
    }
    let body = type_expr(&graph.body, &env, graph_signatures, &graph.name)?;
    if body.ty != graph.output {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "return type mismatch: inferred {:?}, declared {:?}",
                body.ty, graph.output
            ),
        ));
    }
    Ok(TypedGraph {
        name: graph.name,
        backend: graph.backend,
        inputs: graph.inputs,
        output: graph.output,
        lets,
        body,
    })
}

fn type_expr(
    expr: &Expr,
    env: &[(String, TensorType)],
    graph_signatures: &[(String, GraphSignature)],
    current_graph: &str,
) -> syn::Result<TypedExpr> {
    let ty = match expr {
        Expr::Var(name) => env
            .iter()
            .rev()
            .find_map(|(candidate, ty)| (candidate == name).then(|| ty.clone()))
            .ok_or_else(|| syn::Error::new(Span::call_site(), format!("unknown value `{name}`")))?,
        Expr::Const { elem, .. } => TensorType {
            elem: *elem,
            shape: vec![],
        },
        Expr::Unary { value, .. } => type_expr(value, env, graph_signatures, current_graph)?.ty,
        Expr::Binary { lhs, rhs, .. } => {
            let lhs = type_expr(lhs, env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(rhs, env, graph_signatures, current_graph)?.ty;
            binary_result_type(&lhs, &rhs)?
        }
        Expr::Call { op, args } => {
            call_result_type(op, args, env, graph_signatures, current_graph)?
        }
    };
    Ok(TypedExpr {
        kind: expr.clone(),
        ty,
    })
}

fn binary_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_numeric_element(lhs.elem, "arithmetic operators")?;
    expect_numeric_element(rhs.elem, "arithmetic operators")?;
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "elementwise operands must have the same element type, got {lhs:?} and {rhs:?}"
            ),
        ));
    }
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: lhs.elem,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("elementwise operands are not broadcast-compatible: {message}"),
            )
        })
}

fn comparison_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_numeric_element(lhs.elem, "comparison ops")?;
    expect_numeric_element(rhs.elem, "comparison ops")?;
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("comparison operands must have the same element type, got {lhs:?} and {rhs:?}"),
        ));
    }
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: ElementType::Bool,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("comparison operands are not broadcast-compatible: {message}"),
            )
        })
}

fn logical_result_type(lhs: &TensorType, rhs: &TensorType) -> syn::Result<TensorType> {
    expect_bool_element(lhs.elem, "logical ops")?;
    expect_bool_element(rhs.elem, "logical ops")?;
    broadcast_shape(lhs, rhs)
        .map(|shape| TensorType {
            elem: ElementType::Bool,
            shape,
        })
        .map_err(|message| {
            syn::Error::new(
                Span::call_site(),
                format!("logical operands are not broadcast-compatible: {message}"),
            )
        })
}

fn where_result_type(
    condition: &TensorType,
    lhs: &TensorType,
    rhs: &TensorType,
) -> syn::Result<TensorType> {
    expect_bool_element(condition.elem, "where condition")?;
    if lhs.elem != rhs.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "where value operands must have the same element type, got {lhs:?} and {rhs:?}"
            ),
        ));
    }
    let value_shape = broadcast_shape(lhs, rhs).map_err(|message| {
        syn::Error::new(
            Span::call_site(),
            format!("where value operands are not broadcast-compatible: {message}"),
        )
    })?;
    let result = TensorType {
        elem: lhs.elem,
        shape: value_shape,
    };
    broadcast_shape(condition, &result).map_err(|message| {
        syn::Error::new(
            Span::call_site(),
            format!("where condition is not broadcast-compatible with values: {message}"),
        )
    })?;
    Ok(result)
}

fn call_result_type(
    op: &CallOp,
    args: &[Expr],
    env: &[(String, TensorType)],
    graph_signatures: &[(String, GraphSignature)],
    current_graph: &str,
) -> syn::Result<TensorType> {
    match op {
        CallOp::Abs => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(ty.elem, "abs")?;
            Ok(ty)
        }
        CallOp::All(axis) | CallOp::Any(axis) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_bool_element(input.elem, "bool reductions")?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "bool reductions expect a tensor input",
                ));
            }
            reduction_output_type(&input, *axis)
        }
        CallOp::Argmax => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, input.elem)?;
            if input.rank() != 1 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "argmax currently supports rank-1 tensors only",
                ));
            }
            Ok(TensorType {
                elem: input.elem,
                shape: vec![1],
            })
        }
        CallOp::Exp
        | CallOp::IsNan
        | CallOp::Log
        | CallOp::Relu
        | CallOp::Sigmoid
        | CallOp::Sqrt
        | CallOp::Tanh => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, ty.elem)?;
            if matches!(op, CallOp::IsNan) {
                Ok(TensorType {
                    elem: ElementType::Bool,
                    shape: ty.shape,
                })
            } else {
                Ok(ty)
            }
        }
        CallOp::Greater
        | CallOp::GreaterEqual
        | CallOp::Less
        | CallOp::LessEqual
        | CallOp::Equal
        | CallOp::NotEqual => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            comparison_result_type(&lhs, &rhs)
        }
        CallOp::LogicalAnd | CallOp::LogicalOr | CallOp::LogicalXor => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            logical_result_type(&lhs, &rhs)
        }
        CallOp::LogicalNot => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_bool_element(ty.elem, "logical_not")?;
            Ok(ty)
        }
        CallOp::Minimum | CallOp::Maximum => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(lhs.elem, "min/max ops")?;
            expect_numeric_element(rhs.elem, "min/max ops")?;
            binary_result_type(&lhs, &rhs)
        }
        CallOp::Clip => {
            expect_arity(op, args, 3)?;
            let value = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let min = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            let max = type_expr(&args[2], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(value.elem, "clip")?;
            expect_numeric_element(min.elem, "clip")?;
            expect_numeric_element(max.elem, "clip")?;
            let value = binary_result_type(&value, &min)?;
            binary_result_type(&value, &max)
        }
        CallOp::Pow => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            expect_float(op, lhs.elem)?;
            expect_float(op, rhs.elem)?;
            binary_result_type(&lhs, &rhs)
        }
        CallOp::Concat(axis) => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            concat_result_type(&lhs, &rhs, *axis)
        }
        CallOp::Softmax(axis) => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, ty.elem)?;
            if let Some(axis) = axis {
                expect_axis(&ty, *axis)?;
            }
            Ok(ty)
        }
        CallOp::Transpose => {
            expect_arity(op, args, 1)?;
            let mut ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            if ty.rank() != 2 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "transpose currently supports rank-2 tensors only",
                ));
            }
            ty.shape.swap(0, 1);
            Ok(ty)
        }
        CallOp::Reshape(target) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            if input.elem != target.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "reshape input and output element types must match",
                ));
            }
            if element_count(&input) != element_count(target) {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "reshape element counts must match, got {} and {}",
                        element_count(&input),
                        element_count(target)
                    ),
                ));
            }
            Ok(target.clone())
        }
        CallOp::Broadcast(target) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            if input.elem != target.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "broadcast input and output element types must match",
                ));
            }
            if let Err(message) = broadcast_shape(&input, target) {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!("broadcast input and output shapes are incompatible: {message}"),
                ));
            }
            Ok(target.clone())
        }
        CallOp::Slice { target, starts } => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            validate_slice(&input, target, starts)?;
            Ok(target.clone())
        }
        CallOp::Squeeze(target) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            validate_squeeze(&input, target)?;
            Ok(target.clone())
        }
        CallOp::Stack(axis) => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            stack_result_type(&lhs, &rhs, *axis)
        }
        CallOp::Sum(axis) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_numeric_element(input.elem, "sum")?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "sum expects a tensor input",
                ));
            }
            Ok(reduction_output_type(&input, *axis)?)
        }
        CallOp::Mean(axis) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, input.elem)?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "mean expects a tensor input",
                ));
            }
            Ok(reduction_output_type(&input, *axis)?)
        }
        CallOp::Take { axis, index } => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            take_result_type(&input, *axis, *index)
        }
        CallOp::Matmul => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            if lhs.elem != rhs.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "matmul expects operands with the same element type",
                ));
            }
            expect_numeric_element(lhs.elem, "matmul")?;
            match (lhs.rank(), rhs.rank()) {
                (2, 2) => {
                    if lhs.shape[1] != rhs.shape[0] {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            format!(
                                "matmul inner dimensions must match, got {} and {}",
                                lhs.shape[1], rhs.shape[0]
                            ),
                        ));
                    }
                    Ok(TensorType {
                        elem: lhs.elem,
                        shape: vec![lhs.shape[0], rhs.shape[1]],
                    })
                }
                (3, 3) => {
                    if lhs.shape[0] != rhs.shape[0] {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            format!(
                                "batched matmul batch dimensions must match, got {} and {}",
                                lhs.shape[0], rhs.shape[0]
                            ),
                        ));
                    }
                    if lhs.shape[2] != rhs.shape[1] {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            format!(
                                "batched matmul inner dimensions must match, got {} and {}",
                                lhs.shape[2], rhs.shape[1]
                            ),
                        ));
                    }
                    Ok(TensorType {
                        elem: lhs.elem,
                        shape: vec![lhs.shape[0], lhs.shape[1], rhs.shape[2]],
                    })
                }
                _ => Err(syn::Error::new(
                    Span::call_site(),
                    "matmul expects rank-2 operands or rank-3 batched operands",
                )),
            }
        }
        CallOp::Conv2d => {
            expect_arity(op, args, 2)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let kernel = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            if input.rank() != 4 || kernel.rank() != 4 || input.elem != kernel.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "conv2d expects NHWC input and HWCF kernel rank-4 tensors with the same element type",
                ));
            }
            expect_numeric_element(input.elem, "conv2d")?;
            if input.shape[3] != kernel.shape[2] {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "conv2d input channels must match kernel channels, got {} and {}",
                        input.shape[3], kernel.shape[2]
                    ),
                ));
            }
            if input.shape[1] < kernel.shape[0] || input.shape[2] < kernel.shape[1] {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "conv2d kernel spatial dimensions must fit inside the input",
                ));
            }
            Ok(TensorType {
                elem: input.elem,
                shape: vec![
                    input.shape[0],
                    input.shape[1] - kernel.shape[0] + 1,
                    input.shape[2] - kernel.shape[1] + 1,
                    kernel.shape[3],
                ],
            })
        }
        CallOp::Unsqueeze(target) => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            validate_unsqueeze(&input, target)?;
            Ok(target.clone())
        }
        CallOp::Where => {
            expect_arity(op, args, 3)?;
            let condition = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let lhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[2], env, graph_signatures, current_graph)?.ty;
            where_result_type(&condition, &lhs, &rhs)
        }
        CallOp::Graph(name) => {
            if name == current_graph {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!("recursive graph call `{name}` is not supported"),
                ));
            }
            let signature = graph_signatures
                .iter()
                .rev()
                .find_map(|(candidate, signature)| (candidate == name).then_some(signature))
                .ok_or_else(|| {
                    syn::Error::new(
                        Span::call_site(),
                        format!(
                            "unknown graph call `{name}`; graph calls must refer to earlier #[knok::graph] functions"
                        ),
                    )
                })?;
            if args.len() != signature.inputs.len() {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "graph `{name}` expects {} arguments, got {}",
                        signature.inputs.len(),
                        args.len()
                    ),
                ));
            }
            for (index, (arg, expected)) in args.iter().zip(&signature.inputs).enumerate() {
                let actual = type_expr(arg, env, graph_signatures, current_graph)?.ty;
                if &actual != expected {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        format!(
                            "graph `{name}` argument {index} type mismatch: expected {expected:?}, got {actual:?}"
                        ),
                    ));
                }
            }
            Ok(signature.output.clone())
        }
    }
}

fn element_count(ty: &TensorType) -> usize {
    ty.shape.iter().product()
}

fn broadcast_shape(lhs: &TensorType, rhs: &TensorType) -> Result<Vec<usize>, String> {
    let rank = lhs.rank().max(rhs.rank());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let lhs_dim = dim_from_trailing(&lhs.shape, rank, offset);
        let rhs_dim = dim_from_trailing(&rhs.shape, rank, offset);
        let dim = match (lhs_dim, rhs_dim) {
            (Some(lhs_dim), Some(rhs_dim)) if lhs_dim == rhs_dim => lhs_dim,
            (Some(1), Some(rhs_dim)) => rhs_dim,
            (Some(lhs_dim), Some(1)) => lhs_dim,
            (None, Some(dim)) | (Some(dim), None) => dim,
            (None, None) => unreachable!("rank is derived from at least one shape"),
            (Some(lhs_dim), Some(rhs_dim)) => {
                return Err(format!(
                    "dimension {} differs: {} vs {}",
                    offset, lhs_dim, rhs_dim
                ));
            }
        };
        shape.push(dim);
    }
    Ok(shape)
}

fn dim_from_trailing(shape: &[usize], rank: usize, offset: usize) -> Option<usize> {
    let pad = rank - shape.len();
    (offset >= pad).then(|| shape[offset - pad])
}

fn validate_slice(input: &TensorType, target: &TensorType, starts: &[usize]) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "slice input and output element types must match",
        ));
    }
    if input.rank() != target.rank() || starts.len() != input.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "slice expects one start const per dimension and equal input/output rank, got input rank {}, output rank {}, starts {:?}",
                input.rank(),
                target.rank(),
                starts
            ),
        ));
    }
    for (axis, ((start, size), input_dim)) in starts
        .iter()
        .zip(&target.shape)
        .zip(&input.shape)
        .enumerate()
    {
        if *start + *size > *input_dim {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "slice dimension {axis} is out of bounds: start {start} + size {size} exceeds input dimension {input_dim}",
                ),
            ));
        }
    }
    Ok(())
}

fn take_result_type(input: &TensorType, axis: usize, index: usize) -> syn::Result<TensorType> {
    expect_axis(input, axis)?;
    if index >= input.shape[axis] {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "take index {index} is out of bounds for axis {axis} with dimension {}",
                input.shape[axis]
            ),
        ));
    }
    let mut shape = input.shape.clone();
    shape.remove(axis);
    if shape.is_empty() {
        shape.push(1);
    }
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn validate_squeeze(input: &TensorType, target: &TensorType) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "squeeze input and output element types must match",
        ));
    }
    let squeezed = input
        .shape
        .iter()
        .copied()
        .filter(|dim| *dim != 1)
        .collect::<Vec<_>>();
    let squeezed = if squeezed.is_empty() {
        vec![1]
    } else {
        squeezed
    };
    if squeezed == target.shape {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "squeeze target shape {:?} does not match input shape {:?} after removing singleton dimensions",
                target.shape, input.shape
            ),
        ))
    }
}

fn validate_unsqueeze(input: &TensorType, target: &TensorType) -> syn::Result<()> {
    if input.elem != target.elem {
        return Err(syn::Error::new(
            Span::call_site(),
            "unsqueeze input and output element types must match",
        ));
    }
    let target_without_singletons = target
        .shape
        .iter()
        .copied()
        .filter(|dim| *dim != 1)
        .collect::<Vec<_>>();
    let input_without_rank1_scalar = if input.shape == [1] {
        Vec::new()
    } else {
        input.shape.clone()
    };
    if target_without_singletons == input.shape
        || target_without_singletons == input_without_rank1_scalar
    {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "unsqueeze target shape {:?} must insert only singleton dimensions into input shape {:?}",
                target.shape, input.shape
            ),
        ))
    }
}

fn concat_result_type(lhs: &TensorType, rhs: &TensorType, axis: usize) -> syn::Result<TensorType> {
    if lhs.elem != rhs.elem || lhs.rank() != rhs.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            "concat expects tensors with the same element type and rank",
        ));
    }
    expect_axis(lhs, axis)?;
    let mut shape = lhs.shape.clone();
    for dim in 0..lhs.rank() {
        if dim == axis {
            shape[dim] = lhs.shape[dim] + rhs.shape[dim];
        } else if lhs.shape[dim] != rhs.shape[dim] {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "concat dimension {dim} must match outside axis {axis}, got {} and {}",
                    lhs.shape[dim], rhs.shape[dim]
                ),
            ));
        }
    }
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn stack_result_type(lhs: &TensorType, rhs: &TensorType, axis: usize) -> syn::Result<TensorType> {
    if lhs != rhs {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("stack expects matching tensor types, got {lhs:?} and {rhs:?}"),
        ));
    }
    if axis > lhs.rank() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "stack axis {axis} is out of bounds for rank-{} tensor",
                lhs.rank()
            ),
        ));
    }
    let mut shape = lhs.shape.clone();
    shape.insert(axis, 2);
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

fn reduction_output_type(input: &TensorType, axis: Option<usize>) -> syn::Result<TensorType> {
    let shape = if let Some(axis) = axis {
        expect_axis(input, axis)?;
        let mut shape = input.shape.clone();
        shape.remove(axis);
        if shape.is_empty() {
            vec![1]
        } else {
            shape
        }
    } else {
        vec![1]
    };
    Ok(TensorType {
        elem: input.elem,
        shape,
    })
}

fn expect_axis(input: &TensorType, axis: usize) -> syn::Result<()> {
    if axis < input.rank() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!(
                "axis {axis} is out of bounds for rank-{} tensor {:?}",
                input.rank(),
                input.shape
            ),
        ))
    }
}

fn expect_arity(op: &CallOp, args: &[Expr], expected: usize) -> syn::Result<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op:?} expects {expected} arguments, got {}", args.len()),
        ))
    }
}

fn expect_float(op: &CallOp, elem: ElementType) -> syn::Result<()> {
    if elem.is_float() {
        Ok(())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op:?} supports floating-point tensors only"),
        ))
    }
}

fn expect_numeric_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
    if elem.is_numeric() {
        Ok(())
    } else {
        let verb = if op_name.ends_with('s') {
            "support"
        } else {
            "supports"
        };
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} {verb} numeric tensors only"),
        ))
    }
}

fn expect_bool_element(elem: ElementType, op_name: &str) -> syn::Result<()> {
    if elem.is_bool() {
        Ok(())
    } else {
        let verb = if op_name.ends_with('s') {
            "support"
        } else {
            "supports"
        };
        Err(syn::Error::new(
            Span::call_site(),
            format!("{op_name} {verb} bool tensors only"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse_quote;

    fn tensor(shape: &[usize]) -> TensorType {
        TensorType {
            elem: ElementType::F32,
            shape: shape.to_vec(),
        }
    }

    fn parse(item: ItemFn) -> syn::Result<TypedGraph> {
        parse_graph(quote!(backend = "llvm-cpu"), item)
    }

    #[test]
    fn parses_and_types_elementwise_graph() {
        let graph = parse(parse_quote! {
            fn add(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                x + y
            }
        })
        .unwrap();

        assert_eq!(graph.name, "add");
        assert_eq!(graph.backend, "llvm-cpu");
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.output, tensor(&[4]));
        assert_eq!(graph.body.ty, tensor(&[4]));
    }

    #[cfg(feature = "half")]
    #[test]
    fn parses_half_element_types() {
        let f16_graph = parse(parse_quote! {
            fn add(x: Tensor1<half::f16, 4>, y: Tensor1<half::f16, 4>) -> Tensor1<half::f16, 4> {
                x + y
            }
        })
        .unwrap();
        assert_eq!(f16_graph.output.elem, ElementType::F16);

        let bf16_graph = parse(parse_quote! {
            fn identity(x: Tensor1<knok::half::bf16, 4>) -> Tensor1<knok::half::bf16, 4> {
                x
            }
        })
        .unwrap();
        assert_eq!(bf16_graph.output.elem, ElementType::BF16);
    }

    #[test]
    fn infers_broadcast_elementwise_graph() {
        let graph = parse(parse_quote! {
            fn add_bias(x: Tensor2<f32, 2, 3>, bias: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
                x + bias
            }
        })
        .unwrap();

        assert_eq!(graph.body.ty, tensor(&[2, 3]));
    }

    #[test]
    fn infers_matmul_shape() {
        let graph = parse(parse_quote! {
            fn mm(x: Tensor2<f32, 2, 3>, y: Tensor2<f32, 3, 4>) -> Tensor2<f32, 2, 4> {
                matmul(x, y)
            }
        })
        .unwrap();

        assert_eq!(graph.output, tensor(&[2, 4]));
        assert_eq!(graph.body.ty, tensor(&[2, 4]));
    }

    #[test]
    fn infers_reshape_broadcast_and_sum_shapes() {
        let reshape = parse(parse_quote! {
            fn reshape4(x: Tensor1<f32, 4>) -> Tensor2<f32, 2, 2> {
                reshape::<Tensor2<f32, 2, 2>>(x)
            }
        })
        .unwrap();
        assert_eq!(reshape.body.ty, tensor(&[2, 2]));

        let broadcast = parse(parse_quote! {
            fn broadcast4(x: Tensor1<f32, 1>) -> Tensor1<f32, 4> {
                broadcast::<Tensor1<f32, 4>>(x)
            }
        })
        .unwrap();
        assert_eq!(broadcast.body.ty, tensor(&[4]));

        let sum = parse(parse_quote! {
            fn sum4(x: Tensor1<f32, 4>) -> Tensor1<f32, 1> {
                sum(x)
            }
        })
        .unwrap();
        assert_eq!(sum.body.ty, tensor(&[1]));

        let axis_sum = parse(parse_quote! {
            fn sum_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
                sum::<1>(x)
            }
        })
        .unwrap();
        assert_eq!(axis_sum.body.ty, tensor(&[2]));

        let axis_mean = parse(parse_quote! {
            fn mean_axis0(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 3> {
                mean::<0>(x)
            }
        })
        .unwrap();
        assert_eq!(axis_mean.body.ty, tensor(&[3]));
    }

    #[test]
    fn infers_static_shape_and_indexing_ops() {
        for item in [
            parse_quote! {
                fn slice_mid(x: Tensor2<f32, 2, 4>) -> Tensor2<f32, 2, 2> {
                    slice::<Tensor2<f32, 2, 2>, 0, 1>(x)
                }
            },
            parse_quote! {
                fn take_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
                    take::<1, 2>(x)
                }
            },
            parse_quote! {
                fn squeeze4(x: Tensor4<f32, 1, 2, 1, 3>) -> Tensor2<f32, 2, 3> {
                    squeeze::<Tensor2<f32, 2, 3>>(x)
                }
            },
            parse_quote! {
                fn unsqueeze2(x: Tensor2<f32, 2, 3>) -> Tensor4<f32, 1, 2, 1, 3> {
                    unsqueeze::<Tensor4<f32, 1, 2, 1, 3>>(x)
                }
            },
            parse_quote! {
                fn concat_axis0(x: Tensor2<f32, 1, 3>, y: Tensor2<f32, 2, 3>) -> Tensor2<f32, 3, 3> {
                    concat::<0>(x, y)
                }
            },
            parse_quote! {
                fn stack_axis0(x: Tensor1<f32, 3>, y: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
                    stack::<0>(x, y)
                }
            },
        ] {
            let graph = parse(item).unwrap();
            assert_eq!(graph.body.ty, graph.output);
        }
    }

    #[test]
    fn parses_higher_rank_tensors_and_infers_inference_ops() {
        let reshape = parse(parse_quote! {
            fn reshape8(x: Tensor1<f32, 8>) -> Tensor3<f32, 2, 2, 2> {
                reshape::<Tensor3<f32, 2, 2, 2>>(x)
            }
        })
        .unwrap();
        assert_eq!(reshape.body.ty, tensor(&[2, 2, 2]));

        let batch_mm = parse(parse_quote! {
            fn batch_mm(x: Tensor3<f32, 1, 2, 3>, y: Tensor3<f32, 1, 3, 2>) -> Tensor3<f32, 1, 2, 2> {
                matmul(x, y)
            }
        })
        .unwrap();
        assert_eq!(batch_mm.body.ty, tensor(&[1, 2, 2]));

        let conv = parse(parse_quote! {
            fn conv(x: Tensor4<f32, 1, 4, 4, 3>, k: Tensor4<f32, 3, 3, 3, 8>) -> Tensor4<f32, 1, 2, 2, 8> {
                conv2d(x, k)
            }
        })
        .unwrap();
        assert_eq!(conv.body.ty, tensor(&[1, 2, 2, 8]));
    }

    #[test]
    fn infers_scalar_classifier_op_shapes() {
        for item in [
            parse_quote! {
                fn abs4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { abs(x) }
            },
            parse_quote! {
                fn exp4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { exp(x) }
            },
            parse_quote! {
                fn log4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { log(x) }
            },
            parse_quote! {
                fn sqrt4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { sqrt(x) }
            },
            parse_quote! {
                fn tanh4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { tanh(x) }
            },
            parse_quote! {
                fn sigmoid4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { sigmoid(x) }
            },
            parse_quote! {
                fn softmax4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { softmax(x) }
            },
            parse_quote! {
                fn softmax_axis1(x: Tensor2<f32, 2, 3>) -> Tensor2<f32, 2, 3> { softmax::<1>(x) }
            },
        ] {
            let graph = parse(item).unwrap();
            assert_eq!(graph.body.ty, graph.output);
        }

        let mean = parse(parse_quote! {
            fn mean4(x: Tensor1<f32, 4>) -> Tensor1<f32, 1> {
                mean(x)
            }
        })
        .unwrap();
        assert_eq!(mean.body.ty, tensor(&[1]));

        let argmax = parse(parse_quote! {
            fn argmax4(x: Tensor1<f32, 4>) -> Tensor1<f32, 1> {
                argmax(x)
            }
        })
        .unwrap();
        assert_eq!(argmax.body.ty, tensor(&[1]));
    }

    #[test]
    fn infers_elementwise_call_shapes() {
        for item in [
            parse_quote! {
                fn minimum4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                    minimum(x, y)
                }
            },
            parse_quote! {
                fn maximum_broadcast(x: Tensor2<f32, 2, 3>, y: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
                    maximum(x, y)
                }
            },
            parse_quote! {
                fn clip4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                    clip(x, 0.0, 1.0)
                }
            },
            parse_quote! {
                fn pow4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                    pow(x, y)
                }
            },
        ] {
            let graph = parse(item).unwrap();
            assert_eq!(graph.body.ty, graph.output);
        }
    }

    #[test]
    fn infers_bool_predicate_selection_and_reduction_shapes() {
        let bool_tensor = |shape: &[usize]| TensorType {
            elem: ElementType::Bool,
            shape: shape.to_vec(),
        };

        for item in [
            parse_quote! {
                fn greater4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
                    greater(x, y)
                }
            },
            parse_quote! {
                fn logical4(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
                    logical_xor(logical_and(x, y), logical_not(y))
                }
            },
            parse_quote! {
                fn any_axis1(x: Tensor2<bool, 2, 3>) -> Tensor1<bool, 2> {
                    any::<1>(x)
                }
            },
            parse_quote! {
                fn all4(x: Tensor1<bool, 4>) -> Tensor1<bool, 1> {
                    all(x)
                }
            },
            parse_quote! {
                fn isnan4(x: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
                    isnan(x)
                }
            },
        ] {
            let graph = parse(item).unwrap();
            assert_eq!(graph.body.ty, graph.output);
        }

        let selected = parse(parse_quote! {
            fn select4(
                c: Tensor1<bool, 4>,
                x: Tensor1<f32, 4>,
                y: Tensor1<f32, 1>,
            ) -> Tensor1<f32, 4> {
                r#where(c, x, y)
            }
        })
        .unwrap();
        assert_eq!(selected.body.ty, tensor(&[4]));

        let comparison = parse(parse_quote! {
            fn less_broadcast(x: Tensor2<i32, 2, 3>, y: Tensor1<i32, 3>) -> Tensor2<bool, 2, 3> {
                less_equal(x, y)
            }
        })
        .unwrap();
        assert_eq!(comparison.body.ty, bool_tensor(&[2, 3]));
    }

    #[test]
    fn accepts_calls_to_earlier_graph_signatures() {
        let signatures = [(
            "layer".to_string(),
            GraphSignature {
                inputs: vec![tensor(&[4])],
                output: tensor(&[4]),
            },
        )];

        let graph = parse_graph_with_signatures(
            quote!(backend = "llvm-cpu"),
            parse_quote! {
                fn outer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                    layer(x)
                }
            },
            &signatures,
        )
        .unwrap();

        assert_eq!(graph.body.ty, tensor(&[4]));
    }

    #[test]
    fn rejects_elementwise_shape_mismatch() {
        let error = parse(parse_quote! {
            fn add(x: Tensor1<f32, 4>, y: Tensor1<f32, 5>) -> Tensor1<f32, 4> {
                x + y
            }
        })
        .unwrap_err();

        assert!(error.to_string().contains("not broadcast-compatible"));
    }

    #[test]
    fn rejects_invalid_reshape_and_broadcast_shapes() {
        let reshape = parse(parse_quote! {
            fn bad_reshape(x: Tensor1<f32, 4>) -> Tensor2<f32, 3, 2> {
                reshape::<Tensor2<f32, 3, 2>>(x)
            }
        })
        .unwrap_err();
        assert!(reshape.to_string().contains("element counts must match"));

        let broadcast = parse(parse_quote! {
            fn bad_broadcast(x: Tensor1<f32, 2>) -> Tensor1<f32, 4> {
                broadcast::<Tensor1<f32, 4>>(x)
            }
        })
        .unwrap_err();
        assert!(broadcast.to_string().contains("incompatible"));

        let slice = parse(parse_quote! {
            fn bad_slice(x: Tensor2<f32, 2, 4>) -> Tensor2<f32, 2, 3> {
                slice::<Tensor2<f32, 2, 3>, 0, 2>(x)
            }
        })
        .unwrap_err();
        assert!(slice.to_string().contains("out of bounds"));

        let take = parse(parse_quote! {
            fn bad_take(x: Tensor2<f32, 2, 4>) -> Tensor1<f32, 2> {
                take::<1, 4>(x)
            }
        })
        .unwrap_err();
        assert!(take.to_string().contains("out of bounds"));
    }

    #[test]
    fn rejects_unknown_graph_calls() {
        let error = parse(parse_quote! {
            fn outer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                missing(x)
            }
        })
        .unwrap_err();

        assert!(error.to_string().contains("unknown graph call `missing`"));
    }

    #[test]
    fn rejects_direct_recursion() {
        let signatures = [(
            "outer".to_string(),
            GraphSignature {
                inputs: vec![tensor(&[4])],
                output: tensor(&[4]),
            },
        )];
        let error = parse_graph_with_signatures(
            quote!(backend = "llvm-cpu"),
            parse_quote! {
                fn outer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                    outer(x)
                }
            },
            &signatures,
        )
        .unwrap_err();

        assert!(error.to_string().contains("recursive graph call `outer`"));
    }
}
