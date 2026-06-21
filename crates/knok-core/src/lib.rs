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
    Argmax,
    Conv2d,
    Exp,
    Log,
    Matmul,
    Mean,
    Relu,
    Reshape(TensorType),
    Broadcast(TensorType),
    Sigmoid,
    Softmax,
    Sqrt,
    Sum,
    Tanh,
    Transpose,
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
    F32,
    F64,
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
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    pub fn zero_literal(self) -> &'static str {
        match self {
            Self::F32 | Self::F64 => "0.0",
            Self::I32 | Self::I64 => "0",
        }
    }

    pub fn one_literal(self) -> &'static str {
        match self {
            Self::F32 | Self::F64 => "1.0",
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
            "tensor element type must be f32, f64, i32, or i64",
        ));
    };
    for (name, elem) in [
        ("f32", ElementType::F32),
        ("f64", ElementType::F64),
        ("i32", ElementType::I32),
        ("i64", ElementType::I64),
    ] {
        if path.path.is_ident(name) {
            return Ok(elem);
        }
    }
    Err(syn::Error::new(
        path.span(),
        "only f32, f64, i32, and i64 tensors are supported; f16/bf16 and quantized integer types are not supported yet",
    ))
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
            _ => Err(syn::Error::new(expr_lit.span(), "expected numeric literal")),
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
            let (op_name, target_ty) = parse_call_path(&path.path)?;
            let op = match op_name.as_str() {
                "argmax" => CallOp::Argmax,
                "conv2d" => CallOp::Conv2d,
                "exp" => CallOp::Exp,
                "log" => CallOp::Log,
                "matmul" => CallOp::Matmul,
                "mean" => CallOp::Mean,
                "relu" => CallOp::Relu,
                "reshape" => CallOp::Reshape(expect_target_type(target_ty, &path.path, "reshape")?),
                "broadcast" => {
                    CallOp::Broadcast(expect_target_type(target_ty, &path.path, "broadcast")?)
                }
                "sigmoid" => CallOp::Sigmoid,
                "softmax" => CallOp::Softmax,
                "sqrt" => CallOp::Sqrt,
                "sum" => {
                    if target_ty.is_some() {
                        return Err(syn::Error::new(
                            path.path.span(),
                            "sum does not accept a target tensor type",
                        ));
                    }
                    CallOp::Sum
                }
                "tanh" => CallOp::Tanh,
                "transpose" => CallOp::Transpose,
                _ => CallOp::Graph(op_name),
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

fn parse_call_path(path: &syn::Path) -> syn::Result<(String, Option<TensorType>)> {
    let Some(segment) = path.segments.last() else {
        return Err(syn::Error::new(path.span(), "expected graph op name"));
    };
    if path.segments.len() != 1 {
        return Err(syn::Error::new(
            path.span(),
            "graph op names must be unqualified identifiers",
        ));
    }
    let target_ty = match &segment.arguments {
        syn::PathArguments::None => None,
        syn::PathArguments::AngleBracketed(args) => {
            let mut args = args.args.iter();
            let Some(GenericArgument::Type(ty)) = args.next() else {
                return Err(syn::Error::new(
                    segment.arguments.span(),
                    "target tensor type must be a type argument",
                ));
            };
            if args.next().is_some() {
                return Err(syn::Error::new(
                    segment.arguments.span(),
                    "graph ops accept at most one target tensor type",
                ));
            }
            Some(parse_tensor_type(ty)?)
        }
        syn::PathArguments::Parenthesized(_) => {
            return Err(syn::Error::new(
                segment.arguments.span(),
                "parenthesized graph op arguments are not supported",
            ));
        }
    };
    Ok((segment.ident.to_string(), target_ty))
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
    reject_mixed_graph_elements(&graph)?;
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

fn reject_mixed_graph_elements(graph: &Graph) -> syn::Result<()> {
    let expected = graph
        .inputs
        .first()
        .map(|input| input.ty.elem)
        .unwrap_or(graph.output.elem);
    for input in &graph.inputs {
        if input.ty.elem != expected {
            return Err(syn::Error::new(
                Span::call_site(),
                "graph inputs and output must use one homogeneous tensor element type",
            ));
        }
    }
    if graph.output.elem != expected {
        return Err(syn::Error::new(
            Span::call_site(),
            "graph inputs and output must use one homogeneous tensor element type",
        ));
    }
    Ok(())
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
    if lhs == rhs {
        Ok(lhs.clone())
    } else if lhs.rank() == 0 && lhs.elem == rhs.elem {
        Ok(rhs.clone())
    } else if rhs.rank() == 0 && rhs.elem == lhs.elem {
        Ok(lhs.clone())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("elementwise operands must have the same shape and element type, got {lhs:?} and {rhs:?}"),
        ))
    }
}

fn call_result_type(
    op: &CallOp,
    args: &[Expr],
    env: &[(String, TensorType)],
    graph_signatures: &[(String, GraphSignature)],
    current_graph: &str,
) -> syn::Result<TensorType> {
    match op {
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
        | CallOp::Log
        | CallOp::Relu
        | CallOp::Sigmoid
        | CallOp::Softmax
        | CallOp::Sqrt
        | CallOp::Tanh => {
            expect_arity(op, args, 1)?;
            let ty = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, ty.elem)?;
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
            if element_count(&input) != 1 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    format!(
                        "broadcast currently expects a scalar-like input with one element, got shape {:?}",
                        input.shape
                    ),
                ));
            }
            Ok(target.clone())
        }
        CallOp::Sum => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "sum expects a tensor input",
                ));
            }
            Ok(TensorType {
                elem: input.elem,
                shape: vec![1],
            })
        }
        CallOp::Mean => {
            expect_arity(op, args, 1)?;
            let input = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            expect_float(op, input.elem)?;
            if input.rank() == 0 {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "mean expects a tensor input",
                ));
            }
            Ok(TensorType {
                elem: input.elem,
                shape: vec![1],
            })
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
        ] {
            let graph = parse(item).unwrap();
            assert_eq!(graph.body.ty, tensor(&[4]));
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

        assert!(error.to_string().contains("same shape"));
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
        assert!(broadcast.to_string().contains("scalar-like input"));
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
