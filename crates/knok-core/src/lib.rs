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
    ConstF32(String),
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
    Matmul,
    Relu,
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
    }
    Err(syn::Error::new(
        Span::call_site(),
        "missing required backend = \"...\" argument",
    ))
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

fn parse_tensor_type(ty: &Type) -> syn::Result<TensorType> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return Err(syn::Error::new(
            ty.span(),
            "expected Tensor1 or Tensor2 type",
        ));
    };
    let segment = path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new(path.span(), "expected Tensor1 or Tensor2 type"))?;
    let rank = match segment.ident.to_string().as_str() {
        "Tensor1" => 1,
        "Tensor2" => 2,
        _ => {
            return Err(syn::Error::new(
                segment.ident.span(),
                "expected Tensor1<T, D0> or Tensor2<T, D0, D1>",
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
            "tensor element type must be f32",
        ));
    };
    if path.path.is_ident("f32") {
        Ok(ElementType::F32)
    } else {
        Err(syn::Error::new(
            path.span(),
            "only f32 tensors are supported",
        ))
    }
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
            Lit::Float(lit) => Ok(Expr::ConstF32(lit.base10_digits().to_string())),
            Lit::Int(lit) => Ok(Expr::ConstF32(lit.base10_digits().to_string())),
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
            let op_name = path.path.require_ident()?.to_string();
            let op = match op_name.as_str() {
                "matmul" => CallOp::Matmul,
                "relu" => CallOp::Relu,
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
        Expr::ConstF32(_) => TensorType {
            elem: ElementType::F32,
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
    } else if lhs.rank() == 0 {
        Ok(rhs.clone())
    } else if rhs.rank() == 0 {
        Ok(lhs.clone())
    } else {
        Err(syn::Error::new(
            Span::call_site(),
            format!("elementwise operands must have the same shape, got {lhs:?} and {rhs:?}"),
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
        CallOp::Relu => {
            expect_arity(op, args, 1)?;
            Ok(type_expr(&args[0], env, graph_signatures, current_graph)?.ty)
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
        CallOp::Matmul => {
            expect_arity(op, args, 2)?;
            let lhs = type_expr(&args[0], env, graph_signatures, current_graph)?.ty;
            let rhs = type_expr(&args[1], env, graph_signatures, current_graph)?.ty;
            if lhs.rank() != 2 || rhs.rank() != 2 || lhs.elem != rhs.elem {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "matmul expects two rank-2 tensors with the same element type",
                ));
            }
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
