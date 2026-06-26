use proc_macro2::Span;
use syn::{
    parse::Parser, spanned::Spanned, Attribute, BinOp, Expr as SynExpr, FnArg, GenericArgument,
    ItemFn, Lit, MetaNameValue, Pat, PatIdent, ReturnType, Stmt, Type, TypePath, UnOp,
};

use crate::{
    type_check, AxisSpec, BinaryOp, CallOp, Conv2dOptions, ElementType, Expr, Graph,
    GraphSignature, Input, Let, Padding2d, TensorType, TypedGraph, UnaryOp,
};

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
            return parse_backend_path(&arg.value);
        }
        if arg.path.is_ident("backends") {
            return parse_first_backend_from_array(&arg.value);
        }
    }
    Err(syn::Error::new(
        Span::call_site(),
        "missing required backend = Backend::... argument",
    ))
}

fn parse_backend_path(value: &SynExpr) -> syn::Result<String> {
    let SynExpr::Path(path) = value else {
        return Err(syn::Error::new(
            value.span(),
            "backend must be a path such as Backend::LlvmCpu or knok::Backend::LlvmCpu",
        ));
    };
    let Some(backend) = backend_from_path(&path.path) else {
        return Err(syn::Error::new(
            path.span(),
            "unsupported backend path; expected Backend::LlvmCpu or Backend::MetalSpirv",
        ));
    };
    Ok(backend.to_string())
}

fn backend_from_path(path: &syn::Path) -> Option<&'static str> {
    let mut segments = path.segments.iter().rev();
    let variant = segments.next()?;
    let ty = segments.next()?;
    if ty.ident != "Backend" {
        return None;
    }
    match variant.ident.to_string().as_str() {
        "LlvmCpu" => Some("llvm-cpu"),
        "MetalSpirv" => Some("metal-spirv"),
        _ => None,
    }
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
    let Some(first) = call.args.first() else {
        return Err(syn::Error::new(call.span(), "backend path is required"));
    };
    parse_backend_path(first)
}

fn parse_item_fn(item: ItemFn, backend: String) -> syn::Result<Graph> {
    reject_unsupported_attrs(&item.attrs)?;
    let name = item.sig.ident.to_string();
    let mut inputs = Vec::new();
    for input in &item.sig.inputs {
        inputs.push(parse_input(input)?);
    }
    let outputs = match &item.sig.output {
        ReturnType::Type(_, ty) => parse_output_types(ty)?,
        ReturnType::Default => {
            return Err(syn::Error::new(
                item.sig.ident.span(),
                "graph functions must return a Tensor type or tuple of Tensor types",
            ));
        }
    };

    let (lets, body) = parse_block(&item.block.stmts)?;
    Ok(Graph {
        name,
        backend,
        inputs,
        outputs,
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
            "expected Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, or Tensor6 type",
        ));
    };
    let segment = path.segments.last().ok_or_else(|| {
        syn::Error::new(
            path.span(),
            "expected Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, or Tensor6 type",
        )
    })?;
    let rank = match segment.ident.to_string().as_str() {
        "Tensor0" => 0,
        "Tensor1" => 1,
        "Tensor2" => 2,
        "Tensor3" => 3,
        "Tensor4" => 4,
        "Tensor5" => 5,
        "Tensor6" => 6,
        _ => {
            return Err(syn::Error::new(
                segment.ident.span(),
                "expected Tensor0<T>, Tensor1<T, D0>, Tensor2<T, D0, D1>, Tensor3<T, D0, D1, D2>, Tensor4<T, D0, D1, D2, D3>, Tensor5<T, D0, D1, D2, D3, D4>, or Tensor6<T, D0, D1, D2, D3, D4, D5>",
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

fn parse_output_types(ty: &Type) -> syn::Result<Vec<TensorType>> {
    match ty {
        Type::Tuple(tuple) => {
            if tuple.elems.is_empty() {
                return Err(syn::Error::new(
                    tuple.span(),
                    "graph output tuple must contain at least one Tensor type",
                ));
            }
            tuple.elems.iter().map(parse_tensor_type).collect()
        }
        _ => Ok(vec![parse_tensor_type(ty)?]),
    }
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

fn parse_block(stmts: &[Stmt]) -> syn::Result<(Vec<Let>, Vec<Expr>)> {
    let mut lets = Vec::new();
    let mut body = None;
    for stmt in stmts {
        match stmt {
            Stmt::Local(local) => {
                let Some(init) = &local.init else {
                    return Err(syn::Error::new(
                        local.span(),
                        "let bindings must have initializers",
                    ));
                };
                lets.push(Let {
                    names: parse_let_names(&local.pat)?,
                    value: parse_expr(&init.expr)?,
                });
            }
            Stmt::Expr(expr, None) => {
                body = Some(parse_body_expr(expr)?);
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
            "graph functions must end with a tensor expression or tuple of tensor expressions",
        )
    })?;
    Ok((lets, body))
}

fn parse_let_names(pat: &Pat) -> syn::Result<Vec<String>> {
    match pat {
        Pat::Ident(PatIdent { ident, .. }) => Ok(vec![ident.to_string()]),
        Pat::Tuple(tuple) => {
            if tuple.elems.is_empty() {
                return Err(syn::Error::new(
                    tuple.span(),
                    "tuple let bindings must contain at least one identifier",
                ));
            }
            tuple
                .elems
                .iter()
                .map(|pat| {
                    let Pat::Ident(PatIdent { ident, .. }) = pat else {
                        return Err(syn::Error::new(
                            pat.span(),
                            "tuple let bindings must contain only simple identifiers",
                        ));
                    };
                    Ok(ident.to_string())
                })
                .collect()
        }
        Pat::Paren(paren) => parse_let_names(&paren.pat),
        _ => Err(syn::Error::new(
            pat.span(),
            "let bindings must use simple identifiers or tuple patterns of simple identifiers",
        )),
    }
}

fn parse_body_expr(expr: &SynExpr) -> syn::Result<Vec<Expr>> {
    match expr {
        SynExpr::Tuple(tuple) => {
            if tuple.elems.is_empty() {
                return Err(syn::Error::new(
                    tuple.span(),
                    "graph output tuple must contain at least one expression",
                ));
            }
            tuple.elems.iter().map(parse_expr).collect()
        }
        SynExpr::Paren(paren) => parse_body_expr(&paren.expr),
        _ => Ok(vec![parse_expr(expr)?]),
    }
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
                    reject_types(&generics, &path.path, "all")?;
                    CallOp::All(optional_axis(&generics, &path.path, "all")?)
                }
                "argmax" => {
                    reject_types(&generics, &path.path, "argmax")?;
                    CallOp::Argmax(optional_axis(&generics, &path.path, "argmax")?)
                }
                "any" => {
                    reject_types(&generics, &path.path, "any")?;
                    CallOp::Any(optional_axis(&generics, &path.path, "any")?)
                }
                "clip" => {
                    reject_any_generics(&generics, &path.path, "clip")?;
                    CallOp::Clip
                }
                "concat" => {
                    reject_types(&generics, &path.path, "concat")?;
                    CallOp::Concat(expect_one_const(&generics, &path.path, "concat")?)
                }
                "conv2d" => CallOp::Conv2d(parse_conv2d_options(&generics, &path.path)?),
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
                "gather" => {
                    let target = expect_target_type(&generics, &path.path, "gather")?;
                    let values = expect_const_count(&generics, &path.path, "gather", 1)?;
                    CallOp::Gather {
                        target,
                        axis: values[0],
                    }
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
                    reject_types(&generics, &path.path, "mean")?;
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
                "permute" => {
                    let target = expect_target_type(&generics, &path.path, "permute")?;
                    CallOp::Permute {
                        target,
                        axes: generics.consts.clone(),
                    }
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
                    CallOp::Reshape(expect_target_type(&generics, &path.path, "reshape")?)
                }
                "broadcast" => {
                    reject_consts(&generics, &path.path, "broadcast")?;
                    CallOp::Broadcast(expect_target_type(&generics, &path.path, "broadcast")?)
                }
                "sigmoid" => {
                    reject_any_generics(&generics, &path.path, "sigmoid")?;
                    CallOp::Sigmoid
                }
                "softmax" => {
                    reject_types(&generics, &path.path, "softmax")?;
                    CallOp::Softmax(optional_axis(&generics, &path.path, "softmax")?)
                }
                "slice" => {
                    let target = expect_target_type(&generics, &path.path, "slice")?;
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
                    CallOp::Squeeze(expect_target_type(&generics, &path.path, "squeeze")?)
                }
                "stack" => {
                    reject_types(&generics, &path.path, "stack")?;
                    CallOp::Stack(expect_one_const(&generics, &path.path, "stack")?)
                }
                "sum" => {
                    reject_types(&generics, &path.path, "sum")?;
                    CallOp::Sum(optional_axis(&generics, &path.path, "sum")?)
                }
                "tanh" => {
                    reject_any_generics(&generics, &path.path, "tanh")?;
                    CallOp::Tanh
                }
                "take" => {
                    reject_types(&generics, &path.path, "take")?;
                    let values = expect_const_count(&generics, &path.path, "take", 2)?;
                    CallOp::Take {
                        axis: values[0],
                        index: values[1],
                    }
                }
                "take_along_axis" => {
                    reject_types(&generics, &path.path, "take_along_axis")?;
                    CallOp::TakeAlongAxis {
                        axis: expect_one_const(&generics, &path.path, "take_along_axis")?,
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
                    CallOp::Unsqueeze(expect_target_type(&generics, &path.path, "unsqueeze")?)
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
    types: Vec<Type>,
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
                        generics.types.push(ty.clone());
                    }
                    GenericArgument::Const(expr) => {
                        generics.consts.push(parse_generic_const_usize(expr)?);
                    }
                    _ => {
                        return Err(syn::Error::new(
                            arg.span(),
                            "graph op generic argument must be a type-like option or integer const",
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
    generics: &CallGenerics,
    path: &syn::Path,
    op_name: &str,
) -> syn::Result<TensorType> {
    if generics.types.len() == 1 {
        parse_tensor_type(&generics.types[0])
    } else {
        Err(syn::Error::new(
            path.span(),
            format!("{op_name} requires exactly one target tensor type, for example {op_name}::<Tensor1<f32, 4>>(x)"),
        ))
    }
}

fn reject_types(generics: &CallGenerics, path: &syn::Path, op_name: &str) -> syn::Result<()> {
    if !generics.types.is_empty() {
        Err(syn::Error::new(
            path.span(),
            format!("{op_name} does not accept type generic arguments"),
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
    if !generics.types.is_empty() || !generics.consts.is_empty() {
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
) -> syn::Result<AxisSpec> {
    match generics.consts.as_slice() {
        [] => Ok(AxisSpec::All),
        [axis] => Ok(AxisSpec::One(*axis)),
        _ => Err(syn::Error::new(
            path.span(),
            format!("{op_name} accepts at most one axis const generic"),
        )),
    }
}

fn parse_conv2d_options(generics: &CallGenerics, path: &syn::Path) -> syn::Result<Conv2dOptions> {
    if !generics.consts.is_empty() {
        return Err(syn::Error::new(
            path.span(),
            "conv2d options use type-style generics, for example conv2d::<Pad<1, 1, 1, 1>, Stride<2, 2>>(x, k)",
        ));
    }
    let mut options = Conv2dOptions::default();
    let mut saw_padding = false;
    let mut saw_stride = false;
    let mut saw_dilation = false;
    let mut saw_groups = false;
    for ty in &generics.types {
        let segment = option_type_segment(ty)?;
        match segment.ident.to_string().as_str() {
            "Pad" | "Padding" => {
                if saw_padding {
                    return Err(syn::Error::new(
                        segment.span(),
                        "duplicate conv2d padding option",
                    ));
                }
                let args = option_const_args(segment, 4, "Pad")?;
                options.padding = Padding2d {
                    top: args[0],
                    bottom: args[1],
                    left: args[2],
                    right: args[3],
                };
                saw_padding = true;
            }
            "Stride" => {
                if saw_stride {
                    return Err(syn::Error::new(
                        segment.span(),
                        "duplicate conv2d stride option",
                    ));
                }
                let args = option_const_args(segment, 2, "Stride")?;
                options.stride = [args[0], args[1]];
                saw_stride = true;
            }
            "Dilation" => {
                if saw_dilation {
                    return Err(syn::Error::new(
                        segment.span(),
                        "duplicate conv2d dilation option",
                    ));
                }
                let args = option_const_args(segment, 2, "Dilation")?;
                options.dilation = [args[0], args[1]];
                saw_dilation = true;
            }
            "Groups" => {
                if saw_groups {
                    return Err(syn::Error::new(
                        segment.span(),
                        "duplicate conv2d groups option",
                    ));
                }
                let args = option_const_args(segment, 1, "Groups")?;
                options.groups = args[0];
                saw_groups = true;
            }
            name => {
                return Err(syn::Error::new(
                    segment.span(),
                    format!(
                        "unknown conv2d option `{name}`; expected Pad<TOP, BOTTOM, LEFT, RIGHT>, Stride<H, W>, Dilation<H, W>, or Groups<N>"
                    ),
                ));
            }
        }
    }
    Ok(options)
}

fn option_type_segment(ty: &Type) -> syn::Result<&syn::PathSegment> {
    let Type::Path(TypePath { path, qself: None }) = ty else {
        return Err(syn::Error::new(
            ty.span(),
            "conv2d options must be type paths such as Pad<1, 1, 1, 1>",
        ));
    };
    path.segments.last().ok_or_else(|| {
        syn::Error::new(
            ty.span(),
            "conv2d options must be type paths such as Pad<1, 1, 1, 1>",
        )
    })
}

fn option_const_args(
    segment: &syn::PathSegment,
    expected: usize,
    option_name: &str,
) -> syn::Result<Vec<usize>> {
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return Err(syn::Error::new(
            segment.span(),
            format!("{option_name} expects {expected} integer const arguments"),
        ));
    };
    if args.args.len() != expected {
        return Err(syn::Error::new(
            args.span(),
            format!(
                "{option_name} expects {expected} integer const arguments, got {}",
                args.args.len()
            ),
        ));
    }
    args.args
        .iter()
        .map(|arg| {
            let GenericArgument::Const(expr) = arg else {
                return Err(syn::Error::new(
                    arg.span(),
                    format!("{option_name} arguments must be integer consts"),
                ));
            };
            parse_generic_const_usize(expr)
        })
        .collect()
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
