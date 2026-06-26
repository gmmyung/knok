use std::collections::BTreeMap;

use proc_macro2::TokenStream;
use syn::{parse::Parser, punctuated::Punctuated, spanned::Spanned, Lit, MetaNameValue, Token};

pub(crate) struct BackendSpec {
    pub(crate) backend: String,
    pub(crate) driver: String,
    pub(crate) extra_flags: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IreeBackend {
    LlvmCpu,
    MetalSpirv,
}

impl IreeBackend {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "llvm-cpu" => Some(Self::LlvmCpu),
            "metal-spirv" => Some(Self::MetalSpirv),
            _ => None,
        }
    }

    fn default_driver(self) -> &'static str {
        match self {
            Self::LlvmCpu => "local-task",
            Self::MetalSpirv => "metal",
        }
    }

    fn target_backend(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
        }
    }

    fn supports_driver(self, driver: &str) -> bool {
        self.default_driver() == driver
    }
}

impl BackendSpec {
    pub(crate) fn new(
        backend: String,
        driver: Option<String>,
        extra_flags: Vec<String>,
        span: proc_macro2::Span,
    ) -> syn::Result<Self> {
        let capability = IreeBackend::parse(&backend).ok_or_else(|| {
            syn::Error::new(
                span,
                format!(
                    "unsupported IREE backend `{backend}`; expected `llvm-cpu` or `metal-spirv`"
                ),
            )
        })?;
        let driver = driver.unwrap_or_else(|| capability.default_driver().to_string());
        if !capability.supports_driver(&driver) {
            return Err(syn::Error::new(
                span,
                format!(
                    "backend `{}` expects runtime driver `{}`, got `{driver}`",
                    capability.target_backend(),
                    capability.default_driver(),
                ),
            ));
        }
        Ok(Self {
            backend,
            driver,
            extra_flags,
        })
    }
}

pub(crate) fn parse_backend_specs(attr: TokenStream) -> syn::Result<Vec<BackendSpec>> {
    let args = Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse2(attr)?;
    let mut backend = None;
    let mut backends = None;
    for arg in args {
        if arg.path.is_ident("backend") {
            if backend.is_some() || backends.is_some() {
                return Err(syn::Error::new(
                    arg.span(),
                    "backend and backends are mutually exclusive",
                ));
            }
            let syn::Expr::Lit(expr_lit) = &arg.value else {
                return Err(syn::Error::new(
                    arg.value.span(),
                    "backend must be a string literal",
                ));
            };
            let Lit::Str(lit) = &expr_lit.lit else {
                return Err(syn::Error::new(
                    expr_lit.span(),
                    "backend must be a string literal",
                ));
            };
            backend = Some(vec![BackendSpec::new(
                lit.value(),
                None,
                Vec::new(),
                lit.span(),
            )?]);
        } else if arg.path.is_ident("backends") {
            if backend.is_some() || backends.is_some() {
                return Err(syn::Error::new(
                    arg.span(),
                    "backend and backends are mutually exclusive",
                ));
            }
            backends = Some(parse_backend_array(&arg.value)?);
        } else {
            return Err(syn::Error::new(
                arg.path.span(),
                "unknown graph attribute argument",
            ));
        }
    }
    let specs = backend.or(backends).ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "missing required backend = \"...\" argument",
        )
    })?;
    reject_duplicate_drivers(&specs)?;
    Ok(specs)
}

pub(crate) fn parse_backend_array(value: &syn::Expr) -> syn::Result<Vec<BackendSpec>> {
    let syn::Expr::Array(array) = value else {
        return Err(syn::Error::new(
            value.span(),
            "backends must be an array of backend(...) declarations",
        ));
    };
    if array.elems.is_empty() {
        return Err(syn::Error::new(
            array.span(),
            "backends must contain at least one backend(...) declaration",
        ));
    }
    array.elems.iter().map(parse_backend_call).collect()
}

fn parse_backend_call(expr: &syn::Expr) -> syn::Result<BackendSpec> {
    let syn::Expr::Call(call) = expr else {
        return Err(syn::Error::new(expr.span(), "expected backend(...)"));
    };
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return Err(syn::Error::new(call.func.span(), "expected backend(...)"));
    };
    if !path.path.is_ident("backend") {
        return Err(syn::Error::new(call.func.span(), "expected backend(...)"));
    }
    let Some(first) = call.args.first() else {
        return Err(syn::Error::new(call.span(), "backend name is required"));
    };
    let syn::Expr::Lit(expr_lit) = first else {
        return Err(syn::Error::new(
            first.span(),
            "backend name must be a string literal",
        ));
    };
    let Lit::Str(backend_lit) = &expr_lit.lit else {
        return Err(syn::Error::new(
            expr_lit.span(),
            "backend name must be a string literal",
        ));
    };

    let mut driver = None;
    let mut extra_flags = Vec::new();
    for arg in call.args.iter().skip(1) {
        let syn::Expr::Assign(assign) = arg else {
            return Err(syn::Error::new(
                arg.span(),
                "backend options must be assignments such as driver = \"...\"",
            ));
        };
        let syn::Expr::Path(key_path) = assign.left.as_ref() else {
            return Err(syn::Error::new(assign.left.span(), "expected option name"));
        };
        let key = key_path.path.require_ident()?.to_string();
        match key.as_str() {
            "driver" => {
                if driver.is_some() {
                    return Err(syn::Error::new(assign.span(), "duplicate driver option"));
                }
                let syn::Expr::Lit(expr_lit) = assign.right.as_ref() else {
                    return Err(syn::Error::new(
                        assign.right.span(),
                        "driver must be a string literal",
                    ));
                };
                let Lit::Str(lit) = &expr_lit.lit else {
                    return Err(syn::Error::new(
                        expr_lit.span(),
                        "driver must be a string literal",
                    ));
                };
                driver = Some(lit.value());
            }
            "flags" => {
                let syn::Expr::Array(array) = assign.right.as_ref() else {
                    return Err(syn::Error::new(
                        assign.right.span(),
                        "flags must be an array of string literals",
                    ));
                };
                for flag in &array.elems {
                    let syn::Expr::Lit(expr_lit) = flag else {
                        return Err(syn::Error::new(
                            flag.span(),
                            "flags must be string literals",
                        ));
                    };
                    let Lit::Str(lit) = &expr_lit.lit else {
                        return Err(syn::Error::new(
                            expr_lit.span(),
                            "flags must be string literals",
                        ));
                    };
                    extra_flags.push(lit.value());
                }
            }
            _ => {
                return Err(syn::Error::new(
                    key_path.path.span(),
                    format!("unknown backend option `{key}`"),
                ));
            }
        }
    }
    BackendSpec::new(backend_lit.value(), driver, extra_flags, backend_lit.span())
}

pub(crate) fn reject_duplicate_drivers(specs: &[BackendSpec]) -> syn::Result<()> {
    let mut drivers = BTreeMap::<&str, &str>::new();
    for spec in specs {
        if let Some(existing_backend) = drivers.insert(&spec.driver, &spec.backend) {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "duplicate runtime driver `{}` for backends `{}` and `{}`",
                    spec.driver, existing_backend, spec.backend
                ),
            ));
        }
    }
    Ok(())
}

pub(crate) fn backend_flags(backend: &str, extra_flags: &[String]) -> Vec<String> {
    let capability = IreeBackend::parse(backend)
        .unwrap_or_else(|| panic!("unsupported IREE backend `{backend}`"));
    let mut flags = vec![
        format!("--iree-hal-target-backends={backend}"),
        "--iree-input-demote-f64-to-f32=false".to_string(),
    ];
    if capability == IreeBackend::MetalSpirv {
        flags.push("--iree-metal-compile-to-metallib=false".to_string());
    }
    flags.extend(extra_flags.iter().cloned());
    flags
}
