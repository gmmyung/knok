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
    pub(crate) fn from_target_backend(name: &str) -> Option<Self> {
        match name {
            "llvm-cpu" => Some(Self::LlvmCpu),
            "metal-spirv" => Some(Self::MetalSpirv),
            _ => None,
        }
    }

    fn from_path(path: &syn::Path) -> Option<Self> {
        match typed_path_variant(path, "Backend")?.as_str() {
            "LlvmCpu" => Some(Self::LlvmCpu),
            "MetalSpirv" => Some(Self::MetalSpirv),
            _ => None,
        }
    }

    pub(crate) fn default_driver(self) -> IreeDriver {
        match self {
            Self::LlvmCpu => IreeDriver::LocalTask,
            Self::MetalSpirv => IreeDriver::Metal,
        }
    }

    fn target_backend(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
        }
    }

    fn supports_driver(self, driver: IreeDriver) -> bool {
        self.default_driver() == driver
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IreeDriver {
    LocalTask,
    Metal,
}

impl IreeDriver {
    fn from_path(path: &syn::Path) -> Option<Self> {
        match typed_path_variant(path, "Driver")?.as_str() {
            "LocalTask" => Some(Self::LocalTask),
            "Metal" => Some(Self::Metal),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::LocalTask => "local-task",
            Self::Metal => "metal",
        }
    }
}

impl BackendSpec {
    pub(crate) fn new(
        backend: IreeBackend,
        driver: Option<IreeDriver>,
        extra_flags: Vec<String>,
        span: proc_macro2::Span,
    ) -> syn::Result<Self> {
        let driver = driver.unwrap_or_else(|| backend.default_driver());
        if !backend.supports_driver(driver) {
            return Err(syn::Error::new(
                span,
                format!(
                    "backend `{}` expects runtime driver `{}`, got `{driver}`",
                    backend.target_backend(),
                    backend.default_driver(),
                ),
            ));
        }
        Ok(Self {
            backend: backend.target_backend().to_string(),
            driver: driver.name().to_string(),
            extra_flags,
        })
    }
}

impl std::fmt::Display for IreeDriver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

fn typed_path_variant(path: &syn::Path, type_name: &str) -> Option<String> {
    let segments = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>();
    match segments.as_slice() {
        [ty, variant] if ty == type_name => Some(variant.clone()),
        [crate_name, ty, variant] if crate_name == "knok" && ty == type_name => {
            Some(variant.clone())
        }
        _ => None,
    }
}

fn parse_backend_path(value: &syn::Expr) -> syn::Result<(IreeBackend, proc_macro2::Span)> {
    let syn::Expr::Path(path) = value else {
        return Err(syn::Error::new(
            value.span(),
            "backend must be a path such as Backend::LlvmCpu or knok::Backend::LlvmCpu",
        ));
    };
    let backend = IreeBackend::from_path(&path.path).ok_or_else(|| {
        syn::Error::new(
            path.span(),
            "unsupported backend path; expected Backend::LlvmCpu or Backend::MetalSpirv",
        )
    })?;
    Ok((backend, path.span()))
}

fn parse_driver_path(value: &syn::Expr) -> syn::Result<(IreeDriver, proc_macro2::Span)> {
    let syn::Expr::Path(path) = value else {
        return Err(syn::Error::new(
            value.span(),
            "driver must be a path such as Driver::LocalTask or knok::Driver::Metal",
        ));
    };
    let driver = IreeDriver::from_path(&path.path).ok_or_else(|| {
        syn::Error::new(
            path.span(),
            "unsupported driver path; expected Driver::LocalTask or Driver::Metal",
        )
    })?;
    Ok((driver, path.span()))
}

pub(crate) fn parse_backend_expr(value: &syn::Expr) -> syn::Result<BackendSpec> {
    let (backend, span) = parse_backend_path(value)?;
    BackendSpec::new(backend, None, Vec::new(), span)
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
            backend = Some(vec![parse_backend_expr(&arg.value)?]);
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
            "missing required backend = Backend::... argument",
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
        return Err(syn::Error::new(call.span(), "backend path is required"));
    };
    let (backend, backend_span) = parse_backend_path(first)?;

    let mut driver = None;
    let mut extra_flags = Vec::new();
    for arg in call.args.iter().skip(1) {
        let syn::Expr::Assign(assign) = arg else {
            return Err(syn::Error::new(
                arg.span(),
                "backend options must be assignments such as driver = Driver::LocalTask",
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
                driver = Some(parse_driver_path(assign.right.as_ref())?.0);
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
    BackendSpec::new(backend, driver, extra_flags, backend_span)
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
    let capability = IreeBackend::from_target_backend(backend)
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
