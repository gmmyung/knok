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
    } else if capability == IreeBackend::LlvmCpu
        && !extra_flags
            .iter()
            .any(|flag| flag.starts_with("--iree-llvmcpu-target-cpu="))
    {
        flags.push("--iree-llvmcpu-target-cpu=generic".to_string());
    }
    flags.extend(extra_flags.iter().cloned());
    flags
}
