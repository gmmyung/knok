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

    pub(crate) fn target_name(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llvm_cpu_flags_include_generic_cpu_by_default() {
        let flags = backend_flags("llvm-cpu", &[]);

        assert_eq!(flags[0], "--iree-hal-target-backends=llvm-cpu");
        assert!(flags.contains(&"--iree-input-demote-f64-to-f32=false".to_string()));
        assert!(flags.contains(&"--iree-llvmcpu-target-cpu=generic".to_string()));
    }

    #[test]
    fn llvm_cpu_flags_do_not_override_explicit_cpu() {
        let flags = backend_flags(
            "llvm-cpu",
            &["--iree-llvmcpu-target-cpu=apple-m1".to_string()],
        );

        assert!(flags.contains(&"--iree-llvmcpu-target-cpu=apple-m1".to_string()));
        assert!(!flags.contains(&"--iree-llvmcpu-target-cpu=generic".to_string()));
    }

    #[test]
    fn metal_flags_disable_metallib_output() {
        let flags = backend_flags("metal-spirv", &["--custom".to_string()]);

        assert!(flags.contains(&"--iree-hal-target-backends=metal-spirv".to_string()));
        assert!(flags.contains(&"--iree-metal-compile-to-metallib=false".to_string()));
        assert!(flags.contains(&"--custom".to_string()));
        assert!(!flags.contains(&"--iree-llvmcpu-target-cpu=generic".to_string()));
    }
}
