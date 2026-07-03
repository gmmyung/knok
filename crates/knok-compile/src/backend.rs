use std::env;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IreeBackend {
    LlvmCpu,
    #[cfg(any(target_os = "macos", doc))]
    MetalSpirv,
    #[cfg(feature = "vulkan")]
    VulkanSpirv,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl IreeBackend {
    pub(crate) fn from_target_backend(name: &str) -> Option<Self> {
        match name {
            "llvm-cpu" => Some(Self::LlvmCpu),
            #[cfg(any(target_os = "macos", doc))]
            "metal-spirv" => Some(Self::MetalSpirv),
            #[cfg(feature = "vulkan")]
            "vulkan-spirv" => Some(Self::VulkanSpirv),
            #[cfg(feature = "cuda")]
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }

    pub(crate) fn target_name(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            #[cfg(any(target_os = "macos", doc))]
            Self::MetalSpirv => "metal-spirv",
            #[cfg(feature = "vulkan")]
            Self::VulkanSpirv => "vulkan-spirv",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
        }
    }
}

pub(crate) fn supported_backend_names() -> &'static [&'static str] {
    &[
        "llvm-cpu",
        #[cfg(any(target_os = "macos", doc))]
        "metal-spirv",
        #[cfg(feature = "vulkan")]
        "vulkan-spirv",
        #[cfg(feature = "cuda")]
        "cuda",
    ]
}

pub(crate) fn backend_flags(backend: &str, extra_flags: &[String]) -> Vec<String> {
    backend_flags_with_opt_level(backend, extra_flags, env::var("OPT_LEVEL").ok().as_deref())
}

fn backend_flags_with_opt_level(
    backend: &str,
    extra_flags: &[String],
    cargo_opt_level: Option<&str>,
) -> Vec<String> {
    let capability = IreeBackend::from_target_backend(backend)
        .unwrap_or_else(|| panic!("unsupported IREE backend `{backend}`"));
    let mut flags = vec![
        format!("--iree-hal-target-backends={backend}"),
        "--iree-input-demote-f64-to-f32=false".to_string(),
    ];
    #[cfg(any(target_os = "macos", doc))]
    if capability == IreeBackend::MetalSpirv {
        flags.push("--iree-metal-compile-to-metallib=false".to_string());
    }
    if capability == IreeBackend::LlvmCpu {
        if let Some(opt_level) = cargo_iree_opt_level(cargo_opt_level) {
            if !extra_flags
                .iter()
                .any(|flag| flag.starts_with("--iree-opt-level="))
            {
                flags.push(format!("--iree-opt-level={opt_level}"));
            }
        }
        if !extra_flags
            .iter()
            .any(|flag| flag.starts_with("--iree-llvmcpu-target-cpu="))
        {
            flags.push("--iree-llvmcpu-target-cpu=generic".to_string());
        }
    }
    flags.extend(extra_flags.iter().cloned());
    flags
}

fn cargo_iree_opt_level(cargo_opt_level: Option<&str>) -> Option<&'static str> {
    match cargo_opt_level {
        Some("0") => Some("O0"),
        Some("1") => Some("O1"),
        Some("2") => Some("O2"),
        Some("3") => Some("O3"),
        Some("s" | "z") => Some("O2"),
        _ => None,
    }
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
    fn llvm_cpu_opt_level_follows_cargo_opt_level() {
        for (cargo, iree) in [
            ("0", "O0"),
            ("1", "O1"),
            ("2", "O2"),
            ("3", "O3"),
            ("s", "O2"),
            ("z", "O2"),
        ] {
            let flags = backend_flags_with_opt_level("llvm-cpu", &[], Some(cargo));

            assert!(
                flags.contains(&format!("--iree-opt-level={iree}")),
                "{cargo:?} should map to {iree:?}: {flags:?}"
            );
        }
    }

    #[test]
    fn llvm_cpu_opt_level_can_be_overridden() {
        let flags = backend_flags_with_opt_level(
            "llvm-cpu",
            &["--iree-opt-level=O1".to_string()],
            Some("3"),
        );

        assert_eq!(
            flags
                .iter()
                .filter(|flag| flag.starts_with("--iree-opt-level="))
                .count(),
            1
        );
        assert!(flags.contains(&"--iree-opt-level=O1".to_string()));
    }

    #[cfg(any(target_os = "macos", doc))]
    #[test]
    fn non_cpu_backends_do_not_inherit_cargo_opt_level() {
        let flags = backend_flags_with_opt_level("metal-spirv", &[], Some("3"));

        assert!(!flags
            .iter()
            .any(|flag| flag.starts_with("--iree-opt-level=")));
    }

    #[cfg(any(target_os = "macos", doc))]
    #[test]
    fn metal_flags_disable_metallib_output() {
        let flags = backend_flags("metal-spirv", &["--custom".to_string()]);

        assert!(flags.contains(&"--iree-hal-target-backends=metal-spirv".to_string()));
        assert!(flags.contains(&"--iree-metal-compile-to-metallib=false".to_string()));
        assert!(flags.contains(&"--custom".to_string()));
        assert!(!flags.contains(&"--iree-llvmcpu-target-cpu=generic".to_string()));
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_flags_select_vulkan_spirv() {
        let flags = backend_flags("vulkan-spirv", &["--iree-vulkan-target=ampere".to_string()]);

        assert!(flags.contains(&"--iree-hal-target-backends=vulkan-spirv".to_string()));
        assert!(flags.contains(&"--iree-vulkan-target=ampere".to_string()));
        assert!(!flags.contains(&"--iree-llvmcpu-target-cpu=generic".to_string()));
        assert!(!flags.contains(&"--iree-metal-compile-to-metallib=false".to_string()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_flags_select_cuda() {
        let flags = backend_flags("cuda", &["--cuda_use_streams=true".to_string()]);

        assert!(flags.contains(&"--iree-hal-target-backends=cuda".to_string()));
        assert!(flags.contains(&"--cuda_use_streams=true".to_string()));
        assert!(!flags.contains(&"--iree-llvmcpu-target-cpu=generic".to_string()));
        assert!(!flags.contains(&"--iree-metal-compile-to-metallib=false".to_string()));
    }
}
