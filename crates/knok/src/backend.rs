//! Supported IREE compiler backends and runtime drivers.

/// IREE target backend used for graph compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// LLVM CPU backend, normally executed with the `local-task` driver.
    LlvmCpu,
    /// Metal/SPIR-V backend, normally executed with the `metal` driver on macOS.
    #[cfg(any(target_os = "macos", doc))]
    MetalSpirv,
    /// Vulkan/SPIR-V backend, normally executed with the `vulkan` driver.
    #[cfg(feature = "vulkan")]
    VulkanSpirv,
    /// CUDA backend, normally executed with the `cuda` driver.
    #[cfg(feature = "cuda")]
    Cuda,
}

/// IREE runtime driver used to execute an artifact variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Driver {
    /// CPU local task driver.
    LocalTask,
    /// Apple Metal driver on macOS.
    #[cfg(any(target_os = "macos", doc))]
    Metal,
    /// Vulkan driver.
    #[cfg(feature = "vulkan")]
    Vulkan,
    /// NVIDIA CUDA driver.
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Backend {
    /// Returns the IREE target backend flag value.
    pub const fn name(self) -> &'static str {
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

    /// Returns the default runtime driver for this backend.
    pub const fn default_driver(self) -> &'static str {
        match self {
            Self::LlvmCpu => Driver::LocalTask.name(),
            #[cfg(any(target_os = "macos", doc))]
            Self::MetalSpirv => Driver::Metal.name(),
            #[cfg(feature = "vulkan")]
            Self::VulkanSpirv => Driver::Vulkan.name(),
            #[cfg(feature = "cuda")]
            Self::Cuda => Driver::Cuda.name(),
        }
    }

    /// Parses a backend from its IREE target backend name.
    pub fn from_name(name: &str) -> Option<Self> {
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

    /// Returns whether `driver` is the expected runtime driver for this backend.
    pub fn supports_driver(self, driver: Driver) -> bool {
        self.default_driver() == driver.name()
    }
}

/// Backends currently exposed by `knok`.
pub const SUPPORTED_BACKENDS: &[Backend] = &[
    Backend::LlvmCpu,
    #[cfg(any(target_os = "macos", doc))]
    Backend::MetalSpirv,
    #[cfg(feature = "vulkan")]
    Backend::VulkanSpirv,
    #[cfg(feature = "cuda")]
    Backend::Cuda,
];

impl Driver {
    /// Returns the IREE runtime driver name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::LocalTask => "local-task",
            #[cfg(any(target_os = "macos", doc))]
            Self::Metal => "metal",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "vulkan",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
        }
    }

    /// Parses a driver from its IREE runtime driver name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "local-task" => Some(Self::LocalTask),
            #[cfg(any(target_os = "macos", doc))]
            "metal" => Some(Self::Metal),
            #[cfg(feature = "vulkan")]
            "vulkan" => Some(Self::Vulkan),
            #[cfg(feature = "cuda")]
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }
}
