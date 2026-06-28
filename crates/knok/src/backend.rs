//! Supported IREE compiler backends and runtime drivers.

/// IREE target backend used for graph compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// LLVM CPU backend, normally executed with the `local-task` driver.
    LlvmCpu,
    /// Metal/SPIR-V backend, normally executed with the `metal` driver.
    MetalSpirv,
}

/// IREE runtime driver used to execute an artifact variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Driver {
    /// CPU local task driver.
    LocalTask,
    /// Apple Metal driver.
    Metal,
}

impl Backend {
    /// Returns the IREE target backend flag value.
    pub const fn name(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
        }
    }

    /// Returns the default runtime driver for this backend.
    pub const fn default_driver(self) -> &'static str {
        match self {
            Self::LlvmCpu => Driver::LocalTask.name(),
            Self::MetalSpirv => Driver::Metal.name(),
        }
    }

    /// Parses a backend from its IREE target backend name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "llvm-cpu" => Some(Self::LlvmCpu),
            "metal-spirv" => Some(Self::MetalSpirv),
            _ => None,
        }
    }

    /// Returns whether `driver` is the expected runtime driver for this backend.
    pub fn supports_driver(self, driver: Driver) -> bool {
        self.default_driver() == driver.name()
    }
}

/// Backends currently exposed by `knok`.
pub const SUPPORTED_BACKENDS: &[Backend] = &[Backend::LlvmCpu, Backend::MetalSpirv];

impl Driver {
    /// Returns the IREE runtime driver name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::LocalTask => "local-task",
            Self::Metal => "metal",
        }
    }

    /// Parses a driver from its IREE runtime driver name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "local-task" => Some(Self::LocalTask),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }
}
