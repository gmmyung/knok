#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Compile-time IREE target backend.
pub enum Backend {
    /// IREE LLVM CPU backend.
    LlvmCpu,
    /// IREE Metal backend through SPIR-V.
    MetalSpirv,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Runtime driver used to execute a compiled artifact variant.
pub enum Driver {
    /// IREE local-task CPU driver.
    LocalTask,
    /// IREE Metal driver.
    Metal,
}

impl Backend {
    /// Returns the IREE backend name used by `iree-compile`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
        }
    }

    /// Returns the default runtime driver expected for this backend.
    pub const fn default_driver(self) -> &'static str {
        match self {
            Self::LlvmCpu => Driver::LocalTask.name(),
            Self::MetalSpirv => Driver::Metal.name(),
        }
    }

    /// Parses an IREE backend name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "llvm-cpu" => Some(Self::LlvmCpu),
            "metal-spirv" => Some(Self::MetalSpirv),
            _ => None,
        }
    }

    /// Returns whether `driver` can execute artifacts compiled for this backend.
    pub fn supports_driver(self, driver: Driver) -> bool {
        self.default_driver() == driver.name()
    }
}

/// Backends accepted by graph and MLIR model macros.
pub const SUPPORTED_BACKENDS: &[Backend] = &[Backend::LlvmCpu, Backend::MetalSpirv];

impl Driver {
    /// Returns the IREE runtime driver name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::LocalTask => "local-task",
            Self::Metal => "metal",
        }
    }

    /// Parses an IREE runtime driver name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "local-task" => Some(Self::LocalTask),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }
}
