#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    LlvmCpu,
    MetalSpirv,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Driver {
    LocalTask,
    Metal,
}

impl Backend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::LlvmCpu => "llvm-cpu",
            Self::MetalSpirv => "metal-spirv",
        }
    }

    pub const fn default_driver(self) -> &'static str {
        match self {
            Self::LlvmCpu => Driver::LocalTask.name(),
            Self::MetalSpirv => Driver::Metal.name(),
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "llvm-cpu" => Some(Self::LlvmCpu),
            "metal-spirv" => Some(Self::MetalSpirv),
            _ => None,
        }
    }

    pub fn supports_driver(self, driver: Driver) -> bool {
        self.default_driver() == driver.name()
    }
}

pub const SUPPORTED_BACKENDS: &[Backend] = &[Backend::LlvmCpu, Backend::MetalSpirv];

impl Driver {
    pub const fn name(self) -> &'static str {
        match self {
            Self::LocalTask => "local-task",
            Self::Metal => "metal",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "local-task" => Some(Self::LocalTask),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }
}
