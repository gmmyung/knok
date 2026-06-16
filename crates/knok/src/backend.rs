#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    LlvmCpu,
    MetalSpirv,
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
            Self::LlvmCpu => "local-task",
            Self::MetalSpirv => "metal",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "llvm-cpu" => Some(Self::LlvmCpu),
            "metal-spirv" => Some(Self::MetalSpirv),
            _ => None,
        }
    }

    pub fn supports_driver(self, driver: &str) -> bool {
        self.default_driver() == driver
    }
}

pub const SUPPORTED_BACKENDS: &[Backend] = &[Backend::LlvmCpu, Backend::MetalSpirv];
