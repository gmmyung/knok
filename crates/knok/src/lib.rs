#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use knok_macros::{graph, mlir_model};

pub mod artifact;
#[doc(hidden)]
pub mod __private {
    pub use crate::private::*;
}
pub mod private;
pub mod runtime;
pub mod tensor;

pub mod prelude {
    pub use crate::tensor::{Tensor1, Tensor2};
    pub use crate::{graph, mlir_model};
}

pub use artifact::{GraphArtifact, GraphArtifactVariant};
pub use runtime::{Engine, RuntimeConfig};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    #[cfg(feature = "host-runtime")]
    Runtime(eerie::runtime::error::RuntimeError),
    Shape {
        expected: &'static [usize],
        actual: alloc::vec::Vec<usize>,
    },
    UnsupportedBackend(&'static str),
    MissingArtifactVariant {
        function_name: &'static str,
        driver: alloc::string::String,
    },
    MissingDefaultArtifactVariant {
        function_name: &'static str,
    },
    RuntimeDriverMismatch {
        backend: &'static str,
        expected_driver: &'static str,
        actual_driver: alloc::string::String,
    },
    EngineLockPoisoned,
    MissingOutput,
    HostedRuntimeDisabled,
}

#[cfg(feature = "host-runtime")]
impl From<eerie::runtime::error::RuntimeError> for Error {
    fn from(error: eerie::runtime::error::RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "host-runtime")]
            Self::Runtime(error) => write!(formatter, "runtime error: {error}"),
            Self::Shape { expected, actual } => {
                write!(
                    formatter,
                    "tensor shape mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::UnsupportedBackend(backend) => {
                write!(formatter, "unsupported backend: {backend}")
            }
            Self::MissingArtifactVariant {
                function_name,
                driver,
            } => {
                write!(
                    formatter,
                    "no artifact variant for function {function_name} and runtime driver {driver}"
                )
            }
            Self::MissingDefaultArtifactVariant { function_name } => {
                write!(
                    formatter,
                    "artifact for function {function_name} has no compiled variants"
                )
            }
            Self::RuntimeDriverMismatch {
                backend,
                expected_driver,
                actual_driver,
            } => {
                write!(
                    formatter,
                    "runtime driver mismatch for backend {backend}: expected {expected_driver}, got {actual_driver}"
                )
            }
            Self::EngineLockPoisoned => formatter.write_str("runtime engine cache lock poisoned"),
            Self::MissingOutput => formatter.write_str("missing runtime output"),
            Self::HostedRuntimeDisabled => formatter.write_str("host runtime feature is disabled"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
