#![cfg_attr(not(feature = "std"), no_std)]
//! Static-shape tensor containers and runtime wrappers for build-time traced
//! IREE VM bytecode artifacts.
//!
//! Graphs are authored from `build.rs` with `knok-build`, which executes traced
//! Rust functions on the compile host and writes generated artifact wrappers to
//! `OUT_DIR`. Target crates import those wrappers with `generated_graphs!`:
//!
//! ```ignore
//! use knok::prelude::*;
//!
//! knok::generated_graphs!(pub mod graphs);
//! ```
//!
//! For repeated hosted inference, construct one [`Engine`] and call generated
//! `graphs::<name>::run(&engine, ...)` wrappers. Local MLIR files can be embedded
//! with `knok::mlir_model!`.
//!
//! With default features disabled, `knok` is `no_std + alloc`; hosted runtime
//! execution is unavailable on the target unless the runtime feature set is
//! enabled.

extern crate alloc;

#[cfg(feature = "macros")]
pub use knok_macros::mlir_model;

#[cfg(feature = "half")]
pub use half;

pub mod artifact;
pub mod backend;
#[doc(hidden)]
pub mod __private {
    pub use crate::private::*;
}
mod private;
pub mod runtime;
pub mod tensor;

pub mod prelude {
    #[cfg(feature = "half")]
    pub use crate::half::{bf16, f16};
    #[cfg(feature = "macros")]
    pub use crate::mlir_model;
    pub use crate::tensor::{
        Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, TensorElement,
    };
    pub use crate::{Backend, Driver};
}

#[macro_export]
macro_rules! generated_graphs {
    () => {
        include!(concat!(env!("OUT_DIR"), "/knok_graphs.rs"));
    };
    ($file:literal) => {
        include!(concat!(env!("OUT_DIR"), "/", $file));
    };
    (mod $name:ident) => {
        mod $name {
            include!(concat!(env!("OUT_DIR"), "/knok_graphs.rs"));
        }
    };
    (mod $name:ident, $file:literal) => {
        mod $name {
            include!(concat!(env!("OUT_DIR"), "/", $file));
        }
    };
    (pub mod $name:ident) => {
        pub mod $name {
            include!(concat!(env!("OUT_DIR"), "/knok_graphs.rs"));
        }
    };
    (pub mod $name:ident, $file:literal) => {
        pub mod $name {
            include!(concat!(env!("OUT_DIR"), "/", $file));
        }
    };
}

pub use artifact::{DType, GraphArtifact, GraphArtifactVariant, TensorDesc};
pub use backend::{Backend, Driver, SUPPORTED_BACKENDS};
pub use runtime::{Engine, RuntimeConfig};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    #[cfg(feature = "host-runtime")]
    Runtime(eerie::runtime::RuntimeError),
    Shape {
        expected: &'static [usize],
        actual: alloc::vec::Vec<usize>,
    },
    InputCountMismatch {
        expected: usize,
        actual: usize,
    },
    InputDTypeMismatch {
        index: usize,
        expected: DType,
        actual: DType,
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
    OutputCountMismatch {
        expected: usize,
        actual: usize,
    },
    OutputIndexOutOfBounds {
        index: usize,
        len: usize,
    },
    HostedRuntimeDisabled,
}

#[cfg(feature = "host-runtime")]
impl From<eerie::runtime::RuntimeError> for Error {
    fn from(error: eerie::runtime::RuntimeError) -> Self {
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
            Self::InputCountMismatch { expected, actual } => {
                write!(
                    formatter,
                    "runtime input count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InputDTypeMismatch {
                index,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "runtime input {index} dtype mismatch: expected {expected:?}, got {actual:?}"
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
            Self::OutputCountMismatch { expected, actual } => {
                write!(
                    formatter,
                    "runtime output count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::OutputIndexOutOfBounds { index, len } => {
                write!(
                    formatter,
                    "runtime output index out of bounds: index {index}, len {len}"
                )
            }
            Self::HostedRuntimeDisabled => formatter.write_str("host runtime feature is disabled"),
        }
    }
}

impl core::error::Error for Error {}
