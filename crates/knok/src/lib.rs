#![cfg_attr(not(feature = "std"), no_std)]
//! Static-shape tensor graph frontend for compiling restricted Rust function
//! bodies to IREE VM bytecode at compile time.
//!
//! The primary entry point is `#[knok::graph]`, which turns a Rust function body into a
//! compiled graph artifact and generated typed wrappers:
//!
//! ```ignore
//! use knok::prelude::*;
//!
//! #[knok::graph(backend = "llvm-cpu")]
//! fn forward(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
//!     relu(x + y)
//! }
//! ```
//!
//! For repeated hosted inference, construct one [`Engine`] and call the
//! generated `forward_run(&engine, ...)` wrapper. Local MLIR files can be
//! embedded with `knok::mlir_model!`.
//!
//! With default features disabled, `knok` is `no_std + alloc`; proc macros still
//! run on the compile host, while hosted runtime execution is unavailable on the
//! target unless the runtime feature set is enabled.

extern crate alloc;

#[cfg(feature = "macros")]
pub use knok_macros::{graph, mlir_model};

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
    pub use crate::tensor::{Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, TensorElement};
    #[cfg(feature = "macros")]
    pub use crate::{graph, mlir_model};
}

pub use artifact::{GraphArtifact, GraphArtifactVariant};
pub use backend::{Backend, SUPPORTED_BACKENDS};
pub use runtime::{Engine, RuntimeConfig, RuntimeInput, RuntimeOutputs};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    #[cfg(feature = "host-runtime")]
    Runtime(eerie::runtime::RuntimeError),
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

#[cfg(feature = "std")]
impl std::error::Error for Error {}
