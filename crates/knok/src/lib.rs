#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
//! Static-shape tensor graph frontend for compiling restricted Rust function
//! bodies to IREE VM bytecode at compile time.
//!
//! The primary entry point is `#[knok::graph]`, which turns a Rust function body into a
//! compiled graph artifact and generated typed wrappers:
//!
//! ```ignore
//! use knok::prelude::*;
//!
//! #[knok::graph(backend = Backend::LlvmCpu)]
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

/// Compiled graph artifact metadata.
pub mod artifact;
/// Backend and runtime driver selection types.
pub mod backend;
#[doc(hidden)]
pub mod __private {
    pub use crate::private::*;
}
/// Graph operations accepted inside `#[knok::graph]` bodies.
pub mod ops;
mod private;
/// Hosted runtime engine and raw invocation support.
pub mod runtime;
/// Static-rank host tensor containers.
pub mod tensor;

/// Common imports for graph definitions and host code.
pub mod prelude {
    #[cfg(feature = "half")]
    pub use crate::half::{bf16, f16};
    pub use crate::tensor::{
        Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, TensorElement,
    };
    #[cfg(feature = "macros")]
    pub use crate::{graph, mlir_model};
    pub use crate::{Backend, Driver};
}

pub use artifact::{DType, GraphArtifact, GraphArtifactVariant, TensorDesc};
pub use backend::{Backend, Driver, SUPPORTED_BACKENDS};
pub use runtime::{Engine, RuntimeConfig};

/// Result type used by `knok` APIs.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
/// Error type returned by tensor constructors and hosted runtime helpers.
pub enum Error {
    /// Error returned by the underlying `eerie` runtime.
    #[cfg(feature = "host-runtime")]
    Runtime(eerie::runtime::RuntimeError),
    /// Tensor shape did not match the statically declared shape.
    Shape {
        /// Expected static shape.
        expected: &'static [usize],
        /// Actual shape or element-count diagnostic.
        actual: alloc::vec::Vec<usize>,
    },
    /// Raw invocation received the wrong number of inputs.
    InputCountMismatch {
        /// Number of inputs recorded in the artifact metadata.
        expected: usize,
        /// Number of inputs supplied by the caller.
        actual: usize,
    },
    /// Raw invocation received an input with the wrong dtype.
    InputDTypeMismatch {
        /// Input index.
        index: usize,
        /// Expected dtype recorded in the artifact metadata.
        expected: DType,
        /// Actual dtype supplied by the caller.
        actual: DType,
    },
    /// The requested backend is not supported by this crate.
    UnsupportedBackend(&'static str),
    /// No artifact variant matches the engine driver.
    MissingArtifactVariant {
        /// Function name stored in the artifact.
        function_name: &'static str,
        /// Engine driver that was requested.
        driver: alloc::string::String,
    },
    /// The artifact does not contain any backend variants.
    MissingDefaultArtifactVariant {
        /// Function name stored in the artifact.
        function_name: &'static str,
    },
    /// The selected artifact variant expects a different runtime driver.
    RuntimeDriverMismatch {
        /// Backend associated with the selected variant.
        backend: &'static str,
        /// Driver required by the selected variant.
        expected_driver: &'static str,
        /// Driver used by the engine.
        actual_driver: alloc::string::String,
    },
    /// The reusable engine's module cache lock was poisoned.
    EngineLockPoisoned,
    /// Invocation produced the wrong number of outputs.
    OutputCountMismatch {
        /// Expected output count.
        expected: usize,
        /// Actual output count.
        actual: usize,
    },
    /// Output index was outside the returned output list.
    OutputIndexOutOfBounds {
        /// Requested output index.
        index: usize,
        /// Number of outputs available.
        len: usize,
    },
    /// Hosted runtime execution was requested without the `host-runtime` feature.
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
