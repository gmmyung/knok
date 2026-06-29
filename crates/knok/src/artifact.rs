//! Static metadata for generated graph artifacts.

/// Embedded graph artifact metadata and VMFB variants.
///
/// Generated graph modules expose an `artifact()` function returning this
/// value. Runtime wrappers use it to validate typed inputs, select a backend
/// variant for an [`Engine`](crate::Engine), and resolve the compiled function
/// inside the embedded VMFB module.
#[derive(Clone, Copy, Debug)]
pub struct GraphArtifact {
    /// Function name inside the IREE VM module.
    pub function_name: &'static str,
    /// Input tensor descriptors for typed runtime validation.
    pub input_descs: &'static [TensorDesc],
    /// Output tensor descriptors for typed runtime validation.
    pub output_descs: &'static [TensorDesc],
    /// Backend-specific VMFB variants.
    pub variants: &'static [GraphArtifactVariant],
}

impl GraphArtifact {
    #[cfg(feature = "runtime")]
    pub(crate) fn has_typed_signature(&self) -> bool {
        !self.input_descs.is_empty() || !self.output_descs.is_empty()
    }

    /// Returns the artifact variant matching a runtime driver name.
    pub fn variant_for_driver(&self, driver: &str) -> Option<GraphArtifactVariant> {
        self.variants
            .iter()
            .copied()
            .find(|variant| variant.driver == driver)
    }

    /// Returns the first available artifact variant.
    ///
    /// Generated one-shot wrappers use this as the default variant when they
    /// create a convenience engine.
    pub fn first_variant(&self) -> Option<GraphArtifactVariant> {
        self.variants.first().copied()
    }
}

/// Static tensor metadata recorded in a compiled graph artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorDesc {
    /// Element dtype.
    pub elem: DType,
    /// Static tensor shape.
    pub shape: &'static [usize],
}

impl TensorDesc {
    /// Creates tensor metadata from dtype and shape.
    pub const fn new(elem: DType, shape: &'static [usize]) -> Self {
        Self { elem, shape }
    }
}

/// Element type metadata recorded in a compiled graph artifact.
///
/// `DType` is metadata only; tensor values are still represented by concrete
/// Rust element types such as `f32`, `i64`, or `bool`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    /// Boolean values represented as IREE bool8 tensors.
    Bool,
    /// 32-bit floating point values.
    F32,
    /// 64-bit floating point values.
    F64,
    /// IEEE 16-bit floating point values.
    F16,
    /// Brain floating point values.
    BF16,
    /// 32-bit signed integer values.
    I32,
    /// 64-bit signed integer values.
    I64,
}

/// One compiled VMFB variant for a backend/driver pair.
///
/// A generated graph can embed multiple variants over time. The runtime selects
/// the variant whose `driver` matches the active [`Engine`](crate::Engine).
#[derive(Clone, Copy, Debug)]
pub struct GraphArtifactVariant {
    /// Embedded VMFB bytes.
    pub vmfb: &'static [u8],
    /// IREE target backend name used at compile time.
    pub backend: &'static str,
    /// IREE runtime driver expected for execution.
    pub driver: &'static str,
    /// IREE compiler flags used to produce this variant.
    pub compile_flags: &'static [&'static str],
}
