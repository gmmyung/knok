//! Static metadata for generated graph artifacts.

/// Embedded graph artifact metadata and VMFB variants.
///
/// Generated graph modules expose an `artifact()` function returning this
/// value. Runtime wrappers use it to validate typed inputs, select a backend
/// variant for an engine, and resolve the compiled function
/// inside the embedded VMFB module.
#[derive(Clone, Copy, Debug)]
pub struct GraphArtifact {
    /// Function name inside the IREE VM module.
    function_name: &'static str,
    /// Input tensor descriptors for typed runtime validation.
    input_descs: &'static [TensorDesc],
    /// Output tensor descriptors for typed runtime validation.
    output_descs: &'static [TensorDesc],
    /// Backend-specific VMFB variants.
    variants: &'static [GraphArtifactVariant],
}

impl GraphArtifact {
    /// Declares the complete static signature of a compiled function.
    /// Empty descriptor slices mean zero inputs or outputs.
    pub const fn new(
        function_name: &'static str,
        input_descs: &'static [TensorDesc],
        output_descs: &'static [TensorDesc],
        variants: &'static [GraphArtifactVariant],
    ) -> Self {
        Self {
            function_name,
            input_descs,
            output_descs,
            variants,
        }
    }

    /// Returns the exported VM function name.
    pub const fn function_name(&self) -> &'static str {
        self.function_name
    }
    /// Returns the complete input signature.
    pub const fn input_descs(&self) -> &'static [TensorDesc] {
        self.input_descs
    }
    /// Returns the complete output signature.
    pub const fn output_descs(&self) -> &'static [TensorDesc] {
        self.output_descs
    }
    /// Returns the compiled backend variants.
    pub const fn variants(&self) -> &'static [GraphArtifactVariant] {
        self.variants
    }

    /// Returns the first variant compatible with the selected runtime driver.
    pub fn variant_for_driver(&self, driver: crate::Driver) -> Option<GraphArtifactVariant> {
        self.variants
            .iter()
            .copied()
            .find(|variant| variant.backend.supports_driver(driver))
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

/// One compiled VMFB variant for a backend.
///
/// A generated graph can embed multiple variants over time. The runtime selects
/// a variant whose backend supports the engine's driver.
#[derive(Clone, Copy, Debug)]
pub struct GraphArtifactVariant {
    /// Embedded VMFB bytes.
    vmfb: &'static [u8],
    /// IREE target backend name used at compile time.
    backend: crate::Backend,
    /// IREE compiler flags used to produce this variant.
    compile_flags: &'static [&'static str],
}

impl GraphArtifactVariant {
    /// Describes one compiled variant. Compatible drivers are determined by its backend.
    pub const fn new(
        vmfb: &'static [u8],
        backend: crate::Backend,
        compile_flags: &'static [&'static str],
    ) -> Self {
        Self {
            vmfb,
            backend,
            compile_flags,
        }
    }
    /// Returns the embedded VMFB bytes.
    pub const fn vmfb(&self) -> &'static [u8] {
        self.vmfb
    }
    /// Returns the compilation backend.
    pub const fn backend(&self) -> crate::Backend {
        self.backend
    }
    /// Returns the compiler flags recorded at build time.
    pub const fn compile_flags(&self) -> &'static [&'static str] {
        self.compile_flags
    }
}
