#[derive(Clone, Copy, Debug)]
pub struct GraphArtifact {
    pub function_name: &'static str,
    pub input_descs: &'static [TensorDesc],
    pub output_descs: &'static [TensorDesc],
    pub variants: &'static [GraphArtifactVariant],
}

impl GraphArtifact {
    #[cfg(feature = "host-runtime")]
    pub(crate) fn has_typed_signature(&self) -> bool {
        !self.input_descs.is_empty() || !self.output_descs.is_empty()
    }

    pub fn variant_for_driver(&self, driver: &str) -> Option<GraphArtifactVariant> {
        self.variants
            .iter()
            .copied()
            .find(|variant| variant.driver == driver)
    }

    pub fn first_variant(&self) -> Option<GraphArtifactVariant> {
        self.variants.first().copied()
    }
}

/// Static tensor metadata recorded in a compiled graph artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorDesc {
    pub elem: DType,
    pub shape: &'static [usize],
}

impl TensorDesc {
    pub const fn new(elem: DType, shape: &'static [usize]) -> Self {
        Self { elem, shape }
    }
}

/// Element type metadata recorded in a compiled graph artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Bool,
    F32,
    F64,
    F16,
    BF16,
    I32,
    I64,
}

#[derive(Clone, Copy, Debug)]
pub struct GraphArtifactVariant {
    pub vmfb: &'static [u8],
    pub backend: &'static str,
    pub driver: &'static str,
    pub compile_flags: &'static [&'static str],
}
