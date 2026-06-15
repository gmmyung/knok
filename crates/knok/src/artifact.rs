#[derive(Clone, Copy, Debug)]
pub struct GraphArtifact {
    pub function_name: &'static str,
    pub input_shapes: &'static [&'static [usize]],
    pub output_shape: &'static [usize],
    pub variants: &'static [GraphArtifactVariant],
}

impl GraphArtifact {
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

#[derive(Clone, Copy, Debug)]
pub struct GraphArtifactVariant {
    pub vmfb: &'static [u8],
    pub backend: &'static str,
    pub driver: &'static str,
    pub compile_flags: &'static [&'static str],
}
