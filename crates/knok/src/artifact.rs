#[derive(Clone, Copy, Debug)]
pub struct GraphArtifact {
    pub vmfb: &'static [u8],
    pub function_name: &'static str,
    pub backend: &'static str,
    pub input_shapes: &'static [&'static [usize]],
    pub output_shape: &'static [usize],
}
