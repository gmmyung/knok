extern crate alloc;

pub use crate::runtime::raw::{Input, Output, Outputs};

pub fn invoke_with_engine(
    engine: &crate::Engine,
    artifact: crate::GraphArtifact,
    inputs: &[Input<'_>],
) -> crate::Result<Outputs> {
    engine.invoke(artifact, inputs)
}

pub fn invoke_one_with_engine<T: Output>(
    engine: &crate::Engine,
    artifact: crate::GraphArtifact,
    inputs: &[Input<'_>],
) -> crate::Result<alloc::vec::Vec<T>> {
    engine.invoke_one(artifact, inputs)
}
