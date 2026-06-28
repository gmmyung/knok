extern crate alloc;

#[doc(hidden)]
pub fn invoke_one_with_engine<T: crate::runtime::raw::Output>(
    engine: &crate::Engine,
    artifact: crate::GraphArtifact,
    inputs: &[crate::runtime::raw::Input<'_>],
) -> crate::Result<alloc::vec::Vec<T>> {
    engine.invoke_one(artifact, inputs)
}
