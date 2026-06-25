extern crate alloc;

pub fn invoke_one_with_engine<T: crate::runtime::RuntimeOutput>(
    engine: &crate::Engine,
    artifact: crate::GraphArtifact,
    inputs: &[crate::runtime::RuntimeInput<'_>],
) -> crate::Result<alloc::vec::Vec<T>> {
    engine.invoke_one(artifact, inputs)
}
