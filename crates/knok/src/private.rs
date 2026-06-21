extern crate alloc;

pub fn invoke<T: crate::runtime::RuntimeElement>(
    artifact: crate::GraphArtifact,
    inputs: &[(&[usize], &[T])],
) -> crate::Result<alloc::vec::Vec<T>> {
    let engine = crate::Engine::for_artifact(artifact)?;
    invoke_with_engine(&engine, artifact, inputs)
}

pub fn invoke_with_engine<T: crate::runtime::RuntimeElement>(
    engine: &crate::Engine,
    artifact: crate::GraphArtifact,
    inputs: &[(&[usize], &[T])],
) -> crate::Result<alloc::vec::Vec<T>> {
    engine.invoke(artifact, inputs)
}
