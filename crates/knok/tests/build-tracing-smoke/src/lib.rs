#![no_std]

knok::generated_graphs!(pub mod graphs);

pub fn artifact() -> knok::GraphArtifact {
    graphs::forward::artifact()
}

pub fn variant_count() -> usize {
    artifact().variants.len()
}
