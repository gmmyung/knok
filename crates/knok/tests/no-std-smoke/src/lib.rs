#![no_std]

knok::generated_graphs!(pub mod graphs, "knok_no_std_graphs.rs");

pub fn artifact() -> knok::GraphArtifact {
    graphs::add4::artifact()
}

pub fn multi_output_artifact() -> knok::GraphArtifact {
    graphs::add_sub4::artifact()
}

pub fn variant_count() -> usize {
    artifact().variants.len()
}

pub fn multi_output_count() -> usize {
    multi_output_artifact().output_descs.len()
}
