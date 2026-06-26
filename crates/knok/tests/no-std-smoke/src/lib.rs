#![no_std]

extern crate alloc;

use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
pub fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[knok::graph(backend = "llvm-cpu")]
pub fn add_sub4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> (Tensor1<f32, 4>, Tensor1<f32, 4>) {
    (x + y, x - y)
}

pub fn artifact() -> knok::GraphArtifact {
    add4_artifact()
}

pub fn multi_output_artifact() -> knok::GraphArtifact {
    add_sub4_artifact()
}

pub fn artifact_variant_count() -> usize {
    artifact().variants.len()
}

pub fn first_variant_driver() -> Option<&'static str> {
    artifact().first_variant().map(|variant| variant.driver)
}
