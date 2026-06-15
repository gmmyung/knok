#![no_std]

extern crate alloc;

use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
pub fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

pub fn artifact() -> knok::GraphArtifact {
    add4_artifact()
}
