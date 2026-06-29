#![allow(unused_imports)]

use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::Cuda)]
fn forward(x: T1<f32, 4>) -> T1<f32, 4> {
    x
}

fn main() {}
