#![allow(unused_imports)]

use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: T1<f32, 4>) {
    let _ = x;
}

fn main() {}
