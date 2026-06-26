#![cfg(target_os = "macos")]

use knok::prelude::*;
use knok::{Engine, RuntimeConfig};

#[knok::graph(backend = Backend::MetalSpirv)]
fn add4_metal(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[knok::graph(backends = [
    backend(Backend::LlvmCpu, driver = Driver::LocalTask),
    backend(Backend::MetalSpirv, driver = Driver::Metal),
])]
fn add4_cpu_metal(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[test]
fn metal_add_graph_runs() {
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);
    let output = add4_metal(x, y).unwrap();
    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn metal_selects_variant_from_backend_bundle() {
    let engine = Engine::new(RuntimeConfig::driver(Driver::Metal)).unwrap();
    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);
    let output = add4_cpu_metal_run(&engine, x, y).unwrap();

    let artifact = add4_cpu_metal_artifact();
    assert_eq!(artifact.variants.len(), 2);
    assert!(artifact.variant_for_driver("local-task").is_some());
    assert!(artifact.variant_for_driver("metal").is_some());
    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}
