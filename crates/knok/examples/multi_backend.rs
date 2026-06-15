use knok::prelude::*;
use knok::{Engine, RuntimeConfig};

#[cfg(target_os = "macos")]
#[knok::graph(backends = [
    backend("llvm-cpu", driver = "local-task"),
    backend("metal-spirv", driver = "metal"),
])]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

#[cfg(not(target_os = "macos"))]
#[knok::graph(backends = [
    backend("llvm-cpu", driver = "local-task"),
])]
fn add4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    x + y
}

fn main() -> knok::Result<()> {
    #[cfg(target_os = "macos")]
    let engine = Engine::new(RuntimeConfig::driver("metal"))?;

    #[cfg(not(target_os = "macos"))]
    let engine = Engine::new(RuntimeConfig::auto())?;

    let x = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);
    let y = Tensor1::from_array([10.0, 20.0, 30.0, 40.0]);
    let output = add4_run(&engine, x, y)?;

    assert_eq!(output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    Ok(())
}
