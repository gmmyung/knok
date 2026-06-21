use knok::prelude::*;
use knok::{Engine, RuntimeConfig};

#[knok::graph(backend = "llvm-cpu")]
fn add4_f64(x: Tensor1<f64, 4>, y: Tensor1<f64, 4>) -> Tensor1<f64, 4> {
    x + y
}

#[knok::graph(backend = "llvm-cpu")]
fn arithmetic4_i32(x: Tensor1<i32, 4>, y: Tensor1<i32, 4>) -> Tensor1<i32, 4> {
    ((x - y) * 2i32) / 4i32
}

#[knok::graph(backend = "llvm-cpu")]
fn add4_i64(x: Tensor1<i64, 4>, y: Tensor1<i64, 4>) -> Tensor1<i64, 4> {
    x + y
}

fn main() -> knok::Result<()> {
    let engine = Engine::new(RuntimeConfig::auto())?;

    let f64_output = add4_f64_run(
        &engine,
        Tensor1::from_array([1.0, 2.0, 3.0, 4.0]),
        Tensor1::from_array([10.0, 20.0, 30.0, 40.0]),
    )?;
    assert_eq!(f64_output.into_vec(), vec![11.0, 22.0, 33.0, 44.0]);

    let i32_output = arithmetic4_i32_run(
        &engine,
        Tensor1::from_array([10, 20, 30, 40]),
        Tensor1::from_array([2, 4, 6, 8]),
    )?;
    assert_eq!(i32_output.into_vec(), vec![4, 8, 12, 16]);

    let i64_output = add4_i64_run(
        &engine,
        Tensor1::from_array([1i64, 2, 3, 4]),
        Tensor1::from_array([10i64, 20, 30, 40]),
    )?;
    assert_eq!(i64_output.into_vec(), vec![11, 22, 33, 44]);

    Ok(())
}
