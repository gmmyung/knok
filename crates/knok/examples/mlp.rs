use knok::prelude::*;
use knok::{Engine, RuntimeConfig};

#[knok::graph(backend = "llvm-cpu")]
fn mlp_block(
    x: Tensor2<f32, 1, 3>,
    w: Tensor2<f32, 3, 2>,
    b: Tensor2<f32, 1, 2>,
) -> Tensor2<f32, 1, 2> {
    relu(matmul(x, w) + b)
}

fn main() -> knok::Result<()> {
    let engine = Engine::new(RuntimeConfig::auto())?;
    let x = Tensor2::from_array([[1.0, 2.0, 3.0]]);
    let w = Tensor2::from_array([[1.0, -1.0], [0.5, 2.0], [-1.0, 0.25]]);
    let b = Tensor2::from_array([[0.25, -0.5]]);

    let output = mlp_block_run(&engine, x, w, b)?;

    assert_eq!(output.into_vec(), vec![0.0, 3.25]);
    Ok(())
}
