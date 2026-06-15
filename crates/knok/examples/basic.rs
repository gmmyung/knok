use knok::prelude::*;

#[knok::graph(backend = "llvm-cpu")]
fn forward(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    relu(x + y)
}

fn main() -> knok::Result<()> {
    let x = Tensor1::from_array([-1.0, 2.0, -3.0, 4.0]);
    let y = Tensor1::from_array([0.5, 1.0, 10.0, -10.0]);
    let output = forward(x, y)?;
    println!("{:?}", output.into_vec());
    Ok(())
}
