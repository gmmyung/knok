use knok::prelude::*;
use knok::{Engine, RuntimeConfig};

#[knok::graph(backend = Backend::LlvmCpu)]
fn classifier_head(logits: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
    softmax(logits)
}

#[knok::graph(backend = Backend::LlvmCpu)]
fn predicted_class(logits: Tensor1<f32, 4>) -> Tensor0<i64> {
    argmax(logits)
}

fn main() -> knok::Result<()> {
    let engine = Engine::new(RuntimeConfig::auto())?;
    let logits = Tensor1::from_array([1.0, 2.0, 3.0, 4.0]);

    let probabilities = classifier_head_run(&engine, logits.clone())?;
    let class = predicted_class_run(&engine, logits)?;

    assert_close(
        &probabilities.into_vec(),
        &[0.032058604, 0.08714432, 0.23688284, 0.6439143],
    );
    assert_eq!(class.into_vec(), vec![3i64]);
    Ok(())
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-4);
    }
}
