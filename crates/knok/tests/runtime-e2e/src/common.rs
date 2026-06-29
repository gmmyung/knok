use core::fmt::Debug;

pub fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1.0e-4_f32.max(expected.abs() * 1.0e-4);
        assert!(
            (actual - expected).abs() <= tolerance,
            "value mismatch at {index}: actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }
}

pub fn assert_exact<T: PartialEq + Debug>(actual: &[T], expected: &[T]) {
    assert_eq!(actual, expected);
}
