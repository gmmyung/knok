use knok::prelude::*;
use knok::Error;

#[test]
fn tensor3_from_array_and_vec_share_row_major_layout() {
    let from_array = Tensor3::from_array([[[1.0, 2.0], [3.0, 4.0]]]);
    let from_vec = Tensor3::<f32, 1, 2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    assert_eq!(from_array.as_slice(), from_vec.as_slice());
    assert_eq!(from_array.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor4_from_array_and_vec_share_row_major_layout() {
    let from_array = Tensor4::from_array([[[[1.0], [2.0]], [[3.0], [4.0]]]]);
    let from_vec = Tensor4::<f32, 1, 2, 2, 1>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    assert_eq!(from_array.as_slice(), from_vec.as_slice());
    assert_eq!(from_array.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn higher_rank_tensor_from_vec_validates_element_count() {
    let error = Tensor4::<f32, 1, 2, 2, 1>::from_vec(vec![1.0, 2.0, 3.0]).unwrap_err();

    assert!(matches!(
        error,
        Error::Shape {
            expected: &[1, 2, 2, 1],
            ..
        }
    ));
}

#[test]
fn tensor_convenience_constructors_work() {
    let zeros = Tensor2::<f32, 2, 2>::zeros();
    assert_eq!(zeros.as_slice(), &[0.0, 0.0, 0.0, 0.0]);

    let ones = Tensor3::<f32, 1, 2, 2>::ones();
    assert_eq!(ones.as_slice(), &[1.0, 1.0, 1.0, 1.0]);

    let f64_ones = Tensor1::<f64, 2>::ones();
    assert_eq!(f64_ones.as_slice(), &[1.0, 1.0]);

    let i64_zeros = Tensor2::<i64, 1, 2>::zeros();
    assert_eq!(i64_zeros.as_slice(), &[0, 0]);

    let filled = Tensor4::<i32, 1, 1, 2, 2>::filled(7);
    assert_eq!(filled.into_vec(), vec![7, 7, 7, 7]);
}

#[cfg(feature = "half")]
#[test]
fn half_tensor_convenience_constructors_work() {
    let f16_ones = Tensor1::<f16, 2>::ones();
    assert_eq!(
        f16_ones.as_slice(),
        &[f16::from_f32(1.0), f16::from_f32(1.0)]
    );

    let bf16_zeros = Tensor2::<bf16, 1, 2>::zeros();
    assert_eq!(
        bf16_zeros.as_slice(),
        &[bf16::from_f32(0.0), bf16::from_f32(0.0)]
    );
}

#[test]
fn tensor_try_from_vec_and_indexing_work() {
    let mut tensor = Tensor3::<f32, 1, 2, 2>::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    assert_eq!(tensor.get(0, 1, 0), Some(&3.0));
    assert_eq!(tensor.get(1, 0, 0), None);

    *tensor.get_mut(0, 1, 1).unwrap() = 9.0;
    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 9.0]);

    tensor.as_mut_slice()[0] = 5.0;
    assert_eq!(tensor.as_slice(), &[5.0, 2.0, 3.0, 9.0]);
}

#[test]
fn tensor_debug_includes_shape() {
    let tensor = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);

    let debug = format!("{tensor:?}");

    assert!(debug.contains("Tensor2"));
    assert!(debug.contains("shape"));
    assert!(debug.contains("[2, 2]"));
}
