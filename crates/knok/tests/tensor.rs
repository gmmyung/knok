use knok::prelude::*;
use knok::Error;

#[test]
fn tensor0_stores_one_scalar() {
    let mut tensor = Tensor0::from_scalar(3.0);

    assert_eq!(Tensor0::<f32>::SHAPE, &[]);
    assert_eq!(tensor.as_slice(), &[3.0]);
    assert_eq!(tensor.get(), &3.0);

    *tensor.get_mut() = 4.0;
    assert_eq!(tensor.into_vec(), vec![4.0]);
    assert_eq!(Tensor0::from_array([5.0]).into_vec(), vec![5.0]);
}

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
fn tensor5_from_array_and_vec_share_row_major_layout() {
    let from_array = Tensor5::from_array([[[[[1.0], [2.0]], [[3.0], [4.0]]]]]);
    let from_vec = Tensor5::<f32, 1, 1, 2, 2, 1>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    assert_eq!(Tensor5::<f32, 1, 1, 2, 2, 1>::SHAPE, &[1, 1, 2, 2, 1]);
    assert_eq!(from_array.as_slice(), from_vec.as_slice());
    assert_eq!(from_array.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor6_from_array_and_vec_share_row_major_layout() {
    let from_array = Tensor6::from_array([[[[[[1.0, 2.0], [3.0, 4.0]]]]]]);
    let from_vec = Tensor6::<f32, 1, 1, 1, 1, 2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    assert_eq!(Tensor6::<f32, 1, 1, 1, 1, 2, 2>::SHAPE, &[1, 1, 1, 1, 2, 2]);
    assert_eq!(from_array.as_slice(), from_vec.as_slice());
    assert_eq!(from_array.into_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn higher_rank_tensor_from_vec_validates_element_count() {
    let error = Tensor6::<f32, 1, 1, 1, 2, 2, 1>::from_vec(vec![1.0, 2.0, 3.0]).unwrap_err();

    assert!(matches!(
        error,
        Error::Shape {
            expected: &[1, 1, 1, 2, 2, 1],
            ..
        }
    ));
}

#[test]
fn tensor0_from_vec_validates_single_element() {
    let tensor = Tensor0::<f32>::from_vec(vec![1.0]).unwrap();
    assert_eq!(tensor.into_vec(), vec![1.0]);

    let error = Tensor0::<f32>::from_vec(Vec::new()).unwrap_err();
    assert!(matches!(error, Error::Shape { expected: &[], .. }));
}

#[test]
fn tensor_convenience_constructors_work() {
    let zeros = Tensor2::<f32, 2, 2>::zeros();
    assert_eq!(zeros.as_slice(), &[0.0, 0.0, 0.0, 0.0]);

    let scalar_one = Tensor0::<i32>::ones();
    assert_eq!(scalar_one.into_vec(), vec![1]);

    let ones = Tensor3::<f32, 1, 2, 2>::ones();
    assert_eq!(ones.as_slice(), &[1.0, 1.0, 1.0, 1.0]);

    let f64_ones = Tensor1::<f64, 2>::ones();
    assert_eq!(f64_ones.as_slice(), &[1.0, 1.0]);

    let i64_zeros = Tensor2::<i64, 1, 2>::zeros();
    assert_eq!(i64_zeros.as_slice(), &[0, 0]);

    let filled = Tensor4::<i32, 1, 1, 2, 2>::filled(7);
    assert_eq!(filled.into_vec(), vec![7, 7, 7, 7]);

    let tensor5_ones = Tensor5::<i32, 1, 1, 1, 2, 2>::ones();
    assert_eq!(tensor5_ones.as_slice(), &[1, 1, 1, 1]);

    let tensor6_zeros = Tensor6::<i64, 1, 1, 1, 1, 2, 2>::zeros();
    assert_eq!(tensor6_zeros.as_slice(), &[0, 0, 0, 0]);
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
    let mut tensor = Tensor6::<f32, 1, 1, 1, 1, 2, 2>::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    assert_eq!(tensor.get(0, 0, 0, 0, 1, 0), Some(&3.0));
    assert_eq!(tensor.get(0, 0, 0, 0, 2, 0), None);

    *tensor.get_mut(0, 0, 0, 0, 1, 1).unwrap() = 9.0;
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
