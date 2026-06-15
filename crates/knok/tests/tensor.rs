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
