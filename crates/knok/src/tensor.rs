use alloc::{vec, vec::Vec};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor1<T, const D0: usize> {
    data: Vec<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor2<T, const D0: usize, const D1: usize> {
    data: Vec<T>,
}

impl<T, const D0: usize> Tensor1<T, D0> {
    pub const SHAPE: &'static [usize] = &[D0];

    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        if data.len() != D0 {
            return Err(crate::Error::Shape {
                expected: Self::SHAPE,
                actual: vec![data.len()],
            });
        }
        Ok(Self { data })
    }

    pub fn from_array(data: [T; D0]) -> Self {
        Self { data: data.into() }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T, const D0: usize, const D1: usize> Tensor2<T, D0, D1> {
    pub const SHAPE: &'static [usize] = &[D0, D1];

    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        let expected_len = D0 * D1;
        if data.len() != expected_len {
            return Err(crate::Error::Shape {
                expected: Self::SHAPE,
                actual: vec![data.len()],
            });
        }
        Ok(Self { data })
    }

    pub fn from_array(data: [[T; D1]; D0]) -> Self {
        Self {
            data: data.into_iter().flat_map(IntoIterator::into_iter).collect(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}
