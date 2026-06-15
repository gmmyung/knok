use alloc::{vec, vec::Vec};

#[derive(Clone, Debug, PartialEq)]
struct TensorData<T> {
    data: Vec<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor1<T, const D0: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor2<T, const D0: usize, const D1: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor3<T, const D0: usize, const D1: usize, const D2: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor4<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize> {
    storage: TensorData<T>,
}

impl<T> TensorData<T> {
    fn from_vec<const R: usize>(data: Vec<T>, shape: &'static [usize; R]) -> crate::Result<Self> {
        let expected_len = shape.iter().product();
        if data.len() != expected_len {
            return Err(crate::Error::Shape {
                expected: shape,
                actual: vec![data.len()],
            });
        }
        Ok(Self { data })
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T, const D0: usize> Tensor1<T, D0> {
    pub const SHAPE: &'static [usize] = &[D0];

    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0])?,
        })
    }

    pub fn from_array(data: [T; D0]) -> Self {
        Self {
            storage: TensorData { data: data.into() },
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }
}

impl<T, const D0: usize, const D1: usize> Tensor2<T, D0, D1> {
    pub const SHAPE: &'static [usize] = &[D0, D1];

    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1])?,
        })
    }

    pub fn from_array(data: [[T; D1]; D0]) -> Self {
        Self {
            storage: TensorData {
                data: data.into_iter().flat_map(IntoIterator::into_iter).collect(),
            },
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize> Tensor3<T, D0, D1, D2> {
    pub const SHAPE: &'static [usize] = &[D0, D1, D2];

    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2])?,
        })
    }

    pub fn from_array(data: [[[T; D2]; D1]; D0]) -> Self {
        Self {
            storage: TensorData {
                data: data
                    .into_iter()
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .collect(),
            },
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    Tensor4<T, D0, D1, D2, D3>
{
    pub const SHAPE: &'static [usize] = &[D0, D1, D2, D3];

    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2, D3])?,
        })
    }

    pub fn from_array(data: [[[[T; D3]; D2]; D1]; D0]) -> Self {
        Self {
            storage: TensorData {
                data: data
                    .into_iter()
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .collect(),
            },
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }
}
