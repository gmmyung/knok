use alloc::{vec, vec::Vec};
use core::fmt;

pub trait TensorElement: Copy + Clone + PartialEq + fmt::Debug {
    const ZERO: Self;
    const ONE: Self;
}

macro_rules! impl_tensor_element {
    ($($ty:ty),* $(,)?) => {
        $(
            impl TensorElement for $ty {
                const ZERO: Self = 0 as $ty;
                const ONE: Self = 1 as $ty;
            }
        )*
    };
}

impl_tensor_element!(f32, f64, i32, i64);

#[cfg(feature = "half")]
impl TensorElement for half::f16 {
    const ZERO: Self = half::f16::from_bits(0);
    const ONE: Self = half::f16::from_bits(0x3c00);
}

#[cfg(feature = "half")]
impl TensorElement for half::bf16 {
    const ZERO: Self = half::bf16::from_bits(0);
    const ONE: Self = half::bf16::from_bits(0x3f80);
}

#[derive(Clone, Debug, PartialEq)]
struct TensorData<T> {
    data: Vec<T>,
}

#[derive(Clone, PartialEq)]
pub struct Tensor1<T, const D0: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
pub struct Tensor2<T, const D0: usize, const D1: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
pub struct Tensor3<T, const D0: usize, const D1: usize, const D2: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
pub struct Tensor4<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize> {
    storage: TensorData<T>,
}

impl<T> TensorData<T> {
    fn from_vec<const R: usize>(data: Vec<T>, shape: &'static [usize; R]) -> crate::Result<Self> {
        let expected_len: usize = shape.iter().product();
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

    fn filled<const R: usize>(value: T, shape: &'static [usize; R]) -> Self
    where
        T: Clone,
    {
        let len: usize = shape.iter().product();
        Self {
            data: vec![value; len],
        }
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

    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0]),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.storage.data.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.storage.data.get_mut(index)
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

    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1]),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        (row < D0 && col < D1).then(|| &self.storage.data[row * D1 + col])
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        (row < D0 && col < D1).then(|| &mut self.storage.data[row * D1 + col])
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

    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2]),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    pub fn get(&self, d0: usize, d1: usize, d2: usize) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2).then(|| &self.storage.data[(d0 * D1 + d1) * D2 + d2])
    }

    pub fn get_mut(&mut self, d0: usize, d1: usize, d2: usize) -> Option<&mut T> {
        (d0 < D0 && d1 < D1 && d2 < D2).then(|| &mut self.storage.data[(d0 * D1 + d1) * D2 + d2])
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

    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2, D3]),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    pub fn get(&self, d0: usize, d1: usize, d2: usize, d3: usize) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3)
            .then(|| &self.storage.data[((d0 * D1 + d1) * D2 + d2) * D3 + d3])
    }

    pub fn get_mut(&mut self, d0: usize, d1: usize, d2: usize, d3: usize) -> Option<&mut T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3)
            .then(|| &mut self.storage.data[((d0 * D1 + d1) * D2 + d2) * D3 + d3])
    }
}

impl<T: TensorElement, const D0: usize> Tensor1<T, D0> {
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T: TensorElement, const D0: usize, const D1: usize> Tensor2<T, D0, D1> {
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T: TensorElement, const D0: usize, const D1: usize, const D2: usize> Tensor3<T, D0, D1, D2> {
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T: TensorElement, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    Tensor4<T, D0, D1, D2, D3>
{
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T, const D0: usize> TryFrom<Vec<T>> for Tensor1<T, D0> {
    type Error = crate::Error;

    fn try_from(data: Vec<T>) -> crate::Result<Self> {
        Self::from_vec(data)
    }
}

impl<T, const D0: usize, const D1: usize> TryFrom<Vec<T>> for Tensor2<T, D0, D1> {
    type Error = crate::Error;

    fn try_from(data: Vec<T>) -> crate::Result<Self> {
        Self::from_vec(data)
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize> TryFrom<Vec<T>>
    for Tensor3<T, D0, D1, D2>
{
    type Error = crate::Error;

    fn try_from(data: Vec<T>) -> crate::Result<Self> {
        Self::from_vec(data)
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize> TryFrom<Vec<T>>
    for Tensor4<T, D0, D1, D2, D3>
{
    type Error = crate::Error;

    fn try_from(data: Vec<T>) -> crate::Result<Self> {
        Self::from_vec(data)
    }
}

impl<T: fmt::Debug, const D0: usize> fmt::Debug for Tensor1<T, D0> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor1")
            .field("shape", &Self::SHAPE)
            .field("data", &self.storage.data)
            .finish()
    }
}

impl<T: fmt::Debug, const D0: usize, const D1: usize> fmt::Debug for Tensor2<T, D0, D1> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor2")
            .field("shape", &Self::SHAPE)
            .field("data", &self.storage.data)
            .finish()
    }
}

impl<T: fmt::Debug, const D0: usize, const D1: usize, const D2: usize> fmt::Debug
    for Tensor3<T, D0, D1, D2>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor3")
            .field("shape", &Self::SHAPE)
            .field("data", &self.storage.data)
            .finish()
    }
}

impl<T: fmt::Debug, const D0: usize, const D1: usize, const D2: usize, const D3: usize> fmt::Debug
    for Tensor4<T, D0, D1, D2, D3>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor4")
            .field("shape", &Self::SHAPE)
            .field("data", &self.storage.data)
            .finish()
    }
}
