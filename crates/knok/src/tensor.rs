use alloc::{vec, vec::Vec};
use core::fmt;

/// Element type that supports host-side zero and one tensor constructors.
pub trait TensorElement: Copy + Clone + PartialEq + fmt::Debug {
    /// Additive identity used by `TensorN::zeros`.
    const ZERO: Self;
    /// Multiplicative identity used by `TensorN::ones`.
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

impl TensorElement for bool {
    const ZERO: Self = false;
    const ONE: Self = true;
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
/// Rank-0 scalar tensor.
pub struct Tensor0<T> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
/// Rank-1 tensor with shape `[D0]`.
pub struct Tensor1<T, const D0: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
/// Rank-2 tensor with shape `[D0, D1]`.
pub struct Tensor2<T, const D0: usize, const D1: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
/// Rank-3 tensor with shape `[D0, D1, D2]`.
pub struct Tensor3<T, const D0: usize, const D1: usize, const D2: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
/// Rank-4 tensor with shape `[D0, D1, D2, D3]`.
pub struct Tensor4<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
/// Rank-5 tensor with shape `[D0, D1, D2, D3, D4]`.
pub struct Tensor5<
    T,
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
> {
    storage: TensorData<T>,
}

#[derive(Clone, PartialEq)]
/// Rank-6 tensor with shape `[D0, D1, D2, D3, D4, D5]`.
pub struct Tensor6<
    T,
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
> {
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

impl<T> Tensor0<T> {
    /// Static shape of this tensor type.
    pub const SHAPE: &'static [usize] = &[];

    /// Creates a scalar tensor from a vector containing exactly one value.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[])?,
        })
    }

    /// Creates a scalar tensor from one value.
    pub fn from_scalar(value: T) -> Self {
        Self {
            storage: TensorData { data: vec![value] },
        }
    }

    /// Creates a scalar tensor from a one-element array.
    pub fn from_array(data: [T; 1]) -> Self {
        Self {
            storage: TensorData { data: data.into() },
        }
    }

    /// Creates a scalar tensor filled with `value`.
    pub fn filled(value: T) -> Self {
        Self {
            storage: TensorData { data: vec![value] },
        }
    }

    /// Returns the row-major tensor storage as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns the row-major tensor storage as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns its row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns the scalar value.
    pub fn get(&self) -> &T {
        &self.storage.data[0]
    }

    /// Returns the scalar value mutably.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.storage.data[0]
    }
}

impl<T, const D0: usize> Tensor1<T, D0> {
    /// Static shape of this tensor type.
    pub const SHAPE: &'static [usize] = &[D0];

    /// Creates a tensor from row-major storage with exactly `D0` elements.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0])?,
        })
    }

    /// Creates a tensor from a nested array matching the static shape.
    pub fn from_array(data: [T; D0]) -> Self {
        Self {
            storage: TensorData { data: data.into() },
        }
    }

    /// Creates a tensor whose elements are all `value`.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0]),
        }
    }

    /// Returns the row-major tensor storage as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns the row-major tensor storage as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns its row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns the element at `index`, or `None` when out of bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.storage.data.get(index)
    }

    /// Returns the element at `index` mutably, or `None` when out of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.storage.data.get_mut(index)
    }
}

impl<T, const D0: usize, const D1: usize> Tensor2<T, D0, D1> {
    /// Static shape of this tensor type.
    pub const SHAPE: &'static [usize] = &[D0, D1];

    /// Creates a tensor from row-major storage with exactly `D0 * D1` elements.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1])?,
        })
    }

    /// Creates a tensor from a nested array matching the static shape.
    pub fn from_array(data: [[T; D1]; D0]) -> Self {
        Self {
            storage: TensorData {
                data: data.into_iter().flat_map(IntoIterator::into_iter).collect(),
            },
        }
    }

    /// Creates a tensor whose elements are all `value`.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1]),
        }
    }

    /// Returns the row-major tensor storage as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns the row-major tensor storage as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns its row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns the element at `[row, col]`, or `None` when out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        (row < D0 && col < D1).then(|| &self.storage.data[row * D1 + col])
    }

    /// Returns the element at `[row, col]` mutably, or `None` when out of bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        (row < D0 && col < D1).then(|| &mut self.storage.data[row * D1 + col])
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize> Tensor3<T, D0, D1, D2> {
    /// Static shape of this tensor type.
    pub const SHAPE: &'static [usize] = &[D0, D1, D2];

    /// Creates a tensor from row-major storage with exactly `D0 * D1 * D2` elements.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2])?,
        })
    }

    /// Creates a tensor from a nested array matching the static shape.
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

    /// Creates a tensor whose elements are all `value`.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2]),
        }
    }

    /// Returns the row-major tensor storage as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns the row-major tensor storage as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns its row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns the element at `[d0, d1, d2]`, or `None` when out of bounds.
    pub fn get(&self, d0: usize, d1: usize, d2: usize) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2).then(|| &self.storage.data[(d0 * D1 + d1) * D2 + d2])
    }

    /// Returns the element at `[d0, d1, d2]` mutably, or `None` when out of bounds.
    pub fn get_mut(&mut self, d0: usize, d1: usize, d2: usize) -> Option<&mut T> {
        (d0 < D0 && d1 < D1 && d2 < D2).then(|| &mut self.storage.data[(d0 * D1 + d1) * D2 + d2])
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    Tensor4<T, D0, D1, D2, D3>
{
    /// Static shape of this tensor type.
    pub const SHAPE: &'static [usize] = &[D0, D1, D2, D3];

    /// Creates a tensor from row-major storage with exactly `D0 * D1 * D2 * D3` elements.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2, D3])?,
        })
    }

    /// Creates a tensor from a nested array matching the static shape.
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

    /// Creates a tensor whose elements are all `value`.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2, D3]),
        }
    }

    /// Returns the row-major tensor storage as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns the row-major tensor storage as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns its row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns the element at `[d0, d1, d2, d3]`, or `None` when out of bounds.
    pub fn get(&self, d0: usize, d1: usize, d2: usize, d3: usize) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3)
            .then(|| &self.storage.data[((d0 * D1 + d1) * D2 + d2) * D3 + d3])
    }

    /// Returns the element at `[d0, d1, d2, d3]` mutably, or `None` when out of bounds.
    pub fn get_mut(&mut self, d0: usize, d1: usize, d2: usize, d3: usize) -> Option<&mut T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3)
            .then(|| &mut self.storage.data[((d0 * D1 + d1) * D2 + d2) * D3 + d3])
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    Tensor5<T, D0, D1, D2, D3, D4>
{
    /// Static shape of this tensor type.
    pub const SHAPE: &'static [usize] = &[D0, D1, D2, D3, D4];

    /// Creates a tensor from row-major storage with exactly `D0 * D1 * D2 * D3 * D4` elements.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2, D3, D4])?,
        })
    }

    /// Creates a tensor from a nested array matching the static shape.
    pub fn from_array(data: [[[[[T; D4]; D3]; D2]; D1]; D0]) -> Self {
        Self {
            storage: TensorData {
                data: data
                    .into_iter()
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .collect(),
            },
        }
    }

    /// Creates a tensor whose elements are all `value`.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2, D3, D4]),
        }
    }

    /// Returns the row-major tensor storage as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns the row-major tensor storage as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns its row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns the element at `[d0, d1, d2, d3, d4]`, or `None` when out of bounds.
    pub fn get(&self, d0: usize, d1: usize, d2: usize, d3: usize, d4: usize) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3 && d4 < D4)
            .then(|| &self.storage.data[(((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4])
    }

    /// Returns the element at `[d0, d1, d2, d3, d4]` mutably, or `None` when out of bounds.
    pub fn get_mut(
        &mut self,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
        d4: usize,
    ) -> Option<&mut T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3 && d4 < D4)
            .then(|| &mut self.storage.data[(((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4])
    }
}

impl<
        T,
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
        const D5: usize,
    > Tensor6<T, D0, D1, D2, D3, D4, D5>
{
    /// Static shape of this tensor type.
    pub const SHAPE: &'static [usize] = &[D0, D1, D2, D3, D4, D5];

    /// Creates a tensor from row-major storage with exactly `D0 * D1 * D2 * D3 * D4 * D5` elements.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2, D3, D4, D5])?,
        })
    }

    #[allow(clippy::type_complexity)]
    /// Creates a tensor from a nested array matching the static shape.
    pub fn from_array(data: [[[[[[T; D5]; D4]; D3]; D2]; D1]; D0]) -> Self {
        Self {
            storage: TensorData {
                data: data
                    .into_iter()
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .flat_map(IntoIterator::into_iter)
                    .collect(),
            },
        }
    }

    /// Creates a tensor whose elements are all `value`.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2, D3, D4, D5]),
        }
    }

    /// Returns the row-major tensor storage as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns the row-major tensor storage as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns its row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns the element at `[d0, d1, d2, d3, d4, d5]`, or `None` when out of bounds.
    pub fn get(
        &self,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
        d4: usize,
        d5: usize,
    ) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3 && d4 < D4 && d5 < D5).then(|| {
            &self.storage.data[((((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4) * D5 + d5]
        })
    }

    /// Returns the element at `[d0, d1, d2, d3, d4, d5]` mutably, or `None` when out of bounds.
    pub fn get_mut(
        &mut self,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
        d4: usize,
        d5: usize,
    ) -> Option<&mut T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3 && d4 < D4 && d5 < D5).then(|| {
            &mut self.storage.data[((((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4) * D5 + d5]
        })
    }
}

impl<T: TensorElement> Tensor0<T> {
    /// Creates a tensor filled with zeros.
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    /// Creates a tensor filled with ones.
    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T: TensorElement, const D0: usize> Tensor1<T, D0> {
    /// Creates a tensor filled with zeros.
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    /// Creates a tensor filled with ones.
    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T: TensorElement, const D0: usize, const D1: usize> Tensor2<T, D0, D1> {
    /// Creates a tensor filled with zeros.
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    /// Creates a tensor filled with ones.
    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T: TensorElement, const D0: usize, const D1: usize, const D2: usize> Tensor3<T, D0, D1, D2> {
    /// Creates a tensor filled with zeros.
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    /// Creates a tensor filled with ones.
    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T: TensorElement, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    Tensor4<T, D0, D1, D2, D3>
{
    /// Creates a tensor filled with zeros.
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    /// Creates a tensor filled with ones.
    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<
        T: TensorElement,
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
    > Tensor5<T, D0, D1, D2, D3, D4>
{
    /// Creates a tensor filled with zeros.
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    /// Creates a tensor filled with ones.
    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<
        T: TensorElement,
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
        const D5: usize,
    > Tensor6<T, D0, D1, D2, D3, D4, D5>
{
    /// Creates a tensor filled with zeros.
    pub fn zeros() -> Self {
        Self::filled(T::ZERO)
    }

    /// Creates a tensor filled with ones.
    pub fn ones() -> Self {
        Self::filled(T::ONE)
    }
}

impl<T> TryFrom<Vec<T>> for Tensor0<T> {
    type Error = crate::Error;

    fn try_from(data: Vec<T>) -> crate::Result<Self> {
        Self::from_vec(data)
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

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    TryFrom<Vec<T>> for Tensor5<T, D0, D1, D2, D3, D4>
{
    type Error = crate::Error;

    fn try_from(data: Vec<T>) -> crate::Result<Self> {
        Self::from_vec(data)
    }
}

impl<
        T,
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
        const D5: usize,
    > TryFrom<Vec<T>> for Tensor6<T, D0, D1, D2, D3, D4, D5>
{
    type Error = crate::Error;

    fn try_from(data: Vec<T>) -> crate::Result<Self> {
        Self::from_vec(data)
    }
}

impl<T: fmt::Debug> fmt::Debug for Tensor0<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor0")
            .field("shape", &Self::SHAPE)
            .field("data", &self.storage.data)
            .finish()
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

impl<
        T: fmt::Debug,
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
    > fmt::Debug for Tensor5<T, D0, D1, D2, D3, D4>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor5")
            .field("shape", &Self::SHAPE)
            .field("data", &self.storage.data)
            .finish()
    }
}

impl<
        T: fmt::Debug,
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
        const D5: usize,
    > fmt::Debug for Tensor6<T, D0, D1, D2, D3, D4, D5>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor6")
            .field("shape", &Self::SHAPE)
            .field("data", &self.storage.data)
            .finish()
    }
}
