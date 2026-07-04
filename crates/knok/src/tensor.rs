use alloc::{vec, vec::Vec};
use core::fmt;
#[cfg(feature = "serde")]
use core::marker::PhantomData;
#[cfg(feature = "serde")]
use serde::ser::SerializeStruct;

/// Element type accepted by fixed-rank tensor containers.
///
/// This trait supplies zero/one constructors for host-side tensor helpers. It
/// does not imply that every operation supports every dtype; graph operation
/// support is checked by `knok-build` and `knok-core`.
pub trait TensorElement: Copy + Clone + PartialEq + fmt::Debug {
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
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

/// Rank-0 scalar tensor containing one element.
#[derive(Clone, PartialEq)]
pub struct Tensor0<T> {
    storage: TensorData<T>,
}

/// Rank-1 tensor with statically known length.
#[derive(Clone, PartialEq)]
pub struct Tensor1<T, const D0: usize> {
    storage: TensorData<T>,
}

/// Rank-2 tensor with statically known shape.
#[derive(Clone, PartialEq)]
pub struct Tensor2<T, const D0: usize, const D1: usize> {
    storage: TensorData<T>,
}

/// Rank-3 tensor with statically known shape.
#[derive(Clone, PartialEq)]
pub struct Tensor3<T, const D0: usize, const D1: usize, const D2: usize> {
    storage: TensorData<T>,
}

/// Rank-4 tensor with statically known shape.
#[derive(Clone, PartialEq)]
pub struct Tensor4<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize> {
    storage: TensorData<T>,
}

/// Rank-5 tensor with statically known shape.
#[derive(Clone, PartialEq)]
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

/// Rank-6 tensor with statically known shape.
#[derive(Clone, PartialEq)]
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

/// Common storage API implemented by all fixed-rank tensor containers.
pub trait FixedTensor<T>: Sized {
    /// Static tensor shape.
    const SHAPE: &'static [usize];

    /// Creates a tensor from row-major data and validates the element count.
    fn from_vec(data: Vec<T>) -> crate::Result<Self>;

    /// Creates a tensor filled with one value.
    fn filled(value: T) -> Self
    where
        T: Clone;

    /// Returns row-major tensor storage.
    fn as_slice(&self) -> &[T];

    /// Returns mutable row-major tensor storage.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Consumes the tensor and returns row-major storage.
    fn into_vec(self) -> Vec<T>;
}

impl<T> Tensor0<T> {
    /// Static tensor shape.
    pub const SHAPE: &'static [usize] = &[];

    /// Creates a tensor from row-major data and validates the element count.
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

    /// Creates a scalar tensor filled with one value.
    pub fn filled(value: T) -> Self {
        Self {
            storage: TensorData { data: vec![value] },
        }
    }

    /// Returns row-major tensor storage.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns mutable row-major tensor storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns row-major storage.
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
    /// Static tensor shape.
    pub const SHAPE: &'static [usize] = &[D0];

    /// Creates a tensor from row-major data and validates the element count.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0])?,
        })
    }

    /// Creates a tensor from an array.
    pub fn from_array(data: [T; D0]) -> Self {
        Self {
            storage: TensorData { data: data.into() },
        }
    }

    /// Creates a tensor filled with one value.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0]),
        }
    }

    /// Returns row-major tensor storage.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns mutable row-major tensor storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns an element by index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.storage.data.get(index)
    }

    /// Returns a mutable element by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.storage.data.get_mut(index)
    }
}

impl<T, const D0: usize, const D1: usize> Tensor2<T, D0, D1> {
    /// Static tensor shape.
    pub const SHAPE: &'static [usize] = &[D0, D1];

    /// Creates a tensor from row-major data and validates the element count.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1])?,
        })
    }

    /// Creates a tensor from a nested row-major array.
    pub fn from_array(data: [[T; D1]; D0]) -> Self {
        Self {
            storage: TensorData {
                data: data.into_iter().flat_map(IntoIterator::into_iter).collect(),
            },
        }
    }

    /// Creates a tensor filled with one value.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1]),
        }
    }

    /// Returns row-major tensor storage.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns mutable row-major tensor storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns an element by row and column.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        (row < D0 && col < D1).then(|| &self.storage.data[row * D1 + col])
    }

    /// Returns a mutable element by row and column.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        (row < D0 && col < D1).then(|| &mut self.storage.data[row * D1 + col])
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize> Tensor3<T, D0, D1, D2> {
    /// Static tensor shape.
    pub const SHAPE: &'static [usize] = &[D0, D1, D2];

    /// Creates a tensor from row-major data and validates the element count.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2])?,
        })
    }

    /// Creates a tensor from a nested row-major array.
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

    /// Creates a tensor filled with one value.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2]),
        }
    }

    /// Returns row-major tensor storage.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns mutable row-major tensor storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns an element by static-dimension indices.
    pub fn get(&self, d0: usize, d1: usize, d2: usize) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2).then(|| &self.storage.data[(d0 * D1 + d1) * D2 + d2])
    }

    /// Returns a mutable element by static-dimension indices.
    pub fn get_mut(&mut self, d0: usize, d1: usize, d2: usize) -> Option<&mut T> {
        (d0 < D0 && d1 < D1 && d2 < D2).then(|| &mut self.storage.data[(d0 * D1 + d1) * D2 + d2])
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    Tensor4<T, D0, D1, D2, D3>
{
    /// Static tensor shape.
    pub const SHAPE: &'static [usize] = &[D0, D1, D2, D3];

    /// Creates a tensor from row-major data and validates the element count.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2, D3])?,
        })
    }

    /// Creates a tensor from a nested row-major array.
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

    /// Creates a tensor filled with one value.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2, D3]),
        }
    }

    /// Returns row-major tensor storage.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns mutable row-major tensor storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns an element by static-dimension indices.
    pub fn get(&self, d0: usize, d1: usize, d2: usize, d3: usize) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3)
            .then(|| &self.storage.data[((d0 * D1 + d1) * D2 + d2) * D3 + d3])
    }

    /// Returns a mutable element by static-dimension indices.
    pub fn get_mut(&mut self, d0: usize, d1: usize, d2: usize, d3: usize) -> Option<&mut T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3)
            .then(|| &mut self.storage.data[((d0 * D1 + d1) * D2 + d2) * D3 + d3])
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    Tensor5<T, D0, D1, D2, D3, D4>
{
    /// Static tensor shape.
    pub const SHAPE: &'static [usize] = &[D0, D1, D2, D3, D4];

    /// Creates a tensor from row-major data and validates the element count.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2, D3, D4])?,
        })
    }

    /// Creates a tensor from a nested row-major array.
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

    /// Creates a tensor filled with one value.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2, D3, D4]),
        }
    }

    /// Returns row-major tensor storage.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns mutable row-major tensor storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns an element by static-dimension indices.
    pub fn get(&self, d0: usize, d1: usize, d2: usize, d3: usize, d4: usize) -> Option<&T> {
        (d0 < D0 && d1 < D1 && d2 < D2 && d3 < D3 && d4 < D4)
            .then(|| &self.storage.data[(((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4])
    }

    /// Returns a mutable element by static-dimension indices.
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
    /// Static tensor shape.
    pub const SHAPE: &'static [usize] = &[D0, D1, D2, D3, D4, D5];

    /// Creates a tensor from row-major data and validates the element count.
    pub fn from_vec(data: Vec<T>) -> crate::Result<Self> {
        Ok(Self {
            storage: TensorData::from_vec(data, &[D0, D1, D2, D3, D4, D5])?,
        })
    }

    /// Creates a tensor from a nested row-major array.
    #[allow(clippy::type_complexity)]
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

    /// Creates a tensor filled with one value.
    pub fn filled(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            storage: TensorData::filled(value, &[D0, D1, D2, D3, D4, D5]),
        }
    }

    /// Returns row-major tensor storage.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns mutable row-major tensor storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage.data
    }

    /// Consumes the tensor and returns row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns an element by static-dimension indices.
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

    /// Returns a mutable element by static-dimension indices.
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

macro_rules! impl_fixed_tensor_common {
    ($name:ident, $debug_name:literal, []) => {
        impl<T> FixedTensor<T> for $name<T> {
            const SHAPE: &'static [usize] = Self::SHAPE;

            fn from_vec(data: Vec<T>) -> crate::Result<Self> {
                Self::from_vec(data)
            }

            fn filled(value: T) -> Self
            where
                T: Clone,
            {
                Self::filled(value)
            }

            fn as_slice(&self) -> &[T] {
                self.as_slice()
            }

            fn as_mut_slice(&mut self) -> &mut [T] {
                self.as_mut_slice()
            }

            fn into_vec(self) -> Vec<T> {
                self.into_vec()
            }
        }

        impl<T: TensorElement> $name<T> {
            /// Creates a tensor filled with the element type's zero value.
            pub fn zeros() -> Self {
                Self::filled(T::ZERO)
            }

            /// Creates a tensor filled with the element type's one value.
            pub fn ones() -> Self {
                Self::filled(T::ONE)
            }
        }

        impl<T> TryFrom<Vec<T>> for $name<T> {
            type Error = crate::Error;

            fn try_from(data: Vec<T>) -> crate::Result<Self> {
                Self::from_vec(data)
            }
        }

        impl<T: fmt::Debug> fmt::Debug for $name<T> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct($debug_name)
                    .field("shape", &Self::SHAPE)
                    .field("data", &self.storage.data)
                    .finish()
            }
        }
    };
    ($name:ident, $debug_name:literal, [$($dim:ident),*]) => {
        impl<T, $(const $dim: usize),*> FixedTensor<T> for $name<T, $($dim),*> {
            const SHAPE: &'static [usize] = Self::SHAPE;

            fn from_vec(data: Vec<T>) -> crate::Result<Self> {
                Self::from_vec(data)
            }

            fn filled(value: T) -> Self
            where
                T: Clone,
            {
                Self::filled(value)
            }

            fn as_slice(&self) -> &[T] {
                self.as_slice()
            }

            fn as_mut_slice(&mut self) -> &mut [T] {
                self.as_mut_slice()
            }

            fn into_vec(self) -> Vec<T> {
                self.into_vec()
            }
        }

        impl<T: TensorElement, $(const $dim: usize),*> $name<T, $($dim),*> {
            /// Creates a tensor filled with the element type's zero value.
            pub fn zeros() -> Self {
                Self::filled(T::ZERO)
            }

            /// Creates a tensor filled with the element type's one value.
            pub fn ones() -> Self {
                Self::filled(T::ONE)
            }
        }

        impl<T, $(const $dim: usize),*> TryFrom<Vec<T>> for $name<T, $($dim),*> {
            type Error = crate::Error;

            fn try_from(data: Vec<T>) -> crate::Result<Self> {
                Self::from_vec(data)
            }
        }

        impl<T: fmt::Debug, $(const $dim: usize),*> fmt::Debug for $name<T, $($dim),*> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct($debug_name)
                    .field("shape", &Self::SHAPE)
                    .field("data", &self.storage.data)
                    .finish()
            }
        }
    };
}

impl_fixed_tensor_common!(Tensor0, "Tensor0", []);
impl_fixed_tensor_common!(Tensor1, "Tensor1", [D0]);
impl_fixed_tensor_common!(Tensor2, "Tensor2", [D0, D1]);
impl_fixed_tensor_common!(Tensor3, "Tensor3", [D0, D1, D2]);
impl_fixed_tensor_common!(Tensor4, "Tensor4", [D0, D1, D2, D3]);
impl_fixed_tensor_common!(Tensor5, "Tensor5", [D0, D1, D2, D3, D4]);
impl_fixed_tensor_common!(Tensor6, "Tensor6", [D0, D1, D2, D3, D4, D5]);

#[cfg(feature = "serde")]
const SERDE_FIELDS: &[&str] = &["shape", "data"];

#[cfg(feature = "serde")]
enum TensorSerdeField {
    Shape,
    Data,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for TensorSerdeField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct FieldVisitor;

        impl<'de> serde::de::Visitor<'de> for FieldVisitor {
            type Value = TensorSerdeField;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("`shape` or `data`")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "shape" => Ok(TensorSerdeField::Shape),
                    "data" => Ok(TensorSerdeField::Data),
                    _ => Err(E::unknown_field(value, SERDE_FIELDS)),
                }
            }
        }

        deserializer.deserialize_identifier(FieldVisitor)
    }
}

#[cfg(feature = "serde")]
struct TensorSerdeVisitor<T, F> {
    name: &'static str,
    expected_shape: &'static [usize],
    _marker: PhantomData<fn() -> (T, F)>,
}

#[cfg(feature = "serde")]
impl<T, F> TensorSerdeVisitor<T, F> {
    fn new(name: &'static str, expected_shape: &'static [usize]) -> Self {
        Self {
            name,
            expected_shape,
            _marker: PhantomData,
        }
    }
}

#[cfg(feature = "serde")]
fn tensor_from_serde_parts<T, F, E>(
    expected_shape: &[usize],
    shape: Vec<usize>,
    data: Vec<T>,
) -> Result<F, E>
where
    F: FixedTensor<T>,
    E: serde::de::Error,
{
    if shape.as_slice() != expected_shape {
        return Err(E::custom(format_args!(
            "tensor shape mismatch: expected {:?}, got {:?}",
            expected_shape, shape
        )));
    }
    F::from_vec(data).map_err(E::custom)
}

#[cfg(feature = "serde")]
impl<'de, T, F> serde::de::Visitor<'de> for TensorSerdeVisitor<T, F>
where
    T: serde::Deserialize<'de>,
    F: FixedTensor<T>,
{
    type Value = F;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} serialized as `{{ shape, data }}`", self.name)
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut shape = None;
        let mut data = None;
        while let Some(key) = map.next_key()? {
            match key {
                TensorSerdeField::Shape => {
                    if shape.is_some() {
                        return Err(serde::de::Error::duplicate_field("shape"));
                    }
                    shape = Some(map.next_value::<Vec<usize>>()?);
                }
                TensorSerdeField::Data => {
                    if data.is_some() {
                        return Err(serde::de::Error::duplicate_field("data"));
                    }
                    data = Some(map.next_value::<Vec<T>>()?);
                }
            }
        }

        let shape = shape.ok_or_else(|| serde::de::Error::missing_field("shape"))?;
        let data = data.ok_or_else(|| serde::de::Error::missing_field("data"))?;
        tensor_from_serde_parts(self.expected_shape, shape, data)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let shape = seq
            .next_element::<Vec<usize>>()?
            .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
        let data = seq
            .next_element::<Vec<T>>()?
            .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
        if seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
            return Err(serde::de::Error::invalid_length(3, &self));
        }
        tensor_from_serde_parts(self.expected_shape, shape, data)
    }
}

#[cfg(feature = "serde")]
fn deserialize_tensor<'de, T, F, D>(
    deserializer: D,
    name: &'static str,
    expected_shape: &'static [usize],
) -> Result<F, D::Error>
where
    T: serde::Deserialize<'de>,
    F: FixedTensor<T>,
    D: serde::Deserializer<'de>,
{
    deserializer.deserialize_struct(
        name,
        SERDE_FIELDS,
        TensorSerdeVisitor::<T, F>::new(name, expected_shape),
    )
}

#[cfg(feature = "serde")]
macro_rules! impl_tensor_serde {
    ($name:ident, $debug_name:literal, []) => {
        impl<T> serde::Serialize for $name<T>
        where
            T: serde::Serialize,
        {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let mut state = serializer.serialize_struct($debug_name, 2)?;
                state.serialize_field("shape", Self::SHAPE)?;
                state.serialize_field("data", self.as_slice())?;
                state.end()
            }
        }

        impl<'de, T> serde::Deserialize<'de> for $name<T>
        where
            T: serde::Deserialize<'de>,
        {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserialize_tensor::<T, Self, D>(deserializer, $debug_name, Self::SHAPE)
            }
        }
    };
    ($name:ident, $debug_name:literal, [$($dim:ident),*]) => {
        impl<T, $(const $dim: usize),*> serde::Serialize for $name<T, $($dim),*>
        where
            T: serde::Serialize,
        {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let mut state = serializer.serialize_struct($debug_name, 2)?;
                state.serialize_field("shape", Self::SHAPE)?;
                state.serialize_field("data", self.as_slice())?;
                state.end()
            }
        }

        impl<'de, T, $(const $dim: usize),*> serde::Deserialize<'de> for $name<T, $($dim),*>
        where
            T: serde::Deserialize<'de>,
        {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserialize_tensor::<T, Self, D>(deserializer, $debug_name, Self::SHAPE)
            }
        }
    };
}

#[cfg(feature = "serde")]
impl_tensor_serde!(Tensor0, "Tensor0", []);
#[cfg(feature = "serde")]
impl_tensor_serde!(Tensor1, "Tensor1", [D0]);
#[cfg(feature = "serde")]
impl_tensor_serde!(Tensor2, "Tensor2", [D0, D1]);
#[cfg(feature = "serde")]
impl_tensor_serde!(Tensor3, "Tensor3", [D0, D1, D2]);
#[cfg(feature = "serde")]
impl_tensor_serde!(Tensor4, "Tensor4", [D0, D1, D2, D3]);
#[cfg(feature = "serde")]
impl_tensor_serde!(Tensor5, "Tensor5", [D0, D1, D2, D3, D4]);
#[cfg(feature = "serde")]
impl_tensor_serde!(Tensor6, "Tensor6", [D0, D1, D2, D3, D4, D5]);

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use serde_test::{assert_de_tokens, assert_de_tokens_error, assert_tokens, Token};

    use super::{Tensor0, Tensor2};

    #[test]
    fn tensor2_serializes_shape_and_row_major_data() {
        let tensor = Tensor2::<i32, 2, 3>::from_array([[1, 2, 3], [4, 5, 6]]);

        assert_tokens(
            &tensor,
            &[
                Token::Struct {
                    name: "Tensor2",
                    len: 2,
                },
                Token::Str("shape"),
                Token::Seq { len: Some(2) },
                Token::U64(2),
                Token::U64(3),
                Token::SeqEnd,
                Token::Str("data"),
                Token::Seq { len: Some(6) },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn tensor0_serializes_empty_shape_and_one_value() {
        let tensor = Tensor0::from_scalar(7i32);

        assert_tokens(
            &tensor,
            &[
                Token::Struct {
                    name: "Tensor0",
                    len: 2,
                },
                Token::Str("shape"),
                Token::Seq { len: Some(0) },
                Token::SeqEnd,
                Token::Str("data"),
                Token::Seq { len: Some(1) },
                Token::I32(7),
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn tensor2_deserializes_sequence_style_struct() {
        let tensor = Tensor2::<i32, 2, 3>::from_array([[1, 2, 3], [4, 5, 6]]);

        assert_de_tokens(
            &tensor,
            &[
                Token::TupleStruct {
                    name: "Tensor2",
                    len: 2,
                },
                Token::Seq { len: Some(2) },
                Token::U64(2),
                Token::U64(3),
                Token::SeqEnd,
                Token::Seq { len: Some(6) },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::TupleStructEnd,
            ],
        );
    }

    #[test]
    fn deserialize_rejects_shape_mismatch() {
        assert_de_tokens_error::<Tensor2<i32, 2, 3>>(
            &[
                Token::Struct {
                    name: "Tensor2",
                    len: 2,
                },
                Token::Str("shape"),
                Token::Seq { len: Some(2) },
                Token::U64(2),
                Token::U64(2),
                Token::SeqEnd,
                Token::Str("data"),
                Token::Seq { len: Some(4) },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::SeqEnd,
                Token::StructEnd,
            ],
            "tensor shape mismatch: expected [2, 3], got [2, 2]",
        );
    }
}
