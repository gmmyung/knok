use knok_core::{ElementType, TensorType};
use melior::ir::{Value as MlirValue, ValueLike};
use mlir_sys::MlirValue as MlirRawValue;

#[derive(Clone)]
pub(super) struct Value {
    pub(super) raw: RawValue,
    pub(super) ty: TensorType,
    pub(super) kind: ValueKind,
}

#[derive(Clone, Copy)]
pub(super) struct RawValue(MlirRawValue);

impl RawValue {
    pub(super) fn from_value(value: MlirValue<'_, '_>) -> Self {
        Self(value.to_raw())
    }

    pub(super) fn as_value<'c>(self) -> MlirValue<'c, 'static> {
        // The lowerer only stores values created in the same MLIR context and
        // appends all uses before the owning operation/block/module is dropped.
        unsafe { MlirValue::from_raw(self.0) }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ValueKind {
    Scalar,
    Tensor,
}

impl Value {
    pub(super) fn scalar(raw: RawValue, elem: ElementType) -> Self {
        Self {
            raw,
            ty: TensorType {
                elem,
                shape: Vec::new(),
            },
            kind: ValueKind::Scalar,
        }
    }

    pub(super) fn tensor(raw: RawValue, ty: TensorType) -> Self {
        Self {
            raw,
            ty,
            kind: ValueKind::Tensor,
        }
    }
}
