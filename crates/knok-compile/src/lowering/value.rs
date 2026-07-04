use knok_core::{ElementType, TensorType};
use melior::ir::Value as MlirValue;

#[derive(Clone)]
pub(super) struct Value {
    pub(super) raw: RawValue,
    pub(super) ty: TensorType,
    pub(super) kind: ValueKind,
}

#[derive(Clone, Copy)]
pub(super) struct RawValue(MlirValue<'static, 'static>);

impl RawValue {
    pub(super) fn from_value(value: MlirValue<'_, '_>) -> Self {
        // SAFETY: MLIR values are non-owning handles. The lowerer only stores
        // values created in the same MLIR context and appends all uses before
        // the owning operation/block/module is dropped, so erasing the handle
        // lifetimes here preserves the existing lowering lifetime invariant.
        Self(unsafe {
            core::mem::transmute::<MlirValue<'_, '_>, MlirValue<'static, 'static>>(value)
        })
    }

    pub(super) fn as_value<'c>(self) -> MlirValue<'c, 'static> {
        // SAFETY: See from_value. This only rebinds the erased handle to the
        // active lowering context while the owning MLIR objects remain alive.
        unsafe {
            core::mem::transmute::<MlirValue<'static, 'static>, MlirValue<'c, 'static>>(self.0)
        }
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
