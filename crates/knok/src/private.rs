extern crate alloc;

use alloc::vec::Vec;

#[cfg(feature = "runtime")]
use crate::runtime::raw;

pub trait Signature {
    const DESCS: &'static [crate::TensorDesc];
}

pub trait Sealed {}

#[cfg(feature = "runtime")]
pub trait RawGraphInputs {
    fn with_raw_inputs<'a, R>(&'a self, run: impl FnOnce(&[raw::Input<'a>]) -> R) -> R;
}

#[cfg(feature = "runtime")]
pub trait RawGraphOutput: Sized {
    fn validate_raw_outputs(outputs: &raw::Outputs) -> crate::Result<()>;
    // Read methods are only called after Model validates the entire output list.
    fn read_raw_outputs(outputs: raw::Outputs) -> crate::Result<Self>;

    fn read_raw_outputs_into(&mut self, outputs: raw::Outputs) -> crate::Result<()>;
}

pub trait RawGraphTensor: Sized {
    type Element: RawGraphElement;

    const SHAPE: &'static [usize];

    fn from_vec(data: Vec<Self::Element>) -> crate::Result<Self>;

    fn as_slice(&self) -> &[Self::Element];

    fn as_mut_slice(&mut self) -> &mut [Self::Element];
}

pub trait RawGraphElement: Copy + 'static {
    const DTYPE: crate::DType;
    #[cfg(feature = "runtime")]
    fn raw_input<'a>(shape: &'static [usize], data: &'a [Self]) -> raw::Input<'a>;
}
