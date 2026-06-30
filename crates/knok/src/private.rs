extern crate alloc;

use alloc::vec::Vec;

use crate::runtime::raw;

pub trait Sealed {}

pub trait RawGraphInputs {
    fn with_raw_inputs<'a, R>(&'a self, run: impl FnOnce(&[raw::Input<'a>]) -> R) -> R;
}

pub trait RawGraphOutput: Sized {
    fn read_raw_outputs(outputs: raw::Outputs) -> crate::Result<Self>;
}

pub trait RawGraphTensor: Sized {
    type Element: RawGraphElement + raw::Element;

    const SHAPE: &'static [usize];

    fn from_vec(data: Vec<Self::Element>) -> crate::Result<Self>;

    fn as_slice(&self) -> &[Self::Element];
}

pub trait RawGraphElement: Copy {
    fn raw_input<'a>(shape: &'static [usize], data: &'a [Self]) -> raw::Input<'a>;
}
