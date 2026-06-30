extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    private::{RawGraphElement, RawGraphInputs, RawGraphOutput, RawGraphTensor},
    runtime::raw,
    Engine, GraphArtifact,
};

/// Typed handle for one generated graph or imported MLIR model.
///
/// Generated modules expose a `GRAPH` value and thin `run` / `call` helpers.
/// The graph owns no runtime resources; it only carries static artifact
/// metadata and the Rust input/output types expected by the wrapper.
#[derive(Clone, Copy, Debug)]
pub struct Graph<I, O> {
    artifact: GraphArtifact,
    _types: PhantomData<fn(I) -> O>,
}

impl<I, O> Graph<I, O> {
    /// Creates a typed graph handle from embedded artifact metadata.
    pub const fn new(artifact: GraphArtifact) -> Self {
        Self {
            artifact,
            _types: PhantomData,
        }
    }

    /// Returns the embedded artifact metadata.
    pub const fn artifact(&self) -> GraphArtifact {
        self.artifact
    }
}

impl<I, O> Graph<I, O>
where
    I: GraphInputs,
    O: GraphOutput,
{
    /// Runs the graph with a reusable hosted runtime engine.
    pub fn run(&self, engine: &Engine, inputs: I) -> crate::Result<O> {
        RawGraphInputs::with_raw_inputs(&inputs, |raw_inputs| {
            RawGraphOutput::read_raw_outputs(engine.invoke(self.artifact, raw_inputs)?)
        })
    }

    /// Runs the graph once by constructing an engine for the artifact's default variant.
    pub fn run_once(&self, inputs: I) -> crate::Result<O> {
        let engine = Engine::for_artifact(self.artifact)?;
        self.run(&engine, inputs)
    }

    /// Alias for [`Graph::run_once`].
    pub fn call(&self, inputs: I) -> crate::Result<O> {
        self.run_once(inputs)
    }
}

#[doc(hidden)]
pub trait GraphInputs: crate::private::Sealed + RawGraphInputs {}

#[doc(hidden)]
pub trait GraphOutput: crate::private::Sealed + RawGraphOutput {}

#[doc(hidden)]
pub trait GraphTensor: crate::private::Sealed + RawGraphTensor {}

#[doc(hidden)]
pub trait GraphElement: crate::private::Sealed + RawGraphElement {}

impl crate::private::Sealed for () {}

impl RawGraphInputs for () {
    fn with_raw_inputs<'a, R>(&'a self, run: impl FnOnce(&[raw::Input<'a>]) -> R) -> R {
        run(&[])
    }
}

impl GraphInputs for () {}

impl RawGraphOutput for () {
    fn read_raw_outputs(outputs: raw::Outputs) -> crate::Result<Self> {
        if outputs.is_empty() {
            Ok(())
        } else {
            Err(crate::Error::OutputCountMismatch {
                expected: 0,
                actual: outputs.len(),
            })
        }
    }
}

impl GraphOutput for () {}

impl<T> RawGraphInputs for T
where
    T: GraphTensor,
{
    fn with_raw_inputs<'a, R>(&'a self, run: impl FnOnce(&[raw::Input<'a>]) -> R) -> R {
        let inputs = [<T::Element as RawGraphElement>::raw_input(
            T::SHAPE,
            self.as_slice(),
        )];
        run(&inputs)
    }
}

impl<T> GraphInputs for T where T: GraphTensor {}

impl<T> RawGraphOutput for T
where
    T: GraphTensor,
{
    fn read_raw_outputs(outputs: raw::Outputs) -> crate::Result<Self> {
        T::from_vec(outputs.one::<T::Element>()?)
    }
}

impl<T> GraphOutput for T where T: GraphTensor {}

macro_rules! impl_graph_element {
    ($type:ty, $variant:ident) => {
        impl crate::private::Sealed for $type {}

        impl RawGraphElement for $type {
            fn raw_input<'a>(shape: &'static [usize], data: &'a [Self]) -> raw::Input<'a> {
                raw::Input::$variant(shape, data)
            }
        }

        impl GraphElement for $type {}
    };
}

impl_graph_element!(bool, Bool);
impl_graph_element!(f32, F32);
impl_graph_element!(f64, F64);
impl_graph_element!(i32, I32);
impl_graph_element!(i64, I64);

#[cfg(feature = "half")]
impl_graph_element!(crate::half::f16, F16);
#[cfg(feature = "half")]
impl_graph_element!(crate::half::bf16, BF16);

macro_rules! impl_graph_tensor {
    ($name:ident <$elem:ident $(, $dim:ident)*>) => {
        impl<$elem $(, const $dim: usize)*> crate::private::Sealed
            for crate::tensor::$name<$elem $(, $dim)*>
        where
            $elem: GraphElement + raw::Element,
        {
        }

        impl<$elem $(, const $dim: usize)*> RawGraphTensor for crate::tensor::$name<$elem $(, $dim)*>
        where
            $elem: GraphElement + raw::Element,
        {
            type Element = $elem;
            const SHAPE: &'static [usize] = Self::SHAPE;

            fn from_vec(data: Vec<Self::Element>) -> crate::Result<Self> {
                Self::from_vec(data)
            }

            fn as_slice(&self) -> &[Self::Element] {
                self.as_slice()
            }
        }

        impl<$elem $(, const $dim: usize)*> GraphTensor for crate::tensor::$name<$elem $(, $dim)*>
        where
            $elem: GraphElement + raw::Element,
        {
        }
    };
}

impl_graph_tensor!(Tensor0<T>);
impl_graph_tensor!(Tensor1<T, D0>);
impl_graph_tensor!(Tensor2<T, D0, D1>);
impl_graph_tensor!(Tensor3<T, D0, D1, D2>);
impl_graph_tensor!(Tensor4<T, D0, D1, D2, D3>);
impl_graph_tensor!(Tensor5<T, D0, D1, D2, D3, D4>);
impl_graph_tensor!(Tensor6<T, D0, D1, D2, D3, D4, D5>);

macro_rules! impl_graph_tuple {
    ($($name:ident : $index:tt),+ $(,)?) => {
        impl<$($name),+> crate::private::Sealed for ($($name,)+)
        where
            $($name: GraphTensor,)+
        {
        }

        impl<$($name),+> RawGraphInputs for ($($name,)+)
        where
            $($name: GraphTensor,)+
        {
            fn with_raw_inputs<'a, R>(&'a self, run: impl FnOnce(&[raw::Input<'a>]) -> R) -> R {
                let inputs = [
                    $(
                        <$name::Element as RawGraphElement>::raw_input(
                            $name::SHAPE,
                            self.$index.as_slice(),
                        ),
                    )+
                ];
                run(&inputs)
            }
        }

        impl<$($name),+> GraphInputs for ($($name,)+)
        where
            $($name: GraphTensor,)+
        {
        }

        impl<$($name),+> RawGraphOutput for ($($name,)+)
        where
            $($name: GraphTensor,)+
        {
            fn read_raw_outputs(outputs: raw::Outputs) -> crate::Result<Self> {
                Ok((
                    $(
                        $name::from_vec(outputs.read::<$name::Element>($index)?)?,
                    )+
                ))
            }
        }

        impl<$($name),+> GraphOutput for ($($name,)+)
        where
            $($name: GraphTensor,)+
        {
        }
    };
}

impl_graph_tuple!(A: 0);
impl_graph_tuple!(A: 0, B: 1);
impl_graph_tuple!(A: 0, B: 1, C: 2);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10);
impl_graph_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11);
