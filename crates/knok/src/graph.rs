extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    private::{RawGraphElement, RawGraphTensor, Signature},
    GraphArtifact, TensorDesc,
};
#[cfg(feature = "runtime")]
use crate::{
    private::{RawGraphInputs, RawGraphOutput},
    runtime::raw,
    Engine, Model,
};

/// Typed handle for one generated graph or imported MLIR model.
///
/// Generated modules expose a `GRAPH` value and thin `run` / `call` helpers.
/// The graph owns no runtime resources; it only carries static artifact
/// metadata and the Rust input/output types expected by the wrapper.
///
/// Most users call generated module functions such as
/// `graphs::forward::run(&engine, &input)` or `graphs::forward::call(&input)`.
/// Use `Graph` directly when passing a generated graph handle through another
/// abstraction.
#[derive(Debug)]
pub struct Graph<I, O> {
    artifact: GraphArtifact,
    _types: PhantomData<fn(I) -> O>,
}

impl<I, O> Copy for Graph<I, O> {}
impl<I, O> Clone for Graph<I, O> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Artifact signature that disagrees with the typed graph handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphSignatureError {
    /// Input tensor types do not match the artifact.
    Input,
    /// Output tensor types do not match the artifact.
    Output,
}

impl core::fmt::Display for GraphSignatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match self {
            Self::Input => "input",
            Self::Output => "output",
        };
        write!(f, "graph {kind} signature does not match the typed handle")
    }
}
impl core::error::Error for GraphSignatureError {}

impl<I: GraphInputs, O: GraphOutput> Graph<I, O> {
    /// Creates a handle after checking all input and output tensor descriptors.
    pub const fn new(artifact: GraphArtifact) -> Result<Self, GraphSignatureError> {
        if !signatures_match(I::DESCS, artifact.input_descs()) {
            return Err(GraphSignatureError::Input);
        }
        if !signatures_match(O::DESCS, artifact.output_descs()) {
            return Err(GraphSignatureError::Output);
        }
        Ok(Self {
            artifact,
            _types: PhantomData,
        })
    }

    /// Returns the embedded artifact metadata.
    pub const fn artifact(&self) -> GraphArtifact {
        self.artifact
    }
}

#[cfg(feature = "runtime")]
impl<I, O> Graph<I, O>
where
    I: GraphInputs,
    O: GraphOutput,
{
    /// Runs the graph with a reusable hosted runtime engine.
    ///
    /// The engine's runtime driver must support a variant embedded in this
    /// graph's artifact. Constructing the engine with [`Engine::for_artifact`]
    /// is the usual way to keep the driver aligned.
    pub fn run(&self, engine: &Engine, inputs: I::Borrowed<'_>) -> crate::Result<O> {
        self.load(engine)?.run(inputs)
    }

    /// Runs the graph and writes results into existing output tensors.
    ///
    /// Reuses the outputs' host storage. Device buffers and invocation metadata
    /// may still allocate. For multiple outputs, pass a mutable tuple of tensors.
    pub fn run_into(
        &self,
        engine: &Engine,
        inputs: I::Borrowed<'_>,
        outputs: &mut O,
    ) -> crate::Result<()> {
        self.load(engine)?.run_into(inputs, outputs)
    }

    /// Runs the graph once by constructing an engine for the artifact's default variant.
    ///
    /// For loops, reuse an [`Engine`] and a prepared model from [`Graph::load`].
    pub fn run_once(&self, inputs: I::Borrowed<'_>) -> crate::Result<O> {
        let engine = Engine::for_artifact(self.artifact)?;
        self.run(&engine, inputs)
    }

    /// Loads this graph and resolves its function once for repeated inference.
    pub fn load<'a>(&self, engine: &'a Engine) -> crate::Result<Model<'a, I, O>> {
        engine.load(self)
    }
}

// Keep validation const so generated GRAPH values are checked at compile time.
const fn signatures_match(expected: &[TensorDesc], actual: &[TensorDesc]) -> bool {
    if expected.len() != actual.len() {
        return false;
    }
    let mut i = 0;
    while i < expected.len() {
        if expected[i].elem as u8 != actual[i].elem as u8
            || expected[i].shape.len() != actual[i].shape.len()
        {
            return false;
        }
        let mut j = 0;
        while j < expected[i].shape.len() {
            if expected[i].shape[j] != actual[i].shape[j] {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

#[doc(hidden)]
pub trait GraphInputs: crate::private::Sealed + Signature + 'static {
    #[cfg(feature = "runtime")]
    type Borrowed<'a>: RawGraphInputs;
}

#[doc(hidden)]
#[cfg(feature = "runtime")]
pub trait GraphOutput: crate::private::Sealed + Signature + RawGraphOutput {}
#[doc(hidden)]
#[cfg(not(feature = "runtime"))]
pub trait GraphOutput: crate::private::Sealed + Signature {}

#[doc(hidden)]
pub trait GraphTensor: crate::private::Sealed + RawGraphTensor {}

#[doc(hidden)]
pub trait GraphElement: crate::private::Sealed + RawGraphElement {}

impl crate::private::Sealed for () {}
impl Signature for () {
    const DESCS: &'static [TensorDesc] = &[];
}

#[cfg(feature = "runtime")]
impl RawGraphInputs for () {
    fn with_raw_inputs<'a, R>(&'a self, run: impl FnOnce(&[raw::Input<'a>]) -> R) -> R {
        run(&[])
    }
}

impl GraphInputs for () {
    #[cfg(feature = "runtime")]
    type Borrowed<'a> = ();
}

#[cfg(feature = "runtime")]
impl RawGraphOutput for () {
    fn validate_raw_outputs(outputs: &raw::Outputs) -> crate::Result<()> {
        outputs.check_count(0)
    }
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

    fn read_raw_outputs_into(&mut self, outputs: raw::Outputs) -> crate::Result<()> {
        Self::read_raw_outputs(outputs)
    }
}

impl GraphOutput for () {}

#[cfg(feature = "runtime")]
impl<T> RawGraphInputs for &T
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

impl<T: GraphTensor + 'static> GraphInputs for T {
    #[cfg(feature = "runtime")]
    type Borrowed<'a> = &'a T;
}

impl<T: GraphTensor> Signature for T {
    const DESCS: &'static [TensorDesc] = &[TensorDesc::new(T::Element::DTYPE, T::SHAPE)];
}

#[cfg(feature = "runtime")]
impl<T> RawGraphOutput for T
where
    T: GraphTensor,
    T::Element: raw::Element,
{
    fn validate_raw_outputs(outputs: &raw::Outputs) -> crate::Result<()> {
        outputs.check_count(1)?;
        outputs.check_shape::<T::Element>(0, T::SHAPE)
    }
    fn read_raw_outputs(outputs: raw::Outputs) -> crate::Result<Self> {
        T::from_vec(outputs.one::<T::Element>()?)
    }

    fn read_raw_outputs_into(&mut self, outputs: raw::Outputs) -> crate::Result<()> {
        if outputs.len() != 1 {
            return Err(crate::Error::OutputCountMismatch {
                expected: 1,
                actual: outputs.len(),
            });
        }
        outputs.read_into::<T::Element>(0, self.as_mut_slice())
    }
}

#[cfg(feature = "runtime")]
impl<T> GraphOutput for T
where
    T: GraphTensor,
    T::Element: raw::Element,
{
}
#[cfg(not(feature = "runtime"))]
impl<T: GraphTensor> GraphOutput for T {}

macro_rules! impl_graph_element {
    ($type:ty, $variant:ident) => {
        impl crate::private::Sealed for $type {}

        impl RawGraphElement for $type {
            const DTYPE: crate::DType = crate::DType::$variant;

            #[cfg(feature = "runtime")]
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
            $elem: GraphElement,
        {
        }

        impl<$elem $(, const $dim: usize)*> RawGraphTensor for crate::tensor::$name<$elem $(, $dim)*>
        where
            $elem: GraphElement,
        {
            type Element = $elem;
            const SHAPE: &'static [usize] = Self::SHAPE;

            fn from_vec(data: Vec<Self::Element>) -> crate::Result<Self> {
                Self::from_vec(data)
            }

            fn as_slice(&self) -> &[Self::Element] {
                self.as_slice()
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Element] {
                self.as_mut_slice()
            }
        }

        impl<$elem $(, const $dim: usize)*> GraphTensor for crate::tensor::$name<$elem $(, $dim)*>
        where
            $elem: GraphElement,
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

        impl<$($name: GraphTensor),+> Signature for ($($name,)+) {
            const DESCS: &'static [TensorDesc] = &[$(TensorDesc::new($name::Element::DTYPE, $name::SHAPE)),+];
        }

        #[cfg(feature = "runtime")]
        impl<$($name),+> RawGraphInputs for ($(&$name,)+)
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
            $($name: GraphTensor + 'static,)+
        {
            #[cfg(feature = "runtime")]
            type Borrowed<'a> = ($(&'a $name,)+);
        }

        #[cfg(feature = "runtime")]
        impl<$($name),+> RawGraphOutput for ($($name,)+)
        where
            $($name: GraphTensor,)+
            $($name::Element: raw::Element,)+
        {
            fn validate_raw_outputs(outputs: &raw::Outputs) -> crate::Result<()> {
                outputs.check_count([$(stringify!($name)),+].len())?;
                $(outputs.check_shape::<$name::Element>($index, $name::SHAPE)?;)+
                Ok(())
            }
            fn read_raw_outputs(outputs: raw::Outputs) -> crate::Result<Self> {
                Ok((
                    $(
                        $name::from_vec(outputs.read::<$name::Element>($index)?)?,
                    )+
                ))
            }

            fn read_raw_outputs_into(&mut self, outputs: raw::Outputs) -> crate::Result<()> {
                let expected = [$(stringify!($name)),+].len();
                if outputs.len() != expected {
                    return Err(crate::Error::OutputCountMismatch {
                        expected,
                        actual: outputs.len(),
                    });
                }
                $(outputs.read_into::<$name::Element>($index, self.$index.as_mut_slice())?;)+
                Ok(())
            }
        }

        #[cfg(feature = "runtime")]
        impl<$($name),+> GraphOutput for ($($name,)+)
        where
            $($name: GraphTensor,)+
            $($name::Element: raw::Element,)+
        {
        }
        #[cfg(not(feature = "runtime"))]
        impl<$($name: GraphTensor),+> GraphOutput for ($($name,)+) {}
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
