//! Runtime device selection, engines, and prepared models.

use std::{cell::RefCell, collections::BTreeMap, marker::PhantomData};

pub use eerie::runtime::DeviceInfo;
use eerie::runtime::{DeviceSpec, Function, Program, Runtime, Value};

use crate::{
    graph::{GraphInputs, GraphOutput},
    private::RawGraphInputs,
    Backend, Driver, Graph, GraphArtifact, GraphArtifactVariant,
};

/// Runtime driver and device selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    driver: Driver,
    device: DeviceSpec,
}

impl RuntimeConfig {
    /// Selects the default CPU driver and device.
    pub fn auto() -> Self {
        Self::driver(Driver::LocalTask)
    }

    /// Selects a driver and its default device.
    pub fn driver(driver: Driver) -> Self {
        Self {
            driver,
            device: DeviceSpec::custom(driver.name()),
        }
    }

    /// Selects the default driver for a compilation backend.
    pub fn backend(backend: Backend) -> Self {
        Self::driver(
            Driver::from_name(backend.default_driver()).expect("backend has a supported driver"),
        )
    }

    /// Selects a device by its zero-based driver ordinal.
    /// Replaces a previous path selection.
    pub fn device_ordinal(mut self, ordinal: usize) -> Self {
        self.device = self.device.ordinal(ordinal);
        self
    }

    /// Selects a driver-specific device path, replacing a previous ordinal.
    pub fn device_path(mut self, path: impl Into<String>) -> Self {
        self.device = self.device.path(path);
        self
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::auto()
    }
}

/// Runtime instance with a cache of loaded static VMFB modules.
///
/// Use [`Graph::load`] to resolve a function once for repeated inference.
/// An engine and its models stay on the thread where they were created.
pub struct Engine {
    driver: Driver,
    runtime: Runtime,
    modules: RefCell<BTreeMap<(usize, usize), Program>>,
}

impl Engine {
    /// Creates an engine on the configured device.
    pub fn new(config: RuntimeConfig) -> crate::Result<Self> {
        Ok(Self {
            driver: config.driver,
            runtime: Runtime::new(config.device)?,
            modules: RefCell::new(BTreeMap::new()),
        })
    }

    /// Lists devices available to a driver, including their ordinals and paths.
    pub fn available_devices(driver: Driver) -> crate::Result<Vec<DeviceInfo>> {
        Ok(Runtime::available_devices(eerie::runtime::Driver::custom(
            driver.name(),
        ))?)
    }

    /// Creates an engine for a backend's default driver.
    pub fn for_backend(backend: Backend) -> crate::Result<Self> {
        Self::new(RuntimeConfig::backend(backend))
    }

    /// Creates an engine for a compiled variant's default driver.
    pub fn for_variant(variant: GraphArtifactVariant) -> crate::Result<Self> {
        Self::for_backend(variant.backend())
    }

    /// Creates an engine for the first embedded variant.
    pub fn for_artifact(artifact: GraphArtifact) -> crate::Result<Self> {
        Self::for_variant(artifact.first_variant().ok_or(
            crate::Error::MissingDefaultArtifactVariant {
                function_name: artifact.function_name(),
            },
        )?)
    }

    /// Returns the selected driver name.
    pub fn driver_name(&self) -> &'static str {
        self.driver.name()
    }

    /// Loads a typed graph and resolves its function for repeated calls.
    pub fn load<I: GraphInputs, O: GraphOutput>(
        &self,
        graph: &Graph<I, O>,
    ) -> crate::Result<Model<'_, I, O>> {
        let artifact = graph.artifact();
        let variant = artifact.variant_for_driver(self.driver).ok_or_else(|| {
            crate::Error::MissingArtifactVariant {
                function_name: artifact.function_name(),
                driver: self.driver.name().into(),
            }
        })?;
        let vmfb = variant.vmfb();
        // Artifacts are static. Their address and length identify the backing
        // storage without copying or comparing entire VMFB archives on calls.
        let key = (vmfb.as_ptr() as usize, vmfb.len());
        let mut modules = self.modules.borrow_mut();
        let program = match modules.entry(key) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(self.runtime.load_vmfb(vmfb)?)
            }
        };
        let function = program.function(artifact.function_name())?;
        Ok(Model {
            engine: self,
            function,
            artifact,
            types: PhantomData,
        })
    }
}

/// Loaded graph with a resolved function and borrowed host inputs.
///
/// A model borrows its engine and keeps function lookup and cache access out of
/// repeated calls. Host-to-device transfers may still copy and allocate.
pub struct Model<'a, I, O> {
    engine: &'a Engine,
    function: Function,
    artifact: GraphArtifact,
    types: PhantomData<fn(I) -> O>,
}

impl<I: GraphInputs, O: GraphOutput> Model<'_, I, O> {
    /// Returns this model's static artifact metadata.
    pub fn artifact(&self) -> GraphArtifact {
        self.artifact
    }

    /// Executes using borrowed tensors and allocates host outputs.
    pub fn run(&self, inputs: I::Borrowed<'_>) -> crate::Result<O> {
        O::read_raw_outputs(self.invoke(inputs)?)
    }

    /// Executes and reuses existing host output storage.
    /// All output shapes and dtypes are checked before any host output is written.
    pub fn run_into(&self, inputs: I::Borrowed<'_>, outputs: &mut O) -> crate::Result<()> {
        outputs.read_raw_outputs_into(self.invoke(inputs)?)
    }

    fn invoke(&self, inputs: I::Borrowed<'_>) -> crate::Result<raw::Outputs> {
        inputs.with_raw_inputs(|inputs| {
            let values = inputs
                .iter()
                .map(|input| input.to_value(&self.engine.runtime))
                .collect::<crate::Result<Vec<_>>>()?;
            let outputs = raw::Outputs {
                values: self.function.invoke(values)?,
            };
            O::validate_raw_outputs(&outputs)?;
            Ok(outputs)
        })
    }
}

pub(crate) mod raw {
    use super::*;

    pub enum Input<'a> {
        Bool(&'a [usize], &'a [bool]),
        F32(&'a [usize], &'a [f32]),
        F64(&'a [usize], &'a [f64]),
        I32(&'a [usize], &'a [i32]),
        I64(&'a [usize], &'a [i64]),
        #[cfg(feature = "half")]
        F16(&'a [usize], &'a [crate::half::f16]),
        #[cfg(feature = "half")]
        BF16(&'a [usize], &'a [crate::half::bf16]),
    }

    impl Input<'_> {
        pub(super) fn to_value(&self, runtime: &Runtime) -> crate::Result<Value> {
            Ok(match self {
                Self::Bool(shape, data) => runtime.buffer_view(shape, data)?.into(),
                Self::F32(shape, data) => runtime.buffer_view(shape, data)?.into(),
                Self::F64(shape, data) => runtime.buffer_view(shape, data)?.into(),
                Self::I32(shape, data) => runtime.buffer_view(shape, data)?.into(),
                Self::I64(shape, data) => runtime.buffer_view(shape, data)?.into(),
                #[cfg(feature = "half")]
                Self::F16(shape, data) => runtime.buffer_view(shape, data)?.into(),
                #[cfg(feature = "half")]
                Self::BF16(shape, data) => runtime.buffer_view(shape, data)?.into(),
            })
        }
    }

    pub struct Outputs {
        pub(super) values: Vec<Value>,
    }

    impl Outputs {
        pub fn check_count(&self, expected: usize) -> crate::Result<()> {
            let actual = self.values.len();
            if actual != expected {
                return Err(crate::Error::OutputCountMismatch { expected, actual });
            }
            Ok(())
        }
        pub fn check_shape<T: Element>(
            &self,
            index: usize,
            expected: &'static [usize],
        ) -> crate::Result<()> {
            let output = T::buffer_from_value(self.values[index].clone())?;
            let actual = output.shape();
            if actual != expected {
                return Err(crate::Error::OutputShapeMismatch {
                    index,
                    expected,
                    actual,
                });
            }
            Ok(())
        }
        pub fn one<T: Element>(mut self) -> crate::Result<Vec<T>> {
            Ok(T::buffer_from_value(self.values.pop().expect("outputs were validated"))?.read()?)
        }
        pub fn read<T: Element>(&self, index: usize) -> crate::Result<Vec<T>> {
            Ok(T::buffer_from_value(self.values[index].clone())?.read()?)
        }
        pub fn read_into<T: Element>(&self, index: usize, target: &mut [T]) -> crate::Result<()> {
            Ok(T::buffer_from_value(self.values[index].clone())?.read_into(target)?)
        }
    }

    pub trait Element: eerie::runtime::BufferElement {
        fn buffer_from_value(
            value: Value,
        ) -> Result<eerie::runtime::BufferView<Self>, eerie::runtime::RuntimeError>;
    }
    macro_rules! impl_element {
        ($($ty:ty),+) => { $(impl Element for $ty {
            fn buffer_from_value(value: Value) -> Result<eerie::runtime::BufferView<Self>, eerie::runtime::RuntimeError> { value.try_into() }
        })+ };
    }
    impl_element!(bool, f32, f64, i32, i64);
    #[cfg(feature = "half")]
    impl_element!(crate::half::f16, crate::half::bf16);
}
