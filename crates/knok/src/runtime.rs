extern crate alloc;

use alloc::string::String;

use crate::Backend;

#[doc(hidden)]
pub enum RuntimeInput<'a> {
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

#[doc(hidden)]
pub trait RuntimeOutput: Copy {
    #[cfg(feature = "host-runtime")]
    fn read_output(
        engine: &crate::Engine,
        function: &eerie::runtime::Function,
        inputs: &[RuntimeInput<'_>],
    ) -> crate::Result<alloc::vec::Vec<Self>>;
}

#[cfg(feature = "host-runtime")]
macro_rules! impl_runtime_output {
    ($ty:ty) => {
        impl RuntimeOutput for $ty {
            fn read_output(
                engine: &crate::Engine,
                function: &eerie::runtime::Function,
                inputs: &[RuntimeInput<'_>],
            ) -> crate::Result<alloc::vec::Vec<Self>> {
                engine.invoke_typed_values::<Self>(function, inputs)
            }
        }
    };
}

#[cfg(feature = "host-runtime")]
impl_runtime_output!(f32);
#[cfg(feature = "host-runtime")]
impl_runtime_output!(f64);
#[cfg(feature = "host-runtime")]
impl_runtime_output!(bool);
#[cfg(feature = "host-runtime")]
impl_runtime_output!(i32);
#[cfg(feature = "host-runtime")]
impl_runtime_output!(i64);

#[cfg(all(feature = "host-runtime", feature = "half"))]
impl_runtime_output!(crate::half::f16);
#[cfg(all(feature = "host-runtime", feature = "half"))]
impl_runtime_output!(crate::half::bf16);

#[cfg(not(feature = "host-runtime"))]
impl<T: Copy> RuntimeOutput for T {}

/// Element types that can be passed through raw runtime buffer views.
#[cfg(feature = "host-runtime")]
#[doc(hidden)]
pub trait RuntimeElement: eerie::runtime::BufferElement {
    fn buffer_to_value(buffer: &eerie::runtime::BufferView<Self>) -> eerie::runtime::Value;

    fn buffer_from_value(
        value: eerie::runtime::Value,
    ) -> Result<eerie::runtime::BufferView<Self>, eerie::runtime::RuntimeError>;
}

#[cfg(feature = "host-runtime")]
macro_rules! impl_runtime_element {
    ($type:ty) => {
        impl RuntimeElement for $type {
            fn buffer_to_value(buffer: &eerie::runtime::BufferView<Self>) -> eerie::runtime::Value {
                buffer.into()
            }

            fn buffer_from_value(
                value: eerie::runtime::Value,
            ) -> Result<eerie::runtime::BufferView<Self>, eerie::runtime::RuntimeError> {
                value.try_into()
            }
        }
    };
}

#[cfg(feature = "host-runtime")]
impl_runtime_element!(bool);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(u8);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(u16);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(u32);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(u64);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(i8);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(i16);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(i32);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(i64);
#[cfg(all(feature = "host-runtime", feature = "half"))]
impl_runtime_element!(half::f16);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(f32);
#[cfg(feature = "host-runtime")]
impl_runtime_element!(f64);
#[cfg(all(feature = "host-runtime", feature = "half"))]
impl_runtime_element!(half::bf16);

/// Element types that can be named by generated runtime wrappers in no-std builds.
#[cfg(not(feature = "host-runtime"))]
#[doc(hidden)]
pub trait RuntimeElement: Copy {}

#[cfg(not(feature = "host-runtime"))]
impl<T: Copy> RuntimeElement for T {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    driver: DriverSelection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DriverSelection {
    Auto,
    Explicit(String),
}

impl RuntimeConfig {
    pub fn auto() -> Self {
        Self {
            driver: DriverSelection::Auto,
        }
    }

    pub fn driver(name: impl Into<String>) -> Self {
        Self {
            driver: DriverSelection::Explicit(name.into()),
        }
    }

    pub fn backend(backend: Backend) -> Self {
        Self::driver(backend.default_driver())
    }

    #[cfg(feature = "host-runtime")]
    fn driver_name(&self) -> &str {
        match &self.driver {
            DriverSelection::Auto => "local-task",
            DriverSelection::Explicit(name) => name,
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::auto()
    }
}

#[cfg(feature = "host-runtime")]
fn driver_for_backend(backend: &'static str) -> crate::Result<&'static str> {
    Backend::from_name(backend)
        .map(Backend::default_driver)
        .ok_or(crate::Error::UnsupportedBackend(backend))
}

#[cfg(feature = "host-runtime")]
mod hosted {
    use alloc::{
        collections::BTreeMap,
        string::{String, ToString},
        vec::Vec,
    };
    use std::sync::Mutex;

    use eerie::runtime::{BufferView, DeviceSpec, Function, Program, Runtime, Value};

    use super::{driver_for_backend, RuntimeConfig, RuntimeElement, RuntimeInput, RuntimeOutput};
    use crate::{GraphArtifact, GraphArtifactVariant};

    pub struct Engine {
        driver_name: String,
        runtime: Runtime,
        modules: Mutex<BTreeMap<Vec<u8>, Program>>,
    }

    impl Engine {
        pub fn new(config: RuntimeConfig) -> crate::Result<Self> {
            let driver_name = config.driver_name().to_string();
            let runtime = Runtime::new(DeviceSpec::custom(driver_name.clone()))?;
            Ok(Self {
                driver_name,
                runtime,
                modules: Mutex::new(BTreeMap::new()),
            })
        }

        pub fn for_backend(backend: &'static str) -> crate::Result<Self> {
            Self::new(RuntimeConfig::driver(driver_for_backend(backend)?))
        }

        pub fn for_backend_kind(backend: crate::Backend) -> crate::Result<Self> {
            Self::new(RuntimeConfig::backend(backend))
        }

        pub fn for_variant(variant: GraphArtifactVariant) -> crate::Result<Self> {
            Self::new(RuntimeConfig::driver(variant.driver))
        }

        pub fn for_artifact(artifact: GraphArtifact) -> crate::Result<Self> {
            let variant =
                artifact
                    .first_variant()
                    .ok_or(crate::Error::MissingDefaultArtifactVariant {
                        function_name: artifact.function_name,
                    })?;
            Self::for_variant(variant)
        }

        pub fn driver_name(&self) -> &str {
            &self.driver_name
        }

        pub(crate) fn invoke<T: RuntimeElement>(
            &self,
            artifact: GraphArtifact,
            inputs: &[(&[usize], &[T])],
        ) -> crate::Result<Vec<T>> {
            let variant = artifact
                .variant_for_driver(&self.driver_name)
                .ok_or_else(|| crate::Error::MissingArtifactVariant {
                    function_name: artifact.function_name,
                    driver: self.driver_name.clone(),
                })?;
            self.invoke_raw(
                variant.vmfb,
                artifact.function_name,
                variant.backend,
                variant.driver,
                inputs,
            )
        }

        pub(crate) fn invoke_raw<T: RuntimeElement>(
            &self,
            vmfb: &[u8],
            function_name: &'static str,
            backend: &'static str,
            driver: &'static str,
            inputs: &[(&[usize], &[T])],
        ) -> crate::Result<Vec<T>> {
            if self.driver_name != driver {
                return Err(crate::Error::RuntimeDriverMismatch {
                    backend,
                    expected_driver: driver,
                    actual_driver: self.driver_name.clone(),
                });
            }

            let function = {
                let mut modules = self
                    .modules
                    .lock()
                    .map_err(|_| crate::Error::EngineLockPoisoned)?;
                if !modules.contains_key(vmfb) {
                    modules.insert(vmfb.to_vec(), self.runtime.load_vmfb(vmfb)?);
                }
                modules
                    .get(vmfb)
                    .expect("module was just inserted")
                    .function(function_name)?
            };

            let input_buffers: Vec<_> = inputs
                .iter()
                .map(|(shape, data)| self.runtime.buffer_view(shape, data))
                .collect::<Result<_, _>>()?;
            let output = self.invoke_buffer_views(&function, &input_buffers)?;
            Ok(output.read()?)
        }

        pub(crate) fn invoke_typed<T: RuntimeOutput>(
            &self,
            artifact: GraphArtifact,
            inputs: &[RuntimeInput<'_>],
        ) -> crate::Result<Vec<T>> {
            let variant = artifact
                .variant_for_driver(&self.driver_name)
                .ok_or_else(|| crate::Error::MissingArtifactVariant {
                    function_name: artifact.function_name,
                    driver: self.driver_name.clone(),
                })?;
            if self.driver_name != variant.driver {
                return Err(crate::Error::RuntimeDriverMismatch {
                    backend: variant.backend,
                    expected_driver: variant.driver,
                    actual_driver: self.driver_name.clone(),
                });
            }
            let function = self.resolve_function(variant.vmfb, artifact.function_name)?;
            T::read_output(self, &function, inputs)
        }

        fn resolve_function(
            &self,
            vmfb: &'static [u8],
            function_name: &'static str,
        ) -> crate::Result<Function> {
            let mut modules = self
                .modules
                .lock()
                .map_err(|_| crate::Error::EngineLockPoisoned)?;
            if !modules.contains_key(vmfb) {
                modules.insert(vmfb.to_vec(), self.runtime.load_vmfb(vmfb)?);
            }
            modules
                .get(vmfb)
                .expect("module was just inserted")
                .function(function_name)
                .map_err(crate::Error::from)
        }

        pub(crate) fn invoke_typed_values<T: RuntimeElement>(
            &self,
            function: &Function,
            inputs: &[RuntimeInput<'_>],
        ) -> crate::Result<Vec<T>> {
            let input_values = inputs
                .iter()
                .map(|input| self.input_value(input))
                .collect::<crate::Result<Vec<_>>>()?;
            let output = self.invoke_values(function, input_values)?;
            Ok(output.read()?)
        }

        fn invoke_buffer_views<T: RuntimeElement>(
            &self,
            function: &Function,
            inputs: &[BufferView<T>],
        ) -> crate::Result<BufferView<T>> {
            let input_values = inputs.iter().map(T::buffer_to_value).collect::<Vec<_>>();
            self.invoke_values(function, input_values)
        }

        fn invoke_values<T: RuntimeElement>(
            &self,
            function: &Function,
            input_values: Vec<Value>,
        ) -> crate::Result<BufferView<T>> {
            let outputs = function.invoke(input_values)?;
            let actual = outputs.len();
            if actual != 1 {
                return Err(crate::Error::OutputCountMismatch {
                    expected: 1,
                    actual,
                });
            }
            let output = outputs
                .into_iter()
                .next()
                .expect("output count was checked");
            T::buffer_from_value(output).map_err(crate::Error::from)
        }

        fn input_value(&self, input: &RuntimeInput<'_>) -> crate::Result<Value> {
            match input {
                RuntimeInput::Bool(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                RuntimeInput::F32(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                RuntimeInput::F64(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                RuntimeInput::I32(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                RuntimeInput::I64(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                #[cfg(feature = "half")]
                RuntimeInput::F16(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                #[cfg(feature = "half")]
                RuntimeInput::BF16(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
            }
        }
    }
}

#[cfg(not(feature = "host-runtime"))]
mod hosted {
    use alloc::vec::Vec;

    use super::{RuntimeConfig, RuntimeElement, RuntimeInput, RuntimeOutput};
    use crate::GraphArtifact;

    pub struct Engine;

    impl Engine {
        pub fn new(_config: RuntimeConfig) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        pub fn for_backend(_backend: &'static str) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        pub fn for_backend_kind(_backend: crate::Backend) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        pub fn for_variant(_variant: crate::GraphArtifactVariant) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        pub fn for_artifact(_artifact: GraphArtifact) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        pub fn driver_name(&self) -> &str {
            ""
        }

        pub(crate) fn invoke<T: RuntimeElement>(
            &self,
            _artifact: GraphArtifact,
            _inputs: &[(&[usize], &[T])],
        ) -> crate::Result<Vec<T>> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        pub(crate) fn invoke_typed<T: RuntimeOutput>(
            &self,
            _artifact: GraphArtifact,
            _inputs: &[RuntimeInput<'_>],
        ) -> crate::Result<Vec<T>> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        #[allow(dead_code)]
        pub(crate) fn invoke_raw<T: RuntimeElement>(
            &self,
            _vmfb: &[u8],
            _function_name: &'static str,
            _backend: &'static str,
            _driver: &'static str,
            _inputs: &[(&[usize], &[T])],
        ) -> crate::Result<Vec<T>> {
            Err(crate::Error::HostedRuntimeDisabled)
        }
    }
}

pub use hosted::Engine;
