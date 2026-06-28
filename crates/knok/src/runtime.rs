extern crate alloc;

use alloc::string::String;

use crate::Backend;

/// Raw runtime invocation types.
///
/// Generated typed wrappers are preferred for normal use. This module is for
/// manually invoking a [`crate::GraphArtifact`] with explicit input buffers and
/// reading output buffers by dtype.
pub mod raw {
    extern crate alloc;

    #[cfg(feature = "host-runtime")]
    use alloc::vec::Vec;

    use crate::DType;

    /// Raw input buffer passed to a compiled graph.
    pub enum Input<'a> {
        /// Boolean input buffer.
        Bool(&'a [usize], &'a [bool]),
        /// `f32` input buffer.
        F32(&'a [usize], &'a [f32]),
        /// `f64` input buffer.
        F64(&'a [usize], &'a [f64]),
        /// `i32` input buffer.
        I32(&'a [usize], &'a [i32]),
        /// `i64` input buffer.
        I64(&'a [usize], &'a [i64]),
        /// `f16` input buffer.
        #[cfg(feature = "half")]
        F16(&'a [usize], &'a [crate::half::f16]),
        /// `bf16` input buffer.
        #[cfg(feature = "half")]
        BF16(&'a [usize], &'a [crate::half::bf16]),
    }

    impl Input<'_> {
        /// Returns the static shape supplied for this raw input buffer.
        pub fn shape(&self) -> &[usize] {
            match self {
                Self::Bool(shape, _)
                | Self::F32(shape, _)
                | Self::F64(shape, _)
                | Self::I32(shape, _)
                | Self::I64(shape, _) => shape,
                #[cfg(feature = "half")]
                Self::F16(shape, _) | Self::BF16(shape, _) => shape,
            }
        }

        /// Returns the element type supplied for this raw input buffer.
        pub fn dtype(&self) -> DType {
            match self {
                Self::Bool(_, _) => DType::Bool,
                Self::F32(_, _) => DType::F32,
                Self::F64(_, _) => DType::F64,
                Self::I32(_, _) => DType::I32,
                Self::I64(_, _) => DType::I64,
                #[cfg(feature = "half")]
                Self::F16(_, _) => DType::F16,
                #[cfg(feature = "half")]
                Self::BF16(_, _) => DType::BF16,
            }
        }
    }

    /// Element types supported by the raw hosted single-output convenience path.
    pub trait Output: Copy {
        /// Reads a raw output collection as this element type.
        #[cfg(feature = "host-runtime")]
        fn read_output(outputs: Outputs) -> crate::Result<alloc::vec::Vec<Self>>;
    }

    #[cfg(feature = "host-runtime")]
    macro_rules! impl_output {
        ($ty:ty) => {
            impl Output for $ty {
                fn read_output(outputs: Outputs) -> crate::Result<alloc::vec::Vec<Self>> {
                    outputs.one::<Self>()
                }
            }
        };
    }

    #[cfg(feature = "host-runtime")]
    impl_output!(f32);
    #[cfg(feature = "host-runtime")]
    impl_output!(f64);
    #[cfg(feature = "host-runtime")]
    impl_output!(bool);
    #[cfg(feature = "host-runtime")]
    impl_output!(i32);
    #[cfg(feature = "host-runtime")]
    impl_output!(i64);

    #[cfg(all(feature = "host-runtime", feature = "half"))]
    impl_output!(crate::half::f16);
    #[cfg(all(feature = "host-runtime", feature = "half"))]
    impl_output!(crate::half::bf16);

    #[cfg(not(feature = "host-runtime"))]
    impl<T: Copy> Output for T {}

    /// Outputs returned by a raw hosted graph invocation.
    #[cfg(feature = "host-runtime")]
    pub struct Outputs {
        pub(super) values: Vec<eerie::runtime::Value>,
    }

    #[cfg(feature = "host-runtime")]
    impl core::fmt::Debug for Outputs {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter
                .debug_struct("Outputs")
                .field("len", &self.values.len())
                .finish()
        }
    }

    #[cfg(feature = "host-runtime")]
    impl Outputs {
        /// Returns the number of values produced by the invoked function.
        pub fn len(&self) -> usize {
            self.values.len()
        }

        /// Returns true when the invoked function produced no values.
        pub fn is_empty(&self) -> bool {
            self.values.is_empty()
        }

        /// Reads the only output as a host vector.
        pub fn one<T: Element>(mut self) -> crate::Result<Vec<T>> {
            let actual = self.values.len();
            if actual != 1 {
                return Err(crate::Error::OutputCountMismatch {
                    expected: 1,
                    actual,
                });
            }
            let value = self.values.pop().expect("output count was checked");
            let output = T::buffer_from_value(value)?;
            Ok(output.read()?)
        }

        /// Reads one output by index as a host vector.
        pub fn read<T: Element>(&self, index: usize) -> crate::Result<Vec<T>> {
            let value = self
                .values
                .get(index)
                .ok_or(crate::Error::OutputIndexOutOfBounds {
                    index,
                    len: self.values.len(),
                })?;
            let output = T::buffer_from_value(value.clone())?;
            Ok(output.read()?)
        }
    }

    #[cfg(not(feature = "host-runtime"))]
    #[doc(hidden)]
    pub struct Outputs;

    #[cfg(not(feature = "host-runtime"))]
    impl Outputs {
        /// Returns zero because hosted execution is disabled.
        pub fn len(&self) -> usize {
            0
        }

        /// Returns true because hosted execution is disabled.
        pub fn is_empty(&self) -> bool {
            true
        }

        /// Always returns [`crate::Error::HostedRuntimeDisabled`].
        pub fn one<T: Element>(self) -> crate::Result<alloc::vec::Vec<T>> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        /// Always returns [`crate::Error::HostedRuntimeDisabled`].
        pub fn read<T: Element>(&self, _index: usize) -> crate::Result<alloc::vec::Vec<T>> {
            Err(crate::Error::HostedRuntimeDisabled)
        }
    }

    /// Element types that can be passed through raw runtime buffer views.
    #[cfg(feature = "host-runtime")]
    #[doc(hidden)]
    pub trait Element: eerie::runtime::BufferElement {
        fn buffer_from_value(
            value: eerie::runtime::Value,
        ) -> Result<eerie::runtime::BufferView<Self>, eerie::runtime::RuntimeError>;
    }

    #[cfg(feature = "host-runtime")]
    macro_rules! impl_element {
        ($type:ty) => {
            impl Element for $type {
                fn buffer_from_value(
                    value: eerie::runtime::Value,
                ) -> Result<eerie::runtime::BufferView<Self>, eerie::runtime::RuntimeError> {
                    value.try_into()
                }
            }
        };
    }

    #[cfg(feature = "host-runtime")]
    impl_element!(bool);
    #[cfg(feature = "host-runtime")]
    impl_element!(u8);
    #[cfg(feature = "host-runtime")]
    impl_element!(u16);
    #[cfg(feature = "host-runtime")]
    impl_element!(u32);
    #[cfg(feature = "host-runtime")]
    impl_element!(u64);
    #[cfg(feature = "host-runtime")]
    impl_element!(i8);
    #[cfg(feature = "host-runtime")]
    impl_element!(i16);
    #[cfg(feature = "host-runtime")]
    impl_element!(i32);
    #[cfg(feature = "host-runtime")]
    impl_element!(i64);
    #[cfg(all(feature = "host-runtime", feature = "half"))]
    impl_element!(half::f16);
    #[cfg(feature = "host-runtime")]
    impl_element!(f32);
    #[cfg(feature = "host-runtime")]
    impl_element!(f64);
    #[cfg(all(feature = "host-runtime", feature = "half"))]
    impl_element!(half::bf16);

    /// Element types that can be named by generated runtime wrappers in no-std builds.
    #[cfg(not(feature = "host-runtime"))]
    #[doc(hidden)]
    pub trait Element: Copy {}

    #[cfg(not(feature = "host-runtime"))]
    impl<T: Copy> Element for T {}
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Configuration used to construct a reusable runtime engine.
pub struct RuntimeConfig {
    driver: DriverSelection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DriverSelection {
    Auto,
    Explicit(String),
}

impl RuntimeConfig {
    /// Selects the default runtime driver.
    ///
    /// The current default is the local-task CPU driver.
    pub fn auto() -> Self {
        Self {
            driver: DriverSelection::Auto,
        }
    }

    /// Selects an explicit runtime driver.
    pub fn driver(driver: crate::Driver) -> Self {
        Self {
            driver: DriverSelection::Explicit(driver.name().into()),
        }
    }

    /// Selects the default driver for a compile-time backend.
    pub fn backend(backend: Backend) -> Self {
        Self {
            driver: DriverSelection::Explicit(backend.default_driver().into()),
        }
    }

    #[cfg(feature = "host-runtime")]
    fn driver_name_literal(name: &'static str) -> Self {
        Self {
            driver: DriverSelection::Explicit(name.into()),
        }
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
mod hosted {
    use alloc::{
        collections::BTreeMap,
        string::{String, ToString},
        vec::Vec,
    };
    use std::sync::Mutex;

    use eerie::runtime::{DeviceSpec, Function, Program, Runtime, Value};

    use super::{raw, RuntimeConfig};
    use crate::{GraphArtifact, GraphArtifactVariant};

    /// Reusable hosted runtime engine.
    ///
    /// An engine owns one IREE runtime and lazily caches loaded VMFB modules.
    /// Reuse it for repeated inference instead of calling generated one-shot
    /// wrappers when latency matters.
    pub struct Engine {
        driver_name: String,
        runtime: Runtime,
        modules: Mutex<BTreeMap<Vec<u8>, Program>>,
    }

    impl Engine {
        /// Creates a runtime engine from a runtime configuration.
        pub fn new(config: RuntimeConfig) -> crate::Result<Self> {
            let driver_name = config.driver_name().to_string();
            let runtime = Runtime::new(DeviceSpec::custom(driver_name.clone()))?;
            Ok(Self {
                driver_name,
                runtime,
                modules: Mutex::new(BTreeMap::new()),
            })
        }

        /// Creates an engine using the default driver for `backend`.
        pub fn for_backend(backend: crate::Backend) -> crate::Result<Self> {
            Self::new(RuntimeConfig::backend(backend))
        }

        /// Creates an engine compatible with a specific artifact variant.
        pub fn for_variant(variant: GraphArtifactVariant) -> crate::Result<Self> {
            Self::new(RuntimeConfig::driver_name_literal(variant.driver))
        }

        /// Creates an engine for the first variant embedded in `artifact`.
        pub fn for_artifact(artifact: GraphArtifact) -> crate::Result<Self> {
            let variant =
                artifact
                    .first_variant()
                    .ok_or(crate::Error::MissingDefaultArtifactVariant {
                        function_name: artifact.function_name,
                    })?;
            Self::for_variant(variant)
        }

        /// Returns the IREE runtime driver name used by this engine.
        pub fn driver_name(&self) -> &str {
            &self.driver_name
        }

        /// Invokes an artifact with raw input buffers.
        ///
        /// Typed graph wrappers call this internally. Direct callers must pass
        /// input buffers whose dtype and shape match the artifact metadata.
        pub fn invoke(
            &self,
            artifact: GraphArtifact,
            inputs: &[raw::Input<'_>],
        ) -> crate::Result<raw::Outputs> {
            if artifact.has_typed_signature() {
                validate_inputs(artifact, inputs)?;
            }
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
            let outputs = self.invoke_typed_values(&function, inputs)?;
            if artifact.has_typed_signature() && outputs.len() != artifact.output_descs.len() {
                return Err(crate::Error::OutputCountMismatch {
                    expected: artifact.output_descs.len(),
                    actual: outputs.len(),
                });
            }
            Ok(outputs)
        }

        /// Invokes an artifact and reads its single output as `T`.
        pub fn invoke_one<T: raw::Output>(
            &self,
            artifact: GraphArtifact,
            inputs: &[raw::Input<'_>],
        ) -> crate::Result<Vec<T>> {
            let outputs = self.invoke(artifact, inputs)?;
            T::read_output(outputs)
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

        fn invoke_typed_values(
            &self,
            function: &Function,
            inputs: &[raw::Input<'_>],
        ) -> crate::Result<raw::Outputs> {
            let input_values = inputs
                .iter()
                .map(|input| self.input_value(input))
                .collect::<crate::Result<Vec<_>>>()?;
            self.invoke_values(function, input_values)
        }

        fn invoke_values(
            &self,
            function: &Function,
            input_values: Vec<Value>,
        ) -> crate::Result<raw::Outputs> {
            Ok(raw::Outputs {
                values: function.invoke(input_values)?,
            })
        }

        fn input_value(&self, input: &raw::Input<'_>) -> crate::Result<Value> {
            match input {
                raw::Input::Bool(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                raw::Input::F32(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                raw::Input::F64(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                raw::Input::I32(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                raw::Input::I64(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                #[cfg(feature = "half")]
                raw::Input::F16(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
                #[cfg(feature = "half")]
                raw::Input::BF16(shape, data) => {
                    Ok(Value::from(self.runtime.buffer_view(shape, data)?))
                }
            }
        }
    }

    fn validate_inputs(artifact: GraphArtifact, inputs: &[raw::Input<'_>]) -> crate::Result<()> {
        if inputs.len() != artifact.input_descs.len() {
            return Err(crate::Error::InputCountMismatch {
                expected: artifact.input_descs.len(),
                actual: inputs.len(),
            });
        }
        for (index, (input, expected)) in inputs.iter().zip(artifact.input_descs).enumerate() {
            let actual_dtype = input.dtype();
            if actual_dtype != expected.elem {
                return Err(crate::Error::InputDTypeMismatch {
                    index,
                    expected: expected.elem,
                    actual: actual_dtype,
                });
            }
            let actual_shape = input.shape();
            if actual_shape != expected.shape {
                return Err(crate::Error::Shape {
                    expected: expected.shape,
                    actual: actual_shape.to_vec(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "host-runtime"))]
mod hosted {
    use alloc::vec::Vec;

    use super::{raw, RuntimeConfig};
    use crate::GraphArtifact;

    /// Placeholder engine used when hosted runtime support is disabled.
    pub struct Engine;

    impl Engine {
        /// Always returns [`crate::Error::HostedRuntimeDisabled`].
        pub fn new(_config: RuntimeConfig) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        /// Always returns [`crate::Error::HostedRuntimeDisabled`].
        pub fn for_backend(_backend: crate::Backend) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        /// Always returns [`crate::Error::HostedRuntimeDisabled`].
        pub fn for_variant(_variant: crate::GraphArtifactVariant) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        /// Always returns [`crate::Error::HostedRuntimeDisabled`].
        pub fn for_artifact(_artifact: GraphArtifact) -> crate::Result<Self> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        /// Returns an empty driver name because hosted execution is disabled.
        pub fn driver_name(&self) -> &str {
            ""
        }

        /// Always returns [`crate::Error::HostedRuntimeDisabled`].
        pub fn invoke(
            &self,
            _artifact: GraphArtifact,
            _inputs: &[raw::Input<'_>],
        ) -> crate::Result<raw::Outputs> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        /// Always returns [`crate::Error::HostedRuntimeDisabled`].
        pub fn invoke_one<T: raw::Output>(
            &self,
            _artifact: GraphArtifact,
            _inputs: &[raw::Input<'_>],
        ) -> crate::Result<Vec<T>> {
            Err(crate::Error::HostedRuntimeDisabled)
        }
    }
}

/// Reusable runtime engine for generated graph artifacts.
pub use hosted::Engine;
