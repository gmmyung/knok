extern crate alloc;

use alloc::string::String;

use crate::Backend;

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

    use eerie::runtime::{hal, vm};
    use vm::ToRef;

    use super::{driver_for_backend, RuntimeConfig};
    use crate::{GraphArtifact, GraphArtifactVariant};

    pub struct Engine {
        driver_name: String,
        modules: Mutex<BTreeMap<Vec<u8>, LoadedModule>>,
        hal_module: vm::Module,
        device: hal::Device,
        _driver: hal::Driver,
        _registry: hal::DriverRegistry,
        instance: vm::Instance,
    }

    struct LoadedModule {
        functions: BTreeMap<String, vm::Function>,
        _context: vm::Context,
        _bytecode_module: vm::Module,
    }

    impl Engine {
        pub fn new(config: RuntimeConfig) -> crate::Result<Self> {
            let driver_name = config.driver_name().to_string();
            let instance = vm::Instance::new()?;
            let registry = hal::DriverRegistry::with_available_drivers()?;
            let driver = registry.create_driver(&driver_name)?;
            let device = driver.create_default_device()?;
            let hal_module = vm::Module::hal(&instance, &device)?;
            Ok(Self {
                driver_name,
                modules: Mutex::new(BTreeMap::new()),
                hal_module,
                device,
                _driver: driver,
                _registry: registry,
                instance,
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

        pub fn invoke_f32(
            &self,
            artifact: GraphArtifact,
            inputs: &[(&[usize], &[f32])],
        ) -> crate::Result<Vec<f32>> {
            let variant = artifact
                .variant_for_driver(&self.driver_name)
                .ok_or_else(|| crate::Error::MissingArtifactVariant {
                    function_name: artifact.function_name,
                    driver: self.driver_name.clone(),
                })?;
            self.invoke_raw_f32(
                variant.vmfb,
                artifact.function_name,
                variant.backend,
                variant.driver,
                inputs,
            )
        }

        pub(crate) fn invoke_raw_f32(
            &self,
            vmfb: &[u8],
            function_name: &'static str,
            backend: &'static str,
            driver: &'static str,
            inputs: &[(&[usize], &[f32])],
        ) -> crate::Result<Vec<f32>> {
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
                    let bytecode_module = vm::Module::bytecode(&self.instance, vmfb)?;
                    let context = vm::Context::with_modules(
                        &self.instance,
                        &[&self.hal_module, &bytecode_module],
                    )?;
                    modules.insert(
                        vmfb.to_vec(),
                        LoadedModule {
                            functions: BTreeMap::new(),
                            _context: context,
                            _bytecode_module: bytecode_module,
                        },
                    );
                }
                let loaded = modules.get_mut(vmfb).expect("module was just inserted");
                if let Some(function) = loaded.functions.get(function_name) {
                    function.clone()
                } else {
                    let function = loaded._context.resolve_function(function_name)?;
                    loaded
                        .functions
                        .insert(function_name.to_string(), function.clone());
                    function
                }
            };

            let input_buffers: Vec<_> = inputs
                .iter()
                .map(|(shape, data)| {
                    hal::BufferView::<f32>::from_host(
                        &self.device,
                        shape,
                        hal::Encoding::DenseRowMajor,
                        data,
                    )
                })
                .collect::<Result<_, _>>()?;
            let output = self.invoke_buffer_views_f32(&function, &input_buffers)?;
            Ok(output.read_to_vec(&self.device)?)
        }

        fn invoke_buffer_views_f32(
            &self,
            function: &vm::Function,
            inputs: &[hal::BufferView<f32>],
        ) -> crate::Result<hal::BufferView<f32>> {
            let mut input_list = vm::List::<vm::Undefined>::new(inputs.len(), &self.instance)?;
            for input in inputs {
                input_list.push_ref(&input.to_ref(&self.instance)?)?;
            }
            let mut output_list = vm::List::<vm::Undefined>::new(1, &self.instance)?;
            function.invoke(&input_list, &mut output_list)?;
            output_list
                .get_ref::<hal::BufferView<f32>>(0)
                .map_err(crate::Error::from)?
                .to_buffer_view()
                .map_err(crate::Error::from)
        }
    }

    pub fn invoke_f32(
        vmfb: &[u8],
        function_name: &'static str,
        backend: &'static str,
        inputs: &[(&[usize], &[f32])],
    ) -> crate::Result<Vec<f32>> {
        let driver = driver_for_backend(backend)?;
        let engine = Engine::new(RuntimeConfig::driver(driver))?;
        engine.invoke_raw_f32(vmfb, function_name, backend, driver, inputs)
    }
}

#[cfg(not(feature = "host-runtime"))]
mod hosted {
    use alloc::vec::Vec;

    use super::RuntimeConfig;
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

        pub fn invoke_f32(
            &self,
            _artifact: GraphArtifact,
            _inputs: &[(&[usize], &[f32])],
        ) -> crate::Result<Vec<f32>> {
            Err(crate::Error::HostedRuntimeDisabled)
        }

        pub(crate) fn invoke_raw_f32(
            &self,
            _vmfb: &[u8],
            _function_name: &'static str,
            _backend: &'static str,
            _driver: &'static str,
            _inputs: &[(&[usize], &[f32])],
        ) -> crate::Result<Vec<f32>> {
            Err(crate::Error::HostedRuntimeDisabled)
        }
    }

    pub fn invoke_f32(
        _vmfb: &[u8],
        _function_name: &'static str,
        _backend: &'static str,
        _inputs: &[(&[usize], &[f32])],
    ) -> crate::Result<Vec<f32>> {
        Err(crate::Error::HostedRuntimeDisabled)
    }
}

pub use hosted::{invoke_f32, Engine};
