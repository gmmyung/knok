extern crate alloc;

use alloc::string::String;

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
    match backend {
        "llvm-cpu" => Ok("local-task"),
        "metal-spirv" => Ok("metal"),
        other => Err(crate::Error::UnsupportedBackend(other)),
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

    use eerie::runtime::{hal, vm};

    use super::{driver_for_backend, RuntimeConfig};
    use crate::GraphArtifact;

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

        pub fn driver_name(&self) -> &str {
            &self.driver_name
        }

        pub fn invoke_f32(
            &self,
            artifact: GraphArtifact,
            inputs: &[(&[usize], &[f32])],
        ) -> crate::Result<Vec<f32>> {
            self.invoke_raw_f32(
                artifact.vmfb,
                artifact.function_name,
                artifact.backend,
                inputs,
            )
        }

        pub(crate) fn invoke_raw_f32(
            &self,
            vmfb: &[u8],
            function_name: &'static str,
            backend: &'static str,
            inputs: &[(&[usize], &[f32])],
        ) -> crate::Result<Vec<f32>> {
            let expected_driver = driver_for_backend(backend)?;
            if self.driver_name != expected_driver {
                return Err(crate::Error::RuntimeDriverMismatch {
                    backend,
                    expected_driver,
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

            let tensors: Vec<_> = inputs
                .iter()
                .map(|(shape, data)| hal::Tensor::<f32>::from_slice(&self.device, shape, data))
                .collect::<Result<_, _>>()?;
            let input_refs: Vec<_> = tensors.iter().collect();
            let outputs = function.invoke_tensors(&input_refs, 1)?;
            let output = outputs
                .into_iter()
                .next()
                .ok_or(crate::Error::MissingOutput)?;
            Ok(output.read_to_vec(&self.device)?)
        }
    }

    pub fn invoke_f32(
        vmfb: &[u8],
        function_name: &'static str,
        backend: &'static str,
        inputs: &[(&[usize], &[f32])],
    ) -> crate::Result<Vec<f32>> {
        let engine = Engine::for_backend(backend)?;
        engine.invoke_raw_f32(vmfb, function_name, backend, inputs)
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
