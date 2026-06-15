use alloc::vec::Vec;
use eerie::runtime::{hal, vm};

pub fn invoke_f32(
    vmfb: &[u8],
    function_name: &'static str,
    backend: &'static str,
    inputs: &[(&[usize], &[f32])],
) -> crate::Result<Vec<f32>> {
    let instance = vm::Instance::new()?;
    let registry = hal::DriverRegistry::with_available_drivers()?;
    let driver_name = match backend {
        "llvm-cpu" => "local-task",
        "metal-spirv" => "metal",
        other => return Err(crate::Error::UnsupportedBackend(other)),
    };
    let driver = registry.create_driver(driver_name)?;
    let device = driver.create_default_device()?;
    let hal_module = vm::Module::hal(&instance, &device)?;
    let bytecode_module = vm::Module::bytecode(&instance, vmfb)?;
    let context = vm::Context::with_modules(&instance, &[&hal_module, &bytecode_module])?;
    let function = context.resolve_function(function_name)?;

    let tensors: Vec<_> = inputs
        .iter()
        .map(|(shape, data)| hal::Tensor::<f32>::from_slice(&device, shape, data))
        .collect::<Result<_, _>>()?;
    let input_refs: Vec<_> = tensors.iter().collect();
    let outputs = function.invoke_tensors(&input_refs, 1)?;
    let output = outputs
        .into_iter()
        .next()
        .ok_or(crate::Error::MissingOutput)?;
    Ok(output.read_to_vec(&device)?)
}
