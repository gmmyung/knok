extern crate alloc;

pub type OutputF32 = alloc::vec::Vec<f32>;

pub fn invoke_f32(
    vmfb: &[u8],
    function_name: &'static str,
    backend: &'static str,
    inputs: &[(&[usize], &[f32])],
) -> crate::Result<OutputF32> {
    crate::runtime::invoke_f32(vmfb, function_name, backend, inputs)
}

pub fn invoke_f32_with_engine(
    engine: &crate::Engine,
    vmfb: &[u8],
    function_name: &'static str,
    backend: &'static str,
    inputs: &[(&[usize], &[f32])],
) -> crate::Result<OutputF32> {
    engine.invoke_raw_f32(vmfb, function_name, backend, inputs)
}
