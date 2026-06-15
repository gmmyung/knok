extern crate alloc;

pub type OutputF32 = alloc::vec::Vec<f32>;

#[cfg(feature = "host-runtime")]
pub fn invoke_f32(
    vmfb: &[u8],
    function_name: &'static str,
    backend: &'static str,
    inputs: &[(&[usize], &[f32])],
) -> crate::Result<OutputF32> {
    crate::runtime::invoke_f32(vmfb, function_name, backend, inputs)
}

#[cfg(not(feature = "host-runtime"))]
pub fn invoke_f32(
    _vmfb: &[u8],
    _function_name: &'static str,
    _backend: &'static str,
    _inputs: &[(&[usize], &[f32])],
) -> crate::Result<OutputF32> {
    Err(crate::Error::HostedRuntimeDisabled)
}
