extern crate alloc;

pub fn invoke<T: crate::RuntimeElement>(
    vmfb: &[u8],
    function_name: &'static str,
    backend: &'static str,
    inputs: &[(&[usize], &[T])],
) -> crate::Result<alloc::vec::Vec<T>> {
    crate::runtime::invoke(vmfb, function_name, backend, inputs)
}

pub fn invoke_with_engine<T: crate::RuntimeElement>(
    engine: &crate::Engine,
    vmfb: &[u8],
    function_name: &'static str,
    backend: &'static str,
    driver: &'static str,
    inputs: &[(&[usize], &[T])],
) -> crate::Result<alloc::vec::Vec<T>> {
    engine.invoke_raw(vmfb, function_name, backend, driver, inputs)
}
