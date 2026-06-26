use knok::{Backend, Driver, Engine, RuntimeConfig, SUPPORTED_BACKENDS};

#[test]
fn backend_capabilities_describe_supported_targets() {
    assert_eq!(SUPPORTED_BACKENDS, &[Backend::LlvmCpu, Backend::MetalSpirv]);
    assert_eq!(Backend::from_name("llvm-cpu"), Some(Backend::LlvmCpu));
    assert_eq!(Backend::from_name("metal-spirv"), Some(Backend::MetalSpirv));
    assert_eq!(Backend::from_name("vulkan-spirv"), None);
    assert_eq!(Backend::LlvmCpu.name(), "llvm-cpu");
    assert_eq!(Backend::LlvmCpu.default_driver(), "local-task");
    assert!(Backend::MetalSpirv.supports_driver(Driver::Metal));
    assert!(!Backend::MetalSpirv.supports_driver(Driver::LocalTask));
    assert_eq!(Driver::from_name("metal"), Some(Driver::Metal));
    assert_eq!(Driver::from_name("vulkan"), None);
}

#[test]
fn runtime_config_can_be_constructed_from_backend() {
    let engine = Engine::new(RuntimeConfig::backend(Backend::LlvmCpu)).unwrap();

    assert_eq!(engine.driver_name(), "local-task");
}

#[test]
fn engine_can_be_constructed_from_backend() {
    let engine = Engine::for_backend(Backend::LlvmCpu).unwrap();

    assert_eq!(engine.driver_name(), "local-task");
}
