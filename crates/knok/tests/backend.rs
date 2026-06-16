use knok::{Backend, Engine, RuntimeConfig, SUPPORTED_BACKENDS};

#[test]
fn backend_capabilities_describe_supported_targets() {
    assert_eq!(SUPPORTED_BACKENDS, &[Backend::LlvmCpu, Backend::MetalSpirv]);
    assert_eq!(Backend::from_name("llvm-cpu"), Some(Backend::LlvmCpu));
    assert_eq!(Backend::from_name("metal-spirv"), Some(Backend::MetalSpirv));
    assert_eq!(Backend::from_name("vulkan-spirv"), None);
    assert_eq!(Backend::LlvmCpu.name(), "llvm-cpu");
    assert_eq!(Backend::LlvmCpu.default_driver(), "local-task");
    assert!(Backend::MetalSpirv.supports_driver("metal"));
    assert!(!Backend::MetalSpirv.supports_driver("local-task"));
}

#[test]
fn runtime_config_can_be_constructed_from_backend() {
    let engine = Engine::new(RuntimeConfig::backend(Backend::LlvmCpu)).unwrap();

    assert_eq!(engine.driver_name(), "local-task");
}

#[test]
fn engine_can_be_constructed_from_backend_kind() {
    let engine = Engine::for_backend_kind(Backend::LlvmCpu).unwrap();

    assert_eq!(engine.driver_name(), "local-task");
}
