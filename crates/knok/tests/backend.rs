use knok::{
    runtime::raw, Backend, DType, Driver, Engine, Error, GraphArtifact, GraphArtifactVariant,
    RuntimeConfig, TensorDesc, SUPPORTED_BACKENDS,
};

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

#[test]
fn graph_artifact_metadata_selects_variants() {
    static VMFB: [u8; 16] = [0; 16];
    static INPUTS: [TensorDesc; 1] = [TensorDesc::new(DType::F32, &[2, 2])];
    static OUTPUTS: [TensorDesc; 1] = [TensorDesc::new(DType::F32, &[2, 2])];
    static VARIANTS: [GraphArtifactVariant; 2] = [
        GraphArtifactVariant {
            vmfb: &VMFB,
            backend: "llvm-cpu",
            driver: "local-task",
            compile_flags: &["--iree-hal-target-backends=llvm-cpu"],
        },
        GraphArtifactVariant {
            vmfb: &VMFB,
            backend: "metal-spirv",
            driver: "metal",
            compile_flags: &["--iree-hal-target-backends=metal-spirv"],
        },
    ];
    let artifact = GraphArtifact {
        function_name: "forward",
        input_descs: &INPUTS,
        output_descs: &OUTPUTS,
        variants: &VARIANTS,
    };

    assert_eq!(artifact.first_variant().unwrap().driver, "local-task");
    assert_eq!(
        artifact.variant_for_driver("metal").unwrap().backend,
        "metal-spirv"
    );
    assert!(artifact.variant_for_driver("vulkan").is_none());
}

#[test]
fn raw_inputs_report_shape_and_dtype() {
    let bools = [true, false];
    let floats = [1.0_f32, 2.0];
    let ints = [1_i64, 2];

    let bool_input = raw::Input::Bool(&[2], &bools);
    let float_input = raw::Input::F32(&[1, 2], &floats);
    let int_input = raw::Input::I64(&[2], &ints);

    assert_eq!(bool_input.shape(), &[2]);
    assert_eq!(bool_input.dtype(), DType::Bool);
    assert_eq!(float_input.shape(), &[1, 2]);
    assert_eq!(float_input.dtype(), DType::F32);
    assert_eq!(int_input.dtype(), DType::I64);
}

#[test]
fn public_errors_have_actionable_display_text() {
    let error = Error::InputDTypeMismatch {
        index: 1,
        expected: DType::F32,
        actual: DType::I32,
    };
    assert_eq!(
        error.to_string(),
        "runtime input 1 dtype mismatch: expected F32, got I32"
    );

    let shape = Error::Shape {
        expected: &[2, 2],
        actual: vec![3],
    };
    assert_eq!(
        shape.to_string(),
        "tensor shape mismatch: expected [2, 2], got [3]"
    );
}

#[test]
fn engine_for_artifact_rejects_empty_variants_before_runtime_setup() {
    let artifact = GraphArtifact {
        function_name: "forward",
        input_descs: &[],
        output_descs: &[],
        variants: &[],
    };

    let error = match Engine::for_artifact(artifact) {
        Ok(_) => panic!("empty artifact unexpectedly constructed an engine"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        Error::MissingDefaultArtifactVariant {
            function_name: "forward"
        }
    ));
}
