use knok::{
    tensor::{FixedTensor, Tensor1, Tensor2},
    Backend, DType, Driver, Engine, Error, Graph, GraphArtifact, GraphArtifactVariant,
    RuntimeConfig, TensorDesc, SUPPORTED_BACKENDS,
};

#[test]
fn backend_capabilities_describe_supported_targets() {
    let expected_backends = [
        Backend::LlvmCpu,
        #[cfg(any(target_os = "macos", doc))]
        Backend::MetalSpirv,
        #[cfg(feature = "vulkan")]
        Backend::VulkanSpirv,
        #[cfg(feature = "cuda")]
        Backend::Cuda,
    ];

    assert_eq!(SUPPORTED_BACKENDS, &expected_backends);
    assert_eq!(Backend::from_name("llvm-cpu"), Some(Backend::LlvmCpu));
    #[cfg(any(target_os = "macos", doc))]
    assert_eq!(Backend::from_name("metal-spirv"), Some(Backend::MetalSpirv));
    #[cfg(not(any(target_os = "macos", doc)))]
    assert_eq!(Backend::from_name("metal-spirv"), None);
    #[cfg(feature = "vulkan")]
    assert_eq!(
        Backend::from_name("vulkan-spirv"),
        Some(Backend::VulkanSpirv)
    );
    #[cfg(not(feature = "vulkan"))]
    assert_eq!(Backend::from_name("vulkan-spirv"), None);
    #[cfg(feature = "cuda")]
    assert_eq!(Backend::from_name("cuda"), Some(Backend::Cuda));
    #[cfg(not(feature = "cuda"))]
    assert_eq!(Backend::from_name("cuda"), None);
    assert_eq!(Backend::LlvmCpu.name(), "llvm-cpu");
    assert_eq!(Backend::LlvmCpu.default_driver(), "local-task");
    #[cfg(any(target_os = "macos", doc))]
    {
        assert!(Backend::MetalSpirv.supports_driver(Driver::Metal));
        assert!(!Backend::MetalSpirv.supports_driver(Driver::LocalTask));
    }
    #[cfg(feature = "vulkan")]
    assert!(Backend::VulkanSpirv.supports_driver(Driver::Vulkan));
    #[cfg(feature = "cuda")]
    assert!(Backend::Cuda.supports_driver(Driver::Cuda));
    #[cfg(any(target_os = "macos", doc))]
    assert_eq!(Driver::from_name("metal"), Some(Driver::Metal));
    #[cfg(not(any(target_os = "macos", doc)))]
    assert_eq!(Driver::from_name("metal"), None);
    #[cfg(feature = "vulkan")]
    assert_eq!(Driver::from_name("vulkan"), Some(Driver::Vulkan));
    #[cfg(not(feature = "vulkan"))]
    assert_eq!(Driver::from_name("vulkan"), None);
    #[cfg(feature = "cuda")]
    assert_eq!(Driver::from_name("cuda"), Some(Driver::Cuda));
    #[cfg(not(feature = "cuda"))]
    assert_eq!(Driver::from_name("cuda"), None);
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
fn typed_graph_handle_preserves_artifact_and_tensor_metadata() {
    static VMFB: [u8; 16] = [0; 16];
    static INPUTS: [TensorDesc; 1] = [TensorDesc::new(DType::F32, &[1, 2])];
    static OUTPUTS: [TensorDesc; 1] = [TensorDesc::new(DType::I64, &[2])];
    static VARIANTS: [GraphArtifactVariant; 1] = [GraphArtifactVariant {
        vmfb: &VMFB,
        backend: "llvm-cpu",
        driver: "local-task",
        compile_flags: &[],
    }];
    let artifact = GraphArtifact {
        function_name: "forward",
        input_descs: &INPUTS,
        output_descs: &OUTPUTS,
        variants: &VARIANTS,
    };
    let graph: Graph<Tensor2<f32, 1, 2>, Tensor1<i64, 2>> = Graph::new(artifact);

    assert_eq!(graph.artifact().function_name, "forward");
    assert_eq!(graph.artifact().input_descs[0].elem, DType::F32);
    assert_eq!(<Tensor2<f32, 1, 2> as FixedTensor<f32>>::SHAPE, &[1, 2]);
    assert_eq!(<Tensor1<i64, 2> as FixedTensor<i64>>::SHAPE, &[2]);
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
