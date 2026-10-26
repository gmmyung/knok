use knok::{
    tensor::{FixedTensor, Tensor1, Tensor2},
    Backend, DType, Driver, Error, Graph, GraphArtifact, GraphArtifactVariant, GraphSignatureError,
    TensorDesc, SUPPORTED_BACKENDS,
};
#[cfg(feature = "runtime")]
use knok::{Engine, RuntimeConfig};

#[test]
fn backend_capabilities_describe_supported_targets() {
    for &backend in SUPPORTED_BACKENDS {
        assert_eq!(Backend::from_name(backend.name()), Some(backend));
        assert!(backend.supports_driver(Driver::from_name(backend.default_driver()).unwrap()));
    }
    assert!(Backend::from_name("unknown").is_none());
    assert!(Driver::from_name("unknown").is_none());
    assert!(Backend::LlvmCpu.supports_driver(Driver::LocalSync));
    assert!(Backend::LlvmCpu.supports_driver(Driver::LocalTask));
    #[cfg(target_os = "macos")]
    assert!(!Backend::LlvmCpu.supports_driver(Driver::Metal));
}

const INPUTS: &[TensorDesc] = &[TensorDesc::new(DType::F32, &[1, 2])];
const OUTPUTS: &[TensorDesc] = &[TensorDesc::new(DType::I64, &[2])];
const VARIANTS: &[GraphArtifactVariant] = &[GraphArtifactVariant::new(
    &[0; 16],
    Backend::LlvmCpu,
    &["--iree-hal-target-backends=llvm-cpu"],
)];
const ARTIFACT: GraphArtifact = GraphArtifact::new("forward", INPUTS, OUTPUTS, VARIANTS);
type Input = Tensor2<f32, 1, 2>;
type Output = Tensor1<i64, 2>;

#[test]
fn artifact_metadata_selects_compatible_drivers() {
    let variant = ARTIFACT.variant_for_driver(Driver::LocalSync).unwrap();
    assert_eq!(variant.backend(), Backend::LlvmCpu);
    assert_eq!(variant.vmfb(), &[0; 16]);
    assert_eq!(
        variant.compile_flags(),
        &["--iree-hal-target-backends=llvm-cpu"]
    );
    assert_eq!(ARTIFACT.variants().len(), 1);
    assert!(ARTIFACT.first_variant().is_some());
}

#[test]
fn graph_checks_complete_static_signature() {
    fn assert_copy<T: Copy>() {}
    assert_copy::<Graph<Input, Output>>();
    const GRAPH: Graph<Input, Output> = match Graph::new(ARTIFACT) {
        Ok(graph) => graph,
        Err(_) => panic!("invalid graph"),
    };
    assert_eq!(GRAPH.artifact().function_name(), "forward");
    assert_eq!(GRAPH.artifact().input_descs(), INPUTS);
    assert_eq!(GRAPH.artifact().output_descs(), OUTPUTS);
    assert_eq!(<Input as FixedTensor<f32>>::SHAPE, &[1, 2]);
    assert!(matches!(
        Graph::<Tensor2<f32, 2, 1>, Output>::new(ARTIFACT),
        Err(GraphSignatureError::Input)
    ));
    assert!(matches!(
        Graph::<Input, Tensor1<f32, 2>>::new(ARTIFACT),
        Err(GraphSignatureError::Output)
    ));
    assert!(matches!(
        Graph::<Input, (Output, Output)>::new(ARTIFACT),
        Err(GraphSignatureError::Output)
    ));
    assert!(matches!(
        Graph::<(), ()>::new(ARTIFACT),
        Err(GraphSignatureError::Input)
    ));
    let empty = GraphArtifact::new("empty", &[], &[], &[]);
    assert!(Graph::<(), ()>::new(empty).is_ok());
    assert!(Graph::<Input, ()>::new(empty).is_err());
    assert!(Graph::<(), Output>::new(empty).is_err());
}

#[test]
fn public_errors_have_actionable_display_text() {
    assert_eq!(
        Error::from(GraphSignatureError::Output).to_string(),
        "graph output signature does not match the typed handle"
    );
    assert_eq!(
        Error::OutputShapeMismatch {
            index: 1,
            expected: &[2, 2],
            actual: vec![4]
        }
        .to_string(),
        "runtime output 1 shape mismatch: expected [2, 2], got [4]"
    );
    assert_eq!(
        Error::Shape {
            expected: &[2, 2],
            actual: vec![3]
        }
        .to_string(),
        "tensor shape mismatch: expected [2, 2], got [3]"
    );
}

#[cfg(feature = "runtime")]
#[test]
fn cpu_device_selection_supports_ordinals_and_paths() {
    let devices = Engine::available_devices(Driver::LocalSync).unwrap();
    assert!(!devices.is_empty());
    let engine = Engine::new(RuntimeConfig::driver(Driver::LocalSync).device_ordinal(0)).unwrap();
    assert_eq!(engine.driver_name(), "local-sync");
    let engine = Engine::new(
        RuntimeConfig::driver(Driver::LocalSync)
            .device_ordinal(usize::MAX)
            .device_path(devices[0].path()),
    )
    .unwrap();
    assert_eq!(engine.driver_name(), "local-sync");
    let engine = Engine::new(RuntimeConfig::backend(Backend::LlvmCpu)).unwrap();
    assert_eq!(engine.driver_name(), "local-task");
}

#[cfg(feature = "runtime")]
#[test]
fn engine_rejects_empty_artifacts_before_runtime_setup() {
    let artifact = GraphArtifact::new("forward", &[], &[], &[]);
    assert!(matches!(
        Engine::for_artifact(artifact),
        Err(Error::MissingDefaultArtifactVariant {
            function_name: "forward"
        })
    ));
}
