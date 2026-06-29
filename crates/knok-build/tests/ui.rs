#[test]
fn graph_macro_diagnostics_are_stable() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/missing_backend.rs");
    tests.compile_fail("tests/ui/missing_return_type.rs");
    tests.compile_fail("tests/ui/old_axis_syntax.rs");
    tests.compile_fail("tests/ui/pattern_argument.rs");

    #[cfg(not(feature = "vulkan"))]
    tests.compile_fail("tests/ui/missing_vulkan_feature.rs");

    #[cfg(not(feature = "cuda"))]
    tests.compile_fail("tests/ui/missing_cuda_feature.rs");

    #[cfg(not(target_os = "macos"))]
    tests.compile_fail("tests/ui/missing_metal_platform.rs");
}
