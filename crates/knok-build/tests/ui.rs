#[test]
fn graph_macro_diagnostics_are_stable() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/*.rs");
}
