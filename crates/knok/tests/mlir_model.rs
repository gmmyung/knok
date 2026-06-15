knok::mlir_model! {
    name: imported_add4,
    path: "tests/fixtures/add4.mlir",
    backend: "llvm-cpu",
    function: "imported.add4",
}

#[test]
fn imported_mlir_model_runs() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [10.0, 20.0, 30.0, 40.0];
    let output = imported_add4::invoke_f32(&[(&[4], &x), (&[4], &y)]).unwrap();
    assert_eq!(output, vec![11.0, 22.0, 33.0, 44.0]);
}
