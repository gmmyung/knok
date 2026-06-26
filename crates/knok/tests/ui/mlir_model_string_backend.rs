knok::mlir_model! {
    name: imported_add4,
    path: "../../../../crates/knok/tests/fixtures/add4.mlir",
    backend: "llvm-cpu",
    function: "imported.add4",
}

fn main() {}
