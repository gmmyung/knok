use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn add4(x: T1<f32, 4>, y: T1<f32, 4>) -> T1<f32, 4> {
    x + y
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn add_sub4(x: T1<f32, 4>, y: T1<f32, 4>) -> (T1<f32, 4>, T1<f32, 4>) {
    (x.clone() + y.clone(), x - y)
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check().output_file("knok_no_std_graphs.rs");
        add4,
        add_sub4
    );
}
