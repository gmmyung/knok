use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use tempfile::TempDir;

struct Fixture {
    name: &'static str,
    build_rs: &'static str,
    expected: &'static [&'static str],
}

#[test]
fn build_script_tracing_failures_are_reported() {
    let fixtures = [
        Fixture {
            name: "shape_mismatch",
            build_rs: r#"
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: Tensor2<f32, 2, 3>, y: Tensor2<f32, 4, 3>) -> Tensor2<f32, 2, 3> {
    x + y
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
"#,
            expected: &[
                "knok build failed",
                "elementwise operands are not broadcast-compatible",
            ],
        },
        Fixture {
            name: "return_type_mismatch",
            build_rs: r#"
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 3> {
    sum_axis(x, 1)
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
"#,
            expected: &["knok build failed", "return type mismatch"],
        },
        Fixture {
            name: "invalid_axis",
            build_rs: r#"
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
    sum_axis(x, 3)
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
"#,
            expected: &["knok build failed", "axis 3 is out of bounds"],
        },
        Fixture {
            name: "avg_pool_integer",
            build_rs: r#"
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: Tensor4<i32, 1, 4, 4, 1>) -> Tensor4<i32, 1, 2, 2, 1> {
    avg_pool2d(x)
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
"#,
            expected: &[
                "knok build failed",
                "avg_pool2d expects a floating-point element type",
            ],
        },
        Fixture {
            name: "pool_zero_kernel",
            build_rs: r#"
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: Tensor4<f32, 1, 4, 4, 1>) -> Tensor4<f32, 1, 2, 2, 1> {
    max_pool2d_options(x, Pool2dOptions::new(0, 2))
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
"#,
            expected: &["knok build failed", "kernel dimensions must be non-zero"],
        },
        Fixture {
            name: "grad_non_scalar_loss",
            build_rs: r#"
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: Tensor1<f32, 3>) -> Tensor1<f32, 3> {
    square(x)
}

fn main() {
    knok_build::grad_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
"#,
            expected: &[
                "knok build failed",
                "autograd MVP expects the graph output to be a rank-0 floating-point loss",
            ],
        },
        Fixture {
            name: "grad_repeat_rank_limit",
            build_rs: r#"
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(x: Tensor6<f32, 1, 1, 1, 1, 1, 2>) -> Tensor0<f32> {
    let repeated: Tensor6<f32, 1, 1, 1, 1, 1, 4> = repeat(x, 5, 2);
    sum(repeated)
}

fn main() {
    knok_build::grad_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
"#,
            expected: &[
                "knok build failed",
                "autograd MVP does not support repeat gradients that require rank above 6",
            ],
        },
        Fixture {
            name: "grad_too_many_inputs",
            build_rs: r#"
use knok_build::prelude::*;

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn forward(
    x0: Tensor0<f32>,
    x1: Tensor0<f32>,
    x2: Tensor0<f32>,
    x3: Tensor0<f32>,
    x4: Tensor0<f32>,
    x5: Tensor0<f32>,
    x6: Tensor0<f32>,
    x7: Tensor0<f32>,
    x8: Tensor0<f32>,
    x9: Tensor0<f32>,
    x10: Tensor0<f32>,
    x11: Tensor0<f32>,
) -> Tensor0<f32> {
    x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11
}

fn main() {
    knok_build::grad_graphs_with_options!(
        BuildOptions::stub_artifacts_for_check();
        forward
    );
}
"#,
            expected: &[
                "knok build failed",
                "autograd MVP supports at most 11 floating-point inputs",
            ],
        },
    ];

    let temp = TempDir::new().expect("create temp fixture root");
    let target_dir = temp.path().join("target");
    let knok_root = workspace_root();
    for fixture in fixtures {
        let project = temp.path().join(fixture.name);
        write_fixture(&project, fixture.name, &knok_root, fixture.build_rs);

        let output = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
            .arg("check")
            .arg("--offline")
            .arg("--manifest-path")
            .arg(project.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", &target_dir)
            .output()
            .expect("run cargo check fixture");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let combined = format!("{stdout}\n{stderr}");

        assert!(
            !output.status.success(),
            "fixture `{}` unexpectedly passed\n{}",
            fixture.name,
            combined
        );
        for expected in fixture.expected {
            assert!(
                combined.contains(expected),
                "fixture `{}` output did not contain `{}`\n{}",
                fixture.name,
                expected,
                combined
            );
        }
    }
}

fn write_fixture(project: &Path, name: &str, knok_root: &Path, build_rs: &str) {
    fs::create_dir_all(project.join("src")).expect("create fixture src");
    fs::write(
        project.join("Cargo.toml"),
        format!(
            r#"[package]
name = "knok-build-negative-{name}"
version = "0.0.0"
edition = "2021"
publish = false

[build-dependencies]
knok-build = {{ path = "{}" }}
"#,
            toml_path(&knok_root.join("crates/knok-build"))
        ),
    )
    .expect("write fixture manifest");
    fs::write(project.join("build.rs"), build_rs).expect("write fixture build script");
    fs::write(project.join("src/lib.rs"), "").expect("write fixture lib");
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("knok-build crate lives two levels below the workspace root")
        .to_path_buf()
}

fn toml_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}
