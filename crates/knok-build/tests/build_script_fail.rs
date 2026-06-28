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
