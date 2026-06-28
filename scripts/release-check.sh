#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo build -p knok-compile --features compiler-helper --bin knok-iree-compile-helper
export KNOK_IREE_COMPILE_HELPER="$PWD/target/debug/knok-iree-compile-helper"
cargo test -p knok-core -p knok-compile -p knok-build -p knok-build-macros
cargo check -p knok --no-default-features
cargo check -p knok-build-tracing-smoke
cargo check -p knok-no-std-smoke
cargo doc -p knok --no-default-features --features std --no-deps
cargo test -p knok
cargo test -p knok-build-tracing-runtime
