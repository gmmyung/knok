#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo test -p knok-core -p knok-compile -p knok-build -p knok-build-macros -p knok-macros
cargo check -p knok --no-default-features
cargo check -p knok-build-tracing-smoke
cargo check -p knok-no-std-smoke
cargo doc -p knok --no-default-features --features std --no-deps
cargo test -p knok
cargo test -p knok-build-tracing-runtime
