#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo test -p knok-core -p knok-compile -p knok-build -p knok-build-macros
cargo check -p knok --no-default-features
cargo check -p knok-build-tracing-smoke
cargo check -p knok-no-std-smoke
RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc \
    -p knok \
    -p knok-build \
    -p knok-build-macros \
    -p knok-core \
    -p knok-compile \
    --no-deps
cargo test -p knok
cargo test -p knok-build-tracing-runtime
cargo test -p knok-runtime-e2e
