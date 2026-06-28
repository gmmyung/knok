#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo test -p knok-core -p knok-compile -p knok-macros
cargo check -p knok --no-default-features
cargo check -p knok-no-std-smoke
RUSTDOCFLAGS="-D missing_docs" cargo doc -p knok-core -p knok-compile -p knok-macros -p knok --no-deps
RUSTDOCFLAGS="-D missing_docs" cargo doc -p knok --no-default-features --features std --no-deps
cargo test -p knok
