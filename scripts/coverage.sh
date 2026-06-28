#!/usr/bin/env bash
set -euo pipefail

output_dir="${CARGO_LLVM_COV_OUTPUT_DIR:-target/coverage}"
mkdir -p "$output_dir"

cargo llvm-cov clean --workspace

cargo build -p knok-compile --features compiler-helper --bin knok-iree-compile-helper
export KNOK_IREE_COMPILE_HELPER="$PWD/target/debug/knok-iree-compile-helper"

cargo llvm-cov \
  -p knok-core \
  -p knok-compile \
  -p knok-build \
  -p knok-build-macros \
  -p knok \
  -p knok-build-tracing-runtime \
  --ignore-filename-regex '(/tests/|/target/)' \
  --lcov \
  --output-path "$output_dir/lcov.info"

cargo llvm-cov report --summary-only >"$output_dir/summary.txt"
cat "$output_dir/summary.txt"

echo "LCOV report written to $output_dir/lcov.info"
