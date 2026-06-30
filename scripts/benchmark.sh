#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../benchmarks/runtime"
cargo run --release
