#!/usr/bin/env bash
set -euo pipefail

output_dir="${CARGO_LLVM_COV_OUTPUT_DIR:-target/coverage}"
html_dir="$output_dir/html"
badge_dir="$output_dir/badge"
badge_path="$badge_dir/coverage.json"
min_lines="${KNOK_COVERAGE_MIN_LINES-70}"
mkdir -p "$output_dir"
rm -rf "$html_dir"
rm -rf "$badge_dir"

cargo llvm-cov clean --workspace

cargo llvm-cov \
  -p knok-core \
  -p knok-compile \
  -p knok-build \
  -p knok-build-macros \
  -p knok \
  -p knok-build-tracing-smoke \
  -p knok-build-tracing-runtime \
  -p knok-runtime-e2e \
  --ignore-filename-regex '(/tests/|/target/)' \
  --lcov \
  --output-path "$output_dir/lcov.info"

cargo llvm-cov report \
  --ignore-filename-regex '(/tests/|/target/)' \
  --summary-only >"$output_dir/summary.txt"
cat "$output_dir/summary.txt"
line_coverage="$(awk '$1 == "TOTAL" { print $10 }' "$output_dir/summary.txt")"
line_coverage_value="${line_coverage%%%}"

if [[ -z "$line_coverage" ]]; then
  echo "failed to parse total line coverage from $output_dir/summary.txt" >&2
  exit 1
fi

badge_color="$(
  awk -v coverage="$line_coverage_value" 'BEGIN {
    if (coverage >= 80) print "brightgreen";
    else if (coverage >= 60) print "yellowgreen";
    else if (coverage >= 40) print "yellow";
    else if (coverage >= 20) print "orange";
    else print "red";
  }'
)"
mkdir -p "$badge_dir"
cat >"$badge_path" <<EOF
{
  "schemaVersion": 1,
  "label": "coverage",
  "message": "$line_coverage",
  "color": "$badge_color"
}
EOF

cargo llvm-cov report \
  --ignore-filename-regex '(/tests/|/target/)' \
  --html \
  --output-dir "$output_dir" >/dev/null

if [[ -n "$min_lines" ]]; then
  cargo llvm-cov report \
    --ignore-filename-regex '(/tests/|/target/)' \
    --fail-under-lines "$min_lines" >/dev/null
  echo "Coverage line threshold: ${min_lines}%"
fi

echo "Line coverage: $line_coverage"
echo "LCOV report written to $output_dir/lcov.info"
echo "HTML report written to $html_dir/index.html"
echo "Coverage badge endpoint written to $badge_path"
