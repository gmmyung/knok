#!/usr/bin/env bash
set -euo pipefail

output_dir="${CARGO_LLVM_COV_OUTPUT_DIR:-target/coverage}"
html_dir="$output_dir/html"
badge_dir="$output_dir/badge"
badge_path="$badge_dir/coverage.svg"
min_lines="${KNOK_COVERAGE_MIN_LINES-20}"
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
<svg xmlns="http://www.w3.org/2000/svg" width="104" height="20" role="img" aria-label="coverage: $line_coverage">
  <title>coverage: $line_coverage</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="104" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="61" height="20" fill="#555"/>
    <rect x="61" width="43" height="20" fill="$badge_color"/>
    <rect width="104" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11">
    <text x="31" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="31" y="14">coverage</text>
    <text x="82" y="15" fill="#010101" fill-opacity=".3">$line_coverage</text>
    <text x="82" y="14">$line_coverage</text>
  </g>
</svg>
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
echo "Coverage badge written to $badge_path"
