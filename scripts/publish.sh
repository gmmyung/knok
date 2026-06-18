#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--execute" ]]; then
  publish_args=()
elif [[ "${1:-}" == "--dry-run" || "${1:-}" == "" ]]; then
  publish_args=(--dry-run --allow-dirty)
else
  echo "usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

crates=(
  knok-core
  knok-compile
  knok-macros
  knok
)

for crate in "${crates[@]}"; do
  cargo publish -p "$crate" "${publish_args[@]}"
done
