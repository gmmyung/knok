#!/usr/bin/env bash
set -euo pipefail

tag="${1:-${GITHUB_REF_NAME:-}}"

if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "expected release tag in vMAJOR.MINOR.PATCH form, got '${tag}'" >&2
  exit 2
fi

version="${tag#v}"

python3 - "$version" <<'PY'
import pathlib
import re
import sys
import tomllib

version = sys.argv[1]
root = pathlib.Path.cwd()
errors: list[str] = []

crate_paths = {
    "knok-core": root / "crates/knok-core/Cargo.toml",
    "knok-compile": root / "crates/knok-compile/Cargo.toml",
    "knok-build-macros": root / "crates/knok-build-macros/Cargo.toml",
    "knok-build": root / "crates/knok-build/Cargo.toml",
    "knok": root / "crates/knok/Cargo.toml",
}

for crate, path in crate_paths.items():
    data = tomllib.loads(path.read_text())
    actual = data["package"]["version"]
    if actual != version:
        errors.append(f"{path.relative_to(root)} package.version is {actual}, expected {version}")

workspace = tomllib.loads((root / "Cargo.toml").read_text())
deps = workspace["workspace"]["dependencies"]
for crate in ["knok-core", "knok-build", "knok-build-macros", "knok-compile"]:
    dep = deps.get(crate)
    actual = dep.get("version") if isinstance(dep, dict) else None
    if actual != version:
        errors.append(f"workspace dependency {crate} version is {actual!r}, expected {version!r}")

changelog = (root / "CHANGELOG.md").read_text()
heading = re.compile(
    rf"^##\s+(?:\[{re.escape(version)}\]|{re.escape(version)})\s+-\s+\d{{4}}-\d{{2}}-\d{{2}}\s*$",
    re.MULTILINE,
)
if not heading.search(changelog):
    errors.append(f"CHANGELOG.md is missing a '## {version} - YYYY-MM-DD' release section")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    sys.exit(1)

print(f"release metadata matches v{version}")
PY
