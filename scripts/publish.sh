#!/usr/bin/env bash
set -euo pipefail

mode="${1:---dry-run}"
case "$mode" in
  --dry-run | --execute) ;;
  *)
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
    ;;
esac

crates=(
  knok-core
  knok-compile
  knok-build-macros
  knok-build
  knok
)

crate_version() {
  local crate="$1"
  python3 - "$crate" <<'PY'
import pathlib
import sys
import tomllib

crate = sys.argv[1]
root = pathlib.Path.cwd()
paths = {
    "knok-core": root / "crates/knok-core/Cargo.toml",
    "knok-compile": root / "crates/knok-compile/Cargo.toml",
    "knok-build-macros": root / "crates/knok-build-macros/Cargo.toml",
    "knok-build": root / "crates/knok-build/Cargo.toml",
    "knok": root / "crates/knok/Cargo.toml",
}
print(tomllib.loads(paths[crate].read_text())["package"]["version"])
PY
}

wait_for_crate() {
  local crate="$1"
  local version="$2"
  python3 - "$crate" "$version" <<'PY'
import sys
import time
import urllib.error
import urllib.request

crate = sys.argv[1]
version = sys.argv[2]
url = f"https://crates.io/api/v1/crates/{crate}/{version}"

for _ in range(60):
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "knok-release-script"})
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                print(f"{crate} {version} is available on crates.io")
                sys.exit(0)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            print(f"waiting for {crate} {version}: HTTP {error.code}", file=sys.stderr)
    except Exception as error:
        print(f"waiting for {crate} {version}: {error}", file=sys.stderr)

    time.sleep(10)

print(f"timed out waiting for {crate} {version} on crates.io", file=sys.stderr)
sys.exit(1)
PY
}

crate_exists() {
  local crate="$1"
  local version="$2"
  python3 - "$crate" "$version" <<'PY'
import sys
import urllib.error
import urllib.request

crate = sys.argv[1]
version = sys.argv[2]
url = f"https://crates.io/api/v1/crates/{crate}/{version}"

try:
    request = urllib.request.Request(url, headers={"User-Agent": "knok-release-script"})
    with urllib.request.urlopen(request, timeout=10) as response:
        sys.exit(0 if response.status == 200 else 1)
except urllib.error.HTTPError as error:
    sys.exit(1 if error.code == 404 else 2)
except Exception:
    sys.exit(2)
PY
}

check_publish_metadata() {
  python3 <<'PY'
import pathlib
import sys
import tomllib

root = pathlib.Path.cwd()
workspace = tomllib.loads((root / "Cargo.toml").read_text())
workspace_package = workspace["workspace"]["package"]
paths = {
    "knok-core": root / "crates/knok-core/Cargo.toml",
    "knok-compile": root / "crates/knok-compile/Cargo.toml",
    "knok-build-macros": root / "crates/knok-build-macros/Cargo.toml",
    "knok-build": root / "crates/knok-build/Cargo.toml",
    "knok": root / "crates/knok/Cargo.toml",
}
errors = []

def package_field(package: dict, field: str):
    value = package.get(field)
    if isinstance(value, dict) and value.get("workspace") is True:
        return workspace_package.get(field)
    return value

for crate, path in paths.items():
    package = tomllib.loads(path.read_text())["package"]
    for field in ["description", "license", "repository"]:
        value = package_field(package, field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{path.relative_to(root)} package.{field} is missing or empty")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    sys.exit(1)

print("publish metadata is present for all crates")
PY
}

if [[ "$mode" == "--dry-run" ]]; then
  check_publish_metadata
  cargo publish -p knok-core --dry-run --allow-dirty
  cat <<'EOF'
Skipping dependent crate publish dry-runs: their new in-workspace dependency
versions are not visible in the crates.io index until the preceding crates are
actually published. scripts/release-check.sh validates the full workspace.
EOF
  exit 0
fi

for crate in "${crates[@]}"; do
  version="$(crate_version "$crate")"
  if crate_exists "$crate" "$version"; then
    echo "$crate $version is already available on crates.io; skipping"
    continue
  fi
  cargo publish -p "$crate"
  wait_for_crate "$crate" "$version"
done
