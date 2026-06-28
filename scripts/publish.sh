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
  knok-macros
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
    "knok-macros": root / "crates/knok-macros/Cargo.toml",
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

if [[ "$mode" == "--dry-run" ]]; then
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
  cargo publish -p "$crate"
  wait_for_crate "$crate" "$version"
done
