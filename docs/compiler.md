# IREE Compiler Setup

`knok-build` compiles traced graphs during the user's `build.rs`. The final
VMFB step is delegated to the `iree-compile` command line tool.

## Lookup Order

`knok-compile` resolves the compiler as follows:

1. `KNOK_IREE_COMPILE`, when set.
2. `iree-compile` on `PATH`.

Examples:

```sh
export KNOK_IREE_COMPILE=/path/to/iree-compile
cargo build
```

```sh
PATH=/path/to/iree-dist/bin:$PATH cargo build
```

`KNOK_CACHE_DIR` can override the VMFB cache directory. By default, artifacts
are cached under `target/knok-cache` for the build-script crate.

## Supported Sources

The Nix development shell is the recommended contributor path. It provisions
the pinned IREE compiler used by CI and local release checks.

Other usable sources are:

- the IREE Python wheel, when it provides `iree-compile`
- a source build of IREE at a compatible release revision
- experimental IREE dist tarballs that include `bin/iree-compile`

Standalone IREE compiler/library artifacts are not yet treated as a stable
upstream distribution contract. Keep this documented as an external requirement
until IREE publishes regular release assets for the compiler tools.

## docs.rs and CI

`docs.rs` builds the target crate documentation. It should not need to execute
user build scripts that compile graphs. Runtime-facing docs are built with the
default `knok` feature set.

CI uses the Nix shell so tests that exercise `knok-build`, runtime E2E fixtures,
and release checks see a pinned compiler.

## Troubleshooting

`failed to run IREE compiler ...`

Set `KNOK_IREE_COMPILE` or put `iree-compile` on `PATH`.

`unsupported IREE backend`

The current compile backends are `llvm-cpu` and `metal-spirv`.

`runtime driver mismatch`

Use an engine whose runtime driver matches the artifact variant. The typed
backend helpers choose the default driver for each supported backend.
