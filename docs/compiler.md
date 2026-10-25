# IREE Compiler Setup

`knok-build` compiles traced graphs during the user's `build.rs`. The final
VMFB step runs in-process through Eerie's shared IREE compiler library. Install
the matching compiler wheel, or use `nix develop` for the complete toolchain:

```sh
python -m pip install 'iree-base-compiler==3.13.0'
```

## Lookup Order

`knok-compile` resolves the compiler as follows:

1. `KNOK_IREE_COMPILER`, when set, selects a shared library file or directory.
2. Otherwise Eerie discovers `iree-base-compiler==3.13.0` using Python on `PATH`.
   `EERIE_PYTHON` can select a specific Python executable.

Examples:

```sh
export KNOK_IREE_COMPILER=/path/to/libIREECompiler.dylib
cargo build
```

```sh
EERIE_PYTHON=python3 cargo build
```

No IREE library paths or rpaths need to be added to Rust flags.

## Supported Sources

The Nix development shell is the recommended contributor path. It provisions
IREE 3.13.0, matching the Eerie 0.7 runtime used by CI and local release checks.

Recompile VMFB artifacts produced by IREE 3.11 or earlier; they use an older HAL
module ABI.

Custom IREE 3.13 libraries can be selected with `KNOK_IREE_COMPILER`.

## Cache

`KNOK_CACHE_DIR` overrides the VMFB cache directory. Build scripts default to
`OUT_DIR/knok-cache`; direct `knok-compile` calls use `target/knok-cache` under
the calling crate. Changes to the source, compiler, options, or target invalidate
the cache. Build scripts track the library file and compiler selection
environment so compiler changes rebuild the generated artifacts.

## macOS Build Hosts

Melior and IREE each carry LLVM/MLIR. Their symbols must stay separate. The Nix
shell selects `scripts/macos-linker.py`, which hides executable exports while
preserving shared-library and proc-macro exports. It also applies to Cargo's
host build scripts when an explicit `--target` is used.

Outside Nix, copy that executable script into your project and configure Cargo
for your **host** triple:

```toml
# .cargo/config.toml; use x86_64-apple-darwin for an Intel build host.
[target.aarch64-apple-darwin]
linker = "scripts/macos-linker.py"
```

The script uses `clang` on `PATH`. Hiding executable exports is unsuitable for
applications exposing a host API to plugins; those need a selective export
policy. Knok reports a setup error if the build host still exports Melior's
MLIR entry points. Emitting linker arguments from a consumer's `build.rs` only
affects its targets, not that build script's own executable.

## Cross-Language LTO

Enable `knok`'s `cross-lang-lto` feature to compile the native runtime with
Clang ThinLTO. Use the matching development shell for the Rust/Clang/LLVM linker
configuration on Linux or macOS:

```sh
nix develop .#cross-lang-lto
cargo test --release -p knok --features cross-lang-lto
```

This feature belongs on the target `knok` dependency; `knok-build` does not need
it. It requires compatible Rust/LLVM tools and a linker configured for joint
Rust/C LTO. The shell handles this, including macOS host export isolation.
Cargo's optimization/debug profile still controls the native build profile.

## docs.rs and CI

`docs.rs` builds the target crate documentation. It should not need to execute
user build scripts that compile graphs. Runtime-facing docs are built with the
default `knok` feature set.

CI uses the Nix shell so tests that exercise `knok-build`, runtime E2E fixtures,
and release checks see a pinned compiler.

## Troubleshooting

`failed to load the IREE compiler ...`

Install the matching wheel in the active Python environment. Use `EERIE_PYTHON`
to select another interpreter or `KNOK_IREE_COMPILER` for a custom library.

`unsupported IREE backend`

`llvm-cpu` is always exposed. `metal-spirv` is exposed on macOS.
`vulkan-spirv` and `cuda` are exposed when the corresponding `vulkan` or `cuda`
crate feature is enabled and the selected compiler library was built with
that target backend.

`runtime driver mismatch`

Use an engine whose runtime driver matches the artifact variant. The typed
backend helpers choose the default driver for each supported backend.
