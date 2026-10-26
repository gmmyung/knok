# Backends

`knok` separates compile-time IREE target backends from runtime drivers.
Generated artifacts record their backend; compatible drivers follow from it.

## Available Backends

| `knok_build::Backend` | IREE target backend | Default runtime driver | Availability |
| --- | --- | --- | --- |
| `Backend::LlvmCpu` | `llvm-cpu` | `local-task` | Always |
| `Backend::MetalSpirv` | `metal-spirv` | `metal` | macOS |
| `Backend::VulkanSpirv` | `vulkan-spirv` | `vulkan` | `vulkan` feature |
| `Backend::Cuda` | `cuda` | `cuda` | `cuda` feature |

Target-side runtime code uses the matching `knok::Backend` and `knok::Driver`
types. Prefer typed backend and driver values over string names.

## Runtime Selection

Generated artifacts record the backend used by `knok-build`. The default
runtime path lets
the artifact choose the engine driver:

```rust
knok::generated_graphs!(pub mod graphs);

let engine = knok::Engine::for_artifact(graphs::forward::artifact())?;
let model = graphs::forward::load(&engine)?;
let y = model.run(&x)?;
```

The one-shot generated wrapper does the same thing internally:

```rust
let y = graphs::forward::call(&x)?;
```

`Engine::for_backend(Backend::LlvmCpu)` and
`Engine::new(RuntimeConfig::backend(...))` are useful when the caller already
knows which driver a group of graphs should share. If the engine driver does
is incompatible with every embedded backend, loading returns a missing-variant
error. LLVM CPU artifacts accept both `Driver::LocalTask` and `Driver::LocalSync`.

When reusing one engine across multiple graphs, construct it from one artifact
and run graphs compiled for compatible backends. Use
`RuntimeConfig::driver(driver).device_ordinal(n)` or `.device_path(path)` to
select a device, and `Engine::available_devices(driver)` to enumerate choices.

GPU backends are intentionally constrained at compile time. `MetalSpirv` is
only available when the build/target OS is macOS. `VulkanSpirv` and `Cuda` are
feature-gated. Enable the same feature on the build-time crate and, when hosted
execution is needed, the target crate:

```toml
[build-dependencies]
knok-build = { version = "...", features = ["vulkan"] }

[dependencies]
knok = { version = "...", features = ["vulkan"] }
```

Use `cuda` instead of `vulkan` for CUDA artifacts.

## Compiler Requirement

Build-time graph compilation uses Eerie's IREE shared library. Install the
matching wheel in your Python environment:

```sh
python -m pip install 'iree-base-compiler==3.13.0'
```

The Nix development shell installs the pinned compiler and adds it to `PATH`.
For installation options, cache knobs, and troubleshooting, see
[compiler.md](compiler.md).

## Runtime Modes

- Default features enable hosted runtime execution with `std`.
- `default-features = false` builds the target crate as `no_std + alloc` and
  keeps tensor types and generated graph metadata available. Runtime engines
  and generated execution helpers are absent.

## Platform Notes

`LlvmCpu` is the primary portable backend and is the default choice for tests.
`MetalSpirv` targets Apple Metal and is exposed only on macOS. `VulkanSpirv`
expects an IREE compiler with the Vulkan target backend and an executable Vulkan
driver. `Cuda` expects an IREE compiler distribution built with CUDA support and
an executable CUDA driver; this is normally a Linux deployment path. Unsupported
or unavailable runtime drivers surface as runtime errors when constructing or
using an `Engine`.

For the full target-side execution flow, see [runtime.md](runtime.md).
