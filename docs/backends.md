# Backends

`knok` separates compile-time IREE target backends from runtime drivers.
Generated artifacts record both pieces of metadata.

## Available Backends

| `knok_build::Backend` | IREE target backend | Default runtime driver |
| --- | --- | --- |
| `Backend::LlvmCpu` | `llvm-cpu` | `local-task` |
| `Backend::MetalSpirv` | `metal-spirv` | `metal` |

Target-side runtime code uses the matching `knok::Backend` and `knok::Driver`
types. Prefer typed backend and driver values over string names.

## Compiler Requirement

Build-time graph compilation invokes `iree-compile`. Put the binary on `PATH`,
or set:

```sh
export KNOK_IREE_COMPILE=/path/to/iree-compile
```

The Nix development shell installs the pinned compiler and adds it to `PATH`.
For installation options, cache knobs, and troubleshooting, see
[compiler.md](compiler.md).

## Runtime Modes

- Default features enable hosted runtime execution with `std`.
- `default-features = false` builds the target crate as `no_std + alloc` and
  keeps generated wrapper types available, but hosted execution returns
  `HostedRuntimeDisabled`.
- `embedded-runtime` enables the IREE runtime dependency without enabling `std`.

## Platform Notes

`LlvmCpu` is the primary portable backend and is the default choice for tests.
`MetalSpirv` targets Apple Metal and expects a Metal runtime driver to be
available on the executing system. Unsupported or unavailable runtime drivers
surface as runtime errors when constructing or using an `Engine`.
