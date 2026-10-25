# Migrating to the Next Breaking Release

## Borrowed Inputs

Pass references to generated `call`, `run`, and `run_into` helpers:

```rust
let y = graphs::forward::run(&engine, &x)?;
```

For a `Graph<I, O>` or prepared model with several inputs, pass a tuple of
references: `model.run((&x, &weights))`. Pass `()` for no inputs. Remove host
tensor clones previously needed to preserve inputs across calls.

## Prepared Models

Resolve the function once before an inference loop:

```rust
let model = graphs::forward::load(&engine)?;
model.run_into(&x, &mut output)?;
```

The model borrows its engine. Engines and models stay on their creating thread.

## Artifact Contracts

`Graph::new(artifact)` now returns `Result<Graph<_, _>, GraphSignatureError>`.
It rejects mismatched input/output counts, dtypes, or shapes. Generated `GRAPH`
values check this at compile time.

Construct metadata through `GraphArtifact::new` and
`GraphArtifactVariant::new`; inspect it through methods such as `variants()`,
`input_descs()`, and `backend()`. Variants carry a typed backend instead of a
separate driver string. Empty signatures describe zero inputs/outputs and
cannot disable validation. Imported MLIR output shape/dtype/count mismatches
now fail before writing host outputs.

## Compiler and Device Selection

Use `BuildOptions::compiler_options(CompileOptions::default()...)` to set CPU
tuning and IREE flags. Options apply to traced graphs, gradients, and imported
models; see [compiler.md](compiler.md#compiler-options).

Use `RuntimeConfig::driver(...).device_ordinal(n)` or `.device_path(path)` for
device selection. Enumerate devices with `Engine::available_devices(driver)`.
LLVM CPU artifacts can run on `LocalTask` or `LocalSync`.

## Runtime Features

Remove `embedded-runtime` from Cargo feature lists. It did not expose a working
execution path. Enable `runtime` for hosted execution, or disable default
features for tensors and generated graph metadata on `no_std + alloc` targets.
Without `runtime`, engines and execution helpers are absent rather than
returning `HostedRuntimeDisabled`.

The public error enum is non-exhaustive. Obsolete placeholder, cache-lock,
driver-mismatch, and raw input/output-index errors have been removed.
