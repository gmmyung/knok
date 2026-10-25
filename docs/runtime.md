# Runtime Workflow

`knok-build` traces graphs on the build host and embeds VMFB bytecode plus
static signatures. Target code imports the generated modules with
`knok::generated_graphs!`. Execution requires the `runtime` feature; tensor
types, `GRAPH`, and `artifact()` remain available in `no_std + alloc` builds.

## Repeated Inference

Load a model once to reuse the engine, loaded module, and resolved function:

```rust
knok::generated_graphs!(pub mod graphs);

let engine = knok::Engine::for_artifact(graphs::forward::artifact())?;
let model = graphs::forward::load(&engine)?;
for x in inputs {
    let y = model.run(&x)?;
}
```

Inputs are borrowed. A model with multiple inputs takes a tuple of references,
such as `model.run((&x, &weights))`; a model with no inputs takes `()`.
Models borrow their engine and stay on its thread.

To reuse host output storage, allocate it once:

```rust
let mut output = knok::tensor::Tensor2::<f32, 2, 2>::zeros();
model.run_into(&x, &mut output)?;
```

Multiple outputs use a mutable tuple of tensors. All output counts, dtypes,
and full shapes are checked before copying any host output. Transfer errors
can still occur during a copy. Device buffers and invocation metadata may
allocate, and borrowed host inputs may still be copied to the device.

## Convenience Calls

Generated modules also provide wrappers that accept each input separately:

```rust
let y = graphs::forward::call(&x)?;
let y = graphs::forward::run(&engine, &x)?;
graphs::forward::run_into(&engine, &x, &mut output)?;
```

`call` creates an engine for one invocation. `run` and `run_into` reuse the
engine's module cache but resolve the function per call. Use `load` for loops.

## Devices

Select a driver and device explicitly when needed:

```rust
use knok::{Driver, Engine, RuntimeConfig};

let devices = Engine::available_devices(Driver::LocalSync)?;
let engine = Engine::new(RuntimeConfig::driver(Driver::LocalSync).device_ordinal(0))?;
```

GPU drivers support the same ordinal selection. `device_path(devices[0].path())`
selects a driver-specific path instead. The last ordinal/path selection wins.
LLVM CPU artifacts support both `LocalTask` and `LocalSync`; GPU artifacts
require a compatible driver. An unavailable device or driver returns an error.

## Typed Artifacts

`Graph<I, O>` checks the complete static input/output signature at construction:

```rust
let graph = knok::Graph::<Input, Output>::new(artifact)?;
let model = graph.load(&engine)?;
```

Generated `GRAPH` values perform this check at compile time. Use
`GraphArtifact::new` and `GraphArtifactVariant::new` for custom artifacts, and
their accessors to inspect metadata. Empty signature slices mean zero inputs
or outputs. The runtime checks actual outputs against the declared shapes
and dtypes, including external MLIR imports.

See [migration.md](migration.md) for changes from the previous API.
