# Runtime Workflow

`knok-build` chooses a compile backend during `build.rs`, compiles the graph to
a VMFB artifact, and records the matching runtime driver in the generated
metadata. Target code imports that generated module with
`knok::generated_graphs!` and executes it through `knok::Graph<I, O>` and
`knok::Engine`.

Build-time graph functions do not run on the target. They are traced by the
build script, lowered through MLIR, and embedded as generated Rust wrappers plus
bytecode.

## Generated Modules

Each generated module exposes:

- `GRAPH`: a typed `knok::Graph<I, O>` handle.
- `artifact()`: static VMFB and signature metadata.
- `run(&Engine, ...)`: repeated inference with a caller-owned engine.
- `call(...)`: one-shot inference with an engine created from the artifact.

Most applications should use `run` in hot paths and `call` for occasional
invocations.

## One-Shot Execution

```rust
knok::generated_graphs!(pub mod graphs);

let y = graphs::forward::call(x)?;
```

`call` constructs an engine with `Engine::for_artifact(graphs::forward::artifact())`.
That keeps the runtime driver aligned with the backend recorded by `knok-build`.

## Reusable Engine Execution

```rust
let engine = knok::Engine::for_artifact(graphs::forward::artifact())?;

for x in inputs {
    let y = graphs::forward::run(&engine, x)?;
}
```

Use this form for repeated inference and benchmarks. It avoids rebuilding the
runtime engine for every invocation and lets the engine cache loaded VMFB
modules.

## Low-Level Graph Handle

Generated modules expose `GRAPH` for code that needs to pass graph handles
around:

```rust
let graph = graphs::forward::GRAPH;
let artifact = graph.artifact();
let engine = knok::Engine::for_artifact(artifact)?;
let y = graph.run(&engine, x)?;
```

The public `Graph` API is intentionally small: `artifact`, `run`, and
`run_once`. Raw runtime inputs and outputs are crate internals.

## Driver Mismatches

An engine can only run artifacts whose variant list contains the engine's
runtime driver. For example, a `local-task` engine cannot run a Metal artifact.
Mismatch failures are reported as missing artifact variants or runtime driver
mismatch errors.
