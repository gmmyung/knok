# knok-build

`knok-build` is the build-time graph tracing frontend for knok.

It records graph functions executed from `build.rs`, compiles them through the
knok compiler pipeline, and emits typed Rust wrappers into `OUT_DIR` for import
by the runtime crate.

See the repository README for setup, examples, feature flags, and current
limitations:

https://github.com/gmmyung/knok
