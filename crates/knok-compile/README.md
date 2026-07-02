# knok-compile

`knok-compile` lowers checked knok graphs to MLIR and invokes the IREE compiler
to produce runtime artifacts.

This crate is primarily used by `knok-build`. End users normally interact with
generated graph wrappers through `knok` and `knok-build`.

Repository and compiler setup documentation:

https://github.com/gmmyung/knok
