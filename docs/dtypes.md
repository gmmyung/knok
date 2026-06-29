# DType Policy

`knok` keeps dtype behavior explicit. Graph operations do not perform implicit
promotion between tensor element types. Inputs to an operation must already have
compatible dtypes, and casts should be added as explicit graph operations when
the API grows them.

## Supported Element Types

| DType | Tensor storage | Trace/typecheck | MLIR lowering | Runtime I/O |
| --- | --- | --- | --- | --- |
| `bool` | yes | yes | yes | yes |
| `f32` | yes | yes | yes | yes |
| `f64` | yes | yes | yes | yes |
| `i32` | yes | yes | yes | yes |
| `i64` | yes | yes | yes | yes |
| `f16` | `half` feature | `half` feature | `half` feature | `half` feature |
| `bf16` | `half` feature | `half` feature | `half` feature | `half` feature |

Quantized and smaller integer tensor types are tracked separately in #20.

## Operation Categories

| Category | DType policy |
| --- | --- |
| Tensor shape/layout ops | preserve input dtype |
| Elementwise `+`, `-`, `*`, `/`, `abs`, `square`, `reciprocal` | numeric dtypes |
| Floating math such as `sin`, `exp`, `log`, `sqrt`, `round` | floating dtypes |
| `minimum`, `maximum`, extrema reductions, `argmin`, `argmax` | ordered dtypes |
| Equality predicates | matching dtypes, returns `bool` |
| Ordered comparisons | ordered matching dtypes, returns `bool` |
| Logical ops and bool reductions | `bool` only |
| `where` | `bool` condition and matching value dtypes |
| Index tensors for `gather` and `take_along_axis` | `i32` or `i64` |
| `arange`, `linspace`, `eye`, `identity` | target dtype validates the literal family |
| Linalg/convolution ops | numeric dtypes where the lowering supports the op |

The type checker should reject unsupported dtype/op combinations at build time
with an actionable diagnostic.

## Casting Direction

Implicit promotion is intentionally absent. Future cast helpers should:

- make the destination dtype explicit in the Rust type
- preserve static shape
- lower to a dedicated MLIR cast operation
- fail during build-time tracing/typechecking for unsupported conversions

This keeps generated graph signatures stable and avoids hidden dtype changes in
runtime artifacts.
