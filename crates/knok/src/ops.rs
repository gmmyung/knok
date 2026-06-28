//! Operations accepted inside `#[knok::graph]` bodies.
//!
//! These names are graph syntax, not host-callable Rust functions. The proc
//! macro parses them inside decorated functions and lowers them to MLIR before
//! Rust type checking needs ordinary function definitions.
//!
//! Graph bodies support `let` bindings, tensor arithmetic, calls to earlier
//! graph functions, and one final expression. They do not support arbitrary
//! Rust control flow.
//!
//! # Elementwise arithmetic
//!
//! `+`, `-`, `*`, `/`, unary `-`, `abs`, `minimum`, `maximum`, `clip`, `pow`,
//! `square`, `reciprocal`, and `relu`.
//!
//! Numeric operands must have matching element types. Trailing broadcasting is
//! supported where the operation defines elementwise behavior.
//!
//! # Comparisons and predicates
//!
//! `greater`, `greater_equal`, `less`, `less_equal`, `equal`, `not_equal`,
//! `logical_and`, `logical_or`, `logical_not`, `logical_xor`, `all`, `any`,
//! and `isnan`.
//!
//! Predicate tensors use `TensorN<bool, ...>` and lower to MLIR `i1`. The
//! selection op is `where(condition, x, y)`, spelled `r#where(...)` in Rust
//! source because `where` is a keyword.
//!
//! # Shape and indexing
//!
//! `reshape::<Target>(x)`, `broadcast::<Target>(x)`, `squeeze::<Target>(x)`,
//! `unsqueeze::<Target>(x)`, `slice::<Target, START...>(x)`,
//! `take::<AXIS, INDEX>(x)`, `gather::<Target, AXIS>(x, indices)`,
//! `take_along_axis::<AXIS>(x, indices)`, `split::<AXIS, SECTION...>(x)`,
//! `concat::<AXIS>(a, b)`, `stack::<AXIS>(a, b)`, `tile::<MULTIPLE...>(x)`,
//! `repeat::<AXIS, COUNT>(x)`, `pad::<Target, LOW...>(x)`, `flip::<AXES...>(x)`,
//! and `roll::<AXIS, SHIFT>(x)`.
//!
//! Shape-changing operations are static and type-directed. Axis parameters are
//! const generics. Runtime index tensors for `gather` and `take_along_axis`
//! must be `i32` or `i64`; negative runtime indices wrap from the end of the
//! selected axis and out-of-bounds indices fail invocation.
//!
//! # Axis movement
//!
//! `transpose::<AXES...>(x)`, `permute::<Target, AXES...>(x)`,
//! `permute_dims::<AXES...>(x)`, `swapaxes::<A, B>(x)`, and
//! `moveaxis::<SRC, DST>(x)`.
//!
//! `transpose(x)` without axes reverses dimensions. Explicit axis forms must
//! name each axis exactly once.
//!
//! # Creation
//!
//! `zeros_like(x)`, `ones_like(x)`, `full_like(x, value)`,
//! `arange::<Target>(...)`, `linspace::<Target>(start, stop)`,
//! `eye::<Target>()`, and `identity::<Target>()`.
//!
//! Creation ops are graph operations, not dynamic host tensor constructors.
//! Their target shape and dtype are known at macro expansion time.
//!
//! # Reductions and statistics
//!
//! Full-tensor and axis-aware forms are supported for `sum`, `prod`, `mean`,
//! `max` / `amax`, `min` / `amin`, `argmax`, `argmin`, `var`, `std`, `ptp`,
//! and `softmax`.
//!
//! Axis-aware reductions use const generic syntax, for example `sum::<1>(x)`.
//! `argmax` and `argmin` return `i64` indices. Empty `sum`, `prod`, `all`, and
//! `any` reductions use identity values; empty value-selecting or denominator
//! reductions are rejected.
//!
//! # Linalg
//!
//! `matmul`, `dot`, `vecdot`, `inner`, `outer`, `trace`, and `diagonal`.
//!
//! `matmul` follows NumPy-style rank-1 through rank-6 behavior. `dot` accepts
//! rank-1 vectors. `vecdot` contracts the last axis by default or a const axis
//! when provided. `trace` and `diagonal` use the last two axes by default and
//! accept explicit axis forms.
//!
//! # Convolution
//!
//! `conv2d(x, kernel)` lowers NHWC input and HWCF kernel tensors. Options use
//! type-style generic markers: `Pad<TOP, BOTTOM, LEFT, RIGHT>`,
//! `Stride<H, W>`, `Dilation<H, W>`, and `Groups<N>`.
//!
//! # Floating-point math
//!
//! `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `floor`,
//! `ceil`, `round`, `rint`, `sin`, `cos`, `tan`, `tanh`, and `sigmoid`.
//!
//! These require floating-point tensors. Backend support for `f16` and `bf16`
//! math can vary.
