mod common;
mod conv;
mod linalg;
mod matmul;
mod permute;
mod rules;
mod static_creation;

pub(crate) use common::{broadcast_shape_slices, expect_numeric_element};
pub(crate) use conv::conv2d_result_type;
pub(crate) use linalg::{
    diagonal_result_type, dot_result_type, inner_result_type, outer_result_type, trace_result_type,
    vecdot_result_type,
};
pub(crate) use matmul::matmul_result_type;
pub(crate) use permute::validate_permute;
pub(crate) use rules::{infer_call_result, infer_call_results};
pub use static_creation::{static_arange_literals, static_eye_literals, static_linspace_literals};
pub(crate) use static_creation::{validate_static_creation_call, validate_static_creation_target};
