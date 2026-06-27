mod conv;
mod linalg;
mod matmul;
mod permute;
mod rules;

pub(crate) use conv::conv2d_result_type;
pub(crate) use linalg::{
    diagonal_result_type, dot_result_type, inner_result_type, outer_result_type, trace_result_type,
    vecdot_result_type,
};
pub(crate) use matmul::matmul_result_type;
pub(crate) use permute::validate_permute;
pub(crate) use rules::{
    broadcast_shape_slices, expect_numeric_element, infer_call_result, infer_call_results,
};
