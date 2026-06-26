mod conv;
mod matmul;
mod permute;
mod rules;

pub(crate) use conv::conv2d_result_type;
pub(crate) use matmul::matmul_result_type;
pub(crate) use permute::validate_permute;
pub(crate) use rules::{
    broadcast_shape_slices, expect_numeric_element, infer_call_result, infer_call_results,
};
