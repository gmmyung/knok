mod conv;
mod matmul;
mod permute;

pub(crate) use conv::conv2d_result_type;
pub(crate) use matmul::matmul_result_type;
pub(crate) use permute::validate_permute;
