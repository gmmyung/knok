use knok_core::{CallOp, Padding2d, Pool2dOptions as CorePool2dOptions};

use super::{expr::call_output, tensor::TraceTensor};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Pool2dOptions {
    kernel: [usize; 2],
    padding: [usize; 4],
    stride: [usize; 2],
    dilation: [usize; 2],
}

impl Pool2dOptions {
    pub const fn new(kernel_height: usize, kernel_width: usize) -> Self {
        Self {
            kernel: [kernel_height, kernel_width],
            padding: [0, 0, 0, 0],
            stride: [kernel_height, kernel_width],
            dilation: [1, 1],
        }
    }

    pub const fn padding(mut self, top: usize, bottom: usize, left: usize, right: usize) -> Self {
        self.padding = [top, bottom, left, right];
        self
    }

    pub const fn stride(mut self, height: usize, width: usize) -> Self {
        self.stride = [height, width];
        self
    }

    pub const fn dilation(mut self, height: usize, width: usize) -> Self {
        self.dilation = [height, width];
        self
    }

    fn into_core(self) -> CorePool2dOptions {
        CorePool2dOptions {
            kernel: self.kernel,
            padding: Padding2d {
                top: self.padding[0],
                bottom: self.padding[1],
                left: self.padding[2],
                right: self.padding[3],
            },
            stride: self.stride,
            dilation: self.dilation,
        }
    }
}

impl Default for Pool2dOptions {
    fn default() -> Self {
        Self::new(2, 2)
    }
}

pub fn max_pool2d<Output: TraceTensor>(input: impl TraceTensor) -> Output {
    max_pool2d_options(input, Pool2dOptions::default())
}

pub fn max_pool2d_options<Output: TraceTensor>(
    input: impl TraceTensor,
    options: Pool2dOptions,
) -> Output {
    call_output(
        CallOp::MaxPool2d(options.into_core()),
        vec![input.into_expr()],
    )
}

pub fn avg_pool2d<Output: TraceTensor>(input: impl TraceTensor) -> Output {
    avg_pool2d_options(input, Pool2dOptions::default())
}

pub fn avg_pool2d_options<Output: TraceTensor>(
    input: impl TraceTensor,
    options: Pool2dOptions,
) -> Output {
    call_output(
        CallOp::AvgPool2d(options.into_core()),
        vec![input.into_expr()],
    )
}
