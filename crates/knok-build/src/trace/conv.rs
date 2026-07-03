use knok_core::{CallOp, Conv2dOptions as CoreConv2dOptions, Padding2d};

use super::{expr::call_output, tensor::TraceTensor};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Static options for traced 2D convolution.
pub struct Conv2dOptions {
    padding: [usize; 4],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl Conv2dOptions {
    /// Creates options with no padding, unit stride, unit dilation, and one group.
    pub const fn new() -> Self {
        Self {
            padding: [0, 0, 0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }

    /// Sets explicit top, bottom, left, and right padding.
    pub const fn padding(mut self, top: usize, bottom: usize, left: usize, right: usize) -> Self {
        self.padding = [top, bottom, left, right];
        self
    }

    /// Sets height and width stride.
    pub const fn stride(mut self, height: usize, width: usize) -> Self {
        self.stride = [height, width];
        self
    }

    /// Sets height and width dilation.
    pub const fn dilation(mut self, height: usize, width: usize) -> Self {
        self.dilation = [height, width];
        self
    }

    /// Sets the number of convolution groups.
    pub const fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    fn into_core(self) -> CoreConv2dOptions {
        CoreConv2dOptions {
            padding: Padding2d {
                top: self.padding[0],
                bottom: self.padding[1],
                left: self.padding[2],
                right: self.padding[3],
            },
            stride: self.stride,
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Records a 2D convolution with default options.
pub fn conv2d<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    trace_conv2d(lhs, rhs)
}

fn trace_conv2d<Output: TraceTensor>(lhs: impl TraceTensor, rhs: impl TraceTensor) -> Output {
    conv2d_options(lhs, rhs, Conv2dOptions::default())
}

/// Records a 2D convolution with explicit static options.
pub fn conv2d_options<Output: TraceTensor>(
    lhs: impl TraceTensor,
    rhs: impl TraceTensor,
    options: Conv2dOptions,
) -> Output {
    call_output(
        CallOp::Conv2d(options.into_core()),
        vec![lhs.into_expr(), rhs.into_expr()],
    )
}
