use knok_core::{Conv2dOptions, TensorType};

use super::lowerer::{Lowerer, Value};

impl Lowerer<'_, '_> {
    pub(super) fn conv2d(
        &mut self,
        input: Value,
        kernel: Value,
        options: &Conv2dOptions,
    ) -> anyhow::Result<Value> {
        let input = self.pad_conv2d_input(input, options)?;
        if options.groups > 1 {
            return self.grouped_conv2d(input, kernel, options);
        }
        self.conv2d_direct(input, kernel, options)
    }

    fn conv2d_direct(
        &mut self,
        input: Value,
        kernel: Value,
        options: &Conv2dOptions,
    ) -> anyhow::Result<Value> {
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![
                input.ty.shape[0],
                (input.ty.shape[1] - ((kernel.ty.shape[0] - 1) * options.dilation[0] + 1))
                    / options.stride[0]
                    + 1,
                (input.ty.shape[2] - ((kernel.ty.shape[1] - 1) * options.dilation[1] + 1))
                    / options.stride[1]
                    + 1,
                kernel.ty.shape[3],
            ],
        };
        let init = self.zero_initialized_tensor(&ty)?;
        let attrs = conv2d_attrs(options);
        self.append_named_linalg(
            "linalg.conv_2d_nhwc_hwcf",
            &[input, kernel],
            init,
            &ty,
            &attrs,
        )
    }

    fn grouped_conv2d(
        &mut self,
        input: Value,
        kernel: Value,
        options: &Conv2dOptions,
    ) -> anyhow::Result<Value> {
        let groups = options.groups;
        let input_channels_per_group = input.ty.shape[3] / groups;
        let output_channels_per_group = kernel.ty.shape[3] / groups;
        let mut output = None;
        for group in 0..groups {
            let input_slice_ty = TensorType {
                elem: input.ty.elem,
                shape: vec![
                    input.ty.shape[0],
                    input.ty.shape[1],
                    input.ty.shape[2],
                    input_channels_per_group,
                ],
            };
            let kernel_slice_ty = TensorType {
                elem: kernel.ty.elem,
                shape: vec![
                    kernel.ty.shape[0],
                    kernel.ty.shape[1],
                    kernel.ty.shape[2],
                    output_channels_per_group,
                ],
            };
            let input_slice = self.slice(
                input.clone(),
                &input_slice_ty,
                &[0, 0, 0, group * input_channels_per_group],
            )?;
            let kernel_slice = self.slice(
                kernel.clone(),
                &kernel_slice_ty,
                &[0, 0, 0, group * output_channels_per_group],
            )?;
            let group_output = self.conv2d_direct(input_slice, kernel_slice, options)?;
            output = Some(match output {
                Some(previous) => self.concat(previous, group_output, 3)?,
                None => group_output,
            });
        }
        output.ok_or_else(|| anyhow::anyhow!("conv2d groups must be non-zero"))
    }

    fn pad_conv2d_input(&mut self, input: Value, options: &Conv2dOptions) -> anyhow::Result<Value> {
        let padding = options.padding;
        if padding.top == 0 && padding.bottom == 0 && padding.left == 0 && padding.right == 0 {
            return Ok(input);
        }
        let ty = TensorType {
            elem: input.ty.elem,
            shape: vec![
                input.ty.shape[0],
                input.ty.shape[1] + padding.top + padding.bottom,
                input.ty.shape[2] + padding.left + padding.right,
                input.ty.shape[3],
            ],
        };
        let zero = self.constant(ty.elem.zero_literal(), ty.elem)?;
        self.append_tensor_pad(
            input,
            &ty,
            &[0, padding.top, padding.left, 0],
            &[0, padding.bottom, padding.right, 0],
            zero,
        )
    }
}

fn conv2d_attrs(options: &Conv2dOptions) -> Vec<(String, String)> {
    if options.dilation == [1, 1] && options.stride == [1, 1] {
        Vec::new()
    } else {
        vec![
            (
                "dilations".to_string(),
                format!(
                    "dense<[{}, {}]> : vector<2xi64>",
                    options.dilation[0], options.dilation[1]
                ),
            ),
            (
                "strides".to_string(),
                format!(
                    "dense<[{}, {}]> : vector<2xi64>",
                    options.stride[0], options.stride[1]
                ),
            ),
        ]
    }
}
