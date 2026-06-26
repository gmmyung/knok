use knok_core::{Conv2dOptions, TensorType};

use super::lowerer::{Lowerer, Value};

impl Lowerer<'_> {
    pub(super) fn conv2d(
        &mut self,
        input: Value,
        kernel: Value,
        options: &Conv2dOptions,
    ) -> anyhow::Result<Value> {
        let input = self.pad_conv2d_input(input, options)?;
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
        let name = self.fresh();
        let attrs = if *options == Conv2dOptions::default() {
            String::new()
        } else {
            format!(
                " {{dilations = dense<[{}, {}]> : vector<2xi64>, strides = dense<[{}, {}]> : vector<2xi64>}}",
                options.dilation[0], options.dilation[1], options.stride[0], options.stride[1]
            )
        };
        self.lines.push(format!(
            "    {name} = linalg.conv_2d_nhwc_hwcf{attrs} ins({}, {} : {}, {}) outs({init} : {}) -> {}",
            input.name,
            kernel.name,
            input.ty.mlir_type(),
            kernel.ty.mlir_type(),
            ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value::tensor(name, ty))
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
        let zero = self.fresh();
        self.lines.push(format!(
            "    {zero} = arith.constant {} : {}",
            ty.elem.zero_literal(),
            ty.elem.mlir_type()
        ));
        let name = self.fresh();
        self.lines.push(format!(
            "    {name} = tensor.pad {} low[0, {}, {}, 0] high[0, {}, {}, 0] {{",
            input.name, padding.top, padding.left, padding.bottom, padding.right
        ));
        self.lines
            .push("    ^bb0(%d0: index, %d1: index, %d2: index, %d3: index):".to_string());
        self.lines.push(format!(
            "      tensor.yield {zero} : {}",
            ty.elem.mlir_type()
        ));
        self.lines.push(format!(
            "    }} : {} to {}",
            input.ty.mlir_type(),
            ty.mlir_type()
        ));
        Ok(Value::tensor(name, ty))
    }
}
