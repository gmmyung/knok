use super::*;

impl<'a> ValueAndGradBuilder<'a> {
    pub(super) fn reverse_node(&mut self, node: &Node, grad: Expr) -> syn::Result<()> {
        match &node.expr {
            Expr::Var(name) => self.add_grad(name, grad),
            Expr::Const { .. } => Ok(()),
            Expr::Unary {
                op: UnaryOp::Neg,
                value,
            } => self.add_grad_to_expr(value, neg(grad)),
            Expr::Binary { op, lhs, rhs } => self.reverse_binary(*op, lhs, rhs, &node.ty, grad),
            Expr::Call { op, args } => self.reverse_call(node, op, args, grad),
            Expr::Node { .. } | Expr::TupleGet { .. } => Err(error(
                "internal autograd error: forward graph was not flattened",
            )),
        }
    }

    fn reverse_binary(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(lhs)?;
        let rhs_ty = self.expr_ty(rhs)?;
        match op {
            BinaryOp::Add => {
                let lhs_grad = self.unbroadcast(grad.clone(), output_ty, &lhs_ty)?;
                let rhs_grad = self.unbroadcast(grad, output_ty, &rhs_ty)?;
                self.add_grad_to_expr(lhs, lhs_grad)?;
                self.add_grad_to_expr(rhs, rhs_grad)
            }
            BinaryOp::Sub => {
                let lhs_grad = self.unbroadcast(grad.clone(), output_ty, &lhs_ty)?;
                let rhs_grad = self.unbroadcast(neg(grad), output_ty, &rhs_ty)?;
                self.add_grad_to_expr(lhs, lhs_grad)?;
                self.add_grad_to_expr(rhs, rhs_grad)
            }
            BinaryOp::Mul => {
                let lhs_grad = binary(BinaryOp::Mul, grad.clone(), rhs.clone());
                let rhs_grad = binary(BinaryOp::Mul, grad, lhs.clone());
                let lhs_grad = self.unbroadcast(lhs_grad, output_ty, &lhs_ty)?;
                let rhs_grad = self.unbroadcast(rhs_grad, output_ty, &rhs_ty)?;
                self.add_grad_to_expr(lhs, lhs_grad)?;
                self.add_grad_to_expr(rhs, rhs_grad)
            }
            BinaryOp::Div => {
                let lhs_grad = binary(BinaryOp::Div, grad.clone(), rhs.clone());
                let rhs_sq = call(CallOp::Square, vec![rhs.clone()]);
                let rhs_grad = neg(binary(
                    BinaryOp::Div,
                    binary(BinaryOp::Mul, grad, lhs.clone()),
                    rhs_sq,
                ));
                let lhs_grad = self.unbroadcast(lhs_grad, output_ty, &lhs_ty)?;
                let rhs_grad = self.unbroadcast(rhs_grad, output_ty, &rhs_ty)?;
                self.add_grad_to_expr(lhs, lhs_grad)?;
                self.add_grad_to_expr(rhs, rhs_grad)
            }
        }
    }

    fn reverse_call(
        &mut self,
        node: &Node,
        op: &CallOp,
        args: &[Expr],
        grad: Expr,
    ) -> syn::Result<()> {
        if call_stops_gradient(op, &node.ty) {
            return Ok(());
        }
        match op {
            CallOp::Abs => {
                let zero = zeros_like(args[0].clone());
                let positive = call(CallOp::Greater, vec![args[0].clone(), zero.clone()]);
                let negative = call(CallOp::Less, vec![args[0].clone(), zero.clone()]);
                let one = ones_like(args[0].clone());
                let local = call(
                    CallOp::Where,
                    vec![
                        positive,
                        one.clone(),
                        call(CallOp::Where, vec![negative, neg(one), zero]),
                    ],
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Exp => self.add_grad_to_expr(
                &args[0],
                binary(BinaryOp::Mul, grad, Expr::Var(node.name.clone())),
            ),
            CallOp::Exp2 => {
                let y = Expr::Var(node.name.clone());
                let local = binary(BinaryOp::Mul, scalar("0.6931471805599453", node.ty.elem), y);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::ExpM1 => {
                let y = Expr::Var(node.name.clone());
                let local = binary(BinaryOp::Add, y, scalar("1.0", node.ty.elem));
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Log => {
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, args[0].clone()))
            }
            CallOp::Log1P => {
                let denom = binary(BinaryOp::Add, args[0].clone(), scalar("1.0", node.ty.elem));
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, denom))
            }
            CallOp::Log2 => {
                let denom = binary(
                    BinaryOp::Mul,
                    args[0].clone(),
                    scalar("0.6931471805599453", node.ty.elem),
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, denom))
            }
            CallOp::Log10 => {
                let denom = binary(
                    BinaryOp::Mul,
                    args[0].clone(),
                    scalar("2.302585092994046", node.ty.elem),
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, denom))
            }
            CallOp::Sin => {
                let local = call(CallOp::Cos, vec![args[0].clone()]);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Cos => {
                let local = neg(call(CallOp::Sin, vec![args[0].clone()]));
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Tan => {
                let y = Expr::Var(node.name.clone());
                let local = binary(
                    BinaryOp::Add,
                    ones_like(y.clone()),
                    call(CallOp::Square, vec![y]),
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Tanh => {
                let y = Expr::Var(node.name.clone());
                let one_minus_y2 = binary(
                    BinaryOp::Sub,
                    ones_like(y.clone()),
                    call(CallOp::Square, vec![y]),
                );
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, one_minus_y2))
            }
            CallOp::Relu => {
                let zero = zeros_like(args[0].clone());
                let condition = call(CallOp::Greater, vec![args[0].clone(), zero]);
                let zero_grad = zeros_like(grad.clone());
                self.add_grad_to_expr(
                    &args[0],
                    call(CallOp::Where, vec![condition, grad, zero_grad]),
                )
            }
            CallOp::Sigmoid => {
                let y = Expr::Var(node.name.clone());
                let one_minus_y = binary(BinaryOp::Sub, ones_like(y.clone()), y.clone());
                let local = binary(BinaryOp::Mul, y, one_minus_y);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Square => {
                let two = scalar("2.0", node.ty.elem);
                let local = binary(BinaryOp::Mul, two, args[0].clone());
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Reciprocal => {
                let y = Expr::Var(node.name.clone());
                let local = neg(call(CallOp::Square, vec![y]));
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Sqrt => {
                let y = Expr::Var(node.name.clone());
                let two_y = binary(BinaryOp::Mul, scalar("2.0", node.ty.elem), y);
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Div, grad, two_y))
            }
            CallOp::Pow => self.reverse_pow(args, Expr::Var(node.name.clone()), &node.ty, grad),
            CallOp::Minimum => self.reverse_minimum_maximum(args, &node.ty, grad, false),
            CallOp::Maximum => self.reverse_minimum_maximum(args, &node.ty, grad, true),
            CallOp::Clip => self.reverse_clip(args, &node.ty, grad),
            CallOp::Where => self.reverse_where(args, &node.ty, grad),
            CallOp::Sum(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let input_grad = self.expand_reduction_grad(grad, &node.ty, &input_ty, *axis)?;
                self.add_grad_to_expr(&args[0], input_grad)
            }
            CallOp::Prod(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let output = Expr::Var(node.name.clone());
                let output = self.expand_reduction_grad(output, &node.ty, &input_ty, *axis)?;
                let grad = self.expand_reduction_grad(grad, &node.ty, &input_ty, *axis)?;
                let local = binary(BinaryOp::Div, output, args[0].clone());
                self.add_grad_to_expr(&args[0], binary(BinaryOp::Mul, grad, local))
            }
            CallOp::Mean(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let count = reduction_count(&input_ty, *axis);
                let scaled = binary(
                    BinaryOp::Div,
                    grad,
                    scalar(&format!("{count}.0"), node.ty.elem),
                );
                let input_grad = self.expand_reduction_grad(scaled, &node.ty, &input_ty, *axis)?;
                self.add_grad_to_expr(&args[0], input_grad)
            }
            CallOp::Max(axis) | CallOp::Min(axis) => {
                let input_ty = self.expr_ty(&args[0])?;
                let output = Expr::Var(node.name.clone());
                let selected = self.expand_reduction_grad(output, &node.ty, &input_ty, *axis)?;
                let grad = self.expand_reduction_grad(grad, &node.ty, &input_ty, *axis)?;
                let mask = call(CallOp::Equal, vec![args[0].clone(), selected]);
                self.add_grad_to_expr(&args[0], mask_grad(mask, grad))
            }
            CallOp::Var(axis) => self.reverse_var(args, &node.ty, *axis, grad),
            CallOp::Std(axis) => {
                self.reverse_std(args, Expr::Var(node.name.clone()), &node.ty, *axis, grad)
            }
            CallOp::Ptp(axis) => self.reverse_ptp(args, &node.ty, *axis, grad),
            CallOp::Softmax(axis) => {
                self.reverse_softmax(args, Expr::Var(node.name.clone()), *axis, grad)
            }
            CallOp::Matmul => self.reverse_matmul(args, &node.ty, grad),
            CallOp::Dot => self.reverse_dot(args, grad),
            CallOp::Vecdot(axis) => self.reverse_vecdot(args, &node.ty, *axis, grad),
            CallOp::Inner => self.reverse_inner(args, &node.ty, grad),
            CallOp::Outer => self.reverse_outer(args, &node.ty, grad),
            CallOp::Trace(axes) => self.reverse_trace(args, &node.ty, *axes, grad),
            CallOp::Diagonal(axes) => self.reverse_diagonal(args, *axes, grad),
            CallOp::Reshape(_) | CallOp::Squeeze(_) | CallOp::Unsqueeze(_) => {
                self.add_reshape_grad_to_arg(&args[0], grad)
            }
            CallOp::Broadcast(_) => {
                let input_ty = self.expr_ty(&args[0])?;
                let input_grad = self.unbroadcast(grad, &node.ty, &input_ty)?;
                self.add_grad_to_expr(&args[0], input_grad)
            }
            CallOp::Pad { lows, .. } => {
                let input_ty = self.expr_ty(&args[0])?;
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Slice {
                            target: input_ty,
                            starts: lows.clone(),
                        },
                        vec![grad],
                    ),
                )
            }
            CallOp::Slice { starts, .. } => {
                let input_ty = self.expr_ty(&args[0])?;
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Pad {
                            target: input_ty,
                            lows: starts.clone(),
                        },
                        vec![grad],
                    ),
                )
            }
            CallOp::Concat(axis) => self.reverse_concat(args, *axis, grad),
            CallOp::Stack(axis) => {
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Take {
                            axis: *axis,
                            index: 0,
                        },
                        vec![grad.clone()],
                    ),
                )?;
                self.add_grad_to_expr(
                    &args[1],
                    call(
                        CallOp::Take {
                            axis: *axis,
                            index: 1,
                        },
                        vec![grad],
                    ),
                )
            }
            CallOp::Take { axis, index } => {
                let input_ty = self.expr_ty(&args[0])?;
                let mut expanded_ty = input_ty.clone();
                expanded_ty.shape[*axis] = 1;
                let mut lows = vec![0; input_ty.rank()];
                lows[*axis] = *index;
                let expanded = call(CallOp::Reshape(expanded_ty), vec![grad]);
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Pad {
                            target: input_ty,
                            lows,
                        },
                        vec![expanded],
                    ),
                )
            }
            CallOp::Tile(multiples) => self.reverse_tile(args, &node.ty, multiples, grad),
            CallOp::Repeat { axis, count } => self.reverse_repeat(args, *axis, *count, grad),
            CallOp::Gather { axis, .. } => self.reverse_gather(args, &node.ty, *axis, grad),
            CallOp::TakeAlongAxis { axis } => {
                self.reverse_take_along_axis(args, &node.ty, *axis, grad)
            }
            CallOp::AvgPool2d(options) => self.reverse_avg_pool2d(args, options, grad),
            CallOp::MaxPool2d(options) => {
                self.reverse_max_pool2d(args, Expr::Var(node.name.clone()), &node.ty, options, grad)
            }
            CallOp::Conv2d(options) => self.reverse_conv2d(args, &node.ty, options, grad),
            CallOp::Transpose(axes) => {
                let input_ty = self.expr_ty(&args[0])?;
                let inverse = inverse_permutation(default_axes(&input_ty, axes));
                self.add_grad_to_expr(&args[0], call(CallOp::Transpose(inverse), vec![grad]))
            }
            CallOp::SwapAxes { axis0, axis1 } => self.add_grad_to_expr(
                &args[0],
                call(
                    CallOp::SwapAxes {
                        axis0: *axis0,
                        axis1: *axis1,
                    },
                    vec![grad],
                ),
            ),
            CallOp::MoveAxis {
                source,
                destination,
            } => self.add_grad_to_expr(
                &args[0],
                call(
                    CallOp::MoveAxis {
                        source: *destination,
                        destination: *source,
                    },
                    vec![grad],
                ),
            ),
            CallOp::PermuteDims(axes) => {
                let inverse = inverse_permutation(axes.clone());
                self.add_grad_to_expr(&args[0], call(CallOp::PermuteDims(inverse), vec![grad]))
            }
            CallOp::Permute { axes, .. } => {
                let inverse = inverse_permutation(axes.clone());
                self.add_grad_to_expr(&args[0], call(CallOp::PermuteDims(inverse), vec![grad]))
            }
            CallOp::Flip(axes) => {
                self.add_grad_to_expr(&args[0], call(CallOp::Flip(axes.clone()), vec![grad]))
            }
            CallOp::Roll { axis, shift } => {
                let input_ty = self.expr_ty(&args[0])?;
                let dim = input_ty.shape[*axis];
                let reverse_shift = if dim == 0 {
                    0
                } else {
                    (dim - (shift % dim)) % dim
                };
                self.add_grad_to_expr(
                    &args[0],
                    call(
                        CallOp::Roll {
                            axis: *axis,
                            shift: reverse_shift,
                        },
                        vec![grad],
                    ),
                )
            }
            CallOp::FullLike => {
                let fill_ty = self.expr_ty(&args[1])?;
                let fill_grad = self.unbroadcast(grad, &node.ty, &fill_ty)?;
                self.add_grad_to_expr(&args[1], fill_grad)
            }
            _ => Err(error(format!(
                "autograd MVP does not support differentiating floating-point `{}`",
                op.name()
            ))),
        }
    }

    fn reverse_matmul(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let spec = MatmulGradSpec::new(&lhs_ty, &rhs_ty, output_ty)?;
        let lhs = reshape_if_needed(args[0].clone(), &lhs_ty, &spec.lhs_matrix_ty);
        let rhs = reshape_if_needed(args[1].clone(), &rhs_ty, &spec.rhs_matrix_ty);
        let grad = reshape_if_needed(grad, output_ty, &spec.grad_matrix_ty);

        let rhs_t_ty = swap_last_two_type(&spec.rhs_matrix_ty)?;
        let rhs_t = swap_last_two(rhs, &spec.rhs_matrix_ty)?;
        let lhs_grad = call(CallOp::Matmul, vec![grad.clone(), rhs_t]);
        let lhs_grad_ty = matmul_type(&spec.grad_matrix_ty, &rhs_t_ty)?;
        let (lhs_grad, lhs_grad_ty) = spec.demote_lhs_grad(lhs_grad, lhs_grad_ty)?;
        let lhs_grad = self.unbroadcast(lhs_grad, &lhs_grad_ty, &lhs_ty)?;
        self.add_grad_to_expr(&args[0], lhs_grad)?;

        let lhs_t_ty = swap_last_two_type(&spec.lhs_matrix_ty)?;
        let lhs_t = swap_last_two(lhs, &spec.lhs_matrix_ty)?;
        let rhs_grad = call(CallOp::Matmul, vec![lhs_t, grad]);
        let rhs_grad_ty = matmul_type(&lhs_t_ty, &spec.grad_matrix_ty)?;
        let (rhs_grad, rhs_grad_ty) = spec.demote_rhs_grad(rhs_grad, rhs_grad_ty)?;
        let rhs_grad = self.unbroadcast(rhs_grad, &rhs_grad_ty, &rhs_ty)?;
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_pow(
        &mut self,
        args: &[Expr],
        output: Expr,
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let lhs_local = binary(
            BinaryOp::Mul,
            args[1].clone(),
            call(
                CallOp::Pow,
                vec![
                    args[0].clone(),
                    binary(
                        BinaryOp::Sub,
                        args[1].clone(),
                        scalar("1.0", output_ty.elem),
                    ),
                ],
            ),
        );
        let lhs_grad = binary(BinaryOp::Mul, grad.clone(), lhs_local);
        let lhs_grad = self.unbroadcast(lhs_grad, output_ty, &lhs_ty)?;
        self.add_grad_to_expr(&args[0], lhs_grad)?;

        let rhs_local = binary(
            BinaryOp::Mul,
            output,
            call(CallOp::Log, vec![args[0].clone()]),
        );
        let rhs_grad = binary(BinaryOp::Mul, grad, rhs_local);
        let rhs_grad = self.unbroadcast(rhs_grad, output_ty, &rhs_ty)?;
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_minimum_maximum(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
        maximum: bool,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let (lhs_op, rhs_op) = if maximum {
            (CallOp::GreaterEqual, CallOp::Greater)
        } else {
            (CallOp::LessEqual, CallOp::Less)
        };
        let lhs_mask = call(lhs_op, vec![args[0].clone(), args[1].clone()]);
        let rhs_mask = call(rhs_op, vec![args[1].clone(), args[0].clone()]);
        let lhs_grad = self.unbroadcast(mask_grad(lhs_mask, grad.clone()), output_ty, &lhs_ty)?;
        let rhs_grad = self.unbroadcast(mask_grad(rhs_mask, grad), output_ty, &rhs_ty)?;
        self.add_grad_to_expr(&args[0], lhs_grad)?;
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_clip(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let value_ty = self.expr_ty(&args[0])?;
        let min_ty = self.expr_ty(&args[1])?;
        let max_ty = self.expr_ty(&args[2])?;
        let above_min = call(CallOp::GreaterEqual, vec![args[0].clone(), args[1].clone()]);
        let below_max = call(CallOp::LessEqual, vec![args[0].clone(), args[2].clone()]);
        let value_mask = call(CallOp::LogicalAnd, vec![above_min, below_max]);
        let min_mask = call(CallOp::Less, vec![args[0].clone(), args[1].clone()]);
        let max_mask = call(CallOp::Greater, vec![args[0].clone(), args[2].clone()]);

        let value_grad =
            self.unbroadcast(mask_grad(value_mask, grad.clone()), output_ty, &value_ty)?;
        let min_grad = self.unbroadcast(mask_grad(min_mask, grad.clone()), output_ty, &min_ty)?;
        let max_grad = self.unbroadcast(mask_grad(max_mask, grad), output_ty, &max_ty)?;
        self.add_grad_to_expr(&args[0], value_grad)?;
        self.add_grad_to_expr(&args[1], min_grad)?;
        self.add_grad_to_expr(&args[2], max_grad)
    }

    fn reverse_where(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let true_ty = self.expr_ty(&args[1])?;
        let false_ty = self.expr_ty(&args[2])?;
        let true_grad = self.unbroadcast(
            mask_grad(args[0].clone(), grad.clone()),
            output_ty,
            &true_ty,
        )?;
        let false_grad = self.unbroadcast(
            call(
                CallOp::Where,
                vec![args[0].clone(), zeros_like(grad.clone()), grad],
            ),
            output_ty,
            &false_ty,
        )?;
        self.add_grad_to_expr(&args[1], true_grad)?;
        self.add_grad_to_expr(&args[2], false_grad)
    }

    fn reverse_concat(&mut self, args: &[Expr], axis: usize, grad: Expr) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let lhs_starts = vec![0; lhs_ty.rank()];
        let mut rhs_starts = vec![0; rhs_ty.rank()];
        rhs_starts[axis] = lhs_ty.shape[axis];
        let lhs_grad = call(
            CallOp::Slice {
                target: lhs_ty,
                starts: lhs_starts,
            },
            vec![grad.clone()],
        );
        let rhs_grad = call(
            CallOp::Slice {
                target: rhs_ty,
                starts: rhs_starts,
            },
            vec![grad],
        );
        self.add_grad_to_expr(&args[0], lhs_grad)?;
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_softmax(
        &mut self,
        args: &[Expr],
        output: Expr,
        axis: AxisSpec,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let reduced_ty = single_call_type(&CallOp::Sum(axis), core::slice::from_ref(&input_ty))?;
        let weighted = binary(BinaryOp::Mul, grad.clone(), output.clone());
        let summed = call(CallOp::Sum(axis), vec![weighted]);
        let summed = self.expand_reduction_grad(summed, &reduced_ty, &input_ty, axis)?;
        let input_grad = binary(BinaryOp::Mul, output, binary(BinaryOp::Sub, grad, summed));
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_var(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: AxisSpec,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if input_ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(());
        }
        let mean = call(CallOp::Mean(axis), vec![args[0].clone()]);
        let mean = self.expand_reduction_grad(mean, output_ty, &input_ty, axis)?;
        let centered = binary(BinaryOp::Sub, args[0].clone(), mean);
        let grad = self.expand_reduction_grad(grad, output_ty, &input_ty, axis)?;
        let local = binary(
            BinaryOp::Div,
            binary(BinaryOp::Mul, scalar("2.0", input_ty.elem), centered),
            scalar(
                &format!("{}.0", reduction_count(&input_ty, axis)),
                input_ty.elem,
            ),
        );
        let input_grad = binary(BinaryOp::Mul, grad, local);
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_std(
        &mut self,
        args: &[Expr],
        output: Expr,
        output_ty: &TensorType,
        axis: AxisSpec,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if input_ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(());
        }
        let mean = call(CallOp::Mean(axis), vec![args[0].clone()]);
        let mean = self.expand_reduction_grad(mean, output_ty, &input_ty, axis)?;
        let centered = binary(BinaryOp::Sub, args[0].clone(), mean);
        let grad = self.expand_reduction_grad(grad, output_ty, &input_ty, axis)?;
        let std = self.expand_reduction_grad(output, output_ty, &input_ty, axis)?;
        let count = scalar(
            &format!("{}.0", reduction_count(&input_ty, axis)),
            input_ty.elem,
        );
        let denom = binary(BinaryOp::Mul, count, std);
        let input_grad = binary(BinaryOp::Mul, grad, binary(BinaryOp::Div, centered, denom));
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_ptp(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: AxisSpec,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if input_ty.rank() == 0 && matches!(axis, AxisSpec::All) {
            return Ok(());
        }
        let max = call(CallOp::Max(axis), vec![args[0].clone()]);
        let min = call(CallOp::Min(axis), vec![args[0].clone()]);
        let max = self.expand_reduction_grad(max, output_ty, &input_ty, axis)?;
        let min = self.expand_reduction_grad(min, output_ty, &input_ty, axis)?;
        let grad = self.expand_reduction_grad(grad, output_ty, &input_ty, axis)?;
        let max_grad = mask_grad(
            call(CallOp::Equal, vec![args[0].clone(), max]),
            grad.clone(),
        );
        let min_grad = mask_grad(call(CallOp::Equal, vec![args[0].clone(), min]), neg(grad));
        self.add_grad_to_expr(&args[0], binary(BinaryOp::Add, max_grad, min_grad))
    }

    fn reverse_dot(&mut self, args: &[Expr], grad: Expr) -> syn::Result<()> {
        self.add_grad_to_expr(
            &args[0],
            binary(BinaryOp::Mul, grad.clone(), args[1].clone()),
        )?;
        self.add_grad_to_expr(&args[1], binary(BinaryOp::Mul, grad, args[0].clone()))
    }

    fn reverse_vecdot(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: Option<usize>,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let axis = AxisSpec::One(axis.unwrap_or(lhs_ty.rank() - 1));
        let grad = self.expand_reduction_grad(grad, output_ty, &lhs_ty, axis)?;
        self.add_grad_to_expr(
            &args[0],
            binary(BinaryOp::Mul, grad.clone(), args[1].clone()),
        )?;
        self.add_grad_to_expr(&args[1], binary(BinaryOp::Mul, grad, args[0].clone()))
    }

    fn reverse_inner(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        if lhs_ty.rank() == 0 || rhs_ty.rank() == 0 {
            let lhs_grad = self.unbroadcast(
                binary(BinaryOp::Mul, grad.clone(), args[1].clone()),
                output_ty,
                &lhs_ty,
            )?;
            let rhs_grad = self.unbroadcast(
                binary(BinaryOp::Mul, grad, args[0].clone()),
                output_ty,
                &rhs_ty,
            )?;
            self.add_grad_to_expr(&args[0], lhs_grad)?;
            return self.add_grad_to_expr(&args[1], rhs_grad);
        }

        let k = lhs_ty.shape[lhs_ty.rank() - 1];
        let lhs_prefix = element_count_slice(&lhs_ty.shape[..lhs_ty.rank() - 1]);
        let rhs_prefix = element_count_slice(&rhs_ty.shape[..rhs_ty.rank() - 1]);
        let matrix_elem = lhs_ty.elem;
        let lhs_matrix_ty = tensor(matrix_elem, vec![lhs_prefix, k]);
        let rhs_matrix_ty = tensor(matrix_elem, vec![rhs_prefix, k]);
        let grad_matrix_ty = tensor(matrix_elem, vec![lhs_prefix, rhs_prefix]);
        let lhs = call(
            CallOp::Reshape(lhs_matrix_ty.clone()),
            vec![args[0].clone()],
        );
        let rhs = call(
            CallOp::Reshape(rhs_matrix_ty.clone()),
            vec![args[1].clone()],
        );
        let grad = reshape_if_needed(grad, output_ty, &grad_matrix_ty);

        let lhs_grad = call(CallOp::Matmul, vec![grad.clone(), rhs]);
        let lhs_grad = call(CallOp::Reshape(lhs_ty.clone()), vec![lhs_grad]);
        self.add_grad_to_expr(&args[0], lhs_grad)?;

        let grad_t = call(CallOp::Transpose(Vec::new()), vec![grad]);
        let rhs_grad = call(CallOp::Matmul, vec![grad_t, lhs]);
        let rhs_grad = call(CallOp::Reshape(rhs_ty), vec![rhs_grad]);
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_outer(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        grad: Expr,
    ) -> syn::Result<()> {
        let lhs_ty = self.expr_ty(&args[0])?;
        let rhs_ty = self.expr_ty(&args[1])?;
        let lhs_len = element_count_slice(&lhs_ty.shape);
        let rhs_len = element_count_slice(&rhs_ty.shape);
        let elem = lhs_ty.elem;
        let lhs_row_ty = tensor(elem, vec![1, lhs_len]);
        let rhs_col_ty = tensor(elem, vec![rhs_len, 1]);
        let grad_ty = tensor(elem, vec![lhs_len, rhs_len]);
        let lhs_row = call(CallOp::Reshape(lhs_row_ty), vec![args[0].clone()]);
        let rhs_col = call(CallOp::Reshape(rhs_col_ty), vec![args[1].clone()]);
        let grad = reshape_if_needed(grad, output_ty, &grad_ty);

        let lhs_grad = call(CallOp::Matmul, vec![grad.clone(), rhs_col]);
        let lhs_grad = call(CallOp::Reshape(lhs_ty), vec![lhs_grad]);
        self.add_grad_to_expr(&args[0], lhs_grad)?;

        let rhs_grad = call(CallOp::Matmul, vec![lhs_row, grad]);
        let rhs_grad = call(CallOp::Reshape(rhs_ty), vec![rhs_grad]);
        self.add_grad_to_expr(&args[1], rhs_grad)
    }

    fn reverse_trace(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axes: Option<[usize; 2]>,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let axes = default_axis_pair(&input_ty, axes)?;
        let compact_ty = diagonal_compact_type(&input_ty, axes);
        let mut grad_shape = output_ty.shape.clone();
        grad_shape.push(1);
        grad_shape.push(1);
        let grad = call(
            CallOp::Reshape(tensor(input_ty.elem, grad_shape)),
            vec![grad],
        );
        let grad = call(CallOp::Broadcast(compact_ty.clone()), vec![grad]);
        let eye = diagonal_eye(&input_ty, axes, &compact_ty);
        let compact = binary(BinaryOp::Mul, grad, eye);
        self.add_grad_to_expr(&args[0], permute_compact_diagonal(compact, &input_ty, axes))
    }

    fn reverse_diagonal(
        &mut self,
        args: &[Expr],
        axes: Option<[usize; 2]>,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let axes = default_axis_pair(&input_ty, axes)?;
        let compact_ty = diagonal_compact_type(&input_ty, axes);
        let mut grad_shape = remove_axes_shape(&input_ty.shape, axes);
        grad_shape.push(input_ty.shape[axes[0]]);
        grad_shape.push(1);
        let grad = call(
            CallOp::Reshape(tensor(input_ty.elem, grad_shape)),
            vec![grad],
        );
        let grad = call(CallOp::Broadcast(compact_ty.clone()), vec![grad]);
        let eye = diagonal_eye(&input_ty, axes, &compact_ty);
        let compact = binary(BinaryOp::Mul, grad, eye);
        self.add_grad_to_expr(&args[0], permute_compact_diagonal(compact, &input_ty, axes))
    }

    fn reverse_tile(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        multiples: &[usize],
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if multiples.contains(&0) {
            return self.add_grad_to_expr(&args[0], zeros_like(args[0].clone()));
        }
        let mut current = grad;
        let mut current_ty = output_ty.clone();
        for (axis, multiple) in multiples.iter().copied().enumerate().rev() {
            if multiple <= 1 {
                continue;
            }
            let mut slice_ty = current_ty.clone();
            slice_ty.shape[axis] = input_ty.shape[axis];
            let mut acc = None;
            for index in 0..multiple {
                let mut starts = vec![0; current_ty.rank()];
                starts[axis] = index * input_ty.shape[axis];
                let slice = call(
                    CallOp::Slice {
                        target: slice_ty.clone(),
                        starts,
                    },
                    vec![current.clone()],
                );
                acc = Some(match acc {
                    Some(existing) => binary(BinaryOp::Add, existing, slice),
                    None => slice,
                });
            }
            current = acc.ok_or_else(|| error("internal autograd error: empty tile gradient"))?;
            current_ty = slice_ty;
        }
        self.add_grad_to_expr(&args[0], current)
    }

    fn reverse_repeat(
        &mut self,
        args: &[Expr],
        axis: usize,
        count: usize,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        if count == 0 || input_ty.shape[axis] == 0 {
            return self.add_grad_to_expr(&args[0], zeros_like(args[0].clone()));
        }
        if count == 1 {
            return self.add_grad_to_expr(&args[0], grad);
        }
        let mut expanded_shape = input_ty.shape.clone();
        expanded_shape.insert(axis + 1, count);
        if expanded_shape.len() > 6 {
            return Err(error(
                "autograd MVP does not support repeat gradients that require rank above 6",
            ));
        }
        let expanded_ty = tensor(input_ty.elem, expanded_shape);
        let expanded = call(CallOp::Reshape(expanded_ty), vec![grad]);
        let result = call(CallOp::Sum(AxisSpec::One(axis + 1)), vec![expanded]);
        self.add_grad_to_expr(&args[0], result)
    }

    fn reverse_gather(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: usize,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let indices_ty = self.expr_ty(&args[1])?;
        let reduce_axes = (axis..axis + indices_ty.rank()).collect::<Vec<_>>();
        let input_grad = indexed_axis_grad(
            &args[0],
            &args[1],
            &indices_ty,
            output_ty,
            &input_ty,
            axis,
            &reduce_axes,
            grad,
        )?;
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_take_along_axis(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        axis: usize,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let indices_ty = self.expr_ty(&args[1])?;
        let input_grad = indexed_axis_grad(
            &args[0],
            &args[1],
            &indices_ty,
            output_ty,
            &input_ty,
            axis,
            &[axis],
            grad,
        )?;
        self.add_grad_to_expr(&args[0], input_grad)
    }

    fn reverse_avg_pool2d(
        &mut self,
        args: &[Expr],
        options: &Pool2dOptions,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let output_ty = single_call_type(
            &CallOp::AvgPool2d(options.clone()),
            core::slice::from_ref(&input_ty),
        )?;
        let unit_ty = pool_unit_type(&input_ty);
        let scale = scalar(
            &format!("{}.0", options.kernel[0] * options.kernel[1]),
            input_ty.elem,
        );
        let mut acc = None;
        for oh in 0..output_ty.shape[1] {
            for ow in 0..output_ty.shape[2] {
                let grad_unit = pool_output_unit(grad.clone(), &output_ty, oh, ow);
                let grad_unit = binary(BinaryOp::Div, grad_unit, scale.clone());
                for kh in 0..options.kernel[0] {
                    for kw in 0..options.kernel[1] {
                        if let Some((ih, iw)) = pool_input_position(options, oh, ow, kh, kw) {
                            if ih >= input_ty.shape[1] || iw >= input_ty.shape[2] {
                                continue;
                            }
                            let padded = pad_pool_unit_to_input(
                                grad_unit.clone(),
                                &unit_ty,
                                &input_ty,
                                ih,
                                iw,
                            );
                            push_sum(&mut acc, padded);
                        }
                    }
                }
            }
        }
        self.add_grad_to_expr(&args[0], acc.unwrap_or_else(|| zeros_like(args[0].clone())))
    }

    fn reverse_max_pool2d(
        &mut self,
        args: &[Expr],
        output: Expr,
        output_ty: &TensorType,
        options: &Pool2dOptions,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let unit_ty = pool_unit_type(&input_ty);
        let mut acc = None;
        for oh in 0..output_ty.shape[1] {
            for ow in 0..output_ty.shape[2] {
                let grad_unit = pool_output_unit(grad.clone(), output_ty, oh, ow);
                let output_unit = pool_output_unit(output.clone(), output_ty, oh, ow);
                for kh in 0..options.kernel[0] {
                    for kw in 0..options.kernel[1] {
                        if let Some((ih, iw)) = pool_input_position(options, oh, ow, kh, kw) {
                            if ih >= input_ty.shape[1] || iw >= input_ty.shape[2] {
                                continue;
                            }
                            let input_unit = call(
                                CallOp::Slice {
                                    target: unit_ty.clone(),
                                    starts: vec![0, ih, iw, 0],
                                },
                                vec![args[0].clone()],
                            );
                            let masked = mask_grad(
                                call(CallOp::Equal, vec![input_unit, output_unit.clone()]),
                                grad_unit.clone(),
                            );
                            let padded =
                                pad_pool_unit_to_input(masked, &unit_ty, &input_ty, ih, iw);
                            push_sum(&mut acc, padded);
                        }
                    }
                }
            }
        }
        self.add_grad_to_expr(&args[0], acc.unwrap_or_else(|| zeros_like(args[0].clone())))
    }

    fn reverse_conv2d(
        &mut self,
        args: &[Expr],
        output_ty: &TensorType,
        options: &Conv2dOptions,
        grad: Expr,
    ) -> syn::Result<()> {
        let input_ty = self.expr_ty(&args[0])?;
        let kernel_ty = self.expr_ty(&args[1])?;
        let groups = options.groups;
        let input_channels_per_group = input_ty.shape[3] / groups;
        let output_channels_per_group = kernel_ty.shape[3] / groups;

        let mut input_acc = None;
        let mut kernel_acc = None;
        for group in 0..groups {
            let output_channel_start = group * output_channels_per_group;
            let input_channel_start = group * input_channels_per_group;
            for kh in 0..kernel_ty.shape[0] {
                for kw in 0..kernel_ty.shape[1] {
                    for oh in 0..output_ty.shape[1] {
                        for ow in 0..output_ty.shape[2] {
                            let Some((ih, iw)) = conv_input_position(options, oh, ow, kh, kw)
                            else {
                                continue;
                            };
                            if ih >= input_ty.shape[1] || iw >= input_ty.shape[2] {
                                continue;
                            }
                            let grad_block = conv_output_channel_block(
                                grad.clone(),
                                output_ty,
                                oh,
                                ow,
                                output_channel_start,
                                output_channels_per_group,
                            );
                            for local_ic in 0..input_channels_per_group {
                                let input_channel = input_channel_start + local_ic;
                                let input_unit = conv_input_unit(
                                    args[0].clone(),
                                    &input_ty,
                                    ih,
                                    iw,
                                    input_channel,
                                );
                                let kernel_block = conv_kernel_output_block(
                                    args[1].clone(),
                                    &kernel_ty,
                                    kh,
                                    kw,
                                    local_ic,
                                    output_channel_start,
                                    output_channels_per_group,
                                );

                                let input_piece =
                                    binary(BinaryOp::Mul, grad_block.clone(), kernel_block);
                                let input_piece =
                                    call(CallOp::Sum(AxisSpec::One(3)), vec![input_piece]);
                                let input_piece = call(
                                    CallOp::Reshape(conv_input_unit_type(&input_ty)),
                                    vec![input_piece],
                                );
                                let input_piece = call(
                                    CallOp::Pad {
                                        target: input_ty.clone(),
                                        lows: vec![0, ih, iw, input_channel],
                                    },
                                    vec![input_piece],
                                );
                                push_sum(&mut input_acc, input_piece);

                                let kernel_piece =
                                    binary(BinaryOp::Mul, input_unit, grad_block.clone());
                                let kernel_piece =
                                    call(CallOp::Sum(AxisSpec::One(2)), vec![kernel_piece]);
                                let kernel_piece =
                                    call(CallOp::Sum(AxisSpec::One(1)), vec![kernel_piece]);
                                let kernel_piece =
                                    call(CallOp::Sum(AxisSpec::One(0)), vec![kernel_piece]);
                                let kernel_unit_ty = tensor(
                                    kernel_ty.elem,
                                    vec![1, 1, 1, output_channels_per_group],
                                );
                                let kernel_piece = call(
                                    CallOp::Reshape(kernel_unit_ty.clone()),
                                    vec![kernel_piece],
                                );
                                let kernel_piece = call(
                                    CallOp::Pad {
                                        target: kernel_ty.clone(),
                                        lows: vec![kh, kw, local_ic, output_channel_start],
                                    },
                                    vec![kernel_piece],
                                );
                                push_sum(&mut kernel_acc, kernel_piece);
                            }
                        }
                    }
                }
            }
        }

        self.add_grad_to_expr(
            &args[0],
            input_acc.unwrap_or_else(|| zeros_like(args[0].clone())),
        )?;
        self.add_grad_to_expr(
            &args[1],
            kernel_acc.unwrap_or_else(|| zeros_like(args[1].clone())),
        )
    }
}
