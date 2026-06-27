use super::*;
use quote::quote;
use syn::{parse_quote, ItemFn};

fn tensor(shape: &[usize]) -> TensorType {
    TensorType {
        elem: ElementType::F32,
        shape: shape.to_vec(),
    }
}

fn parse(item: ItemFn) -> syn::Result<TypedGraph> {
    parse_graph(quote!(backend = Backend::LlvmCpu), item)
}

#[test]
fn parses_and_types_elementwise_graph() {
    let graph = parse(parse_quote! {
        fn add(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
            x + y
        }
    })
    .unwrap();

    assert_eq!(graph.name, "add");
    assert_eq!(graph.backend, "llvm-cpu");
    assert_eq!(graph.inputs.len(), 2);
    assert_eq!(graph.outputs[0], tensor(&[4]));
    assert_eq!(graph.body[0].ty, tensor(&[4]));
}

#[test]
fn parses_and_types_multi_output_graph() {
    let graph = parse(parse_quote! {
        fn add_sub(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> (Tensor1<f32, 4>, Tensor1<f32, 4>) {
            (x + y, x - y)
        }
    })
    .unwrap();

    assert_eq!(graph.outputs, vec![tensor(&[4]), tensor(&[4])]);
    assert_eq!(graph.body.len(), 2);
    assert_eq!(graph.body[0].ty, tensor(&[4]));
    assert_eq!(graph.body[1].ty, tensor(&[4]));
}

#[test]
fn destructures_multi_output_graph_call_in_let_binding() {
    let signatures = [(
        "add_sub".to_string(),
        GraphSignature {
            inputs: vec![tensor(&[4]), tensor(&[4])],
            outputs: vec![tensor(&[4]), tensor(&[4])],
        },
    )];

    let graph = parse_graph_with_signatures(
        quote!(backend = Backend::LlvmCpu),
        parse_quote! {
            fn combine(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                let (sum, diff) = add_sub(x, y);
                sum * diff
            }
        },
        &signatures,
    )
    .unwrap();

    assert_eq!(graph.lets[0].names, vec!["sum", "diff"]);
    assert_eq!(graph.lets[0].value.tys, vec![tensor(&[4]), tensor(&[4])]);
    assert_eq!(graph.body[0].ty, tensor(&[4]));
}

#[test]
fn rejects_multi_output_let_destructuring_arity_mismatch() {
    let signatures = [(
        "add_sub".to_string(),
        GraphSignature {
            inputs: vec![tensor(&[4]), tensor(&[4])],
            outputs: vec![tensor(&[4]), tensor(&[4])],
        },
    )];

    let error = parse_graph_with_signatures(
        quote!(backend = Backend::LlvmCpu),
        parse_quote! {
            fn combine(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                let (sum, diff, extra) = add_sub(x, y);
                sum + diff + extra
            }
        },
        &signatures,
    )
    .unwrap_err();

    assert!(error.to_string().contains("let pattern expects 3 values"));
}

#[cfg(feature = "half")]
#[test]
fn parses_half_element_types() {
    let f16_graph = parse(parse_quote! {
        fn add(x: Tensor1<half::f16, 4>, y: Tensor1<half::f16, 4>) -> Tensor1<half::f16, 4> {
            x + y
        }
    })
    .unwrap();
    assert_eq!(f16_graph.outputs[0].elem, ElementType::F16);

    let bf16_graph = parse(parse_quote! {
        fn identity(x: Tensor1<knok::half::bf16, 4>) -> Tensor1<knok::half::bf16, 4> {
            x
        }
    })
    .unwrap();
    assert_eq!(bf16_graph.outputs[0].elem, ElementType::BF16);
}

#[test]
fn infers_broadcast_elementwise_graph() {
    let graph = parse(parse_quote! {
        fn add_bias(x: Tensor2<f32, 2, 3>, bias: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
            x + bias
        }
    })
    .unwrap();

    assert_eq!(graph.body[0].ty, tensor(&[2, 3]));
}

#[test]
fn infers_matmul_shape() {
    let graph = parse(parse_quote! {
        fn mm(x: Tensor2<f32, 2, 3>, y: Tensor2<f32, 3, 4>) -> Tensor2<f32, 2, 4> {
            matmul(x, y)
        }
    })
    .unwrap();

    assert_eq!(graph.outputs[0], tensor(&[2, 4]));
    assert_eq!(graph.body[0].ty, tensor(&[2, 4]));
}

#[test]
fn infers_linalg_contraction_shapes() {
    let dot = parse(parse_quote! {
        fn vector_dot(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor0<f32> {
            dot(x, y)
        }
    })
    .unwrap();
    assert_eq!(dot.body[0].ty, tensor(&[]));

    let vecdot = parse(parse_quote! {
        fn row_vecdot(x: Tensor2<f32, 2, 3>, y: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
            vecdot::<1>(x, y)
        }
    })
    .unwrap();
    assert_eq!(vecdot.body[0].ty, tensor(&[2]));

    let inner = parse(parse_quote! {
        fn matrix_inner(x: Tensor2<f32, 2, 3>, y: Tensor2<f32, 4, 3>) -> Tensor2<f32, 2, 4> {
            inner(x, y)
        }
    })
    .unwrap();
    assert_eq!(inner.body[0].ty, tensor(&[2, 4]));

    let outer = parse(parse_quote! {
        fn vector_outer(x: Tensor1<f32, 2>, y: Tensor2<f32, 3, 4>) -> Tensor2<f32, 2, 12> {
            outer(x, y)
        }
    })
    .unwrap();
    assert_eq!(outer.body[0].ty, tensor(&[2, 12]));

    let trace = parse(parse_quote! {
        fn batched_trace(x: Tensor3<f32, 2, 3, 3>) -> Tensor1<f32, 2> {
            trace(x)
        }
    })
    .unwrap();
    assert_eq!(trace.body[0].ty, tensor(&[2]));

    let diagonal = parse(parse_quote! {
        fn diagonal_square(x: Tensor2<f32, 2, 2>) -> Tensor1<f32, 2> {
            diagonal(x)
        }
    })
    .unwrap();
    assert_eq!(diagonal.body[0].ty, tensor(&[2]));
}

#[test]
fn infers_reshape_broadcast_and_sum_shapes() {
    let reshape = parse(parse_quote! {
        fn reshape4(x: Tensor1<f32, 4>) -> Tensor2<f32, 2, 2> {
            reshape::<Tensor2<f32, 2, 2>>(x)
        }
    })
    .unwrap();
    assert_eq!(reshape.body[0].ty, tensor(&[2, 2]));

    let broadcast = parse(parse_quote! {
        fn broadcast4(x: Tensor1<f32, 1>) -> Tensor1<f32, 4> {
            broadcast::<Tensor1<f32, 4>>(x)
        }
    })
    .unwrap();
    assert_eq!(broadcast.body[0].ty, tensor(&[4]));

    let sum = parse(parse_quote! {
        fn sum4(x: Tensor1<f32, 4>) -> Tensor0<f32> {
            sum(x)
        }
    })
    .unwrap();
    assert_eq!(sum.body[0].ty, tensor(&[]));

    let axis_sum = parse(parse_quote! {
        fn sum_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
            sum::<1>(x)
        }
    })
    .unwrap();
    assert_eq!(axis_sum.body[0].ty, tensor(&[2]));

    let axis_mean = parse(parse_quote! {
        fn mean_axis0(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 3> {
            mean::<0>(x)
        }
    })
    .unwrap();
    assert_eq!(axis_mean.body[0].ty, tensor(&[3]));

    let axis_sum4d = parse(parse_quote! {
        fn sum_axis3(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor3<f32, 1, 2, 2> {
            sum::<3>(x)
        }
    })
    .unwrap();
    assert_eq!(axis_sum4d.body[0].ty, tensor(&[1, 2, 2]));

    let axis_mean4d = parse(parse_quote! {
        fn mean_axis2(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor3<f32, 1, 2, 3> {
            mean::<2>(x)
        }
    })
    .unwrap();
    assert_eq!(axis_mean4d.body[0].ty, tensor(&[1, 2, 3]));

    for item in [
        parse_quote! {
            fn prod_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
                prod::<1>(x)
            }
        },
        parse_quote! {
            fn max_axis0(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 3> {
                max::<0>(x)
            }
        },
        parse_quote! {
            fn amax_all(x: Tensor2<i32, 2, 3>) -> Tensor0<i32> {
                amax(x)
            }
        },
        parse_quote! {
            fn min_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
                min::<1>(x)
            }
        },
        parse_quote! {
            fn amin_all(x: Tensor2<i64, 2, 3>) -> Tensor0<i64> {
                amin(x)
            }
        },
        parse_quote! {
            fn var_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
                var::<1>(x)
            }
        },
        parse_quote! {
            fn std_all(x: Tensor2<f32, 2, 3>) -> Tensor0<f32> {
                std(x)
            }
        },
        parse_quote! {
            fn ptp_axis0(x: Tensor2<i32, 2, 3>) -> Tensor1<i32, 3> {
                ptp::<0>(x)
            }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }

    let argmin = parse(parse_quote! {
        fn argmin_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<i64, 2> {
            argmin::<1>(x)
        }
    })
    .unwrap();
    assert_eq!(
        argmin.body[0].ty,
        TensorType {
            elem: ElementType::I64,
            shape: vec![2]
        }
    );

    let scalar_var = parse(parse_quote! {
        fn scalar_var(x: Tensor0<f32>) -> Tensor0<f32> {
            var(x)
        }
    })
    .unwrap();
    assert_eq!(scalar_var.body[0].ty, tensor(&[]));
}

#[test]
fn infers_static_creator_shapes() {
    for item in [
        parse_quote! {
            fn zeros_like_rank6(x: Tensor6<i32, 1, 1, 1, 1, 2, 3>) -> Tensor6<i32, 1, 1, 1, 1, 2, 3> {
                zeros_like(x)
            }
        },
        parse_quote! {
            fn ones_like_bool(x: Tensor2<bool, 2, 2>) -> Tensor2<bool, 2, 2> {
                ones_like(x)
            }
        },
        parse_quote! {
            fn full_like_vec(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                full_like(x, 3.5)
            }
        },
        parse_quote! {
            fn arange_i32() -> Tensor1<i32, 4> {
                arange::<Tensor1<i32, 4>>(0, 8, 2)
            }
        },
        parse_quote! {
            fn arange_f32() -> Tensor1<f32, 4> {
                arange::<Tensor1<f32, 4>>(1.5, -0.5, -0.5)
            }
        },
        parse_quote! {
            fn linspace_f32() -> Tensor1<f32, 5> {
                linspace::<Tensor1<f32, 5>>(0.0, 1.0)
            }
        },
        parse_quote! {
            fn linspace_i64() -> Tensor1<i64, 4> {
                linspace::<Tensor1<i64, 4>>(2i64, 8i64)
            }
        },
        parse_quote! {
            fn eye3() -> Tensor2<f32, 3, 3> {
                eye::<Tensor2<f32, 3, 3>>()
            }
        },
        parse_quote! {
            fn identity_bool() -> Tensor2<bool, 2, 2> {
                identity::<Tensor2<bool, 2, 2>>()
            }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }
}

#[test]
fn static_creator_float_literals_preserve_tiny_values() {
    let f32_target = TensorType {
        elem: ElementType::F32,
        shape: vec![3],
    };
    let arange = static_arange_literals(
        &f32_target,
        &[
            Expr::Const {
                value: "1e-20".to_string(),
                elem: ElementType::F32,
            },
            Expr::Const {
                value: "4e-20".to_string(),
                elem: ElementType::F32,
            },
            Expr::Const {
                value: "1e-20".to_string(),
                elem: ElementType::F32,
            },
        ],
    )
    .unwrap();

    assert_eq!(arange.len(), 3);
    for literal in &arange {
        assert_ne!(literal, "0.0");
        assert_ne!(literal.parse::<f64>().unwrap(), 0.0);
    }

    let f64_target = TensorType {
        elem: ElementType::F64,
        shape: vec![3],
    };
    let linspace = static_linspace_literals(
        &f64_target,
        &[
            Expr::Const {
                value: "1e-20".to_string(),
                elem: ElementType::F64,
            },
            Expr::Const {
                value: "3e-20".to_string(),
                elem: ElementType::F64,
            },
        ],
    )
    .unwrap();

    assert_eq!(linspace.len(), 3);
    for literal in &linspace {
        assert_ne!(literal, "0.0");
        assert_ne!(literal.parse::<f64>().unwrap(), 0.0);
    }
}

#[test]
fn infers_static_shape_and_indexing_ops() {
    for item in [
        parse_quote! {
            fn slice_mid(x: Tensor2<f32, 2, 4>) -> Tensor2<f32, 2, 2> {
                slice::<Tensor2<f32, 2, 2>, 0, 1>(x)
            }
        },
        parse_quote! {
            fn take_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<f32, 2> {
                take::<1, 2>(x)
            }
        },
        parse_quote! {
            fn gather_axis0(
                x: Tensor2<f32, 2, 3>,
                indices: Tensor1<i64, 2>,
            ) -> Tensor2<f32, 2, 3> {
                gather::<Tensor2<f32, 2, 3>, 0>(x, indices)
            }
        },
        parse_quote! {
            fn gather_axis1_matrix_indices(
                x: Tensor2<f32, 2, 3>,
                indices: Tensor2<i32, 2, 2>,
            ) -> Tensor3<f32, 2, 2, 2> {
                gather::<Tensor3<f32, 2, 2, 2>, 1>(x, indices)
            }
        },
        parse_quote! {
            fn take_along_axis1(
                x: Tensor2<f32, 2, 3>,
                indices: Tensor2<i64, 2, 2>,
            ) -> Tensor2<f32, 2, 2> {
                take_along_axis::<1>(x, indices)
            }
        },
        parse_quote! {
            fn squeeze4(x: Tensor4<f32, 1, 2, 1, 3>) -> Tensor2<f32, 2, 3> {
                squeeze::<Tensor2<f32, 2, 3>>(x)
            }
        },
        parse_quote! {
            fn unsqueeze2(x: Tensor2<f32, 2, 3>) -> Tensor4<f32, 1, 2, 1, 3> {
                unsqueeze::<Tensor4<f32, 1, 2, 1, 3>>(x)
            }
        },
        parse_quote! {
            fn concat_axis0(x: Tensor2<f32, 1, 3>, y: Tensor2<f32, 2, 3>) -> Tensor2<f32, 3, 3> {
                concat::<0>(x, y)
            }
        },
        parse_quote! {
            fn stack_axis0(x: Tensor1<f32, 3>, y: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
                stack::<0>(x, y)
            }
        },
        parse_quote! {
            fn slice4d(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor4<f32, 1, 1, 2, 2> {
                slice::<Tensor4<f32, 1, 1, 2, 2>, 0, 1, 0, 1>(x)
            }
        },
        parse_quote! {
            fn take4d_axis3(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor3<f32, 1, 2, 2> {
                take::<3, 1>(x)
            }
        },
        parse_quote! {
            fn concat4d_axis3(
                x: Tensor4<f32, 1, 1, 1, 1>,
                y: Tensor4<f32, 1, 1, 1, 2>,
            ) -> Tensor4<f32, 1, 1, 1, 3> {
                concat::<3>(x, y)
            }
        },
        parse_quote! {
            fn stack3d_axis2(
                x: Tensor3<f32, 1, 2, 3>,
                y: Tensor3<f32, 1, 2, 3>,
            ) -> Tensor4<f32, 1, 2, 2, 3> {
                stack::<2>(x, y)
            }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }
}

#[test]
fn infers_expanded_static_layout_ops() {
    for item in [
        parse_quote! {
            fn transpose_axes(x: Tensor3<f32, 2, 3, 4>) -> Tensor3<f32, 3, 4, 2> {
                transpose::<1, 2, 0>(x)
            }
        },
        parse_quote! {
            fn permute_dims4(x: Tensor4<f32, 1, 2, 3, 4>) -> Tensor4<f32, 3, 1, 4, 2> {
                permute_dims::<2, 0, 3, 1>(x)
            }
        },
        parse_quote! {
            fn swapaxes3(x: Tensor3<i32, 2, 3, 4>) -> Tensor3<i32, 4, 3, 2> {
                swapaxes::<0, 2>(x)
            }
        },
        parse_quote! {
            fn moveaxis4(x: Tensor4<bool, 2, 3, 4, 5>) -> Tensor4<bool, 3, 4, 2, 5> {
                moveaxis::<0, 2>(x)
            }
        },
        parse_quote! {
            fn tile2(x: Tensor2<f64, 1, 2>) -> Tensor2<f64, 2, 6> {
                tile::<2, 3>(x)
            }
        },
        parse_quote! {
            fn repeat2(x: Tensor2<i64, 2, 2>) -> Tensor2<i64, 2, 6> {
                repeat::<1, 3>(x)
            }
        },
        parse_quote! {
            fn pad2(x: Tensor2<f32, 2, 2>) -> Tensor2<f32, 4, 5> {
                pad::<Tensor2<f32, 4, 5>, 1, 2>(x)
            }
        },
        parse_quote! {
            fn flip2(x: Tensor2<bool, 2, 3>) -> Tensor2<bool, 2, 3> {
                flip::<1>(x)
            }
        },
        parse_quote! {
            fn flip_all(x: Tensor3<f32, 1, 2, 3>) -> Tensor3<f32, 1, 2, 3> {
                flip(x)
            }
        },
        parse_quote! {
            fn roll2(x: Tensor2<f32, 2, 4>) -> Tensor2<f32, 2, 4> {
                roll::<1, 1>(x)
            }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }

    let split = parse(parse_quote! {
        fn split_cols(x: Tensor2<f32, 2, 5>) -> (Tensor2<f32, 2, 2>, Tensor2<f32, 2, 3>) {
            let (left, right) = split::<1, 2, 3>(x);
            (left, right)
        }
    })
    .unwrap();
    assert_eq!(
        split.lets[0].value.tys,
        vec![tensor(&[2, 2]), tensor(&[2, 3])]
    );
    assert_eq!(split.outputs, vec![tensor(&[2, 2]), tensor(&[2, 3])]);
}

#[test]
fn parses_higher_rank_tensors_and_infers_inference_ops() {
    let reshape = parse(parse_quote! {
        fn reshape8(x: Tensor1<f32, 8>) -> Tensor3<f32, 2, 2, 2> {
            reshape::<Tensor3<f32, 2, 2, 2>>(x)
        }
    })
    .unwrap();
    assert_eq!(reshape.body[0].ty, tensor(&[2, 2, 2]));

    let batch_mm = parse(parse_quote! {
        fn batch_mm(x: Tensor3<f32, 1, 2, 3>, y: Tensor3<f32, 1, 3, 2>) -> Tensor3<f32, 1, 2, 2> {
            matmul(x, y)
        }
    })
    .unwrap();
    assert_eq!(batch_mm.body[0].ty, tensor(&[1, 2, 2]));

    let rank6_broadcast = parse(parse_quote! {
        fn add_bias6d(
            x: Tensor6<f32, 1, 2, 1, 2, 1, 3>,
            bias: Tensor1<f32, 3>,
        ) -> Tensor6<f32, 1, 2, 1, 2, 1, 3> {
            x + bias
        }
    })
    .unwrap();
    assert_eq!(rank6_broadcast.body[0].ty, tensor(&[1, 2, 1, 2, 1, 3]));

    let rank6_sum = parse(parse_quote! {
        fn sum_axis5(x: Tensor6<f32, 1, 2, 1, 2, 1, 3>) -> Tensor5<f32, 1, 2, 1, 2, 1> {
            sum::<5>(x)
        }
    })
    .unwrap();
    assert_eq!(rank6_sum.body[0].ty, tensor(&[1, 2, 1, 2, 1]));

    let rank6_batch_mm = parse(parse_quote! {
        fn batch_mm6d(
            x: Tensor6<f32, 2, 1, 1, 3, 2, 3>,
            y: Tensor5<f32, 1, 3, 3, 3, 2>,
        ) -> Tensor6<f32, 2, 1, 3, 3, 2, 2> {
            matmul(x, y)
        }
    })
    .unwrap();
    assert_eq!(rank6_batch_mm.body[0].ty, tensor(&[2, 1, 3, 3, 2, 2]));

    let conv = parse(parse_quote! {
            fn conv(x: Tensor4<f32, 1, 4, 4, 3>, k: Tensor4<f32, 3, 3, 3, 8>) -> Tensor4<f32, 1, 2, 2, 8> {
                conv2d(x, k)
            }
        })
        .unwrap();
    assert_eq!(conv.body[0].ty, tensor(&[1, 2, 2, 8]));

    let padded_conv = parse(parse_quote! {
            fn conv(x: Tensor4<f32, 1, 3, 3, 3>, k: Tensor4<f32, 2, 2, 3, 8>) -> Tensor4<f32, 1, 2, 2, 8> {
                conv2d::<Pad<1, 1, 1, 1>, Stride<2, 2>, Dilation<1, 1>, Groups<1>>(x, k)
            }
        })
        .unwrap();
    assert_eq!(padded_conv.body[0].ty, tensor(&[1, 2, 2, 8]));

    let grouped_conv = parse(parse_quote! {
            fn conv(x: Tensor4<f32, 1, 3, 3, 4>, k: Tensor4<f32, 2, 2, 2, 8>) -> Tensor4<f32, 1, 2, 2, 8> {
                conv2d::<Groups<2>>(x, k)
            }
        })
        .unwrap();
    assert_eq!(grouped_conv.body[0].ty, tensor(&[1, 2, 2, 8]));
}

#[test]
fn infers_numpy_style_rank_parity_ops_through_rank6() {
    let bool_tensor = |shape: &[usize]| TensorType {
        elem: ElementType::Bool,
        shape: shape.to_vec(),
    };
    let i64_tensor = |shape: &[usize]| TensorType {
        elem: ElementType::I64,
        shape: shape.to_vec(),
    };

    for item in [
        parse_quote! {
            fn unary_rank6(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
                sigmoid(tanh(sqrt(exp(log(abs(x))))))
            }
        },
        parse_quote! {
            fn elementwise_math_rank0(x: Tensor0<f32>) -> Tensor0<f32> {
                rint(round(ceil(floor(tan(cos(sin(exp2(expm1(log2(log10(log1p(square(reciprocal(x))))))))))))))
            }
        },
        parse_quote! {
            fn elementwise_math_rank6(
                x: Tensor6<f32, 1, 1, 1, 1, 2, 3>,
            ) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
                rint(round(ceil(floor(tan(cos(sin(exp2(expm1(log2(log10(log1p(square(reciprocal(x))))))))))))))
            }
        },
        parse_quote! {
            fn binary_rank6(
                x: Tensor6<f32, 1, 1, 1, 1, 2, 3>,
                y: Tensor1<f32, 3>,
            ) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
                clip(pow(maximum(minimum(x + y, x * y), x - y), 2.0), 0.0, 100.0)
            }
        },
        parse_quote! {
            fn predicate_rank6(
                x: Tensor6<f32, 1, 1, 1, 1, 2, 3>,
                y: Tensor1<f32, 3>,
            ) -> Tensor6<bool, 1, 1, 1, 1, 2, 3> {
                logical_or(logical_and(greater_equal(x, y), less(x, 10.0)), isnan(x))
            }
        },
        parse_quote! {
            fn where_rank6(
                x: Tensor6<f32, 1, 1, 1, 1, 2, 3>,
                y: Tensor1<f32, 3>,
            ) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
                r#where(not_equal(x, 0.0), x, y)
            }
        },
        parse_quote! {
            fn broadcast_rank6(x: Tensor1<f32, 3>) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
                broadcast::<Tensor6<f32, 1, 1, 1, 1, 2, 3>>(x)
            }
        },
        parse_quote! {
            fn reshape_rank6(x: Tensor1<f32, 6>) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
                reshape::<Tensor6<f32, 1, 1, 1, 1, 2, 3>>(x)
            }
        },
        parse_quote! {
            fn slice_rank6(x: Tensor6<f32, 1, 1, 1, 1, 2, 4>) -> Tensor6<f32, 1, 1, 1, 1, 1, 2> {
                slice::<Tensor6<f32, 1, 1, 1, 1, 1, 2>, 0, 0, 0, 0, 1, 1>(x)
            }
        },
        parse_quote! {
            fn squeeze_rank6(x: Tensor6<f32, 1, 2, 1, 3, 1, 4>) -> Tensor3<f32, 2, 3, 4> {
                squeeze::<Tensor3<f32, 2, 3, 4>>(x)
            }
        },
        parse_quote! {
            fn unsqueeze_rank6(x: Tensor3<f32, 2, 3, 4>) -> Tensor6<f32, 1, 2, 1, 3, 1, 4> {
                unsqueeze::<Tensor6<f32, 1, 2, 1, 3, 1, 4>>(x)
            }
        },
        parse_quote! {
            fn take_rank6(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<f32, 1, 1, 1, 1, 2> {
                take::<5, 1>(x)
            }
        },
        parse_quote! {
            fn concat_rank6(
                x: Tensor6<f32, 1, 1, 1, 1, 2, 1>,
                y: Tensor6<f32, 1, 1, 1, 1, 2, 2>,
            ) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
                concat::<5>(x, y)
            }
        },
        parse_quote! {
            fn stack_rank6(
                x: Tensor5<f32, 1, 1, 1, 2, 3>,
                y: Tensor5<f32, 1, 1, 1, 2, 3>,
            ) -> Tensor6<f32, 1, 1, 1, 2, 2, 3> {
                stack::<4>(x, y)
            }
        },
        parse_quote! {
            fn transpose_rank6(x: Tensor6<f32, 1, 2, 1, 3, 2, 4>) -> Tensor6<f32, 4, 2, 3, 1, 2, 1> {
                transpose(x)
            }
        },
        parse_quote! {
            fn permute_rank6(x: Tensor6<f32, 1, 2, 1, 3, 2, 4>) -> Tensor6<f32, 1, 3, 2, 2, 4, 1> {
                permute::<Tensor6<f32, 1, 3, 2, 2, 4, 1>, 0, 3, 4, 1, 5, 2>(x)
            }
        },
        parse_quote! {
            fn matmul_rank6(
                x: Tensor6<f32, 1, 2, 1, 3, 2, 3>,
                y: Tensor5<f32, 2, 1, 3, 3, 4>,
            ) -> Tensor6<f32, 1, 2, 1, 3, 2, 4> {
                matmul(x, y)
            }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }

    let sum_axis = parse(parse_quote! {
        fn sum_rank6_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<f32, 1, 1, 1, 1, 2> {
            sum::<5>(x)
        }
    })
    .unwrap();
    assert_eq!(sum_axis.body[0].ty, tensor(&[1, 1, 1, 1, 2]));

    let mean_all = parse(parse_quote! {
        fn mean_rank6(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor0<f32> {
            mean(x)
        }
    })
    .unwrap();
    assert_eq!(mean_all.body[0].ty, tensor(&[]));

    let softmax_axis = parse(parse_quote! {
        fn softmax_rank6_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor6<f32, 1, 1, 1, 1, 2, 3> {
            softmax::<5>(x)
        }
    })
    .unwrap();
    assert_eq!(softmax_axis.body[0].ty, tensor(&[1, 1, 1, 1, 2, 3]));

    let argmax_axis = parse(parse_quote! {
        fn argmax_rank6_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<i64, 1, 1, 1, 1, 2> {
            argmax::<5>(x)
        }
    })
    .unwrap();
    assert_eq!(argmax_axis.body[0].ty, i64_tensor(&[1, 1, 1, 1, 2]));

    let any_axis = parse(parse_quote! {
        fn any_rank6_axis5(x: Tensor6<bool, 1, 1, 1, 1, 2, 3>) -> Tensor5<bool, 1, 1, 1, 1, 2> {
            any::<5>(x)
        }
    })
    .unwrap();
    assert_eq!(any_axis.body[0].ty, bool_tensor(&[1, 1, 1, 1, 2]));

    let all_all = parse(parse_quote! {
        fn all_rank6(x: Tensor6<bool, 1, 1, 1, 1, 2, 3>) -> Tensor0<bool> {
            all(x)
        }
    })
    .unwrap();
    assert_eq!(all_all.body[0].ty, bool_tensor(&[]));

    for item in [
        parse_quote! {
            fn prod_rank6_axis5(x: Tensor6<i32, 1, 1, 1, 1, 2, 3>) -> Tensor5<i32, 1, 1, 1, 1, 2> {
                prod::<5>(x)
            }
        },
        parse_quote! {
            fn max_rank6_axis5(x: Tensor6<bool, 1, 1, 1, 1, 2, 3>) -> Tensor5<bool, 1, 1, 1, 1, 2> {
                max::<5>(x)
            }
        },
        parse_quote! {
            fn min_rank6_axis5(x: Tensor6<bool, 1, 1, 1, 1, 2, 3>) -> Tensor5<bool, 1, 1, 1, 1, 2> {
                min::<5>(x)
            }
        },
        parse_quote! {
            fn argmin_rank6_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<i64, 1, 1, 1, 1, 2> {
                argmin::<5>(x)
            }
        },
        parse_quote! {
            fn var_rank6_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<f32, 1, 1, 1, 1, 2> {
                var::<5>(x)
            }
        },
        parse_quote! {
            fn std_rank6_axis5(x: Tensor6<f32, 1, 1, 1, 1, 2, 3>) -> Tensor5<f32, 1, 1, 1, 1, 2> {
                std::<5>(x)
            }
        },
        parse_quote! {
            fn ptp_rank6_axis5(x: Tensor6<i64, 1, 1, 1, 1, 2, 3>) -> Tensor5<i64, 1, 1, 1, 1, 2> {
                ptp::<5>(x)
            }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }
}

#[test]
fn rejects_invalid_grouped_conv2d_channels() {
    let error = parse(parse_quote! {
        fn conv(x: Tensor4<f32, 1, 3, 3, 4>, k: Tensor4<f32, 2, 2, 4, 8>) -> Tensor4<f32, 1, 2, 2, 8> {
            conv2d::<Groups<2>>(x, k)
        }
    })
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("kernel input channels must equal input channels / groups"));

    let error = parse(parse_quote! {
        fn conv(x: Tensor4<f32, 1, 3, 3, 4>, k: Tensor4<f32, 2, 2, 2, 7>) -> Tensor4<f32, 1, 2, 2, 7> {
            conv2d::<Groups<2>>(x, k)
        }
    })
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("output channels must be divisible by groups"));
}

#[test]
fn infers_scalar_classifier_op_shapes() {
    for item in [
        parse_quote! {
            fn abs4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { abs(x) }
        },
        parse_quote! {
            fn exp4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { exp(x) }
        },
        parse_quote! {
            fn log4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { log(x) }
        },
        parse_quote! {
            fn sqrt4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { sqrt(x) }
        },
        parse_quote! {
            fn tanh4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { tanh(x) }
        },
        parse_quote! {
            fn sigmoid4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { sigmoid(x) }
        },
        parse_quote! {
            fn softmax4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> { softmax(x) }
        },
        parse_quote! {
            fn softmax_axis1(x: Tensor2<f32, 2, 3>) -> Tensor2<f32, 2, 3> { softmax::<1>(x) }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }

    let mean = parse(parse_quote! {
        fn mean4(x: Tensor1<f32, 4>) -> Tensor0<f32> {
            mean(x)
        }
    })
    .unwrap();
    assert_eq!(mean.body[0].ty, tensor(&[]));

    let argmax = parse(parse_quote! {
        fn argmax4(x: Tensor1<f32, 4>) -> Tensor0<i64> {
            argmax(x)
        }
    })
    .unwrap();
    assert_eq!(
        argmax.body[0].ty,
        TensorType {
            elem: ElementType::I64,
            shape: vec![]
        }
    );

    let integer_argmax = parse(parse_quote! {
        fn argmax4_i32(x: Tensor1<i32, 4>) -> Tensor0<i64> {
            argmax(x)
        }
    })
    .unwrap();
    assert_eq!(integer_argmax.body[0].ty, argmax.body[0].ty);

    let matrix_argmax = parse(parse_quote! {
        fn argmax2x3(x: Tensor2<f32, 2, 3>) -> Tensor0<i64> {
            argmax(x)
        }
    })
    .unwrap();
    assert_eq!(
        matrix_argmax.body[0].ty,
        TensorType {
            elem: ElementType::I64,
            shape: vec![]
        }
    );

    let axis_argmax = parse(parse_quote! {
        fn argmax2x3_axis1(x: Tensor2<f32, 2, 3>) -> Tensor1<i64, 2> {
            argmax::<1>(x)
        }
    })
    .unwrap();
    assert_eq!(
        axis_argmax.body[0].ty,
        TensorType {
            elem: ElementType::I64,
            shape: vec![2]
        }
    );
}

#[test]
fn rejects_empty_non_identity_reductions() {
    let empty_tensor = parse(parse_quote! {
        fn argmax_empty(x: Tensor1<f32, 0>) -> Tensor0<i64> {
            argmax(x)
        }
    })
    .unwrap_err();
    assert!(
        empty_tensor
            .to_string()
            .contains("argmax cannot reduce empty tensor shape [0]"),
        "{empty_tensor}"
    );

    let empty_axis = parse(parse_quote! {
        fn argmax_empty_axis(x: Tensor2<f32, 2, 0>) -> Tensor1<i64, 2> {
            argmax::<1>(x)
        }
    })
    .unwrap_err();
    assert!(
        empty_axis
            .to_string()
            .contains("argmax cannot reduce empty axis 1 for tensor shape [2, 0]"),
        "{empty_axis}"
    );

    let empty_mean = parse(parse_quote! {
        fn mean_empty_axis(x: Tensor2<f32, 2, 0>) -> Tensor1<f32, 2> {
            mean::<1>(x)
        }
    })
    .unwrap_err();
    assert!(
        empty_mean
            .to_string()
            .contains("mean cannot reduce empty axis 1 for tensor shape [2, 0]"),
        "{empty_mean}"
    );

    let empty_argmin = parse(parse_quote! {
        fn argmin_empty_axis(x: Tensor2<f32, 2, 0>) -> Tensor1<i64, 2> {
            argmin::<1>(x)
        }
    })
    .unwrap_err();
    assert!(
        empty_argmin
            .to_string()
            .contains("argmin cannot reduce empty axis 1 for tensor shape [2, 0]"),
        "{empty_argmin}"
    );

    let empty_max = parse(parse_quote! {
        fn max_empty(x: Tensor1<f32, 0>) -> Tensor0<f32> {
            max(x)
        }
    })
    .unwrap_err();
    assert!(
        empty_max
            .to_string()
            .contains("max cannot reduce empty tensor shape [0]"),
        "{empty_max}"
    );

    let empty_var = parse(parse_quote! {
        fn var_empty_axis(x: Tensor2<f32, 2, 0>) -> Tensor1<f32, 2> {
            var::<1>(x)
        }
    })
    .unwrap_err();
    assert!(
        empty_var
            .to_string()
            .contains("var cannot reduce empty axis 1 for tensor shape [2, 0]"),
        "{empty_var}"
    );

    let empty_softmax = parse(parse_quote! {
        fn softmax_empty(x: Tensor1<f32, 0>) -> Tensor1<f32, 0> {
            softmax(x)
        }
    })
    .unwrap_err();
    assert!(
        empty_softmax
            .to_string()
            .contains("softmax cannot reduce empty tensor shape [0]"),
        "{empty_softmax}"
    );

    let sum_empty_axis = parse(parse_quote! {
        fn sum_empty_axis(x: Tensor2<f32, 2, 0>) -> Tensor1<f32, 2> {
            sum::<1>(x)
        }
    })
    .unwrap();
    assert_eq!(sum_empty_axis.body[0].ty, tensor(&[2]));

    let prod_empty_axis = parse(parse_quote! {
        fn prod_empty_axis(x: Tensor2<f32, 2, 0>) -> Tensor1<f32, 2> {
            prod::<1>(x)
        }
    })
    .unwrap();
    assert_eq!(prod_empty_axis.body[0].ty, tensor(&[2]));

    let all_empty_axis = parse(parse_quote! {
        fn all_empty_axis(x: Tensor2<bool, 2, 0>) -> Tensor1<bool, 2> {
            all::<1>(x)
        }
    })
    .unwrap();
    assert_eq!(
        all_empty_axis.body[0].ty,
        TensorType {
            elem: ElementType::Bool,
            shape: vec![2]
        }
    );

    let any_empty_axis = parse(parse_quote! {
        fn any_empty_axis(x: Tensor2<bool, 2, 0>) -> Tensor1<bool, 2> {
            any::<1>(x)
        }
    })
    .unwrap();
    assert_eq!(any_empty_axis.body[0].ty, all_empty_axis.body[0].ty);
}

#[test]
fn infers_elementwise_call_shapes() {
    for item in [
        parse_quote! {
            fn minimum4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                minimum(x, y)
            }
        },
        parse_quote! {
            fn maximum_broadcast(x: Tensor2<f32, 2, 3>, y: Tensor1<f32, 3>) -> Tensor2<f32, 2, 3> {
                maximum(x, y)
            }
        },
        parse_quote! {
            fn add_channel_bias4d(
                x: Tensor4<f32, 1, 2, 2, 3>,
                bias: Tensor1<f32, 3>,
            ) -> Tensor4<f32, 1, 2, 2, 3> {
                x + bias
            }
        },
        parse_quote! {
            fn scalar_compare4d(x: Tensor4<f32, 1, 2, 2, 3>) -> Tensor4<bool, 1, 2, 2, 3> {
                greater(x, 0.0)
            }
        },
        parse_quote! {
            fn clip4(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                clip(x, 0.0, 1.0)
            }
        },
        parse_quote! {
            fn pow4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                pow(x, y)
            }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }
}

#[test]
fn infers_bool_predicate_selection_and_reduction_shapes() {
    let bool_tensor = |shape: &[usize]| TensorType {
        elem: ElementType::Bool,
        shape: shape.to_vec(),
    };

    for item in [
        parse_quote! {
            fn greater4(x: Tensor1<f32, 4>, y: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
                greater(x, y)
            }
        },
        parse_quote! {
            fn equal_bool4(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
                equal(x, y)
            }
        },
        parse_quote! {
            fn logical4(x: Tensor1<bool, 4>, y: Tensor1<bool, 4>) -> Tensor1<bool, 4> {
                logical_xor(logical_and(x, y), logical_not(y))
            }
        },
        parse_quote! {
            fn any_axis1(x: Tensor2<bool, 2, 3>) -> Tensor1<bool, 2> {
                any::<1>(x)
            }
        },
        parse_quote! {
            fn all4(x: Tensor1<bool, 4>) -> Tensor0<bool> {
                all(x)
            }
        },
        parse_quote! {
            fn isnan4(x: Tensor1<f32, 4>) -> Tensor1<bool, 4> {
                isnan(x)
            }
        },
    ] {
        let graph = parse(item).unwrap();
        assert_eq!(graph.body[0].ty, graph.outputs[0]);
    }

    let selected = parse(parse_quote! {
        fn select4(
            c: Tensor1<bool, 4>,
            x: Tensor1<f32, 4>,
            y: Tensor1<f32, 1>,
        ) -> Tensor1<f32, 4> {
            r#where(c, x, y)
        }
    })
    .unwrap();
    assert_eq!(selected.body[0].ty, tensor(&[4]));

    let selected_from_predicate = parse(parse_quote! {
        fn select_from_predicate(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
            r#where(greater(x, 0.0), 1.0, 0.0)
        }
    })
    .unwrap();
    assert_eq!(selected_from_predicate.body[0].ty, tensor(&[4]));

    let comparison = parse(parse_quote! {
        fn less_broadcast(x: Tensor2<i32, 2, 3>, y: Tensor1<i32, 3>) -> Tensor2<bool, 2, 3> {
            less_equal(x, y)
        }
    })
    .unwrap();
    assert_eq!(comparison.body[0].ty, bool_tensor(&[2, 3]));
}

#[test]
fn accepts_calls_to_earlier_graph_signatures() {
    let signatures = [(
        "layer".to_string(),
        GraphSignature {
            inputs: vec![tensor(&[4])],
            outputs: vec![tensor(&[4])],
        },
    )];

    let graph = parse_graph_with_signatures(
        quote!(backend = Backend::LlvmCpu),
        parse_quote! {
            fn outer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                layer(x)
            }
        },
        &signatures,
    )
    .unwrap();

    assert_eq!(graph.body[0].ty, tensor(&[4]));
}

#[test]
fn rejects_elementwise_shape_mismatch() {
    let error = parse(parse_quote! {
        fn add(x: Tensor1<f32, 4>, y: Tensor1<f32, 5>) -> Tensor1<f32, 4> {
            x + y
        }
    })
    .unwrap_err();

    assert!(error.to_string().contains("not broadcast-compatible"));
}

#[test]
fn rejects_invalid_reshape_and_broadcast_shapes() {
    let reshape = parse(parse_quote! {
        fn bad_reshape(x: Tensor1<f32, 4>) -> Tensor2<f32, 3, 2> {
            reshape::<Tensor2<f32, 3, 2>>(x)
        }
    })
    .unwrap_err();
    assert!(reshape.to_string().contains("element counts must match"));

    let broadcast = parse(parse_quote! {
        fn bad_broadcast(x: Tensor1<f32, 2>) -> Tensor1<f32, 4> {
            broadcast::<Tensor1<f32, 4>>(x)
        }
    })
    .unwrap_err();
    assert!(broadcast.to_string().contains("incompatible"));

    let slice = parse(parse_quote! {
        fn bad_slice(x: Tensor2<f32, 2, 4>) -> Tensor2<f32, 2, 3> {
            slice::<Tensor2<f32, 2, 3>, 0, 2>(x)
        }
    })
    .unwrap_err();
    assert!(slice.to_string().contains("out of bounds"));

    let take = parse(parse_quote! {
        fn bad_take(x: Tensor2<f32, 2, 4>) -> Tensor1<f32, 2> {
            take::<1, 4>(x)
        }
    })
    .unwrap_err();
    assert!(take.to_string().contains("out of bounds"));

    let permute = parse(parse_quote! {
        fn bad_permute(x: Tensor3<f32, 2, 3, 4>) -> Tensor3<f32, 2, 3, 4> {
            permute_dims::<0, 0, 2>(x)
        }
    })
    .unwrap_err();
    assert!(permute.to_string().contains("permutation"));

    let moveaxis = parse(parse_quote! {
        fn bad_moveaxis(x: Tensor3<f32, 2, 3, 4>) -> Tensor3<f32, 2, 3, 4> {
            moveaxis::<0, 3>(x)
        }
    })
    .unwrap_err();
    assert!(moveaxis
        .to_string()
        .contains("moveaxis source 0 and destination 3"));

    let split = parse(parse_quote! {
        fn bad_split(x: Tensor2<f32, 2, 4>) -> (Tensor2<f32, 2, 1>, Tensor2<f32, 2, 2>) {
            let (a, b) = split::<1, 1, 2>(x);
            (a, b)
        }
    })
    .unwrap_err();
    assert!(split.to_string().contains("split sections"));

    let pad = parse(parse_quote! {
        fn bad_pad(x: Tensor2<f32, 2, 2>) -> Tensor2<f32, 2, 3> {
            pad::<Tensor2<f32, 2, 3>, 1, 0>(x)
        }
    })
    .unwrap_err();
    assert!(pad.to_string().contains("pad dimension 0 is out of bounds"));

    let repeat = parse(parse_quote! {
        fn bad_repeat(x: Tensor2<f32, 2, 2>) -> Tensor2<f32, 2, 2> {
            repeat::<2, 2>(x)
        }
    })
    .unwrap_err();
    assert!(repeat
        .to_string()
        .contains("repeat axis 2 is out of bounds"));

    let gather_dtype = parse(parse_quote! {
        fn bad_gather_dtype(
            x: Tensor2<f32, 2, 4>,
            indices: Tensor1<f32, 2>,
        ) -> Tensor2<f32, 2, 4> {
            gather::<Tensor2<f32, 2, 4>, 0>(x, indices)
        }
    })
    .unwrap_err();
    assert!(gather_dtype.to_string().contains("must be i32 or i64"));

    let gather_shape = parse(parse_quote! {
        fn bad_gather_shape(
            x: Tensor2<f32, 2, 4>,
            indices: Tensor1<i64, 2>,
        ) -> Tensor2<f32, 3, 2> {
            gather::<Tensor2<f32, 3, 2>, 1>(x, indices)
        }
    })
    .unwrap_err();
    assert!(gather_shape.to_string().contains("gather output shape"));

    let take_along_rank = parse(parse_quote! {
        fn bad_take_along_rank(
            x: Tensor2<f32, 2, 4>,
            indices: Tensor1<i64, 2>,
        ) -> Tensor1<f32, 2> {
            take_along_axis::<1>(x, indices)
        }
    })
    .unwrap_err();
    assert!(take_along_rank.to_string().contains("equal rank"));

    let take_along_shape = parse(parse_quote! {
        fn bad_take_along_shape(
            x: Tensor2<f32, 2, 4>,
            indices: Tensor2<i64, 3, 2>,
        ) -> Tensor2<f32, 3, 2> {
            take_along_axis::<1>(x, indices)
        }
    })
    .unwrap_err();
    assert!(take_along_shape
        .to_string()
        .contains("must match outside axis"));
}

#[test]
fn rejects_unknown_graph_calls() {
    let error = parse(parse_quote! {
        fn outer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
            missing(x)
        }
    })
    .unwrap_err();

    assert!(error.to_string().contains("unknown graph call `missing`"));
}

#[test]
fn rejects_direct_recursion() {
    let signatures = [(
        "self_ref".to_string(),
        GraphSignature {
            inputs: vec![tensor(&[4])],
            outputs: vec![tensor(&[4])],
        },
    )];
    let error = parse_graph_with_signatures(
        quote!(backend = Backend::LlvmCpu),
        parse_quote! {
            fn self_ref(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                self_ref(x)
            }
        },
        &signatures,
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("recursive graph call `self_ref`"));
}
