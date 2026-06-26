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
    parse_graph(quote!(backend = "llvm-cpu"), item)
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
        quote!(backend = "llvm-cpu"),
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
        quote!(backend = "llvm-cpu"),
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
}

#[test]
fn rejects_unsupported_grouped_conv2d() {
    let error = parse(parse_quote! {
        fn conv(x: Tensor4<f32, 1, 3, 3, 4>, k: Tensor4<f32, 2, 2, 4, 8>) -> Tensor4<f32, 1, 2, 2, 8> {
            conv2d::<Groups<2>>(x, k)
        }
    })
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("grouped conv2d is not supported yet"));
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
        quote!(backend = "llvm-cpu"),
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
        "outer".to_string(),
        GraphSignature {
            inputs: vec![tensor(&[4])],
            outputs: vec![tensor(&[4])],
        },
    )];
    let error = parse_graph_with_signatures(
        quote!(backend = "llvm-cpu"),
        parse_quote! {
            fn outer(x: Tensor1<f32, 4>) -> Tensor1<f32, 4> {
                outer(x)
            }
        },
        &signatures,
    )
    .unwrap_err();

    assert!(error.to_string().contains("recursive graph call `outer`"));
}
