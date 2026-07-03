use std::{collections::BTreeMap, sync::Arc};

use crate::{
    parse_tensor_type, type_check, value_and_grad_graph, value_and_grad_graph_with_registry,
    AxisSpec, BinaryOp, CallOp, Conv2dOptions, ElementType, Expr, Graph, GraphSignature, Input,
    Let, Pool2dOptions, TensorType, TypedGraph,
};

fn tensor(elem: ElementType, shape: &[usize]) -> TensorType {
    TensorType {
        elem,
        shape: shape.to_vec(),
    }
}

fn input(name: &str, elem: ElementType, shape: &[usize]) -> Input {
    Input {
        name: name.into(),
        ty: tensor(elem, shape),
    }
}

fn var(name: &str) -> Expr {
    Expr::Var(name.into())
}

fn graph(name: &str, inputs: Vec<Input>, outputs: Vec<TensorType>, body: Expr) -> Graph {
    Graph {
        name: name.into(),
        backend: "llvm-cpu".into(),
        inputs,
        outputs,
        lets: Vec::new(),
        body: vec![body],
    }
}

fn graph_signature(graph: &TypedGraph) -> GraphSignature {
    GraphSignature {
        inputs: graph.inputs.iter().map(|input| input.ty.clone()).collect(),
        outputs: graph.outputs.clone(),
    }
}

fn expr_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Var(_) | Expr::Const { .. } => 1,
        Expr::Unary { value, .. } => 1 + expr_node_count(value),
        Expr::Binary { lhs, rhs, .. } => 1 + expr_node_count(lhs) + expr_node_count(rhs),
        Expr::Node { value, .. } => 1 + expr_node_count(value),
        Expr::TupleGet { value, .. } => 1 + expr_node_count(value),
        Expr::Call { args, .. } => 1 + args.iter().map(expr_node_count).sum::<usize>(),
    }
}

fn graph_expr_node_count(graph: &TypedGraph) -> usize {
    graph
        .lets
        .iter()
        .map(|binding| expr_node_count(&binding.value.kind))
        .sum::<usize>()
        + graph
            .body
            .iter()
            .map(|expr| expr_node_count(&expr.kind))
            .sum::<usize>()
}

#[test]
fn typechecks_elementwise_broadcast_graph() {
    let graph = graph(
        "add_bias",
        vec![
            input("x", ElementType::F32, &[2, 3]),
            input("bias", ElementType::F32, &[3]),
        ],
        vec![tensor(ElementType::F32, &[2, 3])],
        Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(var("x")),
            rhs: Box::new(var("bias")),
        },
    );

    let typed = type_check(graph, &[]).unwrap();
    assert_eq!(typed.outputs, vec![tensor(ElementType::F32, &[2, 3])]);
}

#[test]
fn typechecks_matmul_and_relu_graph() {
    let graph = graph(
        "dense",
        vec![
            input("x", ElementType::F32, &[2, 3]),
            input("w", ElementType::F32, &[3, 4]),
        ],
        vec![tensor(ElementType::F32, &[2, 4])],
        Expr::Call {
            op: CallOp::Relu,
            args: vec![Expr::Call {
                op: CallOp::Matmul,
                args: vec![var("x"), var("w")],
            }],
        },
    );

    let typed = type_check(graph, &[]).unwrap();
    assert_eq!(typed.outputs, vec![tensor(ElementType::F32, &[2, 4])]);
}

#[test]
fn typechecks_multi_output_let_binding() {
    let graph = Graph {
        name: "split_sum".into(),
        backend: "llvm-cpu".into(),
        inputs: vec![input("x", ElementType::F32, &[4])],
        outputs: vec![tensor(ElementType::F32, &[2])],
        lets: vec![Let {
            names: vec!["a".into(), "b".into()],
            value: Expr::Call {
                op: CallOp::Split {
                    axis: 0,
                    sections: vec![2, 2],
                },
                args: vec![var("x")],
            },
        }],
        body: vec![Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(var("a")),
            rhs: Box::new(var("b")),
        }],
    };

    let typed = type_check(graph, &[]).unwrap();
    assert_eq!(typed.outputs, vec![tensor(ElementType::F32, &[2])]);
}

#[test]
fn typechecks_reused_node_payload() {
    let value = Arc::new(Expr::Const {
        value: "1.0".into(),
        elem: ElementType::F32,
    });
    let node = Expr::Node { node_id: 1, value };
    let graph = graph(
        "reused_node",
        Vec::new(),
        vec![tensor(ElementType::F32, &[])],
        Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(node.clone()),
            rhs: Box::new(node),
        },
    );

    let typed = type_check(graph, &[]).unwrap();
    assert_eq!(typed.outputs, vec![tensor(ElementType::F32, &[])]);
}

#[test]
fn rejects_conflicting_node_payloads() {
    let graph = Graph {
        name: "conflicting_node".into(),
        backend: "llvm-cpu".into(),
        inputs: Vec::new(),
        outputs: vec![tensor(ElementType::F32, &[]), tensor(ElementType::F32, &[])],
        lets: Vec::new(),
        body: vec![
            Expr::Node {
                node_id: 1,
                value: Arc::new(Expr::Const {
                    value: "1.0".into(),
                    elem: ElementType::F32,
                }),
            },
            Expr::Node {
                node_id: 1,
                value: Arc::new(Expr::Const {
                    value: "2.0".into(),
                    elem: ElementType::F32,
                }),
            },
        ],
    };

    let error = type_check(graph, &[]).unwrap_err();

    assert!(error
        .to_string()
        .contains("node id 1 is used for multiple expression payloads"));
}

#[test]
fn typechecks_graph_call_signature() {
    let graph = graph(
        "caller",
        vec![input("x", ElementType::F32, &[4])],
        vec![tensor(ElementType::F32, &[4])],
        Expr::Call {
            op: CallOp::Graph("layer".into()),
            args: vec![var("x")],
        },
    );
    let signatures = vec![(
        "layer".into(),
        GraphSignature {
            inputs: vec![tensor(ElementType::F32, &[4])],
            outputs: vec![tensor(ElementType::F32, &[4])],
        },
    )];

    let typed = type_check(graph, &signatures).unwrap();
    assert_eq!(typed.outputs, vec![tensor(ElementType::F32, &[4])]);
}

#[test]
fn rejects_shape_mismatch() {
    let graph = graph(
        "bad",
        vec![
            input("x", ElementType::F32, &[2, 3]),
            input("y", ElementType::F32, &[4]),
        ],
        vec![tensor(ElementType::F32, &[2, 3])],
        Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(var("x")),
            rhs: Box::new(var("y")),
        },
    );

    let error = type_check(graph, &[]).unwrap_err();
    assert!(error
        .to_string()
        .contains("elementwise operands are not broadcast-compatible"));
}

#[test]
fn typechecks_axis_reduction() {
    let graph = graph(
        "row_sums",
        vec![input("x", ElementType::F32, &[2, 3])],
        vec![tensor(ElementType::F32, &[2])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::One(1)),
            args: vec![var("x")],
        },
    );

    let typed = type_check(graph, &[]).unwrap();
    assert_eq!(typed.outputs, vec![tensor(ElementType::F32, &[2])]);
}

#[test]
fn parses_tensor_type_aliases_and_rejects_bad_generics() {
    let parsed =
        parse_tensor_type(&syn::parse_str("Tensor6<f32, 1, 2, 3, 4, 5, 6>").unwrap()).unwrap();
    assert_eq!(parsed, tensor(ElementType::F32, &[1, 2, 3, 4, 5, 6]));

    let scalar = parse_tensor_type(&syn::parse_str("knok::Tensor0<bool>").unwrap()).unwrap();
    assert_eq!(scalar, tensor(ElementType::Bool, &[]));

    let alias = parse_tensor_type(&syn::parse_str("T2<i64, 3, 4>").unwrap()).unwrap();
    assert_eq!(alias, tensor(ElementType::I64, &[3, 4]));

    let too_many = parse_tensor_type(&syn::parse_str("Tensor1<f32, 4, 5>").unwrap()).unwrap_err();
    assert!(too_many
        .to_string()
        .contains("too many tensor generic arguments"));

    let unsupported = parse_tensor_type(&syn::parse_str("Tensor1<u8, 4>").unwrap()).unwrap_err();
    assert!(unsupported
        .to_string()
        .contains("unsupported tensor element type"));
}

#[test]
fn typechecks_shape_manipulation_ops() {
    let graph = Graph {
        name: "shape_ops".into(),
        backend: "llvm-cpu".into(),
        inputs: vec![
            input("x", ElementType::F32, &[2, 3]),
            input("idx", ElementType::I64, &[2, 2]),
        ],
        outputs: vec![
            tensor(ElementType::F32, &[6]),
            tensor(ElementType::F32, &[3, 2]),
            tensor(ElementType::F32, &[2, 2, 2]),
            tensor(ElementType::F32, &[4, 6]),
            tensor(ElementType::F32, &[2, 2]),
        ],
        lets: Vec::new(),
        body: vec![
            Expr::Call {
                op: CallOp::Reshape(tensor(ElementType::F32, &[6])),
                args: vec![var("x")],
            },
            Expr::Call {
                op: CallOp::Transpose(Vec::new()),
                args: vec![var("x")],
            },
            Expr::Call {
                op: CallOp::Gather {
                    target: tensor(ElementType::F32, &[2, 2, 2]),
                    axis: 1,
                },
                args: vec![var("x"), var("idx")],
            },
            Expr::Call {
                op: CallOp::Tile(vec![2, 2]),
                args: vec![var("x")],
            },
            Expr::Call {
                op: CallOp::Slice {
                    target: tensor(ElementType::F32, &[2, 2]),
                    starts: vec![0, 1],
                },
                args: vec![var("x")],
            },
        ],
    };

    let typed = type_check(graph, &[]).unwrap();
    assert_eq!(
        typed.outputs,
        vec![
            tensor(ElementType::F32, &[6]),
            tensor(ElementType::F32, &[3, 2]),
            tensor(ElementType::F32, &[2, 2, 2]),
            tensor(ElementType::F32, &[4, 6]),
            tensor(ElementType::F32, &[2, 2]),
        ]
    );
}

#[test]
fn typechecks_linalg_creation_and_predicate_ops() {
    let graph = Graph {
        name: "more_ops".into(),
        backend: "llvm-cpu".into(),
        inputs: vec![
            input("a", ElementType::F32, &[3]),
            input("b", ElementType::F32, &[3]),
            input("mask", ElementType::Bool, &[3]),
            input("image", ElementType::F32, &[1, 4, 4, 2]),
            input("kernel", ElementType::F32, &[3, 3, 2, 1]),
        ],
        outputs: vec![
            tensor(ElementType::Bool, &[3]),
            tensor(ElementType::F32, &[3]),
            tensor(ElementType::F32, &[3]),
            tensor(ElementType::F32, &[4]),
            tensor(ElementType::F32, &[3]),
            tensor(ElementType::F32, &[1, 2, 2, 1]),
            tensor(ElementType::F32, &[1, 2, 2, 2]),
            tensor(ElementType::F32, &[1, 2, 2, 2]),
        ],
        lets: Vec::new(),
        body: vec![
            Expr::Call {
                op: CallOp::LogicalNot,
                args: vec![var("mask")],
            },
            Expr::Call {
                op: CallOp::Where,
                args: vec![var("mask"), var("a"), var("b")],
            },
            Expr::Call {
                op: CallOp::Maximum,
                args: vec![var("a"), var("b")],
            },
            Expr::Call {
                op: CallOp::Arange(tensor(ElementType::F32, &[4])),
                args: vec![Expr::Const {
                    value: "4".into(),
                    elem: ElementType::F32,
                }],
            },
            Expr::Call {
                op: CallOp::Linspace(tensor(ElementType::F32, &[3])),
                args: vec![
                    Expr::Const {
                        value: "0.0".into(),
                        elem: ElementType::F32,
                    },
                    Expr::Const {
                        value: "1.0".into(),
                        elem: ElementType::F32,
                    },
                ],
            },
            Expr::Call {
                op: CallOp::Conv2d(Conv2dOptions::default()),
                args: vec![var("image"), var("kernel")],
            },
            Expr::Call {
                op: CallOp::MaxPool2d(Pool2dOptions::default()),
                args: vec![var("image")],
            },
            Expr::Call {
                op: CallOp::AvgPool2d(Pool2dOptions::default()),
                args: vec![var("image")],
            },
        ],
    };

    let typed = type_check(graph, &[]).unwrap();
    assert_eq!(typed.outputs[0], tensor(ElementType::Bool, &[3]));
    assert_eq!(typed.outputs[5], tensor(ElementType::F32, &[1, 2, 2, 1]));
    assert_eq!(typed.outputs[6], tensor(ElementType::F32, &[1, 2, 2, 2]));
    assert_eq!(typed.outputs[7], tensor(ElementType::F32, &[1, 2, 2, 2]));
}

#[test]
fn rejects_non_static_creation_calls_in_let_bindings() {
    let graph = Graph {
        name: "bad_let_arange".into(),
        backend: "llvm-cpu".into(),
        inputs: vec![input("stop", ElementType::I32, &[])],
        outputs: vec![tensor(ElementType::I32, &[4])],
        lets: vec![Let {
            names: vec!["values".into()],
            value: Expr::Call {
                op: CallOp::Arange(tensor(ElementType::I32, &[4])),
                args: vec![var("stop")],
            },
        }],
        body: vec![var("values")],
    };

    assert!(type_check(graph, &[])
        .unwrap_err()
        .to_string()
        .contains("numeric literals"));
}

#[test]
fn rejects_invalid_static_creation_targets() {
    let arange = graph(
        "bad_arange_target",
        Vec::new(),
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Arange(tensor(ElementType::F32, &[])),
            args: vec![Expr::Const {
                value: "4.0".into(),
                elem: ElementType::F32,
            }],
        },
    );
    assert!(type_check(arange, &[])
        .unwrap_err()
        .to_string()
        .contains("arange target must be rank-1"));

    let eye = graph(
        "bad_eye_target",
        Vec::new(),
        vec![tensor(ElementType::F32, &[2, 3])],
        Expr::Call {
            op: CallOp::Eye(tensor(ElementType::F32, &[2, 3])),
            args: Vec::new(),
        },
    );
    assert!(type_check(eye, &[])
        .unwrap_err()
        .to_string()
        .contains("eye target matrix must be square"));
}

#[test]
fn rejects_invalid_shape_and_axis_ops() {
    let reshape = graph(
        "bad_reshape",
        vec![input("x", ElementType::F32, &[2, 3])],
        vec![tensor(ElementType::F32, &[5])],
        Expr::Call {
            op: CallOp::Reshape(tensor(ElementType::F32, &[5])),
            args: vec![var("x")],
        },
    );
    assert!(type_check(reshape, &[])
        .unwrap_err()
        .to_string()
        .contains("reshape element counts must match"));

    let split = Graph {
        name: "bad_split".into(),
        backend: "llvm-cpu".into(),
        inputs: vec![input("x", ElementType::F32, &[4])],
        outputs: vec![
            tensor(ElementType::F32, &[2]),
            tensor(ElementType::F32, &[1]),
        ],
        lets: Vec::new(),
        body: vec![Expr::Call {
            op: CallOp::Split {
                axis: 0,
                sections: vec![2, 1],
            },
            args: vec![var("x")],
        }],
    };
    assert!(type_check(split, &[])
        .unwrap_err()
        .to_string()
        .contains("split sections"));

    let softmax = graph(
        "bad_softmax",
        Vec::new(),
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Softmax(AxisSpec::All),
            args: vec![Expr::Const {
                value: "1.0".into(),
                elem: ElementType::F32,
            }],
        },
    );
    assert!(type_check(softmax, &[])
        .unwrap_err()
        .to_string()
        .contains("softmax expects a tensor input"));
}

#[test]
fn builds_value_and_grad_graph_for_scalar_loss() {
    let graph = graph(
        "sum_square",
        vec![input("x", ElementType::F32, &[2, 3])],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::All),
            args: vec![Expr::Call {
                op: CallOp::Square,
                args: vec![var("x")],
            }],
        },
    );
    let typed = type_check(graph, &[]).unwrap();

    let grad = value_and_grad_graph(&typed).unwrap();

    assert_eq!(grad.name, "sum_square_value_and_grad");
    assert_eq!(
        grad.outputs,
        vec![
            tensor(ElementType::F32, &[]),
            tensor(ElementType::F32, &[2, 3])
        ]
    );
}

#[test]
fn value_and_grad_unbroadcasts_input_gradients() {
    let graph = graph(
        "bias_loss",
        vec![
            input("x", ElementType::F32, &[2, 3]),
            input("bias", ElementType::F32, &[3]),
        ],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::All),
            args: vec![Expr::Binary {
                op: BinaryOp::Add,
                lhs: Box::new(var("x")),
                rhs: Box::new(var("bias")),
            }],
        },
    );
    let typed = type_check(graph, &[]).unwrap();

    let grad = value_and_grad_graph(&typed).unwrap();

    assert_eq!(
        grad.outputs,
        vec![
            tensor(ElementType::F32, &[]),
            tensor(ElementType::F32, &[2, 3]),
            tensor(ElementType::F32, &[3]),
        ]
    );
}

#[test]
fn value_and_grad_allows_detached_unsupported_ops() {
    let graph = graph(
        "detached_pool_loss",
        vec![input("x", ElementType::F32, &[1, 4, 4, 1])],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::All),
            args: vec![Expr::Call {
                op: CallOp::Detach,
                args: vec![Expr::Call {
                    op: CallOp::MaxPool2d(Pool2dOptions::default()),
                    args: vec![var("x")],
                }],
            }],
        },
    );
    let typed = type_check(graph, &[]).unwrap();

    let grad = value_and_grad_graph(&typed).unwrap();

    assert_eq!(
        grad.outputs,
        vec![
            tensor(ElementType::F32, &[]),
            tensor(ElementType::F32, &[1, 4, 4, 1]),
        ]
    );
}

#[test]
fn value_and_grad_treats_arg_indices_as_no_grad() {
    let argmax = Expr::Call {
        op: CallOp::Argmax(AxisSpec::One(1)),
        args: vec![var("x")],
    };
    let indices = Expr::Call {
        op: CallOp::Reshape(tensor(ElementType::I64, &[2, 1])),
        args: vec![argmax],
    };
    let graph = graph(
        "argmax_index_loss",
        vec![input("x", ElementType::F32, &[2, 3])],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::All),
            args: vec![Expr::Call {
                op: CallOp::TakeAlongAxis { axis: 1 },
                args: vec![var("x"), indices],
            }],
        },
    );
    let typed = type_check(graph, &[]).unwrap();

    let grad = value_and_grad_graph(&typed).unwrap();

    assert_eq!(
        grad.outputs,
        vec![
            tensor(ElementType::F32, &[]),
            tensor(ElementType::F32, &[2, 3])
        ]
    );
}

#[test]
fn value_and_grad_treats_boolean_conditions_as_no_grad() {
    let positive = Expr::Call {
        op: CallOp::Greater,
        args: vec![
            var("x"),
            Expr::Const {
                value: "0.0".into(),
                elem: ElementType::F32,
            },
        ],
    };
    let row_has_positive = Expr::Call {
        op: CallOp::Any(AxisSpec::One(1)),
        args: vec![positive],
    };
    let condition = Expr::Call {
        op: CallOp::All(AxisSpec::All),
        args: vec![row_has_positive],
    };
    let graph = graph(
        "condition_loss",
        vec![input("x", ElementType::F32, &[2, 3])],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Where,
            args: vec![
                condition,
                Expr::Call {
                    op: CallOp::Sum(AxisSpec::All),
                    args: vec![Expr::Call {
                        op: CallOp::Square,
                        args: vec![var("x")],
                    }],
                },
                Expr::Call {
                    op: CallOp::Sum(AxisSpec::All),
                    args: vec![var("x")],
                },
            ],
        },
    );
    let typed = type_check(graph, &[]).unwrap();

    let grad = value_and_grad_graph(&typed).unwrap();

    assert_eq!(
        grad.outputs,
        vec![
            tensor(ElementType::F32, &[]),
            tensor(ElementType::F32, &[2, 3])
        ]
    );
}

#[test]
fn value_and_grad_rejects_unresolved_graph_calls() {
    let graph = graph(
        "graph_call_loss",
        vec![input("x", ElementType::F32, &[2])],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::All),
            args: vec![Expr::Call {
                op: CallOp::Graph("layer".into()),
                args: vec![var("x")],
            }],
        },
    );
    let signatures = vec![(
        "layer".into(),
        GraphSignature {
            inputs: vec![tensor(ElementType::F32, &[2])],
            outputs: vec![tensor(ElementType::F32, &[2])],
        },
    )];
    let typed = type_check(graph, &signatures).unwrap();

    let error = value_and_grad_graph(&typed).unwrap_err();

    assert!(error
        .to_string()
        .contains("cannot resolve graph call `layer` without a graph registry entry"));
}

#[test]
fn value_and_grad_inlines_graph_calls_from_registry() {
    let layer = graph(
        "layer",
        vec![input("x", ElementType::F32, &[2])],
        vec![tensor(ElementType::F32, &[2])],
        Expr::Call {
            op: CallOp::Square,
            args: vec![var("x")],
        },
    );
    let typed_layer = type_check(layer, &[]).unwrap();
    let graph = graph(
        "graph_call_loss",
        vec![input("x", ElementType::F32, &[2])],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::All),
            args: vec![Expr::Call {
                op: CallOp::Graph("layer".into()),
                args: vec![var("x")],
            }],
        },
    );
    let signatures = vec![(
        "layer".into(),
        GraphSignature {
            inputs: typed_layer
                .inputs
                .iter()
                .map(|input| input.ty.clone())
                .collect(),
            outputs: typed_layer.outputs.clone(),
        },
    )];
    let typed = type_check(graph, &signatures).unwrap();
    let registry = BTreeMap::from([("layer".into(), typed_layer)]);

    let grad = value_and_grad_graph_with_registry(&typed, &registry).unwrap();

    assert_eq!(
        grad.outputs,
        vec![
            tensor(ElementType::F32, &[]),
            tensor(ElementType::F32, &[2])
        ]
    );
}

#[test]
fn value_and_grad_inlines_nested_graph_calls_from_registry() {
    let square = type_check(
        graph(
            "square_layer",
            vec![input("x", ElementType::F32, &[2])],
            vec![tensor(ElementType::F32, &[2])],
            Expr::Call {
                op: CallOp::Square,
                args: vec![var("x")],
            },
        ),
        &[],
    )
    .unwrap();
    let relu_square = type_check(
        graph(
            "relu_square_layer",
            vec![input("x", ElementType::F32, &[2])],
            vec![tensor(ElementType::F32, &[2])],
            Expr::Call {
                op: CallOp::Relu,
                args: vec![Expr::Call {
                    op: CallOp::Graph("square_layer".into()),
                    args: vec![var("x")],
                }],
            },
        ),
        &[("square_layer".into(), graph_signature(&square))],
    )
    .unwrap();
    let graph = graph(
        "nested_graph_call_loss",
        vec![input("x", ElementType::F32, &[2])],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::All),
            args: vec![Expr::Call {
                op: CallOp::Graph("relu_square_layer".into()),
                args: vec![var("x")],
            }],
        },
    );
    let signatures = vec![
        ("square_layer".into(), graph_signature(&square)),
        ("relu_square_layer".into(), graph_signature(&relu_square)),
    ];
    let typed = type_check(graph, &signatures).unwrap();
    let registry = BTreeMap::from([
        ("square_layer".into(), square),
        ("relu_square_layer".into(), relu_square),
    ]);

    let grad = value_and_grad_graph_with_registry(&typed, &registry).unwrap();

    assert_eq!(
        grad.outputs,
        vec![
            tensor(ElementType::F32, &[]),
            tensor(ElementType::F32, &[2])
        ]
    );
}

#[test]
fn value_and_grad_rejects_indirect_recursive_graph_calls() {
    let signature = GraphSignature {
        inputs: vec![tensor(ElementType::F32, &[2])],
        outputs: vec![tensor(ElementType::F32, &[2])],
    };
    let graph_a = type_check(
        graph(
            "a",
            vec![input("x", ElementType::F32, &[2])],
            vec![tensor(ElementType::F32, &[2])],
            Expr::Call {
                op: CallOp::Graph("b".into()),
                args: vec![var("x")],
            },
        ),
        &[("b".into(), signature.clone())],
    )
    .unwrap();
    let graph_b = type_check(
        graph(
            "b",
            vec![input("x", ElementType::F32, &[2])],
            vec![tensor(ElementType::F32, &[2])],
            Expr::Call {
                op: CallOp::Graph("a".into()),
                args: vec![var("x")],
            },
        ),
        &[("a".into(), signature)],
    )
    .unwrap();
    let loss = type_check(
        graph(
            "recursive_graph_call_loss",
            vec![input("x", ElementType::F32, &[2])],
            vec![tensor(ElementType::F32, &[])],
            Expr::Call {
                op: CallOp::Sum(AxisSpec::All),
                args: vec![Expr::Call {
                    op: CallOp::Graph("a".into()),
                    args: vec![var("x")],
                }],
            },
        ),
        &[("a".into(), graph_signature(&graph_a))],
    )
    .unwrap();
    let registry = BTreeMap::from([("a".into(), graph_a), ("b".into(), graph_b)]);

    let error = value_and_grad_graph_with_registry(&loss, &registry).unwrap_err();

    assert!(error
        .to_string()
        .contains("recursive graph call `a` is not supported by autograd"));
}

#[test]
fn value_and_grad_static_loop_graphs_have_bounded_expression_size() {
    let conv = type_check(
        graph(
            "conv_loss",
            vec![
                input("x", ElementType::F32, &[1, 4, 4, 1]),
                input("k", ElementType::F32, &[2, 2, 1, 1]),
            ],
            vec![tensor(ElementType::F32, &[])],
            Expr::Call {
                op: CallOp::Sum(AxisSpec::All),
                args: vec![Expr::Call {
                    op: CallOp::Conv2d(Conv2dOptions::default()),
                    args: vec![var("x"), var("k")],
                }],
            },
        ),
        &[],
    )
    .unwrap();
    let conv_grad = value_and_grad_graph(&conv).unwrap();
    assert!(graph_expr_node_count(&conv_grad) <= 20_000);

    let pool = type_check(
        graph(
            "pool_loss",
            vec![input("x", ElementType::F32, &[1, 6, 6, 1])],
            vec![tensor(ElementType::F32, &[])],
            Expr::Call {
                op: CallOp::Sum(AxisSpec::All),
                args: vec![Expr::Call {
                    op: CallOp::MaxPool2d(Pool2dOptions::default()),
                    args: vec![var("x")],
                }],
            },
        ),
        &[],
    )
    .unwrap();
    let pool_grad = value_and_grad_graph(&pool).unwrap();
    assert!(graph_expr_node_count(&pool_grad) <= 20_000);

    let indexed = type_check(
        graph(
            "indexed_loss",
            vec![
                input("x", ElementType::F32, &[3, 4]),
                input("idx", ElementType::I64, &[2, 3]),
            ],
            vec![tensor(ElementType::F32, &[])],
            Expr::Call {
                op: CallOp::Sum(AxisSpec::All),
                args: vec![Expr::Call {
                    op: CallOp::Gather {
                        target: tensor(ElementType::F32, &[3, 2, 3]),
                        axis: 1,
                    },
                    args: vec![var("x"), var("idx")],
                }],
            },
        ),
        &[],
    )
    .unwrap();
    let indexed_grad = value_and_grad_graph(&indexed).unwrap();
    assert!(graph_expr_node_count(&indexed_grad) <= 8_000);
}
