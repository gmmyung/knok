use super::*;

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
fn value_and_grad_inlines_binary_graph_call_operands_from_registry_scope() {
    let layer = graph(
        "add_layer",
        vec![
            input("x", ElementType::F32, &[2]),
            input("y", ElementType::F32, &[2]),
        ],
        vec![tensor(ElementType::F32, &[2])],
        Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(var("x")),
            rhs: Box::new(var("y")),
        },
    );
    let typed_layer = type_check(layer, &[]).unwrap();
    let graph = graph(
        "graph_call_loss",
        vec![
            input("lhs", ElementType::F32, &[2]),
            input("rhs", ElementType::F32, &[2]),
        ],
        vec![tensor(ElementType::F32, &[])],
        Expr::Call {
            op: CallOp::Sum(AxisSpec::All),
            args: vec![Expr::Call {
                op: CallOp::Graph("add_layer".into()),
                args: vec![var("lhs"), var("rhs")],
            }],
        },
    );
    let signatures = vec![(
        "add_layer".into(),
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
    let registry = BTreeMap::from([("add_layer".into(), typed_layer)]);

    let grad = value_and_grad_graph_with_registry(&typed, &registry).unwrap();

    assert_eq!(
        grad.outputs,
        vec![
            tensor(ElementType::F32, &[]),
            tensor(ElementType::F32, &[2]),
            tensor(ElementType::F32, &[2]),
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
