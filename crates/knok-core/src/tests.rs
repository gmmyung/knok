use crate::{
    parse_tensor_type, type_check, AxisSpec, BinaryOp, CallOp, Conv2dOptions, ElementType, Expr,
    Graph, GraphSignature, Input, Let, Pool2dOptions, TensorType,
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
