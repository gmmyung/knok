use crate::{
    type_check, AxisSpec, BinaryOp, CallOp, ElementType, Expr, Graph, GraphSignature, Input, Let,
    TensorType,
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
