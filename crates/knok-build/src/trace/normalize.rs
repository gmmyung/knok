use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use knok_core::{Expr, Let};

use crate::Result;

pub(super) fn let_bind_nodes(
    body: Vec<Expr>,
    reserved_names: impl IntoIterator<Item = String>,
) -> Result<(Vec<Let>, Vec<Expr>)> {
    let mut values = BTreeMap::new();
    let mut order = Vec::new();
    for expr in &body {
        collect_nodes(expr, &mut values, &mut order)?;
    }

    let mut reserved_names = reserved_names.into_iter().collect::<BTreeSet<_>>();
    let mut names = BTreeMap::new();
    for node_id in &order {
        let name = unique_node_name(*node_id, &mut reserved_names);
        names.insert(*node_id, name);
    }

    let lets = order
        .into_iter()
        .map(|node_id| {
            let name = names
                .get(&node_id)
                .ok_or_else(|| anyhow::anyhow!("node {node_id} was not assigned a binding name"))?
                .clone();
            let value = values
                .get(&node_id)
                .ok_or_else(|| anyhow::anyhow!("node {node_id} was not assigned a value"))?
                .as_ref()
                .clone();
            Ok(Let {
                names: vec![name],
                value: rewrite_node_refs(value, &names)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let body = body
        .into_iter()
        .map(|expr| rewrite_node_refs(expr, &names))
        .collect::<Result<Vec<_>>>()?;
    Ok((lets, body))
}

fn collect_nodes(
    expr: &Expr,
    values: &mut BTreeMap<u64, Arc<Expr>>,
    order: &mut Vec<u64>,
) -> Result<()> {
    match expr {
        Expr::Unary { value, .. } => collect_nodes(value, values, order),
        Expr::Binary { lhs, rhs, .. } => {
            collect_nodes(lhs, values, order)?;
            collect_nodes(rhs, values, order)
        }
        Expr::Node { node_id, value } => {
            if let Some(existing) = values.get(node_id) {
                if existing.as_ref() != value.as_ref() {
                    anyhow::bail!("node id {node_id} is used for multiple expression payloads");
                }
                return Ok(());
            }
            collect_nodes(value, values, order)?;
            values.insert(*node_id, value.clone());
            order.push(*node_id);
            Ok(())
        }
        Expr::TupleGet { value, .. } => collect_nodes(value, values, order),
        Expr::Call { args, .. } => {
            for arg in args {
                collect_nodes(arg, values, order)?;
            }
            Ok(())
        }
        Expr::Var(_) | Expr::Const { .. } => Ok(()),
    }
}

fn rewrite_node_refs(expr: Expr, names: &BTreeMap<u64, String>) -> Result<Expr> {
    Ok(match expr {
        Expr::Unary { op, value } => Expr::Unary {
            op,
            value: Box::new(rewrite_node_refs(*value, names)?),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(rewrite_node_refs(*lhs, names)?),
            rhs: Box::new(rewrite_node_refs(*rhs, names)?),
        },
        Expr::Node { node_id, .. } => Expr::Var(
            names
                .get(&node_id)
                .ok_or_else(|| anyhow::anyhow!("node {node_id} has no generated binding name"))?
                .clone(),
        ),
        Expr::TupleGet {
            tuple_id,
            value,
            index,
        } => Expr::TupleGet {
            tuple_id,
            value: Arc::new(rewrite_node_refs(value.as_ref().clone(), names)?),
            index,
        },
        Expr::Call { op, args } => Expr::Call {
            op,
            args: args
                .into_iter()
                .map(|arg| rewrite_node_refs(arg, names))
                .collect::<Result<Vec<_>>>()?,
        },
        Expr::Var(_) | Expr::Const { .. } => expr,
    })
}

fn unique_node_name(node_id: u64, reserved_names: &mut BTreeSet<String>) -> String {
    let base = node_name(node_id);
    let mut candidate = base.clone();
    let mut suffix = 1;
    while reserved_names.contains(&candidate) {
        candidate = format!("{base}_{suffix}");
        suffix += 1;
    }
    reserved_names.insert(candidate.clone());
    candidate
}

fn node_name(node_id: u64) -> String {
    format!("__knok_node_{node_id}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use knok_core::{BinaryOp, ElementType};

    #[test]
    fn generated_node_bindings_do_not_shadow_reserved_names() {
        let input_name = "__knok_node_7".to_string();
        let body = vec![Expr::Node {
            node_id: 7,
            value: Arc::new(Expr::Var(input_name.clone())),
        }];

        let (lets, body) = let_bind_nodes(body, [input_name.clone()]).unwrap();

        assert_eq!(lets[0].names, vec!["__knok_node_7_1"]);
        assert_eq!(lets[0].value, Expr::Var(input_name));
        assert_eq!(body, vec![Expr::Var("__knok_node_7_1".into())]);
    }

    #[test]
    fn reused_node_payloads_share_one_binding() {
        let node = Expr::Node {
            node_id: 4,
            value: Arc::new(Expr::Const {
                value: "1.0".into(),
                elem: ElementType::F32,
            }),
        };
        let body = vec![Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(node.clone()),
            rhs: Box::new(node),
        }];

        let (lets, body) = let_bind_nodes(body, []).unwrap();

        assert_eq!(lets.len(), 1);
        assert_eq!(lets[0].names, vec!["__knok_node_4"]);
        assert_eq!(
            body,
            vec![Expr::Binary {
                op: BinaryOp::Add,
                lhs: Box::new(Expr::Var("__knok_node_4".into())),
                rhs: Box::new(Expr::Var("__knok_node_4".into())),
            }]
        );
    }

    #[test]
    fn conflicting_node_payloads_are_rejected() {
        let body = vec![
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
        ];

        let error = let_bind_nodes(body, []).unwrap_err();

        assert!(error
            .to_string()
            .contains("node id 1 is used for multiple expression payloads"));
    }
}
