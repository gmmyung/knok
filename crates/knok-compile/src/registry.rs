use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

use knok_core::{GraphSignature, TypedGraph};
use proc_macro2::Span;

static GRAPH_REGISTRY: OnceLock<Mutex<BTreeMap<String, TypedGraph>>> = OnceLock::new();

fn graph_registry() -> &'static Mutex<BTreeMap<String, TypedGraph>> {
    GRAPH_REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(crate) fn registered_graphs() -> BTreeMap<String, TypedGraph> {
    graph_registry()
        .lock()
        .expect("knok graph registry lock poisoned")
        .clone()
}

pub(crate) fn registered_signatures() -> Vec<(String, GraphSignature)> {
    registered_graphs()
        .into_iter()
        .map(|(name, graph)| {
            (
                name,
                GraphSignature {
                    inputs: graph.inputs.into_iter().map(|input| input.ty).collect(),
                    outputs: graph.outputs,
                },
            )
        })
        .collect()
}

pub(crate) fn register_graph(graph: TypedGraph) -> syn::Result<()> {
    let mut registry = graph_registry()
        .lock()
        .expect("knok graph registry lock poisoned");
    if let Some(existing) = registry.get(&graph.name) {
        if existing != &graph {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "duplicate #[knok::graph] name `{}`; graph calls currently resolve by function name, so graph names must be unique within a crate",
                    graph.name
                ),
            ));
        }
        return Ok(());
    }
    registry.insert(graph.name.clone(), graph);
    Ok(())
}

#[cfg(test)]
mod tests {
    use knok_core::{ElementType, Input, TensorType};

    use super::*;

    #[test]
    fn register_graph_rejects_conflicting_duplicate_names() {
        graph_registry()
            .lock()
            .expect("knok graph registry lock poisoned")
            .clear();

        let graph = test_graph("duplicate", ElementType::F32);
        register_graph(graph.clone()).unwrap();
        register_graph(graph).unwrap();

        let error = register_graph(test_graph("duplicate", ElementType::I32)).unwrap_err();
        assert!(error
            .to_string()
            .contains("duplicate #[knok::graph] name `duplicate`"));

        graph_registry()
            .lock()
            .expect("knok graph registry lock poisoned")
            .clear();
    }

    fn test_graph(name: &str, elem: ElementType) -> TypedGraph {
        let ty = TensorType {
            elem,
            shape: vec![4],
        };
        TypedGraph {
            name: name.into(),
            backend: "llvm-cpu".into(),
            inputs: vec![Input {
                name: "x".into(),
                ty: ty.clone(),
            }],
            outputs: vec![ty],
            lets: Vec::new(),
            body: Vec::new(),
        }
    }
}
