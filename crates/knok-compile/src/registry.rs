use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

use knok_core::{GraphSignature, TypedGraph};

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

pub(crate) fn register_graph(graph: TypedGraph) {
    graph_registry()
        .lock()
        .expect("knok graph registry lock poisoned")
        .insert(graph.name.clone(), graph);
}
