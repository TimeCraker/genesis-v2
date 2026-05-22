"""12 mutation primitives for GenomeGraph v2 (vector-node version).

v1 → v2 upgrade:
    * Edge weights: float → np.ndarray[D_dst, D_src]
    * 7 v1 primitives vectorized + 5 new v2 primitives
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from genesis_v2.genome.graph import (
    D_NODE, EdgeKind, GenomeGraph, NodeID, NodeType,
)

WEIGHT_PERTURB_SIGMA = 0.1


class MutationKind(IntEnum):
    # v1 upgraded (0-6)
    ADD_FORWARD_EDGE = 0
    ADD_SHORTCUT_EDGE = 1
    ADD_RECURRENT_EDGE = 2
    ADD_HIDDEN_NODE = 3
    ADD_GATING_NODE = 4
    PERTURB_WEIGHT = 5
    DELETE_RANDOM_EDGE = 6
    # v2 new (7-11)
    ADD_ATTENTION_GROUP = 7
    ADD_MODULE = 8
    SPLIT_NODE_DIM = 9
    MERGE_NODES = 10
    ADD_COMM_EDGE = 11


# ==========================================================================
# helpers
# ==========================================================================


def _candidate_sources(g: GenomeGraph) -> list[NodeID]:
    return [n.id for n in g.nodes.values() if n.type != NodeType.OUTPUT]


def _candidate_sinks(g: GenomeGraph) -> list[NodeID]:
    return [n.id for n in g.nodes.values() if n.type != NodeType.INPUT]


def _has_edge(g: GenomeGraph, src: NodeID, dst: NodeID, kind: EdgeKind) -> bool:
    for e in g.edges.values():
        if e.src == src and e.dst == dst and e.kind == kind:
            return True
    return False


def _pick_endpoint_pair(
    g: GenomeGraph, rng: np.random.Generator
) -> tuple[NodeID, NodeID] | None:
    sources = _candidate_sources(g)
    sinks = _candidate_sinks(g)
    if not sources or not sinks:
        return None
    for _ in range(8):
        s = int(rng.choice(sources))
        d = int(rng.choice(sinks))
        if s != d:
            return s, d
    return None


# ==========================================================================
# v1 primitives (vectorized)
# ==========================================================================


def add_forward_edge(g: GenomeGraph, rng: np.random.Generator) -> bool:
    pair = _pick_endpoint_pair(g, rng)
    if pair is None:
        return False
    s, d = pair
    if _has_edge(g, s, d, EdgeKind.FORWARD):
        return False
    src_dim = g.nodes[s].dim
    dst_dim = g.nodes[d].dim
    w = rng.standard_normal((dst_dim, src_dim)).astype(np.float32) * 0.3
    g._add_edge(src=s, dst=d, kind=EdgeKind.FORWARD, weight=w)
    return True


def add_shortcut_edge(g: GenomeGraph, rng: np.random.Generator) -> bool:
    pair = _pick_endpoint_pair(g, rng)
    if pair is None:
        return False
    s, d = pair
    if _has_edge(g, s, d, EdgeKind.SHORTCUT):
        return False
    src_dim = g.nodes[s].dim
    dst_dim = g.nodes[d].dim
    w = rng.standard_normal((dst_dim, src_dim)).astype(np.float32) * 0.3
    g._add_edge(src=s, dst=d, kind=EdgeKind.SHORTCUT, weight=w)
    return True


def add_recurrent_edge(g: GenomeGraph, rng: np.random.Generator) -> bool:
    pair = _pick_endpoint_pair(g, rng)
    if pair is None:
        return False
    s, d = pair
    if _has_edge(g, s, d, EdgeKind.RECURRENT):
        return False
    src_dim = g.nodes[s].dim
    dst_dim = g.nodes[d].dim
    w = rng.standard_normal((dst_dim, src_dim)).astype(np.float32) * 0.3
    g._add_edge(src=s, dst=d, kind=EdgeKind.RECURRENT, weight=w)
    return True


def add_hidden_node(g: GenomeGraph, rng: np.random.Generator) -> bool:
    """NEAT split-edge: replace u→v with u→h→v (preserves approximate function)."""
    candidates = [
        e for e in g.edges.values() if e.kind in (EdgeKind.FORWARD, EdgeKind.SHORTCUT)
    ]
    if not candidates:
        return False
    e = int(rng.choice(len(candidates)))
    edge = candidates[e]
    h = g._add_node(NodeType.HIDDEN, dim=D_NODE)
    old_weight = edge.weight.copy()
    src_dim = g.nodes[edge.src].dim
    del g.edges[edge.id]
    # u→h: identity-like weight
    w1 = np.eye(D_NODE, src_dim, dtype=np.float32) * 0.5
    g._add_edge(src=edge.src, dst=h.id, kind=EdgeKind.FORWARD, weight=w1)
    # h→v: preserve old weight
    dst_dim = g.nodes[edge.dst].dim
    g._add_edge(src=h.id, dst=edge.dst, kind=EdgeKind.FORWARD,
                weight=old_weight.reshape(dst_dim, D_NODE) if old_weight.shape != (dst_dim, D_NODE) else old_weight)
    return True


def add_gating_node(g: GenomeGraph, rng: np.random.Generator) -> bool:
    existing_edges = [
        e for e in g.edges.values() if e.gated_by is None and e.kind != EdgeKind.RECURRENT
    ]
    if not existing_edges:
        return False
    sources = _candidate_sources(g)
    if not sources:
        return False

    s = int(rng.choice(sources))
    e_idx = int(rng.choice(len(existing_edges)))
    edge = existing_edges[e_idx]
    h = g._add_node(NodeType.GATING, dim=D_NODE)
    src_dim = g.nodes[s].dim
    w = rng.standard_normal((D_NODE, src_dim)).astype(np.float32) * 0.3
    g._add_edge(src=s, dst=h.id, kind=EdgeKind.FORWARD, weight=w)
    edge.gated_by = h.id
    return True


def perturb_weight(g: GenomeGraph, rng: np.random.Generator) -> bool:
    if not g.edges:
        return False
    edges_list = list(g.edges.values())
    e = edges_list[int(rng.choice(len(edges_list)))]
    noise = rng.standard_normal(e.weight.shape).astype(np.float32) * WEIGHT_PERTURB_SIGMA
    e.weight += noise
    g.touch_forward_cache()
    return True


def delete_random_edge(g: GenomeGraph, rng: np.random.Generator) -> bool:
    if len(g.edges) <= 1:
        return False
    edges_list = list(g.edges.values())
    e = edges_list[int(rng.choice(len(edges_list)))]
    del g.edges[e.id]
    g.touch_forward_cache()
    return True


# ==========================================================================
# v2 new primitives
# ==========================================================================


def add_attention_group(g: GenomeGraph, rng: np.random.Generator) -> bool:
    """Create Q/K/V nodes + output node (simplified attention mechanism)."""
    # Need at least 1 source node
    sources = _candidate_sources(g)
    if not sources:
        return False

    # Create Q, K, V hidden nodes
    q_node = g._add_node(NodeType.HIDDEN, dim=D_NODE)
    k_node = g._add_node(NodeType.HIDDEN, dim=D_NODE)
    v_node = g._add_node(NodeType.HIDDEN, dim=D_NODE)
    out_node = g._add_node(NodeType.HIDDEN, dim=D_NODE)

    # Connect some random sources to Q, K, V
    for target in [q_node, k_node, v_node]:
        n_src = max(1, min(3, len(sources)))
        chosen = rng.choice(sources, size=n_src, replace=False)
        for s_id in chosen:
            src_dim = g.nodes[int(s_id)].dim
            w = rng.standard_normal((D_NODE, src_dim)).astype(np.float32) * 0.3
            g._add_edge(src=int(s_id), dst=target.id, kind=EdgeKind.FORWARD, weight=w)

    # Q→out, K→out, V→out (simplified: output is weighted combination)
    for src_node in [q_node, k_node, v_node]:
        w = rng.standard_normal((D_NODE, D_NODE)).astype(np.float32) * 0.3
        g._add_edge(src=src_node.id, dst=out_node.id, kind=EdgeKind.FORWARD, weight=w)

    return True


def add_module(g: GenomeGraph, rng: np.random.Generator) -> bool:
    """Inject a 2-5 node subgraph (random topology)."""
    n_nodes = int(rng.integers(2, 6))
    new_hidden = []
    for _ in range(n_nodes):
        h = g._add_node(NodeType.HIDDEN, dim=D_NODE)
        new_hidden.append(h)

    # Wire internally (dense among new nodes)
    for i, src in enumerate(new_hidden):
        for j, dst in enumerate(new_hidden):
            if i == j:
                continue
            if rng.random() > 0.5:
                continue
            w = rng.standard_normal((D_NODE, D_NODE)).astype(np.float32) * 0.3
            g._add_edge(src=src.id, dst=dst.id, kind=EdgeKind.FORWARD, weight=w)

    # Connect to existing graph
    sources = _candidate_sources(g)
    sinks = _candidate_sinks(g)
    if sources and new_hidden:
        s = int(rng.choice(sources))
        src_dim = g.nodes[s].dim
        w = rng.standard_normal((D_NODE, src_dim)).astype(np.float32) * 0.3
        g._add_edge(src=s, dst=new_hidden[0].id, kind=EdgeKind.FORWARD, weight=w)

    if sinks and new_hidden:
        d = int(rng.choice(sinks))
        dst_dim = g.nodes[d].dim
        w = rng.standard_normal((dst_dim, D_NODE)).astype(np.float32) * 0.3
        g._add_edge(src=new_hidden[-1].id, dst=d, kind=EdgeKind.FORWARD, weight=w)

    return True


def split_node_dim(g: GenomeGraph, rng: np.random.Generator) -> bool:
    """Placeholder — high-dim node split (not implemented in Phase 0)."""
    return False


def merge_nodes(g: GenomeGraph, rng: np.random.Generator) -> bool:
    """Placeholder — multi-node merge (not implemented in Phase 0)."""
    return False


def add_comm_edge(g: GenomeGraph, rng: np.random.Generator) -> bool:
    """Connect a hidden/input node to a message-area output node."""
    msg_nodes = g.get_message_nodes()
    if not msg_nodes:
        return False
    sources = _candidate_sources(g)
    if not sources:
        return False

    s = int(rng.choice(sources))
    d = int(rng.choice(msg_nodes))
    if _has_edge(g, s, d, EdgeKind.FORWARD):
        return False
    src_dim = g.nodes[s].dim
    dst_dim = g.nodes[d].dim
    w = rng.standard_normal((dst_dim, src_dim)).astype(np.float32) * 0.3
    g._add_edge(src=s, dst=d, kind=EdgeKind.FORWARD, weight=w)
    return True


# ==========================================================================
# dispatch
# ==========================================================================


_DISPATCH = {
    MutationKind.ADD_FORWARD_EDGE: add_forward_edge,
    MutationKind.ADD_SHORTCUT_EDGE: add_shortcut_edge,
    MutationKind.ADD_RECURRENT_EDGE: add_recurrent_edge,
    MutationKind.ADD_HIDDEN_NODE: add_hidden_node,
    MutationKind.ADD_GATING_NODE: add_gating_node,
    MutationKind.PERTURB_WEIGHT: perturb_weight,
    MutationKind.DELETE_RANDOM_EDGE: delete_random_edge,
    MutationKind.ADD_ATTENTION_GROUP: add_attention_group,
    MutationKind.ADD_MODULE: add_module,
    MutationKind.SPLIT_NODE_DIM: split_node_dim,
    MutationKind.MERGE_NODES: merge_nodes,
    MutationKind.ADD_COMM_EDGE: add_comm_edge,
}


def mutate(
    g: GenomeGraph,
    rng: np.random.Generator,
    allowed: list[MutationKind] | None = None,
) -> MutationKind | None:
    """Apply exactly one primitive, chosen uniformly from `allowed`."""
    pool = allowed if allowed is not None else list(MutationKind)
    if not pool:
        return None
    kind = pool[int(rng.choice(len(pool)))]
    ok = _DISPATCH[kind](g, rng)
    return kind if ok else None
