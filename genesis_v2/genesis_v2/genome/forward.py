"""Deterministic forward pass over a GenomeGraph v2 (vector-node version).

v2 upgrade: each node computes a D_node=64 dim vector; edges carry weight
matrices W ∈ R^{D_dst × D_src}. The BLAS bundle pre-stacks weight matrices
per destination node for efficient numpy.dot aggregation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genesis_v2.genome.graph import (
    D_NODE, D_OUT, EdgeKind, GenomeGraph, NodeID, NodeType,
)


# ==========================================================================
# activations (vectorized)
# ==========================================================================


def _activate(z: np.ndarray, node_type: NodeType) -> np.ndarray:
    if node_type == NodeType.HIDDEN:
        return np.tanh(z)
    if node_type == NodeType.GATING:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
    # OUTPUT and INPUT: linear
    return z


# ==========================================================================
# deterministic topological order over (FORWARD ∪ SHORTCUT)
# ==========================================================================


def _acyclic_topo_order(g: GenomeGraph) -> list[NodeID]:
    incoming: dict[NodeID, list[NodeID]] = {nid: [] for nid in g.nodes}
    for e in g.edges.values():
        if e.kind == EdgeKind.RECURRENT:
            continue
        incoming[e.dst].append(e.src)

    indeg = {nid: len(src_list) for nid, src_list in incoming.items()}
    frontier = sorted([nid for nid, d in indeg.items() if d == 0])
    order: list[NodeID] = []

    successors: dict[NodeID, list[NodeID]] = {nid: [] for nid in g.nodes}
    for eid in sorted(g.edges):
        e = g.edges[eid]
        if e.kind == EdgeKind.RECURRENT:
            continue
        successors[e.src].append(e.dst)

    while frontier:
        nid = frontier.pop(0)
        order.append(nid)
        for nxt in sorted(successors[nid]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                _insert_sorted(frontier, nxt)

    if len(order) < len(g.nodes):
        leftovers = sorted(nid for nid in g.nodes if nid not in set(order))
        order.extend(leftovers)
    return order


def _insert_sorted(lst: list[int], x: int) -> None:
    lo, hi = 0, len(lst)
    while lo < hi:
        mid = (lo + hi) // 2
        if lst[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    lst.insert(lo, x)


# ==========================================================================
# incoming-edge index
# ==========================================================================


def _incoming_edge_ids(g: GenomeGraph) -> dict[NodeID, list[int]]:
    incoming_edges: dict[NodeID, list[int]] = {nid: [] for nid in g.nodes}
    for eid in sorted(g.edges):
        e = g.edges[eid]
        incoming_edges[e.dst].append(eid)
    return incoming_edges


# ==========================================================================
# scalar reference forward (gold / tests)
# ==========================================================================


def forward_scalar(g: GenomeGraph, x: np.ndarray) -> np.ndarray:
    """Reference implementation — explicit Python loops over edges with vector nodes."""
    n_input = len(g.input_nodes)
    d_in = n_input * D_NODE
    if x.ndim != 1 or x.shape[0] != d_in:
        raise ValueError(f"input dim mismatch: expected ({d_in},), got {tuple(x.shape)}")

    incoming_edges = _incoming_edge_ids(g)
    act: dict[NodeID, np.ndarray] = {}

    # Feed input nodes
    for i, in_id in enumerate(g.input_nodes):
        act[in_id] = x[i * D_NODE : (i + 1) * D_NODE].copy()

    order = _acyclic_topo_order(g)

    for nid in order:
        node = g.nodes[nid]
        if node.type == NodeType.INPUT:
            continue

        dst_dim = node.dim
        z = np.zeros(dst_dim, dtype=np.float32)

        for eid in incoming_edges[nid]:
            e = g.edges[eid]
            if e.kind == EdgeKind.RECURRENT:
                src_val = g._last_hidden.get(e.src, np.zeros(g.nodes[e.src].dim, dtype=np.float32))
            else:
                src_val = act.get(e.src, np.zeros(g.nodes[e.src].dim, dtype=np.float32))

            contrib = e.weight @ src_val

            if e.gated_by is not None:
                gate = act.get(
                    e.gated_by,
                    g._last_hidden.get(e.gated_by, np.zeros(g.nodes[e.gated_by].dim, dtype=np.float32)),
                )
                contrib *= gate

            z += contrib

        act[nid] = _activate(z, node.type)

    # Update recurrent state
    g._last_hidden = {nid: act.get(nid, np.zeros(g.nodes[nid].dim, dtype=np.float32)).copy()
                      for nid in g.nodes}

    # Concatenate output activations
    out_parts = []
    for oid in g.output_nodes:
        out_parts.append(act.get(oid, np.zeros(g.nodes[oid].dim, dtype=np.float32)))
    return np.concatenate(out_parts).astype(np.float32)


# ==========================================================================
# BLAS bundle cache + forward
# ==========================================================================


def _rebuild_forward_bundles(g: GenomeGraph) -> None:
    """Compile per-node incoming-edge bundles into stacked numpy arrays."""
    incoming_edges = _incoming_edge_ids(g)
    order = _acyclic_topo_order(g)

    bundles: dict[NodeID, dict[str, Any]] = {}
    for nid in order:
        node = g.nodes[nid]
        if node.type == NodeType.INPUT:
            continue
        eids = incoming_edges[nid]
        if not eids:
            bundles[nid] = {
                "src_ids": np.array([], dtype=np.int32),
                "is_recurrent": np.array([], dtype=np.bool_),
                "weight_stack": np.zeros((node.dim, 0), dtype=np.float32),
                "gate_ids": np.array([], dtype=np.int32),
            }
            continue

        src_ids = []
        is_rec = []
        weights = []
        gate_ids = []
        for eid in eids:
            e = g.edges[eid]
            src_ids.append(e.src)
            is_rec.append(e.kind == EdgeKind.RECURRENT)
            weights.append(e.weight)
            gate_ids.append(-1 if e.gated_by is None else e.gated_by)

        weight_stack = np.stack(weights, axis=1)  # (dst_dim, n_edges, src_dim)
        bundles[nid] = {
            "src_ids": np.asarray(src_ids, dtype=np.int32),
            "is_recurrent": np.asarray(is_rec, dtype=np.bool_),
            "weight_stack": weight_stack,
            "gate_ids": np.asarray(gate_ids, dtype=np.int32),
        }

    g._fwd_bundles = bundles  # type: ignore[attr-defined]
    g._fwd_topo_order = order  # type: ignore[attr-defined]
    g._forward_cache_built_at = g._forward_cache_revision


def _ensure_forward_bundles(g: GenomeGraph) -> None:
    if g._forward_cache_built_at == g._forward_cache_revision:
        return
    _rebuild_forward_bundles(g)


def forward_blas(g: GenomeGraph, x: np.ndarray) -> np.ndarray:
    """Production forward — per-destination aggregation via numpy matrix ops."""
    n_input = len(g.input_nodes)
    d_in = n_input * D_NODE
    if x.ndim != 1 or x.shape[0] != d_in:
        raise ValueError(f"input dim mismatch: expected ({d_in},), got {tuple(x.shape)}")

    _ensure_forward_bundles(g)
    bundles: dict[NodeID, dict[str, Any]] = g._fwd_bundles  # type: ignore[attr-defined]
    order: list[NodeID] = g._fwd_topo_order  # type: ignore[attr-defined]

    if not g.nodes:
        return np.zeros(D_OUT, dtype=np.float32)

    # Build activation array indexed by node id (ragged dim → use dict)
    act: dict[NodeID, np.ndarray] = {}

    # Feed input nodes
    for i, in_id in enumerate(g.input_nodes):
        act[in_id] = x[i * D_NODE : (i + 1) * D_NODE].copy()

    # Pre-populate last_hidden defaults
    last_h = g._last_hidden

    for nid in order:
        node = g.nodes[nid]
        if node.type == NodeType.INPUT:
            continue

        b = bundles[nid]
        src_ids = b["src_ids"]
        n_edges = len(src_ids)

        if n_edges == 0:
            act[nid] = _activate(np.zeros(node.dim, dtype=np.float32), node.type)
            continue

        # Gather source vectors into (src_dim, n_edges) matrix
        src_vecs = []
        for k in range(n_edges):
            sid = int(src_ids[k])
            if b["is_recurrent"][k]:
                vec = last_h.get(sid, np.zeros(g.nodes[sid].dim, dtype=np.float32))
            else:
                vec = act.get(sid, np.zeros(g.nodes[sid].dim, dtype=np.float32))
            src_vecs.append(vec)
        src_mat = np.stack(src_vecs, axis=1)  # (src_dim, n_edges)

        # Apply gating
        gate_ids = b["gate_ids"]
        gate_mask = gate_ids >= 0
        if np.any(gate_mask):
            for k in np.where(gate_mask)[0]:
                gid = int(gate_ids[k])
                gate_vec = act.get(gid, last_h.get(gid, np.zeros(g.nodes[gid].dim, dtype=np.float32)))
                src_mat[:, k] *= gate_vec

        # weight_stack: (dst_dim, n_edges, src_dim)
        # We want: z = sum_k weight_stack[:, k, :] @ src_mat[:, k]
        ws = b["weight_stack"]
        # Efficient: z = sum over edges of W_k @ x_k
        # ws[:, k, :] is (dst_dim, src_dim), src_mat[:, k] is (src_dim,)
        z = np.zeros(node.dim, dtype=np.float32)
        for k in range(n_edges):
            z += ws[:, k, :] @ src_mat[:, k]

        act[nid] = _activate(z, node.type)

    # Update recurrent state
    g._last_hidden = {nid: act.get(nid, np.zeros(g.nodes[nid].dim, dtype=np.float32)).copy()
                      for nid in g.nodes}

    # Concatenate output activations into D_OUT vector
    out_parts = []
    for oid in g.output_nodes:
        out_parts.append(act.get(oid, np.zeros(g.nodes[oid].dim, dtype=np.float32)))
    return np.concatenate(out_parts).astype(np.float32)


# ==========================================================================
# public entry
# ==========================================================================


def forward(g: GenomeGraph, x: np.ndarray) -> np.ndarray:
    """Return OUTPUT activations (BLAS bundle path)."""
    return forward_blas(g, x)


GenomeGraph.forward = forward  # type: ignore[method-assign]
