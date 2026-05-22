"""NEAT-style crossover for GenomeGraph v2.

Aligns two parent genomes by node ID (homologous genes).
- Matched nodes/edges: randomly pick one parent (bias toward fitter parent).
- Disjoint/excess: inherit from fitter parent.
- Output/input partition structure is forced consistent.
"""

from __future__ import annotations

import numpy as np

from genesis_v2.genome.graph import (
    D_NODE,
    EdgeKind,
    GenomeGraph,
    NodeID,
    NodeType,
)


def crossover(
    parent_a: GenomeGraph,
    parent_b: GenomeGraph,
    rng: np.random.Generator,
    fitness_a: float = 0.0,
    fitness_b: float = 0.0,
) -> GenomeGraph:
    """NEAT-style crossover producing one child.

    Args:
        parent_a: First parent genome.
        parent_b: Second parent genome.
        rng: Random generator for reproducibility.
        fitness_a: Fitness of parent A (higher = more genes inherited).
        fitness_b: Fitness of parent B.

    Returns:
        A new child GenomeGraph.
    """
    # Ensure A is the fitter parent (or equal)
    if fitness_b > fitness_a:
        parent_a, parent_b = parent_b, parent_a
        fitness_a, fitness_b = fitness_b, fitness_a

    child = GenomeGraph()

    # Force input/output partition from parent A (both parents must share structure)
    # Note: _output_partition values are indices into the output_nodes list, not node IDs
    child._output_partition = dict(parent_a._output_partition)

    # --- Align nodes by ID ---
    nodes_a = dict(parent_a.nodes)
    nodes_b = dict(parent_b.nodes)
    all_node_ids = sorted(set(nodes_a.keys()) | set(nodes_b.keys()))

    # Bias: probability of picking parent A when both have the gene
    total_fit = fitness_a + fitness_b
    if total_fit > 0:
        p_a = fitness_a / total_fit
    else:
        p_a = 0.5

    node_id_map: dict[NodeID, NodeID] = {}  # old_id → new_id in child

    for nid in all_node_ids:
        in_a = nid in nodes_a
        in_b = nid in nodes_b

        if in_a and in_b:
            # Matched gene: pick one parent
            src_parent = parent_a if rng.random() < p_a else parent_b
            src_node = src_parent.nodes[nid]
        elif in_a:
            # Disjoint/excess from A (fitter)
            src_node = nodes_a[nid]
        else:
            # Disjoint/excess from B (less fit) — only inherit if both are alive
            src_node = nodes_b[nid]

        new_id = child._alloc_node_id()
        from genesis_v2.genome.graph import Node
        child.nodes[new_id] = Node(id=new_id, type=src_node.type, dim=src_node.dim)
        node_id_map[nid] = new_id

    # Remap input/output node lists
    child.input_nodes = [node_id_map[nid] for nid in parent_a.input_nodes]
    child.output_nodes = [node_id_map[nid] for nid in parent_a.output_nodes]
    # _output_partition stores indices into the output_nodes list, not node IDs — keep as-is

    # --- Align edges by (src, dst, kind) signature ---
    def _edge_sig(e) -> tuple[NodeID, NodeID, EdgeKind]:
        return (e.src, e.dst, e.kind)

    edges_a = {_edge_sig(e): e for e in parent_a.edges.values()}
    edges_b = {_edge_sig(e): e for e in parent_b.edges.values()}
    all_sigs = sorted(set(edges_a.keys()) | set(edges_b.keys()))

    for sig in all_sigs:
        in_a = sig in edges_a
        in_b = sig in edges_b

        if in_a and in_b:
            # Matched edge: pick one parent
            src_edge = edges_a[sig] if rng.random() < p_a else edges_b[sig]
        elif in_a:
            src_edge = edges_a[sig]
        else:
            src_edge = edges_b[sig]

        # Remap endpoints to child node IDs
        new_src = node_id_map.get(src_edge.src)
        new_dst = node_id_map.get(src_edge.dst)
        if new_src is None or new_dst is None:
            continue  # endpoint node not in child

        # Check that source and destination nodes exist in child
        if new_src not in child.nodes or new_dst not in child.nodes:
            continue

        # Ensure weight shape matches child node dims
        src_dim = child.nodes[new_src].dim
        dst_dim = child.nodes[new_dst].dim
        w = src_edge.weight
        if w.shape == (dst_dim, src_dim):
            new_weight = w.copy()
        else:
            # Reshape if parent node dims differ (shouldn't happen with standard config)
            new_weight = rng.standard_normal((dst_dim, src_dim)).astype(np.float32) * 0.3

        child._add_edge(
            src=new_src,
            dst=new_dst,
            kind=src_edge.kind,
            weight=new_weight,
            gated_by=node_id_map.get(src_edge.gated_by) if src_edge.gated_by is not None else None,
        )

    # Ensure minimum connectivity: at least one edge to each output node
    _ensure_output_connectivity(child, parent_a, node_id_map, rng)

    child.touch_forward_cache()
    return child


def _ensure_output_connectivity(
    child: GenomeGraph,
    parent_a: GenomeGraph,
    node_id_map: dict[NodeID, NodeID],
    rng: np.random.Generator,
) -> None:
    """Ensure every output node has at least one incoming edge."""
    output_ids = set(child.output_nodes)

    for out_nid in output_ids:
        has_incoming = any(e.dst == out_nid for e in child.edges.values())
        if has_incoming:
            continue

        # Find a source candidate (hidden or input node)
        candidates = [
            nid for nid, n in child.nodes.items()
            if n.type in (NodeType.HIDDEN, NodeType.INPUT) and nid != out_nid
        ]
        if not candidates:
            continue

        src = int(rng.choice(candidates))
        src_dim = child.nodes[src].dim
        dst_dim = child.nodes[out_nid].dim
        w = rng.standard_normal((dst_dim, src_dim)).astype(np.float32) * 0.3
        child._add_edge(src=src, dst=out_nid, kind=EdgeKind.FORWARD, weight=w)


def crossover_pair(
    agents: list,
    rng: np.random.Generator,
) -> GenomeGraph | None:
    """Pick two random agents and perform crossover. Returns child genome or None."""
    if len(agents) < 2:
        return None

    indices = rng.choice(len(agents), size=2, replace=False)
    a = agents[int(indices[0])]
    b = agents[int(indices[1])]

    if a.genome is None or b.genome is None:
        return None

    return crossover(
        a.genome, b.genome,
        rng=rng,
        fitness_a=a.fitness,
        fitness_b=b.fitness,
    )
