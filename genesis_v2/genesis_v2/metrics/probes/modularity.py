"""Topology Modularity Probe — detect functional specialization in genome graph.

Uses a simplified Louvain-like community detection on the genome adjacency
matrix. Q > 0.4 indicates functional partitioning (understanding).
Q < 0.2 indicates random/monolithic topology (memorization).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModularityResult:
    q_score: float  # modularity Q ∈ [-0.5, 1.0]
    n_communities: int
    n_nodes: int
    n_edges: int


def _adjacency_matrix(genome) -> tuple[np.ndarray, dict[int, int]]:
    """Build adjacency matrix from genome graph. Returns (matrix, id_to_idx)."""
    node_ids = sorted(genome.nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    adj = np.zeros((n, n), dtype=np.float64)

    for e in genome.edges.values():
        if e.src in id_to_idx and e.dst in id_to_idx:
            i = id_to_idx[e.src]
            j = id_to_idx[e.dst]
            # Weight = Frobenius norm of edge weight matrix
            w = float(np.linalg.norm(e.weight))
            adj[i, j] += w
            adj[j, i] += w  # undirected for modularity

    return adj, id_to_idx


def _louvain_simple(adj: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Simplified Louvain community detection.

    Returns community assignments for each node.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    m = adj.sum() / 2.0
    if m < 1e-12:
        return np.arange(n)

    communities = np.arange(n)
    degrees = adj.sum(axis=1)

    for _ in range(max_iter):
        improved = False
        for i in range(n):
            best_comm = communities[i]
            best_gain = 0.0

            # Try moving i to each neighbor's community
            neighbor_comms = set(communities[adj[i] > 0])
            neighbor_comms.add(communities[i])

            ki = degrees[i]
            for c in neighbor_comms:
                # Gain from moving i to community c
                nodes_in_c = np.where(communities == c)[0]
                sum_in = adj[i, nodes_in_c].sum()
                sum_tot = degrees[nodes_in_c].sum()

                # Remove i from its current community
                old_c = communities[i]
                old_nodes = np.where(communities == old_c)[0]
                old_sum_in = adj[i, old_nodes].sum()
                old_sum_tot = degrees[old_nodes].sum()

                # Modularity gain (simplified)
                new_q = (sum_in / (2 * m)) - (sum_tot * ki / (2 * m * m)) if c != old_c else 0
                old_q = (old_sum_in / (2 * m)) - (old_sum_tot * ki / (2 * m * m))
                gain = new_q - old_q

                if gain > best_gain:
                    best_gain = gain
                    best_comm = c

            if best_comm != communities[i]:
                communities[i] = best_comm
                improved = True

        if not improved:
            break

    # Renumber communities
    unique = np.unique(communities)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[c] for c in communities])


def _modularity_q(adj: np.ndarray, communities: np.ndarray) -> float:
    """Compute Newman's modularity Q."""
    n = adj.shape[0]
    m = adj.sum() / 2.0
    if m < 1e-12:
        return 0.0

    q = 0.0
    degrees = adj.sum(axis=1)
    for i in range(n):
        for j in range(n):
            if communities[i] == communities[j]:
                q += adj[i, j] - (degrees[i] * degrees[j]) / (2 * m)

    return q / (2 * m)


def probe_modularity(genome) -> ModularityResult:
    """Compute modularity Q of the genome graph's topology."""
    adj, id_to_idx = _adjacency_matrix(genome)
    n = adj.shape[0]

    if n == 0 or adj.sum() < 1e-12:
        return ModularityResult(q_score=0.0, n_communities=0, n_nodes=n, n_edges=genome.edge_count())

    communities = _louvain_simple(adj)
    q = _modularity_q(adj, communities)

    return ModularityResult(
        q_score=float(q),
        n_communities=int(len(np.unique(communities))),
        n_nodes=n,
        n_edges=genome.edge_count(),
    )
