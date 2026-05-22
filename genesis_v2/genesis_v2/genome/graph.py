"""GenomeGraph v2 — vector-node computational graph (the sole internal state of an agent).

v1 → v2 upgrade:
    * Node output:  float → np.ndarray[D_node=64]
    * Edge weight:  float → np.ndarray[D_dst, D_src]  (matrix)
    * Output partition: 528-dim = 4×64 (action) + 2×64 (message) + 2×64 (state) + 1×64 (selfmod)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

NodeID = int
EdgeID = int

# Output partition constants
D_NODE: int = 64
D_ACTION: int = 256      # 4 × 64
D_MESSAGE: int = 128     # 2 × 64
D_STATE: int = 128       # 2 × 64
D_SELFMOD: int = 16      # first 16 of last 64-dim node
D_OUT: int = D_ACTION + D_MESSAGE + D_STATE + D_NODE  # 528


class NodeType(IntEnum):
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2
    GATING = 3


class EdgeKind(IntEnum):
    FORWARD = 0
    SHORTCUT = 1
    RECURRENT = 2


@dataclass
class Node:
    id: NodeID
    type: NodeType
    dim: int = D_NODE


@dataclass
class Edge:
    id: EdgeID
    src: NodeID
    dst: NodeID
    kind: EdgeKind
    weight: np.ndarray  # shape (dst_dim, src_dim)
    gated_by: NodeID | None = None


class GraphConfig(BaseModel):
    node_dim: int = Field(default=D_NODE, gt=0)
    input_nodes: int = Field(default=8, gt=0)
    output_nodes_action: int = Field(default=4, gt=0)
    output_nodes_message: int = Field(default=2, gt=0)
    output_nodes_state: int = Field(default=2, gt=0)
    output_nodes_selfmod: int = Field(default=1, gt=0)
    initial_hidden_nodes: int = Field(default=4, ge=0)
    initial_edge_density: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_scale: float = Field(default=0.3, gt=0.0)

    @property
    def output_nodes(self) -> int:
        return (
            self.output_nodes_action
            + self.output_nodes_message
            + self.output_nodes_state
            + self.output_nodes_selfmod
        )

    @property
    def input_dim(self) -> int:
        return self.node_dim * self.input_nodes

    @property
    def output_dim(self) -> int:
        return D_OUT


@dataclass
class GenomeGraph:
    nodes: dict[NodeID, Node] = field(default_factory=dict)
    edges: dict[EdgeID, Edge] = field(default_factory=dict)
    input_nodes: list[NodeID] = field(default_factory=list)
    output_nodes: list[NodeID] = field(default_factory=list)

    _next_node_id: NodeID = 0
    _next_edge_id: EdgeID = 0

    _last_hidden: dict[NodeID, np.ndarray] = field(default_factory=dict)
    _forward_cache_revision: int = 0
    _forward_cache_built_at: int = -1

    # partition index boundaries (set during construction)
    _output_partition: dict[str, tuple[int, int]] = field(default_factory=dict)

    def touch_forward_cache(self) -> None:
        self._forward_cache_revision += 1

    # ---------- counts ----------

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def hidden_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.type == NodeType.HIDDEN)

    def gating_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.type == NodeType.GATING)

    # ---------- ID issuance ----------

    def _alloc_node_id(self) -> NodeID:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _alloc_edge_id(self) -> EdgeID:
        eid = self._next_edge_id
        self._next_edge_id += 1
        return eid

    def _add_node(self, node_type: NodeType, dim: int = D_NODE) -> Node:
        nid = self._alloc_node_id()
        node = Node(id=nid, type=node_type, dim=dim)
        self.nodes[nid] = node
        self.touch_forward_cache()
        return node

    def _add_edge(
        self,
        src: NodeID,
        dst: NodeID,
        kind: EdgeKind,
        weight: np.ndarray,
        gated_by: NodeID | None = None,
    ) -> Edge:
        eid = self._alloc_edge_id()
        edge = Edge(
            id=eid, src=src, dst=dst, kind=kind,
            weight=np.array(weight, dtype=np.float32),
            gated_by=gated_by,
        )
        self.edges[eid] = edge
        self.touch_forward_cache()
        return edge

    # ---------- output partition helpers ----------

    def get_action_nodes(self) -> list[NodeID]:
        return self.output_nodes[:4]

    def get_message_nodes(self) -> list[NodeID]:
        return self.output_nodes[4:6]

    def get_state_nodes(self) -> list[NodeID]:
        return self.output_nodes[6:8]

    def get_selfmod_nodes(self) -> list[NodeID]:
        return self.output_nodes[8:9]

    # ---------- deep copy ----------

    def copy(self) -> GenomeGraph:
        g = GenomeGraph(
            input_nodes=list(self.input_nodes),
            output_nodes=list(self.output_nodes),
            _next_node_id=self._next_node_id,
            _next_edge_id=self._next_edge_id,
            _output_partition=dict(self._output_partition),
        )
        g.nodes = {nid: Node(id=n.id, type=n.type, dim=n.dim) for nid, n in self.nodes.items()}
        g.edges = {
            eid: Edge(
                id=e.id, src=e.src, dst=e.dst, kind=e.kind,
                weight=e.weight.copy(), gated_by=e.gated_by,
            )
            for eid, e in self.edges.items()
        }
        g._forward_cache_revision = 0
        g._forward_cache_built_at = -1
        return g

    # ---------- entropy ----------

    def entropy(self) -> float:
        if not self.nodes:
            return 0.0
        counts: dict[NodeType, int] = {}
        for n in self.nodes.values():
            counts[n.type] = counts.get(n.type, 0) + 1
        total = float(len(self.nodes))
        h = 0.0
        for c in counts.values():
            p = c / total
            h -= p * math.log(p)
        return h

    # ---------- state ----------

    def reset_state(self) -> None:
        self._last_hidden.clear()

    # ---------- mutation convenience ----------

    def mutate(self, rng: np.random.Generator, allowed=None):
        from genesis_v2.genome.mutate import mutate as _mutate
        return _mutate(self, rng, allowed)

    # ---------- forward convenience ----------

    def forward(self, x: np.ndarray) -> np.ndarray:
        from genesis_v2.genome.forward import forward as _forward
        return _forward(self, x)

    # ---------- persistence ----------

    def to_payload(self) -> dict:
        return {
            "input_nodes": list(self.input_nodes),
            "output_nodes": list(self.output_nodes),
            "next_node_id": self._next_node_id,
            "next_edge_id": self._next_edge_id,
            "output_partition": dict(self._output_partition),
            "nodes": [
                {"id": n.id, "type": int(n.type), "dim": n.dim}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "src": e.src,
                    "dst": e.dst,
                    "kind": int(e.kind),
                    "weight": e.weight.tolist(),
                    "gated_by": e.gated_by,
                }
                for e in self.edges.values()
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> GenomeGraph:
        g = cls(
            input_nodes=[int(x) for x in payload["input_nodes"]],
            output_nodes=[int(x) for x in payload["output_nodes"]],
            _next_node_id=int(payload["next_node_id"]),
            _next_edge_id=int(payload["next_edge_id"]),
            _output_partition={k: tuple(v) for k, v in payload.get("output_partition", {}).items()},
        )
        g.nodes = {
            int(item["id"]): Node(
                id=int(item["id"]),
                type=NodeType(int(item["type"])),
                dim=int(item.get("dim", D_NODE)),
            )
            for item in payload["nodes"]
        }
        g.edges = {
            int(item["id"]): Edge(
                id=int(item["id"]),
                src=int(item["src"]),
                dst=int(item["dst"]),
                kind=EdgeKind(int(item["kind"])),
                weight=np.array(item["weight"], dtype=np.float32),
                gated_by=(None if item["gated_by"] is None else int(item["gated_by"])),
            )
            for item in payload["edges"]
        }
        g.touch_forward_cache()
        return g


# ==========================================================================
# factory
# ==========================================================================


def new_genome_graph(cfg: GraphConfig, rng: np.random.Generator) -> GenomeGraph:
    """Build a fresh GenomeGraph with random sparse wiring (vector node version)."""
    g = GenomeGraph()

    # Input nodes
    for _ in range(cfg.input_nodes):
        n = g._add_node(NodeType.INPUT, dim=cfg.node_dim)
        g.input_nodes.append(n.id)

    # Output nodes (9 total: 4 action + 2 message + 2 state + 1 selfmod)
    out_n = cfg.output_nodes
    for _ in range(out_n):
        n = g._add_node(NodeType.OUTPUT, dim=cfg.node_dim)
        g.output_nodes.append(n.id)

    # Set partition boundaries
    g._output_partition = {
        "action": (0, cfg.output_nodes_action),
        "message": (cfg.output_nodes_action, cfg.output_nodes_action + cfg.output_nodes_message),
        "state": (
            cfg.output_nodes_action + cfg.output_nodes_message,
            cfg.output_nodes_action + cfg.output_nodes_message + cfg.output_nodes_state,
        ),
        "selfmod": (
            cfg.output_nodes_action + cfg.output_nodes_message + cfg.output_nodes_state,
            out_n,
        ),
    }

    # Hidden nodes
    hidden_ids: list[NodeID] = []
    for _ in range(cfg.initial_hidden_nodes):
        n = g._add_node(NodeType.HIDDEN, dim=cfg.node_dim)
        hidden_ids.append(n.id)

    # Wire: {inputs ∪ hidden} → {hidden ∪ outputs}
    sources = list(g.input_nodes) + hidden_ids
    sinks = hidden_ids + list(g.output_nodes)

    for s in sources:
        for d in sinks:
            if s == d:
                continue
            if rng.random() >= cfg.initial_edge_density:
                continue
            src_dim = g.nodes[s].dim
            dst_dim = g.nodes[d].dim
            w = rng.standard_normal((dst_dim, src_dim)).astype(np.float32) * cfg.weight_scale
            g._add_edge(src=s, dst=d, kind=EdgeKind.FORWARD, weight=w)

    return g
