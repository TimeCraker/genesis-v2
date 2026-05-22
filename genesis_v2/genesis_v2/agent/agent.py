"""Agent v2 — digital organism with output partition (256:128:128:16).

v2 upgrades:
    * Output partition: action[256] | message[128] | state[128] | selfmod[16]
    * Social state: inbox, social_memory
    * Self-modification channel
    * Birth memory for generational transfer
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis_v2.genome.graph import (
    D_ACTION, D_MESSAGE, D_NODE, D_OUT, D_SELFMOD, D_STATE, GenomeGraph,
)


@dataclass
class Agent:
    id: str
    generation: int = 0
    island_id: int = 0

    genome: GenomeGraph | None = None
    energy: float = 0.0
    tick_alive: int = 0
    is_alive: bool = True

    # === output partition cache (updated each tick) ===
    last_action: np.ndarray | None = None      # [256]
    last_message: np.ndarray | None = None     # [128]
    last_state: np.ndarray | None = None       # [128]
    state_buffer: np.ndarray | None = None     # [128] previous tick work memory
    last_selfmod: np.ndarray | None = None     # [16]

    # === evaluation caches ===
    prediction_error: float = 0.0
    compression: float = 0.0
    behavioral_variance: float = 0.0
    exploration_bonus: float = 0.0
    fitness: float = 0.0

    # === social state ===
    inbox: list = field(default_factory=list)
    social_memory: dict = field(default_factory=dict)  # {agent_id: trust_score}

    # === generational memory ===
    birth_mem: dict = field(default_factory=dict)

    # === self-modification ===
    selfmod_enabled: bool = False
    selfmod_count: int = 0
    selfmod_survived: int = 0

    # === birthday snapshot ===
    birth_nodes: int = 0
    birth_edges: int = 0

    def to_payload(self) -> dict:
        if self.genome is None:
            raise ValueError("agent.genome required for persistence")
        return {
            "id": self.id,
            "generation": self.generation,
            "island_id": self.island_id,
            "energy": float(self.energy),
            "tick_alive": int(self.tick_alive),
            "is_alive": bool(self.is_alive),
            "prediction_error": float(self.prediction_error),
            "compression": float(self.compression),
            "behavioral_variance": float(self.behavioral_variance),
            "exploration_bonus": float(self.exploration_bonus),
            "fitness": float(self.fitness),
            "selfmod_enabled": bool(self.selfmod_enabled),
            "selfmod_count": int(self.selfmod_count),
            "selfmod_survived": int(self.selfmod_survived),
            "birth_nodes": int(self.birth_nodes),
            "birth_edges": int(self.birth_edges),
            "social_memory": dict(self.social_memory),
            "birth_mem": dict(self.birth_mem),
            "genome": self.genome.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> Agent:
        genome = GenomeGraph.from_payload(payload["genome"])
        return cls(
            id=str(payload["id"]),
            generation=int(payload.get("generation", 0)),
            island_id=int(payload.get("island_id", 0)),
            genome=genome,
            energy=float(payload.get("energy", 0.0)),
            tick_alive=int(payload.get("tick_alive", 0)),
            is_alive=bool(payload.get("is_alive", True)),
            prediction_error=float(payload.get("prediction_error", 0.0)),
            compression=float(payload.get("compression", 0.0)),
            behavioral_variance=float(payload.get("behavioral_variance", 0.0)),
            exploration_bonus=float(payload.get("exploration_bonus", 0.0)),
            fitness=float(payload.get("fitness", 0.0)),
            selfmod_enabled=bool(payload.get("selfmod_enabled", False)),
            selfmod_count=int(payload.get("selfmod_count", 0)),
            selfmod_survived=int(payload.get("selfmod_survived", 0)),
            birth_nodes=int(payload.get("birth_nodes", genome.node_count())),
            birth_edges=int(payload.get("birth_edges", genome.edge_count())),
            social_memory=dict(payload.get("social_memory", {})),
            birth_mem=dict(payload.get("birth_mem", {})),
        )


def new_agent(
    id: str,
    genome: GenomeGraph,
    initial_energy: float = 1000.0,
    generation: int = 0,
    island_id: int = 0,
) -> Agent:
    return Agent(
        id=id,
        genome=genome,
        generation=generation,
        island_id=island_id,
        energy=initial_energy,
        birth_nodes=genome.node_count(),
        birth_edges=genome.edge_count(),
    )


def split_output(output_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split 528-dim output into (action[256], message[128], state[128], selfmod[16])."""
    if output_vec.shape[0] != D_OUT:
        raise ValueError(f"Expected {D_OUT}-dim output, got {output_vec.shape[0]}")
    action = output_vec[:D_ACTION]
    message = output_vec[D_ACTION:D_ACTION + D_MESSAGE]
    state = output_vec[D_ACTION + D_MESSAGE:D_ACTION + D_MESSAGE + D_STATE]
    selfmod = output_vec[D_ACTION + D_MESSAGE + D_STATE:D_ACTION + D_MESSAGE + D_STATE + D_SELFMOD]
    return action, message, state, selfmod
