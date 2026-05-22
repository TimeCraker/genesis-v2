"""GenerationalMemory — cross-generational knowledge transfer.

Each generation's top agents produce a memory record that can bias
the initial state of their descendants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from genesis_v2.agent.agent import Agent
from genesis_v2.genome.graph import D_NODE

MEMORY_DIR = Path(__file__).parent.parent.parent / "data" / "gen_memory"


@dataclass
class GenMemory:
    """Memory record for one agent's generation."""

    agent_id: str
    generation: int
    fitness: float
    behavioral_signature: np.ndarray = field(
        default_factory=lambda: np.zeros(D_NODE, dtype=np.float32)
    )
    successful_patterns: list[np.ndarray] = field(default_factory=list)
    social_partners: list[str] = field(default_factory=list)
    env_adaptation_scores: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "fitness": float(self.fitness),
            "behavioral_signature": self.behavioral_signature.tolist(),
            "successful_patterns": [p.tolist() for p in self.successful_patterns],
            "social_partners": list(self.social_partners),
            "env_adaptation_scores": dict(self.env_adaptation_scores),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> GenMemory:
        return cls(
            agent_id=str(payload["agent_id"]),
            generation=int(payload["generation"]),
            fitness=float(payload["fitness"]),
            behavioral_signature=np.array(payload["behavioral_signature"], dtype=np.float32),
            successful_patterns=[
                np.array(p, dtype=np.float32) for p in payload.get("successful_patterns", [])
            ],
            social_partners=list(payload.get("social_partners", [])),
            env_adaptation_scores=dict(payload.get("env_adaptation_scores", {})),
        )


class GenerationalMemoryBank:
    """Accumulates generation memories and applies them to offspring."""

    def __init__(self, max_per_generation: int = 5) -> None:
        self.max_per_generation = max_per_generation
        self.memories: dict[int, list[GenMemory]] = {}  # gen → [memories]

    def record_generation(
        self,
        agents: list[Agent],
        generation: int,
    ) -> int:
        """Record memories for top agents of a generation. Returns count recorded."""
        alive = [a for a in agents if a.is_alive]
        if not alive:
            return 0

        elites = sorted(alive, key=lambda a: a.fitness, reverse=True)[:self.max_per_generation]
        records = []

        for a in elites:
            # Behavioral signature: average action vector (first D_NODE dims)
            sig = np.zeros(D_NODE, dtype=np.float32)
            if a.last_action is not None:
                sig = a.last_action[:D_NODE].astype(np.float32)

            # Successful patterns: state vectors from high-fitness ticks
            patterns = []
            if a.last_state is not None:
                patterns.append(a.last_state[:D_NODE].astype(np.float32))

            # Social partners from social_memory
            partners = list(a.social_memory.keys())

            mem = GenMemory(
                agent_id=a.id,
                generation=generation,
                fitness=a.fitness,
                behavioral_signature=sig,
                successful_patterns=patterns,
                social_partners=partners,
            )
            records.append(mem)

        self.memories[generation] = records
        return len(records)

    def get_parent_memory(self, generation: int) -> GenMemory | None:
        """Get the best memory from the previous generation."""
        prev_gen = generation - 1
        if prev_gen not in self.memories or not self.memories[prev_gen]:
            return None
        return max(self.memories[prev_gen], key=lambda m: m.fitness)

    def apply_to_offspring(
        self,
        agent: Agent,
        rng: np.random.Generator,
        inherit_prob: float = 0.5,
    ) -> bool:
        """Apply parent memory to offspring. Returns True if memory was applied."""
        parent_mem = self.get_parent_memory(agent.generation)
        if parent_mem is None:
            return False

        if rng.random() >= inherit_prob:
            return False

        # Store behavioral signature in birth_mem
        agent.birth_mem["parent_behavioral_sig"] = parent_mem.behavioral_signature.tolist()
        agent.birth_mem["parent_fitness"] = parent_mem.fitness
        agent.birth_mem["parent_social_partners"] = parent_mem.social_partners

        # Bias recurrent state with parent's behavioral signature
        if parent_mem.successful_patterns and agent.genome is not None:
            pattern = parent_mem.successful_patterns[0]
            # Set as initial bias for hidden nodes' recurrent state
            for hid_nid in sorted(agent.genome.nodes.keys()):
                node = agent.genome.nodes[hid_nid]
                if node.type.value == 2:  # HIDDEN
                    agent.genome._last_hidden[hid_nid] = pattern[:node.dim].copy()
                    break  # only bias one hidden node

        return True

    def save(self, directory: Path | None = None) -> None:
        """Persist all memories to disk."""
        directory = directory or MEMORY_DIR
        directory.mkdir(parents=True, exist_ok=True)

        for gen, mems in self.memories.items():
            path = directory / f"gen_{gen:04d}.json"
            payload = [m.to_payload() for m in mems]
            path.write_text(json.dumps(payload, indent=2))

    def load(self, directory: Path | None = None) -> int:
        """Load memories from disk. Returns count loaded."""
        directory = directory or MEMORY_DIR
        if not directory.exists():
            return 0

        count = 0
        for path in sorted(directory.glob("gen_*.json")):
            try:
                gen = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            payload = json.loads(path.read_text())
            self.memories[gen] = [GenMemory.from_payload(p) for p in payload]
            count += len(self.memories[gen])

        return count
