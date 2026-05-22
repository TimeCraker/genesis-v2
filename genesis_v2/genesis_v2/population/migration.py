"""Cross-LLM migration — transfer elite agents between islands.

Every G generations, top agents from each island migrate to a neighboring
island (ring topology). Agents that adapt quickly to the new environment
earn an adaptation bonus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import PhysicsConfig


@dataclass
class MigrationRecord:
    """Record of a single agent migration."""

    agent_id: str
    from_island: int
    to_island: int
    generation: int
    fitness_at_migration: float
    ticks_since_migration: int = 0
    kl_at_migration: float = 0.0


@dataclass
class MigrationTracker:
    """Tracks migrants and computes adaptation bonuses."""

    records: list[MigrationRecord] = field(default_factory=list)
    migrant_ids: set[str] = field(default_factory=set)

    def register_migration(
        self,
        agent: Agent,
        from_island: int,
        to_island: int,
    ) -> MigrationRecord:
        """Register an agent as having migrated."""
        record = MigrationRecord(
            agent_id=agent.id,
            from_island=from_island,
            to_island=to_island,
            generation=agent.generation,
            fitness_at_migration=agent.fitness,
            kl_at_migration=agent.prediction_error,
        )
        self.records.append(record)
        self.migrant_ids.add(agent.id)
        return record

    def is_migrant(self, agent_id: str) -> bool:
        return agent_id in self.migrant_ids

    def get_record(self, agent_id: str) -> MigrationRecord | None:
        for r in reversed(self.records):
            if r.agent_id == agent_id:
                return r
        return None

    def tick_migrants(self) -> None:
        """Increment tick counter for all active migrants."""
        for r in self.records:
            if r.ticks_since_migration < 100:
                r.ticks_since_migration += 1


def migration_adaptation_bonus(
    agent: Agent,
    tracker: MigrationTracker,
    phy: PhysicsConfig,
) -> float:
    """Compute adaptation bonus for a recent migrant.

    The bonus rewards agents that quickly improve their KL (prediction error)
    after being placed in a new LLM environment. This directly rewards
    "general understanding" rather than memorization of one LLM's patterns.

    The bonus decays linearly over 20 ticks after migration.
    """
    if not tracker.is_migrant(agent.id):
        return 0.0

    record = tracker.get_record(agent.id)
    if record is None:
        return 0.0

    ticks = record.ticks_since_migration
    if ticks > 20:
        return 0.0

    # KL improvement rate: how much prediction error decreased since migration
    kl_improvement = record.kl_at_migration - agent.prediction_error
    if kl_improvement <= 0:
        return 0.0

    # Linear decay over 20 ticks
    decay = 1.0 - (ticks / 20.0)
    return phy.w_adapt * kl_improvement * decay


def select_migrants(
    agents: list[Agent],
    n: int = 3,
) -> list[Agent]:
    """Select top-n agents for migration."""
    alive = [a for a in agents if a.is_alive]
    if not alive:
        return []
    return sorted(alive, key=lambda a: a.fitness, reverse=True)[:n]


def migrate_agents(
    islands: list,
    rng: np.random.Generator,
    cfg_physics: PhysicsConfig,
    n_per_island: int = 3,
) -> MigrationTracker:
    """Perform ring-topology migration across islands.

    Each island sends its top-n agents to the next island (ring).
    Returns a tracker with all migration records.
    """
    if len(islands) < 2:
        return MigrationTracker()

    tracker = MigrationTracker()

    # Collect migrants from each island before moving them
    migrants_per_island: list[list[Agent]] = []
    for isl in islands:
        migrants = select_migrants(isl.agents, n_per_island)
        migrants_per_island.append(migrants)

    # Ring topology: island i sends to island (i+1) % n
    for i, isl in enumerate(islands):
        target_idx = (i + 1) % len(islands)
        target_isl = islands[target_idx]
        migrants = migrants_per_island[i]

        for migrant in migrants:
            # Clone the migrant for the target island
            child = new_agent(
                id=f"mig-{migrant.id}-to-{target_isl.id}",
                genome=migrant.genome.copy(),
                initial_energy=cfg_physics.initial_energy,
                generation=migrant.generation,
                island_id=target_isl.id,
            )
            child.genome.reset_state()

            # Remove one of the weakest agents from target island to make room
            alive_in_target = [a for a in target_isl.agents if a.is_alive]
            if len(alive_in_target) >= target_isl.island_cfg.size:
                weakest = min(alive_in_target, key=lambda a: a.fitness)
                weakest.is_alive = False

            target_isl.agents.append(child)
            tracker.register_migration(child, from_island=isl.id, to_island=target_isl.id)

    return tracker
