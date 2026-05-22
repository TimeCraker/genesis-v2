"""Reputation system — trust scores and social cooperation rewards.

Agents build trust by cooperating (compatible actions within comm_radius).
Trust scores contribute to social energy rewards.
"""

from __future__ import annotations

import numpy as np

from genesis_v2.agent.agent import Agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.genome.graph import D_MESSAGE

# Cooperation threshold: actions are "compatible" if cosine similarity > this
_COOP_THRESHOLD = 0.3


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def update_reputation(
    agent_a: Agent,
    agent_b: Agent,
    cooperation_score: float,
) -> None:
    """Update trust scores between two agents after cooperation."""
    # Exponential moving average: trust = 0.8 * old + 0.2 * new
    old_a = agent_a.social_memory.get(agent_b.id, 0.0)
    old_b = agent_b.social_memory.get(agent_a.id, 0.0)
    agent_a.social_memory[agent_b.id] = 0.8 * old_a + 0.2 * cooperation_score
    agent_b.social_memory[agent_a.id] = 0.8 * old_b + 0.2 * cooperation_score


def detect_cooperation(
    agents: list[Agent],
    comm_radius: int = 2,
) -> list[tuple[str, str, float]]:
    """Detect cooperative pairs: agents with compatible actions.

    Returns list of (agent_a_id, agent_b_id, cooperation_score).
    """
    cooperations = []
    alive = [a for a in agents if a.is_alive and a.last_action is not None]
    for i, a in enumerate(alive):
        for j in range(i + 1, len(alive)):
            b = alive[j]
            sim = cosine_similarity(a.last_action, b.last_action)
            if sim > _COOP_THRESHOLD:
                cooperations.append((a.id, b.id, sim))
    return cooperations


def get_social_reward(agent: Agent, phy: PhysicsConfig) -> float:
    """Compute social cooperation reward from accumulated trust scores."""
    if not agent.social_memory:
        return 0.0
    mean_trust = float(np.mean(list(agent.social_memory.values())))
    return phy.w_social * max(0.0, mean_trust)
