"""Exploration Reward Effectiveness Probe — does exploring lead to higher fitness?

Measures the correlation between exploration bonus and subsequent fitness.
Positive correlation + agents with exploration_bonus > 0 having higher fitness
= exploration is beneficial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExplorationResult:
    exploration_ratio: float  # fraction of agents with exploration_bonus > 0
    mean_exploration_bonus: float
    fitness_explorers: float  # mean fitness of explorers
    fitness_non_explorers: float  # mean fitness of non-explorers
    fitness_delta: float  # explorers - non-explorers
    n_agents: int


def probe_exploration(agents: list) -> ExplorationResult:
    """Measure exploration effectiveness across population."""
    alive = [a for a in agents if a.is_alive]
    if not alive:
        return ExplorationResult(
            exploration_ratio=0.0,
            mean_exploration_bonus=0.0,
            fitness_explorers=0.0,
            fitness_non_explorers=0.0,
            fitness_delta=0.0,
            n_agents=0,
        )

    explorers = [a for a in alive if a.exploration_bonus > 0]
    non_explorers = [a for a in alive if a.exploration_bonus <= 0]

    exploration_ratio = len(explorers) / len(alive)
    mean_bonus = float(np.mean([a.exploration_bonus for a in alive]))

    fit_exp = float(np.mean([a.fitness for a in explorers])) if explorers else 0.0
    fit_non = float(np.mean([a.fitness for a in non_explorers])) if non_explorers else 0.0

    return ExplorationResult(
        exploration_ratio=float(exploration_ratio),
        mean_exploration_bonus=mean_bonus,
        fitness_explorers=fit_exp,
        fitness_non_explorers=fit_non,
        fitness_delta=fit_exp - fit_non,
        n_agents=len(alive),
    )
