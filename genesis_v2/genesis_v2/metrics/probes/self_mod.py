"""Self-Modification Efficiency Probe — measure if selfmod improves fitness.

Tracks selfmod attempts vs survival rate and fitness change. High survival
rate (> 0.3) with fitness improvement = effective self-optimization.
Low survival (< 0.05) = random self-destruction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelfModResult:
    survival_rate: float  # survived / attempted
    total_attempts: int
    total_survived: int
    mean_fitness_survivors: float
    mean_fitness_non_survivors: float
    fitness_delta: float  # survivors - non-survivors
    agents_with_selfmod: int


def probe_selfmod(agents: list) -> SelfModResult:
    """Measure self-modification efficiency across population."""
    alive = [a for a in agents if a.is_alive]
    dead = [a for a in agents if not a.is_alive]

    all_agents = alive + dead

    # Agents that attempted selfmod
    selfmodders = [a for a in all_agents if a.selfmod_count > 0]

    if not selfmodders:
        return SelfModResult(
            survival_rate=0.0,
            total_attempts=0,
            total_survived=0,
            mean_fitness_survivors=0.0,
            mean_fitness_non_survivors=0.0,
            fitness_delta=0.0,
            agents_with_selfmod=0,
        )

    total_attempts = sum(a.selfmod_count for a in selfmodders)
    total_survived = sum(a.selfmod_survived for a in selfmodders)
    survival_rate = total_survived / max(total_attempts, 1)

    # Compare fitness of survivors vs non-survivors
    survivors = [a for a in selfmodders if a.selfmod_survived > 0 and a.is_alive]
    non_survivors = [a for a in selfmodders if a.selfmod_survived == 0 or not a.is_alive]

    mean_fit_surv = (
        float(sum(a.fitness for a in survivors) / len(survivors))
        if survivors else 0.0
    )
    mean_fit_nonsurv = (
        float(sum(a.fitness for a in non_survivors) / len(non_survivors))
        if non_survivors else 0.0
    )

    return SelfModResult(
        survival_rate=float(survival_rate),
        total_attempts=total_attempts,
        total_survived=total_survived,
        mean_fitness_survivors=mean_fit_surv,
        mean_fitness_non_survivors=mean_fit_nonsurv,
        fitness_delta=mean_fit_surv - mean_fit_nonsurv,
        agents_with_selfmod=len(selfmodders),
    )
