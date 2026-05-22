"""Self-modification channel — agent-driven topology mutation.

Agent output [512:528] is interpreted as self-modification instructions:
    [0:7]   → 7 basic mutation tendency weights
    [7:12]  → mutation strength/direction parameters
    [12:16] → trigger threshold (sigmoid activation)

Safety gates:
    - Energy > selfmod_energy_threshold to execute
    - Each selfmod costs selfmod_energy_cost
    - selfmod_death_rate probability of degradation
    - Not inherited (selfmod_enabled resets each generation)
"""

from __future__ import annotations

import numpy as np

from genesis_v2.agent.agent import Agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.genome.mutate import MutationKind

# The 7 basic mutation types that selfmod can influence
SELFMOD_MUTATION_TYPES = [
    MutationKind.ADD_FORWARD_EDGE,
    MutationKind.ADD_SHORTCUT_EDGE,
    MutationKind.ADD_RECURRENT_EDGE,
    MutationKind.ADD_HIDDEN_NODE,
    MutationKind.ADD_GATING_NODE,
    MutationKind.PERTURB_WEIGHT,
    MutationKind.DELETE_RANDOM_EDGE,
]


def interpret_selfmod(selfmod_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpret the 16-dim selfmod vector.

    Returns:
        mutation_weights: [7] softmax weights for mutation type selection
        params: [5] raw parameter vector
        trigger: scalar in [0, 1] — probability of triggering selfmod
    """
    if len(selfmod_vec) < 16:
        selfmod_vec = np.pad(selfmod_vec, (0, 16 - len(selfmod_vec)))

    raw_weights = selfmod_vec[:7].astype(np.float64)
    # Softmax for mutation type selection
    exp_w = np.exp(raw_weights - raw_weights.max())
    mutation_weights = exp_w / exp_w.sum()

    params = selfmod_vec[7:12].astype(np.float64)

    # Trigger threshold: sigmoid of mean of [12:16]
    trigger_raw = float(np.mean(selfmod_vec[12:16].astype(np.float64)))
    trigger = 1.0 / (1.0 + np.exp(-trigger_raw))

    return mutation_weights, params, trigger


def should_selfmod(
    agent: Agent,
    phy: PhysicsConfig,
    trigger: float,
) -> bool:
    """Check if self-modification should be attempted.

    Requirements:
    1. Agent is alive
    2. Energy > selfmod_energy_threshold
    3. Trigger probability exceeded
    """
    if not agent.is_alive:
        return False
    if agent.energy < phy.selfmod_energy_threshold:
        return False
    return trigger > 0.5


def execute_selfmod(
    agent: Agent,
    phy: PhysicsConfig,
    rng: np.random.Generator,
) -> tuple[bool, str]:
    """Attempt self-modification on an agent.

    Returns:
        (success, description) — success means the agent survived the selfmod.
    """
    if agent.last_selfmod is None:
        return False, "no_selfmod_output"

    mutation_weights, params, trigger = interpret_selfmod(agent.last_selfmod)

    if not should_selfmod(agent, phy, trigger):
        return False, "not_triggered"

    # Pay the energy cost
    agent.energy -= phy.selfmod_energy_cost
    agent.selfmod_count += 1
    agent.selfmod_enabled = True

    # Select mutation type based on weights
    mutation_idx = int(rng.choice(len(SELFMOD_MUTATION_TYPES), p=mutation_weights))
    mutation_type = SELFMOD_MUTATION_TYPES[mutation_idx]

    # Apply the mutation
    from genesis_v2.genome.mutate import _DISPATCH
    ok = _DISPATCH[mutation_type](agent.genome, rng)
    if not ok:
        return False, f"mutation_failed:{mutation_type.name}"

    # Death roll: selfmod_death_rate probability of fatal self-modification
    if rng.random() < phy.selfmod_death_rate:
        agent._selfmod_fatal = True  # type: ignore[attr-defined]
        agent.is_alive = False
        return False, f"selfmod_death:{mutation_type.name}"

    agent.selfmod_survived += 1
    return True, f"selfmod_ok:{mutation_type.name}"


def reset_selfmod_inheritance(agents: list[Agent]) -> None:
    """Reset selfmod state for new generation (not inherited)."""
    for a in agents:
        a.selfmod_enabled = False
        a.selfmod_count = 0
        a.selfmod_survived = 0
