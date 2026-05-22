"""Exploration bonus — rewards agents for surprising the LLM while being self-predictable.

ExplorationBonus = w_e · max(0, surprise_to_LLM − surprise_to_self)

When the agent acts in ways the LLM finds unexpected but the agent itself can
predict, it earns exploration energy. This drives the transition from "learning
known" to "discovering unknown."
"""

from __future__ import annotations

import numpy as np

from genesis_v2.agent.agent import Agent
from genesis_v2.config import PhysicsConfig


def surprise_to_distribution(vec: np.ndarray, reference: np.ndarray, eps: float = 1e-9) -> float:
    """Compute surprise (cross-entropy) of vec relative to reference distribution.

    Both vectors are normalized to probability distributions, then cross-entropy
    is computed as the "surprise" of seeing vec given reference as the model.
    """
    v = np.abs(vec.astype(np.float64)) + eps
    r = np.abs(reference.astype(np.float64)) + eps
    p = v / v.sum()
    q = r / r.sum()
    # Cross-entropy: -sum(p * log(q))
    return float(-np.sum(p * np.log(q)))


def compute_exploration_bonus(
    agent: Agent,
    phy: PhysicsConfig,
    env_feedback: np.ndarray,
    self_prediction: np.ndarray | None = None,
) -> float:
    """Compute exploration bonus for one agent.

    Args:
        agent: The agent to evaluate.
        phy: Physics config (contains w_explore).
        env_feedback: The LLM/environment feedback vector.
        self_prediction: Agent's own prediction of its behavior (state buffer).
            If None, uses zero vector (maximum self-surprise).

    Returns:
        Exploration bonus value (unweighted). Multiply by phy.w_explore for energy.
    """
    if agent.last_action is None:
        return 0.0

    action = agent.last_action

    # surprise_to_LLM: how surprised the LLM is by the agent's action
    # Approximated as cross-entropy of action against env feedback
    min_len = min(len(action), len(env_feedback))
    surprise_llm = surprise_to_distribution(action[:min_len], env_feedback[:min_len])

    # surprise_to_self: how surprised the agent is by its own action
    # Use state buffer as self-prediction; if not available, assume high self-surprise
    if self_prediction is not None and len(self_prediction) > 0:
        min_len_self = min(len(action), len(self_prediction))
        surprise_self = surprise_to_distribution(action[:min_len_self], self_prediction[:min_len_self])
    else:
        surprise_self = 0.0  # no self-prediction = zero self-surprise = maximum bonus

    # Exploration bonus: only positive when LLM is more surprised than self
    raw_bonus = max(0.0, surprise_llm - surprise_self)

    return raw_bonus


def update_agent_exploration(
    agent: Agent,
    phy: PhysicsConfig,
    env_feedback: np.ndarray,
) -> float:
    """Compute and store exploration bonus on the agent. Returns weighted bonus."""
    self_pred = agent.state_buffer if agent.state_buffer is not None else None
    raw = compute_exploration_bonus(agent, phy, env_feedback, self_pred)
    agent.exploration_bonus = raw
    return phy.w_explore * raw
