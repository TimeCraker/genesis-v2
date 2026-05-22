"""Energy accounting v2 — with API cost and message cost terms.

Cost = α·T_usage + β·|V|·log|V| + γ·L_latency + δ·|E| + ε·API_cost + ζ·|messages|
Reward = w_p·(-ΔKL) + w_c·ΔCompression + w_b·BVar + w_a·Adaptation + w_s·Social + w_e·Exploration
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import rel_entr

from genesis_v2.agent.agent import Agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.genome.graph import GenomeGraph


def mdl_compression_proxy(g: GenomeGraph) -> float:
    v = g.node_count()
    e = g.edge_count()
    return -(v * math.log(v + 1.0) + e * math.log(e + 1.0))


def kl_action_truth(action: np.ndarray, truth: np.ndarray, eps: float = 1e-9) -> float:
    a = np.abs(action.astype(np.float64)) + eps
    t = np.abs(truth.astype(np.float64)) + eps
    p = a / a.sum()
    q = t / t.sum()
    forward = float(np.sum(rel_entr(p, q)))
    backward = float(np.sum(rel_entr(q, p)))
    return 0.5 * (forward + backward)


@dataclass
class MetabolismTick:
    cost: float
    reward: float
    delta_energy: float
    kl: float
    compression: float
    behavioral_variance: float
    latency_ms: float
    token_usage: int


def tick_cost(
    phy: PhysicsConfig,
    *,
    token_usage: int,
    node_count: int,
    edge_count: int,
    latency_ms: float,
    api_cost: float = 0.0,
    messages_sent: int = 0,
) -> float:
    return (
        phy.alpha * float(token_usage)
        + phy.beta * float(node_count) * math.log(float(node_count + 1))
        + phy.gamma * float(latency_ms)
        + phy.delta * float(edge_count)
        + phy.epsilon * api_cost
        + phy.zeta * float(messages_sent)
    )


def tick_reward(
    phy: PhysicsConfig,
    *,
    prev_kl: float,
    curr_kl: float,
    prev_compression: float,
    curr_compression: float,
    behavioral_variance: float,
    adaptation: float = 0.0,
    social: float = 0.0,
    exploration: float = 0.0,
) -> float:
    delta_kl = curr_kl - prev_kl
    delta_comp = curr_compression - prev_compression
    return (
        phy.w_pred * (-delta_kl)
        + phy.w_comp * delta_comp
        + phy.w_bvar * behavioral_variance
        + phy.w_adapt * adaptation
        + phy.w_social * social
        + phy.w_explore * exploration
    )


def apply_metabolism(
    agent: Agent,
    phy: PhysicsConfig,
    *,
    truth: np.ndarray | None = None,
    pop_mean: np.ndarray,
    feedback: np.ndarray | None = None,
    latency_ms: float | None = None,
    token_usage: int | None = None,
    api_cost: float = 0.0,
    messages_sent: int = 0,
    social: float = 0.0,
    adaptation: float = 0.0,
    exploration: float = 0.0,
) -> MetabolismTick:
    if agent.genome is None:
        raise ValueError("agent.genome required")

    g = agent.genome
    action = agent.last_action
    if action is None:
        raise ValueError("agent.last_action must be set before metabolism")

    # KL: use feedback or truth
    target = feedback if feedback is not None else truth
    if target is not None:
        # Ensure compatible dims for KL computation
        a = action[:len(target)] if len(action) >= len(target) else np.pad(action, (0, len(target) - len(action)))
        curr_kl = kl_action_truth(a, target)
    else:
        curr_kl = 0.0

    curr_comp = mdl_compression_proxy(g)

    prev_kl = float(getattr(agent, "_prev_kl", curr_kl))
    prev_comp = float(getattr(agent, "_prev_comp", curr_comp))

    bvar = float(np.linalg.norm(action.astype(np.float64) - pop_mean[:len(action)]))

    tok = token_usage if token_usage is not None else int(len(action))
    lat = latency_ms if latency_ms is not None else 0.0

    cost = tick_cost(
        phy,
        token_usage=tok,
        node_count=g.node_count(),
        edge_count=g.edge_count(),
        latency_ms=lat,
        api_cost=api_cost,
        messages_sent=messages_sent,
    )

    rew = tick_reward(
        phy,
        prev_kl=prev_kl,
        curr_kl=curr_kl,
        prev_compression=prev_comp,
        curr_compression=curr_comp,
        behavioral_variance=bvar,
        adaptation=adaptation,
        social=social,
        exploration=exploration,
    )

    delta_e = rew - cost

    agent.prediction_error = curr_kl
    agent.compression = curr_comp
    agent.behavioral_variance = bvar
    agent.fitness += delta_e
    agent.energy += delta_e

    agent._prev_kl = curr_kl  # type: ignore[attr-defined]
    agent._prev_comp = curr_comp  # type: ignore[attr-defined]

    return MetabolismTick(
        cost=cost,
        reward=rew,
        delta_energy=delta_e,
        kl=curr_kl,
        compression=curr_comp,
        behavioral_variance=bvar,
        latency_ms=lat,
        token_usage=tok,
    )
