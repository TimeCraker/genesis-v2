"""Death conditions v2 — with self-modification fatal condition."""

from __future__ import annotations

from dataclasses import dataclass

from genesis_v2.agent.agent import Agent
from genesis_v2.config import PhysicsConfig


@dataclass
class DeathReport:
    dead: bool
    reason: str


def evaluate_death(
    agent: Agent,
    phy: PhysicsConfig,
    *,
    context_tokens: int = 0,
    context_limit: int = 10_000_000,
    llm_is_error: bool = False,
) -> DeathReport:
    if not agent.is_alive:
        return DeathReport(True, "already_dead")

    if agent.energy <= 0.0:
        return DeathReport(True, "starvation")

    if llm_is_error:
        return DeathReport(True, "llm_error")

    if context_tokens > context_limit:
        return DeathReport(True, "context_overflow")

    if agent.genome is not None:
        h = agent.genome.entropy()
        if h >= phy.topology_entropy_threshold:
            return DeathReport(True, "topology_entropy")

    # v2: self-modification fatal
    if getattr(agent, "_selfmod_fatal", False):
        return DeathReport(True, "selfmod_fatal")

    return DeathReport(False, "")


def kill_agent(agent: Agent, reason: str) -> None:
    agent.is_alive = False
    agent._death_reason = reason  # type: ignore[attr-defined]


def sweep_island(agents: list[Agent], phy: PhysicsConfig, **kwargs) -> int:
    n = 0
    for a in agents:
        rep = evaluate_death(a, phy, **kwargs)
        if rep.dead and a.is_alive:
            kill_agent(a, rep.reason)
            n += 1
    return n
