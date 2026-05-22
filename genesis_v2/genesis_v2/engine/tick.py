"""Tick engine v2 — with social layer (CommunicationBus + reputation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from genesis_v2.agent.agent import Agent, split_output
from genesis_v2.config import GenesisConfig
from genesis_v2.engine.metabolism import apply_metabolism
from genesis_v2.engine.reaper import evaluate_death, kill_agent, sweep_island
from genesis_v2.genome.graph import D_MESSAGE

if TYPE_CHECKING:
    from typing import Any

    from genesis_v2.env.mock import MockMathEnvironment
    from genesis_v2.social.comm_bus import CommunicationBus
    from genesis_v2.storage.duckdb_store import DuckDBStore

    Environment = Any  # Protocol: observe(), interact(), true_distribution(), close()


def build_social_input(
    obs: np.ndarray,
    inbox: np.ndarray,
    n_input: int,
    d_node: int,
) -> np.ndarray:
    """Merge environment observation and inbox into agent input vector.

    First (n_input - 2) slots get tiled env observation.
    Last 2 slots (D_MESSAGE=128 dims) get inbox vector.
    """
    env_slots = n_input - 2
    d_in = n_input * d_node
    result = np.zeros(d_in, dtype=np.float32)

    # Tile obs across first env_slots slots
    for i in range(env_slots):
        start = i * d_node
        end = start + d_node
        obs_idx = i % len(obs)
        result[start:end] = obs[obs_idx] if len(obs) > 0 else 0.0

    # Place inbox in last 2 slots (128 dims)
    inbox_start = env_slots * d_node
    inbox_len = min(len(inbox), 2 * d_node)
    result[inbox_start:inbox_start + inbox_len] = inbox[:inbox_len]

    return result


def island_step_sync(
    agents: list[Agent],
    env,
    phy,
    store: DuckDBStore | None,
    tick_id: int,
    rng: np.random.Generator | None = None,
    comm_bus: CommunicationBus | None = None,
) -> int:
    """Core single-frame function. Returns number of newly killed agents."""
    alive = [a for a in agents if a.is_alive]
    if not alive:
        return 0

    # 1. Observe
    obs = env.observe()
    n_input = len(alive[0].genome.input_nodes)
    d_node = alive[0].genome.nodes[alive[0].genome.input_nodes[0]].dim
    d_in = n_input * d_node

    # Pre-compute Phase 0 tiled obs (shared across agents when no comm_bus)
    if comm_bus is None:
        obs_tiled = np.tile(obs, n_input).astype(np.float32)
        if obs_tiled.shape[0] > d_in:
            obs_tiled = obs_tiled[:d_in]
        elif obs_tiled.shape[0] < d_in:
            obs_tiled = np.pad(obs_tiled, (0, d_in - obs_tiled.shape[0]))

    # 2. Forward all agents + deliver messages
    pop_actions = []
    for a in alive:
        if comm_bus is not None:
            inbox = comm_bus.get_inbox(a.id)
            agent_input = build_social_input(obs, inbox, n_input, d_node)
        else:
            agent_input = obs_tiled

        out = a.genome.forward(agent_input)
        action, message, state, selfmod = split_output(out)
        a.last_action = action
        a.last_message = message
        a.state_buffer = a.last_state
        a.last_state = state
        a.last_selfmod = selfmod
        pop_actions.append(action)

        # Deliver messages (social only)
        if comm_bus is not None:
            comm_bus.deliver(a.id, message)

    if not pop_actions:
        return 0

    # 3. Population mean
    pop_mean = np.mean(np.stack(pop_actions, axis=0), axis=0).astype(np.float32)

    # 4. Interact with environment
    feedback = env.interact(pop_mean.copy())
    truth = env.true_distribution(feedback.astype(np.float32))

    # 5. Social cooperation detection (if social)
    social_rewards: dict[str, float] = {}
    if comm_bus is not None:
        from genesis_v2.social.reputation import (
            detect_cooperation,
            get_social_reward,
            update_reputation,
        )
        cooperations = detect_cooperation(alive)
        _agent_map = {a.id: a for a in alive}
        for a_id, b_id, score in cooperations:
            update_reputation(_agent_map[a_id], _agent_map[b_id], score)
        for a in alive:
            social_rewards[a.id] = get_social_reward(a, phy)

    # 5b. Exploration bonus
    exploration_rewards: dict[str, float] = {}
    for a in alive:
        from genesis_v2.metrics.exploration import update_agent_exploration
        exp_bonus = update_agent_exploration(a, phy, feedback)
        exploration_rewards[a.id] = exp_bonus

    # 5c. Self-modification
    if rng is not None:
        from genesis_v2.engine.selfmod import execute_selfmod
        for a in alive:
            execute_selfmod(a, phy, rng)

    # Refresh alive list after selfmod (some may have died)
    alive = [a for a in agents if a.is_alive]
    pop_actions = [a.last_action for a in alive if a.last_action is not None]
    if pop_actions:
        pop_mean = np.mean(np.stack(pop_actions, axis=0), axis=0).astype(np.float32)

    # 6. Metabolize each agent
    for a in alive:
        social_r = social_rewards.get(a.id, 0.0) if comm_bus is not None else 0.0
        messages = comm_bus.message_count(a.id) if comm_bus is not None else 0
        explore_r = exploration_rewards.get(a.id, 0.0)

        apply_metabolism(
            a, phy,
            truth=truth,
            pop_mean=pop_mean,
            feedback=feedback,
            messages_sent=messages,
            social=social_r,
            exploration=explore_r,
        )
        if store is not None:
            mean_trust = 0.0
            if a.social_memory:
                mean_trust = float(np.mean(list(a.social_memory.values())))
            store.record_tick(
                tick_id=tick_id, agent=a,
                messages_received=messages,
                mean_trust=mean_trust,
            )

    # 7. Reap
    killed = sweep_island(agents, phy)

    # 8. Clear inboxes (social only)
    if comm_bus is not None:
        comm_bus.clear_all()

    return killed


class TickEngine:
    def __init__(
        self,
        cfg: GenesisConfig,
        agents: list[Agent],
        env,
        store: DuckDBStore | None = None,
        rng: np.random.Generator | None = None,
        comm_bus: CommunicationBus | None = None,
    ) -> None:
        self.cfg = cfg
        self.agents = agents
        self.env = env
        self.store = store
        self.rng = rng or np.random.default_rng(42)
        self.comm_bus = comm_bus
        self.global_tick = 0

    def step_sync(self) -> int:
        killed = island_step_sync(
            self.agents, self.env, self.cfg.physics,
            self.store, self.global_tick, self.rng,
            comm_bus=self.comm_bus,
        )
        self.global_tick += 1
        return killed

    def run_ticks(self, n: int) -> None:
        for _ in range(n):
            self.step_sync()


def multi_island_step(
    islands: list,
    cfg: GenesisConfig,
    store: DuckDBStore | None,
    tick_id: int,
    rng: np.random.Generator,
) -> dict[int, int]:
    """Run one tick across all islands. Returns {island_id: killed_count}."""
    results: dict[int, int] = {}
    for isl in islands:
        killed = island_step_sync(
            isl.agents, isl.env, cfg.physics, store, tick_id, rng,
            comm_bus=isl.comm_bus,
        )
        results[isl.id] = killed

        # Budget fallback: if island budget exceeded, swap to mock
        if isl.budget is not None and not isl.is_mock:
            if isl.budget.should_fallback(isl.id):
                isl.env.close()
                isl.env = MockMathEnvironment(
                    n_cells=cfg.genome.node_dim,
                    rng=np.random.default_rng(rng.integers(2**31)),
                )
                isl.is_mock = True

    return results
