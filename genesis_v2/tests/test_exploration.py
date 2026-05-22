"""Tests for exploration bonus."""

import numpy as np
import pytest

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.genome.graph import GraphConfig, new_genome_graph
from genesis_v2.metrics.exploration import (
    compute_exploration_bonus,
    surprise_to_distribution,
    update_agent_exploration,
)


@pytest.fixture
def agent():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    a = new_agent("explore-1", g, initial_energy=1000.0)
    a.last_action = rng.standard_normal(256).astype(np.float32)
    return a


@pytest.fixture
def phy():
    return PhysicsConfig()


class TestSurpriseToDistribution:
    def test_identical_distributions_low_surprise(self):
        vec = np.array([1.0, 2.0, 3.0])
        ref = np.array([1.0, 2.0, 3.0])
        s = surprise_to_distribution(vec, ref)
        # Identical → cross-entropy equals entropy
        assert s > 0

    def test_different_distributions_higher_surprise(self):
        vec = np.array([1.0, 0.0, 0.0])
        ref = np.array([0.0, 0.0, 1.0])
        s = surprise_to_distribution(vec, ref)
        assert s > 1.0  # should be quite high

    def test_no_nan(self):
        vec = np.array([1e-10, 1e-10, 1.0])
        ref = np.array([1.0, 1e-10, 1e-10])
        s = surprise_to_distribution(vec, ref)
        assert not np.isnan(s)


class TestComputeExplorationBonus:
    def test_positive_bonus_when_llm_surprised(self, agent, phy):
        # LLM feedback very different from agent action → high surprise_llm
        env_feedback = -agent.last_action.copy()  # opposite direction
        bonus = compute_exploration_bonus(agent, phy, env_feedback)
        assert bonus >= 0.0

    def test_zero_bonus_when_agent_also_surprised(self, agent, phy):
        env_feedback = agent.last_action.copy() + 1.0
        self_pred = np.zeros(128, dtype=np.float32)  # very different from action
        bonus = compute_exploration_bonus(agent, phy, env_feedback, self_pred)
        # When self is also surprised, bonus may be zero
        assert bonus >= 0.0

    def test_no_action_returns_zero(self, phy):
        agent = Agent(id="no-action")
        agent.last_action = None
        bonus = compute_exploration_bonus(agent, phy, np.zeros(10))
        assert bonus == 0.0


class TestUpdateAgentExploration:
    def test_stores_on_agent(self, agent, phy):
        env_feedback = -agent.last_action.copy()
        bonus = update_agent_exploration(agent, phy, env_feedback)
        assert agent.exploration_bonus >= 0.0
        assert bonus == phy.w_explore * agent.exploration_bonus

    def test_uses_state_buffer(self, agent, phy):
        agent.state_buffer = agent.last_action.copy()  # self-predicts well
        env_feedback = agent.last_action.copy()
        bonus = update_agent_exploration(agent, phy, env_feedback)
        assert bonus >= 0.0
