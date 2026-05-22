"""Tests for metabolism v2."""

import numpy as np
import pytest

from genesis_v2.agent.agent import new_agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.engine.metabolism import (
    MetabolismTick, apply_metabolism, kl_action_truth, mdl_compression_proxy,
    tick_cost, tick_reward,
)
from genesis_v2.genome.graph import D_ACTION, GraphConfig, new_genome_graph


@pytest.fixture
def agent_with_action():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    agent = new_agent(id="metab-0", genome=g, initial_energy=5000.0)
    agent.last_action = rng.standard_normal(D_ACTION).astype(np.float32)
    return agent, rng


class TestMDLCompression:
    def test_basic(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig(initial_hidden_nodes=4)
        g = new_genome_graph(cfg, rng)
        comp = mdl_compression_proxy(g)
        assert comp < 0.0  # negative means "compressed"

    def test_more_nodes_more_negative(self):
        rng = np.random.default_rng(42)
        g1 = new_genome_graph(GraphConfig(initial_hidden_nodes=2), np.random.default_rng(42))
        g2 = new_genome_graph(GraphConfig(initial_hidden_nodes=8), np.random.default_rng(42))
        assert mdl_compression_proxy(g2) < mdl_compression_proxy(g1)


class TestKLActionTruth:
    def test_same_distribution(self):
        v = np.ones(10, dtype=np.float32)
        assert kl_action_truth(v, v) < 0.01

    def test_different_distributions(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        assert kl_action_truth(a, b) > 0.5


class TestTickCost:
    def test_positive(self):
        phy = PhysicsConfig()
        c = tick_cost(phy, token_usage=10, node_count=10, edge_count=5, latency_ms=1.0)
        assert c > 0

    def test_api_cost_term(self):
        phy = PhysicsConfig()
        c1 = tick_cost(phy, token_usage=10, node_count=10, edge_count=5, latency_ms=1.0, api_cost=0.0)
        c2 = tick_cost(phy, token_usage=10, node_count=10, edge_count=5, latency_ms=1.0, api_cost=1.0)
        assert c2 > c1

    def test_message_cost_term(self):
        phy = PhysicsConfig()
        c1 = tick_cost(phy, token_usage=10, node_count=10, edge_count=5, latency_ms=1.0, messages_sent=0)
        c2 = tick_cost(phy, token_usage=10, node_count=10, edge_count=5, latency_ms=1.0, messages_sent=5)
        assert c2 > c1


class TestTickReward:
    def test_prediction_improvement(self):
        phy = PhysicsConfig()
        r = tick_reward(phy, prev_kl=2.0, curr_kl=1.0,
                       prev_compression=-100, curr_compression=-100,
                       behavioral_variance=0.5)
        assert r > 0  # KL decreased → positive reward


class TestApplyMetabolism:
    def test_basic(self, agent_with_action):
        agent, rng = agent_with_action
        phy = PhysicsConfig()
        pop_mean = rng.standard_normal(D_ACTION).astype(np.float32)
        truth = rng.standard_normal(D_ACTION).astype(np.float32)

        result = apply_metabolism(agent, phy, truth=truth, pop_mean=pop_mean)
        assert isinstance(result, MetabolismTick)
        assert agent.energy != 5000.0  # energy changed

    def test_energy_decrease_over_time(self, agent_with_action):
        agent, rng = agent_with_action
        phy = PhysicsConfig()

        initial_energy = agent.energy
        # Use agent's own action as pop_mean to zero out BVar, and random truth
        for _ in range(50):
            truth = rng.standard_normal(D_ACTION).astype(np.float32)
            pop_mean = agent.last_action.copy()  # BVar ≈ 0
            apply_metabolism(agent, phy, truth=truth, pop_mean=pop_mean)

        # Over many ticks with random truth and zero BVar, cost should exceed reward
        assert agent.energy < initial_energy
