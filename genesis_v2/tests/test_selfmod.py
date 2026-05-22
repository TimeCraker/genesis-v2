"""Tests for self-modification channel."""

import numpy as np
import pytest

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.engine.selfmod import (
    execute_selfmod,
    interpret_selfmod,
    reset_selfmod_inheritance,
    should_selfmod,
    SELFMOD_MUTATION_TYPES,
)
from genesis_v2.genome.graph import GraphConfig, new_genome_graph


@pytest.fixture
def agent():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    a = new_agent("selfmod-1", g, initial_energy=20000.0)
    return a


@pytest.fixture
def phy():
    return PhysicsConfig()


class TestInterpretSelfmod:
    def test_weights_sum_to_one(self):
        vec = np.random.default_rng(42).standard_normal(16).astype(np.float32)
        weights, params, trigger = interpret_selfmod(vec)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_trigger_in_range(self):
        for seed in range(10):
            vec = np.random.default_rng(seed).standard_normal(16).astype(np.float32)
            _, _, trigger = interpret_selfmod(vec)
            assert 0.0 <= trigger <= 1.0

    def test_short_vector_padded(self):
        vec = np.array([1.0, 2.0], dtype=np.float32)
        weights, params, trigger = interpret_selfmod(vec)
        assert len(weights) == 7
        assert len(params) == 5

    def test_high_trigger_value(self):
        vec = np.zeros(16, dtype=np.float32)
        vec[12:16] = 10.0  # very high trigger values
        _, _, trigger = interpret_selfmod(vec)
        assert trigger > 0.9


class TestShouldSelfmod:
    def test_sufficient_energy(self, agent, phy):
        assert should_selfmod(agent, phy, trigger=0.8) is True

    def test_insufficient_energy(self, agent, phy):
        agent.energy = 100.0  # below threshold
        assert should_selfmod(agent, phy, trigger=0.8) is False

    def test_low_trigger(self, agent, phy):
        assert should_selfmod(agent, phy, trigger=0.1) is False

    def test_dead_agent(self, agent, phy):
        agent.is_alive = False
        assert should_selfmod(agent, phy, trigger=0.8) is False


class TestExecuteSelfmod:
    def test_successful_selfmod(self, agent, phy):
        rng = np.random.default_rng(42)
        # Set high-energy agent with high trigger selfmod vector
        vec = np.zeros(16, dtype=np.float32)
        vec[0] = 10.0  # strong bias toward first mutation type
        vec[5] = 5.0   # perturb weight
        vec[12:16] = 10.0  # high trigger
        agent.last_selfmod = vec

        # Run many times; some should succeed (30% survival rate)
        successes = 0
        for seed in range(100):
            a = new_agent(f"test-{seed}", agent.genome.copy(), initial_energy=20000.0)
            a.last_selfmod = vec.copy()
            ok, desc = execute_selfmod(a, phy, np.random.default_rng(seed))
            if ok:
                successes += 1

        # With 30% survival, expect ~30 successes out of 100
        # Allow wide margin for statistical variance
        assert successes > 0, "No successful selfmods in 100 attempts"

    def test_no_selfmod_output(self, agent, phy):
        agent.last_selfmod = None
        ok, desc = execute_selfmod(agent, phy, np.random.default_rng(42))
        assert not ok
        assert desc == "no_selfmod_output"

    def test_energy_cost_paid(self, agent, phy):
        rng = np.random.default_rng(42)
        vec = np.zeros(16, dtype=np.float32)
        vec[12:16] = 10.0  # high trigger
        agent.last_selfmod = vec

        initial_energy = agent.energy
        # Force survival by setting death_rate to 0
        phy_copy = phy.model_copy()
        phy_copy.selfmod_death_rate = 0.0

        ok, desc = execute_selfmod(agent, phy_copy, rng)
        if ok:
            assert agent.energy < initial_energy

    def test_selfmod_fatal_sets_death(self, agent, phy):
        rng = np.random.default_rng(42)
        vec = np.zeros(16, dtype=np.float32)
        vec[12:16] = 10.0  # high trigger
        agent.last_selfmod = vec

        # Force death by setting death_rate to 1.0
        phy_copy = phy.model_copy()
        phy_copy.selfmod_death_rate = 1.0

        ok, desc = execute_selfmod(agent, phy_copy, rng)
        assert not ok
        assert "selfmod_death" in desc
        assert not agent.is_alive


class TestResetSelfmodInheritance:
    def test_resets_state(self):
        agent = Agent(id="test")
        agent.selfmod_enabled = True
        agent.selfmod_count = 5
        agent.selfmod_survived = 3
        reset_selfmod_inheritance([agent])
        assert agent.selfmod_enabled is False
        assert agent.selfmod_count == 0
        assert agent.selfmod_survived == 0


class TestMutationTypes:
    def test_seven_basic_types(self):
        assert len(SELFMOD_MUTATION_TYPES) == 7
