"""Tests for Agent v2."""

import numpy as np
import pytest

from genesis_v2.agent.agent import Agent, new_agent, split_output
from genesis_v2.genome.graph import (
    D_ACTION, D_MESSAGE, D_NODE, D_OUT, D_SELFMOD, D_STATE,
    GraphConfig, new_genome_graph,
)


@pytest.fixture
def sample_agent():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    return new_agent(id="test-0", genome=g, initial_energy=1000.0)


class TestAgent:
    def test_creation(self, sample_agent):
        assert sample_agent.id == "test-0"
        assert sample_agent.energy == 1000.0
        assert sample_agent.is_alive is True
        assert sample_agent.generation == 0
        assert sample_agent.genome is not None

    def test_birth_snapshot(self, sample_agent):
        assert sample_agent.birth_nodes > 0
        assert sample_agent.birth_edges > 0

    def test_default_values(self, sample_agent):
        assert sample_agent.tick_alive == 0
        assert sample_agent.fitness == 0.0
        assert sample_agent.prediction_error == 0.0
        assert sample_agent.compression == 0.0
        assert sample_agent.selfmod_count == 0
        assert sample_agent.selfmod_enabled is False


class TestSplitOutput:
    def test_split_output_shape(self):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(D_OUT).astype(np.float32)
        action, message, state, selfmod = split_output(vec)
        assert action.shape == (D_ACTION,)
        assert message.shape == (D_MESSAGE,)
        assert state.shape == (D_STATE,)
        assert selfmod.shape == (D_SELFMOD,)

    def test_split_output_values(self):
        vec = np.arange(D_OUT, dtype=np.float32)
        action, message, state, selfmod = split_output(vec)
        np.testing.assert_array_equal(action, vec[:D_ACTION])
        np.testing.assert_array_equal(message, vec[D_ACTION:D_ACTION + D_MESSAGE])
        np.testing.assert_array_equal(state, vec[D_ACTION + D_MESSAGE:D_ACTION + D_MESSAGE + D_STATE])
        np.testing.assert_array_equal(selfmod, vec[D_ACTION + D_MESSAGE + D_STATE:D_ACTION + D_MESSAGE + D_STATE + D_SELFMOD])

    def test_split_output_wrong_dim(self):
        vec = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError):
            split_output(vec)


class TestAgentPersistence:
    def test_roundtrip(self, sample_agent):
        payload = sample_agent.to_payload()
        a2 = Agent.from_payload(payload)
        assert a2.id == sample_agent.id
        assert a2.energy == sample_agent.energy
        assert a2.generation == sample_agent.generation
        assert a2.genome.node_count() == sample_agent.genome.node_count()
