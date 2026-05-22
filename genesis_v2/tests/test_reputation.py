"""Tests for reputation system — cooperation detection and social rewards."""

import numpy as np
import pytest

from genesis_v2.agent.agent import Agent, new_agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.genome.graph import D_ACTION, GraphConfig, new_genome_graph
from genesis_v2.social.reputation import (
    _COOP_THRESHOLD,
    cosine_similarity,
    detect_cooperation,
    get_social_reward,
    update_reputation,
)


@pytest.fixture
def two_agents():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    a = new_agent(id="a", genome=new_genome_graph(cfg, rng), initial_energy=1000.0)
    b = new_agent(id="b", genome=new_genome_graph(cfg, rng), initial_energy=1000.0)
    return a, b


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.ones(10, dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        v = np.ones(10, dtype=np.float32)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = np.zeros(10, dtype=np.float32)
        b = np.ones(10, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


class TestReputation:
    def test_update_reputation_creates_entry(self, two_agents):
        a, b = two_agents
        update_reputation(a, b, 0.8)
        assert b.id in a.social_memory
        assert a.id in b.social_memory
        assert a.social_memory[b.id] == pytest.approx(0.16)  # 0.2 * 0.8
        assert b.social_memory[a.id] == pytest.approx(0.16)

    def test_update_reputation_ema(self, two_agents):
        a, b = two_agents
        update_reputation(a, b, 1.0)
        update_reputation(a, b, 1.0)
        # After two updates: 0.8*(0.2*1.0) + 0.2*1.0 = 0.16 + 0.2 = 0.36
        assert a.social_memory[b.id] == pytest.approx(0.36)

    def test_detect_cooperation_similar_actions(self, two_agents):
        a, b = two_agents
        a.last_action = np.ones(D_ACTION, dtype=np.float32)
        b.last_action = np.ones(D_ACTION, dtype=np.float32) * 0.9
        cooperations = detect_cooperation([a, b])
        assert len(cooperations) == 1
        assert cooperations[0][0] == a.id
        assert cooperations[0][1] == b.id

    def test_detect_cooperation_no_match(self, two_agents):
        a, b = two_agents
        a.last_action = np.ones(D_ACTION, dtype=np.float32)
        b.last_action = -np.ones(D_ACTION, dtype=np.float32)
        cooperations = detect_cooperation([a, b])
        assert len(cooperations) == 0

    def test_detect_cooperation_skips_dead(self, two_agents):
        a, b = two_agents
        a.last_action = np.ones(D_ACTION, dtype=np.float32)
        b.last_action = np.ones(D_ACTION, dtype=np.float32)
        b.is_alive = False
        cooperations = detect_cooperation([a, b])
        assert len(cooperations) == 0

    def test_social_reward_positive(self, two_agents):
        a, b = two_agents
        a.social_memory = {"b": 0.5, "c": 0.3}
        phy = PhysicsConfig(w_social=1.0)
        reward = get_social_reward(a, phy)
        assert reward == pytest.approx(0.4)  # mean of 0.5, 0.3

    def test_social_reward_empty(self, two_agents):
        a, _ = two_agents
        phy = PhysicsConfig(w_social=1.0)
        assert get_social_reward(a, phy) == 0.0

    def test_social_reward_negative_trust_capped(self, two_agents):
        a, _ = two_agents
        a.social_memory = {"x": -0.5}
        phy = PhysicsConfig(w_social=1.0)
        assert get_social_reward(a, phy) == 0.0  # max(0, -0.5)
