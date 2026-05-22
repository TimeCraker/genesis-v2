"""Tests for reaper v2."""

import numpy as np
import pytest

from genesis_v2.agent.agent import new_agent
from genesis_v2.config import PhysicsConfig
from genesis_v2.engine.reaper import (
    DeathReport, evaluate_death, kill_agent, sweep_island,
)
from genesis_v2.genome.graph import GraphConfig, new_genome_graph


@pytest.fixture
def agent():
    rng = np.random.default_rng(42)
    cfg = GraphConfig()
    g = new_genome_graph(cfg, rng)
    return new_agent(id="reap-0", genome=g, initial_energy=1000.0)


class TestEvaluateDeath:
    def test_alive_agent(self, agent):
        phy = PhysicsConfig()
        rep = evaluate_death(agent, phy)
        assert rep.dead is False

    def test_starvation(self, agent):
        phy = PhysicsConfig()
        agent.energy = 0.0
        rep = evaluate_death(agent, phy)
        assert rep.dead is True
        assert rep.reason == "starvation"

    def test_already_dead(self, agent):
        phy = PhysicsConfig()
        agent.is_alive = False
        rep = evaluate_death(agent, phy)
        assert rep.dead is True
        assert rep.reason == "already_dead"

    def test_topology_entropy(self, agent):
        phy = PhysicsConfig()
        # Force high entropy by setting threshold very low
        phy.topology_entropy_threshold = 0.01
        rep = evaluate_death(agent, phy)
        assert rep.dead is True
        assert rep.reason == "topology_entropy"


class TestSweepIsland:
    def test_sweep_kills_dead(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        phy = PhysicsConfig()

        agents = []
        for i in range(5):
            g = new_genome_graph(cfg, rng)
            a = new_agent(id=f"sweep-{i}", genome=g, initial_energy=1000.0)
            agents.append(a)

        agents[2].energy = 0.0
        killed = sweep_island(agents, phy)
        assert killed == 1
        assert not agents[2].is_alive
        assert all(a.is_alive for i, a in enumerate(agents) if i != 2)

    def test_sweep_returns_zero_when_all_alive(self):
        rng = np.random.default_rng(42)
        cfg = GraphConfig()
        phy = PhysicsConfig()
        agents = [new_agent(id=f"s-{i}", genome=new_genome_graph(cfg, rng), initial_energy=1000.0)
                  for i in range(3)]
        assert sweep_island(agents, phy) == 0


class TestKillAgent:
    def test_kill(self, agent):
        kill_agent(agent, "test_reason")
        assert not agent.is_alive
        assert agent._death_reason == "test_reason"
